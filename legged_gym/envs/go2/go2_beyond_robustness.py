"""Stage-4 Isaac Lab BR adapter; NOT a production reward/task registration.

Pass resolve_beyond_robustness() output and explicitly select
``reward_mode='smoke_tracking'`` or supply a reward callable. The temporary
reward is raw planar+yaw tracking, not the paper's reward port.
Native 45-D proprioception uses SI units, absolute joint angles, and the previous
clipped issued policy action (before delay). Actor noise is cached per timestamp;
privileged state/load supervision stays clean and unclipped.
History belongs exclusively to the BR runner. observe() is side-effect free.
Only simulator-neutral curriculum/disturbance methods are reused from Go2;
no HardPACT model, loss, QP, or training algorithm is constructed.
"""
from copy import deepcopy
import math
import torch
from rsl_rl.beyond_robustness_env import BRObservation, BRStep
from rsl_rl.beyond_robustness_config import BeyondRobustnessResolvedConfig
from rsl_rl.beyond_robustness_sensors import BRActorObservationCache
from rsl_rl.beyond_robustness_payload import inverse_rotate_wxyz, lost_payload, termination_masks, named_indices
from legged_gym.simulator.isaaclab_simulator_br import IsaacLabSimulatorBR
from legged_gym.envs.go2.go2_hard_pact.domain_rand_curriculum import HardPACTDomainRandCurriculum
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT


class IsaacLabBeyondRobustnessEnv:
    supports_exact_resume = False  # PhysX contact/solver state is not restorable.
    _apply_domain_rand_curriculum_state = Go2HardPACT._apply_domain_rand_curriculum_state
    _expand_command_curriculum_ranges = Go2HardPACT._expand_command_curriculum_ranges
    _update_curriculum_heading_command = Go2HardPACT._update_curriculum_heading_command
    _persistent_component_settings = Go2HardPACT._persistent_component_settings
    _sample_step_range = Go2HardPACT._sample_step_range
    _schedule_next_persistent_event = Go2HardPACT._schedule_next_persistent_event
    _start_due_persistent_events = Go2HardPACT._start_due_persistent_events
    _update_persistent_wrench = Go2HardPACT._update_persistent_wrench
    _update_terrain_curriculum = Go2PACT._update_terrain_curriculum
    _filter_terrain_move_up = Go2HardPACT._filter_terrain_move_up
    _reset_dofs = Go2PACT._reset_dofs
    _reset_root_states = Go2PACT._reset_root_states
    _resample_commands = Go2PACT._resample_commands

    def __init__(self, resolved, device='cuda:0', headless=True, *,
                 reward_mode=None, reward_fn=None):
        if reward_fn is None and reward_mode != 'smoke_tracking':
            raise ValueError("BR reward port pending: explicitly select smoke_tracking or supply reward_fn")
        if not isinstance(resolved,BeyondRobustnessResolvedConfig):
            raise TypeError('Use resolve_beyond_robustness() for the shared environment/model/storage contract')
        cfg=resolved.environment; sim_params=resolved.sim_params
        self.configuration_manifest=deepcopy(resolved.manifest)
        # Shared scripted reset/command samplers require a device string.
        self.cfg=deepcopy(cfg); self.device=str(torch.device(device)); self.num_envs=cfg.env.num_envs
        self.num_actions=12
        import random
        import numpy as np
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        self.cfg.env.num_actions=12; self.cfg.control.control_type='P'
        self.dt=float(cfg.control.dt); self.reward_fn=reward_fn
        if not math.isclose(self.dt,sim_params['dt']*cfg.control.decimation):
            raise ValueError('Physics/control timestep mismatch')
        self.input_schema=resolved.training.input_schema
        self.simulator=IsaacLabSimulatorBR(self.cfg,sim_params,str(device),headless,resolved.payload)
        s=self.simulator
        named_indices(s._robot.joint_names,self.input_schema.joint_order)
        named_indices(s._robot.body_names,[f'{leg}_foot' for leg in self.input_schema.foot_order])
        # Base class initializes gains in articulation order; convert ONCE for
        # canonical controller inputs. Scalar gains remain unchanged numerically.
        s._p_gains=s._p_gains[s._dof_indices]; s._d_gains=s._d_gains[s._dof_indices]
        self.domain_rand_curriculum=HardPACTDomainRandCurriculum(self.cfg,s.payload_cfg.seed)
        self._apply_domain_rand_curriculum_state()
        s.set_payload_progress(self.domain_rand_curriculum.progress['disturbance'])
        self.command_ranges={key:list(getattr(cfg.commands.ranges,key)) for key in ('lin_vel_x','lin_vel_y','ang_vel_yaw','heading')}
        self.commands=torch.zeros(self.num_envs,4,device=device)
        self.forward_vec=torch.tensor([1.,0.,0.],device=device).repeat(self.num_envs,1)
        self.previous_action=torch.zeros(self.num_envs,12,device=device)
        self.episode_steps=torch.zeros(self.num_envs,dtype=torch.long,device=device)
        self.episode_tracking=torch.zeros(self.num_envs,2,device=device)
        self.failure_steps=torch.zeros_like(self.episode_steps)
        self.actor_observation_cache=BRActorObservationCache(self.num_envs,device,cfg.noise,
            self.input_schema.actor_clip,cfg.seed)
        self.max_episode_length_s=float(cfg.env.episode_length_s)
        self.max_episode_length=math.ceil(self.max_episode_length_s/self.dt)
        max_delay=int(cfg.domain_rand.ctrl_delay_step_range[1]) if cfg.domain_rand.randomize_ctrl_delay else 0
        self.action_history=torch.zeros(self.num_envs,max_delay+1,12,device=device)
        self.action_delay=torch.zeros_like(self.episode_steps)
        self.parameters=torch.zeros(self.num_envs,self.input_schema.parameter_dim,device=device)
        self._persistent_wrench_target_world=torch.zeros(self.num_envs,6,device=device)
        self._current_sustained_wrench_world=torch.zeros_like(self._persistent_wrench_target_world)
        self._persistent_component_active=torch.zeros(self.num_envs,2,device=device,dtype=torch.bool)
        self._current_sustained_active_mask=torch.zeros(self.num_envs,1,device=device,dtype=torch.bool)
        for key in ('start_step','end_step','duration_steps','next_event_step'):
            setattr(self,'_persistent_'+key,torch.zeros(self.num_envs,2,device=device,dtype=torch.long))
        self.clock=0; self.init_done=False; self.iteration=0; self.last_completed_iteration=-1
        self.command_streak=0
        self.curriculum_metrics={}
        # Include the added fixed tray link; ground-filtered sensor forces omit
        # box support without disabling physical collisions.
        names=list(s._contact_sensors.body_names)
        self.collision_ids=sorted(set(s._termination_contact_indices+[names.index('br_tray')]))

    def _sample_commands(self, ids):
        self._resample_commands(ids)  # Shared three-component 0.2 deadband.
        self._refresh_heading()

    def _refresh_heading(self):
        if self.cfg.commands.heading_command:
            from legged_gym.utils.math_utils import quat_apply, wrap_to_pi
            forward=quat_apply(self.simulator.base_quat,self.forward_vec)
            bounds=(self.command_ranges['ang_vel_yaw'] if
                getattr(self.cfg.commands,'max_curriculum_ang_vel_yaw',None) is not None
                else self.cfg.commands.ranges.ang_vel_yaw)
            self.commands[:,2]=(0.5*wrap_to_pi(self.commands[:,3]-torch.atan2(forward[:,1],forward[:,0]))).clamp(*bounds)

    def _reset(self, ids):
        if not len(ids): return
        s=self.simulator
        if self.init_done and self.cfg.terrain.curriculum and hasattr(s,'_terrain'):
            self._update_terrain_curriculum(ids)
        self._reset_command_curriculum(ids)
        self._reset_dofs(ids)
        self._reset_root_states(ids)
        s.reset_idx(ids)
        s.reset_payload(ids)
        self.previous_action[ids]=0; self.action_history[ids]=0; self.episode_steps[ids]=0
        self.episode_tracking[ids]=0
        self.failure_steps[ids]=0
        # Delay is a reset property, using the same eligible reset cadence.
        cadence=s.payload_cadence
        cadence.update_ranges({'delay':tuple(self.cfg.domain_rand.ctrl_delay_step_range)})
        due=cadence.select('delay',ids)
        if self.cfg.domain_rand.randomize_ctrl_delay:
            lo,hi=self.cfg.domain_rand.ctrl_delay_step_range
            self.action_delay[due]=torch.randint(lo,hi+1,(len(due),),device=self.device,generator=s.payload_rng)
        self._persistent_component_active[ids]=False
        self._persistent_wrench_target_world[ids]=0; self._current_sustained_wrench_world[ids]=0
        for component in range(2): self._schedule_next_persistent_event(ids,component,self.clock)
        self._sample_commands(ids)
        self.parameters[ids]=self.input_schema.pack('parameters',s.realized_parameters())[ids]
        s._torques[ids]=0
        self._refresh_actor_observation(ids)

    def _reset_command_curriculum(self, ids):
        """Zero patience retains shared reset/global-step cadence, using raw tracking."""
        if (self.init_done and self.cfg.commands.curriculum and
                self.cfg.commands.curriculum_patience_iterations==0 and self.clock%self.max_episode_length==0 and
                self.episode_tracking[ids,0].mean()/self.max_episode_length>self.cfg.commands.curriculum_threshold):
            self._expand_command_curriculum_ranges()

    def reset(self):
        self._reset(torch.arange(self.num_envs,device=self.device))
        self.simulator._resume_samples=False
        self.init_done=True
        return self.observe()

    def _clean_proprioception(self):
        s=self.simulator; d=s._robot.data; schema=self.input_schema
        q=d.root_link_quat_w
        omega=inverse_rotate_wxyz(q,d.root_link_ang_vel_w)
        gravity=inverse_rotate_wxyz(q,s._global_gravity)
        return schema.pack('observations',dict(angular_velocity_B=omega,gravity_B=gravity,
            command_B=self.commands[:,:3],joint_position=s.dof_pos,joint_velocity=s.dof_vel,
            previous_action=self.previous_action))

    def _refresh_actor_observation(self, ids=None, *, redraw=True):
        self.actor_observation_cache.refresh(self._clean_proprioception(),ids,redraw=redraw)

    def observe(self):
        s=self.simulator; d=s._robot.data; schema=self.input_schema
        q=d.root_link_quat_w
        velocity=inverse_rotate_wxyz(q,d.root_link_lin_vel_w)
        clean=self._clean_proprioception()
        obs=self.actor_observation_cache.values.clone()
        force=s.hard_pact_foot_forces_world()
        force=inverse_rotate_wxyz(q[:,None,:].expand(-1,4,-1),force).reshape(self.num_envs,12)
        state=dict(proprioception=clean,linear_velocity_B=velocity,
            joint_torque=s.hard_pact_executed_torque(),joint_acceleration=d.joint_acc[:,s._dof_indices],
            foot_contact_force_B=force)
        if schema.terrain_points_x and schema.terrain_points_y:
            state['terrain_heights']=d.root_link_pos_w[:,2,None]-s._measured_heights
        state=schema.pack('state',state); labels=s.payload_labels()
        valid=torch.isfinite(torch.cat((obs,state,self.parameters,labels),-1)).all(-1)
        return BRObservation(obs,state,self.parameters,labels,valid)

    def step(self, actions):
        if actions.shape != self.previous_action.shape or not torch.isfinite(actions).all():
            raise ValueError('BR expects finite [N,12] position actions')
        s=self.simulator
        clipped=actions.clamp(-self.cfg.normalization.clip_actions,self.cfg.normalization.clip_actions)
        self.action_history[:,1:]=self.action_history[:,:-1].clone()
        self.action_history[:,0]=clipped
        delayed=self.action_history[torch.arange(self.num_envs,device=self.device),self.action_delay]
        self._update_persistent_wrench(self.clock)
        s.hard_pact_apply_base_wrench_world(self._current_sustained_wrench_world)
        s.step(delayed); s.post_physics_step()
        self.previous_action.copy_(clipped); self.episode_steps+=1; self.clock+=1
        # Shared post-physics callback ordering: commands/heading and pushes
        # precede reward/termination/observation construction, then resets.
        due=self.episode_steps.remainder(max(1,int(self.cfg.commands.resampling_time/self.dt)))==0
        self._sample_commands(due.nonzero().flatten())
        if self.cfg.domain_rand.push_robots: s.push_robots()
        self._refresh_actor_observation()
        packet=self.observe()
        linear=Go2PACT._reward_tracking_lin_vel(self)  # Same raw, unrelaxed curriculum evidence.
        angular=Go2PACT._reward_tracking_ang_vel(self)
        self.episode_tracking+=torch.stack((linear,angular),-1)
        lost=lost_payload(packet.load_labels[:,:3],s.payload_cfg)
        collision=(s.ground_total_forces_world[:,self.collision_ids].norm(dim=-1)>10.).any(-1)
        term=self.cfg.termination
        from legged_gym.utils.math_utils import wrap_to_pi
        r,p=wrap_to_pi(s._base_euler[:,0]),wrap_to_pi(s._base_euler[:,1])
        tilt=torch.zeros_like(collision)
        if 'roll' in term.termination_terms: tilt |= r.abs()>term.roll_threshold
        if 'pitch' in term.termination_terms: tilt |= p.abs()>term.pitch_threshold
        height=(s.base_pos[:,2,None]-s._measured_heights).mean(-1)
        height_fail=torch.zeros_like(collision)
        if 'height_min' in term.termination_terms: height_fail |= height<term.height_min
        if 'height_max' in term.termination_terms: height_fail |= height>term.height_max
        robot_failure=collision|tilt|height_fail
        self.failure_steps.copy_(torch.where(robot_failure,self.failure_steps+1,0))
        failure=lost|~packet.supervision_valid|(self.failure_steps>self.cfg.env.fail_to_terminal_time_s/self.dt)
        done,timeout=termination_masks(failure,self.episode_steps,self.max_episode_length)
        self.last_failure_reasons=dict(payload_lost=lost.clone(),ground_collision=collision.clone(),tilt=tilt.clone())
        reward=linear+.5*angular if self.reward_fn is None else self.reward_fn(self,packet)
        terminal=packet.clone()  # BEFORE any reset/curriculum mutation
        ids=done.nonzero().flatten()
        performance=None
        if len(ids):
            mean=(self.episode_tracking[ids]/self.episode_steps[ids,None]).mean(0)
            performance=dict(lin_vel_tracking=float(mean[0]),ang_vel_tracking=float(mean[1]),
                terrain_level=float(s.terrain_levels[ids].float().mean()))
        self._reset(ids)
        return BRStep(self.observe(),reward,done,timeout,terminal,done.clone(),
            performance_metrics=performance,curriculum_tracking=linear)

    def begin_iteration(self, iteration):
        self.iteration=int(iteration)
        self._terrain_curriculum_iteration=self.iteration

    def end_iteration(self, iteration, tracking, count):
        if iteration<=self.last_completed_iteration: return
        consecutive=iteration==self.last_completed_iteration+1
        self.last_completed_iteration=int(iteration)
        valid=count>0 and tracking is not None and math.isfinite(tracking)
        c=self.cfg.commands
        if c.curriculum and c.curriculum_patience_iterations>0:
            self.command_streak=(self.command_streak if consecutive else 0)+1 if valid and tracking>c.curriculum_threshold else 0
            if self.command_streak>=max(1,c.curriculum_patience_iterations):
                self._expand_command_curriculum_ranges(); self.command_streak=0
        if valid and self.simulator.use_domainrand_curriculum:
            self.domain_rand_curriculum.advance(iteration,tracking)
            self._apply_domain_rand_curriculum_state()
            self.simulator.set_payload_progress(self.domain_rand_curriculum.progress['disturbance'])
        self.curriculum_metrics=dict(tracking=tracking if valid else float('nan'),valid_count=count,
            command_streak=self.command_streak,**self.domain_rand_curriculum.progress)

    def iteration_metrics(self):
        return {'curriculum/'+key:value for key,value in self.curriculum_metrics.items()}

    def curriculum_state_dict(self):
        s=self.simulator
        return deepcopy(dict(version=2,domain=self.domain_rand_curriculum.state_dict(),
            command_ranges=self.command_ranges,command_streak=self.command_streak,
            iteration=self.iteration,last_completed_iteration=self.last_completed_iteration,
            sampling=s.sampling_state_dict(),action_delay=self.action_delay.clone(),
            noise=self.actor_observation_cache.state_dict(),
            physical_config=self.resolved_configuration(),
            terrain_levels=s.terrain_levels.clone(),terrain_types=s.terrain_types.clone()))

    def load_curriculum_state_dict(self, state):
        if state['version']!=2: raise ValueError('Incompatible BR physical curriculum state; Stage-3 scene-wide geometry/no-noise resume requires explicit conversion')
        s=self.simulator
        self.domain_rand_curriculum.load_state_dict(state['domain'])
        self.command_ranges=deepcopy(state['command_ranges']); self.command_streak=state['command_streak']
        self.iteration=state['iteration']; self.last_completed_iteration=state['last_completed_iteration']
        self._apply_domain_rand_curriculum_state()
        s.set_payload_progress(self.domain_rand_curriculum.progress['disturbance'])
        if state['physical_config']!=self.resolved_configuration():
            raise ValueError('BR physical configuration differs from checkpoint')
        s.load_sampling_state_dict(state['sampling'])
        self.action_delay.copy_(state['action_delay'])
        self.actor_observation_cache.load_state_dict(state['noise'])
        self._terrain_curriculum_iteration=self.iteration
        self.init_done=False  # Fresh resume is not a completed episode/curriculum event.
        # Shared heading-mode sampler deadbands using the previous yaw command.
        # A fresh scene has zero commands; do not inherit the discarded episode.
        self.commands.zero_()
        self.clock=0; s._push_call_counter=0
        if hasattr(s,'_terrain'):
            s._terrain_levels.copy_(state['terrain_levels']); s._terrain_types.copy_(state['terrain_types'])
            s._env_origins.copy_(s._terrain_origins[s._terrain_levels,s._terrain_types])

    def resolved_configuration(self):
        from dataclasses import asdict
        from legged_gym.utils.helpers import class_to_dict
        return dict(task=class_to_dict(self.cfg),payload=asdict(self.simulator.payload_cfg),
            observation_units=self.input_schema.contract(),manifest=self.configuration_manifest,
            reward='external_callable' if self.reward_fn else 'temporary_smoke_tracking',
            dimensions_lifetime='seeded per-environment size bank, creation-time only',
            geometry=dict(size_bank_m=self.simulator.box_size_bank,
                assignment='canonical env_id modulo size_bank_count'),
            friction='multiply: fixed tray mu times sampled box mu; no friction estimator label',
            curriculum_order='terrain every reset with absolute-iteration promotion delay; completed-update commands then domain disturbance; changed payload ranges at next eligible reset',
            exact_physics_resume=False)

    def close(self):
        self.simulator._app_launcher.app.close()
