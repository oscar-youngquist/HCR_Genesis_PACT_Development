"""MULE rolling-payload adapter with independent reward streams and shared curricula.

Production rewards are default; explicit test callbacks return already
weighted/dt-scaled streams. BR reuse is physical infrastructure and generic
noise/shared-reward metadata, NOT BR native rewards, observations or learning.
"""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace as NS
import math
import random
import numpy as np
import torch
from .go2_mule import Go2MULE, MULEObservation, MULEStep
from .mule_schema import MULEExecution, estimate_foot_forces
from .go2_mule_config import ResolvedGo2MULE
from .rolling_payload import lost_balls
from .mule_rewards import MULERewards
from rsl_rl.beyond_robustness_sensors import BRActorObservationCache
from rsl_rl.beyond_robustness_payload import inverse_rotate_wxyz
from legged_gym.simulator.isaaclab_simulator_mule import IsaacLabSimulatorMULE
from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
from legged_gym.envs.go2.go2_hard_pact.domain_rand_curriculum import HardPACTDomainRandCurriculum
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT


class IsaacLabMULEEnv(Go2MULE):
    supports_exact_resume=False  # No restoration of PhysX contact/solver state.
    _reset_dofs=Go2PACT._reset_dofs
    _reset_root_states=Go2PACT._reset_root_states
    _apply_domain_rand_curriculum_state=Go2HardPACT._apply_domain_rand_curriculum_state

    def __init__(self,resolved,device='cuda:0',headless=True,*,reward_fn=None,reward_contract=None):
        if not isinstance(resolved,ResolvedGo2MULE): raise TypeError('Use resolve_go2_mule()')
        if reward_fn is not None and (not callable(reward_fn) or not isinstance(reward_contract,str) or not reward_contract):
            raise ValueError('Explicit test MULE reward requires callable and versioned reward_contract')
        if reward_fn is None and reward_contract is not None: raise ValueError('Reward contract without callback')
        super().__init__(resolved.environment)
        self.resolved=deepcopy(resolved); self.shared=deepcopy(resolved.shared)
        self.device=str(torch.device(device)); self.num_actions=12; self.dt=self.cfg.control_dt
        self.reward_fn=reward_fn; self.reward_contract=reward_contract or deepcopy(self.shared.mule_reward)
        self.rewards=None if reward_fn is not None else MULERewards(self.shared.mule_reward,self.shared,self.num_envs,device)
        random.seed(resolved.training.seed); np.random.seed(resolved.training.seed); torch.manual_seed(resolved.training.seed)
        self.command_rng=torch.Generator(device=device).manual_seed(resolved.training.seed+739)
        self.simulator=IsaacLabSimulatorMULE(self.shared,resolved.sim_params,self.device,headless,resolved.payload)
        s=self.simulator; n=self.num_envs
        self.domain_rand_curriculum=HardPACTDomainRandCurriculum(self.shared,resolved.training.seed)
        self._apply_domain_rand_curriculum_state()
        self._publish_payload_progress()
        self.command_ranges={k:list(getattr(self.shared.commands.ranges,k)) for k in ('lin_vel_x','lin_vel_y','ang_vel_yaw','heading')}
        self.commands=torch.zeros(n,4,device=device)
        self.previous_action=torch.zeros(n,12,device=device)
        self.episode_id=torch.full((n,),-1,device=device,dtype=torch.long)
        self.episode_steps=torch.zeros_like(self.episode_id); self.failure_steps=torch.zeros_like(self.episode_id)
        self.max_episode_length_s=float(self.shared.env.episode_length_s)
        self.max_episode_length=math.ceil(self.max_episode_length_s/self.dt)
        dr=self.shared.domain_rand
        delay=int(dr.ctrl_delay_step_range[1]) if dr.randomize_ctrl_delay else 0
        self.action_delay=torch.zeros_like(self.episode_id)
        self.target_queue=torch.tensor(self.cfg.standing_position,device=device).expand(n,delay+1,12).clone()
        self.source_queue=torch.full((n,delay+1),-1,dtype=torch.long,device=device)
        self.actor_observation_cache=BRActorObservationCache(n,device,self.shared.noise,float('inf'),resolved.training.seed)
        # Cache utility's physical noise is scaled exactly like MULE's pack().
        sc=self.input_schema
        for sl,scale in ((slice(0,3),sc.angular_velocity_scale),(slice(9,21),sc.joint_position_scale),(slice(21,33),sc.joint_velocity_scale)):
            self.actor_observation_cache.amplitude[sl]*=scale
        names=list(s._contact_sensors.body_names)
        self.collision_ids=sorted(set(s._termination_contact_indices+[names.index('br_tray')]))
        self.clock=0; self.iteration=0; self._packet=None; self._robot_parameters=None
        self.last_completed_iteration=-1; self.command_streak=0; self.curriculum_metrics={}
        self.episode_tracking=torch.zeros(n,device=device)
        self.reward_collision_ids=sorted(set(s._penalized_contact_indices+[names.index('br_tray')]))

    def _sample_commands(self,ids):
        keys=('lin_vel_x','lin_vel_y','heading' if self.shared.commands.heading_command else 'ang_vel_yaw')
        for column,key in zip((0,1,3 if self.shared.commands.heading_command else 2),keys):
            lo,hi=self.command_ranges[key]
            self.commands[ids,column]=lo+(hi-lo)*torch.rand(len(ids),device=self.device,generator=self.command_rng)
        self.commands[ids,:3]*=(self.commands[ids,:3].norm(dim=-1)>.2)[:,None]

    def _heading(self,ids=None):
        if not self.shared.commands.heading_command: return
        from legged_gym.utils.math_utils import quat_apply,wrap_to_pi
        ids=slice(None) if ids is None else ids
        q=self.simulator.base_quat[ids]
        forward=quat_apply(q,q.new_tensor([1.,0.,0.]).expand(len(q),3))
        self.commands[ids,2]=(.5*wrap_to_pi(self.commands[ids,3]-torch.atan2(forward[:,1],forward[:,0]))).clamp(*self.command_ranges['ang_vel_yaw'])

    def _refresh(self,ids=None):
        s=self.simulator; d=s._robot.data; schema=self.input_schema
        q=d.root_link_quat_w
        physical=dict(angular_velocity_B=inverse_rotate_wxyz(q,d.root_link_ang_vel_w),
            gravity_B=inverse_rotate_wxyz(q,s._global_gravity),command_B=self.commands[:,:3],
            joint_position=s.dof_pos,joint_velocity=s.dof_vel,previous_action=self.previous_action)
        clean=schema.pack('actor',physical)
        self.actor_observation_cache.refresh(clean,ids)
        velocity=inverse_rotate_wxyz(q,d.root_link_lin_vel_w)
        critic={**physical,'linear_velocity_B':velocity,
            'disturbance_force_B':inverse_rotate_wxyz(q,s.payload_reaction_world)}
        if schema.terrain_dim: critic['terrain_heights']=d.root_link_pos_w[:,2,None]-s._measured_heights
        critic=schema.pack('critic',critic)
        force=estimate_foot_forces(s.hard_pact_executed_torque(),s.leg_jacobians_b())
        valid=torch.isfinite(torch.cat((clean,critic,velocity),-1)).all(-1)
        packet=MULEObservation(self.actor_observation_cache.values.clone(),critic,velocity,force,
            self.episode_id.clone(),valid,self.clock).snapshot()
        if ids is not None and self._packet is not None:
            old=self._packet
            for name in ('actor','critic','velocity_b','episode_id','supervision_valid'):
                getattr(old,name)[ids]=getattr(packet,name)[ids]
            old.foot_force_estimate.force_b[ids]=packet.foot_force_estimate.force_b[ids]
            old.foot_force_estimate.valid[ids]=packet.foot_force_estimate.valid[ids]
        else: self._packet=packet

    def observe(self):
        if self._packet is None: raise RuntimeError('Reset before observing')
        return self._packet.snapshot()  # No sensors, history, physics, RNG, or curricula.

    @torch.no_grad()
    def reset(self,env_ids=None,*,completed=False):
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids.to(self.device)
        if not len(ids): return self.observe()
        s=self.simulator; resumed=getattr(s,'_resume_samples',False)
        if completed and not resumed:
            context=NS(cfg=self.shared,simulator=s,commands=self.commands,init_done=True,
                max_episode_length_s=self.max_episode_length_s,
                _filter_terrain_move_up=lambda mask:mask & (self.iteration>=self.shared.terrain.curriculum_upward_delay_iterations))
            if self.shared.terrain.curriculum and hasattr(s,'_terrain'): Go2PACT._update_terrain_curriculum(context,ids)
            c=self.shared.commands
            if (c.curriculum and c.curriculum_patience_iterations==0 and self.clock%self.max_episode_length==0
                    and self.episode_tracking[ids].mean()/self.max_episode_length>c.curriculum_threshold):
                self._expand_commands()
        self._reset_dofs(ids); self._reset_root_states(ids)
        s.reset_idx(ids); s.reset_payload(ids)
        realized={k:v.detach().clone() for k,v in s.realized_parameters().items()}
        realized['robot_including_tray_mass']=s._robot.root_physx_view.get_masses().to(self.device).sum(-1,keepdim=True)
        if self._robot_parameters is None: self._robot_parameters=realized
        else:
            for key,value in realized.items(): self._robot_parameters[key][ids]=value[ids]
        self.episode_id[ids]+=1; self.episode_steps[ids]=0; self.failure_steps[ids]=0
        self.episode_tracking[ids]=0
        self.previous_action[ids]=0; self.target_queue[ids]=self.target_queue.new_tensor(self.cfg.standing_position)
        self.source_queue[ids]=-1; s._torques[ids]=0
        if hasattr(s,'_hard_pact_executed_torque'): s._hard_pact_executed_torque[ids]=0
        if not resumed and self.shared.domain_rand.randomize_ctrl_delay:
            due=s._reset_sample_ids('mule_delay',ids) if hasattr(s,'_reset_sample_ids') else ids
            lo,hi=self.shared.domain_rand.ctrl_delay_step_range
            self.action_delay[due]=torch.randint(lo,hi+1,(len(due),),device=self.device,generator=self.command_rng)
        self._sample_commands(ids); self._heading(ids); self._refresh(ids)
        if self.rewards is not None: self.rewards.reset(ids,self._reward_snapshot())
        s._resume_samples=False
        return self.observe()

    @torch.no_grad()
    def step(self,command):
        target=command.position_target
        expected=target.new_tensor(self.cfg.standing_position)+self.cfg.action_scale*command.combined_clipped
        if target.shape!=(self.num_envs,12) or not torch.isfinite(target).all() or not torch.allclose(target,expected):
            raise ValueError('Expected finite, once-scaled absolute MULE command')
        self.target_queue[:,1:]=self.target_queue[:,:-1].clone(); self.target_queue[:,0]=target
        self.source_queue[:,1:]=self.source_queue[:,:-1].clone(); self.source_queue[:,0]=self.clock
        rows=torch.arange(self.num_envs,device=self.device)
        selected=self.target_queue[rows,self.action_delay].clone(); source=self.source_queue[rows,self.action_delay].clone()
        s=self.simulator
        s.step_position_targets(selected); s.post_physics_step()
        execution=MULEExecution(selected,source,s.hard_pact_executed_torque().clone())
        self.previous_action.copy_(command.combined_clipped); self.episode_steps+=1; self.clock+=1
        self._sample_commands((self.episode_steps%max(1,int(self.shared.commands.resampling_time/self.dt))==0).nonzero().flatten())
        if self.shared.domain_rand.push_robots: s.push_robots()
        refresh_after_push(s)
        self._heading(); self._refresh()
        packet=self.observe(); relative,_=s.payload_motion()
        lost=(lost_balls(relative,s.ball_radius,s.payload_cfg)&s.payload_active) if s.balls else torch.zeros_like(self.episode_steps,dtype=torch.bool)
        collision=(s.ground_total_forces_world[:,self.collision_ids].norm(dim=-1)>10).any(-1)
        from legged_gym.utils.math_utils import wrap_to_pi
        term=self.shared.termination
        roll=wrap_to_pi(s._base_euler[:,0]).abs(); pitch=wrap_to_pi(s._base_euler[:,1]).abs()
        height=(s.base_pos[:,2,None]-s._measured_heights).mean(-1)
        false=torch.zeros_like(lost)
        reasons=dict(payload_lost=lost,contact=collision,nonfinite=~packet.supervision_valid,
            roll=roll>term.roll_threshold if 'roll' in term.termination_terms else false,
            pitch=pitch>term.pitch_threshold if 'pitch' in term.termination_terms else false,
            height_min=height<term.height_min if 'height_min' in term.termination_terms else false,
            height_max=height>term.height_max if 'height_max' in term.termination_terms else false)
        robot=collision|reasons['roll']|reasons['pitch']|reasons['height_min']|reasons['height_max']
        self.failure_steps.copy_(torch.where(robot,self.failure_steps+1,0))
        failure=lost|~packet.supervision_valid|(self.failure_steps>self.shared.env.fail_to_terminal_time_s/self.dt)
        timeout=(self.episode_steps>=self.max_episode_length)&~failure
        done=failure|timeout
        # Callable sees the same pre-reset endpoint as labels and termination.
        reward_terms=None
        if self.rewards is None:
            nominal,adaptive=self.reward_fn(self,packet,failure,timeout)
        else:
            nominal,adaptive=self.rewards.evaluate(self._reward_snapshot(),command,selected,self.commands,
                packet.foot_force_estimate,self._robot_parameters['robot_including_tray_mass'],
                s.ball_mass*s.payload_active[:,None],done,timeout,packet.supervision_valid)
            reward_terms=self.rewards.last_weighted
        # Same unscaled fixed-kernel locomotion evidence as HardPACT's gate;
        # all valid env/control samples, never episode totals or adaptive reward.
        linear=torch.exp(-4*(self.commands[:,:2]-s.base_lin_vel[:,:2]).square().sum(-1))
        angular=torch.exp(-4*(self.commands[:,2]-s.base_ang_vel[:,2]).square())
        valid=packet.supervision_valid&torch.isfinite(linear)&torch.isfinite(angular)
        self.episode_tracking+=torch.where(valid,linear,0.)
        count=int(valid.sum())
        performance=None if count==0 else dict(lin_vel_tracking=float(linear[valid].mean()),
            ang_vel_tracking=float(angular[valid].mean()),terrain_level=float(s.terrain_levels[valid].float().mean()),valid_count=count)
        for r in (nominal,adaptive):
            if r.shape!=(self.num_envs,) or not torch.isfinite(r).all(): raise ValueError('Invalid explicit reward stream')
        result=MULEStep(packet,execution,nominal.clone(),adaptive.clone(),failure.clone(),timeout.clone(),
            packet.snapshot(),done&packet.supervision_valid,failure_reasons={k:v.clone() for k,v in reasons.items()},
            performance_metrics=performance,reward_terms=reward_terms)
        self.reset(done.nonzero().flatten(),completed=True)
        return MULEStep(self.observe(),result.execution,result.nominal_reward,result.adaptive_reward,
            result.terminated,result.timed_out,result.terminal,result.terminal_valid,failure_reasons=result.failure_reasons,
            performance_metrics=result.performance_metrics,reward_terms=result.reward_terms)

    def set_payload_severity(self,value): self.simulator.set_payload_severity(value)
    def _publish_payload_progress(self):
        s=self.simulator
        if self.shared.mule.follow_disturbance_curriculum:
            s.set_payload_severity(self.domain_rand_curriculum.progress['disturbance'])
        if s.payload_cfg.mode=='nominal' or s.payload_progress==0:
            s.push_value*=self.shared.mule.nominal_push_scale
            s.vert_value*=self.shared.mule.nominal_push_scale
            s.angular_push_value*=self.shared.mule.nominal_push_scale

    def requested_optimization_phase(self):
        if self.shared.mule.auto_joint_adaptation and self.simulator.balls and self.simulator.payload_progress>0:
            return 'joint_adaptation'
        return None # Never reverse a trained phase; explicit user phase remains independent.

    def begin_iteration(self,iteration):
        self.iteration=int(iteration)
        if self.rewards is not None: self.rewards.begin_iteration(iteration)

    def _expand_commands(self):
        context=NS(cfg=self.shared,command_ranges=self.command_ranges)
        Go2HardPACT._expand_command_curriculum_ranges(context)

    def end_iteration(self,iteration,performance):
        if iteration<=self.last_completed_iteration: return
        consecutive=iteration==self.last_completed_iteration+1
        self.last_completed_iteration=int(iteration)
        count=0 if performance is None else performance.get('valid_count',0)
        tracking=None if performance is None else performance.get('lin_vel_tracking')
        valid=count>0 and tracking is not None and math.isfinite(tracking)
        c=self.shared.commands
        if c.curriculum and c.curriculum_patience_iterations>0:
            self.command_streak=(self.command_streak if consecutive else 0)+1 if valid and tracking>c.curriculum_threshold else 0
            if self.command_streak>=c.curriculum_patience_iterations:
                self._expand_commands(); self.command_streak=0
        if valid and self.simulator.use_domainrand_curriculum:
            self.domain_rand_curriculum.advance(iteration,tracking)
            self._apply_domain_rand_curriculum_state(); self._publish_payload_progress()
        self.curriculum_metrics={'curriculum/commands/valid_count':count,
            'curriculum/commands/patience_count':self.command_streak}
        if valid: self.curriculum_metrics['curriculum/commands/tracking_mean']=tracking

    def _reward_snapshot(self):
        s=self.simulator
        fields=('base_pos','base_quat','projected_gravity','dof_pos','dof_vel','default_dof_pos',
            'dof_pos_limits','dof_pos_limits_hard','torque_limits','feet_pos','feet_vel','measured_heights',
            '_height_around_feet','_max_height_ahead_feet','_normal_vector_around_feet','base_lin_vel','base_ang_vel')
        values={k:getattr(s,k).detach().clone() for k in fields}
        values.update(torques=s.hard_pact_executed_torque().clone(),requested_torque=s.last_requested_torque.clone(),
            link_contact_forces=s.ground_total_forces_world.clone(),feet_contact_indices=s.feet_contact_indices,
            feet_indices=s.feet_indices,penalised_contact_indices=self.reward_collision_ids)
        if self.rewards is not None and any(self.rewards.coefficients()[k]['nominal']!=0 for k in
                ('feet_near_edge','edge_swing_clearance','swing_foot_collision_edge')):
            values['near_edge']=s.calc_feet_near_edge().clone()
        return NS(**values)

    def physical_labels(self):
        """Owned privileged physical evidence for future rewards, NEVER actor inputs.

        Ball COM is its link origin (centered sphere). Robot reference is torso
        link origin, whose velocity includes the offset from randomized COM.
        Applied contact reaction includes force, not an inferred mass*g wrench.
        """
        s=self.simulator; relative,velocity=s.payload_motion()
        return dict(realized_robot={k:v.clone() for k,v in self._robot_parameters.items()},ball_mass_kg=s.ball_mass.clone(),
            total_ball_mass_kg=(s.ball_mass*s.payload_active[:,None]).sum(-1),ball_radius_m=s.ball_radius.clone(),
            payload_active=s.payload_active.clone(),realized_severity=s.realized_severity.clone(),
            ball_relative_position_b=relative.clone(),ball_relative_velocity_b=velocity.clone(),
            payload_reaction_world_n=s.payload_reaction_world.clone(),
            ground_contact_force_world_n=s.ground_total_forces_world.clone(),
            requested_torque_nm=s.last_requested_torque.clone(),executed_torque_nm=s.hard_pact_executed_torque().clone())

    def physical_contract(self):
        from rsl_rl.beyond_robustness_config import settings_dict
        return dict(version=2,shared=settings_dict(self.resolved.shared),payload=asdict(self.resolved.payload),
            schema=self.input_schema.contract(),reward=self.reward_contract,
            critic_disturbance='measured ball-on-robot contact force, normal+tangent, body axes',
            force_proxy='last applied joint torque and endpoint named foot COM Jacobian, body axes')

    def curriculum_state_dict(self):
        s=self.simulator
        return deepcopy(dict(version=2,contract=self.physical_contract(),sampling=s.sampling_state_dict(),
            domain=self.domain_rand_curriculum.state_dict(),command_ranges=self.command_ranges,
            last_completed_iteration=self.last_completed_iteration,command_streak=self.command_streak,
            rewards=None if self.rewards is None else self.rewards.state_dict(),
            command_rng=self.command_rng.get_state(),noise=self.actor_observation_cache.state_dict(),
            action_delay=self.action_delay.clone(),iteration=self.iteration,
            terrain_levels=s.terrain_levels.clone(),terrain_types=s.terrain_types.clone()))

    def validate_curriculum_state_dict(self,state):
        if state['version']!=2 or state['contract']!=self.physical_contract(): raise ValueError('Incompatible MULE physical/reward contract; Stage 3 requires explicit migration')
        self.simulator.validate_sampling_state_dict(state['sampling'])
        candidate=deepcopy(self.domain_rand_curriculum); candidate.load_state_dict(state['domain'])
        if self.rewards is not None: deepcopy(self.rewards).load_state_dict(state['rewards'])
        if state['last_completed_iteration']< -1 or state['command_streak']<0: raise ValueError('Invalid curriculum counters')
        if state['command_ranges'].keys()!=self.command_ranges.keys() or any(
                len(v)!=2 or not all(math.isfinite(x) for x in v) or v[0]>v[1] for v in state['command_ranges'].values()):
            raise ValueError('Invalid restored command ranges')
        for key,generator in (('command_rng',self.command_rng),):
            torch.Generator(device=generator.device).set_state(state[key].cpu())
        torch.Generator(device=self.device).set_state(state['noise']['rng'].cpu())
        delay=state['action_delay']
        if delay.shape!=self.action_delay.shape or delay.dtype!=torch.long or (delay<0).any() or (delay>=self.target_queue.shape[1]).any():
            raise ValueError('Invalid restored delay')
        s=self.simulator
        for name,bound in (('terrain_levels',self.shared.terrain.num_rows),('terrain_types',self.shared.terrain.num_cols)):
            x=state[name]
            if x.shape!=getattr(s,name).shape or x.dtype!=torch.long or (x<0).any() or (x>=bound).any(): raise ValueError('Invalid terrain identity')

    def load_curriculum_state_dict(self,state):
        self.validate_curriculum_state_dict(state)
        s=self.simulator
        self.domain_rand_curriculum.load_state_dict(state['domain']); self._apply_domain_rand_curriculum_state()
        s.load_sampling_state_dict(state['sampling'])
        self._publish_payload_progress()
        self.last_completed_iteration=state['last_completed_iteration']; self.command_streak=state['command_streak']
        if self.rewards is not None: self.rewards.load_state_dict(state['rewards'])
        self.command_ranges=deepcopy(state['command_ranges']); self.command_rng.set_state(state['command_rng'].cpu())
        self.actor_observation_cache.load_state_dict(state['noise']); self.action_delay.copy_(state['action_delay'])
        self.iteration=state['iteration']; self.commands.zero_(); self._packet=None; self.clock=0
        self.episode_id.fill_(-1); s._push_call_counter=0
        if hasattr(s,'_terrain'):
            s._terrain_levels.copy_(state['terrain_levels']); s._terrain_types.copy_(state['terrain_types'])
            s._env_origins.copy_(s._terrain_origins[s._terrain_levels,s._terrain_types])

    def iteration_metrics(self):
        s=self.simulator
        ranges={}
        for name,values in (('reset_mass_kg',s.payload_cfg.mass_range(s.payload_progress)),
                            ('periodic_mass_kg',s.payload_cfg.mass_range(s.payload_progress,periodic=True)),
                            ('friction',s.payload_cfg.friction_at(s.payload_progress))):
            ranges.update({f'Payload/{name}_{side}':value for side,value in zip(('low','high'),values)})
        if self.rewards is not None:
            rows,valid,force_valid,nonfinite=self.rewards.counts.tolist()
            ranges.update({'Train/reward_sample_count':rows,'Train/reward_valid_count':valid,
                'Train/reward_nonfinite_count':nonfinite,'Payload/grf_proxy_valid_count':force_valid})
        if self.domain_rand_curriculum.reward_ema is not None:
            ranges['Values/domain_rand_reward_ema']=self.domain_rand_curriculum.reward_ema
        ranges['Values/required_reward']=self.domain_rand_curriculum.required_reward
        return {**self.curriculum_metrics,**ranges,
            **{f'Values/domain_rand_{k}_progress':v for k,v in self.domain_rand_curriculum.progress.items()},
            'Payload/requested_severity':s.payload_progress,
            'Train/environment_stage_requested':1+int(bool(s.balls) and s.payload_progress>0),
            'Payload/spawn_jitter_bound_m':s.payload_cfg.spawn_jitter_m*s.payload_progress,
            'Payload/relative_linear_velocity_bound_m_s':s.payload_cfg.relative_linear_velocity_max*s.payload_progress,
            'Payload/relative_angular_velocity_bound_rad_s':s.payload_cfg.relative_angular_velocity_max*s.payload_progress,
            'Payload/active_fraction':float(s.payload_active.float().mean()),
            'Payload/realized_severity':float(s.realized_severity.mean()),
            'Payload/total_mass_mean_kg':float((s.ball_mass*s.payload_active[:,None]).sum(-1).mean()),
            'Payload/ball_mass_max_kg':float(s.ball_mass.max()),'Train/environment_rolling':int(bool(s.balls)),
            'Episode/terrain_level':float(s.terrain_levels.float().mean())}

    def validate_stage_transfer(self,state):
        raise ValueError('MULE liquid/stage transfer adapter not implemented; fresh same-contract resume only')

    def close(self): self.simulator._app_launcher.app.close()


def refresh_after_push(s):
    """Refresh root and foot twists after a velocity write, without another step."""
    d=s._robot.data
    delta_v=d.root_link_lin_vel_w-s._base_world_lin_vel
    delta_w=d.root_link_ang_vel_w-s._base_world_ang_vel
    s._feet_vel.add_(delta_v[:,None]+torch.cross(delta_w[:,None].expand(-1,4,-1),
        s.feet_pos-d.root_link_pos_w[:,None],dim=-1))
    s._base_lin_vel.copy_(inverse_rotate_wxyz(d.root_link_quat_w,d.root_link_lin_vel_w))
    s._base_ang_vel.copy_(inverse_rotate_wxyz(d.root_link_quat_w,d.root_link_ang_vel_w))
    s._base_world_lin_vel.copy_(d.root_link_lin_vel_w); s._base_world_ang_vel.copy_(d.root_link_ang_vel_w)
