"""MULE rolling-payload adapter. Production rewards and stage orchestration deferred.

An explicit reward callable returns two already weighted/dt-scaled streams.
The only BR reuse is physical infrastructure and the generic 45-channel noise
cache, NOT its observations, reward, model or optimization phase machinery.
"""
from copy import deepcopy
from dataclasses import asdict
import math
import random
import numpy as np
import torch
from .go2_mule import Go2MULE, MULEObservation, MULEStep
from .mule_schema import MULEExecution, estimate_foot_forces
from .go2_mule_config import ResolvedGo2MULE
from .rolling_payload import lost_balls
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
        if not callable(reward_fn) or not isinstance(reward_contract,str) or not reward_contract:
            raise ValueError('Explicit MULE reward callable and versioned reward_contract required; production rewards deferred')
        super().__init__(resolved.environment)
        self.resolved=deepcopy(resolved); self.shared=deepcopy(resolved.shared)
        self.device=str(torch.device(device)); self.num_actions=12; self.dt=self.cfg.control_dt
        self.reward_fn=reward_fn; self.reward_contract=reward_contract
        random.seed(resolved.training.seed); np.random.seed(resolved.training.seed); torch.manual_seed(resolved.training.seed)
        self.command_rng=torch.Generator(device=device).manual_seed(resolved.training.seed+739)
        self.simulator=IsaacLabSimulatorMULE(self.shared,resolved.sim_params,self.device,headless,resolved.payload)
        s=self.simulator; n=self.num_envs
        self.domain_rand_curriculum=HardPACTDomainRandCurriculum(self.shared,resolved.training.seed)
        self._apply_domain_rand_curriculum_state()
        # Nominal training has no movable payload and very small pushes. This
        # is an explicit physical-mode setting, never an optimization-phase switch.
        if resolved.payload.mode=='nominal':
            s.push_value*=.01; s.vert_value*=.01; s.angular_push_value*=.01
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
    def reset(self,env_ids=None):
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids.to(self.device)
        if not len(ids): return self.observe()
        s=self.simulator; resumed=getattr(s,'_resume_samples',False)
        self._reset_dofs(ids); self._reset_root_states(ids)
        s.reset_idx(ids); s.reset_payload(ids)
        realized={k:v.detach().clone() for k,v in s.realized_parameters().items()}
        realized['robot_including_tray_mass']=s._robot.root_physx_view.get_masses().to(self.device).sum(-1,keepdim=True)
        if self._robot_parameters is None: self._robot_parameters=realized
        else:
            for key,value in realized.items(): self._robot_parameters[key][ids]=value[ids]
        self.episode_id[ids]+=1; self.episode_steps[ids]=0; self.failure_steps[ids]=0
        self.previous_action[ids]=0; self.target_queue[ids]=self.target_queue.new_tensor(self.cfg.standing_position)
        self.source_queue[ids]=-1; s._torques[ids]=0
        if hasattr(s,'_hard_pact_executed_torque'): s._hard_pact_executed_torque[ids]=0
        if not resumed and self.shared.domain_rand.randomize_ctrl_delay:
            due=s._reset_sample_ids('mule_delay',ids) if hasattr(s,'_reset_sample_ids') else ids
            lo,hi=self.shared.domain_rand.ctrl_delay_step_range
            self.action_delay[due]=torch.randint(lo,hi+1,(len(due),),device=self.device,generator=self.command_rng)
        self._sample_commands(ids); self._heading(ids); self._refresh(ids)
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
        self._heading(); self._refresh()
        packet=self.observe(); relative,_=s.payload_motion()
        lost=lost_balls(relative,s.ball_radius,s.payload_cfg) if s.balls else torch.zeros_like(self.episode_steps,dtype=torch.bool)
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
        nominal,adaptive=self.reward_fn(self,packet,failure,timeout)
        for r in (nominal,adaptive):
            if r.shape!=(self.num_envs,) or not torch.isfinite(r).all(): raise ValueError('Invalid explicit reward stream')
        result=MULEStep(packet,execution,nominal.clone(),adaptive.clone(),failure.clone(),timeout.clone(),
            packet.snapshot(),done&packet.supervision_valid,failure_reasons={k:v.clone() for k,v in reasons.items()})
        self.reset(done.nonzero().flatten())
        return MULEStep(self.observe(),result.execution,result.nominal_reward,result.adaptive_reward,
            result.terminated,result.timed_out,result.terminal,result.terminal_valid,failure_reasons=result.failure_reasons)

    def set_payload_severity(self,value): self.simulator.set_payload_severity(value)
    def begin_iteration(self,iteration): self.iteration=int(iteration)
    # No automatic phase/curriculum progression in this implementation stage.

    def physical_labels(self):
        """Owned privileged physical evidence for future rewards, NEVER actor inputs.

        Ball COM is its link origin (centered sphere). Robot reference is torso
        link origin, whose velocity includes the offset from randomized COM.
        Applied contact reaction includes force, not an inferred mass*g wrench.
        """
        s=self.simulator; relative,velocity=s.payload_motion()
        return dict(realized_robot={k:v.clone() for k,v in self._robot_parameters.items()},ball_mass_kg=s.ball_mass.clone(),
            total_ball_mass_kg=s.ball_mass.sum(-1),ball_radius_m=s.ball_radius.clone(),
            ball_relative_position_b=relative.clone(),ball_relative_velocity_b=velocity.clone(),
            payload_reaction_world_n=s.payload_reaction_world.clone(),
            ground_contact_force_world_n=s.ground_total_forces_world.clone(),
            requested_torque_nm=s.last_requested_torque.clone(),executed_torque_nm=s.hard_pact_executed_torque().clone())

    def physical_contract(self):
        from rsl_rl.beyond_robustness_config import settings_dict
        return dict(version=1,shared=settings_dict(self.resolved.shared),payload=asdict(self.resolved.payload),
            schema=self.input_schema.contract(),reward=self.reward_contract,
            critic_disturbance='measured ball-on-robot contact force, normal+tangent, body axes',
            force_proxy='last applied joint torque and endpoint named foot COM Jacobian, body axes')

    def curriculum_state_dict(self):
        s=self.simulator
        return deepcopy(dict(version=1,contract=self.physical_contract(),sampling=s.sampling_state_dict(),
            domain=self.domain_rand_curriculum.state_dict(),command_ranges=self.command_ranges,
            command_rng=self.command_rng.get_state(),noise=self.actor_observation_cache.state_dict(),
            action_delay=self.action_delay.clone(),iteration=self.iteration,
            terrain_levels=s.terrain_levels.clone(),terrain_types=s.terrain_types.clone()))

    def validate_curriculum_state_dict(self,state):
        if state['version']!=1 or state['contract']!=self.physical_contract(): raise ValueError('Incompatible MULE physical contract')
        self.simulator.validate_sampling_state_dict(state['sampling'])
        candidate=deepcopy(self.domain_rand_curriculum); candidate.load_state_dict(state['domain'])
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
        if s.payload_cfg.mode=='nominal': s.push_value*=.01; s.vert_value*=.01; s.angular_push_value*=.01
        s.load_sampling_state_dict(state['sampling'])
        self.command_ranges=deepcopy(state['command_ranges']); self.command_rng.set_state(state['command_rng'].cpu())
        self.actor_observation_cache.load_state_dict(state['noise']); self.action_delay.copy_(state['action_delay'])
        self.iteration=state['iteration']; self.commands.zero_(); self._packet=None; self.clock=0
        self.episode_id.fill_(-1); s._push_call_counter=0
        if hasattr(s,'_terrain'):
            s._terrain_levels.copy_(state['terrain_levels']); s._terrain_types.copy_(state['terrain_types'])
            s._env_origins.copy_(s._terrain_origins[s._terrain_levels,s._terrain_types])

    def iteration_metrics(self):
        s=self.simulator
        return {'Payload/severity':s.payload_progress,'Payload/total_mass_mean_kg':float(s.ball_mass.sum(-1).mean()),
            'Payload/ball_mass_max_kg':float(s.ball_mass.max()),'Train/environment_rolling':int(bool(s.balls)),
            'Episode/terrain_level':float(s.terrain_levels.float().mean())}

    def validate_stage_transfer(self,state):
        raise ValueError('MULE liquid/stage transfer adapter not implemented; fresh same-contract resume only')

    def close(self): self.simulator._app_launcher.app.close()
