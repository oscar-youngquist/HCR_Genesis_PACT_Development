"""Opt-in PILOT continuation experiment; no registered task defaults change."""
from dataclasses import dataclass, asdict
import torch


@dataclass
class Candidate:
    continuous_clock: bool = True
    reference_air_time: bool = True
    strong_fraction: float = .25
    coefficient_range: tuple = (.8, 1.)
    force_range: tuple = (40., 50.)
    plateau_range: tuple = (1.5, 2.)
    onset_s: float = 5.
    ramp_s: float = .5
    repeat_s: float = 8.
    base_planar_gain: tuple = (.060, .040)
    structured_goals: bool = True

    def validate(self):
        if not 0 <= self.strong_fraction <= 1:
            raise ValueError('strong_fraction must be in [0,1]')
        for bounds, lo, hi in ((self.coefficient_range, 0., 1.),
                               (self.force_range, 0., float('inf')),
                               (self.plateau_range, 0., float('inf'))):
            if not lo <= bounds[0] <= bounds[1] <= hi:
                raise ValueError('Invalid candidate range')
        if self.ramp_s <= 0 or self.repeat_s <= 2*self.ramp_s+self.plateau_range[1]:
            raise ValueError('Pulse timing requires a release interval')
        return self


def configure(cfg, candidate=None):
    c = (candidate or Candidate()).validate()
    if not cfg.compliance.enabled or list(cfg.compliance.ee_gain) != [.005]*3:
        raise ValueError('Candidate requires the saved differential-compliance 0.005 m/N task')
    cfg.pilot_candidate = asdict(c)
    cfg.compliance.base_gain = [*c.base_planar_gain, cfg.compliance.base_gain[2]]
    return c


def walking_mask(command, cfg):
    limits = command.new_tensor([cfg.lin_vel_x_clip, cfg.lin_vel_y_clip, cfg.ang_vel_yaw_clip])
    return (command.abs() > limits).any(-1)


def pulse(age, plateau, c):
    # One held force per interval, including ramps; the plateau excludes ramps.
    t = torch.remainder((age-c.onset_s).clamp_min(0), c.repeat_s)
    return torch.minimum((t/c.ramp_s).clamp(0, 1),
        (1-(t-c.ramp_s-plateau)/c.ramp_s).clamp(0, 1))*(age >= c.onset_s)


def measurements(env):
    """Detached interval-end diagnostics, read before automatic reset."""
    from .directional_compliance import heading, smooth
    r=env.compliance.reference; t=env.compliance.target; s=env.simulator
    d=r.diagnostics
    if d is None: return {}
    force=d['force_task']; gated=force*t.gates
    # Request before total-command clipping, reconstructed from the previous eta.
    out={f'reference_stage_{k}':v for k,v in d.items()}
    out.update(applied_force_task=force, recruited_force_task=gated,
        recruited_wrench=t.ee_wrench, direct_torso_wrench=t.torso_wrench,
        combined_wrench=t.combined_wrench, base_reference=t.base,
        base_measured=env.compliance.measured_base(), gait_phase=env.gait_indices[:,None],
        profile_gate=smooth(1-d['selected_coefficients']/env.cfg.compliance.stiffness_threshold),
        gates=t.gates, workspace_gate=t.workspace, joint_gate=t.joint, collision_gate=t.collision,
        directional_progress=d['directional_progress'],
        accepted_offset=t.position-d['nominal_position'], raw_offset=d['raw_displacement'],
        bounded_offset=d['equilibrium_geometry']-d['nominal_position'],
        filtered_offset=d['filtered_candidate']-d['nominal_position'],
        workspace_reduction=d['equilibrium_workspace']-d['nominal_position']-d['raw_displacement'],
        geometry_reduction=d['equilibrium_geometry']-d['equilibrium_workspace'],
        filter_difference=d['filtered_candidate']-d['equilibrium_geometry'],
        final_reduction=t.position-d['filtered_candidate'],
        foot_contacts=s.foot_contacts,
        recovery=r.recovering[:,None], reference_clipped=t.clipped[:,None])
    from .directional_compliance import task_position
    out['ee_measured_task']=task_position(s.ee_pos,s.base_pos,heading(s.base_quat))
    out['ee_measured_velocity_task']=env.compliance.measured_velocity()
    if hasattr(env,'candidate_eta_start'):
        from .directional_compliance import signed_response
        import math
        eq=signed_response(t.combined_wrench,r.profile[:,6:],r.cfg.base_gain)
        request=env.candidate_eta_start+(eq-env.candidate_eta_start)*(-math.expm1(-env.dt/r.cfg.base_time_constant))
        request*=r.profile[:,6:].reshape(-1,3,2).amax(-1)>0
        out['requested_base_correction']=request
        out['requested_base_total']=env.candidate_command_start+request
        out['base_clipped']=(out['requested_base_total']-t.base).abs()>1e-6
    return {k:v.detach().clone() for k,v in out.items()}


def environment_class(base, training=False):
    class CandidatePILOT(base):
        def __init__(self,*args,**kwargs):
            self.candidate_ready=False
            self.candidate_recorder=None
            super().__init__(*args,**kwargs)
            self.candidate=Candidate(**self.cfg.pilot_candidate).validate()
            self.strong=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
            self.strong_axis=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
            self.strong_sign=torch.ones(self.num_envs,device=self.device)
            self.strong_force=torch.zeros(self.num_envs,device=self.device)
            self.strong_plateau=torch.zeros(self.num_envs,device=self.device)
            self.compliance.reference.collect_diagnostics=True
            self.candidate_ready=True

        def _step_contact_targets(self):
            if self.cfg.pilot_candidate['continuous_clock']:
                self.gait_indices=torch.remainder(self.gait_indices+self.dt/self.cfg.rewards.cycle_time,1.)
            else: super()._step_contact_targets()

        def _reward_feet_air_time(self):
            if not self.cfg.pilot_candidate['reference_air_time']:
                return super()._reward_feet_air_time()
            contact=self.simulator.foot_contacts
            filtered=contact | self.last_contacts
            self.last_contacts.copy_(contact)
            first=(self.feet_air_time>0)&filtered
            self.feet_air_time+=self.dt
            reward=((self.feet_air_time-.18)*first).sum(1).clamp_min(0)
            target=self.compliance.target
            command=self.commands[:,:3] if target is None else target.base
            reward*=walking_mask(command,self.cfg.commands)
            self.feet_air_time*=(~filtered).float()
            return reward

        def reset_idx(self,ids):
            super().reset_idx(ids)
            if not self.candidate_ready or not training or not len(ids): return
            c=self.candidate; n=len(ids)
            self.strong[ids]=torch.rand(n,device=self.device)<c.strong_fraction
            # Upward and forward subsets supply interior-yield and recruitment
            # opportunities. The other 75% retain the native signed sampler.
            self.strong_axis[ids]=torch.where(torch.rand(n,device=self.device)<.5,2,0)
            self.strong_sign[ids]=torch.where(torch.rand(n,device=self.device)<.25,-1.,1.)
            draw=lambda bounds: torch.empty(n,device=self.device).uniform_(*bounds)
            self.strong_force[ids]=draw(c.force_range)
            self.strong_plateau[ids]=draw(c.plateau_range)
            selected=ids[self.strong[ids]]
            # Positive signed directions yield; opposite pulls stay protected.
            p=self.compliance.reference.profile
            p[selected,:6]=0
            p[selected,2*self.strong_axis[selected]+1]=torch.empty(len(selected),device=self.device).uniform_(*c.coefficient_range)
            self.compliance.reference.family_ids[selected]=7
            if self.compliance.coverage.weights is not None:
                self.compliance.coverage.ids[selected]=2 # allow actual EE and native torso stream
            if c.structured_goals:
                old=self.ee_goal_sphere[selected].clone()
                # Interior vertical yielding versus near-radius planar handoff;
                # only the first goal of selected episodes changes. Native path
                # rejection and all subsequent nominal sampling are retained.
                goal=old.new_tensor([.55,.2,0.]).expand_as(old).clone()
                restricted=(self.strong_axis[selected]==0)&(torch.rand(len(selected),device=self.device)<.5)
                goal[restricted]=old.new_tensor([.85,0.,0.])
                bounds=self.compliance.bounds
                within=((goal>=bounds[:,0])&(goal<=bounds[:,1])).all(-1)
                self.ee_goal_sphere[selected]=goal
                reject=self.collision_check(selected)|~within
                self.ee_goal_sphere[selected]=torch.where(reject[:,None],old,goal)
                # Do not let a short native hold resample the interior goal before
                # the planned first pull. Keep native interpolation duration;
                # prolong only this subset's first hold through release/recovery.
                hold_end=c.onset_s+2*c.ramp_s+self.strong_plateau[selected]+1.
                self.traj_total_timesteps[selected]=torch.maximum(
                    self.traj_total_timesteps[selected],hold_end/self.dt)

        def _update_force_stream(self,*args,**kwargs):
            super()._update_force_stream(*args,**kwargs)
            if not training or not self.candidate_ready or kwargs['output'] is not self.ee_force_ext_world: return
            from .directional_compliance import heading,rotate_yaw
            c=self.candidate
            amplitude=self.strong_force*self.strong_sign*pulse(self.episode_length_buf*self.dt,self.strong_plateau,c)
            # Preserve the restored curriculum multiplier and physical enable gate.
            amplitude*=self.external_force_scale
            force=torch.zeros_like(self.ee_force_ext_world)
            force.scatter_(1,self.strong_axis[:,None],amplitude[:,None])
            force=rotate_yaw(force,heading(self.simulator.base_quat))
            self.ee_force_ext_world.copy_(torch.where(self.strong[:,None],force,self.ee_force_ext_world))
            # Parent applies this final buffer once, after its coverage mask.

        def step(self,actions):
            self.candidate_eta_start=self.compliance.reference.eta.clone()
            self.candidate_command_start=self.commands[:,:3].clone()
            return super().step(actions)

        def check_termination(self):
            super().check_termination()
            if not self.candidate_ready or self.compliance.target is None: return
            self.candidate_snapshot=measurements(self)
            if self.candidate_recorder is not None:
                self.candidate_recorder.add(self.candidate_snapshot,~self.reset_buf,self.compliance.profile)
    return CandidatePILOT
