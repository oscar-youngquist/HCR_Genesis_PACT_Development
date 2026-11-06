"""Detached reference generation in heading-XY/world-Z coordinates, SI units.

No policy, simulator, force application, or optimizer lives here. All solves
are batched small DLS probes, not iterative IK. Collision geometry is supplied
by the caller; filter state advances once per control interval, never in PPO.
"""
import math
from dataclasses import dataclass

import torch
from .compliance_bounds import bound_reference, feasible


class ComplianceCfg:
    enabled = False
    # Coefficient order: [-x,+x,-y,+y,-z,+z] and [-x,+x,-y,+y,-yaw,+yaw].
    ee_profile_low = [0.] * 6
    ee_profile_high = [1.] * 6
    base_profile_low = [0.] * 6
    base_profile_high = [1.] * 6
    # Initial coverage choices, not empirically optimized mixture weights.
    sampling_mode = "mixture"  # "uniform" retains the original sampler exactly.
    family_weights = [0.20, 0.20, 0.15, 0.20, 0.15, 0.10]
    ee_axis_probability = [0.55, 0.55, 0.50]  # x, y, z
    base_axis_probability = [0.55, 0.55, 0.25]  # x, y, yaw (independent)
    paired_direction_probability = 0.75  # Both signs, conditional on an active axis.
    symmetric_gain_probability = 0.70
    planar_gain_correlation_probability = 0.50
    ee_active_range = [0.20, 1.0]
    base_active_range = [0.20, 1.0]
    payload_probability = 0.50  # Within the asymmetric family.
    reverse_payload_probability = 0.25
    payload_low_range = [0.05, 0.15]
    payload_high_range = [0.40, 0.65]
    payload_planar_range = [0.20, 0.60]
    ee_gain = [0.005, 0.005, 0.005]  # m/N at compliance=1
    base_gain = [0.008, 0.008, 0.015]  # (m/s)/N, (rad/s)/(Nm)
    omega = 6.0  # rad/s; exact critically damped held-input integration
    zeta = 1.0
    base_time_constant = 0.4  # s
    include_torso_disturbances = True  # False reproduces EE-only base recruitment.
    stiffness_gate = True
    stiffness_threshold = 0.15
    workspace_gate = True
    workspace_margin = [0.06, 0.15, 0.15]  # radius m, elevation/azimuth rad
    joint_gate = True
    joint_margin = 0.03  # rad; buffered hard limits
    joint_headroom = 0.25  # rad; directional weighting saturation
    probe_distance = 0.04  # m, sustained yielding direction (not tracking error)
    probe_joint_limit = 0.15  # rad
    dls_damping = 0.025  # m/rad
    progress_full = 0.2
    progress_onset = 0.8
    collision_gate = True
    collision_margin = 0.035  # m
    collision_probe_steps = 3  # fixed configuration interpolation count
    ee_radius = 0.025  # EE-only sphere; not full-arm collision protection.
    arm_radius = 0.035  # Nonadjacent arm obstacle capsules.
    leg_radius = 0.055
    collision_batch_size = 8192  # bounded scratch; amortizes launches at 4096 envs
    compile_geometry = True  # CUDA-only pure kernels; first-use compiler warmup
    # Stable latency by default: the same full batch on every refinement.
    # Opt-in compaction saves work but new subset sizes may compile/capture late.
    collision_compact_refinement = False
    # Zero-based arm skeleton segments: own final segment and wrist neighbour.
    ee_excluded_arm_segments = [4, 5]
    force_threshold = 0.5  # N; avoids undefined zero-force directions


def validate(cfg):
    for part in ("ee", "base"):
        lo, hi = [torch.as_tensor(getattr(cfg, part+"_profile_"+end)) for end in ("low", "high")]
        if lo.shape != (6,) or hi.shape != (6,) or not ((0 <= lo) & (lo <= hi) & (hi <= 1)).all():
            raise ValueError(part+" compliance ranges must contain six ordered values in [0,1]")
        gain = torch.as_tensor(getattr(cfg, part+"_gain"))
        if gain.shape != (3,) or not torch.isfinite(gain).all() or (gain < 0).any():
            raise ValueError(part+"_gain must be three finite nonnegative SI gains")
    if cfg.zeta != 1:
        raise ValueError("Compliance currently implements exact critical damping (zeta=1) only")
    for key in ("omega", "base_time_constant", "stiffness_threshold", "joint_headroom",
                "probe_distance", "probe_joint_limit", "dls_damping", "collision_margin"):
        if not math.isfinite(getattr(cfg,key)) or getattr(cfg,key) <= 0:
            raise ValueError(key+" must be finite and positive")
    if not 0 <= cfg.progress_full < cfg.progress_onset <= 1:
        raise ValueError("Require 0 <= progress_full < progress_onset <= 1")
    if cfg.collision_probe_steps < 1 or any(not math.isfinite(r) or r<=0 for r in (cfg.ee_radius,cfg.arm_radius,cfg.leg_radius)):
        raise ValueError("Collision probes and primitive radii must be positive")
    if cfg.collision_batch_size < 1:
        raise ValueError("collision_batch_size must be positive")


def smooth(x):
    x = x.clamp(0, 1)
    return x.square()*(3-2*x)


def signed_response(force, profile, gain):
    pairs = profile.reshape(*profile.shape[:-1], 3, 2)
    return (pairs[..., 1]*force.clamp_min(0)+pairs[..., 0]*force.clamp_max(0))*force.new_tensor(gain)


def rotate_yaw(value, yaw, inverse=False):
    while yaw.ndim < value.ndim-1:
        yaw = yaw.unsqueeze(-1)
    c, s = yaw.cos(), yaw.sin()*(-1 if inverse else 1)
    return torch.stack((c*value[...,0]-s*value[...,1], s*value[...,0]+c*value[...,1], value[...,2]), -1)


def torso_wrench_heading(force_world, couple_world, application_point_world, base_world, yaw):
    """Forces ON the robot; couple is about the application point (N, Nm).

    Shift once to the base link origin, then rotate into heading XY/world Z.
    If a source already supplies a moment about the base, pass base_world as
    its reference point: adding the application-point moment again is incorrect.
    """
    moment = couple_world + torch.cross(application_point_world-base_world, force_world, dim=-1)
    force = rotate_yaw(force_world, yaw, True)
    moment = rotate_yaw(moment, yaw, True)
    return torch.cat((force[..., :2], moment[..., 2:3]), -1)


def heading(quat, omega_world=None):
    """xyzw orientation; actual Euler yaw derivative, not body-z gyro."""
    x,y,z,w = quat.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    if omega_world is None:
        return yaw
    # Heading is the azimuth of the body x-axis; differentiate that vector.
    axis = torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)), -1)
    derivative = torch.cross(omega_world, axis, dim=-1)
    rate = (axis[...,0]*derivative[...,1]-axis[...,1]*derivative[...,0])/axis[...,:2].square().sum(-1).clamp_min(1e-6)
    return yaw, rate


def task_position(world, base, yaw):
    local = rotate_yaw(world-base, yaw, True)
    return torch.cat((local[...,:2], world[...,2:3]), -1)


def world_position(task, base, yaw):
    local = torch.cat((task[...,:2], task[...,2:3]-base[...,2:3]), -1)
    return base+rotate_yaw(local, yaw)


def task_velocity(world_pos, world_vel, base, base_vel, yaw, yaw_rate):
    local = rotate_yaw(world_pos-base, yaw, True)
    relative = rotate_yaw(world_vel-base_vel, yaw, True)
    transport = torch.stack((-local[...,1],local[...,0],torch.zeros_like(local[...,0])), -1)*yaw_rate[...,None]
    return torch.cat(((relative-transport)[...,:2], world_vel[...,2:3]), -1)


def world_velocity(task, rate, base, base_vel, yaw, yaw_rate):
    local = torch.cat((task[...,:2],task[...,2:3]-base[...,2:3]),-1)
    local_rate = torch.cat((rate[...,:2],rate[...,2:3]-base_vel[...,2:3]),-1)
    transport = torch.stack((-local[...,1],local[...,0],torch.zeros_like(local[...,0])),-1)*yaw_rate[...,None]
    return base_vel+rotate_yaw(local_rate+transport,yaw)


def spherical(x):
    r = x.norm(dim=-1).clamp_min(1e-8)
    return torch.stack((r, torch.atan2(x[...,2], x[...,:2].norm(dim=-1)), torch.atan2(x[...,1],x[...,0])),-1)


def cartesian(s):
    r,p,y = s.unbind(-1)
    return torch.stack((r*p.cos()*y.cos(),r*p.cos()*y.sin(),r*p.sin()),-1)


def sphere_basis(s):
    _,p,y = s.unbind(-1)
    radial = torch.stack((p.cos()*y.cos(),p.cos()*y.sin(),p.sin()),-1)
    pitch = torch.stack((-p.sin()*y.cos(),-p.sin()*y.sin(),p.cos()),-1)
    azimuth = torch.stack((-y.sin(),y.cos(),torch.zeros_like(y)),-1)
    return torch.stack((radial,pitch,azimuth),-2)


def chart(x, bounds):
    s = spherical(x)
    mid = bounds[2].mean()
    s = torch.cat((s[...,:2], (mid+torch.remainder(s[...,2:3]-mid+math.pi,2*math.pi)-math.pi)), -1)
    return s


def collision_approach(start, samples, margin):
    """Compare each obstacle with itself before taking a maximum across pairs."""
    return (smooth(1-samples/margin)*(samples < start-1e-6)).amax(-1)


def workspace_gate(position, demand, center, bounds, margin):
    s = chart(position-center,bounds)
    rates = (sphere_basis(s)*demand.unsqueeze(-2)).sum(-1)
    distance = torch.where(rates >= 0, bounds[:,1]-s, s-bounds[:,0])
    near = smooth(1-distance/position.new_tensor(margin))
    return torch.where(rates.abs()>1e-8,near,torch.zeros_like(near)).amax(-1)


def directional_probe(jacobian, q, limits, demand, cfg):
    """Unweighted probe selects joint direction; weighted probe uses headroom.

    Headroom is selected by each joint's motion sign, so moving away from a
    limit is not penalized merely for being near that limit.
    """
    eye = torch.eye(3,device=q.device,dtype=q.dtype)
    jt = jacobian.transpose(-1,-2)
    solve = lambda j,w: w.unsqueeze(-1)*j.transpose(-1,-2) @ torch.linalg.solve(
        (j*w.unsqueeze(-2)) @ j.transpose(-1,-2)+cfg.dls_damping**2*eye, demand.unsqueeze(-1))
    initial = solve(jacobian,torch.ones_like(q)).squeeze(-1)
    lo,hi = limits[...,0]+cfg.joint_margin,limits[...,1]-cfg.joint_margin
    headroom = torch.where(initial >= 0,hi-q,q-lo).clamp_min(0)
    weights = (headroom/cfg.joint_headroom).clamp(0,1).square()
    dq = solve(jacobian,weights).squeeze(-1)
    dq = dq.clamp(-cfg.probe_joint_limit,cfg.probe_joint_limit)
    dq = torch.minimum(torch.maximum(dq, (lo-q).clamp_max(0)),(hi-q).clamp_min(0))
    achieved = (jacobian@dq.unsqueeze(-1)).squeeze(-1)
    magnitude = demand.norm(dim=-1).clamp_min(1e-8)
    progress = (achieved*demand).sum(-1)/magnitude.square()
    mismatch = (achieved-demand).norm(dim=-1)/magnitude
    gate = torch.maximum(smooth((cfg.progress_onset-progress)/(cfg.progress_onset-cfg.progress_full)),
                         smooth((mismatch-(1-cfg.progress_onset))/(cfg.progress_onset-cfg.progress_full)))
    return dq, gate, progress


@dataclass
class Reference:
    position: torch.Tensor
    velocity: torch.Tensor
    base: torch.Tensor
    gates: torch.Tensor
    workspace: torch.Tensor
    joint: torch.Tensor
    collision: torch.Tensor
    clipped: torch.Tensor
    torso_wrench: torch.Tensor
    ee_wrench: torch.Tensor
    combined_wrench: torch.Tensor


class ComplianceReference:
    def __init__(self,cfg,n,device,dtype=torch.float32):
        validate(cfg)
        self.cfg=cfg
        self.bound_reference=bound_reference
        self.directional_probe=directional_probe
        self.workspace_gate=workspace_gate
        if cfg.compile_geometry and torch.device(device).type=='cuda':
            self.bound_reference=torch.compile(bound_reference,dynamic=True,fullgraph=True)
            self.directional_probe=torch.compile(directional_probe,dynamic=True,fullgraph=True)
            self.workspace_gate=torch.compile(workspace_gate,dynamic=True,fullgraph=True)
        self.delta=torch.zeros(n,3,device=device,dtype=dtype)
        self.rate=torch.zeros_like(self.delta)
        self.eta=torch.zeros_like(self.delta)
        self.accepted=torch.zeros_like(self.delta)
        self.initialized=torch.zeros(n,device=device,dtype=torch.bool)
        self.recovering=torch.zeros_like(self.initialized)
        self.profile=torch.zeros(n,12,device=device,dtype=dtype)
        from .compliance_sampling import ProfileSampler
        self.sampler=ProfileSampler(cfg,device,dtype)
        self.family_ids=torch.full((n,),-1,device=device,dtype=torch.long)
        self.reset(torch.arange(n,device=device))

    def reset(self,ids):
        self.delta[ids]=0; self.rate[ids]=0; self.eta[ids]=0
        self.accepted[ids]=0; self.initialized[ids]=False
        self.recovering[ids]=False
        self.profile[ids],self.family_ids[ids]=self.sampler.sample(len(ids))

    @torch.no_grad()
    def advance(self,*,nominal,nominal_velocity,center,bounds,force,lever,command,command_limits,
                q,limits,jacobian,collision_clearance,dt,target_safe=None,measured_position=None,torso_wrench=None,
                collision_path_gate=None, recovery_anchor=None, collision_start=None):
        c=self.cfg
        previous=torch.where(self.initialized[:,None],self.accepted,nominal)
        recovering=~feasible(previous,center,bounds)
        # A terrain/workspace change is a reference reinitialization, not a
        # zero-length first-exit step. Preserve only enabled Cartesian offsets.
        recovered=previous
        if recovering.any():
            anchor=nominal if recovery_anchor is None else recovery_anchor(nominal,previous,recovering)
            if not (feasible(anchor,center,bounds)|~recovering).all():
                raise RuntimeError('No feasible compliance recovery anchor in current workspace')
            enabled=self.profile[:,:6].reshape(-1,3,2).amax(-1)>0
            recovered,_=self.bound_reference(anchor+self.delta*enabled,torch.zeros_like(previous),center,bounds,start=anchor)
            if target_safe is not None:recovered=target_safe(recovered,anchor)
            previous=torch.where(recovering[:,None],recovered,previous)
            self.delta.copy_(torch.where(recovering[:,None],previous-nominal,self.delta))
            self.rate.copy_(torch.where(recovering[:,None],torch.zeros_like(self.rate),self.rate))
        demand=signed_response(force,self.profile[:,:6],c.ee_gain)
        active=force.abs()>c.force_threshold
        # Three individual signed axes plus combined demand, regardless of offset rate.
        axes=torch.diag_embed(force.sign())*c.probe_distance
        yielding=torch.where(demand.norm(dim=-1,keepdim=True)>1e-8,demand,force)
        combined=yielding/yielding.norm(dim=-1,keepdim=True).clamp_min(1e-8)*c.probe_distance
        probes=torch.cat((axes,combined[:,None]),1)
        repeated_q=q[:,None].expand(-1,probes.shape[1],-1)
        repeated_limits=limits[:,None].expand(-1,probes.shape[1],-1,-1)
        dq,joint,progress=self.directional_probe(jacobian[:,None].expand(-1,probes.shape[1],-1,-1),
                                          repeated_q,repeated_limits,probes,c)
        workspace=self.workspace_gate(previous[:,None],probes,center[:,None],bounds,c.workspace_margin)
        if measured_position is not None:
            workspace=torch.maximum(workspace,self.workspace_gate(measured_position[:,None],probes,
                center[:,None],bounds,c.workspace_margin))
        collision=torch.zeros_like(workspace)
        if c.collision_gate:
            start=collision_clearance(q) if collision_start is None else collision_start
            if start.ndim==1:start=start[:,None]
            fractions=torch.linspace(1/c.collision_probe_steps,1,c.collision_probe_steps,device=q.device)
            probe_q=(repeated_q[:,:,None]+dq[:,:,None]*fractions[None,None,:,None]).flatten(1,2)
            clear=collision_clearance(probe_q).reshape(len(q),probes.shape[1],c.collision_probe_steps,-1)
            collision=collision_approach(start[:,None,None],clear,c.collision_margin).amax(-1)
            if collision_path_gate is not None:
                collision=torch.maximum(collision,collision_path_gate(probes,previous))
        pairs=self.profile[:,:6].reshape(-1,3,2)
        gain=torch.where(force>=0,pairs[...,1],pairs[...,0])
        stiffness=smooth(1-gain/c.stiffness_threshold) if c.stiffness_gate else torch.zeros_like(force)
        contributions=[]
        for gate,on in ((workspace,c.workspace_gate),(joint,c.joint_gate),(collision,c.collision_gate)):
            # Combined-direction blockage recruits only force-active axes.
            contributions.append(torch.maximum(gate[:,:3],gate[:,3:]) if on else torch.zeros_like(force))
        gates=1-(1-stiffness)*(1-contributions[0])*(1-contributions[1])*(1-contributions[2])
        gates=gates*active
        equilibrium,_=self.bound_reference(nominal+demand,torch.zeros_like(demand),center,bounds,start=nominal)
        if target_safe is not None:
            equilibrium=target_safe(equilibrium,previous)
        desired=equilibrium-nominal
        error=self.delta-desired
        b=self.rate+c.omega*error
        decay=math.exp(-c.omega*dt)
        delta=desired+(error+dt*b)*decay
        rate=(self.rate-c.omega*dt*b)*decay
        nominal_only=(self.profile[:,:6].amax(-1)==0)&(force.abs().amax(-1)==0)
        # No admittance state should delay a rigid, unloaded nominal command.
        # Hard path constraints still apply below, so an unsafe goal is not jumped to.
        delta=torch.where(nominal_only[:,None],torch.zeros_like(delta),delta)
        rate=torch.where(nominal_only[:,None],torch.zeros_like(rate),rate)
        position,velocity=self.bound_reference(nominal+delta,nominal_velocity+rate,center,bounds,start=previous)
        if target_safe is not None:
            safe=target_safe(position,previous)
            blocked=(safe-position).norm(dim=-1)>1e-7
            position=safe
            velocity=torch.where(blocked[:,None],torch.zeros_like(velocity),velocity)
        # A recovery jump has no inherited filter momentum. An exact nominal
        # recovery resumes nominal velocity; otherwise hold for this interval.
        position=torch.where(recovering[:,None],recovered,position)
        recovery_velocity=torch.where(((recovered-nominal).norm(dim=-1)<1e-7)[:,None],nominal_velocity,torch.zeros_like(velocity))
        velocity=torch.where(recovering[:,None],recovery_velocity,velocity)
        self.recovering=recovering.detach()
        self.accepted.copy_(position);self.initialized.fill_(True)
        self.delta.copy_(position-nominal)
        self.rate.copy_(velocity-nominal_velocity)
        gated=force*gates
        ee_wrench=torch.cat((gated[:,:2],torch.cross(lever,gated,dim=-1)[:,2:3]),-1)
        direct=(torso_wrench.detach() if c.include_torso_disturbances and torso_wrench is not None
                else torch.zeros_like(ee_wrench))
        # Apply signed gains AFTER summation, so opposing loads cancel before
        # asymmetric gains are chosen. Arm gates affect only the EE term.
        wrench=ee_wrench+direct
        eta_eq=signed_response(wrench,self.profile[:,6:],c.base_gain)
        self.eta.lerp_(eta_eq,-math.expm1(-dt/c.base_time_constant))
        # Disabled axes stay disabled, even after a manually changed profile.
        enabled=self.profile[:,6:].reshape(-1,3,2).amax(-1)>0
        self.eta*=enabled
        base=(command+self.eta).clamp(command_limits[:,0],command_limits[:,1])
        self.eta.copy_(base-command)
        return Reference(position,velocity,base,gates,contributions[0],contributions[1],contributions[2],
                         (position-nominal-delta).norm(dim=-1)>1e-7,direct,ee_wrench,wrench)
