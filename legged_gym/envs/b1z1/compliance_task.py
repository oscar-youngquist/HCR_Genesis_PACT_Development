"""PACT-only adapter: nominal policy commands, privileged compliant objectives.

Snapshots are prepared after physical force events, before simulation, and
retained through terminal resets. PPO only copies them; it never runs filters.
"""
import torch

from .directional_compliance import (ComplianceReference, cartesian, spherical, heading,
    rotate_yaw, task_position, world_position, task_velocity, world_velocity,
    directional_probe, torso_wrench_heading, collision_approach)
from .compliance_geometry import RobotGeometry
from .ee_collision import segment_box


def enabled(cfg):
    return bool(getattr(getattr(cfg,"compliance",None),"enabled",False))


class DisturbanceCoverage:
    """Optional episode stream selection; not an event amplitude multiplier."""
    def __init__(self, weights, n, device):
        self.weights = None if weights is None else torch.as_tensor(weights, device=device, dtype=torch.float)
        if self.weights is not None and (self.weights.shape != (3,) or
                not torch.isfinite(self.weights).all() or (self.weights < 0).any() or self.weights.sum() <= 0):
            raise ValueError("compliance_disturbance_mode_weights needs three finite nonnegative weights with positive sum")
        self.ids = torch.full((n,), -1, device=device, dtype=torch.long)
        self.reset(torch.arange(n, device=device))

    def reset(self, ids):
        if self.weights is not None and len(ids):
            self.ids[ids] = torch.multinomial(self.weights, len(ids), replacement=True)

    def mask(self, output, ids, ee):
        if self.weights is not None:
            # Mask actual outputs before application/labels; retained stream
            # magnitudes and event timers remain the production ones.
            allowed = self.ids[ids] != (1 if ee else 0)
            output[ids] *= allowed[:, None]


def mask_disturbance(env, output, ids, *, ee):
    state = getattr(env, 'compliance', None)
    if state is not None:
        state.coverage.mask(output, ids, ee)


def configure(cfg):
    if not enabled(cfg):return
    from .directional_compliance import validate
    validate(cfg.compliance)
    if cfg.use_force_shifted_target:
        raise ValueError("Differential compliance and legacy force-shifted targets are mutually exclusive")
    if not getattr(cfg,"_compliance_layout_added",False):
        # Insert deployable profiles before the final six nominal command slots.
        # No force-derived reference/filter state is inserted into actor/history.
        cfg.env.num_observations += 12
        cfg.env.num_critic_state_obs += 12
        cfg.env.num_privileged_obs += 12
        cfg.env.num_privileged_recon_obs += 12
        cfg._compliance_layout_added=True
    # These objectives encode rigid rejection or suppress recruited locomotion.
    # Keep general effort, collision, joint safety and uprightness objectives.
    for name in ("tracking_arm_feedforward","impedance_consistency","arm_progress_before_torso",
                 "early_torso_tilt","stand_still","stand_still_contact",
                 "ref_dof_leg","feet_contact_number","walking_ref_dof","walking_ref_swing_dof"):
        if hasattr(cfg.rewards.scales,name):setattr(cfg.rewards.scales,name,0.)


def configure_algorithm(cfg,env_cfg):
    if not enabled(env_cfg):return cfg
    cfg=dict(cfg)
    # Fixed-target energy decay / torque teachers conflict with moving admittance.
    for key in ("actor_phys_arm_rejection_enabled","actor_phys_base_rejection_enabled",
                "actor_phys_arm_feedforward_enabled","actor_phys_base_balance_enabled"):
        cfg[key]=False
    cfg["actor_phys_force_allocation_weight"]=0.
    cfg["actor_phys_ee_stability_weight"]=0.
    # Keep configured actor velocity weights/scales, including ablation overrides.
    return cfg


class ComplianceTask:
    def __init__(self,env):
        from legged_gym import LEGGED_GYM_ROOT_DIR
        self.env=env; c=env.cfg.compliance; sim=env.simulator
        self.reference=ComplianceReference(c,env.num_envs,env.device)
        self.coverage=DisturbanceCoverage(getattr(env.cfg.commands,'compliance_disturbance_mode_weights',None),env.num_envs,env.device)
        self.wrench_sum=torch.zeros(env.num_envs,3,3,device=env.device)
        self.wrench_count=torch.zeros(env.num_envs,device=env.device)
        self.arm_ids=sim._arm_dof_cfg_ids
        self.geometry=RobotGeometry(env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
            env.cfg.asset.dof_names,self.arm_ids,env.cfg.asset.base_name,env.cfg.asset.gripper_name,
            env.cfg.asset.foot_name,c,env.device)
        self.workspace_clearance=segment_box
        if c.compile_geometry and torch.device(env.device).type=='cuda':
            self.workspace_clearance=torch.compile(segment_box,dynamic=True,fullgraph=True)
        ranges=env.cfg.goal_ee.ranges
        self.bounds=torch.tensor([ranges.pos_l,ranges.pos_p,ranges.pos_y],device=env.device)
        if not (self.bounds[:,1]>self.bounds[:,0]).all() or self.bounds[0,0]<=0 or self.bounds[2,1]-self.bounds[2,0]>2*torch.pi:
            raise ValueError("Invalid original spherical sampling bounds for compliance")
        self.command_limits=torch.tensor([env.command_ranges[k] for k in ('lin_vel_x','lin_vel_y','ang_vel_yaw')],device=env.device)
        self.target=None
        self.interval=None

    @property
    def profile(self):return self.reference.profile

    def reset(self,ids):
        self.reference.reset(ids)
        self.coverage.reset(ids)
        self.wrench_sum[ids]=0;self.wrench_count[ids]=0
        if self.target is not None:
            # Do not mutate the terminal replay snapshot in self.interval.
            self.target.position[ids]=self.nominal(0)[0][ids]
            self.target.velocity[ids]=0
            self.target.base[ids]=self.env.commands[ids,:3]

    def center(self):
        env=self.env; yaw=env._get_base_yaw_quat()
        return task_position(env.get_ee_goal_spherical_center(yaw),env.simulator.base_pos,heading(env.simulator.base_quat))

    def nominal(self,advance):
        env=self.env
        u=((env.goal_timer+advance)/env.traj_timesteps).clamp(0,1)
        ratio=10*u**3-15*u**4+6*u**5
        rate=30*u.square()*(1-u).square()/(env.traj_timesteps*env.dt)
        sphere=env.ee_start_sphere+ratio[:,None]*(env.ee_goal_sphere-env.ee_start_sphere)
        sphere_rate=rate[:,None]*(env.ee_goal_sphere-env.ee_start_sphere)
        from .directional_compliance import sphere_basis
        basis=sphere_basis(sphere)
        factors=torch.stack((torch.ones_like(u),sphere[:,0],sphere[:,0]*sphere[:,1].cos()),-1)
        velocity=(basis*(sphere_rate*factors)[:,:,None]).sum(1)
        return self.center()+cartesian(sphere),velocity

    @torch.no_grad()
    def begin(self):
        env=self.env; sim=env.simulator; c=env.cfg.compliance
        nominal,nominal_velocity=self.nominal(1)
        # The first accepted reference is the actual interval-start nominal,
        # not the next nominal reconstructed with yesterday's offset.
        initial=~self.reference.initialized
        self.reference.accepted.copy_(torch.where(initial[:,None],self.nominal(0)[0],self.reference.accepted))
        self.reference.initialized.fill_(True)
        center=self.center()
        yaw=heading(sim.base_quat)
        force=rotate_yaw(env.ee_force_ext_world,yaw,True)
        lever=rotate_yaw(sim.ee_pos-sim.base_pos,yaw,True)
        direct=None
        if c.include_torso_disturbances:
            source=getattr(sim,'applied_torso_wrench_source',None)
            if source is None:
                raise RuntimeError("Direct torso compliance requires a verified applied_torso_wrench_source backend; use include_torso_disturbances=False for EE-only behavior")
            f,t,point=source()
            direct=torso_wrench_heading(f,t,point,sim.base_pos,yaw)
        full=sim.dof_pos.detach()
        q=full.index_select(1,self.arm_ids)
        limits=getattr(sim,"dof_pos_limits_hard",sim.dof_pos_limits)
        if limits.ndim==2:limits=limits.unsqueeze(0).expand(env.num_envs,-1,-1)
        limits=limits.index_select(1,self.arm_ids)
        _,j,arm_points,legs=self.geometry.forward(full)
        # URDF Jacobian is torso-relative. Full roll/pitch rotate geometry;
        # only the final force/workspace axes are heading aligned.
        from legged_gym.utils.math_utils import quat_apply
        quat=sim.base_quat[:,None].expand(-1,j.shape[-1],-1).reshape(-1,4)
        j=rotate_yaw(quat_apply(quat,j.transpose(1,2).reshape(-1,3)).reshape(env.num_envs,-1,3),yaw,True).transpose(1,2)
        clear=lambda arm:self.geometry.clearance_pairs(arm,full,legs)
        measured=task_position(sim.ee_pos,sim.base_pos,yaw)
        start_clear=clear(q)

        from legged_gym.utils.math_utils import quat_rotate_inverse
        def body(point):
            if point.ndim==2:
                return quat_rotate_inverse(sim.base_quat,
                    world_position(point,sim.base_pos,yaw)-sim.base_pos)
            world=world_position(point,sim.base_pos[:,None],yaw[:,None])-sim.base_pos[:,None]
            quat=sim.base_quat[:,None].expand(*point.shape[:-1],4)
            return quat_rotate_inverse(quat.reshape(-1,4),world.reshape(-1,3)).reshape_as(point)

        def path_clear(start,end):
            return self.geometry.path_clearance(body(start),body(end),arm_points[:,None],legs[:,None])

        def path_gate(directions):
            # Test the whole requested Cartesian path, not just the short DLS
            # configuration probe. Each obstacle retains its own approach test.
            start=measured[:,None].expand_as(directions)
            return collision_approach(start_clear[:,None],
                path_clear(start,start+directions),c.collision_margin)

        def safe(candidate,previous):
            # Fixed probes from measured configuration; not an IK feasibility proof.
            demand=candidate-measured
            dq,_,_=directional_probe(j,q,limits,demand,c)
            fractions=torch.linspace(1/c.collision_probe_steps,1,c.collision_probe_steps,device=q.device)
            minimum=clear(q[:,None]+dq[:,None]*fractions[None,:,None]).amin(1)
            allowed=((minimum>=0) | (minimum>start_clear+1e-6)).all(-1)
            # A short collision-free joint probe cannot certify a farther EE
            # reference. Sweep the EE sphere against frozen measured obstacles.
            start=torch.stack((measured,previous),1)
            clearance=self.geometry.point_clearance(body(start),arm_points[:,None],legs[:,None])
            path=path_clear(start,candidate[:,None].expand_as(start))
            end=self.geometry.point_clearance(body(candidate),arm_points,legs)[:,None]
            allowed &= ((path>=0) | ((clearance<0)&(path>=clearance-1e-6)&(end>clearance+1e-6))).all(-1).all(-1)
            # Retain the existing Cartesian torso-box/underground rejection.
            local=candidate-center
            inside=((local>env.collision_lower_limits)&(local<env.collision_upper_limits)).all(-1)
            allowed &= ~inside & (local[:,2]>=env.underground_limit)
            lower,upper=env.collision_lower_limits,env.collision_upper_limits
            # Retain the legacy workspace exclusion over the accepted path,
            # not just at its endpoint (straight Z has no hidden low point).
            old=previous-center
            swept=self.workspace_clearance(old,local,(lower+upper)/2,(upper-lower)/2)
            allowed &= swept>=0
            return torch.where(allowed[:,None],candidate,previous)

        self.target=self.reference.advance(nominal=nominal,nominal_velocity=nominal_velocity,center=center,
            bounds=self.bounds,force=force,lever=lever,command=env.commands[:,:3],command_limits=self.command_limits,
            q=q,limits=limits,jacobian=j,collision_clearance=clear,dt=env.dt,target_safe=safe,
            measured_position=measured,torso_wrench=direct,collision_path_gate=path_gate)
        self.wrench_sum+=torch.stack((self.target.torso_wrench,self.target.ee_wrench,self.target.combined_wrench),1)
        self.wrench_count+=1
        target_world=world_position(self.target.position,sim.base_pos,yaw)
        # Persist before post-step callbacks may resample commands/goals/reset rows.
        self.interval={"compliance_position":self.target.position.clone(),
                       "compliance_velocity":self.target.velocity.clone(),
                       "command":self.target.base.clone(),"ee_target":target_world.clone()}

    def wrench_statistics(self, ids):
        # Mean of actual control-interval reference contributions, collected
        # before terminal reset clears state; fixed control dt gives equal weights.
        count=self.wrench_count[ids].sum()
        mean=self.wrench_sum[ids].sum(0)/count.clamp_min(1)
        stats={f'compliance_wrench/{source}/{axis}':mean[i,j]
               for i,source in enumerate(('torso','ee_gated','combined'))
               for j,axis in enumerate(('fx_N','fy_N','tz_Nm'))}
        stats['compliance_wrench/interval_count']=count
        if self.coverage.weights is not None:
            stats.update({f'compliance_disturbance/{name}_fraction':(self.coverage.ids[ids]==i).float().mean()
                          for i,name in enumerate(('ee_only','torso_only','simultaneous'))})
        return stats

    def world_target(self):
        if self.target is None:return self.env.curr_ee_goal_cart_world
        sim=self.env.simulator
        return world_position(self.target.position,sim.base_pos,heading(sim.base_quat))

    def measured_velocity(self):
        sim=self.env.simulator
        yaw,rate=heading(sim.base_quat,sim._base_world_ang_vel)
        return task_velocity(sim.ee_pos,sim.ee_vel,sim.base_pos,sim._base_world_lin_vel,yaw,rate)

    def measured_base(self):
        sim=self.env.simulator
        yaw,rate=heading(sim.base_quat,sim._base_world_ang_vel)
        return torch.cat((rotate_yaw(sim._base_world_lin_vel,yaw,True)[:,:2],rate[:,None]),-1)


def attach_transition(algorithm,env):
    """Copy a previously advanced target; filters never run during replay."""
    state=getattr(env,"compliance",None)
    packet=getattr(algorithm.transition,"actor_physics",None)
    if state is None or packet is None:return
    packet.update({k:v.detach().to(algorithm.device).clone() for k,v in state.interval.items()})
