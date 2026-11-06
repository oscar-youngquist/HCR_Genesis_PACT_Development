"""Local, detached mixed-coordinate inverse-dynamics teacher (not a controller).

Canonical velocities are world base linear/angular followed by joint rates.
Task x/y are heading-relative to the base; task z is world height in metres.
"""
import torch


def enabled(cfg, action_mode="coupled"):
    from .b1z1_actor_feedforward import enabled as feedforward_enabled
    return (feedforward_enabled(cfg, action_mode)
            and cfg.get("actor_phys_compliance_enabled", False)
            and cfg.get("actor_phys_arm_feedforward_compliant_local_enabled", False))


def world_jacobian(state, position, jacobian):
    """Explicit world-base tangent columns; do not rotate BARD's base tangent twice."""
    j = jacobian[:, :3].clone()
    eye = torch.eye(3, device=j.device, dtype=j.dtype)
    j[:, :, :3] = eye
    r = position-state[:, :3]
    # Each angular column is e_i cross r for WORLD omega_i.
    j[:, :, 3:6] = torch.cross(eye[None].expand(len(j),-1,-1), r[:,None,:].expand(-1,3,-1),dim=-1).transpose(1,2)
    return j


def task_map(state, position, jacobian):
    from legged_gym.envs.b1z1.directional_compliance import heading, rotate_yaw, task_position
    yaw = heading(state[:, 3:7])
    relative = rotate_yaw(position-state[:, :3], yaw, True)
    jacobian = world_jacobian(state, position, jacobian)
    j = jacobian.clone()
    j[:, :, :3] -= torch.eye(3, device=j.device, dtype=j.dtype)
    h = rotate_yaw(j.transpose(1, 2), yaw[:, None], True).transpose(1, 2)
    x,y,z,w = state[:, 3:7].unbind(-1)
    axis = torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)), -1)
    denominator = axis[:, :2].square().sum(-1)
    yaw_map = torch.cat((-axis[:, 2:3]*axis[:, :2]/denominator.clamp_min(1e-6)[:, None],
                         torch.ones_like(axis[:, :1])), -1)
    transport = torch.stack((-relative[:, 1], relative[:, 0]), -1)
    h[:, :2, 3:6] -= transport[:, :, None]*yaw_map[:, None, :]
    h[:, 2] = jacobian[:, 2]  # World height retains base translation.
    return h, task_position(position, state[:, :3], yaw), denominator > 1e-6


@torch.no_grad()
def kinematics(backend, state, jacobian):
    """Cache H and dot(H)v once with mechanics, not once per minibatch.

    H is analytic. Bias uses a centred 1 ms kinematic directional derivative
    along constant WORLD generalized velocity. This includes yaw acceleration
    due to roll/pitch motion, Coriolis transport, and articulated J-dot terms.
    Only FK/twist is evaluated: no second dynamics solve or policy inference.
    """
    from .b1z1_actor_physics import integrate_pose
    from legged_gym.envs.b1z1.directional_compliance import heading, task_velocity
    position = backend.ee_position(state[:, :51])
    h, s, valid = task_map(state, position, jacobian)
    step = .001
    rates = []
    for sign in (-1, 1):
        shifted = integrate_pose(state[:, :51], state[:, 26:51], sign*step)
        p, j = backend.ee_linear_kinematics(shifted)
        v = (world_jacobian(shifted,p,j)@shifted[:,26:51,None]).squeeze(-1)
        yaw, rate = heading(shifted[:, 3:7], shifted[:, 29:32])
        rates.append(task_velocity(p, v, shifted[:, :3], shifted[:, 26:29], yaw, rate))
    bias = (rates[1]-rates[0])/(2*step)
    valid &= torch.isfinite(h).all(dim=(1,2)) & torch.isfinite(s).all(-1) & torch.isfinite(bias).all(-1)
    # A failed optional local calculation must not invalidate retained rollout
    # objectives when the actor's other mechanics/predictions are still finite.
    return dict(local_h=torch.where(valid[:,None,None],h,0.),
                local_s=torch.where(valid[:,None],s,0.),
                local_bias=torch.where(valid[:,None],bias,0.), local_kinematic_valid=valid[:, None])


@torch.no_grad()
def target(a, actions, state, fixed, acceleration, ids, dofs, data):
    from .b1z1_actor_feedforward import compensation_target
    cfg = a.cfg
    g = ids + fixed.mass_matrix.shape[-1]-dofs
    # Calling the shared controller with a position-only action gives the exact
    # weighted, motor-scaled PD contribution, BEFORE total execution clipping.
    pd = a._coupled_torque(actions[:, :a.actor_critic.num_actions].detach().clamp(
        -cfg["clip_actions"], cfg["clip_actions"]), state.detach()).index_select(1, ids)
    mass = fixed.mass_matrix.index_select(1, g).index_select(2, g)
    a0 = torch.linalg.solve(mass, pd.unsqueeze(-1)).squeeze(-1)
    h = fixed.local_h
    ha = h.index_select(2, g)
    au = acceleration.detach().index_fill(1, g, 0.)
    desired = (data["local_ref_acceleration"]
        + cfg["actor_phys_compliance_omega"]**2*(data["local_ref_position_start"]-fixed.local_s)
        + 2*cfg["actor_phys_compliance_zeta"]*cfg["actor_phys_compliance_omega"]
        *(data["local_ref_velocity_start"]-(h@state[:, 26:51,None]).squeeze(-1)))
    rhs = desired-fixed.local_bias-(h@au.unsqueeze(-1)).squeeze(-1)
    eye = torch.eye(3, device=h.device, dtype=h.dtype)
    correction = ha.transpose(1,2)@torch.linalg.solve(
        ha@ha.transpose(1,2)+cfg["actor_phys_compliance_dls_damping"]**2*eye,
        (rhs-(ha@a0.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1))
    desired_arm = a0+correction.squeeze(-1)
    original, bias, coupling, external = compensation_target(
        fixed, acceleration, data["feedforward_applied_force"], ids, dofs)
    inertial = (mass@desired_arm.unsqueeze(-1)).squeeze(-1)
    teacher = original+inertial-pd
    # One-control-interval feasibility check, not a displacement-probe test.
    dt = cfg["dt"]
    q = state[:, 7:7+dofs].index_select(1, ids)
    v = state[:, 32:32+dofs].index_select(1, ids)
    next_q = q+dt*v+.5*dt**2*desired_arm
    next_v = v+dt*desired_arm
    margin = cfg["actor_phys_compliance_joint_margin"]
    feasible = ((next_q >= data["q_min"].index_select(1,ids)+margin)
        & (next_q <= data["q_max"].index_select(1,ids)-margin)
        & (next_v.abs() <= data["qd_max"].index_select(1,ids))).all(-1)
    scale = actions.new_tensor(cfg["torque_action_scale"]).expand(a.actor_critic.num_actions)[ids]
    ff_limit = state[:, 97:97+dofs][:,ids].abs()*state[:,155:156].abs()*scale.abs()*cfg["clip_actions"]
    saturated = ((teacher.abs()>ff_limit) | ((teacher+pd).abs()>data["torque_max"][:,ids])).any(-1)
    valid = data["local_ref_valid"].flatten().bool() & fixed.local_kinematic_valid.flatten().bool()
    valid &= feasible & ~saturated & torch.isfinite(teacher).all(-1)
    metrics = dict(local_invalid_fraction=(~valid).to(state.dtype).mean(),
                   local_saturated_fraction=saturated.to(state.dtype).mean(),
                   local_infeasible_fraction=(~feasible).to(state.dtype).mean())
    if getattr(a, "enable_additional_diagnostics", False):
        residual = (ha@desired_arm.unsqueeze(-1)).squeeze(-1)-rhs
        metrics.update(local_task_acceleration_residual=residual.norm(dim=-1).mean(),
            local_inertial_norm_nm=inertial.norm(dim=-1).mean(), local_pd_norm_nm=pd.norm(dim=-1).mean(),
            local_pd_ff_opposition_nm=((pd.abs()+teacher.abs()-(pd+teacher).abs())*.5).mean())
    return teacher, bias, coupling, external, valid, metrics
