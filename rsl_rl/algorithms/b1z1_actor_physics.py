"""Actor-facing task-predictive physics, separate from observed-transition PINNs."""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .hard_pact_bard import differentiable_bard_rollout_loss
from . import b1z1_ee_stability, b1z1_actor_rejection
from . import b1z1_actor_sampling as sampling


def enabled(cfg):
    return (cfg.get("actor_phys_enabled", False) and cfg.get("actor_phys_coef", 0.0) > 0
            and cfg.get("actor_phys_sample_fraction", 1.0) > 0)


def pos_fk_enabled(cfg):
    return cfg.get("actor_phys_pos_fk_enabled", False) and cfg.get("actor_phys_pos_fk_weight", 0.) > 0


def manipulability_enabled(cfg):
    return (cfg.get("actor_phys_arm_manipulability_enabled", False)
            and cfg.get("actor_phys_arm_manipulability_weight", .02) > 0)


def arm_manipulability(predicted, cfg):
    """Penalize small translational singular values, not the force ellipsoid."""
    zero = predicted.new_zeros(())
    names = ("raw", "weighted", "sigma_min", "violation_fraction")
    metrics = {"arm_manipulability_" + name: zero for name in names}
    if not manipulability_enabled(cfg):
        return zero, metrics
    from legged_gym.envs.b1z1.z1_arm_kinematics import compute_z1_arm_jacobian
    ids = torch.as_tensor(cfg["actor_phys_arm_dof_indices"], device=predicted.device, dtype=torch.long)
    # Use the live BARD successor, never the detached measured configuration.
    joints = predicted[:, 7:26].index_select(1, ids)
    geometry = [predicted.new_tensor(cfg["actor_phys_arm_" + name]).detach()
                for name in ("joint_offsets", "joint_axes", "link00_offset", "ee_offset")]
    jacobian = compute_z1_arm_jacobian(joints, *geometry)
    sigma = torch.linalg.svdvals(jacobian)[..., -1]
    threshold = cfg["actor_phys_arm_manipulability_sigma_min"]
    raw = ((threshold - sigma) / threshold).clamp_min(0).square().mean()
    weighted = cfg["actor_phys_arm_manipulability_weight"] * raw
    values = (raw, weighted, sigma.mean(), (sigma < threshold).to(predicted.dtype).mean())
    metrics.update({"arm_manipulability_" + name: value.detach()
                    for name, value in zip(names, values)})
    return weighted, metrics


def position_fk(a, actions, data):
    """Direct command-to-FK graph: no predicted successor or forward dynamics."""
    ids = torch.as_tensor(a.cfg["actor_phys_arm_indices"], device=actions.device, dtype=torch.long)
    count = a.cfg["actor_phys_num_actions"]
    position = actions[:, :count].clamp(-a.cfg["clip_actions"], a.cfg["clip_actions"])
    scale = actions.new_tensor(a.cfg["position_action_scale"])
    desired = data["fk_default"].detach()[:, :count] + position * scale
    joints = data["fk_default"].detach().clone()
    joints = joints.index_copy(1, ids, desired.index_select(1, ids))
    prediction, reference = a.dynamics_backend.commanded_ee_in_frame(
        data["fk_base_pos"].detach(), data["fk_base_quat"].detach(), joints,
        data["ee_target"].detach(), a.cfg["actor_phys_arm_root_frame"])
    finite = torch.isfinite(prediction).all(-1) & torch.isfinite(reference).all(-1)
    error = prediction[finite] - reference[finite]
    tolerance = a.cfg.get("actor_phys_pos_fk_deadband", 0.)
    residual = error.sign() * (error.abs() - tolerance).clamp_min(0.)
    residual = residual * actions.new_tensor(a.cfg.get("actor_phys_pos_fk_axis_weights", [1.,1.,1.])) / a.cfg["actor_phys_ee_scale"]
    loss = F.huber_loss(residual, torch.zeros_like(residual), reduction="none",
                        delta=a.cfg.get("actor_phys_pos_fk_huber_delta", 1.)).mean(-1)
    raw = loss.mean() if finite.any() else actions[torch.isfinite(actions)].sum()*0.
    weighted = a.cfg["actor_phys_pos_fk_weight"] * raw
    metrics = {"pos_fk_raw": raw.detach(), "pos_fk_weighted": weighted.detach(),
               "pos_fk_finite_count": finite.sum(), "pos_fk_nonfinite_count": (~finite).sum(),
               "pos_fk_error_m": error.norm(dim=-1).mean() if finite.any() else raw.detach()}
    for i, axis in enumerate("xyz"):
        metrics[f"pos_fk_error_{axis}_m"] = error[:, i].abs().mean() if finite.any() else raw.detach()
    return weighted, metrics


def scheduled_coefficient(a):
    """Reuse the checkpoint-aware PINN ramp, retaining the actor's own maximum."""
    maximum = abs(a.cfg.get("pinn_loss_weight", 0.0))
    if not enabled(a.cfg) or maximum == 0:
        return 0.0
    return a.cfg["actor_phys_coef"] * min(1.0, max(0.0, a.pinn_weight / maximum))


def configure(algorithm):
    """Fail early rather than silently substituting a different physics backend."""
    sampling.validate_fraction(algorithm.cfg)
    if not enabled(algorithm.cfg):
        return
    if not algorithm.bard_auxiliary:
        raise ValueError("actor_phys_enabled requires dynamics_backend='bard'")
    cfg = algorithm.cfg
    allocation = cfg.get("actor_phys_force_allocation_weight", 0.)
    if not math.isfinite(allocation) or allocation < 0:
        raise ValueError("actor_phys_force_allocation_weight must be finite and nonnegative")
    for name in ("arm", "base"):
        if cfg.get(f"actor_phys_{name}_rejection_enabled", False):
            weight = cfg[f"actor_phys_{name}_rejection_weight"]
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Rejection weights must be finite and nonnegative")
            scale = cfg[f"actor_phys_{name}_force_scale"]
            axes = torch.as_tensor(cfg[f"actor_phys_{name}_force_axis_weights"])
            if not math.isfinite(scale) or scale <= 0 or axes.shape != (3 if name == "arm" else 2,):
                raise ValueError("Invalid rejection force normalization")
            if not torch.isfinite(axes).all() or (axes < 0).any():
                raise ValueError("Rejection axis weights must be finite and nonnegative")
    if cfg.get("actor_phys_arm_rejection_enabled", False):
        if not math.isfinite(cfg["actor_phys_rejection_damping"]) or cfg["actor_phys_rejection_damping"] <= 0:
            raise ValueError("Projection damping must be finite and positive")
    for key in ("actor_phys_grf_torque_trust_radius", "actor_phys_rejection_active_force"):
        if key in cfg and (not math.isfinite(cfg[key]) or cfg[key] < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
    stability_weight = cfg.get("actor_phys_ee_stability_weight", 0.)
    if not math.isfinite(stability_weight) or stability_weight < 0:
        raise ValueError("actor_phys_ee_stability_weight must be finite and nonnegative")
    if stability_weight > 0:
        orientation = cfg.get("actor_phys_ee_stability_use_orientation", False)
        for key in (("position_radius", "rotation_radius") if orientation else ("position_radius",)):
            value = cfg["actor_phys_ee_stability_" + key]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"EE stability {key} must be positive")
        for key in ("beta", "twist_weight", "energy_weight", "rho", "energy_slack", "target_speed_threshold"):
            value = cfg["actor_phys_ee_stability_" + key]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"EE stability {key} must be nonnegative")
        if cfg["actor_phys_ee_stability_rho"] > 1:
            raise ValueError("EE stability rho must be at most one")
        for key in ("pose_weights", "twist_weights"):
            value = torch.as_tensor(cfg["actor_phys_ee_stability_" + key])
            width = 6 if orientation else 3
            value = value[:width]
            if value.shape != (width,) or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"EE stability {key} must contain {width} finite nonnegative weights")
    if manipulability_enabled(cfg):
        for key in ("weight", "sigma_min"):
            value = cfg["actor_phys_arm_manipulability_" + key]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Arm manipulability {key} must be finite and positive")
    for key in ("velocity_time_constant", "softplus_temperature", "huber_delta",
                "ee_scale", "q_scale", "qd_scale"):
        value = cfg["actor_phys_" + key]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"actor_phys_{key} must be finite and positive")
    for key in ("coef", "vel_weight", "ee_weight", "q_weight", "qd_weight", "q_margin", "qd_margin"):
        value = cfg["actor_phys_" + key]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"actor_phys_{key} must be finite and nonnegative")
    if cfg.get("actor_phys_pos_fk_enabled", False):
        values = [cfg["actor_phys_pos_fk_huber_delta"], cfg["actor_phys_pos_fk_deadband"],
                  cfg["actor_phys_pos_fk_weight"]]
        if not all(math.isfinite(v) for v in values) or values[0] <= 0 or min(values[1:]) < 0:
            raise ValueError("Position-FK requires positive Huber delta and nonnegative deadband")
        weights = torch.as_tensor(cfg["actor_phys_pos_fk_axis_weights"])
        if weights.shape != (3,) or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Position-FK axis weights must be three finite nonnegative values")


@torch.no_grad()
def capture(runner):
    """Snapshot only state_t and the next deterministic command, never x_(t+1)."""
    a = runner.alg
    if not enabled(a.cfg):
        return
    env = runner.env
    from legged_gym.envs.b1z1.force_task_utils import _compute_force_adjusted_ee_target
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import sphere2cart
    from legged_gym.utils.math_utils import quat_apply
    from .b1z1_bard_pinn import yaw_world
    state = env.get_pact_dynamics_state()
    yaw = env._get_base_yaw_quat()
    # Exactly the environment's quintic trajectory, without resampling or RNG use.
    u = ((env.goal_timer + 1) / env.traj_timesteps).clamp(0, 1)
    ratio = (10 * u**3 - 15 * u**4 + 6 * u**5).unsqueeze(-1)
    sphere = env.ee_start_sphere + ratio * (env.ee_goal_sphere - env.ee_start_sphere)
    nominal = env.get_ee_goal_spherical_center(yaw) + quat_apply(yaw, sphere2cart(sphere))
    force = yaw_world(a.actor_critic.last_context["ee_force"].detach() / a.cfg["ee_force_scale"], state[:, 3:7])
    target = _compute_force_adjusted_ee_target(env, external_force=force,
                                             nominal_target=nominal).effective_target
    def expand(value):
        return value.expand(env.num_envs, *value.shape[-1:])
    limits = env.simulator.dof_pos_limits
    if limits.ndim == 2:
        limits = limits.unsqueeze(0).expand(env.num_envs, -1, -1)
    values = dict(state=state, command=env.commands[:, :3], ee_target=target,
                  q_min=limits[..., 0], q_max=limits[..., 1],
                  qd_max=expand(env.simulator.dof_vel_limits),
                  torque_max=expand(env.simulator.torque_limits) * 1.1,
                  mass_wrench=env.get_mass_wrench_label(),
                  # Resampling next step has no deterministic target available yet.
                  valid=((env.goal_timer + 1) <= env.traj_total_timesteps).unsqueeze(-1))
    if pos_fk_enabled(a.cfg):
        values.update(fk_default=env.simulator.default_dof_pos.expand_as(env.simulator.dof_pos),
                      fk_base_pos=env.simulator.base_pos, fk_base_quat=env.simulator.base_quat)
    if "actor_phys_sample_fraction" in a.cfg:
        # Same physical active-force criterion as actor rejection, captured only for stratification.
        base = yaw_world(a.actor_critic.last_context["base_wrench"].detach()
                         / state.new_tensor(a.cfg["base_wrench_scale"]), state[:, 3:7])[:, :3]
        base = base - values["mass_wrench"][:, :3]
        threshold = a.cfg.get("actor_phys_rejection_active_force", 1.)
        values["sampling_active_force"] = ((force.norm(dim=-1) > threshold)
                                            | (base.norm(dim=-1) > threshold)).unsqueeze(-1)
    if b1z1_actor_rejection.term_weight(a, "base") > 0:
        values["stance"] = env.simulator.foot_contacts
    if a.cfg.get("actor_phys_ee_stability_weight", 0.) > 0:
        # Same compliant reference at both times; freeze force and base frame.
        u0 = (env.goal_timer / env.traj_timesteps).clamp(0, 1)
        ratio0 = (10*u0**3-15*u0**4+6*u0**5).unsqueeze(-1)
        sphere0 = env.ee_start_sphere + ratio0*(env.ee_goal_sphere-env.ee_start_sphere)
        nominal0 = env.get_ee_goal_spherical_center(yaw) + quat_apply(yaw, sphere2cart(sphere0))
        values["ee_stability_current_target"] = _compute_force_adjusted_ee_target(
            env, external_force=force, nominal_target=nominal0).effective_target
        if a.cfg.get("actor_phys_ee_stability_use_orientation", False) and hasattr(env, "default_ee_local_quat"):
            from legged_gym.utils.math_utils import quat_mul
            desired = quat_mul(yaw, env.default_ee_local_quat)
            axes = torch.eye(3, device=desired.device, dtype=desired.dtype)
            values["ee_stability_rotation"] = torch.stack([
                quat_apply(desired, axis.expand(env.num_envs, -1)) for axis in axes], -1)
    a.transition.actor_physics = {k: v.detach().to(a.device).clone() for k, v in values.items()}


@torch.no_grad()
def prepare(a, indices=None):
    """Cache pre-state mechanics independently of the representation-PINN warmup."""
    if indices is None:
        a.actor_physics_metrics = {}
    if scheduled_coefficient(a) == 0:
        return
    from .b1z1_bard_pinn import mechanics
    state = a.storage.actor_physics["state"].flatten(0, 1)
    if indices is not None:
        state = state.index_select(0, indices)
    if not len(state):
        a.actor_physics_cache = None
        return
    chunks = {}
    for raw in state.split(a.dynamics_backend.batch_capacity):
        good = torch.isfinite(raw).all(-1) & (raw[:, 3:7].norm(dim=-1) > 1e-6)
        safe = torch.where(good[:, None], raw, torch.zeros_like(raw)).clone()
        safe[~good, 6] = 1
        fixed = mechanics(a.dynamics_backend, {"rollout_initial_state": safe[:, :51], "dynamics_state": safe})
        for name, value in vars(fixed).items():
            chunks.setdefault(name, []).append(value)
    a.actor_physics_cache = SimpleNamespace(**{k: torch.cat(v) for k, v in chunks.items()})


def start_update(a, iteration):
    a.actor_physics_metrics = {}
    a.actor_physics_cache = None
    sampling.start(a, iteration)
    if enabled(a.cfg) and a.cfg.get("actor_phys_sample_fraction", 1.) == 1:
        prepare(a)  # Exact legacy prepare-once path; no extra RNG draws.


def start_epoch(a, epoch):
    selections = getattr(a, "actor_physics_selection", None)
    if selections is None:
        return
    state = a.storage.actor_physics["state"]
    ids = selections[epoch].to(state.device)
    mapping = torch.full((a.storage.steps * a.storage.num_envs,), -1, device=state.device, dtype=torch.long)
    mapping[ids] = torch.arange(len(ids), device=state.device)
    a.actor_physics_index_map = mapping
    prepare(a, ids)


def end_epoch(a):
    if getattr(a, "actor_physics_selection", None) is not None:
        a.actor_physics_cache = None
        a.actor_physics_index_map = None


def integrate_pose(initial, velocity, dt):
    """Semi-implicit pose extension of the existing one-step velocity predictor."""
    pos = initial[:, :3] + dt * velocity[:, :3]
    rotation = dt * velocity[:, 3:6]
    angle = rotation.norm(dim=-1, keepdim=True)
    xyz = rotation * (0.5 * torch.sinc(angle / (2 * torch.pi)))
    w = torch.cos(angle / 2)
    q, qw = initial[:, 3:6], initial[:, 6:7]
    # World angular velocity: left-multiply exp(dt*omega/2) by xyzw orientation.
    quat = torch.cat((w*q + qw*xyz + torch.cross(xyz, q, dim=-1), w*qw - (xyz*q).sum(-1, keepdim=True)), -1)
    quat = F.normalize(quat, dim=-1)
    joints = initial[:, 7:26] + dt * velocity[:, 6:]
    return torch.cat((pos, quat, joints, velocity), -1)


def components(predicted, ee, data, cfg):
    """Normalized per-sample task costs; all references are detached."""
    from legged_gym.utils.math_utils import quat_rotate_inverse
    state = data["state"].detach()
    current_v = quat_rotate_inverse(state[:, 3:7], state[:, 26:29])[:, :2]
    current_w = quat_rotate_inverse(state[:, 3:7], state[:, 29:32])[:, 2:3]
    current = torch.cat((current_v, current_w), -1)
    # Compare both velocities in the measured pre-step body frame.
    future = torch.cat((quat_rotate_inverse(state[:, 3:7], predicted[:, 26:29])[:, :2],
                        quat_rotate_inverse(state[:, 3:7], predicted[:, 29:32])[:, 2:3]), -1)
    alpha = -math.expm1(-cfg["dt"] / cfg["actor_phys_velocity_time_constant"])
    reference = (current + alpha * (data["command"].detach() - current)).detach()
    ve, pe = future - reference, ee - data["ee_target"].detach()
    delta, temp = cfg["actor_phys_huber_delta"], cfg["actor_phys_softplus_temperature"]
    huber = lambda error: F.huber_loss(error, torch.zeros_like(error), reduction="none", delta=delta).mean(-1)
    lo = data["q_min"].detach() + cfg["actor_phys_q_margin"]
    hi = data["q_max"].detach() - cfg["actor_phys_q_margin"]
    q = predicted[:, 7:26]
    qd_violation = predicted[:, 32:51].abs() - (data["qd_max"].detach() - cfg["actor_phys_qd_margin"])
    barrier = lambda x: (temp * F.softplus(x / temp)).square()
    values = {
        "vel": huber(ve * ve.new_tensor(cfg["base_velocity_scale"])),
        "ee": huber(pe / cfg["actor_phys_ee_scale"]),
        "q": (barrier((lo-q) / cfg["actor_phys_q_scale"]) + barrier((q-hi) / cfg["actor_phys_q_scale"])).mean(-1),
        "qd": barrier(qd_violation / cfg["actor_phys_qd_scale"]).mean(-1),
    }
    values["loss"] = sum(cfg[f"actor_phys_{name}_weight"] * value for name, value in values.items())
    values.update(velocity_error=ve.abs().mean(-1), ee_error_m=pe.norm(dim=-1),
                  position_violation_rad=(F.relu(lo-q)+F.relu(q-hi)).mean(-1),
                  velocity_violation_radps=F.relu(qd_violation).mean(-1))
    return values


def objective(a, batch, actions, context):
    """Frozen force predictions, live current actor torque, no observed successor."""
    from legged_gym.utils.math_utils import quat_apply
    from .b1z1_bard_pinn import yaw_world
    zero = actions[torch.isfinite(actions)].sum() * 0
    a.actor_physics_paths = None
    if not enabled(a.cfg):
        return zero, {"active_fraction": zero.detach()}
    batch, actions, context = sampling.select(a, batch, actions, context)
    if not len(actions):
        return zero, {"active_fraction": zero.detach()}
    data = {k[len("actor_phys_"):]: v.detach() for k, v in batch.items() if k.startswith("actor_phys_")}
    ids = batch["indices"]
    mapping = getattr(a, "actor_physics_index_map", None)
    if mapping is not None:
        ids = mapping[ids]
    fixed = {k: v[ids].detach() for k, v in vars(a.actor_physics_cache).items()}
    with torch.no_grad():
        ctx = {k: v.detach() for k, v in context.items()}
        grf = a.actor_critic.predict_grf(ctx, batch["nominal_torque"].detach()).detach()
        wrench, ee = ctx["base_wrench"], ctx["ee_force"]
    valid = ~(batch["dones"].bool() | batch["physics_invalid"].bool()).flatten()
    valid &= data["valid"].flatten().bool()
    for value in (*data.values(), *fixed.values(), grf, wrench, ee, actions, *ctx.values()):
        valid &= torch.isfinite(value).flatten(1).all(-1)
    valid &= (data["state"][:, 3:7].norm(dim=-1) > 1e-6)
    if a.cfg.get("actor_phys_require_force_gate", False) and not a.force_gate_active:
        valid &= False
    names = ("loss", "vel", "ee", "q", "qd", "velocity_error", "ee_error_m", "position_violation_rad", "velocity_violation_radps")
    metrics = {k: zero.detach() for k in names}
    # Cheap zero metrics even for invalid batches; no stability FK when disabled.
    _, stability_metrics = b1z1_ee_stability.objective(None, actions, {}, {})
    metrics.update(stability_metrics)
    _, arm_metrics = arm_manipulability(actions, {})
    metrics.update(arm_metrics)
    metrics["valid_fraction"] = valid.float().mean()
    metrics["active_fraction"] = zero.detach()
    if pos_fk_enabled(a.cfg):
        metrics.update({"pos_fk_" + name: zero.detach() for name in
                        ("raw", "weighted", "finite_count", "nonfinite_count",
                         "error_m", "error_x_m", "error_y_m", "error_z_m")})
    if not valid.any():
        return zero, metrics
    data = {k: v[valid] for k, v in data.items()}
    fixed = SimpleNamespace(**{k: v[valid] for k, v in fixed.items()})
    state = data["state"]
    # Singular/non-positive mechanics must not enter a differentiable solve.
    _, info = torch.linalg.cholesky_ex(fixed.mass_matrix)
    good = info == 0
    if not good.any():
        return zero, metrics
    data = {k: v[good] for k, v in data.items()}
    fixed = SimpleNamespace(**{k: v[good] for k, v in vars(fixed).items()})
    state = data["state"]
    grf = yaw_world(grf[valid][good] / a.cfg["grf_scale"], state[:, 3:7]).reshape(-1, 4, 3)
    wrench = yaw_world(wrench[valid][good] / state.new_tensor(a.cfg["base_wrench_scale"]), state[:, 3:7]) - data["mass_wrench"]
    lever = quat_apply(state[:, 3:7], state[:, 176:179])
    wrench = torch.cat((wrench[:, :3], wrench[:, 3:] + torch.cross(lever, wrench[:, :3], dim=-1)), -1)
    ee = yaw_world(ee[valid][good] / a.cfg["ee_force_scale"], state[:, 3:7])
    torque = b1z1_actor_rejection.bounded_torque(a, actions[valid][good], state, data["torque_max"])
    selected_batch = {name: batch[name][valid][good] for name in ("observations", "nominal_torque")
                      if name in batch}
    grf, rejection_loss, rejection_metrics = b1z1_actor_rejection.prepare(
        a, selected_batch, actions[valid][good],
        {k: v[valid][good] for k, v in ctx.items()}, data, fixed, torque, grf, ee, wrench[:, :3])
    metrics.update({"rejection_" + name: value for name, value in rejection_metrics.items()})
    # Reuse the existing analytic rollout, discarding its observed-state objective.
    # Dummy post_v is the pre-state: no observed x_next is read anywhere here.
    rollout_context = SimpleNamespace(foot_jacobians=fixed.foot_jacobians, base_jacobian=fixed.base_jacobian,
        pre_v_canonical=state[:, 26:51], post_v_canonical=state[:, 26:51],
        forward_dynamics=lambda g: torch.linalg.solve(fixed.mass_matrix, (g-fixed.bias).unsqueeze(-1)).squeeze(-1))
    mask = torch.zeros(len(state), dtype=torch.bool, device=state.device)
    rollout = differentiable_bard_rollout_loss(context=rollout_context, control_torque=torque,
        interval_grf_world=grf, applied_wrench_world=wrench, control_dt=state.new_full((len(state), 1), a.cfg["dt"]),
        additional_generalized_force=torch.einsum("bkn,bk->bn", fixed.ee_jacobian[:, :3], ee),
        push_event_mask=mask, reset_mask=mask, timeout_mask=mask, teleport_mask=mask)
    predicted = integrate_pose(state[:, :51], rollout.predicted_velocity, a.cfg["dt"])
    finite = torch.isfinite(predicted).all(-1)
    if not finite.any():
        return zero, metrics
    data = {k: v[finite] for k, v in data.items()}
    predicted = predicted[finite]
    ee_next = a.dynamics_backend.ee_position(predicted)
    finite_ee = torch.isfinite(ee_next).all(-1)
    if not finite_ee.any():
        return zero, metrics
    values = components(predicted[finite_ee], ee_next[finite_ee],
                        {k: v[finite_ee] for k, v in data.items()}, a.cfg)
    metrics.update({k: v.detach().mean() for k, v in values.items()})
    metrics["active_fraction"] = finite_ee.sum() / len(actions)
    total = values["loss"].mean() + rejection_loss
    stability_loss, stability_metrics = b1z1_ee_stability.objective(
        a.dynamics_backend, predicted[finite_ee],
        {k: v[finite_ee] for k, v in data.items()}, a.cfg)
    total = total + stability_loss
    metrics.update(stability_metrics)
    arm_loss, arm_metrics = arm_manipulability(predicted[finite_ee], a.cfg)
    total = total + arm_loss
    metrics.update(arm_metrics)
    metrics["loss"] = total.detach()
    if pos_fk_enabled(a.cfg):
        fk_loss, fk_metrics = position_fk(a, actions[valid][good][finite][finite_ee],
                                        {k: v[finite_ee] for k, v in data.items()})
        total = total + fk_loss
        metrics.update(fk_metrics)
        metrics["loss"] = total.detach()
    if getattr(a, "enable_additional_diagnostics", False) and grf.requires_grad:
        a.actor_physics_paths = (torque, grf)
    return total, metrics


def backward(a, batch, ppo_loss, actions, context):
    """Actor-owned PCGrad; diagnostic VJPs never accumulate .grad buffers."""
    weight = scheduled_coefficient(a)
    if weight == 0:
        ppo_loss.backward()
        return
    mapping = getattr(a, "actor_physics_index_map", None)
    if mapping is not None and not (mapping[batch["indices"]] >= 0).any():
        ppo_loss.backward()
        return
    loss, metrics = objective(a, batch, actions, context)
    metrics["loss_scaled"] = weight * loss.detach()
    parameters = a.ppo_parameters
    physics_grad = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    ppo_grad = torch.autograd.grad(ppo_loss, parameters, retain_graph=True, allow_unused=True)
    if getattr(a, "actor_physics_paths", None) is not None:
        torque, grf = a.actor_physics_paths
        total_vjp, contact_vjp = torch.autograd.grad(loss, (torque, grf), retain_graph=True, allow_unused=True)
        if total_vjp is not None and contact_vjp is not None:
            mediated, = torch.autograd.grad(grf, torque, contact_vjp, retain_graph=True, allow_unused=True)
            if mediated is not None:
                for name, vjp in (("direct", total_vjp-mediated), ("grf_mediated", mediated)):
                    grads = torch.autograd.grad(torque, parameters, vjp, retain_graph=True, allow_unused=True)
                    metrics[name + "_actor_gradient_norm"] = sum(
                        (g.square().sum() for g in grads if g is not None), loss.new_zeros(())).sqrt()
        a.actor_physics_paths = None
    dot = loss.new_zeros(())
    norm, pnorm = dot.clone(), dot.clone()
    for g, p in zip(physics_grad, ppo_grad):
        if g is not None:
            norm += g.square().sum()
        if p is not None:
            pnorm += p.square().sum()
        if g is not None and p is not None:
            dot += (g*p).sum()
    # Explicit estimation now belongs to the encoder optimizer, not the decoders.
    heads = list(dict.fromkeys([*a.decoder_parameters, *a.actor_critic.explicit_decoder.parameters()]))
    unexpected = torch.autograd.grad(loss, heads, retain_graph=True, allow_unused=True)
    metrics.update(actor_gradient_norm=norm.sqrt(), ppo_gradient_norm=pnorm.sqrt(),
                   actor_to_ppo_gradient_ratio=norm.sqrt() / pnorm.sqrt().clamp_min(1e-12),
                   ppo_gradient_cosine=dot / (norm*pnorm).sqrt().clamp_min(1e-12),
                   unintended_estimator_gradient_max=max((g.abs().max() for g in unexpected if g is not None), default=loss.new_zeros(())))
    if metrics["active_fraction"] > 0:
        from .b1z1_bard_pinn import auxiliary_backward
        auxiliary_backward(a, a.actor_optimizer, ppo_loss, loss, weight=weight)
    else:
        ppo_loss.backward()
    for name, value in metrics.items():
        a.actor_physics_metrics[name] = a.actor_physics_metrics.get(name, 0) + value.detach()


def finish(a, metrics, updates):
    metrics.update({"ActorPhysics/" + k: v for k, v in getattr(a, "actor_sampling_metrics", {}).items()})
    if enabled(a.cfg):
        metrics.update({"ActorPhysics/" + k: v.item() / max(updates, 1)
                        for k, v in a.actor_physics_metrics.items()})
        metrics["ActorPhysics/scheduled_coefficient"] = scheduled_coefficient(a)
        if scheduled_coefficient(a) == 0:
            metrics.update({"ActorPhysics/loss_scaled": 0.0, "ActorPhysics/active_fraction": 0.0})
        a.actor_physics_cache = None
    a.actor_physics_selection = None
    a.actor_physics_index_map = None
