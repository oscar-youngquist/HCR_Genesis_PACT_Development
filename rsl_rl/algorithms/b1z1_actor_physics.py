"""Actor-facing task-predictive physics, separate from observed-transition PINNs."""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .hard_pact_bard import differentiable_bard_rollout_loss


def enabled(cfg):
    return cfg.get("actor_phys_enabled", False) and cfg.get("actor_phys_coef", 0.0) > 0


def configure(algorithm):
    """Fail early rather than silently substituting a different physics backend."""
    if not enabled(algorithm.cfg):
        return
    if not algorithm.bard_auxiliary:
        raise ValueError("actor_phys_enabled requires dynamics_backend='bard'")
    cfg = algorithm.cfg
    for key in ("velocity_time_constant", "softplus_temperature", "huber_delta",
                "ee_scale", "q_scale", "qd_scale"):
        value = cfg["actor_phys_" + key]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"actor_phys_{key} must be finite and positive")
    for key in ("coef", "vel_weight", "ee_weight", "q_weight", "qd_weight", "q_margin", "qd_margin"):
        value = cfg["actor_phys_" + key]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"actor_phys_{key} must be finite and nonnegative")


@torch.no_grad()
def capture(runner):
    """Snapshot only state_t and the next deterministic command, never x_(t+1)."""
    a, env = runner.alg, runner.env
    if not enabled(a.cfg):
        return
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
    a.transition.actor_physics = {k: v.detach().to(a.device).clone() for k, v in values.items()}


@torch.no_grad()
def prepare(a):
    """Cache pre-state mechanics independently of the representation-PINN warmup."""
    a.actor_physics_metrics = {}
    if not enabled(a.cfg):
        return
    from .b1z1_bard_pinn import mechanics
    state = a.storage.actor_physics["state"].flatten(0, 1)
    chunks = {}
    for raw in state.split(a.dynamics_backend.batch_capacity):
        good = torch.isfinite(raw).all(-1) & (raw[:, 3:7].norm(dim=-1) > 1e-6)
        safe = torch.where(good[:, None], raw, torch.zeros_like(raw)).clone()
        safe[~good, 6] = 1
        fixed = mechanics(a.dynamics_backend, {"rollout_initial_state": safe[:, :51], "dynamics_state": safe})
        for name, value in vars(fixed).items():
            chunks.setdefault(name, []).append(value)
    a.actor_physics_cache = SimpleNamespace(**{k: torch.cat(v) for k, v in chunks.items()})


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
    data = {k[len("actor_phys_"):]: v.detach() for k, v in batch.items() if k.startswith("actor_phys_")}
    fixed = {k: v[batch["indices"]].detach() for k, v in vars(a.actor_physics_cache).items()}
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
    metrics["valid_fraction"] = valid.float().mean()
    metrics["active_fraction"] = zero.detach()
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
    torque = a._coupled_torque(actions[valid][good].clamp(-a.cfg["clip_actions"], a.cfg["clip_actions"]), state)
    torque = torque.clamp(-data["torque_max"], data["torque_max"])
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
    return values["loss"].mean(), metrics


def backward(a, batch, ppo_loss, actions, context):
    """Actor-owned PCGrad; diagnostic VJPs never accumulate .grad buffers."""
    if not enabled(a.cfg):
        ppo_loss.backward()
        return
    loss, metrics = objective(a, batch, actions, context)
    metrics["loss_scaled"] = a.cfg["actor_phys_coef"] * loss.detach()
    parameters = a.ppo_parameters
    physics_grad = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    ppo_grad = torch.autograd.grad(ppo_loss, parameters, retain_graph=True, allow_unused=True)
    dot = loss.new_zeros(())
    norm, pnorm = dot.clone(), dot.clone()
    for g, p in zip(physics_grad, ppo_grad):
        if g is not None:
            norm += g.square().sum()
        if p is not None:
            pnorm += p.square().sum()
        if g is not None and p is not None:
            dot += (g*p).sum()
    unexpected = torch.autograd.grad(loss, a.decoder_parameters, retain_graph=True, allow_unused=True)
    metrics.update(actor_gradient_norm=norm.sqrt(), ppo_gradient_cosine=dot / (norm*pnorm).sqrt().clamp_min(1e-12),
                   unintended_estimator_gradient_max=max((g.abs().max() for g in unexpected if g is not None), default=loss.new_zeros(())))
    if metrics["active_fraction"] > 0:
        a.actor_optimizer.pc_backward_ppgrad([ppo_loss, a.cfg["actor_phys_coef"] * loss])
    else:
        ppo_loss.backward()
    for name, value in metrics.items():
        a.actor_physics_metrics[name] = a.actor_physics_metrics.get(name, 0) + value.detach()


def finish(a, metrics, updates):
    if enabled(a.cfg):
        metrics.update({"ActorPhysics/" + k: v.item() / max(updates, 1)
                        for k, v in a.actor_physics_metrics.items()})
        a.actor_physics_cache = None
