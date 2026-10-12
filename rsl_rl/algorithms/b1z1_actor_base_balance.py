"""Single-pass planar contact-force guidance through the frozen GRF decoder."""

import math

import torch
import torch.nn.functional as F


def enabled(cfg):
    from .b1z1_actor_physics import enabled as actor_enabled
    return (actor_enabled(cfg) and cfg.get("actor_phys_base_balance_enabled", False)
            and cfg.get("actor_phys_base_balance_weight", 0.) > 0)


def term_weight(a):
    if not enabled(a.cfg) or not a.cfg.get("actor_phys_live_grf_enabled", False):
        return 0.
    return a.cfg["actor_phys_base_balance_weight"] * getattr(a, "rejection_progress", 0.)


def configure(a):
    weight = a.cfg.get("actor_phys_base_balance_weight", 0.)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("actor_phys_base_balance_weight must be finite and nonnegative")
    if not enabled(a.cfg):
        return
    if not a.cfg.get("actor_phys_live_grf_enabled", False):
        raise ValueError("Base balance requires actor_phys_live_grf_enabled")
    scale = a.cfg["actor_phys_base_force_scale"]
    axes = torch.as_tensor(a.cfg["actor_phys_base_force_axis_weights"])
    if not math.isfinite(scale) or scale <= 0 or axes.shape != (2,):
        raise ValueError("Base balance requires a positive force scale and two axis weights")
    if not torch.isfinite(axes).all() or (axes < 0).any():
        raise ValueError("Base balance axis weights must be finite and nonnegative")
    contacts = a.cfg["actor_phys_rejection_min_contacts"]
    if not math.isfinite(contacts) or contacts < 1 or int(contacts) != contacts:
        raise ValueError("Base balance requires a positive integer minimum contact count")


def zero_metrics(zero):
    return {"base_balance_" + name: zero.detach() for name in (
        "raw", "weighted", "effective_weight", "active_fraction", "target_force_norm_n",
        "predicted_force_norm_n", "external_force_norm_n", "residual_norm_n",
        "residual_x_n", "residual_y_n")}


@torch.no_grad()
def contact_target(state, command, fixed, acceleration, cfg):
    """Q_contact,b* = sg(M_bb a_b_ref + M_bj a_j + h_b - Q_external,b).

    Return world-planar base rows. Reuse the rollout acceleration for joint and
    uncommanded base directions; override only commanded vx/vy/yaw acceleration
    in the same frozen body frame as the existing velocity objective. h already
    includes randomized gravity. No extra rollout, Jacobian, or solve is needed.
    """
    from legged_gym.utils.math_utils import quat_apply, quat_rotate_inverse
    state, command = state.detach(), command.detach()
    quat = state[:, 3:7]
    reference = acceleration.detach().clone()
    linear = quat_rotate_inverse(quat, reference[:, :3])
    angular = quat_rotate_inverse(quat, reference[:, 3:6])
    current_v = quat_rotate_inverse(quat, state[:, 26:29])
    current_w = quat_rotate_inverse(quat, state[:, 29:32])
    rate = -math.expm1(-cfg["dt"] / cfg["actor_phys_velocity_time_constant"]) / cfg["dt"]
    linear[:, :2] = rate * (command[:, :2] - current_v[:, :2])
    angular[:, 2] = rate * (command[:, 2] - current_w[:, 2])
    reference[:, :3] = quat_apply(quat, linear)
    reference[:, 3:6] = quat_apply(quat, angular)
    required = torch.einsum("bin,bn->bi", fixed.mass_matrix.detach()[:, :2], reference)
    required = required + fixed.bias.detach()[:, :2]

    # Applied-only world forces from the captured state_t packet, not estimates.
    ee, base = state[:, 88:91], state[:, 91:97]
    # Match the existing rollout's shift from torso CoM to its Jacobian origin.
    lever = quat_apply(quat, state[:, 176:179])
    base = torch.cat((base[:, :3], base[:, 3:] + torch.cross(lever, base[:, :3], dim=-1)), -1)
    external = torch.einsum("bki,bk->bi", fixed.base_jacobian.detach()[:, :, :2], base)
    external = external + torch.einsum("bki,bk->bi", fixed.ee_jacobian.detach()[:, :3, :2], ee)
    return required - external, external


def objective(a, actions, data, fixed, acceleration, grf):
    """Only candidate-torque -> frozen decoder -> stance GRFs carries gradients."""
    zero = actions[torch.isfinite(actions)].sum() * 0.
    weight = term_weight(a)
    if weight <= 0:
        return zero, {}
    metrics = zero_metrics(zero)
    from .b1z1_actor_physics import scheduled_coefficient
    metrics["base_balance_effective_weight"] = zero.detach() + weight * scheduled_coefficient(a)
    if not len(actions):
        return zero, metrics
    state, command = data["state"].detach(), data["command"].detach()
    stance = data["stance"].detach().bool()
    threshold = a.cfg["actor_phys_rejection_active_force"]
    gate = (state[:, 88:91].norm(dim=-1) > threshold) | (state[:, 91:94].norm(dim=-1) > threshold)
    gate &= stance.sum(-1) >= a.cfg["actor_phys_rejection_min_contacts"]
    for value in (state, command, acceleration, grf, *vars(fixed).values()):
        gate &= torch.isfinite(value.detach()).flatten(1).all(-1)
    if not gate.any():
        return zero, metrics
    from types import SimpleNamespace
    mechanics = SimpleNamespace(**{k: v[gate].detach() for k, v in vars(fixed).items()})
    target, external = contact_target(state[gate], command[gate], mechanics, acceleration[gate], a.cfg)
    prediction = torch.einsum("bfki,bfk->bi", mechanics.foot_jacobians[..., :2],
                              grf[gate] * stance[gate, :, None])
    finite = torch.isfinite(target).all(-1) & torch.isfinite(prediction).all(-1)
    metrics["base_balance_active_fraction"] = finite.sum() / len(actions)
    if not finite.any():
        return zero, metrics
    error = prediction[finite] - target[finite]
    normalized = error * error.new_tensor(a.cfg["actor_phys_base_force_axis_weights"]) / a.cfg["actor_phys_base_force_scale"]
    per_row = F.huber_loss(normalized, torch.zeros_like(normalized),
                           delta=a.cfg["actor_phys_huber_delta"], reduction="none").mean(-1)
    # Preserve rejection's gated mean over selected rows, not a conditional mean.
    raw = per_row.sum() / len(actions)
    metrics.update(base_balance_raw=raw.detach(), base_balance_weighted=(weight * raw).detach(),
                   base_balance_target_force_norm_n=target[finite].norm(dim=-1).mean(),
                   base_balance_predicted_force_norm_n=prediction[finite].detach().norm(dim=-1).mean(),
                   base_balance_external_force_norm_n=external[finite].norm(dim=-1).mean(),
                   base_balance_residual_norm_n=error.detach().norm(dim=-1).mean(),
                   base_balance_residual_x_n=error.detach()[:, 0].abs().mean(),
                   base_balance_residual_y_n=error.detach()[:, 1].abs().mean())
    return weight * raw, metrics
