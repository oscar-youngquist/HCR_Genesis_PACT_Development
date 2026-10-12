"""Detached arm loading teacher; reuse the actor rollout without another solve."""

import math

import torch
import torch.nn.functional as F


def enabled(cfg, action_mode="coupled"):
    from .b1z1_actor_physics import enabled as actor_enabled
    return (actor_enabled(cfg) and action_mode == "coupled"
            and cfg.get("actor_phys_arm_feedforward_enabled", False)
            and cfg.get("actor_phys_arm_feedforward_weight", 0.) > 0)


def configure(a):
    weight = a.cfg.get("actor_phys_arm_feedforward_weight", 0.)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("actor_phys_arm_feedforward_weight must be finite and nonnegative")
    if enabled(a.cfg, a.actor_critic.action_mode):
        ids = a.cfg.get("actor_phys_arm_indices", ())
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= a.actor_critic.num_actions:
            raise ValueError("Feedforward guidance requires unique policy-controlled arm indices")


def zero_metrics(zero):
    return {"arm_feedforward_" + name: zero.detach() for name in (
        "raw", "weighted", "effective_weight", "target_norm_nm", "bias_norm_nm",
        "coupling_norm_nm", "external_norm_nm", "error_mae_nm", "valid_fraction")}


@torch.no_grad()
def compensation_target(fixed, acceleration, applied_ee_force_world, arm_ids, dof_count):
    """tau_FF* = h_a + M_au a_u - J_EE,a^T F_EE,applied, stop-gradient.

    a contains only learned arm joints; u includes the floating base and held
    wrist/gripper/other joints. h already contains gravity AND velocity bias.
    Excluding a's acceleration leaves arm motion generation to PD. Base and
    foot loads reach this tree's arm through a_u, not a second EE-force term.
    F_EE is the actual applied world-frame disturbance, not a decoder estimate.
    """
    ids = arm_ids + fixed.mass_matrix.shape[-1] - dof_count
    other_acceleration = acceleration.detach().index_fill(1, ids, 0.)
    coupling = (fixed.mass_matrix.detach().index_select(1, ids)
                @ other_acceleration.unsqueeze(-1)).squeeze(-1)
    bias = fixed.bias.detach().index_select(1, ids)
    jacobian = fixed.ee_jacobian.detach()[:, :3].index_select(2, ids)
    external = (jacobian.transpose(1, 2) @ applied_ee_force_world.detach().unsqueeze(-1)).squeeze(-1)
    return bias + coupling - external, bias, coupling, external


def objective(a, actions, state, fixed, acceleration, dof_count):
    """Only the effective physical FF output is live, never the teacher dynamics."""
    zero = actions[torch.isfinite(actions)].sum() * 0.
    if not enabled(a.cfg, a.actor_critic.action_mode):
        return zero, {}
    metrics = zero_metrics(zero)
    if not len(actions):
        return zero, metrics
    count = a.actor_critic.num_actions
    ids = torch.as_tensor(a.cfg["actor_phys_arm_indices"], device=actions.device, dtype=torch.long)
    state = state.detach()
    # get_pact_dynamics_state stores applied world-frame EE force in N here.
    # Use the captured pre-action state, never a next-state force or prediction.
    target, bias, coupling, external = compensation_target(fixed, acceleration, state[:, 88:91], ids, dof_count)
    # Match _coupled_torque: motor_strength * w_FF * S_tau * clipped a_tau.
    scale = actions.new_tensor(a.cfg["torque_action_scale"]).expand(count).index_select(0, ids)
    gain = state[:, 97:97 + dof_count].index_select(1, ids) * state[:, 155:156] * scale
    feedforward = actions[:, count:].clamp(-a.cfg["clip_actions"], a.cfg["clip_actions"])
    prediction = feedforward.index_select(1, ids) * gain
    valid = torch.isfinite(target).all(-1) & torch.isfinite(prediction).all(-1)
    # A disabled FF actuator cannot learn this teacher; keep its loss at zero.
    valid &= torch.isfinite(gain).all(-1) & (gain.abs() > 1e-8).all(-1)
    metrics["arm_feedforward_valid_fraction"] = valid.to(actions.dtype).mean()
    if not valid.any():
        return zero, metrics
    error = prediction[valid] - target[valid]
    # Normalize Nm by existing per-joint action scales, not a new uniform scale.
    raw = F.huber_loss(error / scale.abs().clamp_min(1e-8), torch.zeros_like(error),
                       delta=a.cfg["actor_phys_huber_delta"], reduction="mean")
    weight = a.cfg["actor_phys_arm_feedforward_weight"]
    from .b1z1_actor_physics import scheduled_coefficient
    metrics.update(arm_feedforward_raw=raw.detach(), arm_feedforward_weighted=(weight * raw).detach(),
                   arm_feedforward_effective_weight=zero.detach() + weight * scheduled_coefficient(a),
                   arm_feedforward_target_norm_nm=target[valid].norm(dim=-1).mean(),
                   arm_feedforward_bias_norm_nm=bias[valid].norm(dim=-1).mean(),
                   arm_feedforward_coupling_norm_nm=coupling[valid].norm(dim=-1).mean(),
                   arm_feedforward_external_norm_nm=external[valid].norm(dim=-1).mean(),
                   arm_feedforward_error_mae_nm=error.detach().abs().mean())
    return weight * raw, metrics
