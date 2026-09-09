"""Final-substep torque conditioning in canonical joint order (Nm)."""
import torch
import torch.nn.functional as F


def masked_mean(values, valid):
    per_sample = values.reshape(values.shape[0], -1).mean(-1)
    mask = valid.reshape(-1).to(per_sample.dtype)
    return torch.where(mask.bool(), per_sample, 0.).sum() / mask.sum().clamp_min(1.)


def reconstruction(prediction, target, valid, mode="mse", delta=1.):
    if mode not in ("mse", "huber") or delta <= 0:
        raise ValueError("GRF loss requires mse/huber and a positive Huber delta")
    target = target.detach()
    mse = masked_mean((prediction - target).square(), valid)
    loss = mse if mode == "mse" else masked_mean(
        F.huber_loss(prediction, target, reduction="none", delta=delta), valid)
    return loss, mse


def commanded_torque(actions, transition):
    """Replay final PD substep with cached state/gains, retaining command gradients.

    Delayed samples are excluded by the caller, never replaced by cached actions.
    """
    t = transition.detach()
    a = torch.maximum(torch.minimum(actions, t[:, 85:86]), -t[:, 85:86])
    torque = a[:, :12]*t[:, 12:24] + a[:, 12:24]*t[:, 24:36] + t[:, 36:48]
    return torch.maximum(torch.minimum(torque, t[:, 60:72]), t[:, 48:60])


def capture_substep(sim, motor_strength=None, position_only=False):
    """Capture the affine PD law before the last integration substep."""
    kp = sim._kp_scale * sim._p_gains
    kd = sim._kd_scale * sim._d_gains
    strength = sim._motor_strength if motor_strength is None else motor_strength
    fb = strength * (1.0 if position_only else sim.feedback_tau_weight)
    pos = fb * kp * sim._cfg.control.action_scale
    ff = 0.0 if position_only else strength * sim.feedforward_tau_weight * sim._cfg.control.torque_scale
    offset = fb * (kp*(sim._default_dof_pos-sim._dof_pos)-kd*sim._dof_vel)
    lower, upper = sim._robot.get_dofs_force_range(sim._dof_indices)
    shape = sim._dof_pos.shape
    expand = lambda x: torch.broadcast_to(torch.as_tensor(x, device=sim._device), shape)
    sim._grf_transition = torch.cat([expand(x) for x in (
        torch.maximum(torch.minimum(sim._torques, upper), lower), pos, ff, offset,
        lower, upper, torch.zeros_like(sim._dof_vel))] + [
        sim._grf_current_causal.reshape(-1, 1).float(),
        torch.full((shape[0], 1), sim._cfg.normalization.clip_actions, device=sim._device)], -1).detach()
    sim._grf_start_vel = sim._dof_vel.clone()
    sim._grf_start_world_lin = sim._robot.get_vel().clone()
    sim._grf_start_world_ang = sim._robot.get_ang().clone()
