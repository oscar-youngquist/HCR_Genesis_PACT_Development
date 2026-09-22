"""Canonical per-joint feedforward action units shared by B1Z1 control/training."""
import torch


def resolve_torque_action_scale(cfg, reference):
    """Scalar or learned-joint vector, with exact joint-name overrides [Nm/action]."""
    names = list(cfg.asset.dof_names[:cfg.env.num_actions])
    scale = torch.as_tensor(cfg.control.torque_scale, device=reference.device,
                            dtype=reference.dtype)
    if scale.ndim == 0:
        scale = scale.expand(len(names)).clone()
    elif scale.shape != (len(names),):
        raise ValueError(f"control.torque_scale must be scalar or length {len(names)}")
    else:
        scale = scale.clone()
    overrides = getattr(cfg.control, "torque_scale_overrides", {})
    unknown = set(overrides) - set(names)
    if unknown:
        raise ValueError(f"Torque-scale overrides must name learned joints: {sorted(unknown)}")
    for name, value in overrides.items():
        scale[names.index(name)] = value
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Torque action scales must be finite and positive")
    return scale


def simulator_torque_action_scale(simulator):
    """Cache once, including Lab's inherited Gym controller and play-only paths."""
    if not hasattr(simulator, "_torque_action_scale"):
        # A first call may occur under rollout inference_mode.
        with torch.inference_mode(False):
            simulator._torque_action_scale = resolve_torque_action_scale(
                simulator._cfg, simulator.dof_pos)
    return simulator._torque_action_scale
