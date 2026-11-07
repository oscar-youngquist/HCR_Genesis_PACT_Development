"""B1Z1-style per-action exploration profiles, shared by HardPACT policies."""
import torch
from torch import nn


def std_profile(value, size, name):
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.repeat(size)
    if tensor.ndim != 1 or tensor.numel() != size:
        raise ValueError(f"{name} must be a scalar or a flat list of {size} values")
    if not torch.isfinite(tensor).all() or (tensor <= 0).any():
        raise ValueError(f"{name} must contain finite positive values")
    return tensor.clone()


def initialize_action_std(module, size, initial, minimum, maximum):
    initial = std_profile(initial, size, "init_noise_std")
    lower = std_profile(minimum, size, "min_noise_std")
    upper = std_profile(maximum, size, "max_noise_std")
    if (lower > upper).any():
        raise ValueError("min_noise_std must not exceed max_noise_std")
    module.std = nn.Parameter(initial)
    # Bounds belong to the resolved training config, not learned policy state.
    # Nonpersistent buffers follow device/dtype while preserving old checkpoint
    # keys and strict Pos->coupled migration (12 versus 24 actions).
    module.register_buffer("_std_clip_lwr", lower, persistent=False)
    module.register_buffer("_std_clip_upr", upper, persistent=False)
    clip_action_std(module)


@torch.no_grad()
def clip_action_std(module):
    # Preserve legacy/B1Z1 projection semantics across repeated policy replay:
    # this is not an optimizer step between the shared graph's owned backwards.
    module.std.data.copy_(torch.maximum(torch.minimum(module.std.data, module._std_clip_upr),
                                        module._std_clip_lwr))


@torch.no_grad()
def set_std_lower(module, value):
    lower = std_profile(value, module.std.numel(), "min_noise_std").to(module.std)
    if (lower > module._std_clip_upr).any():
        raise ValueError("min_noise_std must not exceed max_noise_std")
    module._std_clip_lwr.copy_(lower)
