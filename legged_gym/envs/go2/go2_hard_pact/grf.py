"""Measured-GRF conditioning primitives for HardPACT."""

from dataclasses import dataclass
import torch

from legged_gym.utils.math_utils import quat_rotate_inverse


@dataclass(frozen=True)
class GRFProcessingConfig:
    vertical_deadband_n: float = 3.0
    clip_min_n: float = -250.0
    clip_max_n: float = 250.0
    ema_alpha: float = 0.20
    contact_threshold_n: float = 5.0

    def validate(self):
        if self.vertical_deadband_n < 0.0:
            raise ValueError("vertical GRF deadband must be nonnegative")
        if self.clip_min_n > self.clip_max_n:
            raise ValueError("GRF clip bounds are reversed")
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("GRF EMA alpha must lie in [0,1]")
        if self.contact_threshold_n < 0.0:
            raise ValueError("contact threshold must be nonnegative")


class IntervalGRFProcessor:
    """Preserve force stages and average conditioned physics-substep forces."""

    def __init__(self, num_envs, num_feet, device, dtype, config):
        config.validate()
        self.config = config
        shape = (num_envs, num_feet, 3)
        self.raw = torch.zeros(shape, device=device, dtype=dtype)
        self.complete = torch.zeros_like(self.raw)
        self.deadbanded = self.complete
        self.clipped = torch.zeros_like(self.raw)
        self.ema = torch.zeros_like(self.raw)
        self.contacts = torch.zeros(shape[:-1], device=device, dtype=torch.bool)
        self.interval_sum = torch.zeros_like(self.raw)
        self.interval_count = torch.zeros(num_envs, 1, 1, device=device, dtype=dtype)
        self.interval_average = torch.zeros_like(self.raw)

    def begin_interval(self):
        self.interval_sum.zero_()
        self.interval_count.zero_()
        self.interval_average.zero_()

    def update_substep(self, raw_force_n):
        if raw_force_n.shape != self.raw.shape:
            raise ValueError(
                f"raw GRF expected {tuple(self.raw.shape)}, got {tuple(raw_force_n.shape)}"
            )
        self.raw.copy_(raw_force_n)
        active = raw_force_n[..., 2].abs() > self.config.vertical_deadband_n
        self.complete.copy_(torch.where(
            active.unsqueeze(-1), raw_force_n, torch.zeros_like(raw_force_n)
        ))
        self.clipped.copy_(torch.clamp(
            self.complete, min=self.config.clip_min_n, max=self.config.clip_max_n
        ))
        self.ema.mul_(1.0 - self.config.ema_alpha).add_(
            self.clipped, alpha=self.config.ema_alpha
        )
        self.contacts.copy_(self.clipped[..., 2] > self.config.contact_threshold_n)
        self.interval_sum.add_(self.clipped)
        self.interval_count.add_(1.0)
        return self.clipped

    def end_interval(self):
        self.interval_average.copy_(
            self.interval_sum / self.interval_count.clamp_min(1.0)
        )
        return self.interval_average

    def reset(self, env_ids):
        for value in (
            self.raw, self.complete, self.clipped, self.ema, self.contacts,
            self.interval_sum, self.interval_count, self.interval_average,
        ):
            value[env_ids] = 0

    def flattened_stages(self):
        return {
            "raw": self.raw.flatten(1),
            "complete": self.complete.flatten(1),
            "deadbanded": self.deadbanded.flatten(1),
            "clipped": self.clipped.flatten(1),
            "ema": self.ema.flatten(1),
            "interval_average": self.interval_average.flatten(1),
        }


def world_to_yaw_local(vector_world, base_quat_xyzw):
    """Rotate world-frame vectors into the roll/pitch-free base-yaw frame."""
    x, y, z, w = base_quat_xyzw.unbind(-1)
    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )
    half_yaw = 0.5 * yaw
    yaw_quat = torch.stack((
        torch.zeros_like(half_yaw), torch.zeros_like(half_yaw),
        torch.sin(half_yaw), torch.cos(half_yaw),
    ), dim=-1)
    if vector_world.ndim == yaw_quat.ndim + 1:
        yaw_quat = yaw_quat.unsqueeze(-2).expand(*vector_world.shape[:-1], 4)
    return quat_rotate_inverse(yaw_quat, vector_world)
