"""Measured-GRF conditioning and control-interval accumulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GRFProcessingConfig:
    vertical_deadband_n: float = 3.0
    clip_min_n: float = -250.0
    clip_max_n: float = 250.0
    ema_alpha: float = 0.20
    contact_threshold_n: float = 5.0

    def validate(self) -> None:
        if self.vertical_deadband_n < 0.0:
            raise ValueError("vertical GRF deadband must be nonnegative")
        if self.clip_min_n > self.clip_max_n:
            raise ValueError("GRF clip bounds are reversed")
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("GRF EMA alpha must lie in [0,1]")
        if self.contact_threshold_n < 0.0:
            raise ValueError("contact threshold must be nonnegative")


class IntervalGRFProcessor:
    """Preserve every force stage and average conditioned substep forces.

    ``interval_average`` is the clipped/deadbanded average and deliberately
    does not use the lagged EMA. It is the only supervised/PINN force label.
    """

    def __init__(self, num_envs: int, num_feet: int, device, dtype, config: GRFProcessingConfig):
        config.validate()
        self.config = config
        shape = (num_envs, num_feet, 3)
        self.raw = torch.zeros(shape, device=device, dtype=dtype)
        self.deadbanded = torch.zeros_like(self.raw)
        self.clipped = torch.zeros_like(self.raw)
        self.ema = torch.zeros_like(self.raw)
        self.contacts = torch.zeros(shape[:-1], device=device, dtype=torch.bool)
        self.interval_sum = torch.zeros_like(self.raw)
        self.interval_average = torch.zeros_like(self.raw)
        self.interval_count = torch.zeros(num_envs, 1, 1, device=device, dtype=dtype)

    def begin_interval(self) -> None:
        self.interval_sum.zero_()
        self.interval_average.zero_()
        self.interval_count.zero_()

    def update_substep(self, raw_force_n: torch.Tensor) -> torch.Tensor:
        if raw_force_n.shape != self.raw.shape:
            raise ValueError(f"raw GRF expected {tuple(self.raw.shape)}, got {tuple(raw_force_n.shape)}")
        self.raw.copy_(raw_force_n)
        active = raw_force_n[..., 2].abs() > self.config.vertical_deadband_n
        self.deadbanded.copy_(torch.where(active.unsqueeze(-1), raw_force_n, torch.zeros_like(raw_force_n)))
        self.clipped.copy_(torch.clamp(
            self.deadbanded, min=self.config.clip_min_n, max=self.config.clip_max_n
        ))
        self.ema.mul_(1.0 - self.config.ema_alpha).add_(self.clipped, alpha=self.config.ema_alpha)
        self.contacts.copy_(self.clipped[..., 2] > self.config.contact_threshold_n)
        self.interval_sum.add_(self.clipped)
        self.interval_count.add_(1.0)
        return self.clipped

    def end_interval(self) -> torch.Tensor:
        self.interval_average.copy_(
            self.interval_sum / self.interval_count.clamp_min(1.0)
        )
        return self.interval_average

    def reset(self, env_ids: torch.Tensor) -> None:
        for value in (
            self.raw, self.deadbanded, self.clipped, self.ema,
            self.contacts, self.interval_sum, self.interval_average, self.interval_count,
        ):
            value[env_ids] = 0

    def flattened_stages(self):
        return {
            "raw": self.raw.flatten(1),
            "deadbanded": self.deadbanded.flatten(1),
            "clipped": self.clipped.flatten(1),
            "ema": self.ema.flatten(1),
            "interval_average": self.interval_average.flatten(1),
        }

