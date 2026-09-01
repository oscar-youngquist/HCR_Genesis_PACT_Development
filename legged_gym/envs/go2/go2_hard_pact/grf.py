"""Measured-GRF conditioning for the legacy-compatible HardPACT aliases."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from legged_gym.utils.math_utils import quat_rotate_inverse
from rsl_rl.modules.hard_pact_physics import compose_explicit_estimator_target


@dataclass(frozen=True)
class GRFProcessingConfig:
    """Calibrated Go2 force-conditioning parameters, expressed in Newtons."""

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
    """Preserve force stages and average conditioned physics-substep forces."""

    def __init__(self, num_envs, num_feet, device, dtype, config):
        config.validate()
        self.config = config
        shape = (num_envs, num_feet, 3)
        self.raw = torch.zeros(shape, device=device, dtype=dtype)
        self.complete = torch.zeros_like(self.raw)
        # ICLR's name is retained as an alias for compatibility/documentation.
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
        self.complete.copy_(
            torch.where(active.unsqueeze(-1), raw_force_n, torch.zeros_like(raw_force_n))
        )
        self.clipped.copy_(torch.clamp(
            self.complete,
            min=self.config.clip_min_n,
            max=self.config.clip_max_n,
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
            self.raw,
            self.complete,
            self.clipped,
            self.ema,
            self.contacts,
            self.interval_sum,
            self.interval_count,
            self.interval_average,
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
    yaw_quat = torch.stack(
        (
            torch.zeros_like(half_yaw),
            torch.zeros_like(half_yaw),
            torch.sin(half_yaw),
            torch.cos(half_yaw),
        ),
        dim=-1,
    )
    if vector_world.ndim == yaw_quat.ndim + 1:
        yaw_quat = yaw_quat.unsqueeze(-2).expand(*vector_world.shape[:-1], 4)
    return quat_rotate_inverse(yaw_quat, vector_world)


class HardPACTGRFMixin:
    """Add interval GRF targets without altering legacy task tensors or logic."""

    def _init_buffers(self):
        super()._init_buffers()
        grf_cfg = self.cfg.sim.grf
        self.grf_processor = IntervalGRFProcessor(
            self.num_envs,
            len(self.simulator.feet_indices),
            self.device,
            torch.float32,
            GRFProcessingConfig(
                vertical_deadband_n=float(grf_cfg.vertical_deadband_n),
                clip_min_n=float(grf_cfg.clip_min_n),
                clip_max_n=float(grf_cfg.clip_max_n),
                ema_alpha=float(grf_cfg.ema_alpha),
                contact_threshold_n=float(grf_cfg.contact_threshold_n),
            ),
        )
        self.simulator._hard_pact_grf_post_physics_substep = (
            self._hard_pact_grf_post_physics_substep
        )

    def _hard_pact_grf_post_physics_substep(self):
        raw = self.simulator._robot.get_links_net_contact_force()[
            :, self.simulator.feet_indices, :
        ]
        self.grf_processor.update_substep(raw)

    def _update_legacy_grfs_buf_input(self):
        """Optionally replace the legacy raw GRF input with deployment EMA."""
        if bool(self.cfg.sim.grf.use_ema_grfs_buf):
            self.simulator._grfs_buf.copy_(self.grf_processor.ema.flatten(1))

    def compute_observations(self):
        # simulator.post_physics_step() refreshes _grfs_buf immediately before
        # this method, making this the precise opt-in boundary for legacy
        # critic/decoder observation construction.
        self._update_legacy_grfs_buf_input()
        result = super().compute_observations()
        clearance = (
            self.simulator.feet_pos[:, :, 2]
            - torch.mean(self.simulator.height_around_feet, dim=-1)
            - self.cfg.rewards.foot_height_offset
        )
        self.explicit_labels_buf = compose_explicit_estimator_target(
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            self.grf_processor.contacts.float(),
            clearance,
        )
        return result

    def step(self, actions):
        """Run the legacy lifecycle with a control-interval GRF target."""
        actions = self._pre_sim_step(actions)
        pre_step_base_quat = self.simulator.base_quat.clone()
        self.grf_processor.begin_interval()
        self.simulator.step(actions)
        interval_grf = world_to_yaw_local(
            self.grf_processor.end_interval(), pre_step_base_quat
        ).clone().flatten(1)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.obs_history,
            self.explicit_labels_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            interval_grf * self.obs_scales.grf,
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "grf_processor") and env_ids.numel() > 0:
            self.grf_processor.reset(env_ids)
