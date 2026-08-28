"""Shared force-task utilities for B1Z1 training environments."""

import math
from typing import NamedTuple

import torch

from legged_gym.utils.math_utils import quat_apply, quat_rotate_inverse


FORCE_TENSOR_NAMES = (
    "current_Fxyz_gripper_cmd",
    "current_Fxyz_base_cmd",
    "ee_force_ext_world",
    "base_force_ext_world",
)


class ForceAdjustedEETarget(NamedTuple):
    """One consistent view of the force-adjusted Cartesian EE target."""

    effective_target: torch.Tensor
    raw_offset: torch.Tensor
    norm_limited_offset: torch.Tensor
    applied_offset: torch.Tensor
    clipped: torch.Tensor
    projection_alpha: torch.Tensor
    workspace_projected: torch.Tensor
    projection_correction_norm: torch.Tensor
    preprojection_radius: torch.Tensor
    effective_radius: torch.Tensor


EE_FORCE_TARGET_DIAGNOSTIC_NAMES = (
    "EE/raw_force_offset_norm_mean",
    "EE/norm_limited_offset_norm_mean",
    "EE/applied_force_offset_norm_mean",
    "EE/force_offset_clipped_fraction",
    "EE/workspace_projected_fraction",
    "EE/projection_alpha_mean",
    "EE/projection_correction_norm_mean",
    "EE/preprojection_radius_mean",
    "EE/effective_radius_mean",
    "EE/tracking_l1_mean",
    "EE/tracking_l2_mean",
    "EE/relative_velocity_error_mean",
)


FORCE_CURRICULUM_LOG_NAMES = (
    "ForceCurriculum/command_scale",
    "ForceCurriculum/external_scale",
    "ForceCurriculum/stage",
    "ForceCurriculum/ee_l1_ema",
    "ForceCurriculum/roll_termination_ema",
    "ForceCurriculum/episode_length_ema",
    "ForceCurriculum/gate_patience",
    "ForceCurriculum/gate_latched",
    "ForceCurriculum/trigger_iteration",
)


class B1Z1StagedForceCurriculum:
    """Iteration-level command ramp followed by a performance-gated force ramp."""

    def __init__(self, cfg):
        self.command_start = int(cfg.force_curriculum_command_start_iteration)
        self.command_ramp = int(cfg.force_curriculum_command_ramp_iterations)
        self.gate_start = int(cfg.force_curriculum_gate_start_iteration)
        self.external_ramp = int(cfg.force_curriculum_external_ramp_iterations)
        self.ee_l1_threshold = float(cfg.force_curriculum_ee_l1_threshold)
        self.roll_threshold = float(cfg.force_curriculum_roll_termination_threshold)
        self.episode_length_threshold = float(cfg.force_curriculum_episode_length_threshold)
        self.required_patience = int(cfg.force_curriculum_gate_patience)
        self.ema_alpha = float(cfg.force_curriculum_metric_ema_alpha)
        self.use_latest_start = bool(cfg.force_curriculum_use_latest_start_fallback)
        self.latest_start = int(cfg.force_curriculum_latest_start_iteration)
        if min(self.command_start, self.command_ramp, self.gate_start, self.external_ramp) < 0:
            raise ValueError("force curriculum iteration values must be nonnegative")
        if self.gate_start < self.command_start + self.command_ramp:
            raise ValueError("force gate cannot start before the commanded-force ramp finishes")
        if self.required_patience < 1:
            raise ValueError("force_curriculum_gate_patience must be positive")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("force_curriculum_metric_ema_alpha must be in (0, 1]")
        if min(self.ee_l1_threshold, self.roll_threshold, self.episode_length_threshold) < 0.0:
            raise ValueError("force curriculum metric thresholds must be nonnegative")
        if self.use_latest_start and self.latest_start < self.gate_start:
            raise ValueError("force curriculum latest-start fallback must follow the gate start")

        self.ee_l1_ema = None
        self.roll_termination_ema = None
        self.episode_length_ema = None
        self.gate_patience = 0
        self.gate_latched = False
        self.trigger_iteration = -1
        self.last_update_iteration = -1

    @staticmethod
    def _linear_ramp(iteration, start, duration):
        if iteration < start:
            return 0.0
        if duration == 0:
            return 1.0
        return min(max((iteration - start) / duration, 0.0), 1.0)

    def command_scale(self, iteration):
        return self._linear_ramp(int(iteration), self.command_start, self.command_ramp)

    def external_scale(self, iteration):
        if not self.gate_latched:
            return 0.0
        return self._linear_ramp(int(iteration), self.trigger_iteration, self.external_ramp)

    def _update_ema(self, current, sample):
        if sample is None or not math.isfinite(float(sample)):
            return current
        sample = float(sample)
        return sample if current is None else (1.0 - self.ema_alpha) * current + self.ema_alpha * sample

    def update(self, iteration, ee_l1=None, roll_termination_rate=None, mean_episode_length=None):
        """Consume at most one aggregate metric sample for each PPO iteration."""
        iteration = int(iteration)
        if iteration == self.last_update_iteration:
            return
        if iteration < self.last_update_iteration:
            raise ValueError("force curriculum iterations must be monotonically increasing")
        self.last_update_iteration = iteration
        self.ee_l1_ema = self._update_ema(self.ee_l1_ema, ee_l1)
        self.roll_termination_ema = self._update_ema(
            self.roll_termination_ema, roll_termination_rate
        )
        self.episode_length_ema = self._update_ema(
            self.episode_length_ema, mean_episode_length
        )

        complete_sample = all(
            value is not None and math.isfinite(float(value))
            for value in (ee_l1, roll_termination_rate, mean_episode_length)
        )
        criteria_met = (
            complete_sample
            and self.ee_l1_ema is not None
            and self.roll_termination_ema is not None
            and self.episode_length_ema is not None
            and self.ee_l1_ema < self.ee_l1_threshold
            and self.roll_termination_ema < self.roll_threshold
            and self.episode_length_ema > self.episode_length_threshold
        )
        if not self.gate_latched and iteration >= self.gate_start:
            self.gate_patience = self.gate_patience + 1 if criteria_met else 0
            fallback = self.use_latest_start and iteration >= self.latest_start
            if self.gate_patience >= self.required_patience or fallback:
                self.gate_latched = True
                self.trigger_iteration = iteration

    def metrics(self, iteration):
        command_scale = self.command_scale(iteration)
        external_scale = self.external_scale(iteration)
        if int(iteration) < self.command_start:
            stage = 0
        elif int(iteration) < self.gate_start:
            stage = 1
        elif not self.gate_latched:
            stage = 2
        elif external_scale < 1.0:
            stage = 3
        else:
            stage = 4
        return {
            "ForceCurriculum/command_scale": command_scale,
            "ForceCurriculum/external_scale": external_scale,
            "ForceCurriculum/stage": float(stage),
            "ForceCurriculum/ee_l1_ema": float(self.ee_l1_ema or 0.0),
            "ForceCurriculum/roll_termination_ema": float(self.roll_termination_ema or 0.0),
            "ForceCurriculum/episode_length_ema": float(self.episode_length_ema or 0.0),
            "ForceCurriculum/gate_patience": float(self.gate_patience),
            "ForceCurriculum/gate_latched": float(self.gate_latched),
            "ForceCurriculum/trigger_iteration": float(self.trigger_iteration),
        }

    def state_dict(self):
        return {
            "ee_l1_ema": self.ee_l1_ema,
            "roll_termination_ema": self.roll_termination_ema,
            "episode_length_ema": self.episode_length_ema,
            "gate_patience": self.gate_patience,
            "gate_latched": self.gate_latched,
            "trigger_iteration": self.trigger_iteration,
            "last_update_iteration": self.last_update_iteration,
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.ee_l1_ema = state.get("ee_l1_ema")
        self.roll_termination_ema = state.get("roll_termination_ema")
        self.episode_length_ema = state.get("episode_length_ema")
        self.gate_patience = int(state.get("gate_patience", 0))
        self.gate_latched = bool(state.get("gate_latched", False))
        self.trigger_iteration = int(state.get("trigger_iteration", -1))
        self.last_update_iteration = int(state.get("last_update_iteration", -1))


def init_staged_force_curriculum(env):
    env._staged_force_curriculum = B1Z1StagedForceCurriculum(env.cfg.commands)


def update_staged_force_curriculum(env, iteration, ee_l1, roll_rate, episode_length):
    env._staged_force_curriculum.update(iteration, ee_l1, roll_rate, episode_length)
    return env._staged_force_curriculum.metrics(iteration)


def staged_force_curriculum_state_dict(env):
    return env._staged_force_curriculum.state_dict()


def load_staged_force_curriculum_state_dict(env, state):
    env._staged_force_curriculum.load_state_dict(state)


def _mean_episode_metric(ep_infos, name):
    values = [
        torch.as_tensor(info[name]).detach().float().mean()
        for info in ep_infos
        if name in info
    ]
    return torch.stack(values).mean().item() if values else None


def update_force_curriculum_from_rollout(env, iteration, ep_infos, mean_episode_length):
    """Reduce the logged episode metrics once and advance the shared gate."""
    if not hasattr(env, "_staged_force_curriculum"):
        return {}
    return update_staged_force_curriculum(
        env,
        iteration,
        _mean_episode_metric(ep_infos, "EE/tracking_l1_mean"),
        _mean_episode_metric(ep_infos, "roll_termination_rate"),
        mean_episode_length,
    )


def _workspace_coordinates(env, targets, base_yaw_quat, env_ids):
    """Transform world targets into the yaw-aligned Z1 workspace frame."""
    center = env.get_ee_goal_spherical_center(base_yaw_quat, env_ids)
    sample_count = targets.shape[1]
    expanded_quat = base_yaw_quat[:, None, :].expand(-1, sample_count, -1)
    local_targets = quat_rotate_inverse(
        expanded_quat.reshape(-1, 4),
        (targets - center[:, None, :]).reshape(-1, 3),
    )
    return local_targets.reshape_as(targets)


def _project_force_offset(env, nominal_target, limited_offset, base_yaw_quat, env_ids):
    """Select the largest valid force-offset scale with one vectorized search."""
    radius_min, radius_max = map(float, env.cfg.goal_ee.force_target_radius_limits)
    sample_count = int(env.cfg.goal_ee.force_target_projection_samples)
    if not 0.0 <= radius_min <= radius_max:
        raise ValueError("goal_ee.force_target_radius_limits must be ordered and nonnegative")
    if sample_count < 2:
        raise ValueError("goal_ee.force_target_projection_samples must be at least 2")
    configured_nominal_max = max(
        float(env.cfg.goal_ee.ranges.pos_l[1]),
        float(env.cfg.goal_ee.ranges.init_pos_start[0]),
        float(env.cfg.goal_ee.ranges.init_pos_end[0]),
    )
    if configured_nominal_max > radius_max:
        raise ValueError(
            "configured nominal EE radius exceeds force_target_radius_limits[1]"
        )

    alphas = torch.linspace(
        1.0,
        0.0,
        sample_count,
        device=limited_offset.device,
        dtype=limited_offset.dtype,
    )
    candidates = nominal_target[:, None, :] + alphas[None, :, None] * limited_offset[:, None, :]
    local_candidates = _workspace_coordinates(
        env, candidates, base_yaw_quat, env_ids
    )
    radii = torch.linalg.vector_norm(local_candidates, dim=-1)
    inside_radius = (radii >= radius_min) & (radii <= radius_max)

    lower = env.collision_lower_limits.to(
        device=local_candidates.device, dtype=local_candidates.dtype
    )
    upper = env.collision_upper_limits.to(
        device=local_candidates.device, dtype=local_candidates.dtype
    )
    inside_body = torch.all(local_candidates > lower, dim=-1) & torch.all(
        local_candidates < upper, dim=-1
    )
    above_terrain = local_candidates[..., 2] >= float(env.underground_limit)
    valid = inside_radius & ~inside_body & above_terrain

    # Nominal commands are sampled from this same valid workspace. Marking the
    # alpha=0 fallback valid avoids a reduction-side synchronization and keeps
    # the selection entirely on device.
    valid[:, -1] = True
    selected_index = valid.to(torch.int64).argmax(dim=1)
    alpha = alphas[selected_index]
    applied_offset = limited_offset * alpha.unsqueeze(-1)
    effective_target = nominal_target + applied_offset
    effective_radius = radii.gather(1, selected_index.unsqueeze(1)).squeeze(1)
    preprojection_radius = radii[:, 0]
    correction_norm = torch.linalg.vector_norm(
        limited_offset - applied_offset, dim=-1
    )
    projected = alpha < (1.0 - 1.0e-6)
    return (
        effective_target,
        applied_offset,
        alpha,
        projected,
        correction_norm,
        preprojection_radius,
        effective_radius,
    )


def _compute_force_adjusted_ee_target(env, env_ids=None):
    """Return the capped UniFP impedance-equilibrium EE target.

    UniFP contributes its yaw-frame commanded EE force. PACT and PACT-Pos do
    not expose a force-command channel, so their command term is identically
    zero while external-force handling remains otherwise identical.
    """
    base_yaw_quat = env._get_base_yaw_quat(env_ids)
    external_force = env.ee_force_ext_world
    nominal_target = env.curr_ee_goal_cart_world
    force_kp = env.gripper_force_kps
    commanded_force = getattr(env, "current_Fxyz_gripper_cmd", None)

    if env_ids is not None:
        external_force = external_force[env_ids]
        nominal_target = nominal_target[env_ids]
        force_kp = force_kp[env_ids]
        if commanded_force is not None:
            commanded_force = commanded_force[env_ids]
    if commanded_force is None:
        commanded_force = torch.zeros_like(external_force)

    force_world = external_force + quat_apply(base_yaw_quat, commanded_force)
    raw_offset = force_world / force_kp
    raw_norm = torch.linalg.vector_norm(raw_offset, dim=-1, keepdim=True)
    max_offset_value = float(env.cfg.goal_ee.max_ee_force_offset)
    if max_offset_value < 0.0:
        raise ValueError("goal_ee.max_ee_force_offset must be nonnegative")
    max_offset = raw_offset.new_tensor(max_offset_value)
    scale = torch.clamp(
        max_offset / raw_norm.clamp_min(1.0e-8),
        max=1.0,
    )
    norm_limited_offset = raw_offset * scale
    if bool(env.cfg.goal_ee.project_force_adjusted_ee_target):
        (
            effective_target,
            applied_offset,
            projection_alpha,
            workspace_projected,
            projection_correction_norm,
            preprojection_radius,
            effective_radius,
        ) = _project_force_offset(
            env,
            nominal_target,
            norm_limited_offset,
            base_yaw_quat,
            env_ids,
        )
    else:
        applied_offset = norm_limited_offset
        effective_target = nominal_target + applied_offset
        projection_alpha = raw_norm.new_ones(raw_norm.shape[0]).squeeze(-1)
        workspace_projected = torch.zeros_like(projection_alpha, dtype=torch.bool)
        projection_correction_norm = torch.zeros_like(projection_alpha)
        local_target = _workspace_coordinates(
            env, effective_target[:, None, :], base_yaw_quat, env_ids
        ).squeeze(1)
        effective_radius = torch.linalg.vector_norm(local_target, dim=-1)
        preprojection_radius = effective_radius
    return ForceAdjustedEETarget(
        effective_target=effective_target,
        raw_offset=raw_offset,
        norm_limited_offset=norm_limited_offset,
        applied_offset=applied_offset,
        clipped=(raw_norm.squeeze(-1) > max_offset),
        projection_alpha=projection_alpha,
        workspace_projected=workspace_projected,
        projection_correction_norm=projection_correction_norm,
        preprojection_radius=preprojection_radius,
        effective_radius=effective_radius,
    )


def invalidate_force_adjusted_ee_target_cache(env):
    """Invalidate the full-environment target cached for the current control step."""
    env._force_adjusted_ee_target_cache = None


def get_force_adjusted_ee_target(env, env_ids=None, *, use_cache=True):
    """Return one consistent force-adjusted target for the current control step.

    Full-batch calls share one cached projection. Subset calls are deliberately
    uncached because they are used while resetting selected environments and
    must observe the newly written reset state without contaminating the normal
    full-batch cache. ``use_cache=False`` is available for assertions and tools.
    The underlying computation remains stateless.
    """
    if env_ids is not None or not use_cache:
        return _compute_force_adjusted_ee_target(env, env_ids)
    cached = getattr(env, "_force_adjusted_ee_target_cache", None)
    if cached is None:
        cached = _compute_force_adjusted_ee_target(env)
        env._force_adjusted_ee_target_cache = cached
        env._force_adjusted_ee_target_compute_count = (
            getattr(env, "_force_adjusted_ee_target_compute_count", 0) + 1
        )
    return cached


def _per_env_ee_force_offset_diagnostics(target):
    """Return stateless force-target diagnostics before batch reduction."""
    return {
        "EE/raw_force_offset_norm_mean": torch.linalg.vector_norm(
            target.raw_offset, dim=-1
        ),
        "EE/norm_limited_offset_norm_mean": torch.linalg.vector_norm(
            target.norm_limited_offset, dim=-1
        ),
        "EE/applied_force_offset_norm_mean": torch.linalg.vector_norm(
            target.applied_offset, dim=-1
        ),
        "EE/force_offset_clipped_fraction": target.clipped.float(),
        "EE/workspace_projected_fraction": target.workspace_projected.float(),
        "EE/projection_alpha_mean": target.projection_alpha,
        "EE/projection_correction_norm_mean": target.projection_correction_norm,
        "EE/preprojection_radius_mean": target.preprojection_radius,
        "EE/effective_radius_mean": target.effective_radius,
    }


def summarize_ee_force_offset(target):
    """Return aggregate stateless diagnostics for a force-target batch."""
    return {
        name: value.mean()
        for name, value in _per_env_ee_force_offset_diagnostics(target).items()
    }


def init_ee_force_target_diagnostics(env):
    """Allocate per-environment sums; projection itself remains stateless."""
    env.ee_force_target_diagnostic_sums = {
        name: torch.zeros(env.num_envs, device=env.device)
        for name in EE_FORCE_TARGET_DIAGNOSTIC_NAMES
    }
    env.ee_force_target_diagnostic_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env.previous_projected_ee_target = torch.zeros(
        env.num_envs, 3, device=env.device
    )
    env.previous_projected_ee_target_valid = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._force_adjusted_ee_target_cache = None
    env._force_adjusted_ee_target_compute_count = 0


def accumulate_ee_force_target_diagnostics(env):
    """Accumulate one sample per control step, including target-relative speed."""
    target = get_force_adjusted_ee_target(env)
    target_velocity = (
        target.effective_target - env.previous_projected_ee_target
    ) / env.dt
    target_velocity = torch.where(
        env.previous_projected_ee_target_valid[:, None],
        target_velocity,
        torch.zeros_like(target_velocity),
    )
    tracking_error = env.simulator.ee_pos - target.effective_target
    values = _per_env_ee_force_offset_diagnostics(target)
    values.update(
        {
            "EE/tracking_l1_mean": torch.sum(torch.abs(tracking_error), dim=-1),
            "EE/tracking_l2_mean": torch.linalg.vector_norm(tracking_error, dim=-1),
            "EE/relative_velocity_error_mean": torch.linalg.vector_norm(
                env.simulator.ee_vel - target_velocity, dim=-1
            ),
        }
    )
    for name, value in values.items():
        env.ee_force_target_diagnostic_sums[name] += value
    env.ee_force_target_diagnostic_steps += 1
    env.previous_projected_ee_target.copy_(target.effective_target)
    env.previous_projected_ee_target_valid[:] = True


def summarize_accumulated_ee_force_target_diagnostics(env, env_ids):
    """Average diagnostics by each completed episode's actual control steps."""
    steps = env.ee_force_target_diagnostic_steps[env_ids].clamp_min(1).float()
    return {
        name: (total[env_ids] / steps).mean()
        for name, total in env.ee_force_target_diagnostic_sums.items()
    }


def reset_ee_force_target_diagnostics(env, env_ids):
    """Reset sums and suppress finite-difference target velocity after reset."""
    for total in env.ee_force_target_diagnostic_sums.values():
        total[env_ids] = 0.0
    env.ee_force_target_diagnostic_steps[env_ids] = 0
    env.previous_projected_ee_target[env_ids] = get_force_adjusted_ee_target(
        env, env_ids
    ).effective_target
    env.previous_projected_ee_target_valid[env_ids] = False


def force_curriculum_active(training_iteration, ramp_start_iteration):
    """Activate the force-task regime when its scale can leave the hold value."""
    return int(training_iteration) >= int(ramp_start_iteration)


def zero_velocity_probability(active, default_probability, force_probability):
    """Select the unchanged command sampler probability for the active regime."""
    return force_probability if active else default_probability


def force_neutral_mask(env, threshold=None):
    """Return environments with no available commanded or applied force task."""
    if threshold is None:
        threshold = getattr(env.cfg.rewards, "force_neutral_threshold", 1.0e-3)
    threshold = float(threshold)
    if threshold < 0.0:
        raise ValueError("rewards.force_neutral_threshold must be nonnegative")

    neutral = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for name in FORCE_TENSOR_NAMES:
        force = getattr(env, name, None)
        if force is not None:
            neutral &= torch.linalg.vector_norm(force, dim=-1) < threshold
    return neutral


def strict_standing_mask(env):
    """Require both a stationary base command and a force-neutral task."""
    return (~env.get_walking_cmd_mask()) & force_neutral_mask(env)
