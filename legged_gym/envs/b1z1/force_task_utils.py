"""Shared force-task utilities for B1Z1 training environments."""

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


def get_force_adjusted_ee_target(env, env_ids=None):
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
