"""Shared force-task utilities for B1Z1 training environments."""

from typing import NamedTuple

import torch

from legged_gym.utils.math_utils import quat_apply


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
    applied_offset: torch.Tensor
    clipped: torch.Tensor


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
    applied_offset = raw_offset * scale
    return ForceAdjustedEETarget(
        effective_target=nominal_target + applied_offset,
        raw_offset=raw_offset,
        applied_offset=applied_offset,
        clipped=(raw_norm.squeeze(-1) > max_offset),
    )


def summarize_ee_force_offset(target):
    """Return only the requested aggregate EE force-offset diagnostics."""
    return {
        "EE/raw_force_offset_norm_mean": torch.linalg.vector_norm(
            target.raw_offset, dim=-1
        ).mean(),
        "EE/applied_force_offset_norm_mean": torch.linalg.vector_norm(
            target.applied_offset, dim=-1
        ).mean(),
        "EE/force_offset_clipped_fraction": target.clipped.float().mean(),
    }


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
