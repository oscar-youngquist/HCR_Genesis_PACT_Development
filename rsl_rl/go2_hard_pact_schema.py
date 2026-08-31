"""Simulator-independent rollout field contract for Go2 HardPACT."""

from __future__ import annotations

from typing import Dict, Mapping

import torch


TRANSITION_FIELD_DIMS: Dict[str, int] = {
    "delayed_nominal_action": 24,
    "nominal_torque": 12,
    "feedback_branch_weight": 1,
    "feedforward_branch_weight": 1,
    "previous_safe_torque": 12,
    "safe_torque": 12,
    "executed_torque": 12,
    "qp_base_linear_velocity_body": 3,
    "qp_base_quaternion_xyzw": 4,
    "qp_base_angular_velocity_world": 3,
    "qp_joint_position": 12,
    "qp_joint_velocity": 12,
    "predicted_contact_probability": 4,
    # Learned predictions use cfg.normalization.obs_scales; interval labels
    # below remain physical N/Nm for validation and explicit conversion.
    "predicted_grf_yaw_scaled": 12,
    "predicted_base_wrench_yaw_scaled": 6,
    "pre_q": 19,
    "pre_v": 18,
    "post_q": 19,
    "post_v": 18,
    "average_torque": 12,
    "peak_torque": 12,
    "interval_grf_yaw": 12,
    "interval_wrench_yaw": 6,
    "instantaneous_push_delta_world": 6,
    "instantaneous_push_mask": 1,
    "sustained_wrench_world": 6,
    "added_mass_wrench_world": 6,
    "sustained_wrench_yaw_scaled": 6,
    "sustained_wrench_active_mask": 1,
    "reset_mask": 1,
    "timeout_mask": 1,
    "teleport_mask": 1,
    "physics_valid_mask": 1,
    "explicit_estimator_target": 11,
    "reconstruction_target": 79,
    "realized_randomized_parameters": 46,
    "qp_correction": 12,
    "qp_grf_yaw_scaled": 12,
    "qp_contact_slack": 12,
    "qp_residuals": 3,
    "qp_active_constraints": 1,
    "qp_fallback": 1,
    "qp_status": 1,
    "qp_forward_time_ms": 1,
}


def validate_transition(transition: Mapping[str, torch.Tensor]) -> None:
    """Validate the additional HardPACT physics fields."""
    missing = sorted(set(TRANSITION_FIELD_DIMS) - set(transition))
    if missing:
        raise KeyError(f"missing HardPACT transition fields: {missing}")
    batch = None
    for name, width in TRANSITION_FIELD_DIMS.items():
        value = transition[name]
        expected_width = width
        if value.ndim != 2 or value.shape[-1] != expected_width:
            raise ValueError(
                f"{name} must be (batch,{expected_width}), got {tuple(value.shape)}"
            )
        batch = value.shape[0] if batch is None else batch
        if value.shape[0] != batch:
            raise ValueError("transition batch sizes differ")
