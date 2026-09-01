"""HardPACT wrench-label utilities and stable transition schema."""

from __future__ import annotations

from typing import Mapping

import torch

from legged_gym.utils.math_utils import quat_apply, quat_rotate_inverse


def _world_to_yaw_local(vector_world, base_quat_xyzw):
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
    return quat_rotate_inverse(yaw_quat, vector_world)


def added_mass_gravity_wrench_world(
    added_mass_kg, gravity_world_mps2, com_shift_body_m, base_quat_xyzw
):
    """Equivalent gravity wrench of randomized mass/CoM (label only)."""
    mass = added_mass_kg.reshape(-1, 1)
    gravity = torch.as_tensor(
        gravity_world_mps2, device=mass.device, dtype=mass.dtype
    ).reshape(1, 3)
    force_world = mass * gravity
    com_world = quat_apply(base_quat_xyzw, com_shift_body_m)
    moment_world = torch.cross(com_world, force_world, dim=-1)
    return torch.cat((force_world, moment_world), dim=-1)


def wrench_world_to_scaled_yaw_local(wrench_world, base_quat_xyzw, scale):
    local = torch.cat((
        _world_to_yaw_local(wrench_world[:, :3], base_quat_xyzw),
        _world_to_yaw_local(wrench_world[:, 3:], base_quat_xyzw),
    ), dim=-1)
    return local * float(scale)


def physics_transition_mask(reset, timeout, teleport):
    return ~(reset.bool() | timeout.bool() | teleport.bool())


DISTURBANCE_FIELD_DIMS = (
    ("applied_sustained_wrench_world", 6),
    ("sustained_wrench_active_mask", 1),
    ("equivalent_mass_com_wrench_world", 6),
    ("total_external_wrench_label_world", 6),
    ("total_external_wrench_label_yaw_scaled", 6),
    ("realized_added_mass", 1),
    ("realized_com_shift_body", 3),
    ("reset_mask", 1),
    ("timeout_mask", 1),
    ("teleport_mask", 1),
    ("physics_valid_mask", 1),
)
DISTURBANCE_CRITIC_DIM = sum(width for _, width in DISTURBANCE_FIELD_DIMS)


def pack_disturbance_fields(fields: Mapping[str, torch.Tensor]):
    """Pack named transition values in the stable appended critic order."""
    packed = []
    batch_size = None
    for name, width in DISTURBANCE_FIELD_DIMS:
        value = fields[name]
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"{name} must have shape (N,{width}), got {tuple(value.shape)}")
        batch_size = value.shape[0] if batch_size is None else batch_size
        if value.shape[0] != batch_size:
            raise ValueError("disturbance fields have inconsistent batch sizes")
        packed.append(value.float())
    return torch.cat(packed, dim=-1)
