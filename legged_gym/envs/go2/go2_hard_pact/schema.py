"""Canonical state, transition, and named target schemas for Go2 HardPACT.

No simulator-specific tensor ordering is allowed past this module's boundary.
The canonical conventions are documented in :class:`CanonicalConvention` and
validated at runtime by the backend adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch

from rsl_rl.go2_hard_pact_schema import TRANSITION_FIELD_DIMS, validate_transition


GO2_JOINT_NAMES: Tuple[str, ...] = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)
GO2_FOOT_NAMES: Tuple[str, ...] = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")


@dataclass(frozen=True)
class CanonicalConvention:
    quaternion: str = "xyzw"
    base_twist: str = "world_linear_then_world_angular"
    generalized_velocity: str = "base_linear_base_angular_joints"
    force_frame: str = "yaw_local"
    wrench_order: str = "force_xyz_torque_xyz"
    force_units: str = "newton"
    torque_units: str = "newton_meter"
    joint_names: Tuple[str, ...] = GO2_JOINT_NAMES
    foot_names: Tuple[str, ...] = GO2_FOOT_NAMES


CANONICAL = CanonicalConvention()


def permutation_by_name(source: Sequence[str], target: Sequence[str], label: str) -> Tuple[int, ...]:
    """Return indices which reorder ``source`` values into ``target`` order."""
    source = tuple(source)
    target = tuple(target)
    if len(source) != len(set(source)):
        raise ValueError(f"{label} source names contain duplicates: {source}")
    if set(source) != set(target):
        missing = sorted(set(target) - set(source))
        extra = sorted(set(source) - set(target))
        raise ValueError(f"{label} names mismatch; missing={missing}, extra={extra}")
    return tuple(source.index(name) for name in target)


def reorder_named(value: torch.Tensor, source: Sequence[str], target: Sequence[str], label: str) -> torch.Tensor:
    order = torch.as_tensor(
        permutation_by_name(source, target, label), device=value.device, dtype=torch.long
    )
    return value.index_select(-2 if value.ndim >= 3 else -1, order)


def quat_wxyz_to_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have four coordinates")
    return quaternion[..., (1, 2, 3, 0)]


def quat_xyzw_to_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have four coordinates")
    return quaternion[..., (3, 0, 1, 2)]


def quat_conjugate_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)


def quat_apply_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by normalized XYZW quaternions without backend helpers."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    qvec = quaternion[..., :3]
    uv = torch.cross(qvec, vector, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., 3:4] * uv + uuv)


def world_to_body(vector_world: torch.Tensor, base_quat_xyzw: torch.Tensor) -> torch.Tensor:
    return quat_apply_xyzw(quat_conjugate_xyzw(base_quat_xyzw), vector_world)


def body_to_world(vector_body: torch.Tensor, base_quat_xyzw: torch.Tensor) -> torch.Tensor:
    return quat_apply_xyzw(base_quat_xyzw, vector_body)


def yaw_quaternion_xyzw(base_quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Extract a normalized, roll/pitch-free XYZW yaw quaternion."""
    x, y, z, w = base_quat_xyzw.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack((zeros, zeros, torch.sin(half), torch.cos(half)), dim=-1)


def world_to_yaw_local(vector_world: torch.Tensor, base_quat_xyzw: torch.Tensor) -> torch.Tensor:
    yaw = yaw_quaternion_xyzw(base_quat_xyzw)
    if vector_world.ndim == yaw.ndim + 1:
        yaw = yaw.unsqueeze(-2).expand(*vector_world.shape[:-1], 4)
    return quat_apply_xyzw(quat_conjugate_xyzw(yaw), vector_world)


def yaw_local_to_world(vector_local: torch.Tensor, base_quat_xyzw: torch.Tensor) -> torch.Tensor:
    yaw = yaw_quaternion_xyzw(base_quat_xyzw)
    if vector_local.ndim == yaw.ndim + 1:
        yaw = yaw.unsqueeze(-2).expand(*vector_local.shape[:-1], 4)
    return quat_apply_xyzw(yaw, vector_local)


def fixed_gravity_normal(gravity_world: torch.Tensor, force_frame_quat_xyzw: torch.Tensor | None = None) -> torch.Tensor:
    """Return ``-g/|g|`` in the QP force frame; never infer terrain normals."""
    normal_world = -gravity_world / gravity_world.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    if force_frame_quat_xyzw is None:
        return normal_world
    return world_to_body(normal_world, force_frame_quat_xyzw)


@dataclass(frozen=True)
class NamedField:
    name: str
    width: int
    scale: Tuple[float, ...]
    offset: float = 0.0

    def scale_tensor(self, like: torch.Tensor) -> torch.Tensor:
        values = self.scale if len(self.scale) > 1 else self.scale * self.width
        return like.new_tensor(values)


class Go2ReconstructionSchema:
    """The documented 79-D, non-force privileged reconstruction target."""

    fields: Tuple[NamedField, ...] = (
        NamedField("next_base_linear_velocity_body", 3, (1.0,)),
        NamedField("next_projected_gravity", 3, (1.0,)),
        NamedField("next_base_angular_velocity_body", 3, (1.0,)),
        NamedField("next_joint_position_error", 12, (1.0,)),
        NamedField("next_joint_velocity", 12, (1.0,)),
        NamedField("added_base_mass", 1, (8.0,)),
        NamedField("base_com_shift", 3, (0.20, 0.15, 0.15)),
        NamedField("kp_scale", 12, (0.20,), 1.0),
        NamedField("kd_scale", 12, (0.20,), 1.0),
        NamedField("motor_strength_scale", 12, (0.10,), 1.0),
        NamedField("joint_armature", 1, (0.03,)),
        NamedField("joint_friction", 1, (0.20,)),
        NamedField("joint_stiffness", 1, (0.02,)),
        NamedField("joint_damping", 1, (0.80,)),
        NamedField("ground_friction", 1, (1.0,)),
        NamedField("control_delay", 1, (1.0,)),
    )

    def __init__(self):
        start = 0
        slices: Dict[str, slice] = {}
        for field in self.fields:
            slices[field.name] = slice(start, start + field.width)
            start += field.width
        if start != 79:
            raise RuntimeError(f"Go2 reconstruction schema must be 79-D, got {start}")
        self.slices = slices
        self.width = start
        self.next_state_width = 33

    def _validate(self, values: Mapping[str, torch.Tensor]) -> None:
        missing = [field.name for field in self.fields if field.name not in values]
        if missing:
            raise KeyError(f"missing reconstruction fields: {missing}")
        batch = None
        for field in self.fields:
            tensor = values[field.name]
            if tensor.ndim != 2 or tensor.shape[-1] != field.width:
                raise ValueError(
                    f"{field.name} must have shape (batch,{field.width}), got {tuple(tensor.shape)}"
                )
            batch = tensor.shape[0] if batch is None else batch
            if tensor.shape[0] != batch:
                raise ValueError("all reconstruction fields must have the same batch size")

    def build(self, values: Mapping[str, torch.Tensor], *, normalized: bool = True) -> torch.Tensor:
        self._validate(values)
        blocks = []
        for field in self.fields:
            value = values[field.name]
            if normalized:
                value = (value - field.offset) / field.scale_tensor(value)
            blocks.append(value)
        return torch.cat(blocks, dim=-1)

    def unpack(self, target: torch.Tensor, *, normalized: bool = True) -> Dict[str, torch.Tensor]:
        if target.shape[-1] != self.width:
            raise ValueError(f"target must be {self.width}-D, got {target.shape[-1]}")
        result: Dict[str, torch.Tensor] = {}
        for field in self.fields:
            value = target[..., self.slices[field.name]]
            if normalized:
                value = value * field.scale_tensor(value) + field.offset
            result[field.name] = value
        return result

    def system_identification_vector(self, target: torch.Tensor) -> torch.Tensor:
        """Select named system-ID fields without positional target slicing."""
        if target.shape[-1] != self.width:
            raise ValueError(f"target must be {self.width}-D, got {target.shape[-1]}")
        return torch.cat([
            target[..., self.slices[field.name]]
            for field in self.fields
            if not field.name.startswith("next_")
        ], dim=-1)


RECONSTRUCTION_SCHEMA = Go2ReconstructionSchema()


@dataclass(frozen=True)
class RandomizedDynamicsParameters:
    """Exact realized plant constants associated with one transition.

    These values are detached simulator metadata, never decoder outputs.  The
    packed representation is only a storage format; users consume named
    tensors through :meth:`unpack` or :meth:`bard_parameters`.
    """

    added_base_mass: torch.Tensor
    base_com_shift: torch.Tensor
    kp_scale: torch.Tensor
    kd_scale: torch.Tensor
    motor_strength_scale: torch.Tensor
    joint_armature: torch.Tensor
    joint_friction: torch.Tensor
    joint_stiffness: torch.Tensor
    joint_damping: torch.Tensor
    ground_friction: torch.Tensor
    control_delay_steps: torch.Tensor

    FIELD_WIDTHS = (
        ("added_base_mass", 1), ("base_com_shift", 3),
        ("kp_scale", 12), ("kd_scale", 12),
        ("motor_strength_scale", 12), ("joint_armature", 1),
        ("joint_friction", 1), ("joint_stiffness", 1),
        ("joint_damping", 1), ("ground_friction", 1),
        ("control_delay_steps", 1),
    )
    width = 46

    def __post_init__(self):
        batch = self.added_base_mass.shape[0]
        for name, width in self.FIELD_WIDTHS:
            value = getattr(self, name)
            if value.ndim != 2 or value.shape != (batch, width):
                raise ValueError(
                    f"{name} must be ({batch},{width}), got {tuple(value.shape)}"
                )

    def detached(self) -> "RandomizedDynamicsParameters":
        return RandomizedDynamicsParameters(**{
            name: getattr(self, name).detach()
            for name, _ in self.FIELD_WIDTHS
        })

    def pack(self) -> torch.Tensor:
        return torch.cat([getattr(self, name) for name, _ in self.FIELD_WIDTHS], dim=-1)

    @classmethod
    def unpack(cls, packed: torch.Tensor) -> "RandomizedDynamicsParameters":
        if packed.ndim != 2 or packed.shape[-1] != cls.width:
            raise ValueError(f"realized parameters must be (batch,{cls.width})")
        values = {}
        start = 0
        for name, width in cls.FIELD_WIDTHS:
            values[name] = packed[:, start:start + width]
            start += width
        return cls(**values)

    def bard_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            name: getattr(self, name).detach()
            for name in (
                "added_base_mass", "base_com_shift", "joint_armature",
                "joint_friction", "joint_stiffness", "joint_damping",
            )
        }


@dataclass(frozen=True)
class QPStateEstimate:
    """Deployment-state boundary accepted by the QP/BARD builder.

    Predicted GRFs and base wrenches have already been inverse-scaled to
    physical N/Nm before entering this boundary.
    """

    base_linear_velocity_body: torch.Tensor
    base_quaternion_xyzw: torch.Tensor
    base_angular_velocity_world: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    previous_safe_torque: torch.Tensor
    contact_probability: torch.Tensor
    predicted_grf_yaw: torch.Tensor
    predicted_base_wrench_yaw: torch.Tensor

    def __post_init__(self):
        batch = self.base_linear_velocity_body.shape[0]
        widths = {
            "base_linear_velocity_body": 3, "base_quaternion_xyzw": 4,
            "base_angular_velocity_world": 3, "joint_position": 12,
            "joint_velocity": 12, "previous_safe_torque": 12,
            "contact_probability": 4, "predicted_grf_yaw": 12,
            "predicted_base_wrench_yaw": 6,
        }
        for name, width in widths.items():
            value = getattr(self, name)
            if value.ndim != 2 or value.shape != (batch, width):
                raise ValueError(f"{name} must be ({batch},{width})")

    @property
    def local_q_xyzw(self) -> torch.Tensor:
        # Floating-base dynamics are translation invariant.  Absolute simulator
        # position is intentionally unavailable at this interface.
        return torch.cat((
            torch.zeros_like(self.base_linear_velocity_body),
            self.base_quaternion_xyzw,
            self.joint_position,
        ), dim=-1)

    @property
    def velocity_world(self) -> torch.Tensor:
        return torch.cat((
            body_to_world(
                self.base_linear_velocity_body, self.base_quaternion_xyzw
            ),
            self.base_angular_velocity_world,
            self.joint_velocity,
        ), dim=-1)


def reconstruct_coupled_nominal_torque(
    coupled_action: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    default_joint_position: torch.Tensor,
    base_kp: torch.Tensor,
    base_kd: torch.Tensor,
    parameters: RandomizedDynamicsParameters,
    feedback_weight: torch.Tensor,
    feedforward_weight: torch.Tensor,
    position_scale: float,
    torque_scale: float,
    *,
    position_pretraining: bool = False,
):
    """Reconstruct the exact realized coupled PACT actuator command."""
    desired = default_joint_position + float(position_scale) * coupled_action[:, :12]
    feedback = (
        base_kp * parameters.kp_scale.detach() * (desired - joint_position)
        - base_kd * parameters.kd_scale.detach() * joint_velocity
    )
    feedforward = float(torque_scale) * coupled_action[:, 12:]
    if position_pretraining:
        feedforward = torch.zeros_like(feedforward)
    motor = parameters.motor_strength_scale.detach()
    weighted_feedback = motor * feedback_weight.detach() * feedback
    weighted_feedforward = motor * feedforward_weight.detach() * feedforward
    return (
        weighted_feedback + weighted_feedforward,
        feedback,
        feedforward,
        weighted_feedback,
        weighted_feedforward,
    )


@dataclass
class CanonicalState:
    base_position_world: torch.Tensor
    base_quaternion_xyzw: torch.Tensor
    velocity_world: torch.Tensor
    joint_position: torch.Tensor

    def validate(self) -> None:
        batch = self.base_position_world.shape[0]
        expected = {
            "base_position_world": (batch, 3),
            "base_quaternion_xyzw": (batch, 4),
            "velocity_world": (batch, 18),
            "joint_position": (batch, 12),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} expected {shape}, got {tuple(value.shape)}")

    @property
    def q_bard(self) -> torch.Tensor:
        return torch.cat((
            self.base_position_world,
            quat_xyzw_to_wxyz(self.base_quaternion_xyzw),
            self.joint_position,
        ), dim=-1)
