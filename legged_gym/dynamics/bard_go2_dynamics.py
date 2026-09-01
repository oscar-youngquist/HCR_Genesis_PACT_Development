"""BARD Go2 adapter matching the legacy Pinocchio PINN conventions.

Genesis state is accepted in its native order (FR, FL, RR, RL), with an XYZW
quaternion and world-frame base twist.  BARD receives WXYZ, its URDF joint
order, and the body-frame floating-base twist used by Pinocchio free flyers.
Public dynamics terms are returned in simulator joint order while Jacobian
rows are LOCAL_WORLD_ALIGNED so they multiply world-frame forces directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


SIMULATOR_FOOT_ORDER = ("FR", "FL", "RR", "RL")
BARD_FOOT_ORDER = ("FR", "FL", "RR", "RL")
SIMULATOR_JOINT_ORDER = tuple(
    f"{leg}_{joint}_joint"
    for leg in SIMULATOR_FOOT_ORDER
    for joint in ("hip", "thigh", "calf")
)
BARD_JOINT_ORDER = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)


def _name_permutation(source: Sequence[str], destination: Sequence[str], what: str):
    source, destination = tuple(source), tuple(destination)
    if len(source) != len(destination) or set(source) != set(destination):
        raise ValueError(f"{what} names do not describe the same ordered set")
    return [source.index(name) for name in destination]


def _quat_wxyz_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(*quaternion.shape[:-1], 3, 3)


def simulator_state_to_bard(
    q_xyzw: torch.Tensor,
    v_world: torch.Tensor,
    simulator_joint_names: Sequence[str] = SIMULATOR_JOINT_ORDER,
    bard_joint_names: Sequence[str] = BARD_JOINT_ORDER,
):
    """Convert a Genesis state to BARD q/v conventions.

    ``v_world`` is ``[world linear, world angular, simulator-order joints]``.
    BARD/Pinocchio floating-base velocities are expressed in the body frame.
    """
    if q_xyzw.ndim != 2 or q_xyzw.shape[-1] != 19:
        raise ValueError("q_xyzw must have shape (batch,19)")
    if v_world.ndim != 2 or v_world.shape[-1] != 18:
        raise ValueError("v_world must have shape (batch,18)")
    joint_order = torch.as_tensor(
        _name_permutation(simulator_joint_names, bard_joint_names, "joint"),
        device=q_xyzw.device, dtype=torch.long,
    )
    quat_wxyz = q_xyzw[:, 3:7][:, (3, 0, 1, 2)]
    rotation_world_body = _quat_wxyz_rotation(quat_wxyz)
    world_to_body = rotation_world_body.transpose(-1, -2)
    base_body = torch.cat((
        torch.einsum("bij,bj->bi", world_to_body, v_world[:, :3]),
        torch.einsum("bij,bj->bi", world_to_body, v_world[:, 3:6]),
    ), dim=-1)
    q_bard = torch.cat((
        q_xyzw[:, :3], quat_wxyz,
        q_xyzw[:, 7:].index_select(1, joint_order),
    ), dim=-1)
    v_bard = torch.cat((
        base_body, v_world[:, 6:].index_select(1, joint_order)
    ), dim=-1)
    return q_bard, v_bard


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *vector.shape[:-1], 3, 3
    )


def build_linear_first_spatial_inertia(
    mass: torch.Tensor, com: torch.Tensor, rotational_inertia_com: torch.Tensor
) -> torch.Tensor:
    """Construct ``[linear; angular]`` spatial inertia about the link origin."""
    mass = mass.reshape(-1)
    if com.shape != (mass.numel(), 3):
        raise ValueError("com must have shape (batch,3)")
    if rotational_inertia_com.shape != (mass.numel(), 3, 3):
        raise ValueError("rotational inertia must have shape (batch,3,3)")
    skew = _skew(com)
    eye = torch.eye(3, device=com.device, dtype=com.dtype).expand(mass.numel(), -1, -1)
    weighted = mass[:, None, None]
    return torch.cat((
        torch.cat((weighted * eye, -weighted * skew), dim=-1),
        torch.cat((weighted * skew, rotational_inertia_com - weighted * (skew @ skew)), dim=-1),
    ), dim=-2)


def wrench_at_point(
    wrench_world: torch.Tensor,
    source_point_world: torch.Tensor,
    destination_point_world: torch.Tensor,
) -> torch.Tensor:
    """Express a world-aligned wrench about ``destination_point_world``."""
    force, torque = wrench_world[..., :3], wrench_world[..., 3:]
    arm = source_point_world - destination_point_world
    return torch.cat((force, torque + torch.cross(arm, force, dim=-1)), dim=-1)


class _BatchedSpatialInertia:
    """Adapter for BARD kernels that call ``I_spatial.unsqueeze(0)``."""

    def __init__(self, value: torch.Tensor):
        self.value = value

    def unsqueeze(self, dim: int):
        return self.value if dim == 0 else self.value.unsqueeze(dim)


@dataclass
class Go2DynamicsTerms:
    rnea: torch.Tensor
    foot_jacobians: torch.Tensor
    base_jacobian: torch.Tensor


class BardGo2Dynamics:
    """Batched BARD RNEA and world-aligned Jacobians for canonical Go2."""

    nq, nv = 19, 18
    supports_batched_inertial_randomization = True

    def __init__(
        self,
        urdf_path: str,
        simulator_joint_names: Sequence[str] = SIMULATOR_JOINT_ORDER,
        foot_frames: Sequence[str] = ("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
        base_frame: str = "base",
        *,
        device="cpu",
        dtype=torch.float32,
        batch_capacity=4096,
        default_joint_position=None,
        randomize_base_inertia=True,
        scale_rotational_inertia=True,
    ):
        import bard

        self.bard = bard
        self.device, self.dtype = torch.device(device), dtype
        self.batch_capacity = int(batch_capacity)
        self.randomize_base_inertia = bool(randomize_base_inertia)
        self.scale_rotational_inertia = bool(scale_rotational_inertia)
        self.model = bard.build_model_from_urdf(
            urdf_path, floating_base=True, device=self.device, dtype=dtype
        )
        if (self.model.nq, self.model.nv) != (self.nq, self.nv):
            raise RuntimeError("Go2 BARD model must have nq=19 and nv=18")
        self.data = bard.create_data(self.model, max_batch_size=self.batch_capacity)
        self.simulator_joint_names = tuple(simulator_joint_names)
        self.bard_joint_names = tuple(self.model.get_joint_names())
        self._sim_from_bard_joint = torch.tensor(
            _name_permutation(self.bard_joint_names, self.simulator_joint_names, "joint"),
            device=self.device, dtype=torch.long,
        )
        self._canonical_from_bard = torch.cat((
            torch.arange(6, device=self.device), self._sim_from_bard_joint + 6
        ))
        self.foot_frame_ids = [self.model.get_frame_id(name) for name in foot_frames]
        self.base_frame_id = self.model.get_frame_id(base_frame)
        self.base_inertia_node = int(self.model.urdf_root_idx)
        self._nominal_inertias = self.model.I_spatial.detach().clone()
        self._nominal_bard_root_inertia = self._nominal_inertias[
            self.base_inertia_node
        ].clone()
        # Pinocchio collapses the base's fixed children while BARD retains
        # them as nodes. Genesis randomizes the collapsed base body, so apply
        # the *change* in Pinocchio base inertia to BARD's root node. This
        # preserves BARD's nominal fixed-link distribution and exactly matches
        # the existing Pinocchio randomization semantics.
        import pinocchio as pin
        pin_model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        pin_base = pin_model.inertias[1]
        self.nominal_base_mass = torch.tensor(
            pin_base.mass, device=self.device, dtype=dtype
        )
        self.nominal_base_com = torch.as_tensor(
            pin_base.lever, device=self.device, dtype=dtype
        ).clone()
        self.nominal_base_inertia_com = torch.as_tensor(
            pin_base.inertia, device=self.device, dtype=dtype
        ).clone()
        self._nominal_pin_base_spatial = build_linear_first_spatial_inertia(
            self.nominal_base_mass.reshape(1),
            self.nominal_base_com.reshape(1, 3),
            self.nominal_base_inertia_com.reshape(1, 3, 3),
        )[0]
        if default_joint_position is None:
            default_joint_position = torch.zeros(12, device=self.device, dtype=dtype)
        self.default_joint_position = torch.as_tensor(
            default_joint_position, device=self.device, dtype=dtype
        )

    def _canonical(self, value):
        return value.index_select(-1, self._canonical_from_bard)

    def _install_inertias(self, batch, parameters):
        inertias = self._nominal_inertias.unsqueeze(0).expand(batch, -1, -1, -1).clone()
        added = parameters.get("added_base_mass", inertias.new_zeros(batch, 1)).reshape(-1)
        shift = parameters.get("base_com_shift", inertias.new_zeros(batch, 3))
        if not self.randomize_base_inertia:
            added, shift = torch.zeros_like(added), torch.zeros_like(shift)
        mass = (self.nominal_base_mass + added).clamp_min(1e-6)
        com = self.nominal_base_com.unsqueeze(0) + shift
        inertia_com = self.nominal_base_inertia_com.unsqueeze(0).expand(batch, -1, -1)
        if self.scale_rotational_inertia:
            inertia_com = inertia_com * (mass / self.nominal_base_mass)[:, None, None]
        randomized_pin = build_linear_first_spatial_inertia(
            mass, com, inertia_com
        )
        inertias[:, self.base_inertia_node] = (
            self._nominal_bard_root_inertia.unsqueeze(0)
            + randomized_pin
            - self._nominal_pin_base_spatial.unsqueeze(0)
        )
        self.model.I_spatial = _BatchedSpatialInertia(inertias)

    def _world_jacobian(self, frame_id):
        # Pinocchio's legacy PINN uses LOCAL_WORLD_ALIGNED: world-aligned axes
        # at the frame origin, without WORLD's translation-to-world-origin
        # adjoint term. Rotate BARD's LOCAL row blocks without translating.
        local, pose = self.bard.jacobian(
            self.model, self.data, frame_id,
            reference_frame="local", return_pose=True,
        )
        rotation = pose[:, :3, :3]
        jacobian = torch.cat((
            rotation @ local[:, :3], rotation @ local[:, 3:]
        ), dim=1)
        return jacobian.index_select(-1, self._canonical_from_bard)

    def evaluate(self, q_bard, v_bard, acceleration_bard, *, parameters=None):
        parameters = {} if parameters is None else {
            name: value.detach() for name, value in parameters.items()
        }
        batch = q_bard.shape[0]
        if batch > self.batch_capacity:
            chunks = []
            for start in range(0, batch, self.batch_capacity):
                stop = min(start + self.batch_capacity, batch)
                chunks.append(self.evaluate(
                    q_bard[start:stop], v_bard[start:stop],
                    acceleration_bard[start:stop],
                    parameters={
                        name: value[start:stop]
                        for name, value in parameters.items()
                    },
                ))
            return Go2DynamicsTerms(
                torch.cat([chunk.rnea for chunk in chunks]),
                torch.cat([chunk.foot_jacobians for chunk in chunks]),
                torch.cat([chunk.base_jacobian for chunk in chunks]),
            )
        self._install_inertias(batch, parameters)
        self.bard.update_kinematics(self.model, self.data, q_bard, v_bard)
        gravity = q_bard.new_tensor([0.0, 0.0, -9.81])
        result = self._canonical(self.bard.rnea(
            self.model, self.data, acceleration_bard, gravity=gravity
        ))
        armature = parameters.get("joint_armature")
        friction = parameters.get("joint_friction")
        stiffness = parameters.get("joint_stiffness")
        damping = parameters.get("joint_damping")
        q_joint = q_bard[:, 7:].index_select(1, self._sim_from_bard_joint)
        v_joint = self._canonical(v_bard)[:, 6:]
        a_joint = self._canonical(acceleration_bard)[:, 6:]
        def expanded(value):
            if value is None:
                return result.new_zeros(batch, 12)
            return value.expand(-1, 12) if value.shape[-1] == 1 else value
        result[:, 6:] += expanded(armature) * a_joint
        result[:, 6:] += expanded(friction) * torch.tanh(v_joint / 0.01)
        result[:, 6:] += expanded(stiffness) * (q_joint - self.default_joint_position)
        result[:, 6:] += expanded(damping) * v_joint
        foot = torch.stack([self._world_jacobian(fid)[:, :3] for fid in self.foot_frame_ids], dim=1)
        base = self._world_jacobian(self.base_frame_id)
        return Go2DynamicsTerms(result, foot, base)
