"""GPU-native BARD implementation of the B1/Z1 dynamics contract."""

from __future__ import annotations

import os
from typing import Sequence

import torch

from .whole_body_dynamics import WholeBodyDynamicsBackend, WholeBodyTerms


class _BatchedSpatialInertia:
    """Adapt BARD's static-inertia access to a prebuilt (B, links, 6, 6) tensor."""

    def __init__(self, value: torch.Tensor):
        self.value = value

    def unsqueeze(self, dim: int) -> torch.Tensor:
        # Official CRBA/RNEA/ABA all request I_spatial.unsqueeze(0) before
        # expanding across the batch. The batch already exists here.
        if dim != 0:
            return self.value.unsqueeze(dim)
        return self.value


class BardB1Z1DynamicsBackend(WholeBodyDynamicsBackend):
    """Batched differentiable B1/Z1 dynamics backed by official BARD APIs.

    BARD uses floating-base configurations [xyz, qw, qx, qy, qz, joints]
    and world-frame base twists [linear, angular]. Public inputs retain the
    simulator/Pinocchio convention [xyz, qx, qy, qz, qw, Genesis joints].
    Forces use LOCAL_WORLD_ALIGNED coordinates, matching Pinocchio.

    Official BARD v0.3 exposes one spatial inertia per link. A narrow adapter
    supplies a vectorized per-environment inertia batch to its unchanged
    CRBA/RNEA/ABA implementations so base and gripper randomization are retained.
    """

    supports_batched_inertial_randomization = True

    def __init__(
        self,
        urdf_path: str,
        genesis_dof_names: Sequence[str],
        foot_frames: Sequence[str],
        ee_frame: str,
        base_frame: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        batch_capacity: int = 1024,
    ):
        try:
            import bard
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "B1Z1 PACT selected dynamics_backend='bard', but the official "
                "bard-pytorch-dynamics package is unavailable. Install it from "
                "https://github.com/YueWang996/bard-pytorch-dynamics"
            ) from exc

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"B1Z1 BARD URDF does not exist: {urdf_path}")
        self.bard = bard
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("BardB1Z1DynamicsBackend requires a CUDA training device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = dtype
        self.batch_capacity = int(batch_capacity)
        if self.batch_capacity <= 0:
            raise ValueError("BARD batch_capacity must be positive")

        self.model = bard.build_model_from_urdf(urdf_path, floating_base=True)
        if self.model.nv != 25 or self.model.nq != 26:
            raise RuntimeError(
                f"Expected floating B1Z1 BARD dimensions nq=26,nv=25; got "
                f"nq={self.model.nq},nv={self.model.nv}"
            )
        self.data = bard.create_data(self.model, max_batch_size=self.batch_capacity)

        genesis_names = list(genesis_dof_names)
        bard_names = list(self.model.get_joint_names())
        if len(genesis_names) != 19 or set(genesis_names) != set(bard_names):
            missing = sorted(set(genesis_names) - set(bard_names))
            extra = sorted(set(bard_names) - set(genesis_names))
            raise RuntimeError(
                f"BARD/Genesis B1Z1 joint mismatch; missing={missing}, extra={extra}"
            )

        # Pinocchio's randomized inertias are active-joint aggregate inertias:
        # the free-flyer aggregate for the torso and the wrist-joint aggregate
        # for the fixed gripper links. Read those nominal constants once during
        # construction; all training-time reconstruction remains batched CUDA.
        try:
            import pinocchio as pin
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "BARD B1Z1 inertia randomization requires Pinocchio at model "
                "construction to match the reference aggregate inertias."
            ) from exc
        pin_model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        pin_base_frame = pin_model.getFrameId(base_frame)
        pin_ee_frame = pin_model.getFrameId(ee_frame)
        pin_base_joint = pin_model.frames[pin_base_frame].parentJoint
        pin_gripper_joint = pin_model.frames[pin_ee_frame].parentJoint
        gripper_joint_name = pin_model.names[pin_gripper_joint]
        gripper_bard_joint = bard_names.index(gripper_joint_name)
        gripper_node = int(self.model._actuated_nodes_t[gripper_bard_joint])
        self._base_inertia_node = self.model.get_frame_id(base_frame)
        self._gripper_inertia_node = gripper_node
        base_constants = self._pin_inertia_constants(
            pin_model.inertias[pin_base_joint], dtype
        )
        gripper_constants = self._pin_inertia_constants(
            pin_model.inertias[pin_gripper_joint], dtype
        )

        self.model.to(dtype=dtype, device=self.device)
        # BARD consumes URDF order; public tensors remain in simulator order.
        self._genesis_to_bard_joints = torch.tensor(
            [genesis_names.index(name) for name in bard_names],
            dtype=torch.long,
            device=self.device,
        )
        self._canonical_from_bard = torch.tensor(
            list(range(6)) + [6 + bard_names.index(name) for name in genesis_names],
            dtype=torch.long,
            device=self.device,
        )
        self._bard_from_canonical = torch.argsort(self._canonical_from_bard)

        self.foot_frame_ids = [self._frame_id(name) for name in foot_frames]
        if len(self.foot_frame_ids) != 4:
            raise RuntimeError(f"Expected four B1 feet, got {len(self.foot_frame_ids)}")
        self.ee_frame_id = self._frame_id(ee_frame)
        self.base_frame_id = self._frame_id(base_frame)
        self.gravity = torch.tensor(
            [0.0, 0.0, -9.81], device=self.device, dtype=self.dtype
        )
        self._nominal_spatial_inertias = self.model.I_spatial.detach().clone()
        self._base_nominal = tuple(
            value.to(device=self.device, dtype=self.dtype)
            for value in base_constants
        )
        self._gripper_nominal = tuple(
            value.to(device=self.device, dtype=self.dtype)
            for value in gripper_constants
        )

    def _frame_id(self, name: str) -> int:
        try:
            return self.model.get_frame_id(name)
        except (KeyError, ValueError) as exc:
            raise KeyError(f"Frame {name} is missing from B1Z1 BARD model") from exc

    def _check_batch(self, tensor: torch.Tensor) -> None:
        if tensor.device != self.device:
            raise RuntimeError(
                f"BARD input is on {tensor.device}, expected training GPU {self.device}"
            )
        if tensor.shape[0] > self.batch_capacity:
            raise RuntimeError(
                f"BARD batch {tensor.shape[0]} exceeds capacity {self.batch_capacity}"
            )

    @staticmethod
    def _pin_inertia_constants(inertia, dtype):
        return (
            torch.tensor(float(inertia.mass), dtype=dtype),
            torch.tensor(inertia.lever.tolist(), dtype=dtype),
            torch.tensor(inertia.inertia.tolist(), dtype=dtype),
        )

    @staticmethod
    def _skew(vector: torch.Tensor) -> torch.Tensor:
        x, y, z = vector.unbind(-1)
        zero = torch.zeros_like(x)
        return torch.stack((
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ), dim=-1).reshape(*vector.shape[:-1], 3, 3)

    def _spatial_inertia(self, mass, com, rotational):
        batch = mass.shape[0]
        identity = torch.eye(
            3, device=mass.device, dtype=mass.dtype
        ).expand(batch, -1, -1)
        com_skew = self._skew(com)
        mass_matrix = mass.view(batch, 1, 1)
        upper = torch.cat((
            mass_matrix * identity,
            -mass_matrix * com_skew,
        ), dim=-1)
        lower = torch.cat((
            mass_matrix * com_skew,
            rotational - mass_matrix * (com_skew @ com_skew),
        ), dim=-1)
        return torch.cat((upper, lower), dim=-2)

    def _randomized_spatial_inertias(
        self, batch, base_added_mass, base_com_shift, gripper_added_mass
    ):
        zeros_1 = self._nominal_spatial_inertias.new_zeros(batch, 1)
        zeros_3 = self._nominal_spatial_inertias.new_zeros(batch, 3)
        base_added_mass = zeros_1 if base_added_mass is None else base_added_mass
        base_com_shift = zeros_3 if base_com_shift is None else base_com_shift
        gripper_added_mass = (
            zeros_1 if gripper_added_mass is None else gripper_added_mass
        )
        inertias = self._nominal_spatial_inertias.unsqueeze(0).expand(
            batch, -1, -1, -1
        ).clone()

        def randomized_delta(constants, added_mass, com_shift):
            nominal_mass, nominal_com, nominal_rotational = constants
            mass = (nominal_mass + added_mass.squeeze(-1)).clamp_min(1.0e-6)
            com = nominal_com.unsqueeze(0) + com_shift
            rotational = (
                nominal_rotational.unsqueeze(0)
                * (mass / nominal_mass.clamp_min(1.0e-6))[:, None, None]
            )
            nominal = self._spatial_inertia(
                nominal_mass.expand(batch),
                nominal_com.expand(batch, -1),
                nominal_rotational.expand(batch, -1, -1),
            )
            return self._spatial_inertia(mass, com, rotational) - nominal

        inertias[:, self._base_inertia_node] += randomized_delta(
            self._base_nominal, base_added_mass, base_com_shift
        )
        inertias[:, self._gripper_inertia_node] += randomized_delta(
            self._gripper_nominal, gripper_added_mass, zeros_3
        )
        self.model.I_spatial = _BatchedSpatialInertia(inertias)

    def _pack_state(
        self, base_pos, base_quat_xyzw, dof_pos,
        base_linear_velocity, base_angular_velocity, dof_velocity,
    ):
        self._check_batch(base_pos)
        joints_q = dof_pos.index_select(1, self._genesis_to_bard_joints)
        joints_v = dof_velocity.index_select(1, self._genesis_to_bard_joints)
        # Simulator xyzw -> BARD wxyz; base twist remains world-aligned.
        quat_wxyz = base_quat_xyzw[:, (3, 0, 1, 2)]
        q = torch.cat((base_pos, quat_wxyz, joints_q), dim=-1)
        v = torch.cat((base_linear_velocity, base_angular_velocity, joints_v), dim=-1)
        return q, v

    def _to_canonical_vector(self, value: torch.Tensor) -> torch.Tensor:
        return value.index_select(-1, self._canonical_from_bard)

    def _to_bard_vector(self, value: torch.Tensor) -> torch.Tensor:
        return value.index_select(-1, self._bard_from_canonical)

    def _to_canonical_matrix(self, value: torch.Tensor) -> torch.Tensor:
        order = self._canonical_from_bard
        return value.index_select(-2, order).index_select(-1, order)

    def _lwa_jacobian(self, frame_id: int) -> torch.Tensor:
        # Rotate LOCAL rows into world without translating the frame origin.
        jac_local, pose = self.bard.jacobian(
            self.model, self.data, frame_id,
            reference_frame="local", return_pose=True,
        )
        rotation = pose[:, :3, :3]
        jac_lwa = torch.cat((
            rotation @ jac_local[:, :3],
            rotation @ jac_local[:, 3:],
        ), dim=1)
        return jac_lwa.index_select(-1, self._canonical_from_bard)

    def _contact_terms(self, grfs_world, ee_force_world, base_wrench_world):
        foot_jacobians = torch.stack(
            [self._lwa_jacobian(frame_id) for frame_id in self.foot_frame_ids],
            dim=1,
        )
        ee_jacobian = self._lwa_jacobian(self.ee_frame_id)
        base_jacobian = self._lwa_jacobian(self.base_frame_id)
        generalized = torch.einsum(
            "bfkn,bfk->bn", foot_jacobians[:, :, :3], grfs_world
        )
        generalized = generalized + torch.einsum(
            "bkn,bk->bn", ee_jacobian[:, :3], ee_force_world
        )
        generalized = generalized + torch.einsum(
            "bkn,bk->bn", base_jacobian, base_wrench_world
        )
        return generalized, foot_jacobians, ee_jacobian, base_jacobian

    def evaluate(
        self, base_pos, base_quat_xyzw, dof_pos, base_linear_velocity,
        base_angular_velocity, dof_velocity, grfs_world, ee_force_world,
        base_wrench_world, base_added_mass=None, base_com_shift=None,
        gripper_added_mass=None,
    ) -> WholeBodyTerms:
        self._randomized_spatial_inertias(
            base_pos.shape[0], base_added_mass, base_com_shift,
            gripper_added_mass,
        )
        q, v = self._pack_state(
            base_pos, base_quat_xyzw, dof_pos, base_linear_velocity,
            base_angular_velocity, dof_velocity,
        )
        self.bard.update_kinematics(self.model, self.data, q, v)
        mass = self._to_canonical_matrix(self.bard.crba(self.model, self.data))
        bias = self._to_canonical_vector(self.bard.rnea(
            self.model, self.data, torch.zeros_like(v), gravity=self.gravity
        ))
        contacts, foot_jacobians, ee_jacobian, base_jacobian = (
            self._contact_terms(grfs_world, ee_force_world, base_wrench_world)
        )
        return WholeBodyTerms(
            mass, bias, contacts, foot_jacobians, ee_jacobian, base_jacobian
        )

    def forward_dynamics(
        self, base_pos, base_quat_xyzw, dof_pos, base_linear_velocity,
        base_angular_velocity, dof_velocity, generalized_joint_torque,
        grfs_world, ee_force_world, base_wrench_world, base_added_mass=None,
        base_com_shift=None, gripper_added_mass=None,
    ) -> torch.Tensor:
        """Apply BARD ABA to S^T tau + J^T F and return canonical qdd."""
        self._randomized_spatial_inertias(
            base_pos.shape[0], base_added_mass, base_com_shift,
            gripper_added_mass,
        )
        q, v = self._pack_state(
            base_pos, base_quat_xyzw, dof_pos, base_linear_velocity,
            base_angular_velocity, dof_velocity,
        )
        self.bard.update_kinematics(self.model, self.data, q, v)
        contacts, _, _, _ = self._contact_terms(
            grfs_world, ee_force_world, base_wrench_world
        )
        applied = generalized_joint_torque + contacts
        acceleration = self.bard.aba(
            self.model, self.data, self._to_bard_vector(applied),
            gravity=self.gravity,
        )
        return self._to_canonical_vector(acceleration)
