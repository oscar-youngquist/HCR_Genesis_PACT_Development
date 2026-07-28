"""Pinocchio implementation of the B1/Z1 whole-body dynamics contract."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch

import pinocchio as pin

from .whole_body_dynamics import WholeBodyDynamicsBackend, WholeBodyTerms
from .b1z1_parallel_pino_workers import B1Z1PinocchioAsync


class PinocchioWholeBodyDynamics(WholeBodyDynamicsBackend):
    """Batched Pinocchio evaluator with optional persistent worker processes.

    Pinocchio evaluates the observed-state side of
    ``M vdot + h - J^T F``. The returned tensors are constants with respect to
    autograd because model evaluation happens on CPU/shared-memory workers;
    the PINN loss still differentiates through the reconstructed policy torque.
    Keeping that boundary explicit lets a future BARD backend expose the same
    API with fully differentiable model terms.
    """

    def __init__(
        self,
        urdf_path: str,
        genesis_dof_names: Sequence[str],
        foot_frames: Sequence[str],
        ee_frame: str,
        base_frame: str,
        *,
        num_workers: int = 0,
        batch_capacity: int | None = None,
        worker_start_method: str = "spawn",
    ):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"B1Z1 Pinocchio URDF does not exist: {urdf_path}")
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        if self.model.nv != 25:
            raise RuntimeError(f"Expected B1Z1 Pinocchio nv=25, got {self.model.nv}")
        self.genesis_dof_names = list(genesis_dof_names)
        self.pino_velocity_indices = []
        self.pino_position_indices = []
        for name in self.genesis_dof_names:
            jid = self.model.getJointId(name)
            if jid == 0:
                raise KeyError(f"Joint {name} is missing from Pinocchio model")
            joint = self.model.joints[jid]
            self.pino_velocity_indices.append(joint.idx_v)
            self.pino_position_indices.append(joint.idx_q)
        self.foot_frame_ids = [self._frame_id(name) for name in foot_frames]
        self.ee_frame_id = self._frame_id(ee_frame)
        self.base_frame_id = self._frame_id(base_frame)
        self.base_joint_id = self.model.frames[self.base_frame_id].parentJoint
        self.gripper_joint_id = self.model.frames[self.ee_frame_id].parentJoint
        self.nominal_base_inertia = self._clone_inertia(self.model.inertias[self.base_joint_id])
        self.nominal_gripper_inertia = self._clone_inertia(self.model.inertias[self.gripper_joint_id])
        self._async = None
        if num_workers:
            if batch_capacity is None:
                raise ValueError("Pinocchio multiprocessing requires an explicit batch_capacity")
            self._async = B1Z1PinocchioAsync(
                urdf_path,
                self.pino_position_indices,
                self.pino_velocity_indices,
                self.foot_frame_ids,
                self.ee_frame_id,
                self.base_frame_id,
                self.base_joint_id,
                self.gripper_joint_id,
                capacity=batch_capacity,
                num_workers=num_workers,
                start_method=worker_start_method,
            )

    def _frame_id(self, name: str) -> int:
        fid = self.model.getFrameId(name)
        if fid >= len(self.model.frames):
            raise KeyError(f"Frame {name} is missing from B1Z1 Pinocchio model")
        return fid

    @staticmethod
    def _clone_inertia(inertia):
        return pin.Inertia(
            float(inertia.mass),
            np.asarray(inertia.lever, dtype=np.float64).copy(),
            np.asarray(inertia.inertia, dtype=np.float64).copy(),
        )

    @staticmethod
    def _apply_mass_and_com(model, joint_id, nominal_inertia, added_mass, com_shift):
        # Genesis randomizes a link by adding dm and shifting its COM. Rebuild
        # from nominal values on every sample, then scale rotational inertia
        # with mass as the same practical approximation used by PACT workers.
        mass = max(float(nominal_inertia.mass) + float(added_mass), 1.0e-6)
        nominal_mass = max(float(nominal_inertia.mass), 1.0e-6)
        model.inertias[joint_id] = pin.Inertia(
            mass,
            np.asarray(nominal_inertia.lever, dtype=np.float64) + np.asarray(com_shift, dtype=np.float64),
            np.asarray(nominal_inertia.inertia, dtype=np.float64) * (mass / nominal_mass),
        )

    def evaluate(self, base_pos, base_quat_xyzw, dof_pos, base_linear_velocity, base_angular_velocity, dof_velocity, grfs_world, ee_force_world, base_wrench_world, base_added_mass=None, base_com_shift=None, gripper_added_mass=None):
        batch = base_pos.shape[0]
        device, dtype = base_pos.device, base_pos.dtype
        if base_added_mass is None:
            base_added_mass = torch.zeros(batch, 1, device=device, dtype=dtype)
        if base_com_shift is None:
            base_com_shift = torch.zeros(batch, 3, device=device, dtype=dtype)
        if gripper_added_mass is None:
            gripper_added_mass = torch.zeros(batch, 1, device=device, dtype=dtype)
        if self._async is not None:
            if batch > self._async.capacity:
                raise RuntimeError(
                    f"B1Z1 Pinocchio batch {batch} exceeds configured capacity {self._async.capacity}; "
                    "increase algorithm.pino_batch_capacity or reduce PPO minibatch size."
                )
            inputs = {
                "base_pos": base_pos, "base_quat": base_quat_xyzw, "dof_pos": dof_pos,
                "base_linear_velocity": base_linear_velocity,
                "base_angular_velocity": base_angular_velocity,
                "dof_velocity": dof_velocity, "grfs": grfs_world,
                "ee_force": ee_force_world, "base_wrench": base_wrench_world,
                "base_added_mass": base_added_mass, "base_com_shift": base_com_shift,
                "gripper_added_mass": gripper_added_mass,
            }
            for name, tensor in inputs.items():
                # The shared buffers are float32 because the simulator rollout
                # is float32. Detaching is intentional: Pinocchio is the
                # observed-state side of this initial consistency loss. The
                # worker receives one row per PPO sample, including its own
                # domain-randomized inertia parameters.
                np.copyto(self._async.shared.arrays[name][:batch], tensor.detach().float().cpu().numpy())
            self._async.evaluate(batch)
            return WholeBodyTerms(
                torch.as_tensor(self._async.shared.arrays["mass"][:batch].copy(), device=device, dtype=dtype),
                torch.as_tensor(self._async.shared.arrays["bias"][:batch].copy(), device=device, dtype=dtype),
                torch.as_tensor(self._async.shared.arrays["contacts"][:batch].copy(), device=device, dtype=dtype),
            )
        nv = self.model.nv
        mass = torch.empty(batch, nv, nv, device=device, dtype=dtype)
        bias = torch.empty(batch, nv, device=device, dtype=dtype)
        contacts = torch.empty(batch, nv, device=device, dtype=dtype)
        values = zip(
            base_pos.detach().cpu().numpy(), base_quat_xyzw.detach().cpu().numpy(), dof_pos.detach().cpu().numpy(),
            base_linear_velocity.detach().cpu().numpy(), base_angular_velocity.detach().cpu().numpy(), dof_velocity.detach().cpu().numpy(),
            grfs_world.detach().cpu().numpy(), ee_force_world.detach().cpu().numpy(), base_wrench_world.detach().cpu().numpy(),
        )
        for index, (p, quat, qj, vl, va, vj, grfs, ee_force, base_wrench) in enumerate(values):
            self._apply_mass_and_com(self.model, self.base_joint_id, self.nominal_base_inertia, base_added_mass[index].item(), base_com_shift[index].detach().cpu().numpy())
            self._apply_mass_and_com(self.model, self.gripper_joint_id, self.nominal_gripper_inertia, gripper_added_mass[index].item(), np.zeros(3))
            q = np.zeros(self.model.nq)
            q[:3], q[3:7] = p, quat
            q[self.pino_position_indices] = qj

            v = np.zeros(nv)
            v[:3], v[3:6] = vl, va
            v[self.pino_velocity_indices] = vj

            data = self.model.createData()
            h = pin.nonLinearEffects(self.model, data, q, v)
            m = pin.crba(self.model, data, q)

            # CRBA fills one triangle in some Pinocchio builds.
            m = np.triu(m) + np.triu(m, 1).T

            generalized = np.zeros(nv)
            for frame_id, force in zip(self.foot_frame_ids, grfs):
                jac = pin.computeFrameJacobian(self.model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                generalized += jac[:3].T @ force

            ee_jac = pin.computeFrameJacobian(self.model, data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            generalized += ee_jac[:3].T @ ee_force

            base_jac = pin.computeFrameJacobian(self.model, data, q, self.base_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            generalized += base_jac.T @ base_wrench

            mass[index] = torch.as_tensor(m, device=device, dtype=dtype)
            bias[index] = torch.as_tensor(h, device=device, dtype=dtype)

            contacts[index] = torch.as_tensor(generalized, device=device, dtype=dtype)
        return WholeBodyTerms(mass, bias, contacts)

    def close(self) -> None:
        if self._async is not None:
            self._async.close()
            self._async = None
