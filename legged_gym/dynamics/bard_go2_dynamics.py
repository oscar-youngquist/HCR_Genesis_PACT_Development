"""Official-BARD Go2 dynamics with canonical ordering and batched inertias."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from legged_gym.envs.go2.go2_hard_pact.schema import (
    GO2_FOOT_NAMES,
    GO2_JOINT_NAMES,
    permutation_by_name,
)


class _BatchedSpatialInertia:
    """Adapt BARD's static-inertia access to ``(batch,links,6,6)``."""

    def __init__(self, value: torch.Tensor):
        self.value = value

    def unsqueeze(self, dim: int) -> torch.Tensor:
        return self.value if dim == 0 else self.value.unsqueeze(dim)


@dataclass
class Go2DynamicsTerms:
    mass: torch.Tensor
    bias: torch.Tensor
    foot_jacobians: torch.Tensor
    base_jacobian: torch.Tensor
    foot_jdot_v: torch.Tensor


class BardGo2Dynamics:
    """BARD dynamics for Go2 ``nq=19``, ``nv=18``.

    Public generalized vectors always use ``[base linear, base angular,
    canonical Go2 joints]``. BARD state quaternions use WXYZ; all public state
    quaternions use XYZW. Frame Jacobians are LOCAL_WORLD_ALIGNED.
    """

    nq = 19
    nv = 18
    supports_batched_inertial_randomization = True

    def __init__(
        self,
        urdf_path: str,
        simulator_joint_names: Sequence[str] = GO2_JOINT_NAMES,
        foot_frames: Sequence[str] = GO2_FOOT_NAMES,
        base_frame: str = "base",
        *,
        device="cuda",
        dtype=torch.float32,
        batch_capacity: int = 4096,
        default_joint_position: torch.Tensor | None = None,
    ):
        try:
            import bard
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Go2 HardPACT BARD losses require bard-pytorch-dynamics; install "
                "the official BARD package before enabling bard.enabled"
            ) from exc
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"Go2 URDF does not exist: {urdf_path}")
        if batch_capacity <= 0:
            raise ValueError("BARD batch_capacity must be positive")
        self.bard = bard
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_capacity = int(batch_capacity)
        self.model = bard.build_model_from_urdf(urdf_path, floating_base=True)
        if self.model.nq != self.nq or self.model.nv != self.nv:
            raise RuntimeError(
                f"expected Go2 BARD nq={self.nq},nv={self.nv}; got "
                f"nq={self.model.nq},nv={self.model.nv}"
            )
        # BARD data buffers inherit the model dtype/device at allocation time.
        self.model.to(device=self.device, dtype=dtype)
        self.data = bard.create_data(self.model, max_batch_size=self.batch_capacity)
        bard_names = tuple(self.model.get_joint_names())
        simulator_joint_names = tuple(simulator_joint_names)
        permutation_by_name(bard_names, GO2_JOINT_NAMES, "BARD Go2 joint")
        permutation_by_name(simulator_joint_names, GO2_JOINT_NAMES, "simulator Go2 joint")
        # BARD consumes its URDF order. Canonical values consume the paper order.
        self._canonical_from_bard = torch.tensor(
            list(range(6)) + [6 + bard_names.index(name) for name in GO2_JOINT_NAMES],
            device=self.device,
            dtype=torch.long,
        )
        self._bard_from_canonical = torch.argsort(self._canonical_from_bard)
        self._bard_joint_q_order = torch.tensor(
            [GO2_JOINT_NAMES.index(name) for name in bard_names],
            device=self.device,
            dtype=torch.long,
        )
        self.foot_frame_ids = [self._frame_id(name) for name in foot_frames]
        if len(self.foot_frame_ids) != 4:
            raise RuntimeError("Go2 BARD requires exactly four foot frames")
        self.base_frame_id = self._frame_id(base_frame)
        self._base_inertia_node = self.base_frame_id
        self._nominal_inertias = self.model.I_spatial.detach().clone()
        self._nominal_base_inertia = self._nominal_inertias[self._base_inertia_node].clone()
        self.gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device, dtype=dtype)
        if default_joint_position is None:
            default_joint_position = torch.zeros(12, device=self.device, dtype=dtype)
        self.default_joint_position = default_joint_position.to(self.device, self.dtype)

    def _frame_id(self, name: str) -> int:
        try:
            return self.model.get_frame_id(name)
        except (KeyError, ValueError) as exc:
            raise KeyError(f"frame {name!r} is missing from the Go2 BARD model") from exc

    def _check(self, value: torch.Tensor, width: int) -> None:
        if value.ndim != 2 or value.shape[-1] != width:
            raise ValueError(f"expected (batch,{width}), got {tuple(value.shape)}")
        if value.device != self.device:
            raise ValueError(f"BARD input is on {value.device}, expected {self.device}")
        if value.shape[0] > self.batch_capacity:
            raise ValueError("BARD batch exceeds configured capacity")

    @staticmethod
    def _skew(vector: torch.Tensor) -> torch.Tensor:
        x, y, z = vector.unbind(-1)
        zero = torch.zeros_like(x)
        return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
            *vector.shape[:-1], 3, 3
        )

    def _randomized_inertias(self, batch: int, added_mass=None, com_shift=None, armature=None):
        inertias = self._nominal_inertias.unsqueeze(0).expand(batch, -1, -1, -1).clone()
        zero_mass = inertias.new_zeros(batch, 1)
        zero_com = inertias.new_zeros(batch, 3)
        added_mass = zero_mass if added_mass is None else added_mass
        com_shift = zero_com if com_shift is None else com_shift

        # Recover spatial-inertia mass and COM from the nominal linear-first form.
        nominal = self._nominal_base_inertia
        nominal_mass = nominal[:3, :3].diagonal().mean().clamp_min(1.0e-6)
        nominal_com_skew = -nominal[:3, 3:] / nominal_mass
        nominal_com = torch.stack((
            nominal_com_skew[2, 1], nominal_com_skew[0, 2], nominal_com_skew[1, 0]
        ))
        mass = (nominal_mass + added_mass.squeeze(-1)).clamp_min(1.0e-5)
        com = nominal_com.unsqueeze(0) + com_shift
        skew = self._skew(com)
        eye = torch.eye(3, device=self.device, dtype=self.dtype).expand(batch, -1, -1)
        rotational_nominal = nominal[3:, 3:] + nominal_mass * (
            nominal_com_skew @ nominal_com_skew
        )
        rotational = rotational_nominal.unsqueeze(0) * (mass / nominal_mass)[:, None, None]
        upper = torch.cat((mass[:, None, None] * eye, -mass[:, None, None] * skew), dim=-1)
        lower = torch.cat((
            mass[:, None, None] * skew,
            rotational - mass[:, None, None] * (skew @ skew),
        ), dim=-1)
        inertias[:, self._base_inertia_node] = torch.cat((upper, lower), dim=-2)
        self.model.I_spatial = _BatchedSpatialInertia(inertias)
        self._armature = zero_mass if armature is None else armature

    def _pack_q(self, q_xyzw: torch.Tensor) -> torch.Tensor:
        self._check(q_xyzw, 19)
        position = q_xyzw[:, :3]
        quat = q_xyzw[:, 3:7][:, (3, 0, 1, 2)]
        joints = q_xyzw[:, 7:].index_select(1, self._bard_joint_q_order)
        return torch.cat((position, quat, joints), dim=-1)

    def _to_bard_v(self, value: torch.Tensor) -> torch.Tensor:
        return value.index_select(-1, self._bard_from_canonical)

    def _to_canonical_v(self, value: torch.Tensor) -> torch.Tensor:
        return value.index_select(-1, self._canonical_from_bard)

    def _to_canonical_matrix(self, value: torch.Tensor) -> torch.Tensor:
        order = self._canonical_from_bard
        return value.index_select(-2, order).index_select(-1, order)

    def _lwa_jacobian(self, frame_id: int) -> torch.Tensor:
        local, pose = self.bard.jacobian(
            self.model, self.data, frame_id, reference_frame="local", return_pose=True
        )
        rotation = pose[:, :3, :3]
        result = torch.cat((rotation @ local[:, :3], rotation @ local[:, 3:]), dim=1)
        return result.index_select(-1, self._canonical_from_bard)

    @staticmethod
    def _quat_multiply_wxyz(left, right):
        lw, lx, ly, lz = left.unbind(-1)
        rw, rx, ry, rz = right.unbind(-1)
        return torch.stack((
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ), dim=-1)

    def _integrate_bard_state(self, q, v, dt):
        result = q.clone()
        result[:, :3] = result[:, :3] + dt * v[:, :3]
        omega = v[:, 3:6]
        angle = omega.norm(dim=-1, keepdim=True) * abs(dt)
        axis = omega / omega.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        delta = torch.cat((torch.cos(0.5 * angle), axis * torch.sin(0.5 * angle) * (1 if dt >= 0 else -1)), dim=-1)
        result[:, 3:7] = self._quat_multiply_wxyz(delta, result[:, 3:7])
        result[:, 3:7] /= result[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        result[:, 7:] = result[:, 7:] + dt * v[:, 6:]
        return result

    def _jdot_v(self, q_bard, v_bard, epsilon=1.0e-4):
        """Directional finite difference of official BARD frame Jacobians."""
        plus = self._integrate_bard_state(q_bard, v_bard, epsilon)
        minus = self._integrate_bard_state(q_bard, v_bard, -epsilon)
        values = []
        for frame_id in self.foot_frame_ids:
            self.bard.update_kinematics(self.model, self.data, plus, v_bard)
            j_plus = self._lwa_jacobian(frame_id)[:, :3]
            self.bard.update_kinematics(self.model, self.data, minus, v_bard)
            j_minus = self._lwa_jacobian(frame_id)[:, :3]
            jdot = (j_plus - j_minus) / (2.0 * epsilon)
            values.append(torch.einsum("bij,bj->bi", jdot, self._to_canonical_v(v_bard)))
        self.bard.update_kinematics(self.model, self.data, q_bard, v_bard)
        return torch.stack(values, dim=1)

    def terms(self, q_xyzw, v, *, parameters: Mapping[str, torch.Tensor] | None = None):
        self._check(v, 18)
        parameters = {} if parameters is None else parameters
        self._randomized_inertias(
            q_xyzw.shape[0], parameters.get("added_base_mass"),
            parameters.get("base_com_shift"), parameters.get("joint_armature"),
        )
        q_bard = self._pack_q(q_xyzw)
        v_bard = self._to_bard_v(v)
        self.bard.update_kinematics(self.model, self.data, q_bard, v_bard)
        mass = self._to_canonical_matrix(self.bard.crba(self.model, self.data))
        if self._armature is not None:
            mass[:, 6:, 6:] += torch.diag_embed(self._armature.expand(-1, 12))
        bias = self._to_canonical_v(self.bard.rnea(
            self.model, self.data, torch.zeros_like(v_bard), gravity=self.gravity
        ))
        q_joint = q_xyzw[:, 7:]
        v_joint = v[:, 6:]
        friction = parameters.get("joint_friction")
        stiffness = parameters.get("joint_stiffness")
        damping = parameters.get("joint_damping")
        if friction is not None:
            bias[:, 6:] += friction.expand(-1, 12) * torch.tanh(v_joint / 0.01)
        if stiffness is not None:
            bias[:, 6:] += stiffness.expand(-1, 12) * (q_joint - self.default_joint_position)
        if damping is not None:
            bias[:, 6:] += damping.expand(-1, 12) * v_joint
        foot_jacobians = torch.stack(
            [self._lwa_jacobian(frame_id)[:,:3] for frame_id in self.foot_frame_ids], dim=1
        )
        base_jacobian = self._lwa_jacobian(self.base_frame_id)
        foot_jdot_v = self._jdot_v(q_bard, v_bard)
        return Go2DynamicsTerms(mass, bias, foot_jacobians, base_jacobian, foot_jdot_v)

    def rnea(self, q_xyzw, v, acceleration, *, parameters=None):
        self._check(acceleration, 18)
        terms = self.terms(q_xyzw, v, parameters=parameters)
        # Calling official RNEA keeps its differentiable implementation in the
        # path; the passive contributions already included in bias are added.
        q_bard = self._pack_q(q_xyzw)
        v_bard = self._to_bard_v(v)
        a_bard = self._to_bard_v(acceleration)
        self.bard.update_kinematics(self.model, self.data, q_bard, v_bard)
        dynamic = self._to_canonical_v(self.bard.rnea(
            self.model, self.data, a_bard, gravity=self.gravity
        ))
        official_bias = self._to_canonical_v(self.bard.rnea(
            self.model, self.data, torch.zeros_like(a_bard), gravity=self.gravity
        ))
        return dynamic + (terms.bias - official_bias)

    def aba(self, q_xyzw, v, generalized_force, *, parameters=None):
        self._check(generalized_force, 18)
        terms = self.terms(q_xyzw, v, parameters=parameters)
        # BARD ABA receives active force. Subtract extra passive terms not in
        # the URDF before calling the official differentiable implementation.
        q_bard = self._pack_q(q_xyzw)
        v_bard = self._to_bard_v(v)
        self.bard.update_kinematics(self.model, self.data, q_bard, v_bard)
        official_bias = self._to_canonical_v(self.bard.rnea(
            self.model, self.data, torch.zeros_like(v_bard), gravity=self.gravity
        ))
        adjusted = generalized_force - (terms.bias - official_bias)
        acceleration = self.bard.aba(
            self.model, self.data, self._to_bard_v(adjusted), gravity=self.gravity
        )
        return self._to_canonical_v(acceleration)
