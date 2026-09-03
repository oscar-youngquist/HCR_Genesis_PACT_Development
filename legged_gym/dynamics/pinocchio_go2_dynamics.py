"""CPU-parallel Pinocchio adapter for the common Go2 dynamics contract.

This adapter intentionally delegates all rigid-body work to the repository's
existing :class:`PinocchioAsync` persistent worker pool. Its responsibilities
are limited to canonical ordering and the explicit GPU/CPU boundary.
"""

from __future__ import annotations

import os
import time
import multiprocessing as mp

import numpy as np
import torch

from .parallel_pino_workers import PinocchioAsync
from .bard_go2_dynamics import (
    BARD_JOINT_ORDER,
    SIMULATOR_JOINT_ORDER,
    Go2BardContext,
    simulator_state_to_bard,
)


class PinocchioGo2Dynamics:
    """Expose Pinocchio through the same context API as ``BardGo2Dynamics``."""

    nq, nv = 19, 18
    supports_batched_inertial_randomization = True

    def __init__(
        self, urdf_path, simulator_joint_names=SIMULATOR_JOINT_ORDER,
        foot_frames=("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
        base_frame="base", *, device="cpu", dtype=torch.float32,
        batch_capacity=4096, default_joint_position=None,
        randomize_base_inertia=True, scale_rotational_inertia=True,
        num_workers=None, profile_timing=False,
    ):
        import pinocchio as pin

        self.device, self.dtype = torch.device(device), dtype
        self.batch_capacity = int(batch_capacity)
        self.randomize_base_inertia = bool(randomize_base_inertia)
        self.scale_rotational_inertia = bool(scale_rotational_inertia)
        self.profile_timing = bool(profile_timing)
        self.simulator_joint_names = tuple(simulator_joint_names)
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.bard_joint_names = tuple(str(name) for name in self.model.names[2:])
        if set(self.bard_joint_names) != set(self.simulator_joint_names):
            raise ValueError("Pinocchio and simulator joint names differ")
        canonical_indices = [0, 1, 2, 3, 4, 5]
        canonical_indices.extend(
            self.model.joints[self.model.getJointId(name)].idx_v
            for name in self.simulator_joint_names
        )
        self._canonical_indices = tuple(canonical_indices)
        self._canonical_from_bard = torch.tensor(
            canonical_indices, device=self.device, dtype=torch.long
        )
        inverse = np.argsort(np.asarray(canonical_indices))
        self._bard_from_canonical = torch.tensor(
            inverse, device=self.device, dtype=torch.long
        )
        requested_workers = (
            int(os.environ.get("HARD_PACT_PINOCCHIO_WORKERS", "0"))
            if num_workers is None else int(num_workers)
        )
        # Preserve the established Go2 PACT policy: reserve approximately two
        # percent of logical CPUs for the training/simulator process.
        self.num_workers = requested_workers or max(1, int(mp.cpu_count() * 0.98))
        numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
        self.manager = PinocchioAsync(
            self.model, self.batch_capacity, tuple(foot_frames),
            canonical_indices, self.nv, (4, 3), self.num_workers,
            base_joint_id=1, shared_name_prefix="hard_pact_dynamics",
            base_frame_name=base_frame, shared_dtype=numpy_dtype,
        )
        self.default_joint_position = torch.zeros(
            12, device=self.device, dtype=dtype
        ) if default_joint_position is None else torch.as_tensor(
            default_joint_position, device=self.device, dtype=dtype
        )
        self._pin_memory = self.device.type == "cuda" and torch.cuda.is_available()
        self._staging = {}
        self._timing = {"dynamics_ms": 0.0, "transfer_ms": 0.0, "calls": 0}

    def _canonical(self, value):
        return value.index_select(-1, self._canonical_from_bard)

    def _bard_order(self, value):
        return value.index_select(-1, self._bard_from_canonical)

    def _stage(self, name, shape):
        tensor = self._staging.get(name)
        full_shape = (self.batch_capacity, *shape)
        if tensor is None:
            tensor = torch.empty(
                full_shape, dtype=self.dtype, device="cpu",
                pin_memory=self._pin_memory,
            )
            self._staging[name] = tensor
        return tensor

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _to_shared(self, name, value, batch):
        started = time.perf_counter()
        stage = self._stage(f"input_{name}", tuple(value.shape[1:]))
        stage[:batch].copy_(value.detach(), non_blocking=self._pin_memory)
        self._synchronize()
        np.copyto(getattr(self.manager.shared, name)[:batch], stage[:batch].numpy())
        self._timing["transfer_ms"] += 1000.0 * (time.perf_counter() - started)

    def _from_shared(self, name, batch):
        started = time.perf_counter()
        source = torch.from_numpy(getattr(self.manager.shared, name)[:batch])
        stage = self._stage(f"output_{name}", tuple(source.shape[1:]))
        stage[:batch].copy_(source)
        result = stage[:batch].to(
            device=self.device, dtype=self.dtype, non_blocking=self._pin_memory
        )
        if self.profile_timing:
            self._synchronize()
        self._timing["transfer_ms"] += 1000.0 * (time.perf_counter() - started)
        return result

    @staticmethod
    def _expanded_parameter(value, reference, batch, width):
        if value is None:
            return reference.new_zeros(batch, width)
        value = value.detach().reshape(batch, -1)
        return value.expand(-1, width) if value.shape[-1] == 1 else value

    def _install_inputs(self, q_pin, v_pin, parameters):
        batch = q_pin.shape[0]
        self._to_shared("q", q_pin, batch)
        self._to_shared("qd", v_pin, batch)
        added = parameters.get("added_base_mass")
        shift = parameters.get("base_com_shift")
        if not self.randomize_base_inertia:
            added, shift = None, None
        self._to_shared(
            "base_added_mass",
            self._expanded_parameter(added, q_pin, batch, 1), batch,
        )
        self._to_shared(
            "base_com_shift",
            self._expanded_parameter(shift, q_pin, batch, 3), batch,
        )
        for shared_name, parameter_name in (
            ("armature", "joint_armature"),
            ("joint_friction", "joint_friction"),
            ("joint_stiffness", "joint_stiffness"),
            ("joint_damping", "joint_damping"),
        ):
            self._to_shared(
                shared_name,
                self._expanded_parameter(
                    parameters.get(parameter_name), q_pin, batch, 12
                ), batch,
            )
        np.copyto(
            self.manager.shared.default_joint_position,
            self.default_joint_position.detach().cpu().numpy(),
        )
        self.manager.shared.scale_rotational_inertia[0] = float(
            self.scale_rotational_inertia
        )

    def build_context(
        self, pre_q_simulator, pre_v_world, *, parameters=None,
        post_v_world=None, mass_com_wrench_world=None, need_jacobians=True,
        need_qp=False, need_forward_dynamics=False,
    ):
        parameters = {} if parameters is None else {
            name: value.detach() for name, value in parameters.items()
        }
        q_internal, v_internal = simulator_state_to_bard(
            pre_q_simulator.detach(), pre_v_world.detach(),
            self.simulator_joint_names, self.bard_joint_names,
        )
        # The shared conversion produces BARD WXYZ; Pinocchio free flyers use
        # XYZW while sharing the same body-frame base twist and joint order.
        q_internal = torch.cat((
            q_internal[:, :3], q_internal[:, 3:7][:, (1, 2, 3, 0)],
            q_internal[:, 7:],
        ), dim=-1)
        batch = q_internal.shape[0]
        if batch > self.batch_capacity:
            raise ValueError("Pinocchio minibatch exceeds configured capacity")
        self._install_inputs(q_internal, v_internal, parameters)
        started = time.perf_counter()
        need_mass = bool(need_qp or need_forward_dynamics)
        need_jacobians = bool(need_jacobians or need_qp or need_forward_dynamics)
        self.manager.compute_context_async(
            batch, need_mass=need_mass, need_jacobians=need_jacobians,
            need_jdot=bool(need_qp),
        )
        self.manager.wait()
        self._timing["dynamics_ms"] += 1000.0 * (time.perf_counter() - started)
        self._timing["calls"] += 1
        foot = self._from_shared("foot_jacobians", batch).detach() if need_jacobians else None
        base = self._from_shared("base_jacobian", batch).detach() if need_jacobians else None
        post_internal = None
        if post_v_world is not None:
            _, post_internal = simulator_state_to_bard(
                pre_q_simulator.detach(), post_v_world.detach(),
                self.simulator_joint_names, self.bard_joint_names,
            )
        context = Go2BardContext(
            self, q_internal.detach(), v_internal.detach(), parameters,
            q_internal.new_tensor([0.0, 0.0, -9.81]), foot, base,
            None if post_internal is None else post_internal.detach(),
            None if mass_com_wrench_world is None else mass_com_wrench_world.detach(),
        )
        if need_mass:
            context.mass_matrix = self._from_shared("mass_mat", batch).detach()
            context.bias = self._from_shared("bias", batch).detach()
        if need_qp:
            context.foot_acceleration_bias = self._from_shared(
                "foot_acceleration_bias", batch
            ).detach()
        return context

    def _rnea_cached(self, context, acceleration_internal):
        batch = acceleration_internal.shape[0]
        self._to_shared(
            "acceleration", self._canonical(acceleration_internal), batch
        )
        started = time.perf_counter()
        self.manager.compute_rnea_async(batch)
        self.manager.wait()
        self._timing["dynamics_ms"] += 1000.0 * (time.perf_counter() - started)
        self._timing["calls"] += 1
        return self._from_shared("rnea", batch).detach()

    def _aba_cached(self, context, generalized_force):
        # Pinocchio mechanics are deliberately detached. The common RHS-only
        # solve supplies gradients solely to learned generalized force.
        return context.forward_dynamics(generalized_force)

    def timing_metrics(self, reset=False):
        result = dict(self._timing)
        if reset:
            self._timing = {"dynamics_ms": 0.0, "transfer_ms": 0.0, "calls": 0}
        return result

    def shutdown(self):
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
