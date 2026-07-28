"""Persistent shared-memory Pinocchio workers for the B1/Z1 PACT loss.

This follows the worker lifecycle used by ``simulator/parallel_pino_workers``
but carries the additional end-effector force and base-wrench terms required by
the legged-manipulation dynamics residual.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import queue
import uuid
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pinocchio as pin


# Every row is one PPO transition endpoint. Together these inputs let a worker
# evaluate the observed side of r = M(q) vdot + h(q,v) - J(q)^T F. Acceleration
# and policy torque remain in PPO because only that side needs autograd.
_INPUT_SHAPES = {
    "base_pos": (3,),
    "base_quat": (4,),
    "dof_pos": (19,),
    "base_linear_velocity": (3,),
    "base_angular_velocity": (3,),
    "dof_velocity": (19,),
    "grfs": (4, 3),
    "ee_force": (3,),
    "base_wrench": (6,),
    "base_added_mass": (1,),
    "base_com_shift": (3,),
    "gripper_added_mass": (1,),
}
_OUTPUT_SHAPES = {"mass": (25, 25), "bias": (25,), "contacts": (25,)}


class B1Z1PinoSharedBuffers:
    """Owner-side shared float32 arrays. Workers only attach and close them.

    Shared buffers avoid pickling a large PPO minibatch to every worker. The
    main process writes rows ``[0:batch_size]``, enqueues row ranges, waits for
    completion, and copies the three output arrays back to the policy device.
    """

    def __init__(self, capacity: int, prefix: str):
        self.capacity = capacity
        self._segments: dict[str, SharedMemory] = {}
        self.arrays: dict[str, np.ndarray] = {}
        for name, shape in {**_INPUT_SHAPES, **_OUTPUT_SHAPES}.items():
            full_shape = (capacity, *shape)
            segment = SharedMemory(
                create=True,
                size=int(np.prod(full_shape)) * np.dtype(np.float32).itemsize,
                name=f"{prefix}_{name}",
            )
            self._segments[name] = segment
            self.arrays[name] = np.ndarray(full_shape, dtype=np.float32, buffer=segment.buf)

    @property
    def descriptors(self):
        return {
            name: (segment.name, self.arrays[name].shape)
            for name, segment in self._segments.items()
        }

    def close(self):
        for segment in self._segments.values():
            segment.close()
            try:
                segment.unlink()
            except FileNotFoundError:
                pass
        self._segments.clear()
        self.arrays.clear()


def _attach_buffers(descriptors):
    segments, arrays = {}, {}
    for name, (shared_name, shape) in descriptors.items():
        segment = SharedMemory(name=shared_name)
        segments[name] = segment
        arrays[name] = np.ndarray(shape, dtype=np.float32, buffer=segment.buf)
    return segments, arrays


def _clone_inertia(inertia):
    return pin.Inertia(
        float(inertia.mass),
        np.asarray(inertia.lever, dtype=np.float64).copy(),
        np.asarray(inertia.inertia, dtype=np.float64).copy(),
    )


def _apply_mass_and_com(model, joint_id, nominal_inertia, added_mass, com_shift):
    """Reset one body to nominal, then mirror Genesis mass/COM variation.

    For nominal mass m0, lever c0, and inertia I0, use
      m = m0 + dm,  c = c0 + dc,  I = I0 * m/m0.
    Resetting from nominal on every row is essential: workers process many
    environments and must not accumulate one environment's randomization into
    the next. For the gripper, dc is zero and only its added mass is applied.
    """
    mass = max(float(nominal_inertia.mass) + float(added_mass), 1.0e-6)
    nominal_mass = max(float(nominal_inertia.mass), 1.0e-6)
    model.inertias[joint_id] = pin.Inertia(
        mass,
        np.asarray(nominal_inertia.lever, dtype=np.float64) + np.asarray(com_shift, dtype=np.float64),
        np.asarray(nominal_inertia.inertia, dtype=np.float64) * (mass / nominal_mass),
    )


def _worker_loop(urdf_path, position_indices, velocity_indices, foot_frame_ids, ee_frame_id,
                 base_frame_id, base_joint_id, gripper_joint_id, descriptors, task_queue, done_queue):
    """Compute assigned row ranges using a worker-local Pinocchio model/data.

    The worker owns mutable Pinocchio ``Model``/``Data`` instances. This is why
    mass/COM randomization is safe despite each PPO row having different
    inertias: no model object is shared between processes or row ranges.
    """
    segments, arrays = _attach_buffers(descriptors)
    try:
        model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        data = model.createData()
        nominal_base_inertia = _clone_inertia(model.inertias[base_joint_id])
        nominal_gripper_inertia = _clone_inertia(model.inertias[gripper_joint_id])
        position_indices = np.asarray(position_indices, dtype=np.int64)
        velocity_indices = np.asarray(velocity_indices, dtype=np.int64)
        q = np.zeros(model.nq)
        v = np.zeros(model.nv)
        generalized = np.zeros(model.nv)
        while True:
            try:
                task = task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if task is None:
                return
            start, end = task
            for index in range(start, end):
                # Apply the same inertial parameters that Genesis used for the
                # stored post-action state before evaluating M, h, and J^T F.
                _apply_mass_and_com(
                    model, base_joint_id, nominal_base_inertia,
                    arrays["base_added_mass"][index, 0], arrays["base_com_shift"][index],
                )
                _apply_mass_and_com(
                    model, gripper_joint_id, nominal_gripper_inertia,
                    arrays["gripper_added_mass"][index, 0], np.zeros(3),
                )
                q.fill(0.0)
                q[:3] = arrays["base_pos"][index]
                q[3:7] = arrays["base_quat"][index]
                q[position_indices] = arrays["dof_pos"][index]
                v.fill(0.0)
                v[:3] = arrays["base_linear_velocity"][index]
                v[3:6] = arrays["base_angular_velocity"][index]
                v[velocity_indices] = arrays["dof_velocity"][index]

                # h(q,v) contains gravity, Coriolis, and centrifugal terms;
                # M(q) maps PPO's finite-difference vdot into generalized
                # inertial force. Pinocchio may fill only one CRBA triangle.
                bias = pin.nonLinearEffects(model, data, q, v)
                mass = pin.crba(model, data, q)
                mass = np.triu(mass) + np.triu(mass, 1).T
                # Map measured/predicted spatial disturbances to generalized
                # coordinates. Feet and EE use linear forces (top 3 Jacobian
                # rows); the base input is a full 6-D wrench.
                generalized.fill(0.0)
                for frame_id, force in zip(foot_frame_ids, arrays["grfs"][index]):
                    jacobian = pin.computeFrameJacobian(
                        model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                    )
                    generalized += jacobian[:3].T @ force
                ee_jacobian = pin.computeFrameJacobian(
                    model, data, q, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                generalized += ee_jacobian[:3].T @ arrays["ee_force"][index]
                base_jacobian = pin.computeFrameJacobian(
                    model, data, q, base_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                generalized += base_jacobian.T @ arrays["base_wrench"][index]
                arrays["mass"][index] = mass
                arrays["bias"][index] = bias
                arrays["contacts"][index] = generalized
            done_queue.put((start, end))
    finally:
        for segment in segments.values():
            segment.close()


class B1Z1PinocchioAsync:
    """Shared-memory, persistent-worker evaluator for fixed-size PPO batches."""

    def __init__(self, urdf_path, position_indices, velocity_indices, foot_frame_ids,
                 ee_frame_id, base_frame_id, base_joint_id, gripper_joint_id,
                 capacity: int, num_workers: int,
                 start_method: str = "spawn"):
        if capacity < 1:
            raise ValueError("B1Z1 Pinocchio worker capacity must be positive")
        self.capacity = capacity
        self.num_workers = max(1, min(int(num_workers), capacity))
        self._context = mp.get_context(start_method)
        prefix = f"b1z1_pact_pino_{uuid.uuid4().hex[:12]}"
        self.shared = B1Z1PinoSharedBuffers(capacity, prefix)
        self.task_queue = self._context.Queue()
        self.done_queue = self._context.Queue()
        args = (
            urdf_path, list(position_indices), list(velocity_indices), list(foot_frame_ids),
            ee_frame_id, base_frame_id, base_joint_id, gripper_joint_id,
            self.shared.descriptors, self.task_queue, self.done_queue,
        )
        self.workers = [self._context.Process(target=_worker_loop, args=args, daemon=True) for _ in range(self.num_workers)]
        for worker in self.workers:
            worker.start()

    def evaluate(self, batch_size: int):
        if batch_size > self.capacity:
            raise ValueError(f"Batch size {batch_size} exceeds worker capacity {self.capacity}")
        # Divide the batch into contiguous rows. No output rows overlap, so
        # workers can write shared outputs without locks.
        chunks = min(self.num_workers, batch_size)
        chunk_size = math.ceil(batch_size / chunks)
        pending = 0
        for start in range(0, batch_size, chunk_size):
            self.task_queue.put((start, min(start + chunk_size, batch_size)))
            pending += 1
        for _ in range(pending):
            self.done_queue.get()

    def close(self):
        for _ in self.workers:
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        self.workers.clear()
        self.task_queue.close()
        self.done_queue.close()
        self.shared.close()
