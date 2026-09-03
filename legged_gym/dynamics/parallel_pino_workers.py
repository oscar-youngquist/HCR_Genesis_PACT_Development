# Shared worker implementation for batch-parallel Pinocchio dynamics.
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import pinocchio as pn
import time
import os
import queue
import math


def shared_memory_exists(name):
    try:
        shm = SharedMemory(name=name)
    except FileNotFoundError:
        return False
    else:
        shm.close()
        return True

class SharedTensors:
    """
    Allocates shared memory buffers used by both the main process and worker processes.
    """

    def __init__(
        self,
        num_envs,
        num_dof,
        foot_dim=(4,3),
        shared_name_prefix="go1_pact_02",
        max_name_attempts=1000,
        dtype=np.float32,
    ):
        self.num_envs = num_envs
        self.DOF = num_dof
        self.FOOT_DIM = foot_dim
        self.shared_name_prefix = shared_name_prefix
        self.dtype = np.dtype(dtype)

        self.shm = {}

        for prefix in self.available_prefixes(shared_name_prefix, max_name_attempts):
            self.shared_name_prefix = prefix
            try:
                self._create_all()
                break
            except FileExistsError:
                self.close()
        else:
            raise RuntimeError(
                f"Could not allocate shared memory names for prefix {shared_name_prefix}"
            )

    def _create_all(self):
        # Inputs
        #      the plus (1) accounts for the quat. representation of orientation used by pinocchio
        self.q       = self.make("q",       (self.num_envs, self.DOF+1))
        self.qd      = self.make("qd",      (self.num_envs, self.DOF))
        self.qd_prev = self.make("qd_prev", (self.num_envs, self.DOF))
        self.grf     = self.make("grf",     (self.num_envs, *self.FOOT_DIM))
        self.dt      = self.make("dt",      (1,))

        # Outputs
        self.wb_dynamics = self.make("wb_dynamics", (self.num_envs, self.DOF))
        self.wb_contacts = self.make("wb_contacts", (self.num_envs, self.DOF))
        self.mass_mat    = self.make("mass_mat",    (self.num_envs, self.DOF, self.DOF))
        self.bias        = self.make("bias",        (self.num_envs, self.DOF))
        self.acc6d       = self.make("acc6d",       (self.num_envs, 6))

        # Domain-randomized inertial parameters
        self.base_added_mass = self.make("base_added_mass", (self.num_envs, 1))
        self.base_com_shift = self.make("base_com_shift", (self.num_envs, 3))

        # Simulator-neutral HardPACT context inputs/outputs.  These extend the
        # same persistent worker pool used by legacy PACT; legacy jobs simply
        # leave them untouched. Arrays use the established float32 shared
        # transport while Pinocchio performs its internal calculations in its
        # native scalar type.
        self.acceleration = self.make("acceleration", (self.num_envs, self.DOF))
        self.armature = self.make("armature", (self.num_envs, 12))
        self.joint_friction = self.make("joint_friction", (self.num_envs, 12))
        self.joint_stiffness = self.make("joint_stiffness", (self.num_envs, 12))
        self.joint_damping = self.make("joint_damping", (self.num_envs, 12))
        self.default_joint_position = self.make("default_joint_position", (12,))
        self.scale_rotational_inertia = self.make("scale_rotational_inertia", (1,))
        self.rnea = self.make("rnea", (self.num_envs, self.DOF))
        self.foot_jacobians = self.make(
            "foot_jacobians", (self.num_envs, 4, 3, self.DOF)
        )
        self.base_jacobian = self.make(
            "base_jacobian", (self.num_envs, 6, self.DOF)
        )
        self.foot_acceleration_bias = self.make(
            "foot_acceleration_bias", (self.num_envs, 4, 3)
        )

    def available_prefixes(self, base_prefix, max_attempts):
        for counter in range(max_attempts):
            prefix = base_prefix if counter == 0 else f"{base_prefix}_{counter}"
            if self.prefix_available(prefix):
                yield prefix

    def prefix_available(self, prefix):
        return not any(shared_memory_exists(self.name_for(prefix, key)) for key in self.keys())

    def keys(self):
        return (
            "q",
            "qd",
            "qd_prev",
            "grf",
            "dt",
            "wb_dynamics",
            "wb_contacts",
            "mass_mat",
            "bias",
            "acc6d",
            "base_added_mass",
            "base_com_shift",
            "acceleration",
            "armature",
            "joint_friction",
            "joint_stiffness",
            "joint_damping",
            "default_joint_position",
            "scale_rotational_inertia",
            "rnea",
            "foot_jacobians",
            "base_jacobian",
            "foot_acceleration_bias",
        )

    def name_for(self, prefix, key):
        return f"{prefix}_{key}"

    def make(self, key, shape, dtype=None):
        dtype = self.dtype if dtype is None else np.dtype(dtype)
        nbytes = np.prod(shape) * np.dtype(dtype).itemsize
        name = self.name_for(self.shared_name_prefix, key)
        shm = SharedMemory(create=True, size=nbytes, name=name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        self.shm[key] = (shm, arr)
        return arr

    def close(self):
        for shm, _ in self.shm.values():
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self.shm.clear()

# Clone the inertial values from the original Pinocchio model to avoid any issues with shared memory and mutability.
def clone_inertia(I):
    return pn.Inertia(
        float(I.mass),
        np.array(I.lever, dtype=np.float64).copy(),
        np.array(I.inertia, dtype=np.float64).copy()
    )
    
def apply_base_inertia_randomization(
    pino_model,
    nominal_base_inertia,
    base_joint_id,
    added_mass,
    com_shift,
    scale_rotational_inertia=True,
):
    """
    Apply randomized torso mass/COM to Pinocchio model.

    IMPORTANT:
    This is computed relative to nominal_base_inertia, not the currently
    mutated model inertia, to avoid accumulating changes over calls.
    """

    m0 = float(nominal_base_inertia.mass)
    lever0 = np.array(nominal_base_inertia.lever, dtype=np.float64)
    inertia0 = np.array(nominal_base_inertia.inertia, dtype=np.float64)

    dm = float(added_mass)
    dc = np.asarray(com_shift, dtype=np.float64)

    new_mass = max(m0 + dm, 1e-6)
    new_lever = lever0 + dc

    if scale_rotational_inertia:
        # Simple approximation: preserve rotational inertia / mass ratio.
        new_inertia = inertia0 * (new_mass / m0)
    else:
        # Leaves rotational inertia unchanged.
        new_inertia = inertia0.copy()

    pino_model.inertias[base_joint_id] = pn.Inertia(
        new_mass,
        new_lever,
        new_inertia
    )

def compute_single_env(i, shared, pino_model, pino_data, 
                       pino_foot_frame_ids,
                       correct_idxs,
                       base_joint_id,
                       nominal_base_inertia,
                       aq0,
                       contact_tau):
    """
    Compute WB dynamics, mass matrix, bias forces, and contact forces
    for a single environment index i.
    """

    q       = shared.q[i]
    qd      = shared.qd[i]
    qdprev  = shared.qd_prev[i]
    grf     = shared.grf[i]      # shaped as (12)
    dt      = shared.dt[0]
    
    # ------------------------------------------------------------
    # Apply per-env randomized torso mass / COM to Pinocchio model
    # ------------------------------------------------------------
    added_mass = shared.base_added_mass[i, 0]
    com_shift  = shared.base_com_shift[i]

    apply_base_inertia_randomization(
        pino_model=pino_model,
        nominal_base_inertia=nominal_base_inertia,
        base_joint_id=base_joint_id,
        added_mass=added_mass,
        com_shift=com_shift,
        scale_rotational_inertia=True,
    )

    # Acceleration (backward finite difference)
    acc = (qd - qdprev) / dt

    # 6-DoF torso accel
    shared.acc6d[i] = acc[:6]

    # Pinocchio general coords
    aq0.fill(0.0)

    b = pn.rnea(pino_model, pino_data, q, qd, aq0)
    M = pn.crba(pino_model, pino_data, q)

    wb_dyn = M @ acc + b
    shared.wb_dynamics[i] = wb_dyn[correct_idxs]

    # Also save reduced mass matrix + bias
    M_ = M[np.ix_(correct_idxs, correct_idxs)]
    shared.mass_mat[i] = M_
    shared.bias[i] = b[correct_idxs]

    # Contact forces
    contact_tau.fill(0.0)
    grf_flat = grf.reshape(-1)
    for foot_idx, fid in enumerate(pino_foot_frame_ids):
        J = pn.computeFrameJacobian(
            pino_model,
            pino_data,
            q,
            fid,
            pn.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[0:3, :]
        start = 3 * foot_idx
        contact_tau += J.T @ grf_flat[start:start + 3]

    shared.wb_contacts[i] = contact_tau[correct_idxs]


def _passive_torque(shared, i, q, qd, correct_idxs):
    """Legacy-sign passive generalized force in canonical joint order."""
    joint_v_indices = np.asarray(correct_idxs[6:], dtype=np.int64)
    # A free-flyer has nq=nv+1, hence each 1-DoF joint's q index is idx_v+1.
    joint_q = q[joint_v_indices + 1]
    joint_v = qd[joint_v_indices]
    return (
        shared.joint_friction[i] * np.tanh(joint_v / 0.01)
        + shared.joint_stiffness[i]
        * (joint_q - shared.default_joint_position)
        + shared.joint_damping[i] * joint_v
    )


def compute_hard_pact_context_env(
    i, shared, pino_model, pino_data, pino_foot_frame_ids, base_frame_id,
    correct_idxs, base_joint_id, nominal_base_inertia,
    need_mass, need_jacobians, need_jdot,
):
    """Compute one row of the common HardPACT dynamics context."""
    q, qd = shared.q[i], shared.qd[i]
    apply_base_inertia_randomization(
        pino_model, nominal_base_inertia, base_joint_id,
        shared.base_added_mass[i, 0], shared.base_com_shift[i],
        bool(shared.scale_rotational_inertia[0]),
    )
    zero = np.zeros(pino_model.nv, dtype=np.float64)
    pn.forwardKinematics(pino_model, pino_data, q, qd, zero)
    pn.updateFramePlacements(pino_model, pino_data)
    ordering = np.asarray(correct_idxs, dtype=np.int64)

    if need_mass:
        mass = pn.crba(pino_model, pino_data, q)
        canonical_mass = mass[np.ix_(ordering, ordering)]
        canonical_mass[6:, 6:] += np.diag(shared.armature[i])
        shared.mass_mat[i] = canonical_mass
        bias = pn.rnea(pino_model, pino_data, q, qd, zero)[ordering]
        bias[6:] += _passive_torque(shared, i, q, qd, correct_idxs)
        shared.bias[i] = bias

    if need_jacobians:
        for foot, frame_id in enumerate(pino_foot_frame_ids):
            jacobian = pn.computeFrameJacobian(
                pino_model, pino_data, q, frame_id,
                pn.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            shared.foot_jacobians[i, foot] = jacobian[:3, ordering]
        base = pn.computeFrameJacobian(
            pino_model, pino_data, q, base_frame_id,
            pn.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        shared.base_jacobian[i] = base[:, ordering]

    if need_jdot:
        for foot, frame_id in enumerate(pino_foot_frame_ids):
            acceleration = pn.getFrameAcceleration(
                pino_model, pino_data, frame_id,
                pn.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            shared.foot_acceleration_bias[i, foot] = acceleration.linear


def compute_hard_pact_rnea_env(
    i, shared, pino_model, pino_data, correct_idxs, base_joint_id,
    nominal_base_inertia,
):
    """Evaluate randomized RNEA for an acceleration stored in canonical order."""
    q, qd = shared.q[i], shared.qd[i]
    apply_base_inertia_randomization(
        pino_model, nominal_base_inertia, base_joint_id,
        shared.base_added_mass[i, 0], shared.base_com_shift[i],
        bool(shared.scale_rotational_inertia[0]),
    )
    ordering = np.asarray(correct_idxs, dtype=np.int64)
    acceleration = np.empty(pino_model.nv, dtype=np.float64)
    acceleration[ordering] = shared.acceleration[i]
    result = pn.rnea(pino_model, pino_data, q, qd, acceleration)[ordering]
    result[6:] += shared.armature[i] * shared.acceleration[i, 6:]
    result[6:] += _passive_torque(shared, i, q, qd, correct_idxs)
    shared.rnea[i] = result

def worker_loop(worker_id, shared_names, shapes, dtypes,
                pino_model, pino_foot_names, correct_idxs,
                task_queue, done_queue,
                base_joint_id=1, base_frame_name="base",
                idle_sleep=0.01):
    """
    Long-running asynchronous worker.
    """

    # Reattach shared memory
    class S:
        pass

    shared = S()
    # shared.q = np.ndarray(shapes["q"], dtype=dtypes["q"],
    #                       buffer=SharedMemory(name=shared_names["q"]).buf)
    # shared.qd = np.ndarray(shapes["qd"], dtype=dtypes["qd"],
    #                        buffer=SharedMemory(name=shared_names["qd"]).buf)
    # shared.qd_prev = np.ndarray(shapes["qd_prev"], dtype=dtypes["qd_prev"],
    #                        buffer=SharedMemory(name=shared_names["qd_prev"]).buf)
    # shared.grf = np.ndarray(shapes["grf"], dtype=dtypes["grf"],
    #                        buffer=SharedMemory(name=shared_names["grf"]).buf)
    # shared.dt = np.ndarray(shapes["dt"], dtype=dtypes["dt"],
    #                        buffer=SharedMemory(name=shared_names["dt"]).buf)

    # shared.wb_dynamics = np.ndarray(shapes["wb_dynamics"], dtype=dtypes["wb_dynamics"],
    #                        buffer=SharedMemory(name=shared_names["wb_dynamics"]).buf)
    # shared.wb_contacts = np.ndarray(shapes["wb_contacts"], dtype=dtypes["wb_contacts"],
    #                        buffer=SharedMemory(name=shared_names["wb_contacts"]).buf)
    # shared.mass_mat = np.ndarray(shapes["mass_mat"], dtype=dtypes["mass_mat"],
    #                        buffer=SharedMemory(name=shared_names["mass_mat"]).buf)
    # shared.bias = np.ndarray(shapes["bias"], dtype=dtypes["bias"],
    #                        buffer=SharedMemory(name=shared_names["bias"]).buf)
    # shared.acc6d = np.ndarray(shapes["acc6d"], dtype=dtypes["acc6d"],
    #                        buffer=SharedMemory(name=shared_names["acc6d"]).buf)

    shared_shms = {}

    for name in shared_names:
        shm = SharedMemory(name=shared_names[name])
        shared_shms[name] = shm
        setattr(shared, name, np.ndarray(shapes[name], dtype=dtypes[name], buffer=shm.buf))


    pino_data = pino_model.createData()

    # Store nominal inertia once per worker.
    # All env-specific randomization is applied relative to this.
    nominal_base_inertia = clone_inertia(
        pino_model.inertias[base_joint_id]
    )
    pino_foot_frame_ids = [
        pino_model.getFrameId(foot_name) for foot_name in pino_foot_names
    ]
    base_frame_id = pino_model.getFrameId(base_frame_name)
    aq0 = np.zeros(shapes["qd"][1], dtype=dtypes["qd"])
    contact_tau = np.zeros(shapes["qd"][1], dtype=dtypes["qd"])

    while True:
        try:
            # Try to get a task, but timeout if none available
            task = task_queue.get(timeout=idle_sleep)
        except queue.Empty:
            # No task, yield CPU
            time.sleep(idle_sleep)  # optional extra sleep
            continue

        if task == "STOP":
            break

        start_env, end_env, *job = task
        operation = job[0] if job else "legacy"
        for env_id in range(start_env, end_env):
            if operation == "context":
                compute_hard_pact_context_env(
                    env_id, shared, pino_model, pino_data,
                    pino_foot_frame_ids, base_frame_id, correct_idxs,
                    base_joint_id, nominal_base_inertia,
                    bool(job[1]), bool(job[2]), bool(job[3]),
                )
            elif operation == "rnea":
                compute_hard_pact_rnea_env(
                    env_id, shared, pino_model, pino_data, correct_idxs,
                    base_joint_id, nominal_base_inertia,
                )
            else:
                compute_single_env(
                    env_id, shared, pino_model, pino_data,
                    pino_foot_frame_ids, correct_idxs, base_joint_id,
                    nominal_base_inertia, aq0, contact_tau,
                )

        done_queue.put(task)

    # Close handles (don't unlink, main process owns them)
    for shm in shared_shms.values():
        shm.close()


class PinocchioAsync:
    def __init__(self, pino_model, num_envs,
                 pino_foot_names,
                 correct_idxs,
                 wb_dim,
                 foot_dim,
                 num_cpu,
                 base_joint_id=1,
                 shared_name_prefix="pact",
                 base_frame_name="base",
                 shared_dtype=np.float32):

        self.pino_model = pino_model
        self.num_envs = num_envs
        self.pino_foot_names = pino_foot_names
        self.correct_idxs = correct_idxs
        self.base_joint_id = base_joint_id
        self.pending_chunks = 0
        self.active_envs = num_envs

        # Shared memory
        self.shared = SharedTensors(
            num_envs,
            num_dof=wb_dim,
            foot_dim=foot_dim,
            shared_name_prefix=shared_name_prefix,
            dtype=shared_dtype,
        )

        # Spawn workers
        self.task_q = mp.Queue()
        self.done_q = mp.Queue()

        # Shared memory descriptors
        self.shared_names = {k: shm.name for k,(shm,arr) in self.shared.shm.items()}
        self.shapes = {k: arr.shape for k,(shm,arr) in self.shared.shm.items()}
        self.dtypes = {k: arr.dtype for k,(shm,arr) in self.shared.shm.items()}

        self.workers = []
        for w in range(num_cpu):
            p = mp.Process(
                target=worker_loop,
                args=(w, 
                      self.shared_names, 
                      self.shapes, 
                      self.dtypes,
                      pino_model,
                      pino_foot_names,
                      correct_idxs,
                      self.task_q, 
                      self.done_q, 
                      base_joint_id,
                      base_frame_name)
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

    def _dispatch(self, operation="legacy", flags=(), active_envs=None):
        self.pending_chunks = 0
        active_envs = self.num_envs if active_envs is None else int(active_envs)
        if not self.workers or active_envs == 0:
            return

        chunk_size = math.ceil(active_envs / len(self.workers))
        for start_env in range(0, active_envs, chunk_size):
            end_env = min(start_env + chunk_size, active_envs)
            self.task_q.put((start_env, end_env, operation, *flags))
            self.pending_chunks += 1

    def compute_async(self):
        self._dispatch()

    def compute_context_async(
        self, active_envs, *, need_mass, need_jacobians, need_jdot
    ):
        self._dispatch(
            "context", (need_mass, need_jacobians, need_jdot), active_envs
        )

    def compute_rnea_async(self, active_envs):
        self._dispatch("rnea", (), active_envs)

    def wait(self):
        completed = 0
        while completed < self.pending_chunks:
            _ = self.done_q.get()
            completed += 1
        self.pending_chunks = 0

    def shutdown(self):
        for _ in self.workers:
            self.task_q.put("STOP")
        for p in self.workers:
            p.join()
        self.shared.close()
