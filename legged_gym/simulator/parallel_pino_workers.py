# Helper functions used to parallelize calls to the dynamics functions in Pinocchio with shared input/output memory buffers and reused-workers.
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import pinocchio as pn
import time
import os
import queue


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
    ):
        self.num_envs = num_envs
        self.DOF = num_dof
        self.FOOT_DIM = foot_dim
        self.shared_name_prefix = shared_name_prefix

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
        )

    def name_for(self, prefix, key):
        return f"{prefix}_{key}"

    def make(self, key, shape, dtype=np.float32):
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
                       pino_foot_names,
                       correct_idxs,
                       base_joint_id,
                       nominal_base_inertia):
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
    aq0 = np.zeros_like(qd)

    b = pn.rnea(pino_model, pino_data, q, qd, aq0)
    M = pn.crba(pino_model, pino_data, q)

    wb_dyn = M @ acc + b
    shared.wb_dynamics[i] = wb_dyn[correct_idxs]

    # Also save reduced mass matrix + bias
    M_ = M[np.ix_(correct_idxs, correct_idxs)]
    shared.mass_mat[i] = M_
    shared.bias[i] = b[correct_idxs]

    # Contact forces
    J_list = []
    for foot_name in pino_foot_names:
        fid = pino_model.getFrameId(foot_name)
        J = pn.computeFrameJacobian(pino_model, pino_data, q, fid, pn.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, :]
        J_list.append(J)

    J_total = np.concatenate(J_list, axis=0)
    tau = (J_total.transpose() @ grf).squeeze()

    shared.wb_contacts[i] = tau[correct_idxs]

def worker_loop(worker_id, shared_names, shapes, dtypes,
                pino_model, pino_foot_names, correct_idxs,
                task_queue, done_queue,
                base_joint_id=1,
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

    while True:
        try:
            # Try to get a task, but timeout if none available
            env_id = task_queue.get(timeout=idle_sleep)
        except queue.Empty:
            # No task, yield CPU
            time.sleep(idle_sleep)  # optional extra sleep
            continue

        if env_id == "STOP":
            break

        compute_single_env(
            env_id,
            shared,
            pino_model,
            pino_data,
            pino_foot_names,
            correct_idxs,
            base_joint_id,
            nominal_base_inertia,
        )

        done_queue.put(env_id)

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
                 shared_name_prefix="pact"):

        self.pino_model = pino_model
        self.num_envs = num_envs
        self.pino_foot_names = pino_foot_names
        self.correct_idxs = correct_idxs
        self.base_joint_id = base_joint_id

        # Shared memory
        self.shared = SharedTensors(
            num_envs,
            num_dof=wb_dim,
            foot_dim=foot_dim,
            shared_name_prefix=shared_name_prefix,
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
                      base_joint_id)
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

    def compute_async(self):
        for env_id in range(self.num_envs):
            self.task_q.put(env_id)

    def wait(self):
        completed = 0
        while completed < self.num_envs:
            _ = self.done_q.get()
            completed += 1

    def shutdown(self):
        for _ in self.workers:
            self.task_q.put("STOP")
        for p in self.workers:
            p.join()
        self.shared.close()
