import os
from datetime import datetime

import h5py
import numpy as np
import torch

from legged_gym.simulator import genesis_simulator_pact_water as _water_sim


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class WaterDataLogger:
    FLUSH_INTERVAL_STEPS = 200

    def __init__(self, env, args, output_dir):
        self._env = env
        self._sim = env.simulator
        self._num_envs = env.cfg.env.num_envs
        self._args = args
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)

        liq = self._sim.liquid_properties
        sx, sy, sz = float(liq["scale_x"]), float(liq["scale_y"]), float(liq["scale_z"])
        inner_w = _water_sim.container_outer_x * sx - 2.0 * (sx * _water_sim.container_wall_thickness)
        inner_d = _water_sim.container_outer_y * sy - 2.0 * (sy * _water_sim.container_wall_thickness)
        inner_h = _water_sim.container_outer_z * sz - (sz * _water_sim.container_bottom_thickness)
        liq_off = float(liq["offset"])
        vol_m3 = max(inner_w - liq_off, 0.0) * max(inner_d - liq_off, 0.0) * max(inner_h - liq_off, 0.0)

        self._meta = {
            "robot_id": args.task.split("_")[0],
            "task": args.task,
            # Liquid physical properties
            "liquid_type": args.liquid_type,
            "liquid_volume_L": float(args.liquid_volume),
            "liquid_tank": args.liquid_tank,
            "rho":   float(liq["rho"]),
            "mu":    float(liq["mu"]),
            "gamma": float(liq["gamma"]),
            # Container geometry — raw config
            "scale_x": sx, "scale_y": sy, "scale_z": sz,
            "offset": liq_off,
            "mount_offset_xy": np.asarray(liq.get("mount_offset", [0.0, 0.0]), dtype=np.float32),
            # Container geometry — derived
            "tank_inner_dims_m": np.asarray([inner_w, inner_d, inner_h], dtype=np.float32),
            "liquid_volume_m3_effective": float(vol_m3),
            "liquid_mass_kg_effective": float(liq["rho"]) * float(vol_m3),
            # Fixed container constants (needed to reconstruct full tank pose if desired)
            "container_outer_m":   np.asarray([_water_sim.container_outer_x,
                                               _water_sim.container_outer_y,
                                               _water_sim.container_outer_z], dtype=np.float32),
            "container_wall_thickness_m":   float(_water_sim.container_wall_thickness),
            "container_bottom_thickness_m": float(_water_sim.container_bottom_thickness),
            "bucket_offset_z_m": float(_water_sim.bucket_offset),
            "has_lid": True,
            # SPH fidelity tags
            "sph_particle_size_m": float(_water_sim.liquid_particle_size),
            "sph_substeps": int(_water_sim.liquid_substeps),
            # Timing
            "control_dt": float(env.dt),
            "sim_dt": float(env.dt) / float(getattr(env.cfg.control, "decimation", 1)),
            "start_timestamp": datetime.now().isoformat(),
        }

        self._buf = [self._new_buf() for _ in range(self._num_envs)]
        self._ep_idx = [0] * self._num_envs
        self._files = [None] * self._num_envs
        self._n_flushed = 0
        print(f"[water_logger] init: num_envs={self._num_envs} out={self._output_dir} "
              f"flush_every={self.FLUSH_INTERVAL_STEPS} steps", flush=True)
        print(f"[water_logger]   liquid={self._meta['liquid_type']} {self._meta['liquid_volume_L']:g}L "
              f"tank={self._meta['liquid_tank']} mass={self._meta['liquid_mass_kg_effective']:.3f}kg "
              f"control_dt={self._meta['control_dt']:.4f}s", flush=True)

    def _new_buf(self):
        return {k: [] for k in (
            "step", "failed",
            # NN inputs — torso body frame where noted
            "base_lin_vel_body", "base_ang_vel_body", "projected_gravity_body",
            "base_quat_xyzw",
            "dof_pos", "dof_vel", "q_des", "commands",
            "feet_pos_world",
            # Pinocchio label inputs — world frame
            "base_pos_world", "base_vel_world", "base_ang_vel_world",
            "grf_world", "tau_motor",
        )}

    def log_step(self, step_idx, actions, dones):
        if step_idx > 0 and step_idx % 200 == 0:
            print(f"[water_logger] step={step_idx} flushed_so_far={self._n_flushed} "
                  f"buffered_envs={sum(1 for b in self._buf if b['step'])}", flush=True)
        sim, env = self._sim, self._env
        N = self._num_envs

        base_lin_body = _np(sim.base_lin_vel)
        base_ang_body = _np(sim.base_ang_vel)
        proj_g        = _np(sim.projected_gravity)
        base_quat     = _np(sim.base_quat)             # already xyzw
        dof_pos       = _np(sim.dof_pos)
        dof_vel       = _np(sim.dof_vel)
        feet_pos      = _np(sim.feet_pos)
        grfs          = _np(sim._grfs_buf).reshape(N, 4, 3)
        tau_motor     = _np(getattr(sim, "_dof_tau", sim.torques))
        base_pos      = _np(sim.base_pos)
        base_vel_w    = _np(sim._robot.get_vel())
        base_ang_w    = _np(sim._robot.get_ang())
        commands      = _np(env.commands)
        q_des         = (_np(env.get_scaled_pos_actions())
                         if hasattr(env, "get_scaled_pos_actions") else _np(actions))

        dones_np = _np(dones).astype(bool)

        for e in range(N):
            b = self._buf[e]
            b["step"].append(step_idx)
            b["failed"].append(bool(dones_np[e]))
            b["base_lin_vel_body"].append(base_lin_body[e])
            b["base_ang_vel_body"].append(base_ang_body[e])
            b["projected_gravity_body"].append(proj_g[e])
            b["base_quat_xyzw"].append(base_quat[e])
            b["dof_pos"].append(dof_pos[e])
            b["dof_vel"].append(dof_vel[e])
            b["q_des"].append(q_des[e])
            b["commands"].append(commands[e])
            b["feet_pos_world"].append(feet_pos[e])
            b["base_pos_world"].append(base_pos[e])
            b["base_vel_world"].append(base_vel_w[e])
            b["base_ang_vel_world"].append(base_ang_w[e])
            b["grf_world"].append(grfs[e])
            b["tau_motor"].append(tau_motor[e])

            if dones_np[e]:
                self._finalize_episode(e)

        if step_idx > 0 and step_idx % self.FLUSH_INTERVAL_STEPS == 0:
            for e in range(N):
                if self._buf[e]["step"]:
                    self._flush_chunk(e)

    def _ensure_open(self, env_id):
        if self._files[env_id] is not None:
            return self._files[env_id]
        try:
            fname = (f"{self._meta['robot_id']}_{self._meta['liquid_tank']}_"
                     f"{self._meta['liquid_type']}_{int(self._meta['liquid_volume_L'])}L_"
                     f"env{env_id:02d}_ep{self._ep_idx[env_id]:04d}.h5")
        except KeyError as ke:
            print(f"[water_logger] OPEN FAILED env{env_id}: missing meta key {ke!r}; "
                  f"available = {sorted(self._meta.keys())}", flush=True)
            raise
        path = os.path.join(self._output_dir, fname)
        f = h5py.File(path, "w")
        for k, v in self._meta.items():
            f.attrs[k] = v
        f.attrs["env_id"] = env_id
        f.attrs["episode_idx"] = self._ep_idx[env_id]
        self._files[env_id] = f
        return f

    def _flush_chunk(self, env_id):
        buf = self._buf[env_id]
        if not buf["step"]:
            return
        f = self._ensure_open(env_id)
        n_new = len(buf["step"])
        for k, lst in buf.items():
            arr = np.asarray(lst)
            if k in f:
                ds = f[k]
                old_n = ds.shape[0]
                ds.resize((old_n + n_new,) + ds.shape[1:])
                ds[old_n:] = arr
            else:
                maxshape = (None,) + arr.shape[1:]
                f.create_dataset(k, data=arr, maxshape=maxshape, chunks=True)
        f.flush()
        self._buf[env_id] = self._new_buf()

    def _finalize_episode(self, env_id):
        if self._buf[env_id]["step"]:
            self._flush_chunk(env_id)
        f = self._files[env_id]
        if f is None:
            return
        path = f.filename
        n_steps = f["step"].shape[0]
        f.close()
        self._files[env_id] = None
        self._n_flushed += 1
        sz_kb = os.path.getsize(path) // 1024
        print(f"[water_logger] finalized env{env_id:02d} ep{self._ep_idx[env_id]:04d} "
              f"({n_steps} steps, {sz_kb} KB) -> {os.path.basename(path)}  [total={self._n_flushed}]", flush=True)
        self._ep_idx[env_id] += 1

    def close(self):
        n_pending = sum(1 for b in self._buf if b["step"])
        n_open = sum(1 for f in self._files if f is not None)
        print(f"[water_logger] close: {n_pending} buffered envs, {n_open} open files "
              f"(already finalized {self._n_flushed})", flush=True)
        for e in range(self._num_envs):
            if self._buf[e]["step"] or self._files[e] is not None:
                self._finalize_episode(e)
        print(f"[water_logger] close: done. final flushed total = {self._n_flushed}", flush=True)
