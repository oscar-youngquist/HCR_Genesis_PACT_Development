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
                self._flush(e)
                self._buf[e] = self._new_buf()
                self._ep_idx[e] += 1

    def _flush(self, env_id):
        buf = self._buf[env_id]
        if not buf["step"]:
            return
        fname = (f"{self._meta['robot_id']}_{self._meta['liquid_tank']}_"
                 f"{self._meta['liquid_type']}_{int(self._meta['liquid_volume'])}L_"
                 f"env{env_id:02d}_ep{self._ep_idx[env_id]:04d}.h5")
        path = os.path.join(self._output_dir, fname)
        with h5py.File(path, "w") as f:
            for k, v in self._meta.items():
                f.attrs[k] = v
            f.attrs["env_id"] = env_id
            f.attrs["episode_idx"] = self._ep_idx[env_id]
            for k, lst in buf.items():
                f.create_dataset(k, data=np.asarray(lst))

    def close(self):
        for e in range(self._num_envs):
            if self._buf[e]["step"]:
                self._flush(e)
                self._buf[e] = self._new_buf()
