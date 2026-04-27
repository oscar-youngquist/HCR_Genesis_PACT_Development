"""Post-process water-eval episodes into 6-DOF liquid wrench labels.

Physics: for a floating-base rigid-robot Pinocchio model (no liquid),
    M(q) q̈ + b(q, q̇) = τ_motor + Σ J_cᵀ F_c + τ_ext
the first 6 components of τ_ext are the wrench the liquid applies to the
torso. Express in body frame for the NN target.

Sanity check: run one episode with liquid_volume=0 — wrench_body should be
~0 throughout. Non-zero mean = the pinocchio URDF and the Genesis robot
disagree about what's rigid (e.g. Genesis has the tank attached but the
URDF doesn't, so tank weight leaks into the label).
"""
import argparse
import glob
import os

import h5py
import numpy as np
import pinocchio as pin


# TODO: verify URDF paths, foot frame names, and joint-order maps per robot.
ROBOT_CFGS = {
    "go1": dict(
        urdf="resources/robots/go1/urdf/go1.urdf",
        foot_frames=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        joint_map_model_to_pino=list(range(12)),
    ),
    "go2": dict(
        urdf="resources/robots/go2/urdf/go2.urdf",
        foot_frames=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        joint_map_model_to_pino=list(range(12)),
    ),
    "a1": dict(
        urdf="resources/robots/a1/urdf/a1.urdf",
        foot_frames=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        joint_map_model_to_pino=list(range(12)),
    ),
}


def quat_xyzw_to_R(q):
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz), 2*(xy - wz),   2*(xz + wy)],
        [2*(xy + wz),   1 - 2*(xx+zz), 2*(yz - wx)],
        [2*(xz - wy),   2*(yz + wx),   1 - 2*(xx+yy)],
    ])


def compute_labels_for_file(h5_path, robot_cfgs=ROBOT_CFGS):
    with h5py.File(h5_path, "a") as f:
        robot_id = f.attrs["robot_id"]
        if isinstance(robot_id, bytes):
            robot_id = robot_id.decode()
        cfg = robot_cfgs[robot_id]

        model = pin.buildModelFromUrdf(cfg["urdf"], pin.JointModelFreeFlyer())
        data = model.createData()
        foot_ids = [model.getFrameId(n) for n in cfg["foot_frames"]]
        jmap = np.asarray(cfg["joint_map_model_to_pino"])
        dt = float(f.attrs["control_dt"])

        base_pos = f["base_pos_world"][:]
        base_quat = f["base_quat_xyzw"][:]
        base_vel_w = f["base_vel_world"][:]
        base_ang_w = f["base_ang_vel_world"][:]
        dof_pos_pino = f["dof_pos"][:, jmap]
        dof_vel_pino = f["dof_vel"][:, jmap]
        tau_motor_pino = f["tau_motor"][:, jmap]
        grf = f["grf_world"][:]

        T = base_pos.shape[0]
        w_world = np.zeros((T, 6), dtype=np.float32)
        w_body = np.zeros((T, 6), dtype=np.float32)

        for t in range(1, T):
            q = np.concatenate([base_pos[t], base_quat[t], dof_pos_pino[t]])
            qd = np.concatenate([base_vel_w[t], base_ang_w[t], dof_vel_pino[t]])
            qd_prev = np.concatenate([base_vel_w[t-1], base_ang_w[t-1], dof_vel_pino[t-1]])
            acc = (qd - qd_prev) / dt

            M = pin.crba(model, data, q)
            b = pin.rnea(model, data, q, qd, np.zeros_like(qd))
            wb_dyn = M @ acc + b

            pin.computeJointJacobians(model, data, q)
            pin.updateFramePlacements(model, data)
            tau_c = np.zeros(model.nv)
            for k, fid in enumerate(foot_ids):
                J = pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
                tau_c += J.T @ grf[t, k]

            tau_m = np.zeros(model.nv)
            tau_m[6:] = tau_motor_pino[t]

            tau_ext = wb_dyn - tau_c - tau_m
            w_world[t] = tau_ext[:6]
            R = quat_xyzw_to_R(base_quat[t])
            w_body[t, :3] = R.T @ w_world[t, :3]
            w_body[t, 3:] = R.T @ w_world[t, 3:]

        if "labels" in f:
            del f["labels"]
        g = f.create_group("labels")
        g.create_dataset("wrench_world", data=w_world)
        g.create_dataset("wrench_body", data=w_body)
        g.attrs["note"] = "wrench_body[0] invalid (no prior qd for finite diff)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--pattern", default="*.h5")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    print(f"Labelling {len(files)} episodes")
    for i, p in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {os.path.basename(p)}")
        compute_labels_for_file(p)


if __name__ == "__main__":
    main()
