#!/usr/bin/env python3
"""
Create contained, partitioned, seaborn-ready CSV files from quadruped evaluation logs.

Expected raw input filename format:
    {approach}_{terrain}_{disturbance}.csv

The script recursively searches an experiment folder, filters by approach,
disturbance, and optionally terrain, then writes five CSV files:

    tracking_metrics.csv
    torso_stability_metrics.csv
    robot_state.csv
    forces_power.csv
    disturbances.csv

All output files use a mostly seaborn-ready long format:

    approach, terrain, disturbance, source_file, source_relpath,
    timestep, env_id, metric, component, value, row_type, is_failure

The extra column `is_failure` intentionally does not follow the minimal seaborn
format, but is useful for excluding or highlighting failure timesteps in plots.

Example usage:
    python build_contained_seaborn_eval_csvs.py \
        --exp_folder exp_data/output \
        --approaches pact tau pos \
        --disturbances none payload push \
        --output_dir plot_data
"""

import argparse
import ast
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Component naming
# -----------------------------------------------------------------------------
# Adjust these if your logger uses a different joint ordering.
JOINT_NAMES_12 = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

FOOT_COMPONENT_NAMES = [
    "FR_x", "FR_y", "FR_z",
    "FL_x", "FL_y", "FL_z",
    "RR_x", "RR_y", "RR_z",
    "RL_x", "RL_y", "RL_z",
]

BASE_CMD_NAMES = ["cmd_x", "cmd_y", "cmd_yaw", "cmd_heading"]
XYZ_NAMES = ["x", "y", "z"]
RPY_NAMES = ["roll", "pitch", "yaw"]
LIN_VEL_NAMES = ["vx", "vy", "vz"]
ANG_VEL_NAMES = ["wx", "wy", "wz"]
PROJ_GRAV_NAMES = ["gx", "gy", "gz"]

COMPONENT_NAMES = {
    "base_cmd": BASE_CMD_NAMES,
    "base_pose": XYZ_NAMES,
    "base_rpy": RPY_NAMES,
    "base_lin_vel": LIN_VEL_NAMES,
    "base_ang_vel": ANG_VEL_NAMES,
    "proj_grav": PROJ_GRAV_NAMES,
    "dof_pose": JOINT_NAMES_12,
    "dof_vel": JOINT_NAMES_12,
    "q_des": JOINT_NAMES_12,
    "tau_act": JOINT_NAMES_12,
    "tau_ff": JOINT_NAMES_12,
    "tau_pd": JOINT_NAMES_12,
    "joint_power_total": JOINT_NAMES_12,
    "joint_power_ff": JOINT_NAMES_12,
    "joint_power_pd": JOINT_NAMES_12,
    "grf": FOOT_COMPONENT_NAMES,
    "payload": ["payload_mass"],
    "com_shift": ["com_x", "com_y", "com_z"],
    "rand_push": ["push_x", "push_y", "push_z"],
    "rand_wrench": ["wrench_x", "wrench_y", "wrench_z"],
}

RAW_COLUMNS_NEEDED = [
    "base_cmd", "base_pose", "base_rpy", "base_lin_vel", "base_ang_vel",
    "proj_grav", "dof_pose", "dof_vel", "q_des",
    "tau_act", "tau_ff", "tau_pd", "grf",
    "payload", "com_shift", "rand_push", "rand_wrench", "failure",
]


# -----------------------------------------------------------------------------
# Generic parsing helpers
# -----------------------------------------------------------------------------
def parse_array_cell(value):
    """Parse one Python-list-like CSV cell into a numpy array."""
    if isinstance(value, str):
        try:
            return np.asarray(ast.literal_eval(value), dtype=float)
        except Exception:
            return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.asarray([value], dtype=float)

    return None


def normalize_array_shape(arr, max_envs=None):
    """
    Normalize a logged array to shape [num_envs, flat_dim].

    Common logged shapes:
        [num_envs]
        [num_envs, dim]
        [num_envs, feet, dim]
    """
    if arr is None:
        return None

    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[:, None]

    if max_envs is not None:
        arr = arr[:max_envs]

    num_envs = arr.shape[0]
    return arr.reshape(num_envs, -1)


def component_names_for(metric_name, flat_dim):
    """Return human-readable component names, falling back to metric_i names."""
    names = COMPONENT_NAMES.get(metric_name)

    if names is None or len(names) != flat_dim:
        return [f"{metric_name}_{i}" for i in range(flat_dim)]

    return names


def parse_eval_filename(path, approaches):
    """
    Parse filenames of form {approach}_{terrain}_{disturbance}.csv.

    Approach names may contain underscores, so the function first matches
    against the provided approach list and prefers the longest match.
    """
    stem = Path(path).stem

    for approach in sorted(approaches, key=len, reverse=True):
        prefix = f"{approach}_"
        if not stem.startswith(prefix):
            continue

        suffix = stem[len(prefix):]
        parts = suffix.split("_")

        if len(parts) < 2:
            return None

        return {
            "approach": approach,
            "terrain": "_".join(parts[:-1]),
            "disturbance": parts[-1],
            "path": Path(path),
        }

    return None


def should_skip_path(path, exp_folder, output_dir, exclude_dirs):
    """Skip generated outputs and blacklisted directories."""
    path = Path(path)
    exp_folder = Path(exp_folder)
    output_dir = Path(output_dir)

    try:
        rel_parts = path.relative_to(exp_folder).parts
    except ValueError:
        rel_parts = path.parts

    if exclude_dirs:
        excluded = set(exclude_dirs)
        if any(part in excluded for part in rel_parts[:-1]):
            return True

    try:
        path.relative_to(output_dir)
        return True
    except ValueError:
        return False


def load_required_arrays(row, max_envs=None):
    """Parse all columns needed by the downstream metric builders."""
    arrays = {}

    for col in RAW_COLUMNS_NEEDED:
        if col not in row.index:
            continue

        arr = parse_array_cell(row[col])
        flat = normalize_array_shape(arr, max_envs=max_envs)

        if flat is not None:
            arrays[col] = flat

    return arrays


def get_num_envs(arrays):
    """Infer number of parallel environments from the first parsed array."""
    for arr in arrays.values():
        return arr.shape[0]
    return 0


def get_failure_flags(arrays, num_envs):
    """Return one boolean failure flag per env for the current timestep."""
    if "failure" not in arrays:
        return np.zeros(num_envs, dtype=bool)

    failure = arrays["failure"].reshape(arrays["failure"].shape[0], -1)
    flags = np.zeros(num_envs, dtype=bool)
    n = min(num_envs, failure.shape[0])
    flags[:n] = failure[:n, 0].astype(bool)
    return flags


# -----------------------------------------------------------------------------
# Output row helpers
# -----------------------------------------------------------------------------
def append_scalar_row(rows, meta, env_id, metric, component, value, row_type, is_failure):
    """Append one long-format output row."""
    rows.append({
        **meta,
        "env_id": int(env_id),
        "metric": metric,
        "component": component,
        "value": float(value),
        "row_type": row_type,
        "is_failure": bool(is_failure),
    })


def append_component_metric(rows, meta, metric, values, component_names, failure_flags, row_type):
    """Append one row per component per env."""
    if values is None:
        return

    values = np.asarray(values, dtype=float)
    num_envs, dim = values.shape

    if len(component_names) != dim:
        component_names = [f"{metric}_{i}" for i in range(dim)]

    for env_id in range(num_envs):
        for j, component in enumerate(component_names):
            append_scalar_row(
                rows, meta, env_id,
                metric=metric,
                component=component,
                value=values[env_id, j],
                row_type=row_type,
                is_failure=failure_flags[env_id],
            )


def append_l2_norm_metric(rows, meta, metric, values, failure_flags):
    """Append one L2 norm row per env for vector-valued metrics."""
    if values is None or values.shape[1] <= 1:
        return

    norms = np.linalg.norm(values, axis=1)

    for env_id, value in enumerate(norms):
        append_scalar_row(
            rows, meta, env_id,
            metric=f"{metric}_l2_norm",
            component="l2_norm",
            value=value,
            row_type="aggregate_l2_norm",
            is_failure=failure_flags[env_id],
        )


def append_metric_with_components_and_l2(rows, meta, metric, values, failure_flags, row_type):
    """Append component-wise values and a vector L2 norm."""
    if values is None:
        return

    names = component_names_for(metric, values.shape[1])

    append_component_metric(rows, meta, metric, values, names, failure_flags, row_type)
    append_l2_norm_metric(rows, meta, metric, values, failure_flags)


# -----------------------------------------------------------------------------
# Tracking CSV
# -----------------------------------------------------------------------------
def build_tracking_rows(arrays, meta, failure_flags, base_height_target):
    """
    Build rows for tracking_metrics.csv.

    Metrics:
      - height_tracking_mae: |base_z - target_z|
      - linear_velocity_tracking_mae: |cmd_x - vx|, |cmd_y - vy|, mean over x/y
      - angular_velocity_tracking_mae: |cmd_yaw - wz|
      - dof_position_tracking_mae: |q_des - q| per joint, mean over all joints
    """
    rows = []

    base_pose = arrays.get("base_pose")
    base_cmd = arrays.get("base_cmd")
    base_lin_vel = arrays.get("base_lin_vel")
    base_ang_vel = arrays.get("base_ang_vel")
    q_des = arrays.get("q_des")
    dof_pose = arrays.get("dof_pose")

    if base_pose is not None and base_pose.shape[1] >= 3:
        height_abs_err = np.abs(base_pose[:, 2] - base_height_target)[:, None]
        append_component_metric(
            rows, meta, "height_tracking_mae", height_abs_err,
            ["z"], failure_flags, "component_mae"
        )

    if base_cmd is not None and base_lin_vel is not None:
        n = min(base_cmd.shape[0], base_lin_vel.shape[0], len(failure_flags))
        if base_cmd.shape[1] >= 2 and base_lin_vel.shape[1] >= 2:
            lin_abs_err = np.abs(base_cmd[:n, 0:2] - base_lin_vel[:n, 0:2])
            lin_mean_mae = np.mean(lin_abs_err, axis=1)

            append_component_metric(
                rows, meta, "linear_velocity_tracking_mae", lin_abs_err,
                ["x", "y"], failure_flags[:n], "component_mae"
            )

            for env_id, value in enumerate(lin_mean_mae):
                append_scalar_row(
                    rows, meta, env_id, "linear_velocity_tracking_mae",
                    "xy_mean", value, "aggregate_mae", failure_flags[env_id]
                )

    if base_cmd is not None and base_ang_vel is not None:
        n = min(base_cmd.shape[0], base_ang_vel.shape[0], len(failure_flags))
        if base_cmd.shape[1] >= 3 and base_ang_vel.shape[1] >= 3:
            yaw_abs_err = np.abs(base_cmd[:n, 2] - base_ang_vel[:n, 2])[:, None]
            append_component_metric(
                rows, meta, "angular_velocity_tracking_mae", yaw_abs_err,
                ["yaw"], failure_flags[:n], "component_mae"
            )

    if q_des is not None and dof_pose is not None:
        n = min(q_des.shape[0], dof_pose.shape[0], len(failure_flags))
        dim = min(q_des.shape[1], dof_pose.shape[1])

        if dim > 0:
            dof_abs_err = np.abs(q_des[:n, :dim] - dof_pose[:n, :dim])
            names = component_names_for("dof_pose", dim)
            dof_mean_mae = np.mean(dof_abs_err, axis=1)

            append_component_metric(
                rows, meta, "dof_position_tracking_mae", dof_abs_err,
                names, failure_flags[:n], "component_mae"
            )

            for env_id, value in enumerate(dof_mean_mae):
                append_scalar_row(
                    rows, meta, env_id, "dof_position_tracking_mae",
                    "all_joints_mean", value, "aggregate_mae", failure_flags[env_id]
                )

    return rows


# -----------------------------------------------------------------------------
# Torso stability CSV
# -----------------------------------------------------------------------------
def build_stability_rows(arrays, meta, failure_flags):
    """
    Build rows for torso_stability_metrics.csv.

    Metrics:
      - projected_gravity_roll_pitch_l1: |gx| + |gy| plus component values
      - torso_vertical_velocity_mae: |vz|
      - torso_roll_pitch_ang_vel_l1: |wx| + |wy| plus component values
    """
    rows = []

    proj_grav = arrays.get("proj_grav")
    base_lin_vel = arrays.get("base_lin_vel")
    base_ang_vel = arrays.get("base_ang_vel")

    if proj_grav is not None and proj_grav.shape[1] >= 2:
        pg_abs = np.abs(proj_grav[:, 0:2])
        pg_l1 = np.sum(pg_abs, axis=1)

        append_component_metric(
            rows, meta, "projected_gravity_roll_pitch_l1", pg_abs,
            ["gx_abs", "gy_abs"], failure_flags, "component_abs"
        )

        for env_id, value in enumerate(pg_l1):
            append_scalar_row(
                rows, meta, env_id, "projected_gravity_roll_pitch_l1",
                "rp_l1", value, "aggregate_l1", failure_flags[env_id]
            )

    if base_lin_vel is not None and base_lin_vel.shape[1] >= 3:
        z_vel_abs = np.abs(base_lin_vel[:, 2])[:, None]
        append_component_metric(
            rows, meta, "torso_vertical_velocity_mae", z_vel_abs,
            ["vz_abs"], failure_flags, "component_mae"
        )

    if base_ang_vel is not None and base_ang_vel.shape[1] >= 2:
        ang_abs = np.abs(base_ang_vel[:, 0:2])
        ang_l1 = np.sum(ang_abs, axis=1)

        append_component_metric(
            rows, meta, "torso_roll_pitch_ang_vel_l1", ang_abs,
            ["wx_abs", "wy_abs"], failure_flags, "component_abs"
        )

        for env_id, value in enumerate(ang_l1):
            append_scalar_row(
                rows, meta, env_id, "torso_roll_pitch_ang_vel_l1",
                "rp_l1", value, "aggregate_l1", failure_flags[env_id]
            )

    return rows


# -----------------------------------------------------------------------------
# Robot state CSV
# -----------------------------------------------------------------------------
def build_robot_state_rows(arrays, meta, failure_flags):
    """
    Build rows for robot_state.csv.

    Saved variables:
      - base_pose
      - base_rpy
      - dof_pose
      - base_lin_vel
      - base_ang_vel
      - dof_vel

    Each vector is saved component-wise and with an L2 norm row.
    """
    rows = []

    for metric in ["base_pose", "base_rpy", "dof_pose", "base_lin_vel", "base_ang_vel", "dof_vel"]:
        values = arrays.get(metric)
        if values is not None:
            append_component_metric(rows,
                                    meta,
                                    metric,
                                    values,
                                    component_names_for(metric, values.shape[1]),
                                    failure_flags,
                                    "component_state")

    return rows


# -----------------------------------------------------------------------------
# Forces / power CSV
# -----------------------------------------------------------------------------
def build_force_power_rows(arrays, meta, failure_flags, eps=1e-8):
    """
    Build rows for forces_power.csv.

    Saved vector values:
      - tau_act, tau_ff, tau_pd
      - joint_power_total = tau_act * dof_vel
      - joint_power_ff    = tau_ff  * dof_vel
      - joint_power_pd    = tau_pd  * dof_vel
      - grf

    For each vector value, saves component-wise values and L2 norm.

    Saved aggregate-only values:
      - ff_tau_ratio
      - pd_tau_ratio
      - pd_to_ff_tau_norm_ratio
      - power_alignment
      - fraction_antagonistic_energy
      - internal_power_cancellation
    """
    rows = []

    tau_act = arrays.get("tau_act")
    tau_ff = arrays.get("tau_ff")
    tau_pd = arrays.get("tau_pd")
    q_vel = arrays.get("dof_vel")
    grf = arrays.get("grf")

    for metric, values in [
        ("tau_act", tau_act),
        ("tau_ff", tau_ff),
        ("tau_pd", tau_pd),
        ("grf", grf),
    ]:
        if values is not None:
            append_metric_with_components_and_l2(
                rows, meta, metric, values, failure_flags, "component_force"
            )

    if q_vel is not None:
        for metric, tau in [
            ("joint_power_total", tau_act),
            ("joint_power_ff", tau_ff),
            ("joint_power_pd", tau_pd),
        ]:
            if tau is None:
                continue

            n = min(tau.shape[0], q_vel.shape[0], len(failure_flags))
            dim = min(tau.shape[1], q_vel.shape[1])
            power = tau[:n, :dim] * q_vel[:n, :dim]

            append_metric_with_components_and_l2(
                rows, meta, metric, power, failure_flags[:n], "component_power"
            )

    if tau_ff is not None and tau_pd is not None:
        n = min(tau_ff.shape[0], tau_pd.shape[0], len(failure_flags))
        dim = min(tau_ff.shape[1], tau_pd.shape[1])

        ff = tau_ff[:n, :dim]
        pd_tau = tau_pd[:n, :dim]

        ff_norm = np.linalg.norm(ff, axis=1)
        pd_norm = np.linalg.norm(pd_tau, axis=1)
        denom = ff_norm + pd_norm + eps

        derived = {
            "ff_tau_ratio": ff_norm / denom,
            "pd_tau_ratio": pd_norm / denom,
            "pd_to_ff_tau_norm_ratio": pd_norm / (ff_norm + eps),
        }

        for metric, values in derived.items():
            for env_id, value in enumerate(values):
                append_scalar_row(
                    rows, meta, env_id, metric, "aggregate",
                    value, "derived_force_aggregate", failure_flags[env_id]
                )

    if tau_ff is not None and tau_pd is not None and q_vel is not None:
        n = min(tau_ff.shape[0], tau_pd.shape[0], q_vel.shape[0], len(failure_flags))
        dim = min(tau_ff.shape[1], tau_pd.shape[1], q_vel.shape[1])

        ff = tau_ff[:n, :dim]
        pd_tau = tau_pd[:n, :dim]
        qd = q_vel[:n, :dim]

        ff_power = ff * qd
        pd_power = pd_tau * qd
        total_power = (ff + pd_tau) * qd

        dot = np.sum(ff_power * pd_power, axis=1)
        ff_power_norm = np.linalg.norm(ff_power, axis=1)
        pd_power_norm = np.linalg.norm(pd_power, axis=1)

        power_alignment = dot / (ff_power_norm * pd_power_norm + eps)
        fraction_antagonistic_energy = np.maximum(-dot, 0.0) / (np.abs(dot) + eps)

        numerator = np.abs(ff_power) + np.abs(pd_power) - np.abs(total_power)
        denominator = np.abs(ff_power) + np.abs(pd_power) + eps
        internal_power_cancellation = np.mean(numerator / denominator, axis=1)

        derived = {
            "power_alignment": power_alignment,
            "fraction_antagonistic_energy": fraction_antagonistic_energy,
            "internal_power_cancellation": internal_power_cancellation,
        }

        for metric, values in derived.items():
            for env_id, value in enumerate(values):
                append_scalar_row(
                    rows, meta, env_id, metric, "aggregate",
                    value, "derived_power_aggregate", failure_flags[env_id]
                )

    return rows


# -----------------------------------------------------------------------------
# Disturbance CSV
# -----------------------------------------------------------------------------
def build_disturbance_rows(arrays, meta, failure_flags):
    """
    Build rows for disturbances.csv.

    Saved variables:
      - payload
      - com_shift
      - rand_push
      - rand_wrench

    Each is saved component-wise and vector-valued disturbances also receive an
    L2 norm row.
    """
    rows = []

    for metric in ["payload", "com_shift", "rand_push", "rand_wrench"]:
        values = arrays.get(metric)
        if values is not None:
            append_metric_with_components_and_l2(
                rows, meta, metric, values, failure_flags, "component_disturbance"
            )

    return rows

def process_one_csv_from_args(args):
    """Unpack arguments for process/thread pool execution."""
    return process_one_csv(*args)

# -----------------------------------------------------------------------------
# Per-file and top-level pipeline
# -----------------------------------------------------------------------------
def process_one_csv(path, exp_folder, approaches, disturbances, terrain_filter, base_height_target, max_envs):
    """Process one raw evaluation CSV into partition-specific DataFrames."""
    info = parse_eval_filename(path, approaches)
    if info is None:
        return None

    approach = info["approach"]
    terrain = info["terrain"]
    disturbance = info["disturbance"]

    if disturbance not in disturbances:
        return None

    if terrain_filter is not None and terrain not in terrain_filter:
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[skip] failed to read {path}: {exc}")
        return None

    partition_rows = {
        "tracking_metrics": [],
        "torso_stability_metrics": [],
        "robot_state": [],
        "forces_power": [],
        "disturbances": [],
    }

    for timestep, row in df.iterrows():
        arrays = load_required_arrays(row, max_envs=max_envs)
        num_envs = get_num_envs(arrays)

        if num_envs == 0:
            continue

        failure_flags = get_failure_flags(arrays, num_envs)

        meta = {
            "approach": approach,
            "terrain": terrain,
            "disturbance": disturbance,
            "source_file": path.name,
            "source_relpath": str(path.relative_to(exp_folder)),
            "timestep": int(timestep),
        }

        partition_rows["tracking_metrics"].extend(
            build_tracking_rows(arrays, meta, failure_flags, base_height_target)
        )
        partition_rows["torso_stability_metrics"].extend(
            build_stability_rows(arrays, meta, failure_flags)
        )
        partition_rows["robot_state"].extend(
            build_robot_state_rows(arrays, meta, failure_flags)
        )
        partition_rows["forces_power"].extend(
            build_force_power_rows(arrays, meta, failure_flags)
        )
        partition_rows["disturbances"].extend(
            build_disturbance_rows(arrays, meta, failure_flags)
        )

    out = {
        name: pd.DataFrame(rows)
        for name, rows in partition_rows.items()
        if rows
    }

    if not out:
        return None

    print(f"[load] {path}")
    return out


def build_contained_seaborn_csvs(
    exp_folder,
    approaches,
    disturbances,
    output_dir="plot_data",
    terrains=None,
    exclude_dirs=None,
    max_envs=None,
    max_workers=8,
    base_height_target=0.30,
):
    """Main pipeline: discover, process, concatenate, and save CSV partitions."""
    exp_folder = Path(exp_folder)
    approaches = list(approaches)
    disturbances = set(disturbances)
    terrain_filter = None if terrains is None else set(terrains)

    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = exp_folder / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = [
        path for path in sorted(exp_folder.rglob("*.csv"))
        if not should_skip_path(path, exp_folder, output_dir, exclude_dirs)
    ]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {exp_folder}")

    print(f"Discovered CSV files: {len(csv_files)}")
    print(f"Approach filter:      {approaches}")
    print(f"Disturbance filter:   {sorted(disturbances)}")
    print(f"Terrain filter:       {sorted(terrain_filter) if terrain_filter else 'all'}")
    print(f"Output directory:     {output_dir}")

    worker_args = [
        (path, exp_folder, approaches, disturbances, terrain_filter, base_height_target, max_envs)
        for path in csv_files
    ]

    processed = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process_one_csv_from_args, worker_args):
            if result is not None:
                processed.append(result)

    if not processed:
        raise FileNotFoundError("No matching evaluation CSV files were loaded.")

    partitions = [
        "tracking_metrics",
        "torso_stability_metrics",
        "robot_state",
        "forces_power",
        "disturbances",
    ]

    output_counts = {}

    for partition in partitions:
        dfs = [result[partition] for result in processed if partition in result]

        if not dfs:
            print(f"[skip] {partition}: no rows")
            continue

        partition_df = pd.concat(dfs, ignore_index=True)
        output_path = output_dir / f"{partition}.csv"
        partition_df.to_csv(output_path, index=False)

        output_counts[partition] = len(partition_df)
        print(f"[save] {partition:24s} {len(partition_df):12,} rows -> {output_path}")

    print("\nDone.")
    print(f"Loaded source files: {len(processed)}")

    return output_counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create contained seaborn-ready CSV partitions from evaluation logs."
    )

    parser.add_argument("--exp_folder", type=str, required=True)
    parser.add_argument("--approaches", type=str, nargs="+", required=True)
    parser.add_argument("--disturbances", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="plot_data")
    parser.add_argument("--terrains", type=str, nargs="+", default=None)
    parser.add_argument("--exclude_dirs", type=str, nargs="+", default=["results_analysis", "plot_data"])
    parser.add_argument("--max_envs", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--base_height_target", type=float, default=0.30)

    return parser.parse_args()


def main():
    args = parse_args()

    build_contained_seaborn_csvs(
        exp_folder=args.exp_folder,
        approaches=args.approaches,
        disturbances=args.disturbances,
        output_dir=args.output_dir,
        terrains=args.terrains,
        exclude_dirs=args.exclude_dirs,
        max_envs=args.max_envs,
        max_workers=args.max_workers,
        base_height_target=args.base_height_target,
    )


if __name__ == "__main__":
    main()
