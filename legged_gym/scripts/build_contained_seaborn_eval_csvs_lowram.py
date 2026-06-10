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
import csv
import gzip
import os
import shutil
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


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


def normalize_array_shape(arr, metric_name=None, max_envs=None):
    """
    Normalize a logged array to shape [num_envs, flat_dim].

    Common logged shapes:
        [dim]
        [num_envs]
        [num_envs, dim]
        [num_envs, feet, dim]
    """
    if arr is None:
        return None

    arr = np.asarray(arr, dtype=float)
    expected_dim = None
    if metric_name is not None and metric_name != "failure":
        names = COMPONENT_NAMES.get(metric_name)
        if names is not None:
            expected_dim = len(names)

    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if expected_dim is not None and arr.size == expected_dim:
            arr = arr.reshape(1, expected_dim)
        else:
            arr = arr[:, None]
    elif expected_dim is not None:
        trailing_dim = int(np.prod(arr.shape[1:]))
        if trailing_dim != expected_dim and arr.size == expected_dim:
            arr = arr.reshape(1, expected_dim)

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
    """Parse all columns needed by the downstream metric builders.

    `row` may be either a pandas Series from the old implementation or, in the
    low-RAM path below, a plain dictionary from csv.DictReader. Missing columns
    and empty cells are skipped.
    """
    arrays = {}

    row_keys = row.keys() if hasattr(row, "keys") else row.index

    for col in RAW_COLUMNS_NEEDED:
        if col not in row_keys:
            continue

        value = row[col]
        if value is None or value == "":
            continue

        arr = parse_array_cell(value)
        flat = normalize_array_shape(arr, metric_name=col, max_envs=max_envs)

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

OUTPUT_COLUMNS = [
    "approach", "terrain", "disturbance", "source_file", "source_relpath",
    "timestep", "env_id", "metric", "component", "value", "row_type", "is_failure",
]

ALL_PARTITIONS = [
    "tracking_metrics",
    "torso_stability_metrics",
    "robot_state",
    "forces_power",
    "disturbances",
]


def open_text_maybe_gzip(path, mode="rt", newline=""):
    """Open regular text files or gzip files using a common interface."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline=newline)
    return open(path, mode, newline=newline)


def make_partition_writers(output_paths):
    """Create CSV DictWriters for all requested partition shard paths."""
    handles = {}
    writers = {}

    try:
        for partition, path in output_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(path, "w", newline="")
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            handles[partition] = handle
            writers[partition] = writer
    except Exception:
        for handle in handles.values():
            handle.close()
        raise

    return handles, writers


def close_handles(handles):
    for handle in handles.values():
        handle.close()


def write_rows_buffered(writer, rows, count):
    """Write rows immediately and return the updated output row count."""
    if rows:
        writer.writerows(rows)
        count += len(rows)
    return count


def process_one_csv_from_args(args):
    """Unpack arguments for process-pool execution."""
    return process_one_csv_to_shards(*args)


# -----------------------------------------------------------------------------
# Per-file and top-level pipeline
# -----------------------------------------------------------------------------
def process_one_csv_to_shards(
    path,
    exp_folder,
    approaches,
    disturbances,
    terrain_filter,
    base_height_target,
    max_envs,
    shard_dir,
    partitions,
):
    """Stream one raw evaluation CSV and write partition-specific shard CSVs.

    This function intentionally never builds a full raw DataFrame, never stores a
    full processed DataFrame, and never returns rows to the parent process. It
    reads one source row at a time and writes expanded long-format rows directly
    to temporary partition shards.
    """
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

    path = Path(path)
    exp_folder = Path(exp_folder)
    shard_dir = Path(shard_dir)
    partitions = list(partitions)

    # Include the source stem in the shard filename for easier debugging, but add
    # pid to avoid collisions when repeated filenames exist in different folders.
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in path.stem)
    prefix = f"{safe_stem}.{os.getpid()}"
    shard_paths = {p: shard_dir / f"{prefix}.{p}.csv" for p in partitions}

    handles, writers = make_partition_writers(shard_paths)
    counts = {p: 0 for p in partitions}
    source_rows = 0

    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return None

            # Keep parsing narrow. DictReader still reads all CSV fields, but the
            # downstream parser touches only the columns in RAW_COLUMNS_NEEDED.
            for timestep, row in enumerate(reader):
                source_rows += 1
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

                if "tracking_metrics" in writers:
                    counts["tracking_metrics"] = write_rows_buffered(
                        writers["tracking_metrics"],
                        build_tracking_rows(arrays, meta, failure_flags, base_height_target),
                        counts["tracking_metrics"],
                    )

                if "torso_stability_metrics" in writers:
                    counts["torso_stability_metrics"] = write_rows_buffered(
                        writers["torso_stability_metrics"],
                        build_stability_rows(arrays, meta, failure_flags),
                        counts["torso_stability_metrics"],
                    )

                if "robot_state" in writers:
                    counts["robot_state"] = write_rows_buffered(
                        writers["robot_state"],
                        build_robot_state_rows(arrays, meta, failure_flags),
                        counts["robot_state"],
                    )

                if "forces_power" in writers:
                    counts["forces_power"] = write_rows_buffered(
                        writers["forces_power"],
                        build_force_power_rows(arrays, meta, failure_flags),
                        counts["forces_power"],
                    )

                if "disturbances" in writers:
                    counts["disturbances"] = write_rows_buffered(
                        writers["disturbances"],
                        build_disturbance_rows(arrays, meta, failure_flags),
                        counts["disturbances"],
                    )

    except Exception as exc:
        print(f"[skip] failed while processing {path}: {exc}")
        return None
    finally:
        close_handles(handles)

    # Delete empty shard files so final merge does less work.
    nonempty_paths = {}
    for partition, shard_path in shard_paths.items():
        if counts.get(partition, 0) > 0:
            nonempty_paths[partition] = str(shard_path)
        else:
            try:
                shard_path.unlink()
            except FileNotFoundError:
                pass

    if not nonempty_paths:
        return None

    print(f"[processed] {path} ({source_rows:,} source rows)")
    return {
        "path": str(path),
        "counts": counts,
        "shard_paths": nonempty_paths,
    }


def merge_partition_shards(shard_paths, output_path, gzip_outputs=False):
    """Concatenate partition shard CSVs into one final CSV without loading them."""
    output_path = Path(output_path)
    if gzip_outputs:
        output_path = output_path.with_suffix(output_path.suffix + ".gz")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open_text_maybe_gzip(output_path, "wt", newline="") as fout:
        fout.write(",".join(OUTPUT_COLUMNS) + "\n")

        for shard_path in shard_paths:
            shard_path = Path(shard_path)
            with open(shard_path, "r", newline="") as fin:
                # Drop the shard header before streaming into the final file.
                next(fin, None)
                shutil.copyfileobj(fin, fout, length=1024 * 1024)

    return output_path


def build_contained_seaborn_csvs(
    exp_folder,
    approaches,
    disturbances,
    output_dir="plot_data",
    terrains=None,
    exclude_dirs=None,
    max_envs=None,
    max_workers=1,
    base_height_target=0.30,
    partitions=None,
    gzip_outputs=False,
    keep_shards=False,
):
    """Main pipeline: discover, stream-process, and save CSV partitions.

    RAM-saving design choices:
      - raw input files are read row-by-row with csv.DictReader
      - processed rows are written immediately to temporary shard CSVs
      - workers return only shard paths and row counts
      - final outputs are produced by streaming shard concatenation
      - no full raw or processed pandas DataFrames are created

    `max_workers=1` is the safest default for very large datasets. Increasing it
    can improve throughput, but peak RAM scales roughly with the number of active
    workers because each worker parses one timestep's arrays independently.
    """
    exp_folder = Path(exp_folder)
    approaches = list(approaches)
    disturbances = set(disturbances)
    terrain_filter = None if terrains is None else set(terrains)
    partitions = ALL_PARTITIONS if partitions is None else list(partitions)

    invalid = sorted(set(partitions) - set(ALL_PARTITIONS))
    if invalid:
        raise ValueError(f"Unknown partition(s): {invalid}. Valid options: {ALL_PARTITIONS}")

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
    print(f"Partitions:           {partitions}")
    print(f"Output directory:     {output_dir}")
    print(f"Max workers:          {max_workers}")
    print(f"Gzip outputs:         {gzip_outputs}")

    output_counts = {p: 0 for p in partitions}
    shard_paths_by_partition = {p: [] for p in partitions}
    loaded_files = 0

    with tempfile.TemporaryDirectory(prefix="seaborn_csv_shards_", dir=str(output_dir)) as tmp:
        shard_dir = Path(tmp)

        worker_args = [
            (
                path,
                exp_folder,
                approaches,
                disturbances,
                terrain_filter,
                base_height_target,
                max_envs,
                shard_dir,
                partitions,
            )
            for path in csv_files
        ]

        if max_workers == 1:
            for args in worker_args:
                result = process_one_csv_from_args(args)
                if result is None:
                    continue
                loaded_files += 1
                for partition, count in result["counts"].items():
                    output_counts[partition] += count
                for partition, shard_path in result["shard_paths"].items():
                    shard_paths_by_partition[partition].append(shard_path)
        else:
            # Keep only futures and small metadata in memory; never collect rows.
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_one_csv_from_args, args) for args in worker_args]
                for future in as_completed(futures):
                    result = future.result()
                    if result is None:
                        continue
                    loaded_files += 1
                    for partition, count in result["counts"].items():
                        output_counts[partition] += count
                    for partition, shard_path in result["shard_paths"].items():
                        shard_paths_by_partition[partition].append(shard_path)

        if loaded_files == 0:
            raise FileNotFoundError("No matching evaluation CSV files were loaded.")

        final_paths = {}
        for partition in partitions:
            shard_paths = shard_paths_by_partition[partition]
            if not shard_paths:
                print(f"[skip] {partition}: no rows")
                continue

            output_path = merge_partition_shards(
                shard_paths,
                output_dir / f"{partition}.csv",
                gzip_outputs=gzip_outputs,
            )
            final_paths[partition] = str(output_path)
            print(f"[save] {partition:24s} {output_counts[partition]:12,} rows -> {output_path}")

        if keep_shards:
            keep_dir = output_dir / "_debug_shards"
            keep_dir.mkdir(parents=True, exist_ok=True)
            for paths in shard_paths_by_partition.values():
                for shard_path in paths:
                    shutil.copy2(shard_path, keep_dir / Path(shard_path).name)
            print(f"[debug] copied temporary shards to {keep_dir}")

    print("\nDone.")
    print(f"Loaded source files: {loaded_files}")

    return output_counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create contained seaborn-ready CSV partitions from evaluation logs with low RAM usage."
    )

    parser.add_argument("--exp_folder", type=str, required=True)
    parser.add_argument("--approaches", type=str, nargs="+", required=True)
    parser.add_argument("--disturbances", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="largescale_plot_data")
    parser.add_argument("--terrains", type=str, nargs="+", default=None)
    parser.add_argument("--exclude_dirs", type=str, nargs="+", default=["results_analysis", "plot_data", "payload_max", "wrench_max"])
    parser.add_argument("--max_envs", type=int, default=None)
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Use 1 for lowest RAM. Higher values can be faster but multiply peak RAM.",
    )
    parser.add_argument("--base_height_target", type=float, default=0.30)
    parser.add_argument(
        "--partitions",
        type=str,
        nargs="+",
        default=None,
        choices=ALL_PARTITIONS,
        help="Optional subset of output partitions to generate. Useful for reducing disk/RAM/time.",
    )
    parser.add_argument(
        "--gzip_outputs",
        action="store_true",
        help="Write final outputs as .csv.gz files. Saves disk, usually costs CPU.",
    )
    parser.add_argument(
        "--keep_shards",
        action="store_true",
        help="Copy temporary per-source shard files into output_dir/_debug_shards before cleanup.",
    )

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
        partitions=args.partitions,
        gzip_outputs=args.gzip_outputs,
        keep_shards=args.keep_shards,
    )


if __name__ == "__main__":
    main()
