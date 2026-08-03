#!/usr/bin/env python3
"""Sweep B1 UniFP reference-gait and PD parameters in isolated Genesis workers.

The worker deliberately reuses the environment gait clock, stance schedule,
reference pose, and simulator PD controller.  Its only trajectory modification
is an additive fore-aft thigh sweep applied after ``compute_ref_state()``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


TASK_NAME = "b1_unifp"
RESULT_SENTINEL = "B1_GAIT_SWEEP_RESULT="
FOOT_NAMES = ("fr", "fl", "rr", "rl")
PARAMETER_FIELDS = (
    "hip_kp", "thigh_kp", "calf_kp",
    "hip_kd", "thigh_kd", "calf_kd",
    "sweep_phase_lead", "sweep_amplitude", "cycle_time",
    "target_joint_pos_scale", "target_joint_pos_thd",
)

# All score terms are dimensionless before weighting. Positive weights reward
# behavior; negative weights penalize it. Raw values are always retained in CSV.
SCORE_WEIGHTS = {
    "velocity_tracking": 4.0,
    "positive_swing": 2.0,
    "backward_velocity": -4.0,
    "contact_mismatch": -2.0,
    "invalid_touchdowns": -2.0,
    "joint_tracking": -1.0,
    "torque_over_90": -2.0,
    "torque_saturation": -4.0,
    "reset": -20.0,
    "excessive_tilt": -8.0,
    "insufficient_height": -8.0,
}

CONSTRAINT_THRESHOLDS = {
    "minimum_forward_vx": 0.05,
    "maximum_torque_ratio": 0.90,
    "maximum_contact_mismatch": 0.25,
    "maximum_abs_tilt": 0.50,
    "minimum_base_height": 0.35,
}

DEFAULTS = {
    "hip_kp": [150.0, 200.0, 250.0, 300.0],
    "thigh_kp": [150.0, 200.0, 250.0, 300.0],
    "calf_kp": [300.0, 400.0, 500.0],
    "hip_kd": [4.0, 5.0, 6.0, 7.5],
    "thigh_kd": [4.0, 5.0, 6.0, 7.5],
    "calf_kd": [8.0, 10.0, 12.5],
    "sweep_phase_lead": [0.0, 0.025, 0.05, 0.075, 0.10],
    "sweep_amplitude": [0.08, 0.10, 0.12],
    "cycle_time": [0.56, 0.64, 0.72, 0.80],
    "target_joint_pos_scale": [0.17, 0.20, 0.23],
    "target_joint_pos_thd": [0.4, 0.5, 0.6],
}

METRIC_FIELDS = (
    "reset_count", "mean_body_frame_vx", "std_body_frame_vx",
    "mean_abs_vx_command_error", "mean_abs_roll", "max_abs_roll",
    "mean_abs_pitch", "max_abs_pitch", "mean_base_height",
    "min_base_height", "joint_tracking_rmse",
    "mean_max_joint_tracking_error", "mean_commanded_torque_ratio",
    "max_commanded_torque_ratio", "torque_saturation_fraction",
    "overall_contact_mismatch", "mean_contacting_feet",
    "mean_valid_swing_displacement", "min_valid_swing_displacement",
    "positive_valid_swing_fraction",
) + tuple(
    f"{prefix}_{foot}"
    for prefix in (
        "contact_mismatch", "valid_touchdown_count",
        "invalid_touchdown_count", "mean_valid_swing_displacement",
        "min_valid_swing_displacement",
    )
    for foot in FOOT_NAMES
)

CONSTRAINT_FIELDS = (
    "is_forward", "all_feet_positive_swing", "no_reset",
    "torque_safe", "contact_consistent", "upright",
)
CSV_FIELDS = (
    "trial_id", *PARAMETER_FIELDS, "status", "error", "score",
    *METRIC_FIELDS, *CONSTRAINT_FIELDS,
)


def _float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("candidate lists cannot be empty")
    return list(dict.fromkeys(values))


def _command_vector(text: str) -> list[float]:
    """Parse x, y, yaw without deduplicating repeated zero components."""
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("command must contain exactly x,y,yaw")
    return values


def _gain_profiles(text: str) -> list[tuple[float, ...]]:
    """Parse ``hipKp:thighKp:calfKp:hipKd:thighKd:calfKd;...``."""
    profiles = []
    for raw_profile in text.split(";"):
        if not raw_profile.strip():
            continue
        values = tuple(float(value.strip()) for value in raw_profile.split(":"))
        if len(values) != 6:
            raise argparse.ArgumentTypeError(
                "each gain profile must contain six colon-separated values"
            )
        profiles.append(values)
    if not profiles:
        raise argparse.ArgumentTypeError("gain profile list cannot be empty")
    return list(dict.fromkeys(profiles))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep the existing B1 UniFP scripted reference gait.",
        epilog=(
            "Repository runtime arguments such as --headless, --cpu, --gpu, "
            "and --task are forwarded to legged_gym.utils.get_args()."
        ),
    )
    parser.add_argument("--strategy", choices=("random", "grid"), default="random")
    parser.add_argument("--max-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--settling-time", type=float, default=1.5)
    parser.add_argument("--evaluation-time", type=float, default=5.0)
    parser.add_argument("--command", type=_command_vector, default=[0.5, 0.0, 0.0])
    parser.add_argument("--max-abs-action", type=float, default=2.0)
    parser.add_argument("--trial-timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())

    parser.add_argument("--hip-kp-values", type=_float_list, default=DEFAULTS["hip_kp"])
    parser.add_argument("--thigh-kp-values", type=_float_list, default=DEFAULTS["thigh_kp"])
    parser.add_argument("--calf-kp-values", type=_float_list, default=DEFAULTS["calf_kp"])
    parser.add_argument("--hip-kd-values", type=_float_list, default=DEFAULTS["hip_kd"])
    parser.add_argument("--thigh-kd-values", type=_float_list, default=DEFAULTS["thigh_kd"])
    parser.add_argument("--calf-kd-values", type=_float_list, default=DEFAULTS["calf_kd"])
    parser.add_argument(
        "--phase-leads", "--sweep-phase-lead-values", dest="sweep_phase_lead_values",
        type=_float_list, default=DEFAULTS["sweep_phase_lead"],
    )
    parser.add_argument(
        "--sweep-amplitudes", "--sweep-amplitude-values", dest="sweep_amplitude_values",
        type=_float_list, default=DEFAULTS["sweep_amplitude"],
    )
    parser.add_argument(
        "--cycle-times", "--cycle-time-values", dest="cycle_time_values",
        type=_float_list, default=DEFAULTS["cycle_time"],
    )
    parser.add_argument(
        "--target-joint-pos-scales", dest="target_joint_pos_scale_values",
        type=_float_list, default=DEFAULTS["target_joint_pos_scale"],
    )
    parser.add_argument(
        "--target-joint-pos-thds", dest="target_joint_pos_thd_values",
        type=_float_list, default=DEFAULTS["target_joint_pos_thd"],
    )
    parser.add_argument(
        "--gain-profiles", type=_gain_profiles, default=None,
        help=(
            "Optional semicolon-separated paired gain profiles. Each profile "
            "is hipKp:thighKp:calfKp:hipKd:thighKd:calfKd."
        ),
    )

    # Internal worker arguments are intentionally hidden from normal help.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trial-json", type=str, help=argparse.SUPPRESS)
    return parser


def candidate_spaces(args: argparse.Namespace) -> tuple[list[str], list[list[Any]]]:
    if args.gain_profiles:
        names = ["gain_profile"]
        spaces: list[list[Any]] = [args.gain_profiles]
    else:
        names = list(PARAMETER_FIELDS[:6])
        spaces = [
            args.hip_kp_values, args.thigh_kp_values, args.calf_kp_values,
            args.hip_kd_values, args.thigh_kd_values, args.calf_kd_values,
        ]
    names.extend(PARAMETER_FIELDS[6:])
    spaces.extend([
        args.sweep_phase_lead_values, args.sweep_amplitude_values,
        args.cycle_time_values, args.target_joint_pos_scale_values,
        args.target_joint_pos_thd_values,
    ])
    return names, spaces


def unpack_candidate(names: list[str], values: Iterable[Any]) -> dict[str, float]:
    raw = dict(zip(names, values))
    if "gain_profile" in raw:
        gains = raw.pop("gain_profile")
        raw.update(dict(zip(PARAMETER_FIELDS[:6], gains)))
    return {name: float(raw[name]) for name in PARAMETER_FIELDS}


def generate_candidates(args: argparse.Namespace) -> tuple[list[dict[str, float]], int]:
    names, spaces = candidate_spaces(args)
    total = math.prod(len(space) for space in spaces)
    limit = total if args.max_trials <= 0 else min(args.max_trials, total)

    if args.strategy == "grid":
        combinations = itertools.islice(itertools.product(*spaces), limit)
        return [unpack_candidate(names, values) for values in combinations], total

    if args.max_trials <= 0:
        raise ValueError("random search requires a positive --max-trials")
    rng = random.Random(args.seed)
    candidates: list[dict[str, float]] = []
    seen: set[tuple[Any, ...]] = set()
    while len(candidates) < limit:
        sampled = tuple(rng.choice(space) for space in spaces)
        hashable = tuple(tuple(value) if isinstance(value, tuple) else value for value in sampled)
        if hashable in seen:
            continue
        seen.add(hashable)
        candidates.append(unpack_candidate(names, sampled))
    return candidates, total


def parameter_key(parameters: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"{float(parameters[name]):.12g}" for name in PARAMETER_FIELDS)


def trial_id(parameters: dict[str, Any]) -> str:
    payload = json.dumps(
        {name: float(parameters[name]) for name in PARAMETER_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def empty_metrics(reset_count: int = 0) -> dict[str, Any]:
    metrics = {name: 0.0 for name in METRIC_FIELDS}
    metrics["reset_count"] = int(reset_count)
    return metrics


def constraints_for(metrics: dict[str, Any]) -> dict[str, bool]:
    all_feet_positive = all(
        int(metrics[f"valid_touchdown_count_{foot}"]) > 0
        and float(metrics[f"min_valid_swing_displacement_{foot}"]) > 0.0
        for foot in FOOT_NAMES
    )
    return {
        "is_forward": metrics["mean_body_frame_vx"] > CONSTRAINT_THRESHOLDS["minimum_forward_vx"],
        "all_feet_positive_swing": all_feet_positive,
        "no_reset": int(metrics["reset_count"]) == 0,
        "torque_safe": metrics["max_commanded_torque_ratio"] <= CONSTRAINT_THRESHOLDS["maximum_torque_ratio"],
        "contact_consistent": (
            metrics["overall_contact_mismatch"] <= CONSTRAINT_THRESHOLDS["maximum_contact_mismatch"]
            and sum(int(metrics[f"invalid_touchdown_count_{foot}"]) for foot in FOOT_NAMES) == 0
        ),
        "upright": (
            metrics["max_abs_roll"] <= CONSTRAINT_THRESHOLDS["maximum_abs_tilt"]
            and metrics["max_abs_pitch"] <= CONSTRAINT_THRESHOLDS["maximum_abs_tilt"]
            and metrics["min_base_height"] >= CONSTRAINT_THRESHOLDS["minimum_base_height"]
        ),
    }


def score_metrics(metrics: dict[str, Any], command_vx: float) -> float:
    command_scale = max(abs(command_vx), 0.1)
    velocity_tracking = max(0.0, 1.0 - metrics["mean_abs_vx_command_error"] / command_scale)
    positive_swing = max(-1.0, min(1.0, metrics["mean_valid_swing_displacement"] / 0.10))
    backward_velocity = max(0.0, -metrics["mean_body_frame_vx"] / command_scale)
    invalid_count = sum(int(metrics[f"invalid_touchdown_count_{foot}"]) for foot in FOOT_NAMES)
    valid_count = sum(int(metrics[f"valid_touchdown_count_{foot}"]) for foot in FOOT_NAMES)
    invalid_touchdowns = invalid_count / max(1, invalid_count + valid_count)
    joint_tracking = min(2.0, metrics["joint_tracking_rmse"] / 0.25)
    torque_over_90 = min(2.0, max(0.0, (metrics["max_commanded_torque_ratio"] - 0.90) / 0.30))
    torque_saturation = min(2.0, metrics["torque_saturation_fraction"] / 0.10)
    tilt = max(metrics["max_abs_roll"], metrics["max_abs_pitch"])
    excessive_tilt = min(2.0, max(0.0, (tilt - 0.40) / 0.25))
    insufficient_height = min(2.0, max(0.0, (0.35 - metrics["min_base_height"]) / 0.15))
    components = {
        "velocity_tracking": velocity_tracking,
        "positive_swing": positive_swing,
        "backward_velocity": backward_velocity,
        "contact_mismatch": metrics["overall_contact_mismatch"],
        "invalid_touchdowns": invalid_touchdowns,
        "joint_tracking": joint_tracking,
        "torque_over_90": torque_over_90,
        "torque_saturation": torque_saturation,
        "reset": float(int(metrics["reset_count"]) > 0),
        "excessive_tilt": excessive_tilt,
        "insufficient_height": insufficient_height,
    }
    return sum(SCORE_WEIGHTS[name] * value for name, value in components.items())


def configure_trial(env_cfg: Any, parameters: dict[str, float], total_time: float) -> None:
    """Apply deterministic test settings before Genesis constructs the robot."""
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = max(float(env_cfg.env.episode_length_s), total_time + 2.0)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False
    env_cfg.terrain.obtain_terrain_info_around_feet = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = total_time + 100.0
    env_cfg.commands.push_robot_base = False
    env_cfg.commands.apply_base_external_forces = False
    env_cfg.noise.add_noise = False
    env_cfg.rewards.scales.foot_clearance_terrain_aware = 0.0
    env_cfg.init_state.leg_dof_pos_perturb_range = [0.0, 0.0]

    for name in (
        "randomize_friction", "randomize_base_mass", "randomize_com_displacement",
        "randomize_ctrl_delay", "randomize_pd_gain", "randomize_motor_strength",
        "randomize_joint_armature", "randomize_joint_friction",
        "randomize_joint_stiffness", "randomize_joint_damping", "push_robots",
    ):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)
    if hasattr(env_cfg.commands, "randomize_base_force_gains"):
        env_cfg.commands.randomize_base_force_gains = False

    env_cfg.control.stiffness = {
        "hip": parameters["hip_kp"],
        "thigh": parameters["thigh_kp"],
        "calf": parameters["calf_kp"],
    }
    env_cfg.control.damping = {
        "hip": parameters["hip_kd"],
        "thigh": parameters["thigh_kd"],
        "calf": parameters["calf_kd"],
    }
    env_cfg.rewards.cycle_time = parameters["cycle_time"]
    env_cfg.rewards.target_joint_pos_scale = parameters["target_joint_pos_scale"]
    env_cfg.rewards.target_joint_pos_thd = parameters["target_joint_pos_thd"]


def run_worker_trial(
    args: argparse.Namespace,
    parameters: dict[str, float],
    repository_args: list[str],
) -> dict[str, Any]:
    """Construct one fresh environment and return one complete trial result."""
    import torch

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *repository_args]
        # Existing shell launchers export this before importing legged_gym.
        # Direct CLI use should select the same B1-specific simulator backend.
        os.environ.setdefault("SIMULATOR", "genesis_b1_unifp")
        from legged_gym import SIMULATOR, gs
        import legged_gym.envs  # noqa: F401 -- registers tasks
        from legged_gym.utils import get_args, init_genesis, task_registry
        from legged_gym.utils.math_utils import quat_rotate_inverse

        repo_args = get_args()
        repo_args.task = TASK_NAME
        repo_args.seed = args.seed
        if "genesis" in SIMULATOR:
            init_genesis(repo_args, gs)

        env_cfg, _ = task_registry.get_cfgs(name=TASK_NAME, args=repo_args)
        configure_trial(env_cfg, parameters, args.settling_time + args.evaluation_time)
        env, _ = task_registry.make_env(name=TASK_NAME, args=repo_args, env_cfg=env_cfg)

        # B1 observations expect a terrain-relative foot-height tensor even on
        # a flat plane. Terrain sampling is disabled above, so provide its exact
        # zero-height equivalent without changing the environment implementation.
        if not hasattr(env.simulator, "_height_around_feet"):
            env.simulator._height_around_feet = torch.zeros(
                env.num_envs, len(env.simulator.feet_indices), 9,
                device=env.device,
            )
        env.reset()
        env.reset_buf.zero_()

        if env.num_actions != 12:
            raise RuntimeError(f"expected 12 B1 actions, got {env.num_actions}")
        if len(args.command) != 3:
            raise ValueError("--command must contain exactly x,y,yaw")

        command = torch.tensor(args.command, device=env.device, dtype=env.commands.dtype)
        idx = env.leg_dof_indices
        total_steps = int(round((args.settling_time + args.evaluation_time) / env.dt))
        settling_steps = int(round(args.settling_time / env.dt))

        def set_command() -> None:
            env.commands.zero_()
            env.commands[:, :3] = command

        def reference_action() -> tuple[torch.Tensor, torch.Tensor]:
            env.compute_ref_state()
            phase = env._get_phase()
            sweep_phase = torch.remainder(
                phase + parameters["sweep_phase_lead"], 1.0
            )
            direction = torch.sign(env.commands[:, 0])
            command_scale = torch.clamp(env.commands[:, 0].abs() / 0.5, 0.0, 1.0)
            sweep = (
                parameters["sweep_amplitude"] * command_scale * direction
                * torch.cos(2.0 * torch.pi * sweep_phase)
            )
            # The phase lead applies only to this additive fore-aft sweep.
            env.ref_dof_pos[:, idx["FR_thigh_joint"]] += sweep
            env.ref_dof_pos[:, idx["RL_thigh_joint"]] += sweep
            env.ref_dof_pos[:, idx["FL_thigh_joint"]] -= sweep
            env.ref_dof_pos[:, idx["RR_thigh_joint"]] -= sweep
            q_ref = env.ref_dof_pos[:, :12].clone()
            action_scale = float(env.cfg.control.action_scale)
            if action_scale <= 0.0:
                raise ValueError(f"control.action_scale must be positive: {action_scale}")
            action = (q_ref - env.simulator.default_dof_pos[:, :12]) / action_scale
            return action.clamp(-args.max_abs_action, args.max_abs_action), q_ref

        def contacts() -> torch.Tensor:
            return (
                env.simulator.link_contact_forces[:, env.simulator.feet_indices, 2] > 5.0
            )

        def feet_in_base() -> torch.Tensor:
            offset = env.simulator.feet_pos - env.simulator.base_pos.unsqueeze(1)
            quats = env.simulator.base_quat.unsqueeze(1).expand(-1, offset.shape[1], -1)
            return quat_rotate_inverse(
                quats.reshape(-1, 4), offset.reshape(-1, 3)
            ).view(env.num_envs, -1, 3)

        vx_values: list[float] = []
        vx_errors: list[float] = []
        abs_roll: list[float] = []
        abs_pitch: list[float] = []
        heights: list[float] = []
        joint_sq_errors: list[float] = []
        max_joint_errors: list[float] = []
        torque_ratios: list[float] = []
        torque_maxima: list[float] = []
        saturation_counts = 0
        torque_samples = 0
        mismatch_counts = [0, 0, 0, 0]
        contact_counts: list[float] = []
        valid_touchdowns = [0, 0, 0, 0]
        invalid_touchdowns = [0, 0, 0, 0]
        swing_displacements: list[list[float]] = [[], [], [], []]
        reset_count = 0
        previous_contact = None
        liftoff_x = torch.zeros(1, 4, device=env.device)
        swing_active = torch.zeros(1, 4, dtype=torch.bool, device=env.device)
        swing_overlapped_desired = torch.zeros_like(swing_active)

        status = "success"
        error = ""
        for step in range(total_steps):
            print(step)
            set_command()
            actions, q_ref = reference_action()
            print(actions)
            if not torch.isfinite(actions).all() or not torch.isfinite(q_ref).all():
                status, error = "failed", "nonfinite action or reference position"
                break

            env.step(actions)
            if bool(env.reset_buf.any()):
                reset_count += int(env.reset_buf.sum().item())
                status, error = "failed", "environment reset/fall during trial"
                break

            state_tensors = (
                env.simulator.base_lin_vel, env.simulator.base_euler,
                env.simulator.base_pos, env.simulator.dof_pos,
                env.simulator.unclipped_torques,
            )
            if not all(torch.isfinite(value).all() for value in state_tensors):
                status, error = "failed", "nonfinite simulator state"
                break

            if step < settling_steps:
                continue

            current_contact = contacts()
            foot_pos_base = feet_in_base()
            desired_stance = env._get_gait_phase().bool()
            if previous_contact is None:
                previous_contact = current_contact.clone()
            else:
                liftoff = previous_contact & ~current_contact
                touchdown = ~previous_contact & current_contact
                liftoff_x[liftoff] = foot_pos_base[:, :, 0][liftoff]
                swing_active[liftoff] = True
                swing_overlapped_desired[liftoff] = False
                swing_overlapped_desired |= swing_active & ~desired_stance
                valid_touchdown = touchdown & swing_active & swing_overlapped_desired
                invalid_touchdown = touchdown & ~valid_touchdown
                displacement = foot_pos_base[:, :, 0] - liftoff_x
                for foot in range(4):
                    if bool(valid_touchdown[0, foot]):
                        valid_touchdowns[foot] += 1
                        swing_displacements[foot].append(float(displacement[0, foot].item()))
                    if bool(invalid_touchdown[0, foot]):
                        invalid_touchdowns[foot] += 1
                swing_active[touchdown] = False
                swing_overlapped_desired[touchdown] = False
                previous_contact.copy_(current_contact)

            vx = float(env.simulator.base_lin_vel[0, 0].item())
            roll = abs(float(env.simulator.base_euler[0, 0].item()))
            pitch = abs(float(env.simulator.base_euler[0, 1].item()))
            height = float(env.simulator.base_pos[0, 2].item())
            joint_error = q_ref - env.simulator.dof_pos[:, :12]
            torque_ratio = (
                env.simulator.unclipped_torques[:, :12].abs()
                / env.simulator._torque_limits[:12].clamp_min(1.0e-6)
            )

            vx_values.append(vx)
            vx_errors.append(abs(vx - float(command[0].item())))
            abs_roll.append(roll)
            abs_pitch.append(pitch)
            heights.append(height)
            joint_sq_errors.extend(joint_error.square().flatten().tolist())
            max_joint_errors.append(float(joint_error.abs().max().item()))
            torque_ratios.extend(torque_ratio.flatten().tolist())
            torque_maxima.append(float(torque_ratio.max().item()))
            saturation_counts += int((torque_ratio >= 1.0).sum().item())
            torque_samples += torque_ratio.numel()
            mismatch = current_contact != desired_stance
            for foot in range(4):
                mismatch_counts[foot] += int(mismatch[0, foot].item())
            contact_counts.append(float(current_contact.float().sum().item()))

        if not vx_values:
            metrics = empty_metrics(reset_count)
            if status == "success":
                status, error = "failed", "no evaluation samples collected"
        else:
            def mean(values: list[float]) -> float:
                return sum(values) / len(values) if values else 0.0

            vx_mean = mean(vx_values)
            vx_std = math.sqrt(mean([(value - vx_mean) ** 2 for value in vx_values]))
            all_swings = [value for per_foot in swing_displacements for value in per_foot]
            metrics = {
                "reset_count": reset_count,
                "mean_body_frame_vx": vx_mean,
                "std_body_frame_vx": vx_std,
                "mean_abs_vx_command_error": mean(vx_errors),
                "mean_abs_roll": mean(abs_roll),
                "max_abs_roll": max(abs_roll),
                "mean_abs_pitch": mean(abs_pitch),
                "max_abs_pitch": max(abs_pitch),
                "mean_base_height": mean(heights),
                "min_base_height": min(heights),
                "joint_tracking_rmse": math.sqrt(mean(joint_sq_errors)),
                "mean_max_joint_tracking_error": mean(max_joint_errors),
                "mean_commanded_torque_ratio": mean(torque_ratios),
                "max_commanded_torque_ratio": max(torque_maxima),
                "torque_saturation_fraction": saturation_counts / max(1, torque_samples),
                "overall_contact_mismatch": sum(mismatch_counts) / (4.0 * len(vx_values)),
                "mean_contacting_feet": mean(contact_counts),
                "mean_valid_swing_displacement": mean(all_swings),
                "min_valid_swing_displacement": min(all_swings) if all_swings else 0.0,
                "positive_valid_swing_fraction": (
                    sum(value > 0.0 for value in all_swings) / len(all_swings)
                    if all_swings else 0.0
                ),
            }
            for foot, name in enumerate(FOOT_NAMES):
                values = swing_displacements[foot]
                metrics[f"contact_mismatch_{name}"] = mismatch_counts[foot] / len(vx_values)
                metrics[f"valid_touchdown_count_{name}"] = valid_touchdowns[foot]
                metrics[f"invalid_touchdown_count_{name}"] = invalid_touchdowns[foot]
                metrics[f"mean_valid_swing_displacement_{name}"] = mean(values)
                metrics[f"min_valid_swing_displacement_{name}"] = min(values) if values else 0.0

        constraints = constraints_for(metrics)
        score = score_metrics(metrics, float(command[0].item()))
        return {
            "trial_id": trial_id(parameters), **parameters,
            "status": status, "error": error, "score": score,
            **metrics, **constraints,
        }
    finally:
        sys.argv = original_argv


def failed_result(parameters: dict[str, float], error: str) -> dict[str, Any]:
    metrics = empty_metrics()
    return {
        "trial_id": trial_id(parameters), **parameters,
        "status": "failed", "error": error, "score": -1.0e9,
        **metrics, **constraints_for(metrics),
    }


def worker_main(
    args: argparse.Namespace,
    repository_args: list[str],
) -> int:
    parameters = json.loads(args.trial_json)
    try:
        result = run_worker_trial(args, parameters, repository_args)
    except Exception as exc:  # A trial failure must not terminate the sweep.
        result = failed_result(parameters, f"{type(exc).__name__}: {exc}")
    print(RESULT_SENTINEL + json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


def read_existing(path: Path) -> tuple[list[dict[str, str]], set[tuple[str, ...]]]:
    if not path.exists():
        return [], set()
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows, {parameter_key(row) for row in rows}


def append_result(path: Path, result: dict[str, Any], write_header: bool) -> None:
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in CSV_FIELDS})
        stream.flush()


def parse_worker_output(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_SENTINEL):
            return json.loads(line[len(RESULT_SENTINEL):])
    return None


def display_top(rows: list[dict[str, Any]]) -> None:
    successful = [row for row in rows if row.get("status") == "success"]
    successful.sort(key=lambda row: float(row["score"]), reverse=True)
    if not successful:
        print("No successful trials were recorded.")
        return
    columns = (
        ("rank", 4), ("trial", 12), ("score", 8), ("vx", 7),
        ("vx_err", 7), ("swing", 7), ("mismatch", 9), ("tau_max", 7),
    )
    print("\nTop successful configurations")
    print(" ".join(name.rjust(width) for name, width in columns))
    for rank, row in enumerate(successful[:10], 1):
        values = (
            str(rank), str(row["trial_id"]), f"{float(row['score']):.3f}",
            f"{float(row['mean_body_frame_vx']):.3f}",
            f"{float(row['mean_abs_vx_command_error']):.3f}",
            f"{float(row['mean_valid_swing_displacement']):.3f}",
            f"{float(row['overall_contact_mismatch']):.3f}",
            f"{float(row['max_commanded_torque_ratio']):.3f}",
        )
        print(" ".join(value.rjust(width) for value, (_, width) in zip(values, columns)))


def write_best_json(
    path: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    successful = [row for row in rows if row.get("status") == "success"]
    if not successful:
        payload = {
            "best_parameters": {}, "metrics": {}, "score": 0.0,
            "constraints": {}, "command": args.command,
            "settling_time": args.settling_time,
            "evaluation_time": args.evaluation_time, "seed": args.seed,
        }
    else:
        best = max(successful, key=lambda row: float(row["score"]))
        payload = {
            "best_parameters": {name: float(best[name]) for name in PARAMETER_FIELDS},
            "metrics": {name: float(best[name]) for name in METRIC_FIELDS},
            "score": float(best["score"]),
            "constraints": {
                name: str(best[name]).lower() in ("true", "1")
                if isinstance(best[name], str) else bool(best[name])
                for name in CONSTRAINT_FIELDS
            },
            "command": args.command,
            "settling_time": args.settling_time,
            "evaluation_time": args.evaluation_time,
            "seed": args.seed,
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parent_main(args: argparse.Namespace, repository_args: list[str]) -> int:
    candidates, total = generate_candidates(args)
    print(
        f"Candidate space: {total:,} unique combinations; "
        f"scheduled: {len(candidates):,} ({args.strategy}, seed={args.seed})"
    )
    for candidate in candidates[:5]:
        print("  " + json.dumps(candidate, sort_keys=True))
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "b1_gait_sweep_results.csv"
    json_path = args.output_dir / "b1_gait_sweep_best.json"
    if args.resume:
        existing_rows, completed = read_existing(csv_path)
    else:
        existing_rows, completed = [], set()
        if csv_path.exists():
            csv_path.unlink()

    results: list[dict[str, Any]] = list(existing_rows)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    pending = [candidate for candidate in candidates if parameter_key(candidate) not in completed]
    print(f"Pending trials: {len(pending):,}; skipped by resume: {len(candidates) - len(pending):,}")

    for index, parameters in enumerate(pending, 1):
        identifier = trial_id(parameters)
        print(f"[{index}/{len(pending)}] trial {identifier}", flush=True)
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker", "--trial-json", json.dumps(parameters, separators=(",", ":")),
            "--seed", str(args.seed),
            "--settling-time", str(args.settling_time),
            "--evaluation-time", str(args.evaluation_time),
            "--command", ",".join(str(value) for value in args.command),
            "--max-abs-action", str(args.max_abs_action),
            *repository_args,
        ]
        try:
            completed_process = subprocess.run(
                command, capture_output=True, text=True,
                timeout=args.trial_timeout, check=False,
            )
            result = parse_worker_output(completed_process.stdout)
            if result is None:
                detail = (completed_process.stderr or completed_process.stdout)[-2000:].strip()
                result = failed_result(
                    parameters,
                    f"worker exited {completed_process.returncode} without result: {detail}",
                )
        except subprocess.TimeoutExpired:
            result = failed_result(parameters, f"worker timeout after {args.trial_timeout:.1f}s")

        append_result(csv_path, result, write_header)
        write_header = False
        results.append(result)
        print(
            f"  {result['status']}: score={float(result['score']):.3f} "
            f"vx={float(result['mean_body_frame_vx']):.3f} "
            f"error={result.get('error', '')}"
        )

    write_best_json(json_path, results, args)
    display_top(results)
    print(f"\nCSV:  {csv_path}")
    print(f"Best: {json_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args, repository_args = parser.parse_known_args()
    if args.settling_time < 0.0 or args.evaluation_time <= 0.0:
        parser.error("settling time must be nonnegative and evaluation time positive")
    if args.max_abs_action <= 0.0:
        parser.error("--max-abs-action must be positive")
    if len(args.command) != 3:
        parser.error("--command must contain exactly three comma-separated values")
    if args.worker:
        if not args.trial_json:
            parser.error("internal worker requires --trial-json")
        return worker_main(args, repository_args)
    return parent_main(args, repository_args)


if __name__ == "__main__":
    raise SystemExit(main())
