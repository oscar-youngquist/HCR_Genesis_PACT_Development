#!/usr/bin/env python3
"""Successive-refinement search built on the B1 scripted gait evaluator.

Legacy grid/random invocations delegate to the original sweep unchanged. The
subclass in this file adds continuous staged search, robust multi-seed ranking,
and resumable refinement without duplicating simulator evaluation code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import b1_scripted_gait_parameter_sweep as base


SR_METADATA_FIELDS = (
    "evaluation_id", "stage", "round", "evaluation_seed",
    "evaluation_duration", "early_rejection_reason", "parent_elite_trial_id",
)
SR_DETAILED_FIELDS = (*SR_METADATA_FIELDS, *base.CSV_FIELDS)
AGGREGATE_FIELDS = (
    "trial_id", *base.PARAMETER_FIELDS, "aggregate_stage", "aggregate_round",
    "evaluation_count", "score_mean", "score_std", "score_min",
    "failure_rate", "feasible_fraction", "is_robust_feasible", "robust_score",
    *(f"mean_{name}" for name in base.METRIC_FIELDS),
)
FEASIBILITY_FIELDS = ("is_forward", "no_reset", "torque_safe", "contact_consistent", "upright")


def int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("integer list cannot be empty")
    return list(dict.fromkeys(values))


def trial_count_list(text: str) -> list[int]:
    values = int_list(text)
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("per-round trial counts must be positive")
    return values


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def evaluation_id(
    parameters: dict[str, float], stage: str, round_index: int,
    seed: int, duration: float,
) -> str:
    payload = {
        "parameters": {name: float(parameters[name]) for name in base.PARAMETER_FIELDS},
        "stage": stage, "round": int(round_index), "seed": int(seed),
        "duration": float(duration),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def quantile(values: list[float], q: float) -> float:
    return base.percentile(values, q)


class B1GaitSweepController:
    """Compatibility controller that delegates established sweep behavior."""

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        parser = base.build_parser()
        strategy_action = next(action for action in parser._actions if action.dest == "strategy")
        strategy_action.choices = ("random", "grid", "successive")

        group = parser.add_argument_group("successive refinement")
        group.add_argument("--sr-rounds", type=int, default=None,
                           help="Number of local-refinement rounds (required for successive).")
        group.add_argument("--sr-trials-per-round", type=trial_count_list, default=None,
                           help="One count for every stage, or broad plus one count per local round.")
        group.add_argument("--sr-elite-count", type=int, default=None,
                           help="Top candidate count used for validation and local centers.")
        group.add_argument("--sr-elite-fraction", type=float, default=None,
                           help="Top candidate fraction; mutually exclusive with --sr-elite-count.")
        group.add_argument("--sr-global-fraction", type=float, default=0.25,
                           help="Fraction of each local round sampled globally (default: 0.25).")
        group.add_argument("--sr-shrink-factor", type=float, default=0.5,
                           help="Neighborhood-width multiplier after each local round.")
        group.add_argument("--sr-min-width-fraction", type=float, default=0.05,
                           help="Minimum local width as a fraction of each original bound.")
        group.add_argument("--sr-broad-seed", type=int, default=None,
                           help="Deterministic continuous-search seed (required for successive).")
        group.add_argument("--sr-validation-seeds", type=int_list, default=None,
                           help="Comma-separated seeds for robust and optional final validation.")
        group.add_argument("--sr-broad-evaluation-time", type=float, default=None)
        group.add_argument("--sr-validation-evaluation-time", type=float, default=None)
        group.add_argument("--sr-refinement-evaluation-time", type=float, default=None)
        group.add_argument("--sr-risk-weight", type=float, default=1.0)
        group.add_argument("--sr-failure-weight", type=float, default=20.0)
        group.add_argument("--sr-required-feasible-fraction", type=float, default=0.8)
        group.add_argument("--sr-early-reject", action="store_true")
        group.add_argument("--sr-early-reject-grace-period", type=float, default=None)
        group.add_argument("--sr-early-reject-torque-window", type=float, default=None)
        group.add_argument("--sr-early-reject-torque-fraction", type=float, default=0.25)
        group.add_argument("--sr-early-reject-penalty", type=float, default=20.0)
        group.add_argument("--sr-final-validation-count", type=int, default=0,
                           help="Number of refined candidates to validate over all validation seeds.")
        group.add_argument("--sr-output-prefix", type=str, default="b1_gait_successive")
        group.add_argument("--sr-summary-lower-quantile", type=float, default=0.10)
        group.add_argument("--sr-summary-upper-quantile", type=float, default=0.90)
        group.add_argument("--sr-summary-margin-fraction", type=float, default=0.05)

        # Metadata is passed only to isolated workers spawned by this controller.
        group.add_argument("--sr-evaluation-stage", default=None, help=argparse.SUPPRESS)
        group.add_argument("--sr-evaluation-round", type=int, default=0, help=argparse.SUPPRESS)
        group.add_argument("--sr-evaluation-seed", type=int, default=None, help=argparse.SUPPRESS)
        group.add_argument("--sr-evaluation-duration", type=float, default=None, help=argparse.SUPPRESS)
        group.add_argument("--sr-parent-elite", default="", help=argparse.SUPPRESS)
        return parser

    def __init__(self, args: argparse.Namespace, repository_args: list[str]):
        self.args = args
        self.repository_args = repository_args

    def run(self) -> int:
        if self.args.worker:
            return self.worker_main()
        return base.parent_main(self.args, self.repository_args)

    def worker_main(self) -> int:
        return base.worker_main(self.args, self.repository_args)


class SuccessiveRefinementSweep(B1GaitSweepController):
    """Continuous broad/robust/local search with deterministic resume."""

    def __init__(self, args: argparse.Namespace, repository_args: list[str]):
        super().__init__(args, repository_args)
        # Isolated workers only evaluate one fully specified trial. Parent-only
        # search budgets and bounds are intentionally not copied to subprocesses.
        if args.worker:
            required = {
                "--sr-evaluation-stage": args.sr_evaluation_stage,
                "--sr-evaluation-seed": args.sr_evaluation_seed,
                "--sr-evaluation-duration": args.sr_evaluation_duration,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError("successive worker requires " + ", ".join(missing))
            return
        self.validate_args()
        self.bounds = self.infer_bounds()
        self.fixed = {name for name, (low, high) in self.bounds.items() if low == high}
        self.counts = self.round_counts()
        prefix = args.sr_output_prefix
        self.output_dir = args.output_dir
        self.detailed_path = self.output_dir / f"{prefix}_results.csv"
        self.aggregate_path = self.output_dir / f"{prefix}_aggregate.csv"
        self.state_path = self.output_dir / f"{prefix}_state.json"
        self.best_path = self.output_dir / f"{prefix}_best.json"
        self.summary_path = self.output_dir / f"{prefix}_summary.json"
        self.rows: list[dict[str, Any]] = []
        self.completed_ids: set[str] = set()
        self.state: dict[str, Any] = {
            "version": 1,
            "settings_signature": self.settings_signature(),
            "original_bounds": self.bounds,
            "fixed_parameters": sorted(self.fixed),
            "plans": [],
            "current_stage": "not_started",
            "final_local_bounds": self.bounds,
        }

    def validate_args(self) -> None:
        a = self.args
        required = {
            "--sr-rounds": a.sr_rounds,
            "--sr-trials-per-round": a.sr_trials_per_round,
            "--sr-broad-seed": a.sr_broad_seed,
            "--sr-validation-seeds": a.sr_validation_seeds,
            "--sr-broad-evaluation-time": a.sr_broad_evaluation_time,
            "--sr-validation-evaluation-time": a.sr_validation_evaluation_time,
            "--sr-refinement-evaluation-time": a.sr_refinement_evaluation_time,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("successive strategy requires " + ", ".join(missing))
        if a.sr_rounds < 0:
            raise ValueError("--sr-rounds must be nonnegative")
        if len(a.sr_trials_per_round) not in (1, a.sr_rounds + 1):
            raise ValueError(
                "--sr-trials-per-round needs one value or broad + one value per refinement round"
            )
        if (a.sr_elite_count is None) == (a.sr_elite_fraction is None):
            raise ValueError("set exactly one of --sr-elite-count or --sr-elite-fraction")
        if a.sr_elite_count is not None and a.sr_elite_count <= 0:
            raise ValueError("--sr-elite-count must be positive")
        if a.sr_elite_fraction is not None and not 0.0 < a.sr_elite_fraction <= 1.0:
            raise ValueError("--sr-elite-fraction must be in (0, 1]")
        if not 0.0 <= a.sr_global_fraction <= 1.0:
            raise ValueError("--sr-global-fraction must be in [0, 1]")
        if not 0.0 < a.sr_shrink_factor <= 1.0:
            raise ValueError("--sr-shrink-factor must be in (0, 1]")
        if not 0.0 <= a.sr_min_width_fraction <= 1.0:
            raise ValueError("--sr-min-width-fraction must be in [0, 1]")
        if any(value <= 0.0 for value in (
            a.sr_broad_evaluation_time, a.sr_validation_evaluation_time,
            a.sr_refinement_evaluation_time,
        )):
            raise ValueError("successive evaluation durations must be positive")
        if not 0.0 <= a.sr_required_feasible_fraction <= 1.0:
            raise ValueError("--sr-required-feasible-fraction must be in [0, 1]")
        if a.sr_risk_weight < 0.0 or a.sr_failure_weight < 0.0:
            raise ValueError("risk and failure weights must be nonnegative")
        if a.sr_final_validation_count < 0:
            raise ValueError("--sr-final-validation-count cannot be negative")
        if a.sr_early_reject:
            if a.sr_early_reject_grace_period is None or a.sr_early_reject_grace_period < 0.0:
                raise ValueError("early rejection requires nonnegative --sr-early-reject-grace-period")
            if a.sr_early_reject_torque_window is None or a.sr_early_reject_torque_window <= 0.0:
                raise ValueError("early rejection requires positive --sr-early-reject-torque-window")
            if not 0.0 <= a.sr_early_reject_torque_fraction <= 1.0:
                raise ValueError("--sr-early-reject-torque-fraction must be in [0, 1]")
        if not 0.0 <= a.sr_summary_lower_quantile <= a.sr_summary_upper_quantile <= 1.0:
            raise ValueError("summary quantiles must satisfy 0 <= lower <= upper <= 1")
        if a.sr_summary_margin_fraction < 0.0:
            raise ValueError("--sr-summary-margin-fraction must be nonnegative")

    def round_counts(self) -> list[int]:
        values = self.args.sr_trials_per_round
        return values * (self.args.sr_rounds + 1) if len(values) == 1 else values

    def infer_bounds(self) -> dict[str, tuple[float, float]]:
        if self.args.gain_profiles:
            gain_values = list(zip(*self.args.gain_profiles))
            values = {name: list(gain_values[index]) for index, name in enumerate(base.PARAMETER_FIELDS[:6])}
        else:
            values = {
                "hip_kp": self.args.hip_kp_values,
                "thigh_kp": self.args.thigh_kp_values,
                "calf_kp": self.args.calf_kp_values,
                "hip_kd": self.args.hip_kd_values,
                "thigh_kd": self.args.thigh_kd_values,
                "calf_kd": self.args.calf_kd_values,
            }
        values.update({
            "sweep_phase_lead": self.args.sweep_phase_lead_values,
            "sweep_amplitude": self.args.sweep_amplitude_values,
            "cycle_time": self.args.cycle_time_values,
            "target_joint_pos_scale": self.args.target_joint_pos_scale_values,
            "target_joint_pos_thd": self.args.target_joint_pos_thd_values,
        })
        return {name: (float(min(values[name])), float(max(values[name]))) for name in base.PARAMETER_FIELDS}

    def settings_signature(self) -> str:
        payload = {
            "bounds": self.bounds,
            "rounds": self.args.sr_rounds,
            "counts": self.counts,
            "elite_count": self.args.sr_elite_count,
            "elite_fraction": self.args.sr_elite_fraction,
            "global_fraction": self.args.sr_global_fraction,
            "shrink": self.args.sr_shrink_factor,
            "min_width": self.args.sr_min_width_fraction,
            "broad_seed": self.args.sr_broad_seed,
            "validation_seeds": self.args.sr_validation_seeds,
            "durations": [
                self.args.sr_broad_evaluation_time,
                self.args.sr_validation_evaluation_time,
                self.args.sr_refinement_evaluation_time,
            ],
            "final_validation_count": self.args.sr_final_validation_count,
            "risk_weight": self.args.sr_risk_weight,
            "failure_weight": self.args.sr_failure_weight,
            "required_feasible_fraction": self.args.sr_required_feasible_fraction,
            "early_reject": [
                self.args.sr_early_reject,
                self.args.sr_early_reject_grace_period,
                self.args.sr_early_reject_torque_window,
                self.args.sr_early_reject_torque_fraction,
                self.args.sr_early_reject_penalty,
            ],
            "trial": {
                "settling_time": self.args.settling_time,
                "command": self.args.command,
                "max_abs_action": self.args.max_abs_action,
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def elite_count(self, population: int) -> int:
        if population <= 0:
            return 0
        if self.args.sr_elite_count is not None:
            return min(population, self.args.sr_elite_count)
        return min(population, max(1, int(math.ceil(population * self.args.sr_elite_fraction))))

    def sample_continuous(
        self, count: int, bounds: dict[str, tuple[float, float]],
        rng: random.Random, seen: set[tuple[str, ...]],
    ) -> list[dict[str, float]]:
        candidates = []
        attempts = 0
        while len(candidates) < count:
            attempts += 1
            if attempts > max(1000, 100 * count):
                raise ValueError(
                    "unable to generate enough unique continuous candidates; "
                    "reduce the requested budget or unfix a parameter"
                )
            candidate = {
                name: low if low == high else rng.uniform(low, high)
                for name, (low, high) in bounds.items()
            }
            key = base.parameter_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        return candidates

    def local_bounds(
        self, center: dict[str, float], round_index: int,
    ) -> dict[str, tuple[float, float]]:
        width_fraction = max(
            self.args.sr_min_width_fraction,
            self.args.sr_shrink_factor ** round_index,
        )
        result = {}
        for name, (original_low, original_high) in self.bounds.items():
            if original_low == original_high:
                result[name] = (original_low, original_high)
                continue
            half_width = 0.5 * (original_high - original_low) * width_fraction
            result[name] = (
                max(original_low, center[name] - half_width),
                min(original_high, center[name] + half_width),
            )
        return result

    def broad_candidates(self) -> list[dict[str, Any]]:
        seen: set[tuple[str, ...]] = set()
        rng = random.Random(stable_seed(self.args.sr_broad_seed, "broad"))
        return [
            {"parameters": candidate, "parent_elite_trial_id": ""}
            for candidate in self.sample_continuous(self.counts[0], self.bounds, rng, seen)
        ]

    def refinement_candidates(
        self, elites: list[dict[str, Any]], round_index: int,
        all_seen: set[tuple[str, ...]],
    ) -> tuple[list[dict[str, Any]], int, int]:
        count = self.counts[round_index]
        global_count = min(count, int(round(count * self.args.sr_global_fraction)))
        local_count = count - global_count
        rng = random.Random(stable_seed(self.args.sr_broad_seed, "refinement", round_index))
        planned: list[dict[str, Any]] = []
        for candidate in self.sample_continuous(global_count, self.bounds, rng, all_seen):
            planned.append({"parameters": candidate, "parent_elite_trial_id": ""})
        for index in range(local_count):
            elite = elites[index % len(elites)]
            bounds = self.local_bounds(elite, round_index)
            candidate = self.sample_continuous(1, bounds, rng, all_seen)[0]
            planned.append({
                "parameters": candidate,
                "parent_elite_trial_id": base.trial_id(elite),
            })
        return planned, global_count, local_count

    def make_evaluations(
        self, candidates: list[dict[str, Any]], stage: str, round_index: int,
        seeds: Iterable[int], duration: float,
    ) -> list[dict[str, Any]]:
        evaluations = []
        for candidate in candidates:
            for seed in seeds:
                parameters = candidate["parameters"]
                evaluations.append({
                    "evaluation_id": evaluation_id(parameters, stage, round_index, seed, duration),
                    "parameters": parameters,
                    "stage": stage,
                    "round": round_index,
                    "seed": int(seed),
                    "duration": float(duration),
                    "parent_elite_trial_id": candidate.get("parent_elite_trial_id", ""),
                })
        return evaluations

    def persist_state(self, stage: str, evaluations: list[dict[str, Any]] | None = None) -> None:
        self.state["current_stage"] = stage
        if evaluations:
            known = {entry["evaluation_id"] for entry in self.state["plans"]}
            self.state["plans"].extend(entry for entry in evaluations if entry["evaluation_id"] not in known)
        self.state_path.write_text(
            json.dumps(self.state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def planned_evaluations(self, stage: str, round_index: int) -> list[dict[str, Any]]:
        """Return the immutable saved plan for one stage/round during resume."""
        return [
            entry for entry in self.state["plans"]
            if entry["stage"] == stage and int(entry["round"]) == round_index
        ]

    def load_or_initialize(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.args.resume:
            for path in (
                self.detailed_path, self.aggregate_path, self.state_path,
                self.best_path, self.summary_path,
            ):
                if path.exists():
                    path.unlink()
            return
        if self.state_path.exists():
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            if loaded.get("settings_signature") != self.state["settings_signature"]:
                raise ValueError("resume settings do not match the persisted successive-refinement plan")
            self.state = loaded
        if self.detailed_path.exists():
            with self.detailed_path.open(newline="", encoding="utf-8") as stream:
                self.rows = list(csv.DictReader(stream))
            self.completed_ids = {row["evaluation_id"] for row in self.rows}

    def append_result(self, result: dict[str, Any]) -> None:
        write_header = not self.detailed_path.exists() or self.detailed_path.stat().st_size == 0
        with self.detailed_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SR_DETAILED_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({field: result.get(field, "") for field in SR_DETAILED_FIELDS})
            stream.flush()
        self.rows.append(result)
        self.completed_ids.add(result["evaluation_id"])

    def worker_command(self, evaluation: dict[str, Any]) -> list[str]:
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker", "--trial-json", json.dumps(evaluation["parameters"], separators=(",", ":")),
            "--seed", str(evaluation["seed"]),
            "--settling-time", str(self.args.settling_time),
            "--evaluation-time", str(evaluation["duration"]),
            "--command", ",".join(str(value) for value in self.args.command),
            "--max-abs-action", str(self.args.max_abs_action),
            "--sr-evaluation-stage", evaluation["stage"],
            "--sr-evaluation-round", str(evaluation["round"]),
            "--sr-evaluation-seed", str(evaluation["seed"]),
            "--sr-evaluation-duration", str(evaluation["duration"]),
            "--sr-parent-elite", evaluation["parent_elite_trial_id"],
        ]
        if self.args.sr_early_reject:
            command.extend([
                "--sr-early-reject",
                "--sr-early-reject-grace-period", str(self.args.sr_early_reject_grace_period),
                "--sr-early-reject-torque-window", str(self.args.sr_early_reject_torque_window),
                "--sr-early-reject-torque-fraction", str(self.args.sr_early_reject_torque_fraction),
                "--sr-early-reject-penalty", str(self.args.sr_early_reject_penalty),
            ])
        command.extend(self.repository_args)
        return command

    def execute(self, evaluations: list[dict[str, Any]]) -> None:
        pending = [entry for entry in evaluations if entry["evaluation_id"] not in self.completed_ids]
        if not pending:
            return
        started = time.monotonic()
        runtime = 0.0
        for index, evaluation in enumerate(pending, 1):
            trial_started = time.monotonic()
            print(
                f"[{evaluation['stage']} r{evaluation['round']} {index}/{len(pending)}] "
                f"{evaluation['evaluation_id']}", flush=True,
            )
            try:
                process = subprocess.run(
                    self.worker_command(evaluation), capture_output=True, text=True,
                    timeout=self.args.trial_timeout, check=False,
                )
                result = base.parse_worker_output(process.stdout)
                if result is None:
                    detail = (process.stderr or process.stdout)[-2000:].strip()
                    result = base.failed_result(
                        evaluation["parameters"],
                        f"worker exited {process.returncode} without result: {detail}",
                    )
            except subprocess.TimeoutExpired:
                result = base.failed_result(
                    evaluation["parameters"],
                    f"worker timeout after {self.args.trial_timeout:.1f}s",
                )
            result.update({
                "evaluation_id": evaluation["evaluation_id"],
                "stage": evaluation["stage"], "round": evaluation["round"],
                "evaluation_seed": evaluation["seed"],
                "evaluation_duration": evaluation["duration"],
                "parent_elite_trial_id": evaluation["parent_elite_trial_id"],
                "early_rejection_reason": result.get("early_rejection_reason", ""),
            })
            self.append_result(result)
            runtime += time.monotonic() - trial_started
            elapsed = time.monotonic() - started
            eta = runtime / index * (len(pending) - index)
            print(
                f"  {result['status']} score={float(result['score']):.3f} "
                f"elapsed={base.format_duration(elapsed)} eta={base.format_duration(eta)}"
            )

    @staticmethod
    def as_float(row: dict[str, Any], field: str) -> float:
        try:
            value = float(row.get(field, 0.0))
            return value if math.isfinite(value) else 0.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def as_bool(row: dict[str, Any], field: str) -> bool:
        return str(row.get(field, "")).lower() in ("true", "1")

    def aggregate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["trial_id"]), []).append(row)
        aggregates = []
        for identifier, evaluations in grouped.items():
            scores = [self.as_float(row, "score") for row in evaluations]
            score_mean = sum(scores) / len(scores)
            score_std = math.sqrt(sum((score - score_mean) ** 2 for score in scores) / len(scores))
            failures = [row.get("status") != "success" for row in evaluations]
            feasible = [
                row.get("status") == "success"
                and all(self.as_bool(row, field) for field in FEASIBILITY_FIELDS)
                for row in evaluations
            ]
            failure_rate = sum(failures) / len(evaluations)
            feasible_fraction = sum(feasible) / len(evaluations)
            first = evaluations[0]
            aggregate = {
                "trial_id": identifier,
                **{name: self.as_float(first, name) for name in base.PARAMETER_FIELDS},
                "aggregate_stage": first["stage"],
                "aggregate_round": int(float(first["round"])),
                "evaluation_count": len(evaluations),
                "score_mean": score_mean,
                "score_std": score_std,
                "score_min": min(scores),
                "failure_rate": failure_rate,
                "feasible_fraction": feasible_fraction,
                "is_robust_feasible": feasible_fraction >= self.args.sr_required_feasible_fraction,
                "robust_score": (
                    score_mean - self.args.sr_risk_weight * score_std
                    - self.args.sr_failure_weight * failure_rate
                ),
            }
            aggregate.update({
                f"mean_{name}": sum(self.as_float(row, name) for row in evaluations) / len(evaluations)
                for name in base.METRIC_FIELDS
            })
            aggregates.append(aggregate)
        return aggregates

    @staticmethod
    def rank(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            aggregates,
            key=lambda row: (bool(row["is_robust_feasible"]), float(row["robust_score"])),
            reverse=True,
        )

    def rows_for(self, stage: str, round_index: int | None = None) -> list[dict[str, Any]]:
        return [
            row for row in self.rows
            if row.get("stage") == stage
            and (round_index is None or int(float(row.get("round", 0))) == round_index)
        ]

    def write_aggregates(self) -> list[dict[str, Any]]:
        # Each candidate is represented by its most informative/latest stage.
        precedence = {"broad": 0, "robust_validation": 1, "refinement": 2, "final_validation": 3}
        selected: dict[str, list[dict[str, Any]]] = {}
        for row in self.rows:
            identifier = str(row["trial_id"])
            current = selected.get(identifier)
            row_key = (precedence.get(str(row["stage"]), -1), int(float(row.get("round", 0))))
            current_key = (
                precedence.get(str(current[0]["stage"]), -1), int(float(current[0].get("round", 0)))
            ) if current else (-1, -1)
            if row_key > current_key:
                selected[identifier] = [row]
            elif row_key == current_key:
                selected[identifier].append(row)
        aggregates = self.aggregate([row for values in selected.values() for row in values])
        with self.aggregate_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=AGGREGATE_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in self.rank(aggregates):
                writer.writerow({field: row.get(field, "") for field in AGGREGATE_FIELDS})
        return aggregates

    def candidate_from_aggregate(self, aggregate: dict[str, Any]) -> dict[str, float]:
        return {name: float(aggregate[name]) for name in base.PARAMETER_FIELDS}

    def summary_bounds(self, ranked: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
        feasible = [row for row in ranked if row["is_robust_feasible"]]
        source = feasible or ranked
        source = source[:max(1, self.elite_count(len(source)))]
        suggested = {}
        for name, (original_low, original_high) in self.bounds.items():
            if original_low == original_high:
                suggested[name] = (original_low, original_high)
                continue
            values = [float(row[name]) for row in source]
            margin = (original_high - original_low) * self.args.sr_summary_margin_fraction
            suggested[name] = (
                max(original_low, quantile(values, self.args.sr_summary_lower_quantile) - margin),
                min(original_high, quantile(values, self.args.sr_summary_upper_quantile) + margin),
            )
        return suggested

    def write_final_outputs(
        self, final_ranked: list[dict[str, Any]], final_pool_rows: list[dict[str, Any]],
    ) -> None:
        all_aggregates = self.write_aggregates()
        if not final_ranked:
            self.best_path.write_text(json.dumps({"best_parameters": {}, "robust_score": 0.0}, indent=2) + "\n")
            return
        best = final_ranked[0]
        best_rows = [row for row in final_pool_rows if row["trial_id"] == best["trial_id"]]
        best_payload = {
            "best_parameters": self.candidate_from_aggregate(best),
            "robust_score": best["robust_score"],
            "aggregate": best,
            "aggregate_metrics": {
                name: best[f"mean_{name}"] for name in base.METRIC_FIELDS
            },
            "feasible": bool(best["is_robust_feasible"]),
            "per_seed_results": best_rows,
            "original_bounds": self.bounds,
            "final_local_bounds": self.state["final_local_bounds"],
            "command": self.args.command,
        }
        self.best_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True) + "\n")
        summary = {
            "suggested_parameter_bounds": self.summary_bounds(final_ranked),
            "original_bounds": self.bounds,
            "source_candidate_count": len(final_ranked),
            "robust_feasible_count": sum(bool(row["is_robust_feasible"]) for row in final_ranked),
            "lower_quantile": self.args.sr_summary_lower_quantile,
            "upper_quantile": self.args.sr_summary_upper_quantile,
            "margin_fraction": self.args.sr_summary_margin_fraction,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def dry_run(self) -> int:
        broad = self.broad_candidates()
        elite_count = self.elite_count(len(broad))
        validation_count = elite_count * len(self.args.sr_validation_seeds)
        refinement_count = sum(self.counts[1:])
        final_count = min(self.args.sr_final_validation_count, self.counts[-1]) * len(
            self.args.sr_validation_seeds
        )
        print("Successive-refinement dry run (Genesis will not start)")
        print("Original bounds:")
        for name in base.PARAMETER_FIELDS:
            fixed = " fixed" if name in self.fixed else ""
            print(f"  {name}: {self.bounds[name]}{fixed}")
        print(f"Broad candidates/evaluations: {len(broad)} / {len(broad)}")
        print(f"Robust validation: {elite_count} candidates x seeds {self.args.sr_validation_seeds} = {validation_count}")
        for round_index, count in enumerate(self.counts[1:], 1):
            global_count = min(count, int(round(count * self.args.sr_global_fraction)))
            print(
                f"Refinement round {round_index}: {count} evaluations "
                f"({global_count} global, {count - global_count} local), "
                f"width_fraction={max(self.args.sr_min_width_fraction, self.args.sr_shrink_factor ** round_index):.6f}"
            )
        print(f"Optional final validation evaluations: {final_count}")
        print(f"Expected simulator evaluations: {len(broad) + validation_count + refinement_count + final_count}")
        print("Example broad candidates:")
        for candidate in broad[:5]:
            print("  " + json.dumps(candidate["parameters"], sort_keys=True))
        return 0

    def worker_main(self) -> int:
        parameters = json.loads(self.args.trial_json)
        self.args.seed = self.args.sr_evaluation_seed
        self.args.evaluation_time = self.args.sr_evaluation_duration
        try:
            result = base.run_worker_trial(self.args, parameters, self.repository_args)
        except Exception as exc:
            result = base.failed_result(parameters, f"{type(exc).__name__}: {exc}")
        result.update({
            "evaluation_id": evaluation_id(
                parameters, self.args.sr_evaluation_stage,
                self.args.sr_evaluation_round, self.args.sr_evaluation_seed,
                self.args.sr_evaluation_duration,
            ),
            "stage": self.args.sr_evaluation_stage,
            "round": self.args.sr_evaluation_round,
            "evaluation_seed": self.args.sr_evaluation_seed,
            "evaluation_duration": self.args.sr_evaluation_duration,
            "parent_elite_trial_id": self.args.sr_parent_elite,
            "early_rejection_reason": result.get("early_rejection_reason", ""),
        })
        print(base.RESULT_SENTINEL + json.dumps(result, sort_keys=True, allow_nan=False))
        return 0

    def run(self) -> int:
        if self.args.worker:
            return self.worker_main()
        if self.args.dry_run:
            return self.dry_run()

        self.load_or_initialize()
        self.persist_state("broad_planned")
        broad_evaluations = self.planned_evaluations("broad", 0)
        if not broad_evaluations:
            broad_evaluations = self.make_evaluations(
                self.broad_candidates(), "broad", 0, [self.args.sr_broad_seed],
                self.args.sr_broad_evaluation_time,
            )
            self.persist_state("broad_planned", broad_evaluations)
        self.execute(broad_evaluations)

        broad_ranked = self.rank(self.aggregate(self.rows_for("broad", 0)))
        validation_elites = broad_ranked[:self.elite_count(len(broad_ranked))]
        validation_candidates = [
            {"parameters": self.candidate_from_aggregate(row), "parent_elite_trial_id": row["trial_id"]}
            for row in validation_elites
        ]
        validation_evaluations = self.planned_evaluations("robust_validation", 0)
        if not validation_evaluations:
            validation_evaluations = self.make_evaluations(
                validation_candidates, "robust_validation", 0,
                self.args.sr_validation_seeds, self.args.sr_validation_evaluation_time,
            )
            self.persist_state("validation_planned", validation_evaluations)
        self.execute(validation_evaluations)
        robust_ranked = self.rank(self.aggregate(self.rows_for("robust_validation", 0)))
        current_ranked = robust_ranked
        current_rows = self.rows_for("robust_validation", 0)
        current_elites = [
            self.candidate_from_aggregate(row)
            for row in robust_ranked[:self.elite_count(len(robust_ranked))]
        ]

        all_seen = {
            base.parameter_key(entry["parameters"])
            for entry in self.state["plans"]
        }
        for round_index in range(1, self.args.sr_rounds + 1):
            evaluations = self.planned_evaluations("refinement", round_index)
            if not evaluations:
                candidates, _, _ = self.refinement_candidates(
                    current_elites, round_index, all_seen
                )
                evaluations = self.make_evaluations(
                    candidates, "refinement", round_index,
                    [self.args.sr_broad_seed], self.args.sr_refinement_evaluation_time,
                )
                self.persist_state(f"refinement_{round_index}_planned", evaluations)
            self.execute(evaluations)
            current_rows = self.rows_for("refinement", round_index)
            current_ranked = self.rank(self.aggregate(current_rows))
            current_elites = [
                self.candidate_from_aggregate(row)
                for row in current_ranked[:self.elite_count(len(current_ranked))]
            ]
            if current_elites:
                self.state["final_local_bounds"] = self.local_bounds(current_elites[0], round_index)
                self.persist_state(f"refinement_{round_index}_complete")

        if self.args.sr_final_validation_count > 0 and current_ranked:
            finalists = current_ranked[:min(self.args.sr_final_validation_count, len(current_ranked))]
            final_candidates = [
                {"parameters": self.candidate_from_aggregate(row), "parent_elite_trial_id": row["trial_id"]}
                for row in finalists
            ]
            final_round = self.args.sr_rounds + 1
            final_evaluations = self.planned_evaluations("final_validation", final_round)
            if not final_evaluations:
                final_evaluations = self.make_evaluations(
                    final_candidates, "final_validation", final_round,
                    self.args.sr_validation_seeds, self.args.sr_validation_evaluation_time,
                )
                self.persist_state("final_validation_planned", final_evaluations)
            self.execute(final_evaluations)
            current_rows = self.rows_for("final_validation", final_round)
            current_ranked = self.rank(self.aggregate(current_rows))

        self.persist_state("complete")
        self.write_final_outputs(current_ranked, current_rows)
        base.display_top([
            {
                "status": "success", "trial_id": row["trial_id"],
                "score": row["robust_score"],
                "mean_body_frame_vx": row["mean_mean_body_frame_vx"],
                "mean_abs_vx_command_error": row["mean_mean_abs_vx_command_error"],
                "mean_valid_swing_displacement": row["mean_mean_valid_swing_displacement"],
                "overall_contact_mismatch": row["mean_overall_contact_mismatch"],
                "max_commanded_torque_ratio": row["mean_max_commanded_torque_ratio"],
            }
            for row in current_ranked
        ])
        print(f"Detailed:  {self.detailed_path}")
        print(f"Aggregate: {self.aggregate_path}")
        print(f"State:     {self.state_path}")
        print(f"Best:      {self.best_path}")
        print(f"Summary:   {self.summary_path}")
        return 0


def main() -> int:
    parser = B1GaitSweepController.build_parser()
    args, repository_args = parser.parse_known_args()
    try:
        if args.strategy == "successive" or args.worker and args.sr_evaluation_stage:
            controller = SuccessiveRefinementSweep(args, repository_args)
        else:
            controller = B1GaitSweepController(args, repository_args)
        return controller.run()
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
