"""Simulator-independent Experiment 1 scenarios, numeric recording and metrics.

No simulator, global RNG, or training imports: these contracts can be tested on CPU.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def materialize(obj):
    """Copy inherited class attributes too; deepcopy alone retains nested classes."""
    if isinstance(obj, dict):
        return {k: materialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(materialize(v) for v in obj)
    if isinstance(obj, type) or hasattr(obj, "__dict__"):
        return SimpleNamespace(**{k: materialize(getattr(obj, k)) for k in dir(obj)
                                 if not k.startswith("_") and
                                 (isinstance(getattr(obj, k), type) or not callable(getattr(obj, k)))})
    return copy.deepcopy(obj)


def seed_for(seed, trial, stream, step=0):
    payload = f"b1z1-exp1-v1/{seed}/{trial}/{stream}/{step}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def batches(episodes, num_envs):
    if episodes < 1 or num_envs < 1:
        raise ValueError("episodes and num-envs must be positive")
    for first in range(0, episodes, num_envs):
        yield list(range(first, min(first + num_envs, episodes)))


class Scenario:
    """Small event descriptors, not horizon-sized tapes; all loads are world-frame."""
    def __init__(self, seed, trial, condition, load_range, motion_mode, config):
        self.config = copy.deepcopy(config)
        self.mode = ("stationary" if trial % 2 == 0 else "locomotion") if motion_mode == "mixed" else motion_mode
        rng = np.random.default_rng(seed_for(seed, trial, "events"))
        bounds = np.array([50.] * 3 + [35.] * 3 + [10.] * 3)
        limits = bounds if load_range == "in" else np.array([70.] * 3 + [55.] * 3 + [20.] * 3)
        enabled = np.array([condition != "nominal"] * 3 + [condition == "combined-disturbance"] * 6)
        events = []
        for i, start in enumerate(config["event_times"]):
            load = rng.uniform(-limits, limits) if i < 2 else np.zeros(9)
            if load_range == "out" and i < 2:
                # Guarantee a genuinely out-of-training load in each enabled block.
                for offset in (0, 3, 6):
                    axis = offset + int(rng.integers(3))
                    load[axis] = rng.choice([-1., 1.]) * rng.uniform(bounds[axis] + 1., limits[axis])
            load *= enabled
            previous = np.array(events[-1]["load"]) if events else np.zeros(9)
            events.append(dict(time=float(start), load=load.tolist(), changed=bool(np.any(load != previous)),
                               exceeds_training_bounds=(np.abs(load) > bounds).tolist()))
        command_rng = np.random.default_rng(seed_for(seed, trial, "command"))
        command = [0., 0., 0.] if self.mode == "stationary" else [
            float(command_rng.uniform(*r)) for r in config["velocity_ranges"]]
        self.metadata = dict(trial_id=trial, seed=seed, motion_mode=self.mode,
                             condition=condition, load_range=load_range, command=command,
                             events=events, initial_seed=seed_for(seed, trial, "initial"),
                             mechanics_seed=seed_for(seed, trial, "mechanics"), config=self.config)
        self.metadata["scenario_hash"] = digest(self.metadata)

    def at(self, time):
        events = self.metadata["events"]
        index = sum(time >= e["time"] for e in events) - 1
        load = np.zeros(9) if index < 0 else np.array(events[index]["load"])
        # Smooth spherical trajectory in the existing yaw-aligned workspace frame.
        omega = 2 * math.pi / self.config["ee_period"]
        sphere = np.array(self.config["ee_center_sphere"]) + np.array(self.config["ee_amplitude_sphere"]) * math.sin(omega * time)
        transition = index >= 0 and events[index]["changed"] and time - events[index]["time"] < self.config["transition_seconds"]
        phase = 1 if transition else (2 if np.any(load) else 0)
        return np.array(self.metadata["command"]), sphere, load, phase


class NumericWriter:
    """Bounded chunks of homogeneous rows, compressed without object/pickle arrays."""
    def __init__(self, directory, chunk_rows=256):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.chunk_rows, self.rows, self.chunks, self.schema = chunk_rows, [], [], None
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")

    def append(self, row):
        row = {k: np.asarray(v).copy() for k, v in row.items()}
        if any(v.dtype.hasobject for v in row.values()):
            raise TypeError("Only numeric fields are supported; metadata belongs in JSON")
        schema = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in row.items()}
        if self.schema is None:
            self.schema = schema
        if schema != self.schema:
            raise ValueError("Record schema changed during evaluation")
        self.rows.append(row)
        if len(self.rows) >= self.chunk_rows:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        name = f"transitions-{len(self.chunks):06d}.npz"
        np.savez_compressed(self.directory / name,
                            **{key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]})
        self.chunks.append(dict(file=name, rows=len(self.rows)))
        self.rows.clear()

    def close(self):
        self.flush()
        (self.directory / "records.json").write_text(json.dumps(
            dict(schema_version=1, fields=self.schema, chunks=self.chunks), indent=2))


class TrialMetrics:
    """Online sufficient statistics; failed tails contribute zero to S_track."""
    def __init__(self, horizon, dt, scenario, config):
        self.K, self.dt, self.scenario, self.config = horizon, dt, scenario, config
        self.count = self.passed = 0
        self.sums = np.zeros(6)  # EE norm, planar norm, yaw absolute, signed EE xyz.
        self.phases = {str(i): dict(count=0, ee_sum=0., velocity_sum=0.) for i in range(3)}
        self.events = [dict(start=e["time"], count=0, peak=0., iae=0.,
                            velocity_peak=0., velocity_iae=0., yaw_peak=0., yaw_iae=0.,
                            scheduled_within_horizon=e["changed"] and e["time"] < horizon * dt,
                            dwell=0, recovery=None)
                       for e in scenario.metadata["events"]]
        self.limit_counts = np.zeros(3, dtype=np.int64)
        self.limit_denominators = np.zeros(3, dtype=np.int64)

    def add(self, time, ee_error, velocity_error, yaw_error, phase, valid=True,
            violations=(0, 0, 0), denominators=(0, 0, 0)):
        if not valid:
            return
        ee, velocity, yaw = float(np.linalg.norm(ee_error)), float(np.linalg.norm(velocity_error)), abs(float(yaw_error))
        success = ee <= self.config["ee_tolerance_m"] and velocity <= self.config["velocity_tolerance_m_s"] and yaw <= self.config["yaw_tolerance_rad_s"]
        self.count += 1
        self.passed += int(success)
        self.sums += [ee, velocity, yaw, *ee_error]
        p = self.phases[str(phase)]
        p["count"] += 1
        p["ee_sum"] += ee
        p["velocity_sum"] += velocity
        self.limit_counts += violations
        self.limit_denominators += denominators
        for event in self.events:
            elapsed = time - event["start"]
            if 0 <= elapsed < self.config["recovery_window_seconds"]:
                event["count"] += 1
                event["peak"] = max(event["peak"], ee)
                event["iae"] += ee * self.dt
                event["velocity_peak"] = max(event["velocity_peak"], velocity)
                event["velocity_iae"] += velocity * self.dt
                event["yaw_peak"] = max(event["yaw_peak"], yaw)
                event["yaw_iae"] += yaw * self.dt
                event["dwell"] = event["dwell"] + 1 if success else 0
                dwell_steps = max(1, math.ceil(self.config["recovery_dwell_seconds"] / self.dt))
                if event["recovery"] is None and event["dwell"] >= dwell_steps:
                    event["recovery"] = max(0., elapsed - (dwell_steps - 1) * self.dt)

    def finish(self, failure_reason):
        names = ("ee_error_m", "planar_velocity_error_m_s", "yaw_error_rad_s", "ee_signed_x_m", "ee_signed_y_m", "ee_signed_z_m")
        result = dict(self.scenario.metadata)
        result.pop("config")
        result.update(dict(zip(names, (self.sums / self.count).tolist() if self.count else [None] * 6)))
        result.update(valid_steps=self.count, prescribed_steps=self.K, tracking_pass_steps=self.passed,
                      S_track=self.passed / self.K, physical_failure=bool(failure_reason), failure_reason=failure_reason,
                      failure_time_seconds=(self.count + 1) * self.dt if failure_reason else None,
                      phase_statistics=self.phases)
        for i, name in enumerate(("position_violation", "velocity_violation", "actuator_saturation")):
            n, d = int(self.limit_counts[i]), int(self.limit_denominators[i])
            result[name + "_count"], result[name + "_denominator"] = n, d
            result[name + "_frequency"] = n / d if d else None
        expected = math.ceil(self.config["recovery_window_seconds"] / self.dt)
        for event in self.events:
            event["complete_window"] = event["count"] >= expected
            event["observed_iae_m_s"] = event.pop("iae")
            # Never interpret a short/failure-censored window as low integrated error.
            event["iae_m_s"] = event["observed_iae_m_s"] if event["complete_window"] else None
            for key, unit in (("velocity_iae", "m"), ("yaw_iae", "rad")):
                observed = event.pop(key)
                event["observed_" + key + "_" + unit] = observed
                event[key + "_" + unit] = observed if event["complete_window"] else None
            if failure_reason:
                event["recovery"] = None
            event["recovered"] = event["recovery"] is not None
        result["event_metrics"] = self.events
        return result


def aggregate_seed(trials):
    """Equal-trial tracking; pooled valid joint-substep physical limits per seed."""
    groups = {}
    for trial in trials:
        key = "/".join(str(trial[k]) for k in ("seed", "condition", "load_range", "motion_mode"))
        groups.setdefault(key, []).append(trial)
    result = {}
    for key, rows in groups.items():
        events = [e for row in rows for e in row["event_metrics"] if e["scheduled_within_horizon"]]
        result[key] = dict(trials=len(rows), physical_failures=sum(r["physical_failure"] for r in rows),
                           physical_failure_rate=float(np.mean([r["physical_failure"] for r in rows])),
                           recovery_event_count=len(events), recovery_count=sum(e["recovered"] for e in events),
                           recovery_fraction=sum(e["recovered"] for e in events) / len(events) if events else None,
                           complete_response_windows=sum(e["complete_window"] for e in events))
        for name in ("S_track", "ee_error_m", "planar_velocity_error_m_s", "yaw_error_rad_s",
                     "ee_signed_x_m", "ee_signed_y_m", "ee_signed_z_m"):
            values = [row[name] for row in rows if row[name] is not None]
            result[key][name] = float(np.mean(values)) if values else None
            result[key][name + "_trial_count"] = len(values)
        # Pool exposure, not trial percentages: short failed trials must not
        # receive the same physical-limit weight as long valid trials.
        for name in ("position_violation", "velocity_violation", "actuator_saturation"):
            count = sum(int(row[name + "_count"]) for row in rows)
            denominator = sum(int(row[name + "_denominator"]) for row in rows)
            result[key][name + "_count"] = count
            result[key][name + "_denominator"] = denominator
            result[key][name + "_frequency"] = count / denominator if denominator else None
            result[key][name + "_frequency_trial_count"] = sum(row[name + "_denominator"] > 0 for row in rows)
        result[key]["phases"] = {}
        for phase in ("0", "1", "2"):
            present = [r["phase_statistics"][phase] for r in rows if r["phase_statistics"][phase]["count"]]
            result[key]["phases"][phase] = dict(
                contributing_trials=len(present), valid_steps=sum(p["count"] for p in present),
                ee_error_m=float(np.mean([p["ee_sum"] / p["count"] for p in present])) if present else None,
                planar_velocity_error_m_s=float(np.mean([p["velocity_sum"] / p["count"] for p in present])) if present else None)
    return result


def aggregate_across_seeds(seed_summaries):
    """Equal seed weighting, never pool transitions from differently sized runs."""
    groups = {}
    for summary in seed_summaries:
        for key, row in summary.items():
            seed, group = key.split("/", 1)
            if seed in groups.setdefault(group, {}):
                raise ValueError(f"Duplicate evaluation seed {seed} in {group}")
            groups[group][seed] = row
    return {group: {metric: dict(mean=float(np.mean(values)), std=float(np.std(values)), seeds=len(values))
                    for metric in ("S_track", "ee_error_m", "planar_velocity_error_m_s", "yaw_error_rad_s", "physical_failure_rate",
                                   "position_violation_frequency", "velocity_violation_frequency", "actuator_saturation_frequency")
                    # Physical limits are pooled WITHIN seeds, then seeds are equal-weighted.
                    for values in [[r[metric] for r in rows.values() if r.get(metric) is not None]] if values}
            for group, rows in groups.items()}


def regenerate_seed_summary(directory):
    """Rebuild only seed_summary.json from a completed run's saved trial counts."""
    import importlib.util
    import os
    import tempfile
    directory = Path(directory)
    if not (directory / "COMPLETE").is_file():
        raise ValueError(f"Refusing to summarize an incomplete run: {directory}")
    metadata = json.loads((directory / "metadata.json").read_text())
    trials = json.loads((directory / "trials.json").read_text())
    experiment = metadata.get("experiment", 1)  # Original Exp1 metadata has no experiment key.
    if experiment == 2:
        spec = importlib.util.spec_from_file_location("exp2_summary", Path(__file__).with_name("b1z1_experiment2.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        summary = module.aggregate_seed(trials)
    elif experiment == 1:
        summary = aggregate_seed(trials)
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")
    payload = json.dumps(summary, indent=2, allow_nan=False)
    target = directory / "seed_summary.json"
    # Atomic replacement keeps a failed regeneration from damaging an existing summary.
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.close()
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate Exp1/Exp2 seed summaries from saved trials, without simulation.")
    parser.add_argument("directories", nargs="+", type=Path, help="Completed evaluation run directories")
    for directory in parser.parse_args().directories:
        print(regenerate_seed_summary(directory))
