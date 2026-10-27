"""CPU-only Experiment 1 analysis: NumPy/Pandas, no training-package imports."""
import copy
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 1
CONDITIONS = ("nominal", "ee-disturbance", "combined-disturbance")
MAIN = ("ee_error_m", "planar_velocity_error_m_s", "yaw_error_rad_s", "S_track", "failure_rate")
SPLIT_TRACK = ("S_track_velocity", "S_track_manipulation")
LIMITS = {"actuator_saturation_pct": "saturation", "position_violation_pct": "position_violation",
          "velocity_violation_pct": "velocity_violation"}
UNITS = dict(zip(MAIN, ("m", "m/s", "rad/s", "fraction", "fraction")))
UNITS.update({k: "%" for k in LIMITS})
UNITS.update({k: "fraction" for k in SPLIT_TRACK})
UNITS.update(ee_peak_m="m", recovery_time_s="s", recovery_fraction="fraction", ee_iae_m_s="m*s",
             recovery_fraction_ee="fraction", recovery_fraction_velocity="fraction",
             torso_angular_acceleration_rms="rad/s^2", ee_linear_acceleration_rms="m/s^2")
REQUIRED = ("trial_id", "step", "successor_time_seconds", "dt", "alive_for_metrics", "valid",
            "physical_failure", "artificial_reset", "ee_position_world", "ee_reference_world",
            "base_velocity_body", "base_angular_velocity_body", "command_nominal",
            "base_quaternion_xyzw", "actual_clipping_limits", "joint_position",
            "saturation_substep_count", "position_violation_substep_count", "velocity_violation_substep_count")


def read_json(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False))


def fingerprint(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True).encode()).hexdigest()


def mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def seed_statistics(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return dict(mean=mean(values), sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                contributing_seeds=len(values))


def protocol(meta, ignore_evaluator_source_hash=False):
    evaluation = copy.deepcopy(meta["evaluation"])
    evaluation.pop("chunk_rows", None)  # Buffering does not change the physical protocol.
    result = dict(evaluation=evaluation, dt=meta["dt"], physics_dt=meta["physics_dt"],
                # PACT/UniFP backend family names differ, but use the same engine.
                horizon_steps=meta["prescribed_steps"], backend=meta["backend"].split("_b1z1_")[0],
                evaluator_sha256=meta.get("evaluator_sha256"), utilities_sha256=meta.get("utilities_sha256"),
                failure_reason_bits=meta["failure_reason_bits"],
                failure_definition="persistent" if "failure_patience_seconds" in evaluation else "immediate",
                conventions={k: meta["conventions"][k] for k in ("quaternion", "world", "joint")})
    if ignore_evaluator_source_hash:
        result.pop("evaluator_sha256")  # Ignore only this source fingerprint, not protocol settings.
    return result


def discover(roots, filename):
    paths = set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        paths.update([root] if root.is_file() and root.name == filename else root.rglob(filename))
    return sorted(paths)


def compatibility_error(runs):
    """Explain incompatible cohorts without weakening source/protocol checks."""
    groups = {}
    for directory, meta in runs:
        p = protocol(meta)
        key = (meta["checkpoint_sha256"], fingerprint(p))
        groups.setdefault(key, dict(protocol=p, paths=[]))["paths"].append(str(directory))

    def flatten(value, prefix=""):
        if isinstance(value, dict):
            return {key: item for name, child in value.items()
                    for key, item in flatten(child, f"{prefix}.{name}" if prefix else name).items()}
        return {prefix: value}

    fields = [flatten(group["protocol"]) for group in groups.values()]
    differing = [key for key in sorted(set().union(*(set(f) for f in fields)))
                 if len({json.dumps(f[key], sort_keys=True) if key in f else "<missing>"
                         for f in fields}) > 1]
    lines = ["Mixed checkpoint versions or incompatible evaluation protocols",
             f"Found {len({key[0] for key in groups})} checkpoint hash(es) and "
             f"{len({key[1] for key in groups})} protocol hash(es).",
             "Differing protocol fields: " + (", ".join(differing) or "none")]
    for (checkpoint, digest), group in groups.items():
        lines.append(f"Checkpoint {checkpoint}; protocol {digest} ({len(group['paths'])} runs):")
        flat = flatten(group["protocol"])
        lines.extend(f"  {key} = {flat.get(key, '<missing>')!r}" for key in differing)
        lines.extend(f"  {path}" for path in group["paths"])
    lines.append("Use input_roots selecting one compatible cohort, or rerun mismatched evaluations "
                 "with the same evaluator/utilities and settings. checkpoint_sha256 filters weights "
                 "only; it does not resolve evaluator-source differences. Do not edit recorded hashes.")
    return "\n".join(lines)


def load_records(directory, smoothness=False):
    """Load analysis columns only, excluding large observations/history arrays."""
    manifest = read_json(directory / "records.json")
    fields = list(REQUIRED)
    if smoothness:
        fields += ["base_angular_velocity_world", "ee_velocity_world"]
    missing = set(fields) - manifest["fields"].keys()
    if missing:
        raise ValueError(f"{directory}: missing recorded fields {sorted(missing)}")
    parts, seen = {k: [] for k in fields}, set()
    for chunk in manifest["chunks"]:
        path = (directory / chunk["file"]).resolve()
        if path.parent != directory.resolve() or path in seen:
            raise ValueError("Unsafe or duplicate chunk path")
        seen.add(path)
        with np.load(path, allow_pickle=False) as data:
            for field in fields:
                array, spec = data[field], manifest["fields"][field]
                if array.shape != (chunk["rows"], *spec["shape"]) or str(array.dtype) != spec["dtype"]:
                    raise ValueError(f"Chunk/schema mismatch: {path}, {field}")
                parts[field].append(array)
    if not seen:
        raise ValueError(f"No chunks: {directory}")
    return {k: np.concatenate(v) for k, v in parts.items()}


def yaw_frame(error, quaternion):
    """World error -> successor base-yaw frame; xyzw quaternion, z-up, metres."""
    q = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    x, y, z, w = q.T
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack((c*error[:, 0] + s*error[:, 1], -s*error[:, 0] + c*error[:, 1], error[:, 2]))


def acceleration_rms(velocity, time, step, valid, dt):
    # Backward differences only within contiguous valid endpoints; no filtering.
    good = (valid[1:] & valid[:-1] & (np.diff(step) == 1) &
            np.isclose(np.diff(time), dt, rtol=1e-5, atol=1e-8) &
            np.isfinite(velocity[1:]).all(-1) & np.isfinite(velocity[:-1]).all(-1))
    delta = np.diff(time)[good]
    acceleration = np.diff(velocity, axis=0)[good] / delta[:, None]
    return (float(np.sqrt(np.mean(np.sum(acceleration**2, axis=-1)))) if len(delta) else None, int(len(delta)))


def trial_metrics(data, trial, meta, smoothness=False):
    order = np.argsort(data["step"])
    d = {k: v[order] for k, v in data.items()}
    steps, time = d["step"], d["successor_time_seconds"]
    K, dt, cfg = int(meta["prescribed_steps"]), float(meta["dt"]), meta["evaluation"]
    if not len(steps) or len(np.unique(steps)) != len(steps) or steps.min() < 0 or steps.max() >= K:
        raise ValueError("Empty, duplicate, or out-of-horizon trial transitions")
    if not np.allclose(time, (steps + 1) * dt) or not np.allclose(d["dt"], dt):
        raise ValueError("Inconsistent successor timestamps/dt")
    failed = bool(np.any(d["physical_failure"]))
    if failed != bool(trial["physical_failure"]):
        raise ValueError("Trial failure disagrees with transitions")
    if failed and (d["physical_failure"].sum() != 1 or not d["physical_failure"][-1]):
        raise ValueError("Rows after physical failure")
    if not failed and steps[-1] != K - 1:
        raise ValueError("Nonfailed trial does not reach its horizon")
    error = d["ee_position_world"] - d["ee_reference_world"]
    ee = np.linalg.norm(error, axis=-1)
    vel = np.linalg.norm(d["base_velocity_body"][:, :2] - d["command_nominal"][:, :2], axis=-1)
    yaw = np.abs(d["base_angular_velocity_body"][:, 2] - d["command_nominal"][:, 2])
    valid = d["alive_for_metrics"].astype(bool) & d["valid"] & ~d["physical_failure"] & ~d["artificial_reset"]
    alive = valid.copy()
    # Independent task scores: only the relevant errors must be finite/in tolerance.
    velocity_pass = valid & np.isfinite(vel) & np.isfinite(yaw) & (vel <= cfg["velocity_tolerance_m_s"]) & (yaw <= cfg["yaw_tolerance_rad_s"])
    manipulation_pass = valid & np.isfinite(ee) & (ee <= cfg["ee_tolerance_m"])
    finite = np.isfinite(ee) & np.isfinite(vel) & np.isfinite(yaw)
    valid &= finite
    passes = valid & (ee <= cfg["ee_tolerance_m"]) & (vel <= cfg["velocity_tolerance_m_s"]) & (yaw <= cfg["yaw_tolerance_rad_s"])
    result = dict(trial_id=int(trial["trial_id"]), motion_mode=trial["motion_mode"], valid_steps=int(valid.sum()),
                  prescribed_steps=K, tracking_pass_steps=int(passes.sum()), physical_failure=failed,
                  missing_steps=int(steps[-1] + 1 - len(steps)), nonfinite_tracking_rows=int((d["alive_for_metrics"] & ~finite).sum()),
                  ee_error_m=mean(ee[valid]), planar_velocity_error_m_s=mean(vel[valid]),
                  yaw_error_rad_s=mean(yaw[valid]), S_track=float(passes.sum() / K), failure_rate=float(failed))
    # Full prescribed horizon, including zero credit for failed/missing future steps.
    result.update(S_track_velocity=float(velocity_pass.sum() / K),
                  S_track_manipulation=float(manipulation_pass.sum() / K),
                  velocity_tracking_pass_steps=int(velocity_pass.sum()),
                  manipulation_tracking_pass_steps=int(manipulation_pass.sum()))
    substeps = int(meta["resolved_environment"]["control"]["decimation"])
    result["limits"] = {}
    for metric, prefix in LIMITS.items():
        counts = d[prefix + "_substep_count"]
        if counts.ndim != 2 or np.any(counts < 0) or np.any(counts > substeps) or not np.isfinite(counts[valid]).all():
            raise ValueError("Invalid joint/substep counts")
        good = valid & np.isfinite(counts).all(-1)
        # Recorded counts use actual clipping limits, not averaged torques.
        n, den = int(counts[good].sum()), int(good.sum() * counts.shape[1] * substeps)
        result["limits"][metric] = dict(count=n, denominator=den)
    if np.any(d["actual_clipping_limits"][valid] <= 0) or not np.isfinite(d["actual_clipping_limits"][valid]).all():
        raise ValueError("Invalid actuator clipping limits")
    events = []
    for i, event in enumerate(trial["events"]):
        start = float(event["time"])
        elapsed = time - start
        window = valid & (elapsed >= 0) & (elapsed < cfg["recovery_window_seconds"])
        indices = np.flatnonzero(window)
        dwell_steps = max(1, math.ceil(cfg["recovery_dwell_seconds"] / dt))
        def recovery_for(pass_mask, valid_mask):
            # Same dwell/window and whole-trial failure censoring for all three criteria.
            if failed:
                return None
            dwell, previous = 0, None
            for j in np.flatnonzero(valid_mask & (elapsed >= 0) & (elapsed < cfg["recovery_window_seconds"])):
                dwell = dwell + 1 if pass_mask[j] and previous == steps[j] - 1 else int(pass_mask[j])
                previous = steps[j]
                if dwell >= dwell_steps:
                    return max(0., float(elapsed[j] - (dwell_steps - 1) * dt))
            return None
        recovery = recovery_for(passes, valid)
        recovery_ee = recovery_for(manipulation_pass, alive & np.isfinite(ee))
        recovery_velocity = recovery_for(velocity_pass, alive & np.isfinite(vel) & np.isfinite(yaw))
        complete = len(indices) >= math.ceil(cfg["recovery_window_seconds"] / dt)
        peak, iae = (float(ee[window].max()), float(ee[window].sum() * dt)) if len(indices) else (None, None)
        events.append(dict(event_type=("onset" if i == 0 else "removal" if not np.any(event["load"]) else "change"),
                           start=start, eligible=bool(event["changed"] and start < K*dt), complete_window=complete,
                           observed_samples=len(indices), observed_peak_m=peak, observed_iae_m_s=iae,
                           peak_m=peak if complete else None, iae_m_s=iae if complete else None,
                           recovery_time_s=recovery, recovered=recovery is not None,
                           recovery_time_ee_s=recovery_ee, recovered_ee=recovery_ee is not None,
                           recovery_time_velocity_s=recovery_velocity, recovered_velocity=recovery_velocity is not None))
    result["events"] = events
    if smoothness:
        for metric, field in (("torso_angular_acceleration_rms", "base_angular_velocity_world"),
                              ("ee_linear_acceleration_rms", "ee_velocity_world")):
            result[metric], result[metric + "_samples"] = acceleration_rms(d[field], time, steps, valid, dt)
    qgood = np.isfinite(d["base_quaternion_xyzw"]).all(-1) & (np.linalg.norm(d["base_quaternion_xyzw"], axis=-1) > 0)
    return result, error[valid & qgood], d["base_quaternion_xyzw"][valid & qgood], int((valid & ~qgood).sum())


def histogram_rows(errors, edges, seed, condition, load_range, motion):
    rows = []
    for axis, values in zip("xyz", errors.T):
        counts, _ = np.histogram(values, edges)
        n = len(values)
        under, over = int((values < edges[0]).sum()), int((values > edges[-1]).sum())
        for i, count in enumerate(counts):
            rows.append(dict(seed=seed, condition=condition, load_range=load_range, motion=motion, axis=axis,
                             bin_index=i, left=edges[i], right=edges[i+1], count=int(count), valid_samples=n,
                             underflow=under, overflow=over, probability=float(count/n) if n else None))
    return rows


def equal_seed_box_stats(seed_errors):
    """Exact inverse weighted ECDF quantiles; each nonempty seed has equal mass.

    Whiskers are 5th/95th percentiles, not Tukey fences. All raw values (including
    histogram tails) participate. These describe error spread, not uncertainty.
    """
    arrays = [values for values in seed_errors if len(values)]
    if not arrays:
        return []
    weights = np.concatenate([np.full(len(v), 1. / (len(arrays) * len(v))) for v in arrays])
    data = np.concatenate(arrays)
    result = []
    for i, axis in enumerate("xyz"):
        order = np.argsort(data[:, i], kind="stable")
        cdf = np.cumsum(weights[order])
        indices = np.minimum(np.searchsorted(cdf, np.array([.05, .25, .5, .75, .95]) - 1e-12), len(order)-1)
        quantiles = data[order[indices], i]
        result.append(dict(axis=axis, **dict(zip(("whislo", "q1", "med", "q3", "whishi"), quantiles)),
                           contributing_seeds=len(arrays), valid_samples=len(data),
                           minimum=float(data[:, i].min()), maximum=float(data[:, i].max())))
    return result


def binned_ecdf(histogram):
    """Equal-seed mean cumulative mass at bin edges, including underflow.

    Locations within bins/tails are unknown; this is a binned ECDF approximation.
    The left endpoint is P(error < first edge), the right includes the final bin.
    """
    curves = []
    edges = None
    for _, rows in histogram.groupby("seed"):
        rows = rows.sort_values("bin_index")
        n = int(rows.valid_samples.iloc[0])
        if not n:
            continue
        edges = np.r_[rows.left.iloc[0], rows.right.to_numpy()]
        curves.append((rows.underflow.iloc[0] + np.r_[0, np.cumsum(rows['count'])]) / n)
    if not curves:
        return None
    curves = np.asarray(curves)
    return edges, curves.mean(0), curves.std(0, ddof=1) if len(curves) > 1 else None, len(curves)


def summarize(config):
    """One task/checkpoint; strict validation prevents accidental protocol mixing."""
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "COMPLETE").unlink(missing_ok=True)
    edges = np.asarray(config["histogram_edges"], dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Histogram edges must be finite and increasing")
    frame = config.get("histogram_frame", "yaw")
    if frame not in ("yaw", "world"):
        raise ValueError("histogram_frame must be yaw or world")
    diagnostics, runs, identities, hashes, protocols = [], [], set(), set(), {}
    ignore_source = config.get("ignore_evaluator_source_hash", False)
    if not isinstance(ignore_source, bool):
        raise ValueError("ignore_evaluator_source_hash must be a boolean")
    for path in discover(config["input_roots"], "metadata.json"):
        meta = read_json(path)
        if meta.get("task") != config["task"]:
            continue
        if config.get("checkpoint_sha256") and meta.get("checkpoint_sha256") != config["checkpoint_sha256"]:
            continue
        directory = path.parent
        if not (directory / "COMPLETE").exists():
            diagnostics.append(f"Skipped incomplete evaluation: {directory}")
            continue
        for name in ("records.json", "trials.json", "seed_summary.json"):
            if not (directory / name).exists():
                raise ValueError(f"Missing {name}: {directory}")
        if not meta.get("weights_unchanged"):
            raise ValueError(f"Frozen policy not verified: {directory}")
        args = meta["arguments"]
        if args["condition"] not in CONDITIONS or args["load_range"] not in ("in", "out"):
            raise ValueError("Unknown condition or loading range")
        identity = (meta["evaluation_seed"], args["condition"], args["load_range"])
        if identity in identities:
            raise ValueError(f"Duplicate seed/condition/range: {identity}")
        identities.add(identity)
        hashes.add(meta["checkpoint_sha256"])
        p = protocol(meta, ignore_evaluator_source_hash=ignore_source)
        protocols[fingerprint(p)] = p
        runs.append((directory, meta))
    if not runs:
        raise ValueError("No completed evaluations matched task/checkpoint selection")
    if len(hashes) != 1 or len(protocols) != 1:
        raise ValueError(compatibility_error(runs))
    effective = set()
    for _, meta in runs:
        cfg = copy.deepcopy(meta["resolved_environment"])
        cfg.pop("seed", None)
        cfg.get("env", {}).pop("num_envs", None)
        cfg.get("viewer", {}).pop("rendered_envs_idx", None)  # Random display-only selection.
        effective.add(fingerprint(cfg))
    if len(effective) != 1:
        raise ValueError("Effective environment configuration differs within this approach")
    if ignore_source:
        message = ("Evaluator-source hash validation disabled by ignore_evaluator_source_hash=True; "
                   "evaluation-code equivalence has not been verified. Original hashes are retained in provenance.")
        diagnostics.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    per_seed, all_trials, histograms, provenance = [], [], [], []
    box_samples = {}
    smoothness = config.get("acceleration_metrics", False)
    for directory, meta in runs:
        seed, args = meta["evaluation_seed"], meta["arguments"]
        condition, load_range = args["condition"], args["load_range"]
        trials = read_json(directory / "trials.json")
        ids = [t["trial_id"] for t in trials]
        if len(set(ids)) != len(ids) or set(ids) != set(range(args["episodes"])):
            raise ValueError("Duplicate/missing attempted trial IDs")
        records = load_records(directory, smoothness)
        if set(np.unique(records["trial_id"])) != set(ids):
            raise ValueError("Recorded trial identities differ from trials.json")
        calculated, errors = [], {}
        for trial in trials:
            if trial["seed"] != seed or trial["condition"] != condition or trial["load_range"] != load_range:
                raise ValueError("Trial metadata disagrees with run metadata")
            d = {k: v[records["trial_id"] == trial["trial_id"]] for k, v in records.items()}
            result, e, q, missing_orientation = trial_metrics(d, trial, meta, smoothness)
            if result["valid_steps"] == 0 or result["missing_steps"] or result["nonfinite_tracking_rows"]:
                diagnostics.append(f"{directory}, trial {trial['trial_id']}: valid={result['valid_steps']}, gaps={result['missing_steps']}, nonfinite={result['nonfinite_tracking_rows']}")
            if missing_orientation:
                diagnostics.append(f"{directory}, trial {trial['trial_id']}: {missing_orientation} missing histogram orientations")
            errors[trial["trial_id"]] = yaw_frame(e, q) if frame == "yaw" else e
            for key in MAIN[:-1]:
                a, b = result[key], trial.get(key)
                if (a is None) != (b is None) or (a is not None and not np.isclose(a, b, rtol=1e-5, atol=1e-7)):
                    diagnostics.append(f"Online trial mismatch: {directory}, trial {trial['trial_id']}, {key}")
            result.update(seed=seed, condition=condition, load_range=load_range)
            calculated.append(result)
        all_trials.extend(calculated)
        online = read_json(directory / "seed_summary.json")
        motions = ["stationary", "locomotion"] + (["combined"] if config.get("include_combined", False) else [])
        for motion in motions:
            rows = [r for r in calculated if motion == "combined" or r["motion_mode"] == motion]
            if not rows:
                continue
            events = [e for r in rows for e in r["events"] if e["eligible"]]
            values = {key: mean([r[key] for r in rows]) for key in MAIN + SPLIT_TRACK}
            totals = {key: dict(count=sum(r["limits"][key]["count"] for r in rows),
                                denominator=sum(r["limits"][key]["denominator"] for r in rows)) for key in LIMITS}
            values.update({key: 100*t["count"]/t["denominator"] if t["denominator"] else None for key, t in totals.items()})
            values.update(ee_peak_m=mean([e["peak_m"] for e in events]), ee_iae_m_s=mean([e["iae_m_s"] for e in events]),
                          recovery_time_s=mean([e["recovery_time_s"] for e in events if e["recovered"]]),
                          recovery_fraction=sum(e["recovered"] for e in events)/len(events) if events else None,
                          recovery_fraction_ee=sum(e["recovered_ee"] for e in events)/len(events) if events else None,
                          recovery_fraction_velocity=sum(e["recovered_velocity"] for e in events)/len(events) if events else None)
            if smoothness:
                for key in ("torso_angular_acceleration_rms", "ee_linear_acceleration_rms"):
                    values[key] = mean([r[key] for r in rows])
            counts = {key: sum(r.get(key) is not None for r in rows) for key in MAIN + SPLIT_TRACK}
            counts.update({key: sum(r["limits"][key]["denominator"] > 0 for r in rows) for key in LIMITS})
            entry = dict(seed=seed, condition=condition, load_range=load_range, motion=motion, values=values,
                         attempted_trials=len(rows), metric_trial_counts=counts, valid_steps=sum(r["valid_steps"] for r in rows),
                         eligible_events=len(events), recovered_events=sum(e["recovered"] for e in events),
                         recovered_ee_events=sum(e["recovered_ee"] for e in events),
                         recovered_velocity_events=sum(e["recovered_velocity"] for e in events),
                         complete_windows=sum(e["complete_window"] for e in events), limit_totals=totals)
            entry["event_types"] = {kind: dict(eligible=sum(e["event_type"] == kind for e in events),
                recovered=sum(e["event_type"] == kind and e["recovered"] for e in events),
                recovered_ee=sum(e["event_type"] == kind and e["recovered_ee"] for e in events),
                recovered_velocity=sum(e["event_type"] == kind and e["recovered_velocity"] for e in events),
                peak_m=mean([e["peak_m"] for e in events if e["event_type"] == kind]),
                recovery_time_s=mean([e["recovery_time_s"] for e in events if e["event_type"] == kind and e["recovered"]]))
                for kind in ("onset", "change", "removal")}
            per_seed.append(entry)
            old = online.get(f"{seed}/{condition}/{load_range}/{motion}", {})
            for key in MAIN:
                old_key = "physical_failure_rate" if key == "failure_rate" else key
                if motion != "combined" and old.get(old_key) is not None and values[key] is not None and not np.isclose(values[key], old[old_key], rtol=1e-5, atol=1e-7):
                    diagnostics.append(f"Online seed mismatch: {directory}, {motion}, {key}")
            if condition in ("nominal", "combined-disturbance"):
                e = np.concatenate([errors[r["trial_id"]] for r in rows])
                histograms.extend(histogram_rows(e, edges, seed, condition, load_range, motion))
                box_samples.setdefault((condition, load_range, motion), []).append(e)
        provenance.append(dict(directory=str(directory), task=meta["task"], checkpoint=meta["checkpoint"],
                               evaluator_sha256=meta.get("evaluator_sha256"), utilities_sha256=meta.get("utilities_sha256"),
                               checkpoint_sha256=meta["checkpoint_sha256"], training_seed=meta.get("checkpoint_training_seed"),
                               evaluation_seed=seed, backend=meta["backend"], git_commit=meta.get("git_commit"), git_dirty=meta.get("git_dirty"),
                               scenario_hashes=[t.get("scenario_hash") for t in trials], resolved_environment=meta["resolved_environment"]))
    aggregate = []
    strata = sorted({(r["condition"], r["load_range"], r["motion"]) for r in per_seed})
    for condition, load_range, motion in strata:
        rows = [r for r in per_seed if (r["condition"], r["load_range"], r["motion"]) == (condition, load_range, motion)]
        seeds = {r["seed"] for r in rows}
        missing = set(config.get("expected_seeds", [0, 1, 2])) - seeds
        if missing or len(seeds) < 3:
            diagnostics.append(f"{condition}/{load_range}/{motion}: seeds={sorted(seeds)}, missing={sorted(missing)}; fewer than three={len(seeds)<3}")
        for key in rows[0]["values"]:
            aggregate.append(dict(condition=condition, load_range=load_range, motion=motion, metric=key, unit=UNITS[key],
                                  **seed_statistics([r["values"][key] for r in rows]),
                                  attempted_trials=sum(r["attempted_trials"] for r in rows),
                                  contributing_trials=sum(r["metric_trial_counts"].get(key, 0) for r in rows) if key in MAIN + SPLIT_TRACK or key in LIMITS else None))
    for condition in CONDITIONS:
        if not any(r["condition"] == condition for r in per_seed):
            diagnostics.append(f"Missing condition: {condition}")
    summary = dict(schema_version=VERSION, status="complete", task=config["task"], display_name=config["display_name"],
                   checkpoint_sha256=next(iter(hashes)), protocol=next(iter(protocols.values())), protocol_hash=next(iter(protocols)),
                   units=UNITS, provenance=provenance, analysis=dict(histogram_frame=frame, histogram_edges=edges.tolist(),
                   ignore_evaluator_source_hash=ignore_source,
                   acceleration_metrics=smoothness, acceleration_processing="backward difference; no filter; contiguous valid endpoints; per-trial RMS then equal-trial/seed means",
                   include_combined=config.get("include_combined", False), expected_seeds=config.get("expected_seeds", [0, 1, 2])),
                   per_seed=per_seed, aggregates=aggregate, trials=all_trials, diagnostics=diagnostics,
                   ee_boxplots=[dict(condition=c, load_range=l, motion=m, **row)
                                for (c, l, m), arrays in box_samples.items() for row in equal_seed_box_stats(arrays)],
                   boxplot_convention="Equal-seed weighted empirical quantiles; boxes 25-75%, median; whiskers 5-95%; all tails included")
    pd.DataFrame(aggregate).to_csv(out / "summary.csv", index=False)
    columns = ["seed", "condition", "load_range", "motion", "axis", "bin_index", "left", "right", "count", "valid_samples", "underflow", "overflow", "probability"]
    pd.DataFrame(histograms, columns=columns).to_csv(out / "ee_histograms.csv", index=False)
    write_json(out / "summary.json", summary)  # Published last; require COMPLETE too.
    (out / "COMPLETE").write_text("Experiment 1 analysis exports complete\n")
    return summary


LABELS = {"ee_error_m": "EE [m]", "planar_velocity_error_m_s": "Velocity [m/s]",
          "yaw_error_rad_s": "Yaw [rad/s]", "S_track": "Tracking [%]", "failure_rate": "Failure [%]",
          "actuator_saturation_pct": "Saturation [%]", "position_violation_pct": "Position [%]",
          "velocity_violation_pct": "Velocity [%]", "ee_peak_m": "Peak EE [m]",
          "recovery_time_s": "Recovery [s]", "recovery_fraction": "Recovered [%]",
          "recovery_fraction_ee": "EE recovered [%]", "recovery_fraction_velocity": "Velocity recovered [%]"}


def formatted(row):
    if row is None or row["mean"] is None:
        return "\u2014"
    scale = 100 if row["unit"] == "fraction" else 1
    sd = "\u2014" if row["sd"] is None else f"{scale*row['sd']:.3f}"
    return f"{scale*row['mean']:.3f} \u00b1 {sd}"


def motion_modes_table(summaries, config, names, diagnostics):
    """Read mode-specific aggregates; never derive a mode from combined results."""
    condition = config.get("motion_modes_condition", "combined-disturbance")
    if condition not in CONDITIONS:
        raise ValueError("Invalid motion_modes_condition")
    load_range = config["load_range"]
    focused = config.get("focused_methods", ["b1z1_pact", "b1z1_unifp_original", "b1z1_ppo_pos"])
    if len(set(focused)) != len(focused):
        raise ValueError("Duplicate focused methods")
    metrics = ("ee_error_m", "planar_velocity_error_m_s", "yaw_error_rad_s", "S_track",
               "S_track_velocity", "S_track_manipulation", "failure_rate")
    labels = ("Mean EE position error [m] \u2193", "Planar velocity error [m/s] \u2193",
              "Yaw-rate error [rad/s] \u2193", "S_track [%] \u2191", "Velocity S_track [%] \u2191",
              "Manipulation S_track [%] \u2191", "Failure rate [%] \u2193")
    rows, index, numeric = [], [], []
    for mode, group in (("stationary", "Stationary manipulation"),
                        ("locomotion", "Locomotion with manipulation")):
        coverage = None
        for task in focused:
            summary = summaries.get(task, {})
            seeds = sorted(r["seed"] for r in summary.get("per_seed", []) if
                           (r["condition"], r["load_range"], r["motion"]) == (condition, load_range, mode))
            if seeds:
                if coverage is not None and coverage != seeds:
                    raise ValueError(f"Unequal seed coverage for motion_modes/{mode}: {task}")
                coverage = seeds
            name = names.get(task, config.get("display_names", {}).get(task, task))
            index.append((group, name))
            cells = []
            for metric in metrics:
                matches = [r for r in summary.get("aggregates", []) if
                           (r["condition"], r["load_range"], r["motion"], r["metric"]) ==
                           (condition, load_range, mode, metric)]
                if len(matches) > 1:
                    raise ValueError("Duplicate aggregate metric")
                row = matches[0] if matches and seeds else None
                if row is None or row["mean"] is None:
                    diagnostics.append(f"Missing {task}/{condition}/{load_range}/{mode}/{metric}")
                    if metric in SPLIT_TRACK:
                        diagnostics.append(f"{task}: rerun summarization to calculate split S_track from transitions")
                cells.append(formatted(row))
                # CSV stores display units, converting fractional mean AND SD exactly once.
                scale = 100 if UNITS[metric] == "fraction" else 1
                record = dict(row or {})
                record.update(task=task, method=name, group=group, condition=condition,
                              load_range=load_range, motion=mode, metric=metric,
                              unit="%" if scale == 100 else UNITS[metric], evaluation_seeds=json.dumps(seeds),
                              mean=None if row is None or row["mean"] is None else row["mean"] * scale,
                              sd=None if row is None or row["sd"] is None else row["sd"] * scale)
                record.setdefault("contributing_seeds", 0)
                record.setdefault("contributing_trials", 0)
                numeric.append(record)
            rows.append(cells)
    table = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=["Motion", "Method"]), columns=labels)
    out = Path(config["output_dir"])
    pd.DataFrame(numeric).to_csv(out / "motion_modes.csv", index=False)
    # Group labels are explicit rows, avoiding an extra multirow LaTeX dependency.
    tex = table.droplevel("Motion").to_latex(escape=True)
    body = tex.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0].splitlines()
    grouped = []
    for i, line in enumerate(body):
        if i % len(focused) == 0:
            group = index[i][0]
            grouped.append(r"\multicolumn{" + str(len(metrics) + 1) + r"}{l}{\textit{" + group + r"}} \\")
        grouped.append(line)
    tex = tex.split("\\midrule\n", 1)[0] + "\\midrule\n" + "\n".join(grouped) + "\n\\bottomrule\n\\end{tabular}\n"
    tex = tex.replace("\u00b1", r"$\pm$").replace("\u2014", "---").replace("\u2193", r"$\downarrow$").replace("\u2191", r"$\uparrow$")
    caption = (f"Motion-mode comparison, {condition}, {load_range}-range loads. "
               r"Equal-weight evaluation-seed means $\pm$ sample SD; missing results are ---.")
    (out / "motion_modes.tex").write_text(
        "% Requires booktabs and graphicx\n\\begin{table}[t]\n\\centering\n"
        + "\\caption{" + caption + "}\n\\label{tab:exp1_motion_modes}\n"
        + "\\resizebox{\\columnwidth}{!}{%\n" + tex + "}\n\\end{table}\n")
    return table


def compare(config):
    """Read portable summaries only: never open evaluation metadata or NPZ files."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    summaries, histograms = {}, {}
    for path in discover(config["summary_roots"], "summary.json"):
        s = read_json(path)
        if s.get("status") != "complete" or not (path.parent / "COMPLETE").exists():
            continue
        if s["schema_version"] != VERSION:
            raise ValueError("Unsupported analysis schema")
        task = s["task"]
        if task in summaries:
            raise ValueError(f"Duplicate approach summary: {task}; narrow summary_roots")
        if s["protocol_hash"] != fingerprint(s["protocol"]):
            raise ValueError("Invalid summary protocol fingerprint")
        summaries[task] = s
        histograms[task] = pd.read_csv(path.parent / "ee_histograms.csv")
    if not summaries:
        raise ValueError("No completed analysis summaries found")
    signatures = {fingerprint((s["protocol"], s["units"], s["analysis"]["histogram_frame"],
                              s["analysis"]["histogram_edges"])) for s in summaries.values()}
    if len(signatures) != 1:
        raise ValueError("Incompatible protocols, units, histogram frames or bins")
    load_range, motion = config["load_range"], config["motion"]
    if load_range not in ("in", "out") or motion not in ("stationary", "locomotion", "combined"):
        raise ValueError("Select one loading range and one motion stratum")
    coverage = {}
    for task, s in summaries.items():
        keys = [(r["seed"], r["condition"], r["load_range"], r["motion"]) for r in s["per_seed"]]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate per-seed summary identities")
        for condition in CONDITIONS:
            seeds = frozenset(r["seed"] for r in s["per_seed"] if (r["condition"], r["load_range"], r["motion"]) == (condition, load_range, motion))
            if seeds:
                if condition in coverage and coverage[condition] != seeds:
                    raise ValueError(f"Unequal seed coverage for {condition}: {task}")
                coverage[condition] = seeds
        h = histograms[task]
        for condition in ("nominal", "combined-disturbance"):
            expected = {r["seed"] for r in s["per_seed"] if (r["condition"], r["load_range"], r["motion"]) == (condition, load_range, motion)}
            for axis in "xyz":
                actual = set(h[(h.condition == condition) & (h.load_range == load_range) & (h.motion == motion) & (h.axis == axis)].seed)
                if actual != expected:
                    raise ValueError("Histogram seed coverage disagrees with summary")
        if len(h):
            if h.duplicated(["seed", "condition", "load_range", "motion", "axis", "bin_index"]).any():
                raise ValueError("Duplicate histogram identity")
            edges = np.asarray(s["analysis"]["histogram_edges"])
            for _, g in h.groupby(["seed", "condition", "load_range", "motion", "axis"]):
                g = g.sort_values("bin_index")
                if len(g) != len(edges)-1 or not np.allclose(g.left, edges[:-1]) or not np.allclose(g.right, edges[1:]):
                    raise ValueError("Histogram CSV bins disagree with summary")
                if any(g[k].nunique() != 1 for k in ("valid_samples", "underflow", "overflow")):
                    raise ValueError("Inconsistent histogram denominators/tails")
                n = int(g.valid_samples.iloc[0])
                if g['count'].sum() + g.underflow.iloc[0] + g.overflow.iloc[0] != n:
                    raise ValueError("Histogram tails/counts do not sum to denominator")
                if n and not np.allclose(g.probability, g['count']/n):
                    raise ValueError("Histogram probabilities disagree with counts")
    names = {task: config.get("display_names", {}).get(task, s["display_name"]) for task, s in summaries.items()}
    methods = config.get("method_order") or list(summaries)
    if len(set(methods)) != len(methods):
        raise ValueError("Duplicate method order entries")
    focused = config.get("focused_methods", ["b1z1_pact", "b1z1_unifp_original", "b1z1_ppo_pos"])
    missing = [f"Missing method: {t}" for t in methods if t not in summaries]
    for task, s in summaries.items():
        missing.extend(f"{task}: {message}" for message in s["diagnostics"])

    def lookup(task, condition, metric):
        rows = [r for r in summaries.get(task, {}).get("aggregates", []) if
                (r["condition"], r["load_range"], r["motion"], r["metric"]) == (condition, load_range, motion, metric)]
        if len(rows) > 1:
            raise ValueError("Duplicate aggregate metric")
        if not rows:
            missing.append(f"Missing {task}/{condition}/{load_range}/{motion}/{metric}")
        return rows[0] if rows else None

    def export_table(name, table, wide=False):
        table.to_csv(out / f"{name}.csv")
        latex = table.to_latex(escape=True, multicolumn=True, multicolumn_format="c")
        # graphicx/booktabs required; main table spans both publication columns.
        size = r"\textwidth" if wide else r"\columnwidth"
        (out / f"{name}.tex").write_text("% Requires graphicx and booktabs\n" +
            r"\resizebox{" + size + "}{!}{%\n" + latex + "}\n")

    columns = pd.MultiIndex.from_tuples([(c, LABELS[k]) for c in CONDITIONS for k in MAIN])
    main = pd.DataFrame([[formatted(lookup(t, c, k)) for c in CONDITIONS for k in MAIN] for t in methods],
                        index=[names.get(t, config.get("display_names", {}).get(t, t)) for t in methods], columns=columns)
    export_table("main_table", main, True)
    tables = {"main": main}
    if config.get("motion_modes_table", True):
        tables["motion_modes"] = motion_modes_table(summaries, config, names, missing)
    groups = {"physical_limits": list(LIMITS)}
    if config.get("response_table", True):
        groups["disturbance_response"] = ["ee_peak_m", "recovery_time_s", "recovery_fraction",
                                          "recovery_fraction_ee", "recovery_fraction_velocity"]
    if config.get("optional_metrics"):
        groups["optional_metrics"] = config["optional_metrics"]
    for name, metrics in groups.items():
        conditions = CONDITIONS[1:] if name == "disturbance_response" else CONDITIONS
        index = pd.MultiIndex.from_tuples([(names.get(t, t), c) for t in focused for c in conditions], names=["Method", "Condition"])
        table = pd.DataFrame([[formatted(lookup(t, c, k)) for k in metrics] for t in focused for c in conditions],
                             index=index, columns=[LABELS.get(k, f"{k} [{UNITS[k]}]") for k in metrics])
        export_table(name, table)
        tables[name] = table
    # Histograms show equal-seed mean percentage per bin, not probability density.
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = "Times New Roman" if "Times New Roman" in fonts else "DejaVu Serif"
    plot_data = []
    with plt.rc_context({"font.family": font, "font.size": 8, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(3, 2, figsize=(3.5, 5.2), sharex=True, sharey="row")
        for col, condition in enumerate(("nominal", "combined-disturbance")):
            axes[0, col].set_title("Nominal" if col == 0 else "EE + torso")
            for row, axis in enumerate("xyz"):
                ax = axes[row, col]
                for task in focused:
                    if task not in histograms:
                        continue
                    h = histograms[task]
                    h = h[(h.condition == condition) & (h.load_range == load_range) & (h.motion == motion) & (h.axis == axis)]
                    h = h[h.valid_samples > 0]
                    if h.empty:
                        continue
                    values = h.groupby("bin_index").agg(left=("left", "first"), right=("right", "first"), probability=("probability", "mean"), sd=("probability", "std"))
                    tails = h.groupby("seed").first()
                    tail_fraction = float(((tails.underflow + tails.overflow)/tails.valid_samples).mean())
                    x = (values.left + values.right)/2
                    width = values.right - values.left
                    ax.bar(x, 100 * values.probability, width=width, alpha=.4,
                           label=names.get(task, task), edgecolor="black", linewidth=.25,
                           yerr=100 * values.sd if len(tails) > 1 else None,
                           error_kw={"elinewidth": .6, "alpha": .6})
                    for i, v in values.iterrows():
                        plot_data.append(dict(task=task, condition=condition, load_range=load_range, motion=motion, axis=axis,
                                              bin_index=i, left=v.left, right=v.right, probability_mean=v.probability,
                                              probability_sd=v.sd, contributing_seeds=len(tails), outside_fraction=tail_fraction))
                ax.set_xlabel(f"{axis} error [m]")
                if col == 0:
                    ax.set_ylabel("Samples per bin [%]")
                ax.set_ylim(bottom=0)
                ax.grid(alpha=.2)
        handles, labels = [], []
        for ax in axes.flat:
            hs, ls = ax.get_legend_handles_labels()
            for h, label in zip(hs, ls):
                if label not in labels:
                    handles.append(h)
                    labels.append(label)
        fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=7)
        fig.suptitle(f"{load_range}-range; {motion}; {next(iter(summaries.values()))['analysis']['histogram_frame']} frame", fontsize=8)
        fig.tight_layout(rect=(0, .13, 1, .95))
        fig.savefig(out / "ee_distributions.pdf", bbox_inches="tight")
        fig.savefig(out / "ee_distributions.png", dpi=300, bbox_inches="tight")
    pd.DataFrame(plot_data).to_csv(out / "ee_distribution_data.csv", index=False)
    extra_warnings = plot_alternatives(summaries, histograms, config, names, font)
    missing.extend(extra_warnings)
    write_json(out / "comparison.json", dict(load_range=load_range, motion=motion, font=font,
               diagnostics=sorted(set(missing)), seed_coverage={k: sorted(v) for k, v in coverage.items()},
               methods={t: dict(checkpoint_sha256=s["checkpoint_sha256"], protocol_hash=s["protocol_hash"]) for t, s in summaries.items()},
               note="Bars show equal-seed mean percent per bin; error bars are sample SD across seeds. Tails are retained in ee_distribution_data.csv and per-seed exports; probabilities are not renormalized within the plotting range."))
    return tables, fig, sorted(set(missing))


def plot_alternatives(summaries, histograms, config, names, font):
    """Additional summary-only figures; old summaries need regenerating for boxes."""
    import matplotlib.pyplot as plt
    requested = config.get("additional_plots", ["step", "ecdf", "box"])
    if set(requested) - {"step", "ecdf", "box"}:
        raise ValueError("additional_plots supports step, ecdf, box")
    out = Path(config["output_dir"])
    focused = config.get("focused_methods", ["b1z1_pact", "b1z1_unifp_original", "b1z1_ppo_pos"])
    load_range, motion = config["load_range"], config["motion"]
    warnings = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for kind in requested:
        rows = []
        with plt.rc_context({"font.family": font, "font.size": 8, "pdf.fonttype": 42}):
            fig, axes = plt.subplots(3, 2, figsize=(3.5, 5.2), sharex=True, sharey="row")
            for col, condition in enumerate(("nominal", "combined-disturbance")):
                axes[0, col].set_title("Nominal" if col == 0 else "EE + torso")
                for row, axis in enumerate("xyz"):
                    ax = axes[row, col]
                    for i, task in enumerate(focused):
                        if task not in summaries:
                            continue
                        color, label = colors[i % len(colors)], names.get(task, task)
                        if kind == "box":
                            matches = [r for r in summaries[task].get("ee_boxplots", []) if
                                       (r["condition"], r["load_range"], r["motion"], r["axis"]) == (condition, load_range, motion, axis)]
                            if len(matches) > 1:
                                raise ValueError("Duplicate boxplot statistics")
                            if not matches:
                                warnings.append(f"Missing exact box statistics: {task}/{condition}/{motion}/{axis}; rerun summarization notebook")
                                continue
                            stats = matches[0]
                            rows.append(dict(task=task, **stats))
                            ax.bxp([stats], positions=[i], vert=False, showfliers=False, patch_artist=True,
                                   manage_ticks=False, boxprops={"facecolor": color, "alpha": .45},
                                   medianprops={"color": "black"}, whiskerprops={"color": color}, capprops={"color": color})
                        else:
                            h = histograms[task]
                            h = h[(h.condition == condition) & (h.load_range == load_range) & (h.motion == motion) & (h.axis == axis)]
                            h = h[h.valid_samples > 0]
                            if h.empty:
                                continue
                            if kind == "step":
                                values = h.groupby("bin_index").agg(left=("left", "first"), right=("right", "first"), p=("probability", "mean"))
                                edges = np.r_[values.left.iloc[0], values.right.to_numpy()]
                                ax.stairs(values.p * 100, edges, label=label, color=color, linewidth=1.1)
                            else:
                                edges, probability, sd, n = binned_ecdf(h)
                                ax.step(edges, probability * 100, where="post", label=label, color=color, linewidth=1.1)
                                if sd is not None:
                                    ax.fill_between(edges, np.maximum(0, probability-sd)*100,
                                                    np.minimum(1, probability+sd)*100, step="post", color=color, alpha=.12)
                                rows.extend(dict(task=task, condition=condition, load_range=load_range, motion=motion,
                                                 axis=axis, edge_m=x, cumulative_probability=p,
                                                 probability_sd=None if sd is None else sd[j], contributing_seeds=n)
                                            for j, (x, p) in enumerate(zip(edges, probability)))
                    ax.set_xlabel(f"{axis} error [m]")
                    ax.grid(alpha=.2)
                    if kind == "box":
                        ax.set_yticks(range(len(focused)), [names.get(t, config.get("display_names", {}).get(t, t)) for t in focused])
                        ax.axvline(0, color="gray", linewidth=.5, linestyle=":")
                    else:
                        ax.set_ylim(bottom=0, top=100 if kind == "ecdf" else None)
                        if col == 0:
                            ax.set_ylabel("Cumulative [%]" if kind == "ecdf" else "Samples per bin [%]")
            handles, labels = [], []
            for ax in axes.flat:
                hs, ls = ax.get_legend_handles_labels()
                for h, label in zip(hs, ls):
                    if label not in labels:
                        handles.append(h)
                        labels.append(label)
            if handles:
                fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=7)
            title = {"step": "Step histograms", "ecdf": "Binned ECDF approximation", "box": "Boxes: 25-75%; whiskers: 5-95%"}[kind]
            fig.suptitle(f"{title}\n{load_range}-range; {motion}", fontsize=8)
            fig.tight_layout(rect=(0, .13 if kind != "box" else 0, 1, .93))
            for extension in ("pdf", "png"):
                fig.savefig(out / f"ee_{kind}.{extension}", dpi=300, bbox_inches="tight")
            plt.close(fig)
        if rows:
            pd.DataFrame(rows).to_csv(out / f"ee_{kind}_data.csv", index=False)
    return warnings
