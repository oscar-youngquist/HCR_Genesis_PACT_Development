"""Observational, native-trajectory FiLM diagnostics. No simulator/model construction."""
from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import signal

import exp1_analysis as shared
import exp2_analysis as exp2
from exp3_representation import chunks, file_hash
from b1z1_paper_names import DISPLAY_NAMES

FILM = ("film_condition_t", "film_features_pre_t", "film_features_post_t", "film_gamma_t", "film_beta_t")
REQUIRED = (*FILM, "trial_id", "step", "time_seconds", "successor_time_seconds", "valid",
    "alive_for_metrics", "physical_failure", "artificial_reset", "obs_t", "ee_position_t_world",
    "ee_reference_t_world", "ee_reference_sphere_t", "base_quaternion_t", "base_velocity_world_t",
    "base_angular_velocity_world_t", "ee_position_world", "ee_reference_world", "base_quaternion_xyzw", "action_raw")
OPTIONAL = ("prediction_base_velocity", "prediction_ee_position", "contacts", "ee_force_applied_world_interval_mean",
    "executed_torque_interval_mean", "saturation_substep_count", "nominal_kp", "nominal_kd",
    "mechanics_kp_scale", "mechanics_kd_scale", "mechanics_motor_strength", "mechanics_feedback_tau_weight",
    "mechanics_feedforward_tau_weight", "joint_position_t", "joint_velocity_t", "actual_clipping_limits",
    "ee_velocity_world", "ee_velocity_t_world")


def inverse_rotate(v, q, yaw_only=False):
    q = np.asarray(q, dtype=float)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norms < 1e-12):
        raise ValueError("Invalid recorded xyzw quaternion")
    q = q/norms
    if yaw_only:
        x, y, z, w = q.T
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        q = np.column_stack((np.zeros((len(q), 2)), np.sin(yaw/2), np.cos(yaw/2)))
    xyz, w = -q[:, :3], q[:, 3:]
    return v+2*np.cross(xyz, np.cross(xyz, v)+w*v)


def clean(value):
    """Strict JSON: undefined statistics are null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return str(value) if isinstance(value, Path) else value


def film_decomposition(pre, post, gamma, beta, atol=2e-5, rtol=2e-4):
    if not (pre.shape == post.shape == gamma.shape == beta.shape) or pre.ndim != 2:
        raise ValueError("Incompatible FiLM feature shapes")
    if not np.allclose(post, (1+gamma)*pre+beta, atol=atol, rtol=rtol):
        raise ValueError("FiLM residual-scale reconstruction failed")
    # Parameter-change term uses CURRENT pre-features, previous residual scale.
    feature_change = (1+gamma[:-1])*np.diff(pre, axis=0)
    parameter_change = np.diff(gamma, axis=0)*pre[1:]+np.diff(beta, axis=0)
    if not np.allclose(np.diff(post, axis=0), feature_change+parameter_change, atol=2*atol, rtol=rtol):
        raise ValueError("FiLM temporal identity failed")
    return post-pre, feature_change, parameter_change


def measured_condition(d, meta):
    """All inputs are actor-time t. Cartesian errors use measured-minus-reference.

    FiLM uses the opposite sign: command-minus-state. Recover the workspace
    center implicitly from the recorded spherical/world reference pair; do not
    approximate the terrain-invariant Z1 waist origin from base height.
    """
    q = d["base_quaternion_t"]
    error_world = d["ee_position_t_world"]-d["ee_reference_t_world"]
    error_yaw = inverse_rotate(error_world, q, yaw_only=True)
    sphere = d["ee_reference_sphere_t"]
    r, pitch, yaw = sphere.T
    target_cart = np.stack((r*np.cos(pitch)*np.cos(yaw), r*np.cos(pitch)*np.sin(yaw), r*np.sin(pitch)), axis=1)
    local = target_cart+error_yaw
    radius = np.maximum(np.linalg.norm(local, axis=1), 1e-6)
    measured_sphere = np.stack((radius, np.arcsin(np.clip(local[:, 2]/radius, -1, 1)),
                                np.arctan2(local[:, 1], local[:, 0])), axis=1)
    scales = meta["resolved_environment"]["normalization"]["obs_scales"]
    base_scale = np.array([scales["lin_vel"], scales["lin_vel"], scales["ang_vel"]])
    ee_scale = np.array([scales[k] for k in ("ee_sphe_radius_cmd", "ee_sphe_pitch_cmd", "ee_sphe_yaw_cmd")])
    body_v = inverse_rotate(d["base_velocity_world_t"], q)
    body_w = inverse_rotate(d["base_angular_velocity_world_t"], q)
    measured = np.column_stack((body_v[:, :2], body_w[:, 2]))*base_scale
    command = d["obs_t"][:, -6:]
    if not np.allclose(command[:, 3:], sphere*ee_scale, atol=2e-5, rtol=2e-4):
        raise ValueError("Policy EE command differs from recorded nominal spherical reference")
    return np.column_stack((command[:, :3]-measured, command[:, 3:]-measured_sphere*ee_scale)), error_yaw


def load_heads(meta, config):
    """Only deserialize weights on CPU; do not instantiate actor/runner/simulator."""
    name = meta["checkpoint_sha256"]
    path = Path(config.get("checkpoint_paths", {}).get(name, meta["checkpoint"]))
    if not path.is_file():
        return None, f"Checkpoint unavailable: {path}; exact action attribution unavailable"
    if file_hash(path) != name:
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    try:
        import torch
    except ImportError:
        return None, "PyTorch unavailable for checkpoint deserialization; feature analysis retained"
    import pickle
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        return None, f"Safe checkpoint deserialization unavailable: {exc}"
    state = state.get("model_state_dict", state)
    # V02 has a latent-width modulation layer; metadata and head width must agree.
    heads = {}
    for head in ("position", "torque"):
        key = head+"_head.weight"
        if key in state:
            heads[head] = (state[key].detach().numpy().copy(), state[head+"_head.bias"].detach().numpy().copy())
    return (heads, None) if "position" in heads else (None, "Linear position head unavailable")


def attribute(pre, post, parameter, heads, actions, meta, atol=3e-5, rtol=3e-4):
    n = int(meta["resolved_environment"]["env"]["num_actions"])
    control = meta["resolved_environment"]["control"]
    names = meta["dof_names"][:n]
    scales = {"position": np.broadcast_to(control["action_scale"], (n,)),
              "torque": np.broadcast_to(control["torque_scale"], (n,)).copy()}
    for name, scale in control.get("torque_scale_overrides", {}).items():
        scales["torque"][names.index(name)] = scale
    out, reconstructed, neutral = {}, [], []
    for head in ("position", "torque"):
        if head not in heads:
            continue
        w, b = heads[head]
        actual, original = post@w.T+b, pre@w.T+b
        reconstructed.append(actual); neutral.append(original)
        out[head+"_film"] = ((post-pre)@w.T)*scales[head]
        out[head+"_parameter"] = (parameter@w.T)*scales[head]
    reconstructed = np.concatenate(reconstructed, axis=1)
    if not np.allclose(reconstructed, actions, atol=atol, rtol=rtol):
        error = float(np.max(np.abs(reconstructed-actions)))
        raise ValueError(f"Linear heads do not reconstruct recorded deterministic raw actions: max_abs_error={error:.8g}, atol={atol}, rtol={rtol}")
    # This algebraic unmodulated-feature reference is NOT a simulated trajectory.
    return out, np.concatenate(neutral, axis=1), scales


def nominal_torque_delta(actions, unmodulated, d, meta, scales):
    required = ("nominal_kp", "nominal_kd", "mechanics_kp_scale", "mechanics_kd_scale",
        "mechanics_motor_strength", "mechanics_feedback_tau_weight", "mechanics_feedforward_tau_weight",
        "joint_position_t", "joint_velocity_t", "actual_clipping_limits")
    if not all(k in d for k in required):
        return None
    cfg = meta["resolved_environment"]
    n = cfg["env"]["num_actions"]
    names = meta["dof_names"]
    defaults = np.array([cfg["init_state"]["default_joint_angles"][k] for k in names])
    clip = cfg["normalization"]["clip_actions"]
    def torque(a):
        a = np.clip(a, -clip, clip)
        target = np.broadcast_to(defaults, d["joint_position_t"].shape).copy()
        target[:, :n] += a[:, :n]*scales["position"]
        pd = d["nominal_kp"]*d["mechanics_kp_scale"]*(target-d["joint_position_t"])
        pd -= d["nominal_kd"]*d["mechanics_kd_scale"]*d["joint_velocity_t"]
        ff = np.zeros_like(pd)
        if a.shape[1] == 2*n:
            ff[:, :n] = a[:, n:]*scales["torque"]
        return (pd*d["mechanics_feedback_tau_weight"]+ff*d["mechanics_feedforward_tau_weight"])*d["mechanics_motor_strength"]
    request, original = torque(actions), torque(unmodulated)
    limit = d["actual_clipping_limits"]
    return request-original, np.clip(request, -limit, limit)-np.clip(original, -limit, limit)


def spectrum(x, dt, cfg):
    """Welch on signed columns, constant detrend, Hann window; never vector norms."""
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    n = int(cfg["nperseg"])
    overlap = int(n*cfg["overlap_fraction"])
    segments = 0 if len(x) < n else 1+(len(x)-n)//(n-overlap)
    if segments < cfg["min_spectral_segments"]:
        return None
    f, psd = signal.welch(x, fs=1/dt, nperseg=n, noverlap=overlap, axis=0, detrend="constant", window="hann")
    band = cfg.get("frequency_band_hz") or [f[1], .5/dt]
    keep = (f >= band[0]) & (f <= band[1]) & (f > 0)
    if not keep.any():
        raise ValueError("Oscillation frequency band has no Welch bins")
    power = psd[keep].sum(axis=0)*(f[1]-f[0])
    dominant = f[keep][np.argmax(psd[keep], axis=0)]
    dominant = np.where(power > 1e-20, dominant, np.nan)
    return dict(frequency=f, psd=psd, power=power, dominant=dominant,
                dominant_mean=float(f[keep][np.argmax(psd[keep].mean(1))]) if power.mean() > 1e-20 else np.nan, segments=segments)


def relationship(x, y, dt, cfg):
    """Positive lag means y follows x. Phase reported only above coherence cutoff."""
    if spectrum(x, dt, cfg) is None:
        return None
    x, y = np.asarray(x)-np.mean(x), np.asarray(y)-np.mean(y)
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    n = cfg["nperseg"]; overlap = int(n*cfg["overlap_fraction"])
    kw = dict(fs=1/dt, nperseg=n, noverlap=overlap, window="hann", detrend="constant")
    f, c = signal.coherence(x, y, **kw)
    _, cross = signal.csd(x, y, **kw)
    lags = signal.correlation_lags(len(y), len(x))*dt
    corr = signal.correlate(y, x, mode="full")/(len(x)*np.std(x)*np.std(y))
    allowed = np.abs(lags) <= cfg["max_lag_seconds"]
    peak = np.flatnonzero(allowed)[np.argmax(np.abs(corr[allowed]))]
    phase = np.where((c >= cfg["coherence_threshold"]) & (f > 0), np.angle(cross), np.nan)
    return dict(frequency=f, coherence=c, phase=phase, lag_seconds=lags[peak], correlation=corr[peak])


def window_specs(meta, cfg):
    e = meta["evaluation"]
    end = meta["prescribed_steps"]*meta["dt"]
    # Use RECORDED load/motion times, not the originally proposed paper schedule.
    loads = e.get("loads_n")
    events = []
    previous = 0.
    for i, t in enumerate(e["load_times_s"]):
        value = loads[i] if loads is not None else None
        label = "load" if value is None else "load_removal" if value == 0 else "load_onset" if previous == 0 else "load_change"
        events.append((t, label)); previous = value
    events += [(e["move_start_s"], "motion"), (e["stop_s"], "stop")]
    cuts = sorted(set([0., end, e["move_start_s"], e["move_start_s"]+e["velocity_ramp_s"],
                       e["stop_s"], e["stop_s"]+e["velocity_ramp_s"], *e["load_times_s"]]))
    specs = []
    for left, right in zip(cuts, cuts[1:]):
        start, stop = left+cfg["transient_margin_seconds"], min(right-cfg["transient_margin_seconds"], end)
        while stop-start >= cfg["minimum_window_seconds"]:
            finish = min(start+cfg["sustained_window_seconds"], stop)
            if finish-start < cfg["minimum_window_seconds"]:
                break
            specs.append(dict(kind="sustained", start=start, end=finish, event=left, stage=cuts.index(left)))
            start = finish
    for t, label in events:
        lo, hi = cfg["event_window_seconds"]
        specs.append(dict(kind=label+"_response", start=max(0., t+lo), end=min(end, t+hi), event=t,
                          stage=int(np.searchsorted(cuts,t,side="right")-1)))
    for spec in specs:
        midpoint = (spec["start"]+spec["end"])/2
        spec["motion_phase"] = ("ramp" if any(t <= midpoint < t+e["velocity_ramp_s"] for t in (e["move_start_s"],e["stop_s"]))
                                else "moving" if e["move_start_s"] <= midpoint < e["stop_s"] else "stationary")
    return specs


def contiguous_segments(rows, dt):
    groups, current, failed = [], [], False
    for row in rows:
        failed = failed or bool(row["physical_failure"])
        good = row["valid"] and row["alive_for_metrics"] and not failed and not row["artificial_reset"]
        good = good and all(np.isfinite(row[k]).all() for k in REQUIRED if np.issubdtype(np.asarray(row[k]).dtype, np.number))
        if not good or (current and (int(row["step"]) != int(current[-1]["step"])+1
                or not np.isclose(row["time_seconds"]-current[-1]["time_seconds"], dt, atol=1e-7))):
            if current:
                groups.append(current); current = []
        if good:
            if not np.isclose(row["successor_time_seconds"]-row["time_seconds"], dt, atol=1e-7):
                raise ValueError("Actor/successor timestamps do not span one control interval")
            current.append(row)
    return groups+[current] if current else groups


def stream_windows(run, cfg):
    specs = window_specs(run["meta"], cfg)
    fields = [k for k in (*REQUIRED, *OPTIONAL) if k in run["records"]["fields"]]
    pending, done, last, failed = {}, set(), {}, set()
    wanted = cfg.get("trial_ids")
    for _, arrays in chunks(run, fields):
        for i, raw_tid in enumerate(arrays["trial_id"]):
            tid = int(raw_tid)
            if wanted is not None and tid not in wanted:
                continue
            step, t = int(arrays["step"][i]), float(arrays["time_seconds"][i])
            if tid not in run["trial_ids"]:
                raise ValueError("Unknown recorded trial ID")
            if step <= last.get(tid, -1):
                raise ValueError("Duplicate/out-of-order trial transitions")
            last[tid] = step
            if arrays["physical_failure"][i]:
                failed.add(tid)
            row = None
            for wi, spec in enumerate(specs):
                key = (tid, wi)
                if key in done:
                    continue
                if t >= spec["end"]-1e-8:
                    yield tid, spec, pending.pop(key, [])
                    done.add(key)
                elif t >= spec["start"]-1e-8:
                    if row is None:
                        row = {k: v[i].copy() for k, v in arrays.items()}
                        if tid in failed:
                            row["alive_for_metrics"] = np.bool_(False)
                    pending.setdefault(key, []).append(row)
    for (tid, wi), rows in pending.items():
        yield tid, specs[wi], rows


def discover(cfg):
    runs, exclusions, identities, protocols = [], [], set(), set()
    for path in shared.discover(cfg["recording_roots"], "metadata.json"):
        if not (path.parent/"COMPLETE").exists():
            exclusions.append(dict(run=str(path.parent), reason="Incomplete recording")); continue
        m = shared.read_json(path)
        if m.get("experiment") != 2 or (cfg.get("tasks") and m["task"] not in cfg["tasks"]):
            continue
        args = m["arguments"]
        if m["evaluation_seed"] not in cfg["evaluation_seeds"] or args["profile"] not in cfg["profiles"] or args["loading"] not in cfg["loading"]:
            continue
        if cfg.get("checkpoints") and m["checkpoint_sha256"] not in cfg["checkpoints"]:
            continue
        policy = m["resolved_training"]["policy"]
        reason = None
        if policy.get("film_v02_enabled") is not False:
            reason = "V02 or unknown FiLM architecture (explicit False required)"
        elif policy.get("conditioning_mode", "film") != "film" or not m.get("weights_unchanged"):
            reason = "Not frozen FiLM inference"
        records = shared.read_json(path.parent/"records.json")
        missing = set(REQUIRED)-set(records["fields"])
        if missing:
            reason = "Missing fields: "+", ".join(sorted(missing))
        if reason:
            exclusions.append(dict(run=str(path.parent), reason=reason)); continue
        identity = (m["task"], m["checkpoint_sha256"], m["evaluation_seed"], args["profile"], args["loading"])
        if identity in identities:
            raise ValueError(f"Duplicate recording cohort: {identity}")
        identities.add(identity)
        if m["dt"] <= 0 or not m.get("schema_version"):
            raise ValueError("Invalid recording schema/control timestep")
        protocols.add(shared.fingerprint(exp2.protocol(m)))
        if len(protocols) > 1:
            raise ValueError("Incompatible recorded protocols; analyze separately")
        trials = shared.read_json(path.parent/"trials.json")
        ids = [int(t["trial_id"]) for t in trials]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate trial IDs")
        runs.append(dict(path=path.parent, meta=m, records=records, trial_ids=set(ids)))
    for task in cfg.get("tasks") or sorted({r["meta"]["task"] for r in runs}):
        for profile in cfg["profiles"]:
            for loading in cfg["loading"]:
                seen = {r["meta"]["evaluation_seed"] for r in runs if r["meta"]["task"]==task
                        and r["meta"]["arguments"]["profile"]==profile and r["meta"]["arguments"]["loading"]==loading}
                missing = sorted(set(cfg["evaluation_seeds"])-seen)
                if missing:
                    exclusions.append(dict(task=task, profile=profile, loading=loading, reason=f"Missing compatible evaluation seeds: {missing}"))
    return runs, exclusions


def summary_stats(x, dt):
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    centered = x-x.mean(0)
    return dict(bias=float(np.sqrt(np.mean(x.mean(0)**2))), rms=float(np.sqrt(np.mean(x*x))),
                oscillatory_rms=float(np.sqrt(np.mean(centered**2))),
                peak_to_peak=float(np.mean(np.ptp(x, axis=0))), p50_abs=float(np.percentile(np.abs(x), 50)),
                p95_abs=float(np.percentile(np.abs(x), 95)), p99_abs=float(np.percentile(np.abs(x), 99)),
                variation_rms_per_s=float(np.sqrt(np.mean((np.diff(x, axis=0)/dt)**2))),
                reversal_fraction=float(np.mean(np.sum(x[1:]*x[:-1], axis=1) < 0)))


def analyze_window(d, meta, heads, cfg):
    pre, post, gamma, beta = (d[k].astype(float) for k in FILM[1:])
    increment, feature_change, parameter = film_decomposition(pre, post, gamma, beta)
    measured, cart = measured_condition(d, meta)
    condition = d["film_condition_t"]
    if condition.shape[1] != 6:
        raise ValueError("Only original six-component tracking-error FiLM is supported")
    if "prediction_base_velocity" in d and "prediction_ee_position" in d:
        expected = d["obs_t"][:, -6:]-np.concatenate((d["prediction_base_velocity"], d["prediction_ee_position"]), axis=1)
        if not np.allclose(condition, expected, atol=2e-5, rtol=2e-4):
            raise ValueError("Recorded estimator outputs do not reconstruct FiLM conditioning")
    # Parameter differences describe (t-1 -> t), aligned with rows [1:].
    traces = dict(ee_cartesian_error=cart[1:], measured_base_condition=measured[1:, :3],
        measured_ee_condition=measured[1:, 3:], estimated_base_condition=condition[1:, :3],
        estimated_ee_condition=condition[1:, 3:], base_estimator_mismatch=condition[1:, :3]-measured[1:, :3],
        ee_estimator_mismatch=condition[1:, 3:]-measured[1:, 3:], pre=pre[1:], post=post[1:],
        gamma=gamma[1:], beta=beta[1:], modulation=increment[1:], parameter_change=parameter,
        feature_change=feature_change, post_change=np.diff(post,axis=0),
        successor_ee_error=inverse_rotate(d["ee_position_world"][1:]-d["ee_reference_world"][1:], d["base_quaternion_xyzw"][1:], yaw_only=True))
    units = dict(ee_cartesian_error="m (yaw-aligned)", successor_ee_error="m (successor yaw-aligned)",
                 position_film="rad", position_parameter="rad", torque_film="Nm", torque_parameter="Nm")
    unavailable = []
    if heads:
        try:
            attrib, neutral, scales = attribute(pre, post, parameter, heads, d["action_raw"], meta,
                cfg.get("head_parity_atol",3e-5), cfg.get("head_parity_rtol",3e-4))
        except (KeyError, ValueError) as exc:
            unavailable.append("Exact head attribution unavailable: "+str(exc))
        else:
            for k, v in attrib.items():
                traces[k] = v if k.endswith("parameter") else v[1:]
            try:
                nominal = nominal_torque_delta(d["action_raw"], neutral, d, meta, scales)
            except (KeyError, ValueError) as exc:
                nominal = None; unavailable.append("Nominal torque attribution: "+str(exc))
            if nominal is not None:
                traces["nominal_requested_delta_nm"], traces["nominal_clipped_delta_nm"] = (v[1:] for v in nominal)
            else:
                unavailable.append("Nominal torque attribution unavailable (mechanics/scales incomplete)")
    traces["raw_action"] = d["action_raw"][1:]
    traces["base_motion"] = d["base_velocity_world_t"][1:]
    for key in ("contacts", "executed_torque_interval_mean", "ee_force_applied_world_interval_mean", "ee_velocity_world"):
        if key in d:
            traces[key] = d[key][1:]
    units.update({k: "normalized tracking coordinates" for k in traces if "condition" in k or "mismatch" in k})
    units.update(raw_action="normalized action", base_motion="m/s world", contacts="binary",
                 executed_torque_interval_mean="Nm", ee_force_applied_world_interval_mean="N world",
                 ee_velocity_world="m/s world", nominal_requested_delta_nm="Nm (undelayed algebraic)",
                 nominal_clipped_delta_nm="Nm (undelayed algebraic)")
    for key in traces:
        units.setdefault(key, "checkpoint-native feature units")
    for key in list(traces):
        if not np.isfinite(traces[key]).all():
            unavailable.append("Nonfinite optional signal excluded: "+key)
            del traces[key]
    metrics, spectra = {}, {}
    dt = meta["dt"]
    for name, values in traces.items():
        for stat, value in summary_stats(values, dt).items():
            metrics[name+"/"+stat] = value
        # Preserve signed physical component biases rather than only their RMS.
        if values.shape[1] <= 6:
            metrics.update({f"{name}/bias_axis_{i}": v for i, v in enumerate(values.mean(0))})
        sp = spectrum(values, dt, cfg)
        if sp is not None:
            spectra[name] = sp
            metrics[name+"/band_power"] = float(np.mean(sp["power"]))
            metrics[name+"/dominant_frequency_hz"] = sp["dominant_mean"]
    denominator = np.sqrt(np.mean(pre[1:]**2))
    metrics["relative_modulation_denominator_rms"] = float(denominator)
    metrics["relative_modulation_rms"] = np.sqrt(np.mean(increment[1:]**2))/denominator if denominator > cfg["feature_epsilon"] else np.nan
    norm = np.linalg.norm(pre[1:], axis=1)*np.linalg.norm(post[1:], axis=1)
    good = norm > cfg["feature_epsilon"]
    metrics["pre_post_zero_norm_fraction"] = float(1-good.mean())
    metrics["pre_post_cosine"] = float(np.mean(np.sum(pre[1:][good]*post[1:][good], axis=1)/norm[good])) if good.any() else np.nan
    if "saturation_substep_count" in d:
        metrics["saturation_fraction"] = float(d["saturation_substep_count"][1:].sum()/(len(parameter)*len(meta["dof_names"])*meta["resolved_environment"]["control"]["decimation"]))
    relations = []
    # Choose the most variable signed feature/joint, record its index explicitly.
    # This is within-window exploratory selection, not a confirmatory p-value.
    chain = ["ee_cartesian_error", "estimated_ee_condition", "parameter_change", "raw_action"]
    if "executed_torque_interval_mean" in traces:
        chain.append("executed_torque_interval_mean")
    chain.append("ee_velocity_world" if "ee_velocity_world" in traces else "successor_ee_error")
    stationary = cfg.get("stationary_window", True) and np.max(np.ptp(d["ee_reference_sphere_t"], axis=0)) <= cfg.get("reference_stationarity_tolerance", .02)
    if not stationary:
        unavailable.append("Nonstationary/event window: spectra descriptive only; coherence/lag suppressed")
        chain = []
    for left, right in zip(chain, chain[1:]):
        x, y = traces[left], traces[right]
        ix, iy = int(np.argmax(np.var(x, axis=0))), int(np.argmax(np.var(y, axis=0)))
        rel = relationship(x[:, ix], y[:, iy], dt, cfg)
        if rel is not None:
            # Successor-labelled signals are sampled one dt later than actor-time channels.
            if right in ("ee_velocity_world", "successor_ee_error"):
                rel["lag_seconds"] += dt
                rel["phase"] = np.angle(np.exp(1j*(rel["phase"]-2*np.pi*rel["frequency"]*dt)))
            relations.append(dict(left=left, right=right, left_axis=ix, right_axis=iy, **rel))
            band = cfg.get("frequency_band_hz") or [0., .5/dt]
            selected = (rel["frequency"] >= band[0]) & (rel["frequency"] <= band[1]) & (rel["frequency"] > 0)
            label = f"association/{left}_to_{right}/"
            metrics[label+"mean_band_coherence"] = float(rel["coherence"][selected].mean())
            metrics[label+"peak_lag_seconds"] = float(rel["lag_seconds"])
            metrics[label+"peak_correlation"] = float(rel["correlation"])
    return metrics, traces, spectra, relations, unavailable, units


def aggregate(frame):
    keys = ["task", "checkpoint", "training_seed", "profile", "loading", "kind", "stage", "metric"]
    # Windows -> trial -> evaluation seed. Never use timesteps as replicates.
    trials = frame.groupby(keys+["evaluation_seed", "trial_id"], dropna=False).value.mean().reset_index()
    seeds = trials.groupby(keys+["evaluation_seed"], dropna=False).value.agg(["mean", "count"]).reset_index()
    cohorts = seeds.groupby(keys, dropna=False)["mean"].agg(["mean", "std", "count"]).reset_index()
    cohorts["variability"] = "evaluation-seed SD within checkpoint (not trained-policy SD)"
    return trials, seeds, cohorts


def plot_window(d, traces, spectra, relations, identity, output, cfg):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})
    fig, axes = plt.subplots(5, 1, figsize=(7.0, 8.0), sharex=True)
    time = d["time_seconds"][1:]-identity["event"]
    for i, axis in enumerate("xyz"):
        axes[0].plot(time, traces["ee_cartesian_error"][:, i], label=axis)
    axes[0].set_ylabel("EE error [m]"); axes[0].legend(ncol=3)
    for i, name in enumerate(("radius", "elevation", "azimuth")):
        axes[1].plot(time, traces["measured_ee_condition"][:,i], color=f"C{i}", label=name+" measured")
        axes[1].plot(time, traces["estimated_ee_condition"][:,i], color=f"C{i}", linestyle="--", label=name+" estimated")
    axes[1].set_ylabel("EE condition\n[normalized]"); axes[1].legend(ncol=3, fontsize=6)
    for key in ("gamma", "beta", "modulation"):
        axes[2].plot(time, np.sqrt(np.mean(traces[key]**2, axis=1)), label=key)
    axes[2].set_ylabel("Feature RMS\n(magnitude)"); axes[2].legend(ncol=3)
    for key in ("position_film", "torque_film"):
        if key in traces:
            axes[3].plot(time, np.sqrt(np.mean(traces[key]**2, axis=1)), label=key+" ["+("rad" if key.startswith("position") else "Nm")+"]")
    axes[3].set_ylabel("Head effect RMS")
    if axes[3].lines:
        axes[3].legend()
    else:
        axes[3].plot(time, np.sqrt(np.mean(traces["raw_action"]**2,axis=1)), label="recorded raw action (not attribution)")
        axes[3].set_ylabel("Action RMS\n[normalized]"); axes[3].legend(fontsize=7)
        axes[3].text(.02,.04,"Exact head attribution unavailable",fontsize=7,transform=axes[3].transAxes)
    if "executed_torque_interval_mean" in traces:
        axes[4].plot(time, np.sqrt(np.mean(traces["executed_torque_interval_mean"]**2, axis=1)), label="executed interval-mean torque")
    else:
        axes[4].text(.5,.5,"Executed torque recording unavailable",ha="center",transform=axes[4].transAxes)
    axes[4].set_ylabel("Torque [Nm]"); axes[4].set_xlabel("Time relative to event [s]")
    for ax in axes:
        ax.axvline(0, color="grey", linestyle=":"); ax.grid(alpha=.2)
    fig.suptitle(f"{DISPLAY_NAMES.get(identity['task'], identity['task'])}: {identity['profile']} / trial {identity['trial_id']} / {identity['kind']}")
    fig.tight_layout()
    stem = f"window-{identity['window_id']:05d}"
    for ext in ("pdf", "png"):
        fig.savefig(output/(stem+"-traces."+ext), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for key in ("pre", "post", "modulation", "parameter_change"):
        if key in spectra:
            sp = spectra[key]; axes[0].semilogy(sp["frequency"], sp["psd"].mean(1), label=key)
    for key in ("ee_cartesian_error", "raw_action", "position_film", "torque_film"):
        if key in spectra:
            sp = spectra[key]; axes[1].semilogy(sp["frequency"], sp["psd"].mean(1), label=key)
    for rel in relations:
        axes[2].plot(rel["frequency"], rel["coherence"], label=rel["left"]+" -> "+rel["right"])
    axes[2].set_ylim(0, 1); axes[2].set_ylabel("Coherence")
    for ax in axes:
        ax.set_xlabel("Frequency [Hz]"); ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=5)
        else:
            ax.text(.5,.5,"Unavailable: insufficient averages\nor nonstationary window",ha="center",fontsize=7,transform=ax.transAxes)
    axes[0].set_ylabel("Mean feature PSD [feature²/Hz]")
    axes[1].set_ylabel("PSD [native signal units²/Hz]")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output/(stem+"-spectra."+ext), dpi=160)
    plt.close(fig)


def run_analysis(config):
    cfg = dict(config)
    output = Path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to avoid mixing stale windows/figures")
    output.mkdir(parents=True, exist_ok=True)
    (output/"COMPLETE").unlink(missing_ok=True)
    if not 0 <= cfg["overlap_fraction"] < 1 or cfg["min_spectral_segments"] < 3:
        raise ValueError("Require overlap in [0,1) and at least three Welch segments")
    if cfg["minimum_window_seconds"] <= 0 or cfg["sustained_window_seconds"] < cfg["minimum_window_seconds"]:
        raise ValueError("Invalid analysis window lengths")
    if cfg["nperseg"] < 4 or cfg["error_match_bin_m"] <= 0 or cfg["pca_samples"] < 3:
        raise ValueError("Invalid spectral/comparison/PCA settings")
    if cfg.get("pca_color", "signed_error") not in ("signed_error", "load_stage", "oscillation_phase"):
        raise ValueError("Unknown PCA coloring")
    runs, exclusions = discover(cfg)
    rows, windows, relations_out, provenance, heads_cache = [], [], [], [], {}
    reservoirs, seen = defaultdict(list), defaultdict(int)
    rng = np.random.default_rng(cfg.get("sampling_seed", 0))
    plotted = 0
    for run in runs:
        meta = run["meta"]; ckpt = meta["checkpoint_sha256"]
        print(f"FiLM analysis: {run['path']}", flush=True)
        if ckpt not in heads_cache:
            heads_cache[ckpt] = load_heads(meta, cfg)
        heads, missing = heads_cache[ckpt]
        if missing:
            exclusions.append(dict(run=str(run["path"]), reason=missing))
        for field in ("contacts", "executed_torque_interval_mean", "saturation_substep_count", "ee_force_applied_world_interval_mean"):
            if field not in run["records"]["fields"]:
                exclusions.append(dict(run=str(run["path"]), reason="Alternative-explanation diagnostic unavailable: "+field))
        provenance.append(dict(path=str(run["path"]), metadata_hash=file_hash(run["path"]/"metadata.json"),
            records_hash=file_hash(run["path"]/"records.json"), schema_version=meta["schema_version"],
            checkpoint=ckpt, task=meta["task"], evaluation_seed=meta["evaluation_seed"],
            training_seed=meta.get("checkpoint_training_seed"), dt=meta["dt"], protocol=exp2.protocol(meta),
            dof_names=meta["dof_names"], configuration=meta["resolved_environment"], policy=meta["resolved_training"]["policy"],
            recorded_trials=len(run["trial_ids"]), selected_trial_ids=sorted(run["trial_ids"].intersection(cfg["trial_ids"])) if cfg.get("trial_ids") is not None else sorted(run["trial_ids"]),
            conventions=meta.get("conventions")))
        for tid, spec, raw in stream_windows(run, cfg):
            segments = contiguous_segments(raw, meta["dt"])
            if not segments:
                exclusions.append(dict(run=str(run["path"]), trial_id=tid, window=spec, reason="No valid pre-failure samples"))
            for segment in segments:
                if len(segment)*meta["dt"] < cfg["minimum_window_seconds"] or len(segment) < 4:
                    exclusions.append(dict(run=str(run["path"]), trial_id=tid, window=spec, reason="Short/gapped/failed window")); continue
                d = {k: np.stack([r[k] for r in segment]) for k in segment[0]}
                identity = dict(task=meta["task"], checkpoint=ckpt, training_seed=meta.get("checkpoint_training_seed"),
                    evaluation_seed=meta["evaluation_seed"], trial_id=tid, profile=meta["arguments"]["profile"],
                    loading=meta["arguments"]["loading"], **spec, window_id=len(windows))
                try:
                    metrics, traces, spectra, relations, unavailable, units = analyze_window(d, meta, heads, {**cfg, "stationary_window": spec["kind"]=="sustained"})
                except (ValueError, KeyError) as exc:
                    exclusions.append(dict(**identity, reason=str(exc))); continue
                complete = len(segment) == len(raw) and len(raw)*meta["dt"] >= spec["end"]-spec["start"]-meta["dt"]*.5
                windows.append(dict(**identity, samples=len(segment), complete=complete,
                    actual_start=float(d["time_seconds"][0]), actual_end=float(d["successor_time_seconds"][-1]),
                    actual_load_mean_n=float(np.linalg.norm(d["ee_force_applied_world_interval_mean"],axis=1).mean()) if "ee_force_applied_world_interval_mean" in d else None,
                    spectral_available=bool(spectra), units=units))
                if not spectra:
                    unavailable.append("Insufficient Welch averages; coherence/spectra unavailable")
                exclusions.extend(dict(**identity, reason=reason) for reason in unavailable)
                rows.extend(dict(**identity, metric=k, value=v) for k, v in metrics.items())
                for rel in relations:
                    for i, f in enumerate(rel["frequency"]):
                        relations_out.append(dict(**identity, left=rel["left"], right=rel["right"], left_axis=rel["left_axis"], right_axis=rel["right_axis"],
                            frequency_hz=f, coherence=rel["coherence"][i], phase_rad=rel["phase"][i],
                            lag_seconds=rel["lag_seconds"], lag_correlation=rel["correlation"]))
                # Compact numeric window export makes plots reproducible, without retaining features.
                arrays = {"time_seconds": d["time_seconds"][1:], "successor_time_seconds": d["successor_time_seconds"][1:]}
                arrays.update({"trace_"+k: v for k,v in traces.items() if k not in ("pre", "post", "gamma", "beta", "modulation", "parameter_change", "feature_change", "post_change")})
                for k in ("gamma", "beta", "modulation", "parameter_change"):
                    arrays["rms_"+k] = np.sqrt(np.mean(traces[k]**2, axis=1))
                for k, sp in spectra.items():
                    arrays["frequency_"+k], arrays["psd_"+k] = sp["frequency"], sp["psd"].mean(1)
                np.savez_compressed(output/f"window-{identity['window_id']:05d}.npz", **arrays)
                if plotted < cfg["max_plot_windows"]:
                    plot_window(d, traces, spectra, relations, identity, output, cfg); plotted += 1
                if cfg.get("pca", True):
                    # Bounded paired reservoir, separate feature basis for every checkpoint.
                    signed_error = measured_condition(d, meta)[1][:, 0]
                    phase = np.angle(signal.hilbert(signed_error-signed_error.mean()))
                    for i in range(len(d["film_features_pre_t"])):
                        seen[ckpt] += 1
                        # Copy: row views would retain whole windows in the reservoir.
                        item = (d["film_features_pre_t"][i].copy(), d["film_features_post_t"][i].copy(),
                                signed_error[i], spec["stage"], phase[i])
                        pool = reservoirs[ckpt]; cap = cfg["pca_samples"]
                        j = int(rng.integers(seen[ckpt]))
                        if len(pool) < cap:
                            pool.append(item)
                        elif j < cap:
                            pool[j] = item
                if cfg.get("max_windows") and len(windows) >= cfg["max_windows"]:
                    break
            if cfg.get("max_windows") and len(windows) >= cfg["max_windows"]:
                break
        if cfg.get("max_windows") and len(windows) >= cfg["max_windows"]:
            break
    frame = pd.DataFrame(rows)
    summary = []
    if len(frame):
        trials, seeds, cohorts = aggregate(frame)
        for name, table in (("windows", frame), ("trials", trials), ("seeds", seeds), ("summary", cohorts)):
            table.to_csv(output/(name+".csv"), index=False)
        summary = cohorts.to_dict("records")
        compare_windows(frame, output, cfg)
    else:
        exclusions.append(dict(reason="No usable original actor-feature FiLM windows"))
    pd.DataFrame(relations_out).to_csv(output/"relationships.csv", index=False)
    pca = export_embeddings(reservoirs, output, cfg)
    result = dict(schema_version=1, complete=True, config=cfg, provenance=provenance, windows=windows,
        exclusions=exclusions, summary=summary, pca=pca,
        interpretation="Observational associations only; no causal confirmation or simulated counterfactuals.",
        preprocessing="No filtering/interpolation; constant-detrended Hann Welch on contiguous signed signals. Parameter increments per step, variation per second.")
    (output/"summary.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False))
    (output/"COMPLETE").write_text("complete\n")
    return result


def compare_windows(frame, output, cfg):
    keys = ["checkpoint", "evaluation_seed", "trial_id", "profile", "loading", "kind", "stage", "window_id"]
    wide = frame.pivot(index=keys, columns="metric", values="value").reset_index()
    # Descriptive high/low comparison WITHIN task/load/motion stage and checkpoint.
    groups = ["checkpoint", "profile", "loading", "kind", "stage"]
    rows = []
    for key, g in wide.groupby(groups, dropna=False):
        if len(g) < cfg["minimum_comparison_windows"]:
            continue
        score = "ee_cartesian_error/oscillatory_rms"
        threshold = g[score].median()
        g = g.copy(); g["oscillation_group"] = np.where(g[score] > threshold, "high", "low")
        # Error-magnitude bins are explicit, not nearest-neighbour causal matching.
        g["error_bin"] = np.floor(g["ee_cartesian_error/rms"]/cfg["error_match_bin_m"])
        for label, selected in [("unmatched", g), *[(f"error_bin_{b}", v) for b,v in g.groupby("error_bin")]]:
            if selected.oscillation_group.nunique() < 2 or len(selected) < cfg["minimum_comparison_windows"]:
                continue
            for group, v in selected.groupby("oscillation_group"):
                for metric in ("ee_cartesian_error/oscillatory_rms", "relative_modulation_rms", "parameter_change/oscillatory_rms", "pre/oscillatory_rms", "ee_estimator_mismatch/rms", "saturation_fraction"):
                    if metric in v:
                        per_trial = v.groupby(["evaluation_seed", "trial_id"])[metric].mean()
                        rows.append(dict(zip(groups,key), comparison=label, group=group, metric=metric,
                            mean=per_trial.groupby(level=0).mean().mean(), trials=len(per_trial), windows=len(v)))
    pd.DataFrame(rows).to_csv(output/"high_low_windows.csv", index=False)


def export_embeddings(reservoirs, output, cfg):
    import matplotlib.pyplot as plt
    result = {}
    for ckpt, pool in reservoirs.items():
        if len(pool) < 3:
            continue
        pre, post = np.stack([r[0] for r in pool]), np.stack([r[1] for r in pool])
        x = np.concatenate((pre, post)).astype(float); x -= x.mean(0)
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        projected = x@vt[:2].T
        explained = s[:2]**2/max(np.sum(s*s), 1e-30)
        color_name = cfg.get("pca_color", "signed_error")
        color_index, color_label = {"signed_error": (2, "Signed task-x EE error [m]"),
                                   "load_stage": (3, "Recorded schedule interval"),
                                   "oscillation_phase": (4, "Demeaned task-x Hilbert phase [rad]")}[color_name]
        result[ckpt] = dict(samples=len(pool), explained_variance_ratio=explained,
                            coloring=color_label, basis="pooled pre/post, centered, unscaled",
                            phase_note="Unfiltered Hilbert phase is qualitative, not a narrowband oscillator estimate")
        variants = {"pca": projected}
        if cfg.get("tsne", False):
            from sklearn.manifold import TSNE
            variants["tsne"] = TSNE(n_components=2, perplexity=min(cfg["tsne_perplexity"], (len(x)-1)/3),
                random_state=cfg.get("sampling_seed", 0), init="pca").fit_transform(x)
        for name, xy in variants.items():
            fig, ax = plt.subplots(figsize=(4, 3))
            for a, b in zip(xy[:len(pool)], xy[len(pool):]):
                ax.plot([a[0],b[0]], [a[1],b[1]], color="grey", alpha=.08, lw=.3)
            colors = np.tile([r[color_index] for r in pool], 2)
            points = ax.scatter(xy[:,0], xy[:,1], c=colors, s=3, cmap="coolwarm")
            fig.colorbar(points, ax=ax, label=color_label)
            ax.set_title(name.upper()+" (paired pre/post; qualitative)")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(output/f"{name}-{ckpt[:12]}.{ext}", dpi=160)
            plt.close(fig)
            np.savez_compressed(output/f"{name}-{ckpt[:12]}.npz", coordinates=xy, color=colors,
                                signed_error_m=np.tile([r[2] for r in pool], 2),
                                explained_variance_ratio=explained if name=="pca" else np.array([]))
    return result
