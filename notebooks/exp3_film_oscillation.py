"""Observational, native-trajectory FiLM diagnostics. No simulator/model construction."""
from collections import defaultdict
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy import signal

import exp1_analysis as shared
import exp2_analysis as exp2
from exp3_representation import chunks, file_hash
from b1z1_paper_names import DISPLAY_NAMES
from analysis_runtime import NumericCache, numerical_config, data_signature, ordered_work, optional_thread_limits

FILM = ("film_condition_t", "film_features_pre_t", "film_features_post_t", "film_gamma_t", "film_beta_t")


def default_config(config):
    """Keep tuning defaults out of the notebook; existing overrides remain supported."""
    return dict(dict(tasks=['b1z1_pact'],checkpoints=None,trial_ids=None,
        evaluation_seeds=[0,1,2],profiles=['pushing','pulling'],loading=['loaded','unloaded'],
        sustained_window_seconds=4.,event_window_seconds=[-.5,2.],transient_margin_seconds=.25,
        minimum_window_seconds=1.25,nperseg=64,overlap_fraction=.5,min_spectral_segments=4,
        min_nperseg=24,minimum_cycles_per_window=3,segment_length_policy='adaptive',
        frequency_band_hz=[.5,12.],coherence_threshold=.6,max_lag_seconds=.5,feature_epsilon=1e-8,
        minimum_comparison_windows=6,error_match_bin_m=.02,pca=False,pca_samples=1000,
        tsne=False,tsne_perplexity=30,max_plot_windows=6,max_windows=None,
        advanced_diagnostics=False,unmodulated_comparison=False,physical_comparison=True,
        feature_components=3,feature_basis_samples=2048),**config)
LEGACY_REQUIRED = (*FILM, "trial_id", "step", "time_seconds", "successor_time_seconds", "valid",
    "alive_for_metrics", "physical_failure", "artificial_reset", "obs_t", "ee_position_t_world",
    "ee_reference_t_world", "ee_reference_sphere_t", "base_quaternion_t", "base_velocity_world_t",
    "base_angular_velocity_world_t", "ee_position_world", "ee_reference_world", "base_quaternion_xyzw", "action_raw")
REQUIRED = (*FILM,'trial_id','step','time_seconds','successor_time_seconds','valid','alive_for_metrics',
            'physical_failure','artificial_reset','ee_position_t_world','ee_reference_t_world','base_quaternion_t','action_raw')
OPTIONAL = ("prediction_base_velocity", "prediction_ee_position", "contacts", "ee_force_applied_world_interval_mean",
    "executed_torque_interval_mean", "saturation_substep_count", "nominal_kp", "nominal_kd",
    "mechanics_kp_scale", "mechanics_kd_scale", "mechanics_motor_strength", "mechanics_feedback_tau_weight",
    "mechanics_feedforward_tau_weight", "joint_position_t", "joint_velocity_t", "actual_clipping_limits",
    "ee_velocity_world", "ee_velocity_t_world", "base_position_t", "pd_weighted_motor_interval_mean",
    "ff_weighted_motor_interval_mean", "opposing_channel_cancellation_mean")
OPTIONAL += tuple(k for k in LEGACY_REQUIRED if k not in REQUIRED)


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


def optional_heads(meta, cfg):
    if not cfg.get("unmodulated_comparison", False) and not cfg.get("advanced_diagnostics", False):
        return None, None
    try:
        return load_heads(meta, cfg)
    except (ValueError, OSError, RuntimeError) as exc:
        return None, str(exc)


def action_scales(meta):
    n = int(meta["resolved_environment"]["env"]["num_actions"])
    control = meta["resolved_environment"]["control"]
    names = meta["dof_names"][:n]
    q = np.broadcast_to(control["action_scale"], (n,))
    tau = np.broadcast_to(control.get("torque_scale", 1.), (n,)).copy()
    for name, value in control.get("torque_scale_overrides", {}).items():
        if name in names: tau[names.index(name)] = value
    return q, tau


def fit_feature_bases(runs, cfg):
    """Bounded seeded pooled pre/post PCA, fixed once per recorded checkpoint."""
    rng = np.random.default_rng(cfg.get("sampling_seed", 0)); pools = {}
    cap = cfg.get("feature_basis_samples", 2048)
    for run in runs:
        key = run["meta"]["checkpoint_sha256"]
        for _, data in chunks(run, ["film_features_pre_t", "film_features_post_t", "valid", "alive_for_metrics", "physical_failure", "artificial_reset", "trial_id"]):
            valid = data["valid"] & data["alive_for_metrics"] & ~data["physical_failure"] & ~data["artificial_reset"]
            if cfg.get("trial_ids") is not None: valid &= np.isin(data["trial_id"], cfg["trial_ids"])
            x = np.concatenate([data[name][valid] for name in ("film_features_pre_t", "film_features_post_t")])
            x = x[np.isfinite(x).all(1)]
            weights = rng.random(len(x))
            if key in pools:
                old, scores = pools[key]; x = np.concatenate([old,x]); weights = np.r_[scores,weights]
            keep = np.argsort(weights)[:cap]; pools[key] = x[keep], weights[keep]
    bases = {}
    for key,(x,_) in pools.items():
        if len(x)<2: continue
        center=x.mean(0); _,s,v=np.linalg.svd(x-center, full_matrices=False)
        axes=v[:cfg.get("feature_components", 3)].copy()
        axes *= np.where(axes[np.arange(len(axes)),np.argmax(abs(axes),axis=1)]<0,-1,1)[:,None]
        bases[key]=dict(center=center.tolist(), axes=axes.tolist(), samples=len(x),
                        explained_variance=(s[:len(axes)]**2/max(float((s*s).sum()),1e-30)).tolist())
    return bases


def reduced_precision_operand(x):
    """CPU diagnostic: FP32 exponent, 10-bit fraction, nearest/ties-away rounding.

    This matches the recorded head GEMMs, but does not prove the unrecorded GPU
    precision setting. It is not FP16 storage or a looser allclose tolerance.
    """
    bits = np.asarray(x, dtype=np.float32).copy().view(np.uint32)
    return ((bits + np.uint32(4096)) & np.uint32(0xffffe000)).view(np.float32).astype(float)


def head_parity(post, weight, bias, saved, atol, rtol, allow_precision=True):
    x, w, b = (np.asarray(v, dtype=float) for v in (post, weight, bias))
    exact = x@w.T+b
    qx, qw = reduced_precision_operand(x), reduced_precision_operand(w)
    approximate = qx@qw.T+b
    error, residual = np.abs(exact-saved), np.abs(approximate-saved)
    # Bound operand-rounding plus FP32 accumulation, separately for every action.
    eps = np.finfo(np.float32).eps
    gamma = (x.shape[1]+2)*eps/(1-(x.shape[1]+2)*eps)
    bound = np.abs(x-qx)@np.abs(w).T + np.abs(qx)@np.abs(w-qw).T
    bound += gamma*(np.abs(qx)@np.abs(qw).T+np.abs(b))
    strict = np.allclose(exact, saved, atol=atol, rtol=rtol)
    explained = (allow_precision and np.allclose(approximate, saved, atol=atol, rtol=rtol)
                 and np.all(error <= bound+atol)
                 and np.sqrt(np.mean(residual**2)) < .1*np.sqrt(np.mean(error**2)))
    status = "strict" if strict else "approximate_precision_explained" if explained else "unavailable"
    return dict(status=status, feature_dtype=str(np.asarray(post).dtype),
                max_abs_error=error.max(0).tolist(), rms_error=np.sqrt(np.mean(error**2,0)).tolist(),
                precision_residual_max=residual.max(0).tolist(), error_bound_max=bound.max(0).tolist(),
                model="FP32 operands rounded to 10 fraction bits, nearest/ties-away; FP32 accumulation bound",
                assumption="Arithmetic model supported by recorded actions, not recorded backend precision metadata")


def attribute(pre, post, parameter, heads, actions, meta, atol=3e-5, rtol=3e-4,
              audit=None, allow_precision=False, fixed=None, allow_unverified=False):
    n = int(meta["resolved_environment"]["env"]["num_actions"])
    control = meta["resolved_environment"]["control"]
    names = meta["dof_names"][:n]
    scales = {"position": np.broadcast_to(control["action_scale"], (n,)),
              "torque": np.broadcast_to(control["torque_scale"], (n,)).copy()}
    for name, scale in control.get("torque_scale_overrides", {}).items():
        scales["torque"][names.index(name)] = scale
    out, reconstructed, neutral, failures = {}, [], [], []
    for head in ("position", "torque"):
        if head not in heads:
            continue
        w, b = heads[head]
        actual, original = post@w.T+b, pre@w.T+b
        reconstructed.append(actual); neutral.append(original)
        out[head+"_film"] = ((post-pre)@w.T)*scales[head]
        out[head+"_parameter"] = (parameter@w.T)*scales[head]
        start = 0 if head == "position" else n
        check = head_parity(post, w, b, actions[:, start:start+n], atol, rtol, allow_precision)
        check["joint_names"] = names
        check["physical_scale"] = scales[head].tolist()
        if audit is not None:
            audit[head] = check
        if check["status"] == "unavailable":
            if allow_unverified:
                check["status"] = "unverified_algebraic_only"
            else:
                failures.append(f"{head}, max_abs_error={max(check['max_abs_error']):.8g}")
        if fixed is not None:
            fixed_action = fixed@w.T+b
            out[head+"_actual"] = actual*scales[head]
            out[head+"_fixed"] = fixed_action*scales[head]
            out[head+"_unmodulated"] = original*scales[head]
            out[head+"_persistent_effect"] = (fixed_action-original)*scales[head]
            out[head+"_varying_effect"] = (actual-fixed_action)*scales[head]
    reconstructed = np.concatenate(reconstructed, axis=1)
    if failures:
        raise ValueError("Linear heads do not reconstruct recorded deterministic raw actions: "+"; ".join(failures))
    if reconstructed.shape != actions.shape:
        raise ValueError("Recorded action/head ordering or dimensions are incompatible")
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


def spectral_plan(length, dt, cfg):
    """Shorter segments buy averages, NOT low-frequency resolution or independent replicates."""
    n = min(int(cfg["nperseg"]), length)
    minimum = int(cfg.get("min_nperseg", 32))
    policy = cfg.get("segment_length_policy", "adaptive")
    if policy not in ("adaptive", "fixed"):
        raise ValueError("segment_length_policy must be adaptive or fixed")
    if policy == "fixed":
        n = int(cfg["nperseg"])
    def count(k):
        return 0 if length < k else 1+(length-k)//max(1, k-int(k*cfg["overlap_fraction"]))
    if policy == "adaptive":
        while n >= minimum and count(n) < cfg["min_spectral_segments"]:
            n -= 1
    valid = n >= minimum and count(n) >= cfg["min_spectral_segments"]
    resolution = 1/(max(n, 1)*dt)
    band = cfg.get("frequency_band_hz") or [resolution, .5/dt]
    low = max(band[0], resolution, cfg.get("minimum_cycles_per_window", 3)/(max(length, 1)*dt))
    high = min(band[1], .5/dt)
    valid = valid and low <= high and np.any((np.fft.rfftfreq(max(n, 1), dt) >= low) & (np.fft.rfftfreq(max(n, 1), dt) <= high))
    return dict(available=bool(valid), duration_seconds=length*dt, nperseg=n,
                segments=count(max(n, 1)), resolution_hz=resolution,
                usable_low_hz=low, usable_high_hz=high,
                reason="available" if valid else "Insufficient averages/segment length or usable frequency bins")


def spectrum(x, dt, cfg):
    """Welch on signed columns, constant detrend, Hann window; never vector norms."""
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    plan = spectral_plan(len(x), dt, cfg)
    n = plan["nperseg"]
    overlap = int(n*cfg["overlap_fraction"])
    segments = 0 if len(x) < n else 1+(len(x)-n)//(n-overlap)
    if not plan["available"]:
        return None
    f, psd = signal.welch(x, fs=1/dt, nperseg=n, noverlap=overlap, axis=0, detrend=cfg.get("spectral_detrend", "constant"), window="hann")
    keep = (f >= plan["usable_low_hz"]) & (f <= plan["usable_high_hz"]) & (f > 0)
    if not keep.any():
        raise ValueError("Oscillation frequency band has no Welch bins")
    power = psd[keep].sum(axis=0)*(f[1]-f[0])
    dominant = f[keep][np.argmax(psd[keep], axis=0)]
    dominant = np.where(power > 1e-20, dominant, np.nan)
    return dict(frequency=f, psd=psd, power=power, dominant=dominant,
                dominant_mean=float(f[keep][np.argmax(psd[keep].mean(1))]) if power.mean() > 1e-20 else np.nan, segments=segments, plan=plan)


def relationship(x, y, dt, cfg):
    """Positive lag means y follows x. Phase reported only above coherence cutoff."""
    if not spectral_plan(len(x), dt, cfg)["available"]:
        return None
    detrend = cfg.get("spectral_detrend", "constant")
    x, y = (signal.detrend(np.asarray(v), type=detrend) for v in (x, y))
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    plan = spectral_plan(len(x), dt, cfg)
    n = plan["nperseg"]; overlap = int(n*cfg["overlap_fraction"])
    kw = dict(fs=1/dt, nperseg=n, noverlap=overlap, window="hann", detrend=detrend)
    f, c = signal.coherence(x, y, **kw)
    _, cross = signal.csd(x, y, **kw)
    lags = signal.correlation_lags(len(y), len(x))*dt
    corr = signal.correlate(y, x, mode="full")/(len(x)*np.std(x)*np.std(y))
    allowed = np.abs(lags) <= cfg["max_lag_seconds"]
    peak = np.flatnonzero(allowed)[np.argmax(np.abs(corr[allowed]))]
    usable = (f >= plan["usable_low_hz"]) & (f <= plan["usable_high_hz"])
    phase = np.where((c >= cfg["coherence_threshold"]) & usable, np.angle(cross), np.nan)
    return dict(frequency=f, coherence=c, phase=phase, lag_seconds=lags[peak], correlation=corr[peak], plan=plan)


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
        index = int(np.searchsorted(e["load_times_s"], midpoint, side="right"))-1
        spec["load_stage"] = index
        spec["scheduled_load_n"] = (loads[index] if loads is not None and index >= 0 else 0.) if meta["arguments"]["loading"] == "loaded" else 0.
    return specs if cfg.get('advanced_diagnostics',False) or cfg.get('include_event_windows',False) else [s for s in specs if s['kind']=='sustained']


def contiguous_segments(rows, dt, required=REQUIRED):
    groups, current, failed = [], [], False
    for row in rows:
        failed = failed or bool(row["physical_failure"])
        good = row["valid"] and row["alive_for_metrics"] and not failed and not row["artificial_reset"]
        good = good and all(np.isfinite(row[k]).all() for k in required if np.issubdtype(np.asarray(row[k]).dtype, np.number))
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
    specs = cfg.get("_window_specs", window_specs(run["meta"], cfg))
    fields = [k for k in cfg.get("_fields", (*REQUIRED, *OPTIONAL)) if k in run["records"]["fields"]]
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


def joint_groups(meta, cfg):
    """Resolve recorded names, excluding the gripper; overrides are explicit name lists."""
    names = meta["dof_names"]
    overrides = cfg.get("joint_groups", {})
    groups = {}
    for group in ("arm", "leg"):
        selected = overrides.get(group)
        if selected is None:
            selected = [name for name in names if
                        (name.startswith("z1_") and "gripper" not in name.lower() if group == "arm"
                         else name.startswith(("FR_", "FL_", "RR_", "RL_")))]
        if any(name not in names for name in selected):
            raise ValueError(f"Unknown recorded {group} joint name")
        groups[group] = [names.index(name) for name in selected]
    return groups


def load_projections(d, cfg):
    """Stable mean world load direction; signed perpendicular basis, never a norm spectrum."""
    force = d.get("ee_force_applied_world_interval_mean")
    if force is None:
        return None
    force = force[1:]
    mean = force.mean(0); magnitude = np.linalg.norm(mean)
    norms = np.linalg.norm(force, axis=1)
    threshold = cfg.get("load_direction_min_n", 1.)
    if not np.isfinite(force).all() or magnitude < threshold or np.any(norms < threshold):
        return None
    direction = mean/magnitude
    if np.any((force@direction)/norms < np.cos(np.deg2rad(cfg.get("load_direction_tolerance_deg", 10.)))):
        return None
    # Common world frame for error and load; deterministic transverse axes.
    anchor = np.eye(3)[np.argmin(np.abs(direction))]
    perp1 = np.cross(direction, anchor); perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    basis = np.column_stack((direction, perp1, perp2))
    error = d["ee_position_t_world"][1:]-d["ee_reference_t_world"][1:]
    return error@basis, basis


def analyze_window(d, meta, heads, cfg):
    pre, post, gamma, beta = (d[k].astype(float) for k in FILM[1:])
    increment, feature_change, parameter = film_decomposition(pre, post, gamma, beta)
    early_notes=[]
    cart=inverse_rotate(d['ee_position_t_world']-d['ee_reference_t_world'],d['base_quaternion_t'],yaw_only=True)
    try:
        measured, cart = measured_condition(d, meta)
    except (KeyError,ValueError) as exc:
        measured=np.full((len(pre),6),np.nan)
        early_notes.append('Measured conditioning unavailable: '+str(exc))
    condition = d["film_condition_t"]
    if condition.shape[1] != 6:
        raise ValueError("Only original six-component tracking-error FiLM is supported")
    if cfg.get('advanced_diagnostics',False) and "prediction_base_velocity" in d and "prediction_ee_position" in d and 'obs_t' in d:
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
        successor_ee_error=(inverse_rotate(d["ee_position_world"][1:]-d["ee_reference_world"][1:], d["base_quaternion_xyzw"][1:], yaw_only=True)
                           if all(k in d for k in ('ee_position_world','ee_reference_world','base_quaternion_xyzw')) else np.full_like(cart[1:],np.nan)))
    units = dict(ee_cartesian_error="m (yaw-aligned)", successor_ee_error="m (successor yaw-aligned)",
                 position_film="rad", position_parameter="rad", torque_film="Nm", torque_parameter="Nm")
    unavailable = early_notes
    parity_audit = {}
    advanced = cfg.get("advanced_diagnostics", False)
    if not heads and cfg.get("unmodulated_comparison", False):
        unavailable.append("FiLM-attributed temporal links unavailable: compatible linear-head weights absent")
    if heads and advanced:
        try:
            fixed = (1+gamma.mean(0))*pre+beta.mean(0)
            attrib, neutral, scales = attribute(pre, post, parameter, heads, d["action_raw"], meta,
                cfg.get("head_parity_atol",3e-5), cfg.get("head_parity_rtol",3e-4), audit=parity_audit,
                allow_precision=cfg.get("precision_aware_attribution", True), fixed=fixed,
                allow_unverified=cfg.get("allow_unverified_attribution", False))
        except (KeyError, ValueError) as exc:
            unavailable.append("Exact head attribution unavailable: "+str(exc))
        else:
            if any(v["status"] == "unverified_algebraic_only" for v in parity_audit.values()):
                unavailable.append("Head parity unexplained: algebraic attribution exported as UNVERIFIED, not a validated action decomposition")
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
    q_scale,tau_scale = action_scales(meta); n = len(q_scale)
    if d["action_raw"].shape[1] not in (n,2*n):
        raise ValueError("Unexpected recorded position/torque action layout")
    traces["position_recorded"] = d["action_raw"][1:,:n]*q_scale
    if d["action_raw"].shape[1]==2*n:
        traces["torque_recorded"] = d["action_raw"][1:,n:]*tau_scale
    if heads and not advanced:
        for head,scale in (("position",q_scale),("torque",tau_scale)):
            if head not in heads: continue
            w,b=heads[head]
            try:
                actual,neutral=post@w.T+b,pre@w.T+b
                recorded=traces[head+"_recorded"]
                if actual[1:].shape!=recorded.shape: raise ValueError("Head/action shape mismatch")
                traces[head+"_actual"]=actual[1:]*scale
                traces[head+"_unmodulated"]=neutral[1:]*scale
                traces[head+"_film"]=(actual-neutral)[1:]*scale
                units[head+"_reconstruction_max_error"]=float(np.max(abs(actual[1:]*scale-recorded)))
            except (ValueError,KeyError) as exc:
                unavailable.append(head+" comparison unavailable: "+str(exc))
    basis=cfg.get("feature_bases",{}).get(meta["checkpoint_sha256"])
    if basis:
        center,axes=np.asarray(basis["center"]),np.asarray(basis["axes"])
        traces["pre_projection"]=(pre[1:]-center)@axes.T
        traces["post_projection"]=(post[1:]-center)@axes.T
        traces["modulation_projection"]=increment[1:]@axes.T
    if 'base_velocity_world_t' in d: traces["base_motion"] = d["base_velocity_world_t"][1:]
    for key in ("contacts", "executed_torque_interval_mean", "ee_force_applied_world_interval_mean", "ee_velocity_world",
                "pd_weighted_motor_interval_mean", "ff_weighted_motor_interval_mean", "joint_velocity_t", "joint_position_t",
                "opposing_channel_cancellation_mean"):
        if key in d:
            traces[key] = d[key][1:]
    from exp3_oscillation_physical import motion_signals
    motion, motion_reason = motion_signals(d)
    traces.update({k:v[1:] for k,v in motion.items()})
    if motion_reason:
        unavailable.append(motion_reason)
    groups = joint_groups(meta, cfg)
    components = {"ee_cartesian_error": list("xyz"), "successor_ee_error": list("xyz"),
                  "estimated_ee_condition": ["radius", "elevation", "azimuth"],
                  "measured_ee_condition": ["radius", "elevation", "azimuth"]}
    for name in [k for k in traces if k.startswith(("position_", "torque_"))]+["executed_torque_interval_mean"]:
        if name not in traces:
            continue
        components[name] = meta["dof_names"][:traces[name].shape[1]]
        for group, indices in groups.items():
            indices = [i for i in indices if i < traces[name].shape[1]]
            if indices:
                traces[name+"_"+group] = traces[name][:, indices]
                components[name+"_"+group] = [meta["dof_names"][i] for i in indices]
            else:
                unavailable.append(f"{name}: {group} joint grouping unavailable")
    projection = load_projections(d, cfg) if cfg.get("stationary_window", True) else None
    if projection is not None:
        traces["ee_load_parallel_error"] = projection[0][:, :1]
        traces["ee_load_perpendicular_error"] = projection[0][:, 1:]
        units["load_projection_basis_world"] = projection[1].tolist()
    else:
        unavailable.append("Load projections unavailable: zero/missing/varying load direction or event window")
    units.update({k: "normalized tracking coordinates" for k in traces if "condition" in k or "mismatch" in k})
    units.update(raw_action="normalized action", base_motion="m/s world", contacts="binary",
                 executed_torque_interval_mean="Nm", ee_force_applied_world_interval_mean="N world",
                 ee_velocity_world="m/s world", nominal_requested_delta_nm="Nm (undelayed algebraic)",
                 nominal_clipped_delta_nm="Nm (undelayed algebraic)")
    for key in traces:
        units.setdefault(key, "rad" if key.startswith("position_") else "Nm" if key.startswith(("torque_", "executed_torque")) else
                         "m (world load-aligned basis)" if key.startswith("ee_load_") else "checkpoint-native feature units")
    units["signal_components"] = components
    units["head_parity"] = parity_audit
    units["recorded_feature_dtype"] = str(d["film_features_post_t"].dtype)
    units.update({k:"m/s (actor-time yaw frame; arm relative is measured endpoint residual)" for k in motion})
    for key in list(traces):
        if not np.isfinite(traces[key]).all():
            unavailable.append("Nonfinite optional signal excluded: "+key)
            del traces[key]
    metrics, spectra = {}, {}
    dt = meta["dt"]
    sustained = cfg.get("stationary_window", True)
    stationary = sustained and ('ee_reference_sphere_t' not in d or np.max(np.ptp(d["ee_reference_sphere_t"], axis=0)) <= cfg.get("reference_stationarity_tolerance", .02))
    moving = sustained and not stationary and cfg.get("moving_reference_mode", "linear_detrended") == "linear_detrended"
    analysis_mode = "stationary" if stationary else "moving_reference_detrended" if moving else "event_descriptive"
    spectral_cfg = {**cfg, "spectral_detrend": "linear" if moving and advanced else "constant"}
    units["temporal_analysis_mode"] = analysis_mode
    units["spectral_plan"] = spectral_plan(len(parameter), dt, spectral_cfg)
    for name, values in traces.items():
        for stat, value in summary_stats(values, dt).items():
            metrics[name+"/"+stat] = value
        # Preserve signed physical component biases rather than only their RMS.
        if values.shape[1] <= 6:
            metrics.update({f"{name}/bias_axis_{i}": v for i, v in enumerate(values.mean(0))})
        sp = spectrum(values, dt, spectral_cfg)
        if sp is not None:
            spectra[name] = sp
            metrics[name+"/band_power"] = float(np.mean(sp["power"]))
            metrics[name+"/dominant_frequency_hz"] = sp["dominant_mean"]
        if values.shape[1] <= 6 or name in components:
            labels = components.get(name, list("xyz") if name in ("ee_cartesian_error", "successor_ee_error") else [str(i) for i in range(values.shape[1])])
            for i, label in enumerate(labels):
                prefix = f"{name}/component_{label}/"
                metrics.update({prefix+k: v for k, v in summary_stats(values[:, i:i+1], dt).items()})
                metrics[prefix+"signed_bias"] = float(values[:, i].mean())
                if sp is not None:
                    metrics[prefix+"band_power"] = float(sp["power"][i])
    denominator = np.sqrt(np.mean(pre[1:]**2))
    metrics["relative_modulation_denominator_rms"] = float(denominator)
    for numerator, denominator_name, name in (("post","pre","post_pre_oscillation_ratio"),
                                              ("parameter_change","feature_change","parameter_feature_change_ratio")):
        den=summary_stats(traces[denominator_name],dt)["oscillatory_rms"]
        metrics[name]=summary_stats(traces[numerator],dt)["oscillatory_rms"]/den if den>cfg["feature_epsilon"] else np.nan
    metrics["relative_modulation_rms"] = np.sqrt(np.mean(increment[1:]**2))/denominator if denominator > cfg["feature_epsilon"] else np.nan
    norm = np.linalg.norm(pre[1:], axis=1)*np.linalg.norm(post[1:], axis=1)
    good = norm > cfg["feature_epsilon"]
    metrics["pre_post_zero_norm_fraction"] = float(1-good.mean())
    metrics["pre_post_cosine"] = float(np.mean(np.sum(pre[1:][good]*post[1:][good], axis=1)/norm[good])) if good.any() else np.nan
    if "saturation_substep_count" in d:
        metrics["saturation_fraction"] = float(d["saturation_substep_count"][1:].sum()/(len(parameter)*len(meta["dof_names"])*meta["resolved_environment"]["control"]["decimation"]))
    relations = []
    recorded_pairs=[]
    for head in ("position", "torque"):
        for name in (head+"_recorded",head+"_actual",head+"_unmodulated",head+"_film"):
            if name not in traces: continue
            for j in range(traces[name].shape[1]):
                recorded_pairs += [((name,j),("ee_cartesian_error",i)) for i in range(3)]
                recorded_pairs += [((feature,k),(name,j)) for feature in ("pre_projection","post_projection","modulation_projection")
                                   if feature in traces for k in range(traces[feature].shape[1])]
    # Cartesian x/y/z and native radius/elevation/azimuth are NOT interchangeable.
    axes = cfg.get("temporal_ee_axes", [0, 1, 2])
    pairs = [(("ee_cartesian_error", i), ("estimated_ee_condition", j)) for i in axes for j in range(3)]
    targets = [("ee_velocity_world" if "ee_velocity_world" in traces else "successor_ee_error", i) for i in axes]
    for group, indices in groups.items():
        requested = cfg.get("temporal_joint_names", {}).get(group)
        names = [requested] if isinstance(requested, str) else requested
        # All arm channels; fixed front-right leg channels represent gait timing by default.
        selected = (indices if group == "arm" else [i for i in indices if meta["dof_names"][i].startswith("FR_")]) if names is None else [meta["dof_names"].index(name) for name in names]
        if any(j not in indices for j in selected):
            raise ValueError(f"Invalid temporal {group} joints")
        for joint in selected:
            executed = ("executed_torque_interval_mean", joint)
            for name in ("position_film", "position_parameter", "torque_film", "torque_parameter"):
                if name in traces and joint < traces[name].shape[1]:
                    action = (name, joint)
                    pairs += [(("ee_cartesian_error", i), action) for i in axes]
                    pairs += [(("estimated_ee_condition", j), action) for j in range(3)]
                    pairs += [(action, target) for target in targets]
                    if executed[0] in traces:
                        pairs.append((action, executed))
            for head in ("position", "torque"):
                if head+"_parameter" in traces and joint < traces[head+"_parameter"].shape[1]:
                    pairs.append(((head+"_parameter", joint), (head+"_film", joint)))
            if executed[0] in traces:
                if joint < traces["raw_action"].shape[1]:
                    pairs.append((("raw_action", joint), executed))
                pairs += [(executed, target) for target in targets]
    # Contact/base periodicity is a competing explanation, conditioned descriptively on stage.
    for source in ("contacts", "base_motion"):
        if source in traces:
            pairs += [((source, j), ("ee_cartesian_error", i)) for j in range(traces[source].shape[1]) for i in axes]
    if not (stationary or moving):
        unavailable.append("Nonstationary/event window: spectra descriptive only; coherence/lag suppressed")
        pairs = []
    elif moving:
        unavailable.append("Moving-reference residual coherence is linear-detrended and separate from stationary estimates")
    if not advanced:
        pairs=recorded_pairs if stationary or moving else []
    for (left, ix), (right, iy) in dict.fromkeys(pairs):
        x, y = traces[left], traces[right]
        if ix >= x.shape[1] or iy >= y.shape[1] or min(ix, iy) < 0:
            raise ValueError("Temporal component index outside recorded dimensions")
        rel = relationship(x[:, ix], y[:, iy], dt, spectral_cfg)
        if rel is not None:
            # Successor-labelled signals are sampled one dt later than actor-time channels.
            if right in ("ee_velocity_world", "successor_ee_error"):
                rel["lag_seconds"] += dt
                rel["phase"] = np.angle(np.exp(1j*(rel["phase"]-2*np.pi*rel["frequency"]*dt)))
            relations.append(dict(left=left, right=right, left_axis=ix, right_axis=iy,
                                  left_component=components.get(left, [str(i) for i in range(x.shape[1])])[ix],
                                  right_component=components.get(right, [str(i) for i in range(y.shape[1])])[iy],
                                  analysis_mode=analysis_mode, **rel))
            band = [rel["plan"]["usable_low_hz"], rel["plan"]["usable_high_hz"]]
            selected = (rel["frequency"] >= band[0]) & (rel["frequency"] <= band[1]) & (rel["frequency"] > 0)
            label = f"association/{analysis_mode}/{left}_{ix}_to_{right}_{iy}/"
            metrics[label+"mean_band_coherence"] = float(rel["coherence"][selected].mean())
            metrics[label+"peak_lag_seconds"] = float(rel["lag_seconds"])
            metrics[label+"peak_correlation"] = float(rel["correlation"])
            metrics[label+"zero_lag_correlation"] = float(np.corrcoef(signal.detrend(x[:,ix],type=spectral_cfg["spectral_detrend"]), signal.detrend(y[:,iy],type=spectral_cfg["spectral_detrend"]))[0,1])
            if "ee_cartesian_error" in spectra:
                for i in axes:
                    peak = spectra["ee_cartesian_error"]["dominant"][i]
                    if np.isfinite(peak):
                        k = int(np.argmin(np.abs(rel["frequency"]-peak)))
                        metrics[label+f"coherence_at_ee_{'xyz'[i]}_peak"] = float(rel["coherence"][k])
                        if left in spectra:
                            metrics[f"{left}/component_{ix}/power_at_ee_{'xyz'[i]}_peak"] = float(spectra[left]['psd'][k,ix]*rel['plan']['resolution_hz'])
                        # Phase is undefined as an explanatory diagnostic without adequate coherence.
                        metrics[label+f"phase_at_ee_{'xyz'[i]}_peak"] = (float(rel["phase"][k])
                            if rel["coherence"][k] >= cfg["coherence_threshold"] else np.nan)
    # Preserve the cross-term: persistent + varying contributions sum in signal, not power.
    for head in ("position", "torque"):
        for group in ("arm", "leg"):
            actual_key=f'{head}_actual_{group}/oscillatory_rms'
            neutral_key=f'{head}_unmodulated_{group}/oscillatory_rms'
            if actual_key in metrics and neutral_key in metrics:
                metrics[f'{head}_{group}/modulated_minus_unmodulated_rms']=metrics[actual_key]-metrics[neutral_key]
            names = [f"{head}_{ref}_{group}/oscillatory_rms" for ref in ("actual", "fixed", "unmodulated")]
            if all(k in metrics for k in names):
                actual, constant, unmodulated = (metrics[k] for k in names)
                # Paired differences of amplitudes, NOT additive component RMS/energy.
                metrics[f"{head}_{group}/fixed_minus_unmodulated_rms"] = constant-unmodulated
                metrics[f"{head}_{group}/actual_minus_fixed_rms"] = actual-constant
            total, fixed, varying = (f"{head}_{name}_{group}" for name in ("film", "persistent_effect", "varying_effect"))
            if all(k in spectra for k in (total, fixed, varying)):
                cross = spectra[total]["power"]-spectra[fixed]["power"]-spectra[varying]["power"]
                metrics[f"{head}_{group}/persistent_varying_cross_power"] = float(cross.mean())
    if not spectra:
        unavailable.append("Insufficient Welch averages/usable frequencies; coherence/spectra unavailable")
    elif (stationary or moving) and not relations:
        unavailable.append("No nonconstant valid fixed-component pairs for coherence/lag")
    return metrics, traces, spectra, relations, unavailable, units


def aggregate(frame):
    keys = ["task", "checkpoint", "training_seed", "profile", "loading", "kind", "stage", "metric"]
    if "analysis_mode" in frame:
        keys.append("analysis_mode")
    if "attribution_quality" in frame:
        keys.append("attribution_quality")
    # Windows -> trial -> evaluation seed. Never use timesteps as replicates.
    trials = frame.groupby(keys+["evaluation_seed", "trial_id"], dropna=False).value.mean().reset_index()
    seeds = trials.groupby(keys+["evaluation_seed"], dropna=False).value.agg(["mean", "count"]).reset_index()
    cohorts = seeds.groupby(keys, dropna=False)["mean"].agg(["mean", "std", "count"]).reset_index()
    counts=seeds.groupby(keys,dropna=False)['count'].sum().rename('trial_seed_pairs').reset_index()
    cohorts=cohorts.merge(counts,on=keys)
    windows=frame.groupby(keys,dropna=False).size().rename('windows').reset_index()
    cohorts=cohorts.merge(windows,on=keys)
    cohorts["variability"] = "evaluation-seed SD within checkpoint (not trained-policy SD)"
    return trials, seeds, cohorts


def plot_window(d, traces, spectra, relations, identity, output, cfg):
    if not cfg.get("advanced_diagnostics",False):
        return plot_recorded_window(d,traces,spectra,identity,output,cfg)
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
    for key in ("position_film_arm", "position_film_leg", "torque_film_arm", "torque_film_leg"):
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
        for group in ("arm", "leg"):
            key = "executed_torque_interval_mean_"+group
            if key in traces:
                axes[4].plot(time, np.sqrt(np.mean(traces[key]**2, axis=1)), label=group+" executed interval-mean torque")
        if axes[4].lines:
            axes[4].legend()
        else:
            axes[4].plot(time, np.sqrt(np.mean(traces["executed_torque_interval_mean"]**2, axis=1)), label="all joints (grouping unavailable)")
    else:
        axes[4].text(.5,.5,"Executed torque recording unavailable",ha="center",transform=axes[4].transAxes)
    axes[4].set_ylabel("Torque [Nm]"); axes[4].set_xlabel("Time relative to event [s]")
    for ax in axes:
        ax.axvline(0, color="grey", linestyle=":"); ax.grid(alpha=.2)
    caption = (f"{DISPLAY_NAMES.get(identity['task'], identity['task'])}: {identity['profile']} / {identity['loading']} / stage {identity['stage']}\n"
               f"{identity['checkpoint'][:8]} / seed {identity['evaluation_seed']} / trial {identity['trial_id']} / "
               f"{identity['actual_start']:.2f}-{identity['actual_end']:.2f}s / {identity['analysis_mode']}")
    fig.suptitle(caption)
    shared.style_figure(fig, cfg)
    fig.tight_layout()
    stem = f"window-{identity['window_id']:05d}"
    for ext in ("pdf", "png"):
        fig.savefig(output/(stem+"-traces."+ext), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.8))
    for key in ("pre", "post", "modulation", "parameter_change"):
        if key in spectra:
            sp = spectra[key]; keep = (sp["frequency"] >= sp["plan"]["usable_low_hz"]) & (sp["frequency"] <= sp["plan"]["usable_high_hz"])
            axes[0].semilogy(sp["frequency"][keep], sp["psd"].mean(1)[keep], label=key)
    for key in ("ee_cartesian_error", "raw_action", "position_film", "torque_film"):
        if key in spectra:
            sp = spectra[key]; keep = (sp["frequency"] >= sp["plan"]["usable_low_hz"]) & (sp["frequency"] <= sp["plan"]["usable_high_hz"])
            axes[1].semilogy(sp["frequency"][keep], sp["psd"].mean(1)[keep], label=key)
    # Plot a bounded labelled subset; all component links remain in relationships.csv.
    attributed = [r for r in relations if r["left"].startswith(("position_", "torque_"))]
    plotted_relations = (attributed + [r for r in relations if not r["left"].startswith(("position_", "torque_"))])[:cfg.get("max_plot_relationships", 8)]
    short = {"ee_cartesian_error":"err", "estimated_ee_condition":"est", "parameter_change":"dparam",
             "raw_action":"raw", "executed_torque_interval_mean":"tau", "ee_velocity_world":"EE vel",
             "successor_ee_error":"next err", "position_film":"pos FiLM", "torque_film":"FF FiLM",
             "position_parameter":"dpos", "torque_parameter":"dFF"}
    for rel in plotted_relations:
        keep = (rel["frequency"] >= rel["plan"]["usable_low_hz"]) & (rel["frequency"] <= rel["plan"]["usable_high_hz"])
        label = f"{short.get(rel['left'],rel['left'])}[{rel['left_axis']}] -> {short.get(rel['right'],rel['right'])}[{rel['right_axis']}]"
        axes[2].plot(rel["frequency"][keep], rel["coherence"][keep], label=label)
    axes[2].set_ylim(0, 1); axes[2].set_ylabel("Coherence")
    for ax in axes:
        ax.set_xlabel("Frequency [Hz]"); ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=5, loc="upper center", bbox_to_anchor=(.5, -.28))
        else:
            ax.text(.5,.5,"Unavailable: insufficient averages\nor nonstationary window",ha="center",fontsize=7,transform=ax.transAxes)
    axes[0].set_ylabel("Mean feature PSD [feature²/Hz]")
    axes[1].set_ylabel("PSD [native signal units²/Hz]")
    fig.suptitle(caption)
    shared.style_figure(fig, cfg)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output/(stem+"-spectra."+ext), dpi=160)
    plt.close(fig)
    if cfg.get("plot_action_references", True) and "position_actual" in traces:
        # These head references share the recorded trunk, not simulated alternate trajectories.
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        for row, head in enumerate(("position", "torque")):
            for col, group in enumerate(("arm", "leg")):
                ax = axes[row, col]
                for reference in ("actual", "fixed", "unmodulated"):
                    key = f"{head}_{reference}_{group}"
                    if key in spectra:
                        sp = spectra[key]; f = sp["frequency"]
                        keep = (f >= sp["plan"]["usable_low_hz"]) & (f <= sp["plan"]["usable_high_hz"])
                        ax.semilogy(f[keep], sp["psd"].mean(1)[keep], label=reference)
                ax.set_title(f"{group}: {head} ({identity.get('attribution_quality', 'unknown')})")
                ax.set_xlabel("Frequency [Hz]")
                ax.set_ylabel("PSD ["+("rad" if head == "position" else "Nm")+"²/Hz]")
                ax.grid(alpha=.2)
                if ax.lines: ax.legend()
        fig.suptitle(caption+"\nAlgebraic action references, not executed trajectories")
        shared.style_figure(fig, cfg); fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(output/(stem+"-action-references."+ext), dpi=160)
        plt.close(fig)


def plot_recorded_window(d,traces,spectra,identity,output,cfg):
    import matplotlib.pyplot as plt
    time=d["time_seconds"][1:]
    fig,axes=plt.subplots(5,1,figsize=(9,10),sharex=True)
    selections=[("ee_force_applied_world_interval_mean",), ("pre_projection","post_projection"),
                ("position_recorded_arm","position_recorded_leg"), ("torque_recorded_arm","torque_recorded_leg"), ("ee_cartesian_error",)]
    labels=["World load [N]","Fixed PCA features","Position offsets [rad]","FF requests [Nm]","EE error [m]"]
    for ax,names,label in zip(axes,selections,labels):
        for name in names:
            if name not in traces: continue
            # Signed coordinates, not an amplitude norm used as a frequency signal.
            for j in range(min(traces[name].shape[1],cfg.get("trace_components",3))):
                ax.plot(time,traces[name][:,j],label=f"{name}[{j}]",alpha=.8)
        ax.set_ylabel(label); ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=7,ncol=2)
    axes[-1].set_xlabel("Actor time [s]")
    fig.suptitle(f"{DISPLAY_NAMES.get(identity['task'],identity['task'])}: {identity['profile']} / stage {identity['stage']} / trial {identity['trial_id']}")
    shared.style_figure(fig,cfg); fig.tight_layout()
    stem=f"window-{identity['window_id']:05d}"
    for ext in ('pdf','png'): fig.savefig(output/(stem+'-traces.'+ext))
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,names,unit in zip(axes,[("pre","post","modulation"),("position_recorded_arm","position_actual_arm","position_unmodulated_arm","position_film_arm"),("torque_recorded_arm","torque_actual_arm","torque_unmodulated_arm","torque_film_arm")],['feature²/Hz','rad²/Hz','Nm²/Hz']):
        for name in names:
            if name not in spectra: continue
            sp=spectra[name]; f=sp['frequency']; keep=(f>=sp['plan']['usable_low_hz'])&(f<=sp['plan']['usable_high_hz'])
            ax.semilogy(f[keep],sp['psd'].mean(1)[keep],label=name)
        ax.set_xlabel('Frequency [Hz]'); ax.set_ylabel(unit); ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=7)
    shared.style_figure(fig,cfg); fig.tight_layout()
    for ext in ('pdf','png'): fig.savefig(output/(stem+'-spectra.'+ext))
    plt.close(fig)


def recorded_associations(frame,output,cfg):
    """Trial amplitudes first; correlations stay within checkpoint/stage/seed."""
    keys=['task','checkpoint','profile','loading','stage','kind','analysis_mode','evaluation_seed','trial_id']
    wanted=[m for m in frame.metric.unique() if m.endswith('/oscillatory_rms') and not m.startswith('association/')]
    trial=frame[frame.metric.isin(wanted)].groupby(keys+['metric'],dropna=False).value.mean().unstack('metric').reset_index()
    rows=[]
    for key,g in trial.groupby(keys[:-1],dropna=False):
        for left in ['pre/oscillatory_rms','post/oscillatory_rms','modulation/oscillatory_rms','position_recorded_arm/oscillatory_rms','torque_recorded_arm/oscillatory_rms']:
            for right in ['position_recorded_arm/oscillatory_rms','position_recorded_leg/oscillatory_rms','torque_recorded_arm/oscillatory_rms','torque_recorded_leg/oscillatory_rms','ee_cartesian_error/oscillatory_rms',*[f'ee_cartesian_error/component_{axis}/oscillatory_rms' for axis in 'xyz']]:
                if left==right or left not in g or right not in g: continue
                v=g[[left,right]].dropna()
                corr=v[left].corr(v[right]) if len(v)>=3 and (v.std()>0).all() else np.nan
                rows.append(dict(zip(keys[:-1],key),left=left,right=right,correlation=corr,trials=len(v)))
    table=pd.DataFrame(rows); table.to_csv(output/'amplitude_associations.csv',index=False)
    if len(table):
        by=[k for k in keys[:-1] if k!='evaluation_seed']+['left','right']
        # Equal evaluation-seed weighting; undefined correlations remain missing.
        pooled=table.groupby(by,dropna=False).agg(mean=('correlation','mean'),std=('correlation','std'),seeds=('correlation','count'),trial_seed_pairs=('trials','sum')).reset_index()
        pooled.to_csv(output/'amplitude_association_summary.csv',index=False)
    trial.to_csv(output/'trial_amplitudes.csv',index=False)
    import matplotlib.pyplot as plt
    # Each scatter keeps task/load stage and checkpoint distinct; no pooled correlation.
    for index,(key,g) in enumerate(trial.groupby(keys[:-2],dropna=False)):
        if index>=cfg.get('max_scatter_plots',6): break
        x='post/oscillatory_rms'; y='ee_cartesian_error/oscillatory_rms'
        if x not in g or y not in g: continue
        fig,ax=plt.subplots(figsize=(5,4))
        for seed,part in g.groupby('evaluation_seed'): ax.scatter(part[x],part[y],label=f'eval seed {seed}')
        ax.set(xlabel='Post-feature oscillatory RMS',ylabel='EE oscillatory RMS [m]',title=str(key[1:]))
        ax.legend(); shared.style_figure(fig,cfg); fig.tight_layout()
        for ext in ('pdf','png'): fig.savefig(output/f'amplitudes-{index:03d}.{ext}')
        plt.close(fig)
    return table.to_dict('records')


def select_plot_windows(windows, budget):
    """Greedy deterministic coverage, then severity extremes; no visual cherry-picking."""
    groups = defaultdict(list)
    for w in windows:
        key = tuple(w[k] for k in ("checkpoint", "profile", "loading", "kind", "stage"))
        groups[key].append(w)
    candidates = []
    for group in groups.values():
        ordered = sorted(group, key=lambda w: (w["oscillation_severity"], w["run"], w["trial_id"], w["actual_start"]))
        for i, w in enumerate(ordered):
            candidates.append({**w, "severity_group": "low" if i < (len(ordered)+1)//2 else "high"})
    def categories(w):
        return {(name, str(w[name])) for name in ("profile", "loading", "checkpoint", "evaluation_seed", "severity_group")} | {
            ("stage", str((w["profile"], w["loading"], w["kind"], w["stage"], w["severity_group"])))}
    all_categories = set().union(*(categories(w) for w in candidates)) if candidates else set()
    covered, trials, chosen = set(), set(), []
    while candidates and len(chosen) < budget:
        fresh = [w for w in candidates if (w["run"], w["trial_id"]) not in trials]
        pool = fresh or candidates
        def rank(w):
            # First cover dimensions, then stage-specific low/high examples.
            missing = categories(w)-covered
            score = sum(4 if k != "stage" else 1 for k, _ in missing)
            extremity = w["oscillation_severity"] * (1 if w["severity_group"] == "high" else -1)
            return (-score, -extremity, w["run"], w["trial_id"], w["actual_start"])
        pick = min(pool, key=rank)
        candidates.remove(pick); chosen.append(pick)
        covered |= categories(pick); trials.add((pick["run"], pick["trial_id"]))
    return chosen, [dict(category=k, value=v, reason="plot budget/available-trial coverage") for k, v in sorted(all_categories-covered)]


def render_selected(selected, runs, output, cfg):
    """Second bounded pass: reread only selected trials/windows; no feature histories retained."""
    for run in runs:
        picks = [w for w in selected if w["run"] == str(run["path"])]
        if not picks:
            continue
        specs = window_specs(run["meta"], cfg)
        wanted_specs = [s for s in specs if any(s["start"] == w["start"] and s["kind"] == w["kind"] for w in picks)]
        heads, _ = optional_heads(run["meta"], cfg)
        for tid, spec, raw in stream_windows(run, {**cfg, "trial_ids": sorted({w["trial_id"] for w in picks}), "_window_specs": wanted_specs}):
            for segment in contiguous_segments(raw, run["meta"]["dt"]):
                matches = [w for w in picks if w["trial_id"] == tid and w["start"] == spec["start"]
                           and w["kind"] == spec["kind"] and np.isclose(w["actual_start"], segment[0]["time_seconds"])]
                if not matches:
                    continue
                d = {k: np.stack([r[k] for r in segment]) for k in segment[0]}
                outcome, _ = cached_window(d, run["meta"], heads, {**cfg, "stationary_window": spec["kind"] == "sustained"})
                _, traces, spectra, relations, _, _ = outcome
                plot_window(d, traces, spectra, relations, matches[0], output, cfg)


def cached_window(d, meta, heads, cfg):
    """Only numerical inputs enter this key; plotting choices cannot invalidate it."""
    cache = NumericCache(cfg)
    key, result = cache.lookup('film_windows', (d, meta, heads, numerical_config(cfg),
                                               cfg.get('feature_bases')))
    if result is None:
        result = analyze_window(d, meta, heads, cfg)
        cache.save(key, result)
    return result, cache.hits


def analyze_raw_window(job):
    # CPU-only worker; optional head weights are plain arrays loaded by the parent.
    tid, spec, raw, meta, heads, cfg = job
    results = []
    for segment in contiguous_segments(raw, meta['dt']):
        if len(segment)*meta['dt'] < cfg['minimum_window_seconds'] or len(segment) < 4:
            results.append(None); continue
        d = {k: np.stack([r[k] for r in segment]) for k in segment[0]}
        try:
            result, hits = cached_window(d, meta, heads, {**cfg, 'stationary_window': spec['kind']=='sustained'})
            results.append(dict(result=result, hits=hits))
        except (ValueError, KeyError) as exc:
            results.append(dict(error=str(exc)))
    return results


def run_analysis(config):
    threads = config.get('cpu_blas_threads',1)
    if isinstance(threads,bool) or not isinstance(threads,int) or threads < 1:
        raise ValueError('cpu_blas_threads must be a positive integer')
    # Bound parent/worker BLAS teams when the optional limiter is available.
    with optional_thread_limits(limits=threads,user_api='blas'):
        return _run_analysis(config)


def _run_analysis(config):
    started = time.perf_counter()
    timings = {}
    cfg = default_config(config)
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
    timings['discovery_seconds'] = time.perf_counter()-started
    stage_started = time.perf_counter()
    cache = NumericCache(cfg)
    key, bases = cache.lookup('film_basis', (data_signature(runs), numerical_config(cfg)))
    if bases is None:
        bases = fit_feature_bases(runs,cfg)
        cache.save(key, bases)
    cfg["feature_bases"] = bases
    timings['basis_seconds'] = time.perf_counter()-stage_started
    timings['basis_cache_hits'] = cache.hits
    timings['window_cache_hits'] = 0
    stage_started = time.perf_counter()
    last_progress = stage_started
    rows, windows, provenance, heads_cache = [], [], [], {}
    reservoirs, seen = defaultdict(list), defaultdict(int)
    rng = np.random.default_rng(cfg.get("sampling_seed", 0))
    for run in runs:
        meta = run["meta"]; ckpt = meta["checkpoint_sha256"]
        print(f"FiLM analysis: {run['path']}", flush=True)
        if ckpt not in heads_cache:
            heads_cache[ckpt] = optional_heads(meta, cfg)
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
        jobs = ((tid,spec,raw,meta,heads,cfg) for tid,spec,raw in stream_windows(run,cfg))
        for job, prepared in ordered_work(analyze_raw_window, jobs, cfg.get('num_workers',1)):
            tid, spec, raw = job[:3]
            segments = contiguous_segments(raw, meta["dt"])
            if not segments:
                exclusions.append(dict(run=str(run["path"]), trial_id=tid, window=spec, reason="No valid pre-failure samples"))
            for segment_index, segment in enumerate(segments):
                if len(segment)*meta["dt"] < cfg["minimum_window_seconds"] or len(segment) < 4:
                    exclusions.append(dict(run=str(run["path"]), trial_id=tid, window=spec, reason="Short/gapped/failed window")); continue
                d = {k: np.stack([r[k] for r in segment]) for k in segment[0]}
                identity = dict(task=meta["task"], checkpoint=ckpt, training_seed=meta.get("checkpoint_training_seed"),
                    evaluation_seed=meta["evaluation_seed"], trial_id=tid, profile=meta["arguments"]["profile"],
                    loading=meta["arguments"]["loading"], run=str(run["path"]), **spec, window_id=len(windows))
                outcome = prepared[segment_index]
                if 'error' in outcome:
                    exclusions.append(dict(**identity, reason=outcome['error'])); continue
                metrics, traces, spectra, relations, unavailable, units = outcome['result']
                timings['window_cache_hits'] += outcome['hits']
                complete = len(segment) == len(raw) and len(raw)*meta["dt"] >= spec["end"]-spec["start"]-meta["dt"]*.5
                identity["analysis_mode"] = units["temporal_analysis_mode"]
                statuses=[v["status"] for v in units["head_parity"].values()]
                identity["attribution_quality"] = ("unverified" if "unverified_algebraic_only" in statuses else
                    "unavailable" if "unavailable" in statuses or not statuses else
                    "precision_explained" if "approximate_precision_explained" in statuses else "strict")
                if not cfg.get('advanced_diagnostics',False): identity['attribution_quality']='recorded'
                windows.append(dict(**identity, samples=len(segment), complete=complete,
                    actual_start=float(d["time_seconds"][0]), actual_end=float(d["successor_time_seconds"][-1]),
                    actual_load_mean_n=float(np.linalg.norm(d["ee_force_applied_world_interval_mean"],axis=1).mean()) if "ee_force_applied_world_interval_mean" in d else None,
                    spectral_available=bool(spectra), temporal_available=bool(relations),
                    attribution_available="position_film" in traces,
                    spectral_plan=units["spectral_plan"], unavailable=unavailable.copy(),
                    oscillation_severity=metrics["ee_cartesian_error/oscillatory_rms"], units=units))
                exclusions.extend(dict(**identity, reason=reason) for reason in unavailable)
                rows.extend(dict(**identity, metric=k, value=v) for k, v in metrics.items())
                relations_out = []  # Stream frequency rows; do not retain all windows' cross spectra in RAM.
                for rel in relations:
                    for i, f in enumerate(rel["frequency"]):
                        relations_out.append(dict(**identity, left=rel["left"], right=rel["right"], left_axis=rel["left_axis"], right_axis=rel["right_axis"],
                            left_component=rel["left_component"], right_component=rel["right_component"],
                            frequency_hz=f, coherence=rel["coherence"][i], phase_rad=rel["phase"][i],
                            lag_seconds=rel["lag_seconds"], lag_correlation=rel["correlation"]))
                if relations_out:
                    path = output/"relationships.csv"
                    pd.DataFrame(relations_out).to_csv(path, mode="a", header=not path.exists(), index=False)
                # Compact numeric window export makes plots reproducible, without retaining features.
                arrays = {"time_seconds": d["time_seconds"][1:], "successor_time_seconds": d["successor_time_seconds"][1:]}
                arrays.update({"trace_"+k: v for k,v in traces.items() if k not in ("pre", "post", "gamma", "beta", "modulation", "parameter_change", "feature_change", "post_change")})
                for k in ("gamma", "beta", "modulation", "parameter_change"):
                    arrays["rms_"+k] = np.sqrt(np.mean(traces[k]**2, axis=1))
                for k, sp in spectra.items():
                    arrays["frequency_"+k], arrays["psd_"+k] = sp["frequency"], sp["psd"].mean(1)
                    # Retain signed component spectra; group means are not spectra of norms.
                    if k in units["signal_components"] or traces[k].shape[1] <= 6:
                        arrays["psd_components_"+k] = sp["psd"]
                np.savez_compressed(output/f"window-{identity['window_id']:05d}.npz", **arrays)
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
                if cfg.get('progress',True) and time.perf_counter()-last_progress >= 30:
                    last_progress = time.perf_counter()
                    print(f"  {len(windows):,} windows; {timings['window_cache_hits']:,} cached; "
                          f"{(last_progress-stage_started)/60:.1f} min", flush=True)
            if cfg.get("max_windows") and len(windows) >= cfg["max_windows"]:
                break
        if cfg.get("max_windows") and len(windows) >= cfg["max_windows"]:
            break
    timings['read_compute_export_seconds'] = time.perf_counter()-stage_started
    stage_started = time.perf_counter()
    frame = pd.DataFrame(rows)
    summary = []
    associations = []
    if len(frame):
        trials, seeds, cohorts = aggregate(frame)
        for name, table in (("windows", frame), ("trials", trials), ("seeds", seeds), ("summary", cohorts)):
            table.to_csv(output/(name+".csv"), index=False)
        summary = cohorts.to_dict("records")
        compare_windows(frame, output, cfg)
        associations = recorded_associations(frame,output,cfg)
        cohorts[cohorts.metric.str.startswith('association/')].to_csv(output/'temporal_associations.csv',index=False)
    else:
        exclusions.append(dict(reason="No usable original actor-feature FiLM windows"))
    if not (output/"relationships.csv").exists():
        pd.DataFrame(columns=["left", "right", "analysis_mode", "frequency_hz", "coherence"]).to_csv(output/"relationships.csv", index=False)
    timings['aggregation_and_scatter_seconds'] = time.perf_counter()-stage_started
    stage_started = time.perf_counter()
    selected, omitted = select_plot_windows(windows, cfg["max_plot_windows"])
    pd.DataFrame([{k: v for k, v in w.items() if k not in ("units", "unavailable", "spectral_plan")} for w in selected]).to_csv(output/"plot_selection.csv", index=False)
    (output/"plot_omitted_categories.json").write_text(json.dumps(clean(omitted), indent=2))
    render_selected(selected, runs, output, cfg)
    timings['example_figures_seconds'] = time.perf_counter()-stage_started
    # Explicit coverage includes every planned stage, even when all trials fail before it.
    coverage = []
    for run in runs:
        for spec in window_specs(run["meta"], cfg):
            subset = [w for w in windows if w["run"] == str(run["path"]) and w["start"] == spec["start"] and w["kind"] == spec["kind"]]
            for mode in sorted({w["analysis_mode"] for w in subset}) or ["unavailable"]:
                part = [w for w in subset if w["analysis_mode"] == mode]
                coverage.append(dict(run=str(run["path"]), profile=run["meta"]["arguments"]["profile"], loading=run["meta"]["arguments"]["loading"],
                    evaluation_seed=run["meta"]["evaluation_seed"], **spec, analysis_mode=mode,
                    windows=len(part), trials=len({w["trial_id"] for w in part}),
                    spectral_windows=sum(w["spectral_available"] for w in part), temporal_windows=sum(w["temporal_available"] for w in part),
                    attribution_windows=sum(w["attribution_available"] for w in part),
                    reason="; ".join(sorted({s for w in part for s in w["unavailable"]})) if part else "No usable samples (failure/gap/selection/run limit)",
                    min_duration_seconds=min((w["spectral_plan"]["duration_seconds"] for w in part), default=None),
                    min_segments=min((w["spectral_plan"]["segments"] for w in part), default=None),
                    max_usable_low_hz=max((w["spectral_plan"]["usable_low_hz"] for w in part), default=None),
                    min_usable_high_hz=min((w["spectral_plan"]["usable_high_hz"] for w in part), default=None),
                    max_resolution_hz=max((w["spectral_plan"]["resolution_hz"] for w in part), default=None)))
    pd.DataFrame(coverage).to_csv(output/"coverage.csv", index=False)
    parity_rows=[]
    for w in windows:
        for head,audit in w["units"]["head_parity"].items():
            for i,joint in enumerate(audit["joint_names"]):
                parity_rows.append(dict(window_id=w["window_id"],checkpoint=w["checkpoint"],task=w["task"],
                    evaluation_seed=w["evaluation_seed"],trial_id=w["trial_id"],head=head,joint=joint,status=audit["status"],
                    **{k:audit[k][i] for k in ("max_abs_error","rms_error","precision_residual_max","error_bound_max","physical_scale")}))
    pd.DataFrame(parity_rows).to_csv(output/"action_attribution_audit.csv",index=False)
    physical={}
    stage_started = time.perf_counter()
    if cfg.get("physical_comparison", False):
        from exp3_oscillation_physical import analyze
        physical=analyze(cfg,output/"physical_comparison")
    timings['physical_comparison_seconds'] = time.perf_counter()-stage_started
    evidence=explanatory_report(summary,physical,parity_rows) if cfg.get("advanced_diagnostics",False) else []
    if evidence: pd.DataFrame(evidence).to_csv(output/"evidence.csv",index=False)
    pca = export_embeddings(reservoirs, output, cfg)
    timings['total_seconds'] = time.perf_counter()-started
    (output/'timings.json').write_text(json.dumps(timings, indent=2))
    result = dict(schema_version=2, complete=True, config=cfg, provenance=provenance, windows=windows,
        timings=timings,
        plot_selection=selected, plot_omitted_categories=omitted, coverage=coverage,
        physical_comparison=physical, evidence=evidence, amplitude_associations=associations,
        conclusion="Recorded within-stage feature, action and EE oscillation associations.",
        exclusions=exclusions, summary=summary, pca=pca,
        interpretation="Observational associations only; no causal confirmation or simulated counterfactuals.",
        preprocessing="No filtering/interpolation; core amplitudes and spectra remove window means. Hann Welch with exported segment/band limits. Stage-separated windows average within trials then seeds. Optional advanced moving-reference diagnostics use linear detrending.")
    (output/"summary.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False))
    (output/"COMPLETE").write_text("complete\n")
    return result


def explanatory_report(summary, physical, parity_rows):
    """Evidence inventory, not a causal classifier or a winner chosen from correlations."""
    def spread(metric):
        values=[r["mean"] for r in summary if r["metric"]==metric and r["kind"]=="sustained" and np.isfinite(r["mean"])]
        return f"Stratum means range {min(values):.3g}-{max(values):.3g} ({len(values)} strata)" if values else "Unavailable"
    statuses=sorted({r["status"] for r in parity_rows})
    def head_evidence(suffix):
        rows=[r for r in summary if r["metric"].endswith(suffix) and r["kind"]=="sustained"
              and r.get("attribution_quality") in ("strict", "precision_explained") and np.isfinite(r["mean"])]
        return (f"Amplitude increases in {sum(r['mean'] > 0 for r in rows)}/{len(rows)} validated head/group/stage strata; see paired RMS differences"
                if rows else "Validated head-level comparison unavailable; unverified values excluded from evidence")
    comparison="Matched physical comparison unavailable"
    f=pd.DataFrame(physical.get("matched",[]))
    if len(f):
        f=f[(f.signal=="ee_error")&(f.metric=="oscillatory_rms")&(f.loading=="loaded")&(f.scheduled_load_n>0)]
        higher=total=0
        for _,g in f.groupby(["profile","stage","axis","protocol_hash"]):
            pilot=g[g.task=="b1z1_pact"]
            if len(pilot)!=1: continue
            for _,other in g[g.task!="b1z1_pact"].iterrows():
                total+=1; higher+=int(pilot.iloc[0]["mean"]>other["mean"])
        comparison=f"PILOT has higher mean RMS in {higher}/{total} matched axis/stage/baseline comparisons; inspect seed SD/counts (not a significance test)"
    motion=pd.DataFrame(physical.get("matched",[]))
    transmission="Matched simultaneous velocity decomposition unavailable"
    if len(motion):
        terms=motion[(motion.signal=="base_arm")&(motion.metric=="band_cross_term")&(motion.scheduled_load_n>0)]
        if len(terms):
            transmission=f"Negative base/relative-arm cross-power in {int((terms['mean']<0).sum())}/{len(terms)} matched axis/stage/method strata (consistent with attenuation); positive values indicate reinforcement"
    return [dict(candidate="Larger physical EE oscillation",supporting=comparison,
                 contradictory="Not assumed uniform across axes, stages or baselines",limitations="Survivor-conditioned complete windows; PD gains differ; evaluation seeds are not training replicates"),
            dict(candidate="Persistent FiLM scaling",supporting=spread("post_pre_oscillation_ratio")+"; "+head_evidence("/fixed_minus_unmodulated_rms"),
                 contradictory="Feature amplitude is not harmful control gain; use physical-head actual/fixed/unmodulated comparisons",limitations="Offline algebraic references; no alternative trajectory executed"),
            dict(candidate="Time-varying modulation",supporting=spread("parameter_feature_change_ratio")+"; "+head_evidence("/actual_minus_fixed_rms"),
                 contradictory="Small parameter-change amplitude or inconsistent high/low differences weaken this mechanism",limitations="Cross-terms retained; high/low EE difference is guaranteed by selection"),
            dict(candidate="Action-channel behavior",supporting="Attribution statuses: "+(", ".join(statuses) or "unavailable"),
                 contradictory="Requested channels are not clipped executed torques; unverified attribution cannot support an explanation",limitations="See per-joint arithmetic errors/bounds and gains; persistent/varying effects are not unique compensation torques"),
            dict(candidate="Locomotion transmission",supporting=transmission,
                 contradictory="Negative base-arm cross terms suggest attenuation; shared peaks alone do not establish transmission cause",limitations="Measured relative velocity residual assumes consistent base reference point; no independent Jacobian validation or causal intervention"),
            dict(candidate="Overall conclusion",supporting="Compare physical differences before attributing them to FiLM",
                 contradictory="No unique mechanism established by this observational analysis",limitations="Available evidence remains inconclusive about closed-loop causation")]


def compare_windows(frame, output, cfg):
    keys = ["checkpoint", "evaluation_seed", "trial_id", "profile", "loading", "kind", "stage", "window_id"]
    strata = [k for k in ("analysis_mode", "attribution_quality") if k in frame]
    keys += strata
    wide = frame.pivot(index=keys, columns="metric", values="value").reset_index()
    # Descriptive high/low comparison WITHIN task/load/motion stage and checkpoint.
    groups = ["checkpoint", "profile", "loading", "kind", "stage"] + strata
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
                for metric in ("ee_cartesian_error/oscillatory_rms", "relative_modulation_rms", "parameter_change/oscillatory_rms", "pre/oscillatory_rms", "ee_estimator_mismatch/rms", "saturation_fraction",
                               "position_actual_arm/oscillatory_rms", "position_fixed_arm/oscillatory_rms", "position_unmodulated_arm/oscillatory_rms",
                               "torque_actual_arm/oscillatory_rms", "torque_fixed_arm/oscillatory_rms", "torque_unmodulated_arm/oscillatory_rms"):
                    if metric in v:
                        per_trial = v.groupby(["evaluation_seed", "trial_id"])[metric].mean().dropna()
                        seed_values=per_trial.groupby(level=0).mean()
                        rows.append(dict(zip(groups,key), comparison=label, group=group, metric=metric,
                            mean=seed_values.mean(), sd=seed_values.std(ddof=1), seeds=len(seed_values),
                            trial_sd=per_trial.std(ddof=1),
                            trials=len(per_trial), windows=len(v), variability="evaluation-seed SD after trial averaging",
                            selection_note="EE oscillation differs by construction; other signals are descriptive associations"))
    pd.DataFrame(rows).to_csv(output/"high_low_windows.csv", index=False)


def export_embeddings(reservoirs, output, cfg):
    import matplotlib.pyplot as plt
    result = {}
    for ckpt, pool in reservoirs.items():
        if len(pool) < 3:
            continue
        pre, post = np.stack([r[0] for r in pool]), np.stack([r[1] for r in pool])
        basis=cfg.get('feature_bases',{}).get(ckpt)
        if basis is None:
            result[ckpt]=dict(unavailable='Missing checkpoint-wide fixed PCA basis'); continue
        from exp3_film_figures import paired_projection
        projected=np.concatenate(paired_projection(pre,post,basis))
        explained=np.asarray(basis['explained_variance'][:2])
        x=np.concatenate((pre,post)).astype(float)-np.asarray(basis['center'])
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
            for offset,marker,label in ((0,"o","pre"),(len(pool),"^","post")):
                sl=slice(offset,offset+len(pool))
                points = ax.scatter(xy[sl,0], xy[sl,1], c=colors[sl], s=8, marker=marker,
                                    cmap="coolwarm",vmin=colors.min(),vmax=colors.max(),label=label)
            ax.legend()
            fig.colorbar(points, ax=ax, label=color_label)
            ax.set_title(name.upper()+f" (paired pre/post; qualitative)\nPCA explained variance: {100*explained.sum():.1f}%")
            shared.style_figure(fig, cfg)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(output/f"{name}-{ckpt[:12]}.{ext}", dpi=160)
            plt.close(fig)
            np.savez_compressed(output/f"{name}-{ckpt[:12]}.npz", coordinates=xy, color=colors,
                                signed_error_m=np.tile([r[2] for r in pool], 2),
                                explained_variance_ratio=explained if name=="pca" else np.array([]))
    return result
