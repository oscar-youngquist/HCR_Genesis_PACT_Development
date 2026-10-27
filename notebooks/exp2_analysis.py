"""CPU-only Exp2 adapter; metric definitions remain owned by exp1_analysis."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import exp1_analysis as shared

VERSION = 1
PROFILES = ("pushing", "pulling", "carrying")
CURVES = ("force", "ee_error", "command_vx", "measured_vx")
EXTRA = ("time_seconds", "horizon_complete", "ee_force_applied_world_interval_mean",
         "task_stage_successor")
RESPONSE = ("ee_peak_m", "recovery_time_s", "recovery_fraction",
            "recovery_fraction_ee", "recovery_fraction_velocity", "ee_iae_m_s")
SMOOTH = ("torso_angular_acceleration_rms", "ee_linear_acceleration_rms")


def protocol(meta):
    p = shared.protocol(meta)
    p.update(experiment=2, task_stages=meta["task_stages"],
             exp2_evaluator_sha256=meta.get("exp2_evaluator_sha256"),
             exp2_utilities_sha256=meta.get("exp2_utilities_sha256"),
             experiment2_conventions=meta.get("experiment2_conventions"))
    return p


def stream_trials(directory, meta, trials, smoothness=False):
    """Keep only analysis columns and unfinished trials, never observation tensors."""
    manifest = shared.read_json(directory / "records.json")
    fields = list(shared.REQUIRED + EXTRA)
    if smoothness:
        fields += ["base_angular_velocity_world", "ee_velocity_world"]
    pending, finished, seen_chunks = {}, set(), set()
    for chunk in manifest["chunks"]:
        path = (directory / chunk["file"]).resolve()
        if path.parent != directory.resolve() or path in seen_chunks:
            raise ValueError("Invalid or duplicate record chunk")
        seen_chunks.add(path)
        with np.load(path, allow_pickle=False) as arrays:
            d = {k: arrays[k] for k in fields}
        for key, value in d.items():
            spec = manifest["fields"][key]
            if (len(value) != chunk["rows"] or list(value.shape[1:]) != spec["shape"]
                    or str(value.dtype) != spec["dtype"]):
                raise ValueError(f"Record schema mismatch: {key}")
        for tid in np.unique(d["trial_id"]):
            tid = int(tid)
            if tid not in trials or tid in finished:
                raise ValueError("Unknown trial or transitions after terminal row")
            selected = d["trial_id"] == tid
            part = {k: v[selected] for k, v in d.items()}
            pending.setdefault(tid, []).append(part)
            if np.any(part["physical_failure"] | part["horizon_complete"]):
                parts = pending.pop(tid)
                data = {k: np.concatenate([p[k] for p in parts]) for k in fields}
                order = np.argsort(data["step"])
                data = {k: v[order] for k, v in data.items()}
                terminal = data["physical_failure"] | data["horizon_complete"]
                if terminal.sum() != 1 or not terminal[-1]:
                    raise ValueError("Rows after terminal transition")
                finished.add(tid)
                yield trials[tid], data
    if pending or finished != set(trials):
        raise ValueError("Incomplete trial recordings")


def stage_ids(time, cfg):
    boundaries = [cfg["move_start_s"], cfg["move_start_s"] + cfg["velocity_ramp_s"],
                  *cfg["load_times_s"][:3], cfg["stop_s"],
                  cfg["stop_s"] + cfg["velocity_ramp_s"], cfg["load_times_s"][3]]
    return np.searchsorted(boundaries, time, side="right")


def valid_rows(d):
    return (d["alive_for_metrics"].astype(bool) & d["valid"] &
            ~d["physical_failure"] & ~d["artificial_reset"])


def stage_metrics(d, meta):
    dt, K, cfg = meta["dt"], meta["prescribed_steps"], meta["evaluation"]
    stages = stage_ids(d["successor_time_seconds"], cfg)
    if not np.array_equal(stages, d["task_stage_successor"]):
        raise ValueError("Recorded task stages disagree with successor schedule")
    prescribed = stage_ids((np.arange(K) + 1) * dt, cfg)
    ee = np.linalg.norm(d["ee_position_world"] - d["ee_reference_world"], axis=-1)
    vel = np.linalg.norm(d["base_velocity_body"][:, :2] - d["command_nominal"][:, :2], axis=-1)
    yaw = np.abs(d["base_angular_velocity_body"][:, 2] - d["command_nominal"][:, 2])
    valid = valid_rows(d) & np.isfinite(ee) & np.isfinite(vel) & np.isfinite(yaw)
    passes = valid & (ee <= cfg["ee_tolerance_m"]) & (vel <= cfg["velocity_tolerance_m_s"]) & (yaw <= cfg["yaw_tolerance_rad_s"])
    result = {}
    for key, name in meta["task_stages"].items():
        sid = int(key)
        selected, den = valid & (stages == sid), int((prescribed == sid).sum())
        result[name] = dict(ee_error_m=shared.mean(ee[selected]),
                           planar_velocity_error_m_s=shared.mean(vel[selected]),
                           yaw_error_rad_s=shared.mean(yaw[selected]),
                           S_track=float((passes & (stages == sid)).sum() / den) if den else None,
                           valid_steps=int(selected.sum()), prescribed_steps=den)
    return result


def reduce_trials(rows, smoothness=False):
    metrics = {m: shared.mean([r.get(m) for r in rows]) for m in shared.MAIN + shared.SPLIT_TRACK + (SMOOTH if smoothness else ())}
    totals = {}
    for metric in shared.LIMITS:
        n = sum(r["limits"][metric]["count"] for r in rows)
        den = sum(r["limits"][metric]["denominator"] for r in rows)
        totals[metric] = dict(count=n, denominator=den)
        metrics[metric] = 100 * n / den if den else None  # Pool joint-substeps inside each seed.
    events = [e for r in rows for e in r["events"] if e["eligible"]]
    metrics.update(ee_peak_m=shared.mean([e["peak_m"] for e in events]),
                   ee_iae_m_s=shared.mean([e["iae_m_s"] for e in events]),
                   recovery_time_s=shared.mean([e["recovery_time_s"] for e in events]))
    for metric, flag in (("recovery_fraction", "recovered"), ("recovery_fraction_ee", "recovered_ee"),
                         ("recovery_fraction_velocity", "recovered_velocity")):
        metrics[metric] = sum(e[flag] for e in events) / len(events) if events else None
    counts = dict(attempted_trials=len(rows), valid_steps=sum(r["valid_steps"] for r in rows),
                  metric_trial_counts={m: sum(r.get(m) is not None for r in rows) for m in shared.MAIN + shared.SPLIT_TRACK + SMOOTH},
                  limit_totals=totals, eligible_events=len(events),
                  complete_windows=sum(e["complete_window"] for e in events),
                  recovered_events=sum(e["recovered"] for e in events),
                  recovered_ee_events=sum(e["recovered_ee"] for e in events),
                  recovered_velocity_events=sum(e["recovered_velocity"] for e in events))
    stages = {name: {metric: shared.mean([r["stages"][name][metric] for r in rows])
                     for metric in ("ee_error_m", "planar_velocity_error_m_s", "yaw_error_rad_s", "S_track")}
              for name in rows[0]["stages"]}
    counts["stages"] = {name: dict(valid_steps=sum(r["stages"][name]["valid_steps"] for r in rows),
                                   prescribed_steps=sum(r["stages"][name]["prescribed_steps"] for r in rows),
                                   metric_trial_counts={m: sum(r["stages"][name][m] is not None for r in rows)
                                                        for m in stages[name]}) for name in stages}
    return metrics, counts, stages


def summarize(config):
    output = Path(config["output_dir"]).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETE").unlink(missing_ok=True)
    runs, diagnostics, identities = [], [], set()
    loading = config.get("loading", ["loaded"])
    loading = [loading] if isinstance(loading, str) else loading
    if not set(loading) <= {"loaded", "unloaded"}:
        raise ValueError("loading must select loaded/unloaded")
    for path in shared.discover(config["input_roots"], "metadata.json"):
        meta = shared.read_json(path)
        if meta.get("experiment") != 2 or meta["task"] != config["task"]:
            continue
        if config.get("checkpoint_sha256") and meta["checkpoint_sha256"] != config["checkpoint_sha256"]:
            continue
        profile, load = meta["arguments"]["profile"], meta["arguments"]["loading"]
        if load not in loading:
            continue
        if not (path.parent / "COMPLETE").exists():
            diagnostics.append(f"Incomplete run skipped: {path.parent}")
            continue
        for name in ("records.json", "trials.json", "seed_summary.json"):
            if not (path.parent / name).is_file():
                raise FileNotFoundError(path.parent / name)
        identity = (int(meta["evaluation_seed"]), profile, load)
        if profile not in PROFILES or identity in identities or not meta.get("weights_unchanged", False):
            raise ValueError("Invalid/duplicate run or changed policy weights")
        identities.add(identity)
        runs.append((path.parent, meta))
    if not runs:
        raise ValueError("No completed matching Experiment 2 runs")
    if len({m["checkpoint_sha256"] for _, m in runs}) != 1 or len({shared.fingerprint(protocol(m)) for _, m in runs}) != 1:
        raise ValueError("Mixed checkpoints or incompatible Experiment 2 protocols")
    # Within a method, mechanics overrides must also match; counts/seed/viewer may differ.
    mechanics = []
    for _, meta in runs:
        cfg = copy.deepcopy(meta["resolved_environment"])
        cfg.pop("seed", None)
        cfg.get("env", {}).pop("num_envs", None)
        cfg.pop("viewer", None)
        mechanics.append(shared.fingerprint(cfg))
    if len(set(mechanics)) != 1:
        raise ValueError("Incompatible effective environment overrides within approach")
    edges = np.asarray(config.get("histogram_edges", np.linspace(-.5, .5, 101)), dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Histogram edges must be finite and increasing")
    smoothness = config.get("acceleration_metrics", False)
    seeds, all_trials, histograms, curves, provenance = [], [], [], [], []
    for directory, meta in runs:
        seed, profile, load = meta["evaluation_seed"], meta["arguments"]["profile"], meta["arguments"]["loading"]
        labels = dict(seed=seed, profile=profile, loading=load)
        source = shared.read_json(directory / "trials.json")
        trial_map = {int(t["trial_id"]): t for t in source}
        if len(trial_map) != len(source) or not source:
            raise ValueError("Duplicate or empty trial IDs")
        if meta["arguments"].get("episodes", len(source)) != len(source):
            raise ValueError("Completed run has an unexpected number of trials")
        K, dt = meta["prescribed_steps"], meta["dt"]
        sums, squares, counts = (np.zeros((K, len(CURVES))) for _ in range(3))
        histogram = np.zeros((3, len(edges)-1), dtype=np.int64)
        sample_counts, under, over = (np.zeros(3, dtype=np.int64) for _ in range(3))
        rows = []
        for trial, data in stream_trials(directory, meta, trial_map, smoothness):
            if [event["time"] for event in trial["events"]] != meta["evaluation"]["load_times_s"]:
                raise ValueError("Trial load events disagree with recorded protocol")
            result, error, quaternion, missing = shared.trial_metrics(data, trial, meta, smoothness)
            result["stages"] = stage_metrics(data, meta)
            result.update(labels, pair_id=trial.get("pair_id"), scenario_hash=trial.get("scenario_hash"),
                          motion_hash=trial.get("motion_hash"), profile_hash=trial.get("profile_hash"),
                          base_command_events=trial.get("base_command_events", []))
            rows.append(result)
            if missing:
                diagnostics.append(f"{directory}: trial {trial['trial_id']} missing {missing} orientations")
            error = shared.yaw_frame(error, quaternion)
            for i in range(3):
                histogram[i] += np.histogram(error[:, i], edges)[0]
                sample_counts[i] += len(error)
                under[i] += (error[:, i] < edges[0]).sum()
                over[i] += (error[:, i] > edges[-1]).sum()
            steps = data["step"].astype(int)
            if not np.allclose(data["time_seconds"], steps*dt):
                raise ValueError("Incorrect interval-start timestamps")
            force = data["ee_force_applied_world_interval_mean"]
            heading = trial["initial_task_heading_rad"]
            force_axis = force[:, 2] if profile == "carrying" else np.cos(heading)*force[:, 0] + np.sin(heading)*force[:, 1]
            values = np.column_stack((force_axis, np.linalg.norm(data["ee_position_world"]-data["ee_reference_world"], axis=-1),
                                      data["command_nominal"][:, 0], data["base_velocity_body"][:, 0]))
            good = valid_rows(data)[:, None] & np.isfinite(values)
            # No interpolation or padding failed tails. Per-step counts remain explicit.
            sums[steps] += np.where(good, values, 0.)
            squares[steps] += np.where(good, values**2, 0.)
            counts[steps] += good
        metrics, audit, stages = reduce_trials(rows, smoothness)
        seeds.append(dict(**labels, metrics=metrics, counts=audit, stages=stages))
        all_trials.extend(rows)
        online_groups = shared.read_json(directory / "seed_summary.json")
        online = online_groups.get(f"{seed}/{profile}/{load}/loco-manipulation", {})
        for metric in shared.MAIN:
            # Compare only definitions shared with the online equal-trial summary.
            value = online.get("physical_failure_rate" if metric == "failure_rate" else metric)
            if isinstance(value, (int, float)) and metrics[metric] is not None and not np.isclose(value, metrics[metric]):
                diagnostics.append(f"Online summary mismatch {directory}: {metric}")
        for j, curve in enumerate(CURVES):
            for k in range(K):
                n = int(counts[k, j])
                curves.append(dict(**labels, curve=curve, step=k, time_seconds=(k if j == 0 else k+1)*dt,
                                   count=n, attempted_trials=len(rows), sum=sums[k, j], sum_squares=squares[k, j],
                                   mean=sums[k, j]/n if n else None))
        for axis, i in zip("xyz", range(3)):
            for b in range(len(edges)-1):
                histograms.append(dict(**labels, axis=axis, bin_index=b, left=edges[b], right=edges[b+1],
                                       count=int(histogram[i, b]), valid_samples=int(sample_counts[i]),
                                       underflow=int(under[i]), overflow=int(over[i]),
                                       probability=histogram[i, b]/sample_counts[i] if sample_counts[i] else None))
        provenance.append(dict(directory=str(directory), checkpoint=meta["checkpoint"],
                               checkpoint_sha256=meta["checkpoint_sha256"], training_seed=meta.get("checkpoint_training_seed"),
                               evaluation_seed=seed, profile=profile, loading=load, git_commit=meta.get("git_commit"),
                               effective_environment=meta["resolved_environment"]))
    aggregates, stage_rows = [], []
    for load in loading:
        for profile in PROFILES:
            cohort = [s for s in seeds if s["profile"] == profile and s["loading"] == load]
            actual = sorted(s["seed"] for s in cohort)
            if set(actual) != set(config.get("expected_seeds", [0, 1, 2])) or len(actual) < 3:
                diagnostics.append(f"{profile}/{load}: evaluation seeds {actual}; expected {config.get('expected_seeds', [0, 1, 2])}")
            if not cohort:
                continue
            for metric in cohort[0]["metrics"]:
                aggregates.append(dict(profile=profile, loading=load, metric=metric, unit=shared.UNITS[metric],
                                       **shared.seed_statistics([s["metrics"][metric] for s in cohort]),
                                       attempted_trials=sum(s["counts"]["attempted_trials"] for s in cohort),
                                       contributing_trials=sum(s["counts"]["metric_trial_counts"].get(metric, 0) for s in cohort)
                                       if metric in shared.MAIN + shared.SPLIT_TRACK + SMOOTH else None))
            for stage in cohort[0]["stages"]:
                for metric in cohort[0]["stages"][stage]:
                    stage_rows.append(dict(profile=profile, loading=load, stage=stage, metric=metric,
                                           unit=shared.UNITS[metric], **shared.seed_statistics([s["stages"][stage][metric] for s in cohort])))
    p = protocol(runs[0][1])
    summary = dict(schema_version=VERSION, experiment=2, status="complete", task=config["task"],
                   display_name=config.get("display_name", config["task"]), checkpoint_sha256=runs[0][1]["checkpoint_sha256"],
                   protocol=p, protocol_sha256=shared.fingerprint(p), units=shared.UNITS,
                   histogram_edges=edges, histogram_frame="successor base-yaw frame; measured minus nominal reference; metres",
                   per_seed=seeds, aggregates=aggregates, stage_aggregates=stage_rows, trials=all_trials,
                   provenance=provenance, diagnostics=diagnostics,
                   time_series_units=dict(force="N", ee_error="m", command_vx="m/s", measured_vx="m/s", time_seconds="s"),
                   curve_semantics="Equal trial means within seed; equal seeds. Trial-SD bands use equal-seed mixture second moments; no postfailure interpolation.")
    pd.DataFrame(aggregates).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(output / "stages.csv", index=False)
    pd.DataFrame(histograms).to_csv(output / "ee_histograms.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "time_series.csv", index=False)
    shared.write_json(output / "summary.json", summary)
    (output / "COMPLETE").write_text("Experiment 2 analysis complete\n")
    return shared.clean(summary)


def aggregate_curves(frame, band="trial_sd"):
    if band not in ("trial_sd", "seed_sd"):
        raise ValueError("band must be trial_sd or seed_sd")
    rows = []
    for (curve, step, time), group in frame.groupby(["curve", "step", "time_seconds"], sort=True):
        good = group[group["count"] > 0]
        if good.seed.duplicated().any():
            raise ValueError("Duplicate seed/time curve")
        means = good["sum"] / good["count"]
        average = means.mean() if len(good) else None
        if band == "trial_sd":
            # Equal-seed mixture variance across surviving repeated trials (not SEM).
            spread = np.sqrt(max(0., (good.sum_squares/good["count"]).mean()-average**2)) if len(good) else None
        else:
            spread = means.std(ddof=1) if len(good) > 1 else None
        rows.append(dict(curve=curve, step=step, time_seconds=time, mean=average, sd=spread,
                         contributing_trials=int(good["count"].sum()), contributing_seeds=len(good),
                         attempted_trials=int(group.attempted_trials.sum())))
    return pd.DataFrame(rows)


def compare(config):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    output = Path(config["output_dir"]).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    summaries, diagnostics = {}, []
    for path in shared.discover(config["summary_roots"], "summary.json"):
        s = shared.read_json(path)
        if s.get("experiment") != 2 or s.get("status") != "complete" or not (path.parent / "COMPLETE").exists():
            continue
        if s.get("schema_version") != VERSION:
            raise ValueError("Unsupported Exp2 summary version")
        if s["task"] in summaries:
            raise ValueError(f"Duplicate approach summary: {s['task']}")
        summaries[s["task"]] = (s, path.parent)
    if not summaries:
        raise ValueError("No completed Exp2 summaries")
    values = [s for s, _ in summaries.values()]
    for field in ("protocol_sha256", "units", "histogram_edges", "histogram_frame"):
        if len({shared.fingerprint(s[field]) for s in values}) != 1:
            raise ValueError(f"Incompatible comparison: {field}")
    load = config.get("loading", "loaded")
    if load not in ("loaded", "unloaded"):
        raise ValueError("Select exactly one loading status for comparison")
    for profile in PROFILES:
        coverage = [set(r["seed"] for r in s["per_seed"] if r["profile"] == profile and r["loading"] == load) for s in values]
        available = [c for c in coverage if c]
        if available and any(c != available[0] for c in available):
            raise ValueError(f"Mismatched evaluation seed coverage: {profile}/{load}")
    order = config.get("method_order", list(summaries))
    order = list(dict.fromkeys([*order, *summaries]))
    focused = config.get("focused_methods", ["b1z1_pact", "b1z1_unifp_original", "b1z1_ppo_pos"])
    names = config.get("display_names", {})
    def name(task):
        return names.get(task, summaries[task][0]["display_name"] if task in summaries else task)
    def lookup(task, profile, metric):
        if task not in summaries:
            return None
        return next((r for r in summaries[task][0]["aggregates"] if r["profile"] == profile and r["loading"] == load and r["metric"] == metric), None)
    def table(key, tasks, metrics, wide):
        numeric, display = [], []
        for task in tasks:
            for profile in PROFILES:
                row = {"Method": name(task), "Profile": profile}
                for metric in metrics:
                    item = lookup(task, profile, metric)
                    if item is None or item["mean"] is None:
                        diagnostics.append(f"Missing {task}/{profile}/{load}/{metric}")
                    row[metric] = shared.formatted(item)
                    numeric.append(dict(task=task, method=name(task), profile=profile, loading=load, metric=metric,
                                        unit=shared.UNITS[metric], mean=item["mean"] if item else None,
                                        sd=item["sd"] if item else None,
                                        contributing_seeds=item["contributing_seeds"] if item else 0,
                                        contributing_trials=item.get("contributing_trials") if item else None,
                                        attempted_trials=item.get("attempted_trials", 0) if item else 0))
                display.append(row)
        frame = pd.DataFrame(display).set_index(["Method", "Profile"])
        if wide:
            frame = frame.unstack("Profile").swaplevel(0, 1, axis=1).reindex(columns=pd.MultiIndex.from_product([PROFILES, metrics]))
            frame = frame.reindex([name(task) for task in tasks])
        labels = {m: shared.LABELS.get(m, f"{m} [{shared.UNITS[m]}]") + (" ↑" if m.startswith("S_track") or m.startswith("recovery_fraction") else " ↓") for m in metrics}
        frame = frame.rename(columns=labels)
        pd.DataFrame(numeric).to_csv(output / f"{key}.csv", index=False)
        # Optional manuscript macros affect only LaTeX, not display/CSV identities.
        macros = config.get("latex_method_names", {})
        tokens = {name(task): f"EXP2METHOD{i}TOKEN" for i, task in enumerate(tasks) if task in macros}
        tex = frame.rename(index=tokens, level="Method").to_latex(escape=True, multicolumn_format="c")
        for task in tasks:
            if task in macros:
                tex = tex.replace(tokens[name(task)], macros[task])
        tex = tex.replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$").replace("±", r"$\pm$").replace("—", "---")
        env, width = ("table*", "textwidth") if wide else ("table", "columnwidth")
        caption = f"Experiment 2 {key.replace('_', ' ')}; {load}. Equal evaluation-seed means and sample SD; repeated evaluations of fixed checkpoints, not independently trained policies."
        (output / f"{key}.tex").write_text(f"\\begin{{{env}}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{tab:exp2_{key}}}\n\\resizebox{{\\{width}}}{{!}}{{%\n{tex}}}\n\\end{{{env}}}\n")
        return frame
    tables = {"main": table("main", order, shared.MAIN + tuple(config.get("optional_metrics", [])), True),
              "physical_limits": table("physical_limits", focused, tuple(shared.LIMITS), False),
              "response": table("response", focused, RESPONSE[:5], False)}
    font = "Times New Roman" if "Times New Roman" in {f.name for f in font_manager.fontManager.ttflist} else "DejaVu Serif"
    all_curve_rows = []
    with plt.rc_context({"font.family": font, "font.size": config.get("font_size", 9)}):
        fig, axes = plt.subplots(3, 3, figsize=config.get("figure_size", [11, 7]), sharex="col", constrained_layout=True)
        colors = config.get("colors", ["#0072B2", "#D55E00", "#009E73"])
        for col, profile in enumerate(PROFILES):
            for i, task in enumerate(focused):
                if task not in summaries:
                    continue
                s, directory = summaries[task]
                raw = pd.read_csv(directory / "time_series.csv")
                raw = raw[(raw.profile == profile) & (raw.loading == load)]
                if raw.empty:
                    continue
                curve = aggregate_curves(raw, config.get("band", "trial_sd"))
                all_curve_rows.extend(curve.assign(task=task, profile=profile, loading=load).to_dict("records"))
                color = colors[i % len(colors)]
                for j, field in enumerate(CURVES):
                    row = min(j, 2)
                    d = curve[curve.curve == field].sort_values("step")
                    x, y, sd = d.time_seconds.to_numpy(), d["mean"].to_numpy(dtype=float), d.sd.to_numpy(dtype=float)
                    label = name(task) + (" command" if j == 2 else "")
                    # Loads are interval-held, so do not draw fictitious ramps between samples.
                    axes[row, col].plot(x, y, color=color, linestyle="--" if j == 2 else "-", label=label,
                                        drawstyle="steps-post" if j == 0 else "default")
                    if j != 2:
                        axes[row, col].fill_between(x, y-sd, y+sd, color=color, alpha=.15,
                                                    step="post" if j == 0 else None)
                    if j == 1:
                        n = d.contributing_trials
                        diagnostics.append(f"{name(task)}/{profile}: surviving EE trials {n.min()}..{n.max()} (counts exported)")
                        axes[row, col].text(.02, .97-i*.12, f"{name(task)} n={n.min()}–{n.max()}", transform=axes[row, col].transAxes, va="top", color=color, fontsize=7)
            cfg = values[0]["protocol"]["evaluation"]
            for ax in axes[:, col]:
                for t in [cfg["move_start_s"], *cfg["load_times_s"], cfg["stop_s"]]:
                    ax.axvline(t, color=".5", linewidth=.6, linestyle=":")
                ax.grid(alpha=.15)
            axes[0, col].set_title(profile.title())
            axes[0, col].set_ylabel("World Fz [N]" if profile == "carrying" else "Initial-heading Fx [N]")
            axes[1, col].set_ylabel("EE error [m]")
            axes[2, col].set_ylabel("Body forward velocity [m/s]")
            axes[2, col].set_xlabel("Trial time [s]")
        handles, labels = axes[2, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
        fig.suptitle(f"{load.title()} task response; bands: {config.get('band', 'trial_sd')} (not confidence intervals)")
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"task_response.{suffix}", dpi=config.get("dpi", 200), bbox_inches="tight")
    pd.DataFrame(all_curve_rows).to_csv(output / "task_response.csv", index=False)
    shared.write_json(output / "comparison.json", dict(schema_version=VERSION, loading=load,
                      band=config.get("band", "trial_sd"), protocol=values[0]["protocol"],
                      methods=[dict(task=s["task"], checkpoint_sha256=s["checkpoint_sha256"]) for s in values], diagnostics=diagnostics))
    return tables, fig, diagnostics

