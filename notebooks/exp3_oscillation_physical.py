"""Matched, simulator-free Exp2 oscillation comparisons and evidence reporting."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import signal

import exp1_analysis as shared
import exp2_analysis as exp2
from b1z1_paper_names import DISPLAY_NAMES

REQUIRED = ("trial_id", "step", "time_seconds", "successor_time_seconds", "valid", "alive_for_metrics",
            "physical_failure", "artificial_reset", "ee_position_t_world", "ee_reference_t_world", "base_quaternion_t")
EXTRA = ("base_position_t", "base_velocity_world_t", "base_angular_velocity_world_t", "ee_velocity_t_world",
         "ee_force_applied_world_interval_mean", "joint_velocity_t", "joint_position_t", "contacts",
         "pd_weighted_motor_interval_mean", "ff_weighted_motor_interval_mean", "executed_torque_interval_mean",
         "pd_weighted_motor_mean_square", "ff_weighted_motor_mean_square", "executed_torque_mean_square",
         "opposing_channel_cancellation_mean", "nominal_kp", "nominal_kd")


def motion_signals(d):
    """Instantaneous actor-time world twist identity, then rotate to yaw task axes.

    Relative EE motion is a measured kinematic residual, not an independent J*qdot
    estimate. Assume recorded base velocity is at recorded base position; residual
    also includes any unmodelled compliance/sensor-reference error.
    """
    from exp3_film_oscillation import inverse_rotate
    needed = ("base_position_t", "base_velocity_world_t", "base_angular_velocity_world_t", "ee_velocity_t_world")
    if not all(k in d for k in needed):
        return {}, "Base/EE simultaneous velocities or base position unavailable"
    r = d["ee_position_t_world"]-d["base_position_t"]
    translation = d["base_velocity_world_t"]
    rotation = np.cross(d["base_angular_velocity_world_t"], r)
    relative = d["ee_velocity_t_world"]-translation-rotation
    values = dict(base_translation=translation, base_rotation=rotation,
                  base_induced=translation+rotation, arm_relative=relative, ee_motion=d["ee_velocity_t_world"])
    return {k:inverse_rotate(v, d["base_quaternion_t"], yaw_only=True) for k,v in values.items()}, None


def comparable_config(cfg):
    # Every method gets the SAME length, detrend, Welch bins and integrated band.
    return {**cfg, "sustained_window_seconds":cfg.get("comparison_window_seconds", 2.),
            "minimum_window_seconds":cfg.get("comparison_window_seconds", 2.),
            "nperseg":cfg.get("comparison_nperseg", 32), "segment_length_policy":"fixed",
            "min_nperseg":cfg.get("comparison_nperseg", 32), "spectral_detrend":"linear"}


def physical_window(d, meta, cfg):
    from exp3_film_oscillation import inverse_rotate, spectrum, joint_groups, relationship
    dt = meta["dt"]; time = np.arange(len(d["step"]))*dt
    err = inverse_rotate(d["ee_position_t_world"]-d["ee_reference_t_world"], d["base_quaternion_t"], True)
    traces = {"ee_error":err}; motion, reason = motion_signals(d); traces.update(motion)
    for key in EXTRA:
        if key in d and key not in ("base_quaternion_t", "nominal_kp", "nominal_kd"):
            traces[key] = np.asarray(d[key]).reshape(len(time), -1)
    groups = joint_groups(meta, cfg)
    out, spectral = [], {}
    units = {"ee_error":"m", **{k:"m/s" for k in motion}, "joint_velocity_t":"rad/s",
             "joint_position_t":"rad", "contacts":"binary"}
    def record(name, axis, metric, value, unit):
        out.append(dict(signal=name,axis=axis,metric=metric,value=float(value),unit=unit))
    wanted = ["ee_error", *motion, "joint_velocity_t", "joint_position_t", "contacts",
              "pd_weighted_motor_interval_mean", "ff_weighted_motor_interval_mean", "executed_torque_interval_mean"]
    for name in wanted:
        if name not in traces or not np.isfinite(traces[name]).all():
            continue
        x = traces[name]; detrended = signal.detrend(x, axis=0, type="linear")
        sp = spectrum(x, dt, cfg); spectral[name] = sp
        labels = list("xyz") if name == "ee_error" or name in motion else meta["dof_names"][:x.shape[1]] if name != "contacts" else [f"foot_{i}" for i in range(x.shape[1])]
        slope = np.linalg.lstsq(np.column_stack((time, np.ones(len(time)))), x, rcond=None)[0][0]
        for i,label in enumerate(labels):
            unit=units.get(name,"Nm")
            record(name,label,"bias",x[:,i].mean(),unit)
            record(name,label,"drift_per_s",slope[i],unit+"/s")
            record(name,label,"oscillatory_rms",np.sqrt(np.mean(detrended[:,i]**2)),unit)
            record(name,label,"peak_to_peak",np.ptp(detrended[:,i]),unit)
            if sp is not None:
                record(name,label,"band_power",sp["power"][i],unit+"^2")
                record(name,label,"dominant_frequency",sp["dominant"][i],"Hz")
        if name.endswith("torque_interval_mean") or name.startswith(("pd_", "ff_")):
            for group,ids in groups.items():
                ids=[i for i in ids if i<x.shape[1]]
                if ids:
                    record(name,group,"oscillatory_rms",np.sqrt(np.mean(detrended[:,ids]**2)),"Nm")
    for group,ids in groups.items():
        if not ids:
            continue
        for name in ("pd_weighted_motor", "ff_weighted_motor", "executed_torque"):
            key=name+"_mean_square"
            if key in d:
                record(name,group,"substep_mean_square",np.mean(d[key][:,ids]),"Nm^2")
        if "opposing_channel_cancellation_mean" in d:
            record("channel_opposition",group,"mean",np.mean(d["opposing_channel_cancellation_mean"][:,ids]),"Nm")
    if motion:
        b=signal.detrend(motion["base_induced"],axis=0); a=signal.detrend(motion["arm_relative"],axis=0)
        for i,axis in enumerate("xyz"):
            # Negative cross-power is consistent with attenuation, not proof of control causality.
            record("base_arm",axis,"time_cross_term",2*np.mean(b[:,i]*a[:,i]),"m^2/s^2")
            if all(spectral.get(k) is not None for k in ("base_induced","arm_relative","ee_motion")):
                cross=spectral["ee_motion"]["power"][i]-spectral["base_induced"]["power"][i]-spectral["arm_relative"]["power"][i]
                record("base_arm",axis,"band_cross_term",cross,"m^2/s^2")
                rel=relationship(b[:,i],a[:,i],dt,cfg)
                peak=spectral["ee_error"]["dominant"][i]
                if rel is not None and np.isfinite(peak):
                    j=np.argmin(abs(rel["frequency"]-peak))
                    record("base_arm",axis,"coherence_at_error_peak",rel["coherence"][j],"fraction")
                    record("base_arm",axis,"phase_at_error_peak",rel["phase"][j] if rel["coherence"][j]>=cfg["coherence_threshold"] else np.nan,"rad")
    # Competing gait/channel explanations use signed components, never spectra of norms.
    for name in ("contacts", "base_induced", "arm_relative", "pd_weighted_motor_interval_mean", "ff_weighted_motor_interval_mean"):
        if name not in traces: continue
        for axis,i in zip("xyz",range(3)):
            sp=spectral.get("ee_error")
            if sp is None or not np.isfinite(sp["dominant"][i]): continue
            for j in range(traces[name].shape[1]):
                rel=relationship(traces[name][:,j],err[:,i],dt,cfg)
                if rel is None: continue
                k=np.argmin(abs(rel["frequency"]-sp["dominant"][i]))
                label=f"{j}_to_{axis}"
                record(name+"_error_link",label,"coherence_at_error_peak",rel["coherence"][k],"fraction")
                record(name+"_error_link",label,"lag_seconds",rel["lag_seconds"],"s")
                record(name+"_error_link",label,"phase_at_error_peak",rel["phase"][k] if rel["coherence"][k]>=cfg["coherence_threshold"] else np.nan,"rad")
    return out,traces,spectral,reason


def hierarchical(frame, keys):
    """Equal windows within trial, trials within eval seed; keep checkpoints separate."""
    trials=frame.groupby(keys+["checkpoint","evaluation_seed","trial_id"],dropna=False).value.mean().reset_index()
    seeds=trials.groupby(keys+["checkpoint","evaluation_seed"],dropna=False).value.agg(["mean","count"]).reset_index()
    summary=seeds.groupby(keys+["checkpoint"],dropna=False)["mean"].agg(["mean","std","count"]).reset_index()
    counts=frame.groupby(keys+["checkpoint"],dropna=False).size().rename("windows").reset_index()
    trial_counts=seeds.groupby(keys+["checkpoint"],dropna=False)["count"].sum().rename("trial_seed_pairs").reset_index()
    summary=summary.merge(counts,on=keys+["checkpoint"]).merge(trial_counts,on=keys+["checkpoint"])
    summary["variability"]="evaluation-seed SD within checkpoint, not independently trained-policy SD"
    return trials,seeds,summary


def analyze(cfg, output):
    from exp3_film_oscillation import window_specs, stream_windows, contiguous_segments, clean, select_plot_windows
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    methods=cfg.get("comparison_tasks",["b1z1_pact","b1z1_unifp_original","b1z1_ppo_pos"])
    roots=cfg.get("comparison_recording_roots") or cfg["recording_roots"]
    runs,diagnostics,seen=[],[],set(); provenance=[]
    for path in shared.discover(roots,"metadata.json"):
        if not (path.parent/"COMPLETE").exists(): continue
        m=shared.read_json(path); args=m.get("arguments",{})
        if m.get("experiment")!=2 or m["task"] not in methods or args.get("profile") not in cfg["profiles"] or args.get("loading") not in cfg["loading"] or m["evaluation_seed"] not in cfg["evaluation_seeds"]: continue
        if m["resolved_training"]["policy"].get("film_v02_enabled",False):
            diagnostics.append(f"Excluded latent-FiLM/V02: {path.parent}"); continue
        selected=cfg.get("comparison_checkpoints",{}).get(m["task"])
        if selected and m["checkpoint_sha256"] not in selected: continue
        records=shared.read_json(path.parent/"records.json")
        missing=set(REQUIRED)-set(records["fields"])
        if missing:
            diagnostics.append(f"{path.parent}: missing physical fields {sorted(missing)}"); continue
        key=(m["task"],m["checkpoint_sha256"],m["evaluation_seed"],args["profile"],args["loading"])
        if key in seen: raise ValueError("Duplicate physical comparison identity")
        seen.add(key)
        trials=shared.read_json(path.parent/"trials.json")
        chosen=[t for t in trials if cfg.get("trial_ids") is None or int(t["trial_id"]) in cfg["trial_ids"]]
        run=dict(path=path.parent,meta=m,records=records,trial_ids={int(t["trial_id"]) for t in trials})
        runs.append(run)
        provenance.append(dict(task=m["task"],checkpoint=m["checkpoint_sha256"],training_seed=m.get("checkpoint_training_seed"),
            evaluation_seed=m["evaluation_seed"],profile=args["profile"],loading=args["loading"],
            attempted_trials=len(trials),failures=sum(bool(t.get("physical_failure",t.get("failed",False))) for t in trials),
            selected_trials=len(chosen),selected_failures=sum(bool(t.get("physical_failure",t.get("failed",False))) for t in chosen),
            control=m["resolved_environment"]["control"],protocol_hash=shared.fingerprint(exp2.protocol(m)),path=str(path.parent)))
    for task in methods:
        if not any(r["meta"]["task"]==task for r in runs): diagnostics.append(f"Comparative conclusions unavailable for missing method {task}")
    if not any(r["meta"]["arguments"]["loading"]=="unloaded" for r in runs):
        diagnostics.append("No matched unloaded recordings: pre-load vs loaded also changes trajectory/time, not a controlled load comparison")
    cc=comparable_config(cfg); rows=[]; windows=[]; missing=set()
    for run in runs:
        m=run["meta"]; print("Physical oscillation:",run["path"],flush=True)
        protocol=shared.fingerprint(exp2.protocol(m))
        local={**cc,"_fields":(*REQUIRED,*EXTRA),"_window_specs":[s for s in window_specs(m,cc) if s["kind"]=="sustained"]}
        n=round(cc["sustained_window_seconds"]/m["dt"])
        for tid,spec,raw in stream_windows(run,local):
            for segment in contiguous_segments(raw,m["dt"],REQUIRED):
                if len(segment)!=n:
                    missing.add((str(run["path"]),spec["stage"],"Incomplete/failure/gap window excluded from matched spectra")); continue
                d={k:np.stack([r[k] for r in segment]) for k in segment[0]}
                values,traces,spectra,reason=physical_window(d,m,cc)
                if reason: missing.add((str(run["path"]),spec["stage"],reason))
                if spectra.get("ee_error") is None:
                    missing.add((str(run["path"]),spec["stage"],"Fixed comparative Welch settings unsupported")); continue
                plan=spectra["ee_error"]["plan"]
                spectral_contract=shared.fingerprint({k:plan[k] for k in ("nperseg","resolution_hz","usable_low_hz","usable_high_hz")})
                identity=dict(task=m["task"],checkpoint=m["checkpoint_sha256"],training_seed=m.get("checkpoint_training_seed"),
                    evaluation_seed=m["evaluation_seed"],profile=m["arguments"]["profile"],loading=m["arguments"]["loading"],
                    trial_id=tid,protocol_hash=protocol,spectral_contract=spectral_contract,**spec,window_id=len(windows),run=str(run["path"]),
                    actual_start=float(d["time_seconds"][0]),actual_end=float(d["successor_time_seconds"][-1]))
                rows.extend({**identity,**v} for v in values)
                from exp3_film_oscillation import joint_groups
                windows.append(dict(**identity,oscillation_severity=float(np.sqrt(np.mean(signal.detrend(traces["ee_error"],axis=0)**2))),spectral_plan=plan,joint_groups=joint_groups(m,cfg)))
                arrays={"time_seconds":d["time_seconds"],**traces}
                np.savez_compressed(output/f"physical-{identity['window_id']:05d}.npz",**arrays)
                if cfg.get("comparison_max_windows") and len(windows)>=cfg["comparison_max_windows"]: break
            if cfg.get("comparison_max_windows") and len(windows)>=cfg["comparison_max_windows"]: break
        if cfg.get("comparison_max_windows") and len(windows)>=cfg["comparison_max_windows"]: break
    diagnostics.extend(f"{run}: stage {stage}: {why}" for run,stage,why in sorted(missing))
    frame=pd.DataFrame(rows); summary=[]; matched=[]
    if len(frame):
        keys=["task","profile","loading","stage","motion_phase","scheduled_load_n","protocol_hash","spectral_contract","signal","axis","metric","unit"]
        tables=hierarchical(frame,keys)
        for name,table in zip(("physical_trials","physical_seeds","physical_summary"),tables): table.to_csv(output/(name+".csv"),index=False)
        summary=tables[-1].to_dict("records")
        # Match seeds and scheduled windows, not statistics from incompatible protocols/bands.
        match=["protocol_hash","spectral_contract","profile","loading","stage","start","end","evaluation_seed"]
        availability=frame.groupby(match).task.nunique()
        good=availability[availability==len(methods)].index
        common=frame.set_index(match).loc[lambda x:x.index.isin(good)].reset_index()
        if len(common):
            _,_,table=hierarchical(common,keys); matched=table.to_dict("records")
            table.to_csv(output/"matched_comparison.csv",index=False)
        else: diagnostics.append("No complete cross-method protocol/seed/stage coverage; method-specific results retained")
    selected,omitted=select_plot_windows(windows,cfg.get("comparison_plot_windows",6))
    plot_physical(selected,output,cfg)
    result=dict(provenance=provenance,summary=summary,matched=matched,windows=windows,plot_selection=selected,
                omitted_categories=omitted,diagnostics=diagnostics,config=cc,
                assumptions="Actor-time yaw-aligned vectors; linear detrend separates drift. Fixed window length/FFT/band across methods. Relative arm velocity is measured endpoint residual at the recorded base reference point, not independently validated J*qdot. Channel interval fluctuations are not substep RMS; latter retained as mean-square statistics. Gains differ intentionally and are not a causal explanation.")
    (output/"physical_summary.json").write_text(json.dumps(clean(result),indent=2,allow_nan=False))
    return result


def plot_physical(selected,output,cfg):
    import matplotlib.pyplot as plt
    for w in selected:
        with np.load(output/f"physical-{w['window_id']:05d}.npz") as d:
            fig,axes=plt.subplots(3,3,figsize=(12,8),sharex=True)
            t=d["time_seconds"]
            for i,axis in enumerate("xyz"):
                axes[0,i].plot(t,d["ee_error"][:,i],label=axis)
                axes[0,i].set_title("EE "+axis); axes[0,i].set_ylabel("Tracking error [m]")
                for name in ("base_translation","base_rotation","arm_relative","ee_motion"):
                    if name in d: axes[1,i].plot(t,d[name][:,i],label=name.replace('_',' '))
                axes[1,i].set_ylabel("Velocity [m/s]")
            for col,group in enumerate(("arm","leg")):
                for name,label in (("pd_weighted_motor_interval_mean","weighted PD request"),("ff_weighted_motor_interval_mean","weighted FF request"),("executed_torque_interval_mean","clipped execution")):
                    if name in d:
                        ids=w["joint_groups"].get(group,[])
                        ids=[i for i in ids if i<d[name].shape[1]]
                        if ids:
                            axes[2,col].plot(t,np.sqrt(np.mean(d[name][:,ids]**2,axis=1)),label=label)
                axes[2,col].set_title(group+" channels"); axes[2,col].set_ylabel("Joint RMS of interval mean [Nm]")
            if "contacts" in d:
                for i in range(d["contacts"].shape[1]): axes[2,2].plot(t,d["contacts"][:,i],label=f"foot {i}")
            axes[2,2].set_title("Contact timing"); axes[2,2].set_ylabel("Recorded contact state")
            for ax in axes.flat:
                ax.grid(alpha=.2)
                if ax.lines: ax.legend()
            for ax in axes[-1]: ax.set_xlabel("Actor-time [s]")
            fig.suptitle(f"{DISPLAY_NAMES.get(w['task'],w['task'])}: {w['profile']} {w['loading']} stage {w['stage']}\n{w['checkpoint'][:8]} / seed {w['evaluation_seed']} / trial {w['trial_id']} / {w['severity_group']}")
            shared.style_figure(fig,cfg); fig.tight_layout()
            for ext in ("pdf","png"): fig.savefig(output/f"physical-{w['window_id']:05d}.{ext}")
            plt.close(fig)
