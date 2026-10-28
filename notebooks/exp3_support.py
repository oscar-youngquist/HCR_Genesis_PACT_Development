"""CPU-only recorded action/support analysis; no simulator or policy imports."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import exp1_analysis as shared
import exp2_analysis as exp2
from b1z1_paper_names import DISPLAY_NAMES

TASKS = ("b1z1_pact", "b1z1_unifp_original", "b1z1_ppo_pos",
         "b1z1_pact_ab6_position_film", "b1z1_pact_ab8_coupled_film", "b1z1_pact_ab9_representation")


def passive_bias(row):
    """Joint bias [Nm] at t; this is NOT the already executed explicit PD torque."""
    qd = np.asarray(row["joint_velocity_t"], dtype=float)
    damping = np.broadcast_to(np.asarray(row.get("mechanics_joint_damping", 0)), qd.shape)
    if not np.isfinite(damping).all() or np.any(damping < 0):
        raise ValueError("Invalid recorded passive damping")
    assumptions = ["Frozen-state Euler with interval-average executed torque/loads; passive drive target velocity assumed zero."]
    if np.any(np.asarray(row.get("mechanics_joint_friction", 0)) != 0):
        assumptions.append("Recorded joint friction coefficient retained in provenance; unknown constraint-dependent friction torque omitted, NOT coefficient*sign(velocity).")
    if np.any(np.asarray(row.get("mechanics_joint_stiffness", 0)) != 0):
        assumptions.append("Passive drive stiffness torque omitted: simulator drive target not recorded.")
    return damping * qd, assumptions


def hierarchy(frame, keys):
    """Equal trials -> evaluation seeds -> checkpoints; never timestep SD."""
    if frame.empty:
        return pd.DataFrame()
    frame = frame[np.isfinite(frame.value)].copy()
    trial_keys = keys + ["checkpoint", "seed", "trial"]
    trials = frame.groupby(trial_keys, dropna=False).value.mean().reset_index()
    seeds = trials.groupby(keys + ["checkpoint", "seed"], dropna=False).value.mean().reset_index()
    out = []
    for group_key, group in seeds.groupby(keys, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        ckpt = group.groupby("checkpoint").value.mean()
        vals = ckpt if len(ckpt) > 1 else group.value
        # A checkpoint hash is not evidence of an independent training seed.
        training = frame[frame.checkpoint.isin(ckpt.index)]
        known = training.groupby("checkpoint").training_seed.first() if "training_seed" in training else pd.Series(dtype=float)
        independent = len(known) == len(ckpt) and known.notna().all() and known.nunique() == len(ckpt)
        variability = ("training_seed_sd" if independent else "checkpoint_sd_independence_unverified") if len(ckpt) > 1 else "evaluation_seed_sd"
        stats = shared.seed_statistics(vals)
        stats["contributing_seeds"] = len(group)  # Evaluation repetitions, not checkpoint count.
        out.append(dict(zip(keys, group_key), **stats, variability_replicates=len(vals),
                        checkpoints=len(ckpt), evaluation_repetitions=len(group),
                        variability=variability))
    return pd.DataFrame(out)


def axis_rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)


class RecordedFK:
    """URDF tree FK in metres, using recorded joint names, not assumed arm indices."""
    def __init__(self, filename, base, ee):
        root = ET.parse(filename).getroot()
        joints = {j.find("child").get("link"): j for j in root.findall("joint")}
        links = {link.get("name") for link in root.findall("link")}
        def chain(link):
            result = []
            if links and link not in links:
                raise ValueError(f"Unknown URDF link: {link}")
            while link in joints:
                if len(result) > len(joints):
                    raise ValueError("Cyclic URDF chain")
                joint = joints[link]
                result.append(joint)
                link = joint.find("parent").get("link")
            return link, result[::-1]
        ee_root, self.chain = chain(ee)
        base_root, self.base_chain = chain(base)
        if ee_root != base_root:
            raise ValueError("Disconnected base/EE URDF trees")

    def __call__(self, q, names):
        rotation, position = self.transform(q, names, self.chain)
        base_rotation, base_position = self.transform(q, names, self.base_chain)
        # B1's trunk and arm mounting link can be fixed siblings of the URDF root.
        return base_rotation.T @ (position-base_position)

    def transform(self, q, names, chain):
        positions = dict(zip(names, q))
        rotation, position = np.eye(3), np.zeros(3)
        for joint in chain:
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            position += rotation @ xyz
            rotation = rotation @ axis_rotation([0, 0, 1], rpy[2]) @ axis_rotation([0, 1, 0], rpy[1]) @ axis_rotation([1, 0, 0], rpy[0])
            if joint.get("type") != "fixed":
                name = joint.get("name")
                if name not in positions:
                    raise ValueError(f"Missing commanded joint {name}")
                axis = joint.find("axis")
                axis = np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" ")
                if joint.get("type") == "prismatic":
                    position += rotation @ (axis * positions[name])
                else:
                    rotation = rotation @ axis_rotation(axis, positions[name])
        return rotation, position


def channel_values(row, names):
    """Return joint-mean squared usage; square root only AFTER trial averaging."""
    values = {}
    for region in ("arm", "leg"):
        ids = [i for i, name in enumerate(names) if name.startswith("z1_") == (region == "arm")]
        if not ids:
            continue
        for channel in ("pd_weighted_motor", "ff_weighted_motor", "executed_torque"):
            key = channel + "_mean_square"
            if key in row:
                values[region + "_" + channel + "_ms"] = float(np.mean(np.asarray(row[key])[ids]))
            mean_key = channel + "_interval_mean"
            if mean_key in row:
                values[region + "_" + channel + "_mean_nm"] = float(np.mean(np.asarray(row[mean_key])[ids]))
        if "opposing_channel_cancellation_mean" in row:
            values[region + "_cancellation_nm"] = float(np.mean(np.asarray(row["opposing_channel_cancellation_mean"])[ids]))
    return values


def support_values(row, feet):
    """Use recorded mean cross-products about simultaneous whole-body CoM."""
    values = {}
    if "grf_raw_world_interval_mean" in row:
        grf = np.asarray(row["grf_raw_world_interval_mean"])
        for i, foot in enumerate(feet):
            values[f"foot_{foot}_fz_n"] = grf[i, 2]
        values.update({f"foot_support_f{a}_n": v for a, v in zip("xyz", grf.sum(0))})
    if "body_contact_forces_raw_world_interval_mean" in row:
        values.update({f"all_support_f{a}_n": v for a, v in zip("xyz", np.asarray(row["body_contact_forces_raw_world_interval_mean"]).sum(0))})
    for field, label in (("support_moment_world", "foot_support"),
                         ("all_contact_moment_world_interval_mean", "all_support")):
        if field in row:
            values.update({f"{label}_m{a}_nm": v for a, v in zip("xyz", row[field])})
    return values


def analyze(config):
    """Stream numeric chunks, retain compact per-seed sufficient statistics only."""
    from exp3_representation import chunks, file_hash, rotate
    output = Path(config["output_dir"]) / "action_support"
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETE").unlink(missing_ok=True)
    diagnostics, provenance, scalar_rows, curve_acc = [], [], [], {}
    identities, protocols, gains = set(), set(), {}
    selected = config.get("exp2_tasks", TASKS)
    for path in shared.discover(config.get("exp2_roots", []), "metadata.json"):
        if not (path.parent / "COMPLETE").is_file():
            continue
        meta = shared.read_json(path)
        if meta.get("experiment") != 2 or meta["task"] not in selected:
            continue
        seed, args = meta["evaluation_seed"], meta["arguments"]
        if seed not in config["evaluation_seeds"] or args["loading"] not in config.get("exp2_loading", ["loaded"]):
            continue
        task, checkpoint, profile, loading = meta["task"], meta["checkpoint_sha256"], args["profile"], args["loading"]
        identity = (task, checkpoint, seed, profile, loading)
        if identity in identities:
            raise ValueError(f"Duplicate Exp2 cohort: {identity}")
        identities.add(identity)
        protocols.add(shared.fingerprint(exp2.protocol(meta)))
        if len(protocols) > 1:
            raise ValueError("Incompatible Exp2 protocols; narrow exp2_roots")
        env = meta["resolved_environment"]
        gains[task] = {k: env["control"].get(k) for k in ("stiffness", "damping")}
        trials = {r["trial_id"]: r for r in shared.read_json(path.parent / "trials.json")}
        records = shared.read_json(path.parent / "records.json")
        fields = [k for k in records["fields"] if k in (
            "trial_id", "step", "valid", "alive_for_metrics", "physical_failure", "artificial_reset",
            "time_seconds", "successor_time_seconds", "task_stage_interval", "task_stage_successor",
            "ee_force_applied_world_interval_mean", "ee_position_world", "ee_reference_world",
            "ee_reference_t_world", "base_position_t", "base_quaternion_t", "position_target_interval_mean",
            "base_velocity_body", "command_nominal", "grf_raw_world_interval_mean", "support_moment_world",
            "all_contact_moment_world_interval_mean", "body_contact_forces_raw_world_interval_mean",
            "opposing_channel_cancellation_mean", "pd_weighted_motor_interval_mean",
            "ff_weighted_motor_interval_mean", "executed_torque_interval_mean",
            "torque_request_motor_scaled_interval_mean") or k.endswith("_mean_square")]
        expected = {"position_target_interval_mean", "support_moment_world", "all_contact_moment_world_interval_mean",
                    "pd_weighted_motor_mean_square", "ff_weighted_motor_mean_square", "executed_torque_mean_square",
                    "opposing_channel_cancellation_mean"}
        diagnostics.extend(f"{path.parent}: unavailable {key}" for key in sorted(expected-set(fields)))
        fk = None
        try:
            model_path = config.get("urdf") or meta["urdf_source"]
            if file_hash(model_path) != meta["urdf_sha256"]:
                raise ValueError("URDF hash mismatch")
            fk = RecordedFK(model_path, env["asset"]["base_name"], env["asset"]["gripper_name"])
        except (KeyError, OSError, ValueError) as exc:
            diagnostics.append(f"{task} commanded FK unavailable: {exc}")
        run = dict(path=path.parent, records=records)
        trial_acc, last_step = {}, {}
        for _, data in chunks(run, fields):
            for i in range(len(data["trial_id"])):
                row = {k: v[i] for k, v in data.items()}
                tid, step = int(row["trial_id"]), int(row["step"])
                if tid not in trials or step <= last_step.get(tid, -1):
                    raise ValueError("Unknown/duplicate/out-of-order transition")
                last_step[tid] = step
                if not (row["valid"] and row["alive_for_metrics"] and not row["physical_failure"] and not row["artificial_reset"]):
                    continue  # No failed tails, interpolation or zero padding.
                accounting = ("pd_weighted_motor_interval_mean", "ff_weighted_motor_interval_mean", "torque_request_motor_scaled_interval_mean")
                if all(k in row for k in accounting) and not np.allclose(row[accounting[0]]+row[accounting[1]], row[accounting[2]], atol=1e-4, rtol=1e-5):
                    raise ValueError(f"Recorded requested PD+FF accounting mismatch: {path.parent}/{tid}/{step}")
                values = channel_values(row, meta["dof_names"])
                values.update(support_values(row, env["asset"]["foot_name"]))
                values["ee_error_m"] = np.linalg.norm(row["ee_position_world"]-row["ee_reference_world"])
                values["velocity_error_m_s"] = np.linalg.norm(row["base_velocity_body"][:2]-row["command_nominal"][:2])
                force = row.get("ee_force_applied_world_interval_mean")
                if force is not None:
                    heading = trials[tid]["initial_task_heading_rad"]
                    values["load_n"] = force[2] if profile == "carrying" else np.cos(heading)*force[0]+np.sin(heading)*force[1]
                if fk and all(k in row for k in ("position_target_interval_mean", "base_position_t", "base_quaternion_t", "ee_reference_t_world")):
                    # FK(mean delayed command) at measured base_t is NOT mean substep FK.
                    command = fk(row["position_target_interval_mean"], meta["dof_names"])
                    world = row["base_position_t"] + rotate(command, row["base_quaternion_t"])
                    values["command_geometry_error_m"] = np.linalg.norm(world-row["ee_reference_t_world"])
                for metric, value in values.items():
                    if not np.isfinite(value):
                        diagnostics.append(f"Nonfinite {task}/{tid}/{step}/{metric}")
                        continue
                    successor = metric in ("ee_error_m", "velocity_error_m_s")
                    time = float(row["successor_time_seconds"] if successor else row["time_seconds"])
                    stage = int(row["task_stage_successor"] if successor else row["task_stage_interval"])
                    key = (*identity, metric, step, time)
                    acc = curve_acc.setdefault(key, [0., 0., 0])
                    acc[0] += value; acc[1] += value*value; acc[2] += 1
                    key = (tid, stage, metric)
                    acc = trial_acc.setdefault(key, [0., 0])
                    acc[0] += value; acc[1] += 1
        for (tid, stage, metric), (total, count) in trial_acc.items():
            value = total/count
            if metric.endswith("_ms"):
                value = np.sqrt(value)  # RMS uses recorded substep squares, not interval means.
                metric = metric[:-3]+"_rms_nm"
            scalar_rows.append(dict(task=task, checkpoint=checkpoint, seed=seed, trial=tid,
                training_seed=meta.get("checkpoint_training_seed"), profile=profile, loading=loading,
                stage=stage, metric=metric, value=value, count=count))
        provenance.append(dict(path=str(path.parent), task=task, checkpoint=checkpoint,
            training_seed=meta.get("checkpoint_training_seed"), evaluation_seed=seed,
            profile=profile, loading=loading, evaluation=meta["evaluation"], dt=meta["dt"],
            control=env["control"], domain_randomization=env.get("domain_rand"), urdf=meta.get("urdf_sha256")))
    seen = {r["task"] for r in provenance}
    for task in seen:
        hashes = {r["checkpoint"] for r in provenance if r["task"] == task}
        if len(hashes) > 1:
            diagnostics.append(f"{task}: multiple checkpoint hashes; independence of training runs is not established by hashes alone")
    diagnostics.extend(f"Missing Exp2 method: {task}" for task in selected if task not in seen)
    for task in seen:
        for profile in exp2.PROFILES:
            for loading in config.get("exp2_loading", ["loaded"]):
                seeds = {r["evaluation_seed"] for r in provenance if r["task"] == task and r["profile"] == profile and r["loading"] == loading}
                if seeds != set(config["evaluation_seeds"]):
                    diagnostics.append(f"{task}/{profile}/{loading}: evaluation seeds {sorted(seeds)}, expected {config['evaluation_seeds']}")
    pairs = [(TASKS[3], TASKS[4]), (TASKS[4], TASKS[0]), (TASKS[5], TASKS[0])]
    comparisons = [dict(left=a, right=b, available=a in seen and b in seen,
        gains_match=gains.get(a) == gains.get(b) if a in gains and b in gains else None) for a, b in pairs]
    if comparisons[0]["available"] and not comparisons[0]["gains_match"]:
        diagnostics.append("A1 versus A3 gains differ: NOT a matched-gain attribution comparison")
    scalar = pd.DataFrame(scalar_rows)
    summary = hierarchy(scalar, ["task", "profile", "loading", "stage", "metric"])
    curve_rows = []
    training_seeds = {p["checkpoint"]: p["training_seed"] for p in provenance}
    for key, (total, square, count) in curve_acc.items():
        task, checkpoint, seed, profile, loading, metric, step, time = key
        curve_rows.append(dict(task=task, checkpoint=checkpoint, seed=seed, profile=profile, loading=loading,
            training_seed=training_seeds[checkpoint],
            metric=metric, step=step, time_seconds=time, value=total/count, sum=total, sum_squares=square,
            count=count, trial=0))  # Each curve value is already a within-seed survivor mean.
    curves = pd.DataFrame(curve_rows)
    curve_summary = hierarchy(curves, ["task", "profile", "loading", "metric", "step", "time_seconds"])
    if not curves.empty:
        counts = curves.groupby(["task", "profile", "loading", "metric", "step", "time_seconds"])["count"].sum().reset_index(name="surviving_trials")
        curve_summary = curve_summary.merge(counts)
    scalar.to_csv(output/"trial_stages.csv", index=False)
    summary.to_csv(output/"summary.csv", index=False)
    curves.to_csv(output/"per_seed_curves.csv", index=False)
    curve_summary.to_csv(output/"curves.csv", index=False)
    result = dict(schema_version=1, experiment=3, section="action_support", status="complete", provenance=provenance,
        diagnostics=sorted(set(diagnostics)), comparisons=comparisons,
        configuration=config, units=dict(load="N", support_force="N", support_moment="Nm", torque_mean="Nm",
            torque_ms="Nm^2", torque_rms="Nm", cancellation="Nm", geometry="m", tracking="m", velocity="m/s", time="s"),
        summary=summary.to_dict("records"), conventions={
            "timing": "Loads/channels/support: interval start. Tracking: successor. No failure interpolation.",
            "geometry": "URDF FK(mean executed position target) at measured base_t versus nominal reference_t; approximate under substep delay/base motion.",
            "support": "World N and Nm about whole-robot CoM; recorded mean cross products, point-force approximation, not contact wrenches.",
            "channels": "Weighted motor-scaled requested PD/FF; executed clipped torque separately. RMS from substep squares; cancellation from recorded absolute substep values.",
            "bands": "Sample SD across checkpoint means if multiple checkpoints (training independence NOT inferred from hashes); otherwise evaluation-seed SD. Survivor counts exported/plotted.",
            "settling": "Not reported without an explicitly justified steady support reference; locomotion support is time-varying."})
    plot_responses(curve_summary, provenance, output, config)
    shared.write_json(output/"summary.json", result)
    (output/"COMPLETE").write_text("Offline action/support analysis complete\n")
    return shared.clean(result)


def plot_responses(frame, provenance, output, config):
    if frame.empty:
        return
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    main = TASKS[:3]
    channel = ("load_n", "command_geometry_error_m", "ee_error_m", "arm_pd_weighted_motor_mean_nm",
               "arm_ff_weighted_motor_mean_nm", "arm_executed_torque_mean_nm", "leg_pd_weighted_motor_mean_nm",
               "leg_ff_weighted_motor_mean_nm", "leg_executed_torque_mean_nm", "arm_pd_weighted_motor_ms",
               "arm_ff_weighted_motor_ms", "leg_pd_weighted_motor_ms", "leg_ff_weighted_motor_ms")
    support = ("load_n", "all_support_fx_n", "all_support_fy_n", "all_support_fz_n",
               "all_support_mx_nm", "all_support_my_nm", "all_support_mz_nm", "ee_error_m", "velocity_error_m_s")
    feet = sorted(set(frame.metric[frame.metric.str.startswith("foot_") & frame.metric.str.endswith("_fz_n")]) - {"foot_support_fz_n"})
    support += tuple(feet)
    groups = [("channels", main, channel), ("support", main, support),
              ("support_A6_PILOT", (TASKS[5], TASKS[0]), support)]
    groups += [(f"channels_{DISPLAY_NAMES[a]}_{DISPLAY_NAMES[b]}", (a, b), channel)
               for a, b in ((TASKS[3], TASKS[4]), (TASKS[4], TASKS[0]), (TASKS[5], TASKS[0]))]
    for loading in sorted(frame.loading.unique()):
        for name, methods, metrics in groups:
            metrics = [m for m in metrics if m in set(frame.metric)]
            if not metrics or not set(methods) & set(frame.task):
                continue
            fig, axes = plt.subplots(len(metrics)+1, 3, squeeze=False,
                figsize=(config.get("figure_width", 12), 1.65*(len(metrics)+1)), sharex="col", sharey="row")
            for col, profile in enumerate(exp2.PROFILES):
                for j, task in enumerate(methods):
                    color = dict(zip(TASKS, ("#2277AA", "#CC6677", "#228833", "#AA3377", "#EE7733", "#666666")))[task]
                    for row, metric in enumerate(metrics):
                        d = frame[(frame.task == task) & (frame.profile == profile) & (frame.loading == loading) & (frame.metric == metric)].sort_values("time_seconds")
                        if d.empty:
                            continue
                        # Missing/invalid time points break lines; never bridge a gap.
                        dt = next(p["dt"] for p in provenance if p["task"] == task and p["profile"] == profile and p["loading"] == loading)
                        offset = float(d.time_seconds.iloc[0] - d.step.iloc[0]*dt)
                        d = d.set_index("step").reindex(range(int(d.step.min()), int(d.step.max())+1))
                        d["time_seconds"] = d.index*dt + offset
                        # MS curves remain (Nm)^2; do not average per-step RMS as trial RMS.
                        ax = axes[row, col]
                        x, y, sd = d.time_seconds.to_numpy(float), d["mean"].to_numpy(float), d.sd.to_numpy(dtype=float, na_value=np.nan)
                        ax.plot(x, y, color=color, label=DISPLAY_NAMES[task])
                        ax.fill_between(x, y-sd, y+sd, color=color, alpha=.15)
                        if metric == "ee_error_m":
                            axes[-1, col].plot(x, d.surviving_trials, color=color)
                schedules = [p["evaluation"] for p in provenance if p["profile"] == profile and p["loading"] == loading]
                if schedules:
                    c = schedules[0]
                    for ax in axes[:, col]:
                        for t in [c["move_start_s"], *c["load_times_s"], c["stop_s"]]:
                            ax.axvline(t, color="0.6", lw=.5, ls="--")
                axes[0, col].set_title(profile.title())
                axes[-1, col].set_xlabel("Sequence time [s]; dashed event boundaries")
            for row, metric in enumerate(metrics):
                axes[row, 0].set_ylabel(metric.replace("_", " "), fontsize=8)
            axes[-1, 0].set_ylabel("Surviving trials")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=len(labels))
            fig.tight_layout(rect=(0, 0, 1, .97))
            for suffix in ("pdf", "png"):
                fig.savefig(output/f"{name}_{loading}.{suffix}", dpi=160)
            plt.close(fig)
