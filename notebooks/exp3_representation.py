"""Offline Exp1 representation audit. No simulator, runner, or optimizer is built."""
import hashlib
import heapq
import inspect
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import exp1_analysis as shared

ROOT = Path(__file__).resolve().parents[1]
VERSION = 2
MAIN = ("ee_force_rmse", "base_force_rmse", "base_moment_rmse", "grf_rmse",
        "base_linear_rmse", "base_angular_rmse", "joint_velocity_rmse")
UNITS = dict(zip(MAIN, ("N", "N", "Nm", "N", "m/s", "rad/s", "rad/s")))
LOADS = ("ee_force_world", "base_force_world", "base_moment_world")
INDEX_FIELDS = ("trial_id", "step", "phase", "valid", "alive_for_metrics", "physical_failure", "artificial_reset")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def chunks(run, fields, only=None):
    """Validate and read only requested numeric columns from the recorded manifest."""
    seen = set()
    for chunk in run["records"]["chunks"]:
        if only is not None and chunk["file"] not in only:
            continue
        path = (run["path"] / chunk["file"]).resolve()
        if path.parent != run["path"].resolve() or path in seen:
            raise ValueError("Invalid/duplicate chunk path")
        seen.add(path)
        with np.load(path, allow_pickle=False) as arrays:
            d = {k: arrays[k] for k in fields}
        for k, v in d.items():
            spec = run["records"]["fields"][k]
            if v.shape != (chunk["rows"], *spec["shape"]) or str(v.dtype) != spec["dtype"]:
                raise ValueError(f"Schema mismatch: {path}/{k}")
        yield chunk["file"], d


def family(meta):
    runner = meta["runner"].lower()
    if "unifp" in runner:
        return "unifp_original" if "original" in runner else "unifp"
    if "pact" in runner:
        env = meta["resolved_environment"]["env"]
        return "pact_position" if env["num_policy_actions"] == env["num_actions"] else "pact_coupled"
    raise ValueError(f"Unsupported architecture: {meta['runner']}")


def input_contract(meta):
    """Equal widths are insufficient: position/FF history and force commands differ."""
    cfg = meta["resolved_environment"]
    return shared.fingerprint(dict(family=family(meta), obs=cfg["env"]["num_observations"],
        history=cfg["env"]["num_obs_hist"], normalization=cfg["normalization"],
        dofs=meta["dof_names"], initial=cfg.get("init_state"), control_scales={k: v for k,v in cfg["control"].items() if "scale" in k},
        feet=cfg["asset"]["foot_name"],
        noise=cfg.get("noise"), dt=meta["dt"],
        grf_torque_contract=meta["conventions"].get("prediction_grf_torque_input_nm"),
        # Cross-UniFP compensation histories are policy dependent, even at equal widths.
        compensation_checkpoint=meta["checkpoint_sha256"] if "unifp" in family(meta) else None))


def resolve_methods(config):
    """Discover methods from completed Exp1 metadata, never directory-name guesses."""
    from b1z1_paper_names import DISPLAY_NAMES, LATEX_METHOD_NAMES

    config = dict(config)
    explicit = config.get("methods")
    tasks = config.get("tasks")
    if explicit is not None and tasks is not None:
        raise ValueError("Select tasks or explicit methods, not both")
    if explicit is None:
        available = set()
        for path in shared.discover(config["recording_roots"], "metadata.json"):
            if not (path.parent / "COMPLETE").is_file():
                continue
            meta = shared.read_json(path)
            if (meta.get("experiment", 1) == 1
                    and meta["evaluation_seed"] in config["evaluation_seeds"]
                    and meta["arguments"]["condition"] in config["conditions"]):
                available.add(meta["task"])
        if tasks is not None:
            if not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks):
                raise ValueError("tasks must be a nonempty list of unique task IDs")
            missing = set(tasks) - available
            if missing:
                raise ValueError(f"No completed matching Exp1 recordings for {sorted(missing)}")
        selected = sorted(available) if tasks is None else tasks
        if not selected:
            raise ValueError("No completed matching Exp1 recordings")
        explicit = {task: {"task": task} for task in selected}
    # Copy specs so discovery and paper labels do not mutate caller configuration.
    config["methods"] = {name: dict(spec) for name, spec in explicit.items()}
    config["tasks"] = None  # Resolved mappings can safely pass through discovery again.
    for spec in config["methods"].values():
        spec.setdefault("display_name", DISPLAY_NAMES.get(spec["task"], spec["task"]))
    config["latex_method_names"] = {
        **{name: LATEX_METHOD_NAMES[spec["task"]] for name, spec in config["methods"].items()
           if spec["task"] in LATEX_METHOD_NAMES},
        **config.get("latex_method_names", {}),
    }
    return config


def discover(config):
    config = resolve_methods(config)
    runs, notices, checkpoint_hashes = [], [], {}
    identities = set()
    for path in shared.discover(config["recording_roots"], "metadata.json"):
        m = shared.read_json(path)
        if m.get("experiment", 1) != 1:
            continue
        methods = [name for name, spec in config["methods"].items() if spec["task"] == m["task"]]
        if not methods or m["evaluation_seed"] not in config["evaluation_seeds"] or m["arguments"]["condition"] not in config["conditions"]:
            continue
        if not (path.parent/"COMPLETE").exists():
            notices.append(f"Incomplete recording skipped: {path.parent}")
            continue
        for name in methods:
            spec = config["methods"][name]
            checkpoint = Path(spec.get("checkpoint") or m["checkpoint"]).expanduser().resolve()
            if checkpoint not in checkpoint_hashes:
                checkpoint_hashes[checkpoint] = file_hash(checkpoint)
            if checkpoint_hashes[checkpoint] != m["checkpoint_sha256"]:
                if spec.get("checkpoint"):
                    notices.append(f"Different checkpoint cohort skipped: {path.parent}")
                    continue
                raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
            if not m.get("weights_unchanged", False):
                raise ValueError("Recording did not certify frozen inference")
            for required in ("records.json", "trials.json", "initial_states.jsonl"):
                if not (path.parent/required).is_file():
                    raise FileNotFoundError(path.parent/required)
            trials = shared.read_json(path.parent/"trials.json")
            initial = [json.loads(line) for line in (path.parent/"initial_states.jsonl").read_text().splitlines()]
            mechanics = {int(row["trial_id"]): row for row in initial}
            if len(mechanics) != len(initial) or len({r["trial_id"] for r in trials}) != len(trials):
                raise ValueError("Duplicate trial mechanics or identities")
            for row in initial:
                masses = np.asarray(row["body_masses_kg"])
                inertia = np.asarray(row["body_inertias_kg_m2"])
                com = np.asarray(row["body_local_com_pose"])
                if len(masses) != len(m["body_names"]) or not np.isfinite(inertia).all() or not np.isfinite(com).all() or np.any(masses <= 0):
                    raise ValueError("Invalid recorded inertial parameters")
                inertia = inertia.reshape(len(masses),3,3)
                if not np.allclose(inertia, inertia.swapaxes(-1,-2), atol=1e-6) or np.any(np.linalg.eigvalsh(inertia) < -1e-6):
                    raise ValueError("Nonphysical recorded inertia tensor")
            key = (name, m["evaluation_seed"], m["arguments"]["condition"], m["arguments"]["load_range"])
            if key in identities:
                raise ValueError(f"Duplicate run identity {key}")
            identities.add(key)
            runs.append(dict(path=path.parent, meta=m, name=name, checkpoint=checkpoint,
                records=shared.read_json(path.parent/"records.json"), trials={r["trial_id"]:r for r in trials}, initial=mechanics))
    if not runs:
        raise ValueError("No completed matching Exp1 recordings/checkpoints")
    if len({shared.fingerprint(shared.protocol(r["meta"])) for r in runs}) != 1:
        raise ValueError("Incompatible recorded protocols; select one protocol cohort")
    if len({(r["meta"].get("urdf_sha256"),r["meta"].get("converted_urdf_sha256")) for r in runs}) != 1:
        raise ValueError("Recorded robot-model hashes differ")
    for name in config["methods"]:
        cohort = [r for r in runs if r["name"] == name]
        if not cohort:
            notices.append(f"Missing recordings for {name}")
        if len({r["meta"]["checkpoint_sha256"] for r in cohort}) > 1:
            raise ValueError("Use a separate method ID for each independently trained checkpoint")
        for condition in config["conditions"]:
            seen = {r["meta"]["evaluation_seed"] for r in cohort if r["meta"]["arguments"]["condition"] == condition}
            if seen != set(config["evaluation_seeds"]):
                notices.append(f"{name}/{condition}: available evaluation seeds {sorted(seen)}")
    return runs, notices


def candidates(run):
    last = {}
    h = run["meta"]["resolved_environment"]["env"]["num_obs_hist"]
    for filename, d in chunks(run, INDEX_FIELDS):
        for row, (tid, step) in enumerate(zip(d["trial_id"], d["step"])):
            tid, step = int(tid), int(step)
            previous, length = last.get(tid, (-1, 0))
            if step <= previous:
                raise ValueError("Duplicate or out-of-order trial transitions")
            good = bool(d["valid"][row] and d["alive_for_metrics"][row] and not d["physical_failure"][row] and not d["artificial_reset"][row])
            length = length+1 if good and step == previous+1 else int(good)
            last[tid] = step, length
            if length < h:
                continue  # Exclude reset-filled or incomplete histories, not merely NaNs.
            trial, m = run["trials"][tid], run["meta"]
            stratum = (m["arguments"]["load_range"], m["arguments"]["condition"], trial["motion_mode"],
                       int(d["phase"][row]), m["evaluation_seed"], run["name"])
            yield stratum, dict(run=str(run["path"]), source=run["name"], trial_id=tid, step=step, chunk=filename, row=row)


def select_manifest(runs, budget, seed, per_stratum=None, minimum=8):
    """Water-fill strata, then hash-rank rows: deterministic and bounded memory."""
    if per_stratum is not None and (per_stratum < 1 or minimum < 1):
        raise ValueError("Coverage targets must be positive")
    if per_stratum is None and (budget is None or budget < 1):
        raise ValueError("sampling_budget must be positive (per loading range)")
    counts = defaultdict(int)
    for run in runs:
        for key, _ in candidates(run):
            counts[key] += 1
    quotas = dict.fromkeys(counts, 0)
    if per_stratum is not None:
        quotas = {k: min(per_stratum, n) for k, n in counts.items()}
    for load_range in sorted({k[0] for k in counts}):
        if per_stratum is not None:
            continue
        keys = sorted((k for k in counts if k[0] == load_range), key=lambda k: shared.fingerprint([seed, k]))
        remaining = min(budget, sum(counts[k] for k in keys))
        while remaining:
            for key in keys:
                if quotas[key] < counts[key] and remaining:
                    quotas[key] += 1
                    remaining -= 1
    heaps = defaultdict(list)
    for run in runs:
        for key, entry in candidates(run):
            size = quotas[key]
            if not size:
                continue
            rank = int(shared.fingerprint([seed, entry]), 16)
            item = (-rank, entry["step"], entry["trial_id"], entry)
            if len(heaps[key]) < size:
                heapq.heappush(heaps[key], item)
            elif item[0] > heaps[key][0][0]:
                heapq.heapreplace(heaps[key], item)
    manifest = []
    for key, heap in sorted(heaps.items()):
        for _, _, _, entry in heap:
            entry.update(dict(zip(("load_range", "condition", "motion", "phase", "evaluation_seed", "source"), key)))
            entry["transition_id"] = shared.fingerprint(entry)
            manifest.append(entry)
    return sorted(manifest, key=lambda e: e["transition_id"]), [dict(stratum=list(k), available=counts[k], selected=quotas[k],
        minimum=minimum, sufficient=quotas[k] >= minimum) for k in sorted(counts)]


def selected_data(run, entries):
    selected = {(e["trial_id"], e["step"]):e for e in entries}
    previous, result = {}, []
    # Bounded by the selected budget; large, unrelated observation/privileged outputs are omitted.
    names = run["records"]["fields"]
    fields = [k for k in names if k in INDEX_FIELDS or k in LOADS or k.startswith(("prediction_", "mechanics_")) or k in
              ("history_t", "obs_t", "action_raw", "joint_position_t", "joint_velocity_t", "base_position_t", "base_quaternion_t",
               "base_velocity_world_t", "base_angular_velocity_world_t", "base_quaternion_xyzw", "base_velocity_world",
               "base_angular_velocity_world", "joint_velocity", "executed_torque_interval_mean", "grf_raw_world_interval_mean",
               "base_load_point_world_interval_mean", "ee_load_point_world_interval_mean", "time_seconds", "successor_time_seconds")]
    selected_chunks = {e["chunk"] for e in entries}
    for filename, d in chunks(run, tuple(dict.fromkeys(("trial_id", "step", *LOADS)))):
        full = next(chunks(run, fields, {filename}))[1] if filename in selected_chunks else None
        for row, tid in enumerate(d["trial_id"]):
            tid, step = int(tid), int(d["step"][row])
            if (tid, step) in selected:
                if tid not in previous or previous[tid][0] != step-1:
                    raise ValueError("Missing previous loading for causal force targets")
                item = {k: v[row].copy() for k, v in full.items()}
                for k in LOADS:
                    item["history_"+k] = previous[tid][1][k].copy()
                width = run["meta"]["resolved_environment"]["env"]["num_observations"]
                slots = run["meta"]["resolved_environment"]["env"]["num_obs_hist"]
                if item["history_t"].shape != (width*slots,) or not np.isfinite(item["history_t"]).all() or not np.allclose(item["history_t"][-width:], item["obs_t"], atol=1e-6):
                    raise ValueError("Native history ordering/content mismatch")
                dt = run["meta"]["dt"]
                if not np.isclose(item["time_seconds"], step*dt) or not np.isclose(item["successor_time_seconds"], (step+1)*dt):
                    raise ValueError("Invalid interval/successor timestamp alignment")
                condition = run["meta"]["arguments"]["condition"]
                forbidden = LOADS if condition == "nominal" else LOADS[1:] if condition == "ee-disturbance" else ()
                if any(np.any(np.abs(item[k]) > 1e-5) for k in forbidden):
                    raise ValueError("Applied loads violate the recorded condition")
                item["entry"] = selected[tid, step]
                result.append(item)
            previous[tid] = (step, {k:d[k][row] for k in LOADS})
    if len(result) != len(selected):
        raise ValueError("Selected manifest rows missing")
    return result


def build_model(run, device):
    """Same constructors/aliases as eval_b1z1_exp1.build_actor, without registry imports."""
    import torch
    from rsl_rl import modules
    m, env = run["meta"], run["meta"]["resolved_environment"]["env"]
    p = m["resolved_training"]["policy"]
    obs, history = env["num_observations"], env["num_observations"]*env["num_obs_hist"]
    critic = run["records"]["fields"]["privileged_t"]["shape"][0]
    kind = family(m)
    if kind.startswith("pact"):
        cls = modules.ActorCriticB1Z1PACTPos if m["runner"] == "B1Z1PACTPosRunner" else modules.ActorCriticB1Z1PACT
        aliases = dict(latent_dim=p["cenet_latent_dim"], context_layers=p["cenet_enc_layers"], explicit_dim=env["num_explicit_recon_obs"])
        kwargs = {k:aliases.get(k, p.get(k)) for k in inspect.signature(cls).parameters if k in aliases or k in p}
        model = cls(obs, critic, env["num_actions"], history, **kwargs)
    elif kind == "unifp_original":
        from rsl_rl.modules.actor_critic_unifp_original import ActorCriticUniFPOriginal
        model = ActorCriticUniFPOriginal(history, critic, env["num_pred_obs"], obs, env["num_actions"], min_noise_std=p.get("min_noise_std"))
    else:
        cls = getattr(modules, m["resolved_training"]["runner"]["policy_class_name"])
        model = cls(history, critic, env["num_pred_obs"], obs, env["num_actions"],
                    num_privileged_obs_single=env["num_privileged_recon_obs"], enable_additional_diagnostics=False, **p)
    # Only load trusted local evaluation checkpoints. Never instantiate their optimizers.
    state = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
    if "architecture" in state and hasattr(model, "architecture") and state["architecture"] != model.architecture:
        raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def infer(model, data, device):
    import torch
    tensor = lambda name: torch.as_tensor(np.stack([r[name] for r in data]), dtype=torch.float32, device=device)
    with torch.inference_mode():
        history = tensor("history_t")
        if hasattr(model, "decode_context"):
            context = model.decode_context(model.context_encoder.forward_inf(history))
            result = {"prediction_"+k:v for k,v in context.items() if torch.is_tensor(v)}
            if "prediction_grf_torque_input_nm" in data[0]:
                result["prediction_grf"] = model.predict_grf(context, tensor("prediction_grf_torque_input_nm"))
        elif hasattr(model, "architecture"):
            z = model.adaptation_encoder_module(history)
            result = dict(prediction_z=z, prediction_explicit=model.adaptation_decoder_module(z))
        else:
            _, _, z = model.encode_context(history, sample=False)
            result = dict(prediction_z=z, prediction_explicit=model.adaptation_decoder_module(z))
        return {k:v.cpu().numpy() for k,v in result.items()}


def reconstruct_inputs(data, donor, target):
    """Exact recorded-input replay only where semantics, scales and causality agree.

    Raw states do not reveal observation noise or a different estimator's LPF
    compensation history. Equal widths alone therefore never authorize replay.
    """
    if input_contract(donor) != input_contract(target):
        raise ValueError("Unreconstructable native history: normalization/action semantics or estimator-dependent compensation differ")
    width = target["resolved_environment"]["env"]["num_observations"]
    slots = target["resolved_environment"]["env"]["num_obs_hist"]
    for row in data:
        if (np.shape(row["history_t"]) != (width*slots,) or np.shape(row["obs_t"]) != (width,)
                or not np.isfinite(row["history_t"]).all()
                or not np.allclose(row["history_t"][-width:], row["obs_t"], atol=1e-6)):
            raise ValueError("Invalid causal native input/history reconstruction")
    return data


def parity(predictions, data, atol=2e-5, rtol=2e-4):
    checked = []
    for key, value in predictions.items():
        if key not in data[0]:
            continue
        saved = np.stack([r[key] for r in data])
        if saved.shape != value.shape or not np.allclose(saved, value, atol=atol, rtol=rtol):
            error = float(np.max(np.abs(saved-value))) if saved.shape == value.shape else "shape mismatch"
            raise ValueError(f"Native prediction parity failed: {key}, max error={error}; check device/precision/code before cross-method replay")
        checked.append(key)
    if not checked or "prediction_z" not in checked:
        raise ValueError("No saved latent predictions for native parity")
    return checked


def rotate(v, q, yaw_only=False):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) <= 0:
        raise ValueError("Invalid recorded quaternion")
    q = q / np.linalg.norm(q)
    if yaw_only:
        x,y,z,w = q
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        q = np.array([0., 0., np.sin(yaw/2), np.cos(yaw/2)])
    xyz, w = q[:3], q[3]
    return v + 2*np.cross(xyz, np.cross(xyz, v) + w*v)


def physical_predictions(pred, meta, row):
    scales = meta["resolved_environment"]["normalization"]["obs_scales"]
    for name in ("ee_force", "base_force", "grf"):
        if not np.isfinite(scales[name]) or scales[name] <= 0:
            raise ValueError(f"Invalid native {name} normalization")
    kind, q = family(meta), row["base_quaternion_t"]
    if kind.startswith("pact"):
        ee = pred["prediction_ee_force"]/scales["ee_force"]
        base = pred["prediction_base_wrench"] / scales["base_force"]
        values = dict(ee_force=rotate(ee, q, True), base_force=rotate(base[:3], q, True), base_moment=rotate(base[3:], q, True))
    else:
        explicit = pred["prediction_explicit"]
        base_slice, ee_slice = (slice(9,12), slice(6,9)) if kind == "unifp_original" else (slice(6,9), slice(9,12))
        values = dict(ee_force=rotate(explicit[ee_slice]/scales["ee_force"], q, True),
                      base_force=rotate(explicit[base_slice]/scales["base_force"], q, True))
    if "prediction_grf" in pred:
        # Training GRF targets are successor-yaw vectors, unlike current force heads.
        values["grf"] = rotate(pred["prediction_grf"].reshape(-1,3)/scales["grf"], row["base_quaternion_xyzw"], True)
    return values


def dynamics_step(mass, bias, pre_v, torque, foot_j, ee_j, base_j, grf, ee, wrench, dt):
    """Frozen-mechanics analytic Euler velocity step, identical to BARD rollout objective."""
    generalized = np.r_[np.zeros(6), torque] + np.einsum("fkn,fk->n", foot_j[:,:3], grf)
    generalized += ee_j.T @ ee + base_j.T @ wrench
    return pre_v + dt*np.linalg.solve(mass, generalized-bias)


class OfflineBard:
    def __init__(self, run, device, urdf_override=None):
        # Importing the dynamics package does not construct any simulator.
        os.environ.setdefault("SIMULATOR", "isaaclab_b1z1_pact")
        from legged_gym.dynamics.bard_b1z1_dynamics import BardB1Z1DynamicsBackend
        m, asset = run["meta"], run["meta"]["resolved_environment"]["asset"]
        urdf = Path(urdf_override or m["urdf_source"])
        if file_hash(urdf) != m["urdf_sha256"]:
            raise ValueError("BARD URDF differs from recorded robot model")
        self.backend = BardB1Z1DynamicsBackend(str(urdf), m["dof_names"], asset["foot_name"],
            asset["gripper_name"], asset["base_name"], device=device, batch_capacity=1)
        self.run, self.device = run, device

    def terms(self, row):
        import torch
        b, m = self.backend, self.run["meta"]
        t = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.device).reshape(1,-1)
        initial = self.run["initial"][row["entry"]["trial_id"]]
        with torch.no_grad():
            terms = b.evaluate(t(row["base_position_t"]), t(row["base_quaternion_t"]), t(row["joint_position_t"]),
                t(row["base_velocity_world_t"]), t(row["base_angular_velocity_world_t"]), t(row["joint_velocity_t"]),
                torch.zeros(1,4,3,device=self.device), torch.zeros(1,3,device=self.device), torch.zeros(1,6,device=self.device),
                t(row["mechanics_added_base_mass"]), t(row["mechanics_base_com_bias"]), t(row["mechanics_added_gripper_mass"]))
            # Request FK pose from the already-updated BARD data, at q_t (no new rollout).
            _, pose = b.bard.jacobian(b.model, b.data, b.ee_frame_id, reference_frame="local", return_pose=True)
            pose = pose[0].cpu().numpy().copy()
            result = {k:getattr(terms,k)[0].cpu().numpy().copy() for k in ("mass_matrix","bias","foot_jacobians","ee_jacobian","base_jacobian")}
        actual_mass = np.sum(initial["body_masses_kg"])
        if not np.allclose(np.diag(result["mass_matrix"])[:3], actual_mass, rtol=5e-3, atol=1e-3):
            raise ValueError("BARD total mass does not match recorded body masses")
        # Rotor armature is additive joint-space inertia, not an external wrench.
        armature = np.asarray(row.get("mechanics_joint_armature", np.zeros(len(row["joint_position_t"]))))
        if armature.size == 1:
            # Lab records one per-environment value expanded to every joint at application.
            armature = np.full(len(row["joint_position_t"]), float(armature.reshape(-1)[0]))
        if armature.size != len(row["joint_position_t"]):
            raise ValueError("Unknown armature ordering")
        result["mass_matrix"][6:,6:] += np.diag(armature)
        from exp3_support import passive_bias
        passive, assumptions = passive_bias(row)
        # Executed torques already include explicit PD/motor scaling. Only the
        # simulator's separate passive drive damping is added to the bias here.
        result["bias"][6:] += passive
        result["approximation"] = assumptions
        com = np.asarray(initial["body_local_com_pose"])
        result["base_lever"] = rotate(com[m["base_index"],:3], row["base_quaternion_t"])
        result["ee_lever"] = pose[:3,:3] @ com[m["gripper_index"],:3]
        return result


def velocity_errors(row, values, fixed, dt):
    pre = np.r_[row["base_velocity_world_t"], row["base_angular_velocity_world_t"], row["joint_velocity_t"]]
    post = np.r_[row["base_velocity_world"], row["base_angular_velocity_world"], row["joint_velocity"]]
    if values is None:
        return pre-post
    # Wrenches are labelled at body CoM; shift to the BARD frame origin exactly once.
    ee = np.r_[values["ee_force"], np.cross(fixed["ee_lever"], values["ee_force"])]
    wrench = np.r_[values["base_force"], values["base_moment"] + np.cross(fixed["base_lever"], values["base_force"])]
    return dynamics_step(fixed["mass_matrix"], fixed["bias"], pre, row["executed_torque_interval_mean"],
        fixed["foot_jacobians"], fixed["ee_jacobian"], fixed["base_jacobian"], values["grf"], ee, wrench, dt)-post


def velocity_blocks(error, names):
    blocks = dict(base_linear=error[:3], base_angular=error[3:6], joint_velocity=error[6:])
    for label in ("arm", "leg"):
        indices = [j for j, name in enumerate(names) if name.startswith("z1_") == (label == "arm")]
        blocks[label+"_velocity"] = error[6:][indices]
    return blocks


def metric_rows(entry, method, benchmark, variant, errors, foot_names, dof_names, load_timing):
    rows = []
    for field, error in errors.items():
        error = np.asarray(error, dtype=float)
        if not np.isfinite(error).all():
            continue
        labels = [f"{foot}_{axis}" for foot in foot_names for axis in "xyz"] if field == "grf" else (
            list(dof_names) if field == "joint_velocity" else list("xyz") if error.size == 3 else [str(i) for i in range(error.size)])
        if field in ("arm_velocity", "leg_velocity"):
            labels = [n for n in dof_names if n.startswith("z1_") == (field == "arm_velocity")]
        for axis, value in [("all", None), *zip(labels, error.flatten())]:
            rows.append(dict(**entry, method=method, benchmark=benchmark, variant=variant, metric=field+"_rmse",
                axis=axis, mse=float(np.mean(error**2)) if value is None else float(value**2),
                bias=float(np.mean(error)) if value is None else float(value), load_timing=load_timing))
    return rows


def aggregate(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return [], []
    # Timesteps are not replicates: RMSE within trial, equal trials within seed.
    strata = ["method","benchmark","variant","load_range","condition","motion","phase","load_timing","metric","axis"]
    trials = frame.groupby(strata+["evaluation_seed","source","run","trial_id"], dropna=False).agg(mse=("mse","mean"), bias=("bias","mean"), samples=("mse","size")).reset_index()
    trials["rmse"] = np.sqrt(trials.mse)
    seeds = trials.groupby(strata+["evaluation_seed"], dropna=False).agg(mean=("rmse","mean"), bias=("bias","mean"), trials=("rmse","size"), samples=("samples","sum")).reset_index()
    summary = []
    for key, group in seeds.groupby(strata, dropna=False):
        summary.append(dict(zip(strata,key), **shared.seed_statistics(group["mean"]), bias=shared.mean(group.bias),
                            contributing_trials=int(group.trials.sum()), contributing_samples=int(group.samples.sum())))
    return shared.clean(seeds.to_dict("records")), shared.clean(summary)


def run_analysis(config):
    if config.get("recorded_native_only", False) and not config.get("dynamics", True):
        return _run_analysis(config)
    import torch
    precision = config.get("matmul_precision", "high")
    if precision not in ("highest", "high", "medium"):
        raise ValueError("Unknown float32 matmul precision")
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(precision)
        return _run_analysis(config)
    finally:
        torch.set_float32_matmul_precision(previous)


def _run_analysis(config):
    config = resolve_methods(config)
    output = Path(config["output_dir"]).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output/"COMPLETE").unlink(missing_ok=True)
    runs, notices = discover(config)
    manifest, balance = select_manifest(runs, config.get("sampling_budget"), config.get("sampling_seed", 0),
        config.get("samples_per_stratum", 32) if config.get("sampling_budget") is None else None,
        config.get("minimum_per_stratum", 8))
    notices.extend(f"Insufficient coverage: {b}" for b in balance if not b["sufficient"])
    if not manifest:
        raise ValueError("No complete, valid histories within selected recordings")
    shared.write_json(output/"transition_manifest.json", dict(schema_version=VERSION, transitions=manifest, strata=balance))
    pd.DataFrame(manifest).to_csv(output/"transition_manifest.csv", index=False)
    device = config.get("device", "cpu")
    recorded_only = config.get("recorded_native_only", False)
    models, method_runs, parity_checks, data_by_run = {}, {}, {}, {}
    failed_parity = set()
    for run in runs:
        entries = [e for e in manifest if e["run"] == str(run["path"])]
        if not entries:
            continue
        data = selected_data(run, entries)
        data_by_run[str(run["path"])] = data
        name = run["name"]
        if name not in models:
            models[name] = None if recorded_only else build_model(run, device)
            method_runs[name] = run
        if recorded_only:
            parity_checks[str(run["path"])] = "NOT rerun: explicitly selected recorded-native audit"
        else:
            reconstructed = reconstruct_inputs(data, run["meta"], run["meta"])
            predicted = infer(models[name], reconstructed, device)
            try:
                parity_checks[str(run["path"])] = parity(predicted, data, config.get("parity_atol",2e-5), config.get("parity_rtol",2e-4))
            except ValueError as exc:
                # Failed validation excludes this checkpoint from replay, not a
                # tolerance relaxation or substitution of native trajectories.
                failed_parity.add(name)
                parity_checks[str(run["path"])] = "FAILED: " + str(exc)
                notices.append(f"{name} excluded from replay: {exc}")
    contracts = {name:input_contract(r["meta"]) for name,r in method_runs.items()}
    audit = [dict(method=name, family=family(r["meta"]),
                  observation_width=r["meta"]["resolved_environment"]["env"]["num_observations"],
                  history_slots=r["meta"]["resolved_environment"]["env"]["num_obs_hist"],
                  input_contract=contracts[name],
                  torque_history="executed PD torque" if family(r["meta"])=="pact_position" else
                                 "feedforward actions" if family(r["meta"])=="pact_coupled" else "native UniFP action history + force commands")
             for name,r in method_runs.items()]
    pd.DataFrame(audit).to_csv(output/"input_audit.csv",index=False)
    rows, unavailable, cache_dir = [], [], output/"cache"
    # Every table cell must cover the SAME manifest, not a method-specific subset.
    # Policy-dependent compensation histories cannot be synthesized from forces.
    shared_methods = {name for name in models if not recorded_only and name not in failed_parity and all(
        contracts[name] == input_contract(r["meta"]) for r in runs if str(r["path"]) in data_by_run)}
    for name in models.keys() - shared_methods:
        unavailable.append(dict(method=name, component="shared-transition benchmark", reason=
            "Saved-native audit only" if recorded_only else
            "Native prediction parity failed; checkpoint excluded from replay" if name in failed_parity else
            "Manifest includes incompatible native action/command histories. Counterfactual UniFP compensation commands or position/FF histories are not reconstructable by copying donor inputs."))
    for name,r in method_runs.items():
        if "unifp" in family(r["meta"]):
            unavailable.append(dict(method=name, component="base moment, GRF, decoded-loading dynamics",
                                    reason="Native UniFP architecture has no moment or GRF decoder; no measured substitution"))
    if not config.get("dynamics", True):
        unavailable.append(dict(component="BARD dynamics", reason="Explicitly disabled; constant-velocity reference only"))
    cache_dir.mkdir(exist_ok=True)
    source_hash = shared.fingerprint({str(p.relative_to(ROOT)):file_hash(p) for p in
        [Path(__file__), *sorted((ROOT/"rsl_rl/modules").glob("*b1z1*.py")), ROOT/"rsl_rl/modules/actor_critic_unifp_original.py", ROOT/"rsl_rl/modules/actor_critic_unifp.py"]})
    approximations = []
    for run in runs:
        data = data_by_run.get(str(run["path"]))
        if not data:
            continue
        m = run["meta"]
        compatible = [run["name"]] if recorded_only else [name for name in models if name not in failed_parity and contracts[name] == input_contract(m)]
        bard = None
        if config.get("dynamics", True):
            try:
                bard = OfflineBard(run, device, config.get("urdf"))
            except (ImportError, ValueError, RuntimeError, OSError) as exc:
                unavailable.append(dict(run=str(run["path"]), component="BARD", reason=str(exc)))
        fixed_terms = {}
        for i, row in enumerate(data):
            if bard:
                try:
                    fixed_terms[i] = bard.terms(row)
                    approximations.append(dict(transition=row["entry"]["transition_id"],
                        assumptions=fixed_terms[i].get("approximation", [])))
                except (ValueError, RuntimeError, KeyError) as exc:
                    unavailable.append(dict(transition=row["entry"]["transition_id"], component="BARD mechanics", reason=str(exc)))
            # References require no encoder and remain available even if native
            # parity fails. Every method receives exactly the same donor rows.
            references = {"constant_velocity": None}
            if i in fixed_terms:
                references["measured_loading"] = dict(ee_force=row["ee_force_world"], base_force=row["base_force_world"],
                    base_moment=row["base_moment_world"], grf=row["grf_raw_world_interval_mean"])
            timing = "new_load" if any(not np.allclose(row[k], row["history_"+k], atol=1e-6) for k in LOADS) else "sustained"
            for variant, values in references.items():
                blocks = velocity_blocks(velocity_errors(row, values, fixed_terms.get(i), m["dt"]), m["dof_names"])
                for name in models:
                    rows += metric_rows(row["entry"], name, "shared", variant, blocks, [], m["dof_names"], timing)
                    if name == run["name"]:
                        rows += metric_rows(row["entry"], name, "native_recorded" if recorded_only else "native",
                                            variant, blocks, [], m["dof_names"], timing)
        for name in compatible:
            target = method_runs[name]
            input_hash = hashlib.sha256()
            for row in data:
                for key in ("history_t", "prediction_grf_torque_input_nm"):
                    if key in row:
                        input_hash.update(np.ascontiguousarray(row[key]).tobytes())
            cache_key = shared.fingerprint(dict(source=source_hash, checkpoint=target["meta"]["checkpoint_sha256"],
                recorded_native_only=recorded_only,
                inputs=input_hash.hexdigest(), device=device, matmul_precision=config.get("matmul_precision", "high"),
                metadata=file_hash(run["path"]/"metadata.json"), records=file_hash(run["path"]/"records.json"),
                transitions=[r["entry"]["transition_id"] for r in data]))
            cache = cache_dir/(cache_key+".npz")
            if cache.exists():
                with np.load(cache, allow_pickle=False) as saved:
                    predictions = dict(saved)
            else:
                predictions = ({k:np.stack([r[k] for r in data]) for k in data[0] if k.startswith("prediction_")}
                               if recorded_only else infer(models[name], reconstruct_inputs(data, m, target["meta"]), device))
                np.savez_compressed(cache, **predictions)
                shared.write_json(cache.with_suffix(".json"), dict(checkpoint_sha256=target["meta"]["checkpoint_sha256"], source_hash=source_hash,
                    source_run=str(run["path"]), transitions=[r["entry"]["transition_id"] for r in data]))
            scopes = (["native_recorded" if recorded_only else "native"] if name == run["name"] else []) + (["shared"] if name in shared_methods else [])
            for i,row in enumerate(data):
                physical = physical_predictions({k:v[i] for k,v in predictions.items()}, target["meta"], row)
                changed = any(not np.allclose(row[k], row["history_"+k], atol=1e-6) for k in LOADS)
                timing = "new_load" if changed else "sustained"
                targets = dict(ee_force=row["history_ee_force_world"], base_force=row["history_base_force_world"],
                               base_moment=row["history_base_moment_world"], grf=row["grf_raw_world_interval_mean"])
                errors = {k:physical[k]-targets[k] for k in physical}
                for key, value in errors.items():
                    if not np.isfinite(value).all():
                        unavailable.append(dict(method=name, transition=row["entry"]["transition_id"], component=key,
                                                reason="Nonfinite prediction/target; excluded, never replaced by zero"))
                for scope in scopes:
                    rows += metric_rows(row["entry"], name, scope, "decoded", errors, m["resolved_environment"]["asset"]["foot_name"], m["dof_names"], timing)
                    variants = {}
                    if i in fixed_terms:
                        if set(physical) == {"ee_force","base_force","base_moment","grf"}:
                            variants["decoded"] = physical
                    for variant, values in variants.items():
                        error = velocity_errors(row, values, fixed_terms.get(i), m["dt"])
                        blocks = velocity_blocks(error, m["dof_names"])
                        rows += metric_rows(row["entry"], name, scope, variant, blocks, [], m["dof_names"], timing)
        for name in models:
            if name not in compatible:
                notices.append(f"{name} cannot replay {run['name']}: different native history semantics; no padding/privileged force synthesis")
    # A finite subset is not the shared benchmark. Keep partial diagnostics, but
    # suppress a primary cell unless it covers every prescribed donor transition.
    expected = defaultdict(set)
    for entry in manifest:
        expected[entry["load_range"], entry["condition"] == "nominal"].add(entry["transition_id"])
    actual = defaultdict(set)
    for row in rows:
        if row["benchmark"] == "shared" and row["variant"] == "decoded" and row["axis"] == "all":
            actual[row["method"], row["metric"], row["load_range"], row["condition"] == "nominal"].add(row["transition_id"])
    incomplete = {key for key, ids in actual.items() if ids != expected[key[2:]]}
    for name, metric, load, nominal in incomplete:
        unavailable.append(dict(method=name, component=metric, load_range=load, nominal=nominal,
                                reason="Incomplete/nonfinite shared-manifest coverage; primary cell unavailable"))
    table_input = [row for row in rows if not (row["benchmark"] == "shared" and row["variant"] == "decoded"
        and (row["method"], row["metric"], row["load_range"], row["condition"] == "nominal") in incomplete)]
    per_seed, breakdown = aggregate(rows)
    # Recompute pooled nominal/disturbed trial statistics, never average stratum RMSEs/SDs.
    table_rows = [dict(r, condition="nominal" if r["condition"]=="nominal" else "disturbed", motion="all", phase="all", load_timing="all") for r in table_input]
    table_seeds, table_summary = aggregate(table_rows)
    from exp3_support import hierarchy
    hierarchical_rows = []
    for item in table_seeds:
        source = method_runs[item["method"]]["meta"]
        hierarchical_rows.append(dict(item, task=source["task"], checkpoint=source["checkpoint_sha256"],
            training_seed=source.get("checkpoint_training_seed"), seed=item["evaluation_seed"], trial=0, value=item["mean"]))
    comparative = hierarchy(pd.DataFrame(hierarchical_rows),
        ["task", "benchmark", "variant", "load_range", "condition", "metric", "axis"])
    result = dict(schema_version=VERSION, status="complete", experiment=3, protocol=shared.protocol(runs[0]["meta"]),
        recorded_native_only=recorded_only,
        inference_device=device, matmul_precision=config.get("matmul_precision", "high"),
        input_audit=audit,
        methods={name:dict(task=r["meta"]["task"], checkpoint=str(r["checkpoint"]), checkpoint_sha256=r["meta"]["checkpoint_sha256"],
                          training_seed=r["meta"].get("checkpoint_training_seed"), input_contract=contracts[name]) for name,r in method_runs.items()},
        per_seed=per_seed, breakdown=breakdown, table_per_seed=table_seeds, table_summary=table_summary,
        comparative_summary=comparative.to_dict("records"),
        native_parity=parity_checks, unavailable=unavailable, diagnostics=sorted(set(notices)),
        coverage=balance, dynamics_approximations=approximations,
        units={**UNITS,"arm_velocity_rmse":"rad/s","leg_velocity_rmse":"rad/s"},
        conventions=dict(force_target="previous interval applied load visible to history_t; world SI at each body's CoM",
                         grf_target="raw executed-interval mean, NOT processed training target; successor-yaw decoder conversion",
                         dynamics="q_t frozen mechanics, executed interval-mean torque; Euler velocity; decoded current loads lag newly imposed events",
                         missing="No measured substitution in primary decoded-loading dynamics; UniFP lacks moments/GRFs",
                         reference_points="CoM forces shifted once to BARD frame origins; foot point-force approximation"))
    pd.DataFrame(per_seed).to_csv(output/"per_seed.csv", index=False)
    pd.DataFrame(breakdown).to_csv(output/"breakdown.csv", index=False)
    pd.DataFrame(table_summary).to_csv(output/"summary.csv", index=False)
    comparative.to_csv(output/"checkpoint_summary.csv", index=False)
    shared.write_json(output/"summary.json", result)
    export_tables(result, config, output)
    (output/"COMPLETE").write_text("Experiment 3 analysis complete\n")
    return result


def export_tables(result, config, output):
    for scope in ("shared",):
        for load in sorted({b["stratum"][0] for b in result["coverage"]}):
            records = []
            seen_tasks = set()
            for method in result["methods"]:
                task = result["methods"][method]["task"]
                if task in seen_tasks:
                    continue
                seen_tasks.add(task)
                if scope.startswith("common:") and not scope.endswith(result["methods"][method]["input_contract"][:12]):
                    continue
                row = {"Method":config["methods"][method].get("display_name",method)}
                for condition in ("nominal","disturbed"):
                    for metric in MAIN:
                        item = next((r for r in result["comparative_summary"] if r["task"]==task and r["benchmark"]==scope and r["load_range"]==load and r["condition"]==condition and r["metric"]==metric and r["axis"]=="all" and r["variant"]=="decoded"),None)
                        row[(condition,metric)] = shared.formatted(dict(item,unit=UNITS[metric]) if item else None)
                records.append(row)
            if not records:
                continue
            frame = pd.DataFrame(records).set_index("Method")
            frame.columns = pd.MultiIndex.from_tuples([(c, f"{m.replace('_rmse','').replace('_',' ')} [{UNITS[m]}]") for c,m in frame.columns])
            name = scope.replace(":","_")+"_"+load
            frame.to_csv(output/(name+"_table.csv"))
            macros = config.get("latex_method_names",{})
            tokens = {config["methods"][m].get("display_name",m):f"METHOD{i}TOKEN" for i,m in enumerate(macros) if m in result["methods"]}
            tex = frame.rename(index=tokens).to_latex(escape=True,multicolumn_format="c")
            for m,macro in macros.items():
                if m in result["methods"]:
                    tex=tex.replace(tokens[config["methods"][m].get("display_name",m)],macro)
            tex=tex.replace("±",r"$\pm$").replace("—","---")
            (output/(name+"_table.tex")).write_text("\\begin{table*}[t]\n\\centering\n\\caption{Representation and one-step velocity RMSE on the same manifest; "+load+". Equal trials, evaluation seeds, then checkpoints. SD across checkpoints when available; otherwise evaluation-seed SD. Dashes indicate unavailable shared replay, not native-trajectory substitution.}\n\\resizebox{\\textwidth}{!}{%\n"+tex+"}\n\\end{table*}\n")

