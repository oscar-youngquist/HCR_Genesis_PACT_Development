"""Offline Exp1 representation audit. No simulator, runner, or optimizer is built."""
import hashlib
import heapq
import inspect
import json
import pickle
import time
import gzip
import shutil
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import multiprocessing
from contextlib import ExitStack, nullcontext
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import exp1_analysis as shared
from analysis_runtime import NumericCache, ordered_work, timed_read_chunk
from exp3_representation_batch import (pack_terms, unpack_terms, process_batch, pack_job,
    STATE_KEYS, PREDICTIONS, EXPLICIT_STATE_FIELDS, missing_predictions)
import exp3_rollout_statistics as rollout_stats
import exp3_contact as contact

ROOT = Path(__file__).resolve().parents[1]
VERSION = 5
MAIN = ("ee_force_rmse", "base_force_rmse", "base_moment_rmse", "grf_rmse",
        "base_linear_rmse", "base_angular_rmse", "joint_velocity_rmse",
        "explicit_base_velocity_rmse", "explicit_ee_position_rmse")
UNITS = dict(zip(MAIN, ("N", "N", "Nm", "N", "m/s", "rad/s", "rad/s", "m/s", "m")))
MAIN = MAIN[:3] + ('grf_normal_rmse', 'grf_net_rmse', 'grf_net_horizontal_rmse', 'grf_net_window_rmse') + MAIN[3:]
UNITS.update({k:'N' for k in MAIN if k.startswith('grf_')})
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
    def jobs():
        for chunk in run["records"]["chunks"]:
            if only is not None and chunk["file"] not in only:
                continue
            path = (run["path"] / chunk["file"]).resolve()
            if path.parent != run["path"].resolve() or path in seen:
                raise ValueError("Invalid/duplicate chunk path")
            seen.add(path)
            yield chunk['file'], path, fields, run['records']['fields'], chunk['rows']
    # Only decompression/schema checks leave this process, never Torch/BARD work.
    for _, (filename, d, elapsed) in ordered_work(timed_read_chunk, jobs(), run.get('_num_workers',1)):
        run['_chunk_read_seconds'] = run.get('_chunk_read_seconds',0.)+elapsed
        yield filename, d


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
            if spec.get("checkpoint") and checkpoint.is_file() and checkpoint not in checkpoint_hashes:
                checkpoint_hashes[checkpoint] = file_hash(checkpoint)
            if spec.get("checkpoint") and checkpoint.is_file() and checkpoint_hashes[checkpoint] != m["checkpoint_sha256"]:
                if spec.get("checkpoint"):
                    notices.append(f"Different checkpoint cohort skipped: {path.parent}")
                    continue
            if not m.get("weights_unchanged", False):
                raise ValueError("Recording did not certify frozen inference")
            for required in ("records.json", "trials.json"):
                if not (path.parent/required).is_file():
                    raise FileNotFoundError(path.parent/required)
            trials = shared.read_json(path.parent/"trials.json")
            initial_path = path.parent/"initial_states.jsonl"
            initial = [json.loads(line) for line in initial_path.read_text().splitlines()] if initial_path.exists() else []
            mechanics = {int(row["trial_id"]): row for row in initial}
            if len(mechanics) != len(initial) or len({r["trial_id"] for r in trials}) != len(trials):
                raise ValueError("Duplicate trial mechanics or identities")
            for row in initial:
                try:
                    masses = np.asarray(row["body_masses_kg"])
                    inertia = np.asarray(row["body_inertias_kg_m2"])
                    com = np.asarray(row["body_local_com_pose"])
                    if len(masses) != len(m["body_names"]) or not np.isfinite(masses).all() or not np.isfinite(inertia).all() or not np.isfinite(com).all() or np.any(masses <= 0):
                        raise ValueError("Invalid recorded inertial parameters")
                    inertia = inertia.reshape(len(masses),3,3)
                    if not np.allclose(inertia, inertia.swapaxes(-1,-2), atol=1e-6) or np.any(np.linalg.eigvalsh(inertia) < -1e-6):
                        raise ValueError("Nonphysical recorded inertia tensor")
                except (KeyError,TypeError,ValueError) as exc:
                    mechanics.pop(int(row['trial_id']),None)
                    notices.append(f"{path.parent}, trial {row['trial_id']}: offline mechanics unavailable ({exc}); stored metrics retained")
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
            raise ValueError("Use a separate method ID for each checkpoint hash")
        for condition in config["conditions"]:
            seen = {r["meta"]["evaluation_seed"] for r in cohort if r["meta"]["arguments"]["condition"] == condition}
            if seen != set(config["evaluation_seeds"]):
                notices.append(f"{name}/{condition}: available evaluation seeds {sorted(seen)}")
    return runs, notices


def candidates(run):
    last = {}
    h = 2  # Only the preceding executed interval is needed for causal force targets.
    for filename, d in chunks(run, INDEX_FIELDS):
        for row, (tid, step) in enumerate(zip(d["trial_id"], d["step"])):
            tid, step = int(tid), int(step)
            previous, length = last.get(tid, (-1, 0))
            if step <= previous:
                raise ValueError("Duplicate or out-of-order trial transitions")
            good = bool(d["valid"][row] and d["alive_for_metrics"][row] and not d["physical_failure"][row] and not d["artificial_reset"][row])
            length = length+1 if good and step == previous+1 else int(good)
            last[tid] = step, length
            if length < h or previous < 0 or step != previous+1:
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


def native_batches(run, batch_size, selection, coverage):
    """One numeric chunk + one inference batch; causal state survives chunk edges.

    selection=None means ALL eligible rows. Only explicit balanced subsampling
    passes a set of (trial, step) identities; no other policy supplies inputs.
    """
    meta = run["meta"]
    env = meta["resolved_environment"]["env"]
    width, slots, dt = env["num_observations"], env["num_obs_hist"], meta["dt"]
    fields = [k for k in run["records"]["fields"] if k in INDEX_FIELDS or k in LOADS
        or k.startswith(("prediction_", "mechanics_", "dynamics_")) or k in (
            "history_t", "obs_t", "joint_position_t", "joint_velocity_t", "base_position_t", "base_quaternion_t",
            "base_velocity_world_t", "base_angular_velocity_world_t", "base_quaternion_xyzw", "base_velocity_world",
            "base_angular_velocity_world", "joint_velocity", "executed_torque_interval_mean", "grf_raw_world_interval_mean",
            "time_seconds", "successor_time_seconds", *EXPLICIT_STATE_FIELDS)]
    if run.get('_minimal_fields'):
        fields=[k for k in fields if not k.startswith('prediction_') or k in PREDICTIONS or k=='prediction_grf_torque_input_nm']
        # Prefer small named state heads; legacy explicit vectors are only aliases.
        aliases=('prediction_explicit_condition','prediction_explicit_prediction')
        keep_alias=next((k for k in aliases if k in fields),None) if any(k not in fields for k in ('prediction_base_velocity','prediction_ee_position')) else None
        fields=[k for k in fields if k not in aliases or k==keep_alias]
    if not run.get('_need_history',True):
        fields=[k for k in fields if k not in ('history_t','obs_t','prediction_grf_torque_input_nm')]
    # Reuse this stream; no second trajectory read or forward-dynamics solve.
    fields = sorted(set(fields) | (contact.FIELDS & set(run['records']['fields'])))
    momentum = contact.MomentumContact(meta, run.get('initial',{}), run.get('_contact_config',{}))
    run['_contact_files'] = []
    last, batch = {}, []
    for filename, arrays in chunks(run, fields):
        predicted_net = None
        if 'prediction_grf' in arrays:
            try: predicted_net = physical_predictions({'prediction_grf':arrays['prediction_grf']},meta,arrays)['grf'].sum(1)
            except (KeyError,ValueError): pass  # Only this optional metric is unavailable.
        reconstructed = momentum.process(arrays, predicted_net)
        if run.get('_contact_dir') is not None:
            run['_contact_files'].append(contact.save_reference_chunk(run['_contact_dir'],len(run['_contact_files']),
                {**reconstructed, **{k:arrays[k] for k in ('trial_id','step','time_seconds','successor_time_seconds','phase')}}))
        arrays.update({k:reconstructed[k] for k in contact.OUTPUT_FIELDS})
        run['_contact_report'] = dict(eligibility=dict(momentum.counts),notes=sorted(momentum.notes))
        # Validate dense numeric contracts once per chunk, not with tiny NumPy
        # dispatches for every transition. Only eligible/selected rows are enforced.
        timestamps_match = (np.isclose(arrays["time_seconds"], arrays["step"]*dt)
                            & np.isclose(arrays["successor_time_seconds"], (arrays["step"]+1)*dt))
        inference_ready=np.zeros(len(timestamps_match),bool)
        if ('history_t' in arrays and 'obs_t' in arrays and arrays['history_t'].shape[1:]==(width*slots,)
                and arrays['obs_t'].shape[1:]==(width,)):
            inference_ready=(np.isfinite(arrays['history_t']).all(1)&np.isfinite(arrays['obs_t']).all(1)
                &np.isclose(arrays['history_t'][:,-width:],arrays['obs_t'],atol=1e-6).all(1))
        forbidden=LOADS if meta['arguments']['condition']=='nominal' else LOADS[1:] if meta['arguments']['condition']=='ee-disturbance' else ()
        loads_valid=np.ones(len(timestamps_match),bool)
        for k in forbidden:
            if k in arrays: loads_valid &= ~(abs(arrays[k])>1e-5).any(1)
        for index, (tid, step) in enumerate(zip(arrays["trial_id"], arrays["step"])):
            tid, step = int(tid), int(step)
            previous, length, loads = last.get(tid, (-1, 0, None))
            if tid not in run["trials"] or step <= previous:
                raise ValueError("Unknown trial or duplicate/out-of-order transition")
            good = bool(arrays["valid"][index] and arrays["alive_for_metrics"][index]
                        and not arrays["physical_failure"][index] and not arrays["artificial_reset"][index])
            length = length+1 if good and step == previous+1 else int(good)
            last[tid] = (step, length, {k: arrays[k][index].copy() for k in LOADS if k in arrays})
            if length < 2 or loads is None or previous != step-1:
                continue  # Reset-filled histories and the first unseen interval are not eligible.
            stratum = (meta["arguments"]["load_range"], meta["arguments"]["condition"],
                run["trials"][tid]["motion_mode"], int(arrays["phase"][index]), meta["evaluation_seed"], run["name"])
            counts = coverage.setdefault(stratum, [0, 0])
            counts[0] += 1
            if selection is not None and (tid, step) not in selection:
                continue
            counts[1] += 1
            entry = dict(run=str(run["path"]), source=run["name"], trial_id=tid, step=step, chunk=filename, row=index)
            entry.update(dict(zip(("load_range", "condition", "motion", "phase", "evaluation_seed", "source"), stratum)))
            entry["transition_id"] = shared.fingerprint(entry)
            # Copy selected rows only; views would keep previous chunks alive.
            row = {k: v[index].copy() for k, v in arrays.items()}
            row.update({"history_"+k: loads[k] for k in LOADS if k in loads})
            row["entry"] = entry
            # History validity controls optional inference only, not stored-output metrics.
            row["inference_ready"] = bool(length>=slots and inference_ready[index])
            if not timestamps_match[index]:
                raise ValueError("Invalid interval/successor timestamp alignment")
            if not loads_valid[index]:
                raise ValueError("Applied loads violate the recorded condition")
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def build_model(run, device):
    """Same constructors/aliases as eval_b1z1_exp1.build_actor, without registry imports."""
    if not run["checkpoint"].is_file() or file_hash(run["checkpoint"]) != run["meta"]["checkpoint_sha256"]:
        raise ValueError("Missing or mismatched fallback checkpoint")
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
    norm=np.linalg.norm(q,axis=-1,keepdims=True)
    if not np.isfinite(q).all() or np.any(norm <= 0):
        raise ValueError("Invalid recorded quaternion")
    q = q / norm
    if yaw_only:
        x,y,z,w = np.moveaxis(q,-1,0)
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        q = np.stack([np.zeros_like(yaw),np.zeros_like(yaw),np.sin(yaw/2),np.cos(yaw/2)],axis=-1)
    while q.ndim < np.ndim(v): q=np.expand_dims(q,-2)
    xyz, w = q[...,:3], q[...,3:4]
    return v + 2*np.cross(xyz, np.cross(xyz, v) + w*v)


def physical_predictions(pred, meta, row):
    scales = meta["resolved_environment"]["normalization"]["obs_scales"]
    def scale(name):
        value = scales[name]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid native {name} normalization")
        return value
    kind, q = family(meta), row["base_quaternion_t"]
    values = {}
    if kind.startswith("pact"):
        if "prediction_ee_force" in pred:
            values["ee_force"] = rotate(pred["prediction_ee_force"]/scale("ee_force"), q, True)
        if "prediction_base_wrench" in pred:
            base = pred["prediction_base_wrench"] / scale("base_force")
            values.update(base_force=rotate(base[...,:3], q, True), base_moment=rotate(base[...,3:], q, True))
    elif "prediction_explicit" in pred:
        explicit = pred["prediction_explicit"]
        base_slice, ee_slice = (slice(9,12), slice(6,9)) if kind == "unifp_original" else (slice(6,9), slice(9,12))
        values = dict(ee_force=rotate(explicit[...,ee_slice]/scale("ee_force"), q, True),
                      base_force=rotate(explicit[...,base_slice]/scale("base_force"), q, True))
    if "prediction_grf" in pred:
        # Training GRF targets are successor-yaw vectors, unlike current force heads.
        grf=pred['prediction_grf']
        values["grf"] = rotate(grf.reshape(*grf.shape[:-1],-1,3)/scale("grf"), row["base_quaternion_xyzw"], True)
    return values


def dynamics_step(mass, bias, pre_v, torque, foot_j, ee_j, base_j, grf, ee, wrench, dt):
    """Frozen-mechanics analytic Euler velocity step, identical to BARD rollout objective."""
    generalized = np.r_[np.zeros(6), torque] + np.einsum("fkn,fk->n", foot_j[:,:3], grf)
    generalized += ee_j.T @ ee + base_j.T @ wrench
    return pre_v + dt*np.linalg.solve(mass, generalized-bias)


def offline_bard_class():
    """Load only the two Torch/BARD modules, bypassing simulator package initializers."""
    import importlib
    import importlib.machinery
    import sys
    import types
    name = "_exp3_offline_b1z1_dynamics"
    if name not in sys.modules:
        # A private namespace preserves their relative imports without importing
        # legged_gym.__init__ (which selects/imports Genesis or Isaac Lab).
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT/"legged_gym/dynamics")]
        package.__package__ = name
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
    return importlib.import_module(name+".bard_b1z1_dynamics").BardB1Z1DynamicsBackend


class OfflineBard:
    def __init__(self, run, device, urdf_override=None, batch_capacity=256):
        import torch
        BardB1Z1DynamicsBackend = offline_bard_class()
        m, asset = run["meta"], run["meta"]["resolved_environment"]["asset"]
        urdf = Path(urdf_override or m["urdf_source"])
        if file_hash(urdf) != m["urdf_sha256"]:
            raise ValueError("BARD URDF differs from recorded robot model")
        # Triton launches use the current CUDA device, not just tensor placement.
        # Restore the caller's device afterward; no global training-device change.
        with torch.cuda.device(device):
            self.backend = BardB1Z1DynamicsBackend(str(urdf), m["dof_names"], asset["foot_name"],
                asset["gripper_name"], asset["base_name"], device=device, batch_capacity=batch_capacity)
        self.run, self.device = run, device

    def terms(self, row):
        result = self.terms_batch([row])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def terms_batch(self, rows):
        """One BARD evaluation/device transfer per batch, not per transition."""
        import torch
        if not rows:
            return []
        b, m = self.backend, self.run["meta"]
        t = lambda key: torch.as_tensor(np.stack([r[key] for r in rows]), dtype=torch.float32, device=self.device).reshape(len(rows),-1)
        with torch.no_grad(), (torch.cuda.device(self.device) if torch.device(self.device).type=='cuda' else nullcontext()):
            terms = b.evaluate(t("base_position_t"), t("base_quaternion_t"), t("joint_position_t"),
                t("base_velocity_world_t"), t("base_angular_velocity_world_t"), t("joint_velocity_t"),
                torch.zeros(len(rows),4,3,device=self.device), torch.zeros(len(rows),3,device=self.device), torch.zeros(len(rows),6,device=self.device),
                t("mechanics_added_base_mass"), t("mechanics_base_com_bias"), t("mechanics_added_gripper_mass"))
            # Request FK pose from the already-updated BARD data, at q_t (no new rollout).
            _, pose = b.bard.jacobian(b.model, b.data, b.ee_frame_id, reference_frame="local", return_pose=True)
            poses = pose.cpu().numpy().copy()
            arrays = {k:getattr(terms,k).cpu().numpy().copy() for k in ("mass_matrix","bias","foot_jacobians","ee_jacobian","base_jacobian")}
        results = []
        for i, row in enumerate(rows):
            try:
                results.append(self._finish_terms(row, {k:v[i] for k,v in arrays.items()}, poses[i]))
            except (ValueError, RuntimeError, KeyError) as exc:
                results.append(exc)  # A malformed transition must not discard its valid neighbours.
        return results

    def _finish_terms(self, row, result, pose):
        m = self.run["meta"]
        initial = self.run["initial"][row["entry"]["trial_id"]]
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


def recorded_dynamics(row):
    """Optional SI/world-frame terms matching OfflineBard's contract, not velocities.

    The current collector does not save these fields; absence requests offline BARD.
    Never interpret a current-state decoder velocity as a successor prediction.
    """
    keys = ("mass_matrix", "bias", "foot_jacobians", "ee_jacobian", "base_jacobian", "ee_lever", "base_lever")
    if all("dynamics_"+k in row and np.isfinite(row["dynamics_"+k]).all() for k in keys):
        return {k:row["dynamics_"+k] for k in keys}
    return None


def stored_successor_error(row, variant):
    # Only explicit NEXT generalized world-velocity fields qualify; current estimates do not.
    key='prediction_next_velocity_'+variant
    fields=('base_velocity_world','base_angular_velocity_world','joint_velocity')
    if key not in row or not all(k in row for k in fields): return None
    actual=np.concatenate([row[k] for k in fields]); value=np.asarray(row[key])
    return value-actual if value.shape==actual.shape and np.isfinite(value).all() else None


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
        if field == 'grf_normal': labels = list(foot_names)
        if field == 'grf_net_horizontal': labels = ['x','y']
        if field in ("arm_velocity", "leg_velocity"):
            labels = [n for n in dof_names if n.startswith("z1_") == (field == "arm_velocity")]
        if field == 'explicit_base_velocity': labels = ['x','y']
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


STRATA = ["method", "benchmark", "variant", "load_range", "condition", "motion", "phase", "load_timing", "metric", "axis"]


class TrialStatistics:
    """Sums/counts for one recording, not one Python metric row per timestep."""
    def __init__(self, pooled=False):
        self.pooled = pooled
        self.sums = {}

    def merge(self,sums):
        for key,(squared,signed,count,labels) in sums.items():
            if key not in self.sums: self.sums[key]=[squared.copy(),signed.copy(),count,labels]
            else:
                self.sums[key][0]+=squared; self.sums[key][1]+=signed; self.sums[key][2]+=count

    def add_batch(self,entries,method,variant,errors,feet,dofs,timing):
        groups=defaultdict(list)
        for i,e in enumerate(entries):
            key=(e['trial_id'],e['motion'],e['phase'],timing[i]) if not self.pooled else (e['trial_id'],)
            groups[key].append(i)
        for field,error in errors.items():
            values=np.asarray(error,float).reshape(len(entries),-1)
            if not values.shape[1]: continue
            valid=np.isfinite(values).all(1)
            labels=None
            for ix in groups.values():
                ids=np.asarray(ix); ids=ids[valid[ids]]
                if not len(ids): continue
                e=entries[ids[0]]; x=values[ids]
                condition=('nominal' if e['condition']=='nominal' else 'disturbed') if self.pooled else e['condition']
                key=(method,'native',variant,e['load_range'],condition,
                    'all' if self.pooled else e['motion'],'all' if self.pooled else e['phase'],
                    'all' if self.pooled else timing[ids[0]],field+'_rmse',e['evaluation_seed'],e['source'],e['run'],e['trial_id'])
                sq=np.sum(x*x,axis=0); signed=np.sum(x,axis=0)
                sq=np.r_[sq.mean(),sq]; signed=np.r_[signed.mean(),signed]
                if key not in self.sums:
                    if labels is None: labels=[r['axis'] for r in metric_rows(e,method,'native',variant,{field:error[ids[0]]},feet,dofs,timing[ids[0]])]
                    self.sums[key]=[sq,signed,len(ids),labels]
                else:
                    self.sums[key][0]+=sq; self.sums[key][1]+=signed; self.sums[key][2]+=len(ids)

    def add(self, entry, method, variant, errors, feet, dofs, timing):
        for field, error in errors.items():
            error = np.asarray(error, dtype=float)
            if not error.size or not np.isfinite(error).all():
                continue
            values = error.ravel()
            squared = np.r_[np.mean(values**2), values**2]
            signed = np.r_[np.mean(values), values]
            condition = ("nominal" if entry["condition"] == "nominal" else "disturbed") if self.pooled else entry["condition"]
            key = (method, "native", variant, entry["load_range"], condition,
                   "all" if self.pooled else entry["motion"], "all" if self.pooled else entry["phase"],
                   "all" if self.pooled else timing, field+"_rmse", entry["evaluation_seed"], entry["source"], entry["run"], entry["trial_id"])
            if key not in self.sums:
                # Reuse the existing component/foot/joint labels exactly once per cell.
                labels = [r["axis"] for r in metric_rows(entry, method, "native", variant, {field: error}, feet, dofs, timing)]
                self.sums[key] = [squared, signed, 1, labels]
            else:
                cell = self.sums[key]
                cell[0] += squared
                cell[1] += signed
                cell[2] += 1

    def seeds(self):
        records = []
        for key, (squared, signed, count, labels) in self.sums.items():
            fields = dict(zip(STRATA[:-1]+["evaluation_seed", "source", "run", "trial_id"], key))
            for i, axis in enumerate(labels):
                records.append(dict(fields, axis=axis, rmse=np.sqrt(squared[i]/count), bias=signed[i]/count, samples=count))
        if not records:
            return []
        trials = pd.DataFrame(records)
        seeds = trials.groupby(STRATA+["evaluation_seed"], dropna=False).agg(
            mean=("rmse", "mean"), bias=("bias", "mean"), trials=("rmse", "size"), samples=("samples", "sum")).reset_index()
        return seeds.to_dict("records")


def merge_seed_statistics(records):
    """Merge run-level trial means with trial counts, preserving equal trial weight."""
    if not records:
        return [], []
    frame = pd.DataFrame(records)
    frame["weighted_mean"] = frame["mean"] * frame.trials
    frame["weighted_bias"] = frame.bias * frame.trials
    seeds = frame.groupby(STRATA+["evaluation_seed"], dropna=False)[["weighted_mean", "weighted_bias", "trials", "samples"]].sum().reset_index()
    seeds["mean"] = seeds.weighted_mean / seeds.trials
    seeds["bias"] = seeds.weighted_bias / seeds.trials
    seeds = seeds.drop(columns=["weighted_mean", "weighted_bias"])
    summary = []
    for key, group in seeds.groupby(STRATA, dropna=False):
        summary.append(dict(zip(STRATA, key), **shared.seed_statistics(group["mean"]), bias=shared.mean(group.bias),
                            contributing_trials=int(group.trials.sum()), contributing_samples=int(group.samples.sum())))
    return shared.clean(seeds.to_dict("records")), shared.clean(summary)


def run_analysis(config):
    # Tiny joint-space solves are slower with large BLAS thread teams. Restore
    # the caller's settings on exit; this does not change Torch/CUDA precision.
    threads = config.get("cpu_blas_threads", 1)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("cpu_blas_threads must be a positive integer")
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        import warnings
        warnings.warn("Install threadpoolctl to avoid oversubscribed small offline dynamics solves.")
        return _run_with_precision(config)
    with threadpool_limits(limits=threads, user_api="blas"):
        return _run_with_precision(config)


def _run_with_precision(config):
    # Recorded metrics must not import Torch or require a CUDA runtime.
    return _run_analysis(config)


def representation_signatures():
    """Numerical dependencies only: FiLM plots must not invalidate BARD caches."""
    from exp3_support import passive_bias
    import exp3_representation_batch as batch
    import scipy
    from importlib.metadata import version,PackageNotFoundError
    versions={}
    for package in ('bard-pytorch-dynamics','pin','torch'):
        try: versions[package]=version(package)
        except PackageNotFoundError: versions[package]=None
    physics=shared.fingerprint([inspect.getsource(f) for f in (OfflineBard,rotate,passive_bias)]+
        [file_hash(ROOT/'legged_gym/dynamics'/f) for f in ('bard_b1z1_dynamics.py','whole_body_dynamics.py')]+[versions,np.__version__])
    reduction=shared.fingerprint([inspect.getsource(f) for f in (native_batches,physical_predictions,
        TrialStatistics,recorded_dynamics,stored_successor_error,merge_seed_statistics,select_manifest,candidates,rollout_stats.TrialRMSE)]+
        [file_hash(Path(batch.__file__)),file_hash(Path(contact.__file__)),physics,np.__version__,scipy.__version__])
    return physics,reduction


def recording_signature(run):
    # Recordings are immutable. Hash small schema/mechanics files once; retain
    # size/mtime for numeric chunks so changed recordings cannot reuse results.
    paths=[run['path']/f for f in ('metadata.json','records.json','trials.json','initial_states.jsonl')]
    return shared.fingerprint(dict(files=[(str(p),file_hash(p) if p.exists() else None) for p in paths],
        chunks=[(c['file'],(run['path']/c['file']).stat().st_size,(run['path']/c['file']).stat().st_mtime_ns)
                for c in run['records']['chunks']]))


def export_contact(output, run, files, report):
    """Small reference arrays, independent of BARD/decoder dynamics and their caches."""
    meta=run['meta']
    contact.export_references(output,run['path'],files,dict(**report,task=meta['task'],
        checkpoint=meta['checkpoint_sha256'],evaluation_seed=meta['evaluation_seed'],
        urdf_sha256=meta.get('urdf_sha256'),nonfoot_threshold_n=run.get('_contact_config',{}).get('contact_nonfoot_threshold_n',1e-6),
        window_seconds=run.get('_contact_config',{}).get('contact_window_seconds',.1)))


def _run_analysis(config):
    analysis_started = time.perf_counter()
    timings = dict(decoder_fallback_seconds=0., dynamics_seconds=0., runs=[])
    config = resolve_methods(config)
    physics_signature,reduction_signature=representation_signatures()
    cache = NumericCache(config,signature=reduction_signature)
    mechanics_cache=NumericCache(config,signature=physics_signature)
    balanced = bool(config.get("balanced_resampling", False))
    recorded_only = bool(config.get("recorded_native_only", False))
    mode, source = ("balanced" if balanced else "full"), "recorded_first"
    batch_size = config.get("inference_batch_size", 256)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("inference_batch_size must be a positive integer")
    dynamics_batch_size = config.get("dynamics_batch_size", 256)
    if not isinstance(dynamics_batch_size, int) or isinstance(dynamics_batch_size, bool) or dynamics_batch_size < 1:
        raise ValueError("dynamics_batch_size must be a positive integer")
    minimum = config.get("minimum_per_stratum", 8)
    if minimum < 1:
        raise ValueError("minimum_per_stratum must be positive")
    metric_workers=config.get('metric_workers',1)
    if isinstance(metric_workers,bool) or not isinstance(metric_workers,int) or metric_workers<1:
        raise ValueError('metric_workers must be a positive integer')
    rollout_stats.options(config)
    # Keep earlier reinference outputs separate from recorded-first results.
    output = Path(config["output_dir"]).expanduser() / "representation" / f"{mode}_{source}"
    output.mkdir(parents=True, exist_ok=True)
    (output/"COMPLETE").unlink(missing_ok=True)
    # These generated exports are obsolete; never leave old correlations beside
    # a newly completed RMSE-inference result. Raw recordings/caches are untouched.
    for suffix in ('trials.csv.gz','seeds.csv','summary.csv','matched_seeds.csv','matched_summary.csv'):
        (output/('velocity_agreement_'+suffix)).unlink(missing_ok=True)
    runs, notices = discover(config)
    timings['discovery_seconds'] = time.perf_counter()-analysis_started
    for run in runs: run['_num_workers'] = config.get('num_workers',1)
    device = config.get("device", "cpu")
    groups = defaultdict(list)
    for run in runs:
        groups[run["name"]].append(run)
    method_runs = {name: group[0] for name, group in groups.items()}
    selections={}
    for name,group in groups.items():
        if balanced:
            sampled,_=select_manifest(group,config.get('sampling_budget'),config.get('sampling_seed',0),
                config.get('samples_per_stratum',32) if config.get('sampling_budget') is None else None,minimum)
            selection=defaultdict(set)
            for entry in sampled: selection[entry['run']].add((entry['trial_id'],entry['step']))
            selections[name]=selection
        else: selections[name]=None
    def support(run):
        selected=selections[run['name']]
        if selected is not None:
            ids=selected[str(run['path'])]
        else:
            key,saved=cache.lookup('rollout_valid_steps',recording_signature(run))
            if saved is not None: return saved
            ids=((e['trial_id'],e['step']) for _,e in candidates(run))
        by_trial=defaultdict(list)
        for tid,step in ids: by_trial[int(tid)].append(int(step))
        result={tid:np.sort(np.asarray(steps,dtype=np.int64)) for tid,steps in by_trial.items()}
        if selected is None: cache.save(key,result)
        return result
    preparation=time.perf_counter()
    plan=rollout_stats.plan_pairs(runs,config['methods'],support)
    timings['pair_preparation_seconds']=time.perf_counter()-preparation
    coverage, issues, assumptions, parity_checks = {}, {}, {}, {}
    all_seeds, all_table_seeds, provenance = [], [], []
    all_rmse_trials=[]
    timings.update(metric_seconds=0.,completed_run_hits=0)
    failed_fallback = set()
    prediction_sources = defaultdict(int)
    selected_count = 0

    def report(run, component, reason, entry=None):
        key = (run["name"], str(run["path"]), component, reason)
        cell = issues.setdefault(key, dict(method=run["name"], run=str(run["path"]), component=component,
                                          reason=reason, count=0, first_transition=entry["transition_id"] if entry else None))
        cell["count"] += 1  # Bounded diagnostic summaries, not millions of error strings.

    with ExitStack() as resources:
        manifest_file=resources.enter_context((output/'transition_manifest.jsonl').open('w'))
        trial_file=resources.enter_context(gzip.open(output/'rollout_rmse_trials.csv.gz','wt',compresslevel=1))
        trial_header=True
        def save_rmse(records):
            nonlocal trial_header
            if not records: return
            frame=pd.DataFrame(records)
            frame.to_csv(trial_file,index=False,header=trial_header); trial_header=False
            all_rmse_trials.extend(r for r in records if r['variant']=='decoded')
        # One BARD/GPU owner. Only bounded, pruned CPU metric batches go to workers.
        pool=resources.enter_context(ProcessPoolExecutor(max_workers=metric_workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=__import__('analysis_runtime').worker_init)) if metric_workers>1 else None
        for name, group in groups.items():
            selection=selections[name]
            model = None
            cached_bard, cached_bard_key = None, None
            for run in group:
                started = last_progress = time.monotonic()
                run_count = 0
                if config.get("progress", True):
                    print(f"Exp3: {name}: {run['path']}", flush=True)
                meta = run["meta"]
                stats, table_stats = TrialStatistics(), TrialStatistics(pooled=True)
                rmse=rollout_stats.TrialRMSE()
                windows=plan['windows'].get(str(run['path']))
                bard = None
                bard_attempted = False
                if not config.get('dynamics',True): report(run,'BARD dynamics','Explicitly disabled; constant-velocity reference only')
                parity_state = dict(status="recorded_first", batches=0, samples=0)
                parity_checks[str(run["path"])] = parity_state
                meta_hash, records_hash = file_hash(run["path"]/"metadata.json"), file_hash(run["path"]/"records.json")
                provenance.append(dict(method=name, path=str(run["path"]), checkpoint=str(run["checkpoint"]),
                    checkpoint_sha256=meta["checkpoint_sha256"], training_seed=meta.get("checkpoint_training_seed"),
                    evaluation_seed=meta["evaluation_seed"], metadata_sha256=meta_hash, records_sha256=records_hash))
                data_stamp=recording_signature(run)
                run['_minimal_fields']=True
                run['_contact_config']=config
                run['_need_history']=(config.get('fallback_predictions',True) and not recorded_only
                    and bool(missing_predictions(meta,run['records']['fields'])))
                run_options={k:config.get(k) for k in ('dynamics','fallback_predictions','recorded_native_only','device','urdf')}
                run_options['rmse_windows']=windows  # Counterpart failures/selection invalidate only dependent reductions.
                run_options['contact'] = ({k:config.get(k) for k in ('contact_window_seconds','contact_nonfoot_threshold_n')},
                                          contact.model_signature(meta,config.get('urdf')))
                fallback_dependencies=None
                if run['_need_history']:
                    fallback_dependencies=shared.fingerprint([inspect.getsource(build_model),inspect.getsource(infer)]+
                        [file_hash(ROOT/'rsl_rl/modules'/f) for f in ('actor_critic_b1z1_pact.py','actor_critic_b1z1_pact_pos.py','actor_critic_unifp.py','actor_critic_unifp_original.py')])
                run_key,completed=(cache.lookup('representation_runs',(name,data_stamp,run_options,fallback_dependencies,
                    None if selection is None else sorted(selection[str(run['path'])]))) if config.get('cache_completed_runs',True) else (None,None))
                piece=run_key.with_suffix('.jsonl.gz') if run_key is not None else None
                run['_contact_dir'] = (run_key.with_suffix('') if run_key is not None else
                    output/'contact_intervals'/shared.fingerprint(str(run['path']))[:16])
                if completed is not None and not all(Path(f).is_file() for f in completed.get('contact_files',[])): completed = None
                if completed is not None and piece.is_file():
                    with gzip.open(piece,'rt') as src: shutil.copyfileobj(src,manifest_file)
                    all_seeds.extend(completed['seeds']); all_table_seeds.extend(completed['table_seeds']); save_rmse(completed['rmse_trials'])
                    for k,v in completed['coverage'].items(): coverage[k]=v
                    for k,v in completed['issues'].items(): issues[k]=v
                    assumptions.update(completed['assumptions'])
                    for k,v in completed['prediction_sources'].items(): prediction_sources[k]+=v
                    parity_checks[str(run['path'])]=completed['parity']
                    export_contact(output,run,completed.get('contact_files',[]),completed.get('contact_report',{}))
                    selected_count+=completed['count']; timings['completed_run_hits']+=1
                    timings['runs'].append(dict(path=str(run['path']),transitions=completed['count'],wall_seconds=time.monotonic()-started,cached=True))
                    if config.get('progress',True): print(f"  reused {completed['count']:,} transitions and per-trial statistics",flush=True)
                    continue
                if piece is not None:
                    piece.parent.mkdir(parents=True,exist_ok=True)
                    piece_stream=gzip.open(piece,'wt',compresslevel=1)
                    resources.callback(piece_stream.close)
                else: piece_stream=None
                sources_before=dict(prediction_sources)
                pending=deque()
                def consume(result):
                    stats.merge(result['stats']); table_stats.merge(result['pooled'])
                    rmse.merge(result['rmse_sums'])
                    for component,reason,count,entry in result['issues']:
                        report(run,component,reason,entry); issues[(run['name'],str(run['path']),component,reason)]['count']+=count-1
                    for k,v in result['sources'].items(): prediction_sources[k]+=v
                for batch in native_batches(run, batch_size, None if selection is None else selection[str(run["path"])], coverage):
                    for row in batch:
                        line=json.dumps(row["entry"], separators=(",", ":"), allow_nan=False)+"\n"
                        manifest_file.write(line)
                        if piece_stream is not None: piece_stream.write(line)
                    selected_count += len(batch)
                    usable = batch
                    # Each field keeps its stored value; inference fills missing fields only.
                    predictions = [{k:v for k,v in row.items() if k.startswith("prediction_")} for row in usable]
                    missing = [i for i,row in enumerate(usable) if missing_predictions(meta,predictions[i])
                               and row.get("inference_ready", False)]
                    if missing and config.get("fallback_predictions", True) and not recorded_only and name not in failed_fallback:
                        stage_started = time.perf_counter()
                        try:
                            missing_rows = [usable[i] for i in missing]
                            key, inferred = cache.lookup('decoder_fallback', (fallback_dependencies,meta, missing_rows, device))
                            if inferred is None:
                                if model is None:
                                    model = build_model(run, device)
                                inferred = infer(model, missing_rows, device)
                                cache.save(key, inferred)
                            for j,i in enumerate(missing):
                                for key,value in inferred.items():
                                    if key not in predictions[i]:
                                        predictions[i][key] = value[j]
                                        prediction_sources["fallback:"+key] += 1
                        except (ImportError, OSError, ValueError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError, AttributeError, TypeError) as exc:
                            failed_fallback.add(name)
                            report(run, "optional decoder fallback", str(exc))
                        timings['decoder_fallback_seconds'] += time.perf_counter()-stage_started
                    for row in usable:
                        for key in row:
                            if key.startswith("prediction_"):
                                prediction_sources["stored:"+key] += 1
                    # BARD is lazy and optional. Complete recorded terms bypass construction.
                    stage_started = time.perf_counter()
                    fixed_batch = [recorded_dynamics(row) if config.get('dynamics',True) else None for row in usable]
                    stored_terms=[f is not None for f in fixed_batch]
                    terms_needed = any(fixed is None and not all(stored_successor_error(row,v) is not None for v in ('decoded','measured_loading')) for fixed,row in zip(fixed_batch,usable))
                    terms_key, cached_terms = None, None
                    if config.get('dynamics',True) and terms_needed:
                        # The immutable recording signature already covers mechanics
                        # and states. Do not recursively rehash histories/decoders.
                        ids=np.array([(r['entry']['trial_id'],r['entry']['step']) for r in usable],dtype=np.int64)
                        terms_key, cached_terms = mechanics_cache.lookup('bard_batch_v2', (data_stamp,ids,config.get('urdf'),device))
                        if cached_terms is not None:
                            fixed_batch = unpack_terms(cached_terms)
                            terms_needed = False
                    dynamics_inputs=('joint_position_t','joint_velocity_t','base_position_t','base_quaternion_t','base_velocity_world_t','base_angular_velocity_world_t','executed_torque_interval_mean')
                    can_build=any(all(k in row for k in dynamics_inputs) and int(row['trial_id']) in run['initial'] for row in usable)
                    if config.get("dynamics", True) and terms_needed and not can_build:
                        report(run,'offline dynamics','Required recorded states or mechanics unavailable')
                    if config.get("dynamics", True) and terms_needed and can_build and not bard_attempted:
                        bard_attempted = True
                        try:
                            bard_key=shared.fingerprint(dict(urdf=config.get('urdf') or meta.get('urdf_source'),sha=meta.get('urdf_sha256'),dofs=meta['dof_names'],asset=meta['resolved_environment']['asset']))
                            if cached_bard is None or cached_bard_key!=bard_key:
                                cached_bard=OfflineBard(run, device, config.get("urdf"), batch_capacity=dynamics_batch_size)
                                cached_bard_key=bard_key
                            bard=cached_bard
                            bard.run=run
                        except (ImportError, ValueError, RuntimeError, OSError, KeyError) as exc:
                            report(run, "optional BARD", str(exc))
                    if bard is not None and terms_needed:
                        missing_terms=[i for i,row in enumerate(usable) if fixed_batch[i] is None
                                       and not all(stored_successor_error(row,v) is not None for v in ("decoded","measured_loading"))]
                        for start in range(0,len(missing_terms),dynamics_batch_size):
                            indices=missing_terms[start:start+dynamics_batch_size]
                            rows=[usable[i] for i in indices]
                            try:
                                terms=bard.terms_batch(rows)
                            except (ValueError, RuntimeError, KeyError):
                                terms=[]
                                for row in rows:
                                    try: terms.append(bard.terms(row))
                                    except (ValueError, RuntimeError, KeyError) as exc: terms.append(exc)
                            for i,term in zip(indices,terms): fixed_batch[i]=term
                        if all(isinstance(term,dict) for term in fixed_batch):
                            mechanics_cache.save(terms_key, pack_terms(fixed_batch))
                    timings['dynamics_seconds'] += time.perf_counter()-stage_started
                    for i,fixed in enumerate(fixed_batch):
                        if stored_terms[i]: prediction_sources['stored:dynamics_terms']+=1
                        elif isinstance(fixed,dict):
                            key=(str(run['path']),tuple(fixed.get('approximation',[])))
                            assumptions[key]=assumptions.get(key,0)+1
                            prediction_sources[('cached:' if cached_terms is not None else 'offline:')+'BARD_terms']+=1
                        elif isinstance(fixed,Exception):
                            report(run,'BARD mechanics',str(fixed),usable[i]['entry']); fixed_batch[i]=None
                    stage_started=time.perf_counter()
                    fields={'entry','base_quaternion_t','base_quaternion_xyzw','executed_torque_interval_mean',
                        'grf_raw_world_interval_mean','surface_normals_world',*contact.OUTPUT_FIELDS,*STATE_KEYS,*LOADS,*('history_'+k for k in LOADS),*PREDICTIONS[4:6],*EXPLICIT_STATE_FIELDS}
                    masks=None if windows is None else rollout_stats.batch_windows(windows,[r['entry'] for r in usable])
                    job=([{k:v for k,v in r.items() if k in fields} for r in usable],predictions,fixed_batch,meta,name,config.get('dynamics',True),masks)
                    if pool is None: consume(process_batch(job))
                    else:
                        pending.append(pool.submit(process_batch,pack_job(job)))
                        if len(pending)>=2*metric_workers: consume(pending.popleft().result())
                    timings['metric_seconds']+=time.perf_counter()-stage_started
                    run_count += len(usable)
                    now = time.monotonic()
                    if config.get("progress", True) and now-last_progress >= 30:
                        print(f"  {run_count:,} transitions; {run_count/max(now-started, 1e-9):.1f}/s; {(now-started)/60:.1f} min", flush=True)
                        last_progress = now
                while pending: consume(pending.popleft().result())
                export_contact(output,run,run.get('_contact_files',[]),run.get('_contact_report',{}))
                for note in run.get('_contact_report',{}).get('notes',[]):
                    if any(word in note for word in ('unavailable','unrecorded','mismatch')):
                        report(run,'momentum reference',note)
                if piece_stream is not None: piece_stream.close()
                run_seeds,run_table=stats.seeds(),table_stats.seeds()
                run_rmse=rmse.records(run)
                all_seeds.extend(run_seeds); all_table_seeds.extend(run_table); save_rmse(run_rmse)
                run_issues={k:v for k,v in issues.items() if k[:2]==(name,str(run['path']))}
                # Do not freeze failed optional dependencies into reusable results.
                reusable=not any(k[2] in ('optional BARD','optional decoder fallback','offline dynamics','BARD mechanics') for k in run_issues)
                if reusable:
                    cache.save(run_key,dict(seeds=run_seeds,table_seeds=run_table,rmse_trials=run_rmse,count=run_count,
                        contact_files=run.get('_contact_files',[]),contact_report=run.get('_contact_report',{}),
                        coverage={k:v for k,v in coverage.items() if k[1]==meta['arguments']['condition'] and k[0]==meta['arguments']['load_range'] and k[-2:]==(meta['evaluation_seed'],name)},
                        issues=run_issues,assumptions={k:v for k,v in assumptions.items() if k[0]==str(run['path'])},
                        prediction_sources={k:v-sources_before.get(k,0) for k,v in prediction_sources.items()},parity=parity_state))
                timings['runs'].append(dict(path=str(run['path']), transitions=run_count,
                    wall_seconds=time.monotonic()-started,
                    chunk_read_worker_seconds=run.get('_chunk_read_seconds',0.)))
                del bard
            del model, cached_bard  # At most one checkpoint and one compatible BARD backend.

    if not selected_count:
        raise ValueError("No complete, valid native histories within selected recordings")
    # Optional inference failure never invalidates stored predictions from any batch.
    per_seed, breakdown = merge_seed_statistics(all_seeds)
    table_seeds, table_summary = merge_seed_statistics(all_table_seeds)
    from exp3_support import hierarchy
    hierarchical_rows = []
    for item in table_seeds:
        meta = method_runs[item["method"]]["meta"]
        hierarchical_rows.append(dict(item, task=meta["task"], checkpoint=meta["checkpoint_sha256"],
            training_seed=meta.get("checkpoint_training_seed"), seed=item["evaluation_seed"], trial=0, value=item["mean"]))
    comparative = hierarchy(pd.DataFrame(hierarchical_rows),
        ["task", "benchmark", "variant", "load_range", "condition", "metric", "axis"])
    balance = [dict(stratum=list(key), available=counts[0], selected=counts[1], minimum=minimum,
                    sufficient=counts[1] >= minimum) for key, counts in sorted(coverage.items())]
    notices.extend(f"Insufficient coverage: {b}" for b in balance if not b["sufficient"])
    notices.extend(f"{name}: optional inference unavailable; stored metrics retained" for name in sorted(failed_fallback))
    manifest = dict(schema_version=VERSION, selection_mode=mode, prediction_source=source,
        balanced_resampling=balanced, sampling_seed=config.get("sampling_seed", 0) if balanced else None,
        sampling_budget=config.get("sampling_budget") if balanced else None,
        samples_per_stratum=config.get("samples_per_stratum", 32) if balanced else None,
        available=sum(v[0] for v in coverage.values()), selected=selected_count, strata=balance,
        transitions_file="transition_manifest.jsonl", transitions_sha256=file_hash(output/"transition_manifest.jsonl"))
    shared.write_json(output/"transition_manifest.json", manifest)
    statistics_started=time.perf_counter()
    paired=shared.clean(rollout_stats.paired_results(all_rmse_trials,plan,config))
    pd.DataFrame(paired.pop('trials')).to_csv(output/'paired_rollout_trials.csv.gz',index=False,compression='gzip')
    pd.DataFrame(plan['audit']).to_csv(output/'paired_rollout_matching.csv',index=False)
    paired.update(trial_file='paired_rollout_trials.csv.gz',matching_file='paired_rollout_matching.csv',
        rmse_trial_file='rollout_rmse_trials.csv.gz')
    timings['paired_statistics_seconds']=time.perf_counter()-statistics_started
    timings.update(total_seconds=time.perf_counter()-analysis_started, cache_hits=cache.hits+mechanics_cache.hits, cache_misses=cache.misses+mechanics_cache.misses)
    result = dict(schema_version=VERSION, status="complete", experiment=3, output_dir=str(output),
        timings=timings,
        protocol=shared.protocol(runs[0]["meta"]), configuration=config,
        selection=manifest, selection_mode=mode, prediction_source=source, recorded_native_only=recorded_only,
        prediction_sources=dict(prediction_sources), inference_device=device, inference_batch_size=batch_size,
        methods={name: dict(task=spec["task"], checkpoint=str(method_runs[name]["checkpoint"]) if name in method_runs else spec.get("checkpoint"),
            checkpoint_sha256=method_runs[name]["meta"]["checkpoint_sha256"] if name in method_runs else None,
            training_seed=method_runs[name]["meta"].get("checkpoint_training_seed") if name in method_runs else None)
            for name, spec in config["methods"].items()},
        provenance=provenance, per_seed=per_seed, breakdown=breakdown, table_per_seed=table_seeds, table_summary=table_summary,
        comparative_summary=comparative.to_dict("records"), native_parity=parity_checks,
        unavailable=list(issues.values()), diagnostics=sorted(set(notices)), coverage=balance,
        dynamics_approximations=[dict(run=run, assumptions=list(notes), transitions=count) for (run, notes), count in assumptions.items()],
        paired_rollout=paired,
        units={**UNITS, "arm_velocity_rmse":"rad/s", "leg_velocity_rmse":"rad/s"},
        conventions=dict(benchmark="Each checkpoint's own trajectories; matched conditions, different realized transitions; NOT shared-transition comparison.",
            selection="All eligible native transitions" if not balanced else "Deterministic balanced subsampling WITHOUT replacement within checkpoint and loading range",
            force_target="Previous applied interval visible to native history_t; world SI at body CoM",
            grf_target="Legacy vector metric: Error Against Normal-Only Sensor Forces. New normal metric projects onto flat-ground z (or recorded surface normals). Net metric: Momentum-Derived Net Contact Force, foot-only eligible intervals/windows; componentwise RMSE, not vector-norm RMSE. Recorded nominal torque and successor-yaw decoder conversion retained.",
            dynamics="q_t mechanics, recorded interval-average executed torque; passive damping/armature with labelled friction/stiffness approximations",
            explicit_state="Current t: common body vx/vy in m/s (PACT's third estimate is yaw rate; UniFP's is vz); spherical EE estimate converted to yaw-workspace Cartesian meters. Componentwise RMSE, prediction minus measurement, not target tracking.",
            missing="Missing heads never replaced by measured loading in the decoded-loading variant"))
    pd.DataFrame(per_seed).to_csv(output/"per_seed.csv", index=False)
    for level,key in (('seeds','per_seed'),('summary','summary')):
        pd.DataFrame(paired[key]).to_csv(output/('paired_rollout_'+level+'.csv'),index=False)
    pd.DataFrame(breakdown).to_csv(output/"breakdown.csv", index=False)
    pd.DataFrame(table_summary).to_csv(output/"summary.csv", index=False)
    comparative.to_csv(output/"checkpoint_summary.csv", index=False)
    export_tables(result, config, output)
    timings['total_seconds'] = time.perf_counter()-analysis_started
    shared.write_json(output/'timings.json',timings)
    shared.write_json(output/"summary.json", result)
    (output/"COMPLETE").write_text("Experiment 3 native-trajectory representation analysis complete\n")
    return result


def export_tables(result, config, output):
    for scope in ("native",):
        for load in sorted({b["stratum"][0] for b in result["coverage"]}):
            records = []
            seen_tasks = set()
            for method in result["methods"]:
                task = result["methods"][method]["task"]
                if task in seen_tasks:
                    continue
                seen_tasks.add(task)
                row = {"Method":config["methods"][method].get("display_name",method)}
                for condition in ("nominal","disturbed"):
                    for metric in MAIN:
                        item = next((r for r in result["comparative_summary"] if r["task"]==task and r["benchmark"]==scope and r["load_range"]==load and r["condition"]==condition and r["metric"]==metric and r["axis"]=="all" and r["variant"]=="decoded"),None)
                        row[(condition,metric)] = shared.formatted(dict(item,unit=UNITS[metric]) if item else None)
                records.append(row)
            if not records:
                continue
            frame = pd.DataFrame(records).set_index("Method")
            labels={'explicit_base_velocity_rmse':'explicit base velocity (xy)', 'explicit_ee_position_rmse':'explicit ee position (Cartesian)',
                    'grf_rmse':'Error Against Normal-Only Sensor Forces', 'grf_normal_rmse':'per-foot normal GRF',
                    'grf_net_rmse':'net contact (xyz)', 'grf_net_horizontal_rmse':'net contact (xy)',
                    'grf_net_window_rmse':'net contact (window)'}
            frame.columns = pd.MultiIndex.from_tuples([(c, f"{labels.get(m,m.replace('_rmse','').replace('_',' '))} [{UNITS[m]}]") for c,m in frame.columns])
            name = scope.replace(":","_")+"_"+load
            frame.to_csv(output/(name+"_table.csv"))
            macros = config.get("latex_method_names",{})
            tokens = {config["methods"][m].get("display_name",m):f"METHOD{i}TOKEN" for i,m in enumerate(macros) if m in result["methods"]}
            tex = frame.rename(index=tokens).to_latex(escape=True,multicolumn_format="c")
            for m,macro in macros.items():
                if m in result["methods"]:
                    tex=tex.replace(tokens[config["methods"][m].get("display_name",m)],macro)
            tex=tex.replace("±",r"$\pm$").replace("—","---")
            (output/(name+"_table.tex")).write_text("\\begin{table*}[t]\n\\centering\n\\caption{Native-trajectory representation and one-step velocity RMSE; "+load+". Matched experimental conditions but different realized trajectories, not shared-transition comparisons. Equal trials, evaluation seeds, then checkpoints; sample SD across checkpoints when available, otherwise evaluation-seed SD. Missing heads/failed validation are unavailable.}\n\\resizebox{\\textwidth}{!}{%\n"+tex+"}\n\\end{table*}\n")

