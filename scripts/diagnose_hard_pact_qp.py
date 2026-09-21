#!/usr/bin/env python3
"""Bounded actual-training capture or simulator-free trusted-packet replay."""
import argparse
import copy
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    capture = sub.add_parser("capture", aliases=["timing"])
    capture.add_argument("--checkpoint", type=Path, required=True)
    capture.add_argument("--resolved-config", type=Path, help="defaults to checkpoint sibling JSON")
    capture.add_argument("--backend", choices=("isaaclab", "genesis"), default="isaaclab")
    capture.add_argument("--iteration-limit", type=int, default=3)
    capture.add_argument("--warmup-iterations", type=int, default=2)
    capture.add_argument("--capture-limit", type=int, default=32)
    capture.add_argument("--byte-limit-mib", type=int, default=2048)
    capture.add_argument("--torque-violation-trigger", type=float, default=1000.)
    replay = sub.add_parser("replay")
    replay.add_argument("packets", type=Path, nargs="+")
    replay.add_argument("--individual-row-limit", type=int, default=4)
    replay.add_argument("--kkt-row-limit", type=int, default=0)
    for p in (capture, replay):
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for key in ("iteration_limit", "capture_limit", "byte_limit_mib", "torque_violation_trigger"):
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(key + " must be positive")
    if getattr(args, "individual_row_limit", 0) < 0:
        parser.error("individual-row-limit must be nonnegative")
    if getattr(args,"warmup_iterations",0)<0 or not 0<=getattr(args,"kkt_row_limit",0)<=16:
        parser.error("warmup must be nonnegative; kkt-row-limit must be 0..16")
    return args


def configure_diagnostic_warmup(runner, count):
    """Absolute offset, shared rollout/PPO gate; never reset the resumed epoch."""
    activation = runner.current_learning_iteration + count
    runner.alg.qp_config = replace(runner.alg.qp_config,warmup_iterations=activation)
    runner.alg.hard_pact_qp.cfg = replace(runner.alg.hard_pact_qp.cfg,warmup_iterations=activation)
    runner._set_hard_pact_qp_iteration(runner.current_learning_iteration)
    return activation


def capture_run(args):
    # Reuse normal runtime preparation, task registration, runner load/learn.
    # Do not reduce environment count, rollout length, epochs or solver chunks.
    from scripts.eval_hard_pact_frozen import apply_config, sha_file, dependency_versions, write_json
    import torch
    checkpoint = args.checkpoint.resolve(strict=True)
    config_path = (args.resolved_config or checkpoint.parent / "hard_pact_resolved_config.json").resolve(strict=True)
    output = args.output_dir.resolve()
    if output == checkpoint.parent or checkpoint.parent in output.parents:
        raise ValueError("Use an output directory outside the original run")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Capture output directory must be empty")
    document = json.loads(config_path.read_text())
    task = document["task"]
    if not task.endswith("_" + args.backend):
        raise ValueError("Backend must match the resolved task; do not change experimental settings")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: actual simulator/PPO capture was not run")
    os.environ["SIMULATOR"] = args.backend
    from legged_gym.scripts.train_hard_pact import prepare_solver_runtime
    prepare_solver_runtime(["--qp_solver", "cupiqp"])
    import legged_gym.envs
    from legged_gym.utils import get_args, task_registry, save_hard_pact_resolved_config
    from rsl_rl.algorithms.hard_pact_qp_diagnose import QPCapture
    env_cfg = copy.deepcopy(task_registry.env_cfgs[task])
    train_cfg = copy.deepcopy(task_registry.train_cfgs[task])
    apply_config(env_cfg, document["environment"]["resolved"])
    apply_config(train_cfg, document["training"]["resolved"])
    settings = train_cfg.algorithm.hard_pact_qp
    if settings.get("qp_solver", "qpth") != "cupiqp" or any(
            settings.get(k) not in (None, "cupiqp") for k in ("rollout_qp_solver", "ppo_qp_solver")):
        raise ValueError("Capture requires original cuPIQP configuration")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    activation = int(saved["iter"]) + args.warmup_iterations
    overrides = {"warmup_iterations": activation, "runner.resume": False, "policy.pretrained_path": None,
                 "runtime_observers":"CUDA event timing; gradient hooks only in capture mode"}
    # New captures use current constraints, while offline replay always retains
    # old packets' exact matrices. Record this explicit resolved-config upgrade.
    if "joint_acceleration_limits_rad_s2" in settings:
        overrides["removed_joint_acceleration_limits_rad_s2"] = settings.pop("joint_acceleration_limits_rad_s2")
    old_control = document["environment"]["resolved"].get("control", {})
    if "torque_rate_limit_nm_s" not in old_control and "torque_rate_limit_nm_s" in settings:
        env_cfg.control.torque_rate_limit_nm_s = settings["torque_rate_limit_nm_s"]
        overrides["control.torque_rate_limit_nm_s"] = env_cfg.control.torque_rate_limit_nm_s
    overrides["constraint_schema_version"] = 2
    settings["constraint_schema_version"] = 2
    settings["warmup_iterations"] = activation
    # Our bounded recorder handles exceptions too; avoid a second unbudgeted
    # legacy snapshot (and any writes to the original configured directory).
    settings["exception_capture_enabled"] = False
    overrides["exception_capture_enabled"] = False
    train_cfg.runner.resume = False
    train_cfg.policy.pretrained_path = None
    sys.argv = [sys.argv[0], "--task", task, "--headless", "--gpu", args.device,
                "--seed", str(env_cfg.seed)]
    launch_args = get_args()
    if args.backend == "genesis":
        from legged_gym import gs
        from legged_gym.utils import init_genesis
        init_genesis(launch_args, gs)
    identity = {"checkpoint":str(checkpoint), "checkpoint_sha256":sha_file(checkpoint),
        "resolved_config":document, "config_sha256":sha_file(config_path),
        "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "dependencies":dependency_versions(), "seed":env_cfg.seed, "overrides":overrides}
    identity["joint_names"] = list(env_cfg.asset.dof_names)
    env, _ = task_registry.make_env(task, launch_args, env_cfg=env_cfg)
    try:
        runner, _ = task_registry.make_alg_runner(env, task, launch_args, train_cfg=train_cfg, log_root=str(output / "run"))
        if not runner.alg.hard_pact_features.execution_qp:
            raise ValueError("Resolved ablation disables QP; refusing to silently change it")
        keys = ("act_optimizer_state_dict", "enc_optimizer_state_dict", "decoder_opt_state_dict")
        runner.load(str(checkpoint), load_optimizer=all(k in saved for k in keys))
        # Some historical checkpoints omit curriculum metadata; normal load
        # then resets iteration. Diagnosis must retain the recorded PPO epoch.
        runner.current_learning_iteration = int(saved["iter"])
        runner.alg._last_completed_iteration = runner.current_learning_iteration - 1
        runner._set_hard_pact_qp_iteration(runner.current_learning_iteration)
        # Legacy partial checkpoints are explicit, never silently called full resumes.
        identity["optimizer_states_present"] = {k:k in saved for k in keys}
        if not all(k in saved for k in keys):
            if keys[0] in saved:
                runner.alg.load_actor_optimizer_state(saved[keys[0]])
            runner.alg.load_auxiliary_optimizer_states(
                saved.get(keys[1],runner.alg.enc_optimizer.state_dict()),
                saved.get(keys[2],runner.alg.decoder_optimizer.state_dict()))
            identity["resume_limitation"] = "Incomplete optimizer checkpoint; only compatible available states restored"
        identity["start_iteration"] = runner.current_learning_iteration
        identity["resume_state_limitations"] = "Simulator state, solver caches and training RNG are not restored by normal runner.load"
        recorder = QPCapture(output / "captures", limit=args.capture_limit,
            byte_limit=args.byte_limit_mib*1024**2, trigger_nm=args.torque_violation_trigger, identity=identity)
        qp = runner.alg.hard_pact_qp
        if qp is None:
            raise ValueError("Resolved configuration did not construct a QP; capture cannot continue")
        qp.diagnostic_iteration = lambda: getattr(env, "_terrain_curriculum_iteration", runner.current_learning_iteration)
        activation = configure_diagnostic_warmup(runner,args.warmup_iterations)
        identity["periods"] = dict(warmup_start=runner.current_learning_iteration,qp_activation=activation,
            post_warmup_end=activation+args.iteration_limit,mode=args.mode)
        overrides["warmup_iterations"] = activation
        from rsl_rl.algorithms.hard_pact_qp_diagnose_runtime import diagnostic_runtime
        save_hard_pact_resolved_config(output, task, env_cfg, train_cfg, effective_train_cfg=runner.train_cfg)
        write_json(output / "identity.json", identity)
        with diagnostic_runtime(runner,output,gradients=args.mode!="timing") as runtime:
            try:
                for offset in range(args.warmup_iterations+args.iteration_limit):
                    warmup = offset < args.warmup_iterations
                    period = "warmup_qp_disabled" if warmup else (
                        "steady_state_capture_disabled" if args.mode=="timing" else "capture_overhead_included")
                    if not warmup and args.mode!="timing":
                        qp.diagnostic_capture = recorder
                        for backend in qp._backend_instances.values(): backend.capture_details_enabled=True
                    runtime.run_iteration(period)
                runner.save(str(output / "diagnostic_checkpoint.pt"))
            finally:
                write_json(output / "capture_summary.json",recorder.summary())
    finally:
        app = getattr(getattr(env.simulator, "_app_launcher", None), "app", None)
        if app is not None:
            app.close()


def sliced(packet, row):
    result = dict(packet)
    result["tensors"] = {k:(v[row:row+1] if hasattr(v,"ndim") and v.ndim>0 else v)
                         for k,v in packet["tensors"].items()}
    result["problem"] = {k:(v[row:row+1] if hasattr(v,"ndim") and v.ndim>0 and k!="variable_scale" else v)
                         for k,v in packet["problem"].items()}
    if packet.get("raw_primal") is not None:
        result["raw_primal"] = packet["raw_primal"][row:row+1]
    if "data" in packet: result["data"]={k:v[row:row+1] for k,v in packet["data"].items()}
    for key in ("rows","accepted"):
        if hasattr(packet.get(key), "shape"):
            result[key]=packet[key][row:row+1]
    return result


def replay_one(packet, device, change, row=None, kkt_rows=0):
    import torch
    from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig
    from rsl_rl.algorithms.hard_pact_qp_backends import create_backend
    from rsl_rl.algorithms.hard_pact_qp_diagnose import independent_checks, candidate_assessment, distribution, conditioning_audit
    config = dict(packet["config"])
    if packet.get("schema_version",1) < 3:
        config.pop("joint_acceleration_limits_rad_s2",None)
    cfg = HardPACTQPConfig.from_dict(config)
    diff = bool(packet["differentiable"])
    prefix = "ppo" if diff else "rollout"
    if change == "other_tolerances":
        other = "rollout" if diff else "ppo"
        cfg = replace(cfg, **{prefix+"_"+k:getattr(cfg,other+"_"+k) for k in ("eps_abs","eps_rel")})
    elif change == "gap_toggle":
        cfg = replace(cfg, **{prefix+"_duality_gap_policy":"report" if getattr(cfg,prefix+"_duality_gap_policy")=="require" else "require"})
    elif change == "iterations_x2":
        cfg = replace(cfg, **{prefix+"_max_iter":2*getattr(cfg,prefix+"_max_iter")})
    elif change == "gradient_toggle":
        other = "rollout" if diff else "ppo"
        cfg = replace(cfg, **{other+"_"+k:getattr(cfg,prefix+"_"+k) for k in (
            "eps_abs","eps_rel","duality_gap_abs","duality_gap_rel","duality_gap_policy","max_iter")})
        diff = not diff
    if change == "fresh":
        cfg = replace(cfg,cupiqp_ppo_reuse=False)
    backend = create_backend(packet["solver"], cfg)
    backend.capture_details_enabled = True
    def solve(p, differentiate):
        tensors = {k:(v.to(device=device, dtype=torch.float64 if change=="float64" and v.is_floating_point() else v.dtype).clone()
                      if torch.is_tensor(v) else v) for k,v in p["tensors"].items()}
        for name in ("Q", "p", "G", "h", "A", "b"):
            tensors[name].requires_grad_(differentiate)
        with torch.set_grad_enabled(differentiate):
            out = backend.solve(*(tensors[k] for k in ("Q","p","G","h","A","b")),
                native_lower=tensors.get("native_lower"), native_upper=tensors.get("native_upper"),
                differentiable=differentiate, constant_hessian=False)
        return out, tensors
    # Truncated numerical update sequence, not an assertion of exact cache or
    # outstanding-PCGrad lease replay. Full-batch only; individual replay is fresh.
    if change == "reuse_sequence" and row is None:
        for previous in packet.get("preceding_updates", []):
            prior, _ = solve(previous, bool(previous["differentiable"]))
            del prior
    if row is not None:
        packet = sliced(packet, row)
    out, tensors = solve(packet, diff)
    checks = independent_checks(packet, out.solution)
    assessment = candidate_assessment(packet,out.solution,out.duality_gap,out.duality_gap_rel,cfg,diff)
    production_mask = assessment["production_accepted"]
    backward = {}
    if diff:
        # Same deterministic upstream across variants. Test both all certified
        # rows and a mixed keep/drop mask. Failed-row adjoints are zero BEFORE VJP.
        upstream = torch.arange(1,out.solution.shape[-1]+1,device=device,dtype=out.solution.dtype)[None].expand_as(out.solution)
        for name, mask in (("production_accepted",production_mask),
                           ("mixed",production_mask & (torch.arange(out.solution.shape[0],device=device)%2==0))):
            names = ("Q", "p", "G", "h", "A", "b")
            backward[name+"_vjp_rows"] = int(mask.sum())
            try:
                gradients = torch.autograd.grad(out.solution, [tensors[k] for k in names],
                    grad_outputs=torch.where(mask[:,None],upstream,0.), retain_graph=True, allow_unused=True)
            except Exception as error:
                backward[name+"_backward_exception"] = repr(error)
                continue
            backward[name+"_finite"] = all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
            backward[name+"_excluded_zero"] = all(g is None or bool((g[~mask]==0).all()) for g in gradients)
            for k,g in zip(names,gradients):
                backward[name+"_"+k+"_gradient_max"] = None if g is None or not g.numel() else float(g.abs().max())
                backward[name+"_"+k+"_gradient_distribution"] = None if g is None else distribution(g.abs())
    reference = packet.get("raw_primal")
    reference_valid = None
    difference = None
    if reference is not None:
        reference = reference.to(out.solution)
        reference_valid = independent_checks(packet,reference)["primal_feasible"]
        joint = reference_valid & checks["primal_feasible"]
        if bool(joint.any()):
            difference = float((reference[joint]-out.solution.detach()[joint]).abs().max())
    info = out.snapshot or {}
    iterations = info.get("iter")
    status = info.get("status")
    if iterations is not None:iterations=iterations[:out.solution.shape[0]]
    if status is not None:status=status[:out.solution.shape[0]]
    audit_packet = dict(packet,assessment=assessment,accepted=production_mask,
                        failing_rows=(~production_mask).nonzero().flatten())
    return dict(schema_version=3,stage=packet["stage"],phase=packet["phase"],variant=change,row=row,
        row_identity=packet.get("rows"),captured_acceptance=packet.get("accepted"),assessment=assessment,
        conditioning=conditioning_audit(audit_packet,out.solution,kkt_rows) if kkt_rows else None,
        dtype=str(out.solution.dtype),differentiable=diff,
        effective_eps_abs=getattr(cfg,("ppo" if diff else "rollout")+"_eps_abs"),
        effective_max_iter=getattr(cfg,("ppo" if diff else "rollout")+"_max_iter"),
        effective_gap_policy=getattr(cfg,("ppo" if diff else "rollout")+"_duality_gap_policy"),
        real_rows=out.solution.shape[0],primal_feasible_rows=int(checks["primal_feasible"].sum()),
        production_accepted_rows=int(production_mask.sum()),
        finite_rows=int(checks["finite"].sum()),raw_torque_violation_nm=float(checks["torque_violation_nm"].max()),
        equality_residual=float(checks["equality_residual"].max()),inequality_residual=float(checks["inequality_residual"].max()),
        gap=None if out.duality_gap is None else float(out.duality_gap.max()),
        relative_gap=None if out.duality_gap_rel is None else float(out.duality_gap_rel.max()),
        status=status,iterations=iterations,
        iterations_max=None if iterations is None else int(iterations.max()),
        numerical_failure_rows=None if status is None else int((status==4).sum()),
        reuse_history_complete=False,
        reference_primal_feasible_rows=None if reference_valid is None else int(reference_valid.sum()),
        reference_solution_difference=difference,reference_is_ground_truth=False,**backward)


def replay_csv_row(report):
    """Scalar, physical-unit joint summaries; detailed row arrays stay in JSON.

    All/production-accepted populations are separate. Distribution counts and
    nonfinite counts are explicit; unavailable values become empty CSV cells.
    """
    import math
    import torch
    from rsl_rl.algorithms.hard_pact_qp_diagnose import distribution
    row = {k:v for k,v in report.items() if v is None or isinstance(v,(str,int,float,bool))}
    def scalar(key, value):
        if torch.is_tensor(value): value = value.item()
        row[key] = None if isinstance(value,float) and not math.isfinite(value) else value
    def stats(prefix, values):
        for key,value in values.items():
            if key == "percentiles":
                for i,label in enumerate(("p50","p95","p99")):
                    scalar(prefix+"/"+label,None if value is None else value[i])
            else: scalar(prefix+"/"+key,value)
    assessment=report.get("assessment",{})
    joint=assessment.get("joint")
    row["joint_metrics_available"] = joint is not None
    if joint is None:
        row["joint_metrics_unavailable_reason"] = assessment.get("joint_unavailable",report.get("error","not recorded"))
        return row
    names=joint["names"]
    accepted=assessment["production_accepted"].bool()
    for scope,mask in (("all",torch.ones_like(accepted)),("accepted",accepted)):
        prefix="joint/"+scope
        scalar(prefix+"/rows",mask.sum())
        scalar(prefix+"/empty_intersection_rows",joint["empty"][mask].any(-1).sum())
        scalar(prefix+"/empty_intersection_coordinates",joint["empty"][mask].sum())
        hard=assessment.get("original_hard_joint_satisfied")
        if hard is not None:scalar(prefix+"/original_hard_satisfied_rows",hard[mask].sum())
        for key,unit_name in (("acceleration","acceleration_rad_s2"),("q_next","q_next_rad"),
                             ("dq_next","dq_next_rad_s"),("slack_rad_s2","recovery_slack_rad_s2"),
                             ("rate_slack_nm","recovery_rate_slack_nm"),("rate_violation_nm","rate_violation_nm"),
                             ("conflict_rad_s2","conflict_rad_s2"),("lower","acceleration_lower_rad_s2"),
                             ("upper","acceleration_upper_rad_s2")):
            if key not in joint:  # Historical reports lack torque-rate slack.
                continue
            values=joint[key][mask]
            stats(prefix+"/"+unit_name,distribution(values))
            for j,name in enumerate(names):
                stats(prefix+"/"+name+"/"+unit_name,distribution(values[:,j]))
    # Existing violation statistics already use the correct populations and
    # finite-coordinate reductions. Do not average minibatch means here.
    for family,populations in joint["statistics"].items():
        for scope,values in populations.items():
            prefix="joint/"+scope+"/violation/"+family
            stats(prefix,values["aggregate"])
            for name,value in zip(names,values["per_joint"]): stats(prefix+"/"+name,value)
    return row


def write_replay_csv(path, reports):
    rows=[replay_csv_row(report) for report in reports]
    keys=sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def replay_run(args):
    import torch
    from scripts.eval_hard_pact_frozen import write_json
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in args.packets:
        packet = torch.load(path,map_location="cpu",weights_only=True)
        if packet.get("schema_version") not in (1,2,3) or "problem" not in packet:
            raise ValueError("Unsupported packet; use diagnose_hard_pact_qp capture")
        packet["preceding_updates"] = [torch.load(path.parent/p["packet_file"],map_location="cpu",weights_only=True)
            if "packet_file" in p else p for p in packet.get("preceding_updates",[])]
        from rsl_rl.algorithms.hard_pact_qp_diagnose import select_replay_rows
        for row in [None]+select_replay_rows(packet,args.individual_row_limit):
            for change in ("baseline","other_tolerances","gap_toggle","iterations_x2","gradient_toggle","fresh","reuse_sequence","float64"):
                try:
                    result = replay_one(packet,args.device,change,row,args.kkt_row_limit)
                except Exception as error:
                    result = dict(stage=packet["stage"],phase=packet["phase"],variant=change,row=row,error=repr(error))
                reports.append(dict(packet=str(path),**result))
    write_json(args.output_dir/"replay.json",reports)
    write_replay_csv(args.output_dir/"replay.csv",reports)
    print(f"{len(reports)} replay cases; {sum('error' in r for r in reports)} errors; {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    (replay_run if args.mode == "replay" else capture_run)(args)
