#!/usr/bin/env python3
"""Isolated benchmark harness for the canonical HardPACT QP.

Each cell runs in a fresh child process so a CUDA OOM or optional-backend
failure cannot poison later measurements. The primary solve sets chunk size to
the requested batch size and therefore never silently sub-batches.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types
import importlib.util
import importlib.metadata

os.environ.setdefault("SIMULATOR", "genesis_pact")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_hard_pact_benchmark")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_hard_pact_benchmark")


MODES = (
    "assembly", "fused_bard_assembly", "rollout", "sequential_rollout",
    "ppo_forward", "ppo_backward", "complete_iteration",
)


def load_qp_module_without_simulator():
    """Load the solver modules without importing a simulator framework.

    Captured canonical-QP benchmarks intentionally run in isolated solver
    environments. Importing ``rsl_rl.algorithms`` normally initializes every
    PPO class and therefore Genesis/Isaac Lab, which is unrelated to this
    solver-only measurement. This local package shell loads exactly the two
    implementation files under test and does not alter production imports.
    """
    root = Path(__file__).resolve().parents[1]
    package_paths = {
        "rsl_rl": root / "rsl_rl",
        "rsl_rl.algorithms": root / "rsl_rl" / "algorithms",
    }
    for name, path in package_paths.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    for name in ("hard_pact_qp_backends", "hard_pact_qp"):
        qualified = f"rsl_rl.algorithms.{name}"
        if qualified not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                qualified, package_paths["rsl_rl.algorithms"] / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
    return sys.modules["rsl_rl.algorithms.hard_pact_qp"]


def classify_failure(text):
    lowered = text.lower()
    if "out of memory" in lowered or "cuda_error_out_of_memory" in lowered:
        return "OOM"
    if (
        ("backend" in lowered and ("unavailable" in lowered or "not installed" in lowered))
        or "failed to import" in lowered
        or "modulenotfounderror" in lowered
        or "no license key found" in lowered
    ):
        return "unsupported_dependency"
    if "timeout" in lowered:
        return "timeout"
    if "residual" in lowered or "numer" in lowered or "infeasible" in lowered:
        return "numerical_failure"
    return "unexpected_error"


def synthetic_data(batch, device, dtype, iteration=0):
    import torch
    eye = torch.eye(18, device=device, dtype=dtype).expand(batch, -1, -1).clone()
    foot = torch.zeros(batch, 4, 3, 18, device=device, dtype=dtype)
    # A stable synthetic floating-base contact layout, changed slightly per
    # iteration without changing dimensions or sparsity.
    foot[:, :, :, :3] = torch.eye(3, device=device, dtype=dtype)
    base = torch.zeros(batch, 6, 18, device=device, dtype=dtype)
    base[:, :, :6] = torch.eye(6, device=device, dtype=dtype)
    phase = 0.01 * iteration
    return {
        "mass_matrix": eye,
        "bias": torch.zeros(batch, 18, device=device, dtype=dtype),
        "foot_jacobians": foot,
        "base_jacobian": base,
        "foot_acceleration_bias": torch.zeros(batch, 4, 3, device=device, dtype=dtype),
        "tau_nom": torch.full((batch, 12), phase, device=device, dtype=dtype),
        "force_pred_world": torch.zeros(batch, 4, 3, device=device, dtype=dtype),
        "wrench_pred_world": torch.zeros(batch, 6, device=device, dtype=dtype),
        "contact_probability": torch.full((batch, 4), 0.7, device=device, dtype=dtype),
        "previous_torque": torch.zeros(batch, 12, device=device, dtype=dtype),
        "joint_position": torch.zeros(batch, 12, device=device, dtype=dtype),
        "joint_velocity": torch.zeros(batch, 12, device=device, dtype=dtype),
        "dt": torch.full((batch, 1), 0.002, device=device, dtype=dtype),
    }


def synchronize(device):
    import torch
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory_stats(device):
    """Reset allocator peaks when supported by the installed Torch build."""
    import torch
    if torch.device(device).type != "cuda":
        return False
    try:
        # Torch 2.8 accepts an integer ordinal here but, unlike most CUDA
        # memory APIs, rejects an otherwise valid ``torch.device('cuda:0')``.
        # Normalize explicitly so peak VRAM is reported instead of silently
        # becoming unavailable on the simulator's production Torch build.
        ordinal = torch.device(device).index
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        # The CUDA context must exist before Torch 2.8 accepts a reset.  This
        # initialization is outside every timed solver region.
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(ordinal)
        return True
    except RuntimeError:
        # Some externally managed CUDA allocators expose allocation counters
        # but reject resetPeakMemoryStats. Solves remain benchmarkable; the
        # result marks peak memory unavailable instead of aborting the cell.
        return False


def worker(args):
    import torch
    qp_module = load_qp_module_without_simulator()
    HardPACTDifferentiableQP = qp_module.HardPACTDifferentiableQP
    HardPACTQPConfig = qp_module.HardPACTQPConfig
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    config = HardPACTQPConfig(
        qp_solver=args.solver,
        solver_dtype=args.dtype,
        rollout_chunk_size=args.batch_size,
        ppo_chunk_size=args.batch_size,
        diagnostics_level="minimal",
        qpth_warm_start=bool(args.qpth_warm_start),
        cupiqp_mode=args.cupiqp_mode,
    )
    start = time.perf_counter()
    qp = HardPACTDifferentiableQP(
        config, torch.full((12,), 23.5),
        torch.full((12,), -2.0), torch.full((12,), 2.0),
        torch.full((12,), 30.0),
    )
    construction_ms = (time.perf_counter() - start) * 1000.0
    timings = []
    forward_timings = []
    backward_timings = []
    first_solve_ms = None
    stages = []
    equality = []
    inequality = []
    solver_exceptions = []
    full_equality = []
    memory_stats_available = reset_peak_memory_stats(device)
    total = args.warmup + args.iterations
    for index in range(total):
        data = synthetic_data(args.batch_size, device, dtype, index)
        if args.mode == "fused_bard_assembly":
            raise RuntimeError(
                "backend unavailable: fused BARD benchmark requires a simulator/BARD capture"
            )
        differentiable = args.mode.startswith("ppo") or args.mode == "complete_iteration"
        if differentiable:
            for key in ("tau_nom", "force_pred_world", "wrench_pred_world", "contact_probability"):
                data[key].requires_grad_(True)
        synchronize(device)
        started = time.perf_counter()
        forward_started = time.perf_counter()
        if args.mode == "assembly":
            qp._build(data)
            result = None
        elif args.mode == "sequential_rollout":
            # Four changing substeps model one control interval. The output
            # torque is the next substep's exact per-environment rate center.
            result = None
            for substep in range(4):
                data["tau_nom"] = data["tau_nom"] + 0.01 * substep
                result = qp.solve(differentiable=False, **data)
                data["previous_torque"] = result.tau_safe.detach()
        elif args.mode == "complete_iteration":
            # Solver-only representative: four no-grad rollout substeps then
            # one differentiable sampled-QP forward/backward. This is clearly
            # distinct from the real simulator iteration benchmark.
            with torch.inference_mode():
                rollout_data = {key: value.detach() for key, value in data.items()}
                for _ in range(4):
                    rollout_result = qp.solve(differentiable=False, **rollout_data)
                    rollout_data["previous_torque"] = rollout_result.tau_safe
            result = qp.solve(differentiable=True, **data)
        else:
            result = qp.solve(differentiable=differentiable, **data)
        synchronize(device)
        forward_elapsed = (time.perf_counter() - forward_started) * 1000.0
        backward_elapsed = 0.0
        if args.mode in ("ppo_backward", "complete_iteration"):
            backward_started = time.perf_counter()
            result.tau_safe.square().mean().backward()
            synchronize(device)
            backward_elapsed = (time.perf_counter() - backward_started) * 1000.0
        synchronize(device)
        elapsed = (time.perf_counter() - started) * 1000.0
        if first_solve_ms is None and args.mode != "assembly":
            first_solve_ms = elapsed
        if index >= args.warmup:
            timings.append(elapsed)
            forward_timings.append(forward_elapsed)
            backward_timings.append(backward_elapsed)
            if result is not None:
                stages.append(torch.bincount(result.stage, minlength=3).cpu().tolist())
                equality.append(float(result.diagnostics["selected/equality_max"].nan_to_num().max()))
                inequality.append(float(result.diagnostics["selected/inequality_max"].nan_to_num().max()))
                solver_exceptions.append(float(
                    result.diagnostics.get("full/solver_exception", torch.zeros(1, device=device)).float().mean()
                ))
                if "full/equality_max" in result.diagnostics:
                    full_equality.append(float(
                        result.diagnostics["full/equality_max"].nan_to_num(nan=float("inf")).max()
                    ))
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    stage_total = [sum(row[i] for row in stages) for i in range(3)] if stages else [0, 0, 0]
    denom = max(1, sum(stage_total))
    backend_failed = bool(
        stages and stage_total[2] == denom
        and max(solver_exceptions, default=0.0) >= 1.0
    )
    metadata = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "simulator": "captured canonical QP (none)",
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        ).stdout.strip() or "unknown",
    }
    for package in ("qpth", "cupiqp", "moreau", "bard"):
        try:
            metadata[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata[f"{package}_version"] = None
    return {
        "status": "numerical_failure" if backend_failed else "success",
        "solver": args.solver, "mode": args.mode,
        "batch_size": args.batch_size, "dtype": args.dtype,
        "qpth_warm_start": bool(args.qpth_warm_start),
        "cupiqp_mode": args.cupiqp_mode,
        "construction_ms": construction_ms,
        "first_solve_setup_inclusive_ms": first_solve_ms,
        "mean_ms": statistics.mean(timings), "median_ms": statistics.median(timings),
        "p95_ms": p95, "std_ms": statistics.pstdev(timings),
        "qps_per_second": args.batch_size / (statistics.mean(timings) / 1000.0),
        "ms_per_qp": statistics.mean(timings) / args.batch_size,
        "forward_mean_ms": statistics.mean(forward_timings),
        "backward_mean_ms": statistics.mean(backward_timings),
        "full_fraction": stage_total[0] / denom,
        "relaxed_fraction": stage_total[1] / denom,
        "analytic_fraction": stage_total[2] / denom,
        "equality_residual_max": max(equality, default=0.0),
        "inequality_residual_max": max(inequality, default=0.0),
        "solver_exception_fraction": max(solver_exceptions, default=0.0),
        "full_equality_residual_max": max(full_equality, default=0.0),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated(
                torch.device(device).index or 0
            ) / 2**20
            if memory_stats_available else None
        ),
        "peak_cuda_reserved_mib": (
            torch.cuda.max_memory_reserved(
                torch.device(device).index or 0
            ) / 2**20
            if memory_stats_available else None
        ),
        "metadata": metadata,
    }


def run_cell(script, solver, batch, mode, args):
    command = [
        sys.executable, str(script), "--worker", "--solver", solver,
        "--batch-size", str(batch), "--mode", mode, "--dtype", args.dtype,
        "--device", args.device, "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
    ]
    if args.qpth_warm_start:
        command.append("--qpth-warm-start")
    command.extend(("--cupiqp-mode", args.cupiqp_mode))
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=args.timeout
        )
    except subprocess.TimeoutExpired as error:
        return {"solver": solver, "batch_size": batch, "mode": mode,
                "dtype": args.dtype, "status": "timeout", "error": str(error)}
    if result.returncode:
        message = (result.stderr + "\n" + result.stdout)[-4000:]
        return {"solver": solver, "batch_size": batch, "mode": mode,
                "dtype": args.dtype, "status": classify_failure(message), "error": message}
    return json.loads(result.stdout.strip().splitlines()[-1])


def write_results(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hard_pact_qp_benchmark.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    flat = [{k: v for k, v in row.items() if k != "metadata"} for row in rows]
    keys = sorted({key for row in flat for key in row})
    with (output_dir / "hard_pact_qp_benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(flat)
    lines = [
        "# Go2 HardPACT standalone canonical-QP solver throughput and VRAM benchmark",
        "", "## Experiment", "",
        "- Script: `scripts/benchmark_hard_pact_qp.py`.",
        "- Conditions: captured canonical 54-variable HardPACT QPs; the table records solver, mode, batch, dtype, GPU, package versions, certification, fallback rates, and VRAM; every cell uses a fresh child process.",
        "- Purpose: isolate whether each backend is fast, differentiable, certified, and memory-efficient enough for per-substep rollout and PPO replay. These are solver-only results, not simulator iteration timings.",
        "", "## Results", "",
        "| Solver | Mode | Batch | Status | Mean ms | QPs/s | Peak MiB |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        peak = row.get('peak_cuda_allocated_mib')
        peak_text = "n/a" if peak is None else f"{peak:.1f}"
        lines.append(
            f"| {row['solver']} | {row['mode']} | {row['batch_size']} | {row['status']} | "
            f"{row.get('mean_ms', float('nan')):.3f} | {row.get('qps_per_second', float('nan')):.1f} | "
            f"{peak_text} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_matrix(args, script=None):
    """Run all cells, isolating and containing OOM state per solver/mode."""
    script = Path(__file__).resolve() if script is None else Path(script)
    rows, skipped = [], set()
    for solver in args.solvers.split(","):
        for mode in args.modes.split(","):
            for batch in map(int, args.batch_sizes.split(",")):
                if (solver, mode) in skipped:
                    rows.append({"solver": solver, "mode": mode, "batch_size": batch,
                                 "dtype": args.dtype, "status": "skipped_after_oom"})
                    continue
                row = run_cell(script, solver, batch, mode, args)
                rows.append(row)
                if row["status"] == "OOM":
                    skipped.add((solver, mode))
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solvers", default="qpth,cupiqp,moreau")
    parser.add_argument("--batch-sizes", default="256,512,1024,2048,4096")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--solver")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument(
        "--qpth-warm-start", action="store_true",
        help="enable the repository-local qpth rollout warm start",
    )
    parser.add_argument(
        "--cupiqp-mode", choices=("dense", "sparse"), default="dense",
        help="select the official cuPIQP internal matrix representation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
        return
    rows = run_matrix(args)
    write_results(rows, args.output_dir)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
