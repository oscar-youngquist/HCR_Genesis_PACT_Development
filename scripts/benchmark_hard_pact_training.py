#!/usr/bin/env python3
"""OOM-tolerant real-training benchmark for HardPACT QP backends.

Every cell launches the repository's normal backend-specific HardPACT task in
a fresh process. Rollout and PPO chunk sizes are independent because rollout runs
under inference mode whereas PPO retains an implicit-differentiation graph.
The search starts at the largest requested pair and walks downward only after
failure; thus a successful 4096/4096 cell proves the largest pair immediately.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time

from benchmark_hard_pact_qp import classify_failure


ITERATION_RE = re.compile(r"Iteration time:\s*([0-9.]+)s")
STEP_RE = re.compile(r"Computation:\s*([0-9.]+) steps/s")
BARD_INVERSE_RE = re.compile(
    r"BARD inverse forward:\s*([0-9.]+) ms/update \(([0-9.]+) ms/minibatch\)"
)
BARD_ROLLOUT_RE = re.compile(
    r"BARD rollout forward:\s*([0-9.]+) ms/update \(([0-9.]+) ms/minibatch\)"
)


def command_for(args, solver, rollout_chunk, ppo_chunk):
    launcher = Path(__file__).resolve().parent / "go2_hard_pact.sh"
    command = [
        str(launcher), "--task", args.task, "--headless",
        "--num_envs", str(args.num_envs),
        "--max_iterations", str(args.iterations),
        "--seed", str(args.seed),
        "--gpu", args.device,
        "--qp_solver", solver,
        "--qp_rollout_chunk_size", str(rollout_chunk),
        "--qp_ppo_chunk_size", str(ppo_chunk),
    ]
    # Moreau 0.3's compiled solver supports float64 only. This explicit
    # override leaves the required auto policy (CUDA->float32, CPU->float64)
    # unchanged for qpth/cuPIQP and makes the Moreau registration executable
    # when a licensed Python-3.12 simulator environment is available.
    if solver == "moreau":
        command.extend(("--qp_solver_dtype", "float64"))
    if getattr(args, "profile_bard_timing", False):
        command.append("--profile_bard_timing")
    if getattr(args, "bard_active_from_start", False):
        command.extend(("--benchmark_bard_active", "--pinn_loss_weight", "-1"))
    if getattr(args, "bard_batch_capacity", None) is not None:
        command.extend(("--bard_batch_capacity", str(args.bard_batch_capacity)))
    return command


def run_cell(args, solver, rollout_chunk, ppo_chunk):
    command = command_for(args, solver, rollout_chunk, ppo_chunk)
    started = time.perf_counter()
    logical_gpu_index = int(args.device.rsplit(":", 1)[-1])
    # nvidia-smi addresses physical devices, whereas Torch honors
    # CUDA_VISIBLE_DEVICES. Resolve the logical training ordinal so memory
    # samples describe the GPU that actually owns the simulator process.
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_entries = [entry.strip() for entry in visible_devices.split(",") if entry.strip()]
    if visible_entries and logical_gpu_index < len(visible_entries):
        selected = visible_entries[logical_gpu_index]
        gpu_index = int(selected) if selected.isdigit() else selected
    else:
        gpu_index = logical_gpu_index

    def gpu_memory_mib():
        """Read device memory without importing a second CUDA runtime."""
        try:
            value = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_index),
                 "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5.0, check=True,
            ).stdout.strip().splitlines()[0]
            return float(value)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    baseline_memory = gpu_memory_mib()
    peak_memory = baseline_memory
    # A file avoids a PIPE deadlock from Isaac Sim's large startup log while
    # the parent periodically samples VRAM. start_new_session lets a timeout
    # terminate the launcher's complete simulator process tree.
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=Path(__file__).resolve().parents[1],
            stdout=stream, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            while process.poll() is None:
                if time.perf_counter() - started > args.timeout:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                measured = gpu_memory_mib()
                if measured is not None:
                    peak_memory = measured if peak_memory is None else max(peak_memory, measured)
                time.sleep(1.0)
        except BaseException:
            # Never strand an Isaac Sim child (and its multi-GiB CUDA context)
            # when the benchmark parent is interrupted or errors while
            # sampling telemetry. The child owns a fresh process group.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        stream.seek(0)
        output = stream.read()
    if timed_out:
        return {
            "solver": solver, "rollout_chunk_size": rollout_chunk,
            "ppo_chunk_size": ppo_chunk, "status": "timeout",
            "wall_seconds": time.perf_counter() - started,
            "error": f"exceeded {args.timeout:.1f}s", "command": command,
            "peak_gpu_memory_mib": peak_memory,
        }
    row = {
        "solver": solver, "rollout_chunk_size": rollout_chunk,
        "ppo_chunk_size": ppo_chunk,
        "wall_seconds": time.perf_counter() - started,
        "command": command,
        "baseline_gpu_memory_mib": baseline_memory,
        "peak_gpu_memory_mib": peak_memory,
        "peak_incremental_gpu_memory_mib": (
            peak_memory - baseline_memory
            if peak_memory is not None and baseline_memory is not None else None
        ),
        "python": platform.python_version(),
        "backend": args.task.rsplit("_", 1)[-1],
    }
    for package in ("torch", "qpth", "cupiqp", "moreau", "bard"):
        try:
            row[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            row[f"{package}_version"] = None
    try:
        row["git_sha"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=5.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        row["git_sha"] = None
    try:
        row["gpu_name"] = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=name",
             "--format=csv,noheader"], capture_output=True, text=True,
            check=True, timeout=5.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        row["gpu_name"] = None
    if process.returncode:
        row.update(status=classify_failure(output), error=output[-8000:])
        return row
    iteration_times = [float(value) for value in ITERATION_RE.findall(output)]
    throughputs = [float(value) for value in STEP_RE.findall(output)]
    inverse_timings = [tuple(map(float, values)) for values in BARD_INVERSE_RE.findall(output)]
    rollout_timings = [tuple(map(float, values)) for values in BARD_ROLLOUT_RE.findall(output)]
    row.update(
        status="success",
        completed_iterations=len(iteration_times),
        mean_iteration_seconds=(
            sum(iteration_times) / len(iteration_times)
            if iteration_times else None
        ),
        std_iteration_seconds=(
            statistics.pstdev(iteration_times) if iteration_times else None
        ),
        mean_steps_per_second=(
            sum(throughputs) / len(throughputs) if throughputs else None
        ),
        mean_bard_inverse_ms_per_update=(
            statistics.mean(value[0] for value in inverse_timings)
            if inverse_timings else None
        ),
        mean_bard_inverse_ms_per_minibatch=(
            statistics.mean(value[1] for value in inverse_timings)
            if inverse_timings else None
        ),
        mean_bard_rollout_ms_per_update=(
            statistics.mean(value[0] for value in rollout_timings)
            if rollout_timings else None
        ),
        mean_bard_rollout_ms_per_minibatch=(
            statistics.mean(value[1] for value in rollout_timings)
            if rollout_timings else None
        ),
    )
    if len(iteration_times) != args.iterations:
        classified = classify_failure(output)
        row.update(
            status=(
                classified if classified != "unexpected_error"
                else "unexpected_error"
            ),
            error=(
                f"expected {args.iterations} iteration records, got "
                f"{len(iteration_times)}\n--- output tail ---\n{output[-12000:]}"
            ),
        )
    return row


def search_solver(args, solver, on_result=None):
    """Find the lexicographically largest VRAM-safe rollout/PPO pair."""
    sizes = sorted({int(value) for value in args.chunk_sizes.split(",")}, reverse=True)
    rows = []

    def record(row):
        # Full Isaac cells can take hours. Persist each result as soon as its
        # child exits so a later interruption cannot discard an already
        # measured OOM, capability failure, or successful five-iteration run.
        rows.append(row)
        if on_result is not None:
            on_result(row)
        return row

    # First test matching chunk sizes from largest to smallest. If one fits,
    # all smaller diagonal pairs are dominated for the requested max search.
    diagonal_success = None
    for size in sizes:
        row = record(run_cell(args, solver, size, size))
        if row["status"] == "success":
            diagonal_success = size
            break
        if row["status"] == "unsupported_dependency":
            return rows
    if diagonal_success is None:
        return rows
    # A diagonal success at the maximum already proves both maxima. For a
    # smaller success, independently probe larger rollout and PPO dimensions
    # while holding the other coordinate at the known-safe value.
    safe_anchor = diagonal_success
    best_rollout = safe_anchor
    for rollout in [value for value in sizes if value > safe_anchor]:
        row = record(run_cell(args, solver, rollout, diagonal_success))
        if row["status"] == "success":
            best_rollout = rollout
            break
    # Probe PPO independently at the safe diagonal anchor. Mutating the anchor
    # after a rollout success would incorrectly skip every larger PPO size.
    for ppo in [value for value in sizes if value > safe_anchor]:
        row = record(run_cell(args, solver, safe_anchor, ppo))
        if row["status"] == "success":
            break
    return rows


def write_results(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hard_pact_training_benchmark.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    flat = [{**row, "command": " ".join(row["command"])} for row in rows]
    fields = sorted({field for row in flat for field in row})
    with (output_dir / "hard_pact_training_benchmark.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    command = " ".join(rows[0]["command"]) if rows else "not run"
    lines = [
        "# Go2 HardPACT full Isaac Lab training benchmark: QP and BARD PINN", "",
        "## Experiment", "",
        "- Script: `scripts/benchmark_hard_pact_training.py`.",
        f"- Training command: `{command}`.",
        "- Conditions: registered full HardPACT task, real simulator, the requested environment/iteration counts shown in the command, per-substep QP execution, differentiable QP replay, and both BARD PINN objectives when requested. GPU memory is sampled externally.",
        "- Purpose: measure end-to-end suitability for the intended training workload. Unlike the standalone benchmark, this includes simulation, rollout QPs, BARD dynamics, PPO, PCGrad, and optimizer work.",
        "", "## Results", "",
        "| Solver | Rollout chunk | PPO chunk | Status | Iteration mean s | Iteration std s | Steps/s | BARD inverse ms/update | BARD rollout ms/update |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        mean = row.get("mean_iteration_seconds")
        std = row.get("std_iteration_seconds")
        steps = row.get("mean_steps_per_second")
        inverse = row.get("mean_bard_inverse_ms_per_update")
        rollout = row.get("mean_bard_rollout_ms_per_update")
        lines.append(
            f"| {row['solver']} | {row['rollout_chunk_size']} | "
            f"{row['ppo_chunk_size']} | {row['status']} | "
            f"{float('nan') if mean is None else mean:.3f} | "
            f"{float('nan') if std is None else std:.3f} | "
            f"{float('nan') if steps is None else steps:.1f} | "
            f"{float('nan') if inverse is None else inverse:.3f} | "
            f"{float('nan') if rollout is None else rollout:.3f} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solvers", default="qpth,cupiqp,moreau")
    parser.add_argument("--chunk-sizes", default="256,512,1024,2048,4096")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default="go2_hard_pact_full_genesis")
    parser.add_argument("--profile-bard-timing", action="store_true")
    parser.add_argument("--bard-active-from-start", action="store_true")
    parser.add_argument("--bard-batch-capacity", type=int, default=None)
    # qpth at 4096 Isaac environments can legitimately need well over an
    # hour for five full rollout/PPO iterations. Keep a per-cell four-hour
    # guard so genuine hangs are bounded without misclassifying slow solves.
    parser.add_argument("--timeout", type=float, default=14400.0)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/hard_pact_training"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        for solver in args.solvers.split(","):
            print(" ".join(command_for(args, solver, 4096, 4096)))
        return
    rows = []
    def checkpoint(row):
        rows.append(row)
        write_results(rows, args.output_dir)

    for solver in args.solvers.split(","):
        search_solver(args, solver, on_result=checkpoint)
    write_results(rows, args.output_dir)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
