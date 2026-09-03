#!/usr/bin/env python3
"""Run matched full-training BARD and Pinocchio HardPACT comparisons."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import tempfile
import time


ITERATION = re.compile(r"Iteration time:\s*([0-9.]+)s")
BREAKDOWN = re.compile(r"collection:\s*([0-9.]+)s, learning\s*([0-9.]+)s")
PINN = re.compile(r"PINN loss:\s*([-+0-9.eE]+)")
INVERSE = re.compile(r"Dynamics inverse forward:\s*([0-9.]+) ms/update")
ROLLOUT = re.compile(r"Dynamics rollout forward:\s*([0-9.]+) ms/update")
DYNAMICS = re.compile(r"Dynamics total:\s*([0-9.]+) ms/update")
TRANSFER = re.compile(r"Pinocchio transfer:\s*([0-9.]+) ms/update")


def summaries(values):
    tail = values[1:]
    return {
        "values": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "mean_excluding_iteration_1": statistics.mean(tail),
        "median_excluding_iteration_1": statistics.median(tail),
    }


def run(args, backend):
    command = [
        str(Path(__file__).resolve().parents[1]
            / "legged_gym" / "scripts" / "go2_hard_pact.sh"),
        "--task", args.task, "--headless", "--num_envs", str(args.num_envs),
        "--max_iterations", str(args.iterations), "--seed", str(args.seed),
        "--gpu", args.device, "--dynamics_backend", backend,
        "--bard_batch_capacity", str(args.batch_capacity),
        "--profile_bard_timing", "--benchmark_bard_active",
        "--pinn_loss_weight", "-1",
    ]
    if backend == "pinocchio" and args.pinocchio_workers is not None:
        command += ["--pinocchio_num_workers", str(args.pinocchio_workers)]
    started = time.perf_counter()
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=Path(__file__).resolve().parents[1], stdout=stream,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        peak = 0.0
        while process.poll() is None:
            if time.perf_counter() - started > args.timeout:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                raise TimeoutError(f"{backend} exceeded {args.timeout}s")
            try:
                used = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=used_memory",
                     "--format=csv,noheader,nounits"], capture_output=True,
                    text=True, timeout=5, check=True,
                ).stdout.splitlines()
                peak = max([peak, *(float(value) for value in used)])
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            time.sleep(1)
        stream.seek(0)
        output = stream.read()
    if process.returncode:
        raise RuntimeError(f"{backend} failed:\n{output[-12000:]}")
    iteration = [float(value) for value in ITERATION.findall(output)]
    breakdown = [(float(a), float(b)) for a, b in BREAKDOWN.findall(output)]
    if len(iteration) != args.iterations or len(breakdown) != args.iterations:
        raise RuntimeError(f"{backend} emitted incomplete timing records")
    result = {
        "backend": backend, "command": command,
        "pinocchio_workers": (
            args.pinocchio_workers if args.pinocchio_workers is not None
            else max(1, int((os.cpu_count() or 1) * 0.98))
        ) if backend == "pinocchio" else 0,
        "iteration_seconds": summaries(iteration),
        "collection_seconds": summaries([value[0] for value in breakdown]),
        "ppo_update_seconds": summaries([value[1] for value in breakdown]),
        "pinn_loss": [float(value) for value in PINN.findall(output)],
        "inverse_ms_per_update": [float(value) for value in INVERSE.findall(output)],
        "rollout_ms_per_update": [float(value) for value in ROLLOUT.findall(output)],
        "dynamics_ms_per_update": [float(value) for value in DYNAMICS.findall(output)],
        "pinocchio_transfer_ms_per_update": [float(value) for value in TRANSFER.findall(output)],
        "peak_process_memory_mib": peak,
        "wall_seconds": time.perf_counter() - started,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="go2_hard_pact_soft_isaaclab")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-capacity", type=int, default=4096)
    parser.add_argument("--pinocchio-workers", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark_results/hard_pact_dynamics_training.json"))
    args = parser.parse_args()
    results = [run(args, backend) for backend in ("bard", "pinocchio")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
