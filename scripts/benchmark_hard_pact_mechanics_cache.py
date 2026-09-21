#!/usr/bin/env python3
"""Benchmark the rollout-iteration mechanics cache in real HardPACT training.

Both runs use the same Isaac Lab task, seed, rollout, PPO configuration, and
BARD losses.  The sole training-path difference is the cache feature switch;
the uncached run rebuilds mechanics in every PPO minibatch and epoch.
"""

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


PATTERNS = {
    "iteration_seconds": re.compile(r"Iteration time:\s*([0-9.]+)s"),
    "breakdown": re.compile(
        r"collection:\s*([0-9.]+)s, learning\s*([0-9.]+)s"
    ),
    "pinn_loss": re.compile(r"PINN loss:\s*([-+0-9.eE]+)"),
    "inverse_ms": re.compile(
        r"Dynamics inverse forward:\s*([0-9.]+) ms/update"
    ),
    "rollout_ms": re.compile(
        r"Dynamics rollout forward:\s*([0-9.]+) ms/update"
    ),
    "dynamics_ms": re.compile(r"Dynamics total:\s*([0-9.]+) ms/update"),
    "auxiliary_ms": re.compile(r"Auxiliary update:\s*([0-9.]+) ms/update"),
    "pcgrad_ms": re.compile(r"PCGrad backward:\s*([0-9.]+) ms/update"),
}


def _summary(values):
    if not values:
        return {"values": [], "mean": None, "median": None,
                "mean_excluding_iteration_1": None,
                "median_excluding_iteration_1": None}
    tail = values[1:] or values
    return {
        "values": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "mean_excluding_iteration_1": statistics.mean(tail),
        "median_excluding_iteration_1": statistics.median(tail),
    }


def _process_gpu_memory_mib(pid):
    try:
        lines = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return 0.0
    for line in lines:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                return float(fields[1])
            except ValueError:
                return 0.0
    return 0.0


def _run(args, cached):
    launcher = (Path(__file__).resolve().parents[1] / "legged_gym" /
                "scripts" / "go2_hard_pact.sh")
    command = [
        str(launcher), "--task", args.task, "--headless",
        "--num_envs", str(args.num_envs),
        "--max_iterations", str(args.iterations),
        "--seed", str(args.seed), "--gpu", args.device,
        "--dynamics_backend", "bard",
        "--bard_batch_capacity", str(args.batch_capacity),
        "--profile_bard_timing", "--benchmark_bard_active",
        "--pinn_loss_weight", "-1",
    ]
    if not cached:
        command.append("--disable_rollout_mechanics_cache")
    started = time.perf_counter()
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=Path(__file__).resolve().parents[1], stdout=stream,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        peak_mib = 0.0
        while process.poll() is None:
            if time.perf_counter() - started > args.timeout:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
                raise TimeoutError(
                    f"{'cached' if cached else 'uncached'} run exceeded "
                    f"{args.timeout}s"
                )
            peak_mib = max(peak_mib, _process_gpu_memory_mib(process.pid))
            time.sleep(0.5)
        stream.seek(0)
        output = stream.read()
    if process.returncode:
        raise RuntimeError(
            f"{'cached' if cached else 'uncached'} run failed:\n{output[-16000:]}"
        )

    parsed = {}
    for name, pattern in PATTERNS.items():
        matches = pattern.findall(output)
        if name == "breakdown":
            parsed["collection_seconds"] = _summary(
                [float(pair[0]) for pair in matches]
            )
            parsed["ppo_seconds"] = _summary(
                [float(pair[1]) for pair in matches]
            )
        else:
            parsed[name] = _summary([float(value) for value in matches])
    if len(parsed["iteration_seconds"]["values"]) != args.iterations:
        raise RuntimeError("training output did not contain every iteration")
    parsed.update({
        "mode": "cached" if cached else "uncached",
        "cache_enabled": cached,
        "command": command,
        "peak_cuda_memory_mib": peak_mib,
        "wall_seconds": time.perf_counter() - started,
    })
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="go2_hard_pact_soft_isaaclab")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-capacity", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmark_results/hard_pact_mechanics_cache.json"),
    )
    args = parser.parse_args()
    results = [_run(args, cached=False), _run(args, cached=True)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
