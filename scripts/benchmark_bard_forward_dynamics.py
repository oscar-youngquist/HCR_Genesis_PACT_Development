#!/usr/bin/env python3
"""Measure HardPACT rollout forward dynamics without simulator allocation.

The benchmark constructs the same randomized Go2 BARD context used by PPO,
then times either the training CRBA/RNEA solve or official ABA reference.
Run methods in separate processes so an ABA OOM cannot contaminate a fixed-
solve measurement.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from legged_gym.dynamics import BardGo2Dynamics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("fixed", "aba"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this memory benchmark requires CUDA")

    batch = args.batch_size
    dynamics = BardGo2Dynamics(
        "resources/robots/go2/urdf/go2.urdf", device=device,
        batch_capacity=batch,
    )
    q = torch.zeros(batch, 19, device=device)
    q[:, 2], q[:, 6] = 0.42, 1.0
    q[:, 7:] = torch.linspace(-0.25, 0.35, 12, device=device)
    v = torch.linspace(-0.15, 0.2, 18, device=device).expand(batch, -1).clone()
    parameters = {
        "added_base_mass": torch.linspace(-0.5, 2.0, batch, device=device)[:, None],
        "base_com_shift": torch.linspace(-0.03, 0.03, batch, device=device)[:, None].expand(-1, 3),
        "joint_armature": torch.full((batch, 1), 0.015, device=device),
        "joint_friction": torch.full((batch, 1), 0.03, device=device),
        "joint_stiffness": torch.full((batch, 1), 0.02, device=device),
        "joint_damping": torch.full((batch, 1), 0.12, device=device),
    }
    context = dynamics.build_context(
        q, v, parameters=parameters, need_forward_dynamics=True,
    )
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    timings = []
    for iteration in range(args.warmup + args.repeats):
        generalized_force = torch.randn(batch, 18, device=device, requires_grad=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        acceleration = (
            context.forward_dynamics(generalized_force)
            if args.method == "fixed" else context.aba(generalized_force)
        )
        acceleration.square().mean().backward()
        torch.cuda.synchronize(device)
        elapsed = 1.0e3 * (time.perf_counter() - started)
        if iteration >= args.warmup:
            timings.append(elapsed)
        if not (torch.isfinite(acceleration).all() and torch.isfinite(generalized_force.grad).all()):
            raise RuntimeError("nonfinite forward or backward result")

    result = {
        "method": args.method,
        "batch_size": batch,
        "mean_forward_backward_ms": sum(timings) / len(timings),
        "min_forward_backward_ms": min(timings),
        "max_forward_backward_ms": max(timings),
        "context_memory_mib": baseline / 2**20,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "solve_peak_increment_mib": (
            torch.cuda.max_memory_allocated(device) - baseline
        ) / 2**20,
        "finite": True,
        "dtype": str(acceleration.dtype),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
