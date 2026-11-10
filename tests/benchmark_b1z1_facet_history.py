"""Bounded FACET-only benchmark; no simulator, runner, policy or optimizer.

SIMULATOR=isaaclab PYTHONPATH=. python tests/benchmark_b1z1_facet_history.py --device cuda:1
"""
import argparse
import gc
import json
import statistics
import time

import torch

from legged_gym.envs.b1z1.b1z1_facet.reference import FACETReference, FACETReferenceBank, FACETReferenceCfg
from legged_gym.envs.b1z1.b1z1_facet.tracking import FACETTrackingCfg, COMPONENTS, tracking_values


def benchmark(count, device, repeats, steps):
    results = {}
    if device.startswith("cuda"):
        torch.cuda.set_device(device)  # Initialize the allocator before resetting peak counters.
    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
    for mode in ("stage2", "stage3"):
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        ref = FACETReference(count, device, FACETReferenceCfg())
        seed = torch.zeros(count, 12, device=device)
        seed[:, 6] = .5
        seed[:, 8] = .6
        ref.reset(torch.arange(count, device=device), seed[:, :2], seed[:, 2:4], seed[:, 4], seed[:, 5], seed[:, 6:9], seed[:, 9:12])
        z = torch.zeros(count, 3, device=device)
        inputs = (seed[:, :2], seed[:, 4], z[:, :2]+.1, z[:, 0], z+.1, z, z[:, 0]+.5)
        loads = (z+5, z+2, z[:, 0]+.1)
        bank = FACETReferenceBank(ref) if mode == "stage3" else None
        settings = FACETTrackingCfg()
        widths = torch.tensor([settings.bandwidths[k] for k in COMPONENTS], device=device)
        weights = torch.tensor([settings.weights[k] for k in COMPONENTS], device=device)*.02
        error_sum = torch.zeros(count, 3, 6, device=device)
        valid_sum = torch.zeros(count, 3, device=device)
        def advance():
            ref.advance(.02, *inputs, *loads)
            if bank is not None:
                bank.advance(seed, .02, inputs, loads)
                snapshot = bank.read()
                values, errors = tracking_values(snapshot["targets"], seed, snapshot["valid"], widths)
                # Include reward reduction and the same diagnostic sufficient statistics.
                (values*weights).sum(-1)
                error_sum.add_(torch.where(snapshot["valid"][..., None], errors, 0.))
                valid_sum.add_(snapshot["valid"])
        for _ in range(40):
            advance()  # All requested ages are mature before timing.
        sync()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(steps):
                advance()
            sync()
            timings.append((time.perf_counter()-start)*1000/steps)
        results[mode] = dict(median_ms=statistics.median(timings), block_ms=timings,
            peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.startswith("cuda") else None)
        del advance, ref, bank, seed, z, inputs, loads, widths, weights, error_sum, valid_sum
    return dict(envs=count, device=device, dt=.02, warmup=40, steps_per_block=steps,
                repeats=repeats, modes=results,
                additional_ms=results["stage3"]["median_ms"]-results["stage2"]["median_ms"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.num_envs, args.device, args.repeats, args.steps), indent=2))
