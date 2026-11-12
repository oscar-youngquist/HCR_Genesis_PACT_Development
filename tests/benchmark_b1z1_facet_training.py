"""FACET-only rollout/update cost on fixed synthetic packets; excludes simulation.

Production 237/52/9 packet dimensions and 24 x 4096 rollout, 5 epochs/4 batches.
Both online/EMA recurrent paths and PPO storage are included in collection.
"""
import argparse
import json
from statistics import median
import time
from types import SimpleNamespace
import torch

from legged_gym.envs.b1z1.b1z1_facet import B1Z1FACETCfg, B1Z1FACETCfgPPO
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners.b1z1_facet_runner import B1Z1FACETRunner


def benchmark(n, device):
    torch.cuda.set_device(device)
    packet = dict(observation=torch.randn(n, 237, device=device), privileged=torch.randn(n, 52, device=device),
                  wrench=torch.randn(n, 9, device=device), is_init=torch.zeros(n, 1, dtype=torch.bool, device=device))
    reward, done = torch.ones(n, device=device), torch.zeros(n, dtype=torch.bool, device=device)
    env = SimpleNamespace(cfg=B1Z1FACETCfg(), num_envs=n, device=device,
        get_facet_observations=lambda: {k: v.clone() for k, v in packet.items()},
        get_facet_terminal_observations=lambda: None,
        step=lambda actions: (None, None, None, None, reward, done, {}, None))
    runner = B1Z1FACETRunner(env, class_to_dict(B1Z1FACETCfgPPO()), device=device)
    measurements = {}
    for phase in ('teacher', 'student'):
        if phase == 'student':
            runner.alg.switch_to_student()
        rows = []
        for repeat in range(5):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            storage, _ = runner.collect()
            torch.cuda.synchronize(device)
            collection = time.perf_counter()
            runner.alg.update(storage, 0)  # Includes estimator for both phases (worst teacher cadence).
            runner._finish_normalization()
            torch.cuda.synchronize(device)
            end = time.perf_counter()
            if repeat >= 2:
                rows.append(dict(collection_seconds=collection-start, update_seconds=end-collection))
        measurements[phase] = dict(rows=rows, **{key: median(row[key] for row in rows) for key in rows[0]})
    return dict(envs=n, device=device, warmup_updates_per_phase=2, measured_updates_per_phase=3,
                peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20, measurements=measurements)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.num_envs, args.device), indent=2))
