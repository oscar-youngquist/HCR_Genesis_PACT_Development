"""GPU reset-only sampler timings; not an end-to-end training speed claim."""
import argparse
import json
import time

import torch

from legged_gym.envs.b1z1.directional_compliance import ComplianceCfg, ComplianceReference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()
    results = []
    for mode in ('uniform', 'mixture'):
        cfg = ComplianceCfg(); cfg.sampling_mode = mode
        torch.manual_seed(123)
        torch.cuda.synchronize()
        start = time.perf_counter()
        reference = ComplianceReference(cfg, args.num_envs, args.device)
        torch.cuda.synchronize()
        init_ms = (time.perf_counter()-start)*1000
        for count in (64, args.num_envs):
            ids = torch.arange(min(count, args.num_envs), device=args.device)
            for _ in range(10): reference.reset(ids)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(args.repeats): reference.reset(ids)
            torch.cuda.synchronize()
            results.append(dict(mode=mode, resetting_envs=len(ids), initialization_ms=init_ms,
                wall_ms_per_reset=(time.perf_counter()-start)*1000/args.repeats,
                peak_extra_mib=(torch.cuda.max_memory_allocated()-allocated)/2**20))
        p, family = reference.profile, reference.family_ids
        assert torch.isfinite(p).all() and p.ge(0).all() and p.le(1).all()
        print('SAMPLING_COVERAGE '+json.dumps(dict(mode=mode,
            fractions=(torch.bincount(family,minlength=7)/len(family)).cpu().tolist())), flush=True)
    print('SAMPLING_BENCHMARK '+json.dumps(results), flush=True)


if __name__ == '__main__': main()
