"""Matched diagnostic-on/off timing, including ring and rollout reductions; no simulation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
import torch
from legged_gym.envs.b1z1.b1z1_facet.reference import FACETReference, FACETReferenceBank, FACETReferenceCfg
from legged_gym.envs.b1z1.b1z1_facet.recovery_diagnostics import RecoveryRollout


def benchmark(count, device, steps):
    rows, final = [], []
    for enabled in (False, True):
        cfg = FACETReferenceCfg(bounded=True, formulation_version=3, yaw_reaction=True,
                                sampling_mode='bounded_b1z1', recovery_diagnostics=enabled)
        ref = FACETReference(count, device, cfg)
        z = torch.zeros(count, 3, device=device)
        ee = z+z.new_tensor([.9, 0., .6985])
        ref.reset(torch.arange(count, device=device), z, z, z[:, 0], z[:, 0], ee, z)
        bank, reducer = FACETReferenceBank(ref), RecoveryRollout()
        seed = ref._state.clone()
        inputs = (z[:, :2], z[:, 0], z[:, :2]+.1, z[:, 0], ee-z.new_tensor([0., 0., .6985]), z, z[:, 0]+.6985)
        loads = (z+2, z, z[:, 0])
        def step():
            ref.advance(.02, *inputs, *loads)
            bank.advance(seed, .02, inputs, loads)
            reducer.add(bank.read())
        for _ in range(35):  # Fill all horizons; warm compiled kernels before timing.
            step()
        if device.startswith('cuda'):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(steps):
            step()
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        rows.append(dict(enabled=enabled, ms_per_control_interval=(time.perf_counter()-start)*1000/steps,
            peak_allocated_MB=torch.cuda.max_memory_allocated()/1e6 if device.startswith('cuda') else None,
            configuration=asdict(cfg)))
        final.append(bank._state.clone())
    torch.testing.assert_close(*final, rtol=0, atol=0)
    paths = [Path('legged_gym/envs/b1z1/b1z1_facet')/p for p in ('reference.py', 'recovery_diagnostics.py')]
    return dict(envs=count, device=device, measured_intervals=steps, warmup_intervals=35, results=rows,
        bitwise_reference_parity=True, overhead_ms=rows[1]['ms_per_control_interval']-rows[0]['ms_per_control_interval'],
        hardware=torch.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU',
        torch_version=torch.__version__, source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Reference + history + new rollout reductions, not simulator/PPO; shared GPU timing may vary.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = benchmark(args.num_envs, args.device, args.steps)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))
