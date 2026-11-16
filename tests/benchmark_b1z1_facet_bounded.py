"""Reference-only bounded/legacy benchmark, including all 32 history starts."""
import argparse
import json
import time
import hashlib
import platform
import subprocess
from pathlib import Path
from dataclasses import asdict
import torch
from legged_gym.envs.b1z1.b1z1_facet.reference import FACETReference, FACETReferenceBank, FACETReferenceCfg


def benchmark(count, device, steps=20, compile_constraints=True, yaw_reaction=False):
    result = {}
    for bounded in (False, True):
        cfg = FACETReferenceCfg(bounded=bounded, sampling_mode='bounded_b1z1' if bounded else 'upstream_b1z1',
                                formulation_version=3 if bounded else 1, compile_constraints=compile_constraints,
                                yaw_reaction=bounded and yaw_reaction)
        ref = FACETReference(count, device, cfg)
        z = torch.zeros(count, 3, device=device)
        ee = z+torch.tensor([.9, 0., .6985], device=device)
        ref.reset(torch.arange(count, device=device), z, z, z[:, 0], z[:, 0], ee, z)
        bank = FACETReferenceBank(ref)
        seed = ref._state.clone()
        inputs = (z[:, :2], z[:, 0], z[:, :2]+.1, z[:, 0], ee-torch.tensor([0., 0., .6985], device=device), z, z[:, 0]+.6985)
        loads = (z+2, z, z[:, 0])
        def advance():
            ref.advance(.02, *inputs, *loads)
            bank.advance(seed, .02, inputs, loads)
        warmup_start = time.perf_counter()
        for _ in range(3):
            advance()
        if device.startswith('cuda'):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        warmup_seconds = start-warmup_start
        for _ in range(steps):
            advance()
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        result['bounded' if bounded else 'legacy'] = dict(ms_per_control_interval=1000*(time.perf_counter()-start)/steps,
            configuration=asdict(cfg), warmup_seconds=warmup_seconds,
            numerical_substep_limit=ref._max_dt,
            peak_allocated_MB=torch.cuda.max_memory_allocated()/1e6 if device.startswith('cuda') else None)
    diff = subprocess.check_output(['git', 'diff', 'HEAD'])
    return dict(envs=count, device=device, measured_intervals=steps, results=result,
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_files_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path('legged_gym/envs/b1z1/b1z1_facet/reference.py'), Path('legged_gym/envs/b1z1/b1z1_facet/bounds.py'))},
        working_diff_sha256=hashlib.sha256(diff).hexdigest(), torch_version=torch.__version__,
        hardware=torch.cuda.get_device_name(device) if device.startswith('cuda') else platform.processor(),
        scope='Reference plus 32-start history only; excludes simulator/PPO. Warmup includes compilation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--output')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--yaw-reaction', action='store_true', help='Enable production yaw coupling; old 53.9 ms benchmark omitted it')
    args = parser.parse_args()
    data = json.dumps(benchmark(args.num_envs, args.device, compile_constraints=not args.eager, yaw_reaction=args.yaw_reaction), indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(data)
    print(data)
