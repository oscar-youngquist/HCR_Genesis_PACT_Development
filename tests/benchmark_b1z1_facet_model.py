"""Stage 4 overhead on synthetic buffers, with the production six-joint FK path.

No simulator or optimizer. Includes packet noise, history, reference selection,
frame transforms and packet copies; excludes pre-existing reference integration.
"""
import argparse
import json
import statistics
import time
import torch

from test_b1z1_facet_observations import packet_env
from legged_gym.envs.b1z1.b1z1_facet.observations import FACETObservations
from legged_gym.envs.b1z1.z1_arm_kinematics import compute_z1_arm_fk
from rsl_rl.facet_model import FACETModel


@torch.no_grad()
def benchmark(n, device, steps=30):
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
    env = packet_env(n, device)
    env.cfg.noise.add_noise = True
    # Geometry values do not change this fixed-cost FK operation count. Actual
    # robot geometry/physical parity are checked by the Isaac Lab smoke instead.
    offsets = torch.zeros(6, 3, device=device)
    origin = torch.zeros(3, device=device)
    env.compute_z1_arm_fk = lambda q: compute_z1_arm_fk(q, offsets, offsets, origin, origin)
    collector = FACETObservations(env)
    ids = torch.arange(n, device=device)
    action = torch.zeros(n, 17, device=device)
    collector.collect(env, ids, reset=True)
    p = collector.read()
    model = FACETModel(p['observation'].shape[-1], p['privileged'].shape[-1]).to(device)
    model.update_normalizers(p['observation'], p['privileged'], p['wrench'])
    model.eval()
    hidden = torch.zeros(n, 128, device=device)
    operations = {
        'observation_assembly_and_read': lambda: (collector.collect(env, ids, action), collector.read()),
        'teacher_inference': lambda: model.teacher_action(p['observation'], p['privileged'], p['wrench']),
        'student_inference': lambda: model.student_action(p['observation'], hidden, p['is_init']),
        'critic_inference': lambda: model.value(p['observation'], p['privileged'], p['wrench']),
    }
    def sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
    result = {}
    for name, operation in operations.items():
        for _ in range(30):
            operation()
        sync()
        blocks = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(steps):
                operation()
            sync()
            blocks.append((time.perf_counter()-start)*1000/steps)
        result[name] = dict(median_ms=statistics.median(blocks), block_ms=blocks)
    return dict(envs=n, device=device, warmup=30, steps_per_block=steps, measurements=result,
                peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.startswith('cuda') else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--num-envs', type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.num_envs, args.device), indent=2))
