"""No simulator: container versions, CUDA tensor operation and real CLI config."""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

ENTRIES = {'go2_beyond_robustness_isaaclab': 'train_beyond_robustness.py',
           'go2_mule_isaaclab': 'train_mule.py'}

if __name__ == '__main__':
    task = sys.argv[1]
    if task not in ENTRIES:
        raise SystemExit('Container check supports BR and MULE only')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: check host NVIDIA driver and Apptainer --nv')
    x = torch.ones(8, device='cuda:0', requires_grad=True)
    x.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(root/'legged_gym/scripts'/ENTRIES[task]),
                    '--dry-run', '--headless', '--device', 'cuda:0', *sys.argv[2:]], check=True)
    print('CONTAINER_CHECK_PASS '+json.dumps(dict(task=task, gpu=torch.cuda.get_device_name(0),
          versions={p:metadata.version(p) for p in ('torch','isaacsim','isaaclab','numpy')},
          simulator_tested=False)), flush=True)
