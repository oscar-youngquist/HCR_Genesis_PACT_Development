"""Small GPU/import gate, not a replacement for simulator and optimizer smoke tests."""
import importlib.metadata as metadata
import os
from pathlib import Path
import runpy
import sys

import torch

task = sys.argv[1]
assert torch.cuda.is_available(), "CUDA unavailable inside container"
assert torch.cuda.device_count() == 1, "Expected one scheduler-visible GPU"
print("GPU:", torch.cuda.get_device_name(0), "visibility:", os.environ.get("CUDA_VISIBLE_DEVICES"))
for name in ("isaacsim", "isaaclab", "torch", "numpy", "warp-lang", "cupiqp", "bard", "pin"):
    print(name, metadata.version(name))
x = torch.ones(4, device="cuda:0", requires_grad=True)
x.square().sum().backward()
assert torch.isfinite(x.grad).all()
if task.startswith("go2_hard_pact"):
    # Preserve the aligned branch's Warp import order and missing-alias repair.
    entry = runpy.run_path("/workspace/repo/legged_gym/scripts/train_hard_pact.py")
    entry["prepare_solver_runtime"](["--task", task, "--qp_solver", "cupiqp"])
    import cupiqp
    print("cuPIQP:", cupiqp.__file__)
import isaaclab
assert Path(isaaclab.__file__).is_relative_to("/opt/IsaacLab")
import legged_gym.envs
from legged_gym.utils import task_registry
assert task in task_registry.task_classes, f"Task {task} missing from mounted checkout"
print("PASS: CUDA/autograd/imports/task registration. Next run the simulator smoke job.")
