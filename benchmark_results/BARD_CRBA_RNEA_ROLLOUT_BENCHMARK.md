# HardPACT BARD rollout: CRBA/RNEA fixed-mechanics benchmark

## Purpose and conditions

`scripts/benchmark_bard_forward_dynamics.py` measures the differentiable
forward/backward operation used by the rollout PINN, excluding simulator and
QP allocation. Tests used one RTX 4090, PyTorch 2.8.0+cu126, BARD 0.4.3,
float32, randomized mass/CoM, armature, friction, stiffness, and damping.
Commands have the form:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. SIMULATOR=genesis \
  conda run -n genesis_lr python scripts/benchmark_bard_forward_dynamics.py \
  --method fixed --batch-size 4096 --warmup 1 --repeats 3 --device cuda:0
```

## Results

| Method | Batch | Forward+backward mean | Solve peak increment |
|---|---:|---:|---:|
| fixed CRBA/RNEA | 64 | 0.967 ms | 8.398 MiB |
| official ABA reference | 64 | 296.028 ms | 36.905 MiB |
| fixed CRBA/RNEA | 512 | 1.458 ms | 10.307 MiB |
| fixed CRBA/RNEA | 1024 | 1.087 ms | 12.489 MiB |
| fixed CRBA/RNEA | 2048 | 1.736 ms | 17.666 MiB |
| fixed CRBA/RNEA | 4096 | 2.393 ms | 25.582 MiB |
| official ABA reference | 4096 | 300.833 ms | 1835.246 MiB |

At matched batch 64, the fixed-mechanics solve was about 306x faster and used
77% less incremental solve memory. At batch 4096 it was about 126x faster and
used 98.6% less incremental solve memory. Both paths remained finite.

## Full training check

`scripts/benchmark_hard_pact_training.py` completed five real Isaac Lab
HardPACT iterations with 100 environments, cuPIQP, both BARD losses enabled
from iteration 1, and BARD timing enabled. Mean iteration time was 49.076 s
(population standard deviation 0.331 s); inverse and rollout PINN time was
753.577 and 960.504 ms/update, respectively. Peak process GPU memory was
5151 MiB. Full details and the exact launch command are in
`hard_pact_training_isaaclab_cupiqp_crba_100/`.
