# Go2 HardPACT Isaac Lab qpth streamed-BARD training attempt

## Experiment

- Script: `scripts/benchmark_hard_pact_training.py`.
- Conditions: real Isaac Lab/qpth training with simulator-neutral BARD mechanics streamed through fixed-capacity chunks.
- Purpose: validate the capacity correction and expose the remaining differentiable full-pipeline VRAM limit.

| Solver | Rollout chunk | PPO chunk | Status | Iteration mean s | Iteration std s | Steps/s |
|---|---:|---:|---|---:|---:|---:|
| qpth | 4096 | 4096 | numerical_failure | nan | nan | nan |
