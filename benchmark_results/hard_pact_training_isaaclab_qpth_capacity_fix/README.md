# Go2 HardPACT Isaac Lab qpth BARD-capacity-fix training attempt

## Experiment

- Script: `scripts/benchmark_hard_pact_training.py`.
- Conditions: real Isaac Lab/qpth training after adding a fixed BARD workspace capacity.
- Purpose: determine whether workspace capacity alone permits full PPO; the retained flattened minibatch still exceeded it.

| Solver | Rollout chunk | PPO chunk | Status | Iteration mean s | Iteration std s | Steps/s |
|---|---:|---:|---|---:|---:|---:|
| qpth | 4096 | 4096 | unsupported_dependency | nan | nan | nan |
