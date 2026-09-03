# Go2 HardPACT Isaac Lab Moreau capability benchmark

## Experiment

- Script: `scripts/benchmark_hard_pact_training.py`.
- Conditions: requested real Isaac Lab HardPACT training with Moreau CUDA.
- Purpose: test deployment compatibility; Python-version and CUDA-license requirements make this combination unsupported, so no timing is claimed.

| Solver | Rollout chunk | PPO chunk | Status | Iteration mean s | Iteration std s | Steps/s |
|---|---:|---:|---|---:|---:|---:|
| moreau | 4096 | 4096 | unsupported_dependency | nan | nan | nan |
