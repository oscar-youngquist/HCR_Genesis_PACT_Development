# Go2 HardPACT preliminary Isaac Lab qpth training benchmark

## Experiment

- Script: `scripts/benchmark_hard_pact_training.py`.
- Conditions: real Isaac Lab HardPACT/qpth training; exact command, chunks, and failure text are retained in JSON.
- Purpose: preliminary end-to-end capacity check; later targeted runs supersede incomplete cells.

| Solver | Rollout chunk | PPO chunk | Status | Iteration s | Steps/s |
|---|---:|---:|---|---:|---:|
| qpth | 4096 | 4096 | unexpected_error | 125.715 | 1042.0 |
| qpth | 2048 | 2048 | unexpected_error | 159.375 | 822.5 |
| qpth | 1024 | 1024 | unexpected_error | 211.655 | 624.5 |
| qpth | 512 | 512 | unexpected_error | 268.835 | 487.5 |
| qpth | 256 | 256 | numerical_failure | nan | nan |
