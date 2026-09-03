# Go2 HardPACT Isaac Lab qpth warm-start training attempt

## Experiment

- Script: `scripts/benchmark_hard_pact_training.py` with qpth rollout warm starts.
- Conditions: real Isaac Lab HardPACT training; exact chunks/status are retained in JSON.
- Purpose: test whether substep warm starts improve end-to-end qpth training without changing certified solutions.

| Solver | Rollout chunk | PPO chunk | Status | Iteration mean s | Iteration std s | Steps/s |
|---|---:|---:|---|---:|---:|---:|
| qpth | 4096 | 4096 | numerical_failure | 191.255 | 2.515 | 685.0 |
