# Go2 HardPACT standalone qpth warm-start rollout benchmark

## Experiment

- Script: `scripts/benchmark_hard_pact_qp.py` with qpth warm starts enabled.
- Conditions: sequential captured QPs on an RTX 4090; batches and timing are recorded below.
- Purpose: quantify rollout reuse from the preceding substep relative to cold qpth.

| Solver | Mode | Batch | Status | Mean ms | QPs/s | Peak MiB |
|---|---|---:|---|---:|---:|---:|
| qpth | sequential_rollout | 256 | success | 603.110 | 424.5 | 104.5 |
| qpth | sequential_rollout | 512 | success | 1151.046 | 444.8 | 202.6 |
| qpth | sequential_rollout | 1024 | success | 2045.822 | 500.5 | 390.6 |
| qpth | sequential_rollout | 2048 | success | 3632.999 | 563.7 | 768.3 |
| qpth | sequential_rollout | 4096 | success | 6910.488 | 592.7 | 1529.6 |
