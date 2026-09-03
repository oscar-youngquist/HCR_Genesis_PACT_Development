# Go2 HardPACT corrected complete-iteration canonical-QP benchmark

## Experiment

- Script: `scripts/benchmark_hard_pact_qp.py`.
- Conditions: captured canonical QPs, complete-iteration mode, RTX 4090; exact batch/backend metadata is in JSON.
- Purpose: verify the corrected end-to-end standalone timing path after the preliminary harness failure.

| Solver | Mode | Batch | Status | Mean ms | QPs/s | Peak MiB |
|---|---|---:|---|---:|---:|---:|
| cupiqp | complete_iteration | 256 | success | 373.488 | 685.4 | 55.0 |
| cupiqp | complete_iteration | 512 | success | 517.564 | 989.2 | 91.5 |
| cupiqp | complete_iteration | 1024 | success | 447.069 | 2290.5 | 167.9 |
| cupiqp | complete_iteration | 2048 | success | 496.284 | 4126.7 | 312.5 |
| cupiqp | complete_iteration | 4096 | success | 635.640 | 6443.9 | 610.9 |
