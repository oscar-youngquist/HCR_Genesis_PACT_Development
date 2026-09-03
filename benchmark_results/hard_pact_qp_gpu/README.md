# Go2 HardPACT preliminary qpth/cuPIQP canonical-QP GPU benchmark

## Experiment

- Script: `scripts/benchmark_hard_pact_qp.py`.
- Conditions: captured canonical QPs on an RTX 4090; solver/mode/batch details are in the table and JSON.
- Purpose: preliminary cross-solver throughput/capacity check. Superseded cuPIQP cells are not used for final performance claims.

| Solver | Mode | Batch | Status | Mean ms | QPs/s | Peak MiB |
|---|---|---:|---|---:|---:|---:|
| qpth | assembly | 256 | success | 0.641 | 399423.1 | n/a |
| qpth | assembly | 512 | success | 0.555 | 922479.0 | n/a |
| qpth | rollout | 256 | success | 43.526 | 5881.5 | n/a |
| qpth | rollout | 512 | success | 61.565 | 8316.4 | n/a |
| qpth | ppo_forward | 256 | success | 41.985 | 6097.5 | n/a |
| qpth | ppo_forward | 512 | success | 64.066 | 7991.8 | n/a |
| qpth | ppo_backward | 256 | success | 48.677 | 5259.1 | n/a |
| qpth | ppo_backward | 512 | success | 70.162 | 7297.4 | n/a |
| qpth | sequential_rollout | 256 | success | 167.729 | 1526.3 | n/a |
| qpth | sequential_rollout | 512 | success | 263.066 | 1946.3 | n/a |
| qpth | complete_iteration | 256 | success | 218.224 | 1173.1 | n/a |
| qpth | complete_iteration | 512 | success | 303.232 | 1688.5 | n/a |
| cupiqp | assembly | 256 | success | 0.509 | 502976.2 | n/a |
| cupiqp | assembly | 512 | success | 0.581 | 881573.1 | n/a |
| cupiqp | rollout | 256 | success | 15.864 | 16137.0 | n/a |
| cupiqp | rollout | 512 | success | 16.919 | 30261.1 | n/a |
| cupiqp | ppo_forward | 256 | success | 165.386 | 1547.9 | n/a |
| cupiqp | ppo_forward | 512 | success | 168.664 | 3035.6 | n/a |
| cupiqp | ppo_backward | 256 | success | 169.500 | 1510.3 | n/a |
| cupiqp | ppo_backward | 512 | success | 168.776 | 3033.6 | n/a |
| cupiqp | sequential_rollout | 256 | success | 63.942 | 4003.7 | n/a |
| cupiqp | sequential_rollout | 512 | success | 66.071 | 7749.2 | n/a |
| cupiqp | complete_iteration | 256 | unexpected_error | nan | nan | n/a |
| cupiqp | complete_iteration | 512 | unexpected_error | nan | nan | n/a |
