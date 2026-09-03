# Go2 HardPACT preliminary sparse-cuPIQP rollout benchmark

## Experiment

- Script: `scripts/benchmark_hard_pact_qp.py` with sparse cuPIQP.
- Conditions: captured canonical QPs on an RTX 4090; exact package/batch metadata is in JSON.
- Purpose: initial sparse setup/update check. The later `sparse_fixed` experiment is authoritative.

| Solver | Mode | Batch | Status | Mean ms | QPs/s | Peak MiB |
|---|---|---:|---|---:|---:|---:|
| cupiqp | rollout | 256 | numerical_failure | 100.286 | 2552.7 | 36.3 |
| cupiqp | rollout | 512 | numerical_failure | 105.325 | 4861.1 | 63.7 |
| cupiqp | rollout | 1024 | numerical_failure | 98.196 | 10428.1 | 118.7 |
| cupiqp | rollout | 2048 | numerical_failure | 108.170 | 18933.1 | 228.5 |
| cupiqp | rollout | 4096 | numerical_failure | 135.665 | 30192.0 | 448.2 |
| cupiqp | sequential_rollout | 256 | numerical_failure | 356.473 | 718.1 | 36.3 |
| cupiqp | sequential_rollout | 512 | numerical_failure | 389.502 | 1314.5 | 63.7 |
| cupiqp | sequential_rollout | 1024 | numerical_failure | 400.418 | 2557.3 | 118.7 |
| cupiqp | sequential_rollout | 2048 | numerical_failure | 421.835 | 4855.0 | 228.4 |
| cupiqp | sequential_rollout | 4096 | numerical_failure | 520.819 | 7864.5 | 448.0 |
