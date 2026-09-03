# Go2 HardPACT standalone sparse-cuPIQP rollout throughput and VRAM benchmark

## Experiment

- Script: `scripts/benchmark_hard_pact_qp.py` with `cupiqp_mode=sparse`.
- Conditions: captured 54-variable HardPACT QPs; float32 CUDA; rollout and four-substep sequential-rollout modes; batches 256--4096; RTX 4090; cuPIQP 0.1.0 and cuDSS 0.7.1.6.
- Purpose: determine whether uniform-CSR sparse solves improve the rollout hot path. They preserve certification and memory use, but are much slower than dense cuPIQP for this small, fixed-shape QP, so dense is the training default.

| Solver | Mode | Batch | Status | Mean ms | QPs/s | Peak MiB |
|---|---|---:|---|---:|---:|---:|
| cupiqp | rollout | 256 | success | 109.247 | 2343.3 | 36.3 |
| cupiqp | rollout | 512 | success | 128.774 | 3976.0 | 63.7 |
| cupiqp | rollout | 1024 | success | 226.986 | 4511.3 | 118.7 |
| cupiqp | rollout | 2048 | success | 472.902 | 4330.7 | 228.5 |
| cupiqp | rollout | 4096 | success | 903.084 | 4535.6 | 448.2 |
| cupiqp | sequential_rollout | 256 | success | 310.250 | 825.1 | 36.3 |
| cupiqp | sequential_rollout | 512 | success | 485.846 | 1053.8 | 63.7 |
| cupiqp | sequential_rollout | 1024 | success | 926.054 | 1105.8 | 118.7 |
| cupiqp | sequential_rollout | 2048 | success | 1849.207 | 1107.5 | 228.4 |
| cupiqp | sequential_rollout | 4096 | success | 3549.437 | 1154.0 | 448.0 |
