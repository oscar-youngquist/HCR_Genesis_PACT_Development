# Go2 HardPACT standalone large-batch canonical-QP capacity benchmark

## Experiment

- Script: `scripts/benchmark_hard_pact_qp.py`.
- Conditions: captured canonical QPs at large CUDA batch sizes on an RTX 4090.
- Purpose: locate standalone solver throughput and VRAM limits; it does not include simulator or BARD costs.

| Solver | Mode | Batch | Status | Mean ms | QPs/s | Peak MiB |
|---|---|---:|---|---:|---:|---:|
| qpth | rollout | 1024 | success | 123.995 | 8258.4 | n/a |
| qpth | rollout | 2048 | success | 218.469 | 9374.3 | n/a |
| qpth | rollout | 4096 | success | 423.745 | 9666.2 | n/a |
| qpth | ppo_backward | 1024 | success | 134.988 | 7585.9 | n/a |
| qpth | ppo_backward | 2048 | success | 249.001 | 8224.9 | n/a |
| qpth | ppo_backward | 4096 | success | 471.837 | 8681.0 | n/a |
| cupiqp | rollout | 1024 | success | 24.077 | 42529.6 | n/a |
| cupiqp | rollout | 2048 | success | 34.722 | 58982.0 | n/a |
| cupiqp | rollout | 4096 | success | 64.607 | 63398.8 | n/a |
| cupiqp | ppo_backward | 1024 | success | 153.248 | 6682.0 | n/a |
| cupiqp | ppo_backward | 2048 | success | 194.496 | 10529.8 | n/a |
| cupiqp | ppo_backward | 4096 | success | 218.222 | 18769.9 | n/a |
