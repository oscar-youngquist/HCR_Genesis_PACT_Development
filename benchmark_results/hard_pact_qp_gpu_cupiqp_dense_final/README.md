# Go2 HardPACT standalone dense-cuPIQP canonical-QP throughput and VRAM benchmark

## Experiment

- Script: `scripts/benchmark_hard_pact_qp.py` with the dense cuPIQP backend.
- Conditions: captured 54-variable HardPACT QPs; float32 CUDA; batches 256--4096; RTX 4090; Python 3.11.16; Torch 2.7.0+cu128; cuPIQP 0.1.0; fresh process per cell.
- Purpose: measure the production no-grad rollout and implicit-backward PPO paths. Compared with qpth, these results show substantially lower solver latency and VRAM at the intended 4096-environment training scale while retaining full-stage certified solutions.

| Solver | Mode | Batch | Status | Mean ms | QPs/s | Peak MiB |
|---|---|---:|---|---:|---:|---:|
| cupiqp | assembly | 256 | success | 0.486 | 526661.1 | 35.0 |
| cupiqp | assembly | 512 | success | 0.490 | 1044127.2 | 61.1 |
| cupiqp | assembly | 1024 | success | 0.561 | 1823939.3 | 114.1 |
| cupiqp | assembly | 2048 | success | 0.929 | 2205312.9 | 218.5 |
| cupiqp | assembly | 4096 | success | 1.766 | 2319429.0 | 429.7 |
| cupiqp | rollout | 256 | success | 29.880 | 8567.5 | 36.3 |
| cupiqp | rollout | 512 | success | 30.536 | 16767.3 | 63.7 |
| cupiqp | rollout | 1024 | success | 32.764 | 31253.6 | 118.7 |
| cupiqp | rollout | 2048 | success | 40.094 | 51080.0 | 228.5 |
| cupiqp | rollout | 4096 | success | 63.798 | 64202.7 | 448.2 |
| cupiqp | sequential_rollout | 256 | success | 115.761 | 2211.5 | 36.3 |
| cupiqp | sequential_rollout | 512 | success | 120.540 | 4247.5 | 63.7 |
| cupiqp | sequential_rollout | 1024 | success | 129.311 | 7918.9 | 118.7 |
| cupiqp | sequential_rollout | 2048 | success | 159.609 | 12831.3 | 228.4 |
| cupiqp | sequential_rollout | 4096 | success | 258.191 | 15864.3 | 448.0 |
| cupiqp | ppo_forward | 256 | success | 107.515 | 2381.1 | 47.1 |
| cupiqp | ppo_forward | 512 | success | 110.410 | 4637.3 | 84.8 |
| cupiqp | ppo_forward | 1024 | success | 112.196 | 9126.9 | 161.5 |
| cupiqp | ppo_forward | 2048 | success | 120.216 | 17036.0 | 309.6 |
| cupiqp | ppo_forward | 4096 | success | 146.566 | 27946.4 | 610.3 |
| cupiqp | ppo_backward | 256 | success | 113.044 | 2264.6 | 54.9 |
| cupiqp | ppo_backward | 512 | success | 115.748 | 4423.4 | 93.2 |
| cupiqp | ppo_backward | 1024 | success | 116.168 | 8814.8 | 169.0 |
| cupiqp | ppo_backward | 2048 | success | 117.250 | 17466.9 | 313.5 |
| cupiqp | ppo_backward | 4096 | success | 146.736 | 27914.1 | 609.8 |
| cupiqp | complete_iteration | 256 | success | 246.891 | 1036.9 | 55.0 |
| cupiqp | complete_iteration | 512 | success | 331.907 | 1542.6 | 91.5 |
| cupiqp | complete_iteration | 1024 | success | 295.899 | 3460.6 | 167.9 |
| cupiqp | complete_iteration | 2048 | success | 345.528 | 5927.2 | 312.5 |
| cupiqp | complete_iteration | 4096 | success | 474.658 | 8629.4 | 610.9 |
