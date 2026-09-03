# Go2 HardPACT standalone qpth/cuPIQP canonical-QP throughput and VRAM benchmark

## Experiment

- Script: `scripts/benchmark_hard_pact_qp.py`.
- Conditions: captured 54-variable HardPACT QPs; float32 CUDA; batches 256--4096; RTX 4090; Python 3.11.16; Torch 2.7.0+cu128; qpth 0.0.18 and cuPIQP 0.1.0. Each solver/mode/batch ran in a fresh process.
- Purpose: isolate solver throughput, differentiation cost, certification, fallback rate, and VRAM independently of simulation. It shows whether a solver is practical for per-substep rollout and PPO replay; it is not a full-training timing claim.

| Solver | Mode | Batch | Status | Mean ms | QPs/s | Peak MiB |
|---|---|---:|---|---:|---:|---:|
| qpth | assembly | 256 | success | 1.454 | 176059.4 | 35.0 |
| qpth | assembly | 512 | success | 1.392 | 367733.2 | 61.1 |
| qpth | assembly | 1024 | success | 0.575 | 1779746.9 | 114.1 |
| qpth | assembly | 2048 | success | 1.629 | 1257478.3 | 218.5 |
| qpth | assembly | 4096 | success | 1.742 | 2350952.4 | 429.7 |
| qpth | fused_bard_assembly | 256 | unsupported_dependency | nan | nan | n/a |
| qpth | fused_bard_assembly | 512 | unsupported_dependency | nan | nan | n/a |
| qpth | fused_bard_assembly | 1024 | unsupported_dependency | nan | nan | n/a |
| qpth | fused_bard_assembly | 2048 | unsupported_dependency | nan | nan | n/a |
| qpth | fused_bard_assembly | 4096 | unsupported_dependency | nan | nan | n/a |
| qpth | rollout | 256 | success | 352.137 | 727.0 | 104.3 |
| qpth | rollout | 512 | success | 791.675 | 646.7 | 202.1 |
| qpth | rollout | 1024 | success | 1262.297 | 811.2 | 389.5 |
| qpth | rollout | 2048 | success | 2138.231 | 957.8 | 765.0 |
| qpth | rollout | 4096 | success | 3884.395 | 1054.5 | 1524.6 |
| qpth | sequential_rollout | 256 | success | 1462.758 | 175.0 | 104.2 |
| qpth | sequential_rollout | 512 | success | 3145.970 | 162.7 | 202.0 |
| qpth | sequential_rollout | 1024 | success | 5146.521 | 199.0 | 389.5 |
| qpth | sequential_rollout | 2048 | success | 8465.483 | 241.9 | 766.1 |
| qpth | sequential_rollout | 4096 | success | 14953.431 | 273.9 | 1524.8 |
| qpth | ppo_forward | 256 | success | 361.185 | 708.8 | 145.3 |
| qpth | ppo_forward | 512 | success | 770.445 | 664.6 | 279.4 |
| qpth | ppo_forward | 1024 | success | 1416.309 | 723.0 | 544.6 |
| qpth | ppo_forward | 2048 | success | 2092.258 | 978.8 | 1074.1 |
| qpth | ppo_forward | 4096 | success | 3688.733 | 1110.4 | 2140.8 |
| qpth | ppo_backward | 256 | success | 298.801 | 856.8 | 142.1 |
| qpth | ppo_backward | 512 | success | 499.668 | 1024.7 | 268.2 |
| qpth | ppo_backward | 1024 | success | 953.821 | 1073.6 | 512.0 |
| qpth | ppo_backward | 2048 | success | 1815.570 | 1128.0 | 1001.4 |
| qpth | ppo_backward | 4096 | success | 3518.293 | 1164.2 | 1989.4 |
| qpth | complete_iteration | 256 | success | 1314.761 | 194.7 | 142.9 |
| qpth | complete_iteration | 512 | success | 2320.969 | 220.6 | 268.0 |
| qpth | complete_iteration | 1024 | success | 4532.243 | 225.9 | 512.2 |
| qpth | complete_iteration | 2048 | success | 8509.675 | 240.7 | 1001.1 |
| qpth | complete_iteration | 4096 | success | 16457.480 | 248.9 | 1991.3 |
| cupiqp | assembly | 256 | success | 0.450 | 568544.7 | 35.0 |
| cupiqp | assembly | 512 | success | 0.473 | 1082787.3 | 61.1 |
| cupiqp | assembly | 1024 | success | 0.566 | 1807960.4 | 114.1 |
| cupiqp | assembly | 2048 | success | 0.939 | 2180950.8 | 218.5 |
| cupiqp | assembly | 4096 | success | 1.736 | 2359817.2 | 429.7 |
| cupiqp | fused_bard_assembly | 256 | unsupported_dependency | nan | nan | n/a |
| cupiqp | fused_bard_assembly | 512 | unsupported_dependency | nan | nan | n/a |
| cupiqp | fused_bard_assembly | 1024 | unsupported_dependency | nan | nan | n/a |
| cupiqp | fused_bard_assembly | 2048 | unsupported_dependency | nan | nan | n/a |
| cupiqp | fused_bard_assembly | 4096 | unsupported_dependency | nan | nan | n/a |
| cupiqp | rollout | 256 | success | 16.174 | 15828.1 | 36.3 |
| cupiqp | rollout | 512 | success | 16.866 | 30356.0 | 63.7 |
| cupiqp | rollout | 1024 | success | 22.507 | 45497.3 | 118.7 |
| cupiqp | rollout | 2048 | success | 34.508 | 59348.2 | 228.5 |
| cupiqp | rollout | 4096 | success | 65.669 | 62373.3 | 448.2 |
| cupiqp | sequential_rollout | 256 | success | 61.150 | 4186.4 | 36.3 |
| cupiqp | sequential_rollout | 512 | success | 68.189 | 7508.5 | 63.7 |
| cupiqp | sequential_rollout | 1024 | success | 87.188 | 11744.7 | 118.7 |
| cupiqp | sequential_rollout | 2048 | success | 137.011 | 14947.7 | 228.4 |
| cupiqp | sequential_rollout | 4096 | success | 259.418 | 15789.2 | 448.0 |
| cupiqp | ppo_forward | 256 | success | 177.938 | 1438.7 | 47.1 |
| cupiqp | ppo_forward | 512 | success | 181.539 | 2820.3 | 84.8 |
| cupiqp | ppo_forward | 1024 | success | 185.226 | 5528.4 | 161.5 |
| cupiqp | ppo_forward | 2048 | success | 197.819 | 10352.9 | 309.6 |
| cupiqp | ppo_forward | 4096 | success | 227.646 | 17992.9 | 610.3 |
| cupiqp | ppo_backward | 256 | success | 184.041 | 1391.0 | 54.9 |
| cupiqp | ppo_backward | 512 | success | 185.454 | 2760.8 | 93.2 |
| cupiqp | ppo_backward | 1024 | success | 185.116 | 5531.7 | 169.0 |
| cupiqp | ppo_backward | 2048 | success | 201.479 | 10164.8 | 313.5 |
| cupiqp | ppo_backward | 4096 | success | 227.763 | 17983.6 | 609.8 |
| cupiqp | complete_iteration | 256 | unexpected_error | nan | nan | n/a |
| cupiqp | complete_iteration | 512 | unexpected_error | nan | nan | n/a |
| cupiqp | complete_iteration | 1024 | unexpected_error | nan | nan | n/a |
| cupiqp | complete_iteration | 2048 | unexpected_error | nan | nan | n/a |
| cupiqp | complete_iteration | 4096 | unexpected_error | nan | nan | n/a |
