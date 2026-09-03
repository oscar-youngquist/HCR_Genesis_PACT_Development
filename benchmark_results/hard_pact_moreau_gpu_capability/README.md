# Go2 HardPACT Moreau GPU capability and dependency benchmark

## Experiment

- Script: `scripts/benchmark_hard_pact_qp.py` in the isolated Python-3.12 Moreau environment.
- Conditions: captured canonical HardPACT QPs on an RTX 4090; Moreau 0.3.3; Torch 2.10.0+cu128.
- Purpose: establish whether Moreau can execute the intended GPU rollout/PPO paths. Every cell reports an explicit unsupported capability because no Moreau CUDA license is installed; no CPU or alternate-solver fallback was timed as Moreau GPU performance.

| Solver | Mode | Batch | Status | Mean ms | QPs/s | Peak MiB |
|---|---|---:|---|---:|---:|---:|
| moreau | rollout | 256 | unsupported_dependency | nan | nan | n/a |
| moreau | rollout | 512 | unsupported_dependency | nan | nan | n/a |
| moreau | rollout | 1024 | unsupported_dependency | nan | nan | n/a |
| moreau | rollout | 2048 | unsupported_dependency | nan | nan | n/a |
| moreau | rollout | 4096 | unsupported_dependency | nan | nan | n/a |
| moreau | ppo_backward | 256 | unsupported_dependency | nan | nan | n/a |
| moreau | ppo_backward | 512 | unsupported_dependency | nan | nan | n/a |
| moreau | ppo_backward | 1024 | unsupported_dependency | nan | nan | n/a |
| moreau | ppo_backward | 2048 | unsupported_dependency | nan | nan | n/a |
| moreau | ppo_backward | 4096 | unsupported_dependency | nan | nan | n/a |
| moreau | complete_iteration | 256 | unsupported_dependency | nan | nan | n/a |
| moreau | complete_iteration | 512 | unsupported_dependency | nan | nan | n/a |
| moreau | complete_iteration | 1024 | unsupported_dependency | nan | nan | n/a |
| moreau | complete_iteration | 2048 | unsupported_dependency | nan | nan | n/a |
| moreau | complete_iteration | 4096 | unsupported_dependency | nan | nan | n/a |
