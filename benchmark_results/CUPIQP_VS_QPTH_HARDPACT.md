# Go2 HardPACT: cuPIQP versus qpth

## Purpose and conditions

`scripts/benchmark_hard_pact_qp.py` measured identical captured 54-variable
HardPACT QPs on an RTX 4090. `scripts/benchmark_hard_pact_training.py` launched
the real Isaac Lab full task with 4096 environments, per-substep QPs,
differentiable PPO replay, and inverse/rollout BARD losses active from
iteration zero. Software: Python 3.11.16, Torch 2.7.0+cu128, cuPIQP 0.1.0,
qpth 0.0.18, BARD 0.4.3, Isaac Sim 5.1.0.

## Findings

At standalone batch 4096, dense cuPIQP completed a representative iteration in
`474.66 +/- 96.83 ms` using `610.85 MiB`; qpth required
`16457.48 +/- 500.62 ms` and `1991.27 MiB`. Both had 100% certified full-stage
solutions and zero inequality violation. Dense cuPIQP is therefore the better
QP backend for HardPACT's rollout/PPO hot paths; sparse cuPIQP was slower.

The real 4096-environment BARD-active training attempts did not complete five
iterations: cuPIQP QP chunks 4096, 2048, 1024, and 512 all reached the first
differentiable BARD ABA update and exhausted the 24 GiB GPU. Recorded peaks
were 23529, 24057, 23959, and 23203 MiB respectively. This identifies the
retained BARD graph/workspace—not standalone cuPIQP—as the current full-run
memory limit. No end-to-end iteration mean is claimed.

An 8-environment real Isaac Lab timing smoke completed with both losses active:
inverse dynamics used `96.20 ms/update` (`4.81 ms/minibatch`), while rollout ABA
used `35937.00 ms/update` (`1796.85 ms/minibatch`), over 20 calls each. This
small smoke validates timing/wiring but is not a 4096-environment throughput
measurement.

Detailed data: `hard_pact_qp_gpu_cupiqp_dense_final/`,
`hard_pact_qp_gpu_final/`, and
`hard_pact_training_isaaclab_cupiqp_bard_{final,search}/`.
