# HardPACT active-constraint QP correctness and bounded GPU benchmark

Opt in with `algorithm.qp['qp_update_mode'] = 'active_constraint_update'`
and `qp_solver='cupiqp'`. Existing defaults are unchanged.

## Implementation

cuPIQP **0.1.0** was inspected in `lr_lab_cupiqp`: `DenseSolver.solve()`
takes no warm-start argument; `update(P,c,A,b,G,h_u,h_l,x_u,x_l,...)`
refreshes data. Its returned variables are unpreconditioned. Owned copies of
`x,y,z_u,z_bl,z_bu,s_u,s_bl,s_bu` seed canonical constraint IDs, including
native bounds. No interior-point factor is used by the active update.

At k=0 use normal cuPIQP. At later substeps, refactor the current ECQP
using Cholesky and reject failed/tiny normalized pivots. Contact-slack working
sets propose the nearest of their three faces; other binding candidates use
the configurable margin/dual cutoffs. These are proposals only: finite,
physical/scaled primal, stationarity, multiplier sign, complementarity and
gap checks all remain mandatory. Rejected rows alone enter full cuPIQP and
the unchanged recovery cascade. Cache ownership is by environment ID;
resets, settings, shape/device/dtype and native-bound patterns invalidate it.
Numerical Jacobian zeros are not changes to the fixed matrix structure.

PPO still performs its own differentiable cuPIQP solve. It reconstructs the
held k=0 GRF conditioning separately from live sampled-substep PD. Raw-action
likelihood, PCGrad, objective/constraints, torque clipping, sampling and
checkpoint keys are unchanged. Deployment metadata includes the mode and
actual tolerance configuration. Normal profiling remains opt-in.

Changed code: `hard_pact_active_constraints.py`, `hard_pact_qp.py`,
`hard_pact_qp_backends.py`, `hard_pact_qp_diagnostics.py`, `ppo_hard_pact.py`,
`pact_runner.py`, `go2_hard_pact.py`, `go2_hard_pact_config.py`, and
`deployment.py`. Tests: new `test_hard_pact_active_constraints.py`, updated
`test_hard_pact_action_replay.py` and `test_hard_pact_two_anchor_smoke.py`.
Benchmark: `scripts/benchmark_hard_pact_active_constraints.py`.

## Correctness commands

Commands run from repository root, using physical GPU 0 outside the sandbox:

```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/hardpact_mpl \
  conda run --no-capture-output -n lr_lab_cupiqp python -m pytest -q \
  tests/test_hard_pact_active_constraints.py \
  tests/test_hard_pact_two_anchor_smoke.py \
  tests/test_hard_pact_action_replay.py tests/test_go2_hard_pact_qp.py \
  tests/test_hard_pact_cupiqp_pool.py tests/test_go2_hard_pact_physics_heads.py --tb=short
```

**117 passed, 12 subtests passed, 3 warnings in 12.30 s.** Coverage includes
owned native-bound mapping, changing mechanics, contact transitions,
redundancy/rank rejection, mixed recovery and exceptions, reset/reordering
across chunks, full-solver retry after factor exceptions, sampled PPO
forward/backward, rollout aggregation and existing modes/deployment/migration.
Two warnings are deliberately forced qpth exceptions; one is hppfcl deprecation.

Declared tolerances: ECQP algebra float64 `1e-11`, float32 `2e-5`;
full-cuPIQP physical primal parity float64 `atol=.003`, float32 `.03`, both
`rtol=1e-4`; objective `atol=rtol=1e-4`. Reference convergence and gap
tolerances are tightened to `1e-9`/`1e-6`, respectively, not acceptance
tolerances relaxed. Isolated PPO forward parity uses `atol=rtol=1e-9`;
gradient parity `atol=1e-9, rtol=1e-7`, with finite nonzero gradients to
torque, GRF, wrench and contact predictions.

The additional existing `test_hard_pact_qp_reuse.py` float64 relaxed-stage
capacity test (`[True-False-dtype1]`) fails: max difference **0.0604101**,
tolerance **2e-6**. The identical failure was reproduced using the original
QP class loaded read-only from `git show HEAD:rsl_rl/algorithms/hard_pact_qp.py`.
Its tolerance and legacy implementation were not changed or hidden with xfail.
Final rerun of the command above with `tests/test_hard_pact_qp_reuse.py` also
included: **136 passed, 12 subtests passed, 1 failed, 3 warnings in 35.74 s**.

## Bounded performance measurement

```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/hardpact_mpl \
  conda run --no-capture-output -n lr_lab_cupiqp python \
  scripts/benchmark_hard_pact_active_constraints.py \
  --batch-size 32 --intervals 5 --device cuda:0 \
  --output /tmp/hard_pact_active_constraints_b32_cholesky.json
```

RTX 4090; PyTorch **2.7.0+cu128**; cuPIQP **0.1.0**. Seed 7301, float32,
four 5-ms substeps, one untimed interval, five measured intervals. Identical
canonical state sequence, real BARD mechanics refreshed each substep,
bounded PD, production solver settings, certification and recovery included.
This is **not** an Isaac Lab simulation or training benchmark; no policy
network/simulator stepping time is included. Environments were not modified.

| Per batched control interval | Every-substep full | Active update |
|---|---:|---:|
| Total wall time, mean ± population std (ms) | 148.55 ± 40.30 | 165.78 ± 47.98 |
| BARD mechanics, mean CUDA-event time (ms) | 39.94 | 37.26 |
| Assembly/packing (ms) | 2.79 | 2.60 |
| cuPIQP setup/update (ms) | 2.26 | 15.30 |
| cuPIQP solve, including rejected-row retries (ms) | 87.12 | 75.98 |
| Active factorization/certification (ms) | — | 6.63 |
| Other certification/recovery (ms) | 4.69 | 6.84 |
| Peak Torch allocation (MiB) | 12.82 | 17.09 |
| CuPy pool reserved at end (MiB) | 2.44 | 6.12 |

Event windows are not additive wall-clock partitions. CuPy reserved values
include reusable allocator blocks, not total process peak GPU memory.
Active acceptance: **409/480 eligible rows (85.21%)**; **71** rejected-row
full retries plus **160** mandatory k=0 full solves. All 640 final rows passed
the full-stage certificate; no relaxed/elastic/analytic recoveries in this
benchmark (those paths are separately forced in tests). Exact torque/rate
violations: **0 Nm**. Maximum selected normalized equality/inequality
residual: **2.52e-5 / 8.98e-4**, below the unchanged `1e-3` primal tolerance.
Rejected candidate residuals are logged separately and never executed.

Same-data maximum differences against production full cuPIQP:
torque **0.000474 Nm**, GRF **0.004406 N**, qdd **0.014381**,
slack **0.021049 m/s²**, objective **0.000127733**. This comparison includes
the production interior-point gap-report tolerance, unlike the tighter unit
reference. Full per-interval samples/settings/metrics are in
[hard_pact_active_constraints_gpu_b32.json.txt](hard_pact_active_constraints_gpu_b32.json.txt)
(JSON saved as text because the repository ignores `*.json`).

**No speedup established:** this small batch was 11.6% slower overall despite
fewer cuPIQP rows. Changed fallback capacities/setup and dispatch remain costs.
Real-simulator stability/collapse prevention and large-batch throughput remain
untested. `qp/rollout/active/*` reports attempted/accepted/full counts,
conditional acceptance/fallback, rejection reasons and residuals. Logged
torque/objective changes reference the previous owned snapshot, not an
uncomputed same-state full solve; only the benchmark performs that comparison.
