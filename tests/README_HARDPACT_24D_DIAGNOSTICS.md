# 24-D QP diagnostics, replay/backward correctness and bounded control smoke

## Configuration and interpretation

HardPACT's QP config now exposes `tensorboard_diagnostics_enabled=True` and
`tensorboard_diagnostics_interval=50` (absolute PPO iteration; zero disables).
Existing `diagnostics_level` still controls physical/full diagnostics. Required
certification remains unconditional. Scheduled iterations collect motion stats
and optional CUDA events; TensorBoard receives scalar summaries in one transfer.
Unscheduled iterations do not compute the new motion/physical diagnostics or
export QP TensorBoard tags. Existing basic losses, reward and tracking logs stay.

All phase tags use `qp/rollout/` or `qp/ppo/`. `real_rows` counts actual QP
problems, not dispatches/padding. `solve_calls` counts batched dispatches.
Problems/environment/control interval divides by the sum of environment counts
over control intervals; solved-substep coverage divides by four times that sum.
Sample histograms use environment-control intervals. Final stages are only
`full` (certified) and `analytic` (uncertified); stage/failure fractions use real
QP rows. Nonfinite outputs divide by returned solves, excluding exceptions.

Physical diagnostics split certified and rejected candidates. They report
normalized weighted torque, force, stance and attitude objective components;
torque correction [Nm], force-reference error [N], model stance acceleration
[m/s²], active constraint fractions and physical margins. Torque/rate rows use
Nm; joint acceleration intersections use rad/s²; friction rows use N. Do not
interpret a mixed-row maximum as having one physical unit. Dynamics residual
blocks use N, Nm and joint Nm. Means reduce finite per-row samples across chunks.
Missing samples are NaN, never a false zero-error claim.

Measured state diagnostics use pre-substep joint state samples, actual physics
dt, delta-v/dt acceleration [rad/s²], delta-acceleration/dt jerk [rad/s³], and
applied-command slew [Nm/s]. Reset and diagnostic gaps invalidate derivative
stencils. Position/velocity exceedance frequencies divide by finite environment-
joint samples; magnitudes [rad, rad/s] include nonviolating zeros. They split
all, certified-command and unsolved/fallback-command rows. This association is
not a causal guarantee about the next state. Base tilt [rad], angular rate
[rad/s], and measured stance horizontal slip [m/s] use current canonical states;
slip uses sensor vertical force >5 N and finite stance-foot samples.

Existing projection/PINN losses and scheduled gradient/PCGrad diagnostics are
reused; no extra backward is introduced. Supervised GRF/wrench metrics remain
unchanged. Optimized/raw-to-measured GRF errors are explicitly NaN because the
compact packet does not retain time-aligned per-solve sensor labels; interval
averages must not be substituted. Solver iterations are unavailable where the
adapter exposes none. CUDA-event timings require opt-in profiling and sync at
export only. Memory tags identify the PyTorch allocator; they exclude separate
cuPIQP/CuPy and simulator allocations. Production peak tags are since the last
allocator-stat reset, not attributed to one solve.

## Deployment and cleanup

Schema **13** records torque/force variable ordering, eliminated acceleration,
soft stance/attitude costs, hard limits, canonical frames, four-vs-one execution,
input conditioning/clipping and certification scope. It records detached actual
rollout PINN torque and certified outer-projection gradients. Old schemas and
unknown formulation config keys fail explicitly; policy tensor keys are unchanged.
No held corrections, active-set execution or recovery cascades are restored.
cuPIQP solver-internal regularization and generic backend ownership remain.

Retired single/two-anchor and active-set test/smoke/benchmark scripts were removed
(recoverable from git). The obsolete 54-D test suite was replaced by fixture-name
aliases to the current 24-D tests; ownership/reuse/exception/replay tests were
migrated. Historical experiment reports remain historical records, not executable
instructions for the current controller.

## Commands and results

Environment: `lr_lab_cupiqp`; GPU smoke uses Isaac Lab, cuPIQP and RTX 4090 on
CUDA device 0. No full training, production autotuning or broad benchmark.

```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/hardpact_mpl \
conda run --no-capture-output -n lr_lab_cupiqp python -m pytest -q \
tests/test_hard_pact_current_diagnostics.py tests/test_hard_pact_qp_exception_capture.py \
tests/test_go2_hard_pact_physics_heads.py tests/test_hard_pact_reduced_qp.py \
tests/test_hard_pact_qp_modes.py tests/test_hard_pact_contact_indexing_and_inverse_gate.py \
tests/test_hard_pact_cupiqp_pool.py tests/test_hard_pact_qp_reuse.py \
tests/test_hard_pact_action_replay.py tests/test_isaaclab_pact_torque_limits.py \
tests/test_eval_hard_pact_frozen.py --tb=short

env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/hardpact_mpl \
conda run --no-capture-output -n lr_lab_cupiqp python -m pytest -q \
tests/test_hard_pact_reduced_qp.py tests/test_hard_pact_cupiqp_pool.py --tb=short

env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MPLCONFIGDIR=/tmp/hardpact_mpl \
conda run --no-capture-output -n lr_lab_cupiqp python -u tests/smoke_hard_pact_qp_modes.py \
--task go2_hard_pact_full_isaaclab --headless --num_envs 8 --gpu cuda:0 --qp_solver cupiqp
```

CPU correctness: **114 passed, 14 skipped, 1 warning, 8 subtests passed in 5.55s**.
Skips are CUDA-only coverage in the CPU invocation; warning is hppfcl deprecation.
CUDA correctness: **43 passed, 1 warning in 8.84s**. Covers cuPIQP float32/64,
pool lifetimes/retained backward, changed Hessians, certified-row isolation and
reference parity. Existing tolerances retained. CPU suite includes real PPO
update/backward with compact replay in both modes; simulator smoke runs no
optimizer updates. No claim of a GPU end-to-end PPO training update in this run.

Real GPU control smoke: **PASS**, 8 environments, 3 intervals per mode, reset
between modes. Every-substep: 12 problems/env, 80 certified/16 analytic, 2152.842 ms
total. Random-one: 3 problems/env, 7 certified/17 analytic, 2306.439 ms total.
Both enforce actuator/rate bounds, finite outputs and zero substep neural calls.
PyTorch peak allocated: 187557888 / 187628032 bytes respectively; reserved:
209715200 bytes for both. Timings include Python assertions, cold initialization
and fallback work; not a speedup comparison or trained-policy stability claim.
Neither unsolved rows nor analytic fallbacks are joint/contact certified.

## Changed files

Runtime: `hard_pact_qp.py`, `hard_pact_qp_diagnostics.py`, `hard_pact_qp_capture.py`,
`pact_runner.py`, `go2_hard_pact.py`, `go2_hard_pact_config.py`, `deployment.py`.
Tests: the eleven CPU files above, `smoke_hard_pact_qp_modes.py`, and fixture
`test_go2_hard_pact_qp.py`. Removed `test_hard_pact_single_anchor.py`,
`test_hard_pact_two_anchor_smoke.py`, `test_hard_pact_active_constraints.py`,
`smoke_hard_pact_single_anchor.py`, and `scripts/benchmark_hard_pact_active_constraints.py`.
