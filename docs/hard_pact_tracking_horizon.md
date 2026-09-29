# Outer velocity horizon and tracking-conflict diagnostics

HardPACT's `algorithm.hard_pact_qp` settings include:

```python
qp_velocity_loss_horizon_s = 0.020  # None: captured per-row physics dt
qp_velocity_objective_horizon_s = 0.020  # independent inner xy/yaw horizon
torque_rate_constraint_weight = 0.0      # QP switch: 0 disabled; any positive value enabled
diagnostics_level = 'physical'     # minimal skips these diagnostics entirely
```

The same horizon extrapolates body-frame `[vx, vy, omega_z]` for the outer
actor-facing losses: `y_pred = y_body + H*qdd[[0,1,5]]`. This is a
constant-derivative prediction, not a multi-step dynamics rollout. Inner QP
planar/yaw objectives use the independent `qp_velocity_objective_horizon_s`.
Constraints, certification and observed-transition PINN losses continue using
the actual physics timestep. Increasing either H changes gradients;
`lambda_qp_velocity_xy` and `lambda_qp_velocity_yaw` remain independently
configurable and are not automatically rescaled. Both zero skips outer losses.

Missing horizons in old configs/capture packets mean `None`. Resolved configs,
deployment/capture settings and checkpoint horizon metadata record the choice;
resume uses the supplied training config, as for the other loss settings.

## Scheduled metric guide

Existing physical/full cadence controls
`qp/{rollout,ppo}/model_tracking_conflict/{primary,recovery}/accepted/*`.
These are **full-candidate model diagnostics**; nominal predictions may be
infeasible. Rollout `state/*` describes sampled measured state; PPO describes
stored replay state, not newly measured tracking. Execution alpha/corrections
remain in the existing execution metrics: candidate acceptance never certifies
a partially blended command.

- `state/*`: body velocity/commands, planar L2 error [m/s], signed along-command
  error (negative = underspeed), lateral error and underspeed fraction.
- `{physics_dt,outer_horizon}/*`: nominal, torque-only and full-candidate errors,
  error improvement (positive = better), signed velocity changes, and torque/
  force contributions [m/s]. Torque-only retains nominal GRF and fixed wrench.
- `weighted_cost_delta/*`: candidate minus nominal dimensionless weighted
  stance, attitude, GRF-reference and **configured inner-horizon** planar costs.
  `worse_tracking_better_*_fraction` reports association, not causation.
- `original_bounds/*`: coordinate-weighted bound proximity and raw violation
  magnitudes [Nm, rad, rad/s, N]; `*/worsened/*` conditions proximity on worsened
  tracking. Swing friction/unilateral coordinates are excluded. Recovery
  reports original-bound violations separately from accepted softened
  feasibility and its joint [rad/s²]/rate [Nm] slacks.

All pairs use identical finite, real, accepted rows; padding is excluded before
aggregation. `matched_rows`, `nonfinite_rows` and each metric's `/samples`
expose denominators. Empty means are NaN, not successful zeros. Means combine
sums/counts across chunks. Directional statistics require command norm >0.001
m/s; ordinary errors still include zero commands. Worsening threshold is
0.0001 m/s. Bound proximity is absolute margin <=0.001 Nm/N/(rad/s), or
0.0001 rad for position; exceedance magnitudes are not thresholded.

No additional QP solve, backward, host transfer or synchronization is used.

## Optional QP torque-rate constraints

Legacy/missing `torque_rate_constraint_weight` defaults to 1.0. HardPACT
experiment config explicitly uses 0.0. This is an enable switch, **not** a
continuous softening weight. Any positive value restores the existing hard
primary bound at `torque_rate_limit_nm_s`; recovery penalties remain independent.

With zero, primary native/canonical actuator bounds are absolute magnitude only.
Recovery is genuinely 36 variables / 80 canonical inequalities (44 general
inequalities plus native bounds): torque12, force12, joint-envelope slack12.
There are no rate rows, rate-slack variables/costs or outer rate-slack loss.
With rate enabled, recovery retains 48 variables / 116 inequalities (68 general
plus native bounds). Joint position/velocity, friction and absolute torque limits
are unchanged. No independent acceleration cap is introduced.

Rejected QP rows fall back to sanitized magnitude clipping when disabled, or
the existing centered magnitude/rate clipping when enabled. Accepted recovery
commands bypass deterministic rate clipping. Non-QP substeps, warmup and disabled
ablations still obey the independent `control.clip_torque_rate_without_qp` flag;
the torque-rate RL reward also remains independently configured. Thus disabling
the QP rate switch does **not** disable an explicitly enabled non-QP clip/reward.

Diagnostics expose `torque_rate_constraints_enabled=0`; unavailable rate values
are NaN/empty, not successful enforcement. Capture schema 4 records the layout
and both horizons; replay retains schema 3 and older packet defaults/matrices.
Policy weights, optimizer behavior and PCGrad are unchanged.

## Focused validation

Inner-horizon/optional-rate validation (2026-09-29): **83 passed in 5.61 s**,
including CUDA/cuPIQP forward/backward and 36-variable recovery. One existing
`hppfcl` deprecation warning; no training run.

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests conda run --no-capture-output -n lr_lab_cupiqp \
  python -m pytest -q --tb=short tests/test_hard_pact_optional_rate.py \
  tests/test_hard_pact_tracking_horizon.py tests/test_hard_pact_velocity_objective.py \
  tests/test_hard_pact_qp_diagnose.py tests/test_hard_pact_diagnose_v2.py \
  tests/test_hard_pact_reduced_qp.py
```

Additional regression command: **36 passed, 2 failed in 30.95 s**. Both failures
also reproduce with unchanged HEAD QP code: the recovery-gradient fixture's
candidate passes primary instead of entering recovery, and the reward-config
test asserts -0.01 while HEAD config contains -0.001. These unrelated assertions
and reward values were not changed.

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests conda run --no-capture-output -n lr_lab_cupiqp \
  python -m pytest -q --tb=short tests/test_hard_pact_optional_rate.py \
  tests/test_hard_pact_rate_recovery.py tests/test_hard_pact_soft_joint_recovery.py \
  tests/test_hard_pact_qp_modes.py tests/test_hard_pact_runtime_optimizations.py \
  tests/test_hard_pact_torque_rate_reward.py
```

Earlier outer-horizon validation:

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests conda run --no-capture-output -n lr_lab_cupiqp \
  python -m pytest -q --tb=short tests/test_hard_pact_tracking_horizon.py \
  tests/test_hard_pact_velocity_objective.py tests/test_hard_pact_actor_velocity_loss.py \
  tests/test_hard_pact_contact_indexing_and_inverse_gate.py \
  tests/test_hard_pact_qp_diagnose.py tests/test_hard_pact_diagnose_v2.py
```

Validation: 53 passed in 6.49 s with CUDA/cuPIQP available, including the tiny
PPO-backward tests. One existing `hppfcl`→`coal` deprecation warning. No training
run or performance comparison was performed; no speed/stability claim is made.
