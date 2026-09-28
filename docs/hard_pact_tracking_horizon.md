# Outer velocity horizon and tracking-conflict diagnostics

HardPACT's `algorithm.hard_pact_qp` settings include:

```python
qp_velocity_loss_horizon_s = 0.020  # None: captured per-row physics dt
diagnostics_level = 'physical'     # minimal skips these diagnostics entirely
```

The same horizon extrapolates body-frame `[vx, vy, omega_z]` for the outer
actor-facing losses: `y_pred = y_body + H*qdd[[0,1,5]]`. This is a
constant-derivative prediction, not a multi-step dynamics rollout. Inner QP
objectives, constraints, certification and observed-transition PINN losses
continue using the actual physics timestep. Increasing H changes gradients;
`lambda_qp_velocity_xy` and `lambda_qp_velocity_yaw` remain independently
configurable and are not automatically rescaled. Both zero skips outer losses.

Missing horizon in old configs/capture packets means `None`. Resolved configs,
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
  stance, attitude, GRF-reference and **inner-dt** planar objective costs.
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

## Focused validation

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
