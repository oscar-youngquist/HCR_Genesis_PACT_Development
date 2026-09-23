# Optional body-velocity QP tracking

`algorithm.hard_pact_qp` in `go2_hard_pact_config.py`:

```python
'planar_velocity_weight': 0.0,  # opt in, e.g. 1.0
'yaw_rate_weight': 0.0,         # independently opt in, e.g. 1.0
'planar_velocity_scale_m_s': 1.0,
'yaw_rate_scale_rad_s': 1.0,
```

Both default weights are zero. No constraint, acceptance setting, outer loss,
curriculum or PCGrad ownership changes. Recovery retains the identical tracking
quadratic in its 24-variable leading block. Numeric weight/scale changes refresh
Q/p without invalidating solver pools. Command replay adds three physical values
per sampled transition only when enabled.

## Frames and objective

`x=[tau_12;tilde_f_12]`; physical swing force is zero through the existing mask.
`a=A_dyn*x+b_dyn` uses the existing mass solve and wrench contribution once.
Canonical base acceleration is the derivative of body-frame free-flyer velocity.
Rewards compare body `vx,vy,omega_z` (the last is not Euler yaw-angle derivative).
For root rotation R and body twist `(v,w)`, classical world acceleration is
`R*(a_linear+w cross v)`. Differentiating `R.T*v_world` subtracts `w cross v`,
so transport terms cancel: `H` selects canonical coordinates `[0,1,5]`, `c=0`.

At the actual physics dt, `C=dt*H*A_dyn` and
`e=y_body+dt*H*b_dyn-command_body`. For diagonal
`W=[w_planar/s_planar²,w_planar/s_planar²,w_yaw/s_yaw²]`, add
`2*C.T*W*C` to Q and `2*C.T*W*e` to p before existing variable scaling.
Only learned torque/force/wrench inputs retain gradients. Commands, transforms,
mechanics and state are detached. The same resolved physical yaw-rate command
used by rewards is captured at the sampled substep; PPO never reads later live
commands. Enabled reassembly rejects packets without that command/state.

Scheduled physical diagnostics report current and one-step predicted planar RMS
and yaw absolute errors, weighted dimensionless costs, and baseline-nominal/
predicted-GRF versus full-candidate errors. Prefix:
`model_velocity_tracking/{primary|recovery}/{accepted|rejected}` within each
rollout/PPO metric namespace. Means are row-count weighted. These are model
predictions, not measured improvements or blended-command certificates.

## Validation

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests conda run --no-capture-output -n lr_lab_cupiqp \
python -m pytest -q tests/test_hard_pact_velocity_objective.py \
tests/test_hard_pact_qp_modes.py tests/test_hard_pact_rate_recovery.py \
tests/test_hard_pact_runtime_optimizations.py tests/test_hard_pact_qp_diagnose.py \
tests/test_eval_hard_pact_frozen.py
```

Result: **48 passed, 1 dependency deprecation warning, 6.36 s**, with CUDA
access. Finite differences use float64, step 1e-4, rtol 2e-3 / atol 2e-5;
the initial auto/float32 probe did not resolve that small subtraction reliably.
Production auto precision and solver tolerances were not changed.

Matched frozen-checkpoint comparison command (run twice, w=0 and w=1):

```bash
RUN=logs/hardpact_iclr/go2_pact_rough/Sep23_18-28-31_hard_pact_full_isaaclab
for w in 0 1; do
  SIMULATOR=isaaclab PYTHONPATH=.:tests conda run --no-capture-output -n lr_lab_cupiqp \
  python scripts/eval_hard_pact_frozen.py --checkpoint "$RUN/model_0.pt" \
    --resolved-config "$RUN/hard_pact_resolved_config.json" --solver cupiqp \
    --num-envs 4 --duration 0.5 --seeds 1 --smoke --variant qp_random_one_substep \
    --packet-limit 0 --planar-velocity-weight "$w" --yaw-rate-weight "$w" \
    --output-dir "/tmp/hardpact_velocity_weight_$w"
done
```

Executed artifacts: `/tmp/hardpact_velocity_objective_off` and
`/tmp/hardpact_velocity_objective_on` (summary/metadata JSON, CSV, bounded traces).
Isaac Lab/cuPIQP, CUDA GPU 0; four environments, deterministic policy,
100 common prefix steps plus 25 evaluation steps, explicit smoke reset,
small 2x2 terrain, no optimization. Checkpoint weights, initial-state, scenario,
and prefix-torque hashes match. Full correction is active in frozen evaluation.

| Metric | Off | Both weights 1 |
|---|---:|---:|
| Planar component RMSE (m/s) | 0.422277 | 0.422278 |
| Body yaw-rate RMSE (rad/s) | 0.601523 | 0.592474 |
| Applied correction RMS (Nm, all substeps/joints) | 0.392658 | 0.421569 |
| Primary acceptance / recovery / fallback | 100% / 0% / 0% | 100% / 0% / 0% |
| Absolute torque violation (Nm) | 0 | 0 |
| Rollout wall seconds, including prefix/diagnostics | 13.936 | 13.718 |

Both produced the expected 100 QP dispatches and unchanged model weights.
Single short trials with diagnostic overhead (and other GPU test activity) are
not a speed benchmark or evidence of sustained tracking/stability improvement.
