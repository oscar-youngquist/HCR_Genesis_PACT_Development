# HardPACT QP curricula

Configure `algorithm.hard_pact_qp` in `go2_hard_pact_config.py`:

```python
'correction_ramp_enabled': True,
'correction_ramp_start_offset': 0,  # relative to warmup_iterations
'correction_ramp_duration': 1000,
'objective_curriculum_enabled': True,
'contact_acceleration_weight_initial': None,  # 25% of final
'contact_acceleration_weight_final': None,    # contact_acceleration_weight
'attitude_weight_initial': None,              # 25% of final
'attitude_weight_final': None,                # attitude_weight
'objective_curriculum_start': None,           # ramp completion, absolute iteration
'objective_curriculum_progress_delta': 0.05,
'objective_curriculum_step_interval': 10,
'objective_curriculum_ema_alpha': 0.05,
'objective_curriculum_window': 200,
'objective_curriculum_quantile': 0.9,
'objective_curriculum_recovery_ratio': 0.98,
'objective_curriculum_recovery_iterations': 50,
'objective_curriculum_baseline_override': None,
'projection_contact_acceleration_weight': 0.10,
'objective_curriculum_min_tracking': 0.5,
'objective_curriculum_min_samples': 1,         # environment-control-step samples
```

Each switch is independent. Both disabled reproduce unscheduled execution and
weights. Zero ramp duration means full correction immediately. The performance
gate observes the raw linear-velocity tracking reward before reward scaling,
including when command/domain-randomization curricula are disabled. HardPACT's
domain-randomization scheduler receives this exact same rollout average, rather
than scaled episodic returns. Neither score divides by episode length or command
bounds. The fixed absolute-error kernel is unchanged; larger actual tracking
errors still lower performance. Missing,
insufficient, or nonfinite evidence resets sustained recovery. The quantile of
the final 200 valid pre-QP EMA samples is frozen BEFORE activation (even alpha=0).
If activation occurs sooner, available pre-QP samples are used and their count
is logged. No pre-QP evidence means no automatic baseline. Post-QP EMA continues
updating, but cannot replace or lower the reference. The threshold is
`max(min_tracking, 0.98*frozen_reference)`. Require 50 consecutive valid iterations
strictly above it after the configured earliest start before each increase;
reset the counter after an increase, invalid/missing evidence, skipped iteration,
or tracking failure. No domain-randomization state changes.

Weights and alpha are frozen before rollout and throughout PPO. A completed
iteration advances only the next snapshot. Stance/attitude curricula affect ONLY
inner objectives. Outer stance uses fixed `projection_contact_acceleration_weight`
(0.10); zero removes only its explicit penalty, not the inner QP's gradient effects.
The overall multiplier remains `lambda_projection` (the repository's existing
name for the QP-projection multiplier). Solver caches survive
weight changes, while numeric matrices are updated normally.

Accepted commands execute `base + alpha*(QP-base)`; alpha endpoints are exact.
Base respects the non-QP rate-clipping flag. Rejected deterministic fallbacks and
unsolved substeps are unchanged. Accepted recovery is not strictly rate-reclipped.
Partial corrections are **not certified** by the full candidate certificate.

Replay stores full candidate and actual executed torque separately. The previous
torque/rate center and PINN interval-average actuation use actual blended execution.
PPO retains raw-action likelihood; projection losses/gradients use the full QP,
never alpha-scaled. Existing nominal GRF conditioning and detached PINN actuation
are unchanged.

Version-2 checkpoints save independent progress, pre-QP EMA history, frozen
reference/freeze state, recovery counter, and absolute activation/schedule origin.
Older pre-QP histories remain usable. Older POST-QP histories are untrusted:
preserve progress but pause increases until `objective_curriculum_baseline_override`
is explicitly set to a known raw pre-QP reference. Never estimate it from degraded
post-QP performance. A missing curriculum entry initializes progress to zero and
similarly blocks when resumed after activation without a baseline. Resume with
the same configuration for exact continuation; torque-ramp timing is unchanged.

`curriculum/qp/*` logs alpha, current/next progress, effective weights, EMA,
threshold (NaN until a baseline), recovery count, and advancement. Numeric block
reason: 0=advanced, 1=disabled/invalid, 2=pre-QP collection, 3=missing baseline,
4=earliest start, 5=below/equal threshold, 6=sustained recovery/interval pending,
7=complete. `qp/ppo/projection_components/{torque,stance}_{unweighted,weighted}`
reports normalized row-weighted means; weighted contributions include the fixed
outer coefficient and overall lambda. `qp/rollout/execution/*` reports
candidate/executed absolute correction sums in Nm with coordinate counts, and
partially corrected row counts. Candidate acceptance is distinct from execution.

Focused validation (no training):

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests conda run --no-capture-output -n lr_lab_cupiqp \
  python -m pytest -q tests/test_hard_pact_qp_curriculum.py \
  tests/test_hard_pact_qp_modes.py tests/test_hard_pact_rate_recovery.py
```

Frozen-baseline/coefficient/capture and GPU PPO validation:

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests conda run --no-capture-output -n lr_lab_cupiqp \
python -m pytest -q --tb=short tests/test_hard_pact_frozen_qp_curriculum.py \
tests/test_hard_pact_qp_curriculum.py tests/test_hard_pact_qp_diagnose.py \
tests/test_hard_pact_diagnose_v2.py tests/test_hard_pact_actor_velocity_loss.py \
tests/test_hard_pact_contact_indexing_and_inverse_gate.py
```

Result: 49 passed, one dependency deprecation warning, 6.56 s; CUDA/cuPIQP
tiny PPO backwards included. No training/capture run was started.

Opt-in follow-up capture (run from repository root, choose checkpoint):

```bash
conda activate lr_lab_cupiqp
export SIMULATOR=isaaclab PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
RUN=logs/hardpact_iclr/go2_pact_rough/Sep23_18-28-31_hard_pact_full_isaaclab
OUT=/tmp/hardpact_recovery_diagnosis
python scripts/diagnose_hard_pact_qp.py capture --checkpoint "$RUN/model_0.pt" \
  --resolved-config "$RUN/hard_pact_resolved_config.json" --backend isaaclab \
  --device cuda:0 --warmup-iterations 2 --iteration-limit 1 \
  --capture-limit 32 --byte-limit-mib 2048 --capture-recovery-extremes --output-dir "$OUT"
python scripts/diagnose_hard_pact_qp.py replay "$OUT"/captures/extreme_*.pt \
  --device cuda:0 --individual-row-limit 4 --output-dir "$OUT/replay"
```

Capture runs normal rollout/PPO at the checkpoint's original environment count;
it is NOT a small simulation test. The optional flag reserves half of the existing
byte budget for eight record-max slots (rollout/PPO x correction Nm, rate slack
Nm, predicted position rad, predicted velocity rad/s). Each slot retains a complete
solver batch and original row IDs, not an isolated substitute problem. Only
accepted recovery candidates qualify; joint violations are MODEL predictions,
not measured motion. Bounded preceding history and acceptance metadata remain.
Oversized packets are explicitly counted as budget drops; a maximum is only
guaranteed among retainable packets. Existing evidence is never overwritten by a
new recorder. No extreme files appear if no qualifying recovery was observed.
