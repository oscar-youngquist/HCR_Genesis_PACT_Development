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
'objective_curriculum_recovery_ratio': 0.9,
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
insufficient, or nonfinite evidence pauses progress. The rolling quantile of
tracking EMA defines the recovery reference; no domain-randomization state changes.

Weights and alpha are frozen before rollout and throughout PPO. A completed
iteration advances only the next snapshot. Stance weights affect inner and outer
objectives; attitude weights affect the inner objective only. Solver caches survive
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

Checkpoints save independent progress, EMA/history, counters and absolute schedule
origin. Older checkpoints keep their absolute iteration, initialize performance
progress/history to zero, and derive ramp progress from that iteration/config.
Resume with the same configuration for exact schedule continuation.

`curriculum/qp/*` logs alpha, current/next progress, effective weights, EMA,
threshold (NaN until evidence), and advancement. `qp/rollout/execution/*` reports
candidate/executed absolute correction sums in Nm with coordinate counts, and
partially corrected row counts. Candidate acceptance is distinct from execution.

Focused validation (no training):

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests conda run --no-capture-output -n lr_lab_cupiqp \
  python -m pytest -q tests/test_hard_pact_qp_curriculum.py \
  tests/test_hard_pact_qp_modes.py tests/test_hard_pact_rate_recovery.py
```
