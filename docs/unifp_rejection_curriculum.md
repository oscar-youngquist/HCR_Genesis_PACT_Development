# Estimated-force rejection

`b1z1_unifp_reject` uses the current history encoder/12-D decoder prediction
before stepping physics. The decoder layout is base velocity, EE spherical
position, EE force, base force (three values each). Force predictions are
unscaled into newtons in the existing yaw-local command frame.

The control command is `-beta * LPF(predicted external force)`. The filter uses
the existing EE/base time constants and reset behavior. Ground-truth force
labels never replace predictions, even during warmup; nonfinite predictions
are replaced with zero. Actual applied disturbances still enter the existing
net-force target adaptation, independently of prediction-based cancellation.

## Three stages

1. Hold external force ranges at 25% and beta at zero for at least 500 PPO
   iterations. Existing disturbance probabilities and temporal envelopes remain.
2. Require active-force accuracy and policy stability for the existing
   `force_curriculum_gate_patience` consecutive updates. Increase beta by 1/500
   per qualifying update; pause growth and reset patience when gates fail.
3. Once beta reaches one, ramp external force ranges from 25% to 100% over
   1500 iterations. There is no forced-start timeout or oracle fallback.

All new defaults are `commands.reject_*` fields in `b1z1_unifp_reject_config.py`.
For each enabled stream, active samples have true force norm above 1 N.
At least 32 active samples per update are required. The gate computes
`sqrt(sum(||prediction-target||^2) / sum(||target||^2))` separately for EE/base,
then applies the existing metric EMA coefficient and a 0.25 threshold.
Policy stability uses the existing EE tracking, roll termination, and episode
length thresholds. Missing/nonfinite measurements cannot advance compensation.

## Training and deployment

The runner publishes one prediction from the same encoder evaluation used for
the action. Only training records privileged labels for gate statistics;
inference uses exactly the same predicted-force/filter/compensation path.
Beta stays fixed during each rollout and advances after its PPO update.
`Rejection/*` metrics report beta, stage, force scale, patience, and force-error
EMAs. Curriculum state uses the existing force-curriculum checkpoint slot;
older checkpoints without rejection state restart warmup with a warning.
Inference retains the loaded checkpoint beta rather than silently enabling
full compensation. No ground-truth cancellation/oracle mode is enabled here;
any future oracle ablation must be explicitly labeled `UniFP-Reject-Oracle`.

Retained UniFP and UniFP-Original models, supervised losses, and schedules are
unchanged. Small disturbances mean nonzero sampling ranges from the start,
not constant nonzero force: existing idle intervals/probabilities still apply.
