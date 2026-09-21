# HardPACT transfer, contact indexing and inverse-PINN audit

Inspected 2026-09-09 without stopping or modifying the running training process.
No new simulator/training run or GPU benchmark was launched during this audit.

## Runs and measured symptoms

- Active process: PID 1158464, `train_hard_pact.py --task go2_hard_pact_full_isaaclab --headless --dynamics_backend bard --qp_solver cupiqp --num_envs 4096 --gpu cuda:0`.
- Active log: `logs/hardpact_iclr/go2_pact_rough/Sep09_16-05-50_hard_pact_full_isaaclab`.
- Associated pretraining: `logs/hardpact_iclr/go2_pact_pos_rough/Sep08_20-29-22_hard_pact_pos_isaaclab`, completed iteration 5000.
- The configured `Sep08_hard_pact_start_model_5000.pt` is byte-identical to that run's exported starting checkpoint (SHA256 `ef599cd59be6fafa7c0b8c5128ea7e44a49b13b3f134c0f0f0b33aa8939e5a17`).
- Config comparisons use each run's `hard_pact_resolved_config.json`, not the copied Python config alone.

TensorBoard snapshots, averaged over 50 logged iterations ending at HardPACT
iteration 470 and Pos iteration 4999:

| Metric | HardPACT | HardPACTPos |
|---|---:|---:|
| Support-polygon reward | 0.00005795 | 0.16238 |
| Linear tracking reward | 0.64849 | 0.71643 |
| Angular tracking reward | 0.23916 | 0.32228 |
| Mean terrain level | 4.14783 | 5.90147 |

These are episode reward statistics, not controlled same-terrain evaluations.
HardPACT linear tracking was 0.73449 at iteration 300 (terrain 2.69), then
0.64544 at iteration 400 (terrain 3.78). It has not remained below 0.7 throughout.
QP rollout/replay was disabled by its iteration-2000 warmup at the inspected
iterations, so the newly added anchor mode cannot explain these early symptoms.

## Confirmed bugs fixed

1. HardPACT inherits `Go2PACT._reward_support_polygon`, which used articulation
   `feet_indices` on contact-sensor tensors. Pos already used
   `feet_contact_indices`. Identical synthetic stance geometry produced raw
   rewards 0 versus 1 before the fix, solely due to differing index spaces.
   Fixed every inherited foot contact-force/state access, including critic,
   air-time, slip, edge, VHIP, stumble and support helpers. Articulation state
   indexing is unchanged. Genesis exposes identical body/contact indices.
   HardPACT's explicit labels were already replaced by canonical processor
   contacts; those labels did not need another change.
2. Isaac Lab's `_get_pinn_wb_dynamics()` returns an unpopulated zero legacy
   generalized-contact buffer. PPO passed it to inverse-loss soft contact
   weighting, making that loss and both force-head gradients exactly zero.
   Logs confirmed inverse loss 0 despite physical residual MAE ~13.1 at
   iteration 480. PPO now uses its existing stored conditioned GRF targets,
   denormalizes to Newtons, reverses the pre-step yaw rotation, and computes
   detached `g_contact = J_feet.T @ F_measured` using cached actual mechanics.
   No new state storage or dynamics evaluation is needed. Normalization,
   masks, residual definition and loss weights remain unchanged.

The running Python process does not reload these fixes; restart is required.

## PINN gradient interference: risk, not an established live-run diagnosis

The rollout PINN is active (~0.55 unweighted). The *effective* PINN coefficient
is **0.01**, not the copied config's -1.0: `task_registry.py` applies the
argument parser's default. It therefore selects `pc_backward_pinn`, not the
norm-balanced negative-weight path. This describes the inspected running
process. A subsequent fix now preserves HardPACT's configured weight when
`--pinn_loss_weight` is omitted; explicit overrides still win and legacy
tasks retain their 0.01 CLI default. Restart is required to change the live
run's behavior. The parser/registry fix was validated with
`env SIMULATOR=isaaclab conda run -n lr_lab_cupiqp python -m pytest -q tests/test_hard_pact_pinn_weight_cli.py tests/test_hard_pact_ppo_latent_replay.py --tb=short`:
**24 passed, 1 hppfcl warning, 9.54 s**.

PCGrad orthogonalizes against the complete PPO gradient, including critic
coordinates. Global orthogonality does not guarantee actor/trunk orthogonality
or protection after Adam preconditioning. A CPU demonstration with the actual
projector gives PPO `(actor=1, critic=10)`, physics `(-1,0)`, merged
`(0.00990099,10.09900951)`: actor learning can be strongly reduced even when
the global projection is orthogonal. This demonstrates a possible mechanism,
not the actual run's gradient vectors.

The shared encoder also receives a separate auxiliary update outside this
projection. Adaptive policy KL changes actor-group LR only. The checkpoint
confirms encoder PPO LR 3e-4 and auxiliary LR 2e-4, while the logged actor LR
later reaches 1e-5. Thus a small actor LR does not imply a slowly changing
policy-conditioned encoder.

Both `pcgrad_diagnostics_enabled` and `ppo_latent_diagnostics_enabled` are
False. Logged conflict/cosine values are unavailable NaNs, not evidence of
zero conflict. To establish causality, collect scheduled per-objective/module
gradients and pre/post-auxiliary policy KL, then compare matched seeded
PINN-on/off runs. No diagnostic configuration or optimizer behavior was changed.

## Other material transfer differences (not changed)

Shared contracts match: 57 observations, 10-step history, 16 latent units,
11 explicit fields, network widths, control/physics dt, commands, terrain
suite, decoder scales and checkpoint parameter shapes (except intended std).

However, effective training distributions/objectives do not match:

- Persistent force bounds start at 10 N versus 7 N in Pos, with independent
  force/torque probabilities 0.3 versus 0.1. Persistent disturbance progress
  was still 0; initial disturbances remain active (~10.6% at iteration 480).
- Terrain upward delay is 100 versus 250 iterations.
- Collision weight is -10 versus -1; stumble -1 versus -0.2; contact BCE
  coefficient 1 versus 0.1; rear-foot nominal x -0.25 versus -0.20 m.
- `only_positive_rewards` is False versus True; reward curriculum bounds and
  joint-dynamics randomization ranges also differ.
- Export intentionally resets 24-D action std to 1; trained Pos position std
  was approximately 0.38–0.52. With torque scale 10, newly sampled feedforward
  action noise initially corresponds to 10 Nm per joint before actuator handling.
- Pos observes PD-torque history while HardPACT observes feedforward-action
  history: compatible dimensions do not imply identical input distributions.
- Isaac Lab initializes branch weights to one and does not implement Genesis's
  `randomize_pact_weights` path. With tradeoff curriculum disabled, the configured
  initial `[1,1.6]` weights are not actually applied. No change was made here.

Fix contact indexing and inverse gating before attributing the transfer gap
solely to PINN gradients. A same-difficulty evaluation is necessary to separate
terrain progression from actual policy degradation.

## Validation and changed files

```bash
env SIMULATOR=isaaclab conda run -n lr_lab_cupiqp python -m pytest -q \
  tests/test_hard_pact_contact_indexing_and_inverse_gate.py \
  tests/test_go2_hard_pact_bard.py \
  tests/test_go2_hard_pact_dynamics_backends.py \
  tests/test_hard_pact_speed_optimizations.py \
  tests/test_hard_pact_auxiliary.py \
  tests/test_hard_pact_ppo_latent_replay.py \
  tests/test_pc_grad.py --tb=short
```

Result: **70 passed, 1 skipped, 11 subtests passed, 1 warning, 5.45 s**.
The skip is a CUDA-only test; this run was CPU-only. The warning is the
installed hppfcl-to-coal import deprecation. `git diff --check` passed.

The regression includes an actual CPU PPO update with BARD mechanics and
positive PINN weight 0.01, a zero legacy buffer, nonzero measured interval
GRFs, and finite nonzero inverse gradients to the encoder and both heads.
It also checks frame/scale/order conversion, detachment, all contact accesses,
canonical support rewards and Genesis-equivalent index order. Existing tests
cover masks, dynamics parity, rollout gradients, replay and PCGrad.

Changed: `legged_gym/envs/go2/go2_pact/go2_pact.py`,
`rsl_rl/algorithms/hard_pact_bard.py`, `rsl_rl/algorithms/ppo_hard_pact.py`,
`tests/test_hard_pact_contact_indexing_and_inverse_gate.py`, and this report.
Legacy Go1 files, solver settings, loss weights and active-run state are unchanged.
