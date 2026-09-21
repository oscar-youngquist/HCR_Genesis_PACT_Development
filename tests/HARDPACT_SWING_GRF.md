# HardPACT swing-GRF gating and auxiliary-gradient tests

Configure `deployment_physics` in both `go2_hard_pact_config.py` and
`go2_hard_pact_pos_config.py`:

```python
grf_swing_gating_enabled = False
grf_swing_contact_threshold = 0.5
grf_swing_loss_weight = 0.0
```

Gating and loss weight are independent; these defaults preserve existing
behavior. Enable gating to zero physical XYZ **QP references** for feet with
`contact_probability.detach() < threshold`. There is no additional sigmoid,
observation scaling, or change to QP decision variables/constraints.
Rollout, PPO replay and deployment use `gate_grf_for_qp`. Existing frame,
FR/FL/RR/RL ordering, substep selection and hold semantics are retained.

Positive loss weight adds the mean swing-foot squared physical force divided
by the existing positive `grf_scale_n`, targeting physical zero. Its separate
GRF forward detaches latent, explicit/contact and torque inputs. Full HardPACT
adds it only in the decoder-training phase; Pos adds the isolated branch to
its existing auxiliary update. No new optimizer, sampling, or checkpoint keys.
Raw supervised and PINN predictions are unchanged. Deployment schema 10
records the gate, weight, scales and gradient convention alongside weights.

Device-summed auxiliary metrics use valid rows and globally accumulated foot
counts: `physics/grf/swing_fraction`, `swing_raw_norm_n` (swing-foot mean),
`swing_removed_norm_n` (all-valid-foot mean), and `swing_consistency_loss`
(weighted). Empty auxiliary swing sets produce zero. Actual QP-reference
metrics use `qp/{rollout,ppo}/grf_swing/{fraction,raw_norm_n,removed_norm_n}`;
rollout counts only anchors, not held substeps. With no swing feet, the QP
swing-only norm is unavailable/NaN under the existing diagnostic convention.
No new metrics or additional head forward runs when both options are disabled.

## Validation

CPU unit tests in `lr_lab_cupiqp`; simulator APIs are mocked. No training,
simulation run, or benchmark. Exact zero/stance/forward parity uses zero
tolerance; existing numerical tolerances were not relaxed.

```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n lr_lab_cupiqp python -m pytest -q tests/test_hard_pact_swing_grf.py tests/test_hard_pact_two_anchor_smoke.py tests/test_hard_pact_action_replay.py tests/test_hard_pact_two_stage_auxiliary.py tests/test_hard_pact_auxiliary.py tests/test_hard_pact_pos_auxiliary.py tests/test_go2_hard_pact_physics_heads.py tests/test_go2_hard_pact_qp.py tests/test_hard_pact_contact_indexing_and_inverse_gate.py --tb=short
```

Result: **139 passed, 15 subtests passed**, 3 warnings, 6.92 s. The warnings
are the existing hppfcl deprecation and two intentionally injected QP failures.
Stale history, diagnostic-default and 250-N test assumptions were updated to
validate current configuration-driven behavior, without changing training values.
No remaining failures/blockers.

Changed implementation files:

- `rsl_rl/modules/hard_pact_physics.py`
- `rsl_rl/modules/actor_critic_hard_pact.py`, `actor_critic_hard_pact_pos.py`
- `rsl_rl/algorithms/ppo_hard_pact.py`, `ppo_pact_pos.py`
- `rsl_rl/runners/pact_runner.py`, `pact_pos_runner.py`
- `legged_gym/envs/go2/go2_hard_pact/{go2_hard_pact.py,go2_hard_pact_config.py,deployment.py}`
- `legged_gym/envs/go2/go2_hard_pact_pos/go2_hard_pact_pos_config.py`

Tests: new `test_hard_pact_swing_grf.py`; updated `test_hard_pact_two_anchor_smoke.py`,
`test_hard_pact_action_replay.py`, `test_go2_hard_pact_physics_heads.py`.
Documentation: this report and the schema cross-reference in
`scripts/README_eval_hard_pact_frozen.md`.
