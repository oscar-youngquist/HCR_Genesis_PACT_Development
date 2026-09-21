# Frozen Isaac Lab HardPACT controller evaluation

`eval_hard_pact_frozen.py` compares an unchanged checkpoint under pre-QP control,
analytic projection, every-substep QP, and single-anchor QP. It creates no
optimizer and runs no backward pass. Default evaluation
is 64 environments for 20 seconds **after** a common 100-control-step prefix.

From the repository root, in the existing Isaac Lab cuPIQP environment:

```bash
conda activate lr_lab_cupiqp
SIMULATOR=isaaclab PYTHONPATH=. python scripts/eval_hard_pact_frozen.py \
  --checkpoint logs/hardpact_iclr/go2_pact_rough/Sep09_19-46-02_hard_pact_full_isaaclab/model_1000.pt \
  --resolved-config logs/hardpact_iclr/go2_pact_rough/Sep09_19-46-02_hard_pact_full_isaaclab/hard_pact_resolved_config.json \
  --solver cupiqp --seeds 1 --num-envs 64 --duration 20 --device cuda:0 \
  --output-dir logs/frozen_eval/64env_20s_timeout_fixed_sep10
```

For the small activation/reset smoke, use `--num-envs 4 --duration 0.08 --smoke`
and a separate output directory. `--sample-actions` enables seeded latent/action
sampling; deterministic inference is the default. Use a fresh output directory
for each evaluation.

Each trial saves configuration/version/checkpoint hashes, scalar statistics,
bounded four-environment NPZ traces, and up to the shared ten-packet budget of
state/history/mechanics/QP replay packets. The top-level CSV/JSON checks identical
initial states, scenario tapes, and prefix torques. JSON null means unavailable.
Error `mean_abs` is MAE; `rms` is RMSE. GRF errors are per physics substep in
world-frame Newtons; wrench errors are yaw-local N/Nm. Rate excess is Nm per
physics step, not Nm/s. Physical failures remain in the results.

Curricula are frozen at the restored domain-randomization stage. Velocity/yaw
command, push, sustained-wrench and observation-noise tapes are shared; heading
feedback is disabled explicitly. Subsequent asynchronous resets use the normal
task lifecycle, so post-failure trajectories are not paired simulator states.
Survival concerns the first episode, including the prefix; it is not a
Kaplan–Meier estimate. Timeouts/terrain exits are censored rather than falls.
Only the evaluation instance's episode limit is extended to cover its horizon.

The shared Isaac Lab PACT adapter removes the interactive STOP subscription
only in headless mode, preventing the render-until-Play shutdown hang in both
training and evaluation. Training episode limits and interactive play remain
unchanged. Running training processes need a restart to load this code.

Focused forward-only tests:

```bash
SIMULATOR=isaaclab conda run -n lr_lab_cupiqp python -m pytest -q \
  tests/test_eval_hard_pact_frozen.py tests/test_isaaclab_pact_headless_stop.py \
  tests/test_isaaclab_pact_torque_limits.py --tb=short
```

Historical result before removal of the added QP objective: **26 passed**, one
existing `hppfcl`→`coal` deprecation warning (6.00 s).
Five four-environment GPU smoke trials passed with identical prefix hashes,
unchanged weights, explicit reset, and clean shutdown. Primary batched QP call
counts were **0 / 0 / 16 / 4 / 4**; analytic/QP execution had zero torque/rate
excess. The unprojected baseline exposed up to **18.700001 Nm** rate excess.
Artifacts: `logs/frozen_eval/smoke_timeout_fixed_sep10`.

Environment: RTX 4090, Isaac Sim 5.1.0.0, Isaac Lab 0.54.2, PyTorch 2.7.0+cu128,
cuPIQP 0.1.0, BARD 0.4.3, Warp 1.17.0; no environment changes were made.

## 64-environment, 20-second frozen evaluation

The results below are historical, from before removal of HardPACT's added
proximal objective. The current script runs four controllers; the former
rho=0.1 comparison is no longer supported. Saved config keys for that removed
objective are discarded with a warning. Policy weights still load unchanged.
These historical results have not been rerun following removal.

The command above was executed with this process-launch prefix (instead of
interactive activation), with `--num-envs 4 --duration 0.08 --smoke` and output
`logs/frozen_eval/smoke_timeout_fixed_sep10` for the separate small smoke:

```bash
env SIMULATOR=isaaclab PYTHONPATH=. PYTHONUNBUFFERED=1 \
  NUMBA_CACHE_DIR=/tmp/numba_hard_pact_eval MPLCONFIGDIR=/tmp/matplotlib_hard_pact_eval \
  conda run --no-capture-output -n lr_lab_cupiqp python scripts/eval_hard_pact_frozen.py \
  --checkpoint logs/hardpact_iclr/go2_pact_rough/Sep09_19-46-02_hard_pact_full_isaaclab/model_1000.pt \
  --resolved-config logs/hardpact_iclr/go2_pact_rough/Sep09_19-46-02_hard_pact_full_isaaclab/hard_pact_resolved_config.json \
  --solver cupiqp --seeds 1 --num-envs 64 --duration 20 --device cuda:0 \
  --output-dir logs/frozen_eval/64env_20s_timeout_fixed_sep10
```

All five processes exited 0, stopped normally, preserved checkpoint weights and
curriculum progress, and produced finite execution with identical initial-state,
scenario and prefix-torque hashes. Control/physics dt were 0.02/0.005 seconds.
There were no timeout failures or solver exceptions. This is **one seed and one
checkpoint**, not a training benchmark or evidence of general policy quality.

| Controller | Survivors / 64 | Batched QP calls | Velocity RMSE (m/s) | Yaw RMSE (rad/s) |
|---|---:|---:|---:|---:|
| Pre-QP | 51 | 0 | 0.190328 | 0.338331 |
| Analytic | 55 | 0 | 0.198969 | 0.338255 |
| Every-substep QP, rho=0 | 0 | 4,000 | 0.414044 | 1.068941 |
| Single-anchor QP, rho=0 | 1 | 1,000 | 0.409706 | 1.007547 |
| Single-anchor QP, rho=0.1 | 0 | 1,000 | 0.409535 | 0.943825 |

**The QP controllers failed locomotion stability for this checkpoint/scenario,
despite successful execution/certification checks.** Tracking includes subsequent
episodes; survival counts first-episode physical failures, including the prefix.

No controller exceeded torque magnitude limits. Maximum rate-step excess was
33.889854 Nm for pre-QP and 0.00000190735 Nm for every projected controller
(floating-point roundoff; violation flags use 0.00001 Nm, with raw excess retained).
Final solver-row stages `[full, relaxed, elastic, analytic]` were
`[255984, 2, 14, 0]`, `[63997, 1, 2, 0]`, and `[63927, 52, 21, 0]` respectively
for the three QP controllers. No certification thresholds or controller laws
were changed to improve these results.

Complete CSV/JSON, five bounded NPZ traces, six replay packets, and metadata are
in `logs/frozen_eval/64env_20s_timeout_fixed_sep10`. Earlier directories without
`timeout_fixed` are superseded pilots, including an intentionally interrupted
comparison. Remaining limitation: the QP-related stability degradation requires
separate investigation; the shutdown fix does not correct it.

Changed files: `scripts/eval_hard_pact_frozen.py`, this README,
`legged_gym/simulator/isaaclab_simulator_pact.py`,
`tests/test_eval_hard_pact_frozen.py`, and
`tests/test_isaaclab_pact_headless_stop.py`. Pre-existing PPO/auxiliary edits were
preserved. No training or backward tests were run.

## Added QP objective removal

The added temporal objective, its configuration and reference/replay buffers are
removed. Tracking/slack costs, SPD regularization, constraints, torque history,
solver recovery, and cuPIQP internal regularization are unchanged. Deployment
schema advanced to 8 for that removal. Old config keys warn and are discarded; old captured matrices
with a nonzero added objective are rejected rather than silently replayed.

The current torque-convention correction advances the contract to schema 9:
raw delayed requests feed magnitude/limit penalties; clipped delayed actions
produce the bounded non-QP nominal command, including actuator effects once;
final executed commands feed physics labels and the next torque-rate box.
PPO replay and deployment use `rsl_rl.modules.hard_pact_control` for the same
conversion. The historical simulator results above predate this correction
and have **not** been rerun; this change uses focused tests only.
The subsequent optional swing-GRF gate advances deployment to schema 10;
see [its focused validation](../tests/HARDPACT_SWING_GRF.md).

### Torque-convention validation (CPU, `lr_lab_cupiqp`)

```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n lr_lab_cupiqp python -m pytest -q tests/test_isaaclab_pact_torque_limits.py tests/test_hard_pact_action_replay.py tests/test_hard_pact_contact_indexing_and_inverse_gate.py tests/test_go2_hard_pact_qp.py tests/test_hard_pact_two_anchor_smoke.py tests/test_hard_pact_single_anchor.py tests/test_eval_hard_pact_frozen.py --tb=short
# 88 passed, 3 warnings, 4 subtests passed in 5.59s
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES='' conda run --no-capture-output -n lr_lab_cupiqp python -m pytest -q tests/test_go2_hard_pact_physics_heads.py -k 'migration or json_contents or observation_scale_is' --tb=short
# 4 passed, 25 deselected, 1 warning in 2.16s
```

Coverage: reward sensitivity beyond action/actuator clipping, Genesis/Isaac Lab
controller-formula parity on synthetic states, randomized actuator conversion
exactly once, delayed raw/clipped queues and partial resets, sampled-QP replay
torque equality (`rtol=atol=0`), actual PINN consumers of final interval torque
(existing float32 straight-through arithmetic: `atol=2e-7`, `rtol=0`), QP-off/on
and fallback commands, unchanged anchor schedules, deployment and strict Pos
migration. Simulator APIs are mocked; no real simulator/training/benchmark
was run. Warnings are the installed hppfcl deprecation and two deliberately
injected qpth failures. No test failures or implementation blockers remained.

Files changed for this correction:

- `rsl_rl/modules/hard_pact_control.py` (new shared torque conversion)
- `rsl_rl/algorithms/ppo_hard_pact.py`
- `legged_gym/envs/go2/go2_hard_pact/{go2_hard_pact.py,deployment.py}`
- `legged_gym/simulator/{genesis_simulator_pact.py,genesis_simulator_pact_pos.py,isaaclab_simulator_pact.py}` (HardPACT-only hooks)
- `scripts/{eval_hard_pact_frozen.py,README_eval_hard_pact_frozen.md}`
- Existing tests: `test_isaaclab_pact_torque_limits.py`, `test_hard_pact_action_replay.py`, `test_hard_pact_contact_indexing_and_inverse_gate.py`, `test_go2_hard_pact_qp.py`, `test_go2_hard_pact_physics_heads.py`.

HardPACTPos inherits the shared corrections; no legacy task logic, weights,
PPO likelihood, QP settings/constraints, or supervised GRF/wrench outputs changed.
Policy tensors and HardPACTPos migration are unchanged.

Removal touched `hard_pact_qp.py`, `hard_pact_qp_capture.py`, `ppo_hard_pact.py`,
`go2_hard_pact.py`, `go2_hard_pact_config.py`, `deployment.py`, this evaluator,
six existing test files (`test_go2_hard_pact_qp`, `test_hard_pact_qp_reuse`,
`test_hard_pact_cupiqp_pool`, `test_hard_pact_two_anchor_smoke`,
`test_go2_hard_pact_physics_heads`, `test_eval_hard_pact_frozen`), and this README
plus `tests/HARDPACT_QP_CAPACITY_REUSE.md`. No new tests were added.

Focused existing CPU tests in `lr_lab_cupiqp`:

- QP, action replay, anchor modes, cache reuse and evaluator: **83 passed,
  9 CUDA-only skipped, 4 subtests passed**, 3 warnings, 7.25 s.
- Deployment and migration: **3 passed**, 26 deselected, 1 warning, 2.27 s.
- Failure capture/replay: **3 passed**, 3 deselected, 1 warning, 3.26 s.

The deployment test's stale fixed 250 N expectation was corrected to use the
unchanged current config's GRF scale. Remaining warnings are deliberate solver
failure injections and the installed hppfcl deprecation. No training, simulator
runs, GPU tests, or benchmarks were run for this removal.
