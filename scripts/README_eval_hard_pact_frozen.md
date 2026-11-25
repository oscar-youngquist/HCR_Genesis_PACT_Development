# Frozen Isaac Lab HardPACT controller evaluation

## Targeted attitude / diagnostic recovery probe (evaluation only)

```bash
conda activate lr_lab_cupiqp
export SIMULATOR=isaaclab PYTHONPATH="$PWD"
python -m pytest -q tests/test_hard_pact_recovery_probe.py tests/test_eval_hard_pact_tradeoff.py
python scripts/eval_hard_pact_tradeoff.py --attitude-probe --seeds 1 --steps 24 \
  --variants attitude050_probe --output diagnostics/attitude_probe11000/smoke
python scripts/eval_hard_pact_tradeoff.py --attitude-probe --seeds 1 --steps 2000 \
  --output diagnostics/attitude_probe11000/full
python scripts/report_hard_pact_tradeoff.py diagnostics/attitude_probe11000/full
```

Physical GPU 1, ten environments, sequential workers. This mode resolves the
checkpoint **from** `diagnostics/tradeoff11000/full/protocol.json`, verifies its
hash, and does not select a newer checkpoint. Four conditions cross attitude
weights 0.5/0.1 with normal/probe termination; stance=0.25 and torque=20.
Assembly weights, immutable network hashes, saved schedule probabilities and
per-row execution strengths are recorded. No training/optimizer runs occur.

The previous failures select stairs-down levels 4/5 (three environments),
stairs-up levels 2/7 (three), and discrete-obstacle levels 5/7/8/9 (four).
Saved terrain rows are sampled within these cells, using unchanged training
geometry. Thus this is **targeted exposure**, not the overall training terrain
distribution. Identical initial scenario hashes are checked; trajectories and
later randomization necessarily diverge after asynchronous resets.

`--probe-extension 0.2`, `--probe-confirmation 0.5`, and `--capture-cap 24`
configure diagnostic continuation. An original, debounced geometry failure
starts one latched event only if contact/safety/timeout does not already require
reset. Contact or nonfinite state terminates a live probe immediately. Failure
to return inside the original geometry thresholds within 200 ms, or relapse
during confirmation, is a delayed failure. Remaining continuously inside for
500 ms is recovery; either outcome resets that environment. Timeouts/terrain
relocation and unfinished end-of-evaluation probes are unresolved. Durations
are rounded up to control steps. Production termination code is untouched.

Use **`comparison.csv`, `terrain_comparison.csv`, and `summary.json`'s `probe`
and `ordinary_diagnostics` fields**, not legacy top-level summaries (which
include continuation). Original failure events are counted once; ordinary
exposure excludes probe tails, and ordinary tracking additionally excludes
original reset intervals. Recovery fraction uses all started probes, including
unresolved events in its denominator. A delayed reset is not improved survival.
Overlapping reasons describe observed threshold violations, not root causes.

`probe_histories.npz` keeps at most the configured number of events, each with
at least 0.5 seconds before the original decision (one extra control interval,
truncated at previous reset) plus its
continuation. Same-clock non-failure controls are retained when available.
Aggregate counts include uncaptured events. Four physics samples correspond to
each control observation; post-step measurements precede any reset. Solver
accelerations are full-candidate model predictions, not measured acceleration.
World angular x/y norms equal yaw-local horizontal norms; the logged desired
world vector's vertical component is **not** an attitude-objective target.
Stance acceleration is world-frame, swing entries unavailable. Terrain height
and slip reference samples are control-rate cached. Contact transition timings
use simulator processed contacts, not ideal instantaneous contact truth.

Event plots and `event_timing.csv` align to the original failure decision, which
may lag initial deterioration because of production debouncing. A crossing at
the beginning of capture is left-censored. Temporal association cannot identify
whether a QP correction caused deterioration or responded to it. One seed is
preliminary evidence, with correlated environments/time samples.

Executed collection: `diagnostics/attitude_probe11000/full` (seed 1, ten envs,
2,000 control steps each, physical GPU 1). Nine focused tests passed. All four
initial contracts matched; actor weights stayed unchanged. Actual assembly
weights were `(stance, torque, attitude)=(0.25,20,0.5 or 0.1)`. Normal/probe
executed torques matched exactly through 507/1,529 steps before the first
eligible event. The 24-step GPU smoke also completed.

| Attitude / termination | Original failures / ordinary env-s | RMSE vx / vy / yaw | Correction RMS / max [Nm] |
|---|---|---|---|
| 0.5 normal | 12 / 400 | .241 / .260 / .199 | 2.972 / 44.003 |
| 0.1 normal | 6 / 400 | .188 / .228 / .165 | 2.321 / 33.243 |
| 0.5 probe | 6 / 399.60 | .192 / .214 / .169 | 2.450 / 33.609 |
| 0.1 probe | 6 / 399.94 | .187 / .223 / .161 | 2.296 / 33.243 |

There were **zero recovered events**: two 0.5-probe failures after 200 ms and
one 0.1-probe failure stopped by contact after 60 ms. No unresolved events;
confirmation tracking is unavailable because none reached confirmation.
All 74,000 scheduled QP rows per trial accepted primary solutions; no recovery
or fallback solves occurred. Measured torque-magnitude exceedance was zero.
Thirty original events were captured across the four trials, without cap drops.
Collection took 409.5–416.5 seconds/trial. PyTorch allocator peak was 114.8 MiB;
sampled total GPU occupancy was approximately 4.7 GiB including simulator and
non-PyTorch allocations. These are simulator/capture timings, not deployment
latency. Another training process occupied GPU 0/shared host resources.

Lower attitude weight is a promising follow-up, not a validated training
recommendation from one seed. Stair-up failures did not improve in the normal
comparison. The reduced total in the 0.5 probe is not evidence of recovery:
delayed resets changed subsequent trajectories/randomization. Event plots show
large corrections around already developing roll excursions; temporal
association does not identify causality. No production defaults were changed.

## Saved-schedule tracking/survival objective sweep

`eval_hard_pact_tradeoff.py` is an opt-in evaluation extension. It locates the
exact `events.out.tfevents.1795231251.CICSUNIX-US-003.2659464.0` run and snapshots
its latest checkpoint identity on first invocation. It creates **no optimizer**.
All workers use physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`) and
run sequentially. Production files/settings are not modified.

```bash
conda activate lr_lab_cupiqp
export SIMULATOR=isaaclab PYTHONPATH="$PWD"
# Tiny forward-only simulator check:
python scripts/eval_hard_pact_tradeoff.py \
  --output diagnostics/tradeoff11000/smoke_new --steps 8 --seeds 1 --variants saved
# All four objectives, three seeds, 40 seconds/environment:
python scripts/eval_hard_pact_tradeoff.py \
  --output diagnostics/tradeoff11000/full --steps 2000 --seeds 1 2 3 \
  --checkpoint logs/hardpact_iclr/go2_hard_pact_rss_full/Nov20_22-20-49_hard_pact_rss_full_isaaclab/model_11000.pt
python scripts/report_hard_pact_tradeoff.py diagnostics/tradeoff11000/full
```

This extension retains the saved terrain generator and samples ten level/type
pairs from the saved 4,096-environment joint distribution. Promotions are frozen;
native heading, command/reset, actuator-randomization and disturbance protocols
remain active at saved curriculum difficulty. Sampling/strength/objective
curricula never advance. The saved snapshot has mixture approximately
`(0.1125, 0.1125, 0.775)` for 2/3/4 solves, baseline alpha 1 and extra alpha
0.915625. Deterministic deployment inference and held control-rate predictions
are retained. Objective overrides are applied after resolving saved schedules.

Each trial saves a manifest with saved/resolved settings and initial-state hash,
strict JSON results, and a bounded (2,000-step maximum) detached numeric trace.
No graphs or solver factors are stored. `bounded_histories.npz` retains at most
12 one-second pre-failure windows plus six controls; windows do not cross resets.
`summary.json`, `comparison.csv`, `paired.json`, and PNGs are generated offline.
Ordinary tracking excludes reset intervals. Paired one-second windows exclude
any reset in either condition and report coverage; states after earlier resets
are **not** identical despite a shared seed. Three seeds are descriptive evidence,
not independent environment/time samples for significance testing.

Accepted-candidate residual/cost comparisons use production physical scales.
They exclude unsolved/rejected rows. Actual corrections use final executed torque;
full-QP candidate predictions are kept separate. Missing floating samples are
NaN internally and unavailable summaries are JSON null, not successful zeros.
Measured next-substep differences use fresh body-frame velocity/orientation.
Height-label/slip diagnostics use existing control-rate cached measurements;
wrench error describes the QP-clamped reference rather than the unclipped decoder.
Force labels are processed simulator contacts, not instantaneous hardware truth.
These limitations are recorded in the offline report.

Timing excludes the first 100 steps for steady-state collection but includes
evaluation diagnostics. PyTorch allocated/reserved memory excludes PhysX/CuPy;
`gpu1_telemetry.csv`, when collected, measures total device memory/utilization.
Neither is hardware deployment latency. Completed trials can be skipped on
relaunch; interrupted trials restart from their seed, not exact PhysX state.

Forward-only validation:
```bash
SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m pytest -q \
  tests/test_eval_hard_pact_tradeoff.py --tb=short
```

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
# Four-solve intervention/weight evaluation (November 2026)

Opt-in only; no runner, optimizer, weight update, or training-default change.
`eval_hard_pact_tradeoff.py --intervention-sweep` preserves
`curriculum_substeps`, freezes its frequency at `(0,0,1)`, and retains the
randomized baseline masks (substep zero plus one later step). The conditions are:

- `A_reference`: all four strengths 1.
- `B_extra050`: baseline strengths 1, additional strengths 0.5.
- `C_attitude005`: A execution, inner attitude weight 0.05 only.
- `D_planar10`: A execution, inner planar tracking weight 10 only.
- `B_extra075`: optional follow-up, additional strengths 0.75.

Saved stance remains 0.1. Saved clipping is applied **after** the strengths.
The evaluator asserts the configuration-difference allowlist, four real solves,
two baseline interventions, per-row alpha, reconstructed actual torque, and
unchanged model tensors. The older attitude-probe preset is not used.

From the `rescaled_qp_exps` worktree, with `lr_lab_cupiqp` activated:

```bash
CUDA_VISIBLE_DEVICES=1 SIMULATOR=isaaclab PYTHONPATH="$PWD" \
python scripts/eval_hard_pact_tradeoff.py --intervention-sweep \
  --event events.out.tfevents.1795413624.CICSUNIX-US-003.133079.0 \
  --checkpoint logs/hardpact_iclr/go2_hard_pact_rss_full/Nov23_01-00-22_hard_pact_rss_full_isaaclab/model_10500.pt \
  --output diagnostics/intervention10500/full --steps 1000 --seeds 1
PYTHONPATH="$PWD" python scripts/report_hard_pact_tradeoff.py diagnostics/intervention10500/full
```

The bounded smoke uses `--steps 12 --variants A_reference B_extra050` and a
separate output directory. Follow-ups explicitly select `--variants` and
`--seeds`; they are not launched automatically. Completed trials are skipped.
Changed evaluator code/protocol requires a new output directory.

Terrain geometry is the saved training geometry. Targeting uses the latest
500 checkpoint-era iterations of failure counts, selects the three highest
represented failure levels, and allocates 3 stair-down / 3 stair-up / 4
discrete-obstacle environments. **Training logs contain family and level
marginals, not joint failure counts.** Do not describe selected combinations
as individually proven high-failure cells. Initial-state hashes are checked
across conditions. Native commands/randomization diverge after asynchronous
resets despite shared seeds; this is not exact counterfactual state matching.

Artifacts include checkpoint/config/code hashes, resolved manifests, archived
evaluator sources, per-seed JSON, CSV comparisons and weighted objective costs,
plots, compact traces, and at most 24 episode-safe half-second failure histories
plus six healthy controls. A 100-physics-step device ring captures the histories.
Detailed data and new metrics are evaluation-only. Failure counts include all
samples, even after the history cap; capture drops are reported.

Objective and executed-motion diagnostics reuse the captured affine maps. For
execution reconstruction, **optimized candidate forces remain fixed** while
torque changes; these are predictions, not measured contact-force behavior or
post-clipping feasibility certificates. Body-velocity derivatives use live body
frames at both timestamps. Reset/teleport intervals are excluded from aligned
model errors but retained in failure counts. Numerical solver regularization is
not a physical objective component. Recovery slack penalty is reported
separately. Cause thresholds overlap and are not causal attributions.

The preliminary follow-up screen requires fewer failures, each tracking RMSE
no larger than reference ×1.10 +0.01 (m/s or rad/s), joint violation frequency
increase at most 0.001, and torque-excess maximum increase at most 0.01 Nm.
These are explicit practical margins, not significance tests. Results at one
seed remain preliminary; use seed 2 only for A and the promising candidate.

Observed November 24 results (`diagnostics/intervention10500/full`): checkpoint
10500, SHA256 `d0f1116eb39ee6a0796237c473ae45d6bc30012c0a43e76cc965da3c008097a4`.
Each condition: seed 1, ten environments, 1000 control steps, 200 env-seconds.

| Condition | Failures | Failures/env-s | RMSE vx / vy / wz | Completed episode mean [s] |
|---|---:|---:|---|---:|
| A | 12 | 0.060 | 0.192 / 0.221 / 0.202 | 11.01 |
| B, extra alpha 0.5 | 4 | 0.020 | 0.272 / 0.278 / 0.222 | 15.17 |
| C, attitude 0.05 | 6 | 0.030 | 0.238 / 0.222 / 0.193 | 14.32 |
| D, planar weight 10 | 8 | 0.040 | 0.254 / 0.283 / 0.197 | 13.35 |

Tracking units: m/s, m/s, rad/s. Completed-episode means exclude unfinished
episodes; survival plots retain right-censoring. Family exposure was identical:
60 stair-down, 60 stair-up, 80 discrete-obstacle env-seconds, levels 6/7/8.
Initial state hashes matched, but asynchronous resets changed subsequent
command/randomization streams. Shared pre-reset windows covered 98–115
env-seconds per comparison; B's tracking penalty was smaller there than in
the full evaluation. These are preliminary single-seed results.

All 160000 primary candidates were accepted; zero recovery/fallback rows,
zero measured actuator torque-limit exceedance. Small measured joint-limit
exceedances remained. D had a rare velocity-excess peak of 5.36 rad/s versus
1.81 in A. Post-cap predicted endpoint violations were nonzero, illustrating
why execution cannot inherit candidate certification. Reconstructed executed
torque matched within 5.3e-6 Nm, and model tensors remained unchanged.

Executed-model planar error worsened versus nominal in 9.52%, 5.21%, 8.18%,
and 15.58% of valid A/B/C/D rows. Nearly all such rows improved predicted
attitude cost. These are model cost-tradeoff associations, not proof of the
cause of physical failures; execution costs assume optimized contact forces.

No alternative passed the declared survival-plus-tracking screen, so neither
alpha-0.75 nor seed-2 follow-ups were run. No training continuation is
recommended from this experiment alone. All 30 failures were captured, with
zero capture drops; healthy references and objective CSV/plots are included.
Collection/diagnostic wall time was 215–221 s per condition, excluding startup;
PyTorch peak allocated memory was 81–83 MiB, excluding PhysX/renderer memory.
A sampled GPU-1 total was about 4.6 GiB. GPU 0 hosted ongoing training, so
shared-host interference remains possible; these are not deployment latencies.

Validation executed: seven focused evaluator tests (including production
Q/p cost parity and ring ownership), CUDA execution-cap regressions, and two
12-step A/B simulator smokes. Isaac Lab emitted file-watcher-limit warnings
but completed every run. No production settings or training weights changed.
# Reset-independent wrench-load stress test (model 10500)

Evaluation only, physical GPU 1. This bounded driver runs B/C/E/F at seed 1,
then the selected candidate and QP-off at seed 2, with light/heavier wrench
tapes (eight runs, 10 environments x 1000 control steps). No tracking veto:
selection sorts failures, roll+pitch RMS, height-error RMS, then tracking RMSE.

```bash
conda activate lr_lab_cupiqp
CUDA_VISIBLE_DEVICES=1 SIMULATOR=isaaclab PYTHONPATH="$PWD" \
python -u scripts/run_hard_pact_wrench_stress.py \
  --output diagnostics/wrench_stress10500/full
```

Native command/heading targets and persistent disturbance waveforms are
precomputed against absolute control time. Heavy amplitudes are twice light,
bounded by final saved envelopes **at event amplitude**, preserving waveform
shape. Realized mass/CoM, gains, friction and joint properties remain fixed;
action delays retain their initial samples. Push/noise RNG streams are keyed
by absolute step and isolated from resets/QP selection. Heading feedback still
depends on actual heading, so realized yaw commands can diverge. Reset history,
state, queue contents and trajectories necessarily diverge after failures.
The source checkpoint's applied-wrench frame/origin conversion is unchanged.

`tapes.npz` and manifest hashes allow paired verification. QP-off retains the
same actor/estimator/controller path but bypasses the intervention callback.
Full candidate and post-cap predictions retain candidate GRFs, not measured
forces. They are not certificates of measured motion. Normal termination is
unchanged; failure histories are capped at 24/run and aggregates retain all
events. This is a wrench-load stress test, not moving-liquid transport.

Focused checks:
```bash
python -m pytest -p no:cacheprovider tests/test_hard_pact_wrench_stress.py tests/test_eval_hard_pact_tradeoff.py -q
python scripts/run_hard_pact_wrench_stress.py --output diagnostics/wrench_stress10500/full --report-only
```

### Checkpoint-15000 preservation / oracle diagnostic

`run_hard_pact_preservation_eval.py` pins the Nov23 `model_15000.pt` hash.
It restores that checkpoint's current disturbance envelope (the inherited
`light` selector does not mean checkpoint-10500 bounds). Conditions are `off`
(A), `A_reference` (B, four alpha=1 solves), `C_torque30` (C), and
`D_true_wrench` (D, diagnostic only). The oracle replaces only the QP input,
holding the current applied base-origin wrench yaw-local at control rate.
It includes the application-point moment once, not randomized inertia/gravity.

```bash
CUDA_VISIBLE_DEVICES=1 SIMULATOR=isaaclab PYTHONPATH=. python scripts/run_hard_pact_preservation_eval.py --output diagnostics/preservation15000/smoke --smoke
CUDA_VISIBLE_DEVICES=1 SIMULATOR=isaaclab PYTHONPATH=. python scripts/run_hard_pact_preservation_eval.py --output diagnostics/preservation15000/full
# Only after C passes the preliminary tradeoff assessment; B/C only:
CUDA_VISIBLE_DEVICES=1 SIMULATOR=isaaclab PYTHONPATH=. python scripts/run_hard_pact_preservation_eval.py --output diagnostics/preservation15000/full --seed 3
```

Workers use physical GPU1 / internal cuda:0. Each full condition is 10
environments x 1000 controls. Completed conditions are skipped; reports include
paired input/config assertions, active exposure, and world-vertical velocity.
Seed-specific CSV/JSON and plots, manifests, bounded histories and source copies
are retained. No optimizer is constructed or updated by this evaluation.
