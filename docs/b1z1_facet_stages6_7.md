# FACET Stages 6-7: Phases, Deployment, and Evaluation

This extends Stage 5, not the protected PILOT/Pos/UniFP controllers or runners.
The existing 237/52/9 observation packets, 17 position actions, coupled virtual
references, temporal rewards, PPO protocol and physical task are retained.
Leg gains remain 100/2 and arm gains 40/1. There is no terrain curriculum,
signed-compliance target, recruitment gate, learned FF head, BARD or PINN.
The [Stage 5 reward/configuration exception table](b1z1_facet_stage5.md) still applies.

## Phases and Budget

| Completed global updates | Default phase | Collection / updates |
|---|---|---|
| 0-59,999 | Teacher | Privileged teacher PPO; online estimator latent supervision every other update |
| Optional interval starting at 60,000 | Adaptation | Student with **online** estimator; only estimator/EMA update |
| Until total update 70,000 | Student | Student with **EMA** estimator; actor/critic PPO and separate estimator supervision |

`algorithm.adaptation_iterations=0` by default. A value of 2,000 changes the
last 10,000 updates to 2,000 adaptation + 8,000 student, not 12,000 additional
updates. An adaptation update still collects the same 24 transitions per robot;
it consumes interaction budget even though PPO optimization is skipped.
Teacher exit initializes student actor/std once. Adaptation exit never recopies
them. Both online and EMA recurrent states remain live across phase changes.
Normalizers and teacher encoders freeze at teacher exit; actor/critic optimizers,
curricula, references, episodes, and counters are not restarted at handoff.
EMA tau remains .04, with two separate estimator epochs per adaptation/student
update. Student PPO uses detached collection-time EMA features.

At 4,096 robots, the default budget is 6,881,280,000 training transitions:
5,898,240,000 teacher and 983,040,000 student. Checkpoints and TensorBoard
`Budget/*_interactions` count actual completed environment steps, including
resumed runs. Evaluation steps are recorded separately. Robot-seconds sum
`num_envs * control_dt` per step; they are not elapsed wall time.

## Resume and Metadata

Version-2 checkpoints include model/EMA/normalizers/optimizers, separate student
initialization status, phase update/interaction counters, saved schedule,
curricula, reference settings, observation/joint ordering, FK constants and
control/action/delay conventions. The FACET launcher below restores saved
configuration **before** simulator construction. The runner rejects incompatible
physical/observation contracts rather than loading weights into a changed task.
Explicit schedule overrides are `student_start_iteration`,
`adaptation_iterations`, `max_iterations`, and `save_interval`. An override
never reverses an already completed actor initialization or phase.

Stage 5 checkpoints recover initialization from their old transition flag and
require their original `facet_resolved_config.json` (or explicit verified
`legacy_contract`). Their missing sampling-mode field means **fixed**, not the
new sampled default. Old interaction totals are reconstructed from saved rollout
size and marked `accounting_exact=false`: historic batch-size changes and wall
time were not saved. Portable export additionally requires verified FK metadata;
it will not guess missing geometry. New checkpoints preserve exact counters.
Fresh simulator episodes reset sensor/reference/recurrent histories on resume;
this is not bitwise continuation of the unsaved simulator state.

## Impedance Sampling

New training uses `facet_reference.sampling_mode='upstream_b1z1'`. Set `'fixed'`
with empty `sample_ranges` for deterministic fixed settings. Legacy custom
independent `sample_ranges` remain supported only in that mode.

The pinned [upstream sampler](https://github.com/btx0424/active-adaptation/blob/c0d6f5aa6146814803090b5f6f19aafb8ce6f792/projects/facet/facet_commands/impedance_manip.py)
defines base K uniform [2,30], base D = 1.8 sqrt(K), yaw K/D equal to base K/D;
EE K uniform [4,30], EE D = 1.8 sqrt(K), EE virtual mass 2, yaw inertia 1.5.
Effective base mass is `2**randi(1,4)`, hence **2,4,8**, not the unused YAML
mass-range comment. Alpha remains the completed local formulation's configured
value (default 1); the released sampler supplies no alpha distribution.
Our adaptation samples at episode reset only, not upstream's separate command
resampling timers. This preserves the shared nominal-goal sampler. Dedicated
reference RNG seed 1701 is configurable and does not consume shared task RNG;
partial resets leave other robots untouched. Observations read the same sampled
parameter buffers used by reference dynamics. RK4 substeps are bounded using the
full supported range, with the existing .005 s maximum.

## Standalone Mean-Action Deployment

`python -m rsl_rl.facet_deployment --checkpoint CHECKPOINT --output student.pt`
exports TorchScript with embedded JSON metadata. It contains only observation
normalization, EMA estimator's 160 feature outputs, student MLP/mean head, FK,
and sensor-history assembly. No teacher, critic, privileged encoders, optimizer,
Gaussian std or untrained wrench reconstruction head is exported. Loading and
executing the artifact needs only PyTorch, not this repository or Isaac Lab.

An uninitialized student fails inference/export. `--copy-teacher` explicitly
creates a separate estimated-feature artifact from teacher weights/std-compatible
mean architecture; it does not modify the source checkpoint or imply student
fine-tuning. A trained student cannot be overwritten using that option.

Artifact inputs at each control boundary are nominal commands (9), commanded
impedance (10), projected gravity, physical-order q/qd (19 each), previous issued
position action (17), `is_init`, sensor history [N,4,41], action history [N,3,17],
and GRU state [N,128]. Outputs are raw mean action and next explicit states.
Sensor data use raw SI units; nominal XY is heading-relative and height is world
referenced; FK position is trunk-relative. Histories are newest-first, reset
sensors repeat the first measurement, reset action history is zero. Previous
issued actions are clipped before delay exactly as in Stage 4. The caller still
applies existing position scaling, clipping/delay, held joints and PD execution.
No force labels, reference history, contacts or privileged state are accepted.

## Commands

From the repository root, activate the existing Isaac Lab Python environment.
The ordinary `train.py --task b1z1_facet` remains functional. The FACET-specific
launcher exposes phase overrides and restores saved configuration on resume:

```bash
PYTHONPATH=. python legged_gym/scripts/train_b1z1_facet.py --gpu cuda:0 --headless --seed 0 --num-envs 4096
PYTHONPATH=. python legged_gym/scripts/train_b1z1_facet.py --gpu cuda:0 --headless --adaptation-iterations 2000
PYTHONPATH=. python legged_gym/scripts/train_b1z1_facet.py --gpu cuda:0 --headless --resume RUN/model_62000.pt
python -m rsl_rl.facet_deployment --checkpoint RUN/model_70000.pt --output RUN/student.pt
python legged_gym/scripts/eval_b1z1_facet.py --artifact RUN/student.pt --gpu cuda:0 --headless --seed 0 --episodes 36 --output-dir evaluation/facet_seed0
```

Omit `--headless` and use `--num-envs 1` for visual inspection. Same artifact,
sensor builder, scenarios and physical action processing are used.

Prepare independently trained seeds 0, 1 and 2 for each method, using existing
baseline entry points and the same requested total interaction budget:

```bash
for seed in 0 1 2; do
  PYTHONPATH=. python legged_gym/scripts/train_b1z1_facet.py --seed "$seed" --num-envs 4096 --max-iterations 70000 --gpu cuda:0 --headless
  for task in b1z1_pact b1z1_ppo_pos b1z1_unifp_original; do
    backend=isaaclab_b1z1_pact
    case "$task" in *unifp*) backend=isaaclab_b1z1_unifp ;; esac
    SIMULATOR="$backend" PYTHONPATH=. python legged_gym/scripts/train.py --task "$task" --seed "$seed" --num_envs 4096 --max_iterations 70000 --gpu cuda:0 --headless
  done
done
```

These commands are prepared, **not executed**. Confirm baseline rollout counts
in their resolved configs before claiming equal interaction budgets. Baseline
gains/architecture remain method-specific; their runners are not modified here.
To pair evaluation, invoke `eval_b1z1_facet.py --task BASELINE --checkpoint PATH`
instead of `--artifact`, with the same evaluation seed, horizon and trial IDs.
Original UniFP uses the existing frozen estimator-driven Reject interface.

## Scenario and Report Contract

The 36-trial grid covers standing/walking, EE-only/base-only/combined,
lateral/opposing pulls, and K=4,16,28. Loads step at 3/7/11 s from zero to
20/35/zero N EE, with torso force .7 times those magnitudes and an independent
.1 times magnitude yaw couple in Nm when torso loading is selected. Horizontal
directions are fixed to initial heading. Base command is 0 or .3 m/s and yaw
command zero; nominal EE sphere is [.5,.1,0]. Fixed-impedance checkpoints retain
their fixed parameters instead of being presented as range-trained policies.

Initial conditions/mechanics and observation noise use trial/feature-keyed RNG,
and `scenarios.json` records actual sampled states, masses, gain multipliers and
scenario hashes. Actual interval loads are recorded, not just intended schedules.
Metadata retains production force/domain curriculum state separately from the
explicit evaluation protocol. Seeds alone are not evidence of matching exposure.

`trials.json` retains failures and termination reasons; nominal tracking excludes
failed endpoints and S_track uses the full horizon, with zero failed tails.
FACET horizon MSEs retain per-horizon validity counts separately. Reports include
contacts, survival, integrated force exposure, stage signed errors, initial-heading
displacement/velocity changes and release recovery (joint tolerances for .5 s
dwell within the configured 2 s window). Incomplete windows stay flagged; missing
recovery is null. A recovery fraction is unavailable when live, unrecovered trials
have incomplete windows; the observed lower bound and censored count remain.
Reported overshoot is relative to the observed final .25 s of a
complete plateau, not an assumed equilibrium or a causal force-isolation result.
`records.json` and bounded `transitions-*.npz` retain successor/interval alignment.
`summary.json` contains throughput, configuration, source hashes and counts;
scalar summaries weight trials equally within this evaluation seed, retaining
undefined tracking errors and including failures in full-horizon outcomes.
`COMPLETE` is written only after flushing. All simulated rows, even inactive or
padding rows, count as evaluation interactions, never training interactions.

## Verification

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python -m pytest tests/test_b1z1_facet*.py -q
PYTHONPATH=.:tests OMP_NUM_THREADS=1 python tests/smoke_b1z1_facet_pipeline.py --gpu cuda:0 --output-dir /tmp/facet67-pipeline
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python tests/benchmark_b1z1_facet_training.py --device cuda:0 --num-envs 4096
```

The pipeline performs two teacher, two adaptation and two student updates on six
robots, then exports in a new process and runs 36 three-second student trials in
another. Test-local nonzero forces do not change production curricula. Training
and evaluation JSON results are separate. GPU tests compare scripted and native
actions/hidden states, including partial resets; fresh-process artifact tests
remove repository imports and simulator variables. No full-budget training or
trained-policy performance comparison is claimed by these integration checks.

### Actual Checks on This Machine

- CPU FACET suite: 72 passed, one CUDA-only skip. Focused deployment suite with
  GPU 0 available: 12 passed, including CPU/GPU action/hidden-state parity.
- Protected Pos/UniFP/PCGrad regressions: 14 passed, two pre-existing failures.
  Both were reproduced in a separate pristine archive of reviewed HEAD: Pos's
  test expects `film` while configured `none`; UniFP's early-curriculum test
  expects .25 while the current nominal stage supplies zero. Neither changed.
- Verified Isaac Lab pipeline: 2 teacher + 2 adaptation + 2 student updates,
  288 transitions each, 864 total; collection/update wall time 5.55 s at 6 robots.
  Separate student evaluation: 36 trials, 6 physical failures, 5,400 simulated
  evaluation transitions, 56.43 s including setup/recording, 26.39 s stepping,
  1.82 s sensor-adapter/inference time. Failed trials remain in the report.
- Fresh-process resume CLI: one more student update produced 1,008 cumulative
  training transitions (288/288/432), with exact accounting retained.
- Synthetic 4,096-robot, 24-step benchmark (no simulator): median collection /
  update times were teacher 29.57/157.36 ms, adaptation 26.89/49.19 ms,
  student 27.18/146.91 ms; peak allocated 890.95 MiB. Two warmups and three
  measurements per phase; supervised epochs included. These are not full-physics
  training iteration times, and the machine was not reserved exclusively.

Machine-readable results from the verified run are in
`/tmp/facet67-verified/{training.json,validation.json,evaluation/summary.json}`;
the resumed budget is `/tmp/facet67-resumed/budget.json`. The scripts regenerate
these outputs at any chosen directory. Baseline checkpoint evaluation through
the new entry point and full-budget matched-policy performance are still pending;
this smoke policy is deliberately only six updates old. No monotonic learned
impedance-response or recovery claim follows from its results.

### Seed, Reporting, and Resume Corrections

FACET launchers follow the other B1Z1 methods' local/container conventions:

```bash
bash legged_gym/scripts/b1z1_facet_lab.sh --seed 1
# Inside an allocated Unity Isaac Lab container:
bash legged_gym/scripts/b1z1_facet_unity.sh --seed 1
bash legged_gym/scripts/b1z1_facet_unity.sh --resume /path/to/model_60000.pt --seed 1
```

Both use `train_b1z1_facet.py`, explicitly default to `--seed=1` and headless GPU 0,
and forward arguments (`--seed`, `--num-envs`, `--max-iterations`, `--gpu`, phase
overrides, etc.). A forwarded seed overrides the default, including seed 0.
On resume, pass the checkpoint's seed if it is not 1; conflicting seeds are rejected.
The local script activates the same environment as `b1z1_pact_lab.sh`; override
`FACET_CONDA_SH`/`FACET_CONDA_ENV` for another installation. Unity uses container
Python without changing `CUDA_VISIBLE_DEVICES`. These are training entry scripts,
not Slurm submission/container setup scripts; the shared container wrapper's
existing task allowlist is unchanged. No production environment counts or budgets
are overridden by either script.

The FACET launcher copies the resolved training seed into the environment before
the registry initializes RNGs. Resume retains the checkpoint's training seed
(including when an older checkpoint saved a contradictory environment seed);
an explicitly conflicting seed still fails. Reference/noise RNG seeds are unchanged.

Phase selection uses completed global updates and the resolved schedule before
collection. A teacher checkpoint past `student_start_iteration + adaptation_iterations`
enters student directly; skipped adaptation consumes no interactions. Existing
student weights are never recopied, and no completed phase is reversed.

Evaluation force summaries cover the observed trial up to termination, not an
extrapolated full horizon. If any force measurement is missing, that channel's
`force_peak_N`/`base_force_peak_N` and `force_exposure_N_s` entry is null.
`observed_force_peak_N` and `observed_force_exposure_N_s` retain partial finite
measurements, ordered EE/base; zero observations produce null, not zero.
`force_valid_samples`, `force_observed_seconds`, and `force_measurements_complete`
describe coverage of the observed interval. `wrench_valid_samples` additionally
counts valid base moments (EE force/base force/base moment order). Failed endpoints
can contribute finite loads but never tracking credit; failure-inclusive outcomes
and raw nonfinite transition values/masks are retained. Reference diagnostics have
their own finite, valid horizon counts. JSON remains strict, with null for unavailable
measurements, and COMPLETE follows all successful report writes.

Focused reproductions/checks:

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python -m pytest tests/test_b1z1_facet_regressions.py tests/test_b1z1_facet_evaluation.py -q
python legged_gym/scripts/train_b1z1_facet.py --gpu cuda:0 --headless --seed 2 --num-envs 4 --max-iterations 1 --student-start-iteration 10 --adaptation-iterations 2 --save-interval 1 --log-dir /tmp/facet-f0849-fixes/fresh
python legged_gym/scripts/train_b1z1_facet.py --resume /tmp/facet-f0849-fixes/fresh/model_1.pt --gpu cuda:0 --headless --num-envs 4 --max-iterations 2 --student-start-iteration 0 --adaptation-iterations 1 --save-interval 1 --log-dir /tmp/facet-f0849-fixes/resumed
```

The regression suite checks seeds 0/1/2 through the real launcher/parser/registry
and RNG initializer, replacing only simulator execution and optimizer work. It
checks same-seed initialization, saved/logged seeds, legacy resume seed handling,
phase boundaries/overrides, optimizer continuity, and strict report round trips
with nonfinite loads/states, including first-step failures.

Post-fix validation: 105 CPU tests passed, one CUDA-only skip; the focused
deployment suite also passed all 12 tests with GPU access. The two real Isaac Lab
commands above completed on GPU 0: 96 teacher + 96 student transitions, zero
adaptation transitions. Seed 2 agreed in the initialization log, resolved configs,
and checkpoints. Collection/update wall time was 1.09 s fresh and 1.04 s resumed
(four robots, one update each; excludes simulator startup). Machine-readable
checks are `/tmp/facet-f0849-fixes/validation.json`. Nonfinite failure injection
was tested on CPU recordings/report writing, not by corrupting simulator state.
These are integration checks, not trained-policy performance measurements;
production schedules/settings and protected methods are unchanged.
Both shell launchers passed syntax/argument-forwarding tests (including resume
paths with spaces and GPU visibility); the local script also reached `--help`
through the installed conda environment. Remote Unity execution was not run.
