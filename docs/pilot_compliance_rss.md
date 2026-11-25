# PILOT-only RSS differential-compliance evaluation

This evaluates one frozen checkpoint; it does not establish robustness across
independently trained policies. No training, controller, simulator, compliance
gate, or registered task implementation is changed.

## Commands

Activate the existing Isaac Lab Python environment. From the repository root:

```bash
python -m unittest discover -s tests -p test_pilot_compliance_eval.py -v
bash legged_gym/scripts/eval_b1z1_pilot_compliance.sh \
  --headless --gpu cuda:1 --seeds 0 --episodes-per-condition 2 \
  --conditions nominal --num-envs 2 --output-dir analysis/pilot_compliance_rss/nominal
bash legged_gym/scripts/eval_b1z1_pilot_compliance.sh \
  --headless --gpu cuda:1 --seeds 0 --episodes-per-condition 1 \
  --calibration-episodes 1 --num-envs 10 --output-dir analysis/pilot_compliance_rss/smoke
bash legged_gym/scripts/eval_b1z1_pilot_compliance.sh \
  --headless --gpu cuda:1 --seeds 0 1 2 --episodes-per-condition 100 \
  --calibration-episodes 10 --num-envs 128 --output-dir analysis/pilot_compliance_rss/full_instrumented
python legged_gym/scripts/plot_b1z1_pilot_compliance.py analysis/pilot_compliance_rss/full_instrumented
python legged_gym/scripts/analyze_pilot_reference_stages.py analysis/pilot_compliance_rss/full_instrumented
```

`--run` selects an exact directory; default is
`logs/b1z1/b1z1_pact_lab/Nov19_04-02-42_b1z1_pact_improved`.
`--checkpoint 23200` pins a number explicitly; otherwise the highest numeric
checkpoint in that directory is pinned once. There is no fallback run.
`--validate-only` checks strict loading and finite native inference without
initializing a simulator. `--config settings.json` overrides fields in the small
`EVAL_CONFIG` block; unknown fields are rejected. Use new output directories.

## Protocol and conventions

Default horizon is 16 s. The native nominal EE quintic trajectory lasts 3 s;
force rises over 5-6 s, remains constant until 10 s, falls over 10-11 s, and
remains zero through 16 s. Force magnitudes are independently pre-generated
within 19-21 N for each paired trial. Profiles are held for the episode:
EE `[-x,+x,-y,+y,-z,+z]`, base `[-x,+x,-y,+y,-yaw,+yaw]`.
Active compliance is 0.5, protected/unused directions are exactly zero.
Forward recruitment uses only +x base compliance; yaw remains disabled.

The four vertical conditions cross upward/downward yielding with upward/downward
force. The four planar conditions cross ample/restricted arm commands with
base compliance on/off. The FK grid finds feasible *command witnesses* within
the saved spherical workspace and checks EE clearance. It never initializes
the robot to those joint configurations. The policy reaches each command from
its normal reset posture. Actual settled margins and tracking failures are
reported separately; a feasible witness is not proof of learned reachability.

Forces act ON the robot at its native EE link origin. Planar directions follow
current base heading, z is world vertical. The same held force feeds native
compliance and the simulator's existing per-substep composer. There is no
recruited physical torso force. Native control dt is 0.02 s, with four 0.005 s
physics steps. The recorder checks one reference advance per control interval
and equality between published and applied EE forces. Torso loads must be zero.

Evaluation intentionally uses fixed nominal mechanics, no observation noise,
and fixed zero action delay, while retaining the native delay/history path.
Saved gains/scaling, physical limits, strict architecture, and termination are
retained. These controlled experiments are not domain-randomization robustness
tests. Initial yaw, xy, and joint perturbations come from saved manifests,
shared across paired conditions. Policies can still reach different states.

## Lifecycle and records

The evaluator subclasses only scheduling/reset hooks. It calls native
`check_termination` and snapshots the physical successor before native automatic
reset. The first failure ends scoring, including reference-recovery failures;
failed trials are never replaced. Inactive vector rows may continue simulating
but contribute no additional trials. Prescribed horizon is not a failure.
Profiles are installed before the first observation; actor and environment
histories reset through their native mechanisms.

`metadata.json` records checkpoint/hash, source revisions, saved and resolved
configurations, kinematic witnesses, units/frames, and frozen-weight checks.
`trial_manifest.json` contains all paired draws before collection begins.
`seed_N/condition/trial_NNNN.npz` holds bounded per-trial numeric trajectories;
the adjacent JSON is its episode summary. `time` is successor time;
`interval_start` is action/load time. `raw_action` is native deterministic output,
`executed_torque` is the final physics-substep torque (not interval average).
Task EE xy are base-heading-relative; task z is absolute world height. Task
velocities include frame transport. `clipped` comes from the compliance target.
`valid_transition` retains terminal records; `alive_for_metrics` excludes the
failed successor from tracking statistics. `failure_reasons` names are in the
summary utility. `COMPLETE` is the authority for successful collection, not the
simulator process exit code. Failures of infrastructure produce `ERROR.txt`.

## Statistics and plots

Tracking errors use finite pre-failure successors. Loaded signed displacement
subtracts the trial's mean measured-minus-nominal error during 4-5 s from its
mean during 7-10 s. Reference displacement uses the same operation independently.
Protected deviation is the absolute signed loaded displacement, not absolute
unloaded tracking bias. Recovery error uses the last 2 s relative to baseline.
Missing windows are null, never zero. Surviving seconds, baseline/load/recovery
counts, native failures, setup tracking failures, and margins expose censoring.

Trials are equally weighted within evaluation seeds, seeds equally within each
condition, and conditions equally in the overall summary. Sample SD uses ddof=1;
one seed has unavailable SD. Paired contrasts verify manifest hashes and bootstrap
episodes within each seed (2,000 draws), then equally average seed contrasts.
These CIs describe evaluation randomness conditional on this checkpoint and
available paired windows, not training uncertainty or a matched-state experiment.
All attempted trials remain in failure-rate denominators.

`episodes.csv/json`, `summary.json`, `paired_contrasts.json`, and
`results_table.csv` hold the numerical results. Two PDF/PNG plots show mirrored
EE responses and base responses. Curves average surviving trials within seeds,
then equally average seeds; bands show sample SD across seeds. No failure-tail
zero filling or interpolation. `plot_curves.csv` preserves contributor counts.
`representative_trials.json` selects nearest-median absolute loaded-displacement
trials with deterministic seed/ID tie-breaking, not the most successful trial.

Read reference behavior and measured robot behavior separately. Recruitment
gate activation is not evidence of actual successful base recruitment; clipping
is not evidence of bounded measured arm motion.

## Reference-stage diagnostics and calibration

`reference_diagnostics: true` in the evaluation config opts into detached snapshots
inside the native `ComplianceReference.advance`. Training defaults remain off.
The source audit strips only the new diagnostic assignments/branches and checks
AST equality against the saved checkpoint's reference implementation. Unit tests
also check bitwise-identical position/velocity/base/gate outputs with logging on
and off. There is no second filter call, force application, or actor input.

New NPZ keys have prefix `reference_stage_`: task-frame applied force,
sign-selected coefficients, dimensional gains (m/N), raw displacement,
workspace-bounded equilibrium, combined-geometry-restricted equilibrium,
filtered candidate/velocity, nominal position/velocity, and accepted
position/velocity. These are interval-end references paired with the physical
successor. The geometry stage combines the existing target-safety predicate;
it is not uniquely attributed to collision, joint limits, or another cause.
Final guarding and recovery can change the filter output. Differences are
recorded as vectors/norms, not assumed attenuation ratios. Gates and native
clipping/recovery/termination snapshots remain separate.

Two additional calibration conditions use exactly 30 N and 50 N upward force,
coefficient +z=1, all other EE/base coefficients zero, the same interior nominal
pose, and the same 5-10 s load timing. Default is **10 episodes per level per
seed**, independently configurable with `--calibration-episodes` (zero disables).
They are not pooled into the original eight-condition summary. With the saved
0.005 m/N gain, raw demands are 0.15/0.25 m. Bounds and gains are not relaxed;
measured motion is not assumed equal to the reference. Nominal-only checks
automatically omit calibration.

`reference_stage_trials.csv.gz` contains per-trial conditional mean, median,
p90/p95, maximum, threshold fractions, sample counts and durations.
`reference_stage_seeds.csv` additionally retains exact within-seed sample
distributions, equal-trial means and mean episode maxima.
`reference_stage_summary.csv` reports equal-seed means/sample SD, true maxima,
and separately named mean episode maxima. Quantiles are sample distributions,
not timestep-based inferential confidence intervals. `reference_exposure.csv`
contains raw/accepted exceedance counts and the conditional fraction of raw
requests above 10/15/20 cm that remain above the same threshold after acceptance.
A 1 micrometer tolerance prevents float32 roundoff turning exactly 15 cm into
an apparent exceedance. Vector norms, signed loaded-axis components and absolute
loaded-axis magnitudes are distinct metrics.

Masks are finite pre-failure samples: all; force-active yielding (actual axis
force >1e-6 N in magnitude and selected coefficient strictly >0); sustained
yielding (also phase 2); and protected active (coefficient exactly zero).
Small positive coefficients are counted separately, not treated as disabled.
Every condition/seed retains its own summaries, including ample/restricted
capability. Empty groups are null. Legacy files without snapshots are explicitly
unavailable; aliased legacy offset/clipping fields are never substituted.

`reference_stage_reduction.pdf/png` shows sustained raw, bounded, accepted and
measured **nominal-relative norms**, whereas the original response plots use
baseline-corrected signed motion. They answer different questions.
`calibration_traces.pdf/png` shows applied forces, world-height nominal/reference/
measured traces, intermediate offsets and restrictions. Examples are selected
nearest median sustained accepted offset, with deterministic seed/trial ties.
`reference_diagnostics.json` records definitions, source checkpoint, missing
instrumentation, summaries and selected example IDs.

## Executed evaluation

The completed dataset is `analysis/pilot_compliance_rss/full_instrumented`:
2,400 main trials plus 60 calibration trials, seeds 0/1/2, with 1,968,000
transitions. See its `REPORT.md` for measured outcomes and reproduction commands,
and `validation_report.json` for the independent alignment/count audit.
Collection ran on cuda:1 in 885.109 seconds; 69 focused/existing tests passed.
No physical/reference failures occurred, but tracking was imperfect: calibration
references reached 15/25 cm while baseline-corrected measured shifts were
9.56/13.90 cm. Restricted-arm base recruitment requested 0.079 m/s while actual
forward velocity averaged 0.00490 m/s. These shortcomings are retained, not
reclassified as successful behavior. Earlier uninstrumented and incomplete
attempts remain separate and are documented in that report.
