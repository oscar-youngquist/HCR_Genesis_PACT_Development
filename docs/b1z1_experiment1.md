# B1Z1 Experiment 1: frozen disturbance rejection

CPU-only summary/comparison notebooks and metric conventions are documented in
[b1z1_experiment1_analysis.md](b1z1_experiment1_analysis.md).

Run in the existing Isaac Lab Python environment, from any working directory:

```bash
bash legged_gym/scripts/eval_b1z1_exp1.sh \
  --task b1z1_pact --run /path/to/run --ckpt 5000 \
  --condition combined-disturbance --load-range out \
  --motion-mode mixed --num-envs 2 --episodes 10 \
  --horizon-seconds 12 --seed 0 --gpu cuda:0 --headless \
  --output-dir /path/to/new/evaluation
```

`--run` accepts an existing directory or a run name under the registered
`logs/b1z1/<experiment>` directory. `--ckpt` accepts an integer, checkpoint
filename/path, or `-1` for the existing latest-checkpoint resolution. Loading is
strict. Use a trusted local checkpoint. The evaluator does not restore optimizer,
curriculum, entropy, or training-counter state. Output directories must be empty.
The thin launcher uses the active Python, with no Conda paths or module loads.

## Methods and isolation

The registered runner determines actor construction, without constructing the
runner, training storage, auxiliary optimizers, or BARD. PILOT uses the registered
`b1z1_pact` name. All registered B1Z1 ablations, current `b1z1_ppo_pos`, legacy
`b1z1_pact_pos`, and UniFP models are supported. Current Pos retains its inherited
UniFP gains; ablations 6/7 retain PACT gains. No training files are modified.

For `b1z1_unifp_original`, use its **Original checkpoint**, not a Reject checkpoint.
The evaluator instantiates the existing Reject environment with the Original
observation/model configuration, calls its existing force-estimate publisher,
and retains its filtering and reset behavior. Compensation beta is frozen to 1
by default. Predictions, never ground truth, drive compensation. The native
12-D order is velocity, spherical EE position, EE force, base force. This model
does not estimate a torso moment. Applied moments remain legitimate disturbances.

Every method is scored against the nominal spherical EE trajectory and body-frame
base-velocity command. Force-shifted tracking targets are disabled only in these
isolated evaluation instances. Training rewards are not computed: common metrics
replace them, avoiding training-only BARD reward capture.

## Editable protocol

`EVAL_CONFIG` in `eval_b1z1_exp1.py` is the single override block. The default
protocol is a flat plane, no observation noise/physical randomization, mixed
stationary/locomotion trials, and onset/change/removal at 2/5/8 seconds.
**Timing, tolerance and failure thresholds are provisional.** Review them before
collecting comparative results. With a shortened horizon, events outside the
horizon are explicitly untested; use the smoke harness to compress event times.

External bounds are per world axis, not vector norms:

| Range | EE force | Torso force | Torso moment |
| --- | --- | --- | --- |
| in | +/-50 N | +/-35 N | +/-10 Nm |
| out | +/-70 N | +/-55 N | +/-20 Nm |

Each out-range loaded event exceeds a training bound on at least one axis of
each enabled load block. Nominal disables all loads; EE-only disables torso
loads; combined loads overlap. Application switches remain enabled; training
samplers are bypassed. Records read the actual simulator application buffers.

Nested overrides are materialized independently, including inherited class
sections. Unknown keys fail. Enabled randomizations use configured final bounds
once per trial; equal range endpoints give a fixed value. Disabled randomizations
retain native unrandomized properties (they do not apply their range endpoints).
Add native range fields under `overrides` as needed, including CoM bounds and noise
scales. PACT branch-weight sampling has its own toggle and uses native ranges.
Resolved configs and effective body mechanics are saved. Nominal PD gains remain
method-specific. Curricula, generic pushes and random force commands cannot advance.

Scenarios use separate seed/trial/feature/step RNG keys for events, commands,
initial pose, physical properties, delay, and noise. Native randomization calls
run inside saved/restored RNG contexts. Policy calls do not affect the scenario.
Physical trajectories need not be bit-identical across GPU batch sizes. Terrain
wrapping is disabled; leaving a finite terrain aborts evaluation rather than
silently teleporting. Non-plane terrain and every DR combination require further
simulator validation before a large experiment.

## Record schema

| File | Contents |
| --- | --- |
| `metadata.json` | Resolved environment/policy/evaluation configs, checkpoint/git/URDF hashes, ordering, units/frames, available/missing predictions |
| `initial_states.jsonl` | Per-trial initial physical state, origins, full body masses/inertias/local CoM poses, hash |
| `transitions-*.npz` | Bounded compressed numeric chunks; load with `numpy.load(..., allow_pickle=False)` |
| `records.json` | Field shapes/dtypes and ordered chunk row counts |
| `trials.json`, `trials.csv` | Trial summaries; JSON also includes events and phase statistics |
| `seed_summary.json` | Within-seed metrics, separated by condition/range/motion mode |
| `COMPLETE` | Written only after all trials flush and model weights are verified unchanged |
| `ERROR.txt` | Failure traceback, if evaluation failed after output creation |

Rows align `obs_t/history_t/action_raw` with the executed interval and its physical
successor, **before any reset**. `time_seconds` is interval start, successor time
is start + dt. No automatic environment reset occurs during a trial. Dead slots
are not recorded again; the next synchronous batch uses native buffer resets.
Total trial IDs, including failures, are exactly `0..episodes-1`.

Joint arrays follow configured DOF order; foot/sensor maps and full body names are
recorded separately. Quaternions are xyzw; world z is up. Native observations and
decoder predictions retain normalization documented in the resolved config.
Missing predictions are omitted and listed in metadata, never replaced by truth.
Recorded states/actions, decoder context, references and mechanics can support
Experiments 2–3 offline; those experiment schedules/figures are not implemented.

`*_interval_mean` averages **raw substep** samples; filtered GRFs have a separate
field. PD and FF torques are recorded before and after branch/motor weighting.
UniFP uses its actual feedback torque, not the allocated-but-unused combined-PD
buffer; its FF contribution is zero. Weighted motor-scaled PD plus FF equals
the pre-clipping requested total for every controller family.
`prediction_grf_torque_input_nm` follows the registered training runner: BARD
PACT uses the clipped current action and pre-step dynamics state, with configured
gains, branch weights, motor scaling and 1.1-times-rated clipping. As in training,
this nominal input does **not** apply action delay. Legacy PACT-Pos and non-BARD
PACT instead use the final executed substep torque (which includes delay).
These decoder inputs are distinct from requested/executed interval averages.
`torque_request_pre_motor` is not the actual actuator request: motor scaling
follows it. Actual clipping uses **1.1 times rated limits**. Saturation and joint
limit frequencies use substep-by-DOF denominators, not control-step maxima.
Peaks/extrema and counts are stored alongside interval averages. Negative
`source_action_id` denotes the reset-filled zero action in a delay queue.
Applied wrench points are body CoMs, recorded in world coordinates. Robot CoM
and ground-support moment are derived from measured masses/CoMs and raw GRFs;
they are not assumed equal to the torso origin or filtered-contact wrench.

Physical failure is immediate at the first evaluated control endpoint with
disallowed torso contact, terrain-relative height, or roll/pitch violation.
There is no persistence/patience requirement; horizon completion alone is not failure.
Errors use finite **pre-failure** samples. Terminal rows remain in the recording,
including failure coinciding with horizon completion. `valid` means finite data;
`alive_for_metrics` excludes the failure transition. `S_track` is the conjunction
of EE/planar/yaw tolerances divided by the **full prescribed K**, so failed tails
contribute zero. Recovery requires continuous dwell; failed trials remain
unrecovered. Incomplete response-window IAE is null, with observed partial IAE
and counts retained. Do not compare partial IAE as though it covered a full window.
`aggregate_across_seeds()` averages seed summaries, not pooled transition rows.
The training seed is explicitly unavailable when not stored in the checkpoint;
it is never inferred from the evaluation seed. Optional acceleration-RMS summaries
are not computed online; velocities and reset-safe time indices support them offline.

## Focused checks

### Regenerating seed summaries

Physical-limit frequencies pool each seed's trial violation counts and valid
joint-substep denominators, not trial percentages. Zero total exposure yields
`null`; aggregate `*_count` and `*_denominator` fields retain the audit trail.
Tracking errors, S_track and failure rate retain equal trial weighting, and
across-seed summaries retain equal seed weighting.

Rebuild older Exp1 **or Exp2** summaries without a simulator using completed run
directories (not approach parent directories):

```bash
python legged_gym/utils/b1z1_experiment.py \
  /path/to/exp1/seed_0 /path/to/exp2/seed_0
```

This reads `metadata.json` and `trials.json`, requires `COMPLETE`, and atomically
replaces only `seed_summary.json`. No raw recordings, trials, or metadata are
modified. Exp2 is selected from its metadata and retains its stage summaries.
Re-export any downstream summaries separately if they used old online frequencies.

```bash
SIMULATOR=isaaclab_b1z1_pact python -m pytest tests/test_b1z1_experiment1.py -q
python tests/smoke_b1z1_experiment1.py \
  --pact-run /path/to/pact --pact-ckpt 5000 \
  --pos-run /path/to/current-pos --pos-ckpt 1 \
  --unifp-run /path/to/original-unifp --unifp-ckpt 1000 \
  --gpu cuda:0 --output-dir /tmp/experiment1-smoke
```

The smoke harness runs isolated five-step cases, plus a one-step forced-failure
case, and checks actual loads, finite shapes, successor alignment, exact trial
counts, frozen weights, flush, and completion. Ablation smoke cases reuse
architecture-compatible checkpoints to test dispatch only, not performance.
# FiLM activation records

Exp1 and Exp2 transition chunks additionally capture `film_features_pre_t`,
`film_features_post_t`, `film_gamma_t`, `film_beta_t`, and `film_condition_t`
for FiLM-enabled methods. They are detached snapshots of the actual action_t
inference pass, with no extra forward or RNG draws. The dimensionless transform
is `post = pre * (1 + gamma) + beta` (gamma is a residual scale); the condition
uses native normalized model inputs. Dimensions are inferred from the model and
listed in `records.json`. `metadata.json:film_fields_available` lists available
fields; non-FiLM methods omit them and report unavailability. Existing recordings
without these fields remain valid but cannot recover these activations directly.
