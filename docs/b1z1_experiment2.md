# Experiment 2: task-inspired loco-manipulation

Activate the Isaac Lab Python environment, then run from the repository root:

```bash
bash legged_gym/scripts/eval_b1z1_exp2.sh \
  --task b1z1_pact --run /path/to/run --ckpt 70000 \
  --profile pushing --loading loaded --seed 0 \
  --episodes 50 --num-envs 10 --horizon-seconds 28 \
  --gpu cuda:0 --headless --output-dir evaluation/exp2/pilot/pushing/loaded/seed_0
```

`--run` and `--ckpt` use Experiment 1 resolution (run directory/name and checkpoint
number/file or `-1` for latest). Loading is strict and frozen: no optimizer,
learning, history-storage allocation for training, or checkpoint curriculum restore.
Defaults are 50 total trials including failures, 10 robots, and 28 seconds. A partial
final robot batch records only requested IDs. The shell launcher uses active Python.
Output directories must be empty. The parent checks `COMPLETE`, since Kit shutdown
can hide a Python failure behind a zero exit status.

## Protocol

FiLM-enabled methods additionally record `film_features_pre_t`,
`film_features_post_t`, `film_gamma_t`, `film_beta_t`, and `film_condition_t`
in transition chunks. These come from the actual action-producing pass at time t,
not the successor or a second inference. Features are dimensionless and
`post = pre * (1 + gamma) + beta`; the condition retains native normalized units.
`metadata.json:film_fields_available` lists captured fields. Non-FiLM methods omit
them and explicitly report unavailability, rather than recording artificial zeros.

Heading feedback is enabled for all profiles, including stationary phases, in both
collection and visual playback. Each robot holds **zero relative to its measured
reset heading**, not world yaw zero. `heading_command` in `EVAL_CONFIG` enables it;
wrapped heading error divided by pi maps to `velocity_ranges[2]` (default
[-0.6, 0.6] rad/s), following the native PACT/UniFP convention. Actor-time and
successor nominal commands record their respective feedback yaw rates. Load
directions remain fixed to the initial heading. Experiment 1 is unchanged.

`EVAL_CONFIG` in `eval_b1z1_exp2.py` is the editable block. It deep-copies Experiment
1 initialization, observation noise, physical randomization and failure settings,
then overrides only the task schedule and tolerances. Currently these include
flat terrain, observation noise, friction in [0.8, 1.2], fixed randomized joint
armature/friction/damping, and immediate torso-contact/height/tilt failure. Inspect
the resolved `metadata.json` before comparisons; the current config is authoritative.
Do not assume an older Experiment 1 document's default timings/noise still apply.

| Time [s] | Base motion | Loaded control | EE trajectory |
|---|---|---|---|
| 0–3 | stationary | 0 N | initial pose |
| 3–3.5 | smooth velocity ramp | 0 N | initial pose |
| 3.5–7 | full velocity | 0 N | initial pose |
| 7–11 | full velocity | 25 N | smooth interpolation to final pose |
| 11–15 | full velocity | 45 N | final pose |
| 15–19 | full velocity | 65 N | final pose |
| 19–19.5 | smooth stop | 65 N | final pose |
| 19.5–23 | stationary | 65 N | final pose |
| 23–28 | stationary | abrupt release to 0 N | final pose |

| Profile | Forward velocity [m/s] | EE load direction | EE trajectory |
|---|---:|---|---|
| pushing | +0.40 | negative initial-heading x | spherical radius 0.50 → 0.60 m |
| pulling | −0.40 | positive initial-heading x | spherical radius 0.60 → 0.45 m |
| carrying | +0.40 | world downward | +0.08 m forward, +0.05 m lateral in yaw workspace |

Starting elevation/azimuth are 0.10/0 radians; carrying starts at radius 0.50 m.
Lateral/yaw velocity commands are zero. Interpolation is quintic with zero endpoint
velocity and acceleration; loads change abruptly at interval starts. Final EE poses
remain held. Load indices, base-command events and task stages are separate.
Torso forces/moments and EE moments are always zero. World-axis training-bound
exceedance flags use ±50 N for EE axes (65 N supplies OOD loading).

Horizontal loads use the measured **initial reset heading**, held fixed throughout
the trial. World z remains vertical. EE paths use the existing **current yaw-aligned
manipulation workspace**, not a fixed world waypoint. Native base commands retain
their body-frame convention. Initial task heading is saved per trial.

The common spherical validation envelope rejects invalid waypoints; it never clips
them per method and is not a proof of collision-free reachability. Full sequences
have complete 2-second load-response windows with 0.5-second dwell. Short horizons
are allowed for smoke checks and explicitly marked `shortened_sequence`; unobserved
stages are not evidence of task success.

## Methods and paired controls

PILOT is `b1z1_pact`, Pos is `b1z1_ppo_pos`, and UniFP uses
`b1z1_unifp_original` with its **Original checkpoint**. UniFP receives the existing
estimator-only Reject force-command interface, filtering and reset semantics,
with compensation beta frozen to 1. All methods track nominal motion references;
no force-dependent tracking-reference shifts are introduced. Current Pos retains
UniFP gains; PACT position ablations retain PACT gains.

All registered B1Z1 ablations use the same dispatch. Attribution labels:

| Paper label | Implementation task |
|---|---|
| A1 | `b1z1_pact_ab6_position_film` |
| A3 | `b1z1_pact_ab8_coupled_film` |
| A6 | `b1z1_pact_ab9_representation` |

For matched controls, change only `--loading loaded` to `--loading unloaded` and
the output path. Motion references and feature-keyed initialization/mechanics/noise
streams are independent of loading and method call order. Dynamics and termination
times will naturally differ. `pair_id`, `motion_hash`, `profile_hash`, loading and
`scenario_hash` are saved in trial metadata. Compare initial state/mechanics hashes
as well; matching seeds alone does not prove cross-method mechanic equivalence.

Use 20 shared stable trial IDs per evaluation seed/checkpoint for attribution:

```bash
# Repeat with --loading unloaded and with each selected attribution task/checkpoint.
bash legged_gym/scripts/eval_b1z1_exp2.sh \
  --task b1z1_pact_ab6_position_film --run /path/to/ab6 --ckpt 28000 \
  --profile pulling --loading loaded --seed 0 --episodes 20 --num-envs 10 \
  --trial-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --gpu cuda:0 --headless --output-dir evaluation/exp2/A1/pulling/loaded/seed_0
```

IDs need not be contiguous; `--episodes` must equal their count. Subsetting or
reordering IDs does not change their keyed streams. `--seed` is an **evaluation**
seed, never a training seed. Three evaluation seeds of one checkpoint remain one
trained policy. Resolved checkpoint paths/hashes are saved; unknown training seeds
are explicitly unavailable, not inferred from the evaluation seed or run name.

## Timing, metrics and dataset

Each row stores `obs_t/history_t`, the deterministic action, interval-start
`commands_t/ee_reference_t_world`, executed interval statistics, and the physical
successor **before reset**. `command_nominal_t` and `ee_reference_sphere_t` identify
the nominal start target; `command_nominal`, `ee_reference_world` and
`ee_reference_sphere` are evaluated at the **successor timestamp** for scoring.
`commands_effective` retains native UniFP compensation channels. `task_stage_interval`
labels the applied interval; `task_stage_successor` labels its scored endpoint.
The load at an endpoint starts the next interval, never retroactively.

EE/velocity/yaw errors use valid pre-failure samples. Tracking tolerances are
0.10 m EE, 0.20 m/s planar velocity, 0.30 rad/s yaw rate. `S_track` uses all
prescribed control steps (ceil(horizon/dt)); failed tails contribute zero.
Stage S_track uses that stage's prescribed successor-step count, including failed
tails. Error means are undefined for stages without live samples. Horizon completion
is not physical failure, even though a coincident physical failure remains recorded.
Whole-trial failure censors event recovery, as in Experiment 1. Partial response
IAE is retained separately and never substituted for full-window IAE. Per-seed
summaries average trials equally, including failures. No across-training-seed
statistical claim is made by these outputs.

Files reuse Experiment 1's bounded `transitions-*.npz`, `records.json`, `metadata.json`,
`initial_states.jsonl`, `trials.json/csv`, `seed_summary.json`, `COMPLETE`/`ERROR.txt`.
Exp2 marks `experiment=2`; do not mix its summaries into Experiment 1 notebooks.
The recording schema contains actual shapes/dtypes; metadata describes units/frames.
Incomplete runs have no COMPLETE marker. Use a separate directory per checkpoint,
profile, loading status and evaluation seed.

## Experiment 3 support

All Experiment 1 native observations/history, available predictions, exact
runner-specific GRF-conditioning torque, raw/clipped/delayed actions, executed
position targets, PD/FF channels, mechanics, limits and substep violation counts
remain. Missing estimates are omitted and declared unavailable, never replaced by truth.

Additional fields include:

- Per-joint `pd_weighted_motor_mean_square`, `ff_weighted_motor_mean_square`,
  `executed_torque_mean_square` [(N m)²] and corresponding `_abs_peak` [N m].
- `opposing_channel_cancellation_mean` [N m]: mean of
  `(abs(PD)+abs(FF)-abs(PD+FF))/2` **before substep averaging**. Weighted PD+FF equals
  pre-clipping motor-scaled request; executed torque is separate (clip=1.1×rated).
- Raw per-foot force/position interval means, body net contact forces/contact
  fractions including **non-foot bodies**, and explicit articulation-to-sensor maps.
- Successor and interval body CoM positions/velocities; successor inertial-frame
  quaternions are **wxyz**, explicitly distinct from the usual recorded xyzw.
  Whole-body CoM velocity is mass-weighted body CoM linear velocity.
- Support, all-body-contact and EE-load moments [N m] calculated about the simultaneous
  whole-body CoM **at each physics substep**, then averaged. Support/contact moments
  approximate net forces at body link origins; contact points/free moments are
  unavailable. The imposed EE load acts at gripper body CoM. Do not recompute these
  moments from independently averaged positions and forces.
- Full per-trial body masses, inertias and local CoM poses in `initial_states.jsonl`,
  successor body rotations for momentum analysis, joint/body ordering and configured
  DOF-to-articulation indices, base pose, measured joints, executed desired joints,
  URDF/converted asset hashes. These support offline commanded-arm FK without BARD.

## Validation

Physical-limit seed frequencies inherit Experiment 1's pooled violation counts /
valid joint-substep exposure (null for zero exposure); tracking remains equal-trial.
To correct existing summaries without simulation, run
`python legged_gym/utils/b1z1_experiment.py /path/to/completed/exp2/run`.
Only `seed_summary.json` is replaced; stage summaries and raw recordings are preserved.

CPU tests:

```bash
python -m pytest tests/test_b1z1_experiment2.py -q
```

The optional smoke harness runs isolated 2-environment, 14-step time-compressed
sequences for each main method/profile, ab6/8/9 dispatch, an unloaded pair and a
one-step coincident failure/horizon. It checks actual loads, forbidden torso/EE
moments, torque accounting/statistics, finite data, terminal alignment, frozen
weights and repeated inference after decoder logging. Ablation cases use
architecture-compatible weights solely to test dispatch, not performance.

```bash
python tests/smoke_b1z1_experiment2.py \
  --pact-run /path/to/pact --pact-ckpt 70000 \
  --pos-run /path/to/pos --pos-ckpt 28000 \
  --unifp-run /path/to/original --unifp-ckpt 70000 \
  --gpu cuda:0 --output-dir /tmp/exp2-smoke
# Optional: --case pact_pushing (or pos_pulling, unifp_carrying, ab6, ab8, ab9, ...)
```

These short runs validate wiring only, not 28-second success or trajectory feasibility.
