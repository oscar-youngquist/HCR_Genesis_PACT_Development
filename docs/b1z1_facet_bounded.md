# Bounded FACET formulation (version 3)

This is a FACET-specific training change, based on repository HEAD
`bd0f71fe6a1f527935b0930a2a3faf7356b68e13`. It requires fresh training.
The launcher rejects older formulation checkpoints before constructing Isaac
Lab. Deployment of an existing checkpoint should use its original revision.
PILOT, its ablations, UniFP, the shared simulator, and their defaults are unchanged.

## Sources and adaptations

- [Paper, equations 8--9](https://arxiv.org/html/2505.06883v2): coupled virtual
  EE/base translation through an internal spring reaction.
- [Pinned released PPO](https://github.com/btx0424/active-adaptation/blob/c0d6f5aa6146814803090b5f6f19aafb8ce6f792/active_adaptation/learning/ppo/ppo_facet.py):
  Adam, fixed 5e-4 LR, entropy .002, 32 steps, four epochs/minibatches,
  gamma .99, lambda .95, clip .2, ordinary value MSE. Local recurrent reset,
  normalization, terminal bootstrap, EMA and one-time handoff fixes remain.
- [Released impedance implementation](https://github.com/btx0424/active-adaptation/blob/c0d6f5aa6146814803090b5f6f19aafb8ce6f792/projects/facet/facet_commands/impedance_manip.py):
  its EE physical-force application is commented out. Its gains are softer,
  damping is 1.8*sqrt(K), and its base mass sampler uses 2**randint(1,4),
  not the unused mass-range argument. It is not the complete coupled EE path.
- Local adaptations: bounded workspace reactions, yaw transmission, sustained
  EE pulling, conservative EE stiffness, mass-aware critical damping, and the
  existing nominal command interface rather than a new world-goal sampler.

## Reference conventions

`f_internal = K*(moving_nominal - ee_ref) + D*(nominal_velocity - ee_velocity)
+ f_workspace`. EE acceleration uses `F_ee + f_internal`; base XY acceleration
uses `f_base + F_base - alpha*f_internal_xy`. No second external EE force is
added to the base equation. No reference force is sent to PhysX.

World-frame integration uses heading-aligned horizontal workspace coordinates
and world vertical, with the existing mount/terrain height. Moving setpoint
velocity includes translation, yaw transport and nominal trajectory derivative.
Base/yaw commands remain velocities, not a fixed global heading target.
Optional `yaw_reaction=True` maps the transmitted internal force through the
reference-EE yaw Jacobian: `(r_ee_ref x -alpha*f_internal).z`. The impulse
guard uses the same reference-EE lever, rather than the moving nominal lever.
Direct torso moments retain the verified actual application-point shift once.

`FACETReferenceCfg` exposes original spherical bounds, the original exclusion
box/ground plane, 0.20 m nominal-relative displacement cap, 0.04 m soft margin,
300 N/m boundary spring and 30 N s/m boundary damping. Soft force onset is C1.
Numerical guards project active constraints and remove outward velocity;
the moving displacement-cap boundary uses nominal-relative velocity.
Guard impulses are separately reported and transmitted to the virtual base.
Vertical ground reaction has no planar or yaw component.

Base speed/yaw rate limits are 2 m/s and 2 rad/s.
`compliant_offset_speed_limit=1` m/s limits **additional** heading-frame motion
`arm_relative_velocity - nominal_relative_velocity`. It does not limit total
arm-relative or world EE velocity: the unchanged nominal sampler can exceed
2 m/s. Version 2's `ee_relative_speed_limit` is not a compatible resume setting.
Bounded substeps use semi-implicit accepted positions (`x_next=x_previous+dt*v_next`),
including base/yaw and EE motion, rather than a positional projection followed
by an unrelated velocity. Smooth acceleration is still evaluated with RK4.
All 8/16/32-step history trajectories use the same integrator/guards.

The constraint effective inverse mass is
`W=I/m_ee + alpha*diag(1,1,0)/m_base + alpha*c*c.T/I_yaw`, where
`c=z_cross_r_ee` and the yaw term is optional. A diagonal-plus-rank-one solve
uses `n.T W n` for each frictionless correction direction, producing
equal/opposite velocity impulses; EE velocity is not reconstructed
again from the changed base. For alpha=1, internal corrections conserve planar
momentum. EE=2 kg/base=4 kg and an incoming 1 m/s EE velocity give 1/3 m/s
for both bodies at a stationary outward-offset boundary. Speed saturation is
explicitly dissipative and reported separately from this internal impulse.

**Geometry limitations:** these are EE point/segment command-space checks
against the task's exclusion box, ground and spherical workspace, not
articulated arm/leg collision detection or an IK reachability certificate.
Four fixed coupled geometric sweeps handle intersecting constraints. Incompatible
intersections and initially infeasible measured starts are explicitly flagged;
there is **no teleport to nominal**. Recovery obeys the speed limit and can remain
outside the feasible region temporarily. Final geometry/rates are rechecked.
The guard is a numerical correction, not an exact constrained mechanical solve;
its impulse, saturation and incompatible fraction should be monitored.

## Sampling and budgets

Training defaults use `sampling_mode='bounded_b1z1'`: EE K in [60,120] N/m,
mass 2 kg; base K in [2,30], base mass in {2,4,8} kg; yaw K equals base K,
yaw inertia 1.5. D=2*zeta*sqrt(m*K), zeta=1, alpha=1. To calibrate K=90,
set `sampling_mode='fixed'`; the configured fixed EE damping is critical.
The reference RNG and force RNG are dedicated and do not consume command RNG.

`facet_forces` selects equal no-force/EE-only/base-only/both families per
eight-second cycle. Loads begin at 2 s, EE load lasts four seconds then releases.
Base pulses are triangular, .4--.6 s, signed horizontal component peaks40--200 N
and vertical peaks0--20 N. They are zero-order held per control interval, and
that exact held value feeds physics, references, and privileged packets.
EE direction is sampled in world axes with both signs; magnitude is sampled
stiffness times an equivalent displacement growing from .04 to .24 m over
76,800 control steps. Thus initially 2.4--4.8 N, finally14.4--28.8 N, not50 N
against a very soft spring. Boundary interaction is intentional late in training.
The inherited force samplers, velocity kicks and force-dependent standing
probability changes are bypassed/disabled for this configuration only.

The current branch's 27,500*24-step per-environment budget becomes20,625*32.
Teacher handoff is 16,500 completed updates, followed by 4,125 student updates
within the current 20,625-update total (80/20 interaction split). Adaptation is
disabled. Save cadence scales by 24/32. Explicit total-budget overrides require
an accompanying handoff override to retain this ratio; resumes retain saved
schedules unless explicitly overridden.
Existing reward/domain schedule readers receive the equivalent old24-step
iteration clock. Explicit CLI iteration overrides are **new32-step updates**;
passing an old70,000 update limit intentionally changes the budget.
No terrain curriculum or competing nominal/standstill/rejection reward is added.
Tracking rewards still average nonlinear rewards over valid horizons and get
exactly one dt scaling in the shared reward dispatcher.

## Validation commands

From this worktree, in the existing Isaac Lab environment:

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python -m pytest tests/test_b1z1_facet*.py -q
PYTHONPATH=.:tests OMP_NUM_THREADS=1 python tests/smoke_b1z1_facet_training.py --gpu cuda:0 --num-envs 8 --output-dir /tmp/facet-bounded-smoke
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python tests/benchmark_b1z1_facet_bounded.py --device cuda:0 --num-envs 4096 --output /tmp/facet-bounded-benchmark.json
```

The smoke uses six updates, early test-local combined loading/release, a3-update
handoff, checkpoint reload, then standing/walking deterministic and stochastic
student diagnostics. `smoke.json` contains nominal/reference errors, failure
counts, valid sample counts, force peaks, offset/speed/boundary metrics,
last-substep saturation (not interval saturation), PPO/estimator metrics and
timings. Evaluation interactions are separate from training accounting.
These are integration checks of a nearly untrained policy, not task performance.

The runner saves resolved settings, Git SHA, formulation version and interaction
budget. TensorBoard includes guard impulses, boundary/incompatible fractions,
reference speeds/offsets and per-episode time-averaged applied load norms.
No full run is recommended until these diagnostics and geometry approximations
are acceptable for the intended task.

## Version 2 historical checks (2026-11-16)

- FACET CPU suite: **146 passed, 2 skipped**. The CUDA export test was rerun
  unsandboxed and passed; the isolated upstream-module parity test was skipped
  because its pinned checkout fixture was unavailable. Source files were read
  directly to verify settings, not imported as a replacement training framework.
- Sensor-only packet, TorchScript action and hidden-state parity: CPU and GPU0
  both passed (2 tests). Existing gradient ownership, history/reset, probability
  replay, timeout, phase/adaptation/resume and configuration-isolation tests pass.
- GPU0 Isaac Lab: eight environments, three teacher plus three student updates,
  checkpoint reload, then deterministic/stochastic standing/walking under
  test-local combined loading with 1.2-second EE plateaus and release.
  Historical local report `b1z1_facet_bounded_smoke.json` retained failures:
  3 deterministic and 4 stochastic during 1,024 transitions per evaluation.
  Peak EE/base loads were approximately4.8/56.5 N. The last training rollout
  took2.515 s, its update0.094 s; LR remained5e-4, entropy .002, std .999.
  Last KL was .100 and estimator privileged/force MSE .01265/.00186.
  This is not a converged or deployment-ready policy.
- Historical local `b1z1_facet_bounded_benchmark.json` (4096 environments):
  legacy8.39 ms versus bounded53.90 ms per control interval, including all32
  history starts; peak allocated memory65.5 versus152.9 MB. Additional reference
  cost is approximately1.46 s per32-step rollout. Five measured intervals after
  warmup; this is not a full4096-environment training throughput benchmark.
- Relevant protected-task regressions: three failures reproduced identically
  in an untouched HEAD archive (Pos FiLM expectation, UniFP rejection ramp
  expectation, Pos/ablation config equality). Those files were not changed.
- Python syntax and `git diff --check` passed. No full training was launched.

Resolved smoke configuration/checkpoints remain in `/tmp/facet-bounded-delivery`.
The historical JSON names above refer to local reports that were ignored by git;
use the version 3 report below for portable validation evidence.
Full-range long-episode robustness, full-size training throughput and exhaustive
articulated collision safety remain unverified.

## Version 3 diagnostics and evaluation

`FACETHistory/age_{8,16,32}` reports reductions over **every collected control
interval**, including terminal snapshots. `eligible_count` counts age-valid
robot/interval targets, not unused ring slots. Constraint-valid/incompatible,
recovery, speed saturation, impulse magnitudes, offset and speeds use that same
denominator; incompatible targets are not silently removed from rewards or
statistics. `recovery` means the subtrajectory started the interval infeasible,
not a positional teleport. `constraint_correction_impulse_Ns` and
`constraint_saturation_impulse_Ns` distinguish internal and dissipative changes.
Boundary activity and speeds describe interval-end accepted targets; impulse
norms describe the sum over numerical substeps of that interval. Existing
`FACET/*` persistent-reference logs remain last-step diagnostics, not these
reward-history statistics. Getter calls never accumulate counters.

Scripted evaluation uses saved fixed parameters or separate represented
low/middle/high EE and base stiffnesses. Bounded sampling gives EE 60/90/120
and base/yaw 2/16/30, with saved zeta and correlated mass-aware damping.
The training force sampler is disabled for evaluation. Defaults include standing
and walking with none/EE/base/both loads, sustained K*offset EE forces and
represented triangular base pulses; 48 trials cover all levels/directions.
`--extrapolation` explicitly selects legacy sustained 20/35 N and yaw loads.
Finite actual loads are checked against scenario and reference inputs. Older
bounded artifacts must be evaluated with their original source revision.

`compile_constraints=True` fuses FACET-only fixed-shape reference kernels on
CUDA; CPU uses the same eager expressions. RK times are cached device scalars
to avoid a new compiled specialization per numerical stage. Compilation is a
warmup cost, not part of the reported steady-state timing. Use `--eager` in
the reference benchmark for comparison. These timings exclude simulation/PPO.

## Version 3 checks (2026-11-16)

- CPU FACET suite: **156 passed, 3 skipped**. The two CUDA-dependent tests were
  rerun unsandboxed: compiled/eager reference parity and CPU/GPU portable student
  parity **3 passed**. Isolated upstream module parity still needs its external
  checkout fixture. No protected task source/configuration was changed.
- [Warmed benchmark report](b1z1_facet_v3_benchmark.json), RTX 4090 / Torch 2.7:
  production yaw coupling, reference plus all 32 history starts:

  | Environments | Compiled ms/interval | Peak allocated MB |
  | --- | ---: | ---: |
  | 4,096 | 14.56 | 130.59 |
  | 10,240 | 24.33 | 323.94 |

  The same corrected 4,096-environment eager calculation took 74.82 ms:
  compilation gives a 5.14x warmed speedup without changing the expressions.
  The historical 53.90 ms report omitted yaw coupling and is not a freshly
  rerun baseline. New timings include diagnostics; warmup/compilation and source
  hashes are saved separately. These are not end-to-end training timings.

Reproduce the integration pipeline (six PPO/adaptation updates only):

```bash
PYTHONPATH=.:tests OMP_NUM_THREADS=1 python tests/smoke_b1z1_facet_pipeline.py \
  --gpu cuda:0 --output-dir /tmp/facet-v3-pipeline
```

This covers teacher/adaptation/student, resume at update 4, fresh-process
portable export, and 48 short standing/walking trials across no-force, EE-only,
base-only and combined conditions with release and represented impedance levels.
Production schedules and interaction budgets are not changed by this test.

The [version 3 smoke report](b1z1_facet_v3_smoke.json) records the successful
pipeline, exact source hashes, configuration and failures. Training interactions
were 384 per phase (1,152 total); evaluation simulated 7,200 transitions, including
inactive/padded rows. Six-environment final rollout/update took 1.319/0.099 s.
The 48 nearly untrained student trials had **16 physical failures**. Recovery
windows were incomplete, so recovery statistics are unavailable, not zero.
The final training rollout had incompatible fractions of approximately 39%/53%
at ages 8/16, associated with initially infeasible/recovering reference starts.
Those targets remain counted, and no performance benefit is claimed.

**Another full training run is not yet justified.** Inspect these history
diagnostics on a longer but bounded pilot run before spending the full budget.
The new tests establish the impulse, speed, timing and integration contracts;
they do not establish learned-policy quality or eliminate all incompatible
measured starts. No full training run was launched for this change.
