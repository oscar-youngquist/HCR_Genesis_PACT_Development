# Bounded FACET formulation (version 2)

This is a FACET-specific training change, based on repository HEAD
`8525b6e88a95c7e428a872c30f2b5bf36aff34c1`. It requires fresh training.
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
attached setpoint yaw Jacobian: `(r_setpoint x -alpha*f_internal).z`.
Direct torso moments retain the verified actual application-point shift once.

`FACETReferenceCfg` exposes original spherical bounds, the original exclusion
box/ground plane, 0.20 m nominal-relative displacement cap, 0.04 m soft margin,
300 N/m boundary spring and 30 N s/m boundary damping. Soft force onset is C1.
Numerical guards project active constraints and remove outward velocity;
the moving displacement-cap boundary uses nominal-relative velocity.
Guard impulses are separately reported and transmitted to the virtual base.
Vertical ground reaction has no planar or yaw component.

Base speed/yaw rate limits are 2 m/s and 2 rad/s; relative EE speed is 1 m/s.
Capped base increments use bounded trapezoidal integration rather than
subtracting an uncapped velocity increment (which can reverse motion).
All 8/16/32-step history trajectories use the same integrator/guards.

**Geometry limitations:** these are EE point/segment command-space checks
against the task's exclusion box, ground and spherical workspace, not
articulated arm/leg collision detection or an IK reachability certificate.
Four fixed geometric projections handle intersecting constraints; incompatible
intersections are explicitly flagged and fall back to the already-rejected
nominal trajectory. Initialization/recovery projections may be discontinuous.
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
Teacher handoff is18,000*32 =24,000*24 steps. Save cadence scales by24/32.
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

## Completed checks (2026-11-16)

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
  [Machine-readable results](b1z1_facet_bounded_smoke.json) retain failures:
  3 deterministic and 4 stochastic during 1,024 transitions per evaluation.
  Peak EE/base loads were approximately4.8/56.5 N. The last training rollout
  took2.515 s, its update0.094 s; LR remained5e-4, entropy .002, std .999.
  Last KL was .100 and estimator privileged/force MSE .01265/.00186.
  This is not a converged or deployment-ready policy.
- [4096-environment reference-only benchmark](b1z1_facet_bounded_benchmark.json):
  legacy8.39 ms versus bounded53.90 ms per control interval, including all32
  history starts; peak allocated memory65.5 versus152.9 MB. Additional reference
  cost is approximately1.46 s per32-step rollout. Five measured intervals after
  warmup; this is not a full4096-environment training throughput benchmark.
- Relevant protected-task regressions: three failures reproduced identically
  in an untouched HEAD archive (Pos FiLM expectation, UniFP rejection ramp
  expectation, Pos/ablation config equality). Those files were not changed.
- Python syntax and `git diff --check` passed. No full training was launched.

Resolved smoke configuration/checkpoints remain in `/tmp/facet-bounded-delivery`.
The checked-in JSON files are compact reports, not model artifacts or raw data.
Full-range long-episode robustness, full-size training throughput and exhaustive
articulated collision safety remain unverified.
