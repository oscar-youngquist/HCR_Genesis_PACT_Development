# Signed B1Z1 Compliance

Opt in before constructing a PACT environment (including its shared position
and coupled ablations):

```python
env_cfg.compliance.enabled = True
# Episode-held directional profiles, not negative stiffness.
env_cfg.compliance.ee_profile_low = [0.0] * 6
env_cfg.compliance.ee_profile_high = [1.0] * 6
env_cfg.compliance.base_profile_low = [0.0] * 6
env_cfg.compliance.base_profile_high = [1.0] * 6
```

EE order is `[-x,+x,-y,+y,-z,+z]`; base order is
`[-x,+x,-y,+y,-yaw,+yaw]`. To disable base yielding, set both base profile
bounds to zero. Fixed profiles use identical lower and upper bounds.

### Episode Profile Sampling

`ComplianceCfg.sampling_mode = "mixture"` is the default. Profiles are sampled
independently per resetting robot and held for the episode. The initial weights
are coverage choices, not demonstrated optimal training settings:

| Diagnostic ID | Family | Weight |
| --- | --- | --- |
| 0 | Disturbance rejection (all zero) | 0.20 |
| 1 | Arm-only yielding (base zero) | 0.20 |
| 2 | Base-first yielding (EE zero) | 0.15 |
| 3 | Arm-to-base handoff | 0.20 |
| 4 | Directionally asymmetric interaction | 0.15 |
| 5 | Independent uniform coverage | 0.10 |
| 6 | Fully fixed profile (bounds identical) | Special case |

```python
env_cfg.compliance.sampling_mode = "mixture"  # Or "uniform" for the old sampler.
env_cfg.compliance.family_weights = [0.20, 0.20, 0.15, 0.20, 0.15, 0.10]
env_cfg.compliance.ee_axis_probability = [0.55, 0.55, 0.50]
env_cfg.compliance.base_axis_probability = [0.55, 0.55, 0.25]  # Independent yaw.
env_cfg.compliance.paired_direction_probability = 0.75
env_cfg.compliance.symmetric_gain_probability = 0.70
env_cfg.compliance.planar_gain_correlation_probability = 0.50
env_cfg.compliance.ee_active_range = [0.20, 1.0]
env_cfg.compliance.base_active_range = [0.20, 1.0]
```

Axis probabilities apply before conditioning on each family's constraints and
the configured coefficient bounds. Inactive directions are exactly zero.
Symmetric pairs share a draw from their intersecting ranges; correlated planar
gains share a quantile, without violating differing per-direction bounds.
Handoff matches signed horizontal EE/base activation but samples their gain
magnitudes separately. Vertical EE has no base-z counterpart. Yaw activation
is independent of translation, including yaw-only base-first profiles.

Half of asymmetric profiles initially use payload patterns: moderate EE planar
gains in `[0.20,0.60]`, EE `-z` in `[0.05,0.15]`, and `+z` in `[0.40,0.65]`.
`reverse_payload_probability=0.25` reverses this vertical pattern within payload
draws. Remaining asymmetric draws have independently sampled opposite gains
or one-sided activation. These are dimensionless profile coefficients, not forces.

Bounds always take precedence: a fixed nonzero coefficient cannot be made zero
to fit a family. Generic active magnitudes intersect the existing low/high bounds;
explicit fixed values are retained. Payload subfamilies additionally require
their stated magnitude ranges. Incompatible patterns/subfamilies are excluded;
warnings describe exclusions and valid weights are renormalized. No compatible
positive-weight family is a configuration error, not an implicit uniform fallback.
Fully fixed profiles bypass the mixture. There is no sampling curriculum.

The actor/history still receives only the 12 actual coefficients and nominal
commands. Family IDs live in `reference.family_ids` and reset-event
`extras["compliance_sampling"]`; they are not conditioning inputs. TensorBoard
episode diagnostics include `compliance_sampling/<family>_fraction`, effective
weights, zero-coefficient fraction, EE/base means, yaw-disabled fraction and
draw count. These describe newly sampled episodes, not completed-episode success.

Only reset-time sampling changes. Existing filters, gates, rewards, force
application, and actor-PINN definitions are unchanged. Old checkpoints with the
12-coefficient observation contract remain dimensionally compatible; this is
improved task coverage, not evidence of improved trained performance.

Focused validation (no training):
```bash
SIMULATOR=isaaclab PYTHONPATH=. python -m pytest -q tests/test_b1z1_compliance_sampling.py
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python tests/smoke_b1z1_compliance.py --profile-scenarios --task b1z1_pact --gpu cuda:0 --headless
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab PYTHONPATH=. python tests/benchmark_b1z1_compliance_sampling.py
```

Validation on GPU 0 (RTX 4090): 72 focused sampling/reference/actor/BARD tests
passed. The eight-robot, six-decision Isaac Lab diagnostic passed force-application,
finite/bounded-target, independent yaw, lateral-base, and removal checks. Existing
feasibility safeguards held EE offsets near zero at the initialized pose, so this
smoke does not demonstrate arm yielding or learned payload behavior. Those
reference responses, including reversed vertical asymmetry, are unit-tested in
an interior feasible workspace. No training was run.

Reset-only benchmark, 1,000 warmed repeats (filter clearing and profile assignment,
excluding simulator, geometry and scalar logging):

| Resetting robots | Uniform | Mixture |
| --- | --- | --- |
| 64 | 0.055 ms | 0.220 ms |
| 4096 | 0.054 ms | 0.234 ms |

4096-robot mixture scratch peaked at 1.50 MiB versus 0.38 MiB for uniform.
Pattern construction is one-time; sampling adds no work to non-reset steps.

`ee_gain` is m/N; `base_gain` is (m/s)/N, (m/s)/N, (rad/s)/(Nm).
The gate switches and physical margins live in `ComplianceCfg` in
`legged_gym/envs/b1z1/directional_compliance.py`.
Objective settings live in `b1z1_pact_config.py`: `rewards.scales.compliance_ee_velocity_weight`
and `rewards.compliance_ee_velocity_sigma` default to 0.5 and 0.25 (m/s)^2;
`algorithm.compliance_ee_velocity_weight` and `algorithm.compliance_ee_velocity_scale`
default to 0.25 and 0.5 m/s. Ablations inherit these settings. They only affect
compliance-enabled objectives; the reference generator has no reward/loss settings.

## References and Timing

- XY is heading-aligned, EE horizontal position is base-relative, and EE Z
  is world height. EE orientation never rotates these compliance axes.
- Filters advance once after force-event sampling and before simulation.
  They use actually applied, ramped EE/torso disturbances. Physical forces and
  couples are still applied once through the unchanged simulator interface.
- EE integration is the exact critically damped held-input solution. Nominal
  trajectory derivatives use the existing quintic trajectory. Base/yaw
  transport is included when comparing predicted or measured world velocities.
  Terrain-center height is held over each control interval.
- Force removal preserves filter state and returns toward nominal targets.
  Base velocity corrections decay without position/heading restoration.
- Three axis probes and a combined yielding probe test directional workspace
  and joint capability; short configuration probes check collision clearance.
  Gates inspect measured and reference workspace margins. Bounds are enforced
  independently on equilibrium and integrated references, with outward
  velocity removal and state feedback for anti-windup.
- PPO receives cloned interval targets, including terminal transitions;
  replay does not advance filters. Generated references are detached; actor
  gradients still flow through predicted dynamics, FK, and EE velocity.

### Direct Torso Disturbances

```python
env_cfg.compliance.include_torso_disturbances = True  # Default for differential compliance.
# False retains the prior EE-only REFERENCE, not EE-only physical disturbances.
# Optional episode stream eligibility; None retains the existing independent schedules.
env_cfg.commands.compliance_disturbance_mode_weights = [1., 1., 1.]
# Order: EE-only, torso-only (force + moment), simultaneous-enabled.
```

In heading-aligned XY/world Z, about the base link origin:
`w_ref = [F_B.x, F_B.y, tau_B.z] + [F_E_gated.x, F_E_gated.y, (r_BE x F_E_gated).z]`.
The arm gates apply only to the EE contribution. Direct torso loading bypasses
those gates. Signed base gains are applied **after summation**, then the existing
first-order velocity/yaw-rate filter, command limits and anti-windup operate.
Opposing loads can cancel before selecting asymmetric gains. Zero base gains
still request nominal motion. EE offset generation remains EE-force-only.
Vertical torso forces and roll/pitch couples remain physical disturbances but
do not directly generate planar reference commands. Release decays to nominal
velocity without heading/position restoration.

The verified Isaac Lab source reads the same held buffers and enable flags as
`_apply_external_forces`, after the event ramps/masks. Installed PhysX's
`apply_forces_and_torques_at_position(position_data=None)` applies force at the
link transform, and the installed wrench composer adds no point moment when
positions are omitted. Therefore the torso force acts at its link origin, not
at the randomized CoM. The adapter supplies that actual world point and free
couple. `torso_wrench_heading` shifts the moment once by `(point-base) x force`.
A source already providing a base-referenced moment must supply the base as its
reference point; otherwise the point moment would be counted twice. No gravity,
mass mismatch, EE load, GRF, or instantaneous velocity kick enters this source.
Other simulator backends need this verified source accessor or must use the
EE-only compatibility switch; missing source semantics raise an explicit error.

Coverage selection is independent of the compliance profile and held per episode.
It masks only the excluded streams' actual outputs, before simulator application
and labels. Selected streams retain the original ranges, curriculum scales,
ramp/hold durations, probabilities and timers. "Simultaneous" enables both
schedulers; overlap still depends on their event timing. It does not guarantee
continuous loading, bypass a zero force curriculum, or silently enlarge forces.
Neither disturbance-mode IDs nor privileged wrenches become actor observations.

The existing time-aligned target snapshot already feeds rewards and actor-PINN
rollout objectives; no replay-side integration or additional actor input is added.
`Reference.torso_wrench`, `.ee_wrench`, and `.combined_wrench` expose detached
interval-start planar contributions. Episode TensorBoard diagnostics are
`compliance_wrench/{torso,ee_gated,combined}/{fx_N,fy_N,tz_Nm}`: signed control-step
means accumulated before terminal reset, plus `interval_count`. Optional coverage
logging is `compliance_disturbance/{ee_only,torso_only,simultaneous}_fraction`.
Base forces/couples are never synthesized from the mapped EE reference wrench.

`Episode/EE/applied_force_offset_norm_mean` retains the episode-average accepted
EE target displacement norm in meters. `Episode/EE/applied_force_offset_norm_max`
reports the mean of completed episodes' temporal maxima, captured before reset.
Both measure accepted target minus nominal target, not measured tracking error.
The runner retains its existing averaging across episode-reset batches.

Validation commands (GPU 0, no learning):
```bash
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab PYTHONPATH=. python -m pytest -q tests/test_b1z1_torso_compliance.py tests/test_b1z1_compliance_sampling.py tests/test_b1z1_directional_compliance.py tests/test_b1z1_actor_physics.py tests/test_b1z1_bard_dynamics.py
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python tests/smoke_b1z1_torso_compliance.py --task b1z1_pact --gpu cuda:0 --headless
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python tests/smoke_b1z1_torso_compliance.py --task b1z1_pact_ab7_position_full --gpu cuda:0 --headless
```
Off-center shifts and equivalent already-shifted moments are tested analytically;
the current production simulator applies torso forces at the link origin, so
off-center physical application is not claimed as simulation-tested. These
checks validate reference construction, not learned locomotion/compliance quality.
Verified on GPU 0: 80 focused tests passed (one dependency deprecation warning),
and both full PACT and ablation 7 passed eight-environment, six-decision Isaac Lab
smokes with exact physical-buffer accounting and detached interval targets.
No policy training or performance evaluation was run.

## Observations and Objectives

The actor/history receives 12 episode profile values before the final six
**nominal** command slots. It does not receive the privileged applied force,
compliant target, or filter state. The critic/reconstruction layout includes
the profiles. Enabling the feature therefore needs a newly trained compatible
checkpoint; disabled configurations retain the original widths and behavior.

EE position/velocity and planar velocity/yaw rewards use compliant references.
The actor rollout tracks the same references, and commanded-position FK uses
the captured compliant goal. Safety penalties and randomized BARD mechanics
remain unchanged. Force estimator labels are not altered.

When enabled, the runner disables rigid-rejection and base-balance torque teachers,
force allocation, and the old fixed-target EE stability loss. The environment
disables impedance, arm-before-base/early-tilt, stand-still and nominal gait
reference/contact shaping objectives that conflict with base recruitment.
Generic effort, uprightness, contact safety and terrain clearance remain.
Physics-disabled ablations remain physics-disabled.

### Optional Local Arm Feedforward Teacher

The runner no longer forces `actor_phys_arm_feedforward_enabled=False`.
The separate `tracking_arm_feedforward` RL reward remains disabled by compliance.
In the shared PACT **algorithm** config:

```python
actor_phys_arm_feedforward_enabled = True
actor_phys_arm_feedforward_compliant_local_enabled = True  # default False
```

With the existing enable flag off (or weight zero), neither teacher runs. With
only that flag on, the original load-compensation equations remain unchanged,
including in compliant training. With both flags on, compliant coupled policies
use the local teacher exclusively. Noncompliant policies retain the original
teacher; position-only and physics-disabled ablations remain excluded. The
existing weight, overall coefficient, warmup and selected-row sampling apply.
In particular, removing the old override makes the existing default-on original
teacher active in compliant PACT unless it is explicitly disabled.

The new label is `M_aa a_arm* + M_au a_other + h_a - J_arm^T F_applied - tau_PD`.
All quantities, including the candidate action's effective PD torque and the
existing rollout's non-arm acceleration, are detached. Only the effective arm
FF output receives direct loss gradients. Randomized gains, motor strength and
branch weights come from the same controller reconstruction used by the actor
rollout. Torso loads enter through non-arm coupling, not another EE force term.
This is local inverse-dynamics guidance conditional on predicted non-arm
acceleration, not a feasible whole-body solution or an execution controller.

The desired task acceleration uses `ComplianceCfg.omega/zeta`, accepted position
and velocity at interval start, and accepted endpoint velocity change divided
by the control dt. The endpoint position remains the FK/tracking target. These
detached snapshots, including the actual post-sampling EE force, are captured
before physics and survive terminal reset/shuffled replay. First intervals and
reference reinitializations (including the following interval) are masked.
Workspace/collision holds therefore supply braking, not stale nominal motion.
No compliance filter advances in PPO replay.

`H` includes heading-relative horizontal translation/yaw transport and world-z.
World floating-base tangent columns are explicit; BARD joint columns are reused.
The cached velocity-dependent bias uses a centred **1 ms kinematic derivative**
with constant world generalized velocity. This is a numerical approximation,
tested against independent FK differences, not a second dynamics rollout.
Its accuracy for extreme angular rates remains a limitation. DLS uses the
existing `dls_damping` in Cartesian metres/radian. Joint-limit margins and
one-control-interval position/velocity predictions reject infeasible rows;
FF action-range or combined execution-limit saturation also excludes a label.
No clipped/damped solution is claimed to achieve exact task acceleration.

`ActorPhysics/arm_feedforward_local_*` reports invalid, saturated and infeasible
fractions. With additional diagnostics enabled it also reports acceleration
residual, inertial/PD norms and PD-FF opposition; existing teacher component
norms are retained. The local label can correct PD errors, so FF now has a
broader role even while the position-action FK loss remains active.

Reproducible checks (use the Isaac Lab environment and repository PYTHONPATH):

```bash
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab PYTHONPATH=. python -m pytest -q tests/test_b1z1_local_feedforward.py tests/test_b1z1_actor_feedforward.py
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab PYTHONPATH=. python tests/benchmark_b1z1_local_feedforward.py --count 4096
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python tests/smoke_b1z1_local_feedforward.py --task b1z1_pact --gpu cuda:0 --headless
```

The microbenchmark isolates shared mechanics, extra cached kinematics, and
loss/backward on selected rows. It is not end-to-end training throughput and
does not establish improved learned-policy performance.

Validation on an RTX 4090 (`cuda:0`): 197 focused tests passed, including
articulated BARD finite differences with nonidentity base orientation. A
two-environment/four-step Isaac Lab PPO update produced a finite local loss
(`raw=0.8983`) with 12.5% valid teacher rows; the untrained random policy caused
87.5% saturated/invalid labels, correctly excluded. This is integration
verification, not an estimate of rejection frequency in a trained policy.
The smoke harness uses single-worker Inductor compilation to avoid a post-Kit
fork-pool stall; production compiler settings are unchanged.

At 4096 synthetic selected states using randomized BARD mechanics, ten warmed
repetitions measured shared mechanics at 9.07 ms; disabled/original/local
loss+backward at 0.24/0.87/2.32 ms; extra local kinematic preparation at 9.57 ms
per cache build. These use a common cached acceleration and controlled test
gains, not measured 4096-environment simulation or full PPO throughput. The
one-millisecond bias approximation, contact/model errors and predicted non-arm
acceleration remain sources of teacher error. Long training was not run.

## Verification Commands

Use the installed Isaac Lab Python environment, from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab python -m pytest -q \
  tests/test_b1z1_directional_compliance.py tests/test_b1z1_actor_physics.py \
  tests/test_b1z1_bard_dynamics.py
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python -u \
  tests/smoke_b1z1_compliance.py --task b1z1_pact --gpu cuda:0 --headless
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python -u \
  tests/smoke_b1z1_compliance.py --task b1z1_pact_ab7_position_full --gpu cuda:0 --headless
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab PYTHONPATH=. python \
  tests/benchmark_b1z1_compliance.py --counts 128 4096 --compile-geometry --compile-full-fk
```

Smoke tests use two robots, six control decisions, deterministic lateral load
and removal, and no learning. The microbenchmark includes URDF FK/Jacobian,
all recruitment gates and fixed configuration feasibility probes; it excludes
simulation, policy inference and optimizer work. Collision scratch batches
are bounded by `collision_batch_size`.

## Runtime Controls

`compile_geometry = True` is the inherited default. On CUDA this compiles the
pure FK and collision kernels with the installed PyTorch compiler; CPU stays
eager. No simulator calls, random sampling, or stateful reference filters are
inside the compiled functions. Compilation is lazy and adds startup warmup;
steady-state benchmarks exclude it. Set the flag to `False` for compiler
troubleshooting or eager numerical comparisons. Compiler failures are not
silently swallowed. Disabled compliance constructs no geometry kernels.

The default `collision_batch_size = 8192` bounds flattened probe scratch, not
the number of environments or safety probes. On the tested RTX 4090, increasing
this to 16384 saved only about 1 ms in eager execution while increasing peak
reference-only Torch allocation from 136 to 245 MiB. Reducing to 2048 uses less
memory but increases launch overhead. All configurations retain the same
collision samples, margins, exclusions, update rate and trial coverage.

Reproduce disabled/enabled rollout comparisons in separate processes (same
seed and actions; no learning). These include the configured reward differences
and **exclude** policy inference, rollout storage and PPO/actor-PINN backward:

```bash
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python -u \
  tests/benchmark_b1z1_compliance_sim.py --task b1z1_pact --gpu cuda:0 \
  --num_envs 4096 --steps 16 --output /tmp/compliance-disabled.json
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python -u \
  tests/benchmark_b1z1_compliance_sim.py --task b1z1_pact --gpu cuda:0 \
  --num_envs 4096 --steps 16 --compliance --output /tmp/compliance-enabled.json
```

The disabled benchmark attaches the required BARD backend for its existing
arm-feedforward reward directly, without constructing a training optimizer.
Profile sampling consumes additional RNG in enabled runs, so these are matched
workloads, not guaranteed pointwise identical trajectories. JSON includes
phase timings, reset counts, terrain, and peak Torch memory (not total PhysX
or process VRAM). Do not interpret rollout overhead as full-update overhead.

### Measured GPU 0 Results

2026-11-05, RTX 4090, installed `lr_lab_cupiqp` environment. Reference-only
results use 3 warmup calls and 10 measured calls at 4096 environments:

| Reference implementation | ms/control interval | Peak Torch MiB |
| --- | ---: | ---: |
| Initial full-body FK per probe, 2048 batch | 125.33 | 46.79 |
| Cached legs, arm-only probes, 2048 batch | 38.48 | 54.79 |
| Cached legs, arm-only probes, 8192 batch | 18.17 | 136.31 |
| Compiled probe kernels, 8192 batch | 10.71 | 107.40 |
| Compiled FK and probes (default), 8192 batch | 9.59 | 108.54 |

Separate Isaac Lab runs used the current trimesh terrain, seed 1, 4096 robots,
zero actions, 4 warmup and 16 measured steps; all had zero measured resets:

| Configuration | Total ms/step | Reference ms/step | Peak Torch MiB |
| --- | ---: | ---: | ---: |
| Compliance disabled | 76.89 | 0 | 741.23 |
| Optimized eager compliance | 92.58 | 21.90 | 592.38 |
| Compiled compliance (default) | 80.55 | 13.39 | 572.21 |

This short sample gives approximately 4.8% rollout overhead, or 0.088 seconds
over 24 control steps, **not** a measured full PPO iteration increase. Enabled
compliance removes conflicting torque/gait rewards, explaining why net
overhead is less than reference cost and peak allocation is lower. Compiler
warmup, PhysX memory, policy inference, storage, and PPO backward are excluded.
No safety probes were removed to obtain the speedup. Longer trained-policy
and complete-update throughput measurements remain unverified.

## Limitations

Collision guidance is **EE-only**, not full-arm collision protection. A sphere
(`ee_radius=0.025` m) is checked against the URDF torso box, leg capsules and
nonadjacent arm capsules. `ee_excluded_arm_segments=[4,5]` excludes the Z1's
own terminal segment and wrist attachment neighbour. Analytic sphere/segment
and sphere/box distances have no sampling-spacing inflation. Each obstacle's
approach/retreat is checked separately before combining gates. Fixed-count
joint probes and a swept EE sphere over the complete requested Cartesian path
are both checked; a short joint probe alone cannot certify a farther target.
Obstacle geometry is frozen at the measured pose for the Cartesian sweep;
configuration probes separately approximate its motion. These are primitive
checks, not mesh certification, iterative IK or whole-body feasibility. An
unsafe candidate is shortened to a checked safe prefix (held only if no safe
prefix is available); recovery
from an overlap is allowed only when no obstacle's penetration worsens. This does not
certify reachability, nor guarantee collision-free execution or prevent the
learned robot from overshooting. General effort/damping penalties still trade
off against motion. These tests validate reference generation and integration,
not improved task performance of a trained compliant policy.

## Review Corrections

Workspace bounds now enumerate sphere/cone/azimuth-plane intersections and
clip at the first exit of the requested Cartesian segment, including the
inner-radius exclusion. Clipping shortens displacement; it cannot redirect
a lateral-only command into x/z. Outward velocity is stopped rather than
rotated along a boundary. Equilibrium and integrated state both use these
bounds, in addition to the original torso-box/underground exclusion.

The generator stores the accepted mixed-coordinate reference across control
intervals. Rejection holds that position and sets its mixed-coordinate
velocity to zero, even while the nominal trajectory moves. Anti-windup writes
`delta=accepted-current_nominal` and `delta_dot=accepted_velocity-nominal_velocity`.
World conversion still adds base XY translation and yaw transport. Reset
clears only the resetting rows; the next interval initializes from the actual
interval-start nominal. Replay copies detached targets without advancing filters.

Actor and representation PINNs now share `b1z1_wrench.about_base`. Rollout
storage retains `torso_lever_world`, captured before integration/reset. The
world free couple shifts once by `lever x force`; a base-referenced moment
uses a zero lever. Predicted loads remain predicted, and randomized mass/CoM
remain in inertia/bias only. Old packets without geometry explicitly use the
existing origin-applied convention. No observation or checkpoint dimensions
change. A 35 N forward origin-applied force therefore has zero yaw moment even
with 0.15 m lateral CoM randomization; a genuinely offset 0.15 m application
point instead gives -5.25 Nm.
Other live backends must expose a verified source accessor; missing application
semantics now fail explicitly rather than guessing a randomized CoM lever.

Focused regressions are in `tests/test_b1z1_compliance_review.py`. Reproduce
the GPU-0 yielding/obstacle-handoff smoke without a policy or learning:

```bash
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. \
  python tests/smoke_b1z1_compliance.py --review-collision \
  --task b1z1_pact --gpu cuda:0 --headless
```

The diagnostic load aimed at a nonadjacent arm segment is test-only. It does
not modify the production disturbance distribution or demonstrate learned
collision avoidance. Use `--task b1z1_pact_ab7_position_full` for position mode.

GPU-0 review benchmark (RTX 4090, 4,096 rows, 10 measured intervals, identical
fixed poses/loads, compiled FK and geometry): pre-change 9.76 ms/interval,
revised 9.47 ms/interval. Peak Torch allocation fell from 109.37 to 36.07 MiB.
This is approximately unchanged runtime (3% faster in this short measurement),
with 67% less scratch memory, despite the added full-path tests. Revised cold
compiler warmup was 26.2 s and is excluded from steady-state timing. This is
the reference-only microbenchmark above, not full-training throughput; simulator,
reward, PPO and BARD-backward overhead are not measured by it.

Review verification: 149 focused tests passed (CPU and GPU 0), including BARD
integration, actor gradients, detached application geometry, aligned replay,
profile sampling and partial resets. Syntax and whitespace checks passed.
Isaac Lab GPU-0 smoke checks passed with eight environments and six decisions
each: PACT profile coverage, PACT obstacle approach, position-full ab7 obstacle
approach, and PACT direct/combined torso-wrench cases. Default-pose EE clearance
is +0.1752 m with these primitives. No training or trained-policy performance
evaluation was run, and full-training throughput remains unmeasured.

## Moving Workspace And Progressive Handoff

Every update revalidates the accepted reference against the **current** workspace.
When terrain height moves the workspace around it, ordinary first-exit clipping
is replaced by explicit reference reinitialization (`reference.recovering`).
The preferred anchor is the current nominal command. Old compliance offsets
are retained only on enabled Cartesian axes and shortened along their original
direction to the first workspace/collision limit. No spherical redirection is
used. A zero-profile, unloaded reference follows a safe nominal command directly.

Recovery anchors must also satisfy EE geometry and the original torso/underground
exclusions. If nominal is invalid, the deterministic fallback order is previous
accepted point, measured EE, then spherical-workspace midpoint. These exceptional
anchors can change otherwise disabled components. If none is feasible, the code
raises an explicit error requesting an environment reset rather than silently
retaining an invalid reference. A recovery jump is not smooth motion: stale
filter velocity is discarded; exact nominal recovery resumes nominal velocity,
otherwise the recovered task reference is held for that interval. Accepted
position, delta and rate are synchronized before the next update.

Collision recruitment uses measured/accepted clearance and short directional
probes, not the full force-equilibrium distance. Hard travel limiting still
checks the complete swept path, per obstacle, plus the joint probes. An unsafe
equilibrium is clipped using four fixed batched prefix refinements; the filter
may advance toward that safe equilibrium instead of freezing immediately.
Every retained prefix passes the swept-path predicate (resolution 1/16 of the
requested segment); the all-safe case returns without refinement. Clipping an
actual filter update zeros its velocity and feeds the accepted state back for
anti-windup. Base-disabled profiles still request no base correction.

Numerical regressions on the current URDF:

- Raising workspace/nominal height by 0.01 m changes the stale radius from
  0.900000 to 0.904837 m; recovery restores 0.8999999 m, with zero stale rate.
- The tested path starts with 0.175200 m EE clearance and still has 0.148180 m
  clearance over its first tenth. First-step yielding is 0.001158 m with zero
  collision recruitment. After 90 further reference updates, yielding is
  0.174939 m, clearance remains +0.000261 m, and recruitment reaches 1.0.
- Force reversal/removal, base-disabled profiles, collision-aware recovery
  fallbacks, inner-radius clipping, partial resets and detached replay are tested.

GPU-0 reference microbenchmark at 4,096 rows / 10 measured intervals: standard
case 10.83 ms and 36.07 MiB peak Torch allocation; all-rows obstructed-equilibrium
stress case 23.44 ms and 36.62 MiB. The stress case intentionally exercises the
bounded refinements. These are reference-only timings, not training throughput.
Reproduce the stress case with `tests/benchmark_b1z1_compliance.py --counts 4096
--repeats 10 --compile-geometry --compile-full-fk --collision-limited`.

Optional compact safe-prefix refinement gathers only the initially unsafe
environments after the full-batch first check. Their measured geometry is
gathered once and reused for all four refinements; safe rows retain their candidate exactly. Four refinements
resolve 1/16 of the requested displacement; collision predicates are unchanged.
The benchmark accepts
`--blocked-fraction 0.01` and `--full-batch-refinement` for matched comparisons.
Before reducing eight refinements to four, on GPU 0 (RTX 4090), 4,096 rows
and 30 warmed intervals with 1% obstructed
demands measured 23.13 ms/full-batch versus 21.66 ms/compact. This roughly 6%
reference-only improvement is not a whole-training speedup; fixed launch costs
remain, and first-use compilation for compact shapes is excluded.

### Compiled Reference Kernels

With `compile_geometry=True`, the complete target-feasibility predicate and
heading-frame handoff paths now run as fused kernels. Directional/headroom
probes and workspace gates are compiled too, and the interval's initial
collision clearance is reused. The original FK matrix arithmetic, geometry,
three configuration probes, four clipping refinements, and exclusions remain.

The default `collision_compact_refinement=False` runs every refinement on the
full environment batch, split into fixed bounded geometry chunks. An all-safe
batch still returns immediately. Safe candidates retain their exact value;
only unsafe candidates are shortened. Torch compilation stays enabled, but
the target predicate uses ordinary compiled execution without CUDA-graph
capture. Thus changing the number of unsafe rows cannot introduce a new
compiled shape or graph capture. The rare recovery-only point-check signature
is also queried on the first interval, without advancing filters or RNG.

Set `collision_compact_refinement=True` to restore the previous faster compact
path, including CUDA-graph replay. Selected rows are bucketed to
powers of two, capped by `collision_batch_size // collision_probe_steps`;
padding duplicates only the last selected row and is discarded afterward.
This bounds graph shapes instead of caching every possible unsafe-row count.
Each boolean result is copied before subsequent replays can reuse its storage.
`compile_geometry=False` retains the eager path. First-use compilation/capture
costs more than warmed execution and may recur when a new bucket is encountered.

Historical matched GPU-0 RTX 4090 reference-only measurements (4096 environments,
30 warmed intervals, four refinements, compact refinement enabled):

| Obstructed demand fraction | Previous kernels | Optimized kernels |
| --- | --- | --- |
| 1% | 15.66 ms | 8.30 ms |
| 100% | 17.46 ms | 9.92 ms |

Peak Torch allocation in these isolated benchmarks increased from about 36.6
to 66.3 MiB. These are not PPO/full-training timings. Reproduce with
`tests/benchmark_b1z1_compliance.py --counts 4096 --repeats 30 --compile-geometry
--compile-full-fk --collision-limited --blocked-fraction 0.01 --compact-refinement`; add
`--legacy-kernels` for the unfused comparison. `--profile-output /tmp/trace.json`
exports a warmed CPU/CUDA trace, but its instrumentation changes the timings.
Use `SIMULATOR=isaaclab`, `PYTHONPATH=.`, `CUDA_VISIBLE_DEVICES=0`, and the active
Isaac Lab Python environment; `TORCHINDUCTOR_COMPILE_THREADS=1` avoids forking
compiler workers after simulator startup.

For the latency-oriented default, omit `--compact-refinement`. Add
`--vary-blocked-fraction` to exercise changing obstructed populations and report
median/p95/max interval latency plus new compilation counts after warmup.
This is not a hard real-time guarantee: other processes (notably concurrent
Genesis initialization), simulation, and PPO still contribute to iteration
time. Changes take effect only in a newly started process.

The fixed-batch default measured 16.50 ms/control interval (p95 16.89, max
17.17 ms) on GPU 0 at 4096 environments with 1% obstructed demands, versus
10.29 ms for the prior full-batch CUDA-graph path. Changing the obstructed
population between zero, one, 1%, 10%, 50%, and all environments gave 15.18 ms
mean, 16.61 ms p95, 16.69 ms maximum and **zero new compilations** after
warmup (30 intervals). These short reference-only checks ran alongside the
existing jobs; no complete-training latency guarantee was measured.

Earlier compact-kernel validation: 68 focused CPU/CUDA tests passed, including unfused predicate and
handoff parity, bounded buckets, partial resets, reference recovery and replay.
The GPU-0 Isaac Lab `--review-collision` smoke passed with eight robots and six
control decisions, including load removal and obstacle-triggered handoff. No
learning or full-training throughput experiment was run for this optimization.

The updated `--review-collision` smoke uses real Isaac Lab robot geometry but
explicitly initializes a near-obstacle **reference** to test recruitment, and
changes a terrain-height **measurement** to test recovery. Neither is a claim
that an untrained/learned robot executed that approach or traversed a terrain
step. No training or new task-performance evaluation is involved.

Verification for these remaining fixes: 154 focused tests passed, including
the retained sampling, wrench, BARD, actor-gradient and replay tests. GPU-0
Isaac Lab smoke passed for PACT (approach/release/terrain measurement) and ab7
(also explicit near-reference handoff), eight environments and six decisions
each. Syntax and whitespace checks passed. Long-horizon physical approach,
learned compliance performance and full-training throughput remain unverified.

### Compliant Velocity Loss Logging

`ActorPhysics/ee_velocity` remains the raw, normalized Huber loss, not an
EE speed in m/s. `ActorPhysics/ee_velocity_scaled` uses the same update
averaging and multiplies it by `compliance_ee_velocity_weight` and
`ActorPhysics/scheduled_coefficient` (overall actor coefficient times PINN
ramp). It is the scalar objective contribution before gradient projection
or norm balancing. Logging does not change loss/gradient computation.

The fixed-batch and scaled-log changes passed 70 focused CPU/CUDA-0 tests,
including kernel parity, changing unsafe populations, one-time recovery warmup,
reset/replay behavior, inherited defaults, scalar schedule factors and unchanged
actor gradients. Syntax and changed-file whitespace checks passed. No new
simulation or full-training run was launched for these latency/logging changes.
