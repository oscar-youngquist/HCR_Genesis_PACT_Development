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

When enabled, the runner disables rigid-rejection/feedforward torque teachers,
force allocation, and the old fixed-target EE stability loss. The environment
disables impedance, arm-before-base/early-tilt, stand-still and nominal gait
reference/contact shaping objectives that conflict with base recruitment.
Generic effort, uprightness, contact safety and terrain clearance remain.
Physics-disabled ablations remain physics-disabled.

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
unsafe candidate retains the actual previously accepted reference; recovery
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
