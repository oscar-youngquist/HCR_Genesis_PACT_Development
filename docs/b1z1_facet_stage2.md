# FACET Stage 2: Coupled References Only

This documents the diagnostic-only mode. [Stage 3](b1z1_facet_stage3.md) now
defaults to multi-start tracking rewards; set `facet_tracking.enabled=False`
to retain the Stage 2 behavior described here.

`b1z1_facet` now advances a detached virtual base/EE reference in parallel with
the unchanged Stage 1 Pos task. **These outputs are not actor inputs, reward
targets, PINN targets, or a physical controller.** Stage 3 histories/rewards and
FACET architecture/teacher-student learning are not implemented.

## Equations and Sources

Following [FACET equations (8--9)](https://arxiv.org/html/2505.06883v2):

```
mb * base_acc_xy = spring_base_xy + applied_base_force_xy - alpha * spring_ee_xy
me * ee_acc_world = spring_ee_world + applied_ee_force_world
Iy * yaw_acc = spring_yaw + applied_moment_about_base_z
```

Springs use nonnegative stiffness/damping, positive virtual masses/inertia, and
`alpha` in `[0,1]`. Alpha affects only the virtual EE spring reaction. There is
no EE-to-yaw lever term in this stage, no physical force gating/application, and
no workspace/collision/recruitment constraint on the virtual references.

The [released B1Z1 reference](https://github.com/btx0424/active-adaptation/blob/main/projects/facet/facet_commands/impedance_manip.py)
was inspected at commit `c0d6f5aa6146814803090b5f6f19aafb8ce6f792`.
Its `reference_dynamics` omits applied EE force and transmitted EE spring force;
we include both from the paper. We do not copy its fixed 0.5 m height, new goal
sampler, velocity clamp, or future-reference history mechanism.

## Configuration

Each `B1Z1FACETCfg` owns an independent `facet_reference` section:

| Setting | Default | Units / origin |
| --- | ---: | --- |
| `enabled` | `True` | Diagnostic reference generation only |
| `base_mass` | 4 | kg; representative within upstream 2/4/8 sampling |
| `ee_mass` | 2 | kg; upstream fixed value |
| `yaw_inertia` | 1.5 | kg m^2; upstream fixed value |
| `base_stiffness`, `ee_stiffness` | 16 | N/m; chosen within upstream 2--30 / 4--30 ranges |
| `base_damping`, `ee_damping` | 7.2 | N s/m; upstream rule `1.8*sqrt(16)` |
| `yaw_stiffness`, `yaw_damping` | 16, 7.2 | Nm/rad, Nm s/rad; upstream matches base gains |
| `alpha` | 1 | Full virtual spring transmission; explicit Stage 2 choice |
| `seed` | 1701 | Dedicated reference-parameter generator |
| `sample_ranges` | `{}` | Fixed defaults; optional reset-only uniform ranges |
| `max_substep_seconds` | 0.005 | RK4 upper bound, tightened from parameter ranges |

These are provisional fixed settings, not validated optimal gains or measured
robot mechanics. Example overrides, only in FACET's configuration:

```python
self.facet_reference.enabled = True
self.facet_reference.sample_ranges = {
    "base_stiffness": [8.0, 24.0],
    "ee_stiffness": [8.0, 24.0],
    "alpha": [0.0, 1.0],
}
```

Unspecified parameters remain fixed; sampled damping is not implicitly coupled
to sampled stiffness. Invalid names/ranges fail clearly. Sampling uses a private
device generator and cannot consume the environment/policy random stream.
Partial reset samples and initializes only selected rows. Set `enabled=False`
to bypass reference allocation and advancement entirely.

## Commands, Frames, and Timing

- Existing sampled base XY velocity is heading-relative, rotated using measured
  interval-start yaw. Base force is `Kb*(measured_xy-ref_xy) +
  Db*(command_world_xy-ref_velocity_xy)`: no division by stiffness, including
  `Kb=0`. The measured position is refreshed each interval, not an initial anchor.
- Yaw uses `Ky*wrap(measured_yaw-ref_yaw) + Dy*(command_rate-ref_rate)`;
  it tracks a yaw-rate command without a fixed world-heading setpoint. Heading
  rate is the actual Euler-yaw derivative, not an assumed body-z gyro value.
- EE setpoints use the existing interpolated spherical command, mount XY offset,
  virtual base XY/yaw, and terrain-height mean plus the existing configured
  `z_invariant_offset`. Height is world-referenced; virtual base vertical motion
  is not modeled. The mount's Z is not added a second time.
- The existing quintic trajectory's analytic spherical derivative is converted
  to Cartesian velocity. During one control interval it is a **first-order hold**
  of that local trajectory. Setpoint velocity includes this derivative, virtual
  base translation, and yaw transport of the mount-inclusive offset. Measured
  terrain height and measured anchor pose are held over that interval.
- `_pre_sim_step` snapshots measured state, commands and application-point
  geometry. Before parent post-step callbacks/reset, the adapter reads actual
  simulator EE force and `applied_torso_wrench_source` buffers. Those loads were
  held across the completed interval, after event ramps/curriculum/masks. Disabled
  sources contribute zero. No force sampler or simulator step is invoked here.
- Torso moments are free couples at the source point. Shift once with
  `(application_point-base_link_origin) cross force`, then use world Z.
  Randomized CoM is not a lever arm. Geometry is interval-start, not successor
  geometry; within-interval lever motion is not resolved separately.
- All coupled forces are evaluated from the same virtual state at each RK4
  stage. Numerical substeps cover exactly `env.dt`, once per completed interval.
  Impedance-range bounds tighten the fixed batch-wide integration step for
  stability; no per-robot loop, BARD call, solve, or optimizer is introduced.
- Reset reads measured articulation state, not terminal policy-step caches;
  no reference state or diagnostics are carried into the new episode. Stage 2
  does not preserve terminal reference histories.

## Output API

`env.get_facet_references()` returns owned detached tensor copies (or `{}` when
disabled): `base_position_xy`, `base_velocity_xy`, `yaw`, `yaw_rate`,
`ee_position_world`, `ee_velocity_world`, `elapsed_seconds`, and `parameter_*`.
Positions use meters, velocities m/s, yaw radians and yaw rate rad/s.
Diagnostics `base_spring_force_xy`, `ee_spring_force_world`, `ee_reaction_xy`,
`applied_base_force_world`, `applied_ee_force_world`, `yaw_spring_moment`, and
`applied_yaw_moment` are **interval-averaged**, using the integration quadrature.
Force units are N and moment units Nm. Reads never advance or sample.

## Verification

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_b1z1_facet.py tests/test_b1z1_facet_reference.py
PYTHONPATH=. OMP_NUM_THREADS=1 python tests/smoke_b1z1_facet.py \
  --gpu cuda:1 --output-dir /tmp/facet-stage2-smoke
```

The smoke uses three independent scenes: Pos, FACET disabled, FACET enabled,
with explicitly matched actual initial states/properties/actions/loads and
non-unit gains/motor strengths plus delays 0/2. It compares Stage 1 physics
against Pos and all recorded outputs (including observations/history/labels and
rewards) between FACET enabled/disabled. No optimizer step is run. Reports and
reference traces are saved alongside the scene logs. Production randomized
terrain/long-horizon stability and learned performance are not established by
this short replay.

### Results on the Installed Environment

- FACET CPU tests: **26 passed** in `lr_lab_cupiqp`. Including the existing Pos
  suite gives **31 passed, 1 failed**: its pre-existing assertion expects FiLM,
  whereas the current baseline uses `conditioning_mode="none"`. No protected
  baseline code/test was changed to mask that failure.
- Isaac Lab on `cuda:1`, 2 robots x 6 intervals: Stage 1 physical parity passed;
  FACET reference-enabled/disabled outputs matched exactly, including all recorded
  observations, labels, rewards, torques, actual wrenches and successor states.
  Artifacts: `/tmp/facet-stage2-smoke-matched/`.
- A further enabled replay passed partial-reset assertions: the reset row's
  reference equals fresh measured state, its clock clears, and every reference
  field/parameter for the other robot remains bitwise unchanged.
  Artifacts: `/tmp/facet-stage2-reset-smoke/`.
- Reference-only float32 microbenchmark: 4,096 robots, 20 warmup + 100 measured
  advances, control dt 0.02 s and RK4 step 0.005 s: **4.36 ms/advance**, **3.22 MiB**
  peak PyTorch allocation in the benchmark process. GPU 1 had concurrent work;
  this is not an end-to-end training slowdown measurement.
- The pre-existing additive Stage 1 registration diff was preserved unchanged.
  Stage 2 edits only FACET files, FACET tests and FACET documentation.

The installed Isaac Lab source invalidates link-state caches on reset writes
and refreshes kinematics on link-pose reads; the adapter uses these measured
buffers rather than the terminal observation cache. Other simulator backends
have not been validated; enabled references require the existing applied torso
wrench source contract. Disabled references retain the Stage 1 path.
