# FACET Stage 3: Same-Time Histories and Rewards

FACET now trains against six multi-start tracking rewards, using the unchanged
[Stage 2](b1z1_facet_stage2.md) reference equations, impedance defaults, frames,
and RK4 settings. This is still the temporary position-only PPO architecture,
not the FACET teacher/student implementation. No observation or action channels
are added. Robot/PD settings, terrain (including its existing disabled terrain
curriculum), disturbances/curriculum, randomization, termination and command
sampling/interpolation/collision rejection remain inherited and unchanged.

## Configuration

```python
# In B1Z1FACETCfg; default Stage 3 behavior:
self.facet_reference.enabled = True
self.facet_tracking.enabled = True
self.facet_tracking.horizons = (8, 16, 32)
self.facet_tracking.weights["ee_position"] = 1.0
self.facet_tracking.bandwidths["ee_position"] = 0.10

# Exact Stage 2 diagnostic-only reward behavior:
self.facet_tracking.enabled = False
```

Turning tracking off also skips history allocation and history/reward reductions;
the original single-start reference remains diagnostic-only. Turning both flags
off retains Stage 1. Tracking with references disabled is rejected as invalid.
FACET settings are independent nested configuration copies; no baseline class or
shared dictionary is modified. The temporary runner/model and output names stay
unchanged.

## Timing and Startup

For interval `n -> n+1`, one seed containing measured base XY position/velocity,
yaw/rate, and world EE XYZ position/velocity is inserted at boundary `n`. Every
live start advances through the actual interval-start command packet and actual
held, gated physical loads. The age-`h` target at the end therefore starts from
`X[n+1-h]` and ends at `n+1`, alongside measured successor `X[n+1]`.

These are **different starts ending now**, not future predictions or stored old
targets. Changing commands and forces refresh the integrator inputs each interval;
old starts do not get replayed using just the latest packet. Stage 2's local
nominal-trajectory first-order hold remains unchanged within an interval.

A fixed-size ring stores at most 32 starts per robot. One broadcast RK4 call
advances robot/start dimensions together; insertion changes one slot without
rolling/copying the full history. Episode impedance tensors are shared views,
not sampled per start. Inactive slots undergo fixed-shape arithmetic but are
masked out of state advancement and validity. Only three requested endpoints are
copied for the transition snapshot. There are no robot/horizon integration loops,
new dynamics solves, or policy passes.

After a reset:

| Completed control intervals | Valid requested ages |
| --- | --- |
| 1--7 | none; all six tracking contributions are zero |
| 8--15 | 8 |
| 16--31 | 8, 16 |
| 32 onward | 8, 16, 32 |

Partial resets clear only the selected robots' bank slots. History insertion
never calls `FACETReference.reset()` or consumes randomness. Episode resets retain
the existing dedicated impedance sampling. The completed transition snapshot
survives reset for terminal diagnostics; it is replaced on the next interval.

Advancement remains before the parent's `post_physics_step`. Reward reduction
runs inside the normal reward phase, after simulator measurement refresh and
command callbacks but before resets. It uses the frozen history targets, not
newly resampled nominal commands. Reads and individual reward functions do not
integrate, sample, or mutate reference state.

## Reward Definitions

Let `e2` be each horizon's squared vector error (scalar squared wrapped yaw error
for yaw). Compute the following **per horizon**, average over that robot's valid
horizons, then apply the configured weight and the parent's control-dt scaling
exactly once. Never reward an averaged target.

| Runtime reward | Raw formula | Bandwidth | Weight |
| --- | --- | ---: | ---: |
| `facet_base_position` | `exp(-e2/bandwidth)` | 0.25 m^2 | 0.75 |
| `facet_base_velocity` | `exp(-e2/bandwidth)-0.25*e2` | 0.25 (m/s)^2 | 1.50 |
| `facet_yaw` | `exp(-e2/bandwidth)` | 0.25 rad^2 | 0.75 |
| `facet_yaw_rate` | `exp(-e2/bandwidth)-0.25*e2` | 0.25 (rad/s)^2 | 1.00 |
| `facet_ee_position` | `exp(-e2/bandwidth)` | 0.10 m^2 | 1.00 |
| `facet_ee_velocity` | `exp(-e2/bandwidth)` | 0.25 (m/s)^2 | 1.00 |

Definitions follow the [released FACET rewards at the pinned source commit](https://github.com/btx0424/active-adaptation/blob/c0d6f5aa6146814803090b5f6f19aafb8ce6f792/projects/facet/facet_commands/rewards.py).
Base translation is world XY; EE translation is world XYZ. Yaw errors wrap to
the shortest angular difference. Yaw rate retains Stage 2's Euler-heading
derivative, rather than assuming the world-z angular-velocity component is
identical for a tilted torso. There is no extra EE-velocity quadratic penalty.

## Replacements and Legacy Consumers

`facet_tracking.excluded_rewards` removes the following from the **registered
optimization objective**, only for FACET:

- Nominal `tracking_lin_vel_force_world`, `tracking_ang_vel`, and
  `tracking_ee_force_world`.
- `no_physical_progress`, `stand_still`, `stand_still_contact`, `ref_dof_leg`,
  `feet_contact_number`, `walking_ref_dof`, `walking_ref_swing_dof`,
  `arm_progress_before_torso`, and `early_torso_tilt`.

Nonconflicting physical regularization and its existing schedules remain.
Excluded shaping is not re-enabled by the inherited reward scheduler. Existing
zero-weight orientation/impedance terms remain zero. No new reference-gait,
stationary, feedforward, or PINN objective is added.

The inherited nominal scales stay in `cfg.rewards.scales` intentionally: they
define the old diagnostic units and allow the disabled switch to restore Stage 2.
Runtime `reward_names`/`reward_functions` are the authoritative objective list.
Three legacy tracking scales are retained in runtime `reward_scales` solely for
existing threshold consumers. They are **absent from the optimized function list**.

FACET calls the unchanged parent nominal tracking formulas in the original
post-callback timing, accumulating `raw_nominal_reward * legacy_weight * dt` in
the original `episode_sums` keys. Parent reset logging still divides by
`max_episode_length_s`; command/tradeoff thresholds still divide by maximum
episode steps and use the old dt-scaled coefficient. Domain-randomization and
adaptive-entropy readers therefore receive the original `rew_tracking_*` keys,
units, and aggregation, never FACET rewards. These diagnostic sums do not enter
`rew_buf`, positive-reward clipping, or PPO returns.

The parent's independent `EE/tracking_l1_mean` and per-axis physical error
diagnostics remain unchanged for the external-force curriculum. They still use
the nominal EE target; FACET never overwrites that target or the target helper.

## Reading and Logging

- `get_facet_references()` remains the Stage 2 single-start diagnostic API.
- `get_facet_tracking()` returns owned detached copies for the last completed
  transition: `targets [N,H,12]`, actual `ages [N,H]`, `valid [N,H]`,
  `errors_squared [N,H,6]`, and `rewards_unweighted [N,6]`.
- Target columns: base world XY, base world XY velocity, yaw, yaw rate, EE world
  XYZ, EE world XYZ velocity. Component order is the reward-table order above.
- Parent episode logs emit `Episode/rew_facet_*` with existing weight/time
  normalization. At episode resets, `Episode/facet/h8/...` (also h16/h32) includes
  `valid_fraction`, `valid_steps`, and each component's `*_mse` in squared SI units.
  MSE pools valid samples of resetting robots; validity divides valid samples by
  their actual episode steps. When count is zero, MSE logs zero with zero validity,
  not a claim of perfect tracking. Terminal samples are included before clearing.

## Reproducible Checks

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_b1z1_facet.py tests/test_b1z1_facet_reference.py \
  tests/test_b1z1_facet_history.py tests/test_b1z1_ppo_pos.py

PYTHONPATH=. OMP_NUM_THREADS=1 python tests/smoke_b1z1_facet.py \
  --gpu cuda:1 --steps 40 --output-dir /tmp/facet-stage3-smoke

SIMULATOR=isaaclab PYTHONPATH=. OMP_NUM_THREADS=1 \
  python tests/benchmark_b1z1_facet_history.py --device cuda:1 --num-envs 4096
```

The smoke explicitly matches states, actions, mechanics, delays, commands and
applied loads across separate Pos, FACET-disabled, Stage 2 and Stage 3 scenes.
It compares physical outputs and observations, expects Stage 3 rewards to differ,
checks identical nominal diagnostic sums, matures all horizons, and exercises
partial resets and the real parent timeout/reward/reset path. It performs no
learning. The benchmark compares warmed single-start Stage 2 with single-start
plus history bank/reward/log reductions; it does not measure full training speed.

## Results and Limits

- **38 FACET CPU tests passed** in the installed `lr_lab_cupiqp` environment,
  including independent offline replay, changing command/load packets, masks,
  partial resets/RNG, formulas/wrapping, nonlinear averaging, dt scaling,
  nominal-metric bookkeeping and terminal snapshots. The combined final run
  with existing Pos tests had **43 passes and one failure**, the known stale FiLM
  assertion at `tests/test_b1z1_ppo_pos.py:56`. The protected Pos test was not edited.
- Isaac Lab `cuda:1`: four separate 2-robot scenes completed **40 uninterrupted
  intervals**. All 106 recorded physical/observation/nominal-metric comparison
  fields had zero maximum difference. Stage 3 optimization rewards intentionally
  differed. All requested ages matured; partial reset and timeout assertions
  passed. Logs/configs/NPZ/parity report: `/tmp/facet-stage3-smoke/`.
- An additional Stage 3 simulator replay passed partial-reset, timeout, and
  physical roll-termination checks with unchanged failure thresholds. Both
  terminal paths retained mature transition snapshots while clearing the next
  episode's bank. Artifacts: `/tmp/facet-stage3-terminal-smoke/` and its `.log`.
- Same-device float32 benchmark, 4,096 robots, dt 0.02 s, 40 warmups followed by
  five blocks of 20 calls: Stage 2 median **4.286 ms**, Stage 3 median **9.575 ms**,
  an additional **5.289 ms/control interval**. Block ranges were 4.266--4.351 ms
  and 9.385--9.943 ms respectively. Peak PyTorch allocations (including common
  benchmark fixtures) were **4.364 MiB** and **62.658 MiB**. Both used the same
  integrator/settings; Stage 3 includes the preserved single-start diagnostic.

No simulator, runner, PILOT/PACT/ablation/UniFP implementation or configuration
was changed. No training was run. The smoke uses a controlled flat-terrain fixture
with explicit matched mechanics, not a long randomized-terrain evaluation.
These checks establish reference/reward timing and isolation, not learned task
performance. Stage 2's first-order hold, virtual-model approximations, and lack of
workspace/collision/recruitment constraints continue to apply. FACET architecture,
reference-conditioned observations and teacher/student training remain future work.
