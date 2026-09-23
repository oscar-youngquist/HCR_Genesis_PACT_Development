# B1Z1 PACT FlashSAC

Implemented against checkout `bc8f07b767f8c9e1441430ce1380aeb6cc8b8349` on
`legged_manip_flashsac`. Numerical reference:
[Holiday-Robot/FlashSAC at 87edc90](https://github.com/Holiday-Robot/FlashSAC/tree/87edc9061150ae9e962dd84e6544e27a1554b3ab).
The inspected reference includes agent/network/layer/update, replay, reward
normalizer, scheduler, and the Genesis launch defaults. Its MIT notice is retained
in `rsl_rl/modules/FLASH_SAC_LICENSE`.

## Selection and defaults

`B1Z1PACTCfgPPO.runner.algorithm_class_name = "FlashSAC_B1Z1PACT"` selects the new
learner in the existing B1Z1 runner. Set it to `"PPO_B1Z1PACT"` to retain the PPO
baseline; its original gamma and other PPO settings remain intact. No task
registration changes are needed. FlashSAC uses the separate `sac_gamma=0.95`.

Default settings: batch 2048, two updates per eligible **vectorized** step, actor
and temperature updates every two critic updates (including update zero), critic
EMA tau 0.01, 101 categorical bins on [-5, 5], target sigma 0.15, initial
alpha 0.01. Adam learning rates decay from 3e-4 to 1.5e-4 over the configurable
`sac_lr_decay_updates`. The reference's initial and peak rates are equal, so its
linear warmup is constant. This port requires the existing BARD backend.

The runner accumulates the eligible update count during its existing collection
window, then performs those updates on replay. Warm-up is 10,000 transitions.
There is no multiplication of update count by the environment count, no GAE,
PPO likelihood storage, KL adaptation, value/surrogate loss, or timeout reward
correction in the FlashSAC path.

## Preserved policy and physics

The PACT history encoder, explicit/force/GRF decoders, FiLM, trunk, and coupled
position/torque heads remain. Their concatenated output is the pre-tanh mean.
An additional state-dependent log-std head maps through tanh into [-10, 2]; its
zero weights and inverse-mapped bias reproduce configured initial exploration
scales. SAC uses deterministic context estimates, detached from encoder/decoder
training. Auxiliary representation learning retains its stochastic latent and
VAE KL controller. The model's normalized inference is deterministic `tanh(mean)`;
the runner applies the configured position, leg-torque and arm-torque range
multipliers for environment inference.

Behavior actions use independent per-environment truncated-zeta repeated noise
(mu=2, maximum 16), reset at episode boundaries. SAC samples and replay actions remain normalized. The learner expands behavior
commands by their respective action-range multipliers before passing them to the environment; the
simulator's existing position/torque scaling then runs once.
Legacy Gaussian-scale/value parameters remain frozen in the model solely for
strict PACT-POS checkpoint compatibility; they are excluded from SAC optimizers.

The fused double categorical critic ports the reference's residual blocks,
batch/RMS normalization, unit-weight normalization, minimum-Q categorical
selection, projection, current/next concatenated critic batches, and parameter
EMA. Target batch-normalization statistics evolve from its own batches, as in the
reference. The reward normalizer uses the reference's discounted-return variance
and maximum-return bound. No Hydra/Gym framework is imported.

Delayed actor updates combine SAC and the existing FiLM identity loss with the
actor-facing BARD/FK objectives through the unchanged signed PCGrad helper. The
physics command is the per-channel range multiplied by `tanh(mean)`; entropy uses a reparameterized stochastic action.
Force estimates and mechanics are detached. Auxiliary encoder/explicit and
physics/privileged decoders retain their disjoint optimizers and differentiable
BARD inverse/rollout losses. The force gate and VAE dual controller advance once
per runner update window. Force-gate patience accrues completed environment
steps; KL and physics warmups use that same collection clock.

Replay batches receive local mechanics-cache indices. Actor-physics replay stores
nominal goals and measured workspace geometry; the force-adjusted target is
recomputed with current frozen force predictions using the original projection
helper. Stale predicted forces or latents are never stored as training inputs.

## Replay and checkpoints

Replay is a ring of 32,768 transitions by default, allocated once from the first
complete configured transition and reported before optimization. CPU storage is
pinned when CUDA is available. Normalized observations and histories use FP16;
actions, rewards, physical states, supervision, torques and geometry use FP32;
masks use bool. Sampled floating fields become FP32. Only `n_step=1` is accepted.

The actual default B1Z1 schema, including actor physics and FK snapshots, requires
**634.6 MiB** at capacity 32,768, excluding sampled batches and model/optimizer
memory. Capacity, storage device and pinning are configurable. The two-environment
smoke allocated only 16 transitions (about 0.3 MiB).

A B1Z1-only observation preview captures the final actor observation, privileged
stack, history and dynamics before reset without advancing history slots twice.
Replay distinguishes `terminated = done & ~timeout` and `truncated = timeout`;
only termination masks the Bellman bootstrap. Missing final observations on an
episode boundary are an error. Auxiliary losses retain their existing done masks.

Runner checkpoints contain model/decoder weights, Q and target-Q, temperature,
all five optimizers, SAC schedulers/scaler, reward statistics, counters, VAE dual
state, force gates/blending, and environment curricula. Replay persistence is
optional and off by default. Resume resets per-environment return accumulators
because the simulator starts fresh; global reward statistics are restored.
Hot starts permit only the new log-std keys to be absent and explicitly report
them along with freshly initialized Q/target-Q/temperature.

AMP and compilation default off. Optional AMP covers the critic update; actor
PCGrad and physics solves remain FP32. Optional compilation covers critic/target
forward functions without changing checkpoint key names. These optional paths
have not been validated in this patch.

## Changed files

| File | Change |
| --- | --- |
| `rsl_rl/algorithms/flash_sac_b1z1_pact.py` | SAC learner, auxiliary integration, counters and state |
| `rsl_rl/modules/b1z1_flashsac.py` | Reference critic, projection, reward normalization and repeated noise |
| `rsl_rl/modules/FLASH_SAC_LICENSE` | Reference MIT notice |
| `rsl_rl/modules/actor_critic_b1z1_pact.py` | Optional SAC distribution API and parameter ownership |
| `rsl_rl/storage/replay_buffer_b1z1_pact.py` | Typed transition ring and optional persistence |
| `rsl_rl/algorithms/b1z1_actor_physics.py` | Replay snapshots and fresh force-adjusted targets |
| `rsl_rl/runners/b1z1_pact_runner.py` | Selection, replay collection, metrics and checkpoints |
| `legged_gym/envs/b1z1/b1z1_pact/b1z1_pact_config.py` | B1Z1 SAC defaults; PPO remains selectable |
| `legged_gym/envs/b1z1/b1z1_pact/b1z1_pact.py` | Final pre-reset observation/history preview |
| `legged_gym/simulator/genesis_simulator_b1z1_pact.py` | Three startup compatibility fixes discovered by the real smoke |
| `tests/test_b1z1_flash_sac.py` | Numerical, alignment, ownership, replay target and checkpoint tests |
| `tests/smoke_b1z1_flash_sac.py` | Bounded real two-environment/five-step test |
| `docs/b1z1_flash_sac.md` | This implementation and validation report |

The necessary Genesis startup fixes initialize the first interval-torque buffer
from existing joint state, canonicalize identical batched URDF torque limits to
the existing per-joint contract, and read URDF velocity limits for an empty
override. No rewards, action-scaling formulas, BARD dynamics or task registration
were changed.

## Validation

Used `/home/oyoungquist/.conda/envs/genesis_lr/bin/python` (PyTorch 2.8.0+cu126).
Only targeted tests and bounded smoke attempts were run; no full training.

```bash
SIMULATOR=genesis NUMBA_CACHE_DIR=/tmp/pact-numba OMP_NUM_THREADS=1 \
  /home/oyoungquist/.conda/envs/genesis_lr/bin/python -m pytest \
  tests/test_b1z1_flash_sac.py tests/test_b1z1_sampled_context.py \
  tests/test_b1z1_bard_pinn.py tests/test_b1z1_actor_physics.py \
  tests/test_b1z1_position_fk.py -q \
  -k 'not test_enabled_ppo_update_and_snapshot_alignment'
```

Result: **58 passed, 3 skipped, 1 deselected**. Skips require CUDA access outside
the sandbox. The deselected existing PPO fixture fails with missing
`actor_phys_arm_indices` because FK is enabled but its fixture lacks FK metadata.
The identical failure was reproduced in an unmodified `git archive HEAD` checkout.
During the later completed-step conversion, this fixture was corrected to disable
FK (which it does not model), and the PPO snapshot-alignment test now passes.
The new FlashSAC test file independently passes all **10 tests**.

```bash
SIMULATOR=genesis_b1z1_pact NUMBA_CACHE_DIR=/tmp/pact-numba \
  XDG_CACHE_HOME=/tmp/pact-cache MPLCONFIGDIR=/tmp/pact-mpl \
  TI_OFFLINE_CACHE_FILE_PATH=/tmp/pact-ti PYTHONPATH=. OMP_NUM_THREADS=1 \
  /home/oyoungquist/.conda/envs/genesis_lr/bin/python tests/smoke_b1z1_flash_sac.py
```

The real smoke ran with GPU access on an RTX 4090: **two environments, five
vectorized training steps, eight updates**, PINN ramp enabled, actor physics/FK
active, and a forced timeout. Result: finite critic loss 5.0331, actor loss 0.0461,
temperature 0.00999 in one successful run (random draws can change these values).
The bounded smoke uses a small heightfield, batch size 4, a smaller critic and
replay capacity 16. Genesis rejects the shared config's default trimesh, so the
smoke explicitly selects its supported heightfield; normal Genesis launchers
must likewise select a supported terrain. BARD requires CUDA, so the CPU attempt
could not execute the complete learner.

`git diff --check` and compilation checks also passed. Long-run convergence,
large-batch throughput, AMP/compile, and Isaac Gym execution remain
unvalidated. Compact observation storage introduces FP16 quantization; physics
and supervised targets retain FP32 precision.


## IsaacLab validation

The same bounded smoke also passed in the local **lr_lab_cupiqp** environment:
PyTorch 2.7.0+cu128, IsaacLab 0.54.2, Isaac Sim 5.1.0.0. The harness now supports
both simulator backends and closes the IsaacLab application after completion.

```bash
SIMULATOR=isaaclab_b1z1_pact OMP_NUM_THREADS=1 \
  /home/oyoungquist/.conda/envs/lr_lab_cupiqp/bin/python -m pytest \
  tests/test_b1z1_flash_sac.py -q

SIMULATOR=isaaclab_b1z1_pact PACT_SMOKE_DEVICE=cuda:0 PYTHONPATH=. OMP_NUM_THREADS=1 \
  /home/oyoungquist/.conda/envs/lr_lab_cupiqp/bin/python -u \
  tests/smoke_b1z1_flash_sac.py
```

Results: **10 focused tests passed**; real IsaacLab/BARD smoke **passed with two
environments, five vector steps and eight SAC updates**, PINNs/FK enabled and a
forced timeout. Finite losses were critic 5.0249, actor 0.4235; temperature 0.00999.
Replay estimate was again 634.6 MiB at the default capacity. The first attempt on
GPU 1 encountered PhysX allocation failure because an existing process occupied
about 20 GiB. Retrying on GPU 0 succeeded without changing that process or the
training implementation. The successful run log is
`/tmp/pact_flash_smoke_isaaclab_gpu0.log` (temporary local artifact).


## Configurable environment action range

Three `B1Z1PACTCfgPPO.algorithm` settings control the intermediate range expansion:

| Setting | Normalized action indices | Outputs |
| --- | --- | --- |
| `sac_position_action_range` | `[0:17]` | All 17 learned position offsets |
| `sac_leg_torque_action_range` | `[17:29]` | 12 leg feedforward torques |
| `sac_arm_torque_action_range` | `[29:34]` | 5 learned arm feedforward torques |

The learner uses **1.0** when a range is omitted; the task config can override
each independently. Each must be
finite, positive and no larger than `normalization.clip_actions`. PPO ignores
these settings. The two unlearned arm/gripper joints retain their existing PD control.

```text
normalized SAC action in [-1, 1]
    -> multiply by the corresponding channel's range
    -> existing environment clipping/history/delay
    -> existing position offset / feedforward torque scaling
```

For example, position range 2.0, leg-torque range 3.0 and arm-torque range 4.0
produce position offsets up to ±0.5 rad, leg feedforward torques up to ±90 Nm,
and arm feedforward torques up to ±40 Nm with current physical scales, before
existing motor-strength and branch weights. These are command ranges, not
guaranteed realized joint positions or total actuator torques.

Replay actions, all Q inputs, log probabilities and entropy targets remain in
normalized coordinates. Collection nominal torques, inference from the runner,
and deterministic actor BARD/FK losses receive expanded commands exactly once.
The model's `act_inference` remains a normalized API; use the runner's inference
policy for environment commands. Auxiliary physics uses measured executed torque.

Checkpoints save and restore all three ranges even for weights-only runner loads;
the saved ranges take precedence over the current config to preserve the meaning
of replay actions. Checkpoints with a single `action_range` restore that value
for all three channels; checkpoints without range metadata restore 1.0. PACT-POS
hot starts use the newly configured ranges. Each range is logged under
`SAC/{position,leg_torque,arm_torque}_action_range`.

Targeted validation in `lr_lab_cupiqp`:

```bash
SIMULATOR=isaaclab_b1z1_pact OMP_NUM_THREADS=1 \
  /home/oyoungquist/.conda/envs/lr_lab_cupiqp/bin/python -m pytest \
  tests/test_b1z1_flash_sac.py tests/test_b1z1_torque_action_scaling.py -q
```

The checks cover invalid settings, collection/inference consistency, normalized
critic/replay actions, physical scaling and gradients, actor-physics routing,
and checkpoint restoration.

Split-range validation passed: **34 targeted tests** in `lr_lab_cupiqp`, plus a
real IsaacLab GPU smoke with position=2.0, leg torque=3.0 and arm torque=4.0
(two environments, five vector steps, eight updates, including actor physics
and a forced timeout). Temporary smoke log: `/tmp/pact_flash_smoke_split_ranges.log`.


## Completed environment-step curricula

The coupled `b1z1_pact` PPO and FlashSAC paths use **completed per-environment
control steps** for training curricula. One successful vectorized `env.step()`
advances this clock by one, regardless of `num_envs`, rollout length, or replay
updates. Resets, physics substeps and gradient updates do not advance it.
`total_timesteps` separately counts transitions across all environments.

The config now names curriculum times with `_env_step`/`_env_steps`:

| Schedule | New default in completed control steps |
| --- | ---: |
| Reward warmup / cosine ramp | 720,000 / 240,000 |
| Gait-guidance decay (when enabled) | 240,000 |
| Force gate earliest / fallback start | 192,000 / 192,000 |
| External-force ramp / qualifying patience | 192,000 / 9,600 |
| Domain-randomization warmup / minimum spacing | 312,000 / 240 |
| KL base warmup / band ramp | 24,000 / 12,000 |
| PINN and actor-physics start / ramp | 0 / 12,000 |
| Predicted-force reliability patience | 240 |

These are the former durations multiplied by the historical 24-step rollout.
Changing `num_steps_per_env` no longer changes their duration. Reward/gait scales
refresh after every completed control step, for use on the next transition.
Reward-gated force/domain-randomization statistics are evaluated every
`runner.curriculum_metrics_interval_env_steps` (default 24), including within a
rollout. Episode samples are accumulated across rollout boundaries. Missing
performance samples do not count as qualifying evidence; force fallback still
uses the completed-step clock. Learner physics/KL coefficients are evaluated at
the completed collection count, and extra updates at that count add no patience.

Terrain, command and feedback/feedforward tradeoff curricula already advance
from episode outcomes/control steps. They retain those triggers. Adaptive PPO
learning rate and entropy remain performance-driven; SAC learning-rate decay
retains its explicit optimizer-step duration. Run length/checkpoint cadence still
use learning iterations, and simulator solver iteration counts are unchanged.
UniFP and PACT-pos retain their own schedules.

New checkpoints persist the exact completed-step count and pending curriculum
metric samples, and immediately restore reward, gait and physics schedules.
Legacy SAC timestamps are converted using saved collection counts divided by
completed iterations. Legacy PPO checkpoints lack collection counts, so migration
assumes the historical 24-step rollout and emits a warning. Legacy duration
settings are converted with the historical factor of 24. Hot starts begin at zero.

Validation includes curriculum boundaries, rollout-partition independence,
checkpoint migration, repeated optimizer work at a fixed clock, and a real
IsaacLab smoke that crosses shortened reward/gait/force ramps in five control
steps. Temporary GPU log: `/tmp/pact_flash_smoke_env_step_curricula.log`.

Final curriculum checks: **113 distinct focused tests passed** across the
regression suite and added clock/checkpoint cases; two CUDA-only cases skipped
inside the sandbox. The real IsaacLab GPU smoke passed separately (two
environments, five completed control steps, eight SAC updates).
