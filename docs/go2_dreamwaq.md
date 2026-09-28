# Go2 DreamWaQ: installation, configuration, and training

This guide sets up the Go2 DreamWaQ port in **this repository** using IsaacLab.
The policy learns joint-position commands from proprioceptive history, with a
variational encoder, supervised state estimates, and an asymmetric PPO critic.
The port includes PACT-style domain and reward curricula, physical torque
clipping, and separate optimizers for estimation and locomotion.

## 1. Install the software

### Prerequisites

Use Linux x86-64, Bash, Git, and a working NVIDIA GPU/driver compatible with
Isaac Sim 5.1. Install Anaconda or Miniconda if `conda` is not available, then
initialize your shell with `conda init bash` and open a new terminal.
The pip installation requires glibc 2.35 or newer; Ubuntu 22.04 is a suitable
baseline. Check the host before installing:

```bash
nvidia-smi
ldd --version
conda --version
```

The commands below use Python 3.11, Isaac Sim 5.1.0, and IsaacLab v2.3.2.
For platform prerequisites and the CUDA PyTorch installation, see the
[official IsaacLab 2.3 installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/pip_installation.html).
Keep the versions below pinned when reproducing this setup.

### Create the conda environment

```bash
conda create -n lr_lab python=3.11 -y
conda activate lr_lab
python -m pip install --upgrade pip
```

Use this same environment for every remaining installation command and for
training. The repository's `environment.yml` describes a separate Genesis
Python 3.10 environment; it is not the environment for these IsaacLab steps.

### Install and verify Isaac Sim

```bash
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install 'torch==2.7.0' 'torchvision==0.22.0' --index-url https://download.pytorch.org/whl/cu128
```

On a machine with a desktop, run:

```bash
isaacsim
```

Complete the first-run license prompt and wait for the application window to
open, then close it before continuing. On a headless server, use the headless
IsaacLab check below instead of the GUI check. Initial startup can take longer
while extensions and assets are prepared.

### Clone and install IsaacLab

The following layout places IsaacLab and this repository next to each other:

```bash
mkdir -p ~/Research
cd ~/Research
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install none
```

`none` skips installation of external RL training libraries. This project uses
its own bundled `rsl_rl` implementation, including the DreamWaQ optimizer and
checkpoint changes.

Verify IsaacLab from its repository root:

```bash
python scripts/tutorials/00_sim/create_empty.py
```

A window containing an empty scene confirms startup. Close it when finished.
For a headless check on GPU 0, use:

```bash
python scripts/tutorials/00_sim/create_empty.py --headless --device cuda:0
```

Wait for initialization to complete without an exception, then stop the example
with Ctrl+C; it runs continuously.

### Clone and install this DreamWaQ codebase

Use the research fork containing this port, rather than the upstream
`lupinjia/LeggedGym-Ex` repository. You need access to the fork. The working
branch for this port is `vae_rateband`; use the revision supplied by the project
maintainer when reproducing an experiment.

```bash
cd ~/Research
git clone --branch vae_rateband https://github.com/oscar-youngquist/HCR_Genesis_PACT_Development.git
cd HCR_Genesis_PACT_Development
python -m pip install -e '.[isaaclab]'
python -m pip install pytest
```

If the maintainer distributes a source archive instead, extract it and run the
same installation commands from the directory containing `pyproject.toml`.
The checkout must contain `legged_gym/scripts/go2_dreamwaq.sh` and the standalone
Go2 DreamWaQ config described below. Local, uncommitted changes are not included
when a collaborator clones the remote repository.

Confirm that the custom RL package resolves into this checkout:

```bash
python -c 'import rsl_rl; print(rsl_rl.__file__)'
python -m pip check
```

The printed path should end in this repository's `rsl_rl/__init__.py`. Do not
replace it with an external `rsl-rl` installation. Record the checkout and
resolved package versions with each experiment:

```bash
git rev-parse HEAD
python -m pip freeze > installed-packages.txt
```

### Run a short training check

From the research repository root, with `lr_lab` active:

```bash
bash legged_gym/scripts/go2_dreamwaq.sh --gpu cuda:0 --num_envs 16 --max_iterations 2
```

This selects the actual `go2_dreamwaq` task, creates rough-terrain environments,
and performs two training iterations. Expect simulator startup output, PPO/VAE
losses, episode metrics, and a checkpoint in the log directory described below.
It checks installation and training integration, not learned locomotion quality.

## 2. Launch and monitor training

Activate the environment in every new terminal and enter the repository:

```bash
conda activate lr_lab
cd ~/Research/HCR_Genesis_PACT_Development
bash legged_gym/scripts/go2_dreamwaq.sh --gpu cuda:0
```

The [launcher](../legged_gym/scripts/go2_dreamwaq.sh) selects `go2_dreamwaq`, runs
headless, and defaults to IsaacLab. It changes to the repository root itself.
**The current checked-in script text selects `--gpu cuda:1`.** The explicit
`--gpu cuda:0` above overrides it for a single-GPU workstation. Edit the GPU
argument in the script to set your machine's default, or pass `--gpu cuda:N`
when launching. GPU selection uses the command-line argument; the script does
not set or export `CUDA_VISIBLE_DEVICES`. If a scheduler already masks devices,
CUDA indices refer to the devices visible to that process.

Extra arguments are forwarded to the training program:

```bash
bash legged_gym/scripts/go2_dreamwaq.sh --gpu cuda:0 --num_envs 1024 --max_iterations 10000 --seed 1
```

| Argument | Purpose |
| --- | --- |
| `--gpu cuda:N` | Device used for simulation and learning |
| `--num_envs N` | Override parallel environment count; reduce it if GPU memory is insufficient |
| `--max_iterations N` | Number of learning iterations to execute in this invocation |
| `--seed N` | Random seed; default 1 |
| `--resume` | Load an existing run/checkpoint before learning |
| `--load_run NAME` | Run directory name under the experiment log root |
| `--ckpt N` | Load `model_N.pt`; `-1` selects the latest checkpoint in the chosen run |
| `--sync_wandb` | Optionally enable the repository's Weights & Biases integration |

One learning iteration collects `runner.num_steps_per_env` control steps from
each environment, then performs optimization. It is not one physics step or one
episode. On resume, `--max_iterations` specifies **additional** iterations.

The normal launcher uses 4,096 environments and 10,000 learning iterations unless
overridden. Logs are written under:

```text
logs/go2_rough/<timestamp>_dreamwaq_isaaclab/
```

The training script copies the task implementation and configuration into the
run directory. Checkpoints are named `model_<iteration>.pt`, saved every 500
iterations and at the end of a training invocation.

Start TensorBoard in a second terminal with the same environment active:

```bash
tensorboard --logdir logs/go2_rough
```

Open the URL printed by TensorBoard. Monitor episode reward/tracking, PPO losses,
reconstruction/KL and explicit-estimation losses, `Curriculum/*`, `DomainRand/*`,
and `RewardScale/*`. A stalled domain curriculum means its performance gate has
not been met, or no fresh completed-episode statistics are available.

To resume a specific checkpoint, substitute the actual run directory name:

```bash
bash legged_gym/scripts/go2_dreamwaq.sh --gpu cuda:0 --resume \
  --load_run '<timestamp>_dreamwaq_isaaclab' --ckpt 500 --max_iterations 1000
```

This starts from iteration 500 and executes 1,000 more iterations, writing to a
new run directory. Checkpoints restore the model, both optimizer states, PPO
learning rate, next iteration, and domain-curriculum progress/reward history.
Reward weights are reconstructed from the schedule and restored iteration.
Simulator trajectories, terrain/command curriculum state, and sampled physical
parameters are not restored exactly; resets install the restored domain ranges
and restart the episode resampling cadence. Resume with the same architecture
and curriculum configuration. Old DreamWaQ checkpoints without the 11-output,
two-optimizer format are rejected; start a fresh run for those models.

## 3. Implemented DreamWaQ approach

### Observations, estimates, and actions

The actor receives a 45-value proprioceptive observation: command velocity (3),
projected gravity (3), torso angular velocity (3), joint position offsets (12),
joint velocities (12), and previous actions (12). Five observations form the
225-value history fed into the encoder. The actor does not receive terrain
height measurements or simulator labels directly.

The encoder produces a 16-dimensional Gaussian latent and 11 explicit estimates:

| Slice | Estimate | Training target |
| --- | --- | --- |
| `0:3` | Torso linear velocity | Body-frame velocity multiplied by `normalization.obs_scales.lin_vel` (default 1.0) |
| `3:7` | Foot-contact probability | FR, FL, RR, RL; contact-force norm greater than 1 N |
| `7:11` | Foot height above local terrain | Foot world Z minus mean surrounding terrain height minus the 0.022 m foot offset |

The actor consumes the current observation, latent, and explicit estimates to
produce 12 joint-position actions. The decoder uses the latent and estimates to
reconstruct the next clean 45-value observation. The critic receives privileged
observations: clean proprioception, torso velocity, randomization information,
contact states, and terrain heights. With five critic frames and the default
187 height samples, its input has 1,415 values. The environment derives the
critic width from enabled sensors at initialization.

During training, the implicit latent is sampled; inference/export uses its mean.
The explicit continuous heads are unconstrained. The contact head outputs logits;
the actor and decoder receive `epsilon + (1 - 2*epsilon) * sigmoid(logit)`, with
`epsilon = 1e-6`. Contacts use BCE with logits during training.

### Two independent optimization objectives

One Adam optimizer owns the encoder, latent mean/log-variance heads, explicit
head, and decoder. It minimizes the auxiliary estimation/autoencoding objective:

```text
continuous_loss = MSE across torso velocity (3) and foot heights (4)
contact_loss = mean BCEWithLogits across four feet
explicit_loss = continuous_loss + 0.10 * contact_loss
VAE_loss = next_observation_MSE + 2.0 * KL + 1.0 * explicit_loss
```

KL is summed across latent dimensions and averaged across valid transitions.
The configurable weights are `contact_probability_loss_weight`,
`vae_kld_weight`, and `explicit_loss_weight`. Supervised targets are detached.
History/labels come from time t and the reconstruction target from t+1.
Auto-reset transitions are excluded from auxiliary losses. Valid rows are
selected before arithmetic so masked invalid values cannot contaminate the
loss. An all-terminal minibatch skips the VAE optimizer update, including Adam
momentum changes.

A second Adam optimizer owns the actor, critic, and learned action standard
deviation. Only PPO's policy/value/entropy objective updates those parameters.
PPO uses detached encoder features, so it cannot update the VAE. The VAE objective
cannot update the actor or critic. This is the implemented DreamWaQ variant with
PACT-derived explicit-estimator and curriculum conventions.

### Control, torque clipping, and contact indexing

Actions follow `asset.dof_names`: FR, FL, RR, RL, with hip/thigh/calf per leg.
The position target is `default_joint_position + 0.25 * action`. The policy runs
at 50 Hz; PD torque is recomputed at each 200 Hz physics step using the live joint
state. Randomized PD gains and motor strength are applied once. Executed torque
is clipped to physical effort limits. IsaacLab uses the smaller actuator/solver
limit in policy joint order; its motor model can further reduce applied torque.
There is no additional torque-rate limiter.

The torque-limit reward uses a separate **unclipped delayed action request**,
converted to torque before action/torque saturation. Excessive requests therefore
remain penalized even if execution saturates. Raw and executed actions share the
same delay and clear on reset. Torque magnitude/power rewards use bounded
commanded torque.

Articulation body order and contact-sensor order differ in IsaacLab.
`feet_indices` addresses body positions/velocities; `feet_contact_indices`
addresses sensor forces. They resolve FR, FL, RR, RL independently by name.
Explicit contact targets, overreach rewards, collision penalties, and termination
checks use the appropriate sensor indices.

## 4. Configuration reference

Edit [go2_dreamwaq_config.py](../legged_gym/envs/go2/go2_dreamwaq/go2_dreamwaq_config.py)
for environment and training settings. `Go2DreamwaqCfg` inherits directly from
`LeggedRobotCfg`; `Go2DreamwaqCfgPPO` inherits directly from `LeggedRobotCfgPPO`.
Go2-specific settings are local, with no dependency on the shared DreamWaQ or
Go2 common configuration classes. Remaining generic defaults come from
[legged_robot_config.py](../legged_gym/envs/base/legged_robot_config.py).
Only the CLI options listed above override config values; arbitrary nested
config parameters are not command-line options.

### Environment and controller

| Section / parameter | Current default | Meaning |
| --- | --- | --- |
| `env.num_envs` | 4096 | Parallel simulated robots |
| `env.episode_length_s` | 20 | Maximum episode duration |
| `env.frame_stack` / `num_history_obs` | 5 / 225 | Actor encoder history |
| `env.num_latent_dims` / `num_explicit_dims` | 16 / 11 | Implicit and explicit estimate sizes |
| `env.c_frame_stack` | 5 | Privileged critic history |
| `sim.dt` / `control.decimation` | 0.005 s / 4 | Physics timestep / physics steps per policy action |
| `control.stiffness` / `damping` | 20 / 0.5 | Nominal PD gains |
| `control.action_scale` | 0.25 | Joint-position offset in radians per action unit |
| `init_state.default_joint_angles` | Hip 0, thigh 0.8, calf −1.5 rad | Nominal pose |
| `normalization.clip_actions` / `clip_observations` | 100 / 100 | Numerical action/observation clipping; separate from torque saturation |
| `noise.add_noise` | True | Adds configured noise to actor observations |
| `asset.dof_names`, `feet_names` | FR, FL, RR, RL ordering | Policy joint and explicit foot ordering |
| `asset.terminate_after_contacts_on` | Empty | No contact-triggered termination; other termination conditions remain active |
| `sim.use_dreamwaq_adapter` | True | Selects the DreamWaQ backend adapter |

Keep `sim.dt`, `control.decimation`, and `control.dt` consistent if changing
control frequency. Observation normalization and noise magnitudes are configured
in `normalization.obs_scales` and `noise.noise_scales`.

### Terrain and command curricula

IsaacLab uses a triangle-mesh terrain with 10 difficulty rows, 10 columns,
8 × 8 m patches, a 20 m border, and 4 m platforms. The proportions for smooth
slopes, rough slopes, up stairs, down stairs, and discrete obstacles are
`[0.2, 0.1, 0.25, 0.25, 0.2]`. `terrain.measure_heights=True` enables the 17 × 11
critic height grid; `obtain_terrain_info_around_feet=True` supplies local foot
height references. `terrain.curriculum=True` adjusts terrain difficulty at reset
based on distance traveled relative to the commanded motion.

Commands are resampled every 10 seconds. Initial X velocity is −0.5 to 0.5 m/s,
Y velocity is −1 to 1 m/s, yaw-rate range is −1 to 1 rad/s, and heading range is
−3.14 to 3.14 rad. With `heading_command=True`, yaw rate is computed from heading
error. `commands.curriculum=True` expands the X range toward ±1 m/s when linear
tracking exceeds `curriculum_threshold=0.8`. These terrain/command curricula are
separate from domain randomization and reward-weight schedules.

### Network and optimizer settings

| Parameter in `Go2DreamwaqCfgPPO` | Default | Purpose |
| --- | --- | --- |
| `policy.actor_hidden_dims` | `[512, 256, 128]` | Actor MLP |
| `policy.critic_hidden_dims` | `[1024, 256, 128]` | Critic MLP |
| `policy.encoder_hidden_dims` / `decoder_hidden_dims` | `[256, 128]` each | Auxiliary networks |
| `policy.activation` / `init_noise_std` | `elu` / 1.0 | Hidden activation / initial action exploration |
| `algorithm.learning_rate` | 0.001 | Initial PPO Adam rate |
| `algorithm.schedule` / `desired_kl` | `adaptive` / 0.01 | PPO learning-rate adaptation |
| `algorithm.encoder_lr` | 0.0002 | VAE Adam rate |
| `algorithm.num_learning_epochs` / `num_mini_batches` | 5 / 4 | PPO rollout minibatch passes |
| `algorithm.num_encoder_epochs` | 1 | VAE updates per PPO minibatch |
| `algorithm.clip_param` / `entropy_coef` | 0.2 / 0.01 | PPO clipping / exploration weight |
| `algorithm.gamma` / `lam` | 0.99 / 0.95 | Discount / GAE coefficient |
| `algorithm.value_loss_coef` / `max_grad_norm` | 1.0 / 1.0 | Value loss weight / gradient clipping |
| `algorithm.vae_kld_weight` | 2.0 | Latent regularization |
| `algorithm.explicit_loss_weight` | 1.0 | Overall explicit supervision weight |
| `algorithm.contact_probability_loss_weight` | 0.10 | Contact BCE weight within explicit loss |
| `runner.num_steps_per_env` | 24 | Control steps per rollout per environment |
| `runner.max_iterations` / `save_interval` | 10000 / 500 | Run length / checkpoint interval |
| `runner.resume` | False | Fresh training by default |

### Reward terms and reward curriculum

Positive scales reward behavior; negative scales penalize it. Scales are multiplied
by the control timestep once. `only_positive_rewards=True` clips the total
nonterminal reward below at zero, so large penalties can suppress all positive
reward on a step.

Fixed active scales are:

| Reward term | Scale | Meaning |
| --- | --- | --- |
| `tracking_lin_vel`, `tracking_ang_vel` | 1.0, 0.5 | Command tracking |
| `lin_vel_z` | −2.0 | Vertical torso velocity |
| `dof_pos_limits`, `dof_vel_limits` | −2.0, −1.0 | Joint limit penalties |
| `collision` | −1.0 | Contacts on configured penalized bodies |
| `dof_power`, `dof_acc` | −0.0002, −0.0000002 | Joint power and acceleration |
| `feet_air_time`, `foot_clearance` | 1.0, 0.2 | Swing duration and clearance tracking |
| `feet_contact_stand_still` | 0.5 | Standing contact behavior |
| `front_foot_overreach`, `rear_foot_overreach` | −100.0, −10.0 | PACT-derived stance-foot placement penalties |

Position, velocity, and torque soft limits are all 0.9. The velocity penalty sums
`clamp(abs(qdot) - 0.9 * velocity_limit, 0, 1)` across joints. Foot clearance
tracks 0.09 m, with 0.022 m foot offset and tracking sigma 0.01.

Front overreach is squared torso-frame X excess beyond 0.28 m. It is multiplied
by `1 - 0.5 * nominal_mass / (nominal_mass + max(added_mass, 0))`.
Rear overreach is squared excess outside the −0.25 ± 0.08 m torso-frame X band.
Both require upward foot contact force greater than 5 N, controlled by
`overreach_contact_force_threshold`. This is distinct from the 1 N force-norm
threshold for explicit contact labels. The front reward retains PACT's formula;
its current DreamWaQ weight is −100, rather than PACT's −10000.

`rewards.use_reward_curriculum=True` schedules these six scales:

| Term in `reward_curriculum.curr_reward_bounds` | Initial → final |
| --- | --- |
| `ang_vel_xy` | −0.05 → −0.2 |
| `orientation` | −0.2 → −1.0 |
| `torque_limits` | −0.01 → −1.0 |
| `hip_pos` | −0.2 → −0.4 |
| `action_rate` | −0.001 → −0.01 |
| `action_smoothness` | −0.001 → −0.01 |

`warmup_steps=0` and `curr_steps=5000` are **learning iterations**. At iteration i:

```text
fraction = clamp((i - warmup_steps) / max(curr_steps, 1), 0, 1)
ramp = 0.5 * (1 - cos(pi * fraction))
weight = initial + (final - initial) * ramp
```

The final weights hold from iteration 5000 onward. This schedule is time-based,
not performance-gated. Scheduled values override the corresponding entries in
`rewards.scales` (including the initial −0.001 action penalties). Disabling the
reward curriculum uses the static scales instead. New scheduled reward terms
must also have a nonzero scale so the environment registers their functions.

### Domain randomization curriculum

`domain_rand.use_domainrand_curriculum=True` advances three phases in order:
**joint dynamics → mass/CoM → disturbances**. Each phase interpolates its ranges
linearly from initial to final; earlier phases retain their final ranges.

| Randomization | Initial range | Final range | Parameters controlling endpoints |
| --- | --- | --- | --- |
| Joint friction (N m) | 0–0.05 | 0–0.20 | `joint_friction_range_start/end` |
| Joint stiffness (N m/rad) | 0–0.005 | 0–0.02 | `joint_stiffness_range_start/end` |
| Joint damping (N m s/rad) | 0.20–0.60 | 0–0.80 | `joint_damping_range_start/end` |
| Added torso mass (kg) | −1–2 | −1–4 | `added_mass_min`, `min_added_mass_max`, `max_added_mass_max` |
| CoM X/Y/Z displacement (m) | ±0.05 each | ±0.05 each | `com_displacement_{x,y,z}_min/max` |
| XY velocity impulse per axis (m/s) | ±0.50 | ±1.00 | `min/max_push_vel_xy` |
| Downward velocity impulse (m/s) | −0.10–0 | −0.50–0 | `min/max_vertical_push` |
| Angular velocity impulse per axis (rad/s) | ±0.50 | ±1.00 | `min/max_push_torque` |

Despite the legacy `push_torque` name, the last row changes angular velocity,
not applied torque. XY, downward, and angular events have independent per-robot
5–15 s timers (`push_interval_min/max`, `vert_interval_min/max`,
`wrench_timeout_min/max`). Warmup holds initial disturbance magnitudes; it does
not disable the initial disturbances.

Progression uses newly completed episodes' linear tracking reward:

| Parameter | Default | Effect |
| --- | --- | --- |
| `push_warmup` | 2000 | Hold progression through iteration 2000 |
| `reward_ema_alpha` | 0.05 | Tracking-reward EMA update coefficient |
| `best_reward_window` | 400 | Number of recent valid EMA samples retained |
| `best_reward_quantile` | 0.90 | Recent-performance reference quantile |
| `recovery_ratio` | 0.90 | Required fraction of that reference |
| `min_reward_to_step` | 0.60 | Absolute EMA threshold for progression |
| `step_interval` | 10 | Minimum learning iterations between advances |
| `joint_dynamics_progress_delta` | 0.02 | Increment per successful joint-dynamics advance |
| `mass_com_progress_delta` | 0.01 | Increment per successful payload advance |
| `disturbance_progress_delta` | 0.01 | Increment per successful disturbance advance |

An advance requires EMA ≥ `max(min_reward_to_step, recovery_ratio * reference)`.
Missing/nonfinite episode statistics hold progression. A phase finishes at
progress 1.0. The three `use_*_curriculum` phase flags can skip progression for
individual phases; skipped phases retain initial ranges. Disabling the overall
domain curriculum holds initial ranges while enabled randomizations remain active.
To disable a physical randomization itself, use its `randomize_*` flag; use
`push_robots=False` to disable disturbance events.

Other randomizations have fixed ranges: ground friction 0.2–1.25, Kp and Kd
multipliers 0.8–1.2, motor-strength multiplier 0.9–1.1, joint armature 0–0.015
kg m², and action delay 0–2 control steps (0–40 ms at 50 Hz).
The adapter derives effective mass/CoM/joint-dynamics ranges from the curriculum
endpoints. Edit those endpoints, rather than the legacy `added_mass_range` or
`com_pos_*_range` fields, which the adapter overwrites.

**Episode resampling cadence:** `reset_resample_episodes=25` holds each robot's
friction, mass, CoM, armature, joint friction/damping/stiffness, PD gains, and
motor strength for 25 completed episodes. The first reset initializes sampling
without counting as a completed episode. Values 0 or 1 resample every reset.
A changed curriculum range forces the affected parameter to resample at that
robot's next reset even if its episode interval is not yet due; other parameters
retain their cadence. Control delay and disturbance timers reset every episode.
Runtime disturbances follow their timers rather than the physical-parameter
resampling interval. The same semantics are implemented in IsaacLab and Genesis.

## 5. Validation and common setup issues

The automated checks below use **physical GPU 1**. These command-local device
masks are for validation; they are not needed by the training launcher. In the
masked process, `cuda:0` denotes physical GPU 1. On a single-GPU collaborator
machine, change the mask to 0.

```bash
conda activate lr_lab
CUDA_VISIBLE_DEVICES=1 SIMULATOR=isaaclab python -m pytest tests/test_dreamwaq_training.py -q
CUDA_VISIBLE_DEVICES=1 SIMULATOR=isaaclab python -m legged_gym.scripts.smoke_go2_dreamwaq --headless --gpu cuda:0
```

The integration check uses 16 robots, short episodes, and six learning iterations.
It checks contact/body index mappings, three-episode physical resampling,
accelerated domain/reward schedules, disturbances, finite model/state values,
checkpoint resume, and deliberately saturated torque commands. Look for
`DREAMWAQ_SMOKE_PASSED`; an exit code alone does not establish success because
IsaacLab shutdown can obscure exceptions. This does not test convergence.

| Symptom | What to check |
| --- | --- |
| `conda activate` fails | Initialize Bash with `conda init bash`, open a fresh terminal, and activate `lr_lab` |
| `isaaclab` / `isaacsim` import fails | Verify Python 3.11 and that all installation steps used the active `lr_lab` environment |
| GPU index error | Pass `--gpu cuda:0` on a single-GPU machine; the current script's literal default is `cuda:1` |
| CUDA unavailable | Check `nvidia-smi`, the CUDA PyTorch install, and scheduler device visibility |
| Out of GPU memory | Retry with `--num_envs 16` or 128 before increasing parallelism |
| Wrong runner or missing DreamWaQ optimizer behavior | Check `rsl_rl.__file__` and install this checkout in editable mode |
| No display on a server | Use headless checks and the headless training launcher |
| Incompatible checkpoint | Use a checkpoint from this 11-output, two-optimizer implementation |
| Curriculum does not advance | Inspect tracking EMA, thresholds, warmup, fresh episode statistics, and enabled phase flags |

Genesis is an alternative implemented backend. It requires its own compatible
Genesis environment and dependencies from `pyproject.toml`; the IsaacLab conda
environment above does not install Genesis. With that environment active, select
it using `SIMULATOR=genesis bash legged_gym/scripts/go2_dreamwaq.sh --gpu cuda:0`.

For code navigation: [task and rewards](../legged_gym/envs/go2/go2_dreamwaq/go2_dreamwaq.py),
[VAE](../rsl_rl/modules/vae.py), [actor/critic](../rsl_rl/modules/actor_critic_dreamwaq.py),
[PPO and auxiliary losses](../rsl_rl/algorithms/ppo_dreamwaq.py),
[runner and checkpoints](../rsl_rl/runners/dreamwaq_runner.py),
[domain curriculum](../legged_gym/simulator/go2_domain_rand_curriculum.py), and
[shared backend adapter](../legged_gym/simulator/dreamwaq_adapter.py).
