# Go2 DreamWaQ training

Activate the existing IsaacLab environment and launch from the repository root:

```bash
conda activate lr_lab
bash legged_gym/scripts/go2_dreamwaq.sh
```

The launcher selects IsaacLab and sets `--gpu cuda:0` directly in the script.
To change GPUs, edit that argument or pass `--gpu cuda:1` to the launcher.
Extra training arguments are forwarded, for example
`--num_envs 16 --max_iterations 2`. Genesis remains available through
`SIMULATOR=genesis` with its corresponding environment.

## Estimator and optimization

The history encoder predicts an implicit Gaussian latent (16 dimensions) and
11 deterministic explicit values, in this order:

| Slice | Estimate | Target / units |
| --- | --- | --- |
| `0:3` | Torso linear velocity | Body frame, multiplied by `obs_scales.lin_vel` |
| `3:7` | Foot-contact probability | FR, FL, RR, RL; force norm above 1 N |
| `7:11` | Foot clearance | Foot world Z minus mean local terrain height and foot-radius offset, metres |

The contact head produces logits. The actor and reconstruction decoder receive
`epsilon + (1 - 2*epsilon) * sigmoid(logit)`, with epsilon `1e-6` by default.
Continuous heads are unconstrained. Explicit targets are detached. These
conventions follow the explicit estimator and masked-loss implementations in
`aligned_iclr_2027_qp_pinn` (`hard_pact_physics.py`, `ppo_hard_pact.py`).

One Adam optimizer owns the complete VAE: encoder, explicit head, latent
mean/log-variance heads, and decoder. Its objective is next-observation MSE +
`vae_kld_weight * KL` + `explicit_loss_weight * explicit_loss`. Explicit loss is
the mean squared error over the seven continuous values plus
`contact_probability_loss_weight * BCEWithLogits`, averaged over the four feet.
Both explicit-loss weights default to 1; KL weight defaults to 2.

A separate Adam optimizer owns the actor, critic, and action standard deviation.
Only the locomotion PPO objective updates it. Actor features are computed under
`no_grad`, so PPO does not backpropagate into the VAE. The implicit latent is
sampled during training and its mean is used for inference/export.

Each rollout stores history and labels from time t and clean proprioceptive
reconstruction targets from t+1. Auto-reset transitions are excluded from the
auxiliary losses. Valid rows are selected before arithmetic and normalized by
the valid count, so masked NaNs cannot contaminate the losses. An all-terminal
minibatch skips the VAE optimizer step, including its Adam momentum update.

## Contact indexing

IsaacLab articulation bodies and contact sensors have independent name orderings.
`feet_indices` addresses body positions/velocities; `feet_contact_indices`
addresses raw sensor forces. Both resolve the configured FR, FL, RR, RL names
independently. Contact-state, collision-penalty, and termination indices also
resolve against sensor names. The public force tensor stays in sensor order.
Missing or ambiguous foot/contact-state names fail at initialization. Regression
tests deliberately permute the two orders, and the IsaacLab smoke check verifies
the real mappings and labels after reset, training, and resume.

## Curricula

The domain curriculum follows Go2 PACT's three phases: joint dynamics, mass/CoM,
then disturbances. It starts with a 2,000-iteration warmup and uses tracking
reward EMA (`alpha=0.05`), a 400-entry history, a 90th-percentile reference,
90% recovery threshold, minimum tracking reward 0.6, and a minimum 10-iteration
step interval. Per-step progression is 0.02 / 0.01 / 0.01 for the three phases.
Missing or nonfinite episode statistics hold the curriculum. Only newly
completed episodes supply statistics, including when logging is disabled.

| Randomization | Initial range | Final range |
| --- | --- | --- |
| Joint friction | 0–0.05 | 0–0.20 |
| Joint stiffness | 0–0.005 | 0–0.02 |
| Joint damping | 0.20–0.60 | 0–0.80 |
| Added torso mass | −1–2 kg | −1–3 kg |
| CoM displacement X / Y / Z | ±0.05 m | ±0.05 m |
| XY velocity impulse, per axis | ±0.50 m/s | ±1.20 m/s |
| Downward velocity impulse | −0.10–0 m/s | −0.50–0 m/s |
| Angular velocity impulse, per axis | ±0.50 rad/s | ±1.50 rad/s |

Ground friction (0.2–1.25), PD gain multipliers (0.8–1.2), motor strength
(0.9–1.1), armature (0–0.015), and control delay (0–2 control steps) are also
randomized. Physical parameters use `domain_rand.reset_resample_episodes`, which
now defaults to **100**, matching the aligned branch's position-control task.
Each environment retains its friction, mass, CoM, armature, joint friction,
damping, stiffness, PD gains, and motor strength for that many completed episodes.
The first reset initializes sampling and does not count as a completed episode.
Set the interval to `0` or `1` to resample on every reset. Changed curriculum
ranges force the affected parameters to resample at each environment's next
reset, even before the interval expires; unchanged parameters retain their cadence.
This applies to both IsaacLab and Genesis. Control delay and disturbance timers
still reset each episode, matching the aligned branch. Disturbances use
independent per-environment 5–15 second event timers. Both backend adapters
apply the configured ranges; the controller remains a 12-output position policy.

The reward curriculum holds its initial weights for 6,000 iterations, uses a
500-iteration cosine ramp, then holds the final weights:

| Reward | Initial → final |
| --- | --- |
| `ang_vel_xy` | −0.05 → −0.2 |
| `orientation` | −0.2 → −2.0 |
| `torque_limits` | −0.01 → −1.0 |
| `hip_pos` | −0.2 → −0.4 |
| `action_rate` | −0.001 → −0.01 |
| `action_smoothness` | −0.001 → −0.01 |

The torque-limit penalty starts at 90% of the physical motor limit.
Weights are multiplied by the control timestep once. Position-action terms
correspond to PACT's position branch; there are no separate torque-action terms.
The default training duration is 10,000 iterations. TensorBoard records each
explicit loss, reconstruction/KL losses, curriculum progress, active ranges,
and reward weights.

## Checkpoints and validation

New runs default to `resume=False`. The 11-output architecture is incompatible
with old 24-output DreamWaQ checkpoints; loading one produces an explicit error.
New checkpoints contain both optimizer states, the adaptive PPO learning rate,
the next training iteration, and domain-curriculum progress/reward history.
Reward weights are reconstructed from the configured schedule and that iteration.
Resuming resets the per-environment cadence and samples all restored ranges; simulator trajectories
are not restored bit for bit. Resume with the same curriculum configuration.

```bash
CUDA_VISIBLE_DEVICES=1 SIMULATOR=isaaclab python -m pytest tests/test_dreamwaq_training.py -q
CUDA_VISIBLE_DEVICES=1 SIMULATOR=isaaclab python -m legged_gym.scripts.smoke_go2_dreamwaq --headless --gpu cuda:0
```

The smoke check uses 16 environments, rough terrain, and six training iterations.
It first checks that physical parameters persist for a three-episode test interval
and resample at its boundary. After two training iterations, it accelerates the schedules
only for the test, exercises all phases and a simultaneous disturbance, checks
finite states/parameters, saves and reloads both optimizers, and resumes training.
It prints `DREAMWAQ_SMOKE_PASSED` on success. This checks integration, not policy
convergence.

Torque handling follows aligned PACT position control on both backends. The
execution action is clipped and delayed, then converted to PD torque using the
live state at every physics substep. Motor strength is applied once, followed by
clipping to the physical joint effort limits. IsaacLab limits use the minimum of
actuator and simulation limits, reordered into policy joint order. There is no
additional torque-rate limiter.

The torque-limit penalty uses a separate **unclipped, delayed** action request,
converted to torque before actuator saturation. This preserves penalties for
excessive commands even when action clipping or torque saturation bounds execution.
The raw queue shares execution's per-environment delay and clears on reset.
Torque magnitude and power rewards use the bounded commanded torque, matching
aligned PACT; IsaacLab's motor model may further limit the physically applied torque.

The Go2 environment and PPO configurations inherit directly from `LeggedRobotCfg`
and `LeggedRobotCfgPPO`. Go2 terrain, asset, control, reward, and DreamWaQ runner
settings are defined locally in `go2_dreamwaq_config.py`, without the shared
DreamWaQ or Go2 common configuration classes.

Additional locomotion penalties are enabled in `rewards.scales`:

- `dof_vel_limits = -1.0`: sum of joint speed excess above
  `soft_dof_vel_limit = 0.9` times the physical velocity limit, capped at 1 rad/s
  per joint. IsaacLab uses the smaller actuator/solver limit in policy order.
- `front_foot_overreach = -10000.0`: PACT's squared excess beyond torso-frame
  x = 0.28 m, including its payload-dependent scaling.
- `rear_foot_overreach = -10.0`: PACT's squared excess outside a ±0.08 m band
  around torso-frame x = -0.25 m.

Both overreach penalties require upward contact force greater than 5 N and use
contact-sensor indices independently of articulation body indices. Their weights
are fixed; the existing reward curriculum continues to schedule its listed terms.
