# Additive deterministic UniFP baselines

The retained `b1z1_unifp` task is unchanged. These tasks match the
[upstream policy architecture](https://github.com/unified-force/UniFP/blob/main/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py),
not every upstream environment/reward setting.

| Component | Both new tasks |
| --- | --- |
| History encoder | 32 x 73 -> 512 -> 256 -> 128 -> 64, deterministic |
| Adaptation decoder | 64 -> 128 -> 64 -> 12 |
| Actor | [current observation, latent], 137 -> 512 -> 256 -> 128 -> 17 |
| Critic | Existing privileged contents, 3 frames; hidden 512/256/128 |
| Hidden activation | ELU; linear output layers |
| Exploration | Learned diagonal standard deviation, initially 1 |
| Decoder labels | Base velocity, EE spherical position, EE force, base force; 3 each |
| Adaptation objective | Four MSEs weighted 0.2/0.2/1/1; Adam LR 1e-5 |

PPO gradients reach the actor and history encoder. The adaptation optimizer owns
only the encoder and decoder. There is no VAE, KL regularization, privileged
reconstruction, spectral normalization, or decoded-estimate actor input.
The existing rollout storage, GAE, timeout handling, runner collection loop,
episode logging, and unrelated PPO hyperparameters are reused.

## Tasks and force behavior

`b1z1_unifp_original` retains independent commanded-force and physical disturbance
streams and the existing force-adjusted targets. The inherited 20-D internal
explicit buffer and privileged contents are untouched; a separate reordered
12-D target is returned to this policy's runner.

`b1z1_unifp_reject` disables independently sampled force commands. Each training
or inference action forward pass caches its decoder prediction. The runner
publishes that prediction before stepping, without a second encoder pass.
EE predictions [6:9] and base predictions [9:12] are unscaled into physical
forces in the existing local/yaw frame. The inherited filter implements
`filtered += dt/(tau+dt) * (estimate-filtered)` and writes `command=-filtered`
into command slots 9:12 and 12:15. This adapter never reads true external forces.
Reset clears estimates and filters for the selected environments.

Exact estimates cancel the external-plus-command target offset after the filter
settles (immediately if tau=0). Transients and newly sampled disturbances cannot
be canceled perfectly by a causal estimator. The existing helper has no separate
command clipping/slew limit; its LPF and existing target displacement/workspace
limits are retained rather than inventing new limits.

Both tasks inherit the same rewards, physical disturbance distribution and
curriculum, control settings, and training budget. Inherited zero force-command
ranges remain zero; this implementation does not silently restore upstream
force ranges or reward settings. Their model/checkpoint schema matches each
other, but loading retained extended/VAE checkpoints fails explicitly.

## Launch

From `legged_gym/scripts`, with conda available:

```bash
sh b1z1_unifp_original.sh
sh b1z1_unifp_reject.sh
```

Defaults are IsaacLab, `lr_lab_cupiqp`, headless, and `cuda:1`.
Use the normal unremapped GPU numbering for these launchers. `SIMULATOR` and
`CONDA_ENV` may select another already-supported B1Z1 UniFP backend/environment;
additional arguments pass to `train.py`. Experiment names are task-specific.
Deployment through `runner.get_inference_policy()` retains the rejection
callback; exporting only the actor would require exporting that adapter too.

## Validation

`tests/test_unifp_original_baselines.py`: five CPU tests pass, including a tiny
synthetic PPO/adaptation update, architecture/gradient checks, decoder independence,
schema/scaling/cancellation/filter/reset checks, one-pass train/inference callbacks,
and configuration/registration isolation. No training run or full suite was run.

`tests/smoke_unifp_original_baseline.py` is an optional one-environment plane-terrain
check with two policy decisions and no learning. The attempted rejection smoke
was blocked at IsaacLab GPU-device initialization (`No device could be created`)
under GPU-1-only visibility, and was terminated. Simulator behavior remains
unverified by this check; the original variant was not separately smoke-tested.

Implementation: two new environment/config directories under `legged_gym/envs/b1z1`,
`actor_critic_unifp_original.py`, `ppo_unifp_original.py`, and
`unifp_original_runner.py`. Existing files have additive task/runner registrations only.
