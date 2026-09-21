# B1Z1 PACT sampled context and timing

Both PACT variants sample `z = mean + exp(0.5 * logvar) * epsilon`.
The actor and all decoder branches share that sample within a forward pass.
PPO stores epsilon and reuses it during distribution reconstruction, rather
than drawing a different context for the stored action. Auxiliary optimization
uses a fresh sample after the PPO step. Deployment also samples the context.

## Decoder contracts

| Branch | Inputs | Target |
| --- | --- | --- |
| Explicit | z | Velocity(3), spherical EE position(3), contacts(4), foot heights(4) at t |
| External force | z, detached explicit prediction | Normalized base wrench(6), EE force(3) at t |
| GRF | z, detached explicit prediction, detached commanded torque / 100 | Normalized foot GRFs(12) at t+1 |
| Basic privileged | z | Remaining single-frame privileged fields at t+1 |

Contact supervision uses BCE with logits; downstream explicit conditioning
uses probabilities. Force supervision retains the configured physical-value
normalization. `grf_torque_scale` controls the torque divisor (default 100).
The basic decoder excludes the leading explicit and force blocks (44 values
with the current layout); critic observations and environment labels are unchanged.

## Transition alignment

- Capture current labels, observations, history, and latent noise before stepping.
- Capture PACT-Pos cloning gains/defaults before stepping, before reset randomization.
- Record the simulator's final-substep executed total torque for GRF conditioning.
- Store successor GRFs and privileged targets after stepping.
- Exclude terminal/reset rows before auxiliary decoder arithmetic.
- In coupled PACT, mirror the environment's delay queue with source observation,
  history, latent noise, and Gaussian action noise. Replay that source for physics
  losses while PPO still uses the current transition's policy distribution.
- Keep the source queue across rollout boundaries; clear individual environments
  on reset. Initial zero-action delay slots have no policy source and are excluded
  from PINN supervision.
- Reconstruct controller feedback using q_t and v_t, not the post-step state.
- Rotate predicted current external wrenches with the torso yaw at t, but predicted
  next GRFs with the torso yaw at t+1 before converting to world-frame SI forces.

This does not make inverse dynamics an exact simulator replay. Its acceleration
is a control-interval finite difference; controller reconstruction evaluates at
the start of the interval, while feedback evolves during decimation. GRF input
uses the final torque command, not an interval-average work-equivalent torque.
Existing torque-limit reconstruction and solver/contact approximations remain.

## Compatibility and validation

The explicit decoder output and basic privileged decoder output have changed,
and two decoder branches were added. Old model/optimizer checkpoints are not
shape-compatible. The coupled configuration no longer automatically loads an
old monolithic-decoder pretraining checkpoint; use a matching new PACT-Pos run.

CPU tests cover replayed PPO likelihoods, detached decoder conditioning, torque
normalization, both short PPO updates, invalid auxiliary targets, delayed-source
selection, reset clearing, and replay RNG preservation. No simulator/GPU smoke
test is implied by these unit tests.
