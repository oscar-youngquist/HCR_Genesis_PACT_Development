# Actor-facing task-predictive physics loss

This optional objective applies only to coupled B1Z1 PACT. PACT-Pos is unchanged.
Enable `algorithm.actor_phys_enabled` with a positive `actor_phys_coef` and
`algorithm.dynamics_backend = "bard"`. It is disabled by default. A disabled flag
or zero coefficient bypasses capture, physics evaluation, extra backward passes,
and logging; the previous PPO backward path is retained.

## Purpose and gradients

The existing observed-transition PINNs train representations to explain what
occurred. This objective instead trains actions whose predicted consequences
track commands and respect joint limits. It never compares a prediction with
observed `x_next`. Both observed rollout and inverse-dynamics losses are unchanged.

The current actor's mean `[q_des, tau_ff]` passes through the existing
`PPO_B1Z1PACT._coupled_torque` mapping, including action scales, clipping,
randomized gains, motor strengths, branch weights and unlearned-joint PD.
Torque is bounded at the existing BARD nominal-command limit, 1.1 times the
simulator torque limit. Force/contact context and dynamics parameters are
detached. The decoder's GRF, base wrench and EE force predictions are converted
back to physical units and world coordinates using the pre-action orientation.
The label-only mass wrench is subtracted exactly once; randomized mass remains
in the mechanics. There is no new force-accuracy gate unless explicitly enabled.

Gradients flow through the fixed-mechanics dynamics solve, pose integration,
EE FK and internal PD mapping to actor parameters. Decoder and estimator heads
receive none. The existing PPO context detachment also keeps this objective out
of the history encoder; optimizer ownership is unchanged.

## Prediction and references

The velocity predictor is the existing analytic BARD one-step rollout:

    v_hat_next = v_t + dt * M_t^-1 * (S^T tau + J_f^T F + J_b^T W + J_ee^T F_ee - h_t)

It uses fixed measured pre-state mechanics, not a fresh ABA or simulator-substep
rollout. The existing model predicts velocity only. For this objective, pose is
extended with semi-implicit `q_next = q_t + dt*v_hat_next`: joint/base positions
and a world-angular-velocity quaternion exponential. BARD FK of that pose supplies
the world EE position. This pose extension is not inserted into the observed
rollout loss. As with the existing analytic PINN, this is a local predictor,
not an exact reproduction of substep collision, delay, or solver behavior.

Planar velocity and yaw rate are expressed in the measured pre-step body frame:

    u_ref = u_t + (1 - exp(-dt / T_v)) * (u_command - u_t)

The EE target is snapshotted before simulation: advance the existing quintic
spherical schedule by one timestep, transform through the current yaw-aligned
workspace center, and apply `_compute_force_adjusted_ee_target` using the detached
rollout estimator's EE force. This reuses its positive signed `force / Kp` offset,
norm cap and workspace projection. No measured EE force enters this target.
The current helper has no temporal force filter; none is invented here.
The target is frozen in that pre-action world frame throughout PPO, rather than
moving with an observed or action-dependent successor torso pose. Goal-resampling
boundaries are excluded because the next random target is not yet known.

## Costs and defaults

    L_task = w_vel L_vel + w_ee L_ee + w_q L_q + w_qd L_qd

Velocity and EE costs use Huber losses. Velocity uses existing observation scales;
EE error is divided by `actor_phys_ee_scale`. Position barriers penalize both
sides of `[q_min + margin, q_max - margin]`; velocity barriers penalize
`abs(qd) - (qd_max - margin)`. Each uses
`[temperature * softplus(normalized_violation / temperature)]^2`.
Softplus is smooth and slightly positive inside the boundary, not a hard hinge.
Each component averages its coordinates and then valid samples.

| Parameter suffix (`actor_phys_`) | Default | Meaning |
| --- | --- | --- |
| `enabled`, `coef` | false, 0.01 | Enable and outer actor weight |
| `vel_weight`, `ee_weight` | 1, 1 | Tracking weights |
| `q_weight`, `qd_weight` | 0.1, 0.1 | Barrier weights |
| `velocity_time_constant` | 0.25 s | Reachable velocity response |
| `q_margin`, `qd_margin` | 0.05 rad, 0.5 rad/s | Safety margins |
| `softplus_temperature`, `huber_delta` | 0.05, 1 | Normalized smoothing |
| `ee_scale`, `q_scale`, `qd_scale` | 0.1 m, 1 rad, 10 rad/s | Normalizers |
| `require_force_gate` | false | Require existing representation-quality gate |

The actor shares `pinn_init_steps`, `pinn_warmup`, and the checkpoint-restored
`pinn_updates` counter with the representation PINNs. Its effective coefficient
is `actor_phys_coef * pinn_weight / abs(pinn_loss_weight)`, so it ramps from zero
to its own configured maximum. A zero `pinn_loss_weight` disables it as well.
No extra mechanics or actor-physics backward is performed while this weight is zero.

The actor PCGrad path receives `[L_PPO, effective_coefficient * L_task]` through
the same helper as the encoder/decoder PINNs: positive `pinn_loss_weight` selects
`pc_backward_pinn`; negative selects the norm-bounded `pc_backward_ppgrad`.
The sign never makes the task loss negative. These are the input objectives, not a
promise that projection leaves their unmodified summed gradient. Invalid/reset/
teleport samples, nonfinite inputs/predictions and singular mechanics are excluded.
An entirely inactive minibatch falls back to the original PPO backward operation.

## Storage and logging

`capture()` stores pre-action state, commands, detached next EE reference, limits,
mass label and goal validity in optional rollout fields. Storage copies these
tensors and shuffles them with the same indices as PPO. They do not alter model
or checkpoint layouts. Mechanics are cached per rollout only when enabled.

TensorBoard `ActorPhysics/` contains `loss`, `loss_scaled`, `vel`, `ee`, `q`, `qd`,
`valid_fraction`, `active_fraction`, `velocity_error`, `ee_error_m`,
`position_violation_rad`, `velocity_violation_radps`, `actor_gradient_norm`,
`ppo_gradient_cosine`, `scheduled_coefficient`, and `unintended_estimator_gradient_max`. The last is a
loss-specific diagnostic VJP, not accumulated auxiliary `.grad` buffers, and
should be zero. Velocity error combines planar m/s and yaw rad/s as a diagnostic;
only the normalized Huber values enter the objective. Gradient norm/cosine are
measured before coefficient scaling and projection. Logs average minibatches.

Focused tests exercise gradients, masks, limits, disabled behavior, successor
independence, snapshot alignment, and a complete synthetic PPO update. The real
BARD GPU FK test additionally checks its joint derivative against the EE Jacobian.
