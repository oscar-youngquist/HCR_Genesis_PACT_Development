# B1Z1 BARD PINNs and optimizer ownership

This physics path is selected by `algorithm.dynamics_backend = "bard"` in coupled
B1Z1 PACT. All PACT backends and PACT-Pos use the same disjoint optimizer ownership.
Pinocchio retains its force formulation but trains it in the auxiliary phases,
not through the actor. PACT-Pos does not gain a PINN objective.

## Physics and gradient contract

`rsl_rl/algorithms/b1z1_bard_pinn.py` caches detached realized mechanics at
`q_t, v_t`, in bounded backend batches, once per PPO rollout update. Epochs
index the cache with the same shuffled indices as the transition storage.
The cache is released after updating. BARD's randomized torso and gripper
inertias remain enabled.

With 25 generalized coordinates (6 floating-base + 19 joints), define:

```
a_obs = (v_next - v_pre) / dt
g = S.T @ torque_interval + sum(J_foot.T @ F_pred)
    + J_base.T @ W_applied_pred + J_EE_linear.T @ F_EE_pred
r_ID = M @ a_obs + bias - g
delta_v_pred = dt * solve(M, g - bias)
```

`M`, bias, Jacobians, executed torque, time and measured states are constants
for differentiation. Only predicted GRFs, EE forces and torso wrench carry
gradients, including gradients through their latent input. The analytic solve
has an RHS-only derivative `M^-T`; no differentiable ABA rollout is built.
The GRF head receives detached, bounded nominal torque computed before stepping,
not the future executed torque label. Explicit conditioning is also detached.

The inverse loss reuses HardPACT's contact-weighted relative norm:

```
c = sum(J_foot.T @ F_measured)
w = positive(c) / (max(positive(c)) + 1e-8)
L_ID = mean(norm(w * r_ID) / (norm(S.T @ torque_interval) + norm(c) + 1e-8))
```

The rollout reuses HardPACT's three equally weighted coordinate blocks:
base linear, base angular, and **all 19 joints**. Their increment scales are
`dt * [10, 20, 100]`. Each block contributes normalized residual RMS divided
by `1 + detached normalized observed-increment RMS`. Arm and leg joint counts
do not change the total weight assigned to the joint block.

The shared functions in `hard_pact_bard.py` accept an optional extra generalized
force for the EE. Existing Go2 callers keep the original behavior.

## Timing and mass labels

Simulators record control-interval average executed torque. For the BARD runner
only, a separate accumulator records deadbanded/clipped interval GRFs without
changing the observation EMA. These values supervise the GRF decoder and form
the inverse loss's measured-contact normalizer. The existing backend contact
measurement source is retained; this is not a new force-sensor implementation.
GRFs remain stored in successor yaw coordinates, with that same yaw used to
recover world forces. Current wrench predictions use pre-step yaw.

Torso-wrench targets include a label-only added-load contribution:

```
W_mass = [delta_m_base * gravity, r_com x (delta_m_base * gravity)]
       + [delta_m_gripper * gravity, r_EE x (delta_m_gripper * gravity)]
W_total_label = W_external + W_mass
W_applied_pred = W_total_pred - W_mass
```

The gripper contribution is a task-level equivalent load represented at the EE
reference point, not a claim that the gripper's inertial CoM is exactly there.
Subtracting the identical recorded label prevents double counting, while the
actual randomized gripper inertia stays in BARD. After subtraction, the applied
torso wrench is shifted from randomized torso CoM to the base Jacobian origin,
as in HardPACT. No additional mass force is applied in simulation.

Reset rows are excluded before auxiliary arithmetic. Large position discontinuities
are flagged by the runner and excluded from physics. This is a conservative
teleport check, not HardPACT's complete named push/teleport event instrumentation.
Current B1Z1 sustained-force tasks are not excluded as impulsive pushes.

## Updates and PCGrad

Three disjoint AdamW owners follow HardPACT:

1. Actor/critic: PPO only; no BARD PINN torque gradient.
2. Context encoder: reconstruction plus KL, with all
   decoder parameters frozen; a separate encoder PINN objective trains the latent.
3. Explicit-state, force, GRF and basic privileged decoders: reconstruction plus
   a separate decoder PINN objective, using the same detached latent sample.
   Explicit predictions are recomputed for supervision but detached when
   conditioning force/GRF heads.

Each auxiliary phase has its own owned-parameter PCGrad instance. It combines
the supervised objective with the weighted sum of inverse and enabled rollout
PINNs. Positive `pinn_loss_weight` selects `pc_backward_pinn`; negative selects
`pc_backward_ppgrad`. Its absolute value sets the final scheduled magnitude;
zero disables the PINNs. Both encoder and decoder phases use the same selection,
and neither maximizes the physics residual. Both phase gradients are
computed before stepping either optimizer. Each owned group is clipped separately.
The existing B1Z1 PINN warmup and rollout enable/weight settings are retained.
The phase's physics objective is
`scheduled_pinn_weight * (pinn_inverse_weight * inverse_loss + pinn_rollout_weight * rollout_loss)`.
Both component weights default to 1.0; the inverse default also applies to older
algorithm dictionaries without that key. Raw component logs remain unweighted;
the aggregate PINN log includes component weights but not the outer schedule.
PPO conditioning is detached, preventing PPO gradients from reaching any auxiliary
parameters. Constructor checks enforce complete, pairwise-disjoint ownership.
The existing prediction reliability gate does not mix measured forces into this
BARD objective; learned physical predictions enter directly as in HardPACT.

Logs include `pinn_encoder_inverse`, `pinn_encoder_rollout`,
`pinn_decoder_inverse`, and `pinn_decoder_rollout`. Aggregate PINN logs average
the two phases. Decoder optimizer state is saved separately. Old checkpoints
without optimizer partition version 2 restore model weights but start all optimizers fresh
with a warning; old wrench heads also need adaptation to the new total-load label.

## Validation and boundaries

### TensorBoard PINN metrics

`PINN/inverse/loss_raw` and `PINN/rollout/loss_raw` report normalized,
unweighted BARD objectives. Physical MAEs report the unweighted residual:

For each component, `loss_unscaled` aliases `loss_raw`;
`loss_component_weighted` multiplies by `pinn_inverse_weight` or
`pinn_rollout_weight`; `loss_scaled` additionally multiplies by the scheduled
absolute `pinn_loss_weight`. These are the scalar objectives supplied to PCGrad,
not post-projection gradient magnitudes. "Unscaled" does not undo the dynamics
loss normalization; physical errors are reported separately by the MAE tags.

- `PINN/inverse/base_linear_mae_N`: torso force residual [N].
- `PINN/inverse/base_angular_mae_Nm`: torso moment residual [N m].
- `PINN/inverse/legs_mae_Nm` and `arm_gripper_mae_Nm`: joint torque residuals.
- `PINN/rollout/base_linear_mae_mps`: linear velocity prediction error [m/s].
- `PINN/rollout/base_angular_mae_radps`, `legs_mae_radps`, and
  `arm_gripper_mae_radps`: angular/joint velocity prediction errors [rad/s].

The same metrics appear under `PINN/encoder/...` and `PINN/decoder/...`.
They are detached, valid-sample-weighted averages across minibatches/epochs.
No extra BARD evaluations are performed. These core PINN metrics are not
suppressed by the additional-diagnostics flag. They are emitted only when the
corresponding objective is evaluated on valid samples; warmup/disabled rollout
does not generate a fictitious zero physical error.

`PINN/combined_loss` retains the existing component-weighted phase/minibatch
average, and `PINN/weighted_combined_loss` additionally applies the scheduled
outer magnitude. Component weights, rollout enable state, and scheduled weight
are also logged. Existing `Loss/pinn_*` tags remain available.

Focused CPU tests cover analytic force gradients, detached torque/explicit inputs,
mass-label cancellation, disjoint ownership, and a short active-PINN PPO update.
Small BARD/Pinocchio numerical and CUDA gradient tests run on physical GPU 1.
Those tests exposed and fixed workspace allocation before model dtype/device
conversion. Full simulator training convergence is not established by these tests.

This change retains B1Z1's existing realized-mechanics model. It does not add
HardPACT's joint-passive-parameter calibration or QP/control projection system.
Backend-specific contact measurement limitations remain, including Isaac Gym's
existing net-contact-force source. A physics-engine trajectory is not expected
to exactly equal one fixed-mechanics control-interval step.
