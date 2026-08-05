# B1/Z1 PACT Physics Objectives

## Purpose and Scope

This document explains four complementary objectives for the coupled-position/
torque B1/Z1 policy:

1. The whole-body inverse-dynamics PINN loss.
2. A proposed one-step whole-body rollout PINN loss.
3. The UniFP-inspired soft impedance-consistency reward.
4. The per-joint torque-cancellation reward.

The discussion assumes that the rigid-body dynamics implementation is batched,
accurate enough for training, and preserves all useful gradients. It therefore
focuses on the learning formulation rather than a particular dynamics library.

The inverse-dynamics and rollout terms are differentiable auxiliary losses. The
impedance and cancellation terms are environment rewards: they influence the
policy through PPO returns and advantages rather than by differentiating
through the reward calculation itself.

## Notation

Let

- `q` be the floating-base configuration and all robot joint positions.
- `v` be the generalized velocity.
- `vdot` be the generalized acceleration.
- `M(q)` be the whole-body mass matrix.
- `h(q, v)` contain gravity, Coriolis, and centrifugal effects.
- `S` be the actuator-selection matrix.
- `tau` be the vector of actuator torques.
- `J_f` be the stacked foot-contact Jacobian.
- `F_grf` be the four ground-reaction-force vectors.
- `J_ee` and `F_ee` be the EE Jacobian and external EE force.
- `J_b` and `W_b` be the base Jacobian and external base wrench.
- `x`, `xdot`, and `xddot` be EE Cartesian position, velocity, and acceleration.
- `x_d`, `xdot_d`, and `xddot_d` be the corresponding moving EE target terms.

For B1/Z1, `v` contains the six unactuated floating-base coordinates followed
by all leg and arm joint velocities. The generalized actuator force is

```math
S^T \tau = \begin{bmatrix}0_6 \\ \tau_{joints}\end{bmatrix}.
```

The policy emits two learned action heads:

```math
a_t = [a_t^{pos}, a_t^{ff}].
```

The position head defines a PD target,

```math
q_t^{des} = q^{default} + s_{pos} a_t^{pos},
```

and the two branches produce

```math
\tau_{PD} = K_p(q_t^{des} - q_t) - K_d \dot q_t,
```

```math
\tau_{FF} = s_{tau} a_t^{ff}.
```

After the configured branch weights and actuator randomization are applied,
the whole-body losses use the same combined torque sent to the robot:

```math
\tau_\pi = w_{PD}\tau_{PD} + w_{FF}\tau_{FF}.
```

## Whole-Body Inverse-Dynamics PINN Loss

### Theory

Rigid-body force balance is

```math
M(q)\dot v + h(q,v)
= S^T\tau
+ J_f(q)^T F_{grf}
+ J_{ee}(q)^T F_{ee}
+ J_b(q)^T W_b.
```

The left side is the generalized force required to produce the observed
whole-body acceleration. The right side explains that force using actuator
torques and external forces. Moving everything to one side gives the residual

```math
r_{ID} = M(q)\dot v^{obs} + h(q,v)
- S^T\tau_\pi
- J_f(q)^T \hat F_{grf}
- J_{ee}(q)^T \hat F_{ee}
- J_b(q)^T \hat W_b.
```

Here the forces may be measured labels or predictions from the context and
force decoders. The observed acceleration is commonly estimated from the
transition:

```math
\dot v_t^{obs} = \frac{v_{t+1}^{obs} - v_t^{obs}}{\Delta t}.
```

A basic inverse-dynamics loss is

```math
L_{ID} = \frac{1}{B}\sum_{i=1}^{B}
\left\|W_r r_{ID,i}\right\|_2^2,
```

where `W_r` balances base, leg, and arm residual coordinates. A relative form
can normalize each sample by a detached torque or generalized-force scale:

```math
L_{ID}^{rel} = \frac{1}{B}\sum_i
\frac{\left\|W_r r_{ID,i}\right\|_2}
{\left\|S^T\tau_{\pi,i}\right\|_2 + \epsilon}.
```

Reset and invalid transitions must be masked because their stored next states
may no longer belong to the action that preceded them.

### Gradient Meaning

With predicted forces and an intact graph, the loss supplies gradients through
both the action and force-estimation pathways:

```text
inverse-dynamics residual
  +-> combined policy torque -> position head, torque head, actor/context
  +-> predicted GRFs         -> force decoder -> context encoder
  +-> predicted EE force     -> force decoder/explicit estimator -> encoder
  +-> predicted base wrench  -> explicit estimator -> encoder
```

For a predicted external force, the local derivative has the form

```math
\frac{\partial L_{ID}}{\partial \hat F}
= -J(q)\frac{\partial L_{ID}}{\partial r_{ID}}.
```

The six floating-base residual coordinates are particularly valuable because
they contain no direct actuator torque. The network must explain torso motion
through contact forces, external disturbances, gravity, and dynamic coupling.
The arm and leg rows then constrain whether those same force predictions are
consistent with the corresponding joint motion.

### Motivation

The inverse-dynamics objective encourages the policy to learn:

- Coupled position and torque actions that explain observed acceleration.
- GRF, EE-force, and base-wrench estimates with physically credible effects.
- Compensation for reaction forces transmitted between the Z1 arm and B1.
- Torque commands compatible with randomized inertia and actuator properties.
- A whole-body explanation instead of independent leg and arm explanations.

It does not decide whether a transition is useful. A physically consistent
fall is still a fall. PPO rewards remain responsible for command tracking,
stability, and task performance.

### Ambiguities

The residual constrains the sum of generalized forces. Incorrect GRF, EE-force,
and base-wrench predictions can sometimes compensate for one another. Separate
supervised reconstruction losses are therefore needed to identify each force
source rather than merely finding any combination that balances the equation.

Similarly, if two position/torque-head combinations generate the same final
`tau_pi`, inverse dynamics cannot determine which decomposition is preferable.

## One-Step BARD Rollout PINN Loss

### Theory

The rollout objective evaluates the same dynamics in the forward direction.
Given the observed current state and the policy's combined torque, predict the
generalized acceleration:

```math
\dot v_t^{pred} = M(q_t)^{-1}\left(
S^T\tau_{\pi,t}
+ J_f(q_t)^T\hat F_{grf,t}
+ J_{ee}(q_t)^T\hat F_{ee,t}
+ J_b(q_t)^T\hat W_{b,t}
- h(q_t,v_t)\right).
```

The minimal one-step version integrates velocity:

```math
v_{t+1}^{pred} = v_t + \Delta t\dot v_t^{pred}.
```

It then compares the complete predicted generalized velocity with Genesis:

```math
L_{roll,v} =
w_{b,lin}\|\hat v_{b,lin}-v_{b,lin}^{obs}\|^2
+w_{b,ang}\|\hat v_{b,ang}-v_{b,ang}^{obs}\|^2
+w_{leg}\|\hat v_{leg}-v_{leg}^{obs}\|^2
+w_{arm}\|\hat v_{arm}-v_{arm}^{obs}\|^2.
```

The base, leg, and arm blocks should be normalized and logged separately. A
single unscaled MSE mixes linear velocity, angular velocity, and joint velocity
units and can allow the largest numerical block to dominate.

A later extension may also integrate the configuration:

```math
q_{t+1}^{pred} = integrate(q_t, \Delta t v_{t+1}^{pred}),
```

and add translation, orientation, and joint-position errors. Floating-base
orientation requires manifold-aware quaternion or rotation integration and a
geodesic orientation error, not component-wise quaternion subtraction.

### Relation to Inverse Dynamics

With an exact model and one-step Euler integration,

```math
v_{t+1}^{obs} - v_{t+1}^{pred}
= \Delta t M(q)^{-1} r_{ID}.
```

Therefore,

```math
L_{roll,v} \approx \Delta t^2\|M(q)^{-1}r_{ID}\|^2.
```

The two losses enforce nearly the same physical law using different metrics:

- Inverse dynamics measures unexplained generalized force.
- Rollout measures the state-transition error caused by that force imbalance.
- Inverse dynamics emphasizes force balance, including high-inertia modes.
- Rollout emphasizes errors that create observable motion through `M^-1`.

They are complementary forms of conditioning, but they are not independent
physical laws. Excessive weights can double-count the same model mismatch.

### Motivation

The rollout loss encourages the policy to learn actions whose predicted causal
consequences match the next observed B1/Z1 state. In particular, it promotes:

- Torques that predict the resulting base, leg, and arm motion.
- Force estimates that explain the next transition rather than only matching
  labels independently.
- Awareness that arm acceleration can disturb the torso and foot loading.
- Awareness that foot contact and base disturbances affect arm tracking.
- A locally accurate action-to-motion relationship useful under disturbance.

Unlike PPO, the rollout objective does not prefer forward motion or successful
EE tracking. It only asks that the proposed action and forces correctly explain
what happened.

### Transition-Alignment Caveat

The observed transition was generated by the sampled rollout action under the
behavior policy. During PPO optimization, the current deterministic action mean
may have changed. The rollout loss can therefore partially behave like a
physics-weighted imitation of the stored transition. A modest weight, delayed
activation, valid-transition masking, and PPO's trust-region behavior reduce
this mismatch.

## Soft Impedance-Consistency Reward

### Theory

The impedance reward retains UniFP's combined force/position concept without
using an external impedance controller to modify the command. In the
yaw-aligned Z1 workspace, define

```math
r_{imp} = M_v(\ddot x_d-\ddot x)
+D_v(\dot x_d-\dot x)
+K_v(x_d-x)
-F_{ext}.
```

`M_v`, `D_v`, and `K_v` are configurable virtual Cartesian mass, damping, and
stiffness. EE and target velocities and accelerations are estimated with finite
differences and filtered to reduce noise. The bounded reward is

```math
R_{imp} = \exp\left(
-\frac{\|W_{imp}r_{imp}\|_2^2}{\sigma_{imp}}
\right).
```

It approaches one when the observed EE behavior satisfies the selected virtual
impedance relation and approaches zero as the mismatch grows.

### Motivation

This reward asks:

> Does the EE respond to tracking error and external force like the desired
> virtual mass-spring-damper system?

Under an external force, it rewards a controlled combination of displacement,
velocity, and acceleration rather than requiring rigid zero-error tracking at
all costs. Without external force, it favors smooth convergence to the moving
EE target.

The equation is not used to shift `x_d`, overwrite an action, or create a force
command. The actor must discover position and torque actions that realize the
desired response. This preserves direct coupled control while retaining the
force/position behavior of the UniFP baseline.

### Effect on the Two Action Heads

The reward constrains the resulting EE trajectory, not the individual action
heads. PPO may learn a useful division such as:

- The position head describes the slower equilibrium or arm trajectory.
- The torque head compensates gravity, coupling, and rapid disturbances.

However, this allocation is not guaranteed. Different head outputs can create
the same total torque and EE motion. The cancellation reward addresses one
specific undesirable decomposition: large same-joint outputs that fight each
other.

## Per-Joint Torque-Cancellation Reward

### Theory

Let the effective, post-blend torque contributions for learned joint `j` be

```math
\tilde\tau_{PD,j}=w_{PD}\tau_{PD,j}, \qquad
\tilde\tau_{FF,j}=w_{FF}\tau_{FF,j}.
```

Define same-joint cancellation as

```math
c_j = |\tilde\tau_{PD,j}| + |\tilde\tau_{FF,j}|
-|\tilde\tau_{PD,j}+\tilde\tau_{FF,j}|.
```

This identity has useful exact behavior:

- Same signs: `c_j = 0`.
- Either contribution is zero: `c_j = 0`.
- Opposite signs: `c_j = 2 min(|tau_PD,j|, |tau_FF,j|)`.
- Equal and opposite contributions: the full duplicated effort is exposed.

Normalize by the joint torque limit and apply a deadband `delta`:

```math
\bar c_j = \max\left(
\frac{c_j}{\tau_j^{limit}}-\delta, 0
\right).
```

The implemented penalty is

```math
C_{cancel}=\frac{1}{N_a}\sum_{j=1}^{N_a}\bar c_j^2,
```

and the environment contributes

```math
R_{cancel}=-\lambda_{cancel}C_{cancel}.
```

The current B1/Z1 configuration uses a `0.03` torque-limit deadband and a small
negative reward scale. Only policy-controlled joints are included; passive
joints have no learned feedforward contribution to cancel.

### Motivation

The policy could otherwise exploit an underdetermined action decomposition:

```text
large positive PD torque + large negative feedforward torque = small net torque
```

The robot experiences only the small net torque, but both heads use excessive
authority, become difficult to interpret, and can react badly to saturation or
domain randomization. The penalty favors a lower-conflict decomposition while
allowing small corrective opposition through the deadband.

Unlike a global cosine-alignment reward, the per-joint formulation:

- Does not require the complete torque vectors to point in the same direction.
- Does not penalize the heads for acting on different joints.
- Prevents the twelve leg joints from hiding cancellation in an arm joint.
- Does not reward both branches merely for being simultaneously nonzero.

It should remain lightly weighted because legitimate transient control can
require feedforward torque to oppose an excessive instantaneous PD response.

## Combined Learning Objective

The PPO reward includes task rewards plus impedance and cancellation shaping:

```math
R_t = R_{task,t}
+\lambda_{imp}R_{imp,t}
-\lambda_{cancel}C_{cancel,t}
+\ldots
```

The optimizer additionally receives physics losses:

```math
L_{total} = L_{PPO}
+\lambda_{ID}L_{ID}
+\lambda_{roll}L_{roll}.
```

Their responsibilities are different:

| Objective | Question | Primary behavior encouraged |
| --- | --- | --- |
| PPO task rewards | Is this motion useful? | Tracking, locomotion, stability |
| Inverse dynamics | Do forces and torques balance observed acceleration? | Whole-body generalized-force consistency |
| One-step rollout | Do actions and forces predict the next state? | Action-to-observation consistency |
| Impedance reward | Does the EE respond compliantly to force and error? | Combined force/position behavior |
| Cancellation reward | Are the two torque branches wasting authority? | Efficient, interpretable head cooperation |

The intended combined result is a policy that earns task reward using actions
that are physically explainable, locally predictive, compliant at the EE, and
free of large unnecessary conflict between its position and torque branches.

## Interaction and Failure Modes

### Helpful Alignment

The objectives agree when a high-reward action also explains the measured
transition, balances whole-body forces, tracks the EE with the desired
compliance, and avoids head cancellation. In this regime, the physics terms act
as structured regularizers for PPO rather than competing objectives.

### Force Attribution

Torque and predicted external forces can compensate for each other inside both
PINN objectives. Force reconstruction losses and reliability gating are needed
so the estimator cannot invent disturbances solely to reduce physics residuals.

### Action-Head Attribution

Inverse dynamics and rollout normally see only combined torque. Impedance sees
only resulting EE motion. Cancellation removes wasteful opposition, but does
not uniquely assign slow motion to the position head and fast compensation to
the torque head. Separate action-rate, magnitude, or frequency-shaping terms
would be needed to enforce that stronger interpretation.

### Model Mismatch

If actuator delay, torque clipping, passive dynamics, contact impulses, or
randomized inertial properties differ between the training dynamics and
Genesis, both PINN terms can penalize useful behavior for errors the policy did
not cause. Physics weights should be ramped only after the modeled terms are
validated and a minimally viable task policy begins to emerge.

### Over-Regularization

Large physics or cancellation weights can produce conservative actions and
reduce exploration. Large impedance weight can trade away precise tracking for
compliance even when no meaningful disturbance exists. Each raw component
should be logged separately, normalized, and introduced with controlled scale.

## Recommended Training Order

1. Establish PPO task learning with measured force inputs to physics losses.
2. Ramp the observed inverse-dynamics loss after basic standing and reaching.
3. Confirm separate base, leg, and arm residual statistics are well scaled.
4. Enable predicted forces only after their supervised reconstruction is
   reliably better than a naive predictor.
5. Introduce one-step velocity rollout at a smaller weight than inverse
   dynamics, initially without configuration integration.
6. Keep impedance and torque-cancellation reward scales small enough that task
   tracking remains the dominant definition of success.
7. Monitor PPO-versus-physics gradient conflicts and use the existing gradient
   conflict handling when necessary.

## Implementation Map

- Inverse-dynamics loss: `rsl_rl/algorithms/ppo_b1z1_pact.py::_pinn_loss`
- Coupled-torque reconstruction: `PPO_B1Z1PACT._coupled_torque`
- Dynamics backend contract: `legged_gym/dynamics/whole_body_dynamics.py`
- Impedance reward: `B1Z1PACT._reward_impedance_consistency`
- Cancellation reward: `B1Z1PACT._reward_torque_cancellation`
- Reward parameters and scales: `b1z1_pact_config.py::rewards`
- Executed torque composition: `genesis_simulator_b1z1_pact.py::_compute_torques`

The one-step rollout loss described here is a proposed extension. The current
B1/Z1 PACT algorithm implements the inverse-dynamics objective but does not yet
include the rollout objective.
