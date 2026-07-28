# B1/Z1 PACT Initial Implementation

## Scope

`b1z1_pact` is a standalone Genesis task for the B1 quadruped with the Z1 arm.
It inherits only the generic `LeggedRobot` task contract. The task retains the
B1/Z1-specific UniFP mechanisms that describe goals and disturbances, while it
uses a PACT-style temporal context encoder, asymmetric critic, PPO runner, and
coupled position/torque action interface.

The training entrypoint is `legged_gym/scripts/b1z1_pact.sh`. It selects
`SIMULATOR=genesis_b1z1_pact` and runs `train.py --task=b1z1_pact`.

## Command and Goal Interface

The policy command has exactly six values:

`[base_vx, base_vy, base_yaw_rate, ee_radius, ee_pitch, ee_yaw]`.

The last three values are the UniFP yaw-aligned spherical end-effector target.
They are centered at the Z1 waist frame, transformed into the current B1 yaw
frame, collision-filtered along the interpolated path, and moved gradually
from the previous target to the newly sampled target. There are no force
command slots and no external controller that shifts the EE position target.

## Coupled Action and Simulator Control

The actor emits 34 values:

`[position_action[17], torque_action[17]]`.

The learned DOFs are the twelve leg joints plus five arm joints through
`z1_forearm_roll`. `z1_wrist_rotate` and the gripper preserve their nominal PD
target. In `GenesisSimulatorB1Z1PACT`, the executed torque is

`tau = w_feedback * (Kp * (q_target - q) - Kd * qdot) + w_feedforward * tau_ff`,

where `q_target = q_default + action_scale * position_action` and
`tau_ff = torque_scale * motor_strength * torque_action`. Torque clipping is
applied only after the two branches are combined. The PACT tradeoff curriculum
starts at the configured feedback/feedforward weights and advances an
environment when its episode-average base-velocity tracking reward clears the
configured threshold.

## Disturbances

Only two physical disturbance streams exist:

1. A world-frame force applied at the end-effector link.
2. A world-frame force applied to the B1 trunk.

Each environment independently samples the original UniFP event profile:
wait for an interval, probabilistically select an event, ramp from zero to a
sampled force, hold for the settling interval, then ramp down to zero. Events
are gated until `force_start_step * runner_steps_per_iter`; no generic
velocity-push randomization and no commanded-force stream is enabled.

## Context, Decoders, and FiLM

The context encoder consumes the stacked actor-observation history and predicts
a latent distribution, base velocity, a six-dimensional base wrench, and the
three-dimensional external EE force. PPO uses its deterministic latent mean so
action likelihoods remain well-defined across rollout and update; the auxiliary
VAE/decoder update samples `z`. Two independent decoder heads use
`[z, predicted_base_velocity, predicted_base_wrench, predicted_ee_force]` to
predict the next-step `[GRF_12, EE_force_3]` target and the next single
privileged observation, respectively.

The DreamFLEX-style FiLM condition is intentionally limited to `z`, predicted
base wrench, predicted EE force, base velocity tracking error, and arm EE
position tracking error. The EE error is computed with the task's arm forward
kinematics, rather than taken directly from a force-modified command.

## Pinocchio Whole-Body Consistency Loss

`legged_gym/dynamics/whole_body_dynamics.py` defines the backend interface;
`PinocchioWholeBodyDynamics` is the initial implementation. It uses persistent
shared-memory multiprocessing workers, following the existing PACT
`parallel_pino_workers.py` lifecycle, to process PPO minibatches in parallel.
Each worker owns its Pinocchio model/data and receives environment-index chunks.
It maps Genesis joint ordering to a free-flyer Pinocchio B1/Z1 model (`nv=25`),
applies the per-environment torso added mass, torso COM shift, and gripper added
mass to nominal worker-local inertias, evaluates the mass matrix and nonlinear
effects at the observed state, and forms world/LWA
generalized external forces from four foot GRFs, the linear EE force, and the
base wrench. No EE torque is modeled in this initial formulation.

The residual is

`M(q) * vdot + h(q, v) - S^T * tau_executed - J_feet^T * GRF - J_ee_linear^T * F_ee - J_base^T * W_base`.

Acceleration uses the stored previous and current generalized velocity. The
rollout records the state after the action has advanced the simulator, so that
the finite difference is aligned with the preceding action. Terminal samples
are masked from this loss. The force decoder becomes eligible as a PINN force
source only after its EMA reconstruction loss remains below the configured
threshold for the configured patience; hysteresis controls subsequent gate
behavior. The Pinocchio backend evaluates observed-state terms on CPU, so this
initial loss supplies action-torque gradients only; `predicted_force_detach`
is retained as the backend-independent switch needed by a future
differentiable backend such as BARD.

## Soft Impedance Reward

`_reward_impedance_consistency` evaluates a translational virtual impedance
residual from the interpolated EE target, FK/world EE motion, and external EE
force. Its exponential reward encourages force/position consistency but never
changes the sampled command, PD target, or torque action. This preserves the
combined position/force objective while leaving direct coupled torque control
to the policy.

## Checkpointing and Logging

The runner saves and restores the actor/critic/context encoder, both decoder
heads, the actor optimizer and shared context/decoder auxiliary optimizer,
iteration, and force-gate state. TensorBoard
records PPO/PINN/decoder losses, force-gate status, separate position and
torque exploration standard deviations, episode statistics, and throughput.

## Validation Performed

Source compilation succeeded. In the `genesis_lr` environment, task
registration resolved correctly, Pinocchio loaded the B1/Z1 URDF with `nv=25`
and the configured foot/EE/base frames, and a tensor-only actor/decoder smoke
test confirmed an 82-D actor observation, 1,224-D stacked critic input, and
34-D coupled action. The persistent two-worker shared-memory evaluator was
also checked against the synchronous evaluator for mass, bias, and generalized
contact terms. No training run was started.
