# B1Z1 Isaac Gym Backend Port

## Scope

The B1Z1 UniFP, PACT-Pos, and PACT task environments can now select an Isaac
Gym simulator backend without changing their environment, command, reward,
runner, or model classes. The port uses one shared B1Z1 adapter and three
concrete simulator classes:

- `IsaacGymSimulatorB1Z1UniFP`: 17 learned position-residual actions.
- `IsaacGymSimulatorB1Z1PACTPos`: 17 learned position actions with
  motor-strength scaling on PD feedback torque.
- `IsaacGymSimulatorB1Z1PACT`: 17 position actions plus 17 direct-torque
  actions, combined by the existing PACT feedback/feedforward weights.

The classes live together in
`legged_gym/simulator/isaacgym_simulator_b1z1.py` so state acquisition,
external wrench application, randomization state, and DOF-order translation
are implemented once.

## Selecting The Backend

Run the existing scripts from `legged_gym/scripts`. Their Genesis selectors
remain the default. Override `SIMULATOR` to select Isaac Gym; the scripts then
activate the existing `lr_gym` environment automatically.

```bash
SIMULATOR=isaacgym_b1z1_unifp sh b1z1_unifp.sh --num_envs=256
SIMULATOR=isaacgym_b1z1_pact_pos sh b1z1_pact_pos.sh --num_envs=256
SIMULATOR=isaacgym_b1z1_pact sh b1z1_pact.sh --num_envs=256 --headless
```

The equivalent direct selectors are:

```text
isaacgym_b1z1_unifp
isaacgym_b1z1_pact_pos
isaacgym_b1z1_pact
```

## Preserved Simulator Contract

The adapter provides the B1Z1 fields consumed by all three environments:

- 19 physical DOF states in configured B1/Z1 order while policies control the
  first 17 DOFs.
- Passive PD targets for `z1_wrist_rotate` and `z1_jointGripper`.
- Base-frame and world-frame base velocities.
- Foot and thigh positions, foot velocities, EE position/velocity/quaternion,
  link contacts, GRFs, and measured DOF force.
- Motor-strength and PD-gain randomization buffers used by privileged state.
- Added base mass, torso COM displacement, added gripper mass, friction,
  armature, joint-friction, and joint-damping labels.
- UniFP intermittent base velocity/angular-velocity pushes.
- World-aligned EE force, base force, and PACT/PACT-Pos base torque buffers,
  applied before every physics substep.
- PACT feedback, feedforward, combined, unclipped, and executed torque buffers.

Isaac Gym loads leg DOFs in FL/FR/RL/RR order while the task config uses
FR/FL/RR/RL. The shared adapter translates state, limits, reset targets, PD
gains, measured torque, and actuation torque explicitly between those orders.

The adapter also derives Isaac Gym's physics timestep from
`control.dt / control.decimation`. This keeps the B1Z1 control clock identical
to the Genesis task configuration rather than using the stale standalone
`sim.dt` value.

## Domain Randomization

Runtime-equivalent behavior:

- PD gain and motor-strength values are resampled on reset.
- Base linear, vertical, and angular velocity disturbances use the same
  timeout ranges and curriculum amplitudes.
- External EE/base force schedules remain owned by the task environment and
  are applied by Isaac Gym on every substep.
- The existing performance-gated three-phase curriculum state and its
  progress values remain available to runners and logs.

Construction-time behavior:

- Friction, torso added mass, torso COM shift, gripper added mass, joint
  armature, passive joint friction, damping, and stiffness are sampled once
  per environment while actors are created.
- `domain_rand.isaacgym_use_final_domain_rand_ranges` selects which immutable
  curriculum endpoint is sampled. Its backward-compatible default, `False`,
  uses `min_added_mass_max`, `min_gripper_added_mass_max`, the initial COM
  magnitudes, and the joint-property `*_range_start` values. Setting it to
  `True` uses `max_added_mass_max`, `max_gripper_added_mass_max`, the final COM
  magnitudes, and the corresponding `*_range_end` values. Friction and joint
  armature use their single configured ranges in either mode.

This construction-time restriction is intentional. Isaac Gym GPU actors do
not safely support changing these rigid-body and DOF properties after
`prepare_sim()`. Consequently, later mass/COM and passive-joint curriculum
bounds are tracked but cannot affect already-created actors. The endpoint flag
therefore chooses the fixed distribution for a run. Full per-reset parity
would require either a CPU property-update mode or environment pools
constructed at several curriculum levels.

## Existing Files Changed

- `legged_gym/__init__.py`: accepts the three Isaac Gym B1Z1 selectors in the
  Python 3.8 runtime and imports Isaac Gym for any Isaac selector.
- `legged_gym/simulator/__init__.py`: exports only the active backend family,
  avoiding Genesis-only dependencies in `lr_gym`.
- `legged_gym/envs/base/base_task.py`: routes each new selector to its concrete
  simulator class and imports backend classes conditionally.
- `legged_gym/simulator/isaacgym_simulator.py`: supports ordered foot-name
  lists, configured DOF ordering, full physical-DOF torque/gain tensors, DOF
  force sensors, flattened heightfields, initial root rotation, lazy terrain
  loading, optional Warp imports, and translation from Boolean self-collision
  settings to Isaac Gym's inverted collision-filter bit.
- `legged_gym/utils/__init__.py`: avoids importing Genesis visualization
  helpers under Isaac backends.
- `legged_gym/envs/b1z1/b1z1_pact/b1z1_pact.py`: fixes the PACT tradeoff
  curriculum to use its registered `tracking_lin_vel_force_world` reward key.
  The old key prevented the runner's initial reset on every backend.
- `legged_gym/scripts/b1z1_unifp.sh`, `b1z1_pact_pos.sh`, and `b1z1_pact.sh`:
  retain Genesis defaults, honor a caller-provided `SIMULATOR`, select
  `lr_gym` for Isaac, and forward additional CLI arguments.

## Validation Performed

- Python syntax compilation under the repository interpreter.
- Isaac Gym import test under `/home/oyoungquist/.conda/envs/lr_gym`.
- One-environment construction and initial-reset tests for UniFP, PACT-Pos,
  and coupled PACT.
- A two-environment UniFP PPO smoke test for one complete learning iteration.
  It completed 48 environment steps, all adaptation losses, PPO optimization,
  reward logging, and external-force schedule logging without an error.

These are smoke tests, not simulator-equivalence tests. Before long training,
compare short deterministic Genesis and Isaac Gym rollouts with randomization
and noise disabled, especially EE pose, foot contacts, measured DOF force,
and the commanded/executed torque traces.
