# B1Z1 IsaacLab Startup and Parity Checks

## Startup failure

Native Isaac Sim URDF import stalled while importing the B1 Collada visual
meshes. Stack dumps stopped in the URDF importer, before environment reset or
PPO. Converting only the torso moved the stall to the hip; correcting Collada
timestamp metadata did not resolve it.

`legged_gym/simulator/b1z1_lab_assets.py::prepare_lab_urdf` now creates a cached
derived URDF with the five B1 visual meshes converted to STL without decimation.
The cache is content-addressed and protected by a file lock. Joint definitions,
inertials, collision shapes, visual origins, and mesh scales are preserved.
Original assets are not modified. STL does not preserve embedded Collada
materials/textures, so visual appearance need not match exactly.

The conversion requires `pycollada`, now included in the optional IsaacLab
dependencies and installed in `lr_lab_cupiqp`. Cache files live under
`/tmp/b1z1_isaaclab_assets` and regenerate if the source content changes.

The Lab backend now selects `cfg.asset.isaacgym_file`, falling back to
`cfg.asset.file` if absent. All three B1Z1 variants therefore use the same
`b1z1.urdf` source as Gym. The visual-only conversion still avoids the importer
stall, and startup diagnostics report the selected source asset.

## Other fixes

- `isaaclab_simulator_b1z1.py` shares the Gym B1Z1 configuration parser, including
  final fixed randomization bounds and the positive-only CoM-Z option.
- Default joint positions have shape `(1, num_dof)`, as expected by multi-env
  reset. Joint-velocity history stays in backend order until its accessor
  converts it to policy order; it is no longer permuted twice.
- PhysX property-setter indices are explicitly CPU tensors. Configured link
  contact states and non-plane out-of-bounds handling are updated.
- Torso and gripper mass randomization also updates inertia using IsaacLab's
  standard `recompute_inertia` convention: `I_new = I_default * m_new/m_default`.
  Mass and inertia are derived from defaults, including the actual clamped mass,
  so repeated resets do not compound. Only selected environments and enabled
  bodies are modified. This assumes a fixed shape and uniform density change;
  it is not a model of an attached payload at a separate location.
- Gym and Lab PD-gain caches now use configured joint order for both control
  and PACT dynamics/torque-cloning labels. The control equations are unchanged.
- `isaaclab_simulator.py` disables multi-GPU rendering. Physics and model tests
  use physical GPU 1. Kit may still print cross-device P2P diagnostics.
- `rsl_rl/utils/rollout_timing.py::startup_metadata` no longer assumes Lab
  articulations expose Genesis `links`. Shape count is unavailable and currently
  reports zero for Lab, not a measured absence of collision geometry.
- `PPO_B1Z1PACT.init_storage` forwards keyword arguments, including the runner's
  existing `rollout_state_dim`, to rollout storage.

## Validation scope

Small headless tests use two environments, one triangle-mesh terrain tile, four
rollout steps, one learning epoch and one minibatch. They are startup checks,
not throughput benchmarks or evidence of long-run learning stability.

UniFP passed reset, indexed reset isolation, state read/write, external-force
buffer routing, finite environment outputs, and a PPO/auxiliary update in both
Gym and Lab. At matching prescribed joint states/actions, configured DOF/foot
ordering, torque limits, computed torques and output shapes matched. Soft joint
limits differed by at most approximately `4.77e-7` due to floating-point math.
Lab construction took roughly 23-25 seconds in these small tests; Gym took
roughly six seconds. These figures exclude any claim about 4096-env startup.

PACT-Pos also passed a two-environment Lab PPO update with domain randomization
enabled. The test checks finite outputs, not matching random trajectories.
Coupled PACT passed the same short warmup update after correcting the storage
wrapper. This does not exercise post-warmup PINN losses.

`tests/test_b1z1_lab_assets.py` checks unchanged source files, joints/inertials/
collisions, all converted triangle coordinates, and cache reuse for both URDFs.
`tests/test_b1z1_lab_mass.py` verifies inertia scaling, indexed updates, mass
labels, and repeated-call stability without starting the simulator. Python syntax
checks and `git diff --check` cover modified files.

## Remaining parity limits

- Asset selection and mass-dependent inertia updates are now aligned in intent.
  Lab uses its documented default-inertia scaling convention; identical numeric
  inertia tensors to Gym's collision-based recomputation are not guaranteed by
  that formula alone and have not been established by these tests.
- Native joint friction/damping and collision cooking are engine-dependent.
  Explicit contact/rest-offset parity and long-running contact behavior still
  need dedicated checks; the smoke tests do not establish these.
- Force-buffer checks verify routing and values, not a measured acceleration
  response or full wrench-frame equivalence under arbitrary robot rotations.
- GRF measurement utilities were not expanded or subjected to equivalence tests.
- Full terrain grids, long curricula, viewer/camera behavior, sustained PINN
  optimization after warmup, and large-environment throughput remain untested.

Launch from `legged_gym/scripts` using `sh b1z1_unifp_lab.sh` (or the corresponding
PACT/PACT-Pos Lab launcher). These launchers select `lr_lab_cupiqp` and GPU 1.
