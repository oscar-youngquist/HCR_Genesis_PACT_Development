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
- GRF substep filtering and sensor-order tests are now covered below; physical
  contact-force equivalence across engines remains unverified.
- Full terrain grids, long curricula, viewer/camera behavior, sustained PINN
  optimization after warmup, and large-environment throughput remain untested.

Launch from `legged_gym/scripts` using `sh b1z1_unifp_lab.sh` (or the corresponding
PACT/PACT-Pos Lab launcher). These launchers select `lr_lab_cupiqp` and GPU 1.

## HardPACT Functionality Shared With B1Z1

The Lab adapter now applies the existing B1Z1 reward-gated domain-randomization
curriculum at runtime. Set `domain_rand.use_domainrand_curriculum = True` to
enable it; this update does not override the existing configured enable flag.
Enabled curricula start at initial bounds regardless of Gym's construction-only
`isaacgym_use_final_domain_rand_ranges` setting. Phase progress changes at most
once per PPO iteration. Updated mass (including gripper), CoM and joint-dynamics
bounds are sampled on the next episode reset. Friction and armature retain their
configured fixed ranges. Base/EE command-force curricula remain independent.

All B1Z1 runners save/restore optional `domain_rand_curriculum_state`, including
reward EMA/history, current phase/progress and last processed iteration. Older
checkpoints without it retain initial state. Existing `Values/domain_rand_*`
logs continue to report progression. The implementation reuses B1Z1's existing
phase/gating equations rather than substituting Go2-specific ranges or pushes.

GRF processing reuses HardPACT's `IntervalGRFProcessor`: each physics substep
reads world-frame contact-sensor forces, rejects the entire XYZ vector when
vertical force is inside the deadband, clips it, and updates its EMA. The
control-interval average uses clipped forces, not EMA forces. `_grfs_buf` keeps
the B1Z1 `(N,12)` EMA contract and existing normalization; interval averages are
available separately in `_grfs_interval_buf`, not substituted for PINN labels.
Configured `sim.grf` thresholds and alpha remain unchanged. Alpha now applies
per physics substep in Lab, so the time constant differs from the former
control-rate filter. Reset clears all filter and interval history per environment.

Sensor and articulation indices must not be interchanged: `_feet_contact_indices`
indexes native `ContactSensor.data.net_forces_w`; `_contact_indices` reorders
all sensor forces into articulation order for `_link_contact_forces` and existing
reward/termination consumers. `_feet_indices` indexes only articulation-ordered
tensors. Both mappings are resolved by configured link names.

TensorBoard `GRF/*` reports per-foot world-Z forces and mean vector norms for
raw, deadbanded, clipped, EMA and interval-average stages, plus contact fraction.
These are snapshots of the latest control interval at logging time, not averages
over the PPO rollout. They remain enabled with normal force/contact logging when
`enable_additional_diagnostics` is false. No QP controller, Go2 policy layout,
Go2-specific force normalization or additional PINN objective is transplanted.

Validation: the B1Z1 PACT-Pos two-environment GPU-1 smoke test advanced the
curriculum, resampled physical properties at reset, verified randomized inertia,
collected one force sample per physics substep, and completed a short PPO update.
CPU tests cover sensor-name permutation, filter stages, selective reset, logging,
duplicate-iteration protection and curriculum state restoration. The separately
run existing Go2 GRF suite has a target-scale mismatch in
`test_alias_step_returns_interval_target_and_keeps_ema_separate`; its expected
250-N normalization does not match the current Go2 result. Go2 code/config was
not changed to address that independent failure.
