# B1Z1 Isaac Lab Ground Forces

New B1Z1 Isaac Lab runs default to `sim.grf.force_mode = "total"`. This is shared
by PACT/PILOT, its thin ablations, Pos, UniFP and UniFP-Reject. No network shapes,
normalization scales, or checkpoint tensors change. Older decoders trained on
normal-only labels do not become total-force estimators simply by loading them.

## Installed API and Timing

The inspected Isaac Lab ContactSensor explicitly reports **normal-only**
`net_forces_w`/`force_matrix_w`. Installed PhysX tensors 107.3.26 provides
`get_contact_data(dt)` and `get_friction_data(dt)`. Their scalar normal forces
and vector friction forces are already in N when passed the **physics dt**.
There is no subsequent division by policy dt or physics dt.

Exact USD ground collision prims filter out self/other-robot contacts. Each
normal contact is multiplied by its actual world normal, including slopes.
Friction anchors have independent counts/start indices. This PhysX version
reuses index buffers between queries; the reducer preserves normal indices
before requesting friction. Warp is imported after Kit selects its bundled
version. No installed simulator package is modified.

Reduction visits only live contacts on device. Outputs and body-order gathers
are preallocated; there are no per-contact CPU transfers. Capacity defaults to
`8 * number_of_bodies * num_envs` per view and is adjustable with
`sim.grf.contact_capacity_per_body`. Invalid ranges, nonfinite contacts, API
failures, and capacity exhaustion abort rather than produce partial labels.
Exhaustion is rejected conservatively; the installed API has no dropped-contact
counter. Increase the capacity if this guard fires.

Forces are read after each completed physics substep. Decoder labels, GRF
observations and BARD interval targets use total forces, retaining existing
scales, clipping, EMA and interval timing. Deadband/contact detection uses the
normal component. Collision/termination and VHIP normal-support paths remain
normal-only. Other simulator backends retain their existing behavior.

Contact moments use each normal contact point and friction anchor, then shift
the summed moment to the simultaneous whole-body CoM **before** interval
averaging. Non-foot ground contacts are included in all-body outputs. Separate
free contact couples are not exposed by this API; individual contact arrays are
not persisted. Constant physics timesteps make substep arithmetic means exact
time-weighted interval means.

The load validation also exposed the installed permanent wrench composer's
cached link transform. It is now reset/recomposed each substep so commanded
world loads stay world-aligned. This corrects previously rotated loads; it does
not change nominal commands, target generation, gains or force magnitudes.
PhysX's default application point is the link transform. New evaluation load
points are sampled there before stepping, not at the body's inertial CoM.

## Records and Legacy Runs

Training writes `grf_semantics.json`; evaluation writes `metadata.grf_semantics`
(version 2, source, mode, frame, units, physics dt, ground prims, capacity).
Normal-only reproduction is explicit: `sim.grf.force_mode = "normal_only"`.
Unavailable tangential/total recording fields are omitted in that mode; its
moments use the old link-origin approximation. Missing semantics metadata in
older recordings still means legacy normal-only data, never total forces.

Additional Exp1/Exp2 fields, in configured foot order:

| Field | Meaning |
|---|---|
| `grf_normal_world_interval_mean` | Normal ground-force vectors, N |
| `grf_tangential_world_interval_mean` | Friction vectors, N |
| `grf_total_world_interval_mean` | Normal + friction, N |
| `grf_raw_world_interval_mean` | Selected mode's unfiltered labels, N |
| `support_normal_moment_world_interval_mean` | Foot normal-contact moment about whole CoM, Nm |
| `support_tangential_moment_world_interval_mean` | Foot friction-anchor moment about whole CoM, Nm |

Both experiments also record articulation-ordered `body_contact_normal_world`,
`body_contact_tangential_world`, and `body_contact_total_world` interval means.
`support_moment_world` and `all_contact_moment_world_interval_mean` now include
both components in total mode. Normal redistribution analysis prefers the
explicit normal fields; legacy recordings keep their original interpretation.
Both now retain whole-body CoM endpoint velocity and an interval non-foot-ground
contact flag so later momentum analyses can exclude non-foot support explicitly.
The EE-load moment uses the pre-step application point and successor CoM, a
documented moving-point discretization approximation.

## Reproduce the Checks

Use the existing Isaac Lab Python environment. Each invocation is bounded to
1-8 robots and 2-200 control steps per stage (standing, -25 N world-z EE load,
-25 N world-x EE load). `--training-update` runs exactly one four-step PPO
update with one epoch/minibatch, not a training job. Use a matching checkpoint
for meaningful standing response; without it only interfaces/conservation are
tested. Results, substep force arrays and timings are written to the specified
directory. The launcher reports failure even if Kit exits with status zero.

```bash
python legged_gym/scripts/validate_b1z1_grf.py \
  --task b1z1_pact --gpu cuda:0 --training-update \
  --run /path/to/run --ckpt 70000 --steps-per-stage 75 \
  --output-dir /tmp/grf-pilot

# Repeat with both robot and ground materials made frictionless.
python legged_gym/scripts/validate_b1z1_grf.py \
  --task b1z1_pact --gpu cuda:0 --frictionless \
  --run /path/to/run --ckpt 70000 --steps-per-stage 75 \
  --output-dir /tmp/grf-frictionless
```

Repeat the first command with `b1z1_unifp_original`, `b1z1_ppo_pos`, and
`b1z1_pact_ab7_position_full` and their checkpoints. For older actor-trunk FiLM
checkpoints, explicitly add `--actor-trunk-film`; this validation-only override
does not change registered training configurations or loosen strict loading.

The report checks normal-matrix parity, component sums, frictionless zeros,
partial resets and unchanged policy weights. Whole-body momentum residuals
include gravity, actual imposed loads and **all** ground contacts, not just
feet. Query/reduction timings exclude the first compilation call and compare
with a normal-only query on the same completed step. These are small-batch
overheads, not large-training throughput estimates. PyTorch peak memory does
not include all PhysX allocations.

## Validation Results (2026-11-03)

On `cuda:0` (RTX 4090), two robots:

- PACT, UniFP Original, Pos and ablation 7 each completed one four-step PPO
  update and frozen-checkpoint evaluation. The older Pos/ab7 checkpoints used
  the explicit actor-trunk FiLM validation override. Final PACT parameters
  were checked finite after its update.
- PACT checkpoint 70000 from `Oct29_15-14-37_b1z1_pact_improved`: 75 steps per
  stage, no physical failures. Mean vertical support changed from 568.88 N to
  593.82 N under the -25 N EE load. Under -25 N world-x loading, mean tangential
  support was +25.54 N in x. Maximum momentum residual divided by gravitational
  impulse was `5.43e-8`; observed motion accounts for the small mean imbalance.
- Frictionless control: tangential forces passed a `1e-4 N` absolute-zero
  check. Independent normal-contact aggregation matched the installed filtered
  normal matrix (`atol=.05 N`, `rtol=.002`). Partial resets and frozen weights
  passed across all four family checks.
- Final PACT query/reduction averaged **0.527 ms/substep**, p95 **0.776 ms**;
  the normal-only query on the same steps averaged **0.024 ms**. This includes
  new point-moment reduction and body-order gathering, not just friction lookup.
- Real Exp1 recording: 2 episodes x 3.2 s, 320 rows, nonzero EE loads/friction,
  zero forbidden torso loading, `COMPLETE`. Exp2: 2 episodes x 0.2 s, 20 rows,
  finite component/all-body fields, `COMPLETE`; this short Exp2 check does not
  reach its task loading stage.

Focused CPU command (77 passed):

```bash
SIMULATOR=isaaclab_b1z1_pact python -m pytest -q -p no:cacheprovider \
  tests/test_b1z1_contact_forces.py tests/test_b1z1_lab_curriculum_grf.py \
  tests/test_b1z1_experiment1.py \
  tests/test_b1z1_experiment2.py::test_substep_recorder_moments_sensor_order_and_torque_identity \
  tests/test_exp3_contact.py tests/test_exp3_support.py
```

Related broader checks exposed seven stale Exp2 timing/heading fixtures and one
Go2 normalization fixture. All eight failures were reproduced with the original
HEAD implementations; unrelated settings/tests were not changed to hide them.
Changed Python files passed syntax checks and `git diff --check`.

Reports are in `/tmp/b1z1-grf-{final,frictionless,unifp,pos,ab7}/validation.json`;
recording smoke outputs are `/tmp/b1z1-grf-exp1` and `/tmp/b1z1-grf-exp2`.
Live sloped-terrain, large-environment capacity/throughput, and long training
were not tested. Sloped normals, independent indexing, multiple filters,
overflow/missing data and static/accelerated momentum cases have synthetic tests.
