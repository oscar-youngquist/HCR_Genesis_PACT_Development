# Experiment 3: offline representation analysis

Open `notebooks/exp3_representation_analysis.ipynb`. The helper uses **existing
Experiment 1 recordings only**; it does not construct an environment,
training runner, optimizer, or simulator, and never updates checkpoints.

## Configuration and dependencies

Set recording roots, method IDs/tasks and optional checkpoint paths, evaluation
seeds/conditions, sampling budget, device and output directory. Use a distinct
method ID for each independently trained checkpoint. Checkpoint paths must
hash-match the selected recordings. Only load trusted local Torch checkpoints.
Set `urdf` only to relocate the identical, hash-verified robot model.

Fresh inference requires NumPy, Pandas and PyTorch. Direct BARD additionally
requires CUDA, `bard-pytorch-dynamics`, and Pinocchio for aggregate inertial
initialization. No Isaac Lab runtime is needed. Use the existing
`lr_lab_cupiqp` environment for Torch/BARD; install/register a Jupyter kernel in
that environment separately if it lacks `ipykernel`.

`matmul_precision='high'` matched the inspected real Isaac Lab PACT recording
on cuda:0; `'highest'` did not match its saved predictions. The setting is
recorded and restored after analysis. All fresh native predictions must pass
explicit absolute/relative tolerances before replay. A mismatch stops analysis:
check checkpoint identity, code and inference precision rather than simply
loosening tolerances.

For a CPU-only audit without Torch, explicitly set
`recorded_native_only=True` and `dynamics=False`. This uses saved predictions,
labels outputs `native_recorded`, never claims fresh inference parity, and
never performs cross-method replay. It still validates checkpoint file hashes.

```bash
EXP3_CONFIG='{"recording_roots":["/path/to/exp1"],"device":"cuda:0","sampling_budget":256,"output_dir":"/tmp/exp3"}' \
  jupyter nbconvert --execute --to notebook notebooks/exp3_representation_analysis.ipynb --output-dir=/tmp
```

## Selection and comparability

- Discovery requires `COMPLETE`, metadata, records, trials and initial physical
  states. Protocol hashes/settings, schema shapes/dtypes, unique run/trial
  identities, positive masses and physical inertia tensors are checked.
- The budget is **per in/OOD loading range**, not shared across them. Available
  condition/motion/phase/evaluation-seed/source-method strata receive quotas
  as equally as capacity permits. Hash-ranked selection is deterministic.
  Exhausted strata are not duplicated; the manifest reports available and
  selected counts. Small budgets may omit strata, which remain visible there.
- Invalid, reset, failed and reset-filled/incomplete histories are excluded.
  Histories must have a full contiguous valid control-step window, native
  configured dimensions, and the newest observation in the final slot.
  Errors describe this balanced selected population, not every collected step.
- PACT and current Pos use 81-D observations, but the former contains
  feedforward-action history and the latter executed PD-torque history.
  UniFP uses 73-D observations and policy-specific force compensation commands.
  Equal widths alone never establish compatibility.
- Native normalized histories are reused only for exactly matching audited
  contracts: ordering, normalization, history length, action semantics/scales,
  initial joint offsets and torque-conditioning convention. Native gains may
  differ without changing those input semantics. UniFP compensation histories
  are tied to their source checkpoint. No privileged loads are ever synthesized
  into actor inputs, and no missing channels are padded.
- `common:<contract>` tables use the identical selected source transitions
  for every model in that compatible group. Otherwise comparisons remain
  native-rollout comparisons, not a common-trajectory benchmark. The helper
  deliberately does not attempt an unverified cross-family history translator.

## Timing, frames and metrics

Current EE/base force heads are evaluated from `history_t`. The evaluator applies
new disturbances **after** inference, so their causal target is the previous
interval's applied load. Changed-load rows are labelled `new_load`; sustained
rows remain separate. Constant world loads are rotated from each decoder's
native current yaw frame into world SI units. Torso moments are compared at the
torso CoM, not at an arbitrarily shifted base origin.

GRF predictions use the source transition's recorded
`prediction_grf_torque_input_nm`, never a freshly proposed actor action.
Its metadata distinguishes clipped pre-step nominal torque from legacy
final-substep executed torque. Output conversion uses **successor yaw**, matching
training. Targets are `grf_raw_world_interval_mean` [N], explicitly not filtered
or processed training labels. Individual feet/axes and signed biases are retained.
Original UniFP's EE-before-base explicit-force order is handled separately from
extended UniFP's base-before-EE order. UniFP has no moment or GRF head.

The seven main columns are component RMSE for EE force [N], torso force [N],
torso moment [Nm], foot GRFs [N], and base linear [m/s], base angular [rad/s],
joint velocity [rad/s] prediction. Within a trial, average squared component
errors then take the square root. Average trial RMSEs equally within an
evaluation seed; report equal-seed means and sample SD (`ddof=1`). Timesteps
are not independent replicates. One seed has undefined SD. Axis/foot results,
biases, sample/trial counts and arm-plus-gripper/leg velocity breakdowns remain
in numeric exports. Nominal/disturbed tables are recomputed from selected
transition errors, not averaged from stratum RMSEs or SDs. In/OOD stays separate.

## BARD diagnostic and limitations

Direct BARD evaluates M and h and world-aligned Jacobians at recorded q_t/v_t,
with recorded added-base mass, CoM shift and gripper mass. Recorded armature is
added to the joint-space mass diagonal. The analytic training-style velocity
step is `v_next = v_t + dt * solve(M, tau + J^T loads - h)`, using recorded
interval-average **executed** torque (already motor-scaled/clipped). This is a
frozen-mechanics control-interval approximation, not a simulator substep replay.

Applied CoM forces are shifted to BARD frame origins once with `r x F`, using
initial body-local CoMs and q_t rotations. Foot loads are point-force
approximations; contact moments/non-foot impulses are not reconstructed from
Exp1. Full recorded inertias are validated and retained in source metadata;
the backend reproduces its parameterized base/gripper inertia model, not an
arbitrary per-body PhysX inertia replacement. Total BARD mass must agree with
recorded body masses. These model/physics approximations remain limitations.

Nonzero passive joint friction, damping, or stiffness is currently rejected
for dynamics diagnostics because this BARD path does not model those torques.
Unavailable dependencies, unsupported mechanics, or model mismatches produce
explicit reasons and **unavailable dynamics**, not zero errors. Training PINN
enable flags are never consulted. Thus even physics-disabled ablations get
diagnostics when their recorded mechanics are supported.

Three variants are separate:

1. `decoded`: complete decoded loading only. No missing head is replaced by
   measurements, including for nominal trials.
2. `measured_loading`: actual interval loads/raw GRFs, same M/h and executed
   torques; available only when mechanics are supported.
3. `constant_velocity`: recorded v_t compared with v_(t+1), independent of BARD.

## Exports and caching

- `transition_manifest.json/.csv`: exact run/trial/step/chunk/row identities,
  selection seed strata and quotas; reusable for subsequent replay.
- `input_audit.csv`: actual native dimensions, semantics and compatibility IDs.
- `summary.json`: protocol, method/checkpoint/training-seed provenance, device
  precision, native parity, unavailability reasons and aggregation results.
- `per_seed.csv`, `breakdown.csv`, `summary.csv`: numeric seed/group statistics,
  bias, counts, condition/motion/phase/loading-range distinctions and variants.
- Separate `native_*`, `native_recorded_*`, or `common_*` tables: page-wide
  nominal/disturbed groups, unavailable dashes, optional `latex_method_names`
  manuscript macros. LaTeX requires `booktabs` and `graphicx`.
- `cache/*.npz/.json`: normalized predictions keyed by checkpoint, source code,
  metadata/manifest, selected history/torque content, device and precision.
  Native parity is rerun for fresh inference; compatible replay reuses caches.
- `COMPLETE`: written only after successful export; undefined JSON values are null.

Repeated evaluation seeds of one checkpoint are never labelled independently
trained policies. Report source method coverage alongside common results and
do not combine native and common scopes into one table.

## Validation

```bash
python -m pytest -q -p no:cacheprovider tests/test_exp3_representation.py
```

CPU fixtures test balancing, causal loads, native dimensions, normalization,
yaw rotations, wrench reference points, known dynamics, missing heads, seed
aggregation and notebook exports. The Torch test additionally uses the actual
PACT encoder/decoder constructors, strict checkpoint loading, deterministic
parity, and unchanged RNG/weights. It skips when Torch is unavailable. The
notebook smoke uses explicitly labelled saved-native fixtures without Torch.
