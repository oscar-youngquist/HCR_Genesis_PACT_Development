# Experiment 3 offline analysis

Run `notebooks/exp3_representation_analysis.ipynb`. Exp1 roots supply representation
and dynamics; Exp2 roots supply task/channel/support responses. Nothing initializes
a simulator, optimizer, or collection process. Only trusted local checkpoints are
loaded; hashes must match recording metadata.

## Configuration and coverage

`tasks=None` discovers completed Exp1 methods; explicit method/checkpoint mappings
remain supported. **`balanced_resampling=False` uses every eligible transition**
from each checkpoint's own recordings. Sampling budgets, quotas and seeds have no
effect in this mode. Eligibility requires valid pre-failure, non-reset transitions,
complete contiguous native histories and a recorded preceding interval load.

`balanced_resampling=True` enables deterministic **subsampling without replacement**,
not bootstrap resampling. Each method/checkpoint is sampled independently, stratified
by condition, motion, disturbance phase and evaluation seed within each loading
range. `sampling_budget` sets a cap per checkpoint AND range; when it is `None`,
`samples_per_stratum=32` sets the quotas. `minimum_per_stratum=8` only reports
under-covered strata; it does not exclude them. Missing seed/condition coverage is
also reported. No transitions are transferred between policies.

Representation outputs live in
`<output_dir>/representation/{full|balanced}_{recorded|reinferred}/`, separate from
Exp2's unchanged `action_support/`. `transition_manifest.json` records mode,
available/selected counts, quotas and coverage. Exact reproducible identities are
streamed to `transition_manifest.jsonl`, whose SHA256 is recorded in the manifest.
`inference_batch_size=256` bounds selected histories/predictions in memory. Processing
holds one recorded chunk, one inference batch, and one recording's trial-level
sums/counts. It does not retain timestep metric rows. Nonfinite exclusions and
physics approximation notes are counted rather than accumulated per timestep.
Optional `cache_predictions=True` writes minibatch reinference caches, namespaced
by mode/source and keyed by inputs, checkpoint, code, device and precision.
It is disabled by default.

`exp2_roots`, `exp2_tasks`, and `exp2_loading` select completed task recordings.
The defaults select PILOT, UniFP, Pos, A1 (implementation 6), A3 (8), A6 (9).
Missing recordings do not initiate collection. Loaded/unloaded outputs stay separate.
Run only the representation cells when no Exp2 analysis is wanted.

## Own-trajectory representation benchmark

There is one `native_in_table` and one `native_out_table` (CSV/LaTeX), with
nominal/disturbed columns and all selected methods as rows. These explain each
model's own realized trajectories under matched experimental conditions; they
are **not shared-transition estimator comparisons**. There is no compatibility
grouping or cross-policy replay. Pos's native torque history and UniFP's own
estimator-dependent command history are consumed exactly as recorded. Neither
missing FF actions nor another policy's compensation commands are invented.

`recorded_native_only=True` scores saved predictions. `False` recomputes frozen
predictions with strict checkpoint/hash loading and native parity checks in each
minibatch. This choice is independent of full/subsampled selection. A parity
failure in any batch excludes ALL decoded metrics for that checkpoint, including
earlier batches; measured-loading and constant-velocity references remain valid.
There is no automatic tolerance increase or saved-prediction fallback.

`per_seed.csv`, `breakdown.csv`, `summary.csv`, and `checkpoint_summary.csv` retain
component/axis/foot, arm/leg, bias, phase, motion, load-timing and count information.
Trial squared-error sums/counts yield exact trial RMSE; trials are then equally
weighted within each evaluation seed, followed by equal-seed/checkpoint aggregation.
Nominal/disturbed tables pool trial statistics rather than averaging phase RMSEs.
Variability labels distinguish evaluation-seed SD, known training-seed SD, and
checkpoint SD with unverified training independence. Missing heads are dashes.

Current force labels refer to the previous applied interval visible to history;
GRFs use the same recording's nominal-torque conditioning and successor-yaw
conversion, scored against raw interval-average world forces. Missing heads are
unavailable. UniFP cannot supply complete decoded-loading dynamics without a
GRF and moment head. No substitute actor actions are generated for dynamics.

## Model-consistency approximation

BARD uses measured q_t, recorded mass/CoM/gripper parameters, armature added once
to joint inertia, interval-average executed torque, and world loads shifted from
body CoMs to model frame origins once. Passive simulator damping contributes
`damping * qd_t` to bias, assuming a zero drive velocity target. Explicit motor/PD
terms are already inside executed torque and are not added again.

Joint friction is a constraint-dependent coefficient, not an observed torque;
its torque is omitted with a recorded approximation warning. Nonzero passive
stiffness also remains omitted because its simulator target is not recorded.
Frozen-state Euler, interval averages, and approximate inertias/contact points
remain sources of model mismatch. Compare measured-loading and constant-velocity
references with decoded-loading errors; do not attribute all residual to decoders.
Approximation notes include affected-transition counts. CUDA/BARD is optional; missing
dependencies are explicitly reported and no fictitious zero errors are generated.

## Exp2 channels and support

`action_support/` exports trial-stage summaries, per-seed curve sufficient
statistics, aggregate curves and PDF/PNG figures. Loads/support/channels use the
interval start; tracking uses successor timestamps/references. Event lines come
from metadata. There is no interpolation over gaps/failures and no failed-tail
zero filling. Torques are in Nm, squares in (Nm)^2, force in N, moments in Nm,
geometry/tracking in metres and velocity error in m/s. Horizontal applied load is
in the initial-heading frame; carrying load is world Fz. Support is world-frame.

Commanded geometry is `FK(mean delayed position target)` at the measured base at
t versus reference_t, explicitly **not** the mean of substep FK or a predicted
next pose. Missing/mismatched URDFs make this diagnostic unavailable. RMS is
computed from substep mean-square torques before trial/seed aggregation, not from
mean torque. Cancellation uses the recorded substep absolute-value statistic.
PD/FF are weighted and motor-scaled requests; executed clipped torque is separate.

Support moments use recorded averages of simultaneous cross products about the
whole-robot CoM, never cross-products of interval averages. All-body contact
resultants include non-foot support when available; moments remain link-origin
point-force approximations, not measured contact wrenches. Per-foot vertical
redistribution is retained. No unique PD/FF compensation decomposition is claimed.
Settling/overshoot is deliberately unavailable without a defensible steady
reference; nominal gait support is not steady by default.

Attribution exports include A1/A3, A3/PILOT and A6/PILOT; nominal stiffness/damping
are checked for A1/A3. Recorded randomization settings remain in provenance.
All aggregations first average trials within an evaluation seed and checkpoint.
Checkpoint means then receive equal weight; sample SD is across checkpoints if
multiple hashes are present, otherwise across evaluation seeds. Distinct known
training seeds are labelled explicitly; otherwise checkpoint independence is
marked unverified (two hashes may be epochs of the same training run). Curve bands use
the same rule and surviving-trial counts are exported/plotted. Evaluation seeds
are never relabelled as independently trained policies.

## Focused checks

### Offline throughput

`dynamics_batch_size=256` batches BARD mechanics and device transfers separately
from `inference_batch_size`. Compatible recordings of one checkpoint reuse a BARD
backend; every batch still supplies its own recorded randomized mass/CoM values.
Lower the dynamics batch size if GPU memory is limited. A malformed mechanics
batch falls back to isolated rows so valid neighbours are retained.
`progress=True` prints recording paths and processed transitions/second every
30 seconds. This rate includes loading, inference, dynamics and aggregation;
initial model construction happens before the first throughput report.
`cpu_blas_threads=1` limits small CPU dynamics solves using optional `threadpoolctl`
and restores the previous BLAS settings afterward. Benefits depend on BLAS vendor.
Selection, native parity tolerances, force frames and metric definitions are
unchanged. Full coverage still processes every eligible transition; use explicit
balanced subsampling for a short trial, not as an implicit performance shortcut.
Offline BARD loads the existing dynamics modules in a private import namespace:
it does not execute `legged_gym` package initializers, import Isaac Lab/Genesis,
or change `SIMULATOR`. BARD/Torch and robot-model dependencies are still required;
the training package and its normal imports are untouched.

```bash
python -m pytest tests/test_exp3_representation.py tests/test_exp3_support.py -q
```

The notebook smoke uses synthetic saved predictions and all three Exp2 profiles
on CPU. Real frozen-network replay needs PyTorch; BARD dynamics additionally needs
the BARD/Pinocchio/CUDA dependencies. Synthetic success is not simulator validation.
