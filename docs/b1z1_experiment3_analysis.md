# Experiment 3 offline analysis

Run `notebooks/exp3_representation_analysis.ipynb`. Exp1 roots supply representation
and dynamics; Exp2 roots supply task/channel/support responses. Nothing initializes
a simulator, optimizer, or collection process. Only trusted local checkpoints are
loaded; hashes must match recording metadata.

## Configuration and coverage

`tasks=None` discovers completed Exp1 methods; explicit method/checkpoint mappings
remain supported. **`balanced_resampling=False` uses every eligible transition**
from each checkpoint's own recordings. Sampling budgets, quotas and seeds have no
effect in this mode. Eligibility requires valid pre-failure, non-reset transitions
and a contiguous recorded preceding interval load. Complete native histories are
needed only for optional missing-decoder inference, not stored-output metrics.

`balanced_resampling=True` enables deterministic **subsampling without replacement**,
not bootstrap resampling. Each method/checkpoint is sampled independently, stratified
by condition, motion, disturbance phase and evaluation seed within each loading
range. `sampling_budget` sets a cap per checkpoint AND range; when it is `None`,
`samples_per_stratum=32` sets the quotas. `minimum_per_stratum=8` only reports
under-covered strata; it does not exclude them. Missing seed/condition coverage is
also reported. No transitions are transferred between policies.

Representation outputs live in
`<output_dir>/representation/{full|balanced}_recorded_first/`, separate from
Exp2's unchanged `action_support/`. `transition_manifest.json` records mode,
available/selected counts, quotas and coverage. Exact reproducible identities are
streamed to `transition_manifest.jsonl`, whose SHA256 is recorded in the manifest.
`inference_batch_size=256` bounds selected histories/predictions in memory. Processing
holds one recorded chunk, one inference batch, and one recording's trial-level
sums/counts. It does not retain timestep metric rows. Nonfinite exclusions and
physics approximation notes are counted rather than accumulated per timestep.
`cache_enabled=True` caches optional predictions, compact batched mechanics, and
completed per-recording reductions. `cache_completed_runs=False` disables the last
layer when profiling. Raw recording chunks remain immutable; file signatures and
numerical code dependencies invalidate stale results. Plotting/FiLM-analysis edits
do not invalidate mechanics caches. No cached fallback hides a failed dependency.

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

Stored predictions always take precedence. `fallback_predictions=True` permits
frozen, hash-validated inference only for missing outputs; `recorded_native_only=True`
disables that optional fallback. Missing dependencies affect dependent metrics,
not available recorded predictions. Histories are not read when all required
predictions are already recorded.

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

### Paired prediction RMSE

The main paired result is **PINN-minus-control trial RMSE**, with a pointwise
paired-bootstrap confidence interval. Negative differences favor PINNs.
Correlations have been removed. Existing force/velocity RMSE, signed biases,
axis/foot/arm/leg breakdowns and measured-loading/constant-velocity references
remain unchanged.

Comparisons are PILOT versus A7 (actor PINNs on for both), A6 versus A3
(actor PINNs off for both), and A2 versus A1 (both PINN families change, explicitly
labelled as a confounded comparison). Recorded architecture/gain/enable settings
are checked and mismatches labelled. Trial pairing requires recorded scenario
hashes, commands/events/initialization seeds, timestep/horizon, and matching initial
states/mechanics; the same evaluation-seed number alone is insufficient.

- **Primary, own_pre_failure:** compute each trial's RMSE from its own eligible,
  finite pre-failure transitions, then subtract the counterpart trial RMSE.
- **Sensitivity, common_pre_failure:** use exactly the intersection of selected
  valid control-step IDs, ending before the earlier physical-failure transition.
  Gaps/reset/failed rows are excluded. A nonfinite prediction makes that metric's
  trial pair unavailable rather than changing its support. Sparse balanced
  sampling may have few/no shared steps; counts are exported, never expanded
  silently. Shared time/scenarios do not mean the policies visited identical states.

RMSE is sqrt(mean squared error over the block's coordinates and eligible
timesteps), not an average of coordinate RMSEs. Trials have equal weight within
evaluation seed; seed means have equal weight. Bootstrap resampling keeps trial
pairs together within fixed evaluation-seed strata. These intervals quantify
evaluation uncertainty conditional on the compared checkpoints and seed strata,
not variability across independent training runs. Insufficient within-stratum
replication gives an unavailable CI.

Notebook controls: `rmse_bootstrap_resamples=10000`,
`rmse_confidence_level=0.95`, `rmse_resampling_seed=0`.
Optional `rmse_permutation_resamples` (default 0, disabled) adds a paired sign-flip
test under exchangeability of each pair's labels, with Holm adjustment across all
reported valid contrasts/blocks/scopes/conditions/motions. CIs remain pointwise,
not multiplicity-adjusted. Failure/nonfinite exclusions and configuration
mismatches remain visible; do not interpret a selected CI as general proof that
representation training improves all states.

Exports:
- `rollout_rmse_trials.csv.gz`: compact SSE/count/RMSE for each checkpoint/trial,
  own/common support and decoded/measured/constant-velocity variant.
- `paired_rollout_matching.csv`: scenario/initial-state audit, earliest cutoff,
  exact shared-step count/hash and recorded configuration agreement.
- `paired_rollout_trials.csv.gz`: matched side values, differences, usability and
  denominators, including attempted pairs with missing metrics.
- `paired_rollout_seeds.csv` and `paired_rollout_summary.csv`: equal-seed side
  means, RMSE differences, CI, relative reduction, and contributing counts.
  The same compact result is under `paired_rollout` in `summary.json`.

### Explicit state estimates

Every native nominal/disturbed table (in-range and OOD) also includes
`explicit_base_velocity_rmse` and `explicit_ee_position_rmse`.
These compare estimates with the **current state at t**, not commands or t+1.
The common base metric is body-frame planar vx/vy in m/s: PACT's third output is
yaw rate, whereas UniFP's is vertical velocity, so it is intentionally excluded.
EE estimates are unscaled from native spherical coordinates and converted to
Cartesian meters in the yaw-aligned manipulation workspace. The workspace origin
cancels using recorded spherical/world references at t; no terrain-height guess
is required. Both use the existing componentwise RMSE definition, retaining
per-axis biases/errors. Missing estimates/measurements stay unavailable per metric.
Stored predictions take precedence; neither column requires BARD. The paired
exports include these two current-state metrics as separate blocks as well.

### Runtime

Pairing reads compact eligibility indices once and caches them. The existing
batched physics pass also accumulates SSE/counts for common support: no second
BARD pass, rank sorting, or retained full velocity trajectories. CPU workers
return small sufficient statistics. Bootstrap draws are vectorized in bounded
chunks over trial summaries; changing resampling settings reuses completed-run
and mechanics caches. Old generated `velocity_agreement_*` exports are removed
when a new analysis starts, never raw recordings or unrelated outputs.

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
`metric_workers=2` overlaps pruned CPU metric batches with the single GPU/BARD
producer; `1` is useful for debugging/profiling and can be faster for tiny datasets.
This is independent of `num_workers` (chunk decompression only). Force transforms,
grouped squared-error sums and two-RHS mass solves are batched; measured/decoded
loads share a factorization. Mechanics caches store seven stacked arrays rather
than thousands of ZIP members. Completed-run reuse also restores the exact
transition manifest without re-reading/decompressing raw data in full-coverage
mode; balanced selection still scans its small eligibility/index fields. Existing legacy
mechanics caches are left untouched; the new layout has a separate namespace.
`timings.json` separates dynamics/cache preparation, metric submission/wait time,
paired-statistics work, and completed-run hits. With asynchronous workers these timings
overlap and are not additive. Cache-hit runs still regenerate tables/exports.
Offline BARD loads the existing dynamics modules in a private import namespace:
it does not execute `legged_gym` package initializers, import Isaac Lab/Genesis,
or change `SIMULATOR`. BARD/Torch and robot-model dependencies are still required;
the training package and its normal imports are untouched.

```bash
python -m pytest tests/test_exp3_representation.py tests/test_exp3_representation_speed.py tests/test_exp3_support.py -q
```

The notebook smoke uses synthetic saved predictions and all three Exp2 profiles
on CPU. Real frozen-network replay needs PyTorch; BARD dynamics additionally needs
the BARD/Pinocchio/CUDA dependencies. Synthetic success is not simulator validation.

### Momentum Contact and Normal-Force Metrics

Contact sensors in these recordings omit tangential friction. The historical
`grf_rmse` is retained as **Error Against Normal-Only Sensor Forces**, not total
GRF ground truth. New `grf_normal_rmse` projects each predicted/raw sensor force
onto world z on the recorded plane (otherwise recorded surface normals are
required). Its `all` value is componentwise RMSE across feet; named-foot rows
retain individual RMSE/bias/counts. Unsupported decoder heads remain unavailable.

`exp3_contact.py` reconstructs **Momentum-Derived Net Contact Force** in world N:
`m*(v_COM_end-v_COM_start)/duration - m*g - external_force_average`.
Exp2 uses recorded successor CoM velocity, or mass-weighted recorded body CoM
velocities. Exp1 uses batched URDF kinematics, recorded root-link/joint states,
and per-trial PhysX masses/local CoM poses. The URDF hash must match. This path
requires only NumPy/SciPy, not Torch, BARD, checkpoints, or forward dynamics.
It never substitutes torso or learned velocity. Unknown mechanics/loads produce
unavailable references rather than guessed values.

The previous consecutive valid successor supplies the current interval's start
endpoint. The initial interval is skipped; chunk boundaries/interleaved robots
are supported, gaps/resets/failures are not crossed. Actual applied interval
averages take priority. For older Exp1 fields, the inspected collector sets
`ee_force_world` and `base_force_world` once before each control interval and
holds them throughout; endpoint commands change, but those loads do not. Absent
loads are never implicitly zero. Moments do not enter linear momentum balance.

`grf_net_rmse` compares summed decoded feet with this reference; its x/y/z
breakdowns and `all = sqrt(mean(error_xyz**2))` use N, not norm-RMSE.
`grf_net_horizontal_rmse` uses x/y only. `grf_net_window_rmse` compares identical
duration-averaged predicted/reference windows, integrating loads and using
window endpoint momentum. `contact_window_seconds=0.10` sets the minimum complete
non-overlapping window duration (rounded up to a control endpoint). Windows
restart at stage/load changes, gaps, or invalid samples. Partial windows remain
unavailable. Existing optional sampling selects these diagnostics at the sampled
interval/window endpoints; window averages use their full contiguous support.

Net **foot**-GRF comparisons require recorded exclusion of non-foot contact.
Exp1 recordings without these channels therefore retain aggregate-contact
reconstructions but have unavailable foot-only net metrics. `contact_nonfoot_threshold_n`
(default `1e-6`) applies to recorded non-foot forces; recorded substep contact
flags additionally exclude contact intervals. The resultant is never split
among feet. This is a momentum/dynamics-consistency diagnostic related to rollout
accuracy, not independent sensing. No reconstructed force is fed into any
existing joint-level rollout, and its force-balance closure is not a control score.

Exp2 compact `contact_response_loaded.{pdf,png}` and
`contact_response_A6_PILOT_loaded.{pdf,png}` separate reconstructed interaction
support from measured front/rear normal redistribution, channel usage, tracking,
and survivor counts. Mean-square curves are labelled `(Nm)^2`; stage RMS takes
the square root **after** averaging recorded substep squares. Existing normal-only
moment cross-products are retained. Total angular-momentum reconstruction remains
unavailable without a verified complete external angular impulse about the moving
CoM; requesting `contact_moment_reconstruction=True` reports this without blocking
force analysis. `support_legacy_figures=True` also regenerates the older large grids.

Each section exports `contact_intervals/<recording-id>/intervals-*.npz` and
`metadata.json`: trial/step IDs, interval timestamps, whole-CoM endpoint velocity
in m/s, mass in kg, applied interval force, reference force in N, foot eligibility,
complete-window reference/prediction/start/duration, counts and provenance.
NaNs in numeric arrays mean unavailable; metadata JSON is standards-compliant.
Exp2 also exports `trial_stages.csv`, `per_seed_stages.csv`, `summary.csv`,
`per_seed_curves.csv`, `curves.csv`, and `timings.json`. Trial/seed/checkpoint
weighting and all non-GRF representation/rollout definitions are unchanged.

Support is independently runnable after the notebook imports/config cell:

```python
import exp3_support
support = exp3_support.analyze(CONFIG)
```

Bounded `num_workers` now reaches support's chunk readers. FK transforms and
joint/foot mappings are precomputed, reductions operate on chunk arrays, and
checkpoint/training-seed metadata is grouped once rather than scanned per
metric/time bin. Per-recording caches include code, numerical options, URDF
content, schema/initial-state metadata, and chunk size/mtime signatures. Plotting
options do not invalidate numerical caches. Memory holds one recording's compact
derived references plus bounded chunks, not histories/full trajectories.

Focused checks (including CPU notebook smoke execution):

```bash
python -m pytest tests/test_exp3_contact.py tests/test_exp3_support.py \
  tests/test_exp3_representation.py tests/test_exp3_representation_speed.py \
  tests/test_exp3_rollout_statistics.py -q -p no:cacheprovider
```

### Validation and Benchmark (2026-11-01)

The focused suite passed 57 tests with three optional Torch-dependent checks
skipped in the base Python environment. Those three checks were run separately
in `lr_lab_cupiqp` with `CUDA_VISIBLE_DEVICES=''` and passed. Syntax compilation,
notebook schema validation, and scoped whitespace checks also passed.

On the identical 70,000-row PILOT pushing/loaded/seed-0 recording, unprofiled
numerical analysis plus exports (figures disabled in **both** versions) took
213.9 s before versus approximately 5.5 s with an empty analysis cache, about
39x faster. A warm numerical-cache rerun took 0.93 s. This is not a filesystem
cache-flushing benchmark. The old single-process peak RSS was 323 MiB; optimized
peak summed process RSS was approximately 340 MiB including two bounded readers
(sampled every 20 ms), versus approximately 300 MiB warm. The instrumented old
run spent 206/259 s in hierarchy aggregation; the optimized grouped aggregation
takes approximately 0.18 s. Unchanged scalar/curve results agree within 1.6e-8
(metres for the largest FK difference); counts match exactly.

A real cold smoke processed 12 Exp2 recordings, 840,000 rows, all three profiles
for PILOT/UniFP/Pos/A6 at evaluation seed 0 in 68.1 s including figures/exports.
Both compact PDFs/PNGs were inspected and regenerated from exported curves
without raw data. The URDF CoM fallback matched recorded CoM endpoint velocity
on 128 real samples with 2.23e-7 m/s RMS and 1.29e-6 m/s maximum discrepancy.
An Exp1 smoke scored 512 selected transitions and reconstructed all stream
endpoints on CPU without Torch/BARD/simulator initialization. Exp1 foot-only
net-GRF metrics remained unavailable because non-foot contacts were not recorded;
normal-GRF metrics and aggregate contact references were available.

The real vertical consistency diagnostic is **not uniformly small**: absolute
stage-mean discrepancies reach about 45 N under horizontal loading. These are
retained in `vertical_support_consistency.csv` and reported above the configurable
5 N warning threshold. Do not interpret this as a control advantage or an
independently validated force reference: the calculation assumes the recorded
world loads are complete/frame-correct. The source of that recording/model
discrepancy is not resolved here; no frame/force calibration or collection change
was made. Existing GPU BARD rollouts were not rerun by these real-data smoke tests.
