# Experiment 1 CPU analysis

Use `notebooks/exp1_summarize_approach.ipynb` first, once per approach/checkpoint,
then `notebooks/exp1_compare_approaches.ipynb`. Both use the standalone
`notebooks/exp1_analysis.py`; they do not import legged_gym, PyTorch, Isaac Lab,
BARD, or checkpoint contents. No training/evaluation settings are modified.

## Setup and use

In a CPU Python environment (Python 3.10+):

```bash
python -m pip install numpy pandas matplotlib jinja2 jupyterlab nbformat nbclient ipykernel pytest
jupyter lab notebooks/exp1_summarize_approach.ipynb
```

Edit the first configuration cell: input roots, registered task, optional exact
checkpoint SHA-256 from `metadata.json`, display name, expected evaluation seeds,
output directory, histogram edges/frame, combined-motion view, and optional
accelerations. Run All. Input discovery uses saved metadata, not folder naming.
Use the same histogram edges for every method. Default bins cover +/-0.5 m;
underflow and overflow are retained rather than renormalizing truncated data.

For comparisons, set summary roots, ordered task names, display names, the three
focused methods, **one** loading range and **one** motion selection. Missing
methods/conditions appear as an em dash and are listed in diagnostics. A combined
motion view must first be explicitly enabled in the summarization notebook.
Keep in-range and out-of-range exports in separate comparison directories.

Optional JSON configuration overrides are supported via `EXP1_ANALYSIS_CONFIG`
and `EXP1_COMPARE_CONFIG`. This is also how the notebook tests inject fixtures.

## Contracts and validation

- Require `COMPLETE`, metadata, records manifest, trials and online seed summary.
  Every referenced numeric chunk must exist and match the recorded shapes/dtypes.
- Reject duplicate trial/step or seed/condition/range identities, mixed checkpoint
  hashes, incompatible evaluation settings/timesteps/horizons/failure definitions,
  and changed effective environment settings within an approach.
- Evaluator/utility source hashes are included in protocol compatibility. A
  different hash requires reviewing/reprocessing recordings separately; the
  notebook does not silently assume equivalent code. Git hashes remain provenance.
- Backend family suffixes (`isaaclab_b1z1_pact` versus `isaaclab_b1z1_unifp`) are
  normalized to the same engine for cross-method comparisons. Per-method gains,
  architectures and action dimensions are intentionally not required to match.
- Seed, robot count and viewer-only randomized display indices are excluded from
  within-method config equality. All original resolved configs remain in provenance.
- Comparison rejects duplicate method summaries, mismatched units/bins/frames and
  unequal evaluation-seed coverage for a present condition. Missing conditions and
  fewer than three seeds are warnings, not fabricated observations.
- Evaluation seed and checkpoint training seed are separate fields. Unknown
  training seeds remain null; evaluation repeats are not independent training runs.

## Metric conventions

Tracking errors use recorded `alive_for_metrics`, finite successor states and
nominal references. Terminal failure rows are excluded; failed tails are not
padded. Per-trial means are averaged equally within seed, then equally across
seeds. `S_track` divides joint-tolerance-pass steps by the full prescribed horizon.
Failure rate includes all attempted trials, including those with zero valid
samples. Undefined values are JSON null; SD is sample SD (`ddof=1`), undefined
with only one contributing seed.

Physical-limit percentages pool recorded substep violation counts and valid
joint-substep denominators within a seed. They are not averages of trial
percentages and are not inferred from interval-averaged torques. Saturation uses
the recorder's actual clipping threshold, documented in `actual_clipping_limits`.
Online limit frequencies use a different seed reduction and are not cross-checked
as equivalent. Main tracking metrics and failure rates are checked against online
trial/seed summaries; differences are reported in diagnostics.

Events retain onset/change/removal type, eligibility, observed samples, partial
peaks/integrals and window-completeness flags. Only complete windows contribute
comparable peak/IAE means. Recovery is the first continuously successful dwell
within the configured window; it does not require an initial tolerance excursion.
Any physical failure marks **all** that trial's events unrecovered, matching the
online evaluator. Events after an early failure still count if scheduled within
the prescribed horizon. Recovery time averages recovered events only. Complete
event means and recovery fractions are computed within seed, then across seeds.

Optional smoothness uses backward differences of world-frame velocities at
successor timestamps, only between contiguous valid steps in one trial. No filter
is applied. Per-trial RMS is `sqrt(mean(sum(acceleration**2, axis=-1)))`, then
trial/seed averaging follows the tracking convention. Sample counts are retained.

Signed EE histograms are measured minus nominal reference in metres. The default
frame rotates world errors by the inverse **successor base yaw**, not full body
roll/pitch. Each seed histogram is normalized by all valid samples, including
out-of-bin tails; seed distributions are then averaged equally. Overlaid histogram
bars show percentage per bin (not density); error bars are sample SD across seeds,
not a CI. Bin widths and tail accounting are unchanged.

### Additional distribution plots

The comparison notebook's `additional_plots` setting defaults to `["step", "ecdf",
"box"]`; use an empty list to suppress these exports. The original filled histogram
remains available. Each option writes `ee_<option>.pdf` and `.png`:

- `step`: outline histograms, equal-seed percentage per bin, no smoothing.
- `ecdf`: binned cumulative-distribution approximation at bin edges. Underflow is
  included at the first edge; the last edge ends at one minus overflow. Unrecorded
  within-bin/tail positions are not invented. Bands show sample SD across seeds.
- `box`: exact weighted empirical quantiles from raw signed errors. Each nonempty
  evaluation seed has equal total weight, regardless of its valid sample count.
  The box is 25-75%, the line is the median, and whiskers are 5-95% (not Tukey
  fences). Extreme samples participate in the calculation; their min/max and
  sample/seed counts are exported, but individual outlier markers are not drawn.

Rerun the summarization notebook to populate `summary.json`'s `ee_boxplots` for
existing results, then rerun comparison. No new evaluation is necessary. Older
summaries can still supply outline/ECDF plots; missing box statistics are flagged.
`ee_ecdf_data.csv` and `ee_box_data.csv` preserve plot values for offline reuse.

## Outputs

Summarization writes `summary.json`, `summary.csv`, `ee_histograms.csv`, then an
analysis `COMPLETE` marker. JSON includes per-trial/event results, per-seed values,
counts/denominators, aggregate means/SDs, units, protocol and checkpoint provenance,
and missing-data diagnostics. A failed export removes the completion marker.

Comparison reads only these portable exports, never NPZ recordings. It writes
main, physical-limit and optional response tables as CSV and LaTeX; optional
metrics can have a separate table. Main LaTeX uses text width; focused tables use
column width (`graphicx` and `booktabs` required). Figures use Times New Roman
when installed, otherwise DejaVu Serif, and export vector PDF plus 300-dpi PNG.
`ee_distribution_data.csv` preserves mean/SD probabilities, seed counts and tail
fractions. `comparison.json` records selection, provenance and diagnostics.

## Focused validation

```bash
python -m pytest -q -p no:cacheprovider tests/test_exp1_analysis.py
```

Tests create synthetic recordings with analytically known results, exercise
failure censoring, unequal lengths, empty samples, recovery, tail accounting,
seed weighting, pooled limit denominators, reset-safe differentiation and strict
validation. With nbformat/nbclient installed, they execute the summarization
notebook for three synthetic approaches and the comparison notebook end-to-end.
Jupyter requires local kernel sockets. Executed notebooks and exports are written
under pytest's temporary directory, not into the tracked notebooks.
