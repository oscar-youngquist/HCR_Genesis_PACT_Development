# Experiment 2 analysis

Open `notebooks/exp2_summarize_approach.ipynb` once per approach/checkpoint,
then `notebooks/exp2_compare_approaches.ipynb`. These notebooks use NumPy,
Pandas, Matplotlib, Jupyter/nbclient, and the existing `exp1_analysis.py`;
they do not import PyTorch, Isaac Lab, or BARD. Collection and Exp1 are unchanged.

## Configuration

In the first notebook set `input_roots`, registered `task`, `display_name`,
optional `checkpoint_sha256`, `expected_seeds`, and `output_dir`.
`loading` selects `['loaded']`, `['unloaded']`, or both, always kept separate.
Use identical `histogram_edges` across methods. `acceleration_metrics` opts
into unfiltered, reset/gap-safe velocity differentiation.

The comparison notebook takes summary roots, display names/order, focused
methods (PILOT, UniFP, Pos by default), one loading status, and editable
figure size/colors/font size. All discovered methods enter the main table.
Optional `latex_method_names` maps task IDs to existing manuscript macros
(for example `{"b1z1_pact": "\\\\PILOT{}"}` in a Python configuration literal).
It affects LaTeX names only; define those macros in the manuscript.
Optional metric columns use `optional_metrics`, for example
`['torso_angular_acceleration_rms', 'ee_linear_acceleration_rms']`.
No motion-mode table is produced: the exported stages keep stationary,
moving, start/stop ramps, and EE-trajectory motion distinct.

Run from the repository root, or edit and execute the cells interactively:

```bash
EXP2_ANALYSIS_CONFIG='{"input_roots":["/path/to/exp2"],"task":"b1z1_pact","output_dir":"/tmp/exp2-pilot"}' \
  jupyter nbconvert --execute --to notebook notebooks/exp2_summarize_approach.ipynb --output-dir=/tmp
EXP2_COMPARE_CONFIG='{"summary_roots":["/tmp/exp2-pilot"],"output_dir":"/tmp/exp2-comparison"}' \
  jupyter nbconvert --execute --to notebook notebooks/exp2_compare_approaches.ipynb --output-dir=/tmp
```

## Semantics and validation

- Discovery uses metadata, not directory names. Completed runs require
  `metadata.json`, `records.json`, `trials.json`, `seed_summary.json`, and
  `COMPLETE`. Chunk schemas, trial identities, terminal rows and timestamps
  are validated. Only analysis fields for unfinished trials are buffered.
- One approach summary may contain only one checkpoint hash and compatible
  protocols/effective mechanics. Comparison requires matching protocols,
  units, histogram bins/frames, and available per-profile evaluation seeds.
  Native gains may differ across approaches; their effective environment
  configurations remain in provenance. Duplicate method summaries are errors.
- Tracking errors use finite pre-failure samples, then equal trial means
  within seed, then equal seed means. S_track divides by the entire prescribed
  horizon, including zero credit after failure. Zero-valid-sample trials
  retain failure/score outcomes but have undefined tracking errors.
- Physical-limit percentages pool recorded valid joint-substep counts and
  denominators within each seed, then weight seeds equally. They use recorded
  saturation counts, not rated limits or interval-averaged torques.
- Tables use sample SD across evaluation seeds (`ddof=1`); one seed has
  undefined SD. Fractions become percentages only for table display. CSV
  statistics retain explicit native units. Missing results are not zeros.
- Load-event recovery reuses Exp1's window/dwell and whole-trial failure
  censoring: failed trials are unrecovered even if an earlier window tracked
  successfully. Only complete windows enter comparable peak/integrated-error
  summaries; partial observations remain in trial/event records. Unloaded
  controls have no eligible load-change events. Base-command events remain
  separately identified and are not treated as load recovery events.
- Stage scores use successor-time schedule boundaries and prescribed stage
  denominators, including failed tails. The 25 N, 45 N, 65 N moving, loaded
  stopping ramp, and 65 N holding stages remain separate. Stage names in
  unloaded runs refer to task timing, not actual nonzero loading.
- Histograms use signed measured-minus-nominal EE error in the successor
  base-yaw frame [m]. Counts include explicit tails; probability is normalized
  within seed using all valid samples, never just in-range samples.

## Portable schema (version 1)

Per-approach exports:

| File | Contents |
| --- | --- |
| `summary.json` | Protocol/hash, checkpoint/training/evaluation provenance, per-seed metrics and counts, equal-seed aggregates, stages, trial/event censoring details, diagnostics |
| `summary.csv` | Profile/loading/metric, units, seed mean/sample SD and contributing counts |
| `stages.csv` | Profile/loading/stage-specific equal-seed statistics |
| `ee_histograms.csv` | Seed/profile/loading/axis/bin counts, valid denominators, underflow/overflow and probabilities |
| `time_series.csv` | Seed/profile/loading/curve/step, actual timestamp, survivor count, attempted trials, sum, sum of squares and mean |
| `COMPLETE` | Written only after all exports succeed |

Undefined JSON values are `null`. Checkpoint training seed is recorded only
when known; three evaluation seeds are **not** three independently trained
policies. Online seed summaries are cross-checked where metric definitions agree.

Comparison exports `main`, `physical_limits`, and `response` CSV/LaTeX tables,
`task_response.pdf/.png`, `task_response.csv`, and `comparison.json`.
LaTeX needs `booktabs` and `graphicx`; main uses `table*`, focused tables use
`table`. Display names follow the Exp1 convention. Times New Roman is used
when installed, otherwise DejaVu Serif.

## Response curves and variability

The 3x3 figure uses profile columns. Actual interval-mean applied forces [N]
are plotted at **interval-start timestamps**: initial-heading-frame Fx for
pushing/pulling, world Fz for carrying. EE error norms and body-forward
velocity/commands use **successor timestamps and nominal references**.
Event markers come from the recorded schedule. Curves exclude failed and
post-failure samples and never interpolate across missing data.

At each time, average surviving trials within each seed, then average seeds
equally. Default `band='trial_sd'` uses the equal-seed mixture variance:
`E_seed[E_trial[x^2]] - E_seed[E_trial[x]]^2`. It describes surviving trial
spread, not a confidence interval or SEM. `band='seed_sd'` instead shows
sample SD across seed means (undefined for one seed). Unequal survivors do
not give a seed extra weight. The plot shows survivor-count ranges; exported
curve rows give exact contributing trial/seed counts and attempted counts
at every timestamp. These are survivor-conditioned curves, not estimates
of what failed robots would have done.

## Focused checks

```bash
python -m pytest -q -p no:cacheprovider tests/test_exp2_analysis.py
```

The notebook test executes both notebooks with synthetic recordings and
checks CSV/LaTeX/PDF/PNG exports without simulator imports. Jupyter requires
permission to open local kernel sockets. Synthetic fixtures cover all profiles,
loading separation, unequal trial lengths, failures around events, incomplete
windows, histogram tails, physical-limit pooling, timestamp/frame alignment,
survivors, missing methods, and incompatible protocols/seed coverage.
