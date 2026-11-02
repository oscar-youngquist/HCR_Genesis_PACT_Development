# Fixed-Checkpoint PILOT V02 Comparison

These new notebooks leave the earlier notebooks and output folders unchanged:

- `notebooks/exp1_pilot_v02_comparison.ipynb`: summarize and compare Exp1 task
  performance, motion-mode scores, physical limits, recovery, and EE distributions.
- `notebooks/exp2_pilot_v02_comparison.ipynb`: the same existing Exp2 metrics,
  stage diagnostics, and task-response figures for pushing, pulling, carrying.
- `notebooks/exp3_pilot_v02_comparison.ipynb`: representation quality, explicit
  estimates, one-step dynamics, action channels, and whole-body support. Excludes
  FiLM oscillation analysis. Representation and support cells run independently.

## Checkpoints and Configuration

There are **four** comparison entries (V02 plus three comparators):

| Analysis ID | Display | Recorded training run |
| --- | --- | --- |
| `pilot_v02` | PILOT V02 | `Oct29_15-14-37_b1z1_pact_improved` |
| `pilot_original` | PILOT (Original) | `Oct21_00-46-21_b1z1_pact_improved` |
| `unifp` | UniFP | `Oct23_15-25-20_upstream_architecture` |
| `pos` | Pos | `Oct23_16-58-40_position_only` |

`pilot_v02_comparison.defaults(ROOT)` configures the existing `evaluation/exp1`,
`evaluation/exp2`, and `evaluation/pilot_v02` roots. Edit `CONFIG` or individual
`CONFIG['methods']` selectors in each notebook. Discovery uses the training-run
name in recorded checkpoint metadata, then requires a unique checkpoint hash.
It also checks the recorded V02 architecture flag. It never guesses the newest
checkpoint. An ambiguous run requires an explicit `checkpoint_sha256` selector.
Missing cohorts are reported, not replaced with another policy. Both PILOT
entries keep their native `b1z1_pact` schema, but have separate analysis IDs;
no source metadata is rewritten and their training repetitions are never pooled.

All outputs default to `analysis/pilot_v02_comparison/`. Task summaries are under
`exp1/summaries/<ID>` and `exp2/summaries/<ID>`; comparisons are under the matching
`comparison/` directory. Existing CSV/JSON/LaTeX and PDF/PNG outputs are retained.
Completed task summaries are reused only when settings, analysis source, recording
metadata, and numeric chunk size/mtime fingerprints match. Exp3 reuses existing
per-recording numerical/BARD caches. Full eligible transitions remain the default.

Protocol checks remain strict. Like the current Exp1 notebook, the new notebooks
explicitly permit ignoring the evaluator-source hash only; timing, thresholds,
randomization protocol, utility hashes, and robot conventions still must agree.
Loaded/unloaded cohorts, loading ranges, stages, and motion selections stay separate.

## Confidence Intervals

Every contrast is **PILOT V02 minus comparator**. Negative is favorable for errors,
but positive is favorable for tracking/recovery fractions. Units stay native:
fractional scores are not multiplied by 100 again. Checkpoint and evaluation-seed
identities remain explicit. All intervals are pointwise, not multiplicity-adjusted.

Dedicated files:

- `exp1/exp1_metric_confidence_intervals.csv`
- `exp2/exp2_metric_confidence_intervals.csv`
- `exp3/representation/full_recorded_first/representation_metric_confidence_intervals.csv`
- `exp3/representation/full_recorded_first/representation_rollout_confidence_intervals.csv`
- `exp3/action_support/support_metric_confidence_intervals.csv`

The general metric files bootstrap **paired evaluation-seed summaries**. This
preserves the original equal-trial tracking aggregation, physical-limit pooled
joint-substep denominators, event censoring, and stage definitions. It also keeps
representation axes/biases/reference variants separate. Support intervals cover
stage metrics, not thousands of correlated per-timestep tests. Corresponding JSON
files document settings, units, pairing counts, and missing-metric reasons.
Three evaluation seeds give limited interval resolution: these are exploratory
fixed-policy evaluation intervals, not uncertainty over independent training runs.
Seed-block pairing does not claim identical realized states or verified trial pairs.

The rollout file uses the existing **paired trial-RMSE percentile bootstrap within
fixed evaluation-seed strata**. Its three pairs compare V02 with Original PILOT,
UniFP, and Pos, not with the old PINN-ablation matrix. Recorded scenario and initial
mechanics verification remains required. Both own-pre-failure and exact shared
pre-failure-time sensitivities are exported, with `paired_rollout_matching.csv`
and `paired_rollout_trials.csv.gz` retaining the audits and paired measurements.
These are checkpoint/method comparisons, not isolated representation-PINN effects.
UniFP lacks a complete decoded-loading rollout; those contrasts remain unavailable,
while its supported explicit velocity/EE estimates remain comparable. Missing
predictions are never substituted with measured forces or zero errors.
The sampled real UniFP runs also have different initial joint positions from
PILOT, so they fail the strict trial-pairing check. Those trial-CI rows remain
unavailable with the rejection reason in the matching audit; seed-level supported
metric CIs remain available. No exact pairing is inferred from seed numbers alone.

## Running and Dependencies

Run all cells in each new notebook. NumPy, pandas, SciPy, Matplotlib, and Jupyter
support the recorded-data workflows. Exp3 optionally needs the existing PyTorch,
CUDA/BARD/Pinocchio environment for missing one-step dynamics. Set `dynamics=False`
and `fallback_predictions=False` for a CPU-only stored-data run; unavailable
dynamics remain explicit. No simulator, training runner, or optimizer is launched.
Set `run_representation=False` to run support alone.

For unattended execution, set `PILOT_COMPARISON_CONFIG` to a JSON override file:

```bash
PILOT_COMPARISON_CONFIG=/path/to/overrides.json \
  jupyter nbconvert --to notebook --execute \
  notebooks/exp1_pilot_v02_comparison.ipynb --output /tmp/exp1-v02-executed.ipynb
```

The existing Times New Roman/serif fallback, configurable font sizes, frame/unit
conventions, survivor handling, and momentum-support approximation labels apply.
