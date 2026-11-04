# Native versus 200 Hz collection: corrected-contact validation

Date: 2026-11-03.

**Result:** 200 Hz aggregation preserves the native impulses and passes the tested
200 ms integrated objectives in all six cases, using 89.2% less storage. All 12
environments complete six seconds without escapes. Instantaneous wrench and local
conservation checks fail; input-derivative accuracy remains inconclusive. This
supports interval-average recording but does not validate the complete local
physics/JVP training objective.

Six cases (Go1/Go2 DACT × representative/high-fill/viscous), two environments each, six simulated seconds per environment. The experiment used `genesis_lr` on `cuda:0`, 10 mm particles, four substeps, inward-facing analytic-plane contacts and 5 mm softness. Native recording is 2,000 Hz; compact recording is 200 Hz.

The same native trajectory was reduced offline using the production accumulator and independently checked using NumPy. This compares information preservation without differences between independent physics runs. No raw particle trajectories were saved. Acceptance thresholds and payload seeds match the earlier analysis.

Simulation workers used 15.09 minutes. Native state/impulse chunks total 97.79 MB; compact chunks including sparse events and boundaries total 10.54 MB: **9.28× smaller** (89.2% reduction). These byte counts exclude shared source/provenance files and plots.

| Case | Impulse accounting | Coverage | Instantaneous wrench | Input derivative | Local conservation | JVP plumbing | Integrated |
|---|---|---|---|---|---|---|---|
| go1-dact-high_fill | pass | True | fail | inconclusive | fail | pass | pass |
| go1-dact-representative | pass | True | fail | inconclusive | fail | pass | pass |
| go1-dact-viscous | pass | True | fail | inconclusive | fail | pass | pass |
| go2-dact-high_fill | pass | True | fail | inconclusive | fail | pass | pass |
| go2-dact-representative | pass | True | fail | inconclusive | fail | pass | pass |
| go2-dact-viscous | pass | True | fail | inconclusive | fail | pass | pass |

Worst environment per case; normalized RMS as percentages. Force uses mg, torque mgL; integrated position/momentum/angular-momentum use L, mg·0.2 s and mgL·0.2 s. Integrated values below take the largest of those three channels.

| Case | Force RMS | Torque RMS | Integrated excess RMS | Compact integrated residual RMS |
|---|---|---|---|---|
| go1-dact-high_fill | 11.85% | 4.53% | 2.33% | 2.34% |
| go1-dact-representative | 11.01% | 3.92% | 1.84% | 1.86% |
| go1-dact-viscous | 5.87% | 1.76% | 0.86% | 0.85% |
| go2-dact-high_fill | 13.35% | 5.51% | 2.41% | 2.43% |
| go2-dact-representative | 11.28% | 4.48% | 1.73% | 1.74% |
| go2-dact-viscous | 7.40% | 2.30% | 0.92% | 0.93% |

## Criteria and coverage

The frozen reconstruction limits are 5% normalized RMS and 10% normalized p95.
Integrated residuals must stay below 10% RMS, with at most 5% RMS excess over the
native-resolution result and 10% p95. JVP plumbing uses a 1e-5 tolerance. Each
environment must provide at least 100 histories, 100 collocation points, 40
integrated windows and 20 policy observations in each forward/stop/turn phase.
No thresholds were relaxed after the results were inspected.

Each 5 ms output interval combines ten 0.5 ms native steps. Impulses are summed
in a common world frame and moment reference, then shifted to the endpoint base
origin and rotated. Endpoint states, exact sparse events and partial boundary
intervals are retained. The accounting audit checks sums, durations, frames,
endpoints and events independently of the production reducer.

These are two-environment, six-second format tests. They do not establish
50-environment long-duration reliability; the separate containment/runtime
experiments address that question within their own tested scope.


| Case | Valid end by env (s) | First escape by env | Histories / collocations / windows by env |
|---|---|---|---|
| go1-dact-high_fill | [6.0, 6.0] | [None, None] | [(949, 100, 228), (949, 100, 228)] |
| go1-dact-representative | [6.0, 6.0] | [None, None] | [(949, 100, 228), (949, 100, 228)] |
| go1-dact-viscous | [6.0, 6.0] | [None, None] | [(949, 100, 228), (949, 100, 228)] |
| go2-dact-high_fill | [6.0, 6.0] | [None, None] | [(949, 100, 228), (949, 100, 228)] |
| go2-dact-representative | [6.0, 6.0] | [None, None] | [(949, 100, 228), (949, 100, 228)] |
| go2-dact-viscous | [6.0, 6.0] | [None, None] | [(949, 100, 228), (949, 100, 228)] |

The earlier runs passed impulse accounting but could not establish physics adequacy because escapes truncated valid trajectories and maneuver coverage. This rerun evaluates those questions with the corrected contacts; per-case results above remain separate rather than pooling environments into a pass.

Passing accounting establishes that interval-average wrench labels preserve native impulses and reference-frame transformations. Passing integrated checks applies to the specified 41-endpoint, 200 ms trapezoidal objectives and the tested payloads/maneuvers. It does not establish accurate instantaneous forces or accelerations.

The native kinematic reference remains sensitive to float32 pose quantization when differentiated at 0.5 ms. The report includes the separate endpoint-velocity/interval-acceleration reference checks; an inconclusive derivative result is not a pass. The synthetic JVP plumbing test verifies differentiation mechanics, not the physical accuracy of reconstructed inputs.

Retain 200 Hz aggregation for interval-average labels. The remaining local-residual/instantaneous-wrench issues require a separate reconstruction or selected-channel diagnostic before claiming the complete original training objective is supported. No production recording format or thresholds were changed by this rerun.

## Artifacts

- [Full per-environment report](../runs/liquid_validation_planes10_4_20261103/report.json): errors, p95/max, sensitivity, conservation floors, command coverage and accounting.
- [Comparison summary](../runs/liquid_validation_planes10_4_20261103/comparison_summary.json): selected metrics and earlier verdicts.
- [Summary plot](../runs/liquid_validation_planes10_4_20261103/comparison_summary.png).
- [Frozen criteria](../runs/liquid_validation_planes10_4_20261103/criteria.json) and [execution timing](../runs/liquid_validation_planes10_4_20261103/execution.json).
- Each case retains native chunks, `compact_200hz`, independent validation results, command/force/diagnostic plots and resolved configuration.
- Analysis scripts and source hashes are retained in `analysis_sources` and `analysis_provenance.json`.

## Reproduction

Run from the repository root with a fresh output directory:

```bash
SIMULATOR=genesis_pact_water NUMBA_CACHE_DIR=/tmp/liquid-numba \
MPLCONFIGDIR=/tmp/liquid-mpl \
/home/oyoungquist/.conda/envs/genesis_lr/bin/python \
  -m liquid_collection.validation.run --output runs/new_format_validation

MPLCONFIGDIR=/tmp/liquid-mpl \
/home/oyoungquist/.conda/envs/genesis_lr/bin/python \
  -m liquid_collection.validation.analyze runs/new_format_validation
```

The runner selects `cuda:0`. The recorded experiment manifest contains the exact
plans, seeds and configuration; each attempt includes resolved settings and source
snapshots. This run analyzed completed cases while subsequent GPU cases ran; the
archived orchestration calls the same `analyze_case` implementation as the CLI.

See the [method](liquid_format_experiment.md),
[contact experiment](liquid_containment_experiment.md), and
[previous results](../runs/liquid_validation_20261103_summary/summary.md).
Detailed run artifacts are local under `runs/`; this document remains in the
repository documentation independently of those larger artifacts.
