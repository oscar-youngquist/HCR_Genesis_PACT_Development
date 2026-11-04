> Historical validation results below used the settings recorded for each run.
> Current collection and visualization defaults are 10 mm particles, four substeps,
> analytic-plane contacts and 5 mm softness; see [containment experiments](liquid_containment_experiment.md).

# Local collection validation

Executed November 3, 2026 on branch `liquid_wrench_surrogate`, using the existing
patched `genesis_lr` environment, Genesis 0.3.11, Torch 2.8.0+cu126 and `cuda:0`
(NVIDIA RTX 4090). No installation changes, training, large campaign, commits or
pushes were performed. See [the collection guide](liquid_surrogate_collection.md)
for commands, schema and reconstruction requirements.

## Selected settings and runtime

Collection defaults are 10 mm particles, eight internal substeps per 2 ms PD step,
and six attached box parts. The 10 ms policy clock stays unchanged. Box walls
overlap by 2 mm at corners; contact softness is half the particle size. Wall
density compensates for the overlap's added volume. Mount offsets remain within
the original bounds, including fixed robot-specific mounting Z. Property endpoints
expand by 10% and geometry/fill-offset endpoints by 2% by default.

The final matched benchmark used seed 1, two Go2 environments and 25 policy steps
per profile. Profiles ran sequentially in fresh subprocesses with the same
payload, commands and descending terrain. The reference used the original mesh,
12 mm particles and two substeps, with the same domain/reset corrections.

| Profile | Initialization (s) | Steady seconds/policy step | Sampled process GPU peak (MiB) | Peak process RSS (KiB) |
| --- | ---: | ---: | ---: | ---: |
| Original mesh player loop | 14.706 | 1.816198 | 872 | 9,232,668 |
| Box player loop, 10 mm/eight substeps | 12.679 | 0.207450 | 936 | 9,278,768 |
| Box collection, 10 mm/eight substeps | 12.501 | 0.376886 | 936 | 9,285,672 |

Collection took **20.75% of the original player's steady runtime** (4.82 times
faster). Recording added 81.68% to the much faster box player loop. The automated
runtime-budget check passed. GPU memory includes Genesis allocations outside
Torch and was sampled every 0.5 s; collection's Torch allocated peak was
20,793,856 bytes. These results establish the budget at two environments, not at
100 environments or across all randomized payloads.

Raw benchmark: `/tmp/liquid-final-contact-benchmark/round-00000/benchmark.json`.
Each profile's `attempt-0000` retains metrics, resolved settings and provenance.
The baseline/reference profiles intentionally omit the recorder, so their
`smoke_passed: false` means full trajectory assertions were not run; collection
passed them. Resuming this completed benchmark with the identical CLI and
`--resume` returned success without creating new attempts or launching GPU work.

Reproduce the benchmark and longer smoke in fresh output directories:

```bash
~/.conda/envs/genesis_lr/bin/python -m liquid_collection.plan \
  liquid_collection/example_manifest.json --output /tmp/liquid-benchmark-repeat \
  --seed 1 --num-envs 2 --episodes 2 --max-steps 25 --max-seconds 300 \
  --smoke --benchmark
~/.conda/envs/genesis_lr/bin/python -m liquid_collection.plan \
  liquid_collection/example_manifest.json --output /tmp/liquid-long-repeat \
  --seed 1 --num-envs 2 --episodes 2 --max-steps 180 --max-seconds 300 \
  --episode-seconds 1.6 --smoke
```

## Live trajectory checks

The longer Go2 smoke used two environments, 1.6 s episodes, seed 1 and the final
contact settings. It completed 161 policy steps, with one completed episode per
environment and a forced partial reset. All 12,880 environment/native intervals
were structurally valid. There were no outer-envelope escapes, inactive particles,
domain contacts, attachment failures or nonfinite states. Actual particles reached
world Z = -0.2773 m. Steady collection cost was 0.392256 s/policy step.

There were 6,261 environment/row nominal-cavity penetration flags, with maximum
penetration 9.17 mm into wall material. Thus this is **outer-envelope containment**,
not strict containment inside the nominal cavity. The dedicated penetration fields
remain available for downstream filtering and contact-quality assessment.

Streaming audit of `/tmp/liquid-contact8-long/round-00000/collection/attempt-0000`:

| Audit | Result |
| --- | ---: |
| Stored rows / advancing native intervals | 6,608 / 6,440 |
| Policy / feature events (environment counts) | 322 / 160 |
| Structurally valid 249-sample histories | 138 |
| Structurally valid history plus 200 ms windows | 58 |
| Maximum native force closure residual (N s) | 0.00081648 |
| Maximum native fixed-world-origin moment closure residual (N m s) | 0.0102624 |
| Maximum 200 ms force closure residual (N s) | 0.00500010 |
| Maximum 200 ms fixed-world-origin moment closure residual (N m s) | 0.0587239 |

These counts do not certify pointwise labels or JVP reconstruction. All
instantaneous-wrench and derivative-reconstruction validity flags remain false;
`pinn_ready` remains false. Nonzero momentum residuals are retained for numerical
auditing rather than silently treated as exact conservation.

The final Go1 negative-height smoke also passed with the same particle/contact
defaults: 21 steps, two completed episodes, no validity failures or nominal-cavity
penetrations, 0.413039 steady s/policy step and 1,128 MiB sampled GPU peak.
Raw output: `/tmp/liquid-final-go1/round-00000/collection/attempt-0000`.

A separate Go2 run using randomized training terrain (without `--smoke`), two
environments and a 0.2 s episode override completed 20 steps and two episodes.
All recorded validity/penetration failure counts were zero. Terrain generation
and initialization took 62.642 s; steady collection took 0.381629 s/policy step.
Its GPU memory monitor reported an `nvidia-smi` timeout, so its sampled peak is
not treated as a reliable memory bound. Raw output:
`/tmp/liquid-random-terrain/round-00000/collection/attempt-0000`.

Earlier trials with the original contact softness escaped during longer runs,
including 10 mm particles at six and sixteen substeps. An 8 mm/four-substep trial
also escaped. The selected contact geometry and response passed the longer smoke
at eight substeps. These are local trials, not an exhaustive resolution optimum
or proof of stability over the full parameter envelope. Failed attempts remain
under their original `/tmp/liquid-*` paths.

## Software and provenance checks

Executed successfully:

```bash
SIMULATOR=genesis_pact_water NUMBA_CACHE_DIR=/tmp/liquid-numba \
MPLCONFIGDIR=/tmp/liquid-mpl \
~/.conda/envs/genesis_lr/bin/python -m pytest -q \
  tests/test_liquid_collection.py tests/test_go2_liquid_payload.py \
  tests/test_pact_liquid_checkpoint.py
```

**21 tests passed**. Coverage includes bounds and fixed mounts, safe fill/box
geometry and overlap mass, frames and wrench signs, actual-mass aggregates,
causal timing, partial resets and terminal capture, episode-isolated histories,
JVP tap margins, deterministic multi-policy planning/resume, chunk corruption and
writer failures, and runtime-budget failure handling. Compilation and
`git diff --check` passed. Three-round planning at 100 environments passed as a
dry run only; no 100-environment simulation was launched.

The installed locally modified sources retain their original inspected hashes:

| Genesis source | SHA-256 |
| --- | --- |
| `engine/entities/rigid_entity/rigid_entity.py` | `14eda00b636030a364e35877a31e086a59fca44e38ea7558ae37db457183415a` |
| `engine/solvers/sph_solver.py` | `5b5bc988cb3eebe6bbac367ac21eed5dc73e743cc61d979dc2dd696b04a71143` |

Workers snapshot these sources and other relevant installed/repository files,
package versions, checkpoint/config identities, generated terrain and actual
liquid properties. The specification PDF SHA-256 is
`370a3aac7a68bb5f02f0f6167167151525ede6a2905143e80dc0aa07c65167c7`.

Remaining validation before a large campaign: broader payload/terrain coverage,
longer trajectories, progressive environment-count profiling, acceptable wall
penetration and conservation tolerances, and independently validated continuous
trajectory/wrench reconstruction for the PDF's pointwise losses.

## Visualization follow-up

The active preset-based Go1/Go2 visualizers now also default to the box contacts,
10 mm particles and eight substeps. The benchmark reference still explicitly
selects the pre-swap mesh settings. The new `visualize_liquid_collection.sh` runner
samples from the collection planner and provides a translucent-wall preview.

Verified in `genesis_lr` on `cuda:0`: a two-environment interactive launch, a
three-step headless render, and three-step two-environment `--smoke` runs through
the existing player for both Go1 and Go2. Both preset smoke runs passed finite
state, all-part attachment and partial-reset checks. The final rendered image
`/tmp/liquid-visualization.png` was visually inspected: box walls are translucent
and the blue particles are visible. An explicit wall color was needed for the
installed rasterizer to honor opacity; actual-position particle markers avoid
its glass/particle occlusion. Collection keeps those markers disabled.

The 21 focused tests were rerun successfully after the changes. Python compilation,
launcher shell syntax and `git diff --check` passed. These short visual checks do
not extend the longer trajectory stability evidence above.

## Default 200 Hz saving

Collection now defaults to `--save-mode 200hz`; `--save-mode native` retains the
earlier dense debug format. Physics resolution and controller clocks are unchanged.
The primary stream stores endpoint states every 5 ms, summed native world impulses,
fixed-endpoint-reference interval-average wrenches, endpoint velocity differences,
and conservative interval validity diagnostics. Compact exact-time events and
full reset/partial-interval boundary records are separate, checksummed streams.

A new two-environment, seed-1, 1.6 s Go2 smoke passed in `genesis_lr` on `cuda:0`:
`/tmp/liquid-200hz-validation/round-00000/collection/attempt-0000`.

| Stream | Rows | NPZ bytes |
| --- | ---: | ---: |
| Primary 200 Hz endpoints, including time zero | 323 | 424,903 |
| Exact-time events | 248 | 45,176 |
| Terminal/reset boundaries | 6 | 25,226 |
| Total | 577 | 495,305 |

The earlier equal-duration dense run used 7,914,392 NPZ bytes: the new output is
**15.98 times smaller**, including all auxiliary trajectory streams. Fixed
provenance/assets/metadata are excluded from both sizes. Primary timestamps are
exactly 5 ms apart; all 322 advancing intervals contain 20 native substeps,
accounting for all 6,440 substeps. CPU recomputation of the saved origin shift
agreed within 4.77e-7 N m s (float32 CPU/GPU rounding), and impulse/duration
averages were checked. No particle snapshots were enabled.

The audit retained 138 structurally valid 249-sample histories and 58 complete
history-plus-200 ms windows. Maximum 200 ms closure residuals were 0.00490661 N s
and 0.0571291 N m s about the fixed world origin. Native-resolution closure is
accumulated during collection; individual native residuals are not recoverable
from the reduced output. Pointwise labels/JVP reconstruction remain unvalidated.

Steady collection cost was 0.368685 s/policy step, with 936 MiB sampled GPU peak.
All escape/domain/activation/attachment/nonfinite failure counts were zero;
nominal-cavity penetration remains recorded separately. A 21-step native-debug
smoke also passed at `/tmp/liquid-native-mode-validation`, retaining all 840 native
intervals. Resume of the completed 200 Hz run verified all auxiliary streams and
skipped simulation successfully.

**25 focused tests passed**, including summed impulse signs and shifting under a
changing base orientation, endpoint averaging, transient invalidity, off-grid
reset splitting without lost/double-counted impulses, shifted feature-event
timing, and auxiliary-stream corruption on resume. Python compilation and
`git diff --check` passed. The dense trajectory cannot be reconstructed exactly
from 200 Hz output; use explicit native-debug validation runs when needed.

## Top-level campaign launcher

`launch_liquid_collection.sh` exposes platform selection, environment count,
episodes per environment, payload rounds and episode time limit. It uses a
per-environment quota, marks surplus batched trajectories with
`episode_selected: false`, and tests each sampled payload on every selected
platform. The existing lower-level total-episode budget remains supported.

Executed the launcher with `go1-dact go2-dact`, two environments, one episode per
environment, one round and a 0.05 s episode cap, using `genesis_lr` on `cuda:0`.
Both workers completed five policy steps, reported raw/selected episode counts
`[1, 1]`, and saved trajectories ending at exactly 0.05 s. Both used the same
sampled payload. All containment, activation, attachment, finite-state,
domain-contact and nominal-cavity-penetration failure counts were zero in this
short check. Raw output is under `/tmp/liquid-launcher-validation`.

Verified all stream checksums, post-quota masks, effective episode duration and
successful campaign resume without new worker attempts. CLI dry-run created no
output directories; selecting reserved `go2-hardpact` failed before planning or
starting DACT work. **28 focused tests passed**, including unequal per-environment
completion counts, matched multi-platform rounds, immutable campaign resume and
the explicit unsupported-HardPACT guard. Python compilation, shell syntax and
`git diff --check` passed. No large campaign was started.
