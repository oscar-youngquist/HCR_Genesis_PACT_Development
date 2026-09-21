# cuPIQP pool, retained-gradient, bucket, and diagnostics tests

## Implementation and API checked

Inspected the installed cuPIQP 0.1.0 `DenseSolver.update`, `solve`, `backward`,
and result-info implementation in `lr_lab_cupiqp`. `update` preserves allocations
for fixed structure; every supplied matrix/vector is updated. `solve` initializes
its iterates anew. Backward buffers are mutable. Per-row iteration counts are
already-host-resident NumPy arrays; reading them adds no CUDA transfer.

New `algorithm.hard_pact_qp` settings in `go2_hard_pact_config.py`:

```python
cupiqp_ppo_reuse = True             # False: fresh-instance reference
cupiqp_ppo_pool_size = 8            # pooled instances, busy + idle
cupiqp_rollout_cache_size = 4       # idle capacity buckets, LRU eviction
cuda_event_profiling = False       # True: synchronize queued events at logging
```

The PPO key includes shapes/block presence, device, dtype, CUDA stream, and
settings. A graph owns its solver until its autograd context is destroyed—not
until the first backward. Retained graphs/PCGrad and graph abandonment are covered.
Busy entries cannot be evicted; pool overflow uses private fresh solvers. Failed
reuse retries the same problem fresh; failed instances never return to the pool.
Forward results and every VJP own their storage. All changing PPO data refresh,
including elastic Hessians, with the unchanged PPO preconditioning profile.

Rollout requests select their smallest power-of-two bucket, capped at the existing
chunk limit. Large earlier requests cannot inflate later small ones. Independent
padding QPs are omitted from outputs and statistics. Constant-Q updates retain the
existing fast path; elastic Q updates refresh both Hessian and preconditioning.
Dense mode, CUDA-graph default, numerical policies, and chunk sizes are unchanged.

## Logging definitions

One writer emits `qp/rollout/*` and `qp/ppo/*` once per iteration. Solver calls
accumulate scalar sums on-device; no per-environment matrices are exported.

- `real_rows`: actual input rows across solve calls, before fallback compaction.
- `final/{full,relaxed,elastic,analytic}_{count,fraction}`: mutually exclusive;
  fractions divide accumulated counts by the common `real_rows` denominator.
- `attempt/*`: attempted rows and exceptions per stage, separate from outcomes.
- `certified_fraction` versus `differentiated_fraction`: stopgrad remains certified
  but is not differentiated. PPO stopgrad statistics belong to `qp/ppo`.
- `held/*`: held-correction substeps, excluded from rollout solve denominators.
- `backend/*`: setup/update counts, pool hits/misses, requested/capacity/padded rows,
  fresh retries, and real-row solver iteration mean. Capacity counts include
  solver attempts, not just final outcomes.
- `profiling/*_ms`: optional CUDA-event totals for assembly, packing, setup/update,
  solve, certification/recovery, backward, total collection, and total update.
  Normal mode records no events and does not synchronize for profiling.

Unavailable measurements are NaN, with backend/profiling availability flags.
Merged p95 values are NaN: averaging chunk/minibatch p95s is not a valid global
quantile. Other physical/full summaries retain diagnostics-level gating. Legacy
ambiguous `qp/minimal/*` solve tags are no longer emitted by the training runner.
Internal per-solve results remain available to existing tests and callers.

## Conditions, commands, and results (2026-09-08)

Environment unchanged: Python 3.11.16, Torch 2.7.0+cu128, cuPIQP 0.1.0,
CuPy 13.6.0; GPU tests use physical GPU 1, NVIDIA RTX 4090 (24 GiB).
Tests use small canonical synthetic QPs, not simulator trajectory benchmarks.

```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=1 conda run -n lr_lab_cupiqp \
  python -m pytest -q tests/test_hard_pact_cupiqp_pool.py \
  tests/test_hard_pact_qp_reuse.py tests/test_hard_pact_qp_optional_solvers.py --tb=short
```

40 passed, one `hppfcl` deprecation warning (final repeat: 20.03 s).
Fresh/reused forward and input-VJP tolerances: float64 `rtol=atol=2e-6`,
float32 `rtol=atol=2e-4`; elastic PPO float64 `2e-6`. Profiling on/off and retained
VJP repeats match exactly. Existing qpth-reference tolerances remain unchanged.
No tolerances were relaxed. Three equal-shape PPO forwards use one setup/two
updates. Rollout sizes 17/2/3/2 use capacities 32/2/4/2 with a two-entry bound:
three setups/one update; padded rows never enter final stage counts.

```bash
env SIMULATOR=isaaclab conda run -n lr_lab_cupiqp python -m pytest -q \
  tests/test_hard_pact_cupiqp_pool.py tests/test_go2_hard_pact_qp.py \
  tests/test_hard_pact_qp_reuse.py tests/test_hard_pact_two_anchor_smoke.py \
  tests/test_hard_pact_ppo_qp_sampling.py tests/test_go2_hard_pact_ablations.py \
  tests/test_hard_pact_speed_optimizations.py tests/test_hard_pact_auxiliary.py \
  tests/test_hard_pact_ppo_latent_replay.py tests/test_pc_grad.py --tb=short
```

104 passed, 18 GPU-only skips, 15 subtests passed, three warnings (9.87 s).
Two warnings deliberately exercise solver exceptions; the other is `hppfcl`.
Coverage includes unchanged qpth constraints/fallbacks, network gradients,
two-anchor reset/substeps, disjoint PPO sampling, replay likelihood, and PCGrad.

```bash
env SIMULATOR=isaaclab conda run -n lr_lab_cupiqp python -m pytest -q \
  tests/test_hard_pact_qp_warmup.py --tb=short
```

8 passed / 8 failed (6.72 s): all failures are the existing assertion that the
editable training config has `warmup_iterations == 0`; the user's current value
is `2000`. Neither that value nor the unrelated assertions were changed.

No full training/simulator benchmark was run, and no training-speed improvement
is claimed. These tests establish allocation reuse and numerical/gradient safety,
not a measured end-to-end speedup.

## Files changed for this stage

- `rsl_rl/algorithms/hard_pact_qp_backends.py`, `hard_pact_qp.py`,
  `hard_pact_qp_diagnostics.py`, `ppo_hard_pact.py`.
- `rsl_rl/runners/pact_runner.py`.
- `legged_gym/envs/go2/go2_hard_pact/go2_hard_pact.py` and
  `go2_hard_pact_config.py`.
- `tests/test_hard_pact_cupiqp_pool.py`, `test_hard_pact_qp_reuse.py`,
  `test_go2_hard_pact_qp.py`, `test_go2_hard_pact_ablations.py`.
- This report and a historical-note link in `HARDPACT_QP_CAPACITY_REUSE.md`.

Other pre-existing workspace changes were preserved, including the edited
warmup and gap-policy settings.
