# First Plus Random Substep QP

Set the existing `algorithm.hard_pact_qp` dictionary entry:

```python
'qp_update_mode': 'first_plus_random_substep',
```

The compatibility dataclass default remains `random_one_substep`; the current
HardPACT experiment config selects `first_plus_random_substep`. With four physics substeps per
policy interval, the new mode solves every environment at k=0 and again at
K sampled from {1,2,3}. Later-index counts differ by at most one across
environments; random offsets/permutations refresh every policy interval.
Only selected rows build mechanics and enter each later batched solve.

Before execution, an independent fair coin per environment selects k=0 or K
as the single PPO replay anchor. Selection never depends on solver success;
failed/recovery solves retain their existing masks. Existing sampled-state,
previous executed torque, controller and delayed-action records are reused.
Detached nominal mechanics are copied at the selected anchor and gathered
directly using selected global rollout indices during replay. Missing or
incompatible cache rows use mechanics recomputation. There is no
second stored transition, differentiated solve, loss multiplier, or change to
the existing PPO row/epoch budget or normalization.

Unselected substeps recompute fresh nominal torque with the existing bounds
and optional rate clamp. No correction is held or accumulated. Blending,
recovery/fallback and feedforward reconstruction are unchanged.

Existing rollout solve-count diagnostics report two solves per environment;
`sampled_substep/*_count` reports replay-anchor coverage (approximately half
at zero and one sixth at each later phase). Deployment metadata records the
new selection semantics. Existing modes retain their selection and RNG use.

## Runtime switches

`qp_shared_later_substep=False` preserves the balanced per-environment later
selection above. Opting in selects one uniform later k for the entire interval:
two full rollout dispatches, still two problems per environment. The replay
coin remains independent per environment. This changes cross-environment
correlation, not marginal later-step probabilities or unselected execution.
Sampling uses the existing Torch generator/RNG and resume mechanism.

Primary cuPIQP buckets default to `cupiqp_primary_bucket_policy='tight'`,
`cupiqp_bucket_alignment=128`: 1365→1408 and 4900→4992. Tiny requests retain
smaller power-of-two capacities; chunk limits cap padding. Recovery retains
its power-of-two policy. `power_of_two` is the rollout reference;
`cupiqp_ppo_primary_capacity_reuse=False` restores exact-size PPO allocation.
`direct_qp_mechanics_gather=False` restores full-minibatch mechanics gathering.

`compile_tensor_ops=False` is intentionally opt-in. When enabled, dynamic-batch
Inductor kernels cover affine objective contributions, constraint row scaling,
and certification under no-grad. Solver calls/pools/recovery dispatch stay
eager and CUDA graphs remain off. Differentiable PPO kernels stay eager because
Torch 2.7's donated-buffer backward is incompatible with retained PCGrad VJPs;
no global compiler setting or backward ownership changes. Structural changes
may specialize kernels; numerical curriculum weights are tensor inputs.

Bounded matched-checkpoint comparison (activate `lr_lab_cupiqp`, from repo root):

```bash
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" SIMULATOR=isaaclab PYTHONPATH=. \
python scripts/benchmark_hard_pact_runtime_short.py \
  --task go2_hard_pact_full_isaaclab --headless --num_envs 16 --gpu cuda:0 \
  --seed 42 --qp_solver cupiqp --dynamics_backend bard \
  --runtime-checkpoint /absolute/path/model.pt \
  --runtime-variant reference --warmup-iterations 1 --measurement-iterations 2 \
  --runtime-report /tmp/runtime_reference.json
```

Repeat with `buckets`, `gather`, `compile`, and `combined`, changing the report
path. `shared` is a separate sampling experiment, not behavior parity. Compare
alignments with `--bucket-alignment 64|128|256`. For production shapes append
`--production-shape --num_envs 4096`; optionally set `--rollout-steps 24`.
The benchmark allows at most five total iterations, never changes solver
tolerances, and reports warmup records separately from steady means. Memory
is PyTorch allocator peak, not total cuPIQP/renderer GPU usage. Production
shapes can require substantially more time/VRAM than the bounded smoke.

### Validation on rescaled_qp_exps (2026-11-09)

GPU 0: RTX 4090, Torch 2.7.0+cu128, Isaac Sim 5.1, cuPIQP 0.1,
`lr_lab_cupiqp`. Same `model_3.pt`, seed 42, 16 environments, 8 rollout steps,
one warmup plus two measured iterations; no production-size training run.
Exact benchmark checkpoint:
`/tmp/hard_pact_runtime_short_mnsb3t_x/Nov09_14-46-30_hard_pact_full_isaaclab/model_3.pt`.
All runs: 256 rollout and 128 PPO problems per iteration, 100% primary
acceptance, zero recovery/fallback. These two-sample timings are noisy smoke
measurements, not a production-speedup claim.

| Variant | Collection s | Update s | Torch peak MiB |
|---|---:|---:|---:|
| reference | 1.698 | 2.568 | 294.77 |
| buckets only | 1.552 | 1.814 | 294.81 |
| direct gather only | 1.545 | 2.365 | 294.69 |
| compile only | 1.676 | 2.495 | 294.63 |
| buckets + gather | 1.690 | 2.118 | 294.73 |

Compilation warmup iteration: 9.392 s versus 6.108 s reference; not included
in steady means. Reports/logs: `/tmp/tight_{reference,buckets,gather,compile,combined}.{json,log}`.
Bounded canonical-QP probe: 1365 rows, chunk 4096, two warmup/five measured
calls including assembly/certification; all 1365 primary accepted. Power-of-two
capacity 2048: 14.495 ms; tight 64/128 capacity 1408: 14.351/14.250 ms;
tight 256 capacity 1536: 14.206 ms. Smaller capacity was not uniformly faster;
128 remains a memory-oriented default, not an autotuned optimum.

Focused suite: **41 passed**, one existing hppfcl deprecation warning, 13.28 s:

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests python -m pytest -q --tb=short \
 tests/test_hard_pact_tight_runtime.py \
 tests/test_hard_pact_runtime_optimizations.py \
 tests/test_hard_pact_rollout_qp_cache.py \
 tests/test_hard_pact_qp_modes.py tests/test_hard_pact_cupiqp_pool.py
```

Kernel float64 parity tolerance: `rtol=atol=1e-10` on CPU/CUDA. Real cuPIQP
padded/fresh output and VJP parity retains `2e-6` float64 / `2e-4` float32.
Coverage includes retained solver leases, selective backward copies, owned
anchor storage, global shuffled indices, shared dispatch counts, and eager/
compiled no-grad kernels. Compiled retained backward is explicitly unavailable
on this Torch version and uses the eager reference, not a changed PCGrad path.
