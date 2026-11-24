# Mixed and First Plus Random Substep QP

## Execution correction cap (current experiment)

In `algorithm.hard_pact_qp`:

```python
'attitude_weight': 0.1,
'contact_acceleration_weight': 0.1,
'attitude_weight_initial': 0.1,
'contact_acceleration_weight_initial': 0.1,
'execution_correction_clip_enabled': True,
'execution_correction_hip_limit_nm': 12.0,
'execution_correction_thigh_limit_nm': 12.0,
'execution_correction_calf_limit_nm': 18.0,
```

The component weights are authoritative final targets; no deprecated final
override is needed. Outer actor/PINN coefficients and ramp durations are unchanged.
The cap applies to accepted primary/recovery commands AFTER execution/extra alpha:
`tau_exec = actuator_clamp(base + clamp(alpha*(tau_qp-base), -L, L))`.
Limits are resolved by canonical FR/FL/RR/RL joint names and cached per device/dtype.
All quantities are physical Nm, after actuator scaling and before feedforward
reconstruction. Failed-solve fallback and unsolved substeps are unchanged.

Rollout/play share the execution helper. PPO stores anchor alpha and the baseline
rate-clipping flag, reconstructs the current-policy baseline and uses the same cap
with ordinary clamp derivatives. Full QP candidate/projection/slack losses remain
uncapped. No extra solves or constraints. Execution acceleration reuses the cached
dynamics torque map; a changed command has no candidate joint/endpoint certificate.

Aggregate execution logs include clipped accepted-row fraction, requested/applied
correction mean/max in Nm, and finite denominators. Scheduled `physical`/`full`
profiles add `sampled_health/execution_primary` and `execution_recovery` original
joint-limit predictions, plus endpoint-torque exceedance. Their accepted/rejected
labels refer to the SOURCE candidate, not certification of postprocessed commands.
These are model predictions, not measured guarantees.

Missing legacy flags mean disabled; strict checkpoint contracts reject silently
enabling the cap or changing weights on ordinary resume. Captures retain exact
execution baseline/alpha and configs; deployment metadata documents the operation.

Focused checks (no training):

```bash
SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -m pytest -q \
  tests/test_hard_pact_execution_cap.py tests/test_hard_pact_intervention_curriculum.py \
  tests/test_hard_pact_extra_strength.py tests/test_hard_pact_qp_curriculum.py
```

## Performance-gated intervention curriculum (active Go2 configuration)

Select `qp_update_mode='curriculum_substeps'` in `algorithm.hard_pact_qp`.
No extra enable flag is needed. The default schedule is:

```python
'qp_update_mode': 'curriculum_substeps',
'intervention_settling_iterations': 500,
'intervention_ramp_iterations': 8000,
'intervention_window_iterations': 200,
'intervention_gate_check_interval': 5,  # Any positive integer, e.g. 2.
'intervention_tracking_ratio': 0.90,
'intervention_failure_ratio': 1.30,
'intervention_failure_allowance': 0.0,  # events / simulated environment-second
```

Probabilities for 2/3/4 solves interpolate linearly between progress knots
`0:(1,0,0)`, `.25:(.75,.20,.05)`, `.5:(.5,.3,.2)`,
`.75:(.25,.25,.5)`, `1:(0,0,1)`. Each interval uses largest-remainder integer
group sizes (ties prefer 2 then 3 then 4), randomly assigns environments, and
draws distinct uniform later substeps, always including zero. Tiny batches
therefore approximate the mixture to within one row per group. PPO uniformly
preselects one scheduled solve before outcomes; its budget is unchanged.

### Optional extra-intervention strength

In `algorithm.hard_pact_qp`:

```python
'intervention_extra_strength_enabled': False,  # Opt in with True.
'intervention_extra_strength_start': 0.25,     # Finite [0,1].
```

Only `curriculum_substeps` uses this option. Baseline solves are substep 0
and the first entry in the sampler's existing random later ordering. Remaining
selected solves use `alpha * (start + (1-start)*snapshot_progress)`; baseline
solves retain `alpha`. No extra RNG draw, solve or schedule clock is introduced.
Progress stays frozen throughout rollout/update. Prerequisite ramps and the
500-iteration settling period remain unchanged. Full progress means strength
one, preserving the subsequent disturbance-settling trigger.

Accepted primary/recovery torques blend from the existing sanitized, optionally
rate-projected nominal base. Fallbacks are unchanged; unsolved substeps use
fresh nominal commands, not held corrections. Partial blends are not certified
by the full candidate's certificate. Existing FF reconstruction remains intact.

Enabled PPO replay stores one float alpha and one base-rate-clipping flag per
anchor. Actor velocity losses use the current-policy blend, with acceleration
`qdd_candidate + A_tau*(tau_blend-tau_candidate)`. `A_tau=M^-1 S^T` is reused
from the existing primary/recovery assembly, so this requires no extra solve.
Optimized forces/external wrench remain those of the full candidate; this is
a model prediction, not measured execution. Actor-only routing is unchanged.
Projection/recovery losses and their diagnostics still use the full candidate;
representation PINNs still use measured executed torque. Disabled behavior is
unchanged, including full-candidate actor velocity losses.

Gated physical diagnostics include `execution/extra_strength` and
`execution/{baseline,extra}_applied_correction_rms_nm` (coordinate RMS across
accepted rows, count-weighted; empty groups unavailable). Configuration is
saved in resolved/checkpoint/deployment solver settings. Older checkpoints
default to disabled, including those already carrying intervention progress;
an enabled destination is not silently accepted as an ordinary legacy resume.
No new curriculum state is needed: saved snapshot progress determines strength.

Focused checks: `python -m pytest -q tests/test_hard_pact_extra_strength.py
tests/test_hard_pact_intervention_curriculum.py tests/test_hard_pact_qp_modes.py
tests/test_hard_pact_disturbance_sequence.py` (use the existing cuPIQP environment).
Unselected substeps use fresh nominal torque, never held corrections.

Evidence collection starts after both existing ramps end. Correction ends at
`warmup_iterations + correction_ramp_start_offset + correction_ramp_duration`
when enabled with positive duration (otherwise at QP activation). Actor physics
ends at `warmup_iterations + actor_physics_warmup_iterations`. The later end,
not an independent ramp clock, starts settling. At least one complete evidence
window is necessary even if settling is shorter. Progress increments by
`1/intervention_ramp_iterations` after each valid, gate-open completed iteration
past settling; the updated mixture applies to the next rollout/update only.

Raw linear tracking uses existing all-environment/control-step scores before
reward scaling. Each window computes a sample-weighted tracking mean and
`sum(non-timeout, non-relocation resets) / sum(N * control_dt)`; repeated failures
count as events, not unique environments. This is independent of TensorBoard
reporting intervals and episode reward means. Tracking follows the disturbance
curriculum convention: per-valid-iteration EMA (alpha 0.05), a rolling 200-EMA
history, and its 90th-percentile reference (sorted index floor(0.9*(n-1))).
Every five completed iterations (configurable), the gate uses the latest 200
consecutive valid iterations and requires current EMA >= max(0.50, ratio * reference).
The rolling evidence is NOT cleared at checks. Duplicate iteration callbacks
are ignored; missing/nonfinite evidence closes the gate and requires a new full
consecutive window. Failure rate is the ratio of summed counts to summed
simulated exposure, not the mean of individual rates or TensorBoard totals.
These independent `intervention_tracking_ema_alpha`,
`intervention_tracking_reference_window`, `intervention_tracking_reference_quantile`
and `intervention_min_tracking` settings match the disturbance defaults; they do
not change the disturbance curriculum. Old easier-domain tracking ages out.
Failure reference remains the minimum historical **non-overlapping 200-iteration**
window failure rate: frequent rolling checks cannot independently tighten it. It requires
failures <= ratio * reference + absolute allowance.
No yaw or plateau requirement. With a zero failure reference and zero allowance,
only a zero-failure window passes. Missing/nonfinite/insufficient evidence
invalidates the window and pauses progress; the gate otherwise stays fixed
between checks. Gate decisions apply only to subsequent iterations. Progress still
advances once per gate-open iteration, not once per check.

Checkpoint state under `hard_pact_qp_curriculum.intervention` (also in the shared
training envelope) includes progress, absolute timing, partial evidence, best
references, EMA/history and the gate. Version-1 intervention states retain attained
progress/timing and best failure reference, but recollect tracking evidence with
the gate closed for a complete new window. Version-2 states retain progress and
references but recollect missing per-iteration rolling evidence. Version-3 states
preserve the buffer, disjoint reference accumulators and check timing exactly.
Torch RNG restoration preserves subsequent group and
anchor sampling. A checkpoint lacking this state starts at two solves and
collects a fresh complete window; iteration count never implies intervention
progress. Physical Isaac Lab resume still starts fresh episodes, not exact
PhysX continuation. Objective-weight scheduling is independent.

`curriculum/intervention/*` logs progress, probabilities, expected mean solves
(before integer rounding), gate/reason and completed-window means/references,
plus `tracking_ema` and `tracking_threshold` (the actual tracking gate operands).
`window_ready`, `window_iterations`, and `gate_checks` expose evidence/check cadence.
Unavailable evidence is NaN, not a successful zero. Reason codes: 0 existing
ramps, 1 settling, 2 insufficient evidence, 3 tracking, 4 failures, 5 open,
6 complete. Metrics describe the rollout mixture and the decision for the next
iteration. Deployment may explicitly select `every_substep`; without a restored
training schedule, curriculum mode defaults to two solves.

Old complete checkpoint envelopes predating the new settings support a narrow
sampler migration when the new mode is explicitly selected: only intervention
settings and sampler metadata may differ. Numerical QP, optimizer, actor/loss,
and input contracts still compare strictly; a warning identifies the fresh
schedule. Legacy weight-only files still cannot restore unavailable optimizer
or global training state.

Focused CPU validation (2026-11-15, `lr_lab_cupiqp`): 116 passed. Covers the new
schedule plus unchanged sampling/control, QP-objective curriculum and shared
checkpoint regressions. No GPU simulator/training run for this change.

```bash
conda activate lr_lab_cupiqp
SIMULATOR=isaaclab PYTHONPATH=. LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" \
python -m pytest -q -p no:cacheprovider \
  tests/test_hard_pact_intervention_curriculum.py \
  tests/test_hard_pact_first_plus_random.py tests/test_hard_pact_qp_curriculum.py \
  tests/test_hard_pact_qp_modes.py tests/test_training_state_transfer.py
```

## Fixed mixed mode (unchanged)

The existing `qp_update_mode='mixed_substeps'` remains available:
each interval randomly assigns `floor(0.5*N)` environments to first + one
random later solve, `floor(0.3*N)` to first + two distinct random later solves,
and the remainder to all four. Groups and later substeps are redrawn every
interval using the existing Torch RNG. This averages approximately 2.7 solves
per environment (exactly 2.7 for batch sizes divisible by ten).
PPO uniformly preselects one executed solve per environment, independent of
solver success; its row/epoch budget is unchanged. No held corrections or
additional neural forwards are introduced. `qp_shared_later_substep` applies
only to `first_plus_random_substep`, not the new mixed mode.

For the unchanged first-plus-random mode:

Set the existing `algorithm.hard_pact_qp` dictionary entry:

```python
'qp_update_mode': 'first_plus_random_substep',
```

The compatibility dataclass default remains `random_one_substep`; the current
HardPACT experiment config selects `curriculum_substeps`. With four physics substeps per
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
