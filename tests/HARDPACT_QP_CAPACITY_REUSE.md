# HardPACT cuPIQP capacity reuse, assembly parity, and GPU smoke

Historical results below describe the earlier high-water cache. The current
bounded buckets, PPO pool, and disjoint logging are documented in
[pool and phase-diagnostics tests](HARDPACT_CUPIQP_POOL_AND_PHASE_DIAGNOSTICS.md).

Validated September 8, 2026, on `aligned_iclr_2027_qp_pinn` using
`lr_lab_cupiqp`: Python 3.11, PyTorch 2.7.0+cu128, cuPIQP 0.1.0,
Warp 1.17.0, Isaac Sim 5.1.0, RTX 4090 (24 GiB). No environment/package changes.

## Changes

- `rsl_rl/algorithms/hard_pact_qp_backends.py`: rollout-only power-of-two
  capacities, one retained high-water capacity per device/dtype/structure/profile.
  Padding duplicates independent valid QPs; padded outputs are discarded.
  Capacity never exceeds the configured rollout chunk budget unless a direct
  caller itself supplies a larger batch. Returned results own their storage.
- `rsl_rl/algorithms/hard_pact_qp.py`: cache limits, selectors, physical A/G
  templates, proximal diagonal, and scaled shared Q. Only variable entries
  change. Reuse compact-row indices and fixed native-bound slices instead of
  repeated boolean gathers/scalar decisions. Preserve physical certification.
- `legged_gym/envs/go2/go2_hard_pact/go2_hard_pact_config.py`:
  `algorithm.hard_pact_qp['cupiqp_rollout_capacity_reuse'] = True`;
  set False to retain exact-size rollout allocations.
- `tests/test_hard_pact_qp_reuse.py`: 20 focused regression tests.

Full/relaxed constant Hessians skip cuPIQP P updates. Elastic recovery must
refresh both its state-dependent `Q += 2*w*A.T@A` and Ruiz preconditioning;
reusing stale elastic scaling produced fresh-versus-reused numerical differences.
Elastic still reuses allocations. Differentiable PPO solvers are never pooled:
each outstanding backward graph retains its own solver. No loss, checkpoint,
observation, reward, torque/rate constraint, or solver tolerance changes.

Dynamic `nonzero` compaction still synchronizes to obtain batch sizes, and
cuPIQP itself retains host convergence checks. This removes redundant wrapper
synchronizations, not all GPU synchronization. Diagnostics settings are untouched.

## Tests and exact commands

```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=0 HARDPACT_QP_TEST_BATCH=4096 \
  conda run -n lr_lab_cupiqp python -m pytest -q \
  tests/test_hard_pact_solver_runtime.py \
  tests/test_hard_pact_qp_exception_capture.py \
  tests/test_go2_hard_pact_qp.py tests/test_hard_pact_qp_optional_solvers.py \
  tests/test_hard_pact_qp_reuse.py tests/test_hard_pact_two_anchor_smoke.py \
  tests/test_hard_pact_action_replay.py tests/test_hard_pact_ppo_qp_sampling.py \
  tests/test_go2_hard_pact_ablations.py --tb=short
```

Result: **108 passed, 1 failed, 12 subtests passed, 3 warnings in 26.25 s**.
Includes real 4096-row CUDA rollout/implicit-backward tests. The failure is
`StochasticActionReplayTests.test_frozen_policy_exact_raw_delayed_action_and_torque`:
maximum delayed-action error `0.17380964756`. It reproduces identically with
pre-edit QP/backend copies (6 passed, 1 failed in 2.79 s); left outside this
QP-only change. The warnings are two deliberately forced solver failures and
the installed hppfcl deprecation. Full log: `/tmp/hard_pact_qp_reuse_tests.log`.

The focused new file alone: **20 passed, 1 warning in 22.04 s**. GPU capacity
parity uses the production proximal rho=0.1, unchanged assertion tolerances
`rtol=atol=2e-4` (float32) and `2e-6` (float64); test solver tolerances are
stricter than rollout defaults. PPO outputs and four learned-input gradients
match exactly with reuse on/off. Rollout sizes 7/5/3/8/6 require **1 setup +
4 updates**, versus 5 exact-size setups. Tests also cover growing capacities,
memory caps, result ownership, changed elastic mechanics, invalid-row ordering,
hard bounds, inference-to-PPO cache reuse, and no wrapper `item`/tensor-bool reads.

## Small real training smoke

```bash
./legged_gym/scripts/go2_hard_pact.sh \
  --task go2_hard_pact_full_isaaclab --headless --dynamics_backend bard \
  --qp_solver cupiqp --gpu cuda:1 --smoke --num_envs 128 --max_iterations 3
```

Unchanged training settings except environment/iteration counts. Seed 1,
two anchors, 24 rollout steps, five PPO epochs, four minibatches, 20% QP shard,
current full diagnostics. Completed all three iterations, 9216 transitions,
exit 0. All solver-exception fractions were zero; QP gradient norms were
`0.0061453, 0.0114826, 0.0115497`. PINN schedule activated at iteration 2
(logged loss 1.42522). All **457** checkpoint tensors were finite. Rollout
analytic fallback fractions were approximately 3.16%, 6.01%, 4.10%; fallback
remains enabled. Pre-clamp torque violation peaked at 0.0005274 Nm and the
mandatory exact final torque clamp remains in place.

Run: `logs/hardpact_iclr/go2_pact_rough/Sep08_12-35-07_hard_pact_full_isaaclab`.
Log: `/tmp/hard_pact_qp_capacity_smoke_retry.log`. First launch exited 139 in
Isaac Sim native telemetry/getenv startup before task/QP construction;
an unchanged retry succeeded. No simulator installation changes were made.

## Isolated before/after audit and microbenchmark

`env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n lr_lab_cupiqp python /tmp/compare_hard_pact_qp_reuse.py`

This local audit script imports pre-edit copies from
`/tmp/hard_pact_qp_before_capacity.py` and
`/tmp/hard_pact_qp_backends_before_capacity.py`. All physical/scaled matrices
and assembly VJPs matched **exactly (maximum error 0)** for full/relaxed/elastic
in float32 and float64 on identical seeded coupled inputs.

| Isolated operation | Before | After |
| --- | ---: | ---: |
| 4096-row assembly, 20 repetitions | 1.839 ms | 1.114 ms |
| Additional peak Torch allocation during assembly | 673.6 MiB | 311.0 MiB |
| Relaxed solve, changing batches 57/33/48/64/41, repeated twice | 78.037 ms | 13.440 ms |
| New solver setups in those 10 warmed calls | 10 | 0 |
| Accepted rows in those calls | 486/486 | 486/486 |

Synchronization brackets timing regions, not individual assembly calls. Memory
is Torch allocator memory only, **not** total CUDA/cuPIQP memory. These are
short synthetic microbenchmarks, not a claim of end-to-end training speedup.
Unregularized, ill-conditioned elastic fixtures remain sensitive to finite
iteration/preconditioning choices; no assertion tolerances were relaxed.
