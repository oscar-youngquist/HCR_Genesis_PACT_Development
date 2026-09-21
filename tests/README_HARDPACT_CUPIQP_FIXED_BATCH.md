# cuPIQP fixed-contact batching and 100-environment stall diagnosis

Branch `aligned_iclr_2027_qp_pinn`, reviewed HEAD `ce95d0c`.

## Changes

- One fixed-shape solve per valid chunk, not sixteen contact-pattern solves.
  Variables are `[tau12, tilde_f12]`; physical force is `D_m tilde_f` inside
  dynamics, joint inequalities, soft stance acceleration and returned outputs.
- All configured joint acceleration/position/velocity intersections and empty-
  intersection detection remain. Canonical shape: 68 inequalities, no equalities.
  cuPIQP: native torque bounds plus 44 general rows (24 joint, 20 friction).
  Swing friction rows are strictly inactive `0 <= 1`; swing physical force is zero.
- Existing bounded rollout capacities and exclusive PPO leases are reused;
  state-dependent Hessians/masks/mechanics are updated, never marked constant.
  Installed cuPIQP 0.1.0 `DenseSolver.setup/update` supports optional P/c/A/b/G/
  h_u/h_l/x_u/x_l. No new reuse API or CUDA-graph assumptions were introduced.
- `report` now disables gap stopping while preserving reported gaps and primal
  acceptance. `require` retains stopping/acceptance gap checks. Real GPU tests
  confirm finite reported gaps with stopping disabled; the installed solver
  updates gap residuals independently of `check_duality_gap`.
- Launcher and pre-Isaac Warp preparation resolve effective config defaults and
  CLI solver overrides before simulator imports. Full default cuPIQP selects
  `lr_lab_cupiqp`; disabled-QP tasks retain their ordinary environment selection.
- Deployment schema 14 declares masked-force semantics and rejects older schemas.
  Joint limits, rewards, models, losses, PPO likelihood, PCGrad, and detached
  actual rollout-PINN torque are unchanged. Added backend dispatch/row counters.

Changed runtime files: `hard_pact_qp.py`, `hard_pact_qp_backends.py`,
`qpth_warm_start.py` (local zero-direction fix discovered before the cuPIQP-only
request), `deployment.py`, `go2_hard_pact.sh`, `train_hard_pact.py`, and new
`hard_pact_solver_selection.py`. No installed environment/package modifications.
Tests: `test_hard_pact_fixed_contacts.py`, test-only reviewed builder
`hard_pact_grouped_reference.py`, `smoke_hard_pact_qp_stall.py`, and updated shape/
schema assertions in reduced-QP, modes and physics-head tests.

## Diagnosis and limits of the evidence

The reported >5-minute stall **did not reproduce** in bounded 100-environment
tests. The pre-change control-only reproduction completed three intervals in
19.614 s (full) and 6.462 s (sampled). This is not an identical training benchmark
and is not used to claim a speedup.

The former code demonstrably multiplied backend dispatches by contact pattern,
enabled gap stopping for `report`, and only prepared compatible Warp/selected
the cuPIQP environment when a solver was explicit on the command line. Those
paths are fixed, but the exact cause of the user's five-minute stall is unproven.
No timeout stack dump fired. The diagnostic script flushes phase markers and
dumps all Python stacks every 60 seconds if progress stalls; the command has a
240-second external limit.

## Bounded real Isaac Lab collection + PPO backward

RTX 4090, CUDA device 0, `lr_lab_cupiqp`; 100 environments, trimesh 2x2, four
control steps per rollout, one PPO epoch/minibatch, 100% QP replay shard, two
iterations per mode. QP warmup is forced to zero; PINNs start immediately.
No checkpoint loading. Modes execute sequentially in one process (not matched
trajectory performance comparisons). Timing wrappers synchronize explicitly
only in this diagnostic script, so these are instrumented times.

```bash
timeout 240s env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
MPLCONFIGDIR=/tmp/hardpact_mpl conda run --no-capture-output -n lr_lab_cupiqp \
python -u tests/smoke_hard_pact_qp_stall.py \
--task go2_hard_pact_full_isaaclab --headless --num_envs 100 --gpu cuda:0 --qp_solver cupiqp

# Also run exactly the same command without --qp_solver cupiqp.
```

Both explicit/default selection runs **PASS**, with real collection and PPO
backward. Final default-selection run: startup 21.887 s; two full iterations
3.574 s; two sampled iterations 3.052 s. Full: 3,200 rollout / 800 replay QPs;
sampled: 800 rollout / 800 replay QPs. Finite, nonzero accumulated gradients in
actor position/torque, encoder, GRF and wrench parameters; torque magnitude/rate
checks pass. Full rollout certified 2,825/3,200; PPO 702/800. Sampled rollout and
PPO certified 111/800 each. Remaining rows are **uncertified bounded fallback**;
joint constraints were not relaxed and this is not evidence of training stability.

Full-mode cold / subsequent mean timings (ms): mechanics 236.75 / 9.80;
assembly 22.64 / 1.55; cuPIQP solve 59.79 / 48.95; update 3.83 / 3.32;
PPO update+backward 323.51 / 231.63. Three setups and 31 updates across 34 backend
dispatches. An earlier explicit run's first solve took 1,398.31 ms, consistent
with cold compilation/cache cost, versus 47.52 ms subsequent mean; no 5-minute
pause was observed. Final sampled mode: two setups, 32 updates; solves
55.86 / 53.87 ms; PPO update+backward 250.84 / 140.24 ms.
PyTorch allocator peaks: 478,494,208 bytes full; 424,640,000 bytes sampled.
These exclude simulator/CuPy allocations. No speedup claim is made.

## Correctness tests

```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/hardpact_mpl \
conda run --no-capture-output -n lr_lab_cupiqp python -m pytest -q \
tests/test_hard_pact_fixed_contacts.py tests/test_hard_pact_reduced_qp.py \
tests/test_hard_pact_cupiqp_pool.py tests/test_hard_pact_qp_reuse.py --tb=short
```

**55 passed, 1 warning in 9.36s**, completed before the cuPIQP-only clarification.
Includes mixed masks/one dispatch, native 44-row packing, swing zeros, SPD,
unchanged hard joint bounds, infeasible intersections, reuse, forward/VJP parity,
failed-row isolation and gaps. Reviewed grouped-reference tolerances remain
forward rtol=1e-5/atol=1e-6, VJP rtol=3e-3/atol=2e-5.
Earlier CPU focused replay/PINN suite: 45 passed, 2 CUDA skips, 1 warning in 4.02s.

Final cuPIQP-only rerun:
```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/hardpact_mpl \
conda run --no-capture-output -n lr_lab_cupiqp python -m pytest -q \
tests/test_hard_pact_fixed_contacts.py -k cupiqp --tb=short
```
**2 passed, 8 deselected, 1 warning in 2.18s**: real cuPIQP rollout and backward,
mixed masks, finite reported gaps, exact swing zero, finite nonzero VJPs and reuse.
Warning: existing hppfcl/coal deprecation. No further qpth-specific work after
the cuPIQP-only clarification; no Isaac Gym/Genesis/Moreau support claim here.
