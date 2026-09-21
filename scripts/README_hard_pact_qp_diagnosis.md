# Extreme cuPIQP candidate diagnosis

Run in the existing cuPIQP-compatible Isaac Lab environment, from repository root.
Only load trusted local checkpoints/packets.

```bash
python scripts/diagnose_hard_pact_qp.py capture \
  --checkpoint /path/to/original/run/model_1234.pt \
  --resolved-config /path/to/original/run/hard_pact_resolved_config.json \
  --backend isaaclab --device cuda:0 --warmup-iterations 2 --iteration-limit 3 \
  --capture-limit 32 --byte-limit-mib 2048 --torque-violation-trigger 1000 \
  --output-dir /tmp/hard_pact_qp_capture

python scripts/diagnose_hard_pact_qp.py replay \
  /tmp/hard_pact_qp_capture/captures/*.pt \
  --device cuda:0 --individual-row-limit 4 --kkt-row-limit 2 \
  --output-dir /tmp/hard_pact_qp_replay

python scripts/diagnose_hard_pact_qp.py timing \
  --checkpoint /path/to/original/run/model_1234.pt \
  --resolved-config /path/to/original/run/hard_pact_resolved_config.json \
  --backend isaaclab --device cuda:0 --warmup-iterations 2 --iteration-limit 3 \
  --output-dir /tmp/hard_pact_qp_timing
```

Capture uses the resolved environment count, rollout/PPO settings and normal
runner loading/update path. Two normal PPO iterations first run with QP disabled;
QP activates at `checkpoint iteration + warmup iterations`, and iteration-limit
counts only subsequent iterations. Overrides include this gate and suppressing
automatic pretraining/resume selection in favor of the
explicit checkpoint, and replacing legacy exception-only capture with the bounded
recorder in a separate output directory. Disabled-QP
ablations and non-cuPIQP configs are rejected, not silently converted. No normal
training config enables these hooks. Capture deliberately synchronizes/copies;
do not use this mode to measure production performance.

Artifacts: complete primary/recovery solver-batch `.pt` packets (including up to
two healthy batches per phase/stage), `identity.json`, resolved config, separate runner logs and
`diagnostic_checkpoint.pt`, plus capture byte/count summary. Inputs are copied
before backend setup/update; raw outputs are retained before sanitization/clamp.
Packets contain canonical and native-packed constraints/scales, original physical
bounds, compact sampled inputs, production acceptance and primal rejection
reasons, available owned solver variables/info, and up to two preceding updates.
The total serialized packet budget includes history; oversize batches are dropped
explicitly, never silently truncated. History tensor memory is capped at one
quarter of that budget. Increase the budget if `dropped_for_budget` is nonzero.
Each rollout/PPO × primary/recovery stratum reserves a quarter of the packet/byte
budget. Distinct reasons and healthy controls receive priority over repeats.
Counts cover all dispatch attempts even after packet budget exhaustion; unavailable
statuses are counted explicitly. Stored history packets are referenced by filename
rather than serialized repeatedly; keep packets together when copying captures.

Replay outputs `replay.json` and `replay.csv`. Factors: other-phase epsilon
tolerances, gap checking, doubled iteration cap, gradient support, private fresh
PPO instances, preceding-update reuse, float64. Baseline starts a fresh backend;
reuse is a bounded numerical sequence, **not exact reconstruction** of original
pool identities, outstanding graphs, initialization, or preconditioners. Native
rollout padding repeats the last real problem as in production. Individual rows
start fresh and do not inherit full-batch history. cuPIQP's installed public
`setup/update/solve/backward` APIs are used through the shared backend; no guessed
warm-start API is invoked.

CSV now includes `joint/{all|accepted}/...` columns: predicted acceleration,
next position/velocity, acceleration-envelope bounds/conflicts, recovery slack,
and original-limit exceedances. Aggregate and named per-joint columns include
finite/nonfinite counts, mean, max, p50/p95/p99. Row counts, empty intersections,
and original-hard-joint satisfaction are separate from recovery acceptance.
Units appear in column names. Empty cells mean unavailable (including v1 limits
or an empty accepted population), not zero. Detailed row arrays remain in JSON.

Independent checks use original canonical normalized constraints and float64
arithmetic with a fixed `1e-3` primal tolerance across variants. Raw torque
violations are Nm. Reference differences are in solver coordinates and only
compare rows passing that same check on both sides. A primal-certified captured
candidate is **not optimal ground truth**. Schema v2 separately recomputes raw and
post-projection primal feasibility and production gap acceptance using shared
production logic. V1 packets remain readable; original joint limits unavailable
in v1 are explicitly marked. Deterministic VJPs test production-accepted-only
and alternating-row masks; failed rows get zero upstream gradients, and nonfinite
backward results are reported rather than hidden. JSON null means unavailable.

Per-joint reports include original position/velocity/acceleration intervals, ties
(acceleration, then velocity, then position), limiting family, empty intersections,
predicted acceleration/position/velocity, and recovery slack. Raw/post residual
groups retain normalized and physical units. Original hard-joint violations are
separate from softened recovery feasibility. Distributions use real finite
coordinates, with count/p50/p95/p99/max for all and accepted candidates; padded
rows never contribute. Joint violation units are rad, rad/s and rad/s².
Individual replay rows prioritize failures, conflicts, large slack and healthy
controls with original row identities. Optional bounded KKT audits estimate active
set multipliers/conditioning; they do not prove feasibility or optimality.

`runtime.json` labels warmup, capture-overhead-inclusive, and capture-disabled
steady-state periods. Timing mode does not install capture or gradient hooks.
GPU events measure primary/recovery forward/backward and runner collection/update;
wall time includes per-iteration checkpoint I/O. Peak memory is PyTorch allocated
and reserved memory, **not total GPU/CuPy/Isaac memory**. Capture-mode autograd hooks
observe QP loss/input and parameter VJPs before clipping/PCGrad without extra
backward calls. Parameter VJPs cannot be uniquely attributed to the QP objective;
totals are exact, percentile samples are explicitly bounded to the first 256
magnitudes per tensor. Backward exceptions are retained and re-raised in training.
Compare separate capture/timing runs; no exact counterfactual overhead is claimed.

Resume limitations: ordinary checkpoints do not restore simulator state, training
RNG, solver caches, or pre-checkpoint outstanding autograd graphs. Iteration and
available optimizer states are restored; missing states are reported. A new
capture therefore reproduces the training configuration, not an exact old
trajectory. Recovery's acceptance uses its softened problem; original hard-joint
guarantees are not implied. Normal production behavior is unchanged.

## Validation (2026-09-21)

Schema-v2 focused validation:

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests conda run --no-capture-output -n lr_lab_cupiqp \
  python -m pytest -q -rs tests/test_hard_pact_diagnose_v2.py \
  tests/test_hard_pact_qp_diagnose.py tests/test_hard_pact_qp_exception_capture.py \
  tests/test_hard_pact_reduced_qp.py
```

**47 passed, 5 skipped (CUDA unavailable), 1 hppfcl warning; 4.08 seconds.**
Resumed warmup, gap/projection parity, original joint bounds/recovery slack,
padding exclusion, coverage/budget counts, matched-row KKT, and runtime-observer
gradient parity are covered. CPU synthetic CLI replay with `--kkt-row-limit 2`
completed 24 cases with zero errors in `/tmp/hard_pact_diagnose_v2_replay_final`.
An initial CLI check exposed a duplicate report key; it was corrected before
the final tests/replay. No simulator training or GPU timing run was started.

Earlier schema-v1 validation follows:

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests conda run --no-capture-output -n lr_lab_cupiqp \
  python -m pytest -q -rs tests/test_hard_pact_qp_diagnose.py \
  tests/test_hard_pact_qp_exception_capture.py tests/test_hard_pact_reduced_qp.py \
  tests/test_hard_pact_cupiqp_pool.py
```

44 passed, 14 CUDA tests skipped, 1 pre-existing hppfcl deprecation warning
(3.76 s). Includes finite-extreme/NaN/status/exception triggers, owned round-trip
snapshots, primary/recovery separation, history/budgets, and enabled/disabled
forward/gradient parity. Two existing exception tests now patch the actual
backend dispatch, rather than a qpth entrypoint no longer used by that path.

CPU CLI plumbing replay uses a synthetic qpth packet, not a cuPIQP substitute:

```bash
SIMULATOR=isaaclab PYTHONPATH=. conda run --no-capture-output -n lr_lab_cupiqp \
  python scripts/diagnose_hard_pact_qp.py replay \
  /tmp/pytest-of-oyoungquist/pytest-52/test_disabled_and_enabled_capt0/qp_0000_ppo_primary.pt \
  --device cpu --individual-row-limit 2 --output-dir /tmp/hard_pact_diagnose_cli_cpu_final
```

CLI result: 24 replay cases, 0 errors; CSV/JSON in the output directory above.
An earlier CLI retry referenced an expired pytest temporary directory and was
rerun against the current packet shown above.

Installed cuPIQP `DenseSolver.solve/update`, result status codes and owned
primal/dual/info arrays were inspected. GPU validation and real checkpoint
rollout/PPO capture remain blocked: PyTorch reports CUDA unavailable;
`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` exits 9 because
it cannot communicate with the NVIDIA driver. No simulator/training run was
started and no root cause for the reported extreme cuPIQP values is claimed.
