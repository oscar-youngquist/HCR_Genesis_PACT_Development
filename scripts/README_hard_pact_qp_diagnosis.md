# Extreme cuPIQP candidate diagnosis

Run in the existing cuPIQP-compatible Isaac Lab environment, from repository root.
Only load trusted local checkpoints/packets.

```bash
python scripts/diagnose_hard_pact_qp.py capture \
  --checkpoint /path/to/original/run/model_1234.pt \
  --resolved-config /path/to/original/run/hard_pact_resolved_config.json \
  --backend isaaclab --device cuda:0 --iteration-limit 3 \
  --capture-limit 8 --byte-limit-mib 512 --torque-violation-trigger 1000 \
  --output-dir /tmp/hard_pact_qp_capture

python scripts/diagnose_hard_pact_qp.py replay \
  /tmp/hard_pact_qp_capture/captures/*.pt \
  --device cuda:0 --individual-row-limit 4 \
  --output-dir /tmp/hard_pact_qp_replay
```

Capture uses the resolved environment count, rollout/PPO settings and normal
runner loading/update path. The only diagnostic execution overrides are immediate
QP activation, suppressing automatic pretraining/resume selection in favor of the
explicit checkpoint, and replacing legacy exception-only capture with the bounded
recorder in a separate output directory. Disabled-QP
ablations and non-cuPIQP configs are rejected, not silently converted. No normal
training config enables these hooks. Capture deliberately synchronizes/copies;
do not use this mode to measure production performance.

Artifacts: complete primary/recovery solver-batch `.pt` packets (including up to
two healthy batches), `identity.json`, resolved config, separate runner logs and
`diagnostic_checkpoint.pt`, plus capture byte/count summary. Inputs are copied
before backend setup/update; raw outputs are retained before sanitization/clamp.
Packets contain canonical and native-packed constraints/scales, original physical
bounds, compact sampled inputs, production acceptance and primal rejection
reasons, available owned solver variables/info, and up to two preceding updates.
The total serialized packet budget includes history; oversize batches are dropped
explicitly, never silently truncated. History tensor memory is capped at one
quarter of that budget. Increase the budget if `dropped_for_budget` is nonzero.

Replay outputs `replay.json` and `replay.csv`. Factors: other-phase epsilon
tolerances, gap checking, doubled iteration cap, gradient support, private fresh
PPO instances, preceding-update reuse, float64. Baseline starts a fresh backend;
reuse is a bounded numerical sequence, **not exact reconstruction** of original
pool identities, outstanding graphs, initialization, or preconditioners. Native
rollout padding repeats the last real problem as in production. Individual rows
start fresh and do not inherit full-batch history. cuPIQP's installed public
`setup/update/solve/backward` APIs are used through the shared backend; no guessed
warm-start API is invoked.

Independent checks use original canonical normalized constraints and float64
arithmetic with a fixed `1e-3` primal tolerance across variants. Raw torque
violations are Nm. Reference differences are in solver coordinates and only
compare rows passing that same check on both sides. A primal-certified captured
candidate is **not optimal ground truth**. Deterministic VJPs test certified-only
and alternating-row masks; failed rows get zero upstream gradients, and nonfinite
backward results are reported rather than hidden. JSON null means unavailable.

Resume limitations: ordinary checkpoints do not restore simulator state, training
RNG, solver caches, or pre-checkpoint outstanding autograd graphs. Iteration and
available optimizer states are restored; missing states are reported. A new
capture therefore reproduces the training configuration, not an exact old
trajectory. Recovery's acceptance uses its softened problem; original hard-joint
guarantees are not implied. Normal production behavior is unchanged.

## Validation (2026-09-21)

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
