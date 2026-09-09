# HardPACT cuPIQP exception capture and current-config reproduction

Purpose: distinguish a backend exception from ordinary rejected primal
residuals, and reproduce the exact failing chunk without a simulator.

The training `hard_pact_qp` config enables `exception_capture_enabled`, limits
captures to `exception_capture_limit=1` per solver instance, and writes into
`exception_capture_dir=/tmp/hard_pact_qp_failures`. Set enabled to false to
disable all capture work. Successful solves do not copy tensors to the host.
The first caught backend exception prints its traceback and snapshot path;
fallback behavior is unchanged. Complete chunks can produce large files.

In the same activated training environment, from the repository root:

```bash
SIMULATOR=isaaclab PYTHONPATH=. python legged_gym/scripts/replay_hard_pact_qp_failure.py /tmp/hard_pact_qp_failures/qp_failure_EXAMPLE.pt --device cuda:0
SIMULATOR=isaaclab HARDPACT_QP_TEST_BATCH=4096 python -m pytest -q tests/test_hard_pact_qp_exception_capture.py
```

Snapshots include exact scaled backend Q/p/G/h/A/b, native bounds, full config,
stage, differentiability, exception/traceback, device and package versions.
Replay uses a fresh backend, not the previous CUDA cache/allocator history.
Use only trusted local snapshots. No environment dependencies are changed.

For a runtime/import-order failure, pass `--warp-path` pointing at the
directory containing the captured `warp` package. New snapshots record
`runtime_modules` with the actual imported file/version, since package
metadata alone cannot detect Isaac Sim shadowing the environment runtime.

## Confirmed full-launch failure (2026-09-08)

The user's real 4096-environment launch loaded Isaac Sim's bundled Warp 1.8.2.
cuPIQP 0.1.0 in `lr_lab_cupiqp` requires Warp >=1.12 and its tile multiplication
failed to compile in `preconditioner_kernels.py:376`. The exact captured
4096-row matrices reproduced that error with bundled Warp and returned finite
solutions with the environment's already-installed Warp 1.17.0.

The HardPACT shell launcher now enters `train_hard_pact.py`, which imports
Warp before Isaac startup only when Isaac Lab/cuPIQP is explicitly selected.
It restores missing `warp.types.array/indexedarray` aliases used by Isaac Sim
5.1 to the identical top-level Warp classes. No package files are changed;
the legacy training entrypoint and non-cuPIQP runtime selection are unchanged.

The tests use the current HardPACT training config with the cuPIQP CLI
override, float32 CUDA, configured physics timestep, repeated cached solves,
and both rollout/PPO modes. Dynamics and actuator bounds remain synthetic;
these tests are not a real Isaac Lab/BARD training smoke.

Earlier real training evidence remains valid: the recorded
`hard_pact_training_isaaclab_cupiqp_crba_100` run completed five iterations with
100 environments and both BARD losses (49.076 s mean, 0.331 s standard deviation).
The `hard_pact_training_isaaclab_cupiqp_bard_final` 4096-environment run instead
OOMed with the older ABA graph. Those recorded per-substep runs do not establish
correctness of every later two-anchor/elastic/full-diagnostics combination.
