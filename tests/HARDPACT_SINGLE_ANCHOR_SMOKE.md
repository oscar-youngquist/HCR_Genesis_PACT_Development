# Single-anchor HardPACT execution and PPO replay tests

Enable with `--qp_update_mode single_anchor_held_correction` on the existing
HardPACT launcher, or set `algorithm.hard_pact_qp['qp_update_mode']` in
`go2_hard_pact_config.py`. Existing defaults, warmup and other modes are unchanged.
Anchor 0 replaces the correction each interval; held substeps recompute PD and
project onto the exact actuator/rate box. They are not newly QP-certified.
PPO replays anchor 0 without anchor RNG or a loss multiplier, retaining epoch
row partitioning. Deployment exports the same schedule and execution helper.

## Targeted CPU tests (2026-09-09)

```bash
env SIMULATOR=isaaclab conda run -n lr_lab_cupiqp python -m pytest -q \
  tests/test_hard_pact_single_anchor.py \
  tests/test_hard_pact_two_anchor_smoke.py \
  tests/test_hard_pact_ppo_qp_sampling.py \
  tests/test_go2_hard_pact_qp.py --tb=short
```

**53 passed, 4 subtests passed, 3 warnings, 5.14 s.** Covers all schedules,
partial resets, warmup/mode changes, inference buffers, held PD, per-substep
torque/rate bounds, aggregation, exact frozen anchor replay, disjoint PPO
coverage, gradients, contract, and forced recovery. Real qpth forward/VJP
single-anchor versus default parity uses `rtol=atol=0`. Warnings: hppfcl
deprecation and two deliberately injected solver failures.

Adding `tests/test_go2_hard_pact_physics_heads.py` to that command produced
**79 passed, 4 failed, 11 subtests passed, 3 warnings, 5.54 s**. Failures expect
history 20 instead of configured 10, enabled diagnostics instead of False,
and GRF scales 250 instead of 100 (two tests). All three configured values
were verified unchanged in `git show HEAD:legged_gym/envs/go2/go2_hard_pact/go2_hard_pact_config.py`.
No unrelated config values or test tolerances were changed.

## Real Isaac Lab / cuPIQP GPU smoke

```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  conda run --no-capture-output -n lr_lab_cupiqp \
  python -u tests/smoke_hard_pact_single_anchor.py \
  --task go2_hard_pact_full_isaaclab --headless --qp_solver cupiqp \
  --qp_update_mode single_anchor_held_correction \
  --num_envs 8 --gpu cuda:0 --seed 37
```

Isaac Sim 5.1, PyTorch 2.7.0+cu128, Warp 1.17.0, physical GPU 1 (RTX 4090).
Test-only overrides: warmup 0, four control steps, one PPO minibatch in each
of five epochs, 2x2 terrain grid, initial terrain level at most 1. Retains
registered full-task implementation and pretrained model; this is not a
training benchmark or an assertion that both PINN losses were nonzero.

**PASS:** 4 primary rollout anchor invocations, 12 held substeps, 5 PPO replay
invocations and 5 finite/nonzero solver-output backward checks. Finite
observations/torques and all torque/rate bounds passed (float32 tolerances
1e-6 Nm magnitude, 1e-5 Nm rate). Constructor performs the real reset.
Synthetic tests separately verify head/shared-input gradients and forced
fallback; no claim that every fallback stage occurred in this short rollout.

Pass marker and output: `/tmp/hard_pact_single_anchor_smoke_retry.log`.
Run artifacts: `/tmp/hard_pact_single_anchor_1h8whyq2/Sep09_12-25-06_hard_pact_full_isaaclab`.
First attempt used the wrong working-directory-relative checkpoint path;
the harness now resolves that path relative to the existing script directory.
Isaac Lab stalled in shutdown after the successful assertions; its process
was interrupted with SIGINT and exited. No test process remains running.
Genesis/Isaac Gym were not run. No environments or packages were changed.

## Changed files

- `rsl_rl/algorithms/hard_pact_qp.py`: schedule and shared projection helpers, validation.
- `legged_gym/envs/go2/go2_hard_pact/go2_hard_pact.py`: rollout, replay anchor selection, clearing and solve-only diagnostics.
- `legged_gym/envs/go2/go2_hard_pact/deployment.py`: execution contract.
- `rsl_rl/runners/pact_runner.py`: export that contract.
- `legged_gym/envs/go2/go2_hard_pact/go2_hard_pact_config.py`: document option; selected default unchanged.
- `legged_gym/utils/helpers.py`: CLI choice.
- `tests/test_hard_pact_single_anchor.py`, `tests/test_hard_pact_two_anchor_smoke.py`, `tests/test_hard_pact_ppo_qp_sampling.py`: focused regressions.
- `tests/smoke_hard_pact_single_anchor.py`: opt-in simulator smoke.
- This report.
