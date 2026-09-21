# HardPACT QP mode correctness and Isaac Lab control smoke

Implemented on `aligned_iclr_2027_qp_pinn`, starting at `a08bb51`.

`random_one_substep` is the training default: one balanced, uniformly selected
QP substep per environment. `every_substep` executes four problems per
environment. Unsolved substeps use fresh PD/feedforward torque projected onto
actuator/rate bounds; no correction is held. Neural predictions are prepared
before simulation and held throughout all four substeps. Mechanics and frame
conversions refresh at each solve. PPO replays one selected problem, including
the original control-start torque-conditioning state, with current predictions.

Deployment schema 12 records the two modes and rejects incompatible contracts.
The 24-variable formulation, objective, constraints and analytic fallback remain.

## Focused correctness tests

```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/hardpact_mpl \
conda run --no-capture-output -n lr_lab_cupiqp python -m pytest -q \
tests/test_hard_pact_qp_modes.py tests/test_hard_pact_reduced_qp.py \
tests/test_hard_pact_ppo_qp_sampling.py
```

Result: **31 passed, 2 skipped, 1 warning in 4.56s**. Skips are CUDA tests in
this CPU invocation; warning is the existing hppfcl-to-coal deprecation.
Tests cover algebra/constraints, mode counts, changing PD, bounds, reset,
sampling, replay parity and differentiable replay gradients. No training run.

## Real Isaac Lab GPU reset/control smoke

```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MPLCONFIGDIR=/tmp/hardpact_mpl \
conda run --no-capture-output -n lr_lab_cupiqp python -u \
tests/smoke_hard_pact_qp_modes.py --task go2_hard_pact_full_isaaclab \
--headless --num_envs 8 --gpu cuda:0 --qp_solver cupiqp
```

Result: **PASS**, exit 0, RTX 4090. Eight environments, three control intervals
per mode, reset between modes, random untrained policy, no optimizer updates.
Full mode: 12 problems/environment, 80 certified and 16 analytic fallbacks.
Sampled mode: 3 problems/environment, 7 certified and 17 analytic fallbacks.
Both modes: zero neural forwards during the entire simulator step, finite
outputs and enforced actuator/rate bounds. Analytic fallbacks are not reported
as joint/contact certified. These results are correctness checks, not a
performance or trained-policy feasibility claim.

## Changed implementation files

- `legged_gym/envs/go2/go2_hard_pact/{go2_hard_pact.py,go2_hard_pact_config.py,deployment.py}`
- `legged_gym/utils/helpers.py`
- `rsl_rl/algorithms/{hard_pact_qp.py,hard_pact_qp_backends.py,ppo_hard_pact.py}`
- Removed `rsl_rl/algorithms/hard_pact_active_constraints.py`
- `rsl_rl/runners/pact_runner.py`
- `scripts/eval_hard_pact_frozen.py` (mode/compact-row migration; full evaluator not run)
- `tests/{test_hard_pact_qp_modes.py,test_hard_pact_reduced_qp.py,smoke_hard_pact_qp_modes.py}`

Only the focused suite and the stated control smoke were run. Historical
retired-mode tests, full training and benchmarks were not run.
