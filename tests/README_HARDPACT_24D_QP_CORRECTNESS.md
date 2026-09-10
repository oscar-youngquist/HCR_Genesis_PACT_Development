# HardPACT 24-variable QP correctness

Implemented from `974fd47` on `aligned_iclr_2027_qp_pinn`.

The shared builder solves `x=[total torque12; world FR/FL/RR/RL force12]`.
Acceleration is eliminated with a detached batched mechanics solve. Torque/GRF
tracking, stance acceleration, and yaw-local physical roll/pitch acceleration
are quadratic costs. Swing forces have exact zero equalities and no friction
rows. Position prediction uses the configured beta (Genesis/PhysX: 1).
Failed rows return sanitized actuator/rate projection without a solver VJP or
joint/contact certificate. No relaxed/elastic recovery QPs remain.

## Exact focused commands/results

CPU, `lr_lab_cupiqp`:

```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/hardpact_mpl \
  conda run --no-capture-output -n lr_lab_cupiqp python -m pytest -q \
  tests/test_hard_pact_reduced_qp.py
```

**26 passed, 3 CUDA tests skipped; 1.49 s.**

Real CUDA, GPU 0, cuPIQP 0.1.0 / PyTorch 2.7.0+cu128:

```bash
env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/hardpact_mpl \
  conda run --no-capture-output -n lr_lab_cupiqp python -m pytest -q \
  tests/test_hard_pact_reduced_qp.py -k cuda
```

**3 passed, 26 deselected; 10.68 s.** These test actual native cuPIQP bounds,
qpth canonical parity, implicit gradients, and migrated active-set algebra/cache
ownership—not a simulator or training loop.

Checks include all 16 stance patterns, affine dynamics/objective algebra, SPD,
nonredundant swing equalities, torque/rate/joint intersections, mixed invalid
rows, solver exceptions, detached mechanics/contact masks, diagnostics parity,
and cold/warm qpth behavior. CPU objective-gradient tolerance: `rtol=1e-12,
atol=1e-13`; finite differences: `rtol=3e-3, atol=2e-5`. CUDA cuPIQP versus
float64 qpth: torque `rtol=1e-4, atol=2e-4 Nm`, force `rtol=1e-3, atol=2e-3 N`.
No tolerances were loosened to pass these tests.

Both commands report the existing `hppfcl` → `coal` deprecation warning.
Compilation and `git diff --check` pass.

## Scope / pending validation

No simulator, training, or benchmark was run. Before the request to restrict
testing, the reduced-QP suite plus existing single/two-anchor mocked checks
completed with **35 passed, 3 skipped**. No further smoke checks were started.
Moreau and full training/checkpoint smokes are not validated in this change.
Older tests asserting the removed 54-D/slack/recovery formulation still need
migration; this is not a claim that the complete historical test suite passes.

Changed implementation: `hard_pact_qp.py`, `hard_pact_active_constraints.py`,
`hard_pact_qp_diagnostics.py`, `ppo_hard_pact.py`, `go2_hard_pact.py`,
`go2_hard_pact_config.py`, `deployment.py`, and `eval_hard_pact_frozen.py`.
The existing backend adapters/pools are reused unchanged; cuPIQP setup/update
native-bound signatures were inspected before packing. New correctness tests
are in `test_hard_pact_reduced_qp.py`; the two anchor test files were migrated
off the removed slack output/loss.
