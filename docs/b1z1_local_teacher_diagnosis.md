# Local Teacher Omega/DLS Diagnostic

This diagnostic changes no production parameters or dynamics. On GPU 0, a
headless Isaac Lab smoke ran eight untrained robots for eight control steps
and one PPO update. The first teacher call's 64 frozen inputs were replayed
at 72 combinations of omega and DLS damping. These are correlated transitions
from a small untrained-policy test, not trained-policy performance evidence.

Omega: 0, 0.5, 1, 2, 3, 4, 6, 8, 12 rad/s. DLS damping: 0.001, 0.005,
0.01, 0.025, 0.05, 0.1, 0.2, 0.5 m/rad. Actions, references, mechanics and
predicted non-arm accelerations were identical across the sweep. This isolates
teacher sensitivity; it does not rerun the reference generator at each omega.

| Omega | DLS | Usable rows | Mean task-acceleration residual [m/s^2] |
| --- | --- | --- | --- |
| 6 (current) | 0.025 | 7/64 | 0.670 |
| 1 | 0.025 | 8/64 | 0.676 |
| 3 | 0.025 | 8/64 | 0.673 |
| 6 | 0.05 | 8/64 | 2.623 |
| 6 | 0.1 | 8/64 | 9.685 |

No tested combination exceeded 8/64 usable rows. At current settings,
76.56% failed joint-speed checks, 79.69% failed combined PD+FF execution limits,
and 14.06% failed buffered joint-position checks. These reasons overlap.
No rows exceeded the raw FF action-channel range; its configured action clip
allows a much larger torque than the final actuator clipping limits.

The mean norm of `a0 = solve(M_aa, tau_PD)` was 1496 rad/s^2. Decomposing
`a_arm = (I-H#H)a0 + H# a_task - H#(H_u a_u + b)` gave mean norms of
1475, 29, and 270 rad/s^2 respectively. Norms are NOT additive. Lowering omega
to 1 reduced the reference contribution to 5 rad/s^2, but left the retained
PD component unchanged. DLS cannot suppress the exact task-null-space part
of a0, explaining the weak coverage response to this parameter sweep.

A frozen-state control recomputed mechanics at the same joint positions with
zero velocities, zero PD error and zero prescribed non-arm acceleration.
Holding the measured EE pose passed 59/64 rows. A 10 cm x-reference offset
also passed 59/64 for omega 1, 3 and 6, with no torque/speed failures. The five
rejected rows were already inside a buffered joint-position margin. This is
a local teacher sanity check, not a separately simulated stable hold.

The evidence does not justify weakening feasibility checks or changing shared
omega/DLS defaults. Investigate bounding the retained PD acceleration before
claiming gain tuning resolves the issue. Changing shared omega would also
alter compliance-reference dynamics, which this frozen-input sweep does not
test.

## Reproduction

Use the installed Isaac Lab Python environment from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python -u tests/diagnose_b1z1_local_feedforward.py --task b1z1_pact --gpu cuda:0 --headless
CUDA_VISIBLE_DEVICES=0 SIMULATOR=isaaclab PYTHONPATH=. python tests/diagnose_b1z1_local_feedforward.py --replay
SIMULATOR=isaaclab PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/test_b1z1_teacher_diagnosis.py tests/test_b1z1_local_feedforward.py
```

Outputs: `/tmp/b1z1_teacher_diagnosis/inputs.pt`, `results.json`, `sweep.csv`.
Only load the locally generated `inputs.pt`, which contains trusted Python
configuration objects. The JSON includes per-joint rejection frequencies,
limits, teacher torques and acceleration decomposition. CPU validation:
22 passed, one CUDA-only check skipped. The real BARD sweep and Isaac Lab
capture completed on GPU 0; syntax checks and `git diff --check` passed.
