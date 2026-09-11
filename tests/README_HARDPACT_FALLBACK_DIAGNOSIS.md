# cuPIQP fallback causes: bounded real Isaac Lab collection/PPO test

Purpose: distinguish empty joint intersections, numerical certification/gap
failures and solver exceptions without changing training constraints/settings.
Harness: `tests/diagnose_hard_pact_fallbacks.py`, wrapping the existing bounded
100-environment smoke with read-only counters and independent LP checks.

```bash
timeout 240s env SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
MPLCONFIGDIR=/tmp/hardpact_mpl conda run --no-capture-output -n lr_lab_cupiqp \
python -u tests/diagnose_hard_pact_fallbacks.py \
--task go2_hard_pact_full_isaaclab --headless --num_envs 100 --gpu cuda:0 --qp_solver cupiqp
```

Result: PASS, exit 0. Real RTX 4090 / Isaac Lab / cuPIQP, two iterations per
mode, four control steps per rollout and one PPO epoch/minibatch; QP active from
iteration zero. Random initialized policy, not the user's live checkpoint.
Modes run sequentially, so their rates are not controlled comparisons of modes.
Finite outputs, actuator magnitude/rate bounds, and nonzero finite actor,
encoder and GRF/wrench gradients passed the underlying smoke.

| Mode/phase | Problems | Certified | Empty joint intersection | Primal rejection | Gap-only rejection |
|---|---:|---:|---:|---:|---:|
| Every-substep rollout | 3200 | 2825 | 7 | 368 | 0 |
| Every-substep PPO | 800 | 702 | 1 | 96 | 1 |
| Random-one rollout | 800 | 111 | 0 | 689 | 0 |
| Random-one PPO | 800 | 111 | 0 | 689 | 0 |

All buckets: zero exceptions, nonfinite input/output, bad mechanics, empty
torque intersections or unclassified rejections. Every primal-rejected row
violated joint inequalities; friction also violated on 141/38/376/371 rows
respectively (overlapping categories). No candidate torque/rate violations
remained after actuator projection.

The seven empty rollout joint-coordinate intersections in full mode had active
lower sources acceleration (5), velocity (2), and upper sources position (5),
acceleration (2). Two had current position outside limits, two current velocity
outside limits. PPO had one acceleration-lower/position-upper contradiction.
Most rejected rows had NONEMPTY coordinate-wise acceleration intervals: the
problem was the coupled full QP, not merely a failed interval precheck.

Independent feasibility test: captured the first eight primal-rejected rows
per mode/phase (32 total), retaining all 68 scaled inequalities and 24 free
variables. SciPy HiGHS solved zero-objective LPs with 1e-9 primal/dual feasibility
tolerances and five-second per-case limit. **All 32 were reported infeasible**;
none feasible or undetermined. This supports genuine constraint incompatibility
for those samples, not just insufficient cuPIQP iteration count. It does not
prove all other rejected rows are infeasible or identify a single limit to widen.
Hard predicted swing-contact masks and deployment mechanics remain part of the
tested system; this is not a claim that real robot physics is infeasible.

No production code, limits, objectives, solver tolerances or recovery paths were
changed for this diagnosis. Only the diagnostic harness and this report added.
