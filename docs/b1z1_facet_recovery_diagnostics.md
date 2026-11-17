# FACET seed-aware recovery diagnostics

`facet_reference.recovery_diagnostics = True` enables reporting only. Set it to
`False` to omit the new checks/storage/reductions. Neither setting changes
reference integration, constraints, rewards, eligibility, observations, forces,
or training schedules. Existing `FACETHistory` metrics retain their meanings:
in particular, `constraint_recovery` describes the **interval start**, not the
original history seed, and `constraint_incompatible` is an any-substep flag.

## Definitions

Each ring trajectory retains its measured seed's offset norm (m), excess above
`max_offset` (m), and six independent checked-constraint flags:

- `offset`: total reference-to-current-nominal offset, including inherited error.
- `workspace`: spherical radius/elevation/azimuth bounds.
- `box_ground`: forbidden box or ground exclusion.
- `offset_speed`: additional compliant-offset velocity, not nominal trajectory speed.
- `base_speed`: planar virtual-base speed.
- `yaw_rate`: absolute virtual yaw rate.

Checks use the existing guard tolerance, 1e-4 in the corresponding SI units.
These are geometric/velocity checks, **not full articulated collision or IK
certification**. `end_*` describes the accepted interval endpoint; `any_*` is
the union of accepted numerical-substep endpoints (not unconstrained RK stages).
The seed is classified separately, so an inherited violation is not a new one.

`new_violation` means an initially feasible seed subsequently violated a check.
For initially infeasible seeds, first recovery is the first fully feasible
control endpoint. `recovery_time_s` and `recovery_age` are measured along that
same trajectory; subsequent any-substep violations set the persistent re-entry
flag. Recovery is endpoint-observed, not a continuous-time crossing estimate.
Unrecovered trajectories have no recovery-time observation, not zero duration.
Expiration/reset censor counts include only initially infeasible trajectories
that never recovered. Counts are accumulated once during mutation, never reads.

Offset/excess reduction compares the original seed with that trajectory's
current endpoint. `nominal_motion_m` exposes movement of its nominal task-space
command since seeding; reductions must not automatically be attributed to force
yielding. Never infer individual recovery from different age populations.

## Aggregation and output

`FACETRecovery/age_{8,16,32}/...` TensorBoard tags and
`runner.last_reference_diagnostics` report full-rollout sufficient statistics.
The smoke JSON includes the latter. Each eligible trajectory contributes once
at each requested age; terminal pre-reset snapshots are included. Unused slots
are excluded. Constraint validity does not change age eligibility or rewards.

Groups are `all`, `initially_feasible`, `initially_infeasible`, `zero_load`, and
`loaded`. Counts/fractions use the group's eligible observation count. Recovery
time/age means use **recovered_count** only; counts and sums remain available for
audit. These are horizon observations, not independent robot trials; never sum
across horizons as unique recoveries. Lifecycle censor counts are unique ring
trajectories retired during the rollout, including seeds from earlier rollouts.

`loaded` means any actually applied EE/base force or base moment above 1e-6
since that seed, including roll/pitch couples and test-injected loads. It stays
loaded after release. Exposure integrals count each environment interval once,
not once per history: seconds, integral force norms (N s), and integral
base-referenced moment norm (N m s). No sampled-target proxy is used.

Empty groups retain count zero, JSON means/fractions/maxima are `null`, and
unavailable TensorBoard values are omitted. Reports include control dt, configured
episode duration, limits and collection phase. Short tests do not establish
production-training performance. No histories or diagnostic counters are restored
across simulator restarts, consistent with fresh measured-state episodes.

## Reproduction

From the worktree, using the Isaac Lab Python environment:

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python -m pytest tests/test_b1z1_facet*.py -q
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python tests/benchmark_b1z1_facet_recovery.py --device cuda:0 --num-envs 4096 --output /tmp/facet-recovery-benchmark.json
PYTHONPATH=.:tests OMP_NUM_THREADS=1 python tests/smoke_b1z1_facet_training.py --gpu cuda:0 --num-envs 8 --normal-episode-length --output-dir /tmp/facet-recovery-smoke
SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=.:tests OMP_NUM_THREADS=1 python tests/smoke_b1z1_facet.py --recovery-parity --steps 40 --gpu cuda:0 --output-dir /tmp/facet-recovery-parity
```

The benchmark compares identical bounded references with diagnostics off/on,
fills the history bank, warms kernels, measures new rollout reductions, and
asserts bitwise accepted-state parity. It excludes simulator/PPO and cold compile
time. The training smoke keeps production episode duration but uses test-local
early forces and a six-update handoff; its existing post-training evaluation
retains its separate two-second protocol.

The matched replay explicitly enables its injected force/couple sources and
disables the training sampler in the fixture only. This corrects an older
fixture's assumption that production FACET enabled torso torque sampling.

## Verification on reviewed HEAD 4b0f59c

See [machine-readable report](b1z1_facet_recovery_report.json) for source hashes,
configuration, denominators, load exposure and results. CPU regressions: 162
passed, three skipped; the two CUDA skips were subsequently exercised by three
passing CPU/GPU parity checks. Only the pinned upstream checkout check remained
unavailable. Six actual Isaac Lab updates (three teacher/three student), resume
and the existing short evaluation passed on GPU 0 with a 20 s training episode
limit. A separate 40-interval matched replay found 85 shared arrays bitwise equal
with diagnostics enabled/disabled, including rewards/actions/loads/references.

The spatially admissible, zero-load 0.77 m seed has offsets 0.61/0.45/0.138 m
at ages 8/16/32, recovering at age 29 (0.58 s). No workspace or forbidden-region
violation occurs. The final smoke rollout had no new violations among initially
feasible seeds; its age-8/16 remaining offset violations must not be interpreted
as evidence that feasible references became infeasible. Different horizon
populations are still not individual recovery curves.

Warmed 4096-environment reference/bank/reduction benchmark on RTX 4090:
13.85 ms off versus 15.05 ms on (+1.19 ms/control interval, approximately 38 ms
per 32-step rollout), with bitwise state parity. Peak allocated memory was
130.6 versus 167.7 MB. These are shared-GPU reference-path measurements, not
whole-training throughput or cold-compile timings. No full training was run.
