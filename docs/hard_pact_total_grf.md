# HardPACT Total Ground-Reaction Forces

Isaac Lab's installed `ContactSensor.net_forces_w` is normal-only. HardPACT,
its thin Isaac Lab ablations and HardPACTPos now default to
`cfg.sim.grf.force_mode = "total"`. No network shapes, normalization scales,
reward weights, or installed simulator packages change. Genesis/Gym keep
their existing interfaces and processing.

## Contact Contract

- `ground_contact_forces.py` reuses the B1Z1 implementation: one Warp thread
  per sensor/ground-filter pair visits only live normal contacts and friction
  anchors. Counts/start indices are independent; the installed PhysX API
  reuses their storage between calls.
- Exact ground collision prim filters exclude robot/self contacts. Actual
  contact normals are used, including on slopes. Sensor order is preserved;
  the existing named mapping selects canonical FR/FL/RR/RL feet.
- `get_contact_data(dt)` and `get_friction_data(dt)` use the **physics** dt
  and already return N. No extra division by dt is performed. Normal plus
  tangential equals total. Missing friction fails instead of becoming zero.
- GPU reductions have no per-contact host copies. API errors, invalid ranges,
  nonfinite contacts and capacity exhaustion fail closed. Increase
  `contact_capacity_per_body` (default 8) if needed. Exact capacity exhaustion
  is conservatively rejected because PhysX exposes no dropped-contact count.
- Normal/anchor point cross force yields world-origin moments. The adapter's
  `ground_contact_moment_world(reference_world, feet_only=...)` shifts these
  to a supplied reference in world meters. No foot-center approximation is
  needed in total mode; unexposed free contact couples remain unavailable.
- Persistent world wrenches reset the composer's cached transform each
  substep before setting the load, matching the B1Z1 frame correction.

## Consumers and Timing

Completed-step total forces reach `hard_pact_foot_forces_world`, the existing
GRF processor, GRF observations and interval reconstruction/dynamics labels.
Existing deadband, clipping, EMA and pre-step yaw rotation remain. Equal
physics timesteps make the existing substep mean a time-weighted interval
mean. Contact gates use the magnitude of the separate normal force; collision,
termination and other normal-sensor logic remain normal-only.

The frozen evaluator records raw normal/tangential/total substep components
separately from conditioned training labels. Training writes
`grf_semantics.json`; evaluation embeds the same version-2 semantics in
metadata. It includes units/frame, dt, source, filters, capacity and ordering.
Missing historical metadata means **legacy normal-only**, not total.

Set `force_mode = "normal_only"` explicitly for normal-only labels. Tangential
and exact moment buffers are unavailable (NaN), never fabricated zeros.
Existing checkpoints remain loadable, but their normal-only-trained GRF heads
do not automatically become estimators of tangential force.

## Reproducible Validation

In the installed Isaac Lab/cuPIQP environment, from the repository root:

```bash
python scripts/validate_hard_pact_grf.py --task go2_hard_pact_full_isaaclab \
  --gpu cuda:0 --steps 30 --training-update --output-dir /tmp/hard-grf-full
python scripts/validate_hard_pact_grf.py --task go2_hard_pact_full_isaaclab \
  --gpu cuda:0 --steps 30 --frictionless --output-dir /tmp/hard-grf-frictionless
```

Repeat with `go2_hard_pact_pos_isaaclab`, `go2_hard_pact_baseline_isaaclab`
or another registered ablation. Each process uses two robots, nominal PD
standing, downward and resisted horizontal base loads, and partial reset.
The optional update performs four collection steps and one PPO epoch/minibatch,
not training to convergence. Reports distinguish this from trained-policy
performance. Full-body momentum includes gravity, known external loads and
all ground contacts (not only feet). CUDA-event timings cover contact query
and reduction; they are not full-training throughput measurements.

```bash
SIMULATOR=isaaclab python -m pytest -q -p no:cacheprovider \
  tests/test_ground_contact_forces.py tests/test_hard_pact_total_grf.py \
  tests/test_go2_hard_pact_grf.py tests/test_isaaclab_pact_torque_limits.py \
  tests/test_isaaclab_pact_headless_stop.py tests/test_eval_hard_pact_frozen.py
```

## Verified on This Branch (2026-11-03)

- Installed Isaac Lab 0.54.2 / Isaac Sim 5.1, RTX 4090 `cuda:0`.
- Focused command above: **50 passed**, plus 4 unittest subtests; syntax and
  `git diff --check` passed. Tests include frozen-recorder component accounting.
- `full_isaaclab`, `pos_isaaclab`, and `baseline_isaaclab` each completed the
  two-environment physical checks and one four-step PPO update with finite
  model parameters. These are warmup updates, not activated-QP performance tests.
- Full/Pos: -25 N horizontal load gave +25.13 N mean tangential ground reaction.
  Standing/downward-load normal support was 156.33/182.80 N during these short
  transient windows. Their largest momentum residual was 2.19e-8 of
  gravitational impulse (observed acceleration is included in this check).
- Baseline with `--frictionless`: tangential forces were zero; largest relative
  momentum residual 3.04e-7. Component sums and partial resets passed throughout.
- Contact queries plus GPU reduction took 0.55-0.67 ms/substep at two robots.
  This is not a large-environment throughput benchmark.

Reports are `/tmp/hard-grf-full/validation.json`,
`/tmp/hard-grf-pos/validation.json`, and
`/tmp/hard-grf-baseline-frictionless/validation.json`. The last two commands were:

```bash
python scripts/validate_hard_pact_grf.py --task go2_hard_pact_pos_isaaclab \
  --gpu cuda:0 --steps 30 --training-update --output-dir /tmp/hard-grf-pos
python scripts/validate_hard_pact_grf.py --task go2_hard_pact_baseline_isaaclab \
  --gpu cuda:0 --steps 30 --training-update --frictionless \
  --output-dir /tmp/hard-grf-baseline-frictionless
```

An additional targeted regression selection (`test_hard_pact_contact_indexing_and_inverse_gate`,
`test_hard_pact_action_replay`, `test_hard_pact_swing_grf`) gave 23 passes,
2 skips and 3 failures in unchanged contracts: two QP mocks lack
`endpoint_torque_constraints`, and one test expects deployment schema 10
instead of the current 17. Those unrelated fixtures were not changed.
Live rough-terrain/filter and large-batch overflow/throughput stress tests,
trained-checkpoint frozen evaluation, and full training were not run. Sloped
normals, capacity failures and multi-filter reductions were tested synthetically.
