# FACET Stage 4: Observation Packets and Models

Historical Stage 4 behavior is described below. **Stage 5 now connects this API
to the FACET runner**; see [Stage 5](b1z1_facet_stage5.md) for current training.

Stage 4 added a separate ordinary-PyTorch model and observation API. The then-existing
`--task b1z1_facet` training command used the legacy Pos runner/model.
Teacher/student PPO, adaptation losses, and collection integration are Stage 5.
No reference/reward/controller/terrain settings or shared implementations change.

## Source and Architecture

Ported from [active-adaptation at c0d6f5a](https://github.com/btx0424/active-adaptation/tree/c0d6f5aa6146814803090b5f6f19aafb8ce6f792):
`learning/ppo/{ppo_facet,common}.py`, `learning/modules/{vecnorm,distributions}.py`,
and `projects/facet/task/facet/B1Z1.yaml`. MIT attribution is in
`rsl_rl/FACET_LICENSE.txt`. No TorchRL, TensorDict, Hydra, simulator, or optimizer
is imported by `rsl_rl.facet_model`.

Every hidden MLP layer is Linear, LayerNorm, Mish. Teacher and student actors
are independent `[256,256,256]` networks with 17 means and separate learned
state-independent standard deviations initialized to one. Distributions clamp
std to `1e-6`, sum entropy/log-probability over joints, and are unsquashed.
Deterministic actions are means; controller clipping remains downstream.

Privileged encoder: `[128] -> Linear(128)`; force encoder: `[32] -> Linear(32)`.
Estimator: `[128,128] -> GRUCell(128,128) -> LayerNorm`, add the pre-GRU MLP
features, then Linear split `[128,32,9]`. The last 9 values are a separate wrench
reconstruction head, not the 32-D force latent. No reconstruction loss or
calibration is implemented yet. Critic: `[512,256,128] -> Linear(1)`.

Actors concatenate `[obs, privileged_features, force_features]`. Critic uses
`[wrench, obs, privileged]`, matching upstream's sorted normalized keys.
Teacher/critic/encoders/estimator Linear weights use orthogonal gain `.01` and
zero bias. Student Linear and GRUCell initialization retain PyTorch defaults,
as upstream explicitly does. GRU state is raw hidden output, not LayerNorm output.

## Packet Schema

`env.get_facet_observations()` returns owned, detached copies of
`observation`, `privileged`, `wrench`, and boolean `is_init [N,1]`.
`env.get_facet_observation_schema()` gives block order, joint/foot names, and
horizons. Default dimensions are **237 / 52 / 9**, independent of the unchanged
legacy observation dimensions. These are raw values, not PACT-scaled observations.

Student blocks, in concatenation order:

| Block | Width | Source, Units, and Frame |
|---|---:|---|
| Nominal base command | 3 | `commands[:,:3]`: heading vx/vy [m/s], yaw rate [rad/s] |
| Nominal EE position | 3 | `curr_ee_goal_cart`, mount XY offset, terrain height and configured height offset: heading-relative XY and world Z [m] |
| Nominal EE trajectory velocity | 3 | Analytic derivative of existing quintic spherical interpolation, converted to Cartesian [m/s]; same mixed command coordinates |
| Commanded impedance | 10 | `base_mass, ee_mass, yaw_inertia, base_stiffness, base_damping, ee_stiffness, ee_damping, yaw_stiffness, yaw_damping, alpha`; [kg, kg, kg m2, N/m, Ns/m, N/m, Ns/m, Nm/rad, Nms/rad, 1] |
| Projected-gravity history | 12 | Four 3-vectors, body/trunk frame, dimensionless unit-gravity sensor plus configured noise |
| Joint-position history | 76 | Four sets of 19 absolute encoder angles [rad], not offsets from default pose |
| Joint-velocity history | 76 | Four sets of 19 encoder velocities [rad/s] |
| Issued position-action history | 51 | Three previous 17-vectors, dimensionless; clipped to `normalization.clip_actions` **before delay**, before position scaling/default offset |
| Current kinematic EE position | 3 | Shared analytic six-joint Z1 FK using current noisy joint measurement; trunk-relative XYZ [m] |

Observed joints are explicitly name-mapped: FR/FL/RR/RL hip, thigh, calf, then
`z1_waist, z1_shoulder, z1_elbow, z1_wrist_angle, z1_forearm_roll,
z1_wrist_rotate, z1_jointGripper`. The first 17 are the controller's action
order; the held wrist rotation and gripper are observed too. A configurable
observed-joint ordering may be used, but must include the six FK joints.
The collector never reads Pos's compatibility slots containing executed PD torque.

The command adapter is intentional: upstream's commands are not our B1Z1 command
interface. Nominal trajectory derivatives do not include privileged base motion.
The FK block uses trunk axes, rather than upstream's yaw-only body position;
it needs only encoder measurements and nominal kinematics. It is not a simulator
EE-position sensor. Impedance parameters are task commands, not randomized
physical masses/gains. Force-dependent virtual references never enter the student.

Privileged blocks:

| Block | Width | Source, Units, and Frame |
|---|---:|---|
| Base linear velocity | 3 | Fresh base-link world velocity rotated into heading axes [m/s] |
| EE linear velocity | 3 | Fresh EE-link world velocity rotated into heading axes [m/s], not velocity relative to the base |
| Base angular velocity | 3 | Fresh world angular velocity rotated into heading axes [rad/s] |
| Foot contacts | 4 | Existing normal-contact logic; FR, FL, RR, RL, float 0/1 |
| Reference features + validity | 39 | Three horizons (8/16/32), each 12 values followed by one 0/1 validity |

Per-horizon order: base XY displacement from measured base [m, heading]; base
velocity XY [m/s, heading]; wrapped reference-minus-measured yaw [rad]; reference
yaw rate [rad/s]; EE position relative to measured base [m, heading]; EE velocity
[m/s, heading]. Invalid horizons are all zero including validity. These are the
existing same-time multi-start targets, **not future horizons**. Upstream YAML's
commented-out privileged heading mistakenly puts privileged fields in policy;
this API implements the intended separation explicitly.

Wrench order is `[base Fx,Fy,Fz, EE Fx,Fy,Fz, base Mx,My,Mz]` in current heading
axes, forces [N], moments [Nm]. Read existing held applied buffers after enable
gates; shift source free couple by `(application_point-base_link_origin) x F`
exactly once. No CoM gravity wrench, contact force, virtual spring force, or EE
reaction is added. Values describe loads held at this boundary, not future
disturbance draws. At reset the wrench and contacts are zero to exclude retained
terminal sensor/load buffers.

## Timing, Noise, and Reset

Histories are newest-first: sensors `[t,t-1,t-2,t-3]`, issued actions
`[a(t-1),a(t-2),a(t-3)]`. Initial sensor samples repeat the first reset measurement;
action history is zero. `_pre_sim_step` snapshots clipped issued positions before
delay. After the existing post-step/reward/reset/observation path, the collector
advances once; robots already collected during reset are skipped. Getters only
copy, never sample, normalize, or advance. Manual partial resets update only
selected packet/history rows.

This uses control-boundary encoder/gravity measurements and existing B1Z1
uniform noise amplitudes (`noise_scales * noise_level`, in raw units), not
upstream substep averaging/differentiation or Gaussian joint-noise sampling.
Private `noise_seed=1704` RNG isolates these additional measurements from all
shared task sampling. No extra privileged noise is added. Isaac reset reads use
fresh articulation data, not cached terminal simulator measurements.

`FACETReferenceBank.read_current()` selects the cleared current-episode bank.
Existing `read()` and `get_facet_tracking()` continue retaining completed terminal
snapshots for diagnostics. `is_init` is true only for the first observation of
each episode. Both single-step and sequence estimator APIs zero the matching GRU
rows **before** consuming that observation. Sequence input is `[B,T,D]`, input
hidden `[B,128]`, returned hidden is final `[B,128]`; no external reset framework.

## Standalone Use

```python
import torch
from rsl_rl.facet_model import FACETModel

packet = env.get_facet_observations()
model = FACETModel(packet['observation'].shape[-1], packet['privileged'].shape[-1]).to(env.device)
# Future collector: call exactly once per new batch, NOT per teacher/student query.
model.update_normalizers(packet['observation'], packet['privileged'], packet['wrench'])
hidden = torch.zeros(env.num_envs, 128, device=env.device)
action, hidden, wrench_reconstruction = model.student_action(
    packet['observation'], hidden, packet['is_init'])
teacher_action = model.teacher_action(packet['observation'], packet['privileged'], packet['wrench'])
value = model.value(packet['observation'], packet['privileged'], packet['wrench'])
```

Distribution APIs expose `.log_prob(actions)` and `.entropy()` for Stage 5.
`estimate()` and `student_action()` also accept sequences. Student inference
needs only observation/hidden/is_init, not privileged/wrench inputs or simulator.
Standard `state_dict` includes all weights and three independent normalizers.

Normalization matches upstream VecNorm(decay=1): running first/second moments,
initial moments zero, count 1, variance minimum `1e-5`. Forward never updates,
regardless of train/eval mode. `update_normalizers` is explicit and rejects eval
mode; teacher and student share observation statistics. Unlike upstream policy
construction, there is no implicit fake-observation statistics update.

Disable only the new collector using `cfg.facet_observations.enabled=False`.
This does not disable references/rewards or alter legacy observations. The model
is never automatically instantiated by the environment or existing runner.

## Verification

From repository root, using the active Isaac Lab Python environment:

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python -m pytest \
  tests/test_b1z1_facet.py tests/test_b1z1_facet_reference.py \
  tests/test_b1z1_facet_history.py tests/test_b1z1_facet_model.py \
  tests/test_b1z1_facet_observations.py -q
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python \
  tests/smoke_b1z1_facet.py --gpu cuda:1 --steps 40 --output-dir /tmp/facet-stage4-smoke
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python \
  tests/benchmark_b1z1_facet_model.py --device cuda:1 --num-envs 4096
```

Set `FACET_UPSTREAM_ROOT` to a checkout of the pinned commit for the numerical
upstream comparison. The test extracts only pure-PyTorch AST definitions, avoiding
framework initialization; it skips explicitly if the checkout is unavailable.
Actor/encoder/critic/GRU/normalizer outputs were identical with copied weights.
The final targeted FACET suite passed **48 tests**. The broader position-only
regression run additionally passed nine tests, with the existing failure and
CUDA-only skip described below. Syntax checks passed for all ten changed/new
Python files; the final diff contains only FACET-specific files.

The five-process Isaac Lab smoke (GPU 1, two robots, 40 intervals each plus real
timeout/roll resets) passed. It compares explicitly installed states/mechanics,
commands, forces, delay 0/2, non-unit gains/motors, rather than trusting equal
seeds. Packet collection plus inference on/off produced bitwise identical
torques, physical states, legacy observations and rewards. Student/teacher means,
critic values and reconstruction head were finite. No learning was run.

4096-row RTX 4090 benchmark (30 warmups, five blocks of 30, float32/no gradients):
packet assembly/read **1.759 ms**, teacher **0.268 ms**, student **0.288 ms**,
critic **0.131 ms**; process peak allocated GPU memory **68.87 MiB**. Assembly
includes noise and production six-joint FK operations on synthetic buffers;
excludes existing reference integration, simulator stepping and PPO backward.
These are component timings, not an end-to-end training-speed claim.

Existing Pos regression `test_configuration_inheritance_and_registration` still
expects `conditioning_mode='film'`, while the protected current baseline selects
`'none'`; this pre-existing failure is not changed. The sandbox CPU test run also
skips its CUDA-only FK test; the separate Isaac smoke was run unsandboxed.

Stage 5 remains responsible for choosing teacher/student training phases,
rollout storage, normalization update ownership, recurrence replay, objectives,
and checkpoint-to-runner integration. Stage 4 smoke tests do not establish learned
FACET performance or compatibility with upstream checkpoint observation layouts.
