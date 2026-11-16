# Legacy B1Z1 PACT / PILOT TorchScript Deployment

This export targets `Oct29_15-14-37_b1z1_pact_improved/model_70000.pt`, **before
differential compliance**. It uses the run's archived config, not current task
defaults. No 12-D compliance profile or compliant-reference generator is added.
The saved 81-D input contract is strictly checked; newer checkpoints are rejected.

## Files and Dependencies

- `policy.pt`: `torch.jit.script`, not a trace; dynamic batch size, explicit state.
- `metadata.json`: joint order, gains, scales, source hashes, iteration, validation.
- `README.md`: this API document.

Deployment requires PyTorch (exported/tested with 2.7.0+cu128), or a compatible
LibTorch runtime. It requires no repository Python imports, Isaac Lab, Genesis,
BARD, optimizer, critic, privileged observations or measured external forces.
Posterior mean inference includes the history encoder, explicit-state estimator,
force decoder, V02 preprocessing/LayerNorm/modulation, and both actor heads.
Optional GRF inference is exported, but is not used for choosing actions. Its
legacy training semantics must not be reinterpreted as new total-GRF supervision.

## Observation Contract

All API tensors are float32 on the model's device, except boolean reset flags
and int64 delay indices. Batch dimension N must agree across arguments.
`build_observation(roll_pitch, omega_body, q, qd, phase, previous_issued, commands)`:

| Input | Shape | Units and convention |
| --- | --- | --- |
| roll_pitch | N,2 | Base Euler roll/pitch radians, NOT projected gravity |
| omega_body | N,3 | Base angular velocity in body frame, rad/s |
| q, qd | N,19 | Joint position rad / velocity rad/s in metadata joint order |
| phase | N | Gait phase in [0,1), reset to zero |
| previous_issued | N,34 | Previous clipped position/FF action, BEFORE execution delay; zero on reset |
| commands | N,6 | Native base vx,vy,yaw-rate commands [m/s,m/s,rad/s], EE spherical radius/elevation/azimuth [m,rad,rad] |

Joint order: FR hip/thigh/calf, FL hip/thigh/calf, RR hip/thigh/calf,
RL hip/thigh/calf, Z1 waist/shoulder/elbow/wrist_angle/forearm_roll,
wrist_rotate/gripper. Last two joints are held, not learned.
The native base commands are compared with body-frame velocity estimates.
EE spherical axes rotate with base **yaw**, not EE orientation. Its center has
XY = base XY + yaw-rotated [0.3,0], Z = terrain height + 0.6985 m. For elevation e,
azimuth a, local Cartesian target is `[r*cos(e)*cos(a),r*cos(e)*sin(a),r*sin(e)]`.
Supply the CURRENT smoothly interpolated EE command, not a distant goal jump.
Command planning, workspace rejection and hardware sensing remain caller tasks.

The builder returns `[roll,pitch, omega*.25, (q-q_default)[:17], qd[:17]*.05,
sin(2*pi*phase),cos(2*pi*phase), previous_issued[34], commands*[2,2,.25,.5,1,1.3]]`.
No artificial observation noise is added. Do not normalize these fields again.

## Stateful Control Loop

Control interval is 0.02 s (50 Hz). Initialize on the desired device:

```python
import torch
p = torch.jit.load('policy.pt', map_location='cpu').eval()
history, film_history, previous_issued = p.initial_state(1)
phase = torch.zeros(1)
is_init = torch.ones(1, dtype=torch.bool)

# Each control boundary: supply measured roll_pitch, omega_body, q, qd,
# and the current nominal commands. These are hardware/application inputs.
with torch.inference_mode():
    previous_issued[is_init] = 0
    phase[is_init] = 0
    obs = p.build_observation(roll_pitch, omega_body, q, qd, phase,
                              previous_issued, commands)
    raw, issued, q_target, ff_nm, history, film_history = p.step(
        obs, history, film_history, is_init)
    previous_issued = issued
    phase = p.advance_phase(phase, commands[:, :3])
    is_init.zero_()
```

`step` appends one observation and one tracking-error frame exactly once. Do not
call it twice for diagnostics at the same control boundary. For partial reset,
set only those robots' `is_init=True`, zero their previous issued action and
phase before building the first observation. The other rows retain their state.
First history contains 31 zero frames followed by the first observation; the
first FiLM history contains four zero errors followed by the current error.

History is N,2592, 32 frames oldest-to-newest including current observation.
FiLM history is N,5,6, oldest-to-newest errors, base group then EE group.
Base/EE soft-L1 offsets are 0.10/0.05; each frame also gets two tanh(L1) values,
yielding 40 conditioning features. Only actor z is modulated; decoders use raw z.
No histories are hidden inside the artifact, making concurrent robots explicit.

For existing observation pipelines, `p(obs, history, previous_film_history)`
returns `(raw_action, updated_film_history)`. Here history must ALREADY include
the current frame; this low-level API does not append/reset observation history.
As in training, history stores pre-clipped observations; actor obs is clipped at
100. These differ only for extreme inputs/noise. Finite sensor data and bounds
must be checked by the deployment supervisor before invoking inference.

## Action and Controller API

`raw` is N,34, `[17 position actions,17 feedforward actions]`, without sampling
or tanh. `issued` clips these to +/-50, matching training. `action_targets` maps
issued actions to N,19 targets: `q_default + .25*position_action` on learned
joints; held joints remain at defaults. FF scales are 30 Nm/action for legs,
10 Nm/action for five learned arm joints; held joints have zero FF. These are
UNWEIGHTED FF torques, not already combined motor commands.

Default Kp: legs 100; arm shoulder 60, other arm/held joints 40. Kd: legs 5;
shoulder 1.4, other arm/held joints .8. Exact vectors are in metadata/module buffers.
Training randomized gains/strength/branch weights. A nominal deployment uses
unit Kp/Kd/motor multipliers and unit feedback/feedforward weights, with no added
randomization. These choices are explicit, not learned controller parameters.

`controller(executed_action, q, qd, kp_scale, kd_scale, motor_strength,
feedback_weight, feedforward_weight, execution_limits)` returns
`(effective_pd, effective_ff, pre_clip_total, clipped_total)` in Nm for 19 joints.
Gain/motor multipliers broadcast to N,19; branch weights to N,1 or N,19.
Use fresh q/qd each actuator substep, hold the action for the control interval.
Training used four substeps per 20 ms policy interval and clipped at **1.1 times
rated torque limits after motor scaling**. Supply validated execution limits
explicitly; do not automatically increase hardware-rated limits by 10%.
Applying PD/FF again downstream would double count torques.

Optional `apply_delay(issued, queue, delay_steps, is_init)` returns executed
action and updated newest-first N,L,34 FIFO. Delays are N int64 in [0,L).
Training randomized delays 0/1 control intervals. No artificial delay is needed
unless matching that experiment; do not delay twice if hardware already does so.
Always feed the previous ISSUED action, not delayed action or executed torque,
back to the observation builder. Convert the delayed action via `action_targets`
or `controller` if a delay is used.

The action clip is not a safe joint-position limit. Hardware startup/posture
alignment, calibrated encoders and IMU frames, watchdog, torque/rate/position
limits, fall handling and emergency stop are external responsibilities. This is
an inference-ready artifact, not evidence of safe or successful hardware control.

## Optional Estimates and Numerical Precision

`estimate(history)` returns unmodulated z (N,64), explicit estimates (N,14),
forces (N,9). Explicit first six use command scaling, four contacts are
probabilities, last four foot heights are already meters (NOT multiplied by 5).
Force order is base force XYZ, base moment XYZ, EE force XYZ, heading-frame;
divide by .01 to recover N/Nm. Do not add the later randomized-CoM correction
or change the original learned wrench semantics. These are estimates, not sensors.
`predict_grf(history, nominal_torque_nm[N,19])` returns N,12 normalized outputs,
FR/FL/RR/RL XYZ; divide by .001 for legacy forces in the **successor-heading**
frame. Rotate by successor yaw for world-frame interval-force diagnostics; do
not mistake these for current-time world-frame sensor forces.

Float32 matmul precision is a HOST setting, not embedded in TorchScript.
Historical GPU recordings match `torch.set_float32_matmul_precision('high')`
and their original batch size (25). CPU/full-FP32 and changed GPU batch sizes
can give small numerical differences. Export validation reports each comparison
and tolerances explicitly; no inputs or outputs are corrected to force parity.

## Reproduce the Export

From this worktree (no SIMULATOR environment variable is required):

```bash
python legged_gym/scripts/export_b1z1_legacy_pact_jit.py \
  --checkpoint /path/to/Oct29_15-14-37_b1z1_pact_improved/model_70000.pt \
  --output-dir exports/b1z1_pact/Oct29_15-14-37_b1z1_pact_improved/model_70000
python -m pytest tests/test_b1z1_legacy_deployment.py -q
```

Optional real-recording validation: add `--recording /path/to/transitions-000000.npz
--recording-device cuda:0 --recording-precision high`. The adjacent metadata
must identify the exact checkpoint hash. Only full recorded inference batches
are compared, with excluded partial-tail row counts reported. The checkpoint
is trusted PyTorch pickle input; never run this exporter on an untrusted file.
No simulation, training, model updates, or hardware rollout occurs in export/tests.
