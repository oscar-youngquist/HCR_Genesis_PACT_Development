Oct 7 / checkpoint 20000: portable HardPACT deployment bundle
============================================================

Scope and status
----------------

``hardpact_oct07_20000.pt`` is a **torch.jit.script** artifact, not a traced
policy or a training checkpoint. It contains all neural components needed for
deterministic inference: history encoder/latent mean, explicit estimator,
position and feedforward actor heads, torque-conditioned GRF head and wrench
head. It also exports nominal control conversion, historical QP assembly,
recovery assembly, primal-candidate checking and FF reconstruction. Batch size
is dynamic. There are no simulator, repository, Python callback or custom
autograd dependencies at load/inference time.

It does **not** contain a numerical QP solver, Pinocchio/BARD, a state estimator,
device drivers, or a real-time scheduler. Those are explicit C++ dependencies.
cuPIQP's Python/CuPy solver cannot be embedded as a portable TorchScript module.
Do not mistake ``primary()`` for a solve. Training-only critics, reconstruction
decoder, latent variance and exploration parameters are intentionally excluded.

CPU neural and primary/recovery assembly parity was checked against the
historical source (see ``validation.json``). No C++ build, numerical cuPIQP solve,
MuJoCo rollout, hardware test, GPU timing or real-time deadline validation is
claimed for this artifact. This is a porting/reference bundle, not a certified
hardware controller. Actuator saturation alone does not guarantee safe motion.

Provenance and files
--------------------

* Run: ``Oct07_10-22-08_hard_pact_full_isaaclab``; ``model_20000.pt``.
* ``resolved_deployment.json``: checkpoint SHA256, exact saved configuration,
  final curriculum state, joint names/limits and deployment overrides.
* ``archive/run/``: original resolved config, archived environment/config and
  original deployment contract, unmodified.
* ``archive/``: source from ``41d5b7b7b5ebbcef3d760bf975d1275ebb7410e4``, including
  actor, physics/control helpers, QP/backends/PPO, environment, BARD and URDF.
  Its environment and config byte-match the run's archived files. The original
  complete working-tree hash was not checkpointed: other unarchived dirty
  training changes cannot be conclusively ruled out.
* ``golden_vectors.pt`` and ``golden_vectors.npz``: eight-row reference network
  inputs/outputs and complete primary/recovery assembly inputs/outputs for
  C++ comparisons. These are synthetic algebra fixtures, not real robot states.
* ``standalone.py``, ``export.py``, ``smoke.py``: portable source, reproducible
  CPU exporter/historical-reference checks, and repository-independent smoke.
* ``SHA256.json``: artifact/source hashes. Verify before loading.

The archived schema-17 contract contains stale prose (analytic-only recovery,
68 rows without endpoint rows, and output clearance clipping). The actual
archived implementation includes soft-joint recovery and 24 endpoint torque
rows; clearance labels, not network outputs, are clipped. The equations below,
verified assembly, and new versioned manifest supersede that stale prose.

Deployment override: every substep
----------------------------------

This checkpoint trained with ``qp_update_mode=random_one_substep``: one
uniformly selected 5 ms substep per 20 ms policy interval, with bounded nominal
commands on the other three. PPO recomputed one selected QP per sampled row.

The requested deployment executes a primary QP at **all four 5 ms substeps**;
recovery may add a second numerical solve for a rejected primary. Thus four
primary attempts, not necessarily four total numerical solves. No correction
is accumulated or held across solves. Use full execution correction alpha=1;
the checkpoint's correction ramp and objective curriculum had completed.

Every 20 ms:

1. Obtain current physical sensors/command. Build observation and append one
   history entry, exactly once. Evaluate ``forward`` once.
2. Clip raw actions to +/-50; apply the actual command delay queue, if any.
   Preserve nominal position/FF commands for the interval. Previous-action
   observation uses the previously **issued clipped** policy sample, not the
   delay-selected command or QP output.
3. Form bounded nominal torque at interval start, then run ``physics`` once.
   Hold latent, explicit/contact predictions, raw yaw-local GRFs and clipped
   yaw-local wrench throughout the interval. Do not feed corrected torque back
   through the GRF decoder in this baseline.

Every 5 ms, including k=0:

1. Read fresh q, dq, live xyzw orientation, angular velocity and state estimate.
   Rebuild nominal PD from the original delay-selected nominal commands.
   Do not use a cached control-rate orientation accessor.
2. Refresh nominal-model M, h, J, Jb and Jdot*v. Rotate held yaw-local force and
   wrench predictions with the **current** yaw. Rotate estimated body velocity
   with the current full orientation for height damping. Hold physical command
   inputs at their control-rate values.
3. Assemble ``primary(data)``. Do not dispatch invalid/nonfinite input rows or
   empty primary joint envelopes. Solve with the saved rollout profile below.
4. Post-project/check the primary candidate. On rejection or empty primary
   envelope, solve ``recovery(primary)`` if mechanics/input data are valid.
   On recovery failure/exception, use ``fallback(tau_nom, torque_limits)``.
5. Execute the selected **total** torque. For a position+FF interface, keep
   the nominal position target unchanged and use ``feedforward_command``.
   Replace the previous FF correction; never add the previous correction.
   Record the actual applied torque after all actuator-side limits.

The controller needs measured/calibrated effective gains, motor scaling and
branch weights. Training randomized these; silently setting all factors to 1
is an experimental controller change, not a reproduction of every training
environment. Do not randomize hardware gains to emulate training. Zero FF gain
cannot represent an arbitrary selected torque: use a separately validated
direct-total-torque path or reject the command. The returned availability mask
must not be ignored.

Ordinary hardware command delay remains caller-managed. Clear history, previous
issued action, delay queues and previous-torque state on restart/reset. Fill
history with zeros, then append the first real observation once. Initialize
previous torque from actual actuator state, not a stale QP prediction. There
is no torque-rate constraint in this checkpoint, but logging/state must still
refer to actual execution. No legacy/newer command-pair allocator is used.

TorchScript API
---------------

Load with ``torch.jit.load(path).eval()`` or ``torch::jit::load(path)``. Use
inference/no-grad mode. Default tensors are CPU float32. Move the entire module
and all inputs together if changing device/dtype; GPU/double execution has not
been validated here. Tensors are batched, contiguous where required, and in
canonical joint/foot order. No mutable inference history is hidden in the model.
Weights and QP coefficients are embedded in this export: editing the JSON does
not change the script. Re-export and rerun parity checks when changing them.

``forward(obs[B,57], history[B,570]) -> (raw_action[B,24], latent[B,16], explicit[B,12])``
  Deterministic latent **mean**, not a fresh VAE sample. Training used sampled
  latents and boot masking; deterministic inference uses the unmasked mean.
  Action order is position[0:12], FF[12:24]; neither is a physical torque yet.

``observation(command[B,3], gravity_body[B,3], omega_body[B,3], q[B,12], v[B,12], previous_action[B,24]) -> (obs, history_entry)``
  Packs the exact field order below. Deployment adds no synthetic noise.
  Returns current observation clipped to +/-100 and the unclipped history
  entry: the historical environment appended history before clipping its
  current observation. Enforce independent physical sensor validity checks.

``append_history(history[B,570], history_entry[B,57]) -> history[B,570]``
  Oldest first, newest last, including current timestamp. Call once per policy
  interval, not per QP substep. No implicit reset or delay processing.

``nominal(action, q, v, kp, kd, motor, feedback_weight, feedforward_weight, limits)``
  Returns ``(absolute_q_target, scaled_ff, physical_pd, requested_total,
  bounded_nominal_total)``. q/v/action Bx12/Bx24, gains/weights Bx12 (or
  broadcastable), limits 12 or Bx12. Position scale=.25 rad/action;
  FF scale=10 Nm/action. ``scaled_ff`` is before motor/branch weighting.

``physics(latent, explicit, bounded_nominal_at_interval_start[B,12])``
  Returns ``(raw_grf_yaw[B,4,3], raw_wrench_yaw[B,6], qp_wrench_yaw[B,6])``.
  Physical units N/Nm. GRF input torque is physical bounded total Nm, not
  normalized by torque scale. GRF head scale=100 N/component; wrench head
  scales=[100,100,100,25,25,25]. Wrench is sanitized and clipped once in
  yaw-local physical coordinates to +/-[150,150,150,40,40,40]. Raw outputs are
  retained for diagnostics. Contact gating happens in QP assembly.

``world_inputs(quat_xyzw[B,4], grf_yaw[B,4,3], wrench_yaw[B,6])``
  Returns world-frame forces/wrench using yaw only. ``rotate(quat, vector)``
  instead applies full body-to-world rotation to Bx3 vectors. Normalize valid
  input quaternions upstream; quaternion convention is xyzw, not Isaac wxyz.

``primary(data: Dict[str,Tensor]) -> Dict[str,Tensor]``
  Builds the scaled 24-variable primary. Required fields:

  * ``mass_matrix`` Bx18x18, ``bias`` Bx18, ``foot_jacobians`` Bx4x3x18,
    ``base_jacobian`` Bx6x18, ``foot_acceleration_bias`` Bx4x3.
  * ``tau_nom`` Bx12, ``force_pred_world`` Bx4x3,
    ``wrench_pred_world`` Bx6, ``contact_probability`` Bx4.
  * ``joint_position``, ``joint_velocity``, ``effective_kp``, ``effective_kd``
    Bx12. Effective gains include motor*feedback weight once.
  * ``base_quaternion`` Bx4, ``base_angular_velocity_world`` Bx3,
    ``base_linear_velocity_world`` Bx3, ``velocity_command`` Bx3.
    Body vx/vy and body wz commands are physical m/s, m/s, rad/s, with heading
    conversion already resolved. Do not use scaled observation entries.
    Hardware linear velocity requires a deployable state estimate (the explicit
    velocity prediction is available); training used simulator velocity for
    this inner tracking term. Replacing it with an estimate is a deployment
    uncertainty, not exact state-input parity with simulation truth.
  * ``estimated_height`` Bx1 and ``estimated_base_linear_velocity_world`` Bx3:
    explicit[11] and full-R times explicit[:3]/1.0, respectively.
  * ``dt`` Bx1 = .005 s; ``torque_limits``, ``position_lower``,
    ``position_upper``, ``velocity_limits`` length12 or Bx12. Supplied limits
    must be positive/ordered where appropriate and match the actuator model.

  Outputs: Q[B,24,24], p[B,24], G[B,92,24], h[B,92]; physical_G/physical_h;
  variable_scale[24], row_scale[B,92]; native_lower/upper[B,24];
  acceleration_map[B,18,24]/offset[B,18]; joint_lower/upper[B,12];
  tau_lower/upper[B,12]; mechanics_valid and valid[B].
  ``valid`` detects mechanics failure/empty envelope, NOT all caller input
  validation. Caller must exclude nonfinite state, predictions and matrices.

``recovery(primary_dict) -> recovery_dict``
  36 variables, 104 canonical rows. Same inputs/objectives plus 12 joint slack
  variables. May be built for an empty primary envelope if mechanics are valid.

``candidate(z, problem_dict, contact_probability) -> (physical_x, primal_feasible, max_normalized_violation)``
  z is the solver's scaled primal. Projects total torque into the hard current
  box and zeroes swing force components, then rechecks all canonical rows.
  Uses rollout .001 feasibility tolerance and rejects nonfinite raw outputs.
  This is a **primal feasibility check**, not KKT/duality optimality certification
  or an assertion about measured motion. Require a returned solver candidate;
  never accept a fabricated result after an exception. Apply gap policy
  separately when using the stricter PPO profile.

``fallback(tau_nom[B,12], limits) -> bounded_torque[B,12]``
  Historical deterministic fallback: replace NaN/+Inf/-Inf with zero, then
  clip magnitude. No joint/contact guarantee and no rate bound in this config.

``feedforward_command(selected_total, physical_pd, motor, feedforward_weight)``
  Returns ``(scaled_ff_command, available_mask)``; command units match
  ``nominal``'s scaled_ff, NOT raw actor units. Formula:
  ``ff=(selected_total-physical_pd)/(motor*feedforward_weight)``.
  If a consumer needs raw actor units, divide by10; do not reclamp to actor
  +/-50 or apply scaling a second time. Subsequent actuator limits must be
  modeled/rechecked. The desired position remains nominal, not QP-corrected.

Observation and coordinate contract
-----------------------------------

* [0:3]: body velocity commands [vx,vy,wz] scaled [1,1,.25].
* [3:6]: unit gravity direction expressed in body frame (upright [0,0,-1]).
* [6:9]: body angular velocity * .25.
* [9:21]: q-default_q (radians, scale1).
* [21:33]: joint velocity * .05.
* [33:45], [45:57]: previous issued clipped position/FF policy actions.

Explicit outputs: body velocity[0:3] scaled1; contact probabilities[3:7] =
.01+.98*sigmoid(logits); raw foot clearances[7:11] metres; terrain-relative
torso height[11] metres. No second sigmoid or clearance output clamp.
Height label is mean(base_z - sampled terrain heights), not height error.

Joint ordering: FR, FL, RR, RL; within each leg hip, thigh, calf. Foot ordering
FR_foot, FL_foot, RR_foot, RL_foot. Default pose is
[-.1,.8,-1.5, .1,.8,-1.5, -.1,.8,-1.5, .1,.8,-1.5]. Exact backend-realized
limits saved with the run are exported in the manifest and cross-checked
against the named URDF/config; velocity limits [30.1,30.1,15.7] repeat per leg.
Use named mappings for hardware, MuJoCo, and Pinocchio: URDF traversal order
is NOT a valid joint-order assumption. Do not use the reversed calf pair in
the legacy reset-only ``leg_joint_limits`` config as ordered hard bounds.

BARD/Pinocchio generalized velocity/acceleration ordering is body-frame base
linear, body-frame base angular, then canonical joints. Configuration is world
position, body-to-world xyzw quaternion, canonical joints. J maps this velocity
to world foot linear velocity; Jb maps it to world base linear/angular velocity
at the chosen base-link origin. M and h use the matching canonical coordinates.
Generalized base linear acceleration is the derivative of body velocity, not
world classical acceleration. World acceleration includes omega x v transport.
Jdot*v must use the same representation. See archived BARD implementation for
joint permutation and classical acceleration-bias construction.

Wrench order [Fx,Fy,Fz,Mx,My,Mz] is about that same base origin. Changing wrench
origin requires the appropriate r x F moment exactly once. Include gravity in
h and external wrench through Jb^T W once; do not count estimated payload load
again as extra nominal model mass. Dynamics use the nominal deployment robot
model, not simulator-only randomized inertial truth. Archive URDF is supplied
for porting; hardware-identification changes require separate validation.
URDF visual mesh assets are not included; the kinematic/inertial URDF is.

Historical QP: complete formulation
-----------------------------------

Physical variables x=[tau(12); tilde_f(12)]. Stance m_i=1[p_i>=.5]; physical
force f=D_m tilde_f, where D_m repeats m across each foot's XYZ. Swing latent
force coordinates have positive tracking curvature toward zero; dynamics
columns are masked and returned physical swing force is set exactly to zero.
The floating-base acceleration is eliminated by a batched linear solve::

    M a = S^T tau + J^T D_m tilde_f + Jb^T W - h
    A = solve(M, [S^T, J^T D_m])
    b = solve(M, Jb^T W - h)
    a = A x + b

No explicit inverse, dynamics equalities, acceleration variables, temporal
proximal objective or independent acceleration cap are present.

Minimize the sum below (squared Euclidean norms, no hidden one-half):

* 20 ||(tau - bounded_nominal)/40 Nm||^2.
* 1 ||(tilde_f - D_m raw_predicted_world_GRF)/250 N||^2.
* .25 ||D_m(J a + Jdot*v)/50 m/s^2||^2, soft stance acceleration.
* .5 ||(R_yaw_horizontal Jb_angular a - alpha_tilt_des)/20 rad/s^2||^2.
* .4 ||(Jb_linear_z a + (omega_W x vhat_W)_z - az_des)/20 m/s^2||^2.
* 16 ||([vx_B,vy_B] + .020*a[0:2] - command_xy)/1 m/s||^2.
* 6 ||(wz_B + .020*a[5] - command_wz)/1 rad/s||^2.

Let up=R_WB*[0,0,1], tilt_world=[0,0,1] x up and R_yaw_horizontal be the
first two rows of world-to-yaw rotation. Then
alpha_tilt_des=-20*R_yaw_horizontal*tilt_world -5*R_yaw_horizontal*omega_W.
This is physical angular acceleration in frozen yaw axes, not Euler-angle
second derivatives. Height target az_des=20*(.38-hhat)-5*vhat_W_z. Terrain
reference is frozen per solve, so damping is an approximation to relative
height damping on changing terrain. Inner velocity horizon .020 is a
constant-derivative extrapolation, not four integrated physics steps.

For each weighted affine residual Cx+e, assemble Qphys+=2w C^T C and
pphys+=2w C^T e. Substitute x=Dz, D=diag([40]*12+[250]*12):
Q=D Qphys D + 1e-7 I (symmetrized), p=D pphys. Solve
min .5 z^T Q z + p^T z. SPD ridge is in scaled solver coordinates.

Hard constraints, all with physics dt=.005, beta=1 (semi-implicit endpoint):

* Absolute total torque bounds -T <= tau <= T.
* v_next=v+dt*a_j; q_next=q+dt*v+dt^2*a_j. Intersect position/velocity bounds
  into lower=max((-vmax-v)/dt,(qmin-q-dt*v)/dt^2),
  upper=min((vmax-v)/dt,(qmax-q-dt*v)/dt^2), then lower<=A_j x+b_j<=upper.
* Each stance foot: fz>=0, |fx|<=.6*fz, |fy|<=.6*fz. These use world z as
  normal, not a learned/sloped terrain normal. Inactive swing rows become 0<=1.
* Hard endpoint torque: tau_end=tau - kp_eff*dt*v
  -(kp_eff*dt^2+kd_eff*dt)*a_j, with -T<=tau_end<=T.

Endpoint torque uses the same one-step constant-acceleration approximation.
It is NOT the later two-internal-step PD-aware formulation. Position/velocity
are constrained only at the predicted endpoint; neither initial joint
feasibility nor continuous-time safety is guaranteed. Torque-rate constraints
are disabled (weight0); the stored 2000 Nm/s value has no effect here.

Canonical G/h row layout (92 rows, no equalities): torque upper0:12,
torque lower12:24, joint acceleration-envelope upper24:36, lower36:48,
friction48:68 (5 per foot), endpoint torque upper68:80, lower80:92.
For each scaled row g=Gphys*D, normalize by max(||g||_2,|hphys|,1).
cuPIQP native packing replaces first24 rows with coordinate bounds; general
matrix is G[:,24:] (68 rows). Do not discard endpoint torque rows.

Recovery and acceptance
------------------------

Primary failures use joint-slack recovery x=[tau,tilde_f,s], s>=0 in rad/s^2:
lower-s <= a_j <= upper+s. Cost adds 200||s/100 rad/s^2||^2. There are
36 variables and 104 canonical inequalities; no torque-rate slack. Current
and endpoint torque bounds and friction remain hard. In scaled coordinates
the slack Hessian is (400+1e-7)I. Native packing uses G[:,24:-12], 68 general
rows, plus torque bounds and slack lower bounds0. An empty joint envelope can
recover; invalid mechanics cannot. Soft recovery acceptance does not certify
original joint limits. On failure, fall back to sanitized magnitude-clipped
nominal torque; never label fallback joint/contact certified.

Historical rollout acceptance checks finite raw solver outputs and canonical
normalized primal residual <=.001 AFTER torque projection and swing zeroing.
Rollout gap policy is ``report``: it does not enforce optimality/KKT or require
solver SUCCESS status beyond availability of a usable returned candidate.
PPO additionally requires finite absolute/relative gaps and
(absolute_gap<=.001 OR relative_gap<=.001). Missing required gaps reject.
The bundle's ``candidate`` exports rollout primal checks only. External
backend exceptions must enter recovery/fallback. Extra hardware fail-closed
requirements are advisable but are explicit changes to historical acceptance.

cuPIQP profiles: dense, float32/auto, CUDA graphs disabled; rollout eps_abs=
eps_rel=1e-4, max_iter20; PPO eps_abs=eps_rel=1e-4, max_iter30; feasibility.001
for both; absolute/relative gap thresholds.001. Solver-internal regularization
is distinct from the objective ridge and must not be removed. Full settings,
including cache/chunk parameters and legacy inactive keys, are in the manifest.
Dataclass defaults are resolved explicitly there, alongside the saved overrides
and rollout/PPO profiles. Original installed dependency versions and unmodified
cuPIQP internal defaults were not archived; current export-environment versions
are labeled separately and must not be represented as the original binaries.

Training schedule and learning context
--------------------------------------

QP warmup200 updates. Correction ramp starts at activation+200 (absolute400),
duration1000; final alpha1 at1400. Objective performance gate starts1400 with
EMA alpha.05, window200, quantile.9, recovery ratio.9, min_tracking.5,
50 consecutive recovery iterations, progress step.05, minimum step interval10.
Stance initial=.25/final=.25; torso attitude/height initial defaults to25%
of final (.125/.1), final .5/.4. Checkpoint progress=1, final inner weights are
embedded in the scripted assembly. No curriculum runs on hardware.

Outer learning coefficients are not QP objective weights: lambda_projection=.1,
fixed outer stance coefficient=.1, actor planar/yaw loss weights=.1 each with
independent outer horizon .020. Recovery projection multiplier=.5, joint-slack
outer weight1 (rate-slack setting inactive). Full saved PPO/auxiliary config
and source are archived; inference executes no losses, optimizers or PCGrad.
Every-substep deployment and deterministic latent inference change execution
relative to sampled stochastic training; do not claim closed-loop parity.

Minimal Python usage
---------------------

The solver/dynamics below are intentionally external interfaces, not included
placeholder implementations::

    model = torch.jit.load('hardpact_oct07_20000.pt').eval()
    # At policy rate, after reset history=zeros(B,570), previous_action=zeros(B,24):
    obs, entry = model.observation(command, gravity_B, omega_B, q, dq, previous_action)
    history = model.append_history(history, entry)
    raw_action, z, explicit = model(obs, history)
    issued = raw_action.clamp(-50, 50)
    selected_action = delay_queue.select(issued)  # caller's actual delay semantics
    qd, ff, pd, request, nominal = model.nominal(selected_action, q, dq, kp, kd,
        motor, feedback_weight, feedforward_weight, torque_limits)
    grf_yaw, raw_wrench, wrench_yaw = model.physics(z, explicit, nominal)
    previous_action = issued
    for k in range(4):
        # Refresh state, nominal PD/torque and nominal-model mechanics here.
        grf_world, wrench_world = model.world_inputs(live_quat, grf_yaw, wrench_yaw)
        # Populate ALL fields listed in primary() API, with live dt=.005.
        primary = model.primary(data)
        # Solve valid rows; never put padding in command/acceptance statistics.
        zstar = solver(primary['Q'], primary['p'], primary['G'][:,24:],
            primary['h'][:,24:], primary['native_lower'], primary['native_upper'])
        x, ok, residual = model.candidate(zstar, primary, explicit[:,3:7])
        # Rejected rows: recovery(primary), its G[:,24:-12], then candidate().
        # Exceptions/failed recovery: fallback(nominal, torque_limits).
        # Selected total torque -> feedforward_command(selected, fresh_pd, motor, ff_weight).
        # Send unchanged qd + reconstructed FF, record actually applied total Nm.

In C++ use ``module.forward({obs, history}).toTuple()`` and
``module.get_method("primary")({data})`` with a
``c10::Dict<std::string, torch::Tensor>``. Returned dictionaries use the same
keys; the selected solver consumes row-major tensors only if its own API
allows them. Check matrix layout/precision explicitly. ``golden_vectors.npz``
is an independent porting fixture; LibTorch can instead load the script and
compare generated outputs. Standalone loading requires compatible LibTorch
operators, including ``linalg_solve_ex``; pin the tested PyTorch build from
``validation.json`` and validate the chosen C++ toolchain separately.

Reproduce checks
-----------------

From the repository root::

    /home/oyoungquist/.conda/envs/lr_lab_cupiqp/bin/python deployments/oct07_20000/export.py
    /home/oyoungquist/.conda/envs/lr_lab_cupiqp/bin/python -I deployments/oct07_20000/smoke.py

The first command needs the historical Git objects, saved run and Python
reference dependencies. The second uses only this bundle and PyTorch, on CPU.
It includes four fresh assembly calls, NOT four numerical solves or a robot
rollout. Before hardware: verify named mechanics against Pinocchio, actuator
gain/latency calibration, solver parity on captured real states, failure
handling/watchdog deadlines, then closed-loop MuJoCo and constrained hardware
tests with independent emergency stop and physical safety supervision.
