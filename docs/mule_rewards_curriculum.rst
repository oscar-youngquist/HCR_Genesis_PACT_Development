MULE Stage 4: rewards and shared curriculum
=========================================

Code lives in ``legged_gym/envs/go2/go2_mule/``. The authoritative factory is
``resolve_go2_mule``; ``shared.mule_reward`` and the saved configuration manifest
contain every formula, unit, coefficient per stream, source, exclusion reason
and eligible curriculum bound. No new task registration or production launcher
is added. CENet, dual PPO likelihoods and the three optimizers are unchanged.

Reward contract
---------------

Source: https://arxiv.org/html/2505.00488v1 (Table I and GRF equation).
The published nominal/adaptive coefficients are respectively:

==================== ============ ============
Term                 Nominal      Adaptive
==================== ============ ============
Planar tracking      1            0
Yaw tracking         0.5          0
Vertical velocity    -2           -2
Roll/pitch rate      -0.05        -0.05
Orientation          -0.2         -0.2
Joint acceleration   -2.5e-7       -2.5e-7
Joint power          -2e-5         0
Height               -2           -2
Clearance            -0.01        -0.01
Action rate          -0.001       -0.01
Action smoothness    -0.01        -0.01
GRF                  0            2
==================== ============ ============

Where the paper does not define mathematics, the existing HardPACT/Go2PACT
formulation is used, per the user's instruction, not a guessed author formula.
Tracking is fixed ``exp(-4*error_squared)``: no dynamic sigma. Height is mean
torso-z minus terrain heights. Power uses actual last-substep executed torque.
Acceleration differences use the actual control dt. Clearance literally uses
HardPACT's exponential of world-z clearance error weighted by foot speed;
its combination with the negative native coefficient is unusual but retained,
not silently inverted. Slip also retains the shared square-of-summed-xy-speed
definition. These choices are not a claim of exact author-code reproduction.

GRF is ``.75*I(h>target) + .5*I(h<target)*I(sum_i ||f_i||_2 > m_total*9.81)``.
Equality earns neither branch. Forces are the existing signed torque/Jacobian
proxies in body axes, not simulator GRFs; all four proxies must be finite and
valid for the low-height force test. High-height reward does not require force
evidence. Articulation mass includes realized robot randomization and tray
once; only active ball masses are added. Parked positive-mass bodies are not
part of the supported load. Valid/invalid measurement counts are logged.

Native coefficients, including zeros, never change at the optimization-phase
switch and cannot be scheduled/overridden by the compatible-addition merge.
Compatible speed/torque limits, posture/standstill, position tracking,
slip/stumble/edge and support-geometry terms belong only to nominal locomotion.
Their inherited zero coefficients stay off. Orientation, clearance, power and
height-shaping overlaps are excluded; no channel penalties or aliases restore
adaptive direct tracking/power. The allowlist/exclusions are in the manifest.

Two independent raw normalized action histories drive each stream's rate and
second-difference penalties, before sum/clipping/delay/action scaling. They do
not replace either PPO sample. First one/two unavailable differences are masked
after reset. Rewards are evaluated at the clean post-push/pre-reset endpoint;
contacts refer to the last completed physics step. Base and foot velocity
caches are refreshed after a push without a second physics step. Per-stream
``reward_terms`` survive automatic reset, with episode logging following the
existing runner convention. Multiply coefficient by control dt exactly once;
never clamp the total. Only allowlisted additions use the shared cosine reward
curriculum. There is no reward-coefficient TensorBoard expansion.

Curriculum and activation
-------------------------

Default factory settings::

  resolve_go2_mule(
      curriculum_overrides=dict(
          follow_disturbance_curriculum=True,
          auto_joint_adaptation=True,
          nominal_push_scale=0.01),
      payload_overrides=dict(
          mode='rolling',
          light_friction_range=(0.6, 0.6),
          friction_range=(0.15, 0.6),
          reset_mass_range=(0., 1.),
          periodic_mass_range=(0., 2.),
          mass_change_interval_s=4.))

Shared domain-randomization warmup, EMA/recovery gate, phase ordering, increments
and interval are authoritative. Raw fixed-kernel planar tracking is averaged
over all finite environment/control samples, with counts, not episodic returns
or adaptive rewards. Missing evidence pauses advancement. Updates occur once
after completed PPO/auxiliary updates. Command patience uses this same evidence;
zero patience retains shared reset cadence. Terrain promotions retain the
shared absolute-iteration delay and per-reset promotion/demotion behavior.

At zero disturbance progress, dynamic balls have positive mass but are parked
at distinct depths below terrain with PhysX gravity disabled. They do not load
or collide with the robot, terrain or each other. No tiny-mass surrogate is used.
The tray remains. Fixed ``mode='nominal'`` instead creates no balls at all.
Positive progress requests activation at each eligible reset, restoring gravity
and a non-overlapping torso-relative spawn. Geometry remains creation-time only.
Inactive balls are excluded from loss detection and supported-mass accounting.
Requested progress and realized per-environment activation/severity are distinct.

Progress interpolates light/full reset and periodic masses, spawn jitter,
relative linear/angular velocities and explicit material endpoints. Friction
interpolation is a configured choice, not a claim that increasing friction is
harder. The 4-second mass clock samples at each environment's realized severity;
it never advances the curriculum. Changes preserve pose/velocity (not momentum)
and update inertia together. Positive mass floor remains 0.01 kg; the resampled
maximum remains 2.0 kg per ball. Reset sampling shares the robot's eligible-reset
cadence; a changed severity forces fresh samples at the next reset.

Default automatic optimization transition requests joint adaptation before the
next rollout after positive rolling progress is requested. It never switches
mid-update or reinitializes actors. Physical activation still waits for reset.
Set ``auto_joint_adaptation=False`` for independent explicit phase selection;
set ``follow_disturbance_curriculum=False`` to use explicit payload severity.
Neither setting creates a new ramp clock. Persistent artificial torso wrenches
remain disabled; shared pushes remain, scaled down in the nominal physical stage.

Resume
------

The version-2 physical/sampling contract and version-1 reward contract are saved
with the existing model/three-optimizer/phase checkpoint. They include reward
iteration, domain EMA/history/gates, command ranges/streak, terrain levels,
requested and realized severity/activation, geometry, sampler RNG and timers.
Fresh PhysX resume reinstalls saved samples without promoting curricula and
clears observation/delay/reward histories. Pending activation remains pending.
Previous Stage-3 reward/physical contracts are rejected explicitly: no silent
migration or claim of exact PhysX continuation. Liquid transfer remains unsupported.

Bounded validation (from this worktree)
-------------------------------------

::

  conda activate lr_lab_cupiqp
  export SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
  python -m pytest -q tests/test_mule_rewards_curriculum.py tests/test_mule_payload.py tests/test_mule_stage1.py tests/test_mule_learning.py tests/test_br_mule_logging.py tests/test_beyond_robustness*.py
  CUDA_VISIBLE_DEVICES=0 python tests/smoke_mule_payload.py --gpu cuda:0 --num-envs 4 --steps 4 --curriculum-checks --output-dir /tmp/mule_stage4_curriculum
  CUDA_VISIBLE_DEVICES=0 python tests/smoke_mule_payload.py --gpu cuda:0 --num-envs 4 --steps 4 --physical-checks --output-dir /tmp/mule_stage4_physical

The fast curriculum parameters are test-local overrides, not production changes.
Reports contain actual losses, collection/update times, resume checks, versions
and peak PyTorch allocator memory (not total PhysX/Kit VRAM). These tiny checks
do not establish 4096-environment throughput or locomotion performance.

Executed validation (2026-11-13)
-------------------------------

* CPU MULE/BR regression command above: **146 passed**, one pre-existing
  hppfcl deprecation warning (12.87 seconds).
* CUDA:0, four environments, four steps/update: production reward updates in
  both phases, automatic shared-gate transition, pending/reset activation and
  fresh resume passed. Three-update/checkpoint bundle: 1.728 s, 27.78 environment
  transitions/s including checkpoint operations; collection 0.264--0.409 s and
  update 0.0055--0.0441 s per iteration. Peak PyTorch allocation 110,246,400 bytes.
* Physical regression passed: support reaction -6.086 N, zero ground-filtered
  tray force during normal support, 0.0509 m rolling displacement, ball--ball
  interaction, actual terrain collision (429.15 N peak), reset/environment
  isolation, named Jacobian mapping and terminal ownership. Separate two-update
  production-reward/resume bundle: 1.306 s, peak PyTorch allocation 110,174,208 bytes.
* Runtime: Isaac Lab 0.54.2, Isaac Sim 5.1.0.0, Torch 2.7.0+cu128, bundled Warp
  runtime 1.8.2 (installed warp-lang distribution reports 1.17.0).

Artifacts: ``/tmp/mule_stage4_curriculum_final/report.json`` and
``/tmp/mule_stage4_physical/report.json``; associated model checkpoints and logs
are test artifacts, not production-trained policies. Full-scale throughput,
long-term task performance, liquid transfer and production launchers were not tested.
