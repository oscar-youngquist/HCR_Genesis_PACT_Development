MULE rolling-payload integration
===============================

This stage supplies ``IsaacLabMULEEnv(Go2MULE)`` and
``IsaacLabSimulatorMULE``. Production registration and commands now live in
``mule_production.rst``. Models,
native context losses, dual-PPO ownership, rollout likelihoods and phase
transitions retain their ownership. Stage 4 adds production rewards and shared
curriculum orchestration (see ``mule_rewards_curriculum.rst``). Production
integration is now available; liquid fine-tuning remains deferred.

Evidence and choices
--------------------

Source: https://arxiv.org/html/2505.00488v1 . Paper choices: four dynamic balls,
250 g simulation tray, initial individual mass U[0,1] kg, and mass changes every
four seconds (paper upper bound 2.5 kg). The user explicitly overrides the
periodic maximum to **2.0 kg per ball**. Zero is replaced by a documented 0.01 kg
numerical floor. Nominal mode creates no sphere bodies/colliders; the physical
250 g tray remains. It never creates zero-mass dynamic bodies.

Reproduction assumptions: tray 0.50 x 0.34 m, 16 mm floor, 45 mm walls, 110 mm
torso-relative floor-center height; radii 35--45 mm; multiply-combined friction
(tray 1.0, ball U[0.15,0.6]); max-combined restitution U[0,0.1]. These are explicit
configuration choices, not verified author geometry/material settings.

The tray is an articulation link physically fixed to the torso, not a pose
follower. Its compound-body inertia includes wall masses and parallel-axis
offsets. Its mass is separate from randomized torso mass. Balls are independent
rigid bodies with I=2/5*m*r^2 and real ball--tray/ball--ball collisions. No box
physics, scripted payload forces or HardPACT learning components are involved.

``resolve_go2_mule()`` copies allowlisted live shared terrain, commands, PD,
timing, noise, robot randomization and physical limits. It retains MULE's
existing selective optimization factory/schema. Robot reset distributions are
the shared +/-0.3 rad joint offsets, origin/terrain placement and +/-0.5 root
twists. Persistent torso wrench injection is disabled, not replaced by payload
truth in actors. Shared pushes remain; nominal-mode push magnitudes are 1% of
their effective shared bounds. The existing shared disturbance clock now drives
rolling-load severity; there is no second progression clock.

Lifetimes and timestamps
------------------------

* Geometry: seeded size bank (eight four-ball combinations), env-id modulo
  bank size, before physics initialization; never runtime USD scaling. Clones
  are separated before PhysX initialization to avoid coincident broadphase.
* Mass: eligible reset sampling and separately timed four-second resampling;
  ``set_payload_severity(p)`` interpolates light [0.01,0.05] kg to the configured
  reset/periodic ranges. Changed severity takes effect on the next reset;
  periodic samples use each environment's realized severity. Timers are per environment and
  quantized to control intervals. Interval zero disables periodic changes.
* Friction/restitution and initial placement/twist: shared eligible-reset samples.
  Changed severity forces eligibility at the next reset; otherwise the configured
  per-environment reset cadence retains samples. Periodic mass changes remain
  independent of that cadence. Friction interpolates explicit light/full endpoints.
  Four disjoint spawn cells guarantee non-overlap and floor/wall clearance.
* Mass changes preserve pose and linear/angular velocity, not momentum; this is
  a discontinuous simulator-property experiment. Inertia and mass labels are
  updated/read back together through supported CPU-backed PhysX property APIs.
  Four batched body-property calls synchronize; there are no per-env API loops.
* Robot randomization retains shared eligible-reset cadence. Reset of one robot
  affects only its balls, delay/contact/noise buffers, timers and episode ID.
  Terrain-boundary translation relocates every associated ball equally.

45-D actors use the existing named MULE schema: body angular velocity, gravity,
commands, absolute joint angles, joint velocity, previous clipped combined
issued action. The combined action is scaled exactly once into an absolute
position target. The simulator consumes that target directly, recalculating PD
with fresh state at each physics substep. Delay queues retain target/source ID;
the two raw policy samples remain exclusively PPO likelihood variables.

Actor noise is drawn once at each step/reset, scaled by the schema's physical
units and cached. ``observe()`` only returns owned cached snapshots. Critic and
velocity supervision remain noise-free. Terminal snapshots precede automatic
resets; failures override simultaneous timeouts. Any ball entirely beyond the
expanded tray footprint or below its floor tolerance is load loss. Upward
motion inside the footprint is not by itself loss.

Deployable foot-force proxies use only last-substep applied joint torques and
named PhysX foot-COM Jacobian joint columns, rotated world to torso. Floating
base columns are omitted from the leg inverse; the positive tau=J.T*f proxy is
not an inertially compensated environment GRF. Simulator contacts and payload
state never enter this actor interface. The GPU check independently compares
the full Jacobian's predicted foot velocity with simulator COM velocity.

Critic disturbance means actual ball-on-robot contact force (normal+tangential)
in torso axes, not m*g. Each PhysX filter names exactly one link per environment;
the negative ball-side sum gives robot reaction. Ground-pair filtering excludes
support contact without disabling real robot/tray--terrain collisions.
``physical_labels()`` separately exposes realized robot/tray properties,
individual/total ball masses, relative motion, contacts and torques for future
reward work. Body-relative motion includes the rotating-frame omega-cross-r
term using velocities at the torso link origin and centered sphere COM.

Resume and interfaces
---------------------

The existing three-optimizer runner saves atomically, validates physical and
learning contracts before mutation, and restores learning phase, parameters,
moments, global counters and schedules. Adapter hooks retain payload geometry,
realized properties, sampler RNG, mass timers, severity, robot cadence,
terrain/command/domain progress, delay and noise RNG. A fresh reset reinstalls
samples without advancing the saved reset cadence, clears episode/history and
delay queues, and rebuilds observations. This is NOT exact PhysX continuation.

Fresh CUDA resume explicitly maps the saved training-device RNG to the target
device; identical visible-GPU counts are not required. Unrelated device streams
are not restored. Exact test-environment resume keeps strict device/topology
checks. CPU/CUDA or GPU-architecture changes are not bitwise portable. Liquid
transfer is explicitly rejected until a compatible adapter exists.

The default is now the resolved MULE production reward merge. Explicit test
callbacks still require a versioned ``reward_contract`` and return two finite
[N] rewards, already weighted/dt-scaled. The callback receives the pre-reset
env/observation/failure/timeout. Smoke scripts use production rewards unless
``--test-reward`` is explicitly selected. No BR native rewards are inherited.

Bounded validation commands
---------------------------

From this worktree, with the existing Isaac Lab Conda environment::

  conda activate lr_lab_cupiqp
  export SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
  python -m pytest -q tests/test_mule_payload.py tests/test_mule_stage1.py tests/test_mule_learning.py tests/test_br_mule_logging.py
  python -m pytest -q tests/test_beyond_robustness*.py
  python tests/smoke_mule_payload.py --gpu cuda:0 --num-envs 4 --steps 4 --physical-checks --output-dir /tmp/mule_physics
  python tests/smoke_mule_payload.py --gpu cuda:0 --num-envs 64 --steps 8 --output-dir /tmp/mule_64
  python tests/smoke_mule_payload.py --gpu cuda:0 --num-envs 4 --steps 4 --mode nominal --output-dir /tmp/mule_nominal

Both learning phases and fresh save/load are exercised in each smoke. Reports
include versions, per-update collection/update wall times and peak **PyTorch
allocator** bytes (not total PhysX/Kit VRAM). Total elapsed includes checkpoint
I/O and fresh resets; startup is excluded. No speedup or 4096-env throughput is
inferred from these short checks. See generated report.json for actual results.

Executed validation (2026-11-13)
-------------------------------

The combined MULE/BR suites above passed 137 tests. Actual cuda:0 smokes passed
for four rolling environments with physical checks, 64 rolling environments,
and four nominal/no-ball environments. Each performed one nominal-only and one
joint-adaptation update plus fresh checkpoints/resumes, with finite losses.
The test callback is NOT a production objective.

The physical check measured -5.872 N mean payload support reaction (four 0.15 kg
balls), zero tray-ground force during normal support, 0.0596 m rolling motion,
ball--ball impact response, and a 429.15 N genuine ground-impact peak. Named
Jacobian velocities, boundary translation, cross-env collision isolation,
terminal ownership and failure/timeout precedence passed.

64 envs x 8 steps: collection 0.624/0.427 s, updates 0.058/0.010 s for nominal/joint
phases; peak PyTorch allocation 119,772,160 bytes (114.22 MiB). Including save/load
and reset overhead, 1024 interactions took 1.607 s. This is a bounded startup-
affected measurement, not a speedup or production throughput claim.

Runtime: Isaac Lab 0.54.2, Isaac Sim 5.1.0.0, Torch 2.7.0+cu128, bundled Warp
1.8.2 (the installed warp-lang distribution is separately 1.17.0). Early runs
revealed and fixed premature Warp import and invalid all-links contact filters;
their apparent learning success was not used as physical-force validation.

Not executed: 4096-env physics/training, production-reward training, liquid
transfer, exact PhysX restoration, or a hardware run across changed GPU counts.
Training-GPU RNG remapping across visible-count changes is covered by a focused
CPU selection test; same-device fresh resume was exercised on actual CUDA.
