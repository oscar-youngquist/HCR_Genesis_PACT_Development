MULE production integration
===========================

``go2_mule_isaaclab`` resolves through ``resolve_go2_mule`` and constructs
``IsaacLabMULEEnv`` / ``MULERunner``. Production merged rewards and the existing
shared curriculum are defaults. The model, native CENet loss/routing
conventions, separate nominal/adaptive PPO streams and optimizer ownership are
unchanged. No HardPACT learning components or liquid physics are enabled.

Reward contract v2
------------------

Table I of https://arxiv.org/html/2505.00488v1 specifies clearance weight -0.01
for both streams but no identifiable clearance formula. The adopted positive
cost is::

    sum_feet(||v_foot_world_xy|| *
             (foot_z - mean(local_terrain_z) - foot_offset - target_clearance)**2)

The terrain reference uses the existing nine samples around each foot. Units
are m^3/s; the native coefficient and control-dt scaling apply once. Better
clearance at the same foot speed improves reward, and a common vertical
translation of feet/terrain leaves it invariant. Stationary feet have zero
speed-weighted cost. This is a documented reproduction assumption, not verified
author code. Shared HardPACT/BR helpers are unchanged. Version-1 MULE reward
checkpoints are rejected; no reward migration is supplied.

Configuration and stage semantics
--------------------------------

The default rolling scene creates four spheres before physics initialization.
At zero disturbance progress they are inactive/parked; the fixed physical tray
remains. Shared disturbance progress requests severity and payload activation;
only eligible resets realize the changes. Mass remains capped at 2 kg per ball
for periodic resampling, whose production cadence is four seconds. Geometry is
creation-time only. Requested versus realized severity, pending generations,
sampling properties/RNG/timers and curriculum gates survive ordinary resume.

Optimization defaults to ``nominal_only``. The existing
``auto_joint_adaptation`` hook requests the forward phase transition at a
completed-update boundary once rolling payload progress is positive. Both PPO
owners then train, with no copying/reinitialization. Physical progression does
not require this hook: ``--no-auto-joint-adaptation`` disables it independently.
``--optimization-phase joint_adaptation`` selects a fresh optimization phase.
``--payload-mode nominal`` creates no spheres and is a fixed scene choice, not
the recommended initial stage of sequential rolling training. Keep the default
``rolling`` scene to progress from inactive to active payloads without rebuilding.

Fresh fixed-load experiments can explicitly use
``--no-follow-disturbance-curriculum --payload-severity 0.5``. Those options do
not change optimization phase automatically when auto-transition is disabled.
Resume never silently overrides physical settings or the saved phase; use
``--transition-phase joint_adaptation`` for an explicit forward phase change.

Local commands (repository root)
-------------------------------

Use the installed Isaac Lab Conda environment. Physical GPU 1 is exposed as
logical cuda:0 below; change CUDA_VISIBLE_DEVICES to select another GPU::

    export MULE_CONDA_ENV=lr_lab_cupiqp
    export CUDA_VISIBLE_DEVICES=1
    bash legged_gym/scripts/go2_mule_lab.sh --device cuda:0 --headless --seed 1

    # Total budget; resume into a NEW directory, not the source run.
    bash legged_gym/scripts/go2_mule_lab.sh --device cuda:0 --headless \
      --checkpoint /path/source/model_2000.pt --max-iterations 4000 \
      --log-dir /path/resumed

    # Explicit additional budget and phase transition, preserving all moments.
    bash legged_gym/scripts/go2_mule_lab.sh --device cuda:0 --headless \
      --checkpoint /path/source/model_2000.pt --additional-iterations 100 \
      --transition-phase joint_adaptation --log-dir /path/joint

    # Strict compatible network initialization only; new optimizers/counters.
    bash legged_gym/scripts/go2_mule_lab.sh --device cuda:0 --headless \
      --checkpoint /path/source/model_2000.pt --load-operation weights_only \
      --max-iterations 100 --log-dir /path/initialized

    # No simulator or output writes. Wrapper dry-run prints argv; Python validates.
    bash legged_gym/scripts/go2_mule_lab.sh --dry-run --seed 1
    SIMULATOR=isaaclab PYTHONPATH=. python legged_gym/scripts/train_mule.py --dry-run

    # Bounded production-reward updates.
    bash legged_gym/scripts/go2_mule_lab.sh --device cuda:0 --headless \
      --smoke --no-auto-joint-adaptation --log-dir /tmp/mule_nominal
    bash legged_gym/scripts/go2_mule_lab.sh --device cuda:0 --headless \
      --smoke --optimization-phase joint_adaptation --log-dir /tmp/mule_joint
    bash legged_gym/scripts/go2_mule_lab.sh --device cuda:0 --headless \
      --smoke --smoke-curriculum --save-interval 1 --log-dir /tmp/mule_activation
    bash legged_gym/scripts/go2_mule_lab.sh --device cuda:0 --headless \
      --smoke --num-envs 4096 --max-iterations 1 --log-dir /tmp/mule_4096

Standard smoke uses four rollout steps, one epoch/minibatch, up to two updates.
At more than 64 environments it retains the production terrain bank and initial
level distribution, rather than crowding all robots onto the tiny smoke bank.
MULE resolves ``sim.physx.gpu_found_lost_pairs_capacity`` to at least 2^21,
rounded up to cover 1024 times the environment count (2^22 at 4096), with an
explicit factory override available. This is broadphase allocator capacity,
not a constraint, contact-tolerance or solver-iteration change.
``--smoke-curriculum`` is an explicitly test-local three-update case with fast
shared gates, 0.10-second episodes and a 0.02-second mass-resampling cadence.
It asserts reset activation, joint adaptation and observed mass changes. These
are not production defaults or evidence of long-horizon task performance.

Unity/container commands
------------------------

Use scheduler-assigned CUDA_VISIBLE_DEVICES; never overwrite it in the Unity
wrapper. Container Python is /opt/env/bin/python and the device is logical
cuda:0. With IMAGE, REPO and a unique RUN_DIR configured, and NVIDIA's EULA
accepted through OMNI_KIT_ACCEPT_EULA=YES::

    bash containers/isaaclab/run.sh script go2_mule_unity.sh --seed 1 --dry-run
    bash containers/isaaclab/run.sh script go2_mule_unity.sh --seed 1 \
      --log-dir /output/fresh
    bash containers/isaaclab/run.sh train go2_mule_isaaclab --seed 1 \
      --log-dir /output/fresh_direct
    bash containers/isaaclab/run.sh smoke go2_mule_isaaclab --log-dir /output/smoke

For resume, bind the source checkpoint beneath REPO (visible at /workspace/repo)
or RUN_DIR (visible at /output); forward the same checkpoint/load-operation/
budget options shown above. Use a new destination directory. Job Composer may
select ``go2_mule_unity.sh`` through the existing ``run.sh script`` workflow.
Host Git SHA/status are forwarded for linked worktrees whose administrative
Git directory is outside the container bind. No additional checkout is mounted.

Checkpoints and archives
------------------------

Periodic and final files use atomic replacement. ``next_iteration`` is
authoritative, budgets are total unless explicitly additional, and the final
file marks the end of the requested segment even off the periodic cadence.
Each run archives resolved physical/training/payload/reward/curriculum settings,
schema, CLI, Git/dirty status, dependency versions and source archive. Resume
adds source/target config and restored/reset/overridden state reports plus the
parent checkpoint SHA256. Original source-run directories are protected.

Fresh-physics resume restores all three optimizers, independent schedules,
networks, phase, RNG, counters and curriculum/sampling state, but clears physical
episodes and histories. It is not exact PhysX continuation. The saved training
GPU RNG maps to the selected target device, even with changed GPU visibility;
other hardware differences are not promised bitwise reproducible.
MULE saves NumPy MT19937 state as primitive values rather than NumPy-version-
specific pickled arrays, allowing local NumPy 2 / container NumPy 1 loading.
Older NumPy-2-array checkpoints require explicit serialization-only conversion
in their source runtime (preserve the source and change only both RNG array
representations using ``training_checkpoint.portable_numpy_rng``). This does not
migrate incompatible reward/observation contracts or invent missing state.

Future stage transfer uses the existing common runner API::

    runner.load(checkpoint, operation='stage_transfer', target_stage='liquid3',
                stage_local_initial=0.1, learning_rate=1e-4)

This requires a target adapter with pure compatibility validation,
``import_training_progress(progress, transfer=True)`` and
``initialize_transferred_payload(stage_dict)``. Portable export excludes ball
sampling state; transfer clears source promotion evidence but preserves attained
difficulty/global progress. Only a synthetic target is supported/tested here.
The production CLI rejects liquid transfer before constructing physics.

Validation
----------

Focused CPU command::

    SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python -m pytest -q \
      -p no:cacheprovider tests/test_mule_production.py \
      tests/test_mule_rewards_curriculum.py tests/test_mule_payload.py \
      tests/test_mule_stage1.py tests/test_mule_learning.py tests/test_br_mule_logging.py \
      tests/test_beyond_robustness.py tests/test_beyond_robustness_runner.py \
      tests/test_beyond_robustness_payload.py tests/test_beyond_robustness_inheritance.py \
      tests/test_beyond_robustness_rewards.py tests/test_beyond_robustness_stage8.py \
      tests/test_training_state_transfer.py

Observed results (2026-11-13, RTX 4090)
--------------------------------------

The focused command above passed **183 tests in 19.16 s**, including the final
portable-NumPy encoding (two existing dependency/legacy-handling warnings).
``bash -n`` passed for all three wrappers and ``containers/isaaclab/run.sh``;
local/Unity and direct-container dry-run checks passed without output creation.

Actual production entrypoint checks used Isaac Lab 0.54.2, Isaac Sim 5.1.0.0,
Torch 2.7.0+cu128. Local NumPy was 2.4.6; the actual Apptainer image used 1.26.0.
The initial nominal test used physical GPU 0; all subsequent checks used GPU 1
following the user's request. All listed learning losses were finite.

=============================== ===== ======= ============ ========== ==============
Check                           Envs  Updates Wall seconds Steps/s    Torch peak MB
=============================== ===== ======= ============ ========== ==============
Local nominal                   4     2       1.140        28.1       42.0
Local resume, explicit joint    4     1       0.483        33.1       59.0
Fast gates/reset activation     4     3       1.671        28.7       86.1
Inherited-terrain joint smoke   4096  1       3.099        5287.3     966.5
Actual container direct smoke   2     1       1.840        4.35       41.8
Actual Unity-wrapper resume     2     1       0.785        10.2       58.9
Local-to-container resume       4     1       1.067        15.0       59.1
=============================== ===== ======= ============ ========== ==============

MB means decimal bytes/1e6. Peaks are PyTorch allocator allocations only,
excluding PhysX/Kit and other processes. Timers exclude scene initialization
but include first-use learning/collection work (and periodic writes where
configured). These short, unmatched cases are functional checks, not speedup
comparisons or steady-state training benchmarks. The 4096 check retained
production terrain but used four rollout steps and one epoch/minibatch.

The accelerated check recorded eight periodic per-environment mass updates and
activation at reset. Checkpoint/resume preserved the optimization phase and
global interaction count; CPU tests also verify optimizer moments, dormant
adaptive schedules, pre-update likelihoods and synthetic exact continuation.
The physical regression with production rewards passed support transmission,
rolling/ball interaction, genuine ground contacts versus support filtering,
mass/inertia readback, partial resets, pure observations, named force-proxy
mapping, load-loss/failure-timeout precedence and terminal snapshot ownership.
It ran both learning phases and fresh checkpoint restores.

Bounded physical regression command::

    CUDA_VISIBLE_DEVICES=1 SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 \
      python tests/smoke_mule_payload.py --gpu cuda:0 --num-envs 4 --steps 4 \
      --physical-checks --output-dir /tmp/mule_production_physical_gpu1

Observed failures were not counted as passing evidence: GPU resume initially
exposed a shadowed checkpoint-state variable in CUDA RNG validation (fixed);
linked-worktree Git archiving failed inside the container (host identity
forwarding fixed); an accelerated smoke sampled a reset-cleared mass counter
(test reporting boundary corrected); two 4096 checks with the tiny terrain
bank warned about missing broadphase interactions (excluded). The subsequent
inherited-terrain check had no such warnings. Local-to-container NumPy-array
loading initially failed before physics; portable RNG encoding fixes new saves.
The serialization-only converted local test checkpoint then resumed in the
NumPy-1 container, preserving joint adaptation and advancing iteration 3 to 4
(48 to 64 interactions). No learned tensors or RNG values were changed by that
conversion; the source checkpoint was retained.

Artifacts are under ``/tmp/mule_production_nominal``,
``/tmp/mule_production_resume_fixed_gpu1``,
``/tmp/mule_production_curriculum_fixed_gpu1``,
``/tmp/mule_production_4096_terrain_gpu1``,
``/tmp/mule_production_physical_gpu1`` and
``/tmp/mule_container_production_fixed/{mule_smoke,mule_wrapper_resume}``.
The cross-runtime result is ``/tmp/mule_local_to_container_portable/resumed``.
These contain test checkpoints, not trained policies. No Unity scheduler job,
full-budget run, long-term performance validation or liquid physics was run.
Actual local Apptainer execution is not a claim of execution on Unity hardware.

Changed-file map
----------------

* Production: ``legged_gym/scripts/train_mule.py``, ``go2_mule.sh``,
  ``go2_mule_lab.sh``, ``go2_mule_unity.sh``; ``containers/isaaclab/run.sh``.
* Registry: ``legged_gym/envs/__init__.py``, ``legged_gym/utils/task_registry.py``.
* MULE: ``go2_mule/{mule_rewards,go2_mule,go2_mule_config,go2_mule_isaaclab}.py``;
  ``legged_gym/simulator/isaaclab_simulator_mule.py``.
* State: ``rsl_rl/runners/mule_runner.py``, ``rsl_rl/utils/training_checkpoint.py``.
* Tests: ``test_mule_production.py``, ``test_mule_rewards_curriculum.py``,
  ``test_mule_stage1.py``, ``smoke_mule_payload.py``.
* Docs: this file, ``mule_rewards_curriculum.rst``, ``mule_rolling_payload.rst``,
  ``training_state_transfer.rst``.
