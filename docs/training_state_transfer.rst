Training-state checkpoints and cross-payload transfer
====================================================

Implemented on rescaled_qp_exps from 101e447b. The version-1 ``training_state``
envelope augments existing weight keys, so weight consumers and the existing
Pos-to-coupled bootstrap export keep working. It does not change losses,
feature selection, PCGrad, solver settings, or optimization phases.

Registered coverage
-------------------

The inventory is taken from ``legged_gym.envs`` and ``rsl_rl.runners``; RSS
position-only is NOT routed through the Pos pretraining runner.

============================= ====================== ====================== ========================
Task family                   Registered runner      Algorithm              State adapter
============================= ====================== ====================== ========================
go2_hard_pact + backend names PACTRunner             PPO_HardPACT           hard_pact
8 legacy named ablations      PACTRunner             PPO_HardPACT           hard_pact
10 RSS ablations (Isaac Lab)   PACTRunner             PPO_HardPACT           hard_pact
go2_hard_pact_pos + aliases   PACTPosRunner          PPO_PACT_Pos           hard_pact_pos
go2_beyond_robustness_isaaclab BeyondRobustnessRunner PPOBeyondRobustness    beyond_robustness
go2_mule_isaaclab              MULERunner             PPOMULE                mule
============================= ====================== ====================== ========================

``PACTRunner`` resolves to ``OnPolicyRunnerPACT``; ``PACTPosRunner`` resolves
to ``OnPolicyRunnerPACTPos``. All eight legacy selectors (baseline, soft,
hard, full, stopgrad, soft_penalty, inverse, rollout) retain their features.
All ten RSS selectors (pos_ppo, coupled_ppo, guided_no_qp, qp_no_post,
qp_imitation, full, no_fk, no_ff, no_channels, no_qp_wrench) retain their
features, including disabled representation PINNs and the 12-D pos_ppo policy.
HardPACT aliases cover Genesis/Isaac Gym/Isaac Lab; Pos aliases cover
Genesis/Isaac Lab. This is code-path coverage, not GPU/backend certification.
Unrelated legacy PACT/Go1 tasks retain their existing checkpoint behavior.
The inspected registry contains 38 PACTRunner HardPACT tasks, three
PACTPosRunner HardPACTPos tasks and resolved BR and MULE tasks.

Three operations
----------------

* ``resume`` requires the same method/features, semantic contract, shared
  physical settings and environment contract. Isaac Lab starts fresh physical
  episodes. BR's existing ``resume_mode='exact'`` remains available only for
  fully restorable environments (tested on the synthetic fixture, NOT PhysX).
* ``stage_transfer`` requires the SAME method, action mode, active objectives,
  label meanings, input ordering/scaling/history and controller timing. Only
  the payload environment can change, through a target adapter. It preserves
  weights, moments, global update/interactions, exploration, normalization,
  entropy/LR, compatible schedules and attained curriculum difficulty.
* ``weights_only`` restores strict network state, not learning progress or
  optimizer moments. Existing explicit Pos-to-HardPACT weight conversion is
  still a bootstrap, never a cross-method stage transfer.
  Playback may change environment count, terrain/command ranges and QP use;
  learned observation/head normalization and network/action semantics remain
  validated. Full training checkpoints are trusted pickle files, not untrusted
  model downloads; bootstrap readers explicitly support their RNG metadata.

The configured training budget is TOTAL for HardPACT, BR and MULE entrypoints.
``--additional-iterations`` explicitly adds updates to the loaded next-index;
it is mutually exclusive with an explicit total budget. Runner ``learn(n)``
continues to mean n additional updates. Periodic files retain existing names;
the saved next-index, not the filename, is authoritative. Final checkpoint
writes are atomic and happen even between periodic save boundaries. Final
saves set ``stage_end=True``; this denotes the end of the requested training
segment, not a claim that a physical curriculum has converged.

State and validation
--------------------

The envelope stores all registered model state_dicts (including learned std
and registered normalization buffers), three named optimizers, ordered
parameter names/shapes/dtypes per optimizer group, PCGrad configuration,
adaptive entropy/LR, bootstrap state, PINN-update count and effective weights,
VAE/actor warmups through absolute iteration and configuration, QP curriculum,
Python/NumPy/Torch CPU/all initialized CUDA/local-generator RNG state, global
counters, stage/phase, source Git revision, parent path/hash, and source
environment/training/semantic contracts. Attached scheduler/scaler objects
are captured when present; current HardPACT BF16 autocast uses no GradScaler.
There are no target/EMA networks in the current covered models. Additional
registered module owners are captured explicitly if present.

HardPACT portable progress includes domain-randomization gates and generator,
command ranges/patience, terrain levels/types, reward coefficients already
resolved by the reward schedule, and optional controller tradeoff progress.
BR retains its existing box-property/noise sampling state on ordinary resume;
its portable export deliberately excludes these payload-specific fields.
HardPACT fresh reset resamples robot physical properties at retained ranges
and restarts reset-resampling cadence; it does not claim physical-sample or
trajectory continuation. Solver workspaces/factors, action-delay history,
pending transitions, rollout buffers, episode/contact/shaping histories are
not transferable. Owned states are cleared before collection resumes.

Transfer clears domain EMA/reference/history, command patience, QP reference/
EMA/recovery streak, and other source-domain promotion evidence. It retains
attained progress, schedule origins, and last-update/step accounting. New
evidence must satisfy existing gates. QP activation and correction warmup do
not restart. Target ``training_stage`` holds a separate origin iteration and
initial local severity; the future liquid adapter owns its local severity
rule, not a second locomotion/QP clock.

Semantic and named-ownership checks run before loading any live network or
optimizer. Equal dimensions do not imply compatibility. Runtime differences
are limited to budget/save/log/provenance settings. A positive optional
``--finetune-learning-rate`` is applied AFTER restore: all optimizer groups
are scaled by new_PPO_LR/saved_PPO_LR, preserving group/auxiliary LR ratios and
all moments; scheduler base/last LRs are scaled consistently. No loss changes
are permitted via this override. BR restores its saved active optimization
phase; only ordinary resume accepts explicit ``--transition-phase``.

``training_transfer_report.json`` records source/target configurations,
restored/reset/unavailable components, LR/stage overrides and parent SHA256.
New checkpoints retain this provenance. Source directories are protected:
continuations must write into a different run directory.

Legacy checkpoints
------------------

Old files cannot recover RNG/curricula/semantic ownership that was never
saved. HardPACT legacy files support strict network-only initialization with
``--load-operation weights_only`` (or explicitly acknowledged
``--allow-legacy-checkpoint``). The warning/report explicitly says that
optimizer moments and global progress were NOT restored; unnamed optimizer
IDs are not guessed or silently migrated. Existing bootstrap conversion
utilities remain intact. Old BR runner checkpoints can explicitly use
``--allow-legacy-checkpoint`` to restore their available algorithm/curriculum
state under existing BR validation, with missing common-envelope guarantees
reported. No legacy file is accepted for cross-payload stage transfer.

Target adapter contract (liquid is not implemented)
-------------------------------------------------

The target must implement all of:

* ``validate_stage_transfer(source_envelope)``: pure preflight, including
  semantic critic/supervision compatibility, supported progress version and
  terrain/robot/layout validity. Reject unsupported conversions before mutation.
* ``import_training_progress(progress, transfer=True)``: import ONLY shared
  difficulty/schedules; clear evidence (``clear_gate_evidence`` is shared).
  Never copy source box/ball properties, payload RNG or simulator state.
* ``initialize_transferred_payload(stage)``: initialize the target's own
  severity and payload-specific RNG at the next safe reset boundary.
* HardPACT: ``reset_training_episodes()`` and portable progress export/validation.
  BR: its existing ``reset()/observe()`` and semantic observation interface.

Current production adapters intentionally reject liquid transfers. The
synthetic target in ``tests/test_training_state_transfer.py`` exercises the
interface, not liquid physics. A future liquid task must keep compatible
critic inputs and supervised labels, or introduce an explicitly tested
conversion/version; padding/truncation/strict=False are not migrations.

Local and Unity commands
------------------------

From the rescaled_qp_exps worktree, a normal RSS full run automatically writes
periodic and final complete-state checkpoints::

  bash legged_gym/scripts/go2_hard_pact_rss_full_lab.sh \
    --headless --gpu cuda:0 --qp_solver cupiqp --seed 1

Same-method continuation, 1000 additional updates, new output directory::

  bash legged_gym/scripts/go2_hard_pact_rss_full_lab.sh \
    --headless --gpu cuda:0 --qp_solver cupiqp --seed 1 \
    --state-checkpoint /absolute/source/model_20000.pt --load-operation resume \
    --additional-iterations 1000 --checkpoint-log-root /absolute/new_runs

Weights-only uses the same wrapper with ``--load-operation weights_only``.
Use the task/config/seed that matches the saved contract; changing ablations
is rejected. Optional fine-tuning adds ``--finetune-learning-rate 0.0001``.

BR continuation::

  bash legged_gym/scripts/go2_beyond_robustness_lab.sh --headless --device cuda:0 \
    --checkpoint /absolute/source/model_20000.pt --load-operation resume \
    --additional-iterations 1000 --log-dir /absolute/new_br_run

Unity keeps scheduler visibility and logical cuda:0. In an allocated Unity
job, select the corresponding ``*_unity.sh`` script with the SAME checkpoint
and load arguments, omitting --gpu/--device. For example::

  bash containers/isaaclab/run.sh script go2_hard_pact_rss_full_unity.sh \
    --state-checkpoint /mounted/source/model_20000.pt --load-operation resume \
    --additional-iterations 1000 --checkpoint-log-root /mounted/new_runs

Current refusal/forwarding check, NOT an executable liquid experiment::

  bash legged_gym/scripts/go2_hard_pact_rss_full_lab.sh --dry-run \
    --state-checkpoint /absolute/source/model_20000.pt \
    --load-operation stage_transfer --target-stage liquid3 \
    --stage-local-initial 0.0 --additional-iterations 2

Remove --dry-run only after selecting a registered, tested target adapter.
The current HardPACT task rejects this before simulator construction. BR's
current production resolver similarly rejects liquid transfer. Programmatic
transfer for a supported adapter is::

  runner.load(source, operation='stage_transfer', target_stage='liquid3',
              stage_local_initial=0.0, learning_rate=1e-4)

Validation
----------

Bounded CPU learning/transfer regression command::

  SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 \
  /home/oyoungquist/.conda/envs/lr_lab_cupiqp/bin/python -m pytest -q -p no:cacheprovider \
    tests/test_training_state_transfer.py tests/test_beyond_robustness*.py \
    tests/test_br_mule_logging.py tests/test_hard_pact_rss.py \
    tests/test_hard_pact_two_stage_auxiliary.py tests/test_hard_pact_pos_auxiliary.py \
    tests/test_hard_pact_ppo_latent_replay.py

Tests cover real model/optimizer ownership, both BR phases, deterministic
synthetic continuation, shared progress/fresh histories, ablation identity,
semantic preflight, next-index/final save, legacy warnings, and shell forwarding.
Executed results (RTX 4090, cuda:0, lr_lab_cupiqp, November 13, 2026):

* Focused regression command above with CUDA accessible: **179 passed**, no
  skips, 19.83 s. Includes the cuPIQP matched-forward test. Warnings: existing
  hppfcl deprecation and the deliberately exercised legacy-partial warning.
* HardPACT RSS full/cuPIQP and HardPACTPos: two environments, four control
  steps/update, one update followed by save/fresh-reset/load/one more update.
  Exact network and all three optimizer state comparisons pass before the
  resumed update; final next-index=2, parameters finite.
* Full HardPACT collection/update: 1.410/0.242 s initially, 0.580/0.199 s after
  resume. Pos: 0.193/0.117 s initially, 0.101/0.015 s after resume. Peak PyTorch
  allocated memory was 266330112 / 313462272 bytes respectively, including
  two runner instances, excluding PhysX/Kit. These are tiny validation runs,
  NOT representative performance benchmarks or speedup evidence.
* BR registered production entrypoint: teacher_student AND
  student_reinforcement each ran one update, saved, then resumed in a separate
  process for one update. Both retained their phase and advanced 1->2 and
  8->16 interactions with finite losses. Resumed update wall times were
  0.333 / 0.325 s; peak Torch allocations 51354112 / 50789376 bytes (not total
  GPU usage). Models use the production merged rewards and physical boxes.
* ``bash -n`` passed for shared HardPACT/BR, RSS full local/Unity wrappers and
  container run.sh. Local/Unity dry-run forwarding is tested; no Unity job ran.

The GPU smoke found and fixed fresh-reset inference-tensor handling and
unallocated lazy replay buffers. Pos's first smoke attempt also exposed an
invalid smoke-only HardPACT constructor override; it was removed from Pos.
Isaac Lab emitted host file-watcher exhaustion warnings during startup, but
the bounded checks completed. No system limits or production settings changed.

Reproduce the physical checkpoint smokes from this worktree::

  conda activate lr_lab_cupiqp
  export SIMULATOR=isaaclab PYTHONPATH=. OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
  python tests/smoke_training_checkpoint.py --headless --gpu cuda:0 --num_envs 2 \
    --task go2_hard_pact_rss_full_isaaclab --qp_solver cupiqp
  python tests/smoke_training_checkpoint.py --headless --gpu cuda:0 --num_envs 2 \
    --task go2_hard_pact_pos_isaaclab

BR commands actually executed (choose new output paths for another run)::

  python legged_gym/scripts/train_beyond_robustness.py --headless --device cuda:0 \
    --smoke --num-envs 2 --max-iterations 1 --log-dir /tmp/br_checkpoint_teacher_gpu0
  python legged_gym/scripts/train_beyond_robustness.py --headless --device cuda:0 \
    --smoke --checkpoint /tmp/br_checkpoint_teacher_gpu0/model_1.pt \
    --log-dir /tmp/br_checkpoint_resumed_gpu0
  python legged_gym/scripts/train_beyond_robustness.py --headless --device cuda:0 \
    --smoke --num-envs 2 --max-iterations 1 --training-phase student_reinforcement \
    --log-dir /tmp/br_checkpoint_student_gpu0
  python legged_gym/scripts/train_beyond_robustness.py --headless --device cuda:0 \
    --smoke --checkpoint /tmp/br_checkpoint_student_gpu0/model_1.pt \
    --log-dir /tmp/br_checkpoint_student_resumed_gpu0

No full training, liquid physics, Genesis/Isaac Gym physical reset, or exhaustive
GPU execution of all ablations was performed. Cross-payload transfer remains a
synthetic adapter check; exact continuation is only a deterministic test-env
claim, never a PhysX claim.

MULE production adapter
-----------------------

MULE runner checkpoint v2 embeds the common version-1 envelope with both
actors/critics, CENet/decoder, exploration, named nominal/adaptive/auxiliary
optimizers, independent LR/entropy schedules and exact optimization phase.
Inactive adaptive schedule counters do not advance. Compact fresh-physics saves
omit rollout data, histories and episode rewards; fully restorable synthetic
fixtures separately retain their exact-continuation state.

Ordinary resume retains requested versus realized ball severity/activation,
pending reset generations, actual properties, timers, sampling RNG and curriculum
gates. Portable progress excludes all ball-specific state. Synthetic transfer
tests preserve learned state/difficulty, clear source-domain promotion evidence,
and invoke target ``initialize_transferred_payload(stage_dict)``. The rolling
adapter cannot initialize a liquid target and rejects such requests explicitly.
There is no actor reinitialization or combined-action likelihood migration.

Fresh MULE resume maps the saved training-device CUDA RNG to the new training
device; unrelated visible GPUs need not match. Device/architecture changes do
not promise bitwise continuation. Exact fixtures retain strict device/topology
checks. Legacy same-contract checkpoints require explicit opt-in and report
missing named-envelope/provenance evidence, never claim full continuation.
Incompatible clearance/reward v1 checkpoints are rejected even with that flag.
See ``mule_production.rst`` for commands and observed validation results.
