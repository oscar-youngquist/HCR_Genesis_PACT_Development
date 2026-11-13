MULE dual-policy learning (implementation Stage 2)
================================================

This extends the Stage-1 model and reproduction assumptions in
``mule_stage1.rst``. No physical task, payload scheduler, reward port,
registration, launcher, or liquid fine-tuning is added. Synthetic rewards
and dynamics are tests only, not MULE's production objective.

Locations and use
-----------------

Configuration, normalization and schemas remain under
``legged_gym/envs/go2/go2_mule/``. Resolve shared settings with
``resolve_go2_mule_configs()`` and supply a ``Go2MULE`` adapter::

    from rsl_rl.runners.mule_runner import MULERunner

    runner = MULERunner(env, learning_config, device="cpu", log_dir=output)
    runner.learn(2)                         # two ADDITIONAL completed updates
    runner.transition_to_joint_adaptation() # explicit, clean-boundary, idempotent
    runner.learn(2)
    runner.save(checkpoint)
    runner.load(checkpoint, resume_mode="fresh")
    runner.learn()                          # remaining ABSOLUTE max_iterations

Logging follows the shared HardPACT-style contract in ``br_mule_logging.rst``;
keep it in subsequent physical integration stages. A writer is created when a
log directory is supplied, unless a caller-provided writer is used. ``learn``
returns internal metrics while ``last_logged_metrics`` contains published tags.
Automatic checkpoints
honor save_interval and save the final completed update. Log location, save
interval and maximum budget may change on resume. The saved phase wins over
the configured initial phase. The optional phase_transition_hook(iteration, env)
can request joint adaptation at a clean boundary; it cannot reverse the phase.
There is no implicit payload/phase coupling or new physical-stage clock.

Learning semantics
------------------

Each actor/critic has its own Adam, reward/value/return stream, normalized
advantages (population standard deviation), clipped PPO surrogate, value loss,
entropy and analytical old-to-new diagonal-Gaussian KL. Raw samples, NOT summed,
clipped or delayed commands, determine likelihoods. PPO uses owned rollout
context without re-encoding history or sampling a latent. Both separate PPO
updates complete all epochs before auxiliary context training starts.

Context remains auxiliary-only: physical velocity MSE, next-observation MSE
and beta-weighted KL retain Stage-1 reductions. Invalid/cross-episode/nonfinite
rows are removed before arithmetic. Empty minibatches skip backward AND Adam
step. Nominal-only adaptive correction/exploration is exactly zero; its
objective, optimizer, std projection and scheduler never run. Joint adaptation
trains BOTH pairs. Stored actor std parameters are projected to configured
bounds after each active optimizer step; forward clamping is retained.

Fixed LR stays fixed. Adaptive LR independently follows shared PPO's KL rule:
KL > 2*desired_kl divides LR by 1.5 (floor 1e-5); 0 < KL < desired_kl/2
multiplies LR by 1.5 (ceiling 1e-2). Adaptive entropy uses the existing
method-neutral performance-gap helper with inherited bounds/thresholds and
temperature. Full finite rollout performance is averaged and sets the NEXT
update's coefficient for active owners only. Missing evidence holds it.
These scheduling/ownership choices are experimental inheritance, not claims
about unavailable MULE author training code.

Rolling-ball adapter interface
-----------------------------

* ``observe()`` is pure: no RNG, history, reset or physics advancement. Supply
  packed 45-D deployable observations, declared critic fields, clean physical
  velocity labels, torque/leg-kinematics-only force proxies, episode IDs,
  supervision validity and a monotonically increasing integer control timestamp.
  Seed adapter-owned randomness from the resolved seed. Simulator contact
  forces/payload truth must not enter the deployable force interface.
* ``step(command)`` advances one control timestamp and supplies TWO already
  time-scaled rewards; the runner does not rescale them. Action scaling occurs
  once, on the combined issued action. Report actual delayed target, original
  source transition ID and actual applied torque separately. Delay IDs NEVER
  remap PPO rows. Previous action remains the clipped combined issued action
  before delay/scaling, zero on reset, as defined by the Stage-1 schema.
* Auto-reset only done rows; only their episode IDs change. Supply owned
  pre-reset terminal snapshots at t+1 and a validity mask. Physical failure
  overrides simultaneous timeout. Each critic bootstraps its own terminal
  value on timeout; failures bootstrap zero. GAE stops at either boundary.
  Missing timeout snapshots are errors, not silent zero values.
* Auxiliary targets pair history(t), velocity(t) and observation(t+1).
  Valid same-episode terminals are allowed, including physical terminals.
  Missing/invalid terminals are masked, never replaced by reset observations.
  Histories append once before action sampling and clear only reset rows.
* Optional begin_iteration(iteration)/end_iteration(iteration, performance)
  hooks connect future shared curricula once per completed rollout/update.
  Performance contains unscaled lin_vel_tracking, ang_vel_tracking and
  terrain_level, not episode rewards. Physical scheduling remains deferred.

Checkpoint contract
-------------------

Model contract v2 separates structural schema/history/normalization/action/
architecture compatibility from learning/runtime configuration. Known Stage-1
v1 model metadata is normalized explicitly; unknown versions are rejected.
Full saves include all three optimizers, schedules, phase, iteration and
interaction counts, resolved configs, Python/NumPy/Torch RNG, history/current
observation, episode buffers and available adapter curriculum state.
Compatibility and model/optimizer/RNG states are validated before live mutation.
Pending actions or unconsumed rollouts reject phase changes, saves and loads.

``exact`` requires the same device and a complete restorable adapter declaring
supports_exact_resume and state_dict/validate_state_dict/load_state_dict.
Only the synthetic adapter has been validated for exact continuation.
``fresh`` restores learning and separately validated curriculum state, then
resets physics and clears episode/history buffers. An adapter exporting
curriculum_state_dict must also supply validate_curriculum_state_dict and
load_curriculum_state_dict. Exact PhysX continuation is not claimed.
Only load trusted local checkpoint files.

Validation
----------

From the rescaled_qp_exps worktree::

    SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 \
      /home/oyoungquist/.conda/envs/lr_lab_cupiqp/bin/python -m pytest \
      -q -p no:cacheprovider tests/test_mule_stage1.py \
      tests/test_mule_learning.py tests/test_beyond_robustness*.py

For only the bounded synthetic learning smoke, select
``tests/test_mule_learning.py::test_bounded_nominal_transition_joint_save_resume_smoke``:
two nominal updates, explicit transition, two joint updates, checkpoint/exact
resume, one more joint update. No production or GPU training is exercised.

Remaining work: physical rolling balls, deployable force-observer validation,
production rewards, GPU throughput, registration and launchers.
