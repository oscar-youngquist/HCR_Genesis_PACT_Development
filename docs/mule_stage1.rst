MULE implementation Stage 1
===========================

Baseline: ``rescaled_qp_exps`` at ``37bc639447b86f0de658e94fa00d8fd47d64b984``.
This is a standalone neural/contract implementation, not a trainable task yet.

Locations and use
-----------------

* ``legged_gym/envs/go2/go2_mule/go2_mule_config.py``: Go2 environment and
  learning settings; ``resolve_go2_mule_configs()`` resolves the live shared
  configuration, like the Beyond Robustness factory.
* ``legged_gym/envs/go2/go2_mule/go2_mule.py``: abstract ``Go2MULE`` environment
  boundary and owned observation/transition snapshots. No physical backend.
* ``legged_gym/envs/go2/go2_mule/mule_schema.py``: packing, history, auxiliary
  validity, command/execution records, deployable torque-to-force interface.
* ``rsl_rl/modules/actor_critic_mule.py``: dedicated model and native losses.
  There are no MULE config/schema copies in ``rsl_rl``.

Example (configuration resolution only; no simulator construction)::

    from legged_gym.envs.go2.go2_mule.go2_mule_config import resolve_go2_mule_configs
    from rsl_rl.modules.actor_critic_mule import ActorCriticMULE
    env_cfg, train_cfg = resolve_go2_mule_configs()
    model = ActorCriticMULE(train_cfg)
    model.set_phase("joint_adaptation")  # only between complete future PPO updates

Existing package imports require the repository's Python dependencies and
``SIMULATOR=isaaclab`` selection. CPU checks do not initialize Isaac Lab/PhysX.
No task is registered and no launcher is supplied.

Source audit: paper-specified choices
------------------------------------

`MULE v1 <https://arxiv.org/html/2505.00488v1>`_, Sections II-D/E, III-A and
Fig.2 specify 45-D proprioception, 12-D position actions, nominal plus adaptive
actions, separate critics, both policies trained during adaptation, and
torque/Jacobian force estimates. Hidden widths are 512/256/128 for policies and
critics, 128/64 for CENet, and 64/128 for reconstruction. The figure supplies
shared context to both actors and augmented history to the adaptive actor.
The experiment reports history H=5.

`DreamWaQ v2 <https://arxiv.org/html/2301.10602v2>`_, Figs.1/2 and Sections
II-B/C, III-A provide a 16-D latent plus 3-D velocity, ELU, velocity MSE,
next-observation MSE, latent KL, and privileged state [o,v,d,h]. Its diagram
shows a latent-only reconstruction connection and labels 48 outputs.

The `official repository <https://github.com/StochLab/MULE>`_ inspected on
2026-11-12 contains project-page assets, not training code. The
`project page <https://www.stochlab.com/MULE/>`_ does not resolve implementation
details below. Local DreamWaQ classes are only references: their 24 explicit
outputs, alternative widths and velocity sampling were not imported.

Inherited experimental settings (not source reproduction)
---------------------------------------------------------

The factory copies an explicit allowlist, with immutable tuple-based schema
and exploration profiles. Shared modifications propagate at resolution time:
history length, measured terrain grid, named joints/standing pose, position
action scale/clipping, control period/decimation, environment/rollout sizing,
PPO settings, auxiliary LR/epochs, adaptive entropy settings, first 12 position
exploration values, seed, budget and save interval. Current live history is 20
frames, not source H=5; terrain has 143 samples. These yield actor width 45,
nominal policy input 64, adaptive input 20*57+19=1159, and critic width 194.
No HardPACT privileged channels, reward configuration, QP, PINNs, channel
guidance, physics decoders, PCGrad, or BR distillation are imported.

The flat learning contract has standalone defaults for unit construction;
experiments must use the Go2 resolver to obtain current shared values. No
complete environment configuration is fabricated in this stage.

Explicit reproduction assumptions / unresolved details
------------------------------------------------------

* MULE does not state latent width/activation numerically; use DreamWaQ's
  16-D latent and ELU. Velocity is a deterministic 3-D regression head; only
  latent context is sampled. The Gaussian log variance is bounded [-10,10].
* MULE's combined CENet wiring is interpreted as decoder input [scaled
  estimated velocity, sampled latent], with 45 next-observation outputs.
  DreamWaQ's latent-only/48-output diagram differs. Exact MULE decoder input
  and output packing remain unverified. This is versioned, not hidden.
* Both critics use the same declared [o,v,disturbance-force,terrain] state,
  with independent parameters. MULE's complete critic packing is unspecified.
  No payload state/mass or force labels enter either actor.
* Fixed shared observation multipliers replace unspecified online whitening.
  Joint angles are absolute; gravity is a unit vector. No extra observation
  clipping or running normalizer is introduced. Velocity supervision uses
  physical m/s; actor/decoder velocity features use the shared linear scale.
  Force proxy and critic disturbance-force multiplier default to 0.01/N.
* Reconstruction and velocity MSE average over valid rows and components.
  KL sums 16 dimensions then averages valid rows. Total is velocity MSE +
  reconstruction MSE + beta*KL; beta defaults to 1, independently of HardPACT.
* Context receives only auxiliary gradients. PPO treats the exact rollout
  velocity/latent snapshot as constant. There is no verified author-code
  evidence requiring PPO gradients through CENet. No AdaBoot/teacher injection
  is added. Shared context is reused by both actors.

Deployable observations, force signs, actions and timing
-------------------------------------------------------

Canonical order is FR, FL, RR, RL and hip, thigh, calf; xyz is forward/left/up
in the body frame. Schema fields name physical units and normalization.
Nominal observation is [omega_B, gravity_B, command_B, q, dq, previous_action].
Adaptive history appends four 3-D force proxies to each nominal frame. Terrain
is critic-only, x-major/y-minor, base_z minus ground_z, with live sample points.

``estimate_foot_forces`` accepts only applied/measured joint torques [Nm] and
leg Jacobians [m/rad], J=d(foot_position_B)/dq_leg at fixed base pose. It uses
f=pinv(J.T)*tau, following the source's positive torque-equivalent sign. This
is not a dynamic contact-force observer: gravity, inertia and transmission
uncertainty remain unmodeled. In static balance, environment-on-robot reaction
has the opposite sign when leg gravity is neglected. No simulator contact
force or HardPACT prediction may be substituted. Singular/nonfinite feet are
flagged invalid and supplied as zero sentinels, not claimed zero contact.

Each actor retains its own raw sample, Normal distribution, log probability,
entropy and value. Nominal-only correction is exactly zero and has no active
distribution (not a tiny-variance Gaussian). Combine raw samples, clip the
sum once, then issue q_stand + action_scale*combined_clipped. No individual
action clipping/scaling or position-target likelihood is used. Previous action
means that clipped combined issued action before scale/delay; zero on reset.
Execution is a separate record with the delay-selected target, its source ID,
and actual applied torque. Physical controller/delay integration is deferred.

History holds exactly H frames, oldest first including current. Explicit
monotonic control timestamps prohibit duplicate appends. Episode-ID changes
clear only affected rows; preceding frames are zero. observe()/inference must
not advance history or RNG. Auxiliary pairs use history(t), velocity(t),
observation(t+1). Cross-episode targets are excluded before arithmetic; valid
terminal targets require owned pre-reset snapshots. Nonfinite eligible rows
are counted and excluded. Empty batches return graph-free zeros and require
skipping the auxiliary optimizer step, including momentum/weight decay.

Next-stage integration requirements
-----------------------------------

1. Add dual PPO/storage with two returns/advantages, raw actions, distributions,
   exact context snapshots, matching timestamps and separate reward streams.
   Complete both PPO owners using stored contexts before auxiliary updates.
   Recompute policy likelihoods with current actor weights but never fresh z.
   Keep nominal/adaptive/auxiliary optimizers disjoint and retain moments across
   phase changes. No summed-command probability replaces either likelihood.
2. Preserve phases in model metadata; physical stage must not select optimization
   phase implicitly. Stage 1 has nominal locomotion/no movable payload; Stage 2
   has independently dynamic rolling balls/tray and joint adaptation. Both must
   use the existing disturbance curriculum later. No physical scheduler yet.
3. Implement named robot mappings, deployable torque/kinematics timestamps,
   actor noise, ground truth critic/labels, delayed executed commands, failure
   versus timeout precedence and terminal snapshots. Do not repurpose BR boxes.
4. Model state_dict includes strict config/schema/phase metadata; incompatible
   metadata/shape is rejected before changing weights. This is model-only
   serialization, NOT complete training or simulator resume. Future runner must
   save optimizer/RNG/history/curriculum state and reset stale physical history.
5. Liquid-surrogate fine-tuning, rewards, registration and launchers remain out
   of scope. This stage makes no locomotion-performance or hardware claim.

Validation
----------

Run from the rescaled_qp_exps worktree::

    SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 \
      /home/oyoungquist/.conda/envs/lr_lab_cupiqp/bin/python -m pytest \
      -q -p no:cacheprovider tests/test_mule_stage1.py

Checks cover dimensions, inheritance/isolation, force signs, reset ownership,
action scaling, phase/gradient ownership, stored-context replay, deterministic
inference, native losses/masks and model serialization. No training launched.

Executed on 2026-11-12: 16 MULE tests passed (2.99 s). The same command with
``tests/test_beyond_robustness*.py`` appended passed 93 tests (9.80 s), including
77 existing BR tests. One external hppfcl deprecation warning; no failed checks.
No Isaac Lab instance, payload physics, GPU test, or production PPO run was used.
