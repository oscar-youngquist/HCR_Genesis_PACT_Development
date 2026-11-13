BR and MULE logging contract
============================

Both runners use ``rsl_rl/utils/learning_diagnostics.py``. Preserve this contract
in subsequent rolling-payload/environment stages instead of adding parallel
loggers. Logging performs detached reductions on existing data, not additional
policy/decoder evaluations, backward passes, RNG draws, or curriculum updates.
HardPACT and its existing loggers are unchanged.

Published conventions
---------------------

* ``Episode/rew_<term>``: sum of the actual weighted, dt-scaled per-step term
  over each completed episode, divided by the configured maximum episode
  duration in seconds. This is HardPACT's normalization; do NOT divide by
  actual duration or multiply dt again. Average completed episodes with equal
  episode weight, not equal reset-batch weight. MULE uses
  ``Episode/nominal/rew_<term>`` and ``Episode/adaptive/rew_<term>``.
* ``Train/mean_reward`` and ``Train/mean_episode_length``: last 100 completed
  episode returns and lengths (steps), like HardPACT's rewbuffer/lenbuffer.
  The existing HardPACT ``train/return`` / ``train/episode_length`` aliases are
  retained. MULE keeps nominal/adaptive returns separate, never summed.
* ``Loss/value_function``, ``Loss/surrogate``, ``Loss/learning_rate`` for BR;
  MULE scopes these under ``Loss/nominal`` / ``Loss/adaptive``. BR additionally
  reports latent matching and load estimation. MULE's native auxiliary losses
  use ``Loss/autoenc_function``, ``velo_pred``, ``recon``, ``kl_div``. Inactive
  adaptive losses and student-reinforcement latent matching are NOT published.
* ``Policy/mean_noise_std`` / ``Policy/mean_noise_std_position``: current
  learned exploration std in normalized position-action units. MULE also
  separates ``Policy/nominal`` and active ``Policy/adaptive``; overall std is
  the average of active actors' parameters, not the summed command's variance.
* ``Perf/*``: FPS, collection/learning/iteration time, accumulated time and
  interactions. ``Values/entropy`` and applicable domain-randomization values
  follow HardPACT names. ``curriculum/commands/*`` reports command gating.
* ``EpisodeDiagnostics/*``: completed-episode failure/timeout counts, fractions
  and durations; overlapping reason counts/fractions and availability counts.
  Failure wins over timeout. Reasons describe conditions on the terminal step,
  not causal attribution. Empty groups have counts but no fabricated means.
  Reward-valid/incomplete counts expose exclusions, including nonfinite data.
* BR-only ``LoadEstimation/*`` and ``Payload/*`` retain meaningful estimator
  errors and realized payload summaries; they have no HardPACT equivalent.

The old ``reward/raw``, ``reward/weighted``, ``reward/coefficient``, lower-case
``loss/*``/``load/*`` and duplicate native ``train/*`` series are NOT published.
Reward coefficients remain in saved configuration/reward manifests, not scalar
logs. Internal runner return dictionaries retain their API for validation; only
``last_logged_metrics`` is sent to TensorBoard and the terminal formatter.
No reward changes, including smoothness sign changes, are part of this patch.

Terminal summaries follow HardPACT's learning-iteration block: computation,
value/surrogate and method-native losses, policy std, mean reward/episode length,
episode reward terms, failure counts and total interactions/time. The BR entry
point no longer adds a second abbreviated per-iteration line. A provided writer
is respected; both runners automatically create TensorBoard writers when a
log directory is supplied. ``console=False`` suppresses terminal summaries.

Future adapter requirements
---------------------------

Provide logging-only pre-reset ``failure_reasons`` (boolean per-environment
condition masks), already weighted/dt-scaled ``reward_terms``, actual control
dt and ``max_episode_length_s``. MULE reward terms must be nested by nominal /
adaptive stream. Missing terms/reasons/duration are unavailable, NOT guessed.
No terminal measurement may be reconstructed from an automatically reset state.
Partial resets clear only completed rows. Terms missing for part of an episode
are excluded from that term's episode average and counted as incomplete.
Exact synthetic resume retains diagnostic episode sums and the rolling window;
fresh physical resume clears them. Older checkpoints without these optional
fields remain loadable without fabricating earlier partial-episode rewards.

Tests
-----

From the rescaled_qp_exps worktree::

    SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 \
      /home/oyoungquist/.conda/envs/lr_lab_cupiqp/bin/python -m pytest -q \
      -p no:cacheprovider tests/test_br_mule_logging.py \
      tests/test_beyond_robustness*.py tests/test_mule_stage1.py tests/test_mule_learning.py

The focused tests read actual TensorBoard event files, exercise both learning
phases, validate terminal output/aggregation and resume, and compare model/RNG
states with diagnostics disabled. They use deterministic synthetic environments,
not Isaac Lab physics or production training.
