Actor recovery supervision
==========================

The active Go2 HardPACT ``hard_pact_qp`` dictionary enables:

::

    recovery_slack_huber_enabled = True
    recovery_slack_huber_delta = 1.0
    recovery_loss_budget_enabled = True
    recovery_loss_budget_ratio = 0.5
    recovery_loss_budget_ema_alpha = 0.01
    recovery_loss_budget_reference_floor = 1e-6

Missing enable flags default to False. Internal QP slack objectives, constraints,
execution, coefficients and valid-row denominators are unchanged. Twice-Huber
applies only to normalized outer joint/rate slack; torque correction stays
quadratic. Rate slack remains absent when rate constraints are disabled.

The actor recovery scalar includes recovery_projection_weight and
lambda_projection exactly once. Its detached budget scale is applied before
actor warmup and the existing PCGrad operation. The reference consists of the
other weighted actor physics losses (primary projection, velocity, FK and load),
never PPO or estimator losses. The budget limits scalar contribution, NOT
gradient norm; ordinary scaled recovery gradients remain above budget.

Each iteration freezes the previous EMA, or the first finite nonnegative
minibatch reference while uninitialized. Completed-iteration EMA input is the
minibatch-row-count-weighted mean of these already-normalized loss scalars;
individual component denominators are not changed. Invalid reference scalars
are excluded from EMA statistics, without sanitizing the optimized objective.
No valid evidence leaves the persistent reference unchanged. Empty recovery
has zero contribution. State is detached/on-device, with no minibatch host reads.

Checkpoint envelopes and legacy runner saves retain EMA, initialization and
last-update/completed iteration. Older checkpoints without these fields start
fresh evidence. Only absent new options may adopt explicit legacy-resume
configuration; all other contract checks remain in force, and existing recorded
options must still match. No missing historical EMA is invented. Ablations
retain their existing loss masks; torque imitation does not acquire slack
supervision and metrics-only modes acquire no recovery gradients.

At the existing diagnostic interval, ``Loss/actor_pinn/recovery_budget/`` logs
``uncapped``, ``applied``, ``budget``, ``scale``, and ``quadratic_weighted``.
The latter uses the original quadratic shared helper and valid recovery-row
averaging; budget scalars aggregate over actor minibatch rows. Physical slack
and correction metrics remain. ``total_weighted`` is AFTER budget application;
``total_warmed`` also includes warmup. Neither is a gradient-norm measurement.

Focused validation (no training):

::

    CUDA_VISIBLE_DEVICES='' SIMULATOR=isaaclab python -m pytest -q \
      tests/test_hard_pact_recovery_budget.py tests/test_training_state_transfer.py \
      tests/test_hard_pact_soft_joint_recovery.py tests/test_hard_pact_rate_recovery.py \
      tests/test_hard_pact_actor_loss_warmup.py

The existing CUDA ``test_recovery_outer_loss_heads_retained_backward_masks_and_weights``
fixture selects primary rows instead of its expected recovery rows on this HEAD.
This was reproduced using the unmodified HEAD QP implementation, before any
outer recovery loss is evaluated. It was not fixed by changing solver behavior.
