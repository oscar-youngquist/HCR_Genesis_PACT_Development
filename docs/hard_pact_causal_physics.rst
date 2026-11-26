HardPACT causal physics estimators (deployment schema 19)
=======================================================

Configuration
-------------

``deployment_physics.height_label_mode = 'center_weighted'`` and
``height_label_sigma_m = 0.20`` resolve a normalized Gaussian over the existing
torso-relative terrain sample XY coordinates. Weights are cached at environment
initialization; no patch points are added. ``'uniform'`` retains the old label.
Explicit-height and QP model-height diagnostics use this label. Reward and
termination height remain the uniform patch mean; their metrics are not
estimator-label errors.

The algorithm settings are ``explicit_height_loss_weight = 4.0``,
``active_wrench_loss_weight = 2.0``, ``torso_force_loss_weights = [1,1,2]``,
and ``grf_axis_loss_weights = [1,1,2]``. Height scales only its squared residual
inside the original continuous-component mean. XYZ force weights multiply
elementwise normalized Huber values before the original reduction; moment
weights stay one. Neutral-wrench coefficients and physical normalization are
unchanged. These are supervised-estimator weights, not QP objective weights.
The position variant and thin inherited ablations use the same settings without
enabling any disabled objective.

Runtime and gradients
---------------------

Both GRF and applied-wrench heads now require a detached 12-vector in physical
Nm: the final total torque from the last completed physics substep, after
actuator scaling, QP blending/caps/fallback and saturation. It is zero after
reset. Capture it BEFORE the control-rate decoder update; neither the current
nominal torque nor the upcoming interval-average torque is a valid substitute.
Hold that snapshot for the complete interval. QP anchors retain it in
``sampled_qp_grf_conditioning_torque`` even when a later solve is selected.
The shared rollout packet's ``previous_executed_torque`` supplies supervised
and representation-PINN replay. Position storage reuses its historically named
``executed_torque_targets`` slot for this preceding-substep snapshot.

Heads detach explicit context and torque inputs but retain latent/decoder
learning. There is no direct current-action-to-GRF torque-conditioning edge.
The actor's current nominal torque remains differentiable through dynamics and
the QP. Measured interval-average execution remains the representation dynamics
input, not a decoder conditioning input. No solve or dynamics call is added.
Deployment callers must provide the snapshot to both heads at their existing
control cadence; live play uses the same environment methods and runner.

Compatibility
-------------

Schema 19 records torque-input semantics v2 and height-label semantics v2,
including mode and sigma. Raw model state dictionaries also carry semantic
guards. Wrench input width increases by 12; GRF width stays unchanged but its
meaning changes. Older heads are explicitly rejected, including weights-only
loads; no padding, metadata-only conversion or automatic head reinitialization
is provided. Use the historical code/export for historical checkpoints, or
explicitly implement and validate a migration/retraining procedure. The pinned
October-7 exporter remains a historical model/contract, not an exporter for
these new heads. Uniform height does not restore legacy torque semantics.

Validation
----------

Executed CPU regression result: 242 passed, 19 subtests passed, 5 CUDA-dependent
checks skipped. This includes auxiliary encoder/decoder updates, RSS variants,
BARD gradient ownership, replay and reset checks. No training or simulator run
was started; GPU/simulator execution remains unverified for this change.

Focused CPU checks (from the repository root)::

  SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" python -m pytest -q \
    tests/test_hard_pact_causal_physics.py tests/test_hard_pact_action_replay.py \
    tests/test_hard_pact_auxiliary.py tests/test_hard_pact_pos_auxiliary.py \
    tests/test_hard_pact_two_stage_auxiliary.py tests/test_hard_pact_height.py \
    tests/test_hard_pact_qp_modes.py tests/test_go2_hard_pact_physics_heads.py
