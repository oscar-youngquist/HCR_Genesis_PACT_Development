"""Stable scalar-only TensorBoard schema for HardPACT ablations."""

import math

from .hard_pact_ablations import resolve_hard_pact_features


# NaN means "not computed by this ablation"; zero is reserved for counts,
# enabled flags, or mathematically zero values.  These keys are emitted for
# every variant at every runner log call.
STABLE_HARD_PACT_SCALARS = {
    "physics/inverse/enabled": 0.0,
    "physics/rollout/enabled": 0.0,
    "physics/soft_constraint/enabled": 0.0,
    "physics/loss/inverse": math.nan,
    "physics/loss/rollout": math.nan,
    "physics/loss/soft_constraint": math.nan,
    "physics/loss/pinn_unweighted": math.nan,
    "physics/timing/inverse_forward_ms_per_update": math.nan,
    "physics/timing/inverse_forward_ms_per_minibatch": math.nan,
    "physics/timing/inverse_chunk_count": 0.0,
    "physics/timing/rollout_forward_ms_per_update": math.nan,
    "physics/timing/rollout_forward_ms_per_minibatch": math.nan,
    "physics/timing/rollout_chunk_count": 0.0,
    "physics/inverse/base_linear_mae_physical": math.nan,
    "physics/inverse/base_angular_mae_physical": math.nan,
    "physics/inverse/joints_mae_physical": math.nan,
    "physics/inverse/all_mae_physical": math.nan,
    "physics/rollout/base_linear_mae_physical": math.nan,
    "physics/rollout/base_angular_mae_physical": math.nan,
    "physics/rollout/joints_mae_physical": math.nan,
    "physics/rollout/all_mae_physical": math.nan,
    "physics/grf/loss": math.nan,
    "physics/force_decoder_diagnostics_enabled": 0.0,
    "physics/wrench/active_loss": math.nan,
    "physics/wrench/neutral_loss": math.nan,
    "qp/enabled": 0.0,
    "qp/projection_loss_enabled": 0.0,
    "qp/projection_metric_enabled": 0.0,
    "qp/minimal/projection_loss": math.nan,
    "qp/minimal/full_fraction": 0.0,
    "qp/minimal/relaxed_fraction": 0.0,
    "qp/minimal/fallback_fraction": 0.0,
    "qp/minimal/differentiated_fraction": 0.0,
    "qp/minimal/normalized_equality_residual_max": math.nan,
    "qp/minimal/normalized_inequality_violation_max": math.nan,
    "qp/minimal/pre_clamp_torque_violation_max": math.nan,
    "qp/minimal/intervention_fraction": 0.0,
    "qp/minimal/intervention_torque_correction_rms": math.nan,
    "qp/minimal/rollout_timing_ms": math.nan,
    "qp/minimal/rollout_correction_mean": math.nan,
    "qp/minimal/rollout_correction_p95": math.nan,
    "qp/minimal/rollout_correction_max": math.nan,
    "qp/minimal/rollout_slack_mean": math.nan,
    "qp/minimal/rollout_slack_p95": math.nan,
    "qp/minimal/rollout_slack_max": math.nan,
    "grad/physics/finite_fraction": math.nan,
    "grad/physics/nonfinite_fraction": math.nan,
    "grad/physics/norm": math.nan,
    "grad/pcgrad/conflict_fraction": math.nan,
    "grad/pcgrad/cosine": math.nan,
    "disturbance/persistent_active_fraction": math.nan,
    "domain_rand/joint_dynamics_progress": math.nan,
    "domain_rand/mass_com_progress": math.nan,
    "domain_rand/disturbance_progress": math.nan,
}


def collect_force_decoder_scalars(auxiliary):
    """Map shared auxiliary force metrics to identical TensorBoard names."""
    values = {}
    for name, value in auxiliary.items():
        if name.startswith("grf_"):
            values[f"physics/grf/{name[len('grf_'):]}"] = value
        elif name.startswith("wrench_") and name not in (
            "wrench_active", "wrench_neutral"
        ):
            values[f"physics/wrench/{name[len('wrench_'):]}"] = value
    return values


def collect_contact_estimator_scalars(auxiliary):
    """Map contact-estimator diagnostics to the shared TensorBoard prefix."""
    return {
        f"physics/contact_estimator/{name[len('contact_'):]}": value
        for name, value in auxiliary.items()
        if name.startswith("contact_")
    }


def collect_latent_diagnostics_scalars(diagnostics):
    """Expand per-dimension device tensors into stable scalar TB keys."""
    values = {}
    for name, value in diagnostics.items():
        if getattr(value, "ndim", 0) == 1:
            for dimension in range(value.shape[0]):
                values[f"{name}/dim_{dimension:02d}"] = value[dimension]
        else:
            values[name] = value
    return values


def collect_hard_pact_scalars(algorithm, features):
    """Merge algorithm summaries into the stable schema without host copies."""
    spec = resolve_hard_pact_features(features)
    values = dict(STABLE_HARD_PACT_SCALARS)
    values.update({
        "physics/inverse/enabled": float(spec.inverse_loss),
        "physics/rollout/enabled": float(spec.rollout_loss),
        "physics/soft_constraint/enabled": float(spec.soft_constraint_penalty),
        "qp/enabled": float(spec.execution_qp),
        "qp/projection_loss_enabled": float(spec.projection_loss),
        "qp/projection_metric_enabled": float(spec.projection_metric),
        "physics/force_decoder_diagnostics_enabled": float(
            getattr(algorithm, "force_decoder_diagnostics_enabled", False)
        ),
    })
    for name, value in getattr(algorithm, "last_physics_loss_metrics", {}).items():
        values[name] = value
    inverse = getattr(algorithm, "last_inverse_dynamics_metrics", {})
    rollout = getattr(algorithm, "last_rollout_dynamics_metrics", {})
    for block in ("base_linear", "base_angular", "joints", "all"):
        key = f"inverse_residual/{block}_mae_physical"
        if key in inverse:
            values[f"physics/inverse/{block}_mae_physical"] = inverse[key]
        key = f"rollout_velocity/{block}_mae_physical"
        if key in rollout:
            values[f"physics/rollout/{block}_mae_physical"] = rollout[key]
    auxiliary = getattr(algorithm, "last_auxiliary_metrics", {})
    for source, target in (
        ("grf", "physics/grf/loss"),
        ("wrench_active", "physics/wrench/active_loss"),
        ("wrench_neutral", "physics/wrench/neutral_loss"),
    ):
        if source in auxiliary:
            values[target] = auxiliary[source]
    # Force-regression diagnostics are reduced to device scalars by the
    # auxiliary update; no per-environment tensor reaches the runner.
    values.update(collect_force_decoder_scalars(auxiliary))
    values.update(collect_contact_estimator_scalars(auxiliary))
    if getattr(algorithm, "ppo_latent_diagnostics_enabled", False):
        values.update(collect_latent_diagnostics_scalars(
            getattr(algorithm, "last_latent_diagnostics", {})
        ))
    values.update(getattr(algorithm, "last_qp_metrics", {}))
    gradients = getattr(algorithm, "last_physics_gradient_metrics", {})
    if "physics_gradient/finite_fraction" in gradients:
        finite = gradients["physics_gradient/finite_fraction"]
        values["grad/physics/finite_fraction"] = finite
        values["grad/physics/nonfinite_fraction"] = 1.0 - finite
    if "physics_gradient/finite_norm" in gradients:
        values["grad/physics/norm"] = gradients["physics_gradient/finite_norm"]
    values.update({
        key: value for key, value in gradients.items() if key.startswith("grad/")
    })
    return values
