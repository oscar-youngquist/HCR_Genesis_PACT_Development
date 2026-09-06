"""Fixed HardPACT force normalization and deployment contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os

FOOT_ORDER = ("FR", "FL", "RR", "RL")
WRENCH_ORDER = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
RECONSTRUCTION_INDICES = tuple(range(61)) + tuple(range(73, 145))
RECONSTRUCTION_DIM = len(RECONSTRUCTION_INDICES)


@dataclass(frozen=True)
class PhysicsGainSpec:
    grf_scale_n: tuple[float, ...]
    grf_clip_min_n: float
    grf_clip_max_n: float
    wrench_scale_n_nm: tuple[float, ...]
    wrench_qp_clip_n_nm: tuple[float, ...]


def calculate_physics_head_gains(cfg):
    """Validate and return the configured HardPACT force scales and bounds."""
    grf_scale_n = tuple(float(v) for v in cfg.sim.grf.prediction_scale_n) * 4
    if len(grf_scale_n) != 12 or any(value <= 0.0 for value in grf_scale_n):
        raise ValueError("HardPACT GRF prediction_scale_n must contain 3 positive values")
    grf_clip_min_n = float(cfg.sim.grf.clip_min_n)
    grf_clip_max_n = float(cfg.sim.grf.clip_max_n)
    if grf_clip_min_n > grf_clip_max_n:
        raise ValueError("HardPACT GRF clip_min_n must not exceed clip_max_n")
    grf_obs_scale = float(cfg.normalization.obs_scales.grf)
    wrench_obs_scale = float(cfg.normalization.obs_scales.base_wrench)
    if grf_obs_scale <= 0.0 or wrench_obs_scale <= 0.0:
        raise ValueError("force observation scales must be positive")

    deployment = cfg.deployment_physics
    configured_wrench_scale = tuple(float(v) for v in deployment.wrench_scale)
    configured_qp_clip = tuple(float(v) for v in deployment.wrench_qp_clip)
    if len(configured_wrench_scale) != 6 or any(
        value <= 0.0 for value in configured_wrench_scale
    ):
        raise ValueError("HardPACT wrench_scale must contain 6 positive values")
    if len(configured_qp_clip) != 6 or any(
        value <= 0.0 for value in configured_qp_clip
    ):
        raise ValueError("HardPACT wrench_qp_clip must contain 6 positive values")
    return PhysicsGainSpec(
        grf_scale_n=grf_scale_n,
        grf_clip_min_n=grf_clip_min_n,
        grf_clip_max_n=grf_clip_max_n,
        wrench_scale_n_nm=configured_wrench_scale,
        wrench_qp_clip_n_nm=configured_qp_clip,
    )


def _hidden_linear_widths(network):
    widths = [
        layer.out_features for layer in network if hasattr(layer, "out_features")
    ]
    return widths[:-1]


def build_deployment_contract(cfg, actor, gain_spec):
    """Build the human- and machine-readable frozen deployment contract."""
    grf_buffer = actor.physics_estimator.grf_scale_n.detach().cpu().tolist()
    wrench_buffer = actor.physics_estimator.wrench_scale.detach().cpu().tolist()
    latent_dim = actor.context_encoder.ce_out_mean.out_features
    explicit_dim = actor.explicit_estimator.network[-1].out_features
    contract = {
        "schema_version": 6,
        "explicit_estimator": {
            "dimension": 11,
            "input": "shared_history_encoder_features",
            "input_dimension": actor.context_encoder.feature_dim,
            "hidden_layers": [
                layer.out_features
                for layer in actor.explicit_estimator.network
                if hasattr(layer, "out_features")
            ][:-1],
            "activation": "ELU",
            "fields": [
                {"name": "base_linear_velocity_body", "dimension": 3, "units": "observation_scaled_m_per_s", "scaling": "obs_scales.lin_vel", "clipping": None},
                {"name": "foot_contact_probability", "dimension": 4, "order": list(FOOT_ORDER), "units": "probability", "scaling": "epsilon + (1 - 2*epsilon) * sigmoid(contact_logits)", "clipping": None},
                {"name": "foot_clearance", "dimension": 4, "order": list(FOOT_ORDER), "units": "m", "scaling": 1.0, "clipping": [-1.0, 1.0]},
            ],
        },
        "contact_estimator_supervision": {
            "raw_output": "contact_logits",
            "training_loss": "binary_cross_entropy_with_logits",
            "labels": "canonical_binary_contact_FR_FL_RR_RL",
            "epsilon": float(actor.explicit_estimator.contact_epsilon),
            "runtime_conversion_count": "exactly_once_in_explicit_estimator",
            "shared_runtime_vector": "explicit_for_policy",
            "checkpoint_semantics_key": "explicit_estimator.contact_probability_semantics",
        },
        "latent_dimension": latent_dim,
        "physics_head_latent_semantics": {
            "training": "reparameterized_sample_mu_plus_sigma_epsilon",
            "deployment": "deterministic_mean",
            "explicit_conditioning": "stop_gradient_explicit_for_policy",
        },
        "history": {
            "observation_dimension": int(cfg.env.num_observations),
            "steps": int(cfg.env.num_obs_hist),
        },
        "deployment_heads": {
            "activation": "ELU",
            "grf": {
                "input_order": ["z_t", "stopgrad(explicit_t)", "tau_nom"],
                "input_dimension": latent_dim + explicit_dim + 12,
                "hidden_layers": _hidden_linear_widths(actor.physics_estimator.grf_head),
                "output_dimension": 12,
                "output": "normalized_yaw_local_interval_grf",
                "target_formula": "target_grf_physical_n / grf_scale_n",
                "physical_reconstruction": "predicted_normalized * grf_scale_n",
                "grf_scale_n": grf_buffer,
            },
            "base_wrench": {
                "input_order": ["z_t", "stopgrad(explicit_t)"],
                "input_dimension": latent_dim + explicit_dim,
                "hidden_layers": _hidden_linear_widths(actor.physics_estimator.wrench_head),
                "output_dimension": 6,
                "output": "unbounded_raw_normalized_yaw_local_wrench",
                "activation_after_final_linear": None,
                "learned_final_bias": True,
            },
        },
        "frames_and_units": {
            "grf": {"frame": "yaw_local", "units": "N", "foot_order": list(FOOT_ORDER), "component_order": ["Fx", "Fy", "Fz"], "target": "deadbanded_clipped_control_interval_average"},
            "base_wrench": {"frame": "yaw_local", "units": ["N", "N", "N", "Nm", "Nm", "Nm"], "order": list(WRENCH_ORDER), "target": "total_external_wrench_label"},
        },
        "critic_observation_scales_independent_of_decoders": {
            "grf": float(cfg.normalization.obs_scales.grf),
            "base_wrench": float(cfg.normalization.obs_scales.base_wrench),
        },
        "grf_decoder_normalization": {
            "scale_n": grf_buffer,
            "target": "target_grf_physical_n / scale_n",
            "output": "predicted_normalized",
            "physical_reconstruction": "predicted_normalized * scale_n",
            "decoder_prediction_clipping": None,
            "interval_target_clip_n": {
                "minimum": gain_spec.grf_clip_min_n,
                "maximum": gain_spec.grf_clip_max_n,
                "location": "GRF processor before control-interval averaging",
            },
            "observation_scale_is_independent": True,
        },
        "wrench_decoder_normalization": {
            "scale_n_nm": wrench_buffer,
            "target": "wrench_target_physical / scale_n_nm",
            "output": "unbounded_raw_normalized",
            "physical_reconstruction": "raw_normalized * scale_n_nm",
            "offset": None,
            "output_nonlinearity": None,
            "supervised_prediction_clipping": None,
            "supervised_target_clipping": None,
            "observation_scale_is_independent": True,
        },
        "qp_inputs": {
            "base_wrench": {
                "decoder_scale": wrench_buffer,
                "qp_clip": list(gain_spec.wrench_qp_clip_n_nm),
                "physical_lower": [-v for v in gain_spec.wrench_qp_clip_n_nm],
                "physical_upper": list(gain_spec.wrench_qp_clip_n_nm),
                "ordering": list(WRENCH_ORDER),
                "frame": "yaw_local_about_base_origin",
                "units": ["N", "N", "N", "Nm", "Nm", "Nm"],
                "parameterization": "unbounded_final_linear_output_times_fixed_scale",
                "sanitization": "torch.nan_to_num default, before clamp",
                "clamp_location": "exactly_once_immediately_before_QP_then_yaw_to_world",
                "clamp_gradient": "ordinary_clamp_no_straight_through",
            },
            "contact": {
                "epsilon": float(actor.explicit_estimator.contact_epsilon),
                "input": "explicit_for_policy.foot_contact_probability",
                "output": "unchanged QP contact probability",
                "parameterization": "already converted by explicit estimator",
                "application_count": "no downstream conversion",
                "foot_order": list(FOOT_ORDER),
            },
        },
        "reconstruction_target": {
            "dimension": RECONSTRUCTION_DIM,
            "excluded": ["grf_12", "terrain_heights_143"],
            "critic_input_unchanged": True,
        },
        "conversion": {
            "grf_decoder_output_to_physical": "grf_physical_n = predicted_normalized * grf_scale_n",
            "grf_physical_to_decoder_target": "target_normalized = target_grf_physical_n / grf_scale_n",
            "grf_physical_to_observation": "grf_observation = grf_physical_n * obs_scales.grf",
            "wrench_decoder_output_to_physical": "wrench_raw_physical = wrench_raw_normalized * wrench_scale",
            "wrench_physical_to_decoder_target": "wrench_target_normalized = wrench_target_physical / wrench_scale",
            "wrench_physical_to_qp": "wrench_qp = clamp(nan_to_num(wrench_raw_physical), -wrench_qp_clip, wrench_qp_clip)",
        },
        "checkpoint_buffer_keys": {
            "grf": "physics_estimator.grf_scale_n",
            "base_wrench": "physics_estimator.wrench_scale",
            "base_wrench_qp_clip": "physics_estimator.wrench_qp_clip",
            "contact_semantics": "explicit_estimator.contact_probability_semantics",
        },
    }
    return contract


def write_deployment_contract_once(log_dir, contract):
    """Create the deployment contract once; never overwrite an existing run contract."""
    if not log_dir:
        return None, False
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "hard_pact_deployment_contract.json")
    try:
        with open(path, "x", encoding="utf-8") as stream:
            json.dump(contract, stream, indent=2)
    except FileExistsError:
        return path, False
    return path, True
