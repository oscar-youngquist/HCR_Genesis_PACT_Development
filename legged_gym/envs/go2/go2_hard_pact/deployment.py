"""Fixed HardPACT force normalization and deployment contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os

from rsl_rl.modules.hard_pact_physics import (
    GRF_NORMALIZATION_VERSION,
    GRF_SCALE_N,
    WRENCH_NORMALIZATION_VERSION,
    WRENCH_QP_CLIP_N_NM,
    WRENCH_SCALE_N_NM,
)


FOOT_ORDER = ("FR", "FL", "RR", "RL")
WRENCH_ORDER = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
RECONSTRUCTION_INDICES = tuple(range(61)) + tuple(range(73, 145))
RECONSTRUCTION_DIM = len(RECONSTRUCTION_INDICES)


@dataclass(frozen=True)
class PhysicsGainSpec:
    grf_scale_n: tuple[float, ...]
    wrench_scale_n_nm: tuple[float, ...]
    wrench_qp_clip_n_nm: tuple[float, ...]


def calculate_physics_head_gains(cfg):
    """Validate and return the fixed HardPACT decoder/QP scales."""
    grf_scale_n = tuple(float(v) for v in cfg.sim.grf.prediction_scale_n) * 4
    if grf_scale_n != GRF_SCALE_N:
        raise ValueError(
            "HardPACT GRF decoder normalization must be [250, 250, 250] N "
            "in FR/FL/RR/RL order"
        )
    grf_obs_scale = float(cfg.normalization.obs_scales.grf)
    wrench_obs_scale = float(cfg.normalization.obs_scales.base_wrench)
    if grf_obs_scale <= 0.0 or wrench_obs_scale <= 0.0:
        raise ValueError("force observation scales must be positive")

    deployment = cfg.deployment_physics
    configured_wrench_scale = tuple(float(v) for v in getattr(
        deployment, "wrench_scale", WRENCH_SCALE_N_NM
    ))
    configured_qp_clip = tuple(float(v) for v in getattr(
        deployment, "wrench_qp_clip", WRENCH_QP_CLIP_N_NM
    ))
    if configured_wrench_scale != WRENCH_SCALE_N_NM:
        raise ValueError("HardPACT wrench_scale must be [100,100,100,25,25,25]")
    if configured_qp_clip != WRENCH_QP_CLIP_N_NM:
        raise ValueError("HardPACT wrench_qp_clip must be [150,150,150,40,40,40]")
    return PhysicsGainSpec(
        grf_scale_n=grf_scale_n,
        wrench_scale_n_nm=WRENCH_SCALE_N_NM,
        wrench_qp_clip_n_nm=WRENCH_QP_CLIP_N_NM,
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
        "schema_version": 4,
        "grf_normalization_version": GRF_NORMALIZATION_VERSION,
        "wrench_normalization_version": WRENCH_NORMALIZATION_VERSION,
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
                {"name": "foot_contact_logits", "dimension": 4, "order": list(FOOT_ORDER), "units": "logit", "scaling": 1.0, "clipping": None, "training_loss": "binary_cross_entropy_with_logits"},
                {"name": "foot_clearance", "dimension": 4, "order": list(FOOT_ORDER), "units": "m", "scaling": 1.0, "clipping": [-1.0, 1.0]},
            ],
        },
        "latent_dimension": latent_dim,
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
            "observation_scale_is_independent": True,
        },
        "wrench_decoder_normalization": {
            "version": WRENCH_NORMALIZATION_VERSION,
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
                "epsilon": float(getattr(cfg.deployment_physics, "contact_probability_epsilon", 1.0e-2)),
                "input": "explicit_estimator.foot_contact_logits",
                "output": "QP contact probability",
                "parameterization": "epsilon + (1 - 2*epsilon) * sigmoid(contact_logits)",
                "application_count": "exactly_once_at_QP_boundary",
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
            "grf_normalization_version": "physics_estimator.grf_normalization_version",
            "base_wrench": "physics_estimator.wrench_scale",
            "base_wrench_qp_clip": "physics_estimator.wrench_qp_clip",
            "wrench_normalization_version": "physics_estimator.wrench_normalization_version",
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
