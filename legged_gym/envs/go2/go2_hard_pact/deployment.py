"""Fixed HardPACT force normalization and deployment contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import warnings
from rsl_rl.modules.hard_pact_physics import GRFSwingConfig

FOOT_ORDER = ("FR", "FL", "RR", "RL")
WRENCH_ORDER = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
# Preserve the legacy critic ordering and height-observation scaling. Only
# duplicate GRFs (61:73) belong exclusively to their dedicated decoder;
# the terrain height patch (145:288) is reconstructed here again.
RECONSTRUCTION_INDICES = tuple(range(61)) + tuple(range(73, 288))
RECONSTRUCTION_DIM = len(RECONSTRUCTION_INDICES)


def qp_update_contract(mode, decimation, warmup_iterations=0, qp_config=None):
    """Execution metadata using the same schedule and projection as rollout."""
    from rsl_rl.algorithms.hard_pact_qp import qp_substep_anchors
    from dataclasses import asdict
    from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig
    settings = qp_config or HardPACTQPConfig()

    qp_substep_anchors(mode,decimation)
    result = {
        "mode": mode, "formulation": "masked_torque_force_24",
        "variable_ordering": ["total_actuator_torque_12", "FR_FL_RR_RL_world_XYZ_tilde_force_12"],
        "physical_force": "f=D_m*tilde_f; detached stance inside dynamics, joint rows and soft contact objective; returned forces are physical f",
        "matrix_shape": "24 variables; 68 canonical inequalities, zero equalities; cuPIQP native torque bounds plus 44 general inequalities (24 joint,20 friction)",
        "swing_rows": "friction rows 0<=1, not equality padding; positive force tracking curvature retained",
        "solver_and_objective_settings": dict(asdict(settings),qp_update_mode=mode),
        "stance_threshold": settings.contact_threshold,
        "stance_selection": "detached probability >= threshold; optimized swing force exactly zero",
        "joint_position_beta": settings.position_integration_coefficient,
        "training_warmup_iterations": warmup_iterations,
        "physics_substep_anchors": [0,1,2,3],
        "problems_per_environment_interval": 4 if mode=="every_substep" else 1,
        "execution_selection": "all rows" if mode=="every_substep" else "preselected balanced uniform K per environment; compact K=k at each substep",
        "prediction_horizon": "one physics/PD timestep",
        "prediction_rate": "GRF/wrench/contact/latent/explicit evaluated before first physics substep in both modes; no neural forward during substepping; refresh yaw-to-world at each solve",
        "selection_helper": "rsl_rl.algorithms.hard_pact_qp.qp_substep_mask",
        "grf_conditioning": "bounded k=0 total nominal PD/feedforward torque; store k=0 joint q/v and actuator parameters; recompute using current replayed actions",
        "nominal_torque": "fresh bounded total PD/feedforward each substep, actuator effects exactly once",
        "unsolved_execution_helper": "rsl_rl.algorithms.hard_pact_qp.project_nominal_torque",
        "unsolved_execution": "project fresh nominal torque on actuator magnitude/rate intersection; no held correction",
        "previous_torque": "previous actually applied torque, zero on reset",
        "joint_contact_certification": "stage 0: hard QP; stage 1: soft-joint recovery, no hard joint certificate; stage 2/unsolved: actuator-only",
        "soft_joint_recovery": "optional 36-D [tau12, masked-force12, nonnegative joint-slack12(rad/s²)]; quadratic normalized slack cost; torque/rate/friction/swing-zero stay hard; certified softened replay has a separately weighted torque-correction/slack loss and implicit VJP; failed rows excluded",
        "ppo_anchor_selection": "one balanced uniform executed QP substep per environment in both modes",
        "ppo_projection_loss_multiplier": 1,
        "frames": "world forces and world-aligned wrench about the existing base-Jacobian point; yaw-local head outputs rotated once",
        "limits": "canonical joint-specific magnitude/position/velocity limits from backend; acceleration/rate and objective scales in solver_and_objective_settings",
    }
    return result


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
    swing_config = GRFSwingConfig.from_task(cfg)
    contract = {
        "schema_version": 15,
        "grf_swing_gating": {
            "enabled": swing_config.enabled,
            "contact_probability_threshold": swing_config.threshold,
            "consistency_loss_weight": swing_config.loss_weight,
            "consistency_force_scale_n": grf_buffer,
            "swing": "contact_probability.detach() < threshold; no sigmoid or observation scaling",
            "qp_reference": "QP stance selection is mandatory and independent of this auxiliary switch; see qp_update.stance_threshold",
            "helper": "rsl_rl.algorithms.hard_pact_qp.HardPACTDifferentiableQP._build",
            "deployment_conversion": "physics_estimator.grf_to_physical(normalized_prediction), then yaw-to-world rotation; shared QP builder applies stance/reference gating exactly once",
            "ordering": list(FOOT_ORDER),
            "frame_units": "yaw-local Newtons before existing world rotation",
            "qp_decision_variables_and_constraints": "24-D torque/force variables; swing optimized XYZ force constrained to zero",
            "raw_supervised_predictions": "unchanged and ungated",
            "consistency_formula": "weight * sum_swing ||GRF_raw_N / grf_scale_N||^2 / max(swing_count, 1)",
            "consistency_target": "physical zero, independent of any normalization offset",
            "consistency_parameters": "GRF-exclusive decoder only; latent, explicit/contact and torque inputs detached",
            "metrics": "swing fraction of valid feet; raw norm mean over swing feet (N); removed norm mean over all valid feet (N); weighted consistency loss",
            "checkpoint": "no policy/state-dict changes; load this configuration alongside weights",
        },
        "torque_convention": {
            "conversion_helper": "rsl_rl.modules.hard_pact_control.bounded_nominal_torque",
            "requested_components_helper": "rsl_rl.modules.hard_pact_control.requested_torque_components",
            "position_target": "absolute joint position; default pose included exactly once",
            "raw_requested": "unclipped delayed action converted to physical Nm before action clipping, saturation or QP correction; used by torque-limit/feedforward/feedback magnitude penalties",
            "bounded_nominal": "execution-clipped delayed action converted using PD gains, branch weights and motor strength exactly once, then actuator magnitude bounds",
            "non_qp_rate_limit": "none in the current Genesis/Isaac Lab PACT actuator paths",
            "final_executed": "after QP or analytic magnitude/rate projection and final actuator saturation; authoritative physics label and next torque-rate center",
            "rollout_physics_gradient": "detached actual interval-average executed torque; no actor/QP actuation gradient; GRF/wrench/encoder gradients retained",
            "projection_gradient": "certified-row torque correction plus soft stance acceleration; implicit gradients to learned torque/GRF/wrench and encoder, detached mechanics and stance; stopgrad ablation metric only",
            "supervised_grf_wrench_predictions": "raw decoder predictions, unchanged",
            "units": "Nm in canonical joint order",
        },
        "qp_objective": {
            "implementation": "rsl_rl.algorithms.hard_pact_qp.HardPACTDifferentiableQP._build",
            "terms": ["nominal_torque_tracking", "predicted_grf_tracking",
                      "stance_acceleration_soft_tracking", "yaw_local_roll_pitch_stabilization",
                      "positive_definite_regularization"],
            "variables": "x=[total_actuator_torque_12; world_FR_FL_RR_RL_force_12]",
            "acceleration": "a=solve(M,[S^T,Jf^T])x+solve(M,Jb^T W-h)",
            "attitude_frame": "instantaneous yaw-local physical angular acceleration; not Euler-angle acceleration",
            "attitude_target": "-Kp*(z_world cross z_body)_yaw_xy-Kd*omega_yaw_xy",
            "stance": "detached probability >= configured contact_threshold; swing optimized force exactly zero; no swing friction rows",
            "recovery": "sanitized bounded nominal actuator/rate projection; no joint/contact certification or implicit VJP",
            "torque_history": "previous executed torque centers hard torque-rate constraints",
        },
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
            "excluded": ["grf_12"],
            "terrain_heights": {
                "dimension": 143,
                "critic_slice": [145, 288],
                "target_slice": [133, 276],
                "scaling": "unchanged critic height_measurements observation scale",
            },
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
        with open(path, encoding="utf-8") as stream:
            existing = json.load(stream)
        validate_qp_deployment_contract(existing)
        if existing.get("qp_update") != contract.get("qp_update"):
            raise ValueError("Incompatible existing QP deployment contract; export to a new run directory")
        return path, False
    return path, True


def validate_qp_deployment_contract(contract):
    """Reject old held/active execution contracts rather than reinterpret them."""
    if contract.get("schema_version") != 15:
        raise ValueError("Incompatible HardPACT deployment schema; re-export using the current controller")
    update = contract.get("qp_update")
    if update is not None:
        if update.get("mode") not in ("every_substep", "random_one_substep") or update.get("formulation") != "masked_torque_force_24":
            raise ValueError("Incompatible HardPACT QP execution contract")
    return contract
