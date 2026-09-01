"""Automatic HardPACT physics-head gains and deployment contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os


FOOT_ORDER = ("FR", "FL", "RR", "RL")
WRENCH_ORDER = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
RECONSTRUCTION_INDICES = tuple(range(61)) + tuple(range(73, 145))
RECONSTRUCTION_DIM = len(RECONSTRUCTION_INDICES)


@dataclass(frozen=True)
class PhysicsGainSpec:
    physical_grf: tuple[float, ...]
    model_grf: tuple[float, ...]
    physical_wrench: tuple[float, ...]
    model_wrench: tuple[float, ...]
    source: dict


def _maximum_abs(bounds):
    return max(abs(float(value)) for value in bounds)


def calculate_physics_head_gains(cfg):
    """Calculate and freeze-safe head gains from configured physical ranges."""
    grf_physical = tuple(float(v) for v in cfg.sim.grf.prediction_scale_n) * 4
    grf_obs_scale = float(cfg.normalization.obs_scales.grf)
    wrench_obs_scale = float(cfg.normalization.obs_scales.base_wrench)
    if grf_obs_scale <= 0.0 or wrench_obs_scale <= 0.0:
        raise ValueError("force observation scales must be positive")

    deployment = cfg.deployment_physics
    sustained_force = _maximum_abs(deployment.sustained_force_bounds_n)
    sustained_torque = _maximum_abs(deployment.sustained_torque_bounds_nm)
    planned_mass = tuple(float(v) for v in deployment.planned_added_mass_range_kg)
    maximum_mass_delta = _maximum_abs(planned_mass)
    gravity = tuple(float(v) for v in getattr(cfg.sim, "gravity", (0.0, 0.0, -9.81)))
    com_envelope = (
        float(cfg.domain_rand.com_displacement_x_max),
        float(cfg.domain_rand.com_displacement_y_max),
        float(cfg.domain_rand.com_displacement_z_max),
    )
    include_mass = bool(cfg.domain_rand.randomize_base_mass)
    include_com = bool(cfg.domain_rand.randomize_com_displacement) and include_mass
    mass_force = maximum_mass_delta if include_mass else 0.0
    physical_force = tuple(
        sustained_force + mass_force * abs(component) for component in gravity
    )
    com_moment = 0.0
    if include_com:
        com_moment = (
            maximum_mass_delta
            * math.sqrt(sum(component * component for component in gravity))
            * math.sqrt(sum(component * component for component in com_envelope))
        )
    physical_torque = (sustained_torque + com_moment,) * 3
    physical_wrench = physical_force + physical_torque
    source = {
        "backend": "genesis",
        "backend_supported": {
            "base_mass": True,
            "base_com": True,
            "sustained_force": True,
            "sustained_torque": True,
        },
        "sustained_force_bounds_n": list(deployment.sustained_force_bounds_n),
        "sustained_torque_bounds_nm": list(deployment.sustained_torque_bounds_nm),
        "effective_randomization_ranges": {
            "added_mass_kg": [
                float(cfg.domain_rand.added_mass_min),
                float(cfg.domain_rand.min_added_mass_max),
            ],
            "com_xyz_m": [
                [-float(cfg.domain_rand.com_displacement_x_min), float(cfg.domain_rand.com_displacement_x_min)],
                [-float(cfg.domain_rand.com_displacement_y_min), float(cfg.domain_rand.com_displacement_y_min)],
                [-float(cfg.domain_rand.com_displacement_z_min), float(cfg.domain_rand.com_displacement_z_min)],
            ],
        },
        "maximum_planned_curriculum_ranges": {
            "added_mass_kg": list(planned_mass),
            "com_envelope_xyz_m": list(com_envelope),
        },
        "gravity_m_s2": list(gravity),
        "formulas": {
            "force": "max_abs(sustained_force) + max_abs(delta_mass) * abs(gravity_component)",
            "moment": "max_abs(sustained_torque) + max_abs(delta_mass) * norm(gravity) * norm(com_envelope)",
            "model_gain": "physical_gain * observation_scale",
        },
    }
    return PhysicsGainSpec(
        physical_grf=grf_physical,
        model_grf=tuple(v * grf_obs_scale for v in grf_physical),
        physical_wrench=physical_wrench,
        model_wrench=tuple(v * wrench_obs_scale for v in physical_wrench),
        source=source,
    )


def build_deployment_contract(cfg, actor, gain_spec):
    """Build the human- and machine-readable frozen deployment contract."""
    grf_buffer = actor.physics_estimator.grf_scale.detach().cpu().tolist()
    wrench_buffer = actor.physics_estimator.wrench_scale.detach().cpu().tolist()
    return {
        "schema_version": 1,
        "explicit_estimator": {
            "dimension": 11,
            "input": "deterministic_latent_mean",
            "input_dimension": 16,
            "hidden_layers": [
                layer.out_features
                for layer in actor.explicit_estimator.network
                if hasattr(layer, "out_features")
            ][:-1],
            "activation": "ELU",
            "fields": [
                {"name": "base_linear_velocity_body", "dimension": 3, "units": "observation_scaled_m_per_s", "scaling": "obs_scales.lin_vel", "clipping": None},
                {"name": "foot_contact_probability", "dimension": 4, "order": list(FOOT_ORDER), "units": "probability", "scaling": 1.0, "clipping": [0.0, 1.0]},
                {"name": "foot_clearance", "dimension": 4, "order": list(FOOT_ORDER), "units": "m", "scaling": 1.0, "clipping": [-1.0, 1.0]},
            ],
        },
        "latent_dimension": 16,
        "history": {"observation_dimension": 57, "steps": 20},
        "deployment_heads": {
            "hidden_layers": [128, 128],
            "activation": "ELU",
            "grf": {"input_order": ["z_t", "stopgrad(explicit_t)", "tau_nom"], "input_dimension": 39, "output_dimension": 12},
            "base_wrench": {"input_order": ["z_t", "stopgrad(explicit_t)"], "input_dimension": 27, "output_dimension": 6},
        },
        "frames_and_units": {
            "grf": {"frame": "yaw_local", "units": "N", "foot_order": list(FOOT_ORDER), "component_order": ["Fx", "Fy", "Fz"], "target": "deadbanded_clipped_control_interval_average"},
            "base_wrench": {"frame": "yaw_local", "units": ["N", "N", "N", "Nm", "Nm", "Nm"], "order": list(WRENCH_ORDER), "target": "labeled_sustained_external_wrench"},
        },
        "observation_scales": {
            "grf": float(cfg.normalization.obs_scales.grf),
            "base_wrench": float(cfg.normalization.obs_scales.base_wrench),
        },
        "physical_gains": {"grf_n": list(gain_spec.physical_grf), "base_wrench_n_nm": list(gain_spec.physical_wrench)},
        "model_space_gains": {"grf": grf_buffer, "base_wrench": wrench_buffer},
        "gain_sources": gain_spec.source,
        "reconstruction_target": {
            "dimension": RECONSTRUCTION_DIM,
            "excluded": ["grf_12", "terrain_heights_143"],
            "critic_input_unchanged": True,
        },
        "conversion": {
            "model_output_to_physical": "physical = model_output / observation_scale",
            "physical_target_to_model": "model_target = physical_target * observation_scale",
        },
        "checkpoint_buffer_keys": {
            "grf": "physics_estimator.grf_scale",
            "base_wrench": "physics_estimator.wrench_scale",
        },
    }


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
