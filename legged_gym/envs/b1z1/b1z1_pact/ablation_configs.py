"""Thin PACT selectors; all variants inherit PACT control gains."""
from rsl_rl.b1z1_pact_ablations import B1Z1_PACT_ABLATIONS
from .b1z1_pact_config import B1Z1PACTCfg, B1Z1PACTCfgPPO
from legged_gym.envs.b1z1.b1z1_unifp_original.b1z1_unifp_original_config import copy_sections


def make_b1z1_pact_ablation_configs(variant_id):
    features = B1Z1_PACT_ABLATIONS[variant_id]

    class EnvCfg(B1Z1PACTCfg):
        def __init__(self):
            # Own nested lists/dicts, including profile ranges and control gains.
            self.__dict__.update(vars(copy_sections(type(self))))

        ablation_variant = variant_id
        action_mode = features.action_mode

        class compliance(B1Z1PACTCfg.compliance):
            # Recruitment only: target_path_allowed/bounds/filtering remain active.
            if features.recruitment in ("workspace", "none"):
                joint_gate = False
                collision_gate = False
            if features.recruitment == "none":
                stiffness_gate = False
                workspace_gate = False

        class env(B1Z1PACTCfg.env):
            num_policy_actions = B1Z1PACTCfg.env.num_actions * (1 if features.action_mode == "position" else 2)

    class PPOCfg(B1Z1PACTCfgPPO):
        def __init__(self):
            self.__dict__.update(vars(copy_sections(type(self))))

        ablation_variant = variant_id

        class policy(B1Z1PACTCfgPPO.policy):
            action_mode = features.action_mode
            conditioning_mode = features.conditioning_mode
            if variant_id == 5:
                concat_conditioning_version = 2
            if features.conditioning_mode != "film":
                film_identity_loss_weight = 0.0

        class algorithm(B1Z1PACTCfgPPO.algorithm):
            representation_pinn_enabled = features.representation_pinn_enabled
            actor_phys_enabled = features.actor_phys_enabled
            if features.action_mode == "position" or not features.actor_phys_enabled:
                actor_phys_force_allocation_weight = 0.0
                actor_phys_arm_feedforward_enabled = False
            if not features.actor_phys_enabled:
                actor_phys_base_balance_enabled = False
            if not features.channel_physics_enabled:
                actor_phys_pos_fk_enabled = False
                # Master enable gates both legacy and compliant-local FF teachers.
                actor_phys_arm_feedforward_enabled = False

        class runner(B1Z1PACTCfgPPO.runner):
            run_name = features.task_name
            experiment_name = features.task_name

    EnvCfg.__name__ = f"B1Z1PACTAb{variant_id}Cfg"
    PPOCfg.__name__ = EnvCfg.__name__ + "PPO"
    return EnvCfg, PPOCfg
