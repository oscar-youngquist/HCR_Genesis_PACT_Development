"""Thin dynamic selectors inheriting every task/training value from PACT."""
from rsl_rl.b1z1_pact_ablations import B1Z1_PACT_ABLATIONS
from .b1z1_pact_config import B1Z1PACTCfg, B1Z1PACTCfgPPO


def make_b1z1_pact_ablation_configs(variant_id):
    features = B1Z1_PACT_ABLATIONS[variant_id]

    class EnvCfg(B1Z1PACTCfg):
        ablation_variant = variant_id
        action_mode = features.action_mode

        class env(B1Z1PACTCfg.env):
            num_policy_actions = B1Z1PACTCfg.env.num_actions * (1 if features.action_mode == "position" else 2)

    class PPOCfg(B1Z1PACTCfgPPO):
        ablation_variant = variant_id

        class policy(B1Z1PACTCfgPPO.policy):
            action_mode = features.action_mode
            conditioning_mode = features.conditioning_mode
            if features.conditioning_mode != "film":
                film_identity_loss_weight = 0.0

        class algorithm(B1Z1PACTCfgPPO.algorithm):
            representation_pinn_enabled = features.representation_pinn_enabled
            actor_phys_enabled = features.actor_phys_enabled
            if features.action_mode == "position" or not features.actor_phys_enabled:
                actor_phys_force_allocation_weight = 0.0

        class runner(B1Z1PACTCfgPPO.runner):
            run_name = features.task_name
            experiment_name = features.task_name

    EnvCfg.__name__ = f"B1Z1PACTAb{variant_id}Cfg"
    PPOCfg.__name__ = EnvCfg.__name__ + "PPO"
    return EnvCfg, PPOCfg
