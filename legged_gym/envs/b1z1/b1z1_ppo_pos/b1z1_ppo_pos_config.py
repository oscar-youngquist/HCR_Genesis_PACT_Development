"""Position-only ablation; all non-ablation settings remain inherited from PACT."""
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg, B1Z1PACTCfgPPO


class B1Z1PPOPosCfg(B1Z1PACTCfg):
    position_only = True
    use_force_compensation = False
    use_force_shifted_target = False

    class env(B1Z1PACTCfg.env):
        num_policy_actions = B1Z1PACTCfg.env.num_actions


class B1Z1PPOPosCfgPPO(B1Z1PACTCfgPPO):
    runner_class_name = "B1Z1PPOPosRunner"

    class policy(B1Z1PACTCfgPPO.policy):
        use_context_encoder = False
        use_explicit_decoder = False
        use_physics_decoder = False
        use_privileged_decoder = False
        use_film = False
        use_torque_head = False

    class algorithm(B1Z1PACTCfgPPO.algorithm):
        use_reconstruction_losses = False
        use_vae_kl = False
        use_encoder_pinn = False
        use_actor_physics_loss = False

    class runner(B1Z1PACTCfgPPO.runner):
        policy_class_name = "ActorCriticB1Z1PPOPos"
        algorithm_class_name = "PPO_B1Z1PPOPos"
        experiment_name = "b1z1_ppo_pos"
        run_name = "position_only"
