"""Architecture-compatible neutral position-policy pretraining profile."""

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import (
    GO2HardPACTCfg,
    GO2HardPACTCfgPPO,
)


class GO2HardPACTPosCfg(GO2HardPACTCfg):
    class features(GO2HardPACTCfg.features):
        # Neutral means neutral for every factorial method. A study which wants
        # physics-head/QP pretraining must opt in explicitly for all variants.
        supervised_physics_head_pretraining = False
        use_bard_inverse_loss = False
        use_bard_rollout_loss = False
        use_qp = False
        differentiate_qp = False

    class bard(GO2HardPACTCfg.bard):
        enabled = False
        required = False

    class qp(GO2HardPACTCfg.qp):
        enabled = False


class GO2HardPACTPosCfgPPO(GO2HardPACTCfgPPO):
    runner_class_name = "Go2HardPACTRunner"

    class policy(GO2HardPACTCfgPPO.policy):
        position_pretraining = True

    class algorithm(GO2HardPACTCfgPPO.algorithm):
        supervised_physics_head_pretraining = False
        use_bard_inverse_loss = False
        use_bard_rollout_loss = False
        use_qp = False
        feedforward_clone_weight = 1.0

    class runner(GO2HardPACTCfgPPO.runner):
        policy_class_name = "ActorCriticGo2HardPACTPos"
        algorithm_class_name = "PPOGo2HardPACT"
        experiment_name = "go2_hard_pact_pos"
        run_name = "neutral_pretraining"
        checkpoint_migration = "none"

