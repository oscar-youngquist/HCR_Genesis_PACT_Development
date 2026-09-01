"""Legacy-compatible HardPACT position-control configuration aliases."""

from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos_config import (
    GO2PACTPosCfg,
    GO2PACTPosCfgPPO,
)


class GO2HardPACTPosCfg(GO2PACTPosCfg):
    """Legacy PACTPos configuration plus calibrated Go2 GRF conditioning."""

    class env(GO2PACTPosCfg.env):
        num_explicit_recon_obs = 11

    class sim(GO2PACTPosCfg.sim):
        class grf:
            prediction_scale_n = [120.0, 120.0, 250.0]
            vertical_deadband_n = 3.0
            clip_min_n = -250.0
            clip_max_n = 250.0
            ema_alpha = 0.20
            contact_threshold_n = 5.0
            use_ema_grfs_buf = False

    class normalization(GO2PACTPosCfg.normalization):
        class obs_scales(GO2PACTPosCfg.normalization.obs_scales):
            base_wrench = 0.01

    class deployment_physics:
        sustained_force_bounds_n = [-60.0, 60.0]
        sustained_torque_bounds_nm = [-12.0, 12.0]
        planned_added_mass_range_kg = [-1.0, 4.0]


class GO2HardPACTPosCfgPPO(GO2PACTPosCfgPPO):
    """Legacy PACTPos architecture with the reduced explicit estimator."""

    class policy(GO2PACTPosCfgPPO.policy):
        cenet_velo_dim = 11
        cenet_explicit_layers = [128, 128]
        grf_decoder_layers = [128, 128]
        wrench_decoder_layers = [128, 128]
        cenet_dec_input_dim = 16 + 11
        cenet_dec_out_dim = 133

    class runner(GO2PACTPosCfgPPO.runner):
        policy_class_name = "ActorCritic_HardPACT_Pos"
