"""Legacy-compatible HardPACT configuration aliases."""

from legged_gym.envs.go2.go2_pact.go2_pact_config import (
    GO2PACTCfg,
    GO2PACTCfgPPO,
)


class GO2HardPACTCfg(GO2PACTCfg):
    """Legacy PACT configuration plus calibrated Go2 GRF conditioning."""

    class env(GO2PACTCfg.env):
        num_explicit_recon_obs = 11

    class sim(GO2PACTCfg.sim):
        class grf:
            prediction_scale_n = [120.0, 120.0, 250.0]
            vertical_deadband_n = 3.0
            clip_min_n = -250.0
            clip_max_n = 250.0
            ema_alpha = 0.30
            contact_threshold_n = 5.0
            use_ema_grfs_buf = False

    class normalization(GO2PACTCfg.normalization):
        class obs_scales(GO2PACTCfg.normalization.obs_scales):
            base_wrench = 0.01

    class deployment_physics:
        sustained_force_bounds_n = [-60.0, 60.0]
        sustained_torque_bounds_nm = [-12.0, 12.0]
        # Stable planned envelope used for the entire run. It intentionally
        # does not follow the simulator's instantaneous curriculum progress.
        planned_added_mass_range_kg = [-1.0, 4.0]


class GO2HardPACTCfgPPO(GO2PACTCfgPPO):
    """Legacy PACT architecture with the reduced explicit estimator."""

    class policy(GO2PACTCfgPPO.policy):
        cenet_velo_dim = 11
        cenet_explicit_layers = [128, 128]
        grf_decoder_layers = [128, 128]
        wrench_decoder_layers = [128, 128]
        cenet_dec_input_dim = 16 + 11
        cenet_dec_out_dim = 133
        pretrained_path = ""

    class runner(GO2PACTCfgPPO.runner):
        policy_class_name = "ActorCritic_HardPACT"
