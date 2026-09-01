"""Legacy-compatible HardPACT configuration aliases."""

from legged_gym.envs.go2.go2_pact.go2_pact_config import (
    GO2PACTCfg,
    GO2PACTCfgPPO,
)


class GO2HardPACTCfg(GO2PACTCfg):
    """Legacy PACT configuration plus calibrated Go2 GRF conditioning."""

    class sim(GO2PACTCfg.sim):
        class grf:
            vertical_deadband_n = 3.0
            clip_min_n = -250.0
            clip_max_n = 250.0
            ema_alpha = 0.30
            contact_threshold_n = 5.0
            use_ema_grfs_buf = False


class GO2HardPACTCfgPPO(GO2PACTCfgPPO):
    """Inherit the legacy PACT training configuration unchanged."""
