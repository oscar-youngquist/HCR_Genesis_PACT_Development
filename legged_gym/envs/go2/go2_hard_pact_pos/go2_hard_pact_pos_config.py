"""Legacy-compatible HardPACT position-control configuration aliases."""

from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos_config import (
    GO2PACTPosCfg,
    GO2PACTPosCfgPPO,
)


class GO2HardPACTPosCfg(GO2PACTPosCfg):
    """Legacy PACTPos configuration plus calibrated Go2 GRF conditioning."""

    class sim(GO2PACTPosCfg.sim):
        class grf:
            vertical_deadband_n = 3.0
            clip_min_n = -250.0
            clip_max_n = 250.0
            ema_alpha = 0.20
            contact_threshold_n = 5.0
            use_ema_grfs_buf = False


class GO2HardPACTPosCfgPPO(GO2PACTPosCfgPPO):
    """Inherit the legacy PACTPos training configuration unchanged."""
