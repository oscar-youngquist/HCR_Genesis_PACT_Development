"""Legacy-compatible HardPACT position-control configuration aliases."""

from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos_config import (
    GO2PACTPosCfg,
    GO2PACTPosCfgPPO,
)


class GO2HardPACTPosCfg(GO2PACTPosCfg):
    """Inherit the legacy PACTPos environment configuration unchanged."""


class GO2HardPACTPosCfgPPO(GO2PACTPosCfgPPO):
    """Inherit the legacy PACTPos training configuration unchanged."""

