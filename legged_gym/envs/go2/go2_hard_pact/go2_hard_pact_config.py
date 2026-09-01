"""Legacy-compatible HardPACT configuration aliases."""

from legged_gym.envs.go2.go2_pact.go2_pact_config import (
    GO2PACTCfg,
    GO2PACTCfgPPO,
)


class GO2HardPACTCfg(GO2PACTCfg):
    """Inherit the legacy PACT environment configuration unchanged."""


class GO2HardPACTCfgPPO(GO2PACTCfgPPO):
    """Inherit the legacy PACT training configuration unchanged."""

