"""Legacy-compatible HardPACT task alias."""

from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
from .grf import HardPACTGRFMixin


class Go2HardPACT(HardPACTGRFMixin, Go2PACT):
    """Legacy Go2 PACT with conditioned control-interval GRF targets."""
