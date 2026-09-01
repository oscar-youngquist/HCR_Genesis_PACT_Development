"""Legacy-compatible HardPACT position-control task alias."""

from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos import Go2PACTPos
from legged_gym.envs.go2.go2_hard_pact.grf import HardPACTGRFMixin


class Go2HardPACTPos(HardPACTGRFMixin, Go2PACTPos):
    """Legacy Go2 PACTPos with conditioned control-interval GRF targets."""
