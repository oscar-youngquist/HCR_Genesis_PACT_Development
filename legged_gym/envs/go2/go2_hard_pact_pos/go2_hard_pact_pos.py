"""Legacy-compatible HardPACT position-control task alias."""

from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos import Go2PACTPos


class Go2HardPACTPos(Go2PACTPos):
    """Thin task alias retaining the complete legacy Go2 PACTPos behavior."""

