"""Legacy-compatible HardPACT task alias."""

from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT


class Go2HardPACT(Go2PACT):
    """Thin task alias retaining the complete legacy Go2 PACT behavior."""

