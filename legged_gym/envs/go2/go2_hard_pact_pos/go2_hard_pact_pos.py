"""Thin architecture-compatible Go2 HardPACT position pretraining task."""

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACTCore


class Go2HardPACTPos(Go2HardPACTCore):
    position_pretraining = True

