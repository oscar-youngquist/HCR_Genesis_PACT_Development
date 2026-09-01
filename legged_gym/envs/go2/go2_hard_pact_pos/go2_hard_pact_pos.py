"""Legacy-compatible HardPACT position-control task alias."""

from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos import Go2PACTPos
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import (
    install_hard_pact_environment_methods,
)


@install_hard_pact_environment_methods
class Go2HardPACTPos(Go2PACTPos):
    """Legacy Go2 PACTPos with conditioned control-interval GRF targets."""

    _legacy_task_class = Go2PACTPos
