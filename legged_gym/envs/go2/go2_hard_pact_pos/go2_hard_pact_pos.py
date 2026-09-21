"""Legacy-compatible HardPACT position-control task alias."""

import contextlib
import io

from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos import Go2PACTPos
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import (
    install_hard_pact_environment_methods,
)


@install_hard_pact_environment_methods
class Go2HardPACTPos(Go2PACTPos):
    """Legacy Go2 PACTPos with conditioned control-interval GRF targets."""

    _legacy_task_class = Go2PACTPos

    def _call_legacy_with_console_policy(self, method_name, *args, **kwargs):
        method = getattr(self._legacy_task_class, method_name)
        if bool(getattr(self.cfg.sim, "console_debug", False)):
            return method(self, *args, **kwargs)
        # Legacy PACTPos contains a few unconditional diagnostic prints. Keep
        # its calculations byte-for-byte identical while suppressing only
        # those messages for HardPACTPos.
        with contextlib.redirect_stdout(io.StringIO()):
            return method(self, *args, **kwargs)

    def _parse_cfg(self, cfg, sim_device):
        return self._call_legacy_with_console_policy(
            "_parse_cfg", cfg, sim_device
        )

    def step_reward_curriculum(self, num_iters):
        return self._call_legacy_with_console_policy(
            "step_reward_curriculum", num_iters
        )
