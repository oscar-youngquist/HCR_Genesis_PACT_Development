"""Small synthetic reset tests of both real legacy terrain decision paths."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos import Go2HardPACTPos
from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos import Go2PACTPos


@pytest.mark.parametrize("cls", [Go2HardPACT, Go2HardPACTPos, Go2PACT, Go2PACTPos])
@pytest.mark.parametrize("delay,iteration,allowed", [(0, 0, True), (10, 0, False),
                                                   (10, 9, False), (10, 10, True),
                                                   (10, 25, True)])
def test_upward_delay_preserves_demotions_and_legacy(cls, delay, iteration, allowed):
    env = cls.__new__(cls)
    env.init_done = True
    env.cfg = SimpleNamespace(terrain=SimpleNamespace(curriculum_upward_delay_iterations=delay))
    # Same absolute iteration is supplied at collection start on fresh/resumed runs.
    env._terrain_curriculum_iteration = iteration
    env.commands = torch.tensor([[1., 0.], [1., 0.], [1., 0.]])
    env.max_episode_length_s = 4.
    env.simulator = SimpleNamespace(
        base_pos=torch.tensor([[6., 0., 0.], [.5, 0., 0.], [3., 0., 0.]]),
        env_origins=torch.zeros(3, 3), _terrain=SimpleNamespace(env_length=10.),
        _added_base_mass=torch.zeros(3, 1), _robot_mass=15.,
        update_terrain_curriculum=Mock(),
    )
    ids = torch.arange(3)
    env._update_terrain_curriculum(ids)
    _, up, down = env.simulator.update_terrain_curriculum.call_args.args
    is_hard = cls in (Go2HardPACT, Go2HardPACTPos)
    assert up.tolist() == [allowed if is_hard else True, False, False]
    assert down.tolist() == [False, True, False]
    # Initial reset never advances or demotes, regardless of the delay.
    env.simulator.update_terrain_curriculum.reset_mock()
    env.init_done = False
    env._update_terrain_curriculum(ids)
    env.simulator.update_terrain_curriculum.assert_not_called()
