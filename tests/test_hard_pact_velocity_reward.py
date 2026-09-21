"""Soft joint-speed penalty uses configured physical limits, not torques."""
from types import SimpleNamespace

import pytest
import torch

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg


def test_soft_velocity_reward():
    env = Go2HardPACT.__new__(Go2HardPACT)
    env.cfg = GO2HardPACTCfg()
    limits = torch.tensor(env.cfg.asset.dof_vel_limits)
    threshold = limits * env.cfg.rewards.soft_dof_vel_limit
    velocity = threshold.repeat(4, 1)
    velocity[1, 0] += 0.25
    velocity[1, 2] += 0.5
    velocity[2] *= -1
    velocity[2, 1] -= 0.75
    velocity[3] += 100
    env.simulator = SimpleNamespace(dof_vel=velocity)
    before = velocity.clone()
    result = env._reward_dof_vel_limits()
    torch.testing.assert_close(result, torch.tensor([0., 0.75, 0.75, 12.]))
    assert torch.equal(velocity, before)
    cached = env._reward_velocity_limits
    torch.testing.assert_close(env._reward_dof_vel_limits(), result)
    assert env._reward_velocity_limits is cached
    assert env.cfg.rewards.scales.dof_vel_limits < 0


def test_velocity_reward_obeys_asset_limits_and_soft_factor():
    env = Go2HardPACT.__new__(Go2HardPACT)
    env.cfg = SimpleNamespace(asset=SimpleNamespace(dof_vel_limits=[10.] * 12),
                              rewards=SimpleNamespace(soft_dof_vel_limit=0.5))
    env.simulator = SimpleNamespace(dof_vel=torch.full((2, 12), 5.25))
    torch.testing.assert_close(env._reward_dof_vel_limits(), torch.full((2,), 3.))
    env.cfg.rewards.soft_dof_vel_limit = 1.0
    assert torch.equal(env._reward_dof_vel_limits(), torch.zeros(2))


def test_invalid_limit_count_fails_clearly():
    env = Go2HardPACT.__new__(Go2HardPACT)
    env.cfg = SimpleNamespace(asset=SimpleNamespace(dof_vel_limits=[10.]))
    env.simulator = SimpleNamespace(dof_vel=torch.zeros(1, 12))
    with pytest.raises(ValueError, match="one limit per joint"):
        env._reward_dof_vel_limits()
