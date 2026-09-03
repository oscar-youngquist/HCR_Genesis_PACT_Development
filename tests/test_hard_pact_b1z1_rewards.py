"""Parity checks for the two B1Z1 rewards ported into HardPACT."""

from types import SimpleNamespace

import torch

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import (
    GO2HardPACTCfg,
)
from legged_gym.envs.go2.go2_pact.go2_pact_config import GO2PACTCfg


def test_b1z1_torque_cancellation_and_swing_clearance_port():
    # Reward selection changes only the HardPACT subclass.
    assert GO2PACTCfg.rewards.scales.torque_conflict_symmetric == -0.1
    assert GO2HardPACTCfg.rewards.scales.torque_conflict_symmetric == 0.0
    assert GO2HardPACTCfg.rewards.scales.torque_cancellation == -0.10

    env = Go2HardPACT.__new__(Go2HardPACT)
    feedback = torch.zeros(2, 12)
    feedforward = torch.zeros(2, 12)
    feedback[0, 0], feedforward[0, 0] = 4.0, 2.0   # aligned: no conflict
    feedback[1, 0], feedforward[1, 0] = 4.0, -2.0  # cancellation = 4 Nm
    env.simulator = SimpleNamespace(
        feedback_torques=feedback,
        feedforward_torques=feedforward,
        feedback_tau_weight=torch.ones(2, 1),
        feedforward_tau_weight=torch.ones(2, 1),
        _motor_strength=torch.ones(2, 12),
        torque_limits=torch.full((12,), 10.0),
    )
    env.cfg = SimpleNamespace(rewards=SimpleNamespace(
        torque_cancellation_deadband=0.03,
        foot_clearance_target=0.09,
        foot_height_offset=0.022,
        foot_clearance_excess_margin=0.10,
        foot_clearance_excess_weight=0.25,
        foot_clearance_tracking_sigma=0.01,
    ))
    cancellation = env._reward_torque_cancellation()
    torch.testing.assert_close(cancellation[0], torch.tensor(0.0))
    torch.testing.assert_close(
        cancellation[1], torch.tensor((0.4 - 0.03) ** 2 / 12)
    )

    # The new clearance reward gates by conditioned swing contact rather than
    # horizontal velocity. A stationary swing foot below target is penalized,
    # while the same error on a stance foot contributes nothing.
    terrain = torch.zeros(2, 4, 3, 3)
    terrain[:, :, 1, 1] = 0.05
    desired = 0.09 + 0.022 + 0.05
    feet = torch.full((2, 4, 3), desired)
    feet[0, 0, 2] = 0.0
    feet[1, 0, 2] = 0.0
    env.simulator.feet_pos = feet
    env.simulator._height_around_feet = terrain
    env.grf_processor = SimpleNamespace(contacts=torch.tensor([
        [True, True, True, True],
        [False, True, True, True],
    ]))
    clearance = env._reward_foot_clearance_terrain_aware()
    torch.testing.assert_close(clearance[0], torch.tensor(1.0))
    assert 0.0 <= float(clearance[1]) < 1.0
