"""Regression test for distinct Isaac Lab articulation/contact foot indices."""

from types import SimpleNamespace

import torch

from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos import Go2PACTPos
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT


def test_pact_pos_uses_canonical_contact_indices_for_contacts_and_rewards():
    env = Go2PACTPos.__new__(Go2PACTPos)
    # Both lists represent FR, FL, RR, RL, but address different tensors.
    body_indices = [2, 4, 6, 7]
    contact_indices = [8, 3, 10, 1]
    forces = torch.zeros(1, 12, 3)
    forces[0, contact_indices, 2] = torch.tensor([10.0, 20.0, 30.0, 0.0])
    # A body-index lookup produces the intentionally wrong F,F,F,T pattern.
    forces[0, body_indices, 2] = torch.tensor([0.0, 0.0, 0.0, 50.0])
    feet_pos = torch.tensor([[[0.3, -0.2, 0.0], [0.3, 0.2, 0.0],
                              [-0.3, -0.2, 0.0], [-0.3, 0.2, 0.0]]])
    feet_vel = torch.zeros(1, 4, 3)
    feet_vel[0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    env.simulator = SimpleNamespace(
        feet_indices=body_indices,
        feet_contact_indices=contact_indices,
        link_contact_forces=forces,
        _link_contact_forces=forces,
        feet_pos=feet_pos,
        feet_vel=feet_vel,
        base_pos=torch.zeros(1, 3),
    )
    env.cfg = SimpleNamespace(rewards=SimpleNamespace(
        contact_force_threshold=5.0,
        max_contact_force=15.0,
        support_polygon_sigma=0.01,
    ))

    # Canonical contact order is FR, FL, RR, RL.
    torch.testing.assert_close(
        env._feet_contact_mask(), torch.tensor([[True, True, True, False]])
    )
    # Only canonical FR/FL/RR velocities contribute: 1^2 + 2^2 + 3^2.
    torch.testing.assert_close(env._reward_foot_slip(), torch.tensor([14.0]))
    # Excess magnitudes over 15 N are 0, 5, 15, 0 N.
    torch.testing.assert_close(
        env._reward_feet_contact_forces(), torch.tensor([20.0])
    )
    # Three canonical stance feet form a support region. The erroneous body
    # indices expose only one stance foot and would return exactly zero.
    assert env._reward_support_polygon().item() > 0.0


def test_hard_pact_overreach_uses_contact_not_articulation_indices():
    env = Go2HardPACT.__new__(Go2HardPACT)
    body_indices = [2, 4, 6, 7]
    contact_indices = [8, 3, 10, 1]
    forces = torch.zeros(1, 12, 3)
    # Canonical sensor contacts: FR and RR. Articulation-index lookup would
    # incorrectly report FL and RL instead, whose overreach errors differ.
    forces[0, contact_indices, 2] = torch.tensor([10.0, 0.0, 10.0, 0.0])
    forces[0, body_indices, 2] = torch.tensor([0.0, 10.0, 0.0, 10.0])
    env.simulator = SimpleNamespace(
        feet_indices=body_indices,
        feet_contact_indices=contact_indices,
        link_contact_forces=forces,
        feet_pos=torch.tensor([[
            [0.5, -0.2, 0.0], [0.7, 0.2, 0.0],
            [-0.4, -0.2, 0.0], [-0.7, 0.2, 0.0],
        ]]),
        base_pos=torch.zeros(1, 3),
        base_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    )
    env.cfg = SimpleNamespace(rewards=SimpleNamespace(
        front_foot_x_nominal=0.2,
        foot_x_margin=0.1,
        rear_foot_x_nominal=-0.2,
        rear_foot_x_margin=0.1,
    ))

    torch.testing.assert_close(
        env._reward_front_foot_overreach(), torch.tensor([0.04])
    )
    torch.testing.assert_close(
        env._reward_rear_foot_overreach(), torch.tensor([0.01])
    )
