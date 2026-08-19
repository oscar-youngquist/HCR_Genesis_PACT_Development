"""Focused checks for coupled B1Z1 PACT action and torque-branch rewards."""

import unittest
from types import SimpleNamespace

# Isaac Gym must load before torch in its Python environment.
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass
import torch

from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT


class B1Z1PACTBranchRewardTests(unittest.TestCase):
    def setUp(self):
        self.env = B1Z1PACT.__new__(B1Z1PACT)
        self.env.num_actions = 17
        self.env.actions = torch.zeros(1, 34)
        self.env.last_actions = torch.zeros_like(self.env.actions)
        self.env.llast_actions = torch.zeros_like(self.env.actions)
        self.env.simulator = SimpleNamespace(
            combined_feedback_torques=torch.zeros(1, 19),
            combined_feedforward_torques=torch.zeros(1, 19),
        )

    def test_action_heads_and_joint_groups_are_penalized_independently(self):
        self.env.actions[:, :12] = 1.0
        self.env.actions[:, 12:17] = 2.0
        self.env.actions[:, 17:29] = 3.0
        self.env.actions[:, 29:34] = 4.0

        self.assertEqual(self.env._reward_leg_feedback_action_rate().item(), 12.0)
        self.assertEqual(self.env._reward_arm_feedback_action_rate().item(), 20.0)
        self.assertEqual(self.env._reward_leg_feedforward_action_rate().item(), 108.0)
        self.assertEqual(self.env._reward_arm_feedforward_action_rate().item(), 80.0)
        self.assertEqual(self.env._reward_leg_feedback_action_smoothness().item(), 12.0)
        self.assertEqual(self.env._reward_arm_feedback_action_smoothness().item(), 20.0)
        self.assertEqual(self.env._reward_leg_feedforward_action_smoothness().item(), 108.0)
        self.assertEqual(self.env._reward_arm_feedforward_action_smoothness().item(), 80.0)

    def test_torque_branches_and_joint_groups_are_independent(self):
        self.env.simulator.combined_feedback_torques[:, :12] = 1.0
        self.env.simulator.combined_feedback_torques[:, 12:17] = 2.0
        self.env.simulator.combined_feedforward_torques[:, :12] = 3.0
        self.env.simulator.combined_feedforward_torques[:, 12:17] = 4.0

        self.assertEqual(self.env._reward_leg_feedback_torques().item(), 12.0)
        self.assertEqual(self.env._reward_arm_feedback_torques().item(), 20.0)
        self.assertEqual(self.env._reward_leg_feedforward_torques().item(), 108.0)
        self.assertEqual(self.env._reward_arm_feedforward_torques().item(), 80.0)



if __name__ == "__main__":
    unittest.main()
