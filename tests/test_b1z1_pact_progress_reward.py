"""Focused tests for B1Z1 PACT rolling physical-progress rewards."""

import types
import unittest

import torch

from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
from legged_gym.envs.b1z1.b1z1_pact_pos.b1z1_pact_pos import B1Z1PACTPos


class PACTPhysicalProgressTests(unittest.TestCase):
    ENV_CLASSES = (B1Z1PACT, B1Z1PACTPos)

    @staticmethod
    def _environment(env_class):
        env = object.__new__(env_class)
        env.num_envs = 2
        env.device = torch.device("cpu")
        env.dt = 0.02
        env.common_step_counter = 7
        env._progress_update_step = 7
        env.progress_window_steps = 20
        env.progress_buffer_index = 0
        env.progress_delta_buffer = torch.zeros(2, 20, 2)
        env.progress_desired_buffer = torch.zeros_like(env.progress_delta_buffer)
        env.progress_valid_steps = torch.full((2,), 10, dtype=torch.long)
        env.last_progress_base_pos = torch.zeros(2, 2)
        env.commands = torch.zeros(2, 6)
        env.commands[:, 0] = 0.5
        env.progress_desired_buffer[:, :, 0] = 0.01
        # The first environment reaches 40% of the commanded window distance;
        # the second remains stationary despite receiving the same command.
        env.progress_delta_buffer[0, 0, 0] = 0.08
        env.simulator = types.SimpleNamespace(
            base_pos=torch.zeros(2, 3),
            foot_contacts=torch.ones(2, 4, dtype=torch.bool),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]] * 2),
        )
        return env

    def test_progress_and_stall_are_distinguished(self):
        for env_class in self.ENV_CLASSES:
            env = self._environment(env_class)
            penalty = env._reward_no_physical_progress()
            self.assertTrue(torch.allclose(penalty, torch.tensor([0.0, 1.0])))

    def test_standing_and_unsupported_motion_are_masked_correctly(self):
        for env_class in self.ENV_CLASSES:
            env = self._environment(env_class)
            env.commands[1, :2] = 0.0
            self.assertEqual(env._reward_no_physical_progress()[1].item(), 0.0)

            env = self._environment(env_class)
            env.simulator.foot_contacts[0] = False
            self.assertEqual(env._reward_no_physical_progress()[0].item(), 1.0)

    def test_reset_clears_only_selected_environment_history(self):
        for env_class in self.ENV_CLASSES:
            env = self._environment(env_class)
            env._reset_progress_statistics(torch.tensor([0]))
            self.assertEqual(env.progress_valid_steps.tolist(), [0, 10])
            self.assertEqual(env.progress_delta_buffer[0].count_nonzero().item(), 0)
            self.assertGreater(env.progress_desired_buffer[1].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
