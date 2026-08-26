"""Focused checks for Go2-style B1Z1 PACT branch-weight randomization."""

import unittest
from types import SimpleNamespace

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass
import torch

from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT


class B1Z1PACTWeightRandomizationTests(unittest.TestCase):
    def _make_env(self, balanced_prob, bias=0.2):
        env = B1Z1PACT.__new__(B1Z1PACT)
        env.cfg = SimpleNamespace(control=SimpleNamespace(
            pact_weight_bias_min=bias,
            pact_weight_bias_max=bias,
            pact_balanced_prob=balanced_prob,
        ))
        env.simulator = SimpleNamespace(
            feedforward_tau_weight=torch.full((4, 1), -1.0),
            feedback_tau_weight=torch.full((4, 1), -1.0),
            feedforward_tau_weight_clean=torch.ones(4, 1),
            feedback_tau_weight_clean=torch.ones(4, 1),
        )
        return env

    def test_balanced_environments_keep_clean_weights(self):
        env = self._make_env(balanced_prob=1.0)
        env._randomize_pact_torque_weights(torch.arange(4))

        self.assertTrue(torch.equal(env.simulator.feedforward_tau_weight, torch.ones(4, 1)))
        self.assertTrue(torch.equal(env.simulator.feedback_tau_weight, torch.ones(4, 1)))

    def test_bias_is_equal_opposite_and_subset_scoped(self):
        env = self._make_env(balanced_prob=0.0)
        env_ids = torch.tensor([1, 3])
        env._randomize_pact_torque_weights(env_ids)

        feedforward = env.simulator.feedforward_tau_weight[env_ids, 0]
        feedback = env.simulator.feedback_tau_weight[env_ids, 0]
        self.assertTrue(torch.allclose(feedforward + feedback, torch.full((2,), 2.0)))
        self.assertTrue(torch.allclose((feedforward - feedback).abs(), torch.full((2,), 0.4)))
        self.assertEqual(env.simulator.feedforward_tau_weight[0].item(), -1.0)
        self.assertEqual(env.simulator.feedback_tau_weight[2].item(), -1.0)


if __name__ == "__main__":
    unittest.main()
