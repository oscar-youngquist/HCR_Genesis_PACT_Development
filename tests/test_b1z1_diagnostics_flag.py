"""Focused checks for disabling non-training B1Z1 UniFP diagnostics."""

import contextlib
import io
import types
import unittest

import torch

from legged_gym.envs.b1z1.training_diagnostics import (
    additional_diagnostics_enabled,
    should_log_episode_reward,
)
from rsl_rl.modules.actor_critic_unifp import ActorCriticUniFP


class DiagnosticsFlagTests(unittest.TestCase):
    def test_episode_logging_retains_only_curriculum_rewards_when_disabled(self):
        env = types.SimpleNamespace(enable_additional_diagnostics=False)
        self.assertFalse(additional_diagnostics_enabled(env))
        self.assertTrue(should_log_episode_reward(env, "tracking_lin_vel_force_world"))
        self.assertTrue(should_log_episode_reward(env, "tracking_ang_vel"))
        self.assertFalse(should_log_episode_reward(env, "ee_orientation"))
        env.enable_additional_diagnostics = True
        self.assertTrue(should_log_episode_reward(env, "ee_orientation"))

    def test_policy_skips_diagnostic_cache_and_architecture_prints(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            actor = ActorCriticUniFP(
                num_obs=12,
                num_privileged_obs=8,
                num_obs_pred=20,
                num_single_obs=4,
                num_actions=3,
                num_privileged_obs_single=8,
                actor_hidden_dims=(16, 8),
                critic_hidden_dims=(16, 8),
                enable_additional_diagnostics=False,
            )
        self.assertEqual(output.getvalue(), "")
        actor.update_distribution(torch.randn(2, 12))
        self.assertIsNone(actor._last_latent_mean)
        self.assertIsNone(actor._last_latent_logvar)


if __name__ == "__main__":
    unittest.main()
