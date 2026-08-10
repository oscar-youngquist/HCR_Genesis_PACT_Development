"""Focused checks for disabling non-training B1Z1 UniFP diagnostics."""

import contextlib
import io
import unittest

import torch

from rsl_rl.modules.actor_critic_unifp import ActorCriticUniFP


class DiagnosticsFlagTests(unittest.TestCase):
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
