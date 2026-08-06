"""Focused CPU tests for the PACT-position explicit-context curriculum."""

import math
import random
import unittest

import torch

from rsl_rl.algorithms.ppo_b1z1_pact_pos import PPO_B1Z1PACTPos
from rsl_rl.modules.actor_critic_b1z1_pact_pos import ActorCriticB1Z1PACTPos


class ExplicitBlendTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.actor = ActorCriticB1Z1PACTPos(
            num_actor_obs=12, num_critic_obs=8, num_actions=4, history_dim=10,
            latent_dim=3, actor_layers=(16, 8), critic_layers=(16, 8),
            context_layers=(12, 8), explicit_decoder_layers=(8,),
            film_hidden_dim=8, init_noise_std=0.5, min_noise_std=0.1,
        )
        self.context = self.actor.decode_context(
            self.actor.context_encoder(torch.randn(5, 10), sample=False)
        )
        self.labels = torch.randn(5, 23)
        self.labels[:, 15:19] = torch.randint(0, 2, (5, 4)).float()

    def test_blend_endpoints_and_midpoint(self):
        predicted = self.actor.explicit_vector(self.context)
        ground = self.actor.explicit_vector(
            self.actor.blend_explicit_context(self.context, self.labels, 0.0)
        )
        predicted_endpoint = self.actor.explicit_vector(
            self.actor.blend_explicit_context(self.context, self.labels, 1.0)
        )
        midpoint = self.actor.explicit_vector(
            self.actor.blend_explicit_context(self.context, self.labels, 0.5)
        )
        # Contacts pass through the same finite-logit clamp used by the actor.
        expected_ground = self.labels.clone()
        expected_ground[:, 15:19].clamp_(1.0e-4, 1.0 - 1.0e-4)
        expected_midpoint = 0.5 * (self.labels + predicted)
        expected_midpoint[:, 15:19].clamp_(1.0e-4, 1.0 - 1.0e-4)
        self.assertTrue(torch.allclose(ground, expected_ground, atol=1.0e-7))
        self.assertTrue(torch.allclose(predicted_endpoint, predicted, atol=1.0e-7))
        self.assertTrue(torch.allclose(midpoint, expected_midpoint, atol=1.0e-7))

    def test_stored_alpha_controls_distribution_reconstruction(self):
        obs, history = torch.randn(5, 12), torch.randn(5, 10)
        alpha = torch.full((5, 1), 0.35)
        self.actor.update_distribution(
            obs, history, sample_context=False, explicit_labels=self.labels,
            mask_latent=True, explicit_blend_alpha=alpha,
        )
        first = self.actor.action_mean.detach().clone()
        self.actor.update_distribution(
            obs, history, sample_context=False, explicit_labels=self.labels,
            mask_latent=True, explicit_blend_alpha=alpha,
        )
        self.assertTrue(torch.equal(first, self.actor.action_mean))

    def test_endpoint_policy_kl_is_deterministic_and_changes_only_with_explicit(self):
        algorithm = PPO_B1Z1PACTPos.__new__(PPO_B1Z1PACTPos)
        algorithm.actor_critic = self.actor
        algorithm.use_boot_latent = False
        obs, history = torch.randn(5, 12), torch.randn(5, 10)
        context = self.actor.decode_context(self.actor.context_encoder(history, sample=False))
        labels = self.actor.explicit_vector(context).detach()
        batch = {"observations": obs, "histories": history, "actor_explicit_labels": labels}
        torch_state = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        identical = algorithm._explicit_policy_diagnostics(batch)
        self.assertLess(identical["explicit_policy_kl"], 1.0e-7)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        perturbed = dict(batch)
        perturbed["actor_explicit_labels"] = labels + 0.5
        changed = algorithm._explicit_policy_diagnostics(perturbed)
        self.assertGreater(changed["explicit_policy_kl"], 0.0)


class ExplicitCurriculumTests(unittest.TestCase):
    def make_algorithm(self):
        algorithm = PPO_B1Z1PACTPos.__new__(PPO_B1Z1PACTPos)
        algorithm.explicit_blend_alpha = 0.0
        algorithm.explicit_blend_max_alpha = 1.0
        algorithm.explicit_kl_ema_decay = 0.5
        algorithm.explicit_kl_low_threshold = 0.005
        algorithm.explicit_kl_high_threshold = 0.015
        algorithm.explicit_alpha_increment = 0.1
        algorithm.explicit_alpha_decrement = 0.2
        algorithm.explicit_alpha_warmup_updates = 2
        algorithm.explicit_alpha_required_stable_updates = 2
        algorithm.explicit_kl_ema = None
        algorithm.explicit_kl_stable_updates = 0
        return algorithm

    def test_ema_warmup_hysteresis_and_bounds(self):
        algorithm = self.make_algorithm()
        algorithm._update_explicit_blend_curriculum(0, 0.004)
        self.assertEqual(algorithm.explicit_blend_alpha, 0.0)
        self.assertEqual(algorithm.explicit_kl_ema, 0.004)
        algorithm._update_explicit_blend_curriculum(2, 0.004)
        self.assertEqual(algorithm.explicit_blend_alpha, 0.0)
        algorithm._update_explicit_blend_curriculum(3, 0.004)
        self.assertEqual(algorithm.explicit_blend_alpha, 0.1)
        algorithm._update_explicit_blend_curriculum(4, 0.02)
        self.assertAlmostEqual(algorithm.explicit_kl_ema, 0.012, places=7)
        self.assertEqual(algorithm.explicit_blend_alpha, 0.1)
        algorithm._update_explicit_blend_curriculum(5, 0.03)
        self.assertGreaterEqual(algorithm.explicit_kl_ema, 0.015)
        self.assertEqual(algorithm.explicit_blend_alpha, 0.0)
        algorithm.explicit_blend_alpha = 0.99
        algorithm.explicit_kl_ema = 0.0
        algorithm.explicit_kl_stable_updates = 1
        algorithm._update_explicit_blend_curriculum(6, 0.0)
        self.assertEqual(algorithm.explicit_blend_alpha, 1.0)

    def test_nonfinite_kl_decreases_without_rng_use(self):
        algorithm = self.make_algorithm()
        algorithm.explicit_blend_alpha = 0.5
        state = random.getstate()
        algorithm._update_explicit_blend_curriculum(3, math.nan)
        self.assertEqual(state, random.getstate())
        self.assertEqual(algorithm.explicit_blend_alpha, 0.3)


if __name__ == "__main__":
    unittest.main()
