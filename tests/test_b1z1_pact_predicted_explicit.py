"""Focused checks for predicted-only B1Z1 PACT explicit conditioning."""

import unittest

import torch

from rsl_rl.modules.actor_critic_b1z1_pact_pos import ActorCriticB1Z1PACTPos


class PredictedExplicitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.actor = ActorCriticB1Z1PACTPos(
            num_actor_obs=12, num_critic_obs=8, num_actions=4, history_dim=10,
            latent_dim=3, actor_layers=(16, 8), critic_layers=(16, 8),
            context_layers=(12, 8), explicit_decoder_layers=(8,),
            film_hidden_dim=8, init_noise_std=0.5, min_noise_std=0.1,
        )

    def test_distribution_is_deterministic_for_fixed_history(self):
        obs, history = torch.randn(5, 12), torch.randn(5, 10)
        self.actor.update_distribution(obs, history, sample_context=False)
        first = self.actor.action_mean.detach().clone()
        self.actor.update_distribution(obs, history, sample_context=False)
        self.assertTrue(torch.equal(first, self.actor.action_mean))

    def test_actor_and_film_receive_same_predicted_context(self):
        obs, history = torch.randn(5, 12), torch.randn(5, 10)
        context = self.actor.decode_context(self.actor.context_encoder(history, sample=False))
        actor_input, film_condition = self.actor._actor_inputs(obs, context, context)
        self.assertEqual(actor_input.shape[0], obs.shape[0])
        expected_film = torch.cat((
            context["base_wrench"], context["ee_force"],
            obs[:, -6:-3] - context["base_velocity"],
            obs[:, -3:] - context["ee_position"],
        ), dim=-1)
        self.assertTrue(torch.equal(film_condition, expected_film))


if __name__ == "__main__":
    unittest.main()
