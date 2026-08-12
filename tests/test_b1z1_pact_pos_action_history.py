"""Focused tests for PACT-Pos's combined position/torque action history."""

import unittest
from types import SimpleNamespace

# Isaac Gym requires its extension to load before PyTorch in Python 3.8.
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass
import torch

from legged_gym.envs.b1z1.b1z1_pact_pos.b1z1_pact_pos import B1Z1PACTPos
from rsl_rl.modules.actor_critic_b1z1_pact_pos import ActorCriticB1Z1PACTPos
from rsl_rl.storage.rollout_storage_b1z1_pact import RolloutStorageB1Z1PACT


class PACTPosActionHistoryTests(unittest.TestCase):
    def test_pre_sim_step_stores_both_heads_and_executes_position_only(self):
        env = B1Z1PACTPos.__new__(B1Z1PACTPos)
        env.device = "cpu"
        env.num_actions = 3
        env.cfg = SimpleNamespace(
            env=SimpleNamespace(num_policy_actions=6),
            normalization=SimpleNamespace(clip_actions=1.0),
            domain_rand=SimpleNamespace(randomize_ctrl_delay=False),
        )
        env.actions = torch.zeros(2, 6)
        env.last_actions = torch.zeros_like(env.actions)
        env.llast_actions = torch.zeros_like(env.actions)

        complete_action = torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
        ])
        executed = env._pre_sim_step(complete_action)

        self.assertTrue(torch.equal(env.actions, complete_action))
        self.assertTrue(torch.equal(executed, complete_action[:, :3]))

    def test_inference_returns_position_then_torque_heads(self):
        actor = ActorCriticB1Z1PACTPos(
            num_actor_obs=12, num_critic_obs=8, num_actions=4, history_dim=10,
            latent_dim=3, actor_layers=(16, 8), critic_layers=(16, 8),
            context_layers=(12, 8), explicit_decoder_layers=(8,),
            film_hidden_dim=8,
        )
        actions = actor.act_inference(torch.randn(2, 12), torch.randn(2, 10))

        self.assertEqual(actions.shape, (2, 8))
        self.assertTrue(torch.equal(actions[:, :4], actor.last_position_mean))
        self.assertTrue(torch.equal(actions[:, 4:], actor.last_torque_mean))

    def test_storage_separates_action_and_distribution_widths(self):
        storage = RolloutStorageB1Z1PACT(
            2, 3, 5, 7, 11, 8, 4, 7, 9,
            policy_distribution_dim=4,
        )

        self.assertEqual(storage.actions.shape[-1], 8)
        self.assertEqual(storage.mu.shape[-1], 4)
        self.assertEqual(storage.sigma.shape[-1], 4)


if __name__ == "__main__":
    unittest.main()
