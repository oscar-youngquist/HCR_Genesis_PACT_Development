import unittest

import torch

from rsl_rl.algorithms.ppo_go2_hard_pact import (
    ReliabilityEMA,
    pcgrad_backward_two_objectives,
    supervised_physics_head_losses,
)
from rsl_rl.modules.actor_critic_go2_hard_pact import (
    ActorCriticGo2HardPACT,
    ActorCriticGo2HardPACTPos,
    migrate_hard_pact_pos_checkpoint,
)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.observation = torch.randn(3, 57)
        self.history = torch.randn(3, 57 * 20)

    def test_concatenation_actor_dimensions_and_deterministic_inference(self):
        policy = ActorCriticGo2HardPACT(actor_layers=(32,), critic_layers=(32,), encoder_layers=(32,), physics_head_layers=(16,))
        first = policy.act_inference(self.observation, self.history)
        second = policy.act_inference(self.observation, self.history)
        self.assertEqual(first.shape, (3, 24))
        torch.testing.assert_close(first, second)
        encoded = policy.encode_policy_history(self.history)
        self.assertEqual(encoded.latent.shape[-1], 16)
        self.assertEqual(encoded.explicit.shape[-1], 11)
        self.assertTrue(((encoded.explicit[:, 3:7] >= 0) & (encoded.explicit[:, 3:7] <= 1)).all())
        self.assertEqual(policy.actor_trunk[0].in_features, 57 + 16 + 11)
        self.assertFalse(any("film" in name.lower() for name, _ in policy.named_modules()))

    def test_sampling_is_auxiliary_only_and_physics_heads_are_separate(self):
        policy = ActorCriticGo2HardPACT(actor_layers=(16,), critic_layers=(16,), encoder_layers=(16,), physics_head_layers=(16,))
        deterministic = policy.encode_policy_history(self.history)
        sampled = policy.encode_auxiliary_history(self.history, sample=True)
        torch.testing.assert_close(deterministic.latent, deterministic.latent_mean)
        self.assertFalse(torch.equal(sampled.latent, sampled.latent_mean))
        references = policy.physics_references(
            self.history, torch.zeros(3, 12), encoded=deterministic
        )
        self.assertEqual(references.grf_yaw_n.shape, (3, 12))
        self.assertEqual(references.base_wrench_yaw.shape, (3, 6))
        self.assertEqual(policy._physics_evaluations, 1)

    def test_strict_position_checkpoint_migration(self):
        position = ActorCriticGo2HardPACTPos(actor_layers=(16,), critic_layers=(16,), encoder_layers=(16,), physics_head_layers=(16,))
        coupled = ActorCriticGo2HardPACT(actor_layers=(16,), critic_layers=(16,), encoder_layers=(16,), physics_head_layers=(16,))
        source = {name: value.clone() for name, value in position.state_dict().items()}
        report = migrate_hard_pact_pos_checkpoint(coupled, {"model_state_dict": source})
        self.assertEqual(report.reinitialized, ("std[12:24]",))
        torch.testing.assert_close(coupled.std[:12], position.std)
        bad = dict(source)
        bad.pop("feedforward_head.weight")
        with self.assertRaisesRegex(RuntimeError, "undocumented missing"):
            migrate_hard_pact_pos_checkpoint(coupled, bad)


class LossTests(unittest.TestCase):
    def test_grf_uses_all_transitions_and_wrench_events_are_split(self):
        zeros_grf = torch.zeros(4, 12)
        predicted_grf = zeros_grf.clone()
        predicted_grf[0, 0] = 120.0
        predicted_wrench = torch.ones(4, 6)
        target_wrench = torch.zeros_like(predicted_wrench)
        active = torch.tensor([[1], [0], [1], [0]], dtype=torch.bool)
        losses = supervised_physics_head_losses(
            predicted_grf, zeros_grf, predicted_wrench, target_wrench, active
        )
        self.assertGreater(losses["grf"].item(), 0.0)
        self.assertGreater(losses["wrench_active"].item(), 0.0)
        self.assertGreater(losses["wrench_neutral"].item(), 0.0)

    def test_pcgrad_projects_only_a_conflict(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        conflict = pcgrad_backward_two_objectives(
            parameter.square().sum(), -parameter.sum(), [parameter]
        )
        self.assertTrue(conflict.conflict)
        torch.testing.assert_close(parameter.grad, torch.tensor([2.0]))
        compatible = pcgrad_backward_two_objectives(
            parameter.square().sum(), parameter.sum(), [parameter]
        )
        self.assertFalse(compatible.conflict)
        torch.testing.assert_close(parameter.grad, torch.tensor([3.0]))

    def test_reliability_updates_once_per_iteration(self):
        ema = ReliabilityEMA(alpha=0.5)
        self.assertTrue(ema.update(1, {"grf": 2.0}))
        self.assertFalse(ema.update(1, {"grf": 100.0}))
        self.assertEqual(ema.values["grf"], 2.0)
        ema.update(2, {"grf": 4.0})
        self.assertEqual(ema.values["grf"], 3.0)


if __name__ == "__main__":
    unittest.main()
