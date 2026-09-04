import os
import io
import unittest
from contextlib import redirect_stdout

os.environ.setdefault("SIMULATOR", "genesis_pact")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_hard_pact_aux_tests")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_hard_pact_aux_tests")

import torch

from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
from rsl_rl.algorithms.ppo_pact import PPO_PACT
from rsl_rl.modules.actor_critic_hard_pact import (
    ActorCritic_HardPACT,
    ContextDecoder,
)


def make_modules():
    torch.manual_seed(7)
    actor = ActorCritic_HardPACT(
        num_actor_obs=57, num_critic_obs=95, num_actions=12,
        actor_layers=[32, 16], critic_layers=[32, 16],
        cenet_in_dim=57 * 20, cenet_enc_layers=[32, 16],
        cenet_explicit_layers=[16, 16],
        grf_decoder_layers=[16, 16], wrench_decoder_layers=[16, 16],
    )
    decoder = ContextDecoder(input_dim=27, layers=[32, 24, 16], decode_dim=133)
    return actor, decoder


def make_algorithm(**kwargs):
    actor, decoder = make_modules()
    with redirect_stdout(io.StringIO()):
        return PPO_HardPACT(
            actor, decoder, 181, bard_enabled=False,
            use_adaptive_entropy=False, num_encoder_epochs=1,
            **kwargs,
        )


def make_batch(batch=5):
    torch.manual_seed(11)
    transition = {
        "total_external_wrench_label_yaw_scaled": torch.randn(batch, 6),
        "sustained_wrench_active_mask": torch.tensor(
            [[True], [False], [True], [False], [False]]
        )[:batch],
    }
    explicit_target = torch.randn(batch, 11)
    explicit_target[:, 3:7] = torch.randint(0, 2, (batch, 4)).float()
    return (
        torch.randn(batch, 57 * 20), torch.randn(batch, 133),
        explicit_target, torch.randn(batch, 12),
        torch.ones(batch, 1), torch.randn(batch, 12), transition,
    )


class HardPACTAuxiliaryTests(unittest.TestCase):
    def test_explicit_loss_uses_bce_only_for_contact_probabilities(self):
        prediction = torch.tensor([
            [0.2, -0.3, 0.4, 0.8, 0.2, 0.7, 0.1, 0.5, -0.4, 0.3, -0.2],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        ], requires_grad=True)
        target = torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ], requires_grad=True)
        valid = torch.tensor([[True], [False]])

        actual = PPO_HardPACT._masked_explicit_loss(prediction, target, valid)
        expected_elements = torch.cat((
            (prediction[0, :3] - target[0, :3]).square(),
            torch.nn.functional.binary_cross_entropy(
                prediction[0, 3:7], target[0, 3:7], reduction="none"
            ),
            (prediction[0, 7:11] - target[0, 7:11]).square(),
        ))
        torch.testing.assert_close(actual, expected_elements.mean())
        actual.backward()
        self.assertGreater(prediction.grad.abs().sum().item(), 0)
        self.assertIsNone(target.grad)

    def test_independent_physics_loss_weights_are_configurable(self):
        algorithm = make_algorithm(
            lambda_inverse=0.25, lambda_rollout=1.75,
            lambda_projection=0.5,
        )
        self.assertEqual(algorithm.lambda_inverse, 0.25)
        self.assertEqual(algorithm.lambda_rollout, 1.75)
        self.assertEqual(algorithm.lambda_projection, 0.5)
        inverse = torch.tensor(4.0)
        rollout = torch.tensor(2.0)
        projection = torch.tensor(3.0)
        weighted = algorithm._combine_bard_losses(inverse, rollout, projection)
        torch.testing.assert_close(weighted, torch.tensor(6.0))

        # Reporting retains the per-objective lambda values but excludes both
        # the outer PINN schedule and the separately weighted QP projection.
        unweighted = algorithm._unweighted_pinn_loss(inverse, rollout)
        torch.testing.assert_close(unweighted, torch.tensor(4.5))
        optimized = algorithm._combine_bard_losses(
            inverse, rollout, projection, pinn_weight=0.01
        )
        torch.testing.assert_close(optimized, torch.tensor(1.545))
        for name in ("lambda_inverse", "lambda_rollout", "lambda_projection"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "must be nonnegative"):
                    make_algorithm(**{name: -1.0})

    def test_hard_pact_ppo_is_self_contained(self):
        self.assertNotIn(PPO_PACT, PPO_HardPACT.__mro__)

    def test_output_shapes_ranges_and_stochastic_training_latent(self):
        actor, _ = make_modules()
        history = torch.randn(4, 57 * 20)
        first = actor.cenet_enc_forward(history)
        second = actor.cenet_enc_forward(history)
        self.assertEqual(first[0].shape, (4, 16))
        self.assertEqual(first[3].shape, (4, 11))
        self.assertFalse(torch.equal(first[2], first[0]))
        self.assertFalse(torch.equal(first[2], second[2]))
        self.assertFalse(torch.equal(first[3], second[3]))
        inference_latent, inference_explicit = actor.cenet_enc_inference(history)
        torch.testing.assert_close(inference_latent, first[0])
        torch.testing.assert_close(
            inference_explicit, actor.explicit_estimator(first[0])
        )
        self.assertTrue(torch.all((first[3][:, 3:7] >= 0) & (first[3][:, 3:7] <= 1)))
        self.assertTrue(torch.all(first[3][:, 7:11].abs() <= 1))

    def test_privileged_reconstruction_is_stochastic_and_reparameterized(self):
        algorithm = make_algorithm()
        args = make_batch()
        first = algorithm._compute_auxiliary_loss(*args)
        second = algorithm._compute_auxiliary_loss(*args)
        self.assertFalse(torch.equal(first["reconstruction"], second["reconstruction"]))
        algorithm.auxiliary_optimizer.zero_grad(set_to_none=True)
        first["privileged"].backward()
        mean_grad = algorithm.actor_critic.context_encoder.ce_out_mean.weight.grad
        var_grad = algorithm.actor_critic.context_encoder.ce_out_var[0].weight.grad
        self.assertGreater(mean_grad.abs().sum().item(), 0)
        self.assertGreater(var_grad.abs().sum().item(), 0)

    def test_decoder_gradient_routing_and_explicit_stop_gradient(self):
        algorithm = make_algorithm()
        args = make_batch()
        names = ("explicit", "grf", "wrench_active", "privileged")
        expected = {
            "explicit": (True, False, False, False),
            "grf": (False, True, False, False),
            "wrench_active": (False, False, True, False),
            "privileged": (True, False, False, True),
        }
        modules = (
            algorithm.actor_critic.explicit_estimator,
            algorithm.actor_critic.physics_estimator.grf_head,
            algorithm.actor_critic.physics_estimator.wrench_head,
            algorithm.decoder,
        )
        for name in names:
            algorithm.actor_critic.zero_grad(set_to_none=True)
            algorithm.decoder.zero_grad(set_to_none=True)
            loss = algorithm._compute_auxiliary_loss(*args)[name]
            loss.backward()
            actual = tuple(any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in module.parameters()
            ) for module in modules)
            self.assertEqual(actual, expected[name], name)
        # The physics heads consume stopgrad(e), but both retain z gradients.
        for name in ("grf", "wrench_active"):
            algorithm.actor_critic.zero_grad(set_to_none=True)
            algorithm._compute_auxiliary_loss(*args)[name].backward()
            self.assertIsNone(
                algorithm.actor_critic.explicit_estimator.network[0].weight.grad
            )
            self.assertGreater(
                algorithm.actor_critic.context_encoder.ce_out_mean.weight.grad.abs().sum().item(), 0
            )
            self.assertGreater(
                algorithm.actor_critic.context_encoder.ce_out_var[0].weight.grad.abs().sum().item(), 0
            )

    def test_explicit_decoder_uses_reparameterized_sample_during_training(self):
        algorithm = make_algorithm()
        args = make_batch()
        algorithm.actor_critic.zero_grad(set_to_none=True)
        algorithm._compute_auxiliary_loss(*args)["explicit"].backward()

        # A mean-only explicit estimate has no path to log variance.  A
        # nonzero variance gradient therefore verifies explicit(z), matching
        # the HardPACTPos auxiliary path.
        variance_grad = (
            algorithm.actor_critic.context_encoder.ce_out_var[0].weight.grad
        )
        self.assertIsNotNone(variance_grad)
        self.assertGreater(variance_grad.abs().sum().item(), 0)

    def test_combined_auxiliary_step_updates_shared_trunk_not_actor_or_critic(self):
        algorithm = make_algorithm()
        args = make_batch()
        trunk = algorithm.actor_critic.context_encoder.ce_in.weight
        actor = algorithm.actor_critic.act_trunk[0].weight
        critic = algorithm.actor_critic.critic[0].weight
        before = tuple(value.detach().clone() for value in (trunk, actor, critic))
        algorithm.auxiliary_optimizer.zero_grad(set_to_none=True)
        algorithm._compute_auxiliary_loss(*args)["loss"].backward()
        algorithm.auxiliary_optimizer.step()
        self.assertFalse(torch.equal(before[0], trunk))
        torch.testing.assert_close(before[1], actor)
        torch.testing.assert_close(before[2], critic)

    def test_b1z1_pcgrad_parameter_ownership_topology(self):
        algorithm = make_algorithm()
        ppo_ids = [id(p) for g in algorithm.act_optimizer.optimizer.param_groups for p in g["params"]]
        aux_ids = [id(p) for g in algorithm.auxiliary_optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(ppo_ids), len(set(ppo_ids)))
        self.assertEqual(len(aux_ids), len(set(aux_ids)))
        shared = {
            id(p) for p in (
                list(algorithm.actor_critic.context_encoder.parameters())
                + list(algorithm.actor_critic.explicit_estimator.parameters())
                + list(algorithm.actor_critic.physics_estimator.parameters())
                + list(algorithm.decoder.parameters())
            )
        }
        self.assertEqual(set(aux_ids), shared)
        self.assertTrue(shared <= set(ppo_ids))
        actor_only = {id(p) for p in algorithm.actor_critic.act_trunk.parameters()}
        self.assertTrue(actor_only <= set(ppo_ids))
        self.assertTrue(actor_only.isdisjoint(aux_ids))

    def test_all_auxiliary_decoders_share_configured_learning_rate(self):
        learning_rate = 7.0e-5
        algorithm = make_algorithm(auxiliary_learning_rate=learning_rate)
        parameter_lrs = {
            id(parameter): group["lr"]
            for group in algorithm.auxiliary_optimizer.param_groups
            for parameter in group["params"]
        }
        modules = (
            algorithm.actor_critic.context_encoder,
            algorithm.actor_critic.explicit_estimator,
            algorithm.actor_critic.physics_estimator.grf_head,
            algorithm.actor_critic.physics_estimator.wrench_head,
            algorithm.decoder,
        )
        for module in modules:
            for parameter in module.parameters():
                self.assertAlmostEqual(parameter_lrs[id(parameter)], learning_rate)


if __name__ == "__main__":
    unittest.main()
