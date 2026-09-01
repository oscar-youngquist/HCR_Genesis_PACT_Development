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


def make_algorithm():
    actor, decoder = make_modules()
    with redirect_stdout(io.StringIO()):
        return PPO_HardPACT(
            actor, decoder, 181, bard_enabled=False,
            use_adaptive_entropy=False, num_encoder_epochs=1,
        )


def make_batch(batch=5):
    torch.manual_seed(11)
    transition = {
        "total_external_wrench_label_yaw_scaled": torch.randn(batch, 6),
        "sustained_wrench_active_mask": torch.tensor(
            [[True], [False], [True], [False], [False]]
        )[:batch],
    }
    return (
        torch.randn(batch, 57 * 20), torch.randn(batch, 133),
        torch.randn(batch, 11), torch.randn(batch, 12),
        torch.ones(batch, 1), torch.randn(batch, 12), transition,
    )


class HardPACTAuxiliaryTests(unittest.TestCase):
    def test_hard_pact_ppo_is_self_contained(self):
        self.assertNotIn(PPO_PACT, PPO_HardPACT.__mro__)

    def test_output_shapes_ranges_and_deterministic_runtime(self):
        actor, _ = make_modules()
        history = torch.randn(4, 57 * 20)
        first = actor.cenet_enc_forward(history)
        second = actor.cenet_enc_forward(history)
        self.assertEqual(first[0].shape, (4, 16))
        self.assertEqual(first[3].shape, (4, 11))
        torch.testing.assert_close(first[2], first[0])
        torch.testing.assert_close(first[2], second[2])
        torch.testing.assert_close(first[3], second[3])
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


if __name__ == "__main__":
    unittest.main()
