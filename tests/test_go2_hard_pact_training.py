import unittest

import torch

from rsl_rl.algorithms.ppo_go2_hard_pact import (
    PPOGo2HardPACT,
    combine_physics_losses,
)
from rsl_rl.algorithms.pc_grad import PCGrad
from rsl_rl.go2_hard_pact_schema import TRANSITION_FIELD_DIMS
from rsl_rl.modules.actor_critic_go2_hard_pact import ActorCriticGo2HardPACT


class LegacyPACTTrainingLifecycleTests(unittest.TestCase):
    def test_act_process_storage_update_lifecycle(self):
        torch.manual_seed(19)
        policy = ActorCriticGo2HardPACT(
            actor_layers=(16,), critic_layers=(16,), encoder_layers=(16,),
            physics_head_layers=(16,),
        )
        algorithm = PPOGo2HardPACT(
            policy, num_learning_epochs=1, num_mini_batches=1,
            entropy_coef=0.0,
        )
        algorithm.init_storage(2, 2, 57, 198, 57 * 20)
        initial_position = policy.position_head.weight.detach().clone()
        initial_decoder = policy.privileged_decoder[0].weight.detach().clone()
        for step in range(2):
            observation = torch.randn(2, 57)
            critic = torch.randn(2, 198)
            history = torch.randn(2, 57 * 20)
            action = algorithm.act(observation, critic, history)
            payload = {}
            for name, width in TRANSITION_FIELD_DIMS.items():
                dtype = torch.long if name in {
                    "qp_fallback", "qp_status",
                } else torch.float32
                payload[name] = torch.zeros(2, width, dtype=dtype)
            payload["delayed_nominal_action"].copy_(action)
            algorithm.process_env_step(
                torch.randn(2), torch.zeros(2, dtype=torch.bool), {}, payload
            )
        algorithm.compute_returns(torch.randn(2, 198))

        def objectives(batch, actor):
            physics_scalar = (
                actor.position_head.weight.square().mean()
                + actor.feedforward_head.weight.square().mean()
            )
            zeros = physics_scalar * 0.0
            return {
                "physics": combine_physics_losses(
                    zeros, zeros, physics_scalar, 0.0, 0.0, 1.0
                ),
                "metrics": {},
            }

        actor_updated_before_auxiliary = []

        def auxiliary(batch, actor):
            actor_updated_before_auxiliary.append(
                not torch.equal(actor.position_head.weight, initial_position)
            )
            return {
                "loss": actor.privileged_decoder[0].weight.square().mean(),
                "metrics": {},
            }

        metrics = algorithm.update(objectives, auxiliary, iteration=0)
        self.assertIn("loss/ppo", metrics)
        self.assertEqual(algorithm.storage.step, 0)
        self.assertTrue(all(actor_updated_before_auxiliary))
        self.assertFalse(torch.equal(policy.privileged_decoder[0].weight, initial_decoder))

    def test_two_optimizer_parameter_ownership(self):
        policy = ActorCriticGo2HardPACT(
            actor_layers=(16,), critic_layers=(16,), encoder_layers=(16,),
            physics_head_layers=(16,),
        )
        algorithm = PPOGo2HardPACT(policy)
        actor_ids = {
            id(parameter)
            for group in algorithm.actor_optimizer.optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(actor_ids, {id(parameter) for parameter in policy.parameters()})
        auxiliary_ids = {
            id(parameter)
            for group in algorithm.auxiliary_optimizer.param_groups
            for parameter in group["params"]
        }
        expected_auxiliary_ids = {
            id(parameter)
            for module in (
                policy.history_encoder,
                policy.privileged_decoder,
                policy.physics_estimator,
            )
            for parameter in module.parameters()
        }
        self.assertEqual(auxiliary_ids, expected_auxiliary_ids)
        self.assertEqual(
            {group["name"] for group in algorithm.auxiliary_optimizer.param_groups},
            {"encoder", "decoder", "estimator", "force_estimator"},
        )

    def test_pcgrad_leaves_parameters_unused_by_all_objectives_untouched(self):
        used = torch.nn.Parameter(torch.tensor([1.0]))
        unused = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = PCGrad(torch.optim.Adam([used, unused]), reduction="sum")
        optimizer.pc_backward_pinn([used.square().sum(), used.sum()])
        self.assertIsNotNone(used.grad)
        self.assertIsNone(unused.grad)


if __name__ == "__main__":
    unittest.main()
