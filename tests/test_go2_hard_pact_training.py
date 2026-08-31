import unittest

import torch

from rsl_rl.algorithms.ppo_go2_hard_pact import (
    PhysicsLosses,
    PPOGo2HardPACT,
)
from rsl_rl.algorithms.pc_grad import PCGrad
from rsl_rl.go2_hard_pact_schema import TRANSITION_FIELD_DIMS
from rsl_rl.modules.actor_critic_go2_hard_pact import ActorCriticGo2HardPACT


class LegacyPACTTrainingLifecycleTests(unittest.TestCase):
    def test_adaptive_entropy_matches_go2_pact_schedule(self):
        policy = ActorCriticGo2HardPACT(
            actor_layers=(16,), critic_layers=(16,), encoder_layers=(16,),
            physics_head_layers=(16,),
        )
        algorithm = PPOGo2HardPACT(
            policy,
            entropy_coef=0.01,
            use_adaptive_entropy=True,
            adaptive_ent_bounds=(0.005, 0.01),
            adaptive_ent_lin_threshold=0.75,
            adaptive_ent_ang_threshold=0.35,
            adaptive_ent_ter_threshold=6.0,
            adaptive_ent_softmax_temp=2.0,
        )
        self.assertAlmostEqual(
            algorithm.update_adaptive_entropy_coef({
                "lin_vel_tracking": 0.75,
                "ang_vel_tracking": 0.35,
                "terrain_level": 6.0,
            }),
            0.005,
        )
        self.assertAlmostEqual(
            algorithm.update_adaptive_entropy_coef({
                "lin_vel_tracking": 0.0,
                "ang_vel_tracking": 0.0,
                "terrain_level": 0.0,
            }),
            0.01,
        )
        algorithm.set_entropy_coef(0.007)
        self.assertEqual(algorithm.current_entropy_coef, 0.007)

    def test_ppo_loss_uses_current_adaptive_entropy_coefficient(self):
        torch.manual_seed(7)
        policy = ActorCriticGo2HardPACT(
            actor_layers=(16,), critic_layers=(16,), encoder_layers=(16,),
            physics_head_layers=(16,),
        )
        algorithm = PPOGo2HardPACT(
            policy, use_adaptive_entropy=True, entropy_coef=0.0
        )
        batch = {
            "observation": torch.randn(2, 57),
            "history": torch.randn(2, 57 * 20),
            "critic_observation": torch.randn(2, 198),
            "raw_action": torch.randn(2, 24),
            "raw_action_log_probability": torch.zeros(2, 1),
            "action_mean": torch.zeros(2, 24),
            "action_std": torch.ones(2, 24),
            "advantage": torch.randn(2, 1),
            "value": torch.randn(2, 1),
            "return": torch.randn(2, 1),
        }
        algorithm.set_entropy_coef(0.0)
        zero_entropy_loss = algorithm._compute_ppo_loss(batch)[0]
        entropy = policy.entropy.mean().detach()
        algorithm.set_entropy_coef(0.2)
        adaptive_entropy_loss = algorithm._compute_ppo_loss(batch)[0]
        torch.testing.assert_close(
            adaptive_entropy_loss,
            zero_entropy_loss - 0.2 * entropy,
        )

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

        def outputs(batch, actor):
            return {}

        def physics_objective(batch, outputs):
            physics_scalar = (
                policy.position_head.weight.square().mean()
                + policy.feedforward_head.weight.square().mean()
            )
            zeros = physics_scalar * 0.0
            return {
                "physics": PhysicsLosses(
                    physics_scalar, zeros, zeros, physics_scalar, {}
                ),
                "actor_auxiliary": physics_scalar,
                "metrics": {},
            }
        algorithm._compute_physics_objective = physics_objective

        actor_updated_before_auxiliary = []

        def auxiliary_outputs(batch, actor):
            actor_updated_before_auxiliary.append(
                not torch.equal(actor.position_head.weight, initial_position)
            )
            return {}

        def auxiliary_objective(batch, outputs):
            return {
                "loss": policy.privileged_decoder[0].weight.square().mean(),
                "metrics": {},
            }
        algorithm._compute_auxiliary_objective = auxiliary_objective

        metrics = algorithm.update(outputs, auxiliary_outputs, iteration=0)
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
