"""Focused tests for HardPACT deployment physics heads and contract."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
import tempfile
import unittest

os.environ.setdefault("SIMULATOR", "genesis_pact")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_hard_pact_tests")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_hard_pact_tests")

import torch

from legged_gym.envs.go2.go2_hard_pact.transition import DISTURBANCE_CRITIC_DIM
from legged_gym.envs.go2.go2_hard_pact.deployment import (
    FOOT_ORDER,
    RECONSTRUCTION_DIM,
    RECONSTRUCTION_INDICES,
    build_deployment_contract,
    calculate_physics_head_gains,
    write_deployment_contract_once,
)
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import (
    GO2HardPACTCfg,
    GO2HardPACTCfgPPO,
)
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import (
    GO2HardPACTPosCfg,
    GO2HardPACTPosCfgPPO,
)
from legged_gym.envs.go2.go2_pact.go2_pact_config import GO2PACTCfg
from rsl_rl.modules import ActorCritic_HardPACT, ActorCritic_HardPACT_Pos
from rsl_rl.modules.actor_critic_pact import ActorCritic_PACT
from rsl_rl.modules.actor_critic_pact_pos import ActorCritic_PACT_Pos
from rsl_rl.modules.hard_pact_physics import (
    DeploymentPhysicsHeads,
    ExplicitEstimatorDecoder,
    compose_explicit_estimator_target,
    model_output_to_physical,
    normalized_huber_loss,
    physical_unit_mae,
    scale_head_output,
    scale_physical_target,
)
from rsl_rl.algorithms.ppo_pact import PPO_PACT


def _small_actor(gains=None, actor_class=ActorCritic_HardPACT):
    gains = gains or calculate_physics_head_gains(GO2HardPACTCfg())
    return actor_class(
        num_actor_obs=57,
        num_critic_obs=64,
        num_actions=12,
        actor_layers=[32, 16],
        critic_layers=[32, 16],
        cenet_in_dim=57 * 20,
        cenet_latent_dim=16,
        cenet_velo_dim=11,
        cenet_enc_layers=[32, 16],
        activation="elu",
        init_noise_std=1.0,
        grf_scale=gains.model_grf,
        wrench_scale=gains.model_wrench,
    )


def _gain_cfg(
    force=60.0, torque=12.0, mass=4.0, com=(0.20, 0.15, 0.15),
    gravity=(0.0, 0.0, -9.81), grf=(120.0, 120.0, 250.0),
):
    return SimpleNamespace(
        sim=SimpleNamespace(
            gravity=gravity,
            grf=SimpleNamespace(prediction_scale_n=grf),
        ),
        normalization=SimpleNamespace(
            obs_scales=SimpleNamespace(grf=0.01, base_wrench=0.01)
        ),
        deployment_physics=SimpleNamespace(
            sustained_force_bounds_n=(-force, force),
            sustained_torque_bounds_nm=(-torque, torque),
            planned_added_mass_range_kg=(-1.0, mass),
        ),
        domain_rand=SimpleNamespace(
            randomize_base_mass=True,
            randomize_com_displacement=True,
            added_mass_min=-1.0,
            min_added_mass_max=mass,
            com_displacement_x_min=0.075,
            com_displacement_y_min=0.075,
            com_displacement_z_min=0.075,
            com_displacement_x_max=com[0],
            com_displacement_y_max=com[1],
            com_displacement_z_max=com[2],
        ),
    )


class ExplicitEstimatorAndHeadTests(unittest.TestCase):
    def test_explicit_target_composition_order_scaling_and_clipping(self):
        velocity = torch.tensor([[1.0, -2.0, 3.0]]) * 0.5
        contacts = torch.tensor([[0.0, 0.25, 0.75, 1.0]])
        clearance = torch.tensor([[-2.0, -0.2, 0.4, 2.0]])
        target = compose_explicit_estimator_target(velocity, contacts, clearance)
        self.assertEqual(tuple(target.shape), (1, 11))
        torch.testing.assert_close(target[:, :3], velocity)
        torch.testing.assert_close(target[:, 3:7], contacts)
        torch.testing.assert_close(
            target[:, 7:11], torch.tensor([[-1.0, -0.2, 0.4, 1.0]])
        )

    def test_alias_dimensions_and_reconstruction_schema(self):
        for env_cls, train_cls in (
            (GO2HardPACTCfg, GO2HardPACTCfgPPO),
            (GO2HardPACTPosCfg, GO2HardPACTPosCfgPPO),
        ):
            env, train = env_cls(), train_cls()
            with self.subTest(env=env_cls.__name__):
                self.assertEqual(env.env.num_observations, 57)
                self.assertEqual(env.env.num_obs_hist, 20)
                self.assertEqual(env.env.num_explicit_recon_obs, 11)
                self.assertEqual(train.policy.cenet_enc_latent_dim, 16)
                self.assertEqual(train.policy.cenet_velo_dim, 11)
                self.assertEqual(train.policy.cenet_explicit_layers, [128, 128])
                self.assertEqual(train.policy.grf_decoder_layers, [128, 128])
                self.assertEqual(train.policy.wrench_decoder_layers, [128, 128])
                self.assertEqual(train.policy.cenet_dec_input_dim, 27)
                self.assertEqual(train.policy.cenet_dec_out_dim, 133)
                self.assertEqual(
                    env.env.num_privileged_obs,
                    GO2PACTCfg.env.num_privileged_obs + DISTURBANCE_CRITIC_DIM,
                )
        self.assertEqual(RECONSTRUCTION_DIM, 133)
        self.assertTrue(set(range(61, 73)).isdisjoint(RECONSTRUCTION_INDICES))
        self.assertTrue(set(range(145, 288)).isdisjoint(RECONSTRUCTION_INDICES))

    def test_head_shapes_stop_gradient_and_preserved_gradients(self):
        heads = DeploymentPhysicsHeads((1.0,) * 12, (1.0,) * 6)
        latent = torch.randn(3, 16, requires_grad=True)
        explicit = torch.randn(3, 11, requires_grad=True)
        nominal = torch.randn(3, 12, requires_grad=True)
        output = heads(latent, explicit, nominal)
        self.assertEqual(tuple(output.grf_yaw_scaled.shape), (3, 12))
        self.assertEqual(tuple(output.base_wrench_yaw_scaled.shape), (3, 6))
        (output.grf_yaw_scaled.sum() + output.base_wrench_yaw_scaled.sum()).backward()
        self.assertIsNone(explicit.grad)
        self.assertGreater(latent.grad.abs().sum().item(), 0.0)
        self.assertGreater(nominal.grad.abs().sum().item(), 0.0)

    def test_physics_decoder_hidden_layers_are_independently_configurable(self):
        heads = DeploymentPhysicsHeads(
            (1.0,) * 12,
            (1.0,) * 6,
            grf_hidden_layers=(31, 17),
            wrench_hidden_layers=(23, 19, 13),
        )
        grf_linears = [
            module for module in heads.grf_head if isinstance(module, torch.nn.Linear)
        ]
        wrench_linears = [
            module for module in heads.wrench_head
            if isinstance(module, torch.nn.Linear)
        ]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in grf_linears],
            [(39, 31), (31, 17), (17, 12)],
        )
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in wrench_linears],
            [(27, 23), (23, 19), (19, 13), (13, 6)],
        )

    def test_actor_preserves_policy_shape_and_uses_deterministic_latent(self):
        torch.manual_seed(5)
        actor = _small_actor()
        observation = torch.randn(2, 57)
        history = torch.randn(2, 57 * 20)
        first = actor.cenet_enc_forward(history)
        second = actor.cenet_enc_forward(history)
        torch.testing.assert_close(first[2], first[0])
        torch.testing.assert_close(first[2], second[2])
        self.assertEqual(tuple(first[2].shape), (2, 16))
        self.assertEqual(tuple(first[3].shape), (2, 11))
        action = actor.act(observation, history)
        self.assertEqual(tuple(action.shape), (2, 24))
        self.assertEqual(actor.act_trunk[0].in_features, 57 + 16 + 11)

    def test_explicit_estimator_is_separate_registered_configurable_module(self):
        for actor_class in (ActorCritic_HardPACT, ActorCritic_HardPACT_Pos):
            actor = _small_actor(actor_class=actor_class)
            with self.subTest(actor=actor_class.__name__):
                self.assertIsInstance(actor.explicit_estimator, ExplicitEstimatorDecoder)
                self.assertFalse(any(
                    "explicit" in name or "velo" in name
                    for name, _ in actor.context_encoder.named_modules()
                ))
                self.assertEqual(actor.context_encoder.ce_h2.out_features, 54)
                self.assertEqual(actor.context_encoder.ce_latmean_h.in_features, 54)
                self.assertEqual(actor.context_encoder.ce_out_mean.in_features, 54)
                self.assertEqual(actor.context_encoder.ce_out_mean.out_features, 16)
                linears = [
                    module for module in actor.explicit_estimator.network
                    if isinstance(module, torch.nn.Linear)
                ]
                self.assertEqual(
                    [(layer.in_features, layer.out_features) for layer in linears],
                    [(16, 128), (128, 128), (128, 11)],
                )
                self.assertEqual(
                    sum(isinstance(module, torch.nn.ELU)
                        for module in actor.explicit_estimator.network),
                    2,
                )
                state_keys = set(actor.state_dict())
                self.assertTrue(any(
                    key.startswith("explicit_estimator.") for key in state_keys
                ))
                self.assertFalse(any("ce_velo" in key for key in state_keys))
                optimizer_groups = actor.get_optim_groups()
                registered = {
                    id(parameter)
                    for group_list in optimizer_groups
                    for group in group_list
                    for parameter in group["params"]
                }
                self.assertTrue(all(
                    id(parameter) in registered
                    for parameter in actor.explicit_estimator.parameters()
                ))

    def test_mean_decoder_output_ranges_and_sampling_independence(self):
        actor = _small_actor()
        history = torch.randn(4, 57 * 20)
        mean, logvar = actor.context_encoder(history)
        estimate = actor.explicit_estimator(mean)
        inference_mean, inference_estimate = actor.cenet_enc_inference(history)
        torch.testing.assert_close(inference_mean, mean)
        torch.testing.assert_close(inference_estimate, estimate)
        self.assertEqual(tuple(mean.shape), (4, 16))
        self.assertEqual(tuple(logvar.shape), (4, 16))
        self.assertEqual(tuple(estimate.shape), (4, 11))
        self.assertTrue(torch.all((estimate[:, 3:7] >= 0.0) & (estimate[:, 3:7] <= 1.0)))
        self.assertTrue(torch.all((estimate[:, 7:11] >= -1.0) & (estimate[:, 7:11] <= 1.0)))
        torch.manual_seed(1)
        first = actor.cenet_enc_forward(history)
        torch.manual_seed(999)
        second = actor.cenet_enc_forward(history)
        torch.testing.assert_close(first[2], second[2])
        torch.testing.assert_close(first[3], second[3])

    def test_estimate_is_used_by_actor_decoder_and_physics_consumers(self):
        actor = _small_actor()
        observation = torch.randn(3, 57)
        history = torch.randn(3, 57 * 20)
        mean, _, latent, estimate = actor.cenet_enc_forward(history)

        captured_actor_input = []
        hook = actor.act_trunk[0].register_forward_pre_hook(
            lambda _module, inputs: captured_actor_input.append(inputs[0].detach().clone())
        )
        actor.act_inference(observation, history)
        hook.remove()
        torch.testing.assert_close(captured_actor_input[0][:, -11:], estimate)

        algorithm = PPO_PACT.__new__(PPO_PACT)
        algorithm.actor_critic = actor
        algorithm.decoder = torch.nn.Linear(27, 133)
        algorithm.vae_beta = 0.1
        mask = torch.ones(3, 1)
        _, _, _, _, decoder_input, _, _ = algorithm._compute_vae_loss(
            history,
            torch.zeros(3, 12),
            torch.zeros(3, 133),
            torch.zeros(3, 11),
            mask,
        )
        torch.testing.assert_close(decoder_input[:, :16], mean)
        torch.testing.assert_close(decoder_input[:, 16:], estimate)

        nominal_torque = torch.randn(3, 12)
        expected = actor.physics_heads(latent, estimate, nominal_torque)
        actual = actor.physics_heads_from_history(history, nominal_torque)
        torch.testing.assert_close(actual.grf_yaw_scaled, expected.grf_yaw_scaled)
        torch.testing.assert_close(
            actual.base_wrench_yaw_scaled, expected.base_wrench_yaw_scaled
        )

    def test_estimator_gradient_reaches_decoder_and_shared_latent_mean(self):
        actor = _small_actor()
        history = torch.randn(2, 57 * 20)
        mean, _ = actor.context_encoder(history)
        mean.retain_grad()
        estimate = actor.explicit_estimator(mean)
        estimate[:, :3].square().mean().backward()
        self.assertGreater(mean.grad.abs().sum().item(), 0.0)
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
            for parameter in actor.explicit_estimator.parameters()
        ))
        self.assertGreater(
            actor.context_encoder.ce_out_mean.weight.grad.abs().sum().item(), 0.0
        )

    def test_hard_pact_actors_are_standalone_legacy_copies(self):
        self.assertFalse(issubclass(ActorCritic_HardPACT, ActorCritic_PACT))
        self.assertFalse(issubclass(ActorCritic_HardPACT_Pos, ActorCritic_PACT_Pos))
        self.assertEqual(ActorCritic_HardPACT.__bases__, (torch.nn.Module,))
        self.assertEqual(ActorCritic_HardPACT_Pos.__bases__, (torch.nn.Module,))


class GainAndScalingTests(unittest.TestCase):
    def test_default_expected_physical_and_model_gains(self):
        gains = calculate_physics_head_gains(GO2HardPACTCfg())
        expected_grf = (120.0, 120.0, 250.0) * 4
        self.assertEqual(gains.physical_grf, expected_grf)
        torch.testing.assert_close(
            torch.tensor(gains.model_grf), torch.tensor((1.2, 1.2, 2.5) * 4)
        )
        torch.testing.assert_close(
            torch.tensor(gains.physical_wrench),
            torch.tensor((60.0, 60.0, 99.24, 23.4403276, 23.4403276, 23.4403276)),
        )
        torch.testing.assert_close(
            torch.tensor(gains.model_wrench),
            torch.tensor((0.6, 0.6, 0.9924, 0.23440328, 0.23440328, 0.23440328)),
        )

    def test_gains_follow_ranges_gravity_and_com_envelope(self):
        small = calculate_physics_head_gains(_gain_cfg(force=10, torque=2, mass=1, com=(0, 0, 0)))
        large = calculate_physics_head_gains(_gain_cfg(force=20, torque=3, mass=5, com=(0.3, 0.2, 0.1)))
        self.assertGreater(large.physical_wrench[2], small.physical_wrench[2])
        self.assertGreater(large.physical_wrench[3], small.physical_wrench[3])
        self.assertNotEqual(large.model_wrench, small.model_wrench)

    def test_scaling_round_trip_losses_and_physical_mae(self):
        physical = torch.tensor([[10.0, -20.0, 30.0]])
        model = scale_physical_target(physical, 0.01)
        torch.testing.assert_close(model_output_to_physical(model, 0.01), physical)
        raw = torch.tensor([[0.5, -0.5, 1.0]])
        torch.testing.assert_close(
            scale_head_output(raw, torch.tensor([2.0, 4.0, 8.0])),
            torch.tensor([[1.0, -2.0, 8.0]]),
        )
        self.assertEqual(normalized_huber_loss(model, model, torch.ones(3)).item(), 0.0)
        self.assertEqual(physical_unit_mae(model, model, 0.01).item(), 0.0)


class DeploymentContractTests(unittest.TestCase):
    def test_json_contents_single_write_and_checkpoint_buffer_agreement(self):
        cfg = GO2HardPACTCfg()
        gains = calculate_physics_head_gains(cfg)
        actor = _small_actor(gains)
        contract = build_deployment_contract(cfg, actor, gains)
        with tempfile.TemporaryDirectory() as directory:
            path, wrote = write_deployment_contract_once(directory, contract)
            self.assertTrue(wrote)
            with open(path, encoding="utf-8") as stream:
                first_text = stream.read()
            _, wrote_again = write_deployment_contract_once(directory, {"bad": True})
            self.assertFalse(wrote_again)
            with open(path, encoding="utf-8") as stream:
                self.assertEqual(stream.read(), first_text)
            loaded = json.loads(first_text)

        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(loaded["explicit_estimator"]["dimension"], 11)
        self.assertEqual(
            loaded["explicit_estimator"]["input"], "deterministic_latent_mean"
        )
        self.assertEqual(loaded["explicit_estimator"]["hidden_layers"], [128, 128])
        self.assertEqual(
            loaded["deployment_heads"]["grf"]["hidden_layers"], [128, 128]
        )
        self.assertEqual(
            loaded["deployment_heads"]["base_wrench"]["hidden_layers"],
            [128, 128],
        )
        self.assertEqual(loaded["frames_and_units"]["grf"]["foot_order"], list(FOOT_ORDER))
        self.assertTrue(loaded["reconstruction_target"]["critic_input_unchanged"])
        state = actor.state_dict()
        grf_key = loaded["checkpoint_buffer_keys"]["grf"]
        wrench_key = loaded["checkpoint_buffer_keys"]["base_wrench"]
        torch.testing.assert_close(
            state[grf_key], torch.tensor(loaded["model_space_gains"]["grf"])
        )
        torch.testing.assert_close(
            state[wrench_key], torch.tensor(loaded["model_space_gains"]["base_wrench"])
        )


if __name__ == "__main__":
    unittest.main()
