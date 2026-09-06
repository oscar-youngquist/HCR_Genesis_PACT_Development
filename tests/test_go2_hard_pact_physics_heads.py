"""Focused tests for HardPACT deployment physics heads and contract."""

from __future__ import annotations

import json
import inspect
import os
import tempfile
import unittest
from types import SimpleNamespace

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
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact.ablation_configs import (
    make_hard_pact_variant_configs,
)
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import (
    GO2HardPACTPosCfg,
    GO2HardPACTPosCfgPPO,
)
from legged_gym.envs.go2.go2_pact.go2_pact_config import GO2PACTCfg
from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos import Go2PACTPos
from rsl_rl.modules import ActorCritic_HardPACT, ActorCritic_HardPACT_Pos
from rsl_rl.modules.actor_critic_pact import ActorCritic_PACT
from rsl_rl.modules.actor_critic_pact_pos import ActorCritic_PACT_Pos
from rsl_rl.modules.hard_pact_physics import (
    ContactEstimatorMetricsAccumulator,
    GRFDecoderMetricsAccumulator,
    DeploymentPhysicsHeads,
    ExplicitEstimatorDecoder,
    compose_explicit_estimator_target,
    grf_normalized_to_physical,
    normalize_grf_target,
    normalized_grf_huber_loss,
    normalized_wrench_huber_loss,
    sanitize_and_clip_wrench_for_qp,
    normalize_wrench_target,
    wrench_normalized_to_physical,
)
from rsl_rl.algorithms.ppo_pact import PPO_PACT
from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
from rsl_rl.runners.pact_pos_runner import build_hard_pact_start_checkpoint


def _small_actor(
    gains=None, actor_class=ActorCritic_HardPACT, latent_dim=16,
    contact_epsilon=0.01,
):
    gains = gains or calculate_physics_head_gains(GO2HardPACTCfg())
    return actor_class(
        num_actor_obs=57,
        num_critic_obs=64,
        num_actions=12,
        actor_layers=[32, 16],
        critic_layers=[32, 16],
        cenet_in_dim=57 * 20,
        cenet_latent_dim=latent_dim,
        cenet_velo_dim=11,
        cenet_enc_layers=[32, 16],
        activation="elu",
        init_noise_std=1.0,
        grf_scale_n=gains.grf_scale_n,
        wrench_scale=gains.wrench_scale_n_nm,
        wrench_qp_clip=gains.wrench_qp_clip_n_nm,
        contact_epsilon=contact_epsilon,
    )


class ExplicitEstimatorAndHeadTests(unittest.TestCase):
    def test_grf_decoder_metrics_use_physical_units_masks_and_exact_aggregation(self):
        accumulator = GRFDecoderMetricsAccumulator([250.0] * 12)
        # Two deliberately unequal minibatches ensure final RMSE is formed
        # from global squared-error sums, not an average of minibatch RMSEs.
        accumulator.update(
            torch.tensor([[1.0, 0.0, 0.0] * 4]),
            torch.zeros(1, 12),
            torch.tensor([[1, 0, 1, 0]]),
            torch.ones(1, 1, dtype=torch.bool),
        )
        accumulator.update(
            torch.zeros(2, 12),
            torch.zeros(2, 12),
            torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]]),
            torch.tensor([[True], [False]]),
        )
        metrics = accumulator.finalize()
        # The two valid rows contain 24 components: four 250-N errors.
        self.assertAlmostEqual(
            metrics["grf_mae_physical"].item(), 1000.0 / 24.0, places=4
        )
        self.assertAlmostEqual(
            metrics["grf_rmse_physical"].item(),
            (4.0 * 250.0 ** 2 / 24.0) ** 0.5,
            places=5,
        )
        self.assertAlmostEqual(metrics["grf_mae_Fx_physical"].item(), 125.0)
        self.assertAlmostEqual(metrics["grf_mae_Fy_physical"].item(), 0.0)
        self.assertAlmostEqual(
            metrics["grf_mae_FR_physical"].item(), 125.0 / 3.0, places=4
        )
        self.assertAlmostEqual(
            metrics["grf_stance_mae_physical"].item(), 125.0 / 3.0, places=4
        )
        self.assertAlmostEqual(
            metrics["grf_swing_mae_physical"].item(), 125.0 / 3.0, places=4
        )
        self.assertAlmostEqual(metrics["grf_signed_error_Fx_mean_physical"].item(), 125.0)
        self.assertEqual(metrics["grf_prediction_nonfinite_fraction"].item(), 0.0)
        self.assertIn("grf_predicted_norm_stance_physical", metrics)
        self.assertIn("grf_target_norm_swing_physical", metrics)
        self.assertFalse(any("clipping" in name for name in metrics))

    def test_force_decoder_diagnostic_flags_are_explicit_in_both_configs(self):
        self.assertTrue(GO2HardPACTCfgPPO.algorithm.force_decoder_diagnostics_enabled)
        self.assertTrue(GO2HardPACTPosCfgPPO.algorithm.force_decoder_diagnostics_enabled)

    def test_history_latent_dimension_is_configurable(self):
        for actor_class in (ActorCritic_HardPACT, ActorCritic_HardPACT_Pos):
            with self.subTest(actor=actor_class.__name__):
                actor = _small_actor(actor_class=actor_class, latent_dim=7)
                history = torch.randn(2, 57 * 20)
                noise = torch.randn(2, 7)
                mean, logvar, sample, explicit = actor.cenet_enc_forward(
                    history, latent_noise=noise
                )
                self.assertEqual(tuple(mean.shape), (2, 7))
                self.assertEqual(tuple(logvar.shape), (2, 7))
                self.assertEqual(tuple(sample.shape), (2, 7))
                self.assertEqual(tuple(explicit.shape), (2, 11))

    def test_explicit_target_composition_order_scaling_and_clipping(self):
        velocity = torch.tensor([[1.0, -2.0, 3.0]]) * 0.5
        contacts = torch.tensor([[0.0, 0.25, 0.75, 1.0]])
        clearance = torch.tensor([[-2.0, -0.2, 0.4, 2.0]])
        target = compose_explicit_estimator_target(velocity, contacts, clearance)
        self.assertEqual(tuple(target.shape), (1, 11))
        torch.testing.assert_close(target[:, :3], velocity)
        torch.testing.assert_close(target[:, 3:7], contacts)
        torch.testing.assert_close(target[:, 7:11], clearance)

    def test_alias_dimensions_and_reconstruction_schema(self):
        for env_cls, train_cls, history_steps in (
            (GO2HardPACTCfg, GO2HardPACTCfgPPO, 20),
            (GO2HardPACTPosCfg, GO2HardPACTPosCfgPPO, 10),
        ):
            env, train = env_cls(), train_cls()
            with self.subTest(env=env_cls.__name__):
                self.assertEqual(env.env.num_observations, 57)
                self.assertEqual(env.env.num_obs_hist, history_steps)
                self.assertEqual(env.env.num_explicit_recon_obs, 11)
                self.assertEqual(train.policy.cenet_enc_latent_dim, 16)
                self.assertEqual(train.policy.cenet_velo_dim, 11)
                self.assertEqual(train.policy.contact_epsilon, 0.01)
                self.assertEqual(train.policy.cenet_explicit_layers, [128, 128])
                self.assertEqual(train.policy.grf_decoder_layers, [128, 128])
                self.assertEqual(train.policy.wrench_decoder_layers, [128, 128])
                self.assertEqual(train.policy.cenet_dec_input_dim, 27)
                self.assertEqual(train.policy.cenet_dec_out_dim, 133)
                expected_contact_weight = (
                    1.0 if env_cls is GO2HardPACTCfg else 0.1
                )
                self.assertEqual(
                    train.algorithm.contact_probability_loss_weight,
                    expected_contact_weight,
                )
                self.assertFalse(train.algorithm.ppo_latent_diagnostics_enabled)
                self.assertEqual(train.algorithm.ppo_latent_diagnostics_interval, 100)
                self.assertEqual(train.algorithm.ppo_latent_diagnostics_sample_count, 256)
                self.assertEqual(
                    train.algorithm.latent_active_unit_variance_threshold, 1e-2
                )
                self.assertEqual(
                    env.env.num_privileged_obs,
                    GO2PACTCfg.env.num_privileged_obs + DISTURBANCE_CRITIC_DIM,
                )
        self.assertEqual(RECONSTRUCTION_DIM, 133)
        self.assertTrue(set(range(61, 73)).isdisjoint(RECONSTRUCTION_INDICES))
        self.assertTrue(set(range(145, 288)).isdisjoint(RECONSTRUCTION_INDICES))

    def test_head_shapes_stop_gradient_and_preserved_gradients(self):
        gains = calculate_physics_head_gains(GO2HardPACTCfg())
        heads = DeploymentPhysicsHeads(
            grf_scale_n=gains.grf_scale_n,
            wrench_scale=gains.wrench_scale_n_nm,
            wrench_qp_clip=gains.wrench_qp_clip_n_nm,
        )
        latent = torch.randn(3, 16, requires_grad=True)
        explicit = torch.randn(3, 11, requires_grad=True)
        nominal = torch.randn(3, 12, requires_grad=True)
        output = heads(latent, explicit, nominal)
        self.assertEqual(tuple(output.grf_normalized.shape), (3, 12))
        self.assertEqual(tuple(output.wrench_raw_normalized.shape), (3, 6))
        (output.grf_normalized.sum() + output.wrench_raw_normalized.sum()).backward()
        self.assertIsNone(explicit.grad)
        self.assertGreater(latent.grad.abs().sum().item(), 0.0)
        self.assertGreater(nominal.grad.abs().sum().item(), 0.0)

    def test_both_variants_feed_sampled_latent_and_detached_explicit_to_heads(self):
        history = torch.randn(3, 57 * 20)
        nominal = torch.randn(3, 12)
        noise_a = torch.zeros(3, 16)
        noise_b = torch.ones(3, 16)
        for actor_class in (ActorCritic_HardPACT, ActorCritic_HardPACT_Pos):
            with self.subTest(actor=actor_class.__name__):
                actor = _small_actor(actor_class=actor_class)
                mean, _, sample_a, explicit = actor.cenet_enc_forward(
                    history, latent_noise=noise_a
                )
                _, _, sample_b, _ = actor.cenet_enc_forward(
                    history, latent_noise=noise_b
                )
                self.assertFalse(torch.equal(sample_b, mean))

                captured = []
                hooks = [
                    head[0].register_forward_pre_hook(
                        lambda _module, inputs: captured.append(inputs[0])
                    )
                    for head in (
                        actor.physics_estimator.grf_head,
                        actor.physics_estimator.wrench_head,
                    )
                ]
                actor.physics_heads_from_history(
                    history, nominal, latent_noise=noise_b
                )
                for hook in hooks:
                    hook.remove()
                torch.testing.assert_close(captured[0][:, :16], sample_b)
                torch.testing.assert_close(captured[1][:, :16], sample_b)

                explicit_leaf = explicit.detach().requires_grad_()
                sample_leaf = sample_a.detach().requires_grad_()
                outputs = actor.physics_heads(sample_leaf, explicit_leaf, nominal)
                (outputs.grf_normalized.sum()
                 + outputs.wrench_raw_normalized.sum()).backward()
                self.assertIsNone(explicit_leaf.grad)
                self.assertGreater(sample_leaf.grad.abs().sum().item(), 0.0)

    def test_physics_decoder_hidden_layers_are_independently_configurable(self):
        gains = calculate_physics_head_gains(GO2HardPACTCfg())
        heads = DeploymentPhysicsHeads(
            grf_hidden_layers=(31, 17),
            wrench_hidden_layers=(23, 19, 13),
            grf_scale_n=gains.grf_scale_n,
            wrench_scale=gains.wrench_scale_n_nm,
            wrench_qp_clip=gains.wrench_qp_clip_n_nm,
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

    def test_actor_policy_uses_sample_and_explicit_uses_shared_features(self):
        torch.manual_seed(5)
        actor = _small_actor()
        observation = torch.randn(2, 57)
        history = torch.randn(2, 57 * 20)
        first = actor.cenet_enc_forward(history)
        second = actor.cenet_enc_forward(history)
        self.assertFalse(torch.equal(first[2], first[0]))
        self.assertFalse(torch.equal(first[2], second[2]))
        inference_latent, inference_explicit = actor.cenet_enc_inference(history)
        torch.testing.assert_close(inference_latent, first[0])
        _, _, features = actor.context_encoder.encode_with_features(history)
        torch.testing.assert_close(
            inference_explicit,
            actor.explicit_estimator(features).explicit_for_policy,
        )
        self.assertEqual(tuple(first[2].shape), (2, 16))
        self.assertEqual(tuple(first[3].shape), (2, 11))
        captured_actor_input = []
        hook = actor.act_trunk[0].register_forward_pre_hook(
            lambda _module, inputs: captured_actor_input.append(inputs[0].detach())
        )
        action = actor.act(observation, history, latent_noise=torch.ones_like(first[0]))
        first_action_mean = actor.action_mean.clone()
        first_decoder_latent = actor.cenet_z.clone()
        actor.act(observation, history, latent_noise=-torch.ones_like(first[0]))
        hook.remove()
        self.assertEqual(tuple(action.shape), (2, 24))
        self.assertEqual(actor.act_trunk[0].in_features, 57 + 16 + 11)
        torch.testing.assert_close(
            captured_actor_input[0][:, 57:73], first_decoder_latent
        )
        torch.testing.assert_close(
            captured_actor_input[0][:, 73:],
            actor.explicit_estimator(features).explicit_for_policy,
        )
        self.assertFalse(torch.equal(actor.action_mean, first_action_mean))
        self.assertFalse(torch.equal(actor.cenet_z, first_decoder_latent))
        actor.zero_grad(set_to_none=True)
        actor.action_mean.square().mean().backward()
        self.assertGreater(
            actor.context_encoder.ce_out_mean.weight.grad.abs().sum(), 0.0
        )
        self.assertGreater(
            actor.context_encoder.ce_out_var[0].weight.grad.abs().sum(), 0.0
        )

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
                self.assertEqual(actor.context_encoder.feature_dim, 54)
                self.assertEqual(actor.context_encoder.ce_latmean_h.in_features, 54)
                self.assertEqual(actor.context_encoder.ce_out_mean.in_features, 54)
                self.assertEqual(actor.context_encoder.ce_out_mean.out_features, 16)
                linears = [
                    module for module in actor.explicit_estimator.network
                    if isinstance(module, torch.nn.Linear)
                ]
                self.assertEqual(
                    [(layer.in_features, layer.out_features) for layer in linears],
                    [(54, 128), (128, 128), (128, 11)],
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

    def test_feature_decoder_output_ranges_and_sampling_independence(self):
        actor = _small_actor()
        history = torch.randn(4, 57 * 20)
        mean, logvar, features = actor.context_encoder.encode_with_features(history)
        estimate = actor.explicit_estimator(features)
        inference_mean, inference_estimate = actor.cenet_enc_inference(history)
        torch.testing.assert_close(inference_mean, mean)
        torch.testing.assert_close(
            inference_estimate, estimate.explicit_for_policy
        )
        self.assertEqual(tuple(mean.shape), (4, 16))
        self.assertEqual(tuple(logvar.shape), (4, 16))
        self.assertEqual(tuple(estimate.explicit_for_policy.shape), (4, 11))
        self.assertTrue(torch.all(
            (estimate.contact_probability >= 0.01)
            & (estimate.contact_probability <= 0.99)
        ))
        self.assertTrue(torch.all(
            estimate.explicit_for_policy[:, 7:11].abs() <= 1.0
        ))
        torch.manual_seed(1)
        first = actor.cenet_enc_forward(history)
        torch.manual_seed(999)
        second = actor.cenet_enc_forward(history)
        self.assertFalse(torch.equal(first[2], second[2]))
        torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)

    def test_estimate_is_used_by_actor_decoder_and_physics_consumers(self):
        actor = _small_actor()
        observation = torch.randn(3, 57)
        history = torch.randn(3, 57 * 20)
        latent_noise = torch.randn(3, 16)
        mean, _, latent, estimate = actor.cenet_enc_forward(
            history, latent_noise=latent_noise
        )
        _, _, features = actor.context_encoder.encode_with_features(history)

        captured_actor_input = []
        hook = actor.act_trunk[0].register_forward_pre_hook(
            lambda _module, inputs: captured_actor_input.append(inputs[0].detach().clone())
        )
        actor.act_inference(observation, history)
        hook.remove()
        torch.testing.assert_close(
            captured_actor_input[0][:, -11:],
            actor.explicit_estimator(features).explicit_for_policy,
        )

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
        # The latent input is a fresh sample, while the explicit input is the
        # deterministic sibling branch of the shared history features.
        self.assertFalse(torch.equal(decoder_input[:, :16], mean))
        _, _, _, expected_explicit = actor.cenet_enc_forward(history)
        torch.testing.assert_close(
            decoder_input[:, 16:], expected_explicit,
        )

        nominal_torque = torch.randn(3, 12)
        expected = actor.physics_heads(latent, estimate, nominal_torque)
        actual = actor.physics_heads_from_history(
            history, nominal_torque, latent_noise=latent_noise
        )
        torch.testing.assert_close(actual.grf_normalized, expected.grf_normalized)
        torch.testing.assert_close(
            actual.wrench_raw_normalized, expected.wrench_raw_normalized
        )

    def test_estimator_gradient_reaches_decoder_and_shared_history_trunk(self):
        actor = _small_actor()
        history = torch.randn(2, 57 * 20)
        features = actor.context_encoder.encode_features(history)
        features.retain_grad()
        estimate = actor.explicit_estimator(features)
        estimate.explicit_for_policy[:, :3].square().mean().backward()
        self.assertGreater(features.grad.abs().sum().item(), 0.0)
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
            for parameter in actor.explicit_estimator.parameters()
        ))
        self.assertGreater(actor.context_encoder.ce_h2.weight.grad.abs().sum(), 0.0)
        self.assertIsNone(actor.context_encoder.ce_out_mean.weight.grad)
        self.assertIsNone(actor.context_encoder.ce_out_var[0].weight.grad)

    def test_hard_pact_actors_are_standalone_legacy_copies(self):
        self.assertFalse(issubclass(ActorCritic_HardPACT, ActorCritic_PACT))
        self.assertFalse(issubclass(ActorCritic_HardPACT_Pos, ActorCritic_PACT_Pos))
        self.assertEqual(ActorCritic_HardPACT.__bases__, (torch.nn.Module,))
        self.assertEqual(ActorCritic_HardPACT_Pos.__bases__, (torch.nn.Module,))


class GainAndScalingTests(unittest.TestCase):
    def test_contact_estimator_structures_logits_and_converts_exactly_once(self):
        decoder = ExplicitEstimatorDecoder(8, (4,))
        with torch.no_grad():
            decoder.network[-1].weight.zero_()
            decoder.network[-1].bias.zero_()
            decoder.network[-1].bias[3:7] = torch.tensor([-4., -2., 2., 4.])
        estimate = decoder(torch.zeros(1, 8))
        torch.testing.assert_close(
            estimate.contact_logits, torch.tensor([[-4., -2., 2., 4.]])
        )
        expected = 0.01 + 0.98 * torch.sigmoid(estimate.contact_logits)
        torch.testing.assert_close(estimate.contact_probability, expected)
        torch.testing.assert_close(
            estimate.explicit_for_policy[:, 3:7], estimate.contact_probability
        )
        estimate.contact_probability.sum().backward()
        self.assertGreater(decoder.network[-1].bias.grad[3:7].abs().sum(), 0.0)

    def test_configured_contact_epsilon_reaches_actor_and_contract(self):
        cfg = GO2HardPACTCfg()
        gains = calculate_physics_head_gains(cfg)
        actor = _small_actor(gains, contact_epsilon=0.07)
        estimate = actor.explicit_estimator(torch.zeros(
            1, actor.context_encoder.feature_dim
        ))
        self.assertEqual(actor.explicit_estimator.contact_epsilon, 0.07)
        self.assertTrue(torch.all(estimate.contact_probability >= 0.07))
        self.assertTrue(torch.all(estimate.contact_probability <= 0.93))
        contract = build_deployment_contract(cfg, actor, gains)
        self.assertEqual(contract["contact_estimator_supervision"]["epsilon"], 0.07)
        self.assertEqual(contract["qp_inputs"]["contact"]["epsilon"], 0.07)

    def test_contact_diagnostics_use_logits_probabilities_and_valid_mask(self):
        decoder = ExplicitEstimatorDecoder(2, (2,), contact_epsilon=0.01)
        with torch.no_grad():
            decoder.network[-1].weight.zero_()
            decoder.network[-1].bias.zero_()
        estimate = decoder(torch.zeros(2, 2))
        labels = torch.tensor([[0., 1., 0., 1.], [1., 1., 1., 1.]])
        accumulator = ContactEstimatorMetricsAccumulator(0.01)
        accumulator.update(estimate, labels, torch.tensor([[True], [False]]))
        metrics = accumulator.finalize()
        torch.testing.assert_close(metrics["contact_bce"], torch.log(torch.tensor(2.0)))
        torch.testing.assert_close(metrics["contact_probability_mean"], torch.tensor(0.5))
        torch.testing.assert_close(metrics["contact_classification_accuracy"], torch.tensor(0.5))
        torch.testing.assert_close(metrics["contact_brier_score"], torch.tensor(0.25))

    def test_wrench_head_is_unbounded_linear_and_qp_clips_once(self):
        gains = calculate_physics_head_gains(GO2HardPACTCfg())
        head = DeploymentPhysicsHeads(
            grf_hidden_layers=(4,), wrench_hidden_layers=(4,),
            grf_scale_n=gains.grf_scale_n,
            wrench_scale=gains.wrench_scale_n_nm,
            wrench_qp_clip=gains.wrench_qp_clip_n_nm,
        )
        latent = torch.randn(3, 16, requires_grad=True)
        explicit = torch.randn(3, 11)
        raw = head.predict_wrench(latent, explicit)
        physical = head.wrench_to_physical(raw)
        torch.testing.assert_close(
            physical, raw * torch.tensor(gains.wrench_scale_n_nm)
        )
        forced = torch.tensor([[2.0, -2.0, 1.0, 2.0, -2.0, 1.0]])
        torch.testing.assert_close(
            head.wrench_to_qp_physical(forced),
            torch.tensor([[150.0, -150.0, 100.0, 40.0, -40.0, 25.0]]),
        )
        raw.sum().backward()
        self.assertGreater(latent.grad.abs().sum().item(), 0.0)
        with torch.no_grad():
            head.wrench_head[-1].weight.zero_()
            head.wrench_head[-1].bias.fill_(10.0)
        torch.testing.assert_close(
            head.predict_wrench(latent.detach(), explicit), torch.full((3, 6), 10.0)
        )

    def test_fixed_wrench_contract_is_fully_materialized(self):
        cfg = GO2HardPACTCfg()
        gains = calculate_physics_head_gains(cfg)
        actor = _small_actor(gains)
        contract = build_deployment_contract(cfg, actor, gains)
        qp_inputs = contract["qp_inputs"]
        self.assertEqual(qp_inputs["contact"]["epsilon"], 0.01)
        self.assertEqual(qp_inputs["base_wrench"]["decoder_scale"], [100.0] * 3 + [25.0] * 3)
        self.assertEqual(qp_inputs["base_wrench"]["qp_clip"], [150.0] * 3 + [40.0] * 3)

    def test_default_fixed_scales(self):
        cfg = GO2HardPACTCfg()
        gains = calculate_physics_head_gains(cfg)
        self.assertEqual(gains.grf_scale_n, tuple(cfg.sim.grf.prediction_scale_n) * 4)
        self.assertEqual(gains.wrench_scale_n_nm, tuple(cfg.deployment_physics.wrench_scale))
        self.assertEqual(gains.wrench_qp_clip_n_nm, tuple(cfg.deployment_physics.wrench_qp_clip))

    def test_hard_pact_pos_and_hard_pact_share_force_normalization(self):
        gains = calculate_physics_head_gains(GO2HardPACTCfg())
        pos_gains = calculate_physics_head_gains(GO2HardPACTPosCfg())
        self.assertEqual(pos_gains.grf_scale_n, gains.grf_scale_n)
        self.assertEqual(pos_gains.wrench_scale_n_nm, gains.wrench_scale_n_nm)
        self.assertEqual(pos_gains.wrench_qp_clip_n_nm, gains.wrench_qp_clip_n_nm)
        for cfg in (GO2HardPACTCfg, GO2HardPACTPosCfg):
            self.assertEqual(
                tuple(cfg.deployment_physics.wrench_scale), gains.wrench_scale_n_nm
            )
            self.assertEqual(
                tuple(cfg.deployment_physics.wrench_qp_clip),
                gains.wrench_qp_clip_n_nm,
            )
        for variant in (
            "baseline", "soft", "hard", "full", "stopgrad",
            "soft_penalty", "inverse", "rollout",
        ):
            variant_cfg, variant_train = make_hard_pact_variant_configs(
                variant, "genesis"
            )
            self.assertEqual(
                tuple(variant_cfg.deployment_physics.wrench_scale),
                gains.wrench_scale_n_nm,
            )
            self.assertEqual(
                tuple(variant_cfg.deployment_physics.wrench_qp_clip),
                gains.wrench_qp_clip_n_nm,
            )
            self.assertFalse(
                variant_train.algorithm.ppo_latent_diagnostics_enabled
            )
            self.assertEqual(
                variant_train.algorithm.ppo_latent_diagnostics_interval, 100
            )
        actor = ActorCritic_HardPACT(
            num_actor_obs=57, num_critic_obs=64, num_actions=12,
            actor_layers=[16], critic_layers=[16], cenet_in_dim=57 * 20,
            cenet_enc_layers=[16, 8], grf_scale_n=gains.grf_scale_n,
            wrench_scale=gains.wrench_scale_n_nm,
            wrench_qp_clip=gains.wrench_qp_clip_n_nm,
        )
        torch.testing.assert_close(
            actor.physics_estimator.wrench_scale,
            torch.tensor(gains.wrench_scale_n_nm),
        )
        torch.testing.assert_close(
            actor.physics_estimator.wrench_qp_clip,
            torch.tensor(gains.wrench_qp_clip_n_nm),
        )

    def test_edited_force_config_propagates_to_targets_model_qp_and_contract(self):
        configured_grf_scale = (100., 200., 300.)
        configured_grf_clip = (-321., 432.)
        configured_scale = (80., 90., 100., 20., 22., 24.)
        configured_clip = (120., 130., 140., 30., 32., 34.)
        cfg = SimpleNamespace(
            sim=SimpleNamespace(
                grf=SimpleNamespace(
                    prediction_scale_n=configured_grf_scale,
                    clip_min_n=configured_grf_clip[0],
                    clip_max_n=configured_grf_clip[1],
                )
            ),
            normalization=SimpleNamespace(
                obs_scales=SimpleNamespace(grf=0.01, base_wrench=0.01)
            ),
            deployment_physics=SimpleNamespace(
                wrench_scale=configured_scale,
                wrench_qp_clip=configured_clip,
            ),
        )
        gains = calculate_physics_head_gains(cfg)
        actor = ActorCritic_HardPACT(
            num_actor_obs=57, num_critic_obs=64, num_actions=12,
            actor_layers=[16], critic_layers=[16], cenet_in_dim=57 * 20,
            cenet_enc_layers=[16, 8], wrench_scale=gains.wrench_scale_n_nm,
            wrench_qp_clip=gains.wrench_qp_clip_n_nm,
            grf_scale_n=gains.grf_scale_n,
        )
        pos_actor = ActorCritic_HardPACT_Pos(
            num_actor_obs=57, num_critic_obs=64, num_actions=12,
            actor_layers=[16], critic_layers=[16], cenet_in_dim=57 * 20,
            cenet_enc_layers=[16, 8], wrench_scale=gains.wrench_scale_n_nm,
            wrench_qp_clip=gains.wrench_qp_clip_n_nm,
            grf_scale_n=gains.grf_scale_n,
        )
        expected_grf_scale = torch.tensor([100., 200., 300.] * 4)
        for model in (actor, pos_actor):
            torch.testing.assert_close(
                model.physics_estimator.grf_scale_n, expected_grf_scale
            )
        self.assertEqual(gains.grf_clip_min_n, configured_grf_clip[0])
        self.assertEqual(gains.grf_clip_max_n, configured_grf_clip[1])
        torch.testing.assert_close(
            actor.physics_estimator.wrench_scale, torch.tensor(configured_scale)
        )
        torch.testing.assert_close(
            actor.physics_estimator.wrench_qp_clip, torch.tensor(configured_clip)
        )
        physical = torch.tensor([configured_scale])
        torch.testing.assert_close(
            normalize_wrench_target(physical, configured_scale), torch.ones(1, 6)
        )
        raw = torch.full((1, 6), 2.0)
        torch.testing.assert_close(
            actor.physics_estimator.wrench_to_qp_physical(raw),
            torch.tensor([configured_clip]),
        )
        grf_physical = expected_grf_scale.reshape(1, 12)
        grf_normalized = normalize_grf_target(
            grf_physical, gains.grf_scale_n
        )
        torch.testing.assert_close(grf_normalized, torch.ones_like(grf_normalized))
        torch.testing.assert_close(
            actor.physics_estimator.grf_to_physical(grf_normalized), grf_physical
        )
        torch.testing.assert_close(
            pos_actor.physics_estimator.grf_to_physical(grf_normalized),
            grf_physical,
        )
        contract = build_deployment_contract(GO2HardPACTCfg(), actor, gains)
        self.assertEqual(
            contract["grf_decoder_normalization"]["scale_n"],
            expected_grf_scale.tolist(),
        )
        self.assertEqual(
            contract["grf_decoder_normalization"]["interval_target_clip_n"],
            {
                "minimum": configured_grf_clip[0],
                "maximum": configured_grf_clip[1],
                "location": "GRF processor before control-interval averaging",
            },
        )
        self.assertEqual(
            contract["wrench_decoder_normalization"]["scale_n_nm"],
            list(configured_scale),
        )
        self.assertEqual(
            contract["qp_inputs"]["base_wrench"]["qp_clip"],
            list(configured_clip),
        )
        self.assertIn(
            "self.cfg.deployment_physics.wrench_scale",
            inspect.getsource(Go2HardPACT._end_disturbance_interval),
        )
        self.assertIn(
            "self.cfg.sim.grf.prediction_scale_n",
            inspect.getsource(Go2HardPACT.step),
        )

    def test_grf_normalization_round_trip_and_direct_huber(self):
        gains = calculate_physics_head_gains(GO2HardPACTCfg())
        physical = torch.tensor([[250.0, -125.0, 62.5] * 4])
        normalized = normalize_grf_target(physical, gains.grf_scale_n)
        torch.testing.assert_close(
            normalized, torch.tensor([[1.0, -0.5, 0.25] * 4])
        )
        torch.testing.assert_close(
            grf_normalized_to_physical(normalized, gains.grf_scale_n), physical
        )
        self.assertEqual(
            normalized_grf_huber_loss(normalized, normalized).item(), 0.0
        )

    def test_wrench_normalization_round_trip_and_qp_boundary_clip(self):
        gains = calculate_physics_head_gains(GO2HardPACTCfg())
        physical = torch.tensor([[25.0, -50.0, 200.0, 12.5, -50.0, 0.0]])
        normalized = normalize_wrench_target(physical, gains.wrench_scale_n_nm)
        torch.testing.assert_close(
            normalized, torch.tensor([[0.25, -0.5, 2.0, 0.5, -2.0, 0.0]])
        )
        torch.testing.assert_close(
            wrench_normalized_to_physical(normalized, gains.wrench_scale_n_nm),
            physical,
        )
        self.assertEqual(
            normalized_wrench_huber_loss(normalized, normalized).item(), 0.0
        )
        torch.testing.assert_close(
            sanitize_and_clip_wrench_for_qp(
                physical, gains.wrench_qp_clip_n_nm
            ),
            torch.tensor([[25.0, -50.0, 150.0, 12.5, -40.0, 0.0]]),
        )

    def test_observation_scale_is_independent_and_absent_from_physics_paths(self):
        self.assertEqual(GO2HardPACTCfg.normalization.obs_scales.grf, 0.01)
        self.assertEqual(GO2HardPACTPosCfg.normalization.obs_scales.grf, 0.01)
        observation_source = inspect.getsource(Go2PACTPos.compute_observations)
        self.assertIn("_grfs_buf * self.obs_scales.grf", observation_source)
        self.assertNotIn(
            "grf_observation_scale",
            inspect.signature(PPO_HardPACT.__init__).parameters,
        )
        self.assertNotIn(
            "grf_observation_scale",
            inspect.getsource(PPO_HardPACT._compute_bard_loss),
        )
        bard_source = inspect.getsource(PPO_HardPACT._compute_bard_loss)
        self.assertNotIn("base_wrench_observation_scale", bard_source)
        self.assertIn("wrench_to_physical", bard_source)
        self.assertIn("sanitize_and_clip_wrench_for_qp", bard_source)
        rollout_source = inspect.getsource(
            Go2HardPACT._solve_hard_pact_rollout_qp_substep
        )
        self.assertIn("wrench_to_qp_physical", rollout_source)
        self.assertNotIn("obs_scales.base_wrench", rollout_source)
        self.assertNotIn("torch.sigmoid(", rollout_source)
        self.assertNotIn("torch.sigmoid(", bard_source)
        self.assertNotIn("binary_cross_entropy", rollout_source)
        self.assertIn("binary_cross_entropy_with_logits", inspect.getsource(
            PPO_HardPACT._masked_explicit_loss
        ))

    def test_pos_migration_preserves_physical_grf_and_old_scale_is_rejected(self):
        torch.manual_seed(37)
        pos = _small_actor(actor_class=ActorCritic_HardPACT_Pos)
        hard = _small_actor(actor_class=ActorCritic_HardPACT)
        checkpoint = build_hard_pact_start_checkpoint(
            pos.state_dict(), {}, iteration=3
        )
        hard.load_state_dict(checkpoint["model_state_dict"], strict=True)
        latent = torch.randn(3, 16)
        explicit = torch.randn(3, 11)
        torque = torch.randn(3, 12)
        pos_newtons = pos.physics_estimator.grf_to_physical(
            pos.physics_estimator.predict_grf(latent, explicit, torque)
        )
        hard_newtons = hard.physics_estimator.grf_to_physical(
            hard.physics_estimator.predict_grf(latent, explicit, torque)
        )
        torch.testing.assert_close(hard_newtons, pos_newtons, rtol=0, atol=0)

        old = hard.state_dict()
        old["physics_estimator.grf_scale"] = old.pop(
            "physics_estimator.grf_scale_n"
        )
        with self.assertRaisesRegex(RuntimeError, "grf_scale"):
            hard.load_state_dict(old, strict=True)

    def test_pos_migration_preserves_wrench_and_rejects_legacy_semantics(self):
        torch.manual_seed(39)
        pos = _small_actor(actor_class=ActorCritic_HardPACT_Pos)
        hard = _small_actor(actor_class=ActorCritic_HardPACT)
        checkpoint = build_hard_pact_start_checkpoint(pos.state_dict(), {}, 3)
        hard.load_state_dict(checkpoint["model_state_dict"], strict=True)
        latent, explicit = torch.randn(3, 16), torch.randn(3, 11)
        pos_normalized = pos.physics_estimator.predict_wrench(latent, explicit)
        hard_normalized = hard.physics_estimator.predict_wrench(latent, explicit)
        torch.testing.assert_close(hard_normalized, pos_normalized, rtol=0, atol=0)
        torch.testing.assert_close(
            hard.physics_estimator.wrench_to_physical(hard_normalized),
            pos.physics_estimator.wrench_to_physical(pos_normalized),
            rtol=0, atol=0,
        )
        legacy = hard.state_dict()
        legacy.pop("physics_estimator.wrench_qp_clip")
        with self.assertRaisesRegex(RuntimeError, "wrench_"):
            hard.load_state_dict(legacy, strict=True)

    def test_legacy_logits_in_policy_checkpoint_is_rejected_explicitly(self):
        hard = _small_actor(actor_class=ActorCritic_HardPACT)
        legacy = hard.state_dict()
        legacy.pop("explicit_estimator.contact_probability_semantics")
        with self.assertRaisesRegex(
            RuntimeError, "contact_probability_semantics"
        ):
            hard.load_state_dict(legacy, strict=True)


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

        self.assertEqual(loaded["schema_version"], 6)
        self.assertEqual(loaded["explicit_estimator"]["dimension"], 11)
        self.assertEqual(
            loaded["explicit_estimator"]["input"],
            "shared_history_encoder_features",
        )
        self.assertEqual(loaded["explicit_estimator"]["input_dimension"], 54)
        self.assertEqual(loaded["explicit_estimator"]["hidden_layers"], [128, 128])
        self.assertEqual(
            loaded["deployment_heads"]["grf"]["hidden_layers"], [128, 128]
        )
        self.assertEqual(
            loaded["deployment_heads"]["base_wrench"]["hidden_layers"],
            [128, 128],
        )
        contact_field = loaded["explicit_estimator"]["fields"][1]
        self.assertEqual(contact_field["name"], "foot_contact_probability")
        self.assertEqual(
            loaded["contact_estimator_supervision"]["training_loss"],
            "binary_cross_entropy_with_logits",
        )
        self.assertEqual(
            loaded["qp_inputs"]["contact"]["parameterization"],
            "already converted by explicit estimator",
        )
        self.assertEqual(loaded["frames_and_units"]["grf"]["foot_order"], list(FOOT_ORDER))
        self.assertEqual(
            loaded["grf_decoder_normalization"]["scale_n"], [250.0] * 12
        )
        self.assertTrue(
            loaded["grf_decoder_normalization"]["observation_scale_is_independent"]
        )
        self.assertEqual(
            loaded["critic_observation_scales_independent_of_decoders"]["grf"],
            0.01,
        )
        self.assertTrue(loaded["reconstruction_target"]["critic_input_unchanged"])
        state = actor.state_dict()
        grf_key = loaded["checkpoint_buffer_keys"]["grf"]
        wrench_key = loaded["checkpoint_buffer_keys"]["base_wrench"]
        wrench_clip_key = loaded["checkpoint_buffer_keys"]["base_wrench_qp_clip"]
        contact_semantics_key = loaded["checkpoint_buffer_keys"][
            "contact_semantics"
        ]
        torch.testing.assert_close(
            state[grf_key],
            torch.tensor(loaded["grf_decoder_normalization"]["scale_n"]),
        )
        torch.testing.assert_close(
            state[wrench_key], torch.tensor(gains.wrench_scale_n_nm)
        )
        torch.testing.assert_close(
            state[wrench_clip_key], torch.tensor(gains.wrench_qp_clip_n_nm)
        )
        self.assertEqual(state[contact_semantics_key].item(), 1)


if __name__ == "__main__":
    unittest.main()
