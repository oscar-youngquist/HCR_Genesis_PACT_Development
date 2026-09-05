"""Focused HardPACTPos deployment-head auxiliary-training coverage."""

import torch
import pytest

from rsl_rl.algorithms.ppo_pact_pos import PPO_PACT_Pos
from rsl_rl.modules.actor_critic_hard_pact_pos import (
    ActorCritic_HardPACT_Pos,
    ContextDecoder,
)


def _small_algorithm(**kwargs):
    actor = ActorCritic_HardPACT_Pos(
        num_actor_obs=57, num_critic_obs=64, num_actions=12,
        actor_layers=[32, 16], critic_layers=[32, 16],
        cenet_in_dim=57 * 10, cenet_enc_layers=[32, 16],
        cenet_explicit_layers=[16], grf_decoder_layers=[16],
        wrench_decoder_layers=[16],
    )
    decoder = ContextDecoder(input_dim=27, layers=[32, 24, 16], decode_dim=133)
    return PPO_PACT_Pos(
        actor, decoder, num_priv_obs=181,
        use_adaptive_entropy=False, **kwargs,
    )


def test_hard_pact_pos_explicit_loss_uses_bce_for_contact_slice():
    prediction = torch.tensor([
        [0.2, -0.3, 0.4, 0.8, 0.2, 0.7, 0.1, 0.5, -0.4, 0.3, -0.2]
    ], requires_grad=True)
    target = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], requires_grad=True)
    valid = torch.ones(1, 1, dtype=torch.bool)

    actual = PPO_PACT_Pos._masked_explicit_loss(prediction, target, valid)
    expected = torch.cat((
        (prediction[:, :3] - target[:, :3]).square(),
        torch.nn.functional.binary_cross_entropy(
            prediction[:, 3:7], target[:, 3:7], reduction="none"
        ),
        (prediction[:, 7:11] - target[:, 7:11]).square(),
    ), dim=-1).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert prediction.grad.abs().sum() > 0
    assert target.grad is None


def test_vae_kl_cosine_warmup_uses_absolute_iteration_and_zero_disables_it():
    algorithm = _small_algorithm(
        vae_kld_weight=1.0,
        vae_kl_initial_weight=0.2,
        vae_kl_warmup_start=100,
        vae_kl_warmup_iterations=1000,
    )
    assert algorithm._vae_beta_for_iteration(0) == pytest.approx(0.2)
    assert algorithm._vae_beta_for_iteration(100) == pytest.approx(0.2)
    assert algorithm._vae_beta_for_iteration(600) == pytest.approx(0.6)
    assert algorithm._vae_beta_for_iteration(1100) == pytest.approx(1.0)
    assert algorithm._vae_beta_for_iteration(5000) == pytest.approx(1.0)

    constant = _small_algorithm(
        vae_kld_weight=0.7,
        vae_kl_initial_weight=0.0,
        vae_kl_warmup_iterations=0,
    )
    assert constant._vae_beta_for_iteration(-100) == pytest.approx(0.7)
    assert constant._vae_beta_for_iteration(100000) == pytest.approx(0.7)


def test_hard_pact_pos_auxiliary_trains_both_physics_heads_and_logs_parts():
    torch.manual_seed(17)
    actor = ActorCritic_HardPACT_Pos(
        num_actor_obs=57, num_critic_obs=64, num_actions=12,
        actor_layers=[32, 16], critic_layers=[32, 16],
        cenet_in_dim=57 * 10, cenet_enc_layers=[32, 16],
        cenet_explicit_layers=[16], grf_decoder_layers=[16],
        wrench_decoder_layers=[16],
    )
    decoder = ContextDecoder(input_dim=27, layers=[32, 24, 16], decode_dim=133)
    algorithm = PPO_PACT_Pos(
        actor, decoder, num_priv_obs=181,
        use_adaptive_entropy=False,
    )
    batch = 6
    history = torch.randn(batch, 57 * 10)
    grf_target = torch.randn(batch, 12)
    privileged_target = torch.randn(batch, 133)
    explicit_target = torch.randn(batch, 11)
    explicit_target[:, 3:7] = torch.randint(0, 2, (batch, 4)).float()
    valid = torch.ones(batch, 1)
    executed_torque = torch.randn(batch, 12)
    wrench_target = torch.randn(batch, 6)
    wrench_active = torch.tensor(
        [[True], [False], [True], [False], [True], [False]]
    )

    result = algorithm._compute_vae_loss(
        history, grf_target, privileged_target, explicit_target, valid,
        executed_torque, wrench_target, wrench_active,
    )
    total, metrics = result[0], result[-1]
    total.backward()

    assert torch.isfinite(total)
    assert actor.physics_estimator.grf_head[0].weight.grad.abs().sum() > 0
    assert actor.physics_estimator.wrench_head[0].weight.grad.abs().sum() > 0
    assert actor.context_encoder.ce_out_mean.weight.grad.abs().sum() > 0
    assert actor.explicit_estimator.network[0].weight.grad.abs().sum() > 0
    assert set(metrics) == {
        "total", "privileged_reconstruction", "kl", "explicit", "grf",
        "wrench_active", "wrench_neutral",
        "explicit_base_linear_velocity", "explicit_contact_probabilities",
        "explicit_foot_clearance",
    }
    assert all(torch.isfinite(value) for value in metrics.values())


def test_hard_pact_pos_auxiliary_modules_share_configured_learning_rate():
    learning_rate = 7.0e-5
    algorithm = _small_algorithm(auxiliary_learning_rate=learning_rate)

    encoder_param_lrs = {
        id(parameter): group["lr"]
        for group in algorithm.enc_optimizer.param_groups
        for parameter in group["params"]
    }
    expected_encoder_modules = (
        algorithm.actor_critic.context_encoder,
        algorithm.actor_critic.explicit_estimator,
        algorithm.actor_critic.physics_estimator.grf_head,
        algorithm.actor_critic.physics_estimator.wrench_head,
    )
    for module in expected_encoder_modules:
        for parameter in module.parameters():
            assert encoder_param_lrs[id(parameter)] == pytest.approx(learning_rate)

    assert {group["lr"] for group in algorithm.decoder_optimizer.param_groups} == {
        learning_rate
    }
    decoder_ids = {
        id(parameter)
        for group in algorithm.decoder_optimizer.param_groups
        for parameter in group["params"]
    }
    assert decoder_ids == {id(parameter) for parameter in algorithm.decoder.parameters()}


def test_hard_pact_pos_policy_uses_sample_and_explicit_uses_shared_features():
    torch.manual_seed(23)
    actor = ActorCritic_HardPACT_Pos(
        num_actor_obs=57, num_critic_obs=64, num_actions=12,
        actor_layers=[32, 16], critic_layers=[32, 16],
        cenet_in_dim=57 * 10, cenet_enc_layers=[32, 16],
        cenet_explicit_layers=[16], grf_decoder_layers=[16],
        wrench_decoder_layers=[16],
    )
    observation = torch.randn(4, 57)
    history = torch.randn(4, 57 * 10)
    latent_noise = torch.randn(4, 16)

    actor.act(observation, history, latent_noise=latent_noise)
    first_mean = actor.action_mean.clone()
    first_decoder_latent = actor.cenet_z.clone()
    first_explicit = actor.cenet_torso_velo.clone()
    actor.act(observation, history, latent_noise=latent_noise)
    torch.testing.assert_close(actor.action_mean, first_mean, rtol=0, atol=0)
    actor.act(observation, history, latent_noise=-latent_noise)
    assert not torch.equal(actor.action_mean, first_mean)
    assert not torch.equal(actor.cenet_z, first_decoder_latent)
    torch.testing.assert_close(actor.cenet_torso_velo, first_explicit, rtol=0, atol=0)

    actor.zero_grad(set_to_none=True)
    actor.action_mean.square().mean().backward()
    assert actor.context_encoder.ce_out_mean.weight.grad.abs().sum() > 0
    variance_grad = actor.context_encoder.ce_out_var[0].weight.grad
    assert variance_grad.abs().sum() > 0

    first_inference = actor.act_inference(observation, history)
    second_inference = actor.act_inference(observation, history)
    torch.testing.assert_close(first_inference, second_inference, rtol=0, atol=0)


def test_legacy_pact_pos_auxiliary_does_not_allocate_physics_labels():
    from rsl_rl.modules.actor_critic_pact_pos import ActorCritic_PACT_Pos

    actor = ActorCritic_PACT_Pos(
        num_actor_obs=57, num_critic_obs=64, num_actions=12,
        actor_layers=[32, 16], critic_layers=[32, 16],
        cenet_in_dim=57 * 10, cenet_enc_layers=[32, 16],
    )
    decoder = ContextDecoder(input_dim=51, layers=[32, 24, 16], decode_dim=181)
    algorithm = PPO_PACT_Pos(actor, decoder, num_priv_obs=181)
    algorithm.init_storage(
        2, 2, [57], [64], [181], [57 * 10], [12], [35], [12]
    )
    assert algorithm.storage.hard_pact_auxiliary is False
    assert algorithm.storage.executed_torque_targets is None
    assert algorithm.storage.wrench_targets is None
    # The new HardPACT-only auxiliary rate must not alter legacy PACTPos.
    assert {group["lr"] for group in algorithm.decoder_optimizer.param_groups} == {
        algorithm.learning_rate
    }
