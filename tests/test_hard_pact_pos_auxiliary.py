"""Focused HardPACTPos deployment-head auxiliary-training coverage."""

import torch

from rsl_rl.algorithms.ppo_pact_pos import PPO_PACT_Pos
from rsl_rl.modules.actor_critic_hard_pact_pos import (
    ActorCritic_HardPACT_Pos,
    ContextDecoder,
)


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


def test_hard_pact_pos_policy_uses_reparameterized_training_latent():
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
    actor.act(observation, history, latent_noise=latent_noise)
    torch.testing.assert_close(actor.action_mean, first_mean, rtol=0, atol=0)
    actor.act(observation, history, latent_noise=-latent_noise)
    assert not torch.equal(actor.action_mean, first_mean)

    actor.zero_grad(set_to_none=True)
    actor.action_mean.square().mean().backward()
    assert actor.context_encoder.ce_out_mean.weight.grad.abs().sum() > 0
    assert actor.context_encoder.ce_out_var[0].weight.grad.abs().sum() > 0

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
