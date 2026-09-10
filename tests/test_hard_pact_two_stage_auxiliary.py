"""HardPACT's frozen-decoder encoder step and detached three-decoder step."""
from unittest.mock import patch

import pytest
import torch

from test_hard_pact_auxiliary import make_algorithm, make_batch


def snapshot(algorithm, parameters):
    return [(
        p.detach().clone(),
        {k: v.clone() for k, v in algorithm.auxiliary_optimizer.state[p].items()},
    ) for p in parameters]


def assert_unchanged(algorithm, parameters, before):
    for p, (value, state) in zip(parameters, before):
        torch.testing.assert_close(p, value, rtol=0, atol=0)
        for name, expected in state.items():
            torch.testing.assert_close(
                algorithm.auxiliary_optimizer.state[p][name], expected, rtol=0, atol=0
            )


def test_disjoint_steps_preserve_inactive_weights_moments_and_sample():
    alg = make_algorithm(privileged_loss_weight=0.7, grf_loss_weight=1.3,
                         active_wrench_loss_weight=0.4, neutral_wrench_loss_weight=0.2)
    args = make_batch()
    # Prime AdamW momentum to catch stale gradients/weight decay in either
    # inactive parameter set; optimizer ownership and checkpoint keys stay put.
    alg._compute_auxiliary_loss(*args)["loss"].backward()
    alg.auxiliary_optimizer.step()
    encoder, decoders = alg.auxiliary_encoder_parameters, alg.auxiliary_decoder_parameters
    decoder_before = snapshot(alg, decoders)
    encoder_before = snapshot(alg, encoder)
    actor_before = alg.actor_critic.act_trunk[0].weight.detach().clone()
    critic_before = alg.actor_critic.critic[0].weight.detach().clone()
    groups_before = alg.auxiliary_optimizer.state_dict()["param_groups"]

    alg.auxiliary_optimizer.zero_grad(set_to_none=True)
    with alg._frozen_auxiliary_decoders():
        aux = alg._compute_auxiliary_loss(*args, return_decoder_inputs=True)
        assert all(not p.requires_grad for p in decoders)
        aux["loss"].backward()
        assert all(p.grad is None for p in decoders)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder)
        assert sum(p.grad.abs().sum() for p in encoder) > 0
        alg.auxiliary_optimizer.step()
    assert_unchanged(alg, decoders, decoder_before)
    assert any(not torch.equal(p, old[0]) for p, old in zip(encoder, encoder_before))
    encoder_after = snapshot(alg, encoder)

    inputs = aux["decoder_inputs"]
    assert all(not t.requires_grad for t in inputs.values())
    # Even if a caller supplies differentiable features/labels, phase two must
    # detach them. A second encoder evaluation/sample is expressly forbidden.
    differentiable_inputs = {
        k: v.detach().clone().requires_grad_(v.is_floating_point())
        for k, v in inputs.items()
    }
    rng = torch.random.get_rng_state()
    alg.auxiliary_optimizer.zero_grad(set_to_none=True)
    with patch.object(alg.actor_critic.context_encoder, "encode_with_features",
                      side_effect=AssertionError("decoder phase recomputed encoder")), \
         patch.object(alg.actor_critic.context_encoder, "reparameterization_trick",
                      side_effect=AssertionError("decoder phase resampled latent")):
        dec = alg._compute_auxiliary_decoder_loss(**differentiable_inputs)
        # Decoder weights and the exact sampled input are unchanged, so all
        # supervised values match phase one despite the intervening encoder step.
        for name in ("privileged", "grf", "wrench_active", "wrench_neutral"):
            torch.testing.assert_close(dec[name], aux[name], rtol=0, atol=0)
        expected = (0.7 * aux["privileged"] + 1.3 * aux["grf"]
                    + 0.4 * aux["wrench_active"] + 0.2 * aux["wrench_neutral"])
        torch.testing.assert_close(dec["loss"], expected, rtol=0, atol=0)
        dec["loss"].backward()
        assert all(p.grad is None for p in encoder)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoders)
        assert all(t.grad is None for t in differentiable_inputs.values())
        alg.auxiliary_optimizer.step()
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert_unchanged(alg, encoder, encoder_after)
    for module in (alg.decoder, alg.actor_critic.physics_estimator.grf_head,
                   alg.actor_critic.physics_estimator.wrench_head):
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0
        assert any(not torch.equal(p, old[0]) for p, old in zip(decoders, decoder_before)
                   if any(p is q for q in module.parameters()))
    torch.testing.assert_close(alg.actor_critic.act_trunk[0].weight, actor_before, rtol=0, atol=0)
    torch.testing.assert_close(alg.actor_critic.critic[0].weight, critic_before, rtol=0, atol=0)
    assert alg.auxiliary_optimizer.state_dict()["param_groups"] == groups_before


@pytest.mark.parametrize("term", ["privileged", "grf", "wrench_active", "wrench_neutral"])
def test_frozen_decoders_still_backpropagate_to_encoder(term):
    alg = make_algorithm()
    with alg._frozen_auxiliary_decoders():
        aux = alg._compute_auxiliary_loss(*make_batch())
        aux[term].backward()
        assert all(p.grad is None for p in alg.auxiliary_decoder_parameters)
        encoder = alg.actor_critic.context_encoder
        for p in (encoder.ce_in.weight, encoder.ce_out_mean.weight,
                  encoder.ce_out_var[0].weight):
            assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        if term != "privileged":
            assert all(p.grad is None for p in alg.actor_critic.explicit_estimator.parameters())


def test_freeze_restores_original_flags_on_exception():
    alg = make_algorithm()
    alg.auxiliary_decoder_parameters[0].requires_grad_(False)
    flags = [p.requires_grad for p in alg.auxiliary_decoder_parameters]
    with pytest.raises(RuntimeError, match="test exception"):
        with alg._frozen_auxiliary_decoders():
            raise RuntimeError("test exception")
    assert [p.requires_grad for p in alg.auxiliary_decoder_parameters] == flags


def test_decoder_phase_ignores_appended_invalid_nan_rows():
    alg = make_algorithm()
    args = make_batch(3)
    extended = [torch.cat((t, torch.full_like(t[:1], float("nan")))) for t in args[:-1]]
    extended[4][-1] = 0
    extended.append({
        key: torch.cat((t, torch.full_like(t[:1], float("inf"))
                        if t.is_floating_point() else torch.zeros_like(t[:1])))
        for key, t in args[-1].items()
    })
    outputs = []
    for batch in (args, extended):
        torch.manual_seed(27)
        with alg._frozen_auxiliary_decoders():
            aux = alg._compute_auxiliary_loss(*batch, return_decoder_inputs=True)
        assert aux["decoder_inputs"]["sample"].shape[0] == 3
        loss = alg._compute_auxiliary_decoder_loss(**aux["decoder_inputs"])["loss"]
        outputs.append((loss, torch.autograd.grad(loss, alg.auxiliary_decoder_parameters)))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    for before, after in zip(outputs[0][1], outputs[1][1]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("valid_rows", [0, 2])
def test_real_ppo_update_runs_encoder_then_three_decoders(diagnostics, valid_rows):
    alg = make_algorithm(num_learning_epochs=1, num_mini_batches=1,
                         ppo_latent_diagnostics_enabled=diagnostics,
                         ppo_latent_diagnostics_sample_count=2)
    alg.init_storage(3, 1, [57], [95], [133], [1140], [24], [11], [12], [18])
    storage = alg.storage
    storage.configure_action_replay(0)
    obs, hist, critic = torch.randn(3, 57), torch.randn(3, 1140), torch.randn(3, 95)
    with torch.no_grad():
        alg.act(obs, critic, hist, obs, hist, obs, hist)
    tr = alg.transition
    for name, value in (
        ("observations", obs), ("observation_history", hist), ("critic_observations", critic),
        ("actions", tr.actions), ("actions_log_prob", tr.actions_log_prob[:, None]),
        ("mu", tr.action_mean), ("sigma", tr.action_sigma), ("action_noise", tr.action_noise),
        ("latent_noise", tr.latent_noise), ("latent_boot_mask", tr.latent_boot_mask),
    ):
        getattr(storage, name)[0].copy_(value)
    storage.advantages.normal_()
    storage.observation_targets.normal_()
    storage.dones[0, valid_rows:] = 1
    for field in (storage.observation_targets, storage.explicit_labels, storage.grf_targets):
        field[0, valid_rows:] = float("nan")
    storage.hard_pact_fields = {
        "sampled_action_delay": torch.zeros(1, 3, 1, dtype=torch.long),
        "delayed_action_source_valid": torch.ones(1, 3, 1, dtype=torch.bool),
        "total_external_wrench_label_yaw_normalized": torch.zeros(1, 3, 6),
        "sustained_wrench_active_mask": torch.zeros(1, 3, 1, dtype=torch.bool),
    }
    original_step, stages = alg.auxiliary_optimizer.step, []

    def checked_step():
        enc, dec = alg.auxiliary_encoder_parameters, alg.auxiliary_decoder_parameters
        active, inactive = (enc, dec) if not stages else (dec, enc)
        assert all(p.grad is None for p in inactive)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
        assert sum(p.grad.abs().sum() for p in active) > 0
        before = snapshot(alg, inactive)
        result = original_step()
        assert_unchanged(alg, inactive, before)
        stages.append("encoder" if not stages else "decoders")
        return result

    with patch.object(alg, "spectral_normalization"), \
         patch.object(alg.act_optimizer, "step"), \
         patch.object(alg.auxiliary_optimizer, "step", side_effect=checked_step), \
         patch.object(alg, "_compute_auxiliary_decoder_loss",
                      wraps=alg._compute_auxiliary_decoder_loss) as decoder_phase:
        losses = alg.update(lambda a: (a[:, :12], a[:, 12:]),
                            lambda q, p, v: q - p - v, .02, 0, torch.zeros(12), 1.)
    assert stages == (["encoder", "decoders"] if valid_rows else [])
    assert decoder_phase.call_count == bool(valid_rows)
    assert all(torch.isfinite(torch.as_tensor(x)) for x in losses)
    if valid_rows:
        assert decoder_phase.call_args.kwargs["sample"].shape[0] == valid_rows
    assert all(p.requires_grad for p in alg.auxiliary_decoder_parameters)
