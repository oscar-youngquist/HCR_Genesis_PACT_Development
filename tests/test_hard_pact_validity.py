"""Valid-row auxiliary reductions and boot statistics, without simulation."""
import random
import copy
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest

from test_hard_pact_pos_auxiliary import _small_algorithm
from rsl_rl.algorithms.hard_pact_boot_statistics import ValidBootStatistics
from test_hard_pact_auxiliary import make_algorithm, make_batch


def test_full_hard_pact_auxiliary_compacts_before_forward_and_loss():
    alg = make_algorithm()
    args = make_batch(3)
    torch.manual_seed(31)
    reference = alg._compute_auxiliary_loss(*args)
    torch.testing.assert_close(reference["privileged"], (reference["reconstruction"] - args[1]).square().mean())
    mean, logvar, _ = alg.actor_critic.context_encoder.encode_with_features(args[0])
    torch.testing.assert_close(reference["kl"], -.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1).mean())
    ref_grads = torch.autograd.grad(reference["loss"], alg.auxiliary_parameters)

    extended = []
    for i, value in enumerate(args[:-1]):
        tail = torch.full_like(value[:2], float("nan"))
        if i == 4:  # validity, not a target
            tail.zero_()
        extended.append(torch.cat((value, tail)))
    extended.append({
        key: torch.cat((value, torch.full_like(value[:2], float("inf"))
                        if value.is_floating_point() else torch.zeros_like(value[:2])))
        for key, value in args[-1].items()
    })
    extended[0].requires_grad_()
    torch.manual_seed(31)
    actual = alg._compute_auxiliary_loss(*extended)
    for name in ("loss", "privileged", "kl", "explicit", "grf", "wrench_active", "wrench_neutral"):
        torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
    grads = torch.autograd.grad(actual["loss"], [*alg.auxiliary_parameters, extended[0]])
    for a, b in zip(grads[:-1], ref_grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.isfinite(grads[-1]).all() and grads[-1][3:].eq(0).all()
    assert actual["reconstruction"].shape == (3, 133)


@pytest.mark.parametrize("diagnostics", [False, True])
def test_full_hard_pact_empty_auxiliary_keeps_adam_state_parameters_and_boot(diagnostics):
    alg = make_algorithm(num_learning_epochs=1, num_mini_batches=1,
                         ppo_latent_diagnostics_enabled=diagnostics)
    # Prime real AdamW momentum: stepping on zero gradients would now change
    # these weights and moments. Isolate the auxiliary phase from PPO steps.
    alg.auxiliary_optimizer.zero_grad()
    alg._compute_auxiliary_loss(*make_batch())["loss"].backward()
    alg.auxiliary_optimizer.step()
    parameters = [p.detach().clone() for p in alg.auxiliary_parameters]
    optimizer_state = copy.deepcopy(alg.auxiliary_optimizer.state_dict())
    empty_args = list(make_batch())
    empty_args[4].zero_()
    empty_args[0].fill_(float("nan"))
    empty = alg._compute_auxiliary_loss(*empty_args)
    assert all(value == 0 and torch.isfinite(value) for key, value in empty.items() if key != "reconstruction")
    assert empty["reconstruction"].shape == (0, 133)

    alg.init_storage(2, 1, [57], [95], [133], [1140], [24], [11], [12], [18])
    alg.storage.dones.fill_(1)
    alg.storage.observation_history.fill_(float("nan"))
    alg.storage.observation_targets.fill_(float("nan"))
    alg.storage.max_action_delay = 0
    alg.storage.hard_pact_fields = {
        "standardized_action_noise": torch.zeros(1, 2, 24),
        "delayed_action_source_valid": torch.zeros(1, 2, 1, dtype=torch.bool),
    }
    alg.use_boot = False
    rng_state = random.getstate()

    def rl(obs, *args, **kwargs):
        # Unrelated policy learning still performs its backward/update path.
        loss = alg.actor_critic.act_trunk[0].weight.square().mean()
        alg._ppo_log_ratio = torch.zeros(obs.shape[0])
        return loss, loss, loss, obs.new_zeros(obs.shape[0], 24), None

    with patch.object(alg, "_compute_rl_loss", side_effect=rl) as ppo, \
         patch.object(alg, "spectral_normalization"), \
         patch.object(alg.act_optimizer, "step") as ppo_step, \
         patch.object(alg.auxiliary_optimizer, "step", wraps=alg.auxiliary_optimizer.step) as aux_step, \
         patch.object(alg, "_compute_auxiliary_loss", wraps=alg._compute_auxiliary_loss) as auxiliary:
        losses = alg.update(lambda a: (a[:, :12], a[:, 12:]),
                            lambda q, p, v: q - p - v, .02, 0, torch.zeros(12), 1.)
    assert ppo.call_count == ppo_step.call_count == 1
    auxiliary.assert_not_called()
    aux_step.assert_not_called()
    assert losses[2:7] == (0., 0., 0., 0., 0.)
    assert not alg.use_boot and rng_state == random.getstate()
    for current, previous in zip(alg.auxiliary_parameters, parameters):
        torch.testing.assert_close(current, previous, rtol=0, atol=0)
    current_state = alg.auxiliary_optimizer.state_dict()
    assert current_state["param_groups"] == optimizer_state["param_groups"]
    for parameter_id, state in optimizer_state["state"].items():
        for key, previous in state.items():
            torch.testing.assert_close(current_state["state"][parameter_id][key], previous, rtol=0, atol=0)


def batch(n=3):
    return [torch.randn(n, 570), torch.randn(n, 12), torch.randn(n, 133),
            torch.cat((torch.randn(n, 3), torch.zeros(n, 4), torch.randn(n, 4)), 1),
            torch.ones(n, 1), torch.randn(n, 12), torch.randn(n, 6),
            torch.ones(n, 1, dtype=torch.bool)]


def test_auxiliary_valid_row_parity_gradients_and_decoder_targets():
    alg = _small_algorithm()
    args = batch()
    torch.manual_seed(31)
    reference = alg._compute_vae_loss(*args)
    # All-valid reductions preserve the original MSE and summed-latent KL.
    torch.testing.assert_close(reference[2], (reference[6] - args[2]).square().mean())
    mean, logvar, _ = alg.actor_critic.context_encoder.encode_with_features(args[0])
    torch.testing.assert_close(reference[1], -.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1).mean())
    extended = []
    for i, value in enumerate(args):
        tail = torch.full_like(value[:2], float('nan')) if value.is_floating_point() else torch.zeros_like(value[:2])
        if i == 4:
            tail.zero_()
        extended.append(torch.cat((value, tail)))
    extended[0].requires_grad_()
    torch.manual_seed(31)
    actual = alg._compute_vae_loss(*extended)
    for a, b in zip(reference[:4], actual[:4]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert actual[4].shape[0] == actual[5].shape[0] == 3
    actual[0].backward()
    assert torch.isfinite(extended[0].grad).all()
    assert torch.count_nonzero(extended[0].grad[3:]) == 0
    torch.testing.assert_close(
        torch.nn.functional.mse_loss(alg.decoder(actual[4]), actual[5]), reference[2]
    )


def test_boot_partition_invariance_and_empty_state():
    target = torch.tensor([[1., 2.], [3., 8.], [7., 6.]], dtype=torch.float64)
    recon = target + .5
    whole, split = ValidBootStatistics(), ValidBootStatistics()
    whole.add(target, recon, torch.ones(3))
    split.add(target[:1], recon[:1], torch.ones(1))
    split.add(torch.cat((target[1:], torch.full((1, 2), float('nan')))),
              torch.cat((recon[1:], torch.full((1, 2), float('inf')))), torch.tensor([1, 1, 0]))
    assert whole.count == split.count == 3
    for a, b in zip(whole.errors(), split.errors()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(whole.errors()[0], target.var(0, unbiased=False).mean())
    empty = ValidBootStatistics()
    empty.add(torch.full((2, 3), float('nan')), torch.full((2, 3), float('inf')), torch.zeros(2))
    owner = SimpleNamespace(use_boot=True, boot_mult=1.)
    state = random.getstate()
    empty.update_boot(owner)
    assert owner.use_boot and random.getstate() == state


def test_empty_auxiliary_update_skips_optimizers_and_retains_boot():
    alg = _small_algorithm()
    args = batch(2)
    args[4].zero_()
    args[0].fill_(float('nan'))
    result = alg._compute_vae_loss(*args)
    assert all(torch.isfinite(x) and x == 0 for x in result[:4])
    alg.init_storage(2, 1, [57], [64], [133], [570], [12], [11], [12])
    alg.storage.dones.fill_(1)
    alg.storage.sigma.fill_(1)
    alg.use_boot = True
    state = random.getstate()
    def rl(*args, **kwargs):
        loss = next(alg.actor_critic.parameters()).square().mean()
        return loss, loss, loss, None, loss
    # Keep unrelated PPO alive while spying on supervised optimizers.
    with patch('torch.cuda.synchronize'), \
         patch.object(alg, '_compute_rl_loss', side_effect=rl) as ppo, \
         patch.object(alg, 'spectral_normalization'), \
         patch.object(alg.enc_optimizer, 'step') as enc, \
         patch.object(alg.decoder_optimizer, 'step') as dec:
        losses = alg.update(None, None, .02, 0, torch.zeros(12), 1.)
    assert ppo.call_count == 1
    enc.assert_not_called()
    dec.assert_not_called()
    assert alg.use_boot and random.getstate() == state
    assert losses[2:7] == (0., 0., 0., 0., 0.)
