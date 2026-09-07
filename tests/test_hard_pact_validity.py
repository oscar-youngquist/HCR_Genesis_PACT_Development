"""Valid-row auxiliary reductions and boot statistics, without simulation."""
import random
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_hard_pact_pos_auxiliary import _small_algorithm
from rsl_rl.algorithms.hard_pact_boot_statistics import ValidBootStatistics


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
