"""Exercise stored latent replay through the actual shuffled PPO updates."""
from contextlib import ExitStack, nullcontext
from unittest.mock import patch

import pytest
import torch

from test_hard_pact_auxiliary import make_algorithm
from test_hard_pact_pos_auxiliary import _small_algorithm
from rsl_rl.algorithms.hard_pact_latent_diagnostics import diagonal_gaussian_kl


@pytest.mark.parametrize('pos,delay', [(False, 0), (False, 1), (True, 0)])
@pytest.mark.parametrize('diagnostics', [False, True])
def test_real_update_replays_latent_and_boot_mask(pos, delay, diagnostics):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    factory = _small_algorithm if pos else make_algorithm
    alg = factory(device=device, num_learning_epochs=2, num_mini_batches=2,
                  ppo_latent_diagnostics_enabled=diagnostics,
                  ppo_latent_diagnostics_sample_count=2)
    # Production runners place both modules on the training device.
    alg.decoder.to(device)
    history_dim, critic_dim, action_dim = (570, 64, 12) if pos else (1140, 95, 24)
    shape_args = (4, 2, [57], [critic_dim], [133], [history_dim], [action_dim], [11], [12])
    alg.init_storage(*shape_args, *(() if pos else ([18],)))
    storage = alg.storage
    if not pos:
        storage.configure_action_replay(delay)
        storage.hard_pact_fields = {
            'sampled_action_delay': torch.full((2, 4, 1), delay, device=device, dtype=torch.long),
            'delayed_action_source_valid': torch.ones(2, 4, 1, device=device, dtype=torch.bool),
            'total_external_wrench_label_yaw_normalized': torch.zeros(2, 4, 6, device=device),
            'sustained_wrench_active_mask': torch.zeros(2, 4, 1, device=device, dtype=torch.bool),
        }
        storage.hard_pact_fields['delayed_action_source_valid'][0] = not delay
    for t in range(2):
        # Alternate boot behavior between transitions, then deliberately change
        # the current flag to ensure replay uses stored conditioning.
        alg.use_boot = not bool(t) if delay else bool(t)
        obs = torch.randn(4, 57, device=device)
        hist = torch.randn(4, history_dim, device=device)
        critic = torch.randn(4, critic_dim, device=device)
        with torch.no_grad(), (torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()):
            alg.act(obs, critic, hist, *((obs, hist, obs, hist) if not pos else ()))
        tr = alg.transition
        for name, value in (
            ('observations', obs), ('observation_history', hist),
            ('critic_observations', critic), ('actions', tr.actions),
            ('actions_log_prob', tr.actions_log_prob[:, None]),
            ('mu', tr.action_mean), ('sigma', tr.action_sigma),
            ('latent_noise', tr.latent_noise), ('latent_boot_mask', tr.latent_boot_mask),
        ):
            getattr(storage, name)[t].copy_(value)
        if not pos:
            storage.action_noise[t].copy_(tr.action_noise)
    storage.advantages.normal_()
    storage.observation_targets.normal_()
    storage.explicit_labels.zero_()
    # PPO still consumes this transition's finite observation/action, but its
    # supervised next-state labels are invalid. Exercise compact auxiliary
    # reconstruction and boot statistics in the real mixed-validity update.
    storage.dones[1, 1] = 1
    storage.observation_targets[1, 1] = float('nan')
    storage.explicit_labels[1, 1] = float('nan')
    storage.grf_targets[1, 1] = float('nan')
    alg.use_boot = False
    observations = storage.observations.flatten(0, 1)
    seen, ratios, kls, encoder_grads = [], [], [], []
    original = alg._compute_rl_loss
    original_replay = getattr(alg, '_replay_action_path', None)
    replay_count = 0
    delayed_encoder_grads = []

    def checked_replay(*args, **kwargs):
        nonlocal replay_count
        replay_count += 1
        result = original_replay(*args, **kwargs)
        indices = storage.current_batch_indices
        source_indices = (indices - delay * storage.num_envs).clamp_min(0)
        valid = storage.current_hard_pact_batch['delayed_action_source_valid']
        expected = torch.where(valid, storage.actions.flatten(0, 1)[source_indices].clamp(
            -alg.action_clip, alg.action_clip), 0.)
        # Production uses BF16 rollout versus FP32 replay on CUDA.
        torch.testing.assert_close(result['delayed_action'], expected,
                                   rtol=0, atol=.01 if device == 'cuda' else 2e-6)
        if delay:
            grad = torch.autograd.grad(result['nominal_torque'].square().mean(),
                alg.actor_critic.context_encoder.ce_out_mean.weight, retain_graph=True)[0]
            assert torch.isfinite(grad).all()
            delayed_encoder_grads.append(grad.norm().detach())
        return result

    def checked(*args, **kwargs):
        indices = torch.cdist(args[0], observations).argmin(1)
        seen.extend(indices.cpu().tolist())
        torch.testing.assert_close(kwargs['latent_noise'], storage.latent_noise.flatten(0, 1)[indices], rtol=0, atol=0)
        torch.testing.assert_close(kwargs['latent_boot_mask'], storage.latent_boot_mask.flatten(0, 1)[indices])
        result = original(*args, **kwargs)
        actor = alg.actor_critic
        expected = actor.cenet_mean + torch.exp(.5 * actor.cenet_logvar) * kwargs['latent_noise']
        torch.testing.assert_close(actor.cenet_z, expected, rtol=0, atol=0)
        ratios.append((actor.get_actions_log_prob(args[2]) - args[6].reshape(-1)).exp().detach())
        if diagnostics:
            torch.testing.assert_close(
                alg._ppo_log_ratio,
                (actor.get_actions_log_prob(args[2]) - args[6].reshape(-1)).detach(),
                rtol=0, atol=0,
            )
        kls.append(diagonal_gaussian_kl(args[5], args[4], actor.action_mean, actor.action_std).detach())
        gradient = torch.autograd.grad(result[0], actor.context_encoder.ce_out_mean.weight, retain_graph=True)[0]
        encoder_grads.append(gradient.norm().detach())
        return result

    with ExitStack() as stack:
        if device == 'cpu':
            stack.enter_context(patch('torch.cuda.synchronize'))
        stack.enter_context(patch.object(alg, '_compute_rl_loss', side_effect=checked))
        if not pos:
            stack.enter_context(patch.object(alg, '_replay_action_path', side_effect=checked_replay))
        stack.enter_context(patch.object(alg, 'spectral_normalization'))
        # Frozen-policy comparison still runs real forwards, backwards,
        # clipping, minibatching, auxiliary losses and update diagnostics.
        for optimizer in {alg.act_optimizer, alg.enc_optimizer, alg.decoder_optimizer}:
            stack.enter_context(patch.object(optimizer, 'step'))
        alg.update(lambda a: (a[:, :12], a[:, 12:]),
                   lambda q, p, v: q - p - v, .02, 0, torch.zeros(12, device=device), 1.)
    assert sorted(seen) == sorted(list(range(8)) * 2)
    assert replay_count == (0 if pos else 4)
    if delay:
        assert torch.stack(delayed_encoder_grads).max() > 0
    ratio_error = (torch.cat(ratios) - 1).abs().max().item()
    kl_error = torch.cat(kls).abs().max().item()
    print(f'{pos=} {diagnostics=} {device=} ratio_error={ratio_error:.6g} kl_error={kl_error:.6g}')
    # CUDA rollout is bfloat16 and PPO float32 in the actual runners.
    assert ratio_error < (0.03 if device == 'cuda' else 2e-5)
    assert kl_error < (0.001 if device == 'cuda' else 2e-6)
    assert torch.stack(encoder_grads).max() > 0
    assert torch.isfinite(torch.stack(encoder_grads)).all()
    assert bool(alg.last_latent_diagnostics) == diagnostics


@pytest.mark.parametrize('pos', [False, True])
def test_missing_noise_fails_clearly(pos):
    alg = (_small_algorithm if pos else make_algorithm)()
    args = [None] * (16 if pos else 11)
    with pytest.raises(RuntimeError, match='stored latent noise and boot mask'):
        alg._compute_rl_loss(*args)
