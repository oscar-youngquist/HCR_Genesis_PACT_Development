"""V02 latent modulation and temporal replay, using CPU-only synthetic rollouts."""
from types import SimpleNamespace

import pytest
import torch

from rsl_rl.modules.actor_critic_b1z1_pact import ActorCriticB1Z1PACT
from rsl_rl.storage.b1z1_action_replay import B1Z1ActionReplay
from rsl_rl.algorithms.ppo_b1z1_pact import PPO_B1Z1PACT


def model(enabled=True, mode="film", **kwargs):
    return ActorCriticB1Z1PACT(81, 40, 17, 162, latent_dim=8,
        actor_layers=[32, 16], critic_layers=[32, 16], context_layers=[32, 16],
        explicit_decoder_layers=[16], force_decoder_layers=[16], grf_decoder_layers=[16],
        film_v02_enabled=enabled, conditioning_mode=mode, **kwargs)


def test_latent_only_modulation_and_normalization():
    m = model()
    with torch.no_grad():
        m.film.network[-1].bias[:8] = .5
        m.film.network[-1].bias[8:] = .2
    captured = {}
    def trunk(module, inputs):
        captured["input"] = inputs[0].detach().clone()
    def condition(module, inputs):
        captured["condition"] = inputs[0].detach().clone()
    h1 = m.actor_trunk.register_forward_pre_hook(trunk)
    h2 = m.film.network.register_forward_pre_hook(condition)
    obs, hist = torch.randn(2, 81), torch.randn(2, 162)
    with torch.no_grad():
        m.act(obs, hist)
    h1.remove(); h2.remove()
    raw_z = m.last_context["z"]
    torch.testing.assert_close(captured["input"][:, 81:89], raw_z*1.5+.2)
    raw_history = m.last_context["film_tracking_history"].reshape(2, 5, 6)
    base, ee = raw_history.chunk(2, -1)
    expected = torch.cat((base/(base.abs().sum(-1, keepdim=True)+.10), ee/(ee.abs().sum(-1, keepdim=True)+.05),
                          base.abs().sum(-1, keepdim=True).tanh(), ee.abs().sum(-1, keepdim=True).tanh()), -1)
    torch.testing.assert_close(captured["condition"], expected.flatten(1))
    assert captured["condition"].shape == (2, 40)
    # Decoder inputs remain the original latent, even after policy modulation.
    assert m.last_context["z"] is raw_z
    assert not raw_history[:, :4].any()


def test_v02_soft_l1_with_bounded_raw_l1_magnitude():
    m = model()
    errors = torch.tensor([[0., .08, -.12, 1e6, -1e6, .5]], requires_grad=True)
    bounded = m.film.prepare_condition(errors)
    torch.testing.assert_close(bounded[:, :3], errors[:, :3]/(errors[:, :3].abs().sum(-1, keepdim=True)+.10))
    torch.testing.assert_close(bounded[:, 3:6], errors[:, 3:]/(errors[:, 3:].abs().sum(-1, keepdim=True)+.05))
    torch.testing.assert_close(bounded[:, 6], errors[:, :3].abs().sum(-1).tanh())
    torch.testing.assert_close(bounded[:, 7], errors[:, 3:].abs().sum(-1).tanh())
    assert ((bounded[:, 6:] >= 0) & (bounded[:, 6:] <= 1)).all()
    changed = errors.detach().clone()
    changed[:, 3:] = torch.tensor([.01, .02, .03])
    torch.testing.assert_close(m.film.prepare_condition(changed)[:, [0, 1, 2, 6]], bounded[:, [0, 1, 2, 6]])
    changed = errors.detach().clone()
    changed[:, :3] *= 20
    torch.testing.assert_close(m.film.prepare_condition(changed)[:, [3, 4, 5, 7]], bounded[:, [3, 4, 5, 7]])
    small = torch.full((1, 6), .02, requires_grad=True)
    assert m.film.prepare_condition(small)[0, 6] < m.film.prepare_condition(small * 2)[0, 6]
    m.film.prepare_condition(small)[:, -2:].sum().backward()
    assert (small.grad > 0).all()
    bounded.square().sum().backward()
    assert torch.isfinite(errors.grad).all()
    assert isinstance(m.film.network[0], torch.nn.Linear)
    assert isinstance(m.film.network[1], torch.nn.LayerNorm)
    assert isinstance(m.film.network[2], torch.nn.ELU)
    assert sum(isinstance(layer, torch.nn.LayerNorm) for layer in m.film.modules()) == 1
    zero = torch.zeros(1, 6, requires_grad=True)
    output = m.film.prepare_condition(zero)
    assert not output.any()
    output.sum().backward()
    assert torch.isfinite(zero.grad).all()
    assert m.film.prepare_condition(torch.full((1, 6), 1e-8)).abs().max() < 1e-6


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_soft_scales(scale):
    for key in ("film_base_error_soft_scale", "film_ee_error_soft_scale"):
        with pytest.raises(ValueError, match="finite and positive"):
            model(**{key: scale})


def test_configured_soft_scales():
    m = model(film_base_error_soft_scale=.2, film_ee_error_soft_scale=.4)
    e = torch.tensor([[.2, 0., 0., .4, 0., 0.]])
    torch.testing.assert_close(m.film.prepare_condition(e)[:, :6], torch.tensor([[.5, 0., 0., .5, 0., 0.]]))


def test_history_order_reset_and_ppo_replay():
    m = model()
    with torch.no_grad():
        m.film.network[-1].weight.normal_(std=.01)
    snapshots = []
    for _ in range(7):
        obs, hist = torch.randn(2, 81), torch.randn(2, 162)
        with torch.no_grad():
            action = m.act(obs, hist)
        context = m.last_context
        snapshots.append(m._actor_inputs(obs, context)[1].detach())
    stored = context["film_tracking_history"].clone()
    torch.testing.assert_close(stored.reshape(2, 5, 6), torch.stack(snapshots[-5:], 1))
    mean, logp, running = m.action_mean.clone(), m.get_actions_log_prob(action).clone(), m._film_history.clone()
    m.update_distribution(obs, hist, latent_noise=context["latent_noise"],
                          film_tracking_history=stored, detach_context=True)
    torch.testing.assert_close(m.action_mean, mean)
    torch.testing.assert_close(m.get_actions_log_prob(action), logp)
    torch.testing.assert_close(m._film_history, running)
    m.action_mean.square().sum().backward()
    assert any(p.grad is not None for p in m.film.parameters())
    assert all(p.grad is None for p in m.context_encoder.parameters())
    assert all(p.grad is None for p in m.explicit_decoder.parameters())
    assert all(p.grad is None for p in m.physics_decoder.parameters())
    m.reset(torch.tensor([True, False]))
    assert not m._film_history[0].any()
    torch.testing.assert_close(m._film_history[1], running[1])
    m.reset()
    assert m._film_history is None


def test_delayed_action_replays_source_error_history():
    m, queue = model(), B1Z1ActionReplay(1)
    first = None
    for _ in range(2):
        obs, hist = torch.randn(2, 81), torch.randn(2, 162)
        with torch.no_grad():
            action = m.act(obs, hist)
        t = SimpleNamespace(observations=obs, histories=hist,
            latent_noise=m.last_context["latent_noise"], actions=action,
            mu=m.action_mean, sigma=m.action_std,
            film_tracking_history=m.last_context["film_tracking_history"])
        source = queue.push(t, torch.ones(2, dtype=torch.long))
        if first is None:
            first = action.clone()
    batch = dict(observations=obs, histories=hist, latent_noise=t.latent_noise,
                 actions=action, physics_source=source)
    owner = SimpleNamespace(actor_critic=m, cfg={"clip_actions": 100.})
    replay, valid = PPO_B1Z1PACT._physics_actions(owner, batch)
    torch.testing.assert_close(replay, first)
    assert valid.all()


@pytest.mark.parametrize("mode", ["none", "concat"])
def test_nonfilm_ablation_ignores_v02(mode):
    m = model(mode=mode)
    assert not m.film_v02_enabled and m.film_history_dim == 0 and m.film is None


def test_legacy_default_has_no_new_checkpoint_keys():
    m = model(False)
    assert not any(isinstance(layer, torch.nn.LayerNorm) for layer in m.film.modules())
    assert not any("condition_norm" in k for k in m.state_dict())
    assert m.film.network[-1].out_features == 2*16
    m.load_state_dict(m.state_dict(), strict=True)


@pytest.mark.parametrize("bard_active", [False, True, -1])
def test_v02_short_ppo_update(monkeypatch, bard_active):
    import test_b1z1_sampled_context as fixture
    monkeypatch.setattr(fixture, "make_model", lambda pos: model())
    # This existing fixture explicitly checks the legacy sign-selected PCGrad methods.
    monkeypatch.setattr(fixture.B1Z1PACTCfgPPO.algorithm, "pinn_projection_mode", "legacy")
    fixture.test_short_ppo_update_and_auxiliary_targets(False, bard_active, "bard", monkeypatch)


def test_v02_eval_capture_and_deterministic_episode_reset():
    from test_b1z1_eval_film import evaluation
    m = model().eval()
    obs, history = torch.randn(2, 81), torch.randn(2, 162)
    with torch.no_grad():
        baseline = evaluation.inference(m, obs, history)
        m.reset()
        rng = torch.random.get_rng_state().clone()
        actions, records = evaluation.recorded_inference(m, obs, history)
    torch.testing.assert_close(actions, baseline, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert records["film_features_pre_t"].shape == (2, 8)
    assert records["film_condition_t"].shape == (2, 30)
    torch.testing.assert_close(records["film_features_pre_t"], m.last_context["z"])
    torch.testing.assert_close(records["film_condition_normalized_t"],
        m.film.prepare_condition(records["film_condition_t"]))
