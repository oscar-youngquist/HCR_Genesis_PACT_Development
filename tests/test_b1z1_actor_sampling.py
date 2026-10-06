"""Actor-only coverage, reduction, cache and zero-work checks without a simulator."""
from types import SimpleNamespace

import pytest
import torch

from test_b1z1_actor_physics import setup
from rsl_rl.algorithms import b1z1_actor_sampling as sampling, b1z1_actor_physics as physics


@pytest.mark.parametrize("fraction", [0., .07, .2, .37, .8, 1.])
def test_balanced_stratified_coverage(fraction):
    active = torch.arange(100) < 23
    rng = torch.random.get_rng_state().clone()
    epochs = sampling.epoch_indices(active, fraction, 5, 123)
    assert torch.equal(rng, torch.random.get_rng_state())
    counts = torch.bincount(torch.cat(epochs), minlength=100)
    assert counts.max() - counts.min() <= 1
    for ids in epochs:
        assert len(ids) == round(fraction * 100)
        assert len(ids.unique()) == len(ids)
        assert abs(int(active[ids].sum()) - len(ids) * .23) <= 1
    if fraction == .2:
        assert torch.equal(counts, torch.ones_like(counts))
    assert all(torch.equal(x, y) for x, y in zip(epochs, sampling.epoch_indices(active, fraction, 5, 123)))
    if 0 < fraction < 1:
        assert not torch.equal(epochs[0], sampling.epoch_indices(active, fraction, 5, 124)[0])


@pytest.mark.parametrize("fraction", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="sample_fraction"):
        physics.configure(SimpleNamespace(cfg={"actor_phys_sample_fraction": fraction}))


@pytest.mark.parametrize("n", [3, 103])
def test_once_per_rollout_covers_rounding_remainder(n):
    epochs = sampling.epoch_indices(torch.arange(n) % 3 == 0, .2, 5, 9)
    assert torch.equal(torch.bincount(torch.cat(epochs), minlength=n), torch.ones(n, dtype=torch.long))
    sizes = [len(ids) for ids in epochs]
    assert max(sizes) - min(sizes) <= 1


def test_full_fraction_exact_legacy_and_compact_subset():
    a, batch, actions, context = setup()
    actions = actions[:1].expand_as(actions).detach().clone().requires_grad_()
    context = {k: v[:1].expand_as(v) for k, v in context.items()}
    expected, _ = physics.objective(a, batch, actions, context)
    grad, = torch.autograd.grad(expected, actions, retain_graph=True)
    a.cfg["actor_phys_sample_fraction"] = 1.
    actual, _ = physics.objective(a, batch, actions, context)
    actual_grad, = torch.autograd.grad(actual, actions, retain_graph=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, grad, rtol=0, atol=0)
    # Identical rows: selecting one preserves the scalar mean and summed action gradient.
    a.cfg["actor_phys_sample_fraction"] = 1/3
    a.actor_physics_index_map = torch.tensor([-1, 0, -1])
    a.actor_physics_cache = SimpleNamespace(**{k: v[1:2] for k, v in vars(a.actor_physics_cache).items()})
    selected, _ = physics.objective(a, batch, actions, context)
    selected_grad, = torch.autograd.grad(selected, actions)
    torch.testing.assert_close(selected, expected)
    torch.testing.assert_close(selected_grad.sum(0), grad.sum(0))
    assert not selected_grad[[0, 2]].any()


@pytest.mark.parametrize("fraction", [.1, .2, .7, 1.])
def test_constant_per_row_mean_gradient(fraction):
    parameter = torch.tensor(2., requires_grad=True)
    actions = parameter.expand(100, 1)
    ids = sampling.epoch_indices(torch.arange(100) < 33, fraction, 5, 4)[0]
    mapping = torch.full((100,), -1)
    mapping[ids] = torch.arange(len(ids))
    a = SimpleNamespace(actor_physics_index_map=mapping)
    _, selected, _ = sampling.select(a, {"indices": torch.arange(100)}, actions, {})
    loss = selected.square().mean()
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(4.))
    torch.testing.assert_close(parameter.grad, torch.tensor(4.))


@pytest.mark.parametrize("disabled", [False, True])
def test_zero_fraction_or_disabled_no_physics(disabled):
    cfg = dict(actor_phys_enabled=not disabled, actor_phys_coef=1.,
               actor_phys_sample_fraction=1. if disabled else 0., pinn_loss_weight=1.)
    a = SimpleNamespace(cfg=cfg, pinn_weight=1., epochs=5, mini_batches=1,
                        storage=SimpleNamespace(steps=2, num_envs=3, actor_physics={}))
    physics.start_update(a, 0)  # No backend: any mechanics work would fail.
    physics.start_epoch(a, 0)
    physics.capture(SimpleNamespace(alg=a))  # No environment access permitted.
    action = torch.ones(3, requires_grad=True)
    loss, _ = physics.objective(a, {}, action, {})
    loss.backward()
    assert loss == 0 and not action.grad.any()
    assert a.actor_physics_cache is None


def test_compact_epoch_cache_mapping_and_release(monkeypatch):
    from rsl_rl.algorithms import b1z1_bard_pinn
    sizes = []
    def mechanics(backend, batch):
        state = batch["dynamics_state"]
        sizes.append(len(state))
        return SimpleNamespace(marker=state[:, :1].clone())
    monkeypatch.setattr(b1z1_bard_pinn, "mechanics", mechanics)
    state = torch.zeros(2, 10, 51)
    state[..., 6] = 1
    state[..., 0] = torch.arange(20).reshape(2, 10)
    a = SimpleNamespace(cfg=dict(actor_phys_enabled=True, actor_phys_coef=1.,
                                pinn_loss_weight=1., actor_phys_sample_fraction=.2),
                        pinn_weight=1., epochs=5, mini_batches=2,
                        dynamics_backend=SimpleNamespace(batch_capacity=3),
                        storage=SimpleNamespace(steps=2, num_envs=10, actor_physics={"state": state}))
    physics.start_update(a, 4)
    assert sizes == [] and a.actor_physics_cache is None
    for epoch in range(5):
        physics.start_epoch(a, epoch)
        ids = a.actor_physics_selection[epoch]
        torch.testing.assert_close(a.actor_physics_cache.marker[:, 0], ids.float())
        torch.testing.assert_close(a.actor_physics_index_map[ids], torch.arange(4))
        physics.end_epoch(a)
        assert a.actor_physics_cache is None and a.actor_physics_index_map is None
    assert sum(sizes) == 20 and max(sizes) <= 3
    assert a.actor_sampling_metrics["unique_coverage"] == 1.


def test_empty_selected_minibatch_skips_objective(monkeypatch):
    a = SimpleNamespace(cfg=dict(actor_phys_enabled=True, actor_phys_coef=.1,
                                actor_phys_sample_fraction=.2, pinn_loss_weight=1.),
                        pinn_weight=1., actor_physics_index_map=torch.tensor([-1, -1, 0]))
    def forbidden(*args):
        raise AssertionError("empty actor selection evaluated physics")
    monkeypatch.setattr(physics, "objective", forbidden)
    parameter = torch.tensor(2., requires_grad=True)
    physics.backward(a, {"indices": torch.tensor([0, 1])}, parameter.square(), None, None)
    assert parameter.grad == 4.
