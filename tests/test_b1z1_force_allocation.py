"""Allocation-only gradients and shared projection, without a simulator."""
import pytest
import torch

from test_b1z1_actor_physics import setup
from test_b1z1_actor_rejection import configure_actor
from rsl_rl.algorithms import b1z1_actor_rejection as rejection, b1z1_actor_physics as physics
from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs


def allocation_setup():
    a, batch, actions, context = setup()
    configure_actor(a, batch)
    a.cfg.update(actor_phys_force_allocation_weight=.0025, pinn_loss_weight=-1.)
    a.pinn_weight = .5
    ids = torch.tensor(a.cfg["actor_phys_arm_dof_indices"])
    state = batch["actor_phys_state"].clone().requires_grad_()
    jac = torch.randn(len(actions), 3, len(ids), requires_grad=True)
    mass = torch.eye(len(ids)).expand(len(actions), -1, -1).clone().requires_grad_()
    neutral = torch.zeros_like(actions, requires_grad=True)
    actions = torch.zeros_like(actions).detach().requires_grad_()
    return a, state, ids, mass, jac, actions, neutral


@pytest.mark.parametrize("change", ["none", "ff", "position"])
def test_allocation_routes_only_conditioned_position(change):
    a, state, ids, mass, jac, actions, neutral = allocation_setup()
    count = a.actor_critic.num_actions
    with torch.no_grad():
        if change != "none":
            actions[:, int(ids[0]) + (count if change == "ff" else 0)] = .1
    delta = (a._coupled_torque(actions, state.detach())
             - a._coupled_torque(neutral.detach(), state.detach())).index_select(1, ids)
    generated, valid, _, projection = rejection.project_arm_force(mass, jac, delta, 1e-4, return_projection=True)
    force_gate = torch.ones(len(actions), requires_grad=True)
    loss, metrics = rejection.force_allocation(a, actions, neutral, state, ids, projection,
                                             force_gate * valid, generated)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(actions.grad).all()
    assert not actions.grad[:, count:].any()
    assert all(x.grad is None for x in (neutral, state, mass, jac, force_gate))
    if change == "position":
        assert loss > 0 and actions.grad[:, :count].abs().sum() > 0
    else:
        assert loss == 0 and not actions.grad.any()
    assert metrics["allocation_decomposition_error"] < 1e-4


def test_projection_reused_without_extra_solves(monkeypatch):
    a, state, ids, mass, jac, actions, neutral = allocation_setup()
    delta = torch.randn(len(actions), len(ids), requires_grad=True)
    expected, _, _ = rejection.project_arm_force(mass, jac, delta, 1e-4)
    calls = dict(cholesky_solve=0, solve_ex=0)
    for name in calls:
        original = getattr(torch if name == "cholesky_solve" else torch.linalg, name)
        def wrapped(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch if name == "cholesky_solve" else torch.linalg, name, wrapped)
    force, _, _, projection = rejection.project_arm_force(mass, jac, delta, 1e-4, return_projection=True)
    torch.testing.assert_close(force, expected)
    torch.testing.assert_close((projection @ delta.unsqueeze(-1)).squeeze(-1), force)
    assert calls == dict(cholesky_solve=1, solve_ex=1)
    assert not projection.requires_grad


def test_allocation_weight_and_thin_configs():
    a, *_ = allocation_setup()
    a.rejection_progress = .4
    effective = rejection.allocation_weight(a) * physics.scheduled_coefficient(a)
    assert effective == pytest.approx(.0025 * .5 * .4**2)
    for variant in range(4, 12):
        _, cfg = make_b1z1_pact_ablation_configs(variant)
        active = cfg.algorithm.actor_phys_enabled and cfg.policy.action_mode == "coupled"
        assert cfg.algorithm.actor_phys_force_allocation_weight == (.0025 if active else 0.)


def test_saturation_mismatch_is_reported_not_assigned_to_feedforward():
    a, state, ids, mass, jac, actions, neutral = allocation_setup()
    with torch.no_grad():
        actions[:, int(ids[0])] = 1.
    # Pin the first Cartesian axis to the changed arm joint for a deterministic check.
    jac = torch.zeros_like(jac)
    jac[:, 0, 0] = 1.
    limits = torch.full_like(state[:, 7:26], .01)
    delta = (rejection.bounded_torque(a, actions, state, limits)
             - rejection.bounded_torque(a, neutral.detach(), state, limits)).index_select(1, ids)
    generated, valid, _, projection = rejection.project_arm_force(mass, jac, delta, 1e-4, return_projection=True)
    loss, metrics = rejection.force_allocation(a, actions, neutral, state, ids, projection, valid, generated)
    assert loss > 0
    assert metrics["allocation_ff_force_norm"] == 0.
    assert metrics["allocation_decomposition_error"] > 0.


@pytest.mark.parametrize("mode,weight", [("coupled", 0.), ("position", .0025)])
def test_disabled_allocation_skips_specific_work(monkeypatch, mode, weight):
    a, batch, actions, context = setup()
    configure_actor(a, batch)
    a.cfg["actor_phys_force_allocation_weight"] = weight
    a.actor_critic.action_mode = mode
    if mode == "position":
        actions = actions[:, :a.actor_critic.num_actions]
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled allocation was evaluated")
    monkeypatch.setattr(rejection, "force_allocation", forbidden)
    loss, metrics = physics.objective(a, batch, actions, context)
    assert torch.isfinite(loss)
    assert not any("allocation" in name for name in metrics)


def test_full_actor_backward_allocation_frozen_heads():
    a, batch, actions, context = setup()
    configure_actor(a, batch)
    a.cfg.update(actor_phys_force_allocation_weight=.0025, pinn_loss_weight=-1.)
    a.pinn_weight = 1.
    a.actor_physics_cache.ee_jacobian[:, :3, 18:21] = torch.eye(3)
    loss, metrics = physics.objective(a, batch, actions, context)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["rejection_allocation_effective_weight"] == pytest.approx(.0025)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.actor_critic.position_head.parameters())
    for module in (a.actor_critic.context_encoder, a.actor_critic.explicit_decoder, a.actor_critic.physics_decoder):
        assert all(p.grad is None for p in module.parameters())
