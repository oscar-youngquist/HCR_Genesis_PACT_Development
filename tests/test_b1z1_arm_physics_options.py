"""Focused shared-kinematics and opt-in actor-loss tests; no training launch."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_b1z1_actor_physics import setup
from test_b1z1_ee_stability import inputs
from legged_gym.envs.b1z1.z1_arm_kinematics import compute_z1_arm_fk, compute_z1_arm_jacobian
from rsl_rl.algorithms import b1z1_actor_physics as physics, b1z1_ee_stability as stability


def geometry():
    # Read the unchanged environment geometry instead of maintaining a second model.
    root = Path(__file__).resolve().parents[1]
    source = (root / "legged_gym/envs/b1z1/b1z1_pact/b1z1_pact.py").read_text()
    names = ("joint_offsets", "joint_axes", "link00_offset", "ee_offset")
    values = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Attribute) and target.attr in {"z1_" + n for n in names}:
                values[target.attr[3:]] = eval(compile(ast.Expression(node.value), "<geometry>", "eval"),
                                              {"torch": torch, "self": SimpleNamespace(device="cpu")})
    return [values[n] for n in names]


def arm_config():
    return dict(actor_phys_arm_manipulability_enabled=True,
                actor_phys_arm_manipulability_weight=.02,
                actor_phys_arm_manipulability_sigma_min=.5,
                actor_phys_arm_dof_indices=list(range(12, 18)),
                **{"actor_phys_arm_" + n: v.tolist() for n, v in zip(
                    ("joint_offsets", "joint_axes", "link00_offset", "ee_offset"), geometry())})


def test_translation_stability_ignores_rotations_and_capture():
    backend, state, data, cfg = inputs()
    cfg["actor_phys_ee_stability_use_orientation"] = False
    cfg.pop("actor_phys_ee_stability_rotation_radius")
    data.pop("ee_stability_rotation")
    state[:, 26] = .2
    base, metrics = stability.objective(backend, state, data, cfg)
    state[:, 29:32] = 100
    data["ee_stability_rotation"] = torch.full((2, 3, 3), float("nan"))
    same, other = stability.objective(backend, state, data, cfg)
    torch.testing.assert_close(base, same, atol=0, rtol=0)
    assert metrics["ee_stability_angular_speed"] == other["ee_stability_angular_speed"] == 0


def test_orientation_matches_legacy_energy():
    backend, state, data, cfg = inputs()
    cfg.update(actor_phys_ee_stability_use_orientation=True,
               actor_phys_ee_stability_pose_weights=[1.]*6,
               actor_phys_ee_stability_twist_weights=[1.]*6)
    state[:, 29] = .3
    actual, _ = stability.objective(backend, state, data, cfg)
    # Current energy/error is zero, so the original 6D formula reduces to this.
    kinetic = .3**2
    violation = max(cfg["actor_phys_ee_stability_beta"]*kinetic
                    - cfg["actor_phys_ee_stability_energy_slack"], 0)
    expected = cfg["actor_phys_ee_stability_weight"] * (
        cfg["actor_phys_ee_stability_twist_weight"]*kinetic
        + cfg["actor_phys_ee_stability_energy_weight"]*violation**2)
    assert actual.item() == pytest.approx(expected)


@pytest.mark.parametrize("enabled,weight", [(False, .02), (True, 0.)])
def test_manipulability_disabled_skips_jacobian(monkeypatch, enabled, weight):
    import legged_gym.envs.b1z1.z1_arm_kinematics as arm
    def forbidden(*args):
        raise AssertionError("disabled Jacobian called")
    monkeypatch.setattr(arm, "compute_z1_arm_jacobian", forbidden)
    cfg = dict(actor_phys_arm_manipulability_enabled=enabled,
               actor_phys_arm_manipulability_weight=weight)
    loss, metrics = physics.arm_manipulability(torch.zeros(2, 51), cfg)
    assert loss == 0 and all(v == 0 for v in metrics.values())


def test_shared_jacobian_is_fk_derivative():
    geo = [v.double() for v in geometry()]
    q = torch.randn(4, 6, generator=torch.Generator().manual_seed(42), dtype=torch.float64,
                    requires_grad=True)
    fk = compute_z1_arm_fk(q, *geo)
    automatic = torch.stack([torch.autograd.grad(fk[:, i].sum(), q, retain_graph=True)[0]
                             for i in range(3)], 1)
    torch.testing.assert_close(compute_z1_arm_jacobian(q, *geo), automatic, atol=1e-12, rtol=1e-12)


def test_barrier_threshold_and_actor_only_gradients():
    a, batch, actions, context = setup()
    a.cfg.update(arm_config())
    for name in ("vel", "ee", "q", "qd"):
        a.cfg["actor_phys_" + name + "_weight"] = 0.
    batch["actor_phys_state"].requires_grad_()
    a.actor_physics_cache.mass_matrix.requires_grad_()
    loss, metrics = physics.objective(a, batch, actions, context)
    assert torch.isfinite(loss) and loss > 0
    assert metrics["arm_manipulability_violation_fraction"] == 1
    grad, = torch.autograd.grad(loss, actions, retain_graph=True)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    loss.backward()
    for module in (a.actor_critic.context_encoder, a.actor_critic.explicit_decoder,
                   a.actor_critic.physics_decoder):
        assert all(p.grad is None for p in module.parameters())
    assert batch["actor_phys_state"].grad is None
    assert a.actor_physics_cache.mass_matrix.grad is None
    predicted = torch.zeros(2, 51)
    predicted[:, 19:25] = torch.tensor([.2, -.4, .5, -.3, .1, .2])
    cfg = arm_config()
    cfg["actor_phys_arm_manipulability_sigma_min"] = 1e-6
    inactive, _ = physics.arm_manipulability(predicted, cfg)
    assert inactive == 0


def test_translation_validation_ignores_rotation_radius():
    a, _, _, _ = setup()
    _, _, _, cfg = inputs()
    a.cfg.update(cfg)
    a.cfg["actor_phys_ee_stability_rotation_radius"] = -1.
    a.bard_auxiliary = True
    physics.configure(a)
    a.cfg["actor_phys_ee_stability_use_orientation"] = True
    with pytest.raises(ValueError, match="rotation_radius"):
        physics.configure(a)


def test_manipulability_evaluates_successor_and_both_action_branches(monkeypatch):
    import legged_gym.envs.b1z1.z1_arm_kinematics as arm
    a, batch, actions, context = setup()
    a.cfg.update(arm_config())
    for name in ("vel", "ee", "q", "qd"):
        a.cfg["actor_phys_" + name + "_weight"] = 0.
    recorded, successors = [], []
    jacobian, integrate = arm.compute_z1_arm_jacobian, physics.integrate_pose
    def spy(q, *geometry):
        recorded.append(q.detach().clone())
        return jacobian(q, *geometry)
    def rollout(*args):
        result = integrate(*args)
        successors.append(result)
        return result
    monkeypatch.setattr(arm, "compute_z1_arm_jacobian", spy)
    monkeypatch.setattr(physics, "integrate_pose", rollout)
    loss, _ = physics.objective(a, batch, actions, context)
    ids = torch.tensor(a.cfg["actor_phys_arm_dof_indices"])
    torch.testing.assert_close(recorded[-1], successors[-1][:, 7:26].index_select(1, ids))
    assert not torch.equal(recorded[-1], batch["actor_phys_state"][:, 7:26].index_select(1, ids))
    gradient, = torch.autograd.grad(loss, actions)
    count = a.actor_critic.num_actions
    assert torch.isfinite(gradient).all()
    assert gradient[:, :count].abs().sum() > 0
    assert gradient[:, count:].abs().sum() > 0
    changed = actions.detach().clone()
    changed[:, 13] += .5
    changed_loss, _ = physics.objective(a, batch, changed, context)
    assert changed_loss.item() != loss.item()
    # Holding the successor fixed isolates evaluation from measured q_t. In a
    # real rollout q_t still legitimately influences the successor via dynamics.
    fixed_successor = successors[-1].detach().clone()
    fixed_jacobian_input = recorded[-1].clone()
    monkeypatch.setattr(physics, "integrate_pose", lambda *args: fixed_successor)
    batch["actor_phys_state"][:, 19:25] += .7
    same, _ = physics.objective(a, batch, changed, context)
    torch.testing.assert_close(recorded[-1], fixed_jacobian_input, rtol=0, atol=0)
    torch.testing.assert_close(same, changed_loss, rtol=0, atol=0)
