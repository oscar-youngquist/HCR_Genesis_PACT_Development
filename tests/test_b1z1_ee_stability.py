"""Small actor-stability tests; no simulator or training launch."""
from types import SimpleNamespace

import pytest
import torch

from test_b1z1_actor_physics import setup
from test_b1z1_sampled_context import B1Z1PACTCfgPPO
from rsl_rl.algorithms import b1z1_actor_physics as physics
from rsl_rl.algorithms import b1z1_ee_stability as stability


def config():
    cfg = {k: getattr(B1Z1PACTCfgPPO.algorithm, k) for k in dir(B1Z1PACTCfgPPO.algorithm)
           if k.startswith("actor_phys_ee_stability_")}
    return {**cfg, "actor_phys_ee_stability_weight": 1., "dt": .02}


def pose_twist(state):
    pose = torch.eye(4, device=state.device, dtype=state.dtype).repeat(len(state), 1, 1)
    pose[:, :3, 3] = state[:, :3] + state[:, 19:22]
    return pose, state[:, 26:32] + state[:, 32:38]


def inputs():
    state = torch.zeros(2, 51)
    state[:, 6] = 1
    data = dict(state=state, ee_target=torch.zeros(2, 3),
                ee_stability_current_target=torch.zeros(2, 3),
                ee_stability_rotation=torch.eye(3).repeat(2, 1, 1))
    return SimpleNamespace(ee_pose_twist=pose_twist), state.clone(), data, config()


def test_zero_weight_never_calls_backend():
    backend, state, data, cfg = inputs()
    def forbidden(*args):
        raise AssertionError("disabled stability must not run FK/Jacobian")
    backend.ee_pose_twist = forbidden
    cfg["actor_phys_ee_stability_weight"] = 0.
    loss, metrics = stability.objective(backend, state, {}, cfg)
    assert loss == 0 and all(v == 0 for v in metrics.values())
    a, batch, actions, context = setup()
    base, _ = physics.objective(a, batch, actions, context)
    a.dynamics_backend.ee_pose_twist = forbidden
    a.cfg.update(cfg)
    same, metrics = physics.objective(a, batch, actions, context)
    torch.testing.assert_close(base, same, rtol=0, atol=0)
    assert metrics["ee_stability_raw"] == 0


def test_stationary_and_increasing_energy():
    backend, state, data, cfg = inputs()
    loss, _ = stability.objective(backend, state, data, cfg)
    assert loss == 0
    state[:, 26] = .2
    moving, _ = stability.objective(backend, state, data, cfg)
    assert moving > 0
    state[:, 0] = .1
    worse, _ = stability.objective(backend, state, data, cfg)
    assert worse > moving


@pytest.mark.parametrize("mode", ["far", "moving"])
def test_gates(mode):
    backend, state, data, cfg = inputs()
    state[:, 26] = 1
    if mode == "far":
        data["state"][:, 0] = 10
    else:
        data["ee_target"][:, 0] = .1
    loss, metrics = stability.objective(backend, state, data, cfg)
    assert loss == 0 and metrics["ee_stability_gate"] == 0


def test_actor_gradient_routing_and_reset_mask():
    a, batch, actions, context = setup()
    a.cfg.update(config())
    a.dynamics_backend.ee_pose_twist = pose_twist
    batch["actor_phys_ee_target"].zero_()
    batch["actor_phys_ee_stability_current_target"] = torch.zeros(3, 3, requires_grad=True)
    batch["actor_phys_state"].requires_grad_()
    a.actor_physics_cache.mass_matrix.requires_grad_()
    # Isolate the new term from the existing actor objectives.
    for name in ("vel", "ee", "q", "qd"):
        a.cfg[f"actor_phys_{name}_weight"] = 0.
    loss, metrics = physics.objective(a, batch, actions, context)
    assert loss > 0 and metrics["ee_stability_raw"] > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.actor_critic.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.actor_critic.parameters())
    for module in (a.actor_critic.context_encoder, a.actor_critic.explicit_decoder, a.actor_critic.physics_decoder):
        assert all(p.grad is None for p in module.parameters())
    assert batch["actor_phys_state"].grad is None
    assert batch["actor_phys_ee_stability_current_target"].grad is None
    assert a.actor_physics_cache.mass_matrix.grad is None
    batch["dones"].fill_(True)
    masked, metrics = physics.objective(a, batch, actions, context)
    assert masked == 0 and metrics["ee_stability_raw"] == 0


def test_rotation_log_identity_and_pi():
    rotations = torch.stack((torch.eye(3), torch.diag(torch.tensor([1., -1., -1.])))).requires_grad_()
    log = stability.rotation_log(rotations)
    torch.testing.assert_close(log[0], torch.zeros(3))
    assert log[1].norm().item() == pytest.approx(torch.pi)
    log.square().sum().backward()
    assert torch.isfinite(rotations.grad).all()


def test_tiny_bard_pose_twist_smoke():
    pytest.importorskip("bard")
    if torch.cuda.device_count() < 2:
        pytest.skip("BARD smoke test requires GPU 1; GPU 0 must remain untouched")
    from test_b1z1_bard_dynamics import URDF, DOF_NAMES, FOOT_NAMES
    from legged_gym.dynamics.bard_b1z1_dynamics import BardB1Z1DynamicsBackend
    backend = BardB1Z1DynamicsBackend(URDF, DOF_NAMES, FOOT_NAMES, "ee_gripper_link", "trunk",
                                    device=torch.device("cuda:1"), batch_capacity=2)
    _, state, _, _ = inputs()
    state = state.to("cuda:1")
    state[:, 32:] = .1
    state.requires_grad_()
    pose, twist = backend.ee_pose_twist(state)
    torch.testing.assert_close(pose[:, :3, 3], backend.ee_position(state))
    # World twist must equal the existing canonical world-aligned Jacobian times v.
    terms = backend.evaluate(state[:, :3], state[:, 3:7], state[:, 7:26],
                             state[:, 26:29], state[:, 29:32], state[:, 32:51],
                             state.new_zeros(2, 4, 3), state.new_zeros(2, 3), state.new_zeros(2, 6))
    torch.testing.assert_close(twist, (terms.ee_jacobian @ state[:, 26:51, None]).squeeze(-1),
                               atol=1e-5, rtol=1e-4)
    (pose[:, :3, 3].square().sum()+twist.square().sum()).backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0
