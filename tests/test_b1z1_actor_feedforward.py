"""Small detached-teacher, zero-cost gating, and actor-only training checks."""
from types import SimpleNamespace

import pytest
import torch

from test_b1z1_actor_physics import setup, config
from test_b1z1_actor_rejection import configure_actor
from rsl_rl.algorithms import b1z1_actor_feedforward as feedforward
from rsl_rl.algorithms import b1z1_actor_physics as physics, b1z1_actor_rejection as rejection


def fixture(device="cpu", dtype=torch.float32):
    a, batch, actions, context = setup(device, dtype)
    count = a.actor_critic.num_actions
    # Non-contiguous, reversed joint IDs catch accidental prefix/six-joint assumptions.
    a.cfg.update(actor_phys_arm_feedforward_enabled=True, actor_phys_arm_feedforward_weight=.25,
                 actor_phys_arm_indices=[count - 1, count - 3], pinn_loss_weight=-1.,
                 torque_action_scale=torch.linspace(2., count + 1., count).tolist())
    a.pinn_weight = 1.
    state = batch["actor_phys_state"]
    dof_count = batch["nominal_torque"].shape[-1]
    ids = torch.tensor(a.cfg["actor_phys_arm_indices"], device=device)
    fixed = a.actor_physics_cache
    generalized_ids = ids + fixed.mass_matrix.shape[-1] - dof_count
    fixed.mass_matrix[:, generalized_ids, 0] = actions.new_tensor([.2, .4])
    fixed.mass_matrix[:, 0, generalized_ids] = actions.new_tensor([.2, .4])
    fixed.bias[:, generalized_ids] = actions.new_tensor([2., 4.])
    fixed.ee_jacobian[:, 0, generalized_ids] = actions.new_tensor([1., .5])
    acceleration = torch.zeros_like(fixed.bias)
    acceleration[:, 0] = 3.
    acceleration[:, generalized_ids] = 100.  # Own arm acceleration must NOT enter the teacher.
    ee = actions.new_tensor([1., 0., 0.]).expand(len(actions), -1).clone()
    state[:, 88:91] = ee  # Recorded applied force, distinct from the decoder output.
    return a, batch, actions, context, acceleration, ee, ids, dof_count


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_target_bias_coupling_force_sign_and_detachment(device, dtype):
    a, _, _, _, acceleration, ee, ids, dofs = fixture(device, dtype)
    fixed = a.actor_physics_cache
    for value in (fixed.mass_matrix, fixed.bias, fixed.ee_jacobian, acceleration, ee):
        value.requires_grad_()
    target, bias, coupling, external = feedforward.compensation_target(fixed, acceleration, ee, ids, dofs)
    for value, expected in ((bias, [2., 4.]), (coupling, [.6, 1.2]),
                            (external, [1., .5]), (target, [1.6, 4.7])):
        torch.testing.assert_close(value, value.new_tensor(expected).expand_as(value))
        assert value.dtype == dtype and value.device.type == device and not value.requires_grad
    # Zero disturbance/acceleration retains gravity/bias, rather than using the force-task gate.
    stationary = feedforward.compensation_target(fixed, acceleration * 0, ee * 0, ids, dofs)[0]
    torch.testing.assert_close(stationary, bias)


def test_matching_torque_scaling_and_ff_only_gradients():
    a, batch, actions, _, acceleration, ee, ids, dofs = fixture(dtype=torch.float64)
    state = batch["actor_phys_state"].clone()
    state[:, 97:97 + dofs] = torch.linspace(.8, 1.2, dofs, dtype=state.dtype)
    state[:, 155] = .7
    fixed = a.actor_physics_cache
    target = feedforward.compensation_target(fixed, acceleration, ee, ids, dofs)[0]
    count = a.actor_critic.num_actions
    scale = actions.new_tensor(a.cfg["torque_action_scale"])
    actions = torch.zeros_like(actions)
    gain = state[:, 97:97 + dofs].index_select(1, ids) * state[:, 155:156] * scale[ids]
    actions[:, count + ids] = target / gain
    # Cross-check physical FF against the existing controller, not another scale convention.
    actual = a._coupled_torque(actions, state) - a._coupled_torque(actions[:, :count], state)
    torch.testing.assert_close(actual.index_select(1, ids), target)
    loss, _ = feedforward.objective(a, actions, state, fixed, acceleration, dofs)
    assert loss.abs() < 1e-20
    actions[:, count + ids[0]] += .1
    actions.requires_grad_()
    for value in (state, acceleration, ee, fixed.mass_matrix, fixed.bias, fixed.ee_jacobian):
        value.requires_grad_()
    loss, metrics = feedforward.objective(a, actions, state, fixed, acceleration, dofs)
    loss.backward()
    assert loss > 0 and torch.isfinite(actions.grad).all()
    assert actions.grad[:, count + ids[0]].abs().sum() > 0
    keep = torch.zeros_like(actions, dtype=torch.bool)
    keep[:, count + ids] = True
    assert not actions.grad[~keep].any()
    assert all(value.grad is None for value in (state, acceleration, ee, fixed.mass_matrix, fixed.bias, fixed.ee_jacobian))
    assert metrics["arm_feedforward_effective_weight"] == pytest.approx(.25 * a.cfg["actor_phys_coef"])


@pytest.mark.parametrize("reason", ["flag", "weight", "actor", "coefficient", "sampling", "position"])
def test_disabled_teacher_does_no_work(monkeypatch, reason):
    cfg = {**config(), "actor_phys_arm_feedforward_enabled": True, "actor_phys_arm_feedforward_weight": 1.}
    changes = dict(flag={"actor_phys_arm_feedforward_enabled": False}, weight={"actor_phys_arm_feedforward_weight": 0.},
                   actor={"actor_phys_enabled": False}, coefficient={"actor_phys_coef": 0.},
                   sampling={"actor_phys_sample_fraction": 0.})
    cfg.update(changes.get(reason, {}))
    a = SimpleNamespace(cfg=cfg, actor_critic=SimpleNamespace(action_mode="position" if reason == "position" else "coupled"))
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled teacher touched dynamics")
    monkeypatch.setattr(feedforward, "compensation_target", forbidden)
    actions = torch.ones(2, 4, requires_grad=True)
    loss, metrics = feedforward.objective(a, actions, None, None, None, None)
    loss.backward()
    assert loss == 0 and not actions.grad.any() and metrics == {}


@pytest.mark.parametrize("reason", ["reset", "teleport", "target_nan", "goal_boundary"])
def test_invalid_transition_skips_teacher(monkeypatch, reason):
    a, batch, actions, context, *_ = fixture()
    if reason == "reset":
        batch["dones"].fill_(True)
    elif reason == "teleport":
        batch["physics_invalid"].fill_(1)
    elif reason == "target_nan":
        batch["actor_phys_ee_target"].fill_(float("nan"))
    else:
        batch["actor_phys_valid"].fill_(False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid transition reached teacher")
    monkeypatch.setattr(feedforward, "compensation_target", forbidden)
    loss, metrics = physics.objective(a, batch, actions, context)
    assert loss == 0 and metrics["arm_feedforward_raw"] == 0


def test_invalid_label_and_disabled_actuator_stay_finite():
    a, batch, actions, _, acceleration, ee, _, dofs = fixture()
    state = batch["actor_phys_state"].clone()
    acceleration.fill_(float("nan"))
    loss, metrics = feedforward.objective(a, actions, state, a.actor_physics_cache, acceleration, dofs)
    assert loss == 0 and metrics["arm_feedforward_valid_fraction"] == 0
    acceleration.zero_()
    state[:, 155] = 0.
    loss, metrics = feedforward.objective(a, actions, state, a.actor_physics_cache, acceleration, dofs)
    assert loss == 0 and metrics["arm_feedforward_valid_fraction"] == 0


def test_teacher_uses_recorded_applied_force_not_estimate():
    a, batch, actions, context, *_ = fixture()
    # Remove cross-inertia only to isolate the direct -J^T F teacher term.
    fixed = a.actor_physics_cache
    fixed.mass_matrix.copy_(torch.eye(fixed.mass_matrix.shape[-1]))
    actions = torch.zeros_like(actions, requires_grad=True)
    _, original = physics.objective(a, batch, actions, context)
    expected = actions.new_tensor([1., .5]).norm()
    torch.testing.assert_close(original["arm_feedforward_external_norm_nm"], expected)
    # Even a contradictory next-state force or decoder estimate cannot replace F_t.
    batch["dynamics_state"] = batch["actor_phys_state"].clone()
    batch["dynamics_state"][:, 88:91] = -100.
    context = {**context, "ee_force": torch.full_like(context["ee_force"], -2.)}
    _, estimated_changed = physics.objective(a, batch, actions, context)
    for name in ("arm_feedforward_external_norm_nm", "arm_feedforward_raw"):
        torch.testing.assert_close(estimated_changed[name], original[name], rtol=0, atol=0)
    batch["actor_phys_state"][:, 88:91] *= 2
    _, applied_changed = physics.objective(a, batch, actions, context)
    torch.testing.assert_close(applied_changed["arm_feedforward_external_norm_nm"], 2 * expected)
    assert applied_changed["arm_feedforward_raw"] != original["arm_feedforward_raw"]


def test_additive_loss_preserves_rollout_fk_and_frozen_heads(monkeypatch):
    a, batch, actions, context, *_ = fixture()
    n = a.actor_critic.num_actions
    a.cfg.update(actor_phys_pos_fk_enabled=True, actor_phys_pos_fk_weight=.3,
                 actor_phys_arm_root_frame="arm_root", actor_phys_num_actions=n)
    state = batch["actor_phys_state"]
    batch.update(actor_phys_fk_default=state[:, 156:175], actor_phys_fk_base_pos=state[:, :3],
                 actor_phys_fk_base_quat=state[:, 3:7])
    a.dynamics_backend.commanded_ee_in_frame = lambda p, r, q, t, f: (q[:, n - 3:n], t)
    a.cfg["actor_phys_arm_feedforward_weight"] = 0.
    before, old = physics.objective(a, batch, actions, context)
    original_rollout = physics.differentiable_bard_rollout_loss
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original_rollout(*args, **kwargs)
    monkeypatch.setattr(physics, "differentiable_bard_rollout_loss", counted)
    a.cfg["actor_phys_arm_feedforward_weight"] = .25
    after, metrics = physics.objective(a, batch, actions, context)
    assert calls == [1]  # No extra rollout, inverse solve, or backend evaluation.
    torch.testing.assert_close(after, before + metrics["arm_feedforward_weighted"])
    for name in ("vel", "ee", "q", "qd", "pos_fk_raw", "pos_fk_weighted"):
        torch.testing.assert_close(metrics[name], old[name], rtol=0, atol=0)
    added = after - before
    gradient, = torch.autograd.grad(added, actions, retain_graph=True)
    torch.testing.assert_close(gradient[:, :n], torch.zeros_like(gradient[:, :n]), rtol=0, atol=1e-8)
    added.backward(retain_graph=True)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.actor_critic.torque_head.parameters())
    for module in (a.actor_critic.context_encoder, a.actor_critic.explicit_decoder, a.actor_critic.physics_decoder):
        assert all(p.grad is None for p in module.parameters())
    # Disabling restores the exact previous scalar and action gradient.
    a.cfg["actor_phys_arm_feedforward_weight"] = 0.
    restored, _ = physics.objective(a, batch, actions, context)
    torch.testing.assert_close(restored, before, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(restored, actions, retain_graph=True)[0],
                               torch.autograd.grad(before, actions)[0], rtol=0, atol=0)


@pytest.mark.parametrize("base_active", [False, True])
def test_zero_old_arm_weights_skip_projection_and_allocation(monkeypatch, base_active):
    a, batch, actions, context, *_ = fixture()
    configure_actor(a, batch)
    a.cfg.update(actor_phys_arm_rejection_enabled=True,
                 actor_phys_arm_rejection_weight=0., actor_phys_force_allocation_weight=0.,
                 actor_phys_base_rejection_enabled=base_active)
    def forbidden(*args, **kwargs):
        raise AssertionError("Zero-weight arm loss performed work")
    monkeypatch.setattr(rejection, "project_arm_force", forbidden)
    monkeypatch.setattr(rejection, "force_allocation", forbidden)
    if not base_active:
        monkeypatch.setattr(rejection, "neutral_torque", forbidden)
    loss, metrics = physics.objective(a, batch, actions, context)
    assert loss.isfinite() and metrics["arm_feedforward_raw"] > 0
    assert metrics["rejection_arm_weight"] == 0
    assert not any("allocation" in key for key in metrics)


def test_thin_configs_and_validation():
    from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfgPPO
    from legged_gym.utils.helpers import class_to_dict
    for number in range(4, 12):
        _, cfg = make_b1z1_pact_ablation_configs(number)
        expected = cfg.algorithm.actor_phys_enabled and cfg.policy.action_mode == "coupled"
        assert feedforward.enabled(class_to_dict(cfg().algorithm), cfg.policy.action_mode) == expected
        assert cfg.algorithm.actor_phys_arm_feedforward_weight == B1Z1PACTCfgPPO.algorithm.actor_phys_arm_feedforward_weight
    a, *_ = fixture()
    feedforward.configure(a)
    for value in (-1., float("nan"), float("inf")):
        a.cfg["actor_phys_arm_feedforward_weight"] = value
        with pytest.raises(ValueError, match="finite and nonnegative"):
            feedforward.configure(a)


def test_minimal_ppo_update_with_compact_actor_sampling():
    from test_b1z1_sampled_context import class_to_dict, B1Z1PACTCfgPPO, B1Z1PACTDecoder, PPO_B1Z1PACT
    template, batch, *_ = fixture()
    full = class_to_dict(B1Z1PACTCfgPPO())
    cfg = {**full["algorithm"], **full["policy"], **template.cfg,
           "num_learning_epochs": 2, "num_mini_batches": 1, "actor_phys_sample_fraction": .5,
           "representation_pinn_enabled": False, "dynamics_backend": "bard",
           "privileged_force_start": 23, "privileged_force_dim": 21,
           "pinn_init_steps": 0, "pinn_warmup": 2, "pinn_loss_weight": .1,
           "actor_phys_arm_rejection_weight": 0., "actor_phys_force_allocation_weight": 0.,
           "actor_phys_base_rejection_enabled": False, "actor_phys_ee_stability_weight": 0.}
    fixed = template.actor_physics_cache
    backend = SimpleNamespace(batch_capacity=3, ee_position=template.dynamics_backend.ee_position,
        evaluate=lambda *args: SimpleNamespace(**{k: v[:len(args[0])] for k, v in vars(fixed).items()}))
    a = PPO_B1Z1PACT(template.actor_critic, B1Z1PACTDecoder(22, 188, hidden=[16]), backend, cfg, "cpu")
    a.pinn_updates = 1
    a.init_storage(3, 2, 81, 40, 162, 34, 23, 232, 180, rollout_state_dim=51)
    for _ in range(2):
        a.act(torch.randn(3, 81), torch.randn(3, 40), torch.randn(3, 162), torch.zeros(3, 23))
        snapshots = {k[11:]: v.clone() for k, v in batch.items() if k.startswith("actor_phys_")}
        a.transition.actor_physics = snapshots
        a.transition.nominal_torque = torch.zeros(3, 19)
        with torch.inference_mode():
            a.process_env_step(torch.ones(3), torch.zeros(3, dtype=torch.bool), {}, torch.zeros(3, 232),
                               snapshots["state"], snapshots["state"][:, :51])
    a.compute_returns(torch.zeros(3, 40))
    result = a.update(0)
    assert result["ActorPhysics/arm_feedforward_raw"] > 0
    assert result["ActorPhysics/arm_feedforward_effective_weight"] == pytest.approx(.25 * .01 * .5)
    assert result["ActorPhysics/unintended_estimator_gradient_max"] == 0
    assert a.actor_physics_cache is None
    assert all(torch.isfinite(p).all() for p in a.actor_critic.parameters())
