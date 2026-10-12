"""Applied-force balance, frozen-decoder gradients, and compact PPO smoke tests."""
import math
from types import SimpleNamespace

import pytest
import torch

from test_b1z1_actor_physics import setup
from rsl_rl.algorithms import b1z1_actor_physics as physics, b1z1_actor_rejection as rejection
from rsl_rl.algorithms import b1z1_actor_base_balance as balance


def fixture(device="cpu", dtype=torch.float32):
    a, batch, actions, context = setup(device, dtype)
    a.cfg.update(actor_phys_base_balance_enabled=True, actor_phys_base_balance_weight=.5,
                 actor_phys_live_grf_enabled=True, actor_phys_grf_torque_trust_radius=0.,
                 actor_phys_base_force_scale=100., actor_phys_base_force_axis_weights=[1., 1.],
                 actor_phys_rejection_active_force=1., actor_phys_rejection_min_contacts=2,
                 actor_phys_arm_rejection_enabled=False, actor_phys_base_rejection_enabled=False,
                 actor_phys_force_allocation_weight=0., pinn_loss_weight=-1.)
    a.rejection_progress, a.pinn_weight = .5, 1.
    state = batch["actor_phys_state"]
    state[:, 88] = -20.  # Opposing EE force; target must ask for forward stance force.
    state[:, 91] = -10.  # Separate actual base force.
    batch["actor_phys_command"] = torch.zeros_like(batch["actor_phys_command"])
    batch["actor_phys_stance"] = torch.ones(len(actions), 4, dtype=torch.bool, device=device)
    fixed = a.actor_physics_cache
    fixed.mass_matrix.copy_(torch.eye(fixed.mass_matrix.shape[-1], device=device, dtype=dtype))
    fixed.foot_jacobians[:, :, :3, :3] = torch.eye(3, device=device, dtype=dtype)
    fixed.base_jacobian[:, :, :6] = torch.eye(6, device=device, dtype=dtype)
    fixed.ee_jacobian[:, :3, :3] = torch.eye(3, device=device, dtype=dtype)
    data = {k[11:]: v for k, v in batch.items() if k.startswith("actor_phys_")}
    return a, batch, actions, context, data


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_force_sign_zero_residual_and_detached_teacher(device, dtype):
    a, _, actions, _, data = fixture(device, dtype)
    fixed = a.actor_physics_cache
    acceleration = torch.zeros_like(fixed.bias, requires_grad=True)
    for value in (*vars(fixed).values(), data["state"], data["command"]):
        value.requires_grad_()
    target, external = balance.contact_target(data["state"], data["command"], fixed, acceleration, a.cfg)
    torch.testing.assert_close(target, target.new_tensor([30., 0.]).expand_as(target))
    torch.testing.assert_close(external, -target)
    assert not target.requires_grad
    grf = actions.new_zeros(len(actions), 4, 3)
    grf[:, :, 0] = 30. / grf.shape[1]
    loss, _ = balance.objective(a, actions, data, fixed, acceleration, grf)
    assert loss == 0
    grf = (-grf).requires_grad_()
    loss, metrics = balance.objective(a, actions, data, fixed, acceleration, grf)
    loss.backward()
    assert loss > 0 and loss.dtype == dtype and loss.device.type == device
    assert torch.isfinite(grf.grad).all() and (grf.grad[..., 0] < 0).all()
    assert all(v.grad is None for v in (*vars(fixed).values(), acceleration, data["state"], data["command"]))
    assert metrics["base_balance_effective_weight"] == pytest.approx(.5 * .5 * a.cfg["actor_phys_coef"])


def test_reachable_reference_frames_joint_coupling_and_bias():
    a, _, _, _, data = fixture(dtype=torch.float64)
    state, fixed = data["state"], a.actor_physics_cache
    state[:, 88:97] = 0.
    # Turn the body 90 degrees: commanded body +x is world +y.
    state[:, 3:7] = state.new_tensor([0., 0., math.sqrt(.5), math.sqrt(.5)])
    data["command"][:, 0] = .5
    acceleration = torch.zeros_like(fixed.bias)
    acceleration[:, 6] = 3.
    fixed.mass_matrix[:, 1, 6] = .2
    fixed.bias[:, 1] = 2.
    target, _ = balance.contact_target(state, data["command"], fixed, acceleration, a.cfg)
    rate = -math.expm1(-a.cfg["dt"] / a.cfg["actor_phys_velocity_time_constant"]) / a.cfg["dt"]
    expected = target.new_tensor([0., rate * .5 + .2 * 3. + 2.]).expand_as(target)
    torch.testing.assert_close(target, expected)
    # Mass/CoM/gripper randomization is already in cached M/h, not extra gravity labels.
    state[:, 175:180] = 7.
    torch.testing.assert_close(balance.contact_target(state, data["command"], fixed, acceleration, a.cfg)[0], target)


def test_applied_wrench_origin_shift():
    a, _, _, _, data = fixture(dtype=torch.float64)
    state, fixed = data["state"], a.actor_physics_cache
    state[:, 88:97] = 0.
    state[:, 91] = 3.
    state[:, 96] = 2.
    state[:, 177] = 4.  # lever=(0,4,0); lever x force gives tau_z=-12.
    # A synthetic mapping makes the origin-shift convention observable in planar rows.
    fixed.base_jacobian[:, 5, 1] = 1.
    target, external = balance.contact_target(state, data["command"], fixed, torch.zeros_like(fixed.bias), a.cfg)
    torch.testing.assert_close(external, state.new_tensor([3., -10.]).expand_as(external))
    torch.testing.assert_close(target, -external)


@pytest.mark.parametrize("reason", ["flag", "weight", "actor", "coefficient", "sampling", "progress", "live_grf"])
def test_disabled_skips_teacher(monkeypatch, reason):
    a, _, actions, _, _ = fixture()
    updates = dict(flag={"actor_phys_base_balance_enabled": False}, weight={"actor_phys_base_balance_weight": 0.},
                   actor={"actor_phys_enabled": False}, coefficient={"actor_phys_coef": 0.},
                   sampling={"actor_phys_sample_fraction": 0.}, live_grf={"actor_phys_live_grf_enabled": False})
    a.cfg.update(updates.get(reason, {}))
    if reason == "progress":
        a.rejection_progress = 0.
    def forbidden(*args):
        raise AssertionError("Disabled base teacher was evaluated")
    monkeypatch.setattr(balance, "contact_target", forbidden)
    loss, metrics = balance.objective(a, actions, None, None, None, None)
    assert loss == 0 and metrics == {}
    assert not torch.autograd.grad(loss, actions)[0].any()


@pytest.mark.parametrize("reason", ["neutral", "contacts", "nonfinite"])
def test_gates_skip_teacher(monkeypatch, reason):
    a, _, actions, _, data = fixture()
    acceleration = torch.zeros_like(a.actor_physics_cache.bias)
    if reason == "neutral":
        data["state"][:, 88:97] = 0.
    elif reason == "contacts":
        data["stance"].zero_()
    else:
        acceleration.fill_(float("nan"))
    monkeypatch.setattr(balance, "contact_target", lambda *args: pytest.fail("Gated teacher was evaluated"))
    loss, metrics = balance.objective(a, actions, data, a.actor_physics_cache, acceleration, actions.new_zeros(len(actions), 4, 3))
    assert loss == 0 and metrics["base_balance_active_fraction"] == 0


def test_stance_mask_and_gated_reduction():
    a, _, actions, _, data = fixture()
    grf = actions.new_zeros(len(actions), 4, 3)
    fixed = a.actor_physics_cache
    acceleration = torch.zeros_like(fixed.bias)
    full, _ = balance.objective(a, actions, data, fixed, acceleration, grf)
    data["stance"][:, 2:] = False
    grf[:, 2:, 0] = 1000.  # Non-stance feet do not contribute.
    data["state"][1:, 88:97] = 0.
    partial, metrics = balance.objective(a, actions, data, fixed, acceleration, grf)
    torch.testing.assert_close(partial, full / len(actions))
    assert metrics["base_balance_active_fraction"] == pytest.approx(1 / len(actions))


def test_target_uses_applied_force_not_prediction_or_next_state():
    a, batch, actions, context, _ = fixture()
    _, first = physics.objective(a, batch, actions, context)
    batch["dynamics_state"] = torch.full_like(batch["actor_phys_state"], 99.)
    context = {**context, "ee_force": torch.full_like(context["ee_force"], 1.),
               "base_wrench": torch.full_like(context["base_wrench"], 2.)}
    _, second = physics.objective(a, batch, actions, context)
    for key in ("base_balance_target_force_norm_n", "base_balance_external_force_norm_n"):
        torch.testing.assert_close(second[key], first[key], rtol=0, atol=0)
    batch["actor_phys_state"][:, 88:94] *= 2.
    _, third = physics.objective(a, batch, actions, context)
    torch.testing.assert_close(third["base_balance_target_force_norm_n"],
                               2 * first["base_balance_target_force_norm_n"])


def test_capture_stance_without_counterfactual_during_warmup():
    from test_b1z1_force_target_projection import ForceTargetProjectionTests
    a, batch, _, _, data = fixture()
    a.rejection_progress = 0.
    env = ForceTargetProjectionTests._environment([[.5, 0, 0]] * 3, [[100., 0, 0]] * 3)
    env.ee_start_sphere = env.ee_goal_sphere = torch.tensor([[.5, 0, 0]] * 3)
    env.goal_timer = torch.zeros(3)
    env.traj_timesteps = env.traj_total_timesteps = torch.full((3,), 4.)
    env.commands = torch.zeros(3, 6)
    env.get_pact_dynamics_state = lambda: data["state"]
    env.simulator.dof_pos_limits = torch.stack((data["q_min"], data["q_max"]), -1)
    env.simulator.dof_vel_limits = data["qd_max"]
    env.simulator.torque_limits = data["torque_max"]
    env.simulator.foot_contacts = data["stance"].clone()
    a.device, a.transition = "cpu", SimpleNamespace()
    physics.capture(SimpleNamespace(alg=a, env=env))
    torch.testing.assert_close(a.transition.actor_physics["stance"], data["stance"])
    env.simulator.foot_contacts.zero_()
    assert a.transition.actor_physics["stance"].all()  # Environment mutation cannot change stored contacts.
    a.cfg["actor_phys_base_balance_enabled"] = False
    physics.capture(SimpleNamespace(alg=a, env=env))
    assert "stance" not in a.transition.actor_physics


@pytest.mark.parametrize("position", [False, True])
def test_additive_grf_only_actor_gradients_and_no_extra_pass(monkeypatch, position):
    a, batch, actions, context, _ = fixture()
    if position:
        a.actor_critic.action_mode = "position"
        actions = actions[:, :a.actor_critic.num_actions]
    model = a.actor_critic
    a.cfg["actor_phys_base_balance_weight"] = 0.
    before, old = physics.objective(a, batch, actions, context)
    a.cfg["actor_phys_base_balance_weight"] = .5
    calls = {"rollout": 0, "grf": 0}
    original_rollout = physics.differentiable_bard_rollout_loss
    original_grf = model.physics_decoder.predict_grf_actor
    def rollout(*args, **kwargs):
        calls["rollout"] += 1
        return original_rollout(*args, **kwargs)
    def grf(*args, **kwargs):
        calls["grf"] += 1
        return original_grf(*args, **kwargs)
    monkeypatch.setattr(physics, "differentiable_bard_rollout_loss", rollout)
    monkeypatch.setattr(model.physics_decoder, "predict_grf_actor", grf)
    monkeypatch.setattr(rejection, "neutral_torque", lambda *args, **kwargs: pytest.fail("Extra actor pass"))
    after, metrics = physics.objective(a, batch, actions, context)
    assert calls == {"rollout": 1, "grf": 1}
    torch.testing.assert_close(after, before + metrics["base_balance_weighted"])
    for name in ("vel", "ee", "q", "qd"):
        torch.testing.assert_close(metrics[name], old[name], rtol=0, atol=0)
    # Subtract unchanged objectives to isolate this term's gradient.
    added = after - before
    grad, = torch.autograd.grad(added, actions, retain_graph=True)
    assert torch.isfinite(grad).all() and grad[:, :model.num_actions].abs().sum() > 0
    if not position:
        assert grad[:, model.num_actions:].abs().sum() > 0
    added.backward(retain_graph=True)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.position_head.parameters())
    for module in (model.context_encoder, model.explicit_decoder, model.physics_decoder):
        assert all(p.grad is None for p in module.parameters())
    a.cfg["actor_phys_base_balance_weight"] = 0.
    restored, _ = physics.objective(a, batch, actions, context)
    torch.testing.assert_close(restored, before, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(restored, actions, retain_graph=True)[0],
                               torch.autograd.grad(before, actions)[0], rtol=0, atol=0)


@pytest.mark.parametrize("reason", ["reset", "invalid", "goal_boundary"])
def test_invalid_transition_skips_loss(monkeypatch, reason):
    a, batch, actions, context, _ = fixture()
    key = dict(reset="dones", invalid="physics_invalid", goal_boundary="actor_phys_valid")[reason]
    batch[key].fill_(reason != "goal_boundary")
    monkeypatch.setattr(balance, "objective", lambda *args: pytest.fail("Invalid rows reached loss"))
    loss, metrics = physics.objective(a, batch, actions, context)
    assert loss == 0 and metrics["base_balance_raw"] == 0


def test_config_validation_and_thin_inheritance(monkeypatch):
    from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfgPPO
    from legged_gym.utils.helpers import class_to_dict
    monkeypatch.setattr(B1Z1PACTCfgPPO.algorithm, "actor_phys_base_balance_enabled", True)
    for number in range(4, 12):
        _, cfg = make_b1z1_pact_ablation_configs(number)
        assert balance.enabled(class_to_dict(cfg().algorithm)) == cfg.algorithm.actor_phys_enabled
        assert cfg.algorithm.actor_phys_base_balance_weight == B1Z1PACTCfgPPO.algorithm.actor_phys_base_balance_weight
    a, *_ = fixture()
    balance.configure(a)
    for key, value in (("actor_phys_base_balance_weight", -1.), ("actor_phys_base_balance_weight", float("nan")),
                       ("actor_phys_base_force_scale", 0.), ("actor_phys_base_force_axis_weights", [-1., 1.]),
                       ("actor_phys_rejection_min_contacts", 0), ("actor_phys_live_grf_enabled", False)):
        old = a.cfg[key]
        a.cfg[key] = value
        with pytest.raises(ValueError):
            balance.configure(a)
        a.cfg[key] = old


def test_minimal_ppo_update_with_compact_sampling():
    from test_b1z1_sampled_context import class_to_dict, B1Z1PACTCfgPPO, B1Z1PACTDecoder, PPO_B1Z1PACT
    template, batch, *_ = fixture()
    full = class_to_dict(B1Z1PACTCfgPPO())
    cfg = {**full["algorithm"], **full["policy"], **template.cfg,
           "num_learning_epochs": 2, "num_mini_batches": 1, "actor_phys_sample_fraction": .5,
           "representation_pinn_enabled": False, "dynamics_backend": "bard",
           "privileged_force_start": 23, "privileged_force_dim": 21,
           "pinn_init_steps": 0, "pinn_warmup": 2, "pinn_loss_weight": .1,
           "actor_phys_ee_stability_weight": 0.}
    fixed = template.actor_physics_cache
    backend = SimpleNamespace(batch_capacity=3, ee_position=template.dynamics_backend.ee_position,
        evaluate=lambda *args: SimpleNamespace(**{k: v[:len(args[0])] for k, v in vars(fixed).items()}))
    a = PPO_B1Z1PACT(template.actor_critic, B1Z1PACTDecoder(22, 188, hidden=[16]), backend, cfg, "cpu")
    a.pinn_updates, a.rejection_progress = 1, .5
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
    assert result["ActorPhysics/base_balance_raw"] > 0
    assert result["ActorPhysics/base_balance_effective_weight"] == pytest.approx(.5 * .5 * .01 * .5)
    assert result["ActorPhysics/base_balance_active_fraction"] == 1
    assert result["ActorPhysics/unintended_estimator_gradient_max"] == 0
    assert a.actor_physics_cache is None
    assert all(torch.isfinite(p).all() for p in a.actor_critic.parameters())
