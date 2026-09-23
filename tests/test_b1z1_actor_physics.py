"""Actor-only task prediction with synthetic mechanics, no simulator required."""
from types import SimpleNamespace

import pytest
import torch

from test_b1z1_sampled_context import make_model
from rsl_rl.algorithms import b1z1_actor_physics as physics
from rsl_rl.algorithms.ppo_b1z1_pact import PPO_B1Z1PACT


def config():
    return dict(actor_phys_enabled=True, actor_phys_coef=.01,
                actor_phys_velocity_time_constant=.25, actor_phys_softplus_temperature=.05,
                actor_phys_huber_delta=1., actor_phys_ee_scale=.1, actor_phys_q_scale=1.,
                actor_phys_qd_scale=10., actor_phys_q_margin=.05, actor_phys_qd_margin=.5,
                actor_phys_vel_weight=1., actor_phys_ee_weight=1., actor_phys_q_weight=.1,
                actor_phys_qd_weight=.1, dt=.02, base_velocity_scale=[2., 2., .25],
                position_action_scale=.5, torque_action_scale=10., clip_actions=100.,
                grf_scale=.001, base_wrench_scale=[.01]*6, ee_force_scale=.01)


def setup(device="cpu", dtype=torch.float32):
    model = make_model(False).to(device=device, dtype=dtype)
    obs = torch.zeros(3, 81, device=device, dtype=dtype)
    model.update_distribution(obs, obs.new_zeros(3, 162), detach_context=True)
    state = obs.new_zeros(3, 180)
    state[:, 6] = 1
    state[:, 97:116] = 1
    state[:, 116:135] = 10
    state[:, 135:154] = 1
    state[:, 154:156] = 1
    mass = torch.eye(25, device=device, dtype=dtype).repeat(3, 1, 1)
    # SPD coupling lets joint torques influence both unactuated planar coordinates.
    mass[:, 0, 6] = mass[:, 6, 0] = .1
    mass[:, 1, 7] = mass[:, 7, 1] = .1
    fixed = SimpleNamespace(mass_matrix=mass, bias=obs.new_zeros(3, 25),
                            foot_jacobians=obs.new_zeros(3, 4, 3, 25),
                            base_jacobian=obs.new_zeros(3, 6, 25), ee_jacobian=obs.new_zeros(3, 6, 25))
    data = dict(state=state, command=obs.new_tensor([.5, 0., 0.]).expand(3, -1),
                ee_target=obs.new_ones(3, 3) * .1, q_min=obs.new_full((3, 19), -2.),
                q_max=obs.new_full((3, 19), 2.), qd_max=obs.new_full((3, 19), 20.),
                torque_max=obs.new_full((3, 19), 100.), mass_wrench=obs.new_zeros(3, 6),
                valid=torch.ones(3, 1, device=device, dtype=torch.bool))
    batch = {"actor_phys_"+k: v for k, v in data.items()}
    batch.update(indices=torch.arange(3, device=device), dones=torch.zeros(3, 1, device=device, dtype=torch.bool),
                 physics_invalid=obs.new_zeros(3, 1), nominal_torque=obs.new_zeros(3, 19))
    a = SimpleNamespace(cfg=config(), actor_critic=model, actor_physics_cache=fixed,
                        dynamics_backend=SimpleNamespace(ee_position=lambda x: x[:, :3]+x[:, 19:22]))
    a._coupled_torque = lambda action, s: PPO_B1Z1PACT._coupled_torque(a, action, s)
    return a, batch, model.action_mean, model.last_context


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_finite_actor_only_gradients(device, dtype):
    a, batch, actions, context = setup(device, dtype)
    loss, metrics = physics.objective(a, batch, actions, context)
    assert loss.dtype == dtype and loss.device.type == device
    assert torch.isfinite(loss) and metrics["active_fraction"] == 1
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.actor_critic.parameters())
    for module in (a.actor_critic.explicit_decoder, a.actor_critic.physics_decoder, a.actor_critic.context_encoder):
        assert all(p.grad is None for p in module.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.actor_critic.parameters())


def test_no_observed_successor_dependency():
    a, batch, actions, context = setup()
    one, _ = physics.objective(a, batch, actions, context)
    batch["dynamics_state"] = torch.full((3, 180), float("nan"))
    batch["next_privileged"] = torch.full((3, 232), float("nan"))
    two, _ = physics.objective(a, batch, actions, context)
    torch.testing.assert_close(one, two, rtol=0, atol=0)


@pytest.mark.parametrize("reason", ["reset", "teleport", "force_nan", "singular", "target_nan", "goal_boundary"])
def test_invalid_rows_zero(reason):
    a, batch, actions, context = setup()
    if reason == "reset":
        batch["dones"].fill_(True)
    elif reason == "teleport":
        batch["physics_invalid"].fill_(1)
    elif reason == "force_nan":
        context["ee_force"] = torch.full_like(context["ee_force"], float("nan"))
    elif reason == "target_nan":
        batch["actor_phys_ee_target"].fill_(float("nan"))
    elif reason == "goal_boundary":
        batch["actor_phys_valid"].fill_(False)
    else:
        a.actor_physics_cache.mass_matrix.zero_()
    loss, metrics = physics.objective(a, batch, actions, context)
    assert loss == 0 and metrics["active_fraction"] == 0
    grad, = torch.autograd.grad(loss, actions)
    assert torch.equal(grad, torch.zeros_like(grad))


def test_component_monotonicity():
    a, batch, _, _ = setup()
    data = {k[11:]: v for k, v in batch.items() if k.startswith("actor_phys_")}
    state = data["state"][:, :51].clone()
    ee = data["ee_target"].clone()
    base = physics.components(state, ee, data, a.cfg)
    state[:, 26] = -3
    state[:, 7:26] = 3
    state[:, 32:51] = 30
    worse = physics.components(state, ee+1, data, a.cfg)
    for name in ("vel", "ee", "q", "qd"):
        assert (worse[name] > base[name]).all()


@pytest.mark.parametrize("flag,coef", [(False, 1.), (True, 0.)])
def test_disabled_exact_backward(flag, coef):
    param = torch.tensor(2., requires_grad=True)
    a = SimpleNamespace(cfg={"actor_phys_enabled": flag, "actor_phys_coef": coef})
    # No storage, backend, extra modules or RNG draws are needed when disabled.
    before = torch.random.get_rng_state()
    physics.backward(a, {}, param.square(), None, None)
    assert param.grad == 4
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("maximum,scheduled,expected", [(2., 0., 0.), (2., 1., .005),
    (-2., 1., .005), (-2., 2., .01), (0., 0., 0.)])
def test_shared_schedule(maximum, scheduled, expected):
    a = SimpleNamespace(cfg={**config(), "pinn_loss_weight": maximum}, pinn_weight=scheduled)
    assert physics.scheduled_coefficient(a) == pytest.approx(expected)


def test_warmup_skips_physics_and_preserves_ppo_backward():
    a = SimpleNamespace(cfg={**config(), "pinn_loss_weight": -1.}, pinn_weight=0.)
    physics.prepare(a)  # No storage or backend needed before the common ramp opens.
    p = torch.tensor(2., requires_grad=True)
    physics.backward(a, {}, p.square(), None, None)
    assert p.grad == 4


@pytest.mark.parametrize("sign,method", [(1., "pc_backward_pinn"), (-1., "pc_backward_ppgrad")])
def test_actor_uses_shared_projection(monkeypatch, sign, method):
    from unittest.mock import Mock
    a, batch, actions, context = setup()
    a.cfg["pinn_loss_weight"] = sign
    a.pinn_weight = .5
    a.actor_optimizer = Mock()
    groups, _ = a.actor_critic.get_optim_groups()
    a.ppo_parameters = [p for group in groups for p in group["params"]]
    a.decoder_parameters = list(a.actor_critic.explicit_decoder.parameters())
    a.actor_physics_metrics = {}
    loss = actions.square().mean()
    monkeypatch.setattr(physics, "objective", lambda *args: (loss, {"active_fraction": loss.new_tensor(1.)}))
    ppo = actions.sum()
    physics.backward(a, batch, ppo, actions, context)
    assert [call[0] for call in a.actor_optimizer.mock_calls] == [method]
    objectives = getattr(a.actor_optimizer, method).call_args.args[0]
    assert objectives[0] is ppo
    torch.testing.assert_close(objectives[1], .005 * loss)


def test_pose_integration_stationary_quaternion_backward():
    initial = torch.zeros(2, 51, dtype=torch.float64)
    initial[:, 6] = 1
    v = torch.zeros(2, 25, dtype=torch.float64, requires_grad=True)
    result = physics.integrate_pose(initial, v, .02)
    torch.testing.assert_close(result, initial)
    result.sum().backward()
    assert torch.isfinite(v.grad).all()


def test_enabled_ppo_update_and_snapshot_alignment():
    from test_b1z1_sampled_context import class_to_dict, B1Z1PACTCfgPPO, B1Z1PACTDecoder
    template, batch, _, _ = setup()
    full = class_to_dict(B1Z1PACTCfgPPO())
    cfg = {**full["algorithm"], **full["policy"], **config(), "num_learning_epochs": 1,
           "num_mini_batches": 1, "dynamics_backend": "bard", "privileged_force_start": 23,
           "privileged_force_dim": 21, "pinn_start_env_step": 0, "pinn_warmup_env_steps": 4,
           "actor_phys_pos_fk_enabled": False,  # This fixture supplies mechanics, not an FK model.
           "pinn_loss_weight": .1}
    fixed = template.actor_physics_cache
    backend = SimpleNamespace(batch_capacity=3, ee_position=template.dynamics_backend.ee_position,
        evaluate=lambda *args: SimpleNamespace(**{k: v[:len(args[0])] for k, v in vars(fixed).items()}))
    a = PPO_B1Z1PACT(template.actor_critic, B1Z1PACTDecoder(8 + 14, 188, hidden=[16]), backend, cfg, "cpu")
    a.init_storage(3, 2, 81, 40, 162, 34, 23, 232, 180, rollout_state_dim=51)
    for step in range(2):
        a.act(torch.randn(3, 81), torch.randn(3, 40), torch.randn(3, 162), torch.zeros(3, 23))
        snapshots = {k[11:]: v.clone() for k, v in batch.items() if k.startswith("actor_phys_")}
        a.transition.actor_physics = snapshots
        a.transition.nominal_torque = torch.zeros(3, 19)
        with torch.inference_mode():
            a.process_env_step(torch.ones(3), torch.zeros(3, dtype=torch.bool), {}, torch.zeros(3, 232),
                               snapshots["state"], snapshots["state"][:, :51])
        assert not a.storage.actor_physics["state"].is_inference()
        snapshots["ee_target"].fill_(99.)
        assert (a.storage.actor_physics["ee_target"][step] == .1).all()
    a.compute_returns(torch.zeros(3, 40))
    metrics = a.update(2)  # Two completed control steps: halfway through warmup.
    assert metrics["ActorPhysics/active_fraction"] == 1
    assert metrics["ActorPhysics/actor_gradient_norm"] > 0
    assert metrics["ActorPhysics/unintended_estimator_gradient_max"] == 0
    assert metrics["ActorPhysics/loss"] > 0
    assert a.pinn_weight == .05
    assert metrics["ActorPhysics/scheduled_coefficient"] == cfg["actor_phys_coef"] * .5
    assert all(torch.isfinite(p).all() for p in a.actor_critic.parameters())


def test_capture_reuses_predicted_force_projection_without_mutating_environment():
    from test_b1z1_force_target_projection import ForceTargetProjectionTests
    a, batch, _, _ = setup()
    env = ForceTargetProjectionTests._environment([[.5, 0, 0]]*3, [[100., 0, 0]]*3)
    env.ee_start_sphere = torch.tensor([[.5, 0, 0]]*3)
    env.ee_goal_sphere = torch.tensor([[.7, 0, 0]]*3)
    env.goal_timer = torch.zeros(3)
    env.traj_timesteps = torch.full((3,), 2.)
    env.traj_total_timesteps = torch.full((3,), 4.)
    env.commands = torch.zeros(3, 6)
    env.gripper_force_kps.fill_(1000.)
    env.get_pact_dynamics_state = lambda: batch["actor_phys_state"]
    env.get_mass_wrench_label = lambda: torch.zeros(3, 6)
    env.simulator.dof_pos_limits = torch.stack((batch["actor_phys_q_min"], batch["actor_phys_q_max"]), -1)
    env.simulator.dof_vel_limits = batch["actor_phys_qd_max"]
    env.simulator.torque_limits = batch["actor_phys_torque_max"]
    a.actor_critic.last_context["ee_force"] = torch.tensor([[.01, 0, 0]]*3)
    a.device, a.transition = "cpu", SimpleNamespace()
    before = torch.random.get_rng_state()
    physics.capture(SimpleNamespace(alg=a, env=env))
    # Halfway through the quintic trajectory gives .6 m; predicted force adds .001 m.
    torch.testing.assert_close(a.transition.actor_physics["ee_target"], torch.tensor([[.601, 0, 0]]*3))
    assert torch.equal(env.goal_timer, torch.zeros(3))
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(env.ee_force_ext_world, torch.tensor([[100., 0, 0]]*3))
