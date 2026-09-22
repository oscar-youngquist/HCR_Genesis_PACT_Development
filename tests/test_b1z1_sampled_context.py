"""Sample replay, decoder detach boundaries and transition-aligned training."""

from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfgPPO
from legged_gym.envs.b1z1.b1z1_pact_pos.b1z1_pact_pos_config import B1Z1PACTPosCfgPPO
from rsl_rl.modules.actor_critic_b1z1_pact import ActorCriticB1Z1PACT, B1Z1PACTDecoder
from rsl_rl.modules.actor_critic_b1z1_pact_pos import ActorCriticB1Z1PACTPos
from rsl_rl.algorithms.ppo_b1z1_pact import PPO_B1Z1PACT
from rsl_rl.algorithms.ppo_b1z1_pact_pos import PPO_B1Z1PACTPos
from rsl_rl.storage.b1z1_action_replay import B1Z1ActionReplay


def make_model(pos):
    cls = ActorCriticB1Z1PACTPos if pos else ActorCriticB1Z1PACT
    return cls(81, 40, 17, 162, latent_dim=8, actor_layers=[32, 16],
               critic_layers=[32, 16], context_layers=[32, 16],
               explicit_decoder_layers=[16], force_decoder_layers=[16], grf_decoder_layers=[16])


@pytest.mark.parametrize("pos", [False, True])
def test_sample_replay_and_gradient_boundaries(pos):
    model = make_model(pos)
    obs, history = torch.randn(4, 81), torch.randn(4, 162)
    model.act(obs, history)
    noise = model.last_context["latent_noise"].detach().clone()
    original = model.action_mean.detach().clone()
    rng = torch.random.get_rng_state()
    model.update_distribution(obs, history, latent_noise=noise)
    torch.testing.assert_close(model.action_mean, original)
    assert torch.equal(torch.random.get_rng_state(), rng)
    ctx = model.last_context
    assert ctx["explicit_condition"].shape == (4, 14)
    assert ctx["base_wrench"].shape == (4, 6)
    assert ctx["ee_force"].shape == (4, 3)
    torque = torch.randn(4, 19, requires_grad=True)
    model.zero_grad(set_to_none=True)
    model.predict_grf(ctx, torque).square().mean().backward()
    assert torque.grad is None
    assert all(p.grad is None for p in model.explicit_decoder.parameters())
    assert any(p.grad is not None for p in model.context_encoder.parameters())
    model.zero_grad(set_to_none=True)
    ctx = model.decode_context(model.context_encoder(history))
    ctx["base_wrench"].square().mean().backward()
    assert all(p.grad is None for p in model.explicit_decoder.parameters())
    assert any(p.grad is not None for p in model.context_encoder.parameters())


@pytest.mark.parametrize("pos,bard_active,backend", [
    (False, False, "bard"), (True, False, "bard"),
    (False, True, "bard"), (False, -1, "bard"), (False, False, "pinocchio")])
def test_short_ppo_update_and_auxiliary_targets(pos, bard_active, backend):
    model = make_model(pos)
    config = class_to_dict((B1Z1PACTPosCfgPPO if pos else B1Z1PACTCfgPPO)())
    cfg = {**config["algorithm"], **config["policy"], "num_learning_epochs": 1,
           "actor_phys_enabled": False,  # This fixture does not collect task-predictive snapshots.
           "num_mini_batches": 1, "privileged_force_start": 23, "privileged_force_dim": 21,
           "position_action_scale": 0.25, "torque_action_scale": 1.,
           "dof_pos_obs_scale": 1., "dof_vel_obs_scale": 1., "dt": .02,
           "grf_scale": .001, "ee_force_scale": .01,
           "base_velocity_scale": [1., 1., 1.], "base_wrench_scale": [1.] * 6}
    decoder = B1Z1PACTDecoder(8 + 14, 188, hidden=[16])
    cfg["dynamics_backend"] = backend
    alg = (PPO_B1Z1PACTPos(model, decoder, cfg, "cpu") if pos else
           PPO_B1Z1PACT(model, decoder, SimpleNamespace(), cfg, "cpu"))
    if bard_active:
        def evaluate(*args):
            n = len(args[0])
            return SimpleNamespace(mass_matrix=torch.eye(25).repeat(n, 1, 1),
                bias=torch.zeros(n, 25), foot_jacobians=torch.ones(n, 4, 6, 25),
                ee_jacobian=torch.ones(n, 6, 25), base_jacobian=torch.ones(n, 6, 25))
        alg.dynamics_backend = SimpleNamespace(evaluate=evaluate, batch_capacity=3)
        alg.cfg.update(pinn_init_steps=0, pinn_warmup=1, pinn_loss_weight=.1 * bard_active, use_pinn_rollout_loss=True)
        alg.pinn_updates = 1
        projection = "pc_backward_ppgrad" if bard_active < 0 else "pc_backward_pinn"
        for optimizer in (alg.encoder_pcgrad, alg.decoder_pcgrad):
            setattr(optimizer, projection, Mock(wraps=getattr(optimizer, projection)))
    alg.init_storage(4, 2, 81, 40, 162, 34, 23, 232, 76 if pos else 180,
                     policy_distribution_dim=17 if pos else 34,
                     rollout_state_dim=0 if pos else 51)
    for _ in range(2):
        obs, history, critic = torch.randn(4, 81), torch.randn(4, 162), torch.randn(4, 40)
        labels = torch.randn(4, 23)
        labels[:, 15:19] = torch.randint(0, 2, (4, 4)).float()
        alg.act(obs, critic, history, labels)
        before = alg.transition.explicit_targets.clone()
        labels.add_(100.)
        torch.testing.assert_close(alg.transition.explicit_targets, before)
        alg.transition.nominal_torque = torch.randn(4, 19)
        state = torch.ones(4, 76 if pos else 180)
        if pos:
            alg.process_env_step(torch.randn(4), torch.zeros(4, dtype=torch.bool), {}, torch.randn(4, 232), state)
        else:
            initial = torch.zeros(4, 51)
            initial[:, 6] = 1.
            alg.process_env_step(torch.randn(4), torch.zeros(4, dtype=torch.bool), {}, torch.randn(4, 232), state, initial)
    alg.compute_returns(torch.randn(4, 40))
    # PPO backward has no path into context or decoder parameters.
    batch = next(alg.storage.mini_batches(1, 1))
    alg._compute_rl_loss(batch)[0].backward()
    assert all(p.grad is None for p in alg.enc_parameters + alg.decoder_parameters)
    alg.actor_optimizer.zero_grad()
    owners = [set(map(id, parameters)) for parameters in
              (alg.ppo_parameters, alg.enc_parameters, alg.decoder_parameters)]
    assert not (owners[0] & owners[1] or owners[0] & owners[2] or owners[1] & owners[2])
    assert set(map(id, (*model.context_encoder.parameters(), *model.explicit_decoder.parameters()))) == owners[1]
    metrics = alg.update(0)
    assert metrics["pre_update_mu_rms"] < 1e-6
    assert metrics["pre_update_logprob_rms"] < 1e-5
    assert metrics["grf_decoder"] >= 0
    if bard_active:
        assert alg.pinn_weight == .1
        for optimizer in (alg.encoder_pcgrad, alg.decoder_pcgrad):
            getattr(optimizer, projection).assert_called_once()
        assert metrics["pinn_encoder_inverse"] > 0
        assert metrics["pinn_decoder_rollout"] > 0
        for phase in ("", "encoder/", "decoder/"):
            for name in ("inverse/loss_raw", "rollout/loss_raw",
                         "inverse/arm_gripper_mae_Nm", "rollout/base_linear_mae_mps"):
                assert metrics[f"PINN/{phase}{name}"] >= 0
            for component in ("inverse", "rollout"):
                prefix = f"PINN/{phase}{component}"
                raw = metrics[f"{prefix}/loss_unscaled"]
                assert raw == metrics[f"{prefix}/loss_raw"]
                weighted = raw * alg.cfg.get(f"pinn_{component}_weight", 1.0)
                assert metrics[f"{prefix}/loss_component_weighted"] == pytest.approx(weighted)
                assert metrics[f"{prefix}/loss_scaled"] == pytest.approx(alg.pinn_weight * weighted)
        assert metrics["PINN/weighted_combined_loss"] == pytest.approx(
            alg.pinn_weight * metrics["PINN/combined_loss"])
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # Invalid/reset targets are excluded before NaN-bearing decoder arithmetic.
    empty = alg._compute_vae_loss(history, torch.full((4, 232), float("nan")),
        labels, torch.zeros(4, 1), 0, torch.randn(4, 19))
    assert empty["grf_decoder"] == 0


def test_delayed_source_replay_and_reset():
    model = make_model(False)
    queue = B1Z1ActionReplay(2)
    delay = torch.tensor([0, 1, 2])
    sources = []
    for _ in range(3):
        obs, history = torch.randn(3, 81), torch.randn(3, 162)
        actions = model.act(obs, history).detach()
        transition = SimpleNamespace(observations=obs, histories=history, actions=actions,
            latent_noise=model.last_context["latent_noise"].detach(),
            mu=model.action_mean.detach(), sigma=model.action_std.detach())
        source = queue.push(transition, delay)
        sources.append((obs.clone(), history.clone(), actions.clone()))
    alg = object.__new__(PPO_B1Z1PACT)
    alg.actor_critic, alg.cfg = model, {"clip_actions": 0.5}
    rng = torch.random.get_rng_state()
    replayed, valid = alg._physics_actions({"physics_source": source,
        "observations": obs, "histories": history, "actions": actions,
        "latent_noise": transition.latent_noise})
    expected = torch.stack([sources[2-i][2][i] for i in range(3)]).clamp(-0.5, 0.5)
    torch.testing.assert_close(replayed, expected, atol=1e-6, rtol=1e-5)
    assert valid.all() and torch.equal(rng, torch.random.get_rng_state())
    queue.reset(torch.tensor([False, True, False]))
    source = queue.push(transition, delay)
    assert source[1, -1] == 0


def test_torque_condition_normalization():
    model = make_model(False)
    context = model.decode_context(model.context_encoder(torch.randn(2, 162)))
    inputs = []
    hook = model.physics_decoder.grf.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone()))
    model.predict_grf(context, torch.full((2, 19), 100.))
    hook.remove()
    torch.testing.assert_close(inputs[0][:, -19:], torch.ones(2, 19))


def test_inverse_controller_uses_pre_step_state():
    alg = object.__new__(PPO_B1Z1PACT)
    alg.cfg = {"base_velocity_scale": [1.] * 3, "dt": .02}
    state = torch.ones(2, 180)
    initial = torch.zeros(2, 51)
    initial[:, 6] = 1.
    captured = []
    alg._resolve_pinn_forces = lambda *args: (torch.zeros(2, 12), torch.zeros(2, 3), torch.zeros(2, 6))
    alg.dynamics_backend = SimpleNamespace(evaluate=lambda *args: SimpleNamespace(
        mass_matrix=torch.eye(25).repeat(2, 1, 1), bias=torch.zeros(2, 25),
        generalized_contacts=torch.zeros(2, 25)))
    def torque(actions, controller_state):
        captured.append(controller_state.clone())
        return actions[:, :19]
    alg._coupled_torque = torque
    loss = alg._pinn_loss(torch.ones(2, 34), {"rollout_initial_state": initial},
                         torch.zeros(2, 21), state, torch.ones(2, 1))
    torch.testing.assert_close(captured[0][:, :51], initial)
    torch.testing.assert_close(captured[0][:, 97:], state[:, 97:])
    assert torch.isfinite(loss)
