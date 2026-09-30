"""Focused generated-task and shared-path tests, without simulator construction."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
import legged_gym.envs
from legged_gym.envs.b1z1.b1z1_pact import B1Z1PACT, B1Z1PACTCfg, B1Z1PACTCfgPPO
from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos import B1Z1PPOPos
from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.b1z1_pact_ablations import B1Z1_PACT_ABLATIONS
from rsl_rl.modules.actor_critic_b1z1_pact import ActorCriticB1Z1PACT
from rsl_rl.algorithms import b1z1_actor_physics as actor_physics, b1z1_bard_pinn as representation
from rsl_rl.algorithms.ppo_b1z1_pact import PPO_B1Z1PACT


def make_model(action_mode="coupled", conditioning_mode="film"):
    e, p = B1Z1PACTCfg.env, B1Z1PACTCfgPPO.policy
    return ActorCriticB1Z1PACT(
        e.num_observations, e.num_privileged_obs * e.num_priv_stack,
        e.num_actions, e.num_observations * e.num_obs_hist,
        latent_dim=p.cenet_latent_dim, actor_layers=p.actor_layers,
        critic_layers=p.critic_layers, context_layers=p.cenet_enc_layers,
        explicit_decoder_layers=p.explicit_decoder_layers, explicit_dim=e.num_explicit_recon_obs,
        force_decoder_layers=p.force_decoder_layers, grf_decoder_layers=p.grf_decoder_layers,
        film_hidden_dim=p.film_hidden_dim, activation=p.activation,
        init_noise_std=p.init_noise_std, min_noise_std=p.min_noise_std, max_noise_std=p.max_noise_std,
        action_mode=action_mode, conditioning_mode=conditioning_mode)


def test_complete_matrix_and_registry():
    expected = {
        4: ("coupled", "none", False, False), 5: ("coupled", "concat", False, False),
        6: ("position", "film", False, False), 7: ("position", "film", True, True),
        8: ("coupled", "film", False, False), 9: ("coupled", "film", True, False),
        10: ("coupled", "film", False, True), 11: ("coupled", "film", True, True)}
    names = []
    original = class_to_dict(B1Z1PACTCfg())
    for number, feature in B1Z1_PACT_ABLATIONS.items():
        assert (feature.action_mode, feature.conditioning_mode, feature.representation_pinn_enabled,
                feature.actor_phys_enabled) == expected[number]
        assert task_registry.task_classes[feature.task_name] is (B1Z1PPOPos if feature.action_mode == "position" else B1Z1PACT)
        if number == 11:
            continue
        env, train = make_b1z1_pact_ablation_configs(number)
        assert train.runner_class_name == B1Z1PACTCfgPPO.runner_class_name == "B1Z1PACTRunner"
        assert env.env.num_policy_actions == env.env.num_actions * (1 if feature.action_mode == "position" else 2)
        assert env.env.num_obs_hist == B1Z1PACTCfg.env.num_obs_hist
        assert train.policy.actor_layers == B1Z1PACTCfgPPO.policy.actor_layers
        assert train.algorithm.representation_pinn_enabled == feature.representation_pinn_enabled
        assert train.algorithm.actor_phys_enabled == feature.actor_phys_enabled
        for key in ("rewards", "commands", "domain_rand", "normalization", "control", "terrain"):
            assert class_to_dict(env())[key] == original[key]
        if feature.conditioning_mode != "film":
            assert train.policy.film_identity_loss_weight == 0
        names.append(train.runner.run_name)
    assert len(set(names)) == len(names)
    assert "b1z1_pact_ab11" not in task_registry.task_classes
    with pytest.raises(FrozenInstanceError):
        B1Z1_PACT_ABLATIONS[4].action_mode = "position"
    assert class_to_dict(B1Z1PACTCfg()) == original


@pytest.mark.parametrize("mode", ["none", "concat", "film"])
@pytest.mark.parametrize("action_mode", ["position", "coupled"])
def test_actor_forward(mode, action_mode):
    model = make_model(action_mode, mode)
    e = B1Z1PACTCfg.env
    obs = torch.randn(2, e.num_observations)
    history = torch.randn(2, e.num_observations * e.num_obs_hist)
    seen = []
    handle = model.actor_trunk.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    if mode != "film":
        assert model.film is None
        class Forbidden(torch.nn.Module):
            def forward(self, *args):
                pytest.fail("Disabled FiLM was called")
        model.film = Forbidden()
    action = model.act(obs, history)
    assert action.shape == (2, e.num_actions * (1 if action_mode == "position" else 2))
    assert torch.isfinite(model.get_actions_log_prob(action)).all()
    common, condition = model._actor_inputs(obs, model.last_context)
    expected = torch.cat((common, condition), -1) if mode == "concat" else common
    torch.testing.assert_close(seen[0], expected)
    if action_mode == "position":
        assert model.torque_head is None
        assert torch.count_nonzero(model.last_torque_mean) == 0
        torch.testing.assert_close(model.std, torch.tensor(B1Z1PACTCfgPPO.policy.init_noise_std[:e.num_actions]))
    model.update_distribution(obs, history, latent_noise=model.last_context["latent_noise"], detach_context=True)
    model.action_mean.square().mean().backward()
    assert all(p.grad is None for p in model.context_encoder.parameters())
    handle.remove()


def test_position_torque_mapping_matches_zero_feedforward():
    from test_b1z1_actor_physics import setup
    a, batch, _, _ = setup()
    n = a.actor_critic.num_actions
    position = torch.randn(len(batch["actor_phys_state"]), n)
    torch.testing.assert_close(a._coupled_torque(position, batch["actor_phys_state"]),
        a._coupled_torque(torch.cat((position, torch.zeros_like(position)), -1), batch["actor_phys_state"]))


def test_disabled_physics_and_actor_only_schedule(monkeypatch):
    a = SimpleNamespace(cfg={"representation_pinn_enabled": False})
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled representation PINN was called")
    monkeypatch.setattr(representation, "mechanics", forbidden)
    assert representation.cache_rollout(a) is None
    zero = torch.zeros(1)
    assert representation.losses(None, {"z": zero}, {}, None, a.cfg) == (zero.sum(), zero.sum())
    a.cfg.update(actor_phys_enabled=False)
    actor_physics.capture(SimpleNamespace(alg=a, env=None))
    actor_physics.prepare(a)
    a.cfg.update(actor_phys_enabled=True, actor_phys_coef=.1, pinn_loss_weight=-1.)
    a.pinn_weight = .5
    assert actor_physics.scheduled_coefficient(a) == pytest.approx(.05)


def test_position_environment_retains_history_and_pd_channels(monkeypatch):
    cfg_cls, _ = make_b1z1_pact_ablation_configs(7)
    env = object.__new__(B1Z1PPOPos)
    env.cfg, env.num_actions = cfg_cls(), cfg_cls.env.num_actions
    captured = []
    monkeypatch.setattr(B1Z1PACT, "_init_buffers", lambda self: captured.append(
        (self.cfg.env.num_obs_hist, self.cfg.env.num_policy_actions)))
    env._init_buffers()
    assert captured == [(cfg_cls.env.num_obs_hist, 2*env.num_actions)]
    env.num_envs = 2
    monkeypatch.setattr(B1Z1PACT, "_pre_sim_step", lambda self, actions: actions)
    position = torch.randn(env.num_envs, env.num_actions)
    padded = env._pre_sim_step(position)
    torch.testing.assert_close(padded[:, env.position_history_slice], position)
    assert not padded[:, env.torque_history_slice].any()
    with pytest.raises(ValueError):
        env._pre_sim_step(padded)
    env.actions = padded.clone()
    env.simulator = SimpleNamespace(_cfg=env.cfg,
        dof_pos=torch.zeros(env.num_envs, len(env.cfg.asset.dof_names)),
        _torques=torch.randn(env.num_envs, len(env.cfg.asset.dof_names)))
    monkeypatch.setattr(B1Z1PACT, "post_physics_step", lambda self: None)
    env.post_physics_step()
    from legged_gym.torque_action_scaling import simulator_torque_action_scale
    torch.testing.assert_close(env.actions[:, env.position_history_slice], position)
    torch.testing.assert_close(env.actions[:, env.torque_history_slice],
        env.simulator._torques[:, :env.num_actions] / simulator_torque_action_scale(env.simulator))


def test_no_representation_vjps_when_disabled(monkeypatch):
    """Supervised encoder/decoder steps survive even with a nonzero shared ramp."""
    encoder, decoder = torch.nn.Parameter(torch.ones(())), torch.nn.Parameter(torch.ones(()))
    from rsl_rl.algorithms.pc_grad import PCGrad
    enc_opt, dec_opt = torch.optim.SGD([encoder], lr=.01), torch.optim.SGD([decoder], lr=.01)
    a = SimpleNamespace(cfg={"representation_pinn_enabled": False}, pinn_weight=1.,
        bard_auxiliary=True, encoder_pcgrad=PCGrad(enc_opt), decoder_pcgrad=PCGrad(dec_opt),
        auxiliary_optimizer=enc_opt, decoder_optimizer=dec_opt, enc_parameters=[encoder],
        decoder_parameters=[decoder], max_grad_norm=1.)
    def supervised(**kwargs):
        context = kwargs.get("context_override", {"z": encoder.reshape(1, 1)})
        return {"context": context, "loss": context["z"].square().sum() + decoder.square()}
    a._compute_vae_loss = supervised
    def forbidden(*args, **kwargs):
        pytest.fail("Representation physics/VJP path was called")
    monkeypatch.setattr(representation, "losses", forbidden)
    monkeypatch.setattr(a.encoder_pcgrad, "pc_backward_pinn", forbidden)
    monkeypatch.setattr(a.decoder_pcgrad, "pc_backward_pinn", forbidden)
    batch = {k: torch.zeros(1, 1) for k in ("histories", "next_privileged", "explicit_targets", "nominal_torque")}
    _, inverse, rollout = representation.auxiliary_step(a, batch, torch.ones(1, 1), 0)
    assert inverse == rollout == 0
    assert encoder.item() < 1 and decoder.item() < 1


@pytest.mark.parametrize("variant", [7, 10])
def test_actor_pinn_position_and_actor_only(variant):
    from test_b1z1_actor_physics import setup
    a, batch, actions, context = setup()
    feature = B1Z1_PACT_ABLATIONS[variant]
    a.cfg.update(representation_pinn_enabled=feature.representation_pinn_enabled,
                 actor_phys_enabled=feature.actor_phys_enabled, pinn_loss_weight=-1.)
    a.pinn_weight = 1.
    if feature.action_mode == "position":
        actions = actions[:, :a.actor_critic.num_actions]
    loss, metrics = actor_physics.objective(a, batch, actions, context)
    assert torch.isfinite(loss) and metrics["active_fraction"] == 1
    grad, = torch.autograd.grad(loss, actions)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert actor_physics.scheduled_coefficient(a) > 0


def test_position_ablation_keeps_pinn_cli_override(monkeypatch):
    import importlib
    registry = importlib.import_module("legged_gym.utils.task_registry")
    monkeypatch.setattr(registry, "configure_runtime_device", lambda args: None)
    monkeypatch.setattr(registry, "update_cfg_from_args", lambda env, train, args: (env, train))
    monkeypatch.setattr(registry.runner_registry, "get_runner_class", lambda name:
        lambda env, cfg, log_dir, device: SimpleNamespace(cfg=cfg))
    _, config = make_b1z1_pact_ablation_configs(7)
    runner, _ = task_registry.make_alg_runner(None, B1Z1_PACT_ABLATIONS[7].task_name,
        SimpleNamespace(cpu=True, pinn_loss_weight=-.25), train_cfg=config(), log_root=None)
    assert runner.cfg["policy"]["pinn_loss_weight"] == -.25


@pytest.mark.parametrize("task", [
    "b1z1_pact", "b1z1_pact_pos", "go2_pact", "go2_hard_pact", "b1z1_unifp",
    B1Z1_PACT_ABLATIONS[7].task_name,
])
@pytest.mark.parametrize("explicit", [None, -.25, 0.0, .01])
def test_ablation_parser_preserves_configured_pinn_weight(monkeypatch, explicit, task):
    import sys
    from legged_gym.utils.helpers import get_args
    argv = ["train.py", "--task", task, "--cpu"]
    if explicit is not None:
        argv += ["--pinn_loss_weight", str(explicit)]
    monkeypatch.setattr(sys, "argv", argv)
    assert get_args().pinn_loss_weight == explicit
