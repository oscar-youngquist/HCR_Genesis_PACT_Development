"""Focused CPU contracts; no simulator startup or training run."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils import task_registry
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg, B1Z1PACTCfgPPO
from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos import B1Z1PPOPos
from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos_config import B1Z1PPOPosCfg, B1Z1PPOPosCfgPPO
from legged_gym.envs.b1z1.force_task_utils import _compute_force_adjusted_ee_target
from legged_gym.torque_action_scaling import resolve_torque_action_scale
from rsl_rl.modules.actor_critic_b1z1_ppo_pos import ActorCriticB1Z1PPOPos
from rsl_rl.algorithms.ppo_b1z1_ppo_pos import PPO_B1Z1PPOPos
from rsl_rl.runners.b1z1_ppo_pos_runner import B1Z1PPOPosRunner
from legged_gym.envs.b1z1.force_task_utils import B1Z1StagedForceCurriculum


def build():
    torch.set_num_threads(1)
    cfg, train = B1Z1PPOPosCfg(), B1Z1PPOPosCfgPPO()
    model = ActorCriticB1Z1PPOPos(
        cfg.env.num_observations, cfg.env.num_privileged_obs * cfg.env.num_priv_stack,
        cfg.env.num_actions, **class_to_dict(train.policy))
    return cfg, train, model


def test_model_architecture_and_optimizer():
    cfg, train, model = build()
    obs = torch.randn(2, cfg.env.num_observations)
    assert model.act(obs).shape == (2, cfg.env.num_actions)
    assert model.get_actions_log_prob(model.action_mean).shape == (2,)
    assert model.entropy.shape == (2,)
    assert [m.out_features for m in model.actor_trunk.modules() if isinstance(m, nn.Linear)] == train.policy.actor_layers
    assert [m.out_features for m in model.critic if isinstance(m, nn.Linear)][:-1] == train.policy.critic_layers
    assert torch.equal(model.std, torch.tensor(train.policy.init_noise_std[:cfg.env.num_actions]))
    for name in ("context_encoder", "explicit_decoder", "physics_decoder", "privileged_decoder", "film", "torque_head"):
        assert not hasattr(model, name)
    alg = PPO_B1Z1PPOPos(model, class_to_dict(train.algorithm), "cpu")
    owned = [id(p) for g in alg.optimizer.param_groups for p in g["params"]]
    assert len(owned) == len(set(owned)) == len(list(model.parameters()))
    assert not hasattr(alg, "auxiliary_optimizer")


def test_configuration_inheritance_and_registration():
    cfg, train, _ = build()
    original = B1Z1PACTCfg()
    for name in ("rewards", "control", "terrain", "commands", "domain_rand", "normalization"):
        assert class_to_dict(getattr(cfg, name)) == class_to_dict(getattr(original, name))
    assert cfg.env.num_policy_actions == cfg.env.num_actions
    assert original.env.num_policy_actions == 2 * original.env.num_actions
    for name, value in class_to_dict(B1Z1PACTCfgPPO().algorithm).items():
        assert getattr(train.algorithm, name) == value
    assert cfg.commands.apply_ee_external_forces and cfg.commands.apply_base_external_forces
    assert not cfg.use_force_compensation and not cfg.use_force_shifted_target
    assert task_registry.task_classes["b1z1_ppo_pos"] is B1Z1PPOPos
    assert task_registry.task_classes["b1z1_pact"] is B1Z1PACT


def test_action_contract_and_history(monkeypatch):
    cfg, _, _ = build()
    cfg.domain_rand.randomize_ctrl_delay = False
    env = B1Z1PPOPos.__new__(B1Z1PPOPos)
    env.cfg, env.device, env.num_envs, env.num_actions = cfg, "cpu", 2, cfg.env.num_actions
    env.actions = torch.zeros(2, 2 * env.num_actions)
    env.last_actions = env.actions.clone()
    env.llast_actions = env.actions.clone()
    pos = torch.randn(2, env.num_actions)
    execution = env._pre_sim_step(pos)
    assert torch.equal(execution[:, env.position_history_slice], pos)
    assert execution[:, env.torque_history_slice].count_nonzero() == 0
    with pytest.raises(ValueError):
        env._pre_sim_step(execution)
    torque = torch.randn_like(pos)
    env.simulator = SimpleNamespace(_torques=torque, _cfg=cfg, dof_pos=pos)
    monkeypatch.setattr(B1Z1PACT, "post_physics_step", lambda self: None)
    env.post_physics_step()
    assert torch.equal(env.actions[:, env.position_history_slice], pos)
    assert torch.allclose(env.actions[:, env.torque_history_slice], torque / resolve_torque_action_scale(cfg, pos))
    previous = env.actions.clone()
    env._pre_sim_step(torch.zeros_like(pos))
    assert torch.equal(env.last_actions, previous)


def test_nominal_target_ignores_disturbance():
    cfg = B1Z1PPOPosCfg()
    cfg.goal_ee.project_force_adjusted_ee_target = False
    nominal = torch.tensor([[.5, 0., .2], [.6, .1, .2]])
    quat = torch.tensor([[0.,0.,0.,1.]]).expand(2,-1)
    env = SimpleNamespace(cfg=cfg, ee_force_ext_world=torch.full_like(nominal, 100.),
                          curr_ee_goal_cart_world=nominal, gripper_force_kps=torch.ones_like(nominal),
                          _get_base_yaw_quat=lambda ids: quat if ids is None else quat[ids],
                          get_ee_goal_spherical_center=lambda q, ids: torch.zeros_like(q[:, :3]))
    result = _compute_force_adjusted_ee_target(env)
    assert torch.equal(result.effective_target, nominal)
    assert result.applied_offset.count_nonzero() == 0
    cfg.use_force_shifted_target = True
    assert not torch.equal(_compute_force_adjusted_ee_target(env).effective_target, nominal)


def test_tiny_synthetic_ppo_update():
    cfg, train, model = build()
    settings = class_to_dict(train.algorithm)
    settings.update(num_learning_epochs=1, num_mini_batches=1)
    alg = PPO_B1Z1PPOPos(model, settings, "cpu")
    critic_dim = cfg.env.num_privileged_obs * cfg.env.num_priv_stack
    alg.init_storage(2, 2, [cfg.env.num_observations], [critic_dim], [cfg.env.num_actions])
    before = model.position_head.weight.detach().clone()
    with torch.inference_mode():
        for _ in range(2):
            obs, critic = torch.randn(2, cfg.env.num_observations), torch.randn(2, critic_dim)
            alg.act(obs, critic)
            obs.zero_()  # Transition is an independent snapshot.
            assert alg.transition.observations.count_nonzero() > 0
            alg.process_env_step(torch.ones(2), torch.zeros(2), {})
        alg.compute_returns(critic)
    losses = alg.update()
    assert torch.isfinite(torch.tensor(losses)).all()
    assert not torch.equal(before, model.position_head.weight)
    assert not hasattr(alg.storage, "histories")


def test_runner_adapter_and_checkpoint(tmp_path):
    cfg, train, _ = build()
    # Only test runtime/batch size is reduced; widths and layouts remain inherited.
    settings = class_to_dict(train)
    settings["runner"]["num_steps_per_env"] = 2
    settings["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
    obs = torch.zeros(2, cfg.env.num_observations)
    critic = torch.zeros(2, cfg.env.num_privileged_obs * cfg.env.num_priv_stack)
    env = SimpleNamespace(
        cfg=cfg, num_envs=2, num_obs=cfg.env.num_observations,
        num_privileged_obs=cfg.env.num_privileged_obs, num_crit_obs_stack=cfg.env.num_priv_stack,
        num_actions=cfg.env.num_actions, simulator=SimpleNamespace(),
        reset=lambda: (obs, critic), get_observations=lambda: (obs,None,critic,None),
        _staged_force_curriculum=B1Z1StagedForceCurriculum(cfg.commands),
        step=lambda actions: (obs,critic,None,None,torch.ones(2),torch.zeros(2),{},None),
    )
    iterations = []
    env.set_training_iteration = iterations.append
    runner = B1Z1PPOPosRunner(env, settings, log_dir=None, device="cpu")
    runner.learn(1)
    assert iterations == [0,1]
    assert env.bard_mass_wrench_labels == (train.algorithm.dynamics_backend == "bard")
    assert runner.get_inference_policy()(obs).shape == (2,cfg.env.num_actions)
    checkpoint = tmp_path / "ppo_pos.pt"
    runner.save(checkpoint)
    restored = B1Z1PPOPosRunner(env, settings, log_dir=None, device="cpu")
    restored.load(checkpoint)
    assert restored.current_learning_iteration == 1
    assert torch.equal(runner.actor_critic.std, restored.actor_critic.std)
    assert not hasattr(runner, "dynamics")
