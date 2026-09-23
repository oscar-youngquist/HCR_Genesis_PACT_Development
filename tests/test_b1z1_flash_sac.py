"""Small CPU numerical/ownership tests; no full training runs."""
import copy
from types import SimpleNamespace

import pytest
import torch

from test_b1z1_sampled_context import class_to_dict, B1Z1PACTCfgPPO, B1Z1PACTDecoder
from rsl_rl.modules.actor_critic_b1z1_pact import ActorCriticB1Z1PACT
from rsl_rl.algorithms.flash_sac_b1z1_pact import FlashSAC_B1Z1PACT
from rsl_rl.modules.b1z1_flashsac import RepeatedNoise, _compute_categorical_td_target
from rsl_rl.storage.replay_buffer_b1z1_pact import ReplayBufferB1Z1PACT


def make_algorithm(**overrides):
    config = class_to_dict(B1Z1PACTCfgPPO())
    cfg = {**config['algorithm'], **config['policy'], 'actor_phys_enabled': False,
           'privileged_force_start': 23, 'privileged_force_dim': 21,
           'position_action_scale': .25, 'torque_action_scale': 1., 'clip_actions': 1.,
           'dt': .02, 'grf_scale': .001, 'ee_force_scale': .01,
           'base_velocity_scale': [1.] * 3, 'base_wrench_scale': [1.] * 6,
           'sac_critic_width': 16, 'sac_critic_blocks': 1, 'sac_batch_size': 4,
           'sac_position_action_range': 1., 'sac_leg_torque_action_range': 1., 'sac_arm_torque_action_range': 1.,
           'replay_capacity': 32, 'replay_warmup': 4, **overrides}
    model = ActorCriticB1Z1PACT(81, 40, 17, 162, sac=True, latent_dim=8,
        actor_layers=[32, 16], critic_layers=[32, 16], context_layers=[32, 16],
        explicit_decoder_layers=[16], force_decoder_layers=[16], grf_decoder_layers=[16])
    def evaluate(*args):
        n = len(args[0])
        return SimpleNamespace(mass_matrix=torch.eye(25).repeat(n, 1, 1), bias=torch.zeros(n, 25),
            foot_jacobians=torch.ones(n, 4, 6, 25), ee_jacobian=torch.ones(n, 6, 25),
            base_jacobian=torch.ones(n, 6, 25))
    backend = SimpleNamespace(evaluate=evaluate, batch_capacity=3,
                             ee_position=lambda x: x[:, :3] + x[:, 19:22])
    return FlashSAC_B1Z1PACT(model, B1Z1PACTDecoder(22, 188, hidden=[16]), backend, cfg, 'cpu')


def transition(a, step=0, n=2, done=False):
    obs = torch.full((n, 81), float(step))
    hist = torch.full((n, 162), float(step))
    critic = torch.full((n, 40), float(step))
    labels = torch.zeros(n, 23)
    a.act(obs, critic, hist, labels)
    state = torch.zeros(n, 180)
    state[:, 6] = 1
    state[:, 97:116] = 1
    state[:, 116:135] = 10
    state[:, 135:156] = 1
    a.transition.nominal_torque = torch.randn(n, 19)
    a.transition.interval_torque = torch.randn(n, 19)
    a.transition.mass_wrench = torch.zeros(n, 6)
    a.transition.physics_invalid = torch.zeros(n, 1)
    terminal = torch.zeros(n, dtype=torch.bool)
    timeout = torch.zeros_like(terminal)
    if done:
        terminal[:2] = True
        timeout[1] = True
    final = dict(next_observations=obs+1, next_histories=hist+1, next_critic_observations=critic+1)
    a.process_env_step(torch.ones(n), terminal, {'time_outs': timeout, 'flash_sac_final': final},
        torch.zeros(n, 232), state, state[:, :51].clone(),
        **{k: torch.full_like(v, -99.) if done else v for k, v in final.items()})


def test_squash_and_gradient_ownership():
    a = make_algorithm()
    model = a.actor_critic
    obs, hist = torch.randn(4, 81), torch.randn(4, 162)
    mean, std = model.get_mean_and_std(obs, hist)
    torch.testing.assert_close(std, torch.full_like(std, .65))
    torch.testing.assert_close(model.act_inference(obs, hist), mean.tanh())
    action, lp = model.sample_squashed(obs, hist)
    assert (action.abs() <= 1).all() and torch.isfinite(lp).all()
    lp.mean().backward()
    assert all(p.grad is None for p in a.enc_parameters + a.decoder_parameters + list(a.q.parameters()))
    assert model.log_std_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        model.position_head.bias.fill_(100.)
    _, lp = model.sample_squashed(obs, hist)
    assert torch.isfinite(lp).all()


def test_replay_alignment_timeouts_and_wrap():
    a = make_algorithm(replay_capacity=5)
    transition(a, 0)
    transition(a, 1, done=True)
    torch.testing.assert_close(a.replay.data['next_histories'][3], torch.full((162,), 2., dtype=torch.float16))
    torch.testing.assert_close(a.replay.data['next_observations'][2], torch.full((81,), 2., dtype=torch.float16))
    assert a.replay.data['terminated'][2] and not a.replay.data['truncated'][2]
    assert a.replay.data['truncated'][3] and not a.replay.data['terminated'][3]
    transition(a, 2)
    assert a.replay.size == 5 and a.replay.cursor == 1
    batch = a.replay.sample(8, 'cpu')
    assert batch['histories'].dtype == torch.float32
    assert not {'mu', 'sigma', 'log_probs', 'values', 'latent_noise'} & a.replay.data.keys()
    with pytest.raises(ValueError, match='n_step=1'):
        ReplayBufferB1Z1PACT(n_step=2)
    replay = ReplayBufferB1Z1PACT(3)
    replay.add({'x': torch.arange(8)[:, None]})
    assert set(replay.data['x'].flatten().tolist()) == {5, 6, 7}


def test_noise_reset_and_categorical_projection():
    noise = RepeatedNoise()
    first = noise.sample(torch.zeros(3, 4)).clone()
    noise.remaining.fill_(10)
    noise.reset(torch.tensor([True, False, False]))
    second = noise.sample(torch.zeros(3, 4))
    torch.testing.assert_close(first[1:], second[1:])
    assert not torch.equal(first[0], second[0])
    target = _compute_categorical_td_target(torch.full((2, 101), -torch.log(torch.tensor(101.))),
        torch.tensor([5., 0.]), torch.tensor([1., 0.]), torch.zeros(2), .95, 101, -5., 5.)
    torch.testing.assert_close(target.sum(-1), torch.ones(2))
    assert target[0, -1] == pytest.approx(1.)


def test_updates_auxiliary_gradients_and_checkpoint(tmp_path):
    a = make_algorithm(pinn_start_env_step=0, pinn_warmup_env_steps=1, pinn_loss_weight=-.1)
    transition(a, n=4)
    batch = a.replay.sample(4, 'cpu')
    a.pinn_weight = .1
    a._prepare_batch(batch)
    a._actor_update(batch)
    assert all(p.grad is None for p in a.enc_parameters + a.decoder_parameters + list(a.q.parameters()))
    a.actor_optimizer.zero_grad()
    before = [p.detach().clone() for p in a.target_q.parameters()]
    metrics = a.update_batch(batch, 1)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(not torch.equal(x, y) for x, y in zip(before, a.target_q.parameters()))
    for parameters in (a.enc_parameters, a.decoder_parameters):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)
    old_actor = [p.detach().clone() for p in a.ppo_parameters]
    old_temp = a.temperature.log_temp.detach().clone()
    metrics = a.update_batch(batch, 1)
    assert 'SAC/actor_loss' not in metrics
    for old, current in zip(old_actor, a.ppo_parameters):
        torch.testing.assert_close(old, current)
    torch.testing.assert_close(old_temp, a.temperature.log_temp)
    path = tmp_path / 'checkpoint.pt'
    saved = a.state_dict()
    torch.save(saved, path)
    b = make_algorithm()
    b.load_state_dict(torch.load(path, weights_only=False))
    for name in ('q', 'target_q', 'temperature'):
        for key, value in getattr(a, name).state_dict().items():
            torch.testing.assert_close(value, getattr(b, name).state_dict()[key])
    assert a.update_step == b.update_step and a.pinn_weight == b.pinn_weight
    assert a.schedulers[1].state_dict() == b.schedulers[1].state_dict()
    legacy = dict(saved)
    legacy.pop('action_ranges')
    b._set_action_ranges(dict(position=.5, leg_torque=.5, arm_torque=.5))
    b.load_state_dict(legacy)
    assert b.action_ranges == dict(position=1., leg_torque=1., arm_torque=1.)
    legacy['action_range'] = .75
    b.load_state_dict(legacy)
    assert b.action_ranges == dict(position=.75, leg_torque=.75, arm_torque=.75)
    assert len(b.q_optimizer.state) == len(a.q_optimizer.state)


def test_two_environment_eight_step_collection_smoke():
    a = make_algorithm()
    for step in range(8):
        with torch.inference_mode():
            transition(a, step, done=step == 4)
        metrics = a.update(step)
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert a.env_steps == 8 and a.replay.size == 16
    assert a.update_step == 14  # Seven eligible vector steps * two updates.


@pytest.mark.parametrize('sign', [-1., 1.])
def test_actor_physics_and_auxiliary_ownership_on_replay(sign):
    from test_b1z1_actor_physics import setup
    from rsl_rl.algorithms.b1z1_bard_pinn import auxiliary_step
    template, physical, _, _ = setup()
    a = make_algorithm(actor_phys_enabled=True, actor_phys_pos_fk_enabled=False,
                       pinn_loss_weight=sign, pinn_warmup_env_steps=1)
    a.pinn_weight = .1
    transition(a, n=3)
    batch = {k: v[:3].float() if v.is_floating_point() else v[:3] for k, v in a.replay.data.items()}
    batch.update(physical)
    a._prepare_batch(batch)
    metrics = a._actor_update(batch)
    assert all(p.grad is None for p in a.enc_parameters + a.decoder_parameters + list(a.q.parameters()))
    assert a.actor_physics_metrics['actor_gradient_norm'] > 0
    assert a.actor_physics_metrics['unintended_estimator_gradient_max'] == 0
    a.actor_optimizer.zero_grad()
    a.pinn_metric_sums = {}
    _, inverse, rollout = auxiliary_step(a, batch, torch.ones(3, 1), 1)
    assert torch.isfinite(inverse + rollout)
    assert all(p.grad is None for p in a.ppo_parameters + list(a.q.parameters()))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.enc_parameters)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.decoder_parameters)


@pytest.mark.parametrize("project", [False, True])
def test_replay_force_target_is_recomputed(project):
    from test_b1z1_actor_physics import setup
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
    from rsl_rl.algorithms.b1z1_actor_physics import refresh_replay_target
    _, batch, _, _ = setup()
    a = make_algorithm(actor_phys_goal_ee=class_to_dict(B1Z1PACTCfg.goal_ee), actor_phys_force_shift=True)
    a.cfg['actor_phys_goal_ee']['project_force_adjusted_ee_target'] = project
    batch.update(histories=torch.randn(3, 162), actor_phys_ee_nominal=torch.tensor([[.5, 0., 0.]]).repeat(3, 1),
        actor_phys_target_yaw=torch.tensor([[0., 0., 0., 1.]]).repeat(3, 1),
        actor_phys_target_center=torch.zeros(3, 3), actor_phys_target_force_kp=torch.ones(3, 1),
        actor_phys_target_collision_lower=torch.full((3, 3), -3.),
        actor_phys_target_collision_upper=torch.full((3, 3), -2.),
        actor_phys_target_underground=torch.full((3, 1), -1.))
    refresh_replay_target(a, batch)
    before = batch['actor_phys_ee_target'].clone()
    # Force decoder changes after collection must change the sampled target.
    with torch.no_grad():
        for p in a.actor_critic.physics_decoder.force.parameters():
            p.add_(.1)
    refresh_replay_target(a, batch)
    assert not torch.equal(before, batch['actor_phys_ee_target'])
    assert not batch['actor_phys_ee_target'].requires_grad


def test_runner_checkpoint_and_strict_hotstart(tmp_path):
    from rsl_rl.runners.b1z1_pact_runner import B1Z1PACTRunner
    def runner(a):
        r = B1Z1PACTRunner.__new__(B1Z1PACTRunner)
        r.alg, r.actor_critic, r.privileged_decoder = a, a.actor_critic, a.privileged_decoder
        r.env = SimpleNamespace(set_completed_env_steps=lambda count: None, simulator=SimpleNamespace(), _staged_force_curriculum=SimpleNamespace(
            state_dict=lambda: {}, load_state_dict=lambda state: None))
        r.is_flash_sac, r.device = True, 'cpu'
        r.current_learning_iteration, r.total_timesteps, r.total_time = 3, 16, 1.5
        r.completed_env_steps = 8
        r.curriculum_ep_infos, r.curriculum_episode_lengths = [], []
        return r
    a = make_algorithm(replay_persistence=True, sac_position_action_range=2., sac_leg_torque_action_range=3., sac_arm_torque_action_range=4., clip_actions=100.)
    transition(a, n=4)
    a.update(0)
    r = runner(a)
    path = tmp_path / 'runner.pt'
    r.curriculum_ep_infos = [{'roll_termination_rate': torch.tensor(.01)}]
    r.curriculum_episode_lengths = [950., 975.]
    r.save(path)
    b = make_algorithm(replay_persistence=True, clip_actions=100.)
    restored = runner(b)
    assert restored.load(path) == 3
    assert restored.completed_env_steps == 8 and b.schedule_env_steps == 8
    assert list(restored.curriculum_episode_lengths) == [950., 975.]
    assert restored.curriculum_ep_infos[0]['roll_termination_rate'] == pytest.approx(.01)
    assert b.action_ranges == dict(position=2., leg_torque=3., arm_torque=4.)
    obs, hist = torch.zeros(2, 81), torch.zeros(2, 162)
    torch.testing.assert_close(restored.get_inference_policy()(obs, hist), b.actor_critic.act_inference(obs, hist) * torch.tensor([2.] * 17 + [3.] * 12 + [4.] * 5))
    assert a.replay.size == b.replay.size
    for name in ('actor_critic', 'privileged_decoder', 'q', 'target_q', 'temperature'):
        for key, value in getattr(a, name).state_dict().items():
            torch.testing.assert_close(value, getattr(b, name).state_dict()[key])
    for name in ('auxiliary_optimizer', 'decoder_optimizer', 'q_optimizer', 'temperature_optimizer'):
        assert len(getattr(a, name).state) == len(getattr(b, name).state)
    assert a.force_metric_emas == b.force_metric_emas
    torch.testing.assert_close(a.reward_normalizer.variance, b.reward_normalizer.variance)
    legacy = torch.load(path, weights_only=False)
    legacy.pop('curriculum_clock')
    legacy['runner_counters'].pop('completed_env_steps')
    legacy['force_curriculum_state'] = dict(trigger_iteration=2, last_update_iteration=3, gate_patience=4)
    legacy['force_gate_count'] = 2
    legacy['flash_sac']['counters']['env_steps'] = 15
    loaded_force = {}
    restored.env._staged_force_curriculum.load_state_dict = loaded_force.update
    torch.save(legacy, path)
    with pytest.warns(UserWarning, match='Migrating legacy PACT'):
        restored.load(path)
    assert restored.completed_env_steps == 15
    assert loaded_force == dict(trigger_iteration=10, last_update_iteration=15, gate_patience=20)
    assert b.force_gate_count == 10
    hotstart = {k: v for k, v in a.actor_critic.state_dict().items() if not k.startswith('log_std_head.')}
    torch.save(dict(model_state_dict=hotstart, privileged_decoder_state_dict=a.privileged_decoder.state_dict()), path)
    restored._load_pretrained_model(str(path))
    hotstart.pop('position_head.bias')
    torch.save(dict(model_state_dict=hotstart, privileged_decoder_state_dict=a.privileged_decoder.state_dict()), path)
    with pytest.raises(RuntimeError, match='Unexpected hot-start'):
        restored._load_pretrained_model(str(path))


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf'), 101.])
@pytest.mark.parametrize('channel', ['position', 'leg_torque', 'arm_torque'])
def test_invalid_environment_action_ranges(value, channel):
    key = f'sac_{channel}_action_range'
    with pytest.raises(ValueError, match=key):
        make_algorithm(**{key: value}, clip_actions=100.)


def test_action_range_collection_inference_and_physics(monkeypatch):
    from rsl_rl.algorithms import b1z1_actor_physics
    from rsl_rl.runners.b1z1_pact_runner import B1Z1PACTRunner
    a = make_algorithm(sac_position_action_range=2., sac_leg_torque_action_range=3., sac_arm_torque_action_range=4., clip_actions=100.)
    obs, history = torch.zeros(4, 81), torch.zeros(4, 162)
    commands = a.act(obs, torch.zeros(4, 40), history, torch.zeros(4, 23))
    torch.testing.assert_close(commands, a.transition.actions * torch.tensor([2.] * 17 + [3.] * 12 + [4.] * 5))
    assert a.transition.actions.abs().max() <= 1
    runner = B1Z1PACTRunner.__new__(B1Z1PACTRunner)
    runner.is_flash_sac, runner.alg, runner.actor_critic = True, a, a.actor_critic
    inference = runner.get_inference_policy()
    torch.testing.assert_close(inference(obs, history), a.actor_critic.act_inference(obs, history) * torch.tensor([2.] * 17 + [3.] * 12 + [4.] * 5))

    # Both physical branches expand by the range once before their unit scales.
    state = torch.zeros(4, 180)
    state[:, 97:156] = 1.
    normalized = torch.ones(4, 34, requires_grad=True)
    torques = a._coupled_torque(a.to_env_actions(normalized), state)
    torch.testing.assert_close(torques[:, :17], torch.tensor([3.5] * 12 + [4.5] * 5).expand(4, -1))
    torques.sum().backward()
    torch.testing.assert_close(normalized.grad[:, :17], torch.full((4, 17), .5))
    torch.testing.assert_close(normalized.grad[:, 17:], torch.tensor([3.] * 12 + [4.] * 5).expand(4, -1))

    transition(a, n=4)
    batch = a.replay.sample(4, 'cpu')
    assert batch['actions'].abs().max() <= 1
    seen = []
    def physical_backward(algorithm, batch, loss, actions, context):
        # The existing helper uses this same expanded command for BARD and FK.
        torch.testing.assert_close(actions, algorithm.actor_critic.action_mean.tanh() * torch.tensor([2.] * 17 + [3.] * 12 + [4.] * 5))
        seen.append('physics')
        loss.backward()
    monkeypatch.setattr(b1z1_actor_physics, 'backward', physical_backward)
    original = a.q_forward
    def critic_forward(observations, actions, training):
        assert actions.abs().max() <= 1
        seen.append('critic')
        return original(observations, actions, training)
    a.q_forward = critic_forward
    a._actor_update(batch)
    a._critic_update(batch)
    assert seen.count('physics') == 1 and seen.count('critic') == 2
