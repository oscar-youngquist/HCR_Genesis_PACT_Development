"""Run with SIMULATOR=isaaclab CUDA_VISIBLE_DEVICES=1 in the IsaacLab env."""
from types import SimpleNamespace

import pytest
import torch

from rsl_rl.modules.actor_critic_dreamwaq import ActorCriticDreamWaQ
from rsl_rl.algorithms.ppo_dreamwaq import PPO_DreamWaQ
from rsl_rl.runners.dreamwaq_runner import DreamWaQRunner
from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq_config import Go2DreamwaqCfg
from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq import Go2Dreamwaq
from legged_gym.simulator.go2_domain_rand_curriculum import Go2DomainRandCurriculum
from legged_gym.simulator.dreamwaq_adapter import DreamWaQAdapter
from legged_gym.simulator.isaaclab_simulator_dreamwaq import IsaacLabSimulatorDreamWaQ

DEVICE = 'cuda:0'  # Process-local GPU 0 is physical GPU 1 via CUDA_VISIBLE_DEVICES.


@pytest.fixture
def algorithm():
    actor = ActorCriticDreamWaQ(45, 12, 60, 225, 16, 11, 45,
        actor_hidden_dims=[32], critic_hidden_dims=[32],
        encoder_hidden_dims=[32], decoder_hidden_dims=[32]).to(DEVICE)
    return PPO_DreamWaQ(actor, device=DEVICE)


def labels(n):
    target = torch.randn(n, 11, device=DEVICE)
    target[:, 3:7] = torch.randint(0, 2, (n, 4), device=DEVICE)
    return target


def test_explicit_semantics_and_inference(algorithm):
    vae = algorithm.actor_critic.vae
    history = torch.randn(8, 225, device=DEVICE)
    mu, _, output = vae.encode(history)
    assert output.explicit_for_policy.shape == (8, 11)
    assert torch.all((output.contact_probability > 0) & (output.contact_probability < 1))
    torch.testing.assert_close(output.explicit_for_policy[:, 3:7], output.contact_probability)
    torch.testing.assert_close(vae.inference(history), torch.cat((mu, output.explicit_for_policy), -1))
    # Explicit estimates are deterministic even when the implicit latent samples differ.
    first, _ = vae(history)
    second, _ = vae(history)
    assert not torch.equal(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_optimizer_ownership_and_gradient_isolation(algorithm):
    actor = algorithm.actor_critic
    ppo = {id(p) for g in algorithm.optimizer.param_groups for p in g['params']}
    vae = {id(p) for g in algorithm.vae_optimizer.param_groups for p in g['params']}
    assert not ppo & vae
    assert ppo | vae == {id(p) for p in actor.parameters()}
    before = [p.clone() for p in actor.vae.parameters()]
    actor.act(torch.randn(8, 45, device=DEVICE), torch.randn(8, 225, device=DEVICE))
    loss = actor.action_mean.square().mean() + actor.evaluate(torch.randn(8, 60, device=DEVICE)).square().mean() - actor.entropy.mean()
    loss.backward()
    assert all(p.grad is None for p in actor.vae.parameters())
    algorithm.optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(before, actor.vae.parameters()))
    algorithm.optimizer.zero_grad(set_to_none=True)
    before = [p.clone() for p in algorithm.rl_parameters]
    loss, *_ = algorithm._compute_vae_loss(torch.randn(8, 225, device=DEVICE),
        torch.ones(8, 1, device=DEVICE), labels(8), torch.randn(8, 45, device=DEVICE))
    loss.backward()
    assert all(p.grad is None for p in algorithm.rl_parameters)
    algorithm.vae_optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(before, algorithm.rl_parameters))
    assert all(p.grad is not None for p in actor.vae.parameters())


def test_mask_selects_before_arithmetic_and_normalizes(algorithm):
    history = torch.randn(4, 225, device=DEVICE)
    target, future = labels(4), torch.randn(4, 45, device=DEVICE)
    torch.manual_seed(9)
    expected = algorithm._compute_vae_loss(history, torch.ones(4, 1, device=DEVICE), target, future)
    history = torch.cat((history, torch.full_like(history, float('nan'))))
    target = torch.cat((target, torch.full_like(target, float('nan'))))
    future = torch.cat((future, torch.full_like(future, float('nan'))))
    valid = torch.tensor([1] * 4 + [0] * 4, device=DEVICE).reshape(-1, 1)
    torch.manual_seed(9)
    actual = algorithm._compute_vae_loss(history, valid, target, future)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    actual[0].backward()
    assert all(torch.isfinite(p.grad).all() for p in algorithm.actor_critic.vae.parameters())
    empty = algorithm._compute_vae_loss(history, torch.zeros_like(valid), target, future)
    assert all(value.item() == 0 for value in empty)


def test_contact_bce_uses_logits(algorithm):
    vae = algorithm.actor_critic.vae
    for p in vae.parameters():
        p.data.zero_()
    target = torch.zeros(3, 11, device=DEVICE)
    _, explicit, _, kl = algorithm._compute_vae_loss(
        torch.zeros(3, 225, device=DEVICE), torch.ones(3, 1, device=DEVICE), target,
        torch.zeros(3, 45, device=DEVICE))
    torch.testing.assert_close(explicit, torch.tensor(2., device=DEVICE).log())
    assert kl.item() == 0
    explicit.backward()
    assert torch.all(vae.explicit_head.bias.grad[3:7] > 0)


def test_rollout_snapshots_survive_reset_mutation(algorithm):
    obs, critic = torch.randn(4, 45, device=DEVICE), torch.randn(4, 60, device=DEVICE)
    history, target = torch.randn(4, 225, device=DEVICE), labels(4)
    originals = [t.clone() for t in (obs, critic, history, target)]
    algorithm.act(obs, critic, history, target)
    for value in (obs, critic, history, target):
        value.zero_()
    for saved, original in zip((algorithm.transition.observations,
            algorithm.transition.privileged_observations, algorithm.transition.observation_histories,
            algorithm.transition.explicit_info_labels), originals):
        torch.testing.assert_close(saved, original)


def test_domain_curriculum_phases_gating_and_restore():
    cfg = Go2DreamwaqCfg()
    cfg.domain_rand.push_warmup = 1
    cfg.domain_rand.step_interval = 1
    cfg.domain_rand.joint_dynamics_progress_delta = 1.
    cfg.domain_rand.mass_com_progress_delta = 1.
    cfg.domain_rand.disturbance_progress_delta = 1.
    curriculum = Go2DomainRandCurriculum(cfg)
    assert not curriculum.advance(1, 1.)
    assert not curriculum.advance(2, None)
    assert not curriculum.advance(3, float('nan'))
    assert curriculum.advance(4, 1.)
    assert curriculum.phase == 'mass_com'
    assert not curriculum.advance(4, 1.)
    assert curriculum.advance(5, 1.)
    assert curriculum.phase == 'disturbance'
    other = Go2DomainRandCurriculum(cfg)
    other.load_state_dict(curriculum.state_dict())
    assert other.effective_ranges() == curriculum.effective_ranges()
    assert other.advance(6, 1.)
    assert other.phase == 'complete'
    assert other.effective_ranges()['joint_damping'] == (0., .8)
    assert other.effective_ranges()['added_base_mass'] == (cfg.domain_rand.added_mass_min, cfg.domain_rand.max_added_mass_max)
    assert not other.advance(7, 1.)
    cfg.domain_rand.reward_ema_alpha = 1.
    gated = Go2DomainRandCurriculum(cfg)
    assert not gated.advance(9, .1)


@pytest.mark.parametrize('iteration,blend', [(0, 0.), (6, 0.), (8, .5), (10, 1.), (20, 1.)])
def test_reward_curriculum_boundaries(iteration, blend):
    cfg = Go2DreamwaqCfg()
    # Isolate the schedule contract from user-tuned training defaults.
    cfg.rewards.reward_curriculum.warmup_steps = 6
    cfg.rewards.reward_curriculum.curr_steps = 4
    env = SimpleNamespace(cfg=cfg, dt=.02,
        reward_scales={name: 1. for name in cfg.rewards.reward_curriculum.curr_reward_bounds})
    Go2Dreamwaq.step_reward_curriculum(env, iteration)
    for name, (low, high) in cfg.rewards.reward_curriculum.curr_reward_bounds.items():
        assert env.reward_scales[name] == pytest.approx((low + blend * (high-low)) * .02)


def test_checkpoint_restores_both_optimizers_and_curriculum(algorithm, tmp_path):
    # Populate Adam state for both owners before serializing.
    for optimizer in (algorithm.optimizer, algorithm.vae_optimizer):
        optimizer.zero_grad()
        sum(p.square().sum() for g in optimizer.param_groups for p in g['params']).backward()
        optimizer.step()
    cfg = Go2DreamwaqCfg()
    curriculum = Go2DomainRandCurriculum(cfg)
    curriculum.advance(2010, 1.)
    simulator = SimpleNamespace(domain_rand_curriculum=curriculum, load_domain_randomization=curriculum.load_state_dict)
    env = SimpleNamespace(simulator=simulator, step_reward_curriculum=lambda i: None, reset=lambda: None)
    runner = DreamWaQRunner.__new__(DreamWaQRunner)
    runner.alg, runner.env, runner.device = algorithm, env, DEVICE
    runner.current_learning_iteration = 2011
    checkpoint = tmp_path / 'model.pt'
    runner.save(checkpoint)
    algorithm.optimizer.state.clear()
    algorithm.vae_optimizer.state.clear()
    runner.current_learning_iteration = 0
    runner.load(checkpoint)
    assert runner.current_learning_iteration == 2011
    assert algorithm.optimizer.state and algorithm.vae_optimizer.state
    assert curriculum.progress['joint_dynamics'] == .02
    torch.save({'model_state_dict': {}}, checkpoint)
    with pytest.raises(ValueError, match='Incompatible DreamWaQ'):
        runner.load(checkpoint)


@pytest.fixture
def permuted_contact_adapter():
    adapter = IsaacLabSimulatorDreamWaQ.__new__(IsaacLabSimulatorDreamWaQ)
    adapter._cfg = Go2DreamwaqCfg()
    adapter._cfg.asset.contact_state_link_names = ['base', 'FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
    adapter._cfg.asset.penalize_contacts_on = ['thigh']
    adapter._cfg.asset.terminate_after_contacts_on = ['Head']
    adapter._robot = SimpleNamespace(body_names=['base', 'FR_foot', 'FL_foot', 'RR_foot', 'RL_foot', 'thigh', 'Head'])
    adapter._contact_sensors = SimpleNamespace(
        body_names=['RL_foot', 'Head', 'RR_foot', 'thigh', 'base', 'FL_foot', 'FR_foot'],
        data=SimpleNamespace(net_forces_w=torch.zeros(2, 7, 3, device=DEVICE)))
    adapter._configure_contact_indices()
    return adapter


def test_isaaclab_separate_body_and_sensor_indices(permuted_contact_adapter):
    adapter = permuted_contact_adapter
    assert adapter.feet_indices == [1, 2, 3, 4]
    assert adapter.feet_contact_indices == [6, 5, 2, 0]
    assert adapter._contact_state_link_indices == [4, 6, 5, 2, 0]
    assert adapter.penalized_contact_indices == [3]
    assert adapter.termination_contact_indices == [1]
    raw = adapter._contact_sensors.data.net_forces_w
    raw.copy_(torch.arange(raw.numel(), device=DEVICE).reshape_as(raw))
    assert adapter.link_contact_forces is raw
    torch.testing.assert_close(adapter.link_contact_forces[:, adapter.feet_contact_indices], raw[:, [6, 5, 2, 0]])


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'duplicate_requested'])
def test_isaaclab_rejects_invalid_contact_name_mapping(permuted_contact_adapter, problem):
    adapter = permuted_contact_adapter
    if problem == 'missing':
        adapter._contact_sensors.body_names.remove('FR_foot')
    elif problem == 'duplicate':
        adapter._contact_sensors.body_names.append('FR_foot')
    else:
        adapter._cfg.asset.feet_names = ['FR_foot'] * 4
    with pytest.raises(ValueError):
        adapter._configure_contact_indices()


def test_contact_rewards_and_termination_use_sensor_order(permuted_contact_adapter):
    adapter = permuted_contact_adapter
    raw = adapter.link_contact_forces
    # Row 0: all feet down, thigh collision, no head contact.
    # Row 1: head contact only. Body-order indexing would misclassify these.
    raw[0, [6, 5, 2, 0], 2] = 2.
    raw[0, 3, 2] = 5.
    raw[1, 1, 2] = 20.
    adapter._projected_gravity = torch.tensor([[0., 0., -1.]] * 2, device=DEVICE)
    env = SimpleNamespace(simulator=adapter, cfg=adapter._cfg,
        commands=torch.zeros(2, 3, device=DEVICE), dt=.02,
        fail_buf=torch.zeros(2, device=DEVICE), episode_length_buf=torch.zeros(2, device=DEVICE),
        max_episode_length=100, feet_air_time=torch.ones(2, 4, device=DEVICE),
        last_contacts=torch.zeros(2, 4, device=DEVICE, dtype=torch.bool))
    torch.testing.assert_close(Go2Dreamwaq._reward_feet_contact_stand_still(env), torch.tensor([1., 0.], device=DEVICE))
    torch.testing.assert_close(Go2Dreamwaq._reward_collision(env), torch.tensor([1., 0.], device=DEVICE))
    env.cfg.env.fail_to_terminal_time_s = 0.
    Go2Dreamwaq.check_termination(env)
    assert env.reset_buf.tolist() == [False, True]
    env.commands[:, 0] = 1.
    torch.testing.assert_close(Go2Dreamwaq._reward_feet_air_time(env), torch.tensor([4 * (1. + env.dt - .25), 0.], device=DEVICE))


def test_explicit_contact_labels_use_sensor_order(permuted_contact_adapter):
    from collections import deque
    adapter = permuted_contact_adapter
    raw = adapter.link_contact_forces
    raw[0, [6, 2], 2] = 2.  # FR and RR
    raw[1, [5, 0], 2] = 2.  # FL and RL
    zero = lambda width: torch.zeros(2, width, device=DEVICE)
    simulator = SimpleNamespace(
        link_contact_forces=raw, feet_contact_indices=adapter.feet_contact_indices,
        feet_indices=adapter.feet_indices, projected_gravity=zero(3),
        base_ang_vel=zero(3), base_lin_vel=zero(3), dof_pos=zero(12), dof_vel=zero(12),
        default_dof_pos=zero(12), _friction_values=zero(1), _added_base_mass=zero(1),
        _base_com_bias=zero(3), _rand_push_vels=zero(3), _kp_scale=zero(12), _kd_scale=zero(12),
        feet_pos=torch.zeros(2, 4, 3, device=DEVICE), height_around_feet=torch.zeros(2, 4, 9, device=DEVICE))
    cfg = adapter._cfg
    cfg.asset.obtain_link_contact_states = False
    cfg.terrain.measure_heights = False
    env = SimpleNamespace(cfg=cfg, simulator=simulator, commands=zero(3), commands_scale=1.,
        obs_scales=SimpleNamespace(ang_vel=1., lin_vel=1., dof_pos=1., dof_vel=1.),
        actions=zero(12), friction_value_offset=0., kp_scale_offset=0., kd_scale_offset=0.,
        critic_obs_deque=deque(maxlen=1), obs_history_deque=deque(maxlen=1), add_noise=False)
    Go2Dreamwaq.compute_observations(env)
    torch.testing.assert_close(env.explicit_labels_buf[:, 3:7],
        torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]], device=DEVICE))


def test_policy_export_matches_inference(algorithm):
    from legged_gym.utils.helpers import PolicyExporterWaQ
    actor = algorithm.actor_critic
    exporter = PolicyExporterWaQ(actor).to(DEVICE)
    scripted = torch.jit.script(exporter)
    obs, history = torch.randn(4, 45, device=DEVICE), torch.randn(4, 225, device=DEVICE)
    torch.testing.assert_close(scripted(obs, history), actor.act_inference(obs, history))


def test_torque_limit_reward_uses_simulator_limits():
    cfg = Go2DreamwaqCfg()
    torques = torch.tensor([[0., 20., -40.]], device=DEVICE)
    limits = torch.tensor([10., 20., 30.], device=DEVICE)
    env = SimpleNamespace(cfg=cfg, simulator=SimpleNamespace(requested_torques=torques, torque_limits=limits))
    expected = (torques.abs() - limits * cfg.rewards.soft_torque_limit).clamp_min(0).sum(-1)
    torch.testing.assert_close(Go2Dreamwaq._reward_torque_limits(env), expected)


class _FakeBackend:
    def __init__(self, cfg, sim_params, device, headless):
        self._cfg, self._sim_params, self._device = cfg, sim_params, device
        self._num_envs, self._num_actions = 4, 12
        self._rand_push_vels = torch.zeros(4, 3, device=device)
        self.resets = []
        self.samples = {}
        for name in ('friction', 'base_mass', 'com_displacement', 'joint_armature',
                     'joint_friction', 'joint_damping', 'pd_gain'):
            self.samples[name] = []
            setattr(self, '_randomize_' + name,
                    lambda ids, key=name: self.samples[key].append(ids.clone()))

    def reset_idx(self, env_ids):
        self.resets.append(env_ids.clone())
        self._reset_domain_randomization(env_ids)

    def _reset_domain_randomization(self, env_ids):
        for name in self.samples:
            if getattr(self._cfg.domain_rand, 'randomize_' + name):
                getattr(self, '_randomize_' + name)(env_ids)

    def step(self, actions):
        return actions.clone()


class _FakeAdapter(DreamWaQAdapter, _FakeBackend):
    def _install_joint_stiffness(self, env_ids, values):
        self.installed_stiffness = values.clone()

    def _apply_velocity_push(self, env_ids, linear, angular):
        self.last_push = (env_ids, linear, angular)


def test_adapter_delay_reset_and_disturbances():
    cfg = Go2DreamwaqCfg()
    adapter = _FakeAdapter(cfg, {'dt': .005}, DEVICE, True)
    # Delay belongs to the environment, exactly once, before backend stepping.
    env = SimpleNamespace(cfg=cfg, device=DEVICE, num_envs=4,
        actions=torch.zeros(4, 12, device=DEVICE), last_actions=torch.zeros(4, 12, device=DEVICE),
        llast_actions=torch.zeros(4, 12, device=DEVICE),
        action_queue=torch.zeros(4, 3, 12, device=DEVICE),
        action_delay=torch.full((4,), 2, device=DEVICE, dtype=torch.long))
    fixture = env
    env = object.__new__(Go2Dreamwaq)
    env.__dict__.update(vars(fixture))
    env.simulator = adapter
    env._raw_action_queue = torch.zeros_like(env.action_queue)
    first = torch.ones(4, 12, device=DEVICE)
    assert Go2Dreamwaq._pre_sim_step(env, first).count_nonzero() == 0
    assert Go2Dreamwaq._pre_sim_step(env, 2 * first).count_nonzero() == 0
    torch.testing.assert_close(Go2Dreamwaq._pre_sim_step(env, 3 * first), first)
    # Adapter step delegates directly to the backend and adds no second delay.
    torch.testing.assert_close(adapter.step(first), first)
    adapter.reset_idx(torch.arange(4, device=DEVICE))
    assert (adapter.installed_stiffness >= 0).all()
    assert (adapter.installed_stiffness <= .005).all()
    adapter._push_timers.zero_()
    adapter.push_robots()
    _, linear, angular = adapter.last_push
    assert (linear[:, :2].abs() <= .5).all()
    assert ((linear[:, 2] >= -.1) & (linear[:, 2] <= 0)).all()
    assert (angular.abs() <= .5).all()
    adapter.push_robots()
    assert adapter._rand_push_vels.count_nonzero() == 0
    assert adapter._rand_wrench_vels.count_nonzero() == 0
    adapter.domain_rand_curriculum.progress = {phase: 1. for phase in adapter.domain_rand_curriculum.phases}
    adapter._apply_curriculum_ranges()
    assert cfg.domain_rand.joint_friction_range == (0., .2)
    assert cfg.domain_rand.added_mass_range == (cfg.domain_rand.added_mass_min, cfg.domain_rand.max_added_mass_max)
    assert cfg.domain_rand.com_pos_x_range == (-cfg.domain_rand.com_displacement_x_max, cfg.domain_rand.com_displacement_x_max)


def test_empty_auxiliary_update_does_not_apply_adam_momentum(algorithm):
    # Initialize optimizer momentum with a real update.
    loss, *_ = algorithm._compute_vae_loss(torch.randn(4, 225, device=DEVICE),
        torch.ones(4, 1, device=DEVICE), labels(4), torch.randn(4, 45, device=DEVICE))
    loss.backward()
    algorithm.vae_optimizer.step()
    before = [p.clone() for p in algorithm.actor_critic.vae.parameters()]
    algorithm.init_storage(4, 1, [45], [60], [225], [11], [45], [12])
    with torch.no_grad():
        algorithm.act(torch.randn(4, 45, device=DEVICE), torch.randn(4, 60, device=DEVICE),
            torch.randn(4, 225, device=DEVICE), labels(4))
        algorithm.process_env_step(torch.randn(4, device=DEVICE), torch.ones(4, device=DEVICE), {},
            torch.full((4, 45), float('nan'), device=DEVICE))
        algorithm.compute_returns(torch.randn(4, 60, device=DEVICE))
    values = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value, device=DEVICE)) for value in values)
    assert all(torch.equal(p, old) for p, old in zip(algorithm.actor_critic.vae.parameters(), before))


def test_step_drops_stale_episode_statistics():
    from legged_gym.envs.base.legged_robot_dreamwaq import LeggedRobotDreamwaq
    tensor = torch.zeros(4, 1, device=DEVICE)
    env = SimpleNamespace(cfg=Go2DreamwaqCfg(),
        _pre_sim_step=lambda actions: actions,
        simulator=SimpleNamespace(step=lambda actions: None),
        post_physics_step=lambda: None,
        extras={'episode': {'rew_tracking_lin_vel': 1.}},
        obs_buf=tensor, privileged_obs_buf=tensor, obs_history=tensor,
        explicit_labels_buf=tensor, next_state_buf=tensor, rew_buf=tensor, reset_buf=tensor)
    result = LeggedRobotDreamwaq.step(env, tensor)
    assert 'episode' not in result[-1]


@pytest.mark.parametrize('interval', [0, 1, 3])
def test_reset_cadence_counts_each_environment_and_keeps_samples(interval):
    cfg = Go2DreamwaqCfg()
    cfg.domain_rand.reset_resample_episodes = interval
    adapter = _FakeAdapter(cfg, {'dt': .005}, DEVICE, True)
    all_ids = torch.arange(4, device=DEVICE)
    adapter.reset_idx(all_ids)  # Initialization, not a completed episode.
    if interval > 1:
        assert adapter._reset_cadence.episodes.count_nonzero() == 0
    initial_motor = adapter._motor_strength.clone()
    initial_stiffness = adapter._joint_stiffness.clone()
    for episode in range(1, 4):
        ids = all_ids[:1]
        adapter.reset_idx(ids)
        if interval > 1 and episode < interval:
            torch.testing.assert_close(adapter._motor_strength, initial_motor, rtol=0, atol=0)
            torch.testing.assert_close(adapter._joint_stiffness, initial_stiffness, rtol=0, atol=0)
        else:
            assert not torch.equal(adapter._motor_strength[0], initial_motor[0])
        # Other environments have not completed any episodes and retain values.
        torch.testing.assert_close(adapter._motor_strength[1:], initial_motor[1:], rtol=0, atol=0)
    expected_calls = 2 if interval > 1 else 4
    assert all(len(calls) == expected_calls for calls in adapter.samples.values())
    if interval > 1:
        torch.testing.assert_close(adapter._reset_cadence.episodes, torch.tensor([3, 0, 0, 0], device=DEVICE))
    adapter.reset_idx(all_ids[:0])
    assert all(len(calls) == expected_calls for calls in adapter.samples.values())


def test_changed_ranges_invalidate_only_affected_parameters_on_next_reset():
    cfg = Go2DreamwaqCfg()
    adapter = _FakeAdapter(cfg, {'dt': .005}, DEVICE, True)
    ids = torch.arange(4, device=DEVICE)
    adapter.reset_idx(ids)
    motor = adapter._motor_strength.clone()
    stiffness = adapter._joint_stiffness.clone()
    adapter.domain_rand_curriculum.progress['joint_dynamics'] = 1.
    adapter._apply_curriculum_ranges()
    # The range change is deferred until each environment's next reset.
    torch.testing.assert_close(adapter._joint_stiffness, stiffness)
    adapter.reset_idx(ids[:1])
    assert len(adapter.samples['joint_friction']) == 2
    assert len(adapter.samples['joint_damping']) == 2
    assert len(adapter.samples['base_mass']) == 1
    assert len(adapter.samples['pd_gain']) == 1
    torch.testing.assert_close(adapter._motor_strength, motor, rtol=0, atol=0)
    assert not torch.equal(adapter._joint_stiffness[0], stiffness[0])
    torch.testing.assert_close(adapter._joint_stiffness[1:], stiffness[1:])
    adapter._apply_curriculum_ranges()  # Identical bounds must not re-invalidate.
    adapter.reset_idx(ids[:1])
    assert len(adapter.samples['joint_friction']) == 2
    adapter.reset_idx(ids[1:2])
    torch.testing.assert_close(adapter.samples['joint_friction'][-1], ids[1:2])


def test_checkpoint_curriculum_restore_reinitializes_sampling():
    adapter = _FakeAdapter(Go2DreamwaqCfg(), {'dt': .005}, DEVICE, True)
    ids = torch.arange(4, device=DEVICE)
    adapter.reset_idx(ids)
    adapter.reset_idx(ids)
    assert all(len(calls) == 1 for calls in adapter.samples.values())
    adapter.load_domain_randomization(adapter.domain_rand_curriculum.state_dict())
    adapter.reset_idx(ids)
    assert all(len(calls) == 2 for calls in adapter.samples.values())
    assert adapter._reset_cadence.episodes.count_nonzero() == 0


@pytest.mark.parametrize('interval', [-1, 1.5])
def test_invalid_reset_sampling_interval_fails_before_backend_initialization(interval):
    cfg = Go2DreamwaqCfg()
    cfg.domain_rand.reset_resample_episodes = interval
    with pytest.raises(ValueError, match='nonnegative integer'):
        _FakeAdapter(cfg, {'dt': .005}, DEVICE, True)


@pytest.mark.parametrize('delay_enabled', [False, True])
def test_raw_torque_request_tracks_execution_delay_and_reset(monkeypatch, delay_enabled):
    from legged_gym.envs.base.legged_robot_dreamwaq import LeggedRobotDreamwaq
    env = object.__new__(Go2Dreamwaq)
    env.cfg = Go2DreamwaqCfg()
    env.cfg.normalization.clip_actions = 1.
    env.cfg.domain_rand.randomize_ctrl_delay = delay_enabled
    env.device, env.num_envs, env.num_actions = DEVICE, 2, 12
    env.actions = torch.zeros(2, 12, device=DEVICE)
    env.last_actions = torch.zeros_like(env.actions)
    env.llast_actions = torch.zeros_like(env.actions)
    env.action_queue = torch.zeros(2, 3, 12, device=DEVICE)
    env._raw_action_queue = torch.zeros_like(env.action_queue)
    env.action_delay = torch.tensor([0, 2], device=DEVICE)
    env.simulator = SimpleNamespace(raw_delayed_actions=torch.zeros_like(env.actions))
    for value in (2., -3., 4.):
        executed = env._pre_sim_step(torch.full_like(env.actions, value))
        torch.testing.assert_close(executed, env.simulator.raw_delayed_actions.clamp(-1, 1))
    expected = torch.tensor([4., 2. if delay_enabled else 4.], device=DEVICE)
    torch.testing.assert_close(env.simulator.raw_delayed_actions[:, 0], expected)
    monkeypatch.setattr(LeggedRobotDreamwaq, 'reset_idx', lambda self, ids: None)
    env.reset_idx(torch.tensor([1], device=DEVICE))
    assert not env._raw_action_queue[1].any()
    assert env._raw_action_queue[0].any()


def test_position_torque_saturation_preserves_raw_reward_request():
    sim = SimpleNamespace(
        _cfg=Go2DreamwaqCfg(), default_dof_pos=torch.tensor([[.1, -.2, .3]], device=DEVICE),
        dof_pos=torch.tensor([[.2, -.4, .1]], device=DEVICE),
        dof_vel=torch.tensor([[1., -2., 3.]], device=DEVICE),
        _motor_strength=torch.tensor([[.8, 1.2, 1.1]], device=DEVICE),
        _kp_scale=torch.tensor([[1.1, .9, 1.]], device=DEVICE),
        _kd_scale=torch.tensor([[.9, 1.1, 1.]], device=DEVICE),
        _p_gains=torch.tensor([20., 25., 30.], device=DEVICE),
        _d_gains=torch.tensor([.5, .6, .7], device=DEVICE),
        torque_limits=torch.tensor([2., 3., 4.], device=DEVICE),
        raw_delayed_actions=torch.tensor([[20., -30., 40.]], device=DEVICE),
        requested_torques=torch.zeros(1, 3, device=DEVICE))
    actions = sim.raw_delayed_actions.clamp(-1., 1.)
    def pd(a):
        return sim._motor_strength * (sim._kp_scale * sim._p_gains * (
            sim.default_dof_pos + .25 * a - sim.dof_pos)
            - sim._kd_scale * sim._d_gains * sim.dof_vel)
    for _ in range(2):
        command = DreamWaQAdapter._bounded_position_torques(sim, actions)
        torch.testing.assert_close(command, pd(actions).clamp(-sim.torque_limits, sim.torque_limits))
        torch.testing.assert_close(sim.requested_torques, pd(sim.raw_delayed_actions))
        env = SimpleNamespace(cfg=sim._cfg, simulator=sim)
        expected = (pd(sim.raw_delayed_actions).abs() - .9 * sim.torque_limits).clamp_min(0).sum(-1)
        torch.testing.assert_close(Go2Dreamwaq._reward_torque_limits(env), expected)
        assert (sim.requested_torques.abs() > sim.torque_limits).all()
        sim.dof_pos += .1  # PD must be recomputed from live physics-substep state.


def test_isaaclab_physical_torque_limits_follow_policy_joint_order():
    sim = object.__new__(IsaacLabSimulatorDreamWaQ)
    sim._dof_indices = [2, 0, 3, 1]
    sim._robot = SimpleNamespace(
        data=SimpleNamespace(joint_effort_limits=torch.tensor([[1.e9, 1.e9, 20., 1.e9]], device=DEVICE)),
        actuators={
            'calves': SimpleNamespace(joint_indices=[1, 3], effort_limit=torch.tensor([[45.43, 45.43]], device=DEVICE)),
            'hips': SimpleNamespace(joint_indices=[0, 2], effort_limit=torch.tensor([[23.7, 23.7]], device=DEVICE)),
        })
    torch.testing.assert_close(sim.torque_limits, torch.tensor([20., 23.7, 45.43, 45.43], device=DEVICE))


def test_standalone_config_keeps_dreamwaq_training_classes():
    from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
    from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq_config import Go2DreamwaqCfgPPO
    assert Go2DreamwaqCfg.__bases__ == (LeggedRobotCfg,)
    assert Go2DreamwaqCfgPPO.__bases__ == (LeggedRobotCfgPPO,)
    for config in (Go2DreamwaqCfg, Go2DreamwaqCfgPPO):
        for value in vars(config).values():
            if isinstance(value, type):
                assert all(base.__module__ == LeggedRobotCfg.__module__ for base in value.__bases__ if base is not object)
    cfg, train = Go2DreamwaqCfg(), Go2DreamwaqCfgPPO()
    assert train.runner_class_name == 'DreamWaQRunner'
    assert train.runner.policy_class_name == 'ActorCriticDreamWaQ'
    assert train.runner.algorithm_class_name == 'PPO_DreamWaQ'
    assert cfg.env.num_explicit_dims == 11 and cfg.env.num_history_obs == 225
    assert cfg.rewards.scales.front_foot_overreach == -10000.
    assert cfg.rewards.scales.rear_foot_overreach == -10.
    assert cfg.rewards.scales.dof_vel_limits == -1.


def test_soft_velocity_limit_penalty_is_symmetric_and_capped():
    cfg = Go2DreamwaqCfg()
    limits = torch.tensor([10., 20., 30., 40.], device=DEVICE)
    velocity = torch.tensor([[8., -18., 27.4, -50.]], device=DEVICE)
    env = SimpleNamespace(cfg=cfg, simulator=SimpleNamespace(dof_vel=velocity, dof_vel_limits=limits))
    torch.testing.assert_close(Go2Dreamwaq._reward_dof_vel_limits(env), torch.tensor([1.4], device=DEVICE))


def test_overreach_uses_body_frame_sensor_indices_and_pact_payload_scaling():
    cfg = Go2DreamwaqCfg()
    body_feet = torch.tensor([[[.38, .1, -.3], [.48, -.1, -.3],
                               [-.45, .1, -.3], [-.05, -.1, -.3]]], device=DEVICE).repeat(2, 1, 1)
    base_pos = torch.tensor([[3., 4., 1.], [-2., 3., .5]], device=DEVICE)
    # Rotate the robot +90 degrees in world yaw and translate it.
    world_feet = body_feet.clone()
    world_feet[:, :, 0] = -body_feet[:, :, 1]
    world_feet[:, :, 1] = body_feet[:, :, 0]
    forces = torch.zeros(2, 9, 3, device=DEVICE)
    sensor_ids = [7, 2, 8, 4]
    forces[:, sensor_ids, 2] = torch.tensor([[6., 5., 6., 6.], [6., 5., 6., 0.]], device=DEVICE)
    sim = SimpleNamespace(base_pos=base_pos, feet_pos=world_feet + base_pos[:, None],
        base_quat=torch.tensor([[0., 0., 2**-.5, 2**-.5]], device=DEVICE).repeat(2, 1),
        feet_indices=[0, 1, 3, 5], feet_contact_indices=sensor_ids, link_contact_forces=forces,
        _robot_mass=10., _added_base_mass=torch.tensor([[-1.], [10.]], device=DEVICE))
    env = SimpleNamespace(cfg=cfg, simulator=sim)
    torch.testing.assert_close(Go2Dreamwaq._reward_front_foot_overreach(env),
                               torch.tensor([.005, .0075], device=DEVICE))
    torch.testing.assert_close(Go2Dreamwaq._reward_rear_foot_overreach(env),
                               torch.tensor([2*.12**2, .12**2], device=DEVICE))
    sim.link_contact_forces.zero_()
    assert not Go2Dreamwaq._reward_front_foot_overreach(env).any()
    assert not Go2Dreamwaq._reward_rear_foot_overreach(env).any()


def test_isaaclab_velocity_limits_use_actuator_limits_in_policy_order():
    sim = object.__new__(IsaacLabSimulatorDreamWaQ)
    sim._dof_indices = [2, 0, 3, 1]
    sim._robot = SimpleNamespace(
        data=SimpleNamespace(joint_vel_limits=torch.tensor([[1.e9, 1.e9, 20., 1.e9]], device=DEVICE)),
        actuators={
            'calves': SimpleNamespace(joint_indices=[1, 3], velocity_limit=torch.tensor([[15.7, 15.7]], device=DEVICE)),
            'hips': SimpleNamespace(joint_indices=[0, 2], velocity_limit=torch.tensor([[30.1, 30.1]], device=DEVICE)),
        })
    torch.testing.assert_close(sim.dof_vel_limits, torch.tensor([20., 30.1, 15.7, 15.7], device=DEVICE))
