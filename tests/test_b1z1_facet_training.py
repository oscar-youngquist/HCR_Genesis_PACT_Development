"""Stage 5: on-policy replay, recurrent ownership, and completed-update handoff."""
from copy import deepcopy
from types import SimpleNamespace
import torch
import pytest

from legged_gym.envs.b1z1.b1z1_facet import B1Z1FACETCfg, B1Z1FACETCfgPPO
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.facet_model import FACETModel
from rsl_rl.facet_ppo import FACETPPO
from rsl_rl.facet_storage import gae
from rsl_rl.runners.b1z1_facet_runner import B1Z1FACETRunner


class PacketEnv:
    """Distinct terminal/reset packets, partial timeout and true-failure boundaries."""
    def __init__(self):
        self.cfg = B1Z1FACETCfg()
        self.num_envs, self.device, self.dt = 4, 'cpu', .02
        self.all_env_ids = torch.arange(4)
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.max_episode_length = 10
        self.common_step_counter = 0
        self.reward_scales, self.command_ranges = {}, {}
        self.simulator = SimpleNamespace(use_domainrand_curriculum=False)
        self.resets = 0
        self.iterations = []
        from legged_gym.envs.b1z1.force_task_utils import init_staged_force_curriculum
        init_staged_force_curriculum(self)
        self.packet = dict(observation=torch.randn(4, 7), privileged=torch.randn(4, 5),
                           wrench=torch.randn(4, 9), is_init=torch.ones(4, 1, dtype=torch.bool))
        self.terminal = None
        self.z1_joint_offsets = self.z1_joint_axes = torch.zeros(6, 3)
        self.z1_link00_offset = self.z1_ee_offset = torch.zeros(3)

    def get_facet_observation_schema(self):
        from legged_gym.envs.b1z1.b1z1_facet.observations import JOINTS, ACTION_JOINTS
        return dict(observed_joint_names=list(JOINTS), action_joint_names=list(ACTION_JOINTS),
                    observation_blocks=[['test', 7]], privileged_blocks=[['test', 5]])

    def get_facet_observations(self):
        return {k: v.clone() for k, v in self.packet.items()}

    def get_facet_terminal_observations(self):
        return self.terminal

    def set_training_iteration(self, iteration):
        self.training_iteration = iteration
        self.iterations.append(iteration)

    def reset_idx(self, ids):
        self.resets += 1
        for k, v in self.packet.items():
            v[ids] = 1 if k == 'is_init' else -7
        self.episode_length_buf[ids] = 0

    def step(self, actions):
        self.last_raw_actions = actions.clone()
        self.common_step_counter += 1
        self.episode_length_buf += 1
        for k, v in self.packet.items():
            if k == 'is_init':
                v.zero_()
            else:
                v.add_(.1)
        done = torch.tensor([self.common_step_counter % 2 == 0, self.common_step_counter % 3 == 0, False, False])
        ids = done.nonzero().flatten()
        self.terminal = {k: v[ids].clone() for k, v in self.packet.items()}
        self.terminal.update(ids=ids, timeouts=ids == 0, failures=ids == 1)
        self.reset_idx(ids)
        return (None, None, None, None, torch.arange(4).float()+.5, done, {}, None)


def make_runner(start=2):
    torch.manual_seed(12)
    cfg = class_to_dict(B1Z1FACETCfgPPO())
    cfg['runner']['num_steps_per_env'] = 4
    cfg['algorithm'].update(num_learning_epochs=1, num_mini_batches=2, student_start_iteration=start)
    runner = B1Z1FACETRunner(PacketEnv(), cfg)
    return runner


def snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def unchanged(module, before):
    return all(torch.equal(v, before[k]) for k, v in module.state_dict().items())


@pytest.mark.parametrize('student', [False, True])
def test_collection_replays_exact_gaussian_and_normalization(student):
    runner = make_runner()
    if student:
        runner.alg.switch_to_student()
    norms = snapshot(runner.model.obs_norm)
    storage, _ = runner.collect()
    during = snapshot(runner.model.obs_norm)
    for batch in storage.minibatches(3):
        distribution = runner.alg.distribution(batch)
        torch.testing.assert_close(distribution.mean, batch['means'], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(distribution.stddev, batch['stds'], atol=0, rtol=0)
        torch.testing.assert_close((distribution.log_prob(batch['actions'])-batch['log_probs']).exp(),
                                   torch.ones(len(batch['obs'])), atol=5e-6, rtol=0)
    assert unchanged(runner.model.obs_norm, during)
    assert not unchanged(runner.model.obs_norm, norms)  # Exactly one initial batch update.
    assert runner.model.obs_norm.count == 5
    assert torch.equal(storage.data['actions'][-1], runner.env.last_raw_actions)
    # t=2 has a true failure, t=3 a timeout: only terminal successor V bootstraps.
    assert storage.data['successor_values'][2, 1] == 0
    terminal_v = runner.alg.value(runner._normalize(runner.env.terminal))
    torch.testing.assert_close(storage.data['successor_values'][-1, 0], terminal_v[0])
    assert not torch.equal(terminal_v[0], runner.alg.value(runner._normalize(runner.env.packet))[0])


def test_gae_stops_at_timeout_and_failure():
    rewards = torch.tensor([[1., 2.], [3., 4.]])
    values = torch.tensor([[.5, .7], [.2, .3]])
    successor = torch.tensor([[10., 0.], [2., 3.]])
    done = torch.tensor([[True, True], [False, False]])
    advantage, returns = gae(rewards, values, successor, done, .9, .95)
    torch.testing.assert_close(advantage, torch.tensor([[9.5, 1.3], [4.6, 6.4]]))
    torch.testing.assert_close(returns, advantage+values)


def test_gradient_ownership_and_no_wrench_reconstruction_objective():
    runner = make_runner()
    model, alg = runner.model, runner.alg
    storage, _ = runner.collect()
    batch = next(storage.minibatches(1))
    initial = {name: snapshot(getattr(model, name)) for name in ('teacher', 'critic', 'priv_encoder', 'force_encoder', 'estimator', 'student')}
    alg.optimizer.zero_grad(set_to_none=True)
    alg.ppo_loss(batch)[0].backward()
    alg.optimizer.step()
    for name in ('teacher', 'critic', 'priv_encoder', 'force_encoder'):
        assert not unchanged(getattr(model, name), initial[name])
    for name in ('estimator', 'student'):
        assert unchanged(getattr(model, name), initial[name])
    teacher = snapshot(model.teacher)
    priv = snapshot(model.priv_encoder)
    output_tail = model.estimator.out.weight[-9:].detach().clone()
    sequence, hidden = next(storage.sequences(1))
    alg.estimator_optimizer.zero_grad(set_to_none=True)
    alg.estimator_loss(sequence, hidden)[0].backward()
    alg.estimator_optimizer.step()
    assert not unchanged(model.estimator, initial['estimator'])
    assert unchanged(model.teacher, teacher) and unchanged(model.priv_encoder, priv)
    torch.testing.assert_close(output_tail, model.estimator.out.weight[-9:], rtol=0, atol=0)
    critic_param = next(model.critic.parameters())
    critic_state = alg.optimizer.state[critic_param]
    actor_param = next(model.teacher.parameters())
    actor_state = deepcopy(alg.optimizer.state[actor_param])
    assert alg.switch_to_student()
    assert unchanged(model.student, teacher)
    assert alg.optimizer.state[critic_param] is critic_state
    for key, value in actor_state.items():
        torch.testing.assert_close(alg.optimizer.state[next(model.student.parameters())][key], value)
    with pytest.raises(RuntimeError, match='reuse teacher'):
        alg.update(storage, 2)
    storage, _ = runner.collect()
    frozen = {name: snapshot(getattr(model, name)) for name in ('teacher', 'priv_encoder', 'force_encoder')}
    estimator = snapshot(model.estimator)
    alg.optimizer.zero_grad(set_to_none=True)
    alg.ppo_loss(next(storage.minibatches(1)))[0].backward()
    alg.optimizer.step()
    assert not unchanged(model.student, teacher)
    assert unchanged(model.estimator, estimator)
    assert all(unchanged(getattr(model, name), before) for name, before in frozen.items())


def test_recurrent_sequence_padding_reset_and_rollout_boundary():
    runner = make_runner()
    storage, _ = runner.collect()
    batch, initial = next(storage.sequences(1))
    loss, _ = runner.alg.estimator_loss(batch, initial)
    pred = runner.model.estimator(batch['obs'], initial, batch['is_init'])
    target = torch.cat((runner.model.priv_encoder(batch['priv']), runner.model.force_encoder(batch['wrench'])), -1).detach()
    error = (torch.cat(pred[:2], -1)-target).square()
    expected = ((error[..., :128].mean(-1)+error[..., 128:].mean(-1))*(~batch['is_init'])).mean()
    torch.testing.assert_close(loss, expected)
    padded = {k: torch.cat((v, torch.zeros_like(v[:, :2])), 1) for k, v in batch.items()}
    torch.testing.assert_close(runner.alg.estimator_loss(padded, initial)[0], loss)
    hidden = runner.hidden.clone()
    next_storage, _ = runner.collect()
    torch.testing.assert_close(next_storage.initial_hidden, hidden)
    # A reset observation discards arbitrary stale hidden state before processing.
    reset = torch.ones(4, 1, dtype=torch.bool)
    x = torch.randn(4, 7)
    a = runner.model.estimator(x, torch.randn(4, 128)*100, reset)
    b = runner.model.estimator(x, torch.zeros(4, 128), reset)
    for one, two in zip(a, b):
        torch.testing.assert_close(one, two)


def test_transition_once_resume_and_frozen_latents(tmp_path):
    runner = make_runner(2)
    runner._schedules = lambda iteration, infos: {}
    runner.learn(1)
    assert runner.alg.phase == 'teacher'
    runner.learn(2)
    assert runner.alg.phase == 'student' and runner.current_learning_iteration == 2
    teacher = snapshot(runner.model.teacher)
    assert unchanged(runner.model.student, teacher)
    norms, targets = snapshot(runner.model.obs_norm), snapshot(runner.model.priv_encoder)
    resets, hidden = runner.env.resets, runner.ema_hidden.clone()
    assert not runner.maybe_transition()
    assert runner.env.resets == resets
    torch.testing.assert_close(hidden, runner.ema_hidden)
    runner.learn(3)
    assert not unchanged(runner.model.student, teacher)
    assert unchanged(runner.model.teacher, teacher)
    assert unchanged(runner.model.obs_norm, norms) and unchanged(runner.model.priv_encoder, targets)
    path = tmp_path/'model.pt'
    runner.save(path)
    resumed = make_runner(2)
    resumed.load(path)
    assert resumed.current_learning_iteration == 3 and resumed.alg.phase == 'student'
    assert unchanged(resumed.model, snapshot(runner.model))
    assert unchanged(resumed.alg.ema, snapshot(runner.alg.ema))
    assert resumed.hidden.count_nonzero() == resumed.ema_hidden.count_nonzero() == 0
    assert resumed.env.packet['is_init'].all()
    assert not resumed.maybe_transition()
    resumed._schedules = lambda iteration, infos: {}
    resumed.learn(4)
    assert resumed.current_learning_iteration == 4
    assert unchanged(resumed.model.teacher, teacher)
    assert 3 in resumed.env.iterations and 4 in resumed.env.iterations


def test_student_inference_does_not_use_privilege_or_wrench():
    runner = make_runner()
    runner.collect()
    runner.alg.switch_to_student()
    inference = runner.get_inference_policy()
    packet = runner.env.get_facet_observations()
    h = torch.randn(4, 128)
    action, _ = inference(packet['observation'], h, packet['is_init'])
    runner.env.packet['privileged'].fill_(999)
    runner.env.packet['wrench'].fill_(-999)
    repeat, _ = inference(packet['observation'], h, packet['is_init'])
    torch.testing.assert_close(action, repeat, atol=0, rtol=0)


def test_pilot_protocol_and_effective_reward_audit():
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
    from legged_gym.envs.b1z1.b1z1_unifp_original.b1z1_unifp_original_config import copy_sections
    from legged_gym.envs.b1z1.b1z1_facet.tracking import LEGACY_TRACKING, CONFLICTING_SHAPING, FF_ONLY
    from test_b1z1_facet import differences
    from test_b1z1_facet_history import reward_env
    pilot, facet = copy_sections(B1Z1PACTCfg), B1Z1FACETCfg()
    a, b = class_to_dict(pilot), class_to_dict(facet)
    changes = differences(a, b)
    allowed = {'action_mode', 'ablation_variant', 'env.num_policy_actions', 'compliance.enabled',
               'facet_reference', 'facet_tracking', 'facet_observations'}
    allowed |= {f'control.{kind}.{name}' for kind in ('stiffness', 'damping')
                for name in set(a['control'][kind]) | set(b['control'][kind])}
    assert changes <= allowed  # Some arm gains already coincide with FACET.
    assert changes - {p for p in allowed if p.startswith('control.')} == allowed - {p for p in allowed if p.startswith('control.')}
    assert not facet.terrain.curriculum
    active = reward_env()
    excluded = set(LEGACY_TRACKING+CONFLICTING_SHAPING+FF_ONLY) | {'compliance_ee_velocity_weight'}
    for key, weight in a['rewards']['scales'].items():
        if not weight:
            continue
        if key in excluded:
            assert key not in active.reward_names
        else:
            assert key in active.reward_names
            assert active.reward_scales[key] == pytest.approx(weight*active.dt)
    assert b['rewards']['reward_curriculum'] == a['rewards']['reward_curriculum']


def test_estimator_cadence_and_no_schedule_reset_at_handoff():
    runner = make_runner(2)
    calls = []
    original = runner.alg.update
    def update(storage, iteration):
        metrics = original(storage, iteration)
        calls.append((iteration, storage.phase, 'estimator_priv' in metrics))
        return metrics
    runner.alg.update = update
    schedule_iterations = []
    runner._schedules = lambda iteration, infos: schedule_iterations.append(iteration) or {}
    runner.learn(4)
    assert calls == [(0, 'teacher', True), (1, 'teacher', False),
                     (2, 'student', True), (3, 'student', True)]
    assert schedule_iterations == [0, 1, 2, 3]


def test_terminal_capture_precedes_reset_and_reads_are_pure():
    from test_b1z1_facet_history import reward_env
    env = reward_env()
    env.reward_names, env.reward_functions = [], []
    env.all_env_ids = torch.arange(2)
    env.reset_buf = env.time_out_buf = torch.ones(2, dtype=torch.bool)
    env.fail_buf = torch.tensor([False, True])  # Timeout AND failure: failure wins.
    env._facet_issued_action = torch.zeros(2, 17)
    env._facet_observations = SimpleNamespace(packet=dict(observation=torch.full((2, 7), 123.),
        privileged=torch.ones(2, 5), wrench=torch.ones(2, 9), is_init=torch.zeros(2, 1, dtype=torch.bool)),
        skip=torch.zeros(2, dtype=torch.bool), collect=lambda *args: None)
    env.compute_reward()
    assert env._facet_terminal['timeouts'].tolist() == [True, False]
    assert env._facet_terminal['failures'].tolist() == [False, True]
    # Reset writes and consumer mutation cannot corrupt terminal bootstrapping.
    env._facet_observations.packet['observation'].zero_()
    first = env.get_facet_terminal_observations()
    first['observation'].zero_()
    assert (env.get_facet_terminal_observations()['observation'] == 123.).all()


def test_transition_checkpoint_does_not_shadow_later_checkpoints(tmp_path):
    from legged_gym.utils.helpers import get_load_path
    runner = make_runner(0)
    runner.log_dir = str(tmp_path/'run')
    (tmp_path/'run').mkdir()
    runner.maybe_transition()
    normalizers = snapshot(runner.model.obs_norm)
    runner.collect()
    runner._finish_normalization()
    assert unchanged(runner.model.obs_norm, normalizers)
    runner.current_learning_iteration = 1000
    runner.save(tmp_path/'run'/'model_1000.pt')
    assert (tmp_path/'run'/'transition_0.pt').exists()
    assert get_load_path(str(tmp_path), 'run', -1).endswith('model_1000.pt')
