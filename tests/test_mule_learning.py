"""Bounded CPU learning through the real MULE runner; no production reward/physics."""
from copy import deepcopy
from dataclasses import replace
import random
import numpy as np
import pytest
import torch
from legged_gym.envs.go2.go2_mule.go2_mule_config import Go2MULECfg, Go2MULECfgPPO
from legged_gym.envs.go2.go2_mule.go2_mule import Go2MULE, MULEObservation, MULEStep
from legged_gym.envs.go2.go2_mule.mule_schema import (
    MULESchema, FootForceEstimate, MULEExecution, MULEAuxiliaryBatch)
from rsl_rl.modules.actor_critic_mule import ActorCriticMULE
from rsl_rl.storage.rollout_storage_mule import RolloutStorageMULE
from rsl_rl.runners.mule_runner import MULERunner


@pytest.fixture(autouse=True)
def deterministic_threads():
    old = torch.get_num_threads(); torch.set_num_threads(1); torch.manual_seed(5)
    yield
    torch.set_num_threads(old)


def config(phase='nominal_only', **kw):
    defaults = dict(schema=MULESchema(history_length=3), optimization_phase=phase,
        num_envs=3, num_steps_per_env=4, num_learning_epochs=2, auxiliary_epochs=2,
        num_mini_batches=2, max_iterations=3, save_interval=2, schedule='fixed', seed=17)
    defaults.update(kw)
    return Go2MULECfgPPO(**defaults)


class SyntheticMULE(Go2MULE):
    """Deterministic test dynamics, two independent test rewards, one-step command delay."""
    supports_exact_resume = True

    def __init__(self, c):
        super().__init__(Go2MULECfg(c.schema, c.num_envs, c.control_dt, 4,
            c.action_scale, c.clip_actions, (0.,)*12))
        n = c.num_envs
        self.q = torch.zeros(n, 12); self.previous = self.q.clone(); self.queue = self.q.clone()
        self.torque = self.q.clone()
        self.source = torch.full((n,), -1, dtype=torch.long)
        self.episodes = torch.zeros(n, dtype=torch.long); self.lengths = self.episodes.clone()
        self.timestamp = 0
        self.valid = torch.ones(n, dtype=torch.bool)
        self.omit_terminal = False
        self.begin_calls, self.end_calls = [], []
        self.physical_stage = 1

    def observe(self):
        n = self.num_envs
        o = torch.zeros(n, 45)
        o[:, 0] = self.timestamp * .01; o[:, 1] = self.episodes.float()*.1
        o[:, 6:9] = .2
        o[:, 9:21] = self.q; o[:, 21:33] = self.torque*.1; o[:, 33:] = self.previous
        velocity = torch.stack((self.q.mean(-1), self.lengths.float()*.01, self.episodes.float()*.02), -1)
        critic = torch.cat((o, velocity, torch.zeros(n, 3+self.input_schema.terrain_dim)), -1)
        return MULEObservation(o, critic, velocity,
            FootForceEstimate(self.torque.reshape(n, 4, 3)*.1, torch.ones(n, 4, dtype=torch.bool)),
            self.episodes.clone(), self.valid.clone(), self.timestamp)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs) if env_ids is None else env_ids
        for key in ('q', 'previous', 'queue', 'torque'): getattr(self, key)[ids] = 0.
        self.source[ids] = -1; self.lengths[ids] = 0; self.episodes[ids] += 1
        return self.observe()

    def step(self, command):
        executed = self.queue.clone(); source = self.source.clone()
        self.torque = 2*(executed-self.q)
        self.q += .01*self.torque
        self.queue.copy_(command.position_target); self.source.fill_(self.timestamp)
        self.previous.copy_(command.combined_clipped)
        self.timestamp += 1; self.lengths += 1
        terminal = self.observe().snapshot()
        horizon = 2 + torch.arange(self.num_envs) % 3
        done = self.lengths >= horizon
        terminated = done & (torch.arange(self.num_envs) != 0)
        timeout = done & (torch.arange(self.num_envs) != 2)  # row 1 simultaneously fails/times out
        nominal_reward = 1. - self.q.square().sum(-1)
        adaptive_reward = -.3 - self.torque.square().sum(-1)*.02  # distinct, never summed
        applied_torque = self.torque.clone()
        self.reset(done.nonzero().flatten())
        self.last_terminal = terminal
        return MULEStep(self.observe(), MULEExecution(executed, source, applied_torque),
            nominal_reward, adaptive_reward, terminated, timeout, None if self.omit_terminal else terminal,
            done, {'lin_vel_tracking': .8, 'ang_vel_tracking': .4, 'terrain_level': 2.})

    def begin_iteration(self, iteration): self.begin_calls.append(iteration)
    def end_iteration(self, iteration, evidence): self.end_calls.append((iteration, deepcopy(evidence)))

    def state_dict(self):
        keys = ('q', 'previous', 'queue', 'torque', 'source', 'episodes', 'lengths', 'timestamp',
                'valid', 'begin_calls', 'end_calls', 'physical_stage')
        return deepcopy({k: getattr(self, k) for k in keys})

    def validate_state_dict(self, state):
        if set(state) != set(self.state_dict()): raise ValueError('Synthetic state keys differ')
        for key, value in self.state_dict().items():
            if isinstance(value, torch.Tensor) and (state[key].shape != value.shape or state[key].dtype != value.dtype):
                raise ValueError('Synthetic state shape differs')

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        for key, value in deepcopy(state).items(): setattr(self, key, value)


def runner(phase='nominal_only', **kwargs):
    c = config(phase, **kwargs)
    return MULERunner(SyntheticMULE(c), c)


def assert_equal(a, b):
    if isinstance(a, torch.Tensor): torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a: assert_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_equal(x, y)
    else: assert a == b


def changed(before, module):
    return any(not torch.equal(before[k], v) for k, v in module.state_dict().items() if isinstance(v, torch.Tensor))


def test_two_stream_hand_calculated_gae_and_owned_shuffle():
    c = config('joint_adaptation', num_envs=1, num_steps_per_env=3, num_mini_batches=2, gamma=.9, lam=.8)
    s = RolloutStorageMULE(c)
    for t in range(3):
        row = {k: torch.zeros_like(v[0]) for k, v in s.data.items()}
        row['values'][0] = torch.tensor([t+1, 10*(t+1)])
        row['rewards'].copy_(row['values'])
        row['timestamp'].fill_(t); row['observations'].fill_(t)
        row['context_latent'].fill_(100+t); row['nominal_raw'].fill_(200+t)
        row['adaptive_raw'].fill_(300+t); row['executed_source_id'].fill_(t-1)
        if t == 1:
            row['timed_out'].fill_(True); row['terminal_valid'].fill_(True)
            row['terminal_values'][0] = torch.tensor([4., 40.])
        s.add(row, 'joint_adaptation')
        row['observations'].fill_(-999)  # no mutable alias
    s.compute_returns(torch.tensor([[5., 50.]]))
    torch.testing.assert_close(s.raw_advantages[:, 0, 0], torch.tensor([4.392, 3.6, 4.5]))
    torch.testing.assert_close(s.raw_advantages[:, 0, 1], s.raw_advantages[:, 0, 0]*10)
    torch.testing.assert_close(s.returns[:, 0, 0], torch.tensor([5.392, 5.6, 7.5]))
    seen = []
    for batch in s.batches():
        ids = batch['indices']; seen.extend(ids.tolist())
        assert torch.equal(batch['timestamp'], ids)
        assert torch.equal(batch['executed_source_id'], ids-1)
        assert torch.equal(batch['observations'][:, 0], ids.float())
        assert torch.equal(batch['context_latent'][:, 0], 100+ids.float())
        assert torch.equal(batch['nominal_raw'][:, 0], 200+ids.float())
        assert torch.equal(batch['adaptive_raw'][:, 0], 300+ids.float())
    assert sorted(seen) == [0, 1, 2]


@pytest.mark.parametrize('phase', ['nominal_only', 'joint_adaptation'])
def test_real_collection_likelihoods_gradients_and_auxiliary_order(phase):
    r = runner(phase); r.collect_rollout()
    batch = next(r.alg.storage.batches())
    rng = torch.get_rng_state().clone()
    for owner in r.alg.active_owners:
        r.alg.model.zero_grad(set_to_none=True)
        loss, metrics = r.alg.policy_objective(batch, owner)
        torch.testing.assert_close(metrics['ratio_mean'], torch.tensor(1.), atol=2e-6, rtol=0)
        assert abs(float(metrics['kl'])) < 1e-6
        loss.backward()
        for name, params in r.alg.groups.items():
            if name == owner: assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
            else: assert all(p.grad is None for p in params)
    assert torch.equal(rng, torch.get_rng_state())
    r.alg.model.zero_grad(set_to_none=True)
    b = batch
    aux = MULEAuxiliaryBatch(b['history'], b['velocity_b'], b['next_observation'],
        b['auxiliary_valid'], b['episode_id'], b['next_episode_id'])
    r.alg.model.context_estimator.losses(aux).total.backward()
    assert all(p.grad is None for k in ('nominal', 'adaptive') for p in r.alg.groups[k])
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in r.alg.groups['auxiliary'])
    before = {k: deepcopy(getattr(r.alg.model, k+'_actor').state_dict()) for k in ('nominal', 'adaptive')}
    critics = {k: deepcopy(getattr(r.alg.model, k+'_critic').state_dict()) for k in ('nominal', 'adaptive')}
    order = []
    handles = [opt.register_step_post_hook(lambda opt, args, kwargs, k=k: order.append(k)) for k, opt in r.alg.optimizers.items()]
    result = r.alg.update()
    for handle in handles: handle.remove()
    assert all(np.isfinite(v) for v in result.values())
    assert changed(before['nominal'], r.alg.model.nominal_actor)
    assert changed(before['adaptive'], r.alg.model.adaptive_actor) == (phase == 'joint_adaptation')
    assert changed(critics['nominal'], r.alg.model.nominal_critic)
    assert changed(critics['adaptive'], r.alg.model.adaptive_critic) == (phase == 'joint_adaptation')
    first_aux = order.index('auxiliary')
    assert set(order[first_aux:]) == {'auxiliary'}
    assert set(order[:first_aux]) == set(r.alg.active_owners)


def test_timeout_failure_terminal_alignment_partial_resets_and_delay():
    r = runner('joint_adaptation')
    with torch.no_grad():
        for name, constant in (('nominal_critic', 2.), ('adaptive_critic', 5.)):
            critic = getattr(r.alg.model, name)
            for p in critic.parameters(): p.zero_()
            critic[-1].bias.fill_(constant)
    r.collect_step()
    first = r.alg.storage.data
    old_history_row1 = r.history.observations[1].clone()
    assert torch.equal(first['executed_source_id'][0], torch.full((3,), -1))
    assert torch.count_nonzero(first['executed_target'][0]) == 0
    r.collect_step()  # row 0 timeout, other rows untouched
    terminal = r.env.last_terminal
    torch.testing.assert_close(first['terminal_values'][1, 0], torch.tensor([2., 5.]))
    assert torch.equal(first['next_observation'][1, 0], terminal.actor[0])
    assert not torch.equal(first['next_observation'][1, 0], r.current.actor[0])
    assert torch.equal(first['velocity_b'][1], first['critic'][1, :, 45:48])
    assert torch.count_nonzero(r.history.observations[0]) == 0
    assert torch.equal(r.history.observations[1, :-1], old_history_row1[1:])
    assert torch.equal(first['executed_target'][1], first['issued_target'][0])
    r.collect_step()  # row 1 has simultaneous physical failure/timeout
    assert first['terminated'][2, 1] and first['timed_out'][2, 1]
    assert torch.count_nonzero(first['terminal_values'][2, 1]) == 0
    r.collect_step(); r.alg.compute_returns(r.current.critic)
    torch.testing.assert_close(r.alg.storage.returns[2, 1], first['rewards'][2, 1])
    # Same-episode terminals are valid auxiliary targets even on a physical terminal.
    assert first['auxiliary_valid'][2, 1]
    assert first['next_episode_id'][2, 1] == first['episode_id'][2, 1]
    terminal.actor.fill_(999)
    assert not (first['next_observation'][1] == 999).any()


def test_missing_terminal_fails_before_timeout_bootstrap():
    r = runner(); r.env.omit_terminal = True
    r.collect_step()
    with pytest.raises(ValueError, match='Timeout requires'): r.collect_step()


def test_observe_and_bootstrap_do_not_advance_rng_or_history():
    r = runner(); r.collect_rollout()
    rng, history = torch.get_rng_state().clone(), r.history.snapshot()
    timestamp = r.history.timestamp
    for _ in range(3):
        r.env.observe(); r.alg.model.values(r.current.critic)
        r.get_inference_policy()(r.current.actor, *r.history.snapshot())
    assert torch.equal(rng, torch.get_rng_state())
    assert_equal(history, r.history.snapshot()); assert timestamp == r.history.timestamp


def test_inactive_optimizer_moments_schedule_and_empty_auxiliary_preserved():
    r = runner('joint_adaptation'); r.learn(1)
    # Seed nonempty adaptive Adam moments, then exercise inactive-owner preservation.
    r.alg.model.set_phase('nominal_only')
    before_adaptive = deepcopy(r.alg.optimizers['adaptive'].state_dict())
    before_aux = deepcopy(r.alg.optimizers['auxiliary'].state_dict())
    before_schedule = deepcopy(r.alg.schedulers['adaptive'])
    before_weights = deepcopy(r.alg.model.context_estimator.state_dict())
    r.env.valid.zero_(); r.current = r.env.observe().snapshot()
    logs = r.learn(1)
    assert logs[0]['loss/auxiliary_steps'] == 0
    assert_equal(before_adaptive, r.alg.optimizers['adaptive'].state_dict())
    assert_equal(before_aux, r.alg.optimizers['auxiliary'].state_dict())
    assert_equal(before_weights, r.alg.model.context_estimator.state_dict())
    assert_equal(before_schedule, r.alg.schedulers['adaptive'])


def test_phase_transition_guards_and_parameters_preserved():
    r = runner(); r.collect_step()
    with pytest.raises(RuntimeError, match='completed update'): r.transition_to_joint_adaptation()
    while not r.alg.storage.full: r.collect_step()
    r.alg.compute_returns(r.current.critic)
    with pytest.raises(RuntimeError): r.transition_to_joint_adaptation()
    r.alg.update()
    weights = {k: v.clone() for k, v in r.alg.model.state_dict().items() if isinstance(v, torch.Tensor)}
    moments = deepcopy(r.alg.optimizers['nominal'].state_dict())
    r.transition_to_joint_adaptation(); r.transition_to_joint_adaptation()
    for key, value in weights.items(): assert torch.equal(value, r.alg.model.state_dict()[key])
    assert_equal(moments, r.alg.optimizers['nominal'].state_dict())
    assert r.env.physical_stage == 1
    r.learn(1); assert r.alg.schedulers['adaptive']['optimizer_steps'] > 0


def test_pending_action_and_phase_hook_guards(tmp_path):
    r = runner(); r._fresh_environment()
    o = r.current
    r.history.append(o.actor, o.foot_force_estimate, o.episode_id, o.timestamp)
    r.alg.act(o, *r.history.snapshot(), r.standing_position)
    with pytest.raises(RuntimeError): r.transition_to_joint_adaptation()
    with pytest.raises(RuntimeError): r.save(tmp_path/'pending.pt')
    r = runner()
    r.phase_transition_hook = lambda iteration, env: 'joint_adaptation' if iteration >= 1 else None
    logs = r.learn(2)
    assert logs[0]['train/joint_adaptation'] == 0 and logs[1]['train/joint_adaptation'] == 1
    assert r.env.physical_stage == 1


def test_separate_schedulers_and_missing_evidence():
    r = runner(schedule='adaptive')
    inactive = deepcopy(r.alg.schedulers['adaptive'])
    r.alg._adjust_lr('nominal', r.config.desired_kl*3)
    assert r.alg.schedulers['nominal']['learning_rate'] < r.config.learning_rate
    assert_equal(inactive, r.alg.schedulers['adaptive'])
    before = deepcopy(r.alg.schedulers)
    r.alg.update_entropy(None)
    r.alg.update_entropy({'lin_vel_tracking': float('nan'), 'ang_vel_tracking': 1., 'terrain_level': 0.})
    assert_equal(before, r.alg.schedulers)
    evidence = dict(lin_vel_tracking=.1, ang_vel_tracking=.1, terrain_level=0.)
    r.alg.update_entropy(evidence)
    assert_equal(inactive, r.alg.schedulers['adaptive'])
    assert r.alg.schedulers['nominal']['entropy_updates'] == 1
    r.transition_to_joint_adaptation(); r.alg.update_entropy(evidence)
    assert r.alg.schedulers['adaptive']['entropy_updates'] == 1
    assert r.alg.schedulers['nominal']['entropy_updates'] == 2


def test_missing_failure_terminal_is_not_paired_with_reset():
    r = runner(); r.collect_step(); r.collect_step()
    r.env.omit_terminal = True
    r.collect_step()  # Row 1 is failure + timeout: no bootstrap required.
    d = r.alg.storage.data
    assert d['terminated'][2, 1] and not d['auxiliary_valid'][2, 1]
    assert not d['terminal_valid'][2, 1] and d['terminal_values'][2, 1].sum() == 0


def test_compose_rejects_inactive_corrections_and_std_projection_remains_trainable():
    r = runner()
    with pytest.raises(ValueError, match='zero adaptive'):
        r.alg.model.compose_actions(torch.zeros(3, 12), torch.ones(3, 12), torch.zeros(12))
    r.transition_to_joint_adaptation()
    with torch.no_grad():
        for owner in r.alg.active_owners:
            actor = getattr(r.alg.model, owner+'_actor')
            actor.std[:6] = actor.std_lower[:6]*.1; actor.std[6:] = actor.std_upper[6:]*10
    r.learn(1)
    for owner in r.alg.active_owners:
        actor = getattr(r.alg.model, owner+'_actor')
        assert torch.all(actor.std >= actor.std_lower) and torch.all(actor.std <= actor.std_upper)
    r.collect_rollout(); batch = next(r.alg.storage.batches())
    for owner in r.alg.active_owners:
        r.alg.model.zero_grad(set_to_none=True)
        loss, _ = r.alg.policy_objective(batch, owner); loss.backward()
        grad = getattr(r.alg.model, owner+'_actor').std.grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize('phase', ['nominal_only', 'joint_adaptation'])
def test_checkpoint_exact_and_fresh_resume_with_runtime_overrides(tmp_path, phase):
    r = runner(phase); r.learn(1)
    path = tmp_path/'learning.pt'; r.save(path)
    saved = deepcopy(r.alg.state_dict())
    r.learn(1)
    expected = deepcopy(r.alg.state_dict())
    expected_history = r.history.snapshot()
    rng = (random.random(), np.random.rand(), torch.rand(3))
    c = replace(r.config, max_iterations=10, save_interval=7, optimization_phase='nominal_only')
    restored = MULERunner(SyntheticMULE(c), c, log_dir=tmp_path/'different-log')
    restored.load(path, resume_mode='exact')
    assert restored.alg.model.phase == phase
    restored.learn(1)
    # Only explicitly supported runtime config differences survive checkpoint loading.
    actual = restored.alg.state_dict(); actual['config'] = expected['config']
    assert_equal(actual, expected); assert_equal(restored.history.snapshot(), expected_history)
    assert random.random() == rng[0] and np.random.rand() == rng[1]
    assert torch.equal(torch.rand(3), rng[2])
    restored.load(path, resume_mode='fresh')
    assert restored.alg.iteration == 1 and restored.interactions == r.config.num_envs*r.config.num_steps_per_env
    assert restored.history.timestamp == -1 and torch.count_nonzero(restored.history.observations) == 0
    assert torch.count_nonzero(restored.episode_lengths) == 0
    assert_equal(restored.alg.optimizers['nominal'].state_dict(), saved['optimizers']['nominal'])
    assert restored.alg.model.phase == phase
    restored.learn(1)


def test_checkpoint_incompatibility_rejected_before_live_mutation(tmp_path):
    r = runner(); r.learn(1); path = tmp_path/'state.pt'; r.save(path)
    incompatible = runner(gamma=.8)
    before = deepcopy(incompatible.alg.model.state_dict())
    with pytest.raises(ValueError, match='learning contract'): incompatible.load(path)
    assert_equal(before, incompatible.alg.model.state_dict())
    assert incompatible.current is None
    damaged = torch.load(path, weights_only=False)
    damaged['algorithm']['model']['_extra_state']['contract']['schema']['version'] = 999
    torch.save(damaged, path)
    before = deepcopy(r.alg.model.state_dict())
    with pytest.raises(ValueError, match='contract'): r.load(path)
    assert_equal(before, r.alg.model.state_dict())


@pytest.mark.parametrize('kwargs', [dict(gamma=1.1), dict(lam=float('nan')), dict(schedule='unknown'),
    dict(value_loss_coef=-1), dict(entropy_coef=-1), dict(max_grad_norm=0), dict(max_iterations=-1),
    dict(save_interval=0), dict(adaptive_ent_bounds=(1., .1))])
def test_invalid_learning_configuration(kwargs):
    with pytest.raises(ValueError): config(**kwargs)


def test_bounded_nominal_transition_joint_save_resume_smoke(tmp_path):
    r = runner(); nominal = r.learn(2)
    r.transition_to_joint_adaptation(); joint = r.learn(2)
    path = tmp_path/'joint.pt'; r.save(path)
    restored = runner(); restored.load(path, resume_mode='exact'); continuation = restored.learn(1)
    assert restored.alg.iteration == 5 and restored.interactions == 5*3*4
    assert restored.alg.model.phase == 'joint_adaptation'
    for record in nominal + joint + continuation: assert all(np.isfinite(x) for x in record.values())
    assert 'loss/adaptive/surrogate' not in nominal[0] and 'loss/adaptive/surrogate' in joint[0]
    assert r.env.begin_calls == [0, 1, 2, 3]
    assert [x[0] for x in r.env.end_calls] == [0, 1, 2, 3]
