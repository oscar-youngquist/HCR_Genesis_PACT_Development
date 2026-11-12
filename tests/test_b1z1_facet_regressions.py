"""Launcher RNG initialization and schedule-override regressions, without a simulator."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_b1z1_facet_training import PacketEnv, make_runner, snapshot, unchanged
from rsl_rl.runners.b1z1_facet_runner import B1Z1FACETRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'legged_gym/scripts'))
import train_b1z1_facet as launcher


def test_launcher_registry_initializes_requested_rng_and_preserves_resume_seed(monkeypatch, tmp_path):
    registry = importlib.import_module('legged_gym.utils.task_registry')
    calls, environments = [], []
    real_seed = registry.set_seed

    def seed(value):
        calls.append(value)
        real_seed(value)

    class SeedEnv(PacketEnv):
        def __init__(self, cfg, **kwargs):
            # These are the actual global RNGs seeded inside TaskRegistry.make_env.
            self.initialization = (random.random(), np.random.rand(), torch.rand(3))
            super().__init__()
            self.cfg = cfg
            self.reward_names = []
            self.simulator._app_launcher = SimpleNamespace(app=SimpleNamespace(close=lambda: None))
            environments.append(self)

    def learn(self, **kwargs):
        self.save(Path(self.log_dir)/'model_0.pt')

    real_init = B1Z1FACETRunner.__init__
    def cpu_runner(self, env, cfg, log, device):
        real_init(self, env, cfg, log, 'cpu')

    monkeypatch.setattr(registry, 'set_seed', seed)
    monkeypatch.setitem(registry.task_registry.task_classes, 'b1z1_facet', SeedEnv)
    monkeypatch.setattr(B1Z1FACETRunner, 'learn', learn)
    monkeypatch.setattr(B1Z1FACETRunner, '__init__', cpu_runner)
    monkeypatch.setenv('SIMULATOR', 'isaaclab')

    def run(name, *args):
        out = tmp_path/name
        monkeypatch.setattr(sys, 'argv', ['train_b1z1_facet.py', '--gpu', 'cuda:0', '--num-envs', '4',
                                         '--log-dir', str(out), *map(str, args)])
        launcher.main()  # Real parser, config resolution, registry, set_seed and runner checkpoint/logging.
        state = torch.load(out/'model_0.pt', weights_only=False)
        logged = json.loads((out/'facet_resolved_config.json').read_text())
        return environments[-1], state, logged

    initial = []
    for requested in (0, 1, 2):
        env, state, logged = run(f'seed{requested}', '--seed', requested)
        assert calls[-1] == env.cfg.seed == state['config']['seed'] == requested
        assert state['contract']['environment']['seed'] == requested
        assert logged['environment']['seed'] == requested
        assert logged['training']['seed'] == requested
        repeated, _, _ = run(f'repeat{requested}', '--seed', requested)
        assert env.initialization[:2] == repeated.initialization[:2]
        torch.testing.assert_close(env.initialization[2], repeated.initialization[2], rtol=0, atol=0)
        initial.append(env.initialization)
    assert len({x[0] for x in initial}) == len({x[1] for x in initial}) == 3
    assert all(not torch.equal(initial[0][2], x[2]) for x in initial[1:])
    assert len({e.cfg.facet_reference.seed for e in environments}) == 1
    assert len({e.cfg.facet_observations.noise_seed for e in environments}) == 1

    source = tmp_path/'seed2/model_0.pt'
    # Old checkpoints can have contradictory env seeds; training seed is authoritative.
    old = torch.load(source, weights_only=False)
    old['contract']['environment']['seed'] = 1
    torch.save(old, source)
    resumed, state, logged = run('resume', '--resume', source)
    assert calls[-1] == resumed.cfg.seed == state['config']['seed'] == 2
    assert state['contract']['environment']['seed'] == logged['environment']['seed'] == logged['training']['seed'] == 2
    explicit, _, _ = run('resume_explicit', '--resume', source, '--seed', 2)
    assert explicit.initialization[:2] == resumed.initialization[:2]
    n = len(environments)
    with pytest.raises(ValueError, match='training-seed identity'):
        run('conflict', '--resume', source, '--seed', 0)
    assert len(environments) == n


@pytest.mark.parametrize('start,duration,iteration,expected', [
    (60000, 0, 59999, 'teacher'), (60000, 0, 60000, 'student'),
    (60000, 0, 70000, 'student'), (30000, 2000, 30000, 'adaptation'),
    (30000, 2000, 31999, 'adaptation'), (30000, 2000, 32000, 'student'),
    (30000, 2000, 40000, 'student'), (0, 0, 0, 'student'),
])
def test_resolved_phase_boundaries(start, duration, iteration, expected):
    runner = make_runner(start)
    runner.alg.cfg['adaptation_iterations'] = duration
    runner.current_learning_iteration = iteration
    teacher = snapshot(runner.model.teacher)
    runner.hidden.fill_(.3)
    runner.ema_hidden.fill_(.7)
    counters = deepcopy(runner.phase_interactions)
    assert runner.maybe_transition() == (expected != 'teacher')
    assert runner.alg.phase == expected
    assert not runner.maybe_transition()
    assert runner.phase_interactions == counters and runner.env.resets == 0
    assert torch.all(runner.hidden == .3) and torch.all(runner.ema_hidden == .7)
    if expected != 'teacher':
        assert runner.alg.student_initialized and unchanged(runner.model.student, teacher)


@pytest.mark.parametrize('phase,iteration,start,duration,expected', [
    ('teacher', 40000, 30000, 2000, 'student'),
    ('adaptation', 3, 2, 3, 'adaptation'),
    ('adaptation', 4, 2, 2, 'student'),
    ('adaptation', 3, 2, 0, 'student'),
    ('adaptation', 3, 10, 2, 'adaptation'),
    ('student', 5, 60000, 1000, 'student'),
])
def test_resume_destination_before_first_rollout(tmp_path, phase, iteration, start, duration, expected):
    runner = make_runner(60000)
    storage, _ = runner.collect()
    runner.alg.update(storage, 0)  # Populate actor/critic optimizer state before handoff.
    if phase == 'adaptation':
        runner.alg.switch_to_adaptation()
    elif phase == 'student':
        runner.alg.switch_to_student()
    if phase != 'teacher':
        with torch.no_grad():
            next(runner.model.student.parameters()).add_(.123)
    runner.current_learning_iteration = iteration
    runner.save(tmp_path/'resume.pt')
    resumed = make_runner()
    resumed.load(tmp_path/'resume.pt', schedule_overrides=dict(student_start_iteration=start, adaptation_iterations=duration))
    critic = next(resumed.model.critic.parameters())
    critic_state = deepcopy(resumed.alg.optimizer.state[critic])
    actor = resumed.model.teacher if phase == 'teacher' else resumed.model.student
    actor_state = deepcopy(resumed.alg.optimizer.state[next(actor.parameters())])
    student = snapshot(resumed.model.student)
    teacher = snapshot(resumed.model.teacher)
    norms = [snapshot(n) for n in (resumed.model.obs_norm, resumed.model.priv_norm, resumed.model.wrench_norm)]
    resets, count = resumed.env.resets, deepcopy(resumed.phase_interactions)
    resumed.maybe_transition()
    assert resumed.alg.phase == expected
    assert unchanged(resumed.model.student, teacher if phase == 'teacher' else student)
    assert resumed.env.resets == resets and resumed.phase_interactions == count
    for key, value in critic_state.items():
        torch.testing.assert_close(resumed.alg.optimizer.state[critic][key], value, rtol=0, atol=0)
    for key, value in actor_state.items():
        torch.testing.assert_close(resumed.alg.optimizer.state[next(resumed.model.student.parameters())][key], value, rtol=0, atol=0)
    assert all(unchanged(n, s) for n, s in zip((resumed.model.obs_norm, resumed.model.priv_norm, resumed.model.wrench_norm), norms))
    assert not resumed.maybe_transition()
    rollout, _ = resumed.collect()
    assert rollout.phase == expected
    assert resumed.phase_interactions == {k: v+(16 if k == expected else 0) for k, v in count.items()}
