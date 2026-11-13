"""FACET-only reporting: collection ownership, effective std and elapsed timing."""
from unittest.mock import Mock

import pytest
import torch

from test_b1z1_facet_training import make_runner


@pytest.mark.parametrize('phase', ['teacher', 'adaptation', 'student'])
def test_std_uses_collected_policy_without_mutation(phase):
    runner = make_runner()
    with torch.no_grad():
        runner.model.teacher.std.fill_(.8)
        runner.model.student.std.fill_(.2)
        runner.model.student.std[0] = -1
    before = runner.model.student.std.detach().clone()
    rng = torch.random.get_rng_state().clone()
    result = runner._policy_log_metrics(phase)
    expected = .8 if phase == 'teacher' else (16*.2 + 1e-6)/17
    assert result['Policy/mean_noise_std'] == pytest.approx(expected)
    assert result['Policy/position_noise_std'] == pytest.approx(expected)
    torch.testing.assert_close(runner.model.student.std, before)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert 'Train/mean_reward' not in result


def test_tensorboard_and_terminal_across_handoff(capsys):
    runner = make_runner(start=1)
    runner.writer = Mock()
    runner.learn(2)
    calls = runner.writer.add_scalar.call_args_list
    for iteration in (1, 2):
        for tag in ('Policy/mean_noise_std', 'Policy/position_noise_std',
                    'Perf/total_fps', 'Train/mean_reward', 'Train/mean_episode_length'):
            values = [c.args[1] for c in calls if c.args[0] == tag and c.args[2] == iteration]
            assert len(values) == 1 and torch.isfinite(torch.tensor(values)).all()
    output = capsys.readouterr().out
    for label in ('Learning iteration 1/2', 'Learning iteration 2/2',
                  'FACET rollout phase:', 'Next rollout phase:', 'Surrogate loss:',
                  'Value function loss:', 'Position action noise std:', 'Total timesteps:', 'ETA:'):
        assert label in output
    assert 'PINN loss' not in output and 'Torque action noise std' not in output
    assert runner.phase_interactions == dict(teacher=16, adaptation=0, student=16)
    runner.writer.flush.assert_called_once()


def test_terminal_timing_and_missing_estimator_loss(capsys):
    runner = make_runner()
    runner.training_wall_seconds = 100
    runner.current_learning_iteration = 10
    runner.phase_updates['teacher'] = 10
    metrics = runner._policy_log_metrics('teacher')
    metrics.update({'Perf/collection_seconds': 2., 'Perf/update_seconds': .5,
                    'Loss/learning_rate': .0003, 'Values/entropy': .01,
                    'Budget/total_interactions': 160, 'Episode/rew_alive': .4})
    runner._print_iteration(metrics, 'teacher', 20)
    output = capsys.readouterr().out
    assert '100.0s' in output and '2.50s' in output
    assert 'Mean episode rew_alive:' in output
    assert 'Estimator force-feature loss' not in output
