import io
import subprocess
from unittest.mock import Mock

import pytest

from liquid_collection.progress import episode_progress, format_progress, wait_with_progress
from liquid_collection.storage import atomic_json


def test_per_environment_quota_does_not_finish_early():
    job = dict(num_envs=2, episodes_per_env=5)
    progress = episode_progress([20, 1], job)
    assert progress['completed'] == 6
    assert progress['target'] == 10
    assert progress['environments_complete'] == 1
    assert '60%' in format_progress('round 1/4', progress)
    assert episode_progress([20, 5], job)['completed'] == 10


def test_total_episode_budget():
    progress = episode_progress([4, 8], dict(num_envs=2, episodes=10))
    assert progress['completed'] == progress['target'] == 10
    assert progress['environments_complete'] is None


@pytest.mark.parametrize('returncode,phase', [(0, 'complete'), (2, 'incomplete'), (1, 'finalizing')])
def test_wait_displays_initial_and_final_counts(tmp_path, returncode, phase):
    job = dict(num_envs=2, episodes_per_env=5)
    proc = Mock(args=['worker'])

    def finish(timeout):
        atomic_json(tmp_path/'progress.json', episode_progress([5, 3], job, phase))
        return returncode

    proc.wait.side_effect = finish
    stream = io.StringIO()
    assert wait_with_progress(proc, tmp_path, job, 'go1 round 2/4', 10, stream) == returncode
    output = stream.getvalue()
    assert 'initializing' in output and '8/10 episodes' in output and phase in output
    assert '\r' not in output


def test_polling_preserves_timeout_and_suppresses_duplicate_lines(tmp_path, monkeypatch):
    clock = iter([0., 0., 1., 2.])
    monkeypatch.setattr('liquid_collection.progress.time.monotonic', lambda: next(clock))
    proc = Mock(args=['worker'])
    proc.wait.side_effect = subprocess.TimeoutExpired(['worker'], 1)
    stream = io.StringIO()
    with pytest.raises(subprocess.TimeoutExpired):
        wait_with_progress(proc, tmp_path, dict(num_envs=1, episodes=5), 'round 1/2', 2., stream)
    assert proc.wait.call_count == 2
    assert stream.getvalue().count('initializing') == 1


def test_terminal_progress_ends_line_on_interrupt(tmp_path):
    proc = Mock(args=['worker'])
    proc.wait.side_effect = KeyboardInterrupt
    stream = io.StringIO()
    stream.isatty = lambda: True
    with pytest.raises(KeyboardInterrupt):
        wait_with_progress(proc, tmp_path, dict(num_envs=1, episodes=5), 'round 1/2', 2., stream)
    assert stream.getvalue().startswith('\r')
    assert stream.getvalue().endswith('\n')
