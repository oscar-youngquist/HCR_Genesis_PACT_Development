"""Wait/checkpoint/launcher contracts without running Torch, training, or Isaac Lab."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'legged_gym/scripts'
spec = importlib.util.spec_from_file_location('pact_waiter', SCRIPTS / 'eval_b1z1_pact_after_training.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def checkpoint(run, iteration):
    path = run / f'model_{iteration}.pt'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('archive/data.pkl', b'stub checkpoint, never unpickled')
    return path


def fake_process(tmp_path):
    return dict(pid=17, ticks=40, started=100., argv=['python', 'train.py', '--task=b1z1_pact', '--gpu=cuda:0'],
        environment={'PATH': os.environ['PATH'], 'CONDA_PREFIX': sys.prefix}, executable=Path(sys.executable))


def config(run, resume=False):
    (run / 'b1z1_pact_config.py').write_text(
        'raise AssertionError("must not import config")\n'
        'class B1Z1PACTCfgPPO:\n    class runner:\n        max_iterations = 70\n'
        f'        resume = {resume}\n')


def test_expected_iteration_uses_archived_config_cli_override_and_resume(tmp_path):
    config(tmp_path)
    assert w.expected_checkpoint(tmp_path, []) == 70
    assert w.expected_checkpoint(tmp_path, ['--max_iterations=80']) == 80
    assert w.expected_checkpoint(tmp_path, ['--max_iterations', '90']) == 90
    with pytest.raises(ValueError, match='Resumed'):
        w.expected_checkpoint(tmp_path, ['--resume'])
    config(tmp_path, resume=True)
    with pytest.raises(ValueError, match='Resumed'):
        w.expected_checkpoint(tmp_path, [])
    assert w.expected_checkpoint(tmp_path, ['--resume'], 150) == 150


def test_selects_numeric_latest_and_does_not_fall_back_after_corruption(tmp_path):
    with pytest.raises(ValueError, match='No model'):
        w.latest_checkpoint(tmp_path)
    for i in (9, 100, 20): checkpoint(tmp_path, i)
    (tmp_path / 'model_999.pt.partial').write_bytes(b'ignored')
    number, path = w.latest_checkpoint(tmp_path)
    assert number == 100
    w.validate_checkpoint(path)
    path.write_bytes(b'incomplete final save')
    with pytest.raises(ValueError, match='incomplete'):
        w.validate_checkpoint(w.latest_checkpoint(tmp_path)[1])


@pytest.mark.parametrize('end', [None, (41, 'R'), (40, 'Z'), (40, 'X')])
def test_wait_exit_zombie_pid_reuse_without_signalling(monkeypatch, end):
    states = iter([(40, 'R'), (40, 'S'), end]); sleeps = []
    monkeypatch.setattr(w, 'process_state', lambda *args: next(states))
    monkeypatch.setattr(w.time, 'sleep', sleeps.append)
    w.wait_for_exit(dict(pid=17, ticks=40), .25)
    assert sleeps == [.25, .25]


def test_run_selection_uses_event_pid_and_start_not_folder_recency(tmp_path):
    process = fake_process(tmp_path)
    old = tmp_path / 'logs/b1z1/b1z1_pact_lab/old'
    right = old.parent / 'current'; wrong = old.parent / 'newer'
    for path in (old, right, wrong): path.mkdir(parents=True)
    (old / 'events.out.tfevents.98.host.17.0').touch()
    (right / 'events.out.tfevents.102.host.with.dots.17.0').touch()
    (wrong / 'events.out.tfevents.200.host.18.0').touch()
    assert w.find_run(process, repo=tmp_path) == right
    with pytest.raises(ValueError, match='does not belong'):
        w.find_run(process, wrong, tmp_path)
    (wrong / 'events.out.tfevents.200.host.17.0').touch()
    with pytest.raises(ValueError, match='uniquely'):
        w.find_run(process, repo=tmp_path)
    assert w.find_run(process, right, tmp_path) == right


def test_environment_gpu_and_separate_output_locations(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    monkeypatch.delenv('EVAL_GPU', raising=False)
    process = fake_process(tmp_path); run = tmp_path / 'a run'
    args, env = w.evaluation_command(process, run, 70, None, None, tmp_path)
    assert args[2:4] == [str(run), '70']
    root = tmp_path / 'evaluation/pilot_v02/a run/model_70'
    assert Path(args[4]) == root / 'exp1'
    assert Path(env['EXP2_OUTPUT_BASE']) == root / 'exp2'
    assert env['EVAL_GPU'] == 'cuda:0' and 'CUDA_VISIBLE_DEVICES' not in env
    assert env['PATH'].split(os.pathsep)[0] == str(Path(sys.prefix) / 'bin')
    monkeypatch.setenv('EVAL_GPU', 'cuda:1')
    assert w.evaluation_command(process, run, 70, None, None, tmp_path)[1]['EVAL_GPU'] == 'cuda:1'
    assert w.evaluation_command(process, run, 70, None, 'cuda:2', tmp_path)[1]['EVAL_GPU'] == 'cuda:2'


def setup_main(tmp_path, monkeypatch):
    run = tmp_path / 'training run'; run.mkdir(); config(run); checkpoint(run, 9)
    monkeypatch.setattr(w, 'find_process', lambda *args: fake_process(tmp_path))
    monkeypatch.setattr(w, 'find_run', lambda *args: run)
    monkeypatch.setattr(w.time, 'sleep', lambda *args: None)
    monkeypatch.setattr(w, 'REPO', tmp_path)
    monkeypatch.setattr(w.os, 'chdir', lambda *args: None)
    return run


def test_dry_run_does_not_wait_or_launch(tmp_path, monkeypatch):
    setup_main(tmp_path, monkeypatch)
    monkeypatch.setattr(w, 'wait_for_exit', lambda *args: pytest.fail('dry run waited'))
    monkeypatch.setattr(w.os, 'execve', lambda *args: pytest.fail('dry run launched'))
    assert w.main(['--dry-run']) == 0


def test_final_checkpoint_is_chosen_after_wait(tmp_path, monkeypatch):
    run = setup_main(tmp_path, monkeypatch); calls = []
    monkeypatch.setattr(w, 'wait_for_exit', lambda *args: checkpoint(run, 70))
    monkeypatch.setattr(w.os, 'execve', lambda *args: calls.append(args))
    w.main(['--output-dir', str(tmp_path / 'results'), '--gpu', 'cuda:1'])
    assert len(calls) == 1 and calls[0][1][2:4] == [str(run), '70']
    assert calls[0][2]['EVAL_GPU'] == 'cuda:1'


def test_early_exit_stops_evaluation_unless_explicitly_allowed(tmp_path, monkeypatch):
    setup_main(tmp_path, monkeypatch); calls = []
    monkeypatch.setattr(w, 'wait_for_exit', lambda *args: None)
    monkeypatch.setattr(w.os, 'execve', lambda *args: calls.append(args))
    with pytest.raises(ValueError, match='evaluation NOT started'):
        w.main([])
    assert not calls
    w.main(['--allow-early-exit']); assert calls[0][1][3] == '9'


def test_existing_output_refused(tmp_path, monkeypatch):
    run = setup_main(tmp_path, monkeypatch); checkpoint(run, 70)
    output = tmp_path / 'results/exp2'; output.mkdir(parents=True); (output / 'old').touch()
    monkeypatch.setattr(w, 'wait_for_exit', lambda *args: None)
    monkeypatch.setattr(w.os, 'execve', lambda *args: pytest.fail('overwrote output'))
    with pytest.raises(ValueError, match='not empty'):
        w.main(['--output-dir', str(output.parent)])


def test_existing_suite_runs_all_18_collections_with_pinned_checkpoint(tmp_path):
    scripts = tmp_path / 'legged_gym/scripts'; scripts.mkdir(parents=True)
    shutil.copyfile(SCRIPTS / 'eval_b1z1_pact_three_seeds.sh', scripts / 'eval_b1z1_pact_three_seeds.sh')
    for experiment in (1, 2):
        (scripts / f'eval_b1z1_exp{experiment}.sh').write_text(
            f'printf "exp{experiment}\\0" >> "$CAPTURE"\nprintf "%s\\0" "$@" >> "$CAPTURE"\nprintf "\\n" >> "$CAPTURE"\n')
    run = tmp_path / 'run folder'; run.mkdir(); checkpoint(run, 70)
    command, env = w.evaluation_command(fake_process(tmp_path), run, 70, tmp_path / 'output folder', 'cuda:1', tmp_path)
    capture = tmp_path / 'calls'; env['CAPTURE'] = str(capture)
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    rows = capture.read_bytes().splitlines(); assert len(rows) == 18
    for i, row in enumerate(rows):
        argv = row.decode().rstrip('\0').split('\0')
        assert argv[0] == ('exp1' if i < 9 else 'exp2')
        for flag, expected in (('--ckpt', '70'), ('--run', str(run)), ('--gpu', 'cuda:1'), ('--seed', str(i % 3))):
            assert argv[argv.index(flag) + 1] == expected
    subprocess.run(['bash', '-n', str(SCRIPTS / 'eval_b1z1_pact_after_training.sh')], check=True)


def test_process_discovery_and_identity_use_only_this_repos_pact_trainer(tmp_path):
    repo = tmp_path / 'repo'; scripts = repo / 'legged_gym/scripts'; scripts.mkdir(parents=True)
    (scripts / 'train.py').touch()
    proc = tmp_path / 'proc'; proc.mkdir(); (proc / 'stat').write_text('btime 100\n')
    def add(pid, task):
        root = proc / str(pid); root.mkdir()
        (root / 'stat').write_text(f'{pid} (python (worker)) R ' + '0 ' * 18 + '40')
        (root / 'cmdline').write_bytes(('python\0-u\0train.py\0--task\0' + task + '\0').encode())
        (root / 'environ').write_bytes(b'PATH=/bin\0UNRELATED_PRIVATE_FIELD=not-forwarded\0')
        (root / 'cwd').symlink_to(scripts)
        (root / 'exe').symlink_to(sys.executable)
    add(17, 'b1z1_pact'); add(18, 'b1z1_pact_ab10_actor')
    actual = w.find_process(repo=repo, proc=proc)
    assert actual['pid'] == 17 and actual['ticks'] == 40
    assert actual['environment'] == {'PATH': '/bin'}
    assert actual['started'] == 100 + 40 / os.sysconf('SC_CLK_TCK')
    add(19, 'b1z1_pact')
    with pytest.raises(ValueError, match='Specify --pid'):
        w.find_process(repo=repo, proc=proc)
    assert w.find_process(19, repo, proc)['pid'] == 19
