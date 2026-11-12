"""Real shell dispatch with fake runtimes; never submits a job or starts evaluation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from test_b1z1_facet_launchers import container_dispatch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'legged_gym/scripts'
TASKS = ['b1z1_pact_ab13_workspace', 'b1z1_pact_ab14_no_ee_recruitment',
         'b1z1_pact_ab15_no_channel_physics']


@pytest.mark.parametrize('task', TASKS + ['b1z1_pact_ab5_coupled_concat'])
def test_unity_container_allowlist_and_forwarding(container_dispatch, task):
    run, env = container_dispatch
    name = task + '_unity.sh'
    shutil.copy2(SCRIPTS / name, Path(env['REPO']) / 'legged_gym/scripts' / name)
    subprocess.run(['bash', '-n', str(SCRIPTS / name)], check=True)
    result = run('script', name, '--seed=0', '--num_envs=2', '--max_iterations=1')
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout.splitlines()[-1])
    assert record['simulator'] == 'isaaclab_b1z1_pact'
    assert record['visibility'] == env['CUDA_VISIBLE_DEVICES']
    assert record['argv'] == ['train.py', '--task=' + task, '--seed=1', '--gpu=cuda:0',
                              '--headless', '--seed=0', '--num_envs=2', '--max_iterations=1']


@pytest.mark.parametrize('task', TASKS + ['b1z1_pact_ab5_coupled_concat'])
def test_local_active_environment_and_collection_settings(tmp_path, task):
    binary = tmp_path / 'python'
    binary.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
        'print(json.dumps(dict(args=sys.argv[1:], gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))))\n')
    binary.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'], CUDA_VISIBLE_DEVICES='1')
    local = SCRIPTS / (task + '_lab.sh')
    subprocess.run(['bash', '-n', str(local)], check=True)
    result = json.loads(subprocess.check_output(['bash', str(local), '--seed=0'], env=env, text=True))
    assert result['gpu'] == '1' and result['args'][-1] == '--seed=0'
    assert '--task=' + task in result['args']
    # Copy the real loop; replace just evaluator entry points to inspect argv.
    for name in ('eval_' + task + '_three_seeds.sh', 'eval_b1z1_pact_new_ablations_three_seeds.sh'):
        shutil.copy2(SCRIPTS / name, tmp_path / name)
        subprocess.run(['bash', '-n', str(tmp_path / name)], check=True)
    for exp in (1, 2):
        (tmp_path / f'eval_b1z1_exp{exp}.sh').write_text('exec python "$@"\n')
    run = tmp_path / 'checkpoint with spaces'
    run.mkdir()
    (run / 'model_100.pt').touch()
    output = subprocess.check_output(['bash', str(tmp_path / ('eval_' + task + '_three_seeds.sh')),
                                     str(run), '100', str(tmp_path / 'out with spaces')], env=env, text=True)
    records = [json.loads(line)['args'] for line in output.splitlines() if line.startswith('{')]
    assert len(records) == 18
    for index, args in enumerate(records):
        def values(key):
            return args[args.index(key) + 1]
        assert values('--task') == task and values('--run') == str(run)
        assert values('--seed') == str(index % 3)
        assert values('--num-envs') == '25'
        assert values('--episodes') == ('100' if index < 9 else '50')
        if index < 9:
            assert values('--condition') == ['nominal', 'ee-disturbance', 'combined-disturbance'][index // 3]
            assert values('--load-range') == 'in'
        else:
            assert values('--profile') == ['pushing', 'pulling', 'carrying'][(index - 9) // 3]
            assert values('--loading') == 'loaded'
