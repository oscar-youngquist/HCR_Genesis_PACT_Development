"""Shell dispatch only: no conda environment, simulator or training required."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize('variant', ['lab', 'unity'])
def test_facet_shell_dispatch(tmp_path, variant):
    script = Path(__file__).resolve().parents[1]/f'legged_gym/scripts/b1z1_facet_{variant}.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    python = tmp_path/'python'
    python.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
        'print(json.dumps(dict(argv=sys.argv[1:], cwd=os.getcwd(), '
        'sim=os.environ.get("SIMULATOR"), visible=os.environ.get("CUDA_VISIBLE_DEVICES"))))\n')
    python.chmod(0o755)
    conda = tmp_path/'conda.sh'
    conda.write_text('conda() { :; }\n')
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
               CUDA_VISIBLE_DEVICES='GPU-scheduler-assignment', FACET_CONDA_SH=str(conda))
    for forwarded in ([], ['--seed', '0', '--num-envs', '8', '--gpu', 'cuda:1'],
                      ['--seed=2', '--resume', '/checkpoints/seed 2/model_60000.pt', '--max-iterations', '70000']):
        output = subprocess.check_output(['bash', str(script), *forwarded], cwd=tmp_path, env=env, text=True)
        data = json.loads(output)
        assert data['argv'] == ['-u', 'train_b1z1_facet.py', '--seed=1', '--gpu=cuda:0', '--headless', *forwarded]
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int)
        resolved, _ = parser.parse_known_args(data['argv'][2:])
        assert resolved.seed == (0 if '--seed' in forwarded else 2 if '--seed=2' in forwarded else 1)
        assert data['sim'] == 'isaaclab_b1z1_pact' and data['cwd'] == str(script.parent)
        assert data['visible'] == (None if variant == 'lab' else 'GPU-scheduler-assignment')


@pytest.fixture
def container_dispatch(tmp_path):
    """Execute real shells/parser; replace only the container and training runtime."""
    root = Path(__file__).resolve().parents[1]
    repo = tmp_path/'checkout with spaces'
    scripts = repo/'legged_gym/scripts'
    scripts.mkdir(parents=True)
    for name in ('b1z1_facet_unity.sh', 'train_b1z1_facet.py', 'b1z1_pact_unity.sh'):
        shutil.copy2(root/'legged_gym/scripts'/name, scripts/name)
    image = tmp_path/'existing image.sif'
    image.write_bytes(b'unchanged image fixture')
    apptainer = tmp_path/'apptainer'
    apptainer.write_text(f'#!{sys.executable}\n'+'''
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
print(json.dumps(dict(container_argv=args, simulator=os.environ['APPTAINERENV_SIMULATOR'],
                     visibility=os.environ['APPTAINERENV_CUDA_VISIBLE_DEVICES'])), flush=True)
# Emulate only the environment exports and working-directory bind. No GPU/container.
env = dict(os.environ)
env.update({k[len('APPTAINERENV_'):]: v for k, v in os.environ.items() if k.startswith('APPTAINERENV_')})
command = args[args.index(str(Path(env['IMAGE']).resolve()))+1:]
raise SystemExit(subprocess.call(command, cwd=Path(env['REPO'])/'legged_gym/scripts', env=env))
''')
    python = tmp_path/'python'
    python.write_text(f'#!{sys.executable}\n'+'''
import argparse, json, os, runpy, sys
sys.argv = sys.argv[2:]  # Consume Python's -u; retain the real script and quoted arguments.
def report(parsed=None):
    print(json.dumps(dict(argv=sys.argv, parsed=vars(parsed) if parsed else None,
                         simulator=os.environ['SIMULATOR'], visibility=os.environ['CUDA_VISIBLE_DEVICES']), default=str))
    raise SystemExit(0)
if sys.argv[0] == 'train_b1z1_facet.py':
    parse = argparse.ArgumentParser.parse_args
    def capture(self, *args, **kwargs):
        report(parse(self, *args, **kwargs))
    argparse.ArgumentParser.parse_args = capture
    runpy.run_path(sys.argv[0], run_name='__main__')  # Stop AFTER real parsing, BEFORE simulator imports.
else:
    report()
''')
    for binary in (apptainer, python):
        binary.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH'], IMAGE=str(image),
               REPO=str(repo), RUN_DIR=str(tmp_path/'output with spaces'),
               CUDA_VISIBLE_DEVICES='GPU-scheduler-assignment', OMNI_KIT_ACCEPT_EULA='YES')
    for key in ('RUNTIME_DIR', 'NVIDIA_ICD_FILE'):
        env.pop(key, None)
    launcher = root/'containers/isaaclab/run.sh'
    for script in (launcher, scripts/'b1z1_facet_unity.sh'):
        subprocess.run(['bash', '-n', str(script)], check=True)

    def run(*args):
        return subprocess.run(['/bin/bash', str(launcher), *args], env=env, text=True, capture_output=True)
    return run, env


@pytest.mark.parametrize('spelling', ['hyphen', 'underscore'])
@pytest.mark.parametrize('equals', [False, True])
def test_container_to_facet_parser(container_dispatch, spelling, equals):
    run, env = container_dispatch
    keys = ['--num-envs', '--max-iterations'] if spelling == 'hyphen' else ['--num_envs', '--max_iterations']
    args = [f'{k}={v}' for k, v in zip(keys, (4096, 70000))] if equals else [keys[0], '4096', keys[1], '70000']
    resume = '/checkpoints/seed 0/model 60000.pt'
    result = run('script', 'b1z1_facet_unity.sh', '--seed=0', '--resume', resume, *args)
    assert result.returncode == 0, result.stderr
    container, training = map(json.loads, result.stdout.splitlines())
    assert container['simulator'] == training['simulator'] == 'isaaclab_b1z1_pact'
    assert container['visibility'] == training['visibility'] == env['CUDA_VISIBLE_DEVICES']
    parsed = training['parsed']
    assert (parsed['seed'], parsed['num_envs'], parsed['max_iterations'], parsed['resume']) == (0, 4096, 70000, resume)
    assert parsed['gpu'] == 'cuda:0' and parsed['headless']
    assert parsed['student_start_iteration'] is None and parsed['adaptation_iterations'] is None
    argv = container['container_argv']
    assert argv[argv.index(env['IMAGE'])+1:] == ['/bin/bash', 'b1z1_facet_unity.sh', '--seed=0', '--resume', resume, *args]
    for mount in (f"{env['REPO']}:/workspace/repo", f"{env['RUN_DIR']}:/output",
                  f"{env['RUN_DIR']}/logs:/workspace/repo/logs",
                  f"{env['RUN_DIR']}/runtime/kit-cache:/opt/env/lib/python3.11/site-packages/isaacsim/kit/cache"):
        assert mount in argv
    assert Path(env['IMAGE']).read_bytes() == b'unchanged image fixture'


def test_container_defaults_and_existing_script_dispatch(container_dispatch):
    run, _ = container_dispatch
    result = run('script', 'b1z1_facet_unity.sh')
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout.splitlines()[-1])['parsed']
    assert parsed['seed'] == 1 and parsed['num_envs'] is None and parsed['max_iterations'] is None
    result = run('script', 'b1z1_pact_unity.sh', '--seed=0', '--num_envs=4096')
    assert result.returncode == 0, result.stderr
    training = json.loads(result.stdout.splitlines()[-1])
    assert training['argv'] == ['train.py', '--task=b1z1_pact', '--seed=1', '--gpu=cuda:0', '--headless', '--seed=0', '--num_envs=4096']
    assert training['simulator'] == 'isaaclab_b1z1_pact'


@pytest.mark.parametrize('args,diagnostic', [
    (['script', 'unknown_unity.sh'], 'Unsupported Unity launcher'),
    (['script', 'b1z1_facet_unity.sh', '--gpu=cuda:1'], 'Task/device are managed'),
    (['script', 'b1z1_facet_unity.sh', '--gpu', 'cuda:1'], 'Task/device are managed'),
    (['script', 'b1z1_facet_unity.sh', '--cpu'], 'Task/device are managed'),
    (['script', 'b1z1_facet_unity.sh', '--task=b1z1_pact'], 'Task is managed'),
    (['train', 'b1z1_facet'], 'Unsupported IsaacLab task'),
])
def test_container_validation_retained(container_dispatch, args, diagnostic):
    run, _ = container_dispatch
    result = run(*args)
    assert result.returncode != 0 and diagnostic in result.stderr
    assert not result.stdout  # Rejected before Apptainer or Python runs.
