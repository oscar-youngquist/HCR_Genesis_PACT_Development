"""Shell dispatch only: no conda environment, simulator or training required."""
import argparse
import json
import os
from pathlib import Path
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
