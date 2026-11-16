"""Portable launcher tests: fake runtime only; real GPU checks are separate."""
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT/'containers/isaaclab/comparisons.sh'


def launch(tmp_path, *args, **extra):
    env = dict(os.environ, IMAGE=str(tmp_path/'runtime.sif'), REPO=str(ROOT),
               RUN_DIR=str(tmp_path/'output'), CUDA_VISIBLE_DEVICES='1',
               OMNI_KIT_ACCEPT_EULA='YES', **extra)
    return subprocess.run(['bash',str(WRAPPER),*args],env=env,text=True,capture_output=True)


@pytest.mark.parametrize('method,entry', [('br','train_beyond_robustness.py'),('mule','train_mule.py')])
@pytest.mark.parametrize('mode', ['check','smoke','train'])
def test_dry_run_no_writes(tmp_path,method,entry,mode):
    result = launch(tmp_path,method,mode,'--dry-run','--seed','7')
    assert result.returncode == 0, result.stderr
    assert entry in result.stdout and 'CUDA_VISIBLE_DEVICES=1' in result.stdout
    assert '--seed 7' in result.stdout
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('args', [('br','train','--gpu','cuda:0'),
    ('mule','train','--task','other'),('mule','smoke','--num-envs','4096'),
    ('invalid','train'),('br','invalid')])
def test_reject_invalid_routing(tmp_path,args):
    assert launch(tmp_path,*args,'--dry-run').returncode != 0


@pytest.mark.parametrize('method', ['br','mule'])
def test_runtime_smoke_and_checkpoint_mount(tmp_path,method):
    fake = tmp_path/'apptainer'
    fake.write_text('#!/bin/bash\nprintf "CUDA=%s\\n" "$APPTAINERENV_CUDA_VISIBLE_DEVICES"\nprintf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    (tmp_path/'runtime.sif').touch()
    checkpoint_dir = tmp_path/'checkpoints'; checkpoint_dir.mkdir()
    result = launch(tmp_path,method,'smoke','--seed','2',
                    PATH=str(tmp_path)+':'+os.environ['PATH'],CHECKPOINT_DIR=str(checkpoint_dir))
    assert result.returncode == 0, result.stderr
    for text in ('CUDA=1','--nv','--cleanenv','/opt/env/bin/python','--gpu=cuda:0',
                 '--smoke','--num-envs=2','--max-iterations=1',str(checkpoint_dir)+':/checkpoints:ro'):
        assert text in result.stdout


def test_bundle_excludes_runs_and_rejects_external_symlink(tmp_path):
    spec = importlib.util.spec_from_file_location('bundle', ROOT/'containers/isaaclab/bundle_comparisons.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    subprocess.run(['git','init',str(tmp_path)],check=True,capture_output=True)
    for name in ('rsl_rl/model.py','logs/model.pt','legged_gym/.env','resources/robot.urdf'):
        path=tmp_path/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('fixture')
    assert {str(p) for p in module.source_paths(tmp_path)} == {'rsl_rl/model.py','resources/robot.urdf'}
    (tmp_path/'resources/external').symlink_to('/etc/hosts')
    with pytest.raises(ValueError,match='symlink'): list(module.source_paths(tmp_path))
