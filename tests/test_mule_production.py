"""Production registration, contracts and shell dispatch without PhysX."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import pytest
import torch

from legged_gym.envs.go2.go2_mule.mule_rewards import clearance_cost,VERSION
from legged_gym.envs.go2.go2_mule.go2_mule_config import resolve_go2_mule
from legged_gym.scripts.train_mule import parser,resolve_run
from rsl_rl.runners.mule_runner import MULERunner
from rsl_rl.utils import training_checkpoint as tc
from test_mule_learning import SyntheticMULE,config,runner,assert_equal,deterministic_threads

ROOT=Path(__file__).resolve().parents[1]


def test_clearance_direction_translation_and_contract():
    p=torch.zeros(2,4,3); p[0,:,2]=.12; p[1,:,2]=.32
    v=torch.zeros_like(p); v[:,:,0]=2
    ground=torch.zeros(2,4,9)
    cost=clearance_cost(p,v,ground,.10,.02)
    reward=-.01*cost*.02
    assert reward[0]>reward[1] and reward[0]==pytest.approx(0,abs=1e-10)
    p[:,:,2]+=17.; ground+=17.
    torch.testing.assert_close(clearance_cost(p,v,ground,.10,.02),cost,atol=1e-6,rtol=1e-5)
    assert VERSION==3


def test_mule_broadphase_capacity_is_resolved_and_overridable():
    c=resolve_go2_mule(environment_overrides={'env.num_envs':4096})
    assert c.shared.sim.physx.gpu_found_lost_pairs_capacity==2**22
    other=resolve_go2_mule(environment_overrides={'env.num_envs':4096,'sim.physx.gpu_found_lost_pairs_capacity':2**23})
    assert other.shared.sim.physx.gpu_found_lost_pairs_capacity==2**23
    smoke,_,_=resolve_run(parser().parse_args(['--smoke','--num-envs','4096','--max-iterations','1']))
    assert smoke.shared.terrain.num_cols==c.shared.terrain.num_cols
    assert smoke.shared.terrain.max_init_terrain_level==c.shared.terrain.max_init_terrain_level


class ProductionFixture(SyntheticMULE):
    supports_exact_resume=False
    device='cpu'
    def __init__(self,c):
        super().__init__(c.training); self.resolved=self.resolved_config=deepcopy(c)
        self.progress=dict(version=1,domain=dict(progress={'disturbance':.6},reward_ema=.9,reward_history=[.9]),
            command_streak=8,iteration=1,terrain=[3,4,5])
        self.payload='balls'; self.sampling=dict(requested=.6,realized=[.2,0.,.6],timers=[.1,.2,0.])
    def physical_contract(self): return dict(payload=self.payload,reward=self.resolved.shared.mule_reward)
    def export_training_progress(self): return deepcopy(self.progress)
    def curriculum_state_dict(self): return dict(progress=deepcopy(self.progress),sampling=deepcopy(self.sampling))
    def validate_curriculum_state_dict(self,s):
        if s.keys()!=self.curriculum_state_dict().keys(): raise ValueError('bad physical state')
    def load_curriculum_state_dict(self,s): self.progress=deepcopy(s['progress']); self.sampling=deepcopy(s['sampling'])
    def validate_stage_transfer(self,s):
        if self.payload!='synthetic_target': raise ValueError('unsupported liquid')
        assert 'sampling' not in s['transferable_progress']
    def import_training_progress(self,s,transfer=False): self.progress=deepcopy(s)
    def initialize_transferred_payload(self,stage): self.sampling=dict(initial=stage['local_progress'])


def production(phase='nominal_only'):
    c=resolve_go2_mule(environment_overrides={'env.num_envs':3},training_overrides=dict(
        optimization_phase=phase,num_steps_per_env=4,num_learning_epochs=1,auxiliary_epochs=1,num_mini_batches=1,max_iterations=2))
    return c,MULERunner(ProductionFixture(c),c.training,console=False)


@pytest.mark.parametrize('phase',['nominal_only','joint_adaptation'])
def test_compact_checkpoint_resume_weights_and_budgets(tmp_path,phase):
    c,r=production(phase); r.learn(1); path=tmp_path/'source'/'m.pt'; r.save(path,stage_end=True)
    d=torch.load(path,weights_only=False)
    assert not {'history','current','episode_returns','episode_lengths','episode_diagnostics'}&d.keys()
    assert d[tc.KEY]['stage_end'] and d[tc.KEY]['next_iteration']==1
    assert isinstance(d['numpy_rng'][1],list) and isinstance(d[tc.KEY]['rng']['numpy'][1],list)
    assert set(d[tc.KEY]['optimizers'])=={'nominal','adaptive','auxiliary'}
    assert d[tc.KEY]['contract']['optimizer_ownership']['auxiliary']['groups']
    args=parser().parse_args(['--checkpoint',str(path),'--additional-iterations','2','--save-interval','7'])
    resolved,start,_=resolve_run(args); assert start==1 and resolved.training.max_iterations==3
    target=MULERunner(ProductionFixture(resolved),resolved.training,console=False)
    target.load(path)
    assert target.alg.model.phase==phase and target.alg.iteration==1 and target.history.timestamp==-1
    assert_equal(r.alg.model.state_dict(),target.alg.model.state_dict())
    assert_equal(r.alg.schedulers,target.alg.schedulers)
    for key,opt in r.alg.optimizers.items(): assert_equal(opt.state_dict(),target.alg.optimizers[key].state_dict())
    assert_equal(r.env.sampling,target.env.sampling)
    _,weights=production(); weights.load(path,operation='weights_only')
    assert weights.alg.iteration==0 and weights.alg.model.phase=='nominal_only'
    assert all(not o.state for o in weights.alg.optimizers.values())
    for k,v in r.alg.model.state_dict().items():
        if torch.is_tensor(v): torch.testing.assert_close(v,weights.alg.model.state_dict()[k])
    with pytest.raises(ValueError,match='source runs'): target.save(path)
    for extra in (['--seed','999'],['--num-envs','4'],['--optimization-phase','joint_adaptation'],['--max-iterations','0']):
        with pytest.raises(ValueError): resolve_run(parser().parse_args(['--checkpoint',str(path),*extra]))


def test_synthetic_transfer_and_preflight(tmp_path):
    c,r=production('joint_adaptation'); r.learn(1); path=tmp_path/'source'/'m.pt'; r.save(path)
    _,target=production(); before=deepcopy(target.alg.model.state_dict())
    with pytest.raises(ValueError,match='unsupported liquid'): target.load(path,operation='stage_transfer',target_stage='liquid')
    assert_equal(before,target.alg.model.state_dict())
    target.env.payload='synthetic_target'
    target.load(path,operation='stage_transfer',target_stage='synthetic3',stage_local_initial=.1)
    assert_equal(r.alg.model.state_dict(),target.alg.model.state_dict())
    assert target.alg.model.phase=='joint_adaptation' and target.alg.iteration==1
    assert target.env.progress['domain']['progress']['disturbance']==.6
    assert target.env.progress['domain']['reward_ema'] is None and target.env.progress['command_streak']==0
    assert target.env.sampling=={'initial':.1} and target.history.timestamp==-1
    for k,opt in r.alg.optimizers.items(): assert_equal(opt.state_dict(),target.alg.optimizers[k].state_dict())


def test_final_save_legacy_and_incompatible_rewards(tmp_path):
    c,r=production(); r.log_dir=tmp_path/'run'; r.learn(1)
    path=r.log_dir/'model_1.pt'; assert path.exists()
    data=torch.load(path,weights_only=False); data.pop(tc.KEY); old=tmp_path/'old.pt'; torch.save(data,old)
    _,t=production()
    with pytest.raises(ValueError,match='Legacy'): t.load(old)
    with pytest.warns(RuntimeWarning,match='NOT full continuation'): t.load(old,allow_legacy=True)
    data['resolved_configuration'].shared.mule_reward['version']=1
    torch.save(data,old)
    with pytest.raises(ValueError,match='reward contract'):
        resolve_run(parser().parse_args(['--checkpoint',str(old),'--allow-legacy-checkpoint']))


def test_finetune_moments_and_inactive_schedule(tmp_path):
    _,r=production(); r.learn(1); path=tmp_path/'m.pt'; r.save(path)
    _,target=production(); target.load(path,learning_rate=1e-4)
    assert target.alg.schedulers['nominal']['learning_rate']==pytest.approx(1e-4)
    assert target.alg.schedulers['adaptive']['optimizer_steps']==0
    assert target.alg.schedulers['adaptive']['entropy_updates']==0
    for name,opt in r.alg.optimizers.items():
        assert_equal(opt.state_dict()['state'],target.alg.optimizers[name].state_dict()['state'])
    before=deepcopy(target.alg.schedulers['adaptive'])
    target.learn(1)
    assert target.alg.schedulers['adaptive']==before


def test_worktree_archive_fallback(monkeypatch,tmp_path):
    from legged_gym.scripts import train_mule
    c,_=production(); args=parser().parse_args([])
    monkeypatch.setattr(train_mule,'ROOT',tmp_path)
    monkeypatch.setenv('TRAINING_GIT_SHA','host-commit')
    monkeypatch.setenv('TRAINING_GIT_STATUS',' M source.py')
    train_mule.archive(tmp_path,c,args,{})
    manifest=json.loads((tmp_path/'run_manifest.json').read_text())
    assert manifest['git_sha']=='host-commit' and manifest['git_status']==' M source.py'


def test_registry_default_and_explicit_training(monkeypatch):
    from legged_gym.envs import task_registry
    assert 'go2_mule_isaaclab' in task_registry.resolved_factories
    assert 'go2_beyond_robustness_isaaclab' in task_registry.resolved_factories
    c,r=production()
    monkeypatch.setitem(task_registry.task_classes,'go2_mule_isaaclab',lambda resolved,**kw: ProductionFixture(resolved))
    env,actual=task_registry.make_env('go2_mule_isaaclab',args=NS(device='cpu',headless=True),env_cfg=c)
    assert actual is c
    for explicit in (None,c.training):
        runner,cfg=task_registry.make_alg_runner(env,name='go2_mule_isaaclab',train_cfg=explicit,log_root=None)
        assert isinstance(runner,MULERunner) and cfg==c.training


def test_shell_dispatch_and_no_write_dryrun(tmp_path):
    env=dict(os.environ,SIMULATOR='isaaclab',CUDA_VISIBLE_DEVICES='6',MULE_CONDA_ENV='test-env')
    for mode in ('lab','unity'):
        result=subprocess.run(['bash',str(ROOT/f'legged_gym/scripts/go2_mule_{mode}.sh'),
            '--dry-run','--seed','2','--checkpoint','/tmp/a b.pt'],env=env,text=True,capture_output=True,check=True)
        assert 'train_mule.py' in result.stdout and 'CUDA_VISIBLE_DEVICES=6' in result.stdout
        if mode=='unity': assert '/opt/env/bin/python' in result.stdout and 'cuda:0' in result.stdout
    result=subprocess.run(['bash',str(ROOT/'legged_gym/scripts/go2_mule_unity.sh'),'--dry-run','--gpu','cuda:1'],env=env,capture_output=True)
    assert result.returncode!=0
    env.update(IMAGE='/unavailable.sif',REPO=str(ROOT),RUN_DIR=str(tmp_path/'container'),OMNI_KIT_ACCEPT_EULA='YES')
    for mode,task in [('script','go2_mule_unity.sh'),('train','go2_mule_isaaclab'),('smoke','go2_mule_isaaclab')]:
        r=subprocess.run(['bash',str(ROOT/'containers/isaaclab/run.sh'),mode,task,'--dry-run'],env=env,text=True,capture_output=True,check=True)
        assert ('go2_mule_unity.sh' if mode=='script' else 'train_mule.py') in r.stdout
    assert not (tmp_path/'container').exists()
    from legged_gym.scripts.train_mule import main
    main(['--smoke','--dry-run','--log-dir',str(tmp_path/'never')])
    assert not (tmp_path/'never').exists()
