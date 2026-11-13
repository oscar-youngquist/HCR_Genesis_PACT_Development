"""Learning continuation/transfer without claiming simulator or liquid support."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS
import subprocess
import os

import pytest
import torch

from rsl_rl.utils import training_checkpoint as tc
from rsl_rl.utils import hard_pact_checkpoint as hc
from rsl_rl.algorithms.hard_pact_qp_curriculum import QPCurriculum
from rsl_rl.runners.pact_runner import OnPolicyRunnerPACT
from rsl_rl.runners.pact_pos_runner import OnPolicyRunnerPACTPos
from test_hard_pact_auxiliary import make_algorithm
from test_hard_pact_pos_auxiliary import _small_algorithm
from test_beyond_robustness_runner import make as make_br, SyntheticBR
from test_beyond_robustness import cpu_threads
from test_beyond_robustness_stage8 import assert_equal


class TransferEnv:
    """Synthetic target implementing the explicit portable curriculum contract."""
    num_envs=2
    device='cpu'
    supports_exact_resume=False

    def __init__(self, payload='wrench'):
        self.payload=payload
        self.cfg={'control':{'dt':.020,'physics_dt':.005},'observations':{'order':['q','v'],'units':['rad','rad/s']}}
        self.progress={'version':1,'domain':{'progress':{'disturbance':.8},'reward_ema':.9,'reward_history':[.9]},
                       'commands':{'lin_vel_x':[-2.,2.],'streak':40},'terrain':[4,5]}
        self.delay=torch.ones(2,4)
        self.reset_count=0

    def training_environment_contract(self): return dict(payload=self.payload,config=self.cfg)
    def export_training_progress(self): return deepcopy(self.progress)
    def validate_training_progress(self,p):
        if p['version']!=1: raise ValueError('Portable progress version')
    def import_training_progress(self,p,transfer=False):
        self.validate_training_progress(p)
        self.progress=tc.clear_gate_evidence(p) if transfer else deepcopy(p)
    def validate_stage_transfer(self,state):
        if self.payload!='synthetic_liquid': raise ValueError('No liquid adapter')
        assert 'sampling' not in state['transferable_progress']
        self.validate_training_progress(state['transferable_progress'])
    def initialize_transferred_payload(self,stage): self.payload_stage=stage
    def reset_training_episodes(self): self.delay.zero_(); self.reset_count+=1
    def command_curriculum_state_dict(self): return self.progress['commands']
    def domain_rand_curriculum_state_dict(self): return self.progress['domain']


def hard(pos=False,payload='wrench'):
    r=(OnPolicyRunnerPACTPos if pos else OnPolicyRunnerPACT).__new__(OnPolicyRunnerPACTPos if pos else OnPolicyRunnerPACT)
    r.is_hard_pact=not pos; r.is_hard_pact_pos=pos
    r.alg=_small_algorithm() if pos else make_algorithm()
    # Real networks and all real optimizer owners; minimal synthetic rollout boundary.
    r.alg.storage=NS(step=0,delay=torch.ones(2,3),clear=lambda:None)
    r.alg.transition=NS(actions=None,clear=lambda:None)
    r.env=TransferEnv(payload)
    r.train_cfg={'policy':{'history':20},'algorithm':{'bard_inverse_enabled':False},
                 'runner':{'max_iterations':100,'save_interval':10}}
    r.cfg=r.train_cfg['runner']; r.device='cpu'; r.log_dir=None
    r.current_learning_iteration=17; r.tot_timesteps=800; r.tot_time=4.
    r.hard_pact_features=NS(actions='position' if pos else 'coupled',inverse_loss=False)
    r.deployment_contract={'version':99,'actions':'position' if pos else 'coupled',
                           'observation_units':['rad','rad/s'],'normalization':.1}
    r._set_hard_pact_qp_iteration=lambda i:setattr(r.alg,'_qp_training_iteration',i)
    r.alg.use_boot=True
    if not pos:
        r.qp_curriculum=QPCurriculum(r.alg.qp_config)
        r.qp_curriculum.progress=.6; r.qp_curriculum.ema=.8
        r.qp_curriculum.history.extend([.8,.9]); r.qp_curriculum.pre_qp_reference=.8
        r.qp_curriculum.baseline_frozen=True; r.qp_curriculum.recovery_counter=32
    return r


def prime(r):
    # Initialize all optimizer moments without introducing an environment/QP dependency.
    for opt in tc.optimizers(r).values():
        opt.zero_grad(set_to_none=True)
        for g in opt.param_groups:
            for p in g['params']: p.grad=torch.ones_like(p)*.002
        opt.step(); opt.zero_grad(set_to_none=True)


@pytest.mark.parametrize('pos',[False,True])
def test_hard_transfer_preserves_networks_moments_progress_and_resets(tmp_path,pos):
    r=hard(pos); prime(r)
    path=tmp_path/'source'/'model.pt'; r.save(path,stage_end=True)
    old=torch.load(path,weights_only=False)[tc.KEY]
    target=hard(pos,'synthetic_liquid'); target.log_dir=tmp_path/'target'
    target.load(path,operation='stage_transfer',target_stage='liquid3',stage_local_initial=.1)
    for n,m in tc.modules(target).items(): assert_equal(old['models'][n],m.state_dict())
    for n,opt in tc.optimizers(target).items(): assert_equal(old['optimizers'][n],opt.state_dict())
    assert target.current_learning_iteration==17 and target.tot_timesteps==800
    assert target.alg.use_boot and not target.hard_pact_features.inverse_loss
    assert target.env.delay.eq(0).all() and target.alg.storage.delay.eq(0).all()
    assert target.env.progress['domain']['progress']['disturbance']==.8
    assert target.env.progress['domain']['reward_ema'] is None
    assert target.env.progress['commands']['streak']==0 and target.env.progress['terrain']==[4,5]
    assert target.training_stage==dict(name='liquid3',origin_iteration=17,local_progress=.1)
    if not pos:
        assert target.qp_curriculum.progress==.6
        assert target.qp_curriculum.origin==r.qp_curriculum.origin
        assert target.qp_curriculum.ema is None and not target.qp_curriculum.baseline_frozen
        assert target.qp_curriculum.recovery_counter==0
    report=json.loads((target.log_dir/'training_transfer_report.json').read_text())
    assert report['parent']['sha256'] and 'payload sampling' in report['reset']
    assert report['source_configuration']==report['target_configuration']
    target.save(tmp_path/'target'/'final.pt',stage_end=True)
    assert torch.load(tmp_path/'target'/'final.pt',weights_only=False)[tc.KEY]['stage_end']


@pytest.mark.parametrize('pos',[False,True])
def test_hard_same_env_next_optimizer_update_and_lr_override(tmp_path,pos):
    r=hard(pos); prime(r); r.save(tmp_path/'source.pt')
    target=hard(pos)
    target.alg.storage.hard_pact_fields=None  # lazily allocated on the first QP rollout
    target.load(tmp_path/'source.pt')
    prime(r); prime(target)
    for n,m in tc.modules(r).items(): assert_equal(m.state_dict(),tc.modules(target)[n].state_dict())
    for n,o in tc.optimizers(r).items(): assert_equal(o.state_dict(),tc.optimizers(target)[n].state_dict())
    tuned=hard(pos); tuned.load(tmp_path/'source.pt',learning_rate=.00005)
    saved=torch.load(tmp_path/'source.pt',weights_only=False)[tc.KEY]
    factor=.00005/saved['adaptive']['learning_rate']
    for name,o in tc.optimizers(tuned).items():
        assert_equal(o.state_dict()['state'],saved['optimizers'][name]['state'])
        for g,old in zip(o.param_groups,saved['optimizers'][name]['param_groups']): assert g['lr']==pytest.approx(old['lr']*factor)


@pytest.mark.parametrize('corruption',['semantic','ownership','features','timing','moment','progress','rng'])
def test_incompatibility_fails_before_mutation(tmp_path,corruption):
    r=hard(); prime(r); r.save(tmp_path/'good.pt')
    data=torch.load(tmp_path/'good.pt',weights_only=False); s=data[tc.KEY]
    if corruption=='semantic': s['contract']['semantics']['normalization']=10.
    if corruption=='ownership': s['contract']['optimizer_ownership']['enc_optimizer']['groups'][0][0]['name']='actor_critic.wrong_same_shape'
    if corruption=='features': s['contract']['features']['inverse_loss']=True
    if corruption=='timing': s['contract']['environment']['config']['control']['physics_dt']=.01
    if corruption=='moment':
        opt=s['optimizers']['act_optimizer']; index=next(iter(opt['state']))
        opt['state'][index]['exp_avg']=torch.zeros(999)
    if corruption=='progress': s['transferable_progress']['version']=9
    if corruption=='rng': s['rng']['torch']=torch.zeros(3,dtype=torch.uint8)
    torch.save(data,tmp_path/'bad.pt')
    target=hard(); before=deepcopy(target.alg.actor_critic.state_dict()); rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError): target.load(tmp_path/'bad.pt')
    assert_equal(before,target.alg.actor_critic.state_dict())
    assert torch.equal(rng,torch.get_rng_state()) and target.env.reset_count==0


def test_runtime_overrides_legacy_boundary_and_source_protection(tmp_path):
    r=hard(); prime(r); r.save(tmp_path/'model.pt')
    target=hard(); target.train_cfg['runner'].update(max_iterations=200,save_interval=5)
    target.load(tmp_path/'model.pt')
    with pytest.raises(ValueError,match='source runs'): target.save(tmp_path/'model.pt')
    target._checkpoint_update_in_progress=True
    with pytest.raises(RuntimeError,match='completed'): target.save(tmp_path/'target'/'partial.pt')
    data=torch.load(tmp_path/'model.pt',weights_only=False); data.pop(tc.KEY); torch.save(data,tmp_path/'legacy.pt')
    new=hard()
    with pytest.raises(ValueError,match='Legacy'): new.load(tmp_path/'legacy.pt')
    with pytest.warns(RuntimeWarning,match='NOT full continuation'): new.load(tmp_path/'legacy.pt',allow_legacy=True)
    assert not new.alg.act_optimizer.optimizer.state  # never guess legacy owner IDs
    assert new.last_transfer_report['operation']=='weights_only'
    assert new.last_transfer_report['unavailable']


class PortableBR(SyntheticBR):
    def export_training_progress(self): return dict(progress=self.progress,reward_history=[.9],command_streak=30)
    def validate_stage_transfer(self,s):
        if not getattr(self,'is_target',False): raise ValueError('Target adapter unavailable')
    def import_training_progress(self,s,transfer=False):
        self.progress=s['progress']; self.imported=s
    def initialize_transferred_payload(self,stage): self.local_stage=stage


@pytest.mark.parametrize('phase',['teacher_student','student_reinforcement'])
def test_br_transfer_retains_phase_optimizer_and_final_checkpoint(tmp_path,phase):
    source=make_br(phase=phase,log_dir=tmp_path/'source'); source.console=False
    source.env.__class__=PortableBR
    source.learn(1,save_interval=10)
    path=tmp_path/'source'/'model_1.pt'; assert path.exists()
    s=torch.load(path,weights_only=False)[tc.KEY]
    target=make_br(phase=phase,log_dir=tmp_path/'target'); target.console=False
    target.env.__class__=PortableBR; target.env.is_target=True
    target.load(path,operation='stage_transfer',target_stage='synthetic_liquid')
    assert target.alg.training_phase==phase and target.alg.iteration==1
    assert target.env.progress==source.env.progress
    assert target.env.imported['reward_history']==[] and target.env.imported['command_streak']==0
    assert target.alg.history.frames.eq(0).all() and target.episode_lengths.eq(0).all()
    assert_equal(s['models']['actor_critic'],target.alg.actor_critic.state_dict())
    for n,o in tc.optimizers(target).items(): assert_equal(s['optimizers'][n],o.state_dict())
    target.learn(1,save_interval=10)
    assert (tmp_path/'target'/'model_2.pt').exists()


def test_br_exact_continuation_and_extended_budget(tmp_path):
    source=make_br(); source.console=False; source.learn(1); source.save(tmp_path/'source.pt')
    source.learn(1); expected=deepcopy(source.alg.actor_critic.state_dict())
    target=make_br(max_iterations=30000,save_interval=55); target.console=False
    target.load(tmp_path/'source.pt',resume_mode='exact'); target.learn(1)
    assert_equal(expected,target.alg.actor_critic.state_dict())
    assert target.interactions==source.interactions


def test_registered_coverage_and_launcher_forwarding():
    from legged_gym.envs import task_registry
    from rsl_rl.hard_pact_ablations import RSS_HARD_PACT_ABLATIONS
    for name in task_registry.task_classes:
        if name.startswith('go2_hard_pact'):
            cfg=task_registry.train_cfgs[name]
            assert cfg.runner_class_name in ('PACTRunner','PACTPosRunner')
    assert len(RSS_HARD_PACT_ABLATIONS)==10
    root=Path(__file__).resolve().parents[1]
    for where in ('lab','unity'):
        command=['bash',str(root/'legged_gym/scripts'/f'go2_hard_pact_rss_full_{where}.sh'),
                 '--dry-run','--state-checkpoint','/tmp/source.pt','--load-operation','stage_transfer',
                 '--target-stage','liquid3','--additional-iterations','2']
        out=subprocess.check_output(command,cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':'3'},text=True)
        assert '--state-checkpoint' in out and '--target-stage' in out and 'CUDA_VISIBLE_DEVICES=3' in out
        if where=='unity': assert 'cuda:0' in out and 'skip_conda=1' in out
        command=['bash',str(root/'legged_gym/scripts'/f'go2_beyond_robustness_{where}.sh'),
                 '--dry-run','--checkpoint','/tmp/source.pt','--load-operation','resume',
                 '--additional-iterations','2','--finetune-learning-rate','0.0001']
        out=subprocess.check_output(command,cwd=root,env={**os.environ,'CUDA_VISIBLE_DEVICES':'3'},text=True)
        assert '--load-operation resume' in out and '--finetune-learning-rate' in out
        if where=='unity': assert 'cuda:0' in out and 'CUDA_VISIBLE_DEVICES=3' in out


def test_weights_only_leaves_learning_fresh_and_unknown_transfer_rejected(tmp_path):
    source=hard(); prime(source); source.save(tmp_path/'source.pt')
    target=hard(); target.current_learning_iteration=0
    target.env.payload='different_terrain_for_play'
    target.train_cfg['runner']['num_envs']=10
    target.load(tmp_path/'source.pt',operation='weights_only')
    assert not target.alg.act_optimizer.optimizer.state and target.current_learning_iteration==0
    assert_equal(source.alg.actor_critic.state_dict(),target.alg.actor_critic.state_dict())
    before=deepcopy(target.alg.actor_critic.state_dict())
    target.train_cfg=deepcopy(source.train_cfg)
    with pytest.raises(ValueError,match='No liquid adapter'):
        target.load(tmp_path/'source.pt',operation='stage_transfer',target_stage='liquid')
    assert_equal(before,target.alg.actor_critic.state_dict())


@pytest.mark.parametrize('variant', ['rss_'+v for v in ('pos_ppo','coupled_ppo','guided_no_qp',
    'qp_no_post','qp_imitation','full','no_fk','no_ff','no_channels','no_qp_wrench')])
def test_rss_feature_identity_preserved_and_cross_ablation_rejected(tmp_path,variant):
    from rsl_rl.hard_pact_ablations import resolve_hard_pact_features
    r=hard(); r.hard_pact_features=resolve_hard_pact_features(variant)
    r.alg.hard_pact_features=r.hard_pact_features
    r.save(tmp_path/'source.pt')
    new=hard(); new.hard_pact_features=r.hard_pact_features; new.alg.hard_pact_features=new.hard_pact_features
    new.load(tmp_path/'source.pt')
    assert new.alg.hard_pact_features==r.alg.hard_pact_features
    assert not new.alg.hard_pact_features.inverse_loss and not new.alg.hard_pact_features.rollout_loss
    mismatch=hard()
    with pytest.raises(ValueError,match='features'): mismatch.load(tmp_path/'source.pt')
