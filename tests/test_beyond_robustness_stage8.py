"""Stage/phase independence, production dispatch, safe continuation and launchers."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import subprocess
import os
import sys
import pytest
import torch

from rsl_rl.beyond_robustness_config import resolve_beyond_robustness
from rsl_rl.beyond_robustness_stages import BREnvironmentStages,BRStageState
from rsl_rl.beyond_robustness_payload import SlidingPayloadConfig
from rsl_rl.runners.beyond_robustness_runner import BeyondRobustnessRunner
from test_beyond_robustness_runner import SyntheticBR,Writer
from test_beyond_robustness import cpu_threads
from legged_gym.scripts.train_beyond_robustness import parser,resolve_run,archive

ROOT=Path(__file__).resolve().parents[1]


class StagedSynthetic(SyntheticBR):
    """Real stage controller + existing reset-safe synthetic runner fixture."""
    def __init__(self,resolved,device='cpu',headless=True):
        super().__init__(resolved.training.input_schema,resolved.training.num_envs)
        self.resolved_config=deepcopy(resolved)
        self.stage_state=BRStageState(resolved.stages)
        self.progress=0.
        self.begin_calls=[]

    def begin_iteration(self,iteration):
        self.begin_calls.append(iteration)
        self.stage_state.update(iteration,self.progress)

    def end_iteration(self,iteration,tracking,count): pass

    def curriculum_state_dict(self):
        return dict(version=5,stage=self.stage_state.state_dict(),progress=self.progress)

    def validate_curriculum_state_dict(self,state): self.stage_state.validate(state['stage'])

    def load_curriculum_state_dict(self,state):
        self.validate_curriculum_state_dict(state)
        self.stage_state.load_state_dict(state['stage']); self.progress=state['progress']


def resolved(**stages):
    return resolve_beyond_robustness(environment_overrides={'env.num_envs':3,'env.num_obs_hist':3},
        training_overrides=dict(num_steps_per_env=3,num_learning_epochs=1,num_mini_batches=1,
            auxiliary_epochs=1,max_iterations=2,save_interval=10),stage_overrides=stages)


def runner(c):
    return BeyondRobustnessRunner(StagedSynthetic(c),c.training,writer=Writer())


def assert_equal(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: assert_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_equal(x,y)
    else: assert a==b


def test_disturbance_progress_changes_stage_not_optimization():
    c=resolved(); r=runner(c)
    r.learn(1); assert r.env.stage_state.stage==1
    r.env.progress=.2
    r.learn(1); assert r.env.stage_state.stage==2 and r.alg.training_phase=='teacher_student'
    r.env.begin_iteration(2); assert len(r.env.stage_state.transitions)==1
    assert r.env.begin_calls==[0,1,2]
    assert r.env.progress==.2


@pytest.mark.parametrize('phase',['teacher_student','student_reinforcement'])
def test_stage_transition_resume_preserves_learning_and_all_optimizers(tmp_path,phase):
    c=resolved(); c.training=replace(c.training,training_phase=phase)
    r=runner(c); r.learn(1)
    r.env.progress=.2; r.env.stage_state.update(r.alg.iteration,r.env.progress)
    r.save(tmp_path/'one.pt')
    old=deepcopy(r.alg.checkpoint_state())
    c.training=replace(c.training,max_iterations=5,save_interval=0)
    new=runner(c); new.load(tmp_path/'one.pt')
    assert new.alg.training_phase==phase and new.env.stage_state.stage==2
    assert new.env.progress==r.env.progress and new.interactions==r.interactions
    assert new.alg.history.frames.eq(0).all() and new.alg.reset_pending.all()
    for key in ('model_state_dict','optimizer','history_optimizer','load_optimizer','learning_rate','entropy_coef','iteration'):
        assert_equal(old[key],new.alg.checkpoint_state()[key])
    new.learn(1); new.save(tmp_path/'two.pt')
    again=runner(c); again.load(tmp_path/'two.pt')
    assert again.env.stage_state.stage==2 and again.alg.training_phase==phase
    assert len(again.env.stage_state.transitions)==1
    again.learn(1)


def test_runtime_overrides_and_cli_rejection(tmp_path):
    c=resolved(); r=runner(c); r.learn(1); path=tmp_path/'checkpoint.pt'; r.save(path)
    args=parser().parse_args(['--checkpoint',str(path),'--additional-iterations','2','--save-interval','7'])
    new,start,changes=resolve_run(args)
    assert start==1 and new.training.max_iterations==3 and new.training.save_interval==7
    assert changes['max_iterations']==[2,3]
    for extra in (['--seed','99'],['--num-envs','5'],['--training-phase','student_reinforcement'],['--max-iterations','0']):
        with pytest.raises(ValueError): resolve_run(parser().parse_args(['--checkpoint',str(path),*extra]))
    with pytest.raises(SystemExit): parser().parse_args(['--qp-solver','cupiqp'])
    for flag in ('--transition-stage','--stage2-start-iteration','--environment-stage'):
        with pytest.raises(SystemExit): parser().parse_args([flag,'2'])


def test_preflight_rejects_incompatibility_before_mutation(tmp_path):
    c=resolved(); source=runner(c); source.learn(1); path=tmp_path/'checkpoint.pt'; source.save(path)
    for field,value in [('learning_rate',.002),('history_length',4)]:
        other=deepcopy(c); other.training=replace(other.training,**{field:value})
        target=runner(other); before=deepcopy(target.alg.checkpoint_state())
        with pytest.raises(ValueError): target.load(path)
        assert_equal(before['model_state_dict'],target.alg.checkpoint_state()['model_state_dict'])
        assert target.alg.iteration==0
    data=torch.load(path,weights_only=False); data['curriculum_state']['stage']['config']['light_disturbance_scale']=.5
    torch.save(data,tmp_path/'bad.pt'); target=runner(c); before=deepcopy(target.alg.actor_critic.state_dict())
    with pytest.raises(ValueError,match='stage contract'): target.load(tmp_path/'bad.pt')
    assert_equal(before,target.alg.actor_critic.state_dict())


def test_stage_ranges_and_reset_only_resampling():
    from legged_gym.simulator.reset_randomization_cadence import ResetRandomizationCadence
    c=SlidingPayloadConfig(); stage=BRStageState(BREnvironmentStages())
    assert c.ranges(stage.payload_progress(0.))['mass']==c.mass_initial
    cadence=ResetRandomizationCadence(3,'cpu',100); ids=torch.arange(3)
    cadence.update_ranges(c.ranges(0.)); cadence.begin_reset(ids)
    for key in c.ranges(0.): assert cadence.select(key,ids).tolist()==[0,1,2]
    stage.update(44,.8); cadence.update_ranges(c.ranges(stage.payload_progress(.8)))
    # Only the resetting row consumes the new generation; no active sample is changed.
    for key in c.ranges(.8):
        assert cadence.select(key,ids[1:2]).tolist()==[1]
        assert cadence.select(key,ids[1:2]).numel()==0
        assert cadence.select(key,ids[:1]).tolist()==[0]
    assert c.ranges(1.)['friction']==pytest.approx(c.friction_range)
    assert stage.disturbance_scale==1 and BREnvironmentStages().light_disturbance_scale==.01


def test_shared_disturbance_warmup_gates_survive_stage_boundary():
    from legged_gym.envs.go2.go2_beyond_robustness import IsaacLabBeyondRobustnessEnv as Env
    from legged_gym.envs.go2.go2_hard_pact.domain_rand_curriculum import HardPACTDomainRandCurriculum
    env=Env.__new__(Env); env.cfg=resolved().environment; d=env.cfg.domain_rand
    d.use_joint_dynamics_curriculum=False; d.use_mass_com_curriculum=False; d.use_disturbance_curriculum=True
    d.push_warmup=2; d.step_interval=1; d.reward_ema_alpha=1.; d.min_reward_to_step=.7; d.disturbance_progress_delta=.1
    env.cfg.commands.curriculum=False
    env.domain_rand_curriculum=HardPACTDomainRandCurriculum(env.cfg)
    env.stage_state=BRStageState(BREnvironmentStages()); env.last_completed_iteration=-1; env.command_streak=0
    values=[]; env.simulator=NS(use_domainrand_curriculum=True,set_payload_progress=values.append)
    env._apply_domain_rand_curriculum_state=lambda:None
    for i in range(3): env.end_iteration(i,.9,3)
    assert env.domain_rand_curriculum.progress['disturbance']==0
    env.end_iteration(3,.9,3); assert env.domain_rand_curriculum.progress['disturbance']==.1
    assert values[-1]==.1 and env.stage_state.stage==2
    saved=deepcopy(env.domain_rand_curriculum.state_dict()); env._publish_payload_stage()
    assert values[-1]==.1
    assert_equal(saved,env.domain_rand_curriculum.state_dict())
    env.end_iteration(4,.0,3); assert values[-1]==.1
    env.end_iteration(4,.9,3); assert values[-1]==.1
    env.end_iteration(5,None,0); env.end_iteration(6,float('nan'),3)
    assert values[-1]==.1
    env.end_iteration(7,.9,3); assert values[-1]==.2
    assert len(env.stage_state.transitions)==1


def test_persistent_wrenches_disabled_only_in_br():
    from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_hard_pact_backend_configs
    E,T=make_hard_pact_backend_configs('isaaclab'); e,t=E(),T()
    e.domain_rand.persistent_disturbance=True
    c=resolve_beyond_robustness(e,t)
    assert not c.environment.domain_rand.persistent_disturbance
    assert e.domain_rand.persistent_disturbance
    assert c.environment.domain_rand.use_disturbance_curriculum==e.domain_rand.use_disturbance_curriculum
    assert c.environment.domain_rand.push_robots==e.domain_rand.push_robots
    from legged_gym.envs.go2.go2_beyond_robustness import IsaacLabBeyondRobustnessEnv as Env
    env=NS(cfg=c.environment,_current_sustained_wrench_world=torch.ones(3,6),
           _persistent_component_active=torch.ones(3,2,dtype=torch.bool),
           _current_sustained_active_mask=torch.ones(3,dtype=torch.bool))
    rng=torch.get_rng_state().clone()
    Env._update_persistent_wrench(env,100000)
    assert not env._current_sustained_wrench_world.any()
    assert not env._persistent_component_active.any() and not env._current_sustained_active_mask.any()
    assert torch.equal(rng,torch.get_rng_state())
    with pytest.raises(ValueError,match='physical payloads'):
        resolve_beyond_robustness(environment_overrides={'domain_rand.persistent_disturbance':True})


def test_cli_rejects_old_reward_contract_and_unbounded_smoke(tmp_path):
    c=resolved(); r=runner(c); r.save(tmp_path/'current.pt')
    data=torch.load(tmp_path/'current.pt',weights_only=False)
    data['resolved_configuration'].environment.br_reward['version']=2
    torch.save(data,tmp_path/'old.pt')
    with pytest.raises(ValueError,match='reward contract'):
        resolve_run(parser().parse_args(['--checkpoint',str(tmp_path/'old.pt')]))
    data=torch.load(tmp_path/'current.pt',weights_only=False); data['curriculum_state']['version']=4
    torch.save(data,tmp_path/'old_physics.pt')
    with pytest.raises(ValueError,match='physical curriculum contract'):
        resolve_run(parser().parse_args(['--checkpoint',str(tmp_path/'old_physics.pt')]))
    with pytest.raises(ValueError,match='at most two'):
        resolve_run(parser().parse_args(['--smoke','--max-iterations','1000']))


def test_registered_dispatch_flat_config_and_factory_isolation(monkeypatch):
    from legged_gym.envs import task_registry
    name='go2_beyond_robustness_isaaclab'
    assert name in task_registry.resolved_factories
    a,_=task_registry.get_cfgs(name,None); b,_=task_registry.get_cfgs(name,None)
    a.environment.commands.ranges.lin_vel_x[0]=-123
    assert b.environment.commands.ranges.lin_vel_x[0]!=-123
    monkeypatch.setitem(task_registry.task_classes,name,StagedSynthetic)
    c=resolved()
    env,actual=task_registry.make_env(name,NS(device='cpu',headless=True),c)
    r,t=task_registry.make_alg_runner(env,name,train_cfg=c.training,log_root=None)
    assert actual is c and t is c.training
    r.learn(1); assert r.alg.iteration==1


def test_launchers_and_unity_allowlist():
    scripts=[ROOT/'legged_gym/scripts'/f'go2_beyond_robustness{s}.sh' for s in ('','_lab','_unity')]+[ROOT/'containers/isaaclab/run.sh']
    for path in scripts: subprocess.run(['bash','-n',str(path)],check=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-test-assigned',IMAGE='/image.sif',REPO=str(ROOT),RUN_DIR='/tmp/br-unity-dry',OMNI_KIT_ACCEPT_EULA='YES')
    for kind in ('lab','unity'):
        p=subprocess.run(['bash',str(ROOT/f'legged_gym/scripts/go2_beyond_robustness_{kind}.sh'),'--dry-run','--smoke'],env=env,text=True,capture_output=True,check=True)
        assert 'train_beyond_robustness.py' in p.stdout and 'GPU-test-assigned' in p.stdout
        if kind=='unity': assert '/opt/env/bin/python' in p.stdout and 'cuda:0' in p.stdout
    p=subprocess.run(['bash',str(ROOT/'containers/isaaclab/run.sh'),'script','go2_beyond_robustness_unity.sh','--dry-run','--smoke'],env=env,text=True,capture_output=True,check=True)
    assert 'go2_beyond_robustness_unity.sh' in p.stdout
    p=subprocess.run(['bash',str(ROOT/'legged_gym/scripts/go2_beyond_robustness_unity.sh'),'--dry-run','--gpu','cuda:1'],env=env,capture_output=True)
    assert p.returncode!=0


def test_direct_scripts_directory_entrypoint_uses_this_worktree():
    env=dict(os.environ,SIMULATOR='isaaclab'); env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,'train_beyond_robustness.py','--smoke','--dry-run'],
        cwd=ROOT/'legged_gym/scripts',env=env,text=True,capture_output=True,check=True)
    assert str(ROOT) in result.stdout
    assert '"simulator_initialized": false' in result.stdout
