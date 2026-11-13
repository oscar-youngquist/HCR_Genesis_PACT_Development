"""Shared HardPACT-style logging without simulator construction or extra forwards."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from rsl_rl.utils.learning_diagnostics import LearningEpisodeDiagnostics, policy_diagnostics
from rsl_rl.runners.beyond_robustness_runner import BeyondRobustnessRunner
from rsl_rl.runners.mule_runner import MULERunner
from rsl_rl.beyond_robustness_config import BeyondRobustnessConfig
from rsl_rl.beyond_robustness_schema import BeyondRobustnessInputSchema
from test_beyond_robustness_runner import SyntheticBR, Writer
from test_mule_learning import SyntheticMULE, config, assert_equal


@pytest.fixture(autouse=True)
def cpu():
    previous=torch.get_num_threads(); torch.set_num_threads(1); torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def make_runner(method, phase, log_dir=None, writer=None):
    if method=='BR':
        cfg=BeyondRobustnessConfig(training_phase=phase,history_length=3,num_envs=3,
            num_steps_per_env=3,num_learning_epochs=1,num_mini_batches=1,auxiliary_epochs=1,
            input_schema=BeyondRobustnessInputSchema(terrain_points_x=(0.,),terrain_points_y=(0.,)))
        env=SyntheticBR(cfg.input_schema); env.dt=.02; env.max_episode_length_s=1.
        original=env.step
        def step(actions):
            result=original(actions)
            result.reward_terms={'tracking_lin_vel':result.rewards.clone()}
            result.failure_reasons={'contact':result.dones & ~result.timeouts}
            result.raw_timeouts=result.dones.clone()
            return result
        env.step=step
        return BeyondRobustnessRunner(env,cfg,log_dir=log_dir,writer=writer)
    cfg=config(phase,num_learning_epochs=1,auxiliary_epochs=1,num_mini_batches=1)
    env=SyntheticMULE(cfg); env.max_episode_length_s=1.
    original=env.step
    def step(actions):
        result=original(actions)
        return replace(result,failure_reasons={'contact':result.terminated},reward_terms={
            'nominal':{'tracking_lin_vel':result.nominal_reward.clone()},
            'adaptive':{'tracking_lin_vel':result.adaptive_reward.clone()}})
    env.step=step
    return MULERunner(env,cfg,log_dir=log_dir,writer=writer)


def test_count_weighted_episode_terms_reasons_and_cross_iteration_state():
    log=LearningEpisodeDiagnostics(3,'cpu',dt=.5,max_episode_seconds=4.)
    zeros=torch.zeros(3,dtype=torch.bool)
    lengths=torch.ones(3,dtype=torch.long)
    log.record(zeros,zeros,lengths,reward_terms={'tracking':torch.tensor([1.,10.,100.])})
    done=torch.tensor([True,False,False]); lengths+=1
    log.record(done,done,lengths,reward_terms={'tracking':torch.tensor([3.,20.,200.])})
    done=torch.tensor([False,True,True]); timeout=torch.tensor([False,False,True])
    lengths=torch.tensor([1,3,3])
    log.record(done,timeout,lengths,reward_terms={'tracking':torch.tensor([5.,30.,300.])},
        reasons={'contact':torch.tensor([False,True,False]),'height_min':torch.tensor([False,True,False])},
        raw_timeout=done)
    metrics=log.metrics()
    assert metrics['Episode/rew_tracking']==pytest.approx((4+60+600)/3/4)
    assert metrics['EpisodeDiagnostics/all/all_count']==3
    assert metrics['EpisodeDiagnostics/all/failure_count']==1
    assert metrics['EpisodeDiagnostics/all/timeout_count']==2
    assert metrics['EpisodeDiagnostics/all/all_mean_length_s']==pytest.approx(8/3*.5)
    for key in ('contact','height_min'):
        assert metrics[f'EpisodeDiagnostics/reason/{key}_fraction_of_failures']==1
    assert metrics['EpisodeDiagnostics/failure_timeout_overlap_count']==1
    # Partial episode on row 0 survives iteration and exact-resume boundaries.
    saved=log.state_dict(); restored=LearningEpisodeDiagnostics(3,'cpu',.5,4.)
    restored.load_state_dict(saved); log.begin_iteration()
    for target in (log,restored):
        target.record(torch.tensor([True,False,False]),zeros,torch.tensor([2,1,1]),
            reward_terms={'tracking':torch.tensor([7.,1.,1.])})
    assert_equal(log.metrics(),restored.metrics())
    assert log.metrics()['Episode/rew_tracking']==3.
    restored.clear_episodes(); assert restored.term_sums=={}


def test_nonfinite_missing_terms_and_unknown_duration_are_not_successful_zeros():
    log=LearningEpisodeDiagnostics(2,'cpu',.02,1.)
    no=torch.zeros(2,dtype=torch.bool); yes=~no
    log.record(no,no,torch.ones(2,dtype=torch.long),reward_terms={'x':torch.tensor([float('nan'),1.])})
    log.record(yes,no,torch.full((2,),2),reward_terms={'x':torch.tensor([2.,3.])})
    metrics=log.metrics()
    assert metrics['Episode/rew_x']==4
    assert metrics['EpisodeDiagnostics/reward_valid_count/x']==1
    assert metrics['EpisodeDiagnostics/reward_incomplete_count/x']==1
    assert metrics['EpisodeDiagnostics/reason_unavailable_count']==2
    unknown=LearningEpisodeDiagnostics(2,'cpu')
    unknown.record(yes,no,torch.ones(2),reward_terms={'x':torch.ones(2)})
    assert 'Episode/rew_x' not in unknown.metrics()
    assert 'EpisodeDiagnostics/all/all_mean_length_s' not in unknown.metrics()


def test_rolling_return_window_matches_hardpact_and_no_rng_or_policy_mutation():
    log=LearningEpisodeDiagnostics(3,'cpu',.02,1.)
    for i in range(40):
        log.record(torch.ones(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool),torch.ones(3),
            returns=torch.arange(3).float()+3*i)
    metrics=log.metrics()
    assert metrics['Train/mean_reward']==pytest.approx(torch.arange(20,120).float().mean().item())
    assert metrics['Train/mean_episode_length']==1
    std=torch.arange(1.,13.,requires_grad=True); before=std.clone(); rng=torch.get_rng_state().clone()
    values=policy_diagnostics(std,tuple(str(i) for i in range(12)))
    assert values=={'Policy/mean_noise_std':6.5,'Policy/mean_noise_std_position':6.5}
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(std,before) and std.grad is None


@pytest.mark.parametrize('method,phase', [('BR','teacher_student'),('BR','student_reinforcement'),
    ('MULE','nominal_only'),('MULE','joint_adaptation')])
def test_real_event_files_and_terminal_follow_canonical_tags(tmp_path,capsys,method,phase):
    runner=make_runner(method,phase,tmp_path)
    runner.learn(1); runner.close()
    output=capsys.readouterr().out
    assert 'Learning iteration' in output and 'Computation:' in output
    assert 'Mean action noise std:' in output and 'Mean episode length:' in output
    assert 'Mean episode' in output and 'rew_tracking_lin_vel' in output
    assert 'reward/coefficient' not in output
    events=EventAccumulator(str(tmp_path)); events.Reload()
    tags=set(events.Tags()['scalars'])
    assert {'Policy/mean_noise_std','Policy/mean_noise_std_position','Perf/total_fps',
        'Train/mean_episode_length','EpisodeDiagnostics/all/failure_count'} <= tags
    assert not any(k.startswith(('reward/','loss/','load/','payload/','environment/')) for k in tags)
    assert not any('coefficient' in k or '/raw/' in k or '/weighted/' in k for k in tags)
    assert all(len(events.Scalars(k))==1 for k in tags)  # No duplicate writes.
    if method=='BR':
        assert {'Loss/value_function','Loss/surrogate','Loss/load_estimation','Episode/rew_tracking_lin_vel'} <= tags
        assert ('Loss/latent_matching' in tags)==(phase=='teacher_student')
    else:
        assert {'Loss/nominal/value_function','Loss/autoenc_function','Episode/nominal/rew_tracking_lin_vel'} <= tags
        assert ('Loss/adaptive/value_function' in tags)==(phase=='joint_adaptation')
        assert ('Policy/adaptive/mean_noise_std' in tags)==(phase=='joint_adaptation')


@pytest.mark.parametrize('method,phase',[('BR','teacher_student'),('MULE','joint_adaptation')])
def test_diagnostics_do_not_change_learning_or_rng(method,phase):
    torch.manual_seed(7)
    normal=make_runner(method,phase,writer=Writer()); normal.console=False
    normal.learn(1)
    model=normal.alg.actor_critic if method=='BR' else normal.alg.model
    state=deepcopy(model.state_dict()); rng=torch.get_rng_state().clone()
    torch.manual_seed(7)
    reference=make_runner(method,phase,writer=Writer()); reference.console=False
    reference.episode_diagnostics.record=lambda *a,**kw: None
    reference.episode_diagnostics.metrics=lambda **kw: {}
    reference.learn(1)
    other=reference.alg.actor_critic if method=='BR' else reference.alg.model
    assert_equal(state,other.state_dict()); assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('method,phase',[('BR','teacher_student'),('MULE','joint_adaptation')])
def test_episode_logging_resume_and_fresh_reset(tmp_path,method,phase):
    runner=make_runner(method,phase,writer=Writer()); runner.console=False; runner.learn(1)
    saved=runner.episode_diagnostics.state_dict(); path=tmp_path/'model.pt'; runner.save(path)
    restored=make_runner(method,phase,writer=Writer()); restored.console=False
    restored.load(path,resume_mode='exact')
    assert_equal(saved,restored.episode_diagnostics.state_dict())
    restored.load(path,resume_mode='fresh')
    assert restored.episode_diagnostics.term_sums=={}
    assert restored.episode_diagnostics.recent_lengths.numel()==0
