"""Resolved pretraining configuration and real scheduler checks, no simulator."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import pytest

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg, GO2HardPACTCfgPPO
from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_rss_hard_pact_configs, make_hard_pact_variant_configs
from legged_gym.envs.go2.go2_hard_pact.domain_rand_curriculum import HardPACTDomainRandCurriculum
from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig
from rsl_rl.algorithms.hard_pact_qp_curriculum import QPCurriculum, extra_intervention_strength
from rsl_rl.hard_pact_ablations import RSS_HARD_PACT_ABLATIONS

WEIGHTS=dict(torque_tracking_weight=15.,force_tracking_weight=.5,
    contact_acceleration_weight=.05,attitude_weight=.01,height_weight=.01,
    planar_velocity_weight=5.,yaw_rate_weight=5.)

def schedules():
    cfg=HardPACTQPConfig.from_dict(GO2HardPACTCfgPPO.algorithm.hard_pact_qp)
    return QPCurriculum(cfg,GO2HardPACTCfgPPO.algorithm.actor_physics_warmup_iterations), HardPACTDomainRandCurriculum(GO2HardPACTCfg())

def step(q,d,i,valid=True):
    q.begin(i)
    q.intervention.finish(i,.8 if valid else None,100,1,2000)
    d.advance(i,.8 if valid else None,qp_complete=q.intervention.progress>=1,
        tracking_count=100,failures=1,exposure=2000)

def test_resolved_weights_variants_and_stale_aliases():
    configs=[GO2HardPACTCfgPPO]+[make_rss_hard_pact_configs(v)[1] for v in RSS_HARD_PACT_ABLATIONS]
    configs += [make_hard_pact_variant_configs(v,'isaaclab')[1]
                for v in (None,'baseline','soft','hard','full','stopgrad','soft_penalty','inverse','rollout')]
    for config in configs:
        c=HardPACTQPConfig.from_dict(config.algorithm.hard_pact_qp)
        assert config.runner.max_iterations==16000
        assert not c.objective_curriculum_enabled and not c.torso_stability_curriculum_enabled
        assert (c.intervention_window_iterations,c.intervention_gate_check_interval)==(200,2)
        for key,value in WEIGHTS.items(): assert getattr(c,key)==value
        for key in ('contact_acceleration_weight','attitude_weight','height_weight'):
            assert getattr(c,key+'_initial')==WEIGHTS[key]
        # Legacy endpoint aliases cannot overwrite authoritative component weights.
        c=replace(c,contact_acceleration_weight_final=99.,attitude_weight_final=99.)
        s=QPCurriculum(c,config.algorithm.actor_physics_warmup_iterations)
        s.progress=1.
        effective,_=s.begin(16000)
        for key,value in WEIGHTS.items(): assert getattr(effective,key)==value

def test_open_gate_timeline():
    q,d=schedules();s=q.intervention
    assert (s.ramp_end,s.start,s.duration)==(1000,1500,7000)
    milestones={}
    for i in range(16000):
        step(q,d,i)
        if s.progress>0:milestones.setdefault('first_intervention_advance',i)
        if s.snapshot_progress>=1:
            milestones.setdefault('full_execution',i)
            assert extra_intervention_strength(q.cfg,s.snapshot_progress)==1.
            assert s.begin(i)==pytest.approx((0.,0.,1.))
        if d.progress['disturbance']>0:milestones.setdefault('first_disturbance_advance',i)
        if d.progress['disturbance']>=1:milestones.setdefault('full_disturbance',i)
    assert milestones['first_intervention_advance']==1500
    assert 8500<=milestones['full_execution']<=8501
    assert milestones['first_disturbance_advance']>9500
    assert milestones['first_disturbance_advance']>=d.sequence.completion_iteration+1000
    assert 13490<=milestones['full_disturbance']<=13510
    print('open-gate absolute PPO indices:',milestones)

def test_gate_pause_and_partial_evidence_resume():
    q,d=schedules()
    for i in range(1703):step(q,d,i)
    before=q.intervention.progress
    for i in range(1703,1713):step(q,d,i,False)
    assert q.intervention.progress==before
    for i in range(1713,1781):step(q,d,i)
    qr,dr=schedules()
    qr.load_state_dict(deepcopy(q.state_dict()));dr.load_state_dict(deepcopy(d.state_dict()))
    for i in range(1781,14200):
        step(q,d,i);step(qr,dr,i)
        assert q.intervention.state_dict()==qr.intervention.state_dict()
        assert d.progress==dr.progress and d.sequence.state_dict()==dr.sequence.state_dict()
    assert d.sequence.completion_iteration>8500
    assert q.intervention.progress==1 and d.progress['disturbance']==1

def test_reward_endpoints_are_absolute_ppo_iterations():
    cfg=GO2HardPACTCfg();r=cfg.rewards.reward_curriculum
    assert (r.warmup_steps,r.curr_steps)==(0,12000)
    env=SimpleNamespace(use_reward_curriculum=True,reward_warmup_steps=r.warmup_steps,
        reward_curr_steps=r.curr_steps,reward_curr_keys=r.curr_reward_keys,
        reward_curr_bounds=deepcopy(r.curr_reward_bounds),dt=.02,
        reward_scales={key:1. for key in r.curr_reward_keys},
        cfg=SimpleNamespace(sim=SimpleNamespace(console_debug=False)))
    for i in (0,6000,12000,16000):
        Go2PACT.step_reward_curriculum(env,i)
        for key in r.curr_reward_keys:
            lo,hi=r.curr_reward_bounds[key]
            expected=lo if i==0 else (lo+hi)/2 if i==6000 else hi
            assert env.reward_scales[key]==pytest.approx(expected*env.dt)


def test_downstream_shared_curricula_and_method_exceptions():
    from rsl_rl.beyond_robustness_config import resolve_beyond_robustness
    from legged_gym.envs.go2.go2_mule.go2_mule_config import resolve_go2_mule
    from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_hard_pact_pos_backend_configs, make_hard_pact_backend_configs
    SharedEnv,SharedTraining=make_hard_pact_backend_configs('isaaclab')
    source,training=SharedEnv(),SharedTraining()
    keys=('push_warmup','step_interval','disturbance_progress_delta',
          'best_reward_window','best_reward_quantile','recovery_ratio',
          'reward_ema_alpha','min_reward_to_step')
    configs=[make_rss_hard_pact_configs(v) for v in RSS_HARD_PACT_ABLATIONS]
    configs += [make_hard_pact_pos_backend_configs(b) for b in ('genesis','isaaclab')]
    for E,T in configs:
        e,t=E(),T()
        assert t.runner.max_iterations==training.runner.max_iterations
        for key in keys:assert getattr(e.domain_rand,key)==getattr(source.domain_rand,key)
        for key in ('curr_steps','warmup_steps'):
            assert getattr(e.rewards.reward_curriculum,key)==getattr(source.rewards.reward_curriculum,key)
        assert e.terrain.curriculum_upward_delay_iterations==source.terrain.curriculum_upward_delay_iterations
        assert e.commands.curriculum_patience_iterations==source.commands.curriculum_patience_iterations
    E,_=make_hard_pact_pos_backend_configs('isaaclab')
    assert not E.domain_rand.use_domainrand_curriculum
    assert not E.domain_rand.use_disturbance_curriculum

    # Exercise factories with non-default shared settings, proving inheritance
    # instead of just coincidentally matching constants in the current config.
    source.domain_rand.push_warmup=9876
    source.domain_rand.disturbance_progress_delta=.003
    source.rewards.reward_curriculum.curr_steps=12345
    training.runner.max_iterations=16789
    br=resolve_beyond_robustness(source,training)
    mule=resolve_go2_mule(source,training)
    for e,t,spec in ((br.environment,br.training,br.environment.br_reward),
                     (mule.shared,mule.training,mule.shared.mule_reward)):
        assert t.max_iterations==16789
        for key in keys:assert getattr(e.domain_rand,key)==getattr(source.domain_rand,key)
        assert spec['curriculum']['duration']==12345
        assert spec['curriculum']['warmup']==source.rewards.reward_curriculum.warmup_steps
        # Payload methods must not wait forever for a nonexistent QP ramp.
        assert not e.domain_rand.disturbance_sequence_enabled
        assert not e.domain_rand.persistent_disturbance
    br.environment.domain_rand.push_warmup=1
    assert source.domain_rand.push_warmup==mule.shared.domain_rand.push_warmup==9876
