from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg
from legged_gym.envs.go2.go2_hard_pact.domain_rand_curriculum import (
    HardPACTDomainRandCurriculum, PostInterventionDisturbanceGate, go2_pact_domain_rand_schema)
from rsl_rl.algorithms.hard_pact_actor_regularizers import nominal_support_gravity
from rsl_rl.runners.pact_runner import OnPolicyRunnerPACT as OnPolicyRunner_PACT


def config():
    c = deepcopy(GO2HardPACTCfg())
    d = c.domain_rand
    d.disturbance_settling_iterations = 4
    d.disturbance_gate_window_iterations = 2
    d.disturbance_gate_min_env_seconds = 10
    d.push_warmup = 0
    d.step_interval = 1
    return c


def advance(c, i, score=.8, failures=1, complete=True):
    return c.advance(i, score, qp_complete=complete, tracking_count=100,
                     failures=failures, exposure=100)


def test_fixed_com_bounds_positive_z_and_validation():
    c = config(); s = go2_pact_domain_rand_schema(c)
    for axis, bounds in [('x',(-.05,.05)),('y',(-.05,.05)),('z',(0.,.05))]:
        assert s['base_com_'+axis].initial == s['base_com_'+axis].final == bounds
    curriculum = HardPACTDomainRandCurriculum(c)
    assert 'mass_com' not in curriculum.enabled_phases
    z = curriculum.sample('base_com_z', (1000,))
    assert z.min() >= 0 and z.max() <= .05 and z.std() > 0
    c.domain_rand.com_displacement_z_min_pos = .05
    with pytest.raises(ValueError, match='Positive CoM Z'): go2_pact_domain_rand_schema(c)


def test_com_inheritance_without_br_mule_qp_dependency():
    from rsl_rl.beyond_robustness_config import resolve_beyond_robustness
    from legged_gym.envs.go2.go2_mule.go2_mule_config import resolve_go2_mule
    from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_rss_hard_pact_configs
    from rsl_rl.hard_pact_ablations import RSS_HARD_PACT_ABLATIONS
    br, mule = resolve_beyond_robustness(), resolve_go2_mule()
    for cfg in (br.environment,mule.shared):
        assert not cfg.domain_rand.disturbance_sequence_enabled
        assert not cfg.domain_rand.use_mass_com_curriculum
        assert go2_pact_domain_rand_schema(cfg)['base_com_z'].final == (0.,.05)
    for variant in RSS_HARD_PACT_ABLATIONS:
        E,_=make_rss_hard_pact_configs(variant)
        assert go2_pact_domain_rand_schema(E())['base_com_z'].initial == (0.,.05)


def test_actual_completion_settling_resume_and_improving_references():
    c = HardPACTDomainRandCurriculum(config())
    for i in range(3): assert not advance(c,i,complete=False)
    assert not advance(c,3)
    for i in range(4,7): assert not advance(c,i)
    resumed = HardPACTDomainRandCurriculum(config()); resumed.load_state_dict(c.state_dict())
    for i in range(7,12):
        assert advance(c,i) == advance(resumed,i)
        assert c.progress == resumed.progress
        assert c.sequence.state_dict() == resumed.sequence.state_dict()
    assert c.progress['disturbance'] > 0
    ref, fail = c.sequence.tracking_reference, c.sequence.failure_reference
    for i in range(12,20): advance(c,i,score=.1,failures=10)
    assert c.sequence.tracking_reference >= ref and c.sequence.failure_reference <= fail
    before = c.progress['disturbance']
    for i in range(20,24): assert not advance(c,i,score=None)
    assert c.progress['disturbance'] == before


def test_minimum_evidence_warmup_and_no_qp_dependency():
    cfg = config(); cfg.domain_rand.push_warmup = 10
    cfg.domain_rand.disturbance_gate_min_env_seconds = 1000
    c = HardPACTDomainRandCurriculum(cfg)
    for i in range(15): assert not advance(c,i)
    assert c.sequence.tracking_reference is None
    cfg = config(); cfg.domain_rand.push_warmup = 10
    c = HardPACTDomainRandCurriculum(cfg)
    for i in range(11): assert not advance(c,i)
    assert advance(c,11)
    runner = SimpleNamespace(hard_pact_features=SimpleNamespace(execution_qp=False))
    assert OnPolicyRunner_PACT._disturbance_qp_complete(runner,0)
    runner.hard_pact_features.execution_qp=True
    runner.qp_curriculum=SimpleNamespace(intervention=SimpleNamespace(progress=.999))
    assert not OnPolicyRunner_PACT._disturbance_qp_complete(runner,99999)
    runner.qp_curriculum.intervention.progress=1.
    assert OnPolicyRunner_PACT._disturbance_qp_complete(runner,99999)


def test_step_cadence_and_partial_window_checkpoint():
    cfg=config(); cfg.domain_rand.step_interval=10
    c=HardPACTDomainRandCurriculum(cfg)
    changes=[]
    for i in range(26):
        if advance(c,i): changes.append(i)
    assert changes == [4,14,24]
    assert c.progress['disturbance'] == pytest.approx(.015)
    resumed=HardPACTDomainRandCurriculum(cfg); resumed.load_state_dict(c.state_dict())
    assert resumed.sequence.window_iterations == 1
    for i in range(26,30):
        assert advance(c,i) == advance(resumed,i)
        assert c.sequence.state_dict() == resumed.sequence.state_dict()
    from rsl_rl.utils.training_checkpoint import clear_gate_evidence
    transferred=clear_gate_evidence(c.state_dict())
    assert transferred['progress']==c.state_dict()['progress']
    assert transferred['sequence']['completion_iteration']==0
    assert transferred['sequence']['tracking_reference'] is None
    assert transferred['sequence']['failure_threshold'] is None
    legacy=c.state_dict(); legacy.pop('sequence')
    legacy.pop('enabled_phases'); legacy['phase_index']=2
    resumed=HardPACTDomainRandCurriculum(cfg); resumed.load_state_dict(legacy)
    assert resumed.phase == 'disturbance'
    assert resumed.sequence.completion_iteration is None and not resumed.sequence.gate


def test_realized_gravity_is_detached_batched_and_recomputed():
    # Analytic isolated torso: g_z=mg, moment=r x mg; no external wrench.
    calls=[]
    def context(q,v,parameters,need_jacobians):
        assert not torch.is_grad_enabled() and v.eq(0).all()
        calls.append(parameters)
        mass=10+parameters['added_base_mass']
        r=parameters['base_com_shift']
        force=torch.cat((mass*0,mass*0,mass*9.81),-1)
        g=q.new_zeros(q.shape[0],18); g[:,:3]=force
        g[:,3:6]=torch.cross(r,force,dim=-1)
        return SimpleNamespace(rnea=lambda a:g)
    dynamics=SimpleNamespace(batch_capacity=2,build_context=context)
    q=torch.zeros(3,19,requires_grad=True)
    with torch.no_grad(): q[:,6]=1
    mass=torch.tensor([[0.],[1.],[2.]],requires_grad=True)
    r=torch.tensor([[0.,0.,0.],[.05,-.05,.05],[0.,0.,.05]],requires_grad=True)
    g=nominal_support_gravity(dynamics,q,{'added_base_mass':mass,'base_com_shift':r})
    assert not g.requires_grad and len(calls)==2
    torch.testing.assert_close(g[:,2],(10+mass[:,0]).detach()*9.81)
    torch.testing.assert_close(g[1,3:5],torch.tensor([-5.3955,-5.3955]))
    g2=nominal_support_gravity(dynamics,q,{'added_base_mass':mass+1,'base_com_shift':r})
    assert not torch.equal(g,g2) and mass.grad is None and r.grad is None
