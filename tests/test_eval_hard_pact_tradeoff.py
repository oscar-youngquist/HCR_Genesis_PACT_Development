"""Forward-only evaluation contract checks; no training/optimizer/backward."""
from dataclasses import asdict
from types import SimpleNamespace
import torch
import pytest
from scripts.eval_hard_pact_tradeoff import freeze_settings, Observer
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig
from rsl_rl.algorithms.hard_pact_qp_curriculum import QPCurriculum


def test_frozen_settings_only_requested_weights_change():
    cfg=HardPACTQPConfig(qp_update_mode='curriculum_substeps',contact_acceleration_weight=.25,
        torque_tracking_weight=20.,intervention_extra_strength_enabled=True)
    schedule=QPCurriculum(cfg).state_dict()
    schedule['intervention']['snapshot_progress']=.5
    checkpoint={'training_state':{'qp_curriculum':schedule,'next_iteration':11001}}
    training={'algorithm':{'hard_pact_qp':asdict(cfg)}}
    saved,alpha,p,strength=freeze_settings(training,checkpoint,'saved')
    assert p==(.5,.3,.2) and strength==.625
    for name,stance,torque in [('stance0_torque20',0.,20.),('stance010_torque20',.1,20.),('stance010_torque30',.1,30.)]:
        variant,a,prob,s=freeze_settings(training,checkpoint,name)
        expected=asdict(saved);expected.update(contact_acceleration_weight=stance,torque_tracking_weight=torque)
        assert asdict(variant)==expected
        assert (a,prob,s)==(alpha,p,strength)
    assert checkpoint['training_state']['qp_curriculum']['intervention']['snapshot_progress']==.5


def test_missing_sample_unavailable_not_successful_zero():
    observer=Observer.__new__(Observer)
    observer.t=0;observer.k=1;observer.args=SimpleNamespace(steps=2)
    observer.arrays={};observer.controls={}
    observer.record('objective',torch.tensor([3.,4.]))
    assert torch.isnan(observer.arrays['objective'][0]).all()
    assert torch.equal(observer.arrays['objective'][1],torch.tensor([3.,4.]))
    assert not observer.arrays['objective'].requires_grad


def test_objective_overrides_reach_production_matrices():
    from test_hard_pact_reduced_qp import solver,inputs
    from dataclasses import replace
    qp=solver();d=inputs(2);d['foot_jacobians'][:,:,0,6]=1
    qp.cfg=replace(qp.cfg,contact_acceleration_weight=.25,torque_tracking_weight=20.)
    original=qp._build(d)
    qp.cfg=replace(qp.cfg,contact_acceleration_weight=0.)
    no_stance=qp._build(d)
    assert not torch.equal(original.Q,no_stance.Q)
    assert torch.equal(original.G,no_stance.G) and torch.equal(original.h,no_stance.h)
    qp.cfg=replace(qp.cfg,torque_tracking_weight=30.)
    high_torque=qp._build(d)
    assert not torch.equal(no_stance.Q,high_torque.Q)
    assert torch.equal(no_stance.G,high_torque.G) and torch.equal(no_stance.h,high_torque.h)


@pytest.mark.parametrize('name',['B_planar10','B_yaw5','B_both_lower'])
def test_velocity_stress_weights_reach_assembly_without_constraint_changes(name):
    from test_hard_pact_reduced_qp import inputs,solver
    from dataclasses import replace
    from scripts.eval_hard_pact_tradeoff import INTERVENTION_VARIANTS
    q=solver();q.cfg=replace(q.cfg,planar_velocity_weight=20.,yaw_rate_weight=10.)
    d=inputs();d['base_linear_velocity_world']=torch.zeros(2,3,dtype=torch.float64)
    d['velocity_command']=torch.ones(2,3,dtype=torch.float64)
    d['foot_jacobians'][:,:,:,:3]=torch.eye(3,dtype=torch.float64)*.1
    d['foot_jacobians'][:,:,2,5]=.02
    reference=q._build(d)
    q.cfg=replace(q.cfg,**INTERVENTION_VARIANTS[name][0]);changed=q._build(d)
    assert not torch.equal(reference.Q,changed.Q)
    assert not torch.equal(reference.p,changed.p)
    assert torch.equal(reference.G,changed.G) and torch.equal(reference.h,changed.h)


def test_intervention_sweep_preserves_stance_clipping_and_only_changes_allowlist():
    from scripts.eval_hard_pact_tradeoff import INTERVENTION_VARIANTS
    cfg=HardPACTQPConfig(qp_update_mode='curriculum_substeps',contact_acceleration_weight=.1,
        attitude_weight=.1,planar_velocity_weight=20.,intervention_extra_strength_enabled=True,
        execution_correction_clip_enabled=True,objective_curriculum_enabled=False)
    checkpoint={'training_state':{'qp_curriculum':QPCurriculum(cfg).state_dict(),'next_iteration':10501}}
    training={'algorithm':{'hard_pact_qp':asdict(cfg)}}
    ref,*_=freeze_settings(training,checkpoint,'A_reference')
    for name,(overrides,extra) in INTERVENTION_VARIANTS.items():
        result,alpha,prob,strength=freeze_settings(training,checkpoint,name)
        expected=asdict(ref);expected.update(overrides)
        assert asdict(result)==expected and alpha==1. and prob==(0.,0.,1.) and strength==extra
        assert result.contact_acceleration_weight==.1 and result.execution_correction_clip_enabled
    assert checkpoint['training_state']['qp_curriculum']['intervention']['progress']==0.


def test_targeting_uses_failure_levels_and_balances_families():
    from scripts.hard_pact_intervention_eval import targeted_indices
    terrain={'terrain_levels':torch.tensor([6,7,8]*3),'terrain_types':torch.tensor([6]*3+[10]*3+[14]*3)}
    saved={'environment':{'resolved':{'terrain':{'terrain_proportions':[.1,.2,.2,.2,.2,.1],'num_cols':20}}}}
    evidence={'counts':{f'EpisodeDiagnostics/level/{l}/failure_count':{'count':l} for l in (6,7,8)}}
    indices,plan=targeted_indices(terrain,saved,evidence,1)
    assert len(indices)==10 and [sum(x['family']==f for x in plan) for f in ('stairs_down','stairs_up','discrete_obstacles')]==[3,3,4]
    assert plan[0]['level']==8 and all(x['level'] in (6,7,8) for x in plan)


def test_failure_ring_wraps_owns_samples_and_respects_cap():
    from scripts.hard_pact_intervention_eval import capture_ring
    o=Observer.__new__(Observer);o.args=SimpleNamespace(steps=40,intervention_sweep=True,capture_cap=1)
    o.arrays={};o.controls={};o.e=SimpleNamespace(device='cpu')
    for index in range(120):
        o.t,o.k=divmod(index,4);o.record('value',torch.tensor([float(index),-1.]))
    capture_ring(o,torch.tensor([True,True]))
    assert len(o.failure_ring)==1
    torch.testing.assert_close(o.failure_ring[0]['value'],torch.arange(20.,120.))
    o.history_ring['value'].zero_()
    assert o.failure_ring[0]['value'][-1]==119


def test_reported_weighted_cost_changes_match_production_objective():
    from test_hard_pact_reduced_qp import inputs,solver
    d=inputs(2);d.update(environment_ids=torch.arange(2),velocity_command=torch.ones(2,3,dtype=torch.float64),
        base_linear_velocity_world=torch.zeros(2,3,dtype=torch.float64),estimated_height=torch.full((2,),.3,dtype=torch.float64),
        estimated_base_linear_velocity_world=torch.zeros(2,3,dtype=torch.float64))
    d['base_angular_velocity_world'].fill_(.2)
    d['foot_jacobians'][:,:,:,:3]=torch.eye(3,dtype=torch.float64)
    qp=solver(height_target=.38,height_weight=.5,attitude_weight=.1,contact_acceleration_weight=.1,
        planar_velocity_weight=20.,yaw_rate_weight=10.)
    m=qp._build(d);data={}
    o=Observer.__new__(Observer);o.build=m;o.qp=qp;o.e=SimpleNamespace(num_envs=2);o.probe=None
    o.assembly_weights=(.1,20.);o.record=lambda name,value:data.update({name:value})
    r=SimpleNamespace(tau_safe=d['tau_nom']+1,force_world=d['force_pred_world']+2,
        differentiated_mask=torch.ones(2,dtype=torch.bool),recovery_mask=torch.zeros(2,dtype=torch.bool))
    o.costs(d,r)
    components=('torque','grf','stance','attitude','height','planar','yaw')
    delta=sum(data['objective_candidate_'+k]-data['objective_nominal_'+k] for k in components)
    def value(tau,force):
        z=torch.cat((tau,force.flatten(1)),-1)/m.variable_scale
        return .5*torch.einsum('bi,bij,bj->b',z,m.Q,z)+(m.p*z).sum(-1)-.5*qp.cfg.q_regularization*z.square().sum(-1)
    torch.testing.assert_close(delta,value(r.tau_safe,r.force_world)-value(d['tau_nom'],d['force_pred_world']))
