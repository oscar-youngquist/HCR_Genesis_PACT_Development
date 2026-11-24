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
