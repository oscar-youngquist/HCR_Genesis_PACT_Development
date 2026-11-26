from copy import deepcopy
from dataclasses import replace, asdict
from types import SimpleNamespace
import pytest
import torch
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig, recovery_projection_loss
from rsl_rl.algorithms.hard_pact_recovery_budget import RecoveryLossBudget, legacy_recovery_contract, DEFAULTS


def fixture(rate=True):
    cfg=HardPACTQPConfig(recovery_slack_huber_enabled=True,
        soft_joint_recovery_scale_rad_s2=1.,soft_rate_recovery_scale_nm=1.,
        torque_rate_constraint_weight=float(rate))
    slack=torch.tensor([[.5,3.],[float('nan'),float('nan')]],requires_grad=True)
    result=SimpleNamespace(recovery_mask=torch.tensor([True,False]),
        tau_safe=torch.tensor([[2.,0.],[float('nan'),float('nan')]],requires_grad=True),
        recovery_slack=slack,recovery_rate_slack=slack)
    return cfg,result,torch.zeros(2,2),torch.ones(2),torch.tensor([True,True])


@pytest.mark.parametrize('rate',[False,True])
def test_huber_values_gradients_and_empty(rate):
    cfg,r,tau,limit,valid=fixture(rate)
    loss,rows=recovery_projection_loss(r,tau,limit,valid,cfg)
    assert loss.item()==pytest.approx(4+5.25*(1+rate))
    loss.backward();torch.testing.assert_close(r.recovery_slack.grad[0],torch.tensor([1.,2.])*(1+rate))
    assert rows[1]==0 and r.recovery_slack.grad[1].eq(0).all()
    r.recovery_mask.zero_()
    empty,_=recovery_projection_loss(r,tau,limit,valid,cfg)
    assert empty==0


def test_disabled_and_small_residual_parity():
    cfg,r,tau,limit,valid=fixture()
    old=replace(cfg,recovery_slack_huber_enabled=False)
    a,_=recovery_projection_loss(r,tau,limit,valid,old)
    b,_=recovery_projection_loss(r,tau,limit,valid,cfg,quadratic=True)
    assert a==b==22.5
    with torch.no_grad():r.recovery_slack[0].fill_(.5)
    a,_=recovery_projection_loss(r,tau,limit,valid,old)
    b,_=recovery_projection_loss(r,tau,limit,valid,cfg)
    torch.testing.assert_close(a,b)


def test_budget_identity_scaled_gradient_frozen_reference_and_weighted_ema():
    b=RecoveryLossBudget(HardPACTQPConfig(), 'cpu');b.begin(3)
    other=torch.tensor(4.,requires_grad=True);rec=torch.tensor(1.,requires_grad=True)
    total,m=b.apply(other,rec,2);assert m['scale']==1 and total==5
    total.backward();assert rec.grad==1 and other.grad==1
    rec=torch.tensor(8.,requires_grad=True)
    total,m=b.apply(torch.tensor(10.),rec,1)
    assert m['budget']==2 and m['applied']==2 and m['scale']==.25
    total.backward();assert rec.grad==.25
    b.finish();assert b.ema==6 and b.last_updated_iteration==3
    b.finish();assert b.ema==6
    b.begin(4);_,m=b.apply(torch.tensor(100.),torch.tensor(100.),1)
    assert m['budget']==3
    b.finish();torch.testing.assert_close(b.ema,torch.tensor(6.94))


def test_empty_invalid_zero_reference_and_resume():
    cfg=HardPACTQPConfig(recovery_loss_budget_reference_floor=0.)
    b=RecoveryLossBudget(cfg,'cpu');b.begin(0);b.finish();assert not b.initialized
    b.begin(1);b.apply(torch.tensor(float('nan')),torch.tensor(0.),2);b.finish()
    assert not b.initialized and b.last_updated_iteration==-1
    b.begin(2);total,m=b.apply(torch.tensor(0.),torch.tensor(0.),2)
    assert total==0 and torch.isfinite(m['scale']);b.finish()
    c=RecoveryLossBudget(cfg,'cpu');c.load_state_dict(deepcopy(b.state_dict()))
    for budget in (b,c):
        budget.begin(3);budget.apply(torch.tensor(4.),torch.tensor(20.),3);budget.finish()
    for k in b.state_dict():assert torch.equal(torch.as_tensor(b.state_dict()[k]),torch.as_tensor(c.state_dict()[k]))


def test_legacy_contract_narrow_and_new_flags_strict():
    saved={'algorithm':{'qp_update_mode':'curriculum_substeps','friction_coefficient':.4}}
    current=deepcopy(saved);current['algorithm'].update(recovery_loss_budget_enabled=True,recovery_slack_huber_enabled=True)
    migrated=legacy_recovery_contract(saved,current)
    assert migrated==current and 'recovery_loss_budget_enabled' not in saved['algorithm']
    current['algorithm']['friction_coefficient']=.9
    assert legacy_recovery_contract(saved,current)!=current
    assert legacy_recovery_contract(migrated,{'algorithm':dict(current['algorithm'],recovery_loss_budget_enabled=False)})==migrated


@pytest.mark.parametrize('kw',[{'recovery_slack_huber_delta':0.},
    {'recovery_loss_budget_ratio':float('nan')},{'recovery_loss_budget_ema_alpha':1.1},
    {'recovery_loss_budget_reference_floor':-1.}])
def test_invalid_config(kw):
    with pytest.raises(ValueError):HardPACTQPConfig(**kw)


def test_full_checkpoint_budget_and_legacy_enablement():
    from test_training_state_transfer import hard
    from rsl_rl.utils import training_checkpoint as tc
    r=hard();b=r.alg.recovery_loss_budget
    b.begin(16);b.apply(torch.tensor(4.),torch.tensor(9.),3);b.finish()
    KEY=tc.KEY
    data={KEY:tc.envelope(r)}
    target=hard();tc.validate(target,data);tc.restore_learning(target,data[KEY])
    assert target.alg.recovery_loss_budget.ema==4
    assert target.alg.recovery_loss_budget.last_updated_iteration==16
    legacy=deepcopy(data);legacy[KEY]['extra'].pop('recovery_loss_budget')
    with pytest.warns(RuntimeWarning,match='recovery-loss budget'):
        tc.validate(hard(),legacy)


def test_legacy_explicit_enablement_does_not_relax_other_contracts():
    from test_training_state_transfer import hard
    from rsl_rl.utils import training_checkpoint as tc
    old=hard();target=hard()
    options=asdict(old.alg.qp_config)
    old.train_cfg['algorithm']['hard_pact_qp']={k:v for k,v in options.items() if k not in DEFAULTS}
    target.train_cfg['algorithm']['hard_pact_qp']=dict(options,
        recovery_slack_huber_enabled=True,recovery_loss_budget_enabled=True)
    data={tc.KEY:tc.envelope(old)};data[tc.KEY]['extra'].pop('recovery_loss_budget')
    with pytest.warns(RuntimeWarning,match='recovery-loss budget'):tc.validate(target,data)
    target.train_cfg['algorithm']['hard_pact_qp']['friction_coefficient']=.123
    with pytest.raises(ValueError,match='contract'):tc.validate(target,data)


@pytest.mark.parametrize('variant',['full','qp_no_post','qp_imitation'])
def test_enabled_real_actor_assembly_and_ablation_masks(monkeypatch, variant):
    import test_hard_pact_rss as rss
    original_solver=rss.solver;original_make=rss.make_algorithm;algorithms=[]
    def make(**kwargs):
        alg=original_make(**kwargs);algorithms.append(alg);return alg
    monkeypatch.setattr(rss,'make_algorithm',make)
    monkeypatch.setattr(rss,'solver',lambda **kwargs:original_solver(**kwargs,
        recovery_slack_huber_enabled=True,recovery_loss_budget_enabled=True,
        tensorboard_diagnostics_interval=1))
    rss.test_tiny_real_ppo_update(variant)
    alg=algorithms[0]
    assert alg.recovery_loss_budget.last_completed==1
    assert alg.recovery_loss_budget.initialized
    metrics=alg.last_actor_loss_metrics
    assert torch.isfinite(metrics['Loss/actor_pinn/total_weighted'])
    assert 'Loss/actor_pinn/recovery_budget/applied' in metrics
