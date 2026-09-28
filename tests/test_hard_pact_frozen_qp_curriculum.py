from dataclasses import replace
import torch
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig, projection_loss
from rsl_rl.algorithms.hard_pact_qp_curriculum import QPCurriculum


def cfg(**kw):
    return replace(HardPACTQPConfig(),**dict(dict(objective_curriculum_enabled=True,
        warmup_iterations=200,objective_curriculum_ema_alpha=1.,
        objective_curriculum_start=200,objective_curriculum_step_interval=1,
        objective_curriculum_recovery_iterations=3),**kw))


def tick(s,i,value=.9,count=1):
    s.begin(i);s.finish(i,value,count)


def test_frozen_reference_decline_and_sustained_recovery():
    s=QPCurriculum(cfg())
    for i in range(200):tick(s,i)
    s.begin(200)
    assert s.baseline_frozen and s.pre_qp_reference==.9
    threshold=s.threshold
    for i in range(200,450):tick(s,i,.2)
    assert s.threshold==threshold and s.progress==0 and len(s.history)==200
    tick(s,450);tick(s,451)
    assert s.progress==0 and s.recovery_counter==2
    tick(s,452,None);assert s.recovery_counter==0
    for i in range(453,456):tick(s,i)
    assert s.progress==.05 and s.recovery_counter==0
    tick(s,455);assert s.progress==.05  # once per completed iteration


def test_freeze_at_activation_even_when_alpha_zero_and_resume():
    c=cfg(warmup_iterations=2,objective_curriculum_start=2,correction_ramp_enabled=True)
    s=QPCurriculum(c)
    tick(s,0);tick(s,1)
    assert s.begin(2)[1]==0 and s.baseline_frozen
    tick(s,2)
    clone=QPCurriculum(c);clone.load_state_dict(s.state_dict())
    for i,p in [(3,.9),(4,.9),(5,float('nan')),(6,.9)]:
        assert s.begin(i)==clone.begin(i)
        s.finish(i,p,1);clone.finish(i,p,1)
        assert s.state_dict()==clone.state_dict()


def test_old_checkpoint_preserves_progress_but_requires_baseline():
    c=cfg(warmup_iterations=2,objective_curriculum_start=2)
    old=dict(version=1,origin=2,start=2,progress=.4,ema=.2,threshold=.18,
             last_iteration=900,last_step=899,history=[.2]*200)
    s=QPCurriculum(c);s.load_state_dict(old)
    for i in range(901,1000):tick(s,i,.95)
    assert s.progress==.4 and s.pre_qp_reference is None and s.block_reason==3
    override=QPCurriculum(replace(c,objective_curriculum_baseline_override=.9))
    override.load_state_dict(s.state_dict())
    for i in range(1000,1003):tick(override,i,.95)
    assert override.progress==.45


def test_outer_coefficient_independent_and_zero_only_removes_explicit_stance():
    from test_hard_pact_reduced_qp import solver,inputs
    q=solver();d=inputs(1);d['foot_jacobians'][:,:,0,6]=1
    a=q._build(d);q.cfg=replace(q.cfg,projection_contact_acceleration_weight=0)
    b=q._build(d)
    assert torch.equal(a.Q,b.Q) and torch.equal(a.p,b.p)
    tau=torch.ones(1,12,requires_grad=True);acc=torch.ones(1,18,requires_grad=True)
    def loss(weight):
        parts={}
        value=projection_loss(tau,torch.zeros_like(tau),torch.ones(12),torch.tensor([True]),torch.tensor([True]),
            qdd=acc,foot_jacobians=d['foot_jacobians'].float(),foot_acceleration_bias=torch.zeros(1,4,3),
            stance_mask=torch.ones(1,4,dtype=torch.bool),contact_weight=weight,component_log=parts)
        return value,parts
    value,parts=loss(.1)
    torch.testing.assert_close(value,parts['torque'].mean()+.1*parts['stance'].mean())
    zero,_=loss(0)
    assert torch.autograd.grad(zero,acc,allow_unused=True)[0] is None
    # Inner stance objective still changes assembly independently of outer zero.
    q.cfg=replace(q.cfg,contact_acceleration_weight=2)
    assert not torch.equal(q._build(d).Q,b.Q)
