"""Bounded control/replay smoke and rolling-evidence regressions; no training."""
from dataclasses import replace
from copy import deepcopy
import pytest
import torch
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig
from rsl_rl.algorithms.hard_pact_qp_curriculum import execution_torque, blended_candidate_acceleration, correction_limits_for_joints
from rsl_rl.algorithms.hard_pact_intervention_curriculum import InterventionCurriculum


def schedule(interval):
    return InterventionCurriculum(HardPACTQPConfig(warmup_iterations=0,correction_ramp_enabled=False,
        intervention_settling_iterations=0,intervention_window_iterations=4,
        intervention_gate_check_interval=interval,intervention_ramp_iterations=100))


def tick(s,i,failures=2,exposure=20):
    s.begin(i);s.finish(i,.9,10,failures,exposure)


def test_rolling_weighted_reference_cadence_duplicate_resume_and_gap():
    s=schedule(1)
    for i in range(4):tick(s,i)
    assert s.gate and s.failure_reference==.1
    tick(s,4,0,100)
    assert s.failure_rate==6/160 and s.failure_reference==.1  # no overlapping tightening
    restored=schedule(1);restored.load_state_dict(s.state_dict())
    state=deepcopy(s.state_dict());tick(s,4);assert s.state_dict()==state
    for i in range(5,8):
        tick(s,i,0,100);tick(restored,i,0,100)
        assert s.state_dict()==restored.state_dict()
    assert s.failure_reference==0 and len(s.evidence)==4
    tick(s,9);assert not s.gate and not s.evidence
    for i in range(10,13):tick(s,i,0,100)
    assert not s.gate
    tick(s,13,0,100);assert s.gate


def test_check_frequency_does_not_change_open_ramp_speed_and_legacy_recollects():
    schedules=[schedule(i) for i in (1,2,5)]
    for i in range(20):
        for s in schedules:tick(s,i)
    assert all(s.progress==pytest.approx(.16) for s in schedules)
    old=schedules[0].state_dict();old['version']=2
    for key in ('check_interval','evidence','last_check','check_count'):old.pop(key)
    target=schedule(5);target.load_state_dict(old)
    assert target.progress==schedules[0].progress and not target.gate and not target.evidence


def test_initial_gap_requires_exactly_one_consecutive_window():
    s=schedule(5)
    tick(s,0);s.finish(1,None,None,None,None)
    for i in range(2,5):tick(s,i)
    assert not s.gate
    tick(s,5);assert s.gate and len(s.evidence)==4
    for interval in (0,-1,1.5,float('nan')):
        with pytest.raises(ValueError):schedule(interval)


def test_named_mapping_cached_and_clamp_gradients_primary_recovery_fallback():
    from test_hard_pact_reduced_qp import solver
    cfg=HardPACTQPConfig()
    assert correction_limits_for_joints(['RL_calf_joint','FR_hip_joint','FL_thigh_joint'],cfg)==[18,12,12]
    with pytest.raises(ValueError):correction_limits_for_joints(['unknown'],cfg)
    qp=solver(execution_correction_clip_enabled=True)
    base=torch.zeros(3,12,requires_grad=True);candidate=torch.full((3,12),40.,requires_grad=True)
    cap=qp.execution_correction_limits(base)
    assert cap.data_ptr()==qp.execution_correction_limits(base).data_ptr()
    accepted=torch.tensor([True,True,False]);alpha=torch.tensor([1.,.5,0.])
    actual=execution_torque(base,candidate,accepted,alpha,cap,torch.full((12,),23.5))
    torch.testing.assert_close(actual[:2],cap.expand(2,-1))
    torch.testing.assert_close(actual[2],candidate[2])  # deterministic fallback untouched
    actual[:2].sum().backward()
    assert candidate.grad.eq(0).all() and base.grad[:2].eq(1).all()
    assert torch.equal(execution_torque(base,candidate,accepted,1.),candidate)


def test_real_control_smoke_anchor_replay_and_no_accumulation():
    from test_hard_pact_qp_modes import fixture
    task,heads,qp,solve,counts,q,v,quat,_=fixture('every_substep',4)
    qp.cfg=replace(qp.cfg,execution_correction_clip_enabled=True)
    task._qp_execution_alpha=.5
    task._begin_qp_interval()
    for k in range(4):
        q.fill_(k*.001)
        with torch.no_grad():task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(4,6))
        assert torch.isfinite(task.simulator._torques).all()
        assert (task.simulator._torques.abs()<=qp.torque_limits).all()
    assert counts.tolist()==[4]*4 and heads.calls==1
    sample=task._qp_sampled_transition
    assert sample['sampled_qp_execution_alpha'].eq(.5).all()
    assert sample['sampled_qp_base_rate_clip'].all()


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_rollout_inference_cache_reused_by_repeated_replay_backward(device):
    from test_hard_pact_reduced_qp import solver
    qp=solver(execution_correction_clip_enabled=True)
    with torch.inference_mode():
        rollout=torch.zeros(3,12,device=device)
        cap=qp.execution_correction_limits(rollout)
    for _ in range(2):
        base=torch.zeros(3,12,device=device,requires_grad=True)
        candidate=torch.full_like(base,4.,requires_grad=True)
        accepted=torch.tensor([True,True,False],device=device)
        alpha=base.new_tensor([1.,.5,0.])
        cached=qp.execution_correction_limits(base)
        assert cached.data_ptr()==cap.data_ptr()
        output=execution_torque(base,candidate,accepted,alpha,cached,base.new_full((12,),23.5))
        output[:2].sum().backward()
        torch.testing.assert_close(candidate.grad[:,0],base.new_tensor([1.,.5,0.]))
        torch.testing.assert_close(base.grad[:,0],base.new_tensor([0.,.5,0.]))
        assert not torch.is_inference(cached)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA/cuPIQP unavailable')
def test_cupiqp_capped_replay_mixed_stages_and_backward():
    from test_hard_pact_reduced_qp import inputs,solver
    d=inputs(3,device='cuda');d['joint_position'][1,0]=2.4;d['joint_position'][2,0]=float('nan')
    d['tau_nom'].requires_grad_()
    qp=solver(qp_solver='cupiqp',solver_dtype='float64',rollout_max_iter=100,ppo_max_iter=100,
        execution_correction_clip_enabled=True,diagnostics_level='physical',endpoint_torque_constraints=True)
    d['effective_kp']=torch.zeros_like(d['tau_nom']);d['effective_kd']=torch.zeros_like(d['tau_nom'])
    reference=qp.solve(differentiable=True,**d)
    with torch.inference_mode():
        qp.execution_correction_limits(d['tau_nom'])
    base=d['tau_nom'];alpha=base.new_tensor([[1.],[1.],[0.]])
    out=qp.solve(differentiable=True,execution_base=base,execution_alpha=alpha,**d)
    assert out.stage.tolist()==[0,1,2]
    torch.testing.assert_close(out.tau_safe,reference.tau_safe)
    torch.testing.assert_close(out.qdd,reference.qdd)
    # Execution clipping must not cap either stage's projection supervision.
    from rsl_rl.algorithms.hard_pact_qp import projection_loss, recovery_projection_loss
    valid=torch.ones(3,dtype=torch.bool,device=base.device)
    losses=[]
    for result in (reference,out):
        primary=projection_loss(result.tau_safe,base,qp.torque_limits.to(base),
            valid,result.differentiated_mask,qdd=result.qdd,
            foot_jacobians=d['foot_jacobians'],foot_acceleration_bias=d['foot_acceleration_bias'],
            stance_mask=d['contact_probability']>=qp.cfg.contact_threshold,
            contact_weight=.1)
        recovery,_=recovery_projection_loss(result,base,qp.torque_limits.to(base),valid,qp.cfg)
        losses.append(primary+recovery)
    torch.testing.assert_close(losses[0],losses[1])
    gradients=[torch.autograd.grad(loss,base,retain_graph=True)[0] for loss in losses]
    torch.testing.assert_close(gradients[0],gradients[1])
    accepted=out.differentiated_mask|out.recovery_mask
    selected=execution_torque(base,out.tau_safe,accepted,alpha,qp.execution_correction_limits(base),qp.torque_limits.to(base))
    torch.testing.assert_close(out.execution_qdd[:2,6:],selected[:2])
    assert (selected[1]-out.tau_safe[1]).abs().max()>1  # active recovery cap, not just identical formulas
    assert out.execution_qdd[2].eq(0).all()
    out.execution_qdd.square().sum().backward()
    assert base.grad.isfinite().all() and base.grad[2].eq(0).all() and base.grad[:2].abs().sum()>0
    assert any('execution_recovery' in k for k in qp.iteration_diagnostics['ppo'].sums)
