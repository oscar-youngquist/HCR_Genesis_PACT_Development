"""Focused position/velocity envelope and independent recovery slew slack checks."""
from unittest.mock import patch
import pytest
import torch
from test_hard_pact_reduced_qp import inputs, solver
from test_hard_pact_qp_modes import fixture
from rsl_rl.algorithms.hard_pact_qp import recovery_projection_loss, project_nominal_torque
from rsl_rl.algorithms.hard_pact_qp_backends import QPBackendResult


def test_two_slack_algebra_native_bounds_and_outer_gradients():
    qp=solver(torque_rate_limit_nm_s=100.)
    d=inputs(2); d['joint_position'][:,0]=2.01
    m=qp._build(d); r=qp._soft_joint_problem(m)
    assert r.p.shape==(2,48) and r.G.shape==(2,116,48)
    assert torch.linalg.eigvalsh(r.Q).min()>0
    # No independent 100 rad/s² cap: only position/velocity set the envelope.
    assert m.qdd_upper[0,1]==3000
    assert r.tau_upper.eq(23.5).all() and m.tau_upper.eq(1.).all()
    x=torch.zeros_like(r.p); x[:,:12]=5.; x[:,24:36]=105.; x[:,36:48]=4.
    assert qp._certificate(r,x/r.variable_scale,1e-9)[0].all()
    G,h,lower,upper=qp._cupiqp_native_pack(r)
    z=x/r.variable_scale
    assert (G@z[...,None]).squeeze(-1).sub(h).max()<1e-9
    assert (z>=lower).all() and (z<=upper).all()
    x[:,36:48]=0
    assert not qp._certificate(r,x/r.variable_scale,1e-9)[0].any()
    x[:,36:48]=100; x[:,0]=24
    assert not qp._certificate(r,x/r.variable_scale,1e-9)[0].any()
    from types import SimpleNamespace
    joint=torch.ones(2,12,dtype=torch.float64,requires_grad=True)
    rate=torch.ones_like(joint,requires_grad=True)
    tau=torch.ones_like(joint,requires_grad=True)
    result=SimpleNamespace(recovery_mask=torch.tensor([True,False]),recovery_slack=joint,
                           recovery_rate_slack=rate,tau_safe=tau)
    loss,_=recovery_projection_loss(result,d['tau_nom'],qp.torque_limits,
                                    torch.ones(2,dtype=torch.bool),qp.cfg)
    loss.backward()
    for value in (joint,rate,tau):
        assert value.grad[0].abs().sum()>0 and value.grad[1].eq(0).all()


def test_real_recovery_solve_and_replay_acceptance():
    qp=solver(torque_rate_limit_nm_s=10.,soft_rate_recovery_weight=.1)
    d=inputs(1);d['joint_position'][:,0]=2.0005
    d['tau_nom'].requires_grad_()
    backend=qp._backend_solve
    def force_recovery(m):
        if m.p.shape[1]==24: raise RuntimeError('primary rejected')
        return backend(m)
    with patch.object(qp,'_backend_solve',side_effect=force_recovery):
        out=qp.solve(differentiable=True,**d)
        rollout=qp.solve(differentiable=False,**d)
    assert out.stage.eq(1).all(), out.diagnostics
    torch.testing.assert_close(out.tau_safe,rollout.tau_safe)
    assert out.tau_safe.abs().max()<=23.5
    assert out.recovery_rate_slack.max()>0
    loss,_=recovery_projection_loss(out,d['tau_nom'],qp.torque_limits,torch.ones(1,dtype=torch.bool),qp.cfg)
    loss.backward()
    assert d['tau_nom'].grad.isfinite().all() and d['tau_nom'].grad.abs().sum()>0


@pytest.mark.parametrize('clip',[False,True])
def test_unsolved_substeps_and_recovery_bypass(clip):
    task,_,qp,_,_,q,v,quat,_=fixture('random_one_substep')
    task.cfg.control.clip_torque_rate_without_qp=clip
    task._hard_pact_q_d.fill_(2.)
    task._begin_qp_interval()
    task._qp_sampled_substep_index.zero_()
    # Certified recovery intentionally exceeds rate but not absolute torque.
    def candidate(m):
        if m.p.shape[-1]==24: raise RuntimeError('primary rejected')
        x=torch.zeros_like(m.p);x[:,:12]=5; x[:,36:]=4.9
        return QPBackendResult(x/m.variable_scale)
    with patch.object(qp,'_backend_solve',side_effect=candidate):
        task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(8,6))
    torch.testing.assert_close(task.simulator._torques,torch.full_like(q,5.))
    # No QP at k=1. Fresh PD is 6 Nm, centered on actual previous 5 Nm.
    task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(8,6))
    torch.testing.assert_close(task.simulator._torques,torch.full_like(q,5.1 if clip else 6.))
    task.reset_idx(torch.tensor([0,2]))
    assert task._hard_pact_previous_substep_torque[[0,2]].eq(0).all()


def test_fallback_always_rate_bounded_and_nonfinite_safe():
    qp=solver(torque_rate_limit_nm_s=10.)
    d=inputs(2);d['tau_nom'].fill_(20)
    with patch.object(qp,'_backend_solve',side_effect=RuntimeError('all fail')):
        out=qp.solve(**d)
    assert out.stage.eq(2).all() and not out.differentiated_mask.any()
    torch.testing.assert_close(out.tau_safe,torch.full_like(out.tau_safe,.1))
    previous=torch.full((1,12),float('nan'))
    assert project_nominal_torque(torch.full_like(previous,float('inf')),previous,
                                 torch.ones(12)*23.5,10.,.01).isfinite().all()


@pytest.mark.parametrize('pos',[False,True])
@pytest.mark.parametrize('state',['disabled','warmup'])
def test_no_qp_control_path_history_reset_and_disabled_parity(pos,state):
    from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
    from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos import Go2HardPACTPos
    task,_,qp,_,_,q,_,_,_=fixture('random_one_substep')
    task.simulator.torque_limits=qp.torque_limits.to(q)
    del task._hard_pact_rollout_qp  # This path cannot require a solver/mechanics.
    task._hard_pact_rollout_qp_enabled=False
    task._hard_pact_policy_context_ready=(state=='warmup')
    apply=(Go2HardPACTPos if pos else Go2HardPACT)._apply_non_qp_torque_rate_clip
    for k in range(3):
        task.simulator._torques=torch.full_like(q,20.)
        apply(task)
        torch.testing.assert_close(task.simulator._torques,torch.full_like(q,.1*(k+1)))
        task._hard_pact_previous_substep_torque.copy_(task.simulator._torques)
    task.reset_idx(torch.tensor([0]))
    task.simulator._torques=torch.full_like(q,20.)
    apply(task)
    torch.testing.assert_close(task.simulator._torques[0],torch.full_like(q[0],.1))
    torch.testing.assert_close(task.simulator._torques[1],torch.full_like(q[1],.4))
    task.cfg.control.clip_torque_rate_without_qp=False
    task.simulator._torques=torch.full_like(q,20.)
    original=task.simulator._torques
    apply(task)
    assert task.simulator._torques is original
