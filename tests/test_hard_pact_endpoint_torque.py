"""Hard endpoint PD torque uses the existing single-interval motion model."""
import pytest
import torch
from test_hard_pact_reduced_qp import inputs, solver
from rsl_rl.modules.hard_pact_control import execution_feedforward, effective_feedback_gains
from rsl_rl.modules.hard_pact_control import bounded_nominal_torque


@pytest.mark.parametrize('beta',[.5,1.])
def test_endpoint_algebra_recovery_and_native_pack(beta):
    d=inputs(2);d['dt'].fill_(.005)
    d['effective_kp']=torch.full_like(d['tau_nom'],50.)
    d['effective_kd']=torch.full_like(d['tau_nom'],2.)
    d['joint_velocity'].fill_(4.)
    qp=solver(endpoint_torque_constraints=True,torque_rate_constraint_weight=0,
              position_integration_coefficient=beta)
    m=qp._build(d);old=solver(torque_rate_constraint_weight=0,position_integration_coefficient=beta)._build(d)
    torch.testing.assert_close(m.physical_G[:,:68],old.physical_G)
    torch.testing.assert_close(m.physical_h[:,:68],old.physical_h)
    x=torch.cat((d['tau_nom'],d['force_pred_world'].flatten(1)),1).requires_grad_()
    a=(m.acceleration_map@x[...,None]).squeeze(-1)+m.acceleration_offset
    H=d['dt'];v=d['joint_velocity'];q=d['joint_position']
    qH=q+H*v+beta*H**2*a[:,6:];vH=v+H*a[:,6:]
    pd0=-d['effective_kp']*q-d['effective_kd']*v
    pdH=-d['effective_kp']*qH-d['effective_kd']*vH
    expected=x[:,:12]-pd0+pdH
    actual=(m.physical_G[:,68:80]@x[...,None]).squeeze(-1)-m.physical_h[:,68:80]+23.5
    torch.testing.assert_close(actual,expected)
    actual.square().sum().backward();assert x.grad.isfinite().all()
    recovery=qp._soft_joint_problem(m)
    assert recovery.Q.shape[-1]==36 and recovery.G.shape[1]==104
    torch.testing.assert_close(recovery.physical_G[:,68:92,:24],m.physical_G[:,68:92])
    assert recovery.physical_G[:,68:92,24:].eq(0).all()
    assert qp._cupiqp_native_pack(m)[0].shape[1]==68
    assert qp._cupiqp_native_pack(recovery)[0].shape[1]==68


@pytest.mark.parametrize('device,backend',[('cpu','qpth'),pytest.param('cuda','cupiqp',
    marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_initial_feasible_endpoint_violation_corrected(device,backend):
    d=inputs(1,device=device);d['dt'].fill_(.005);d['joint_velocity'].fill_(4.)
    d['effective_kp']=torch.full_like(d['tau_nom'],100.)
    d['effective_kd']=torch.full_like(d['tau_nom'],2.)
    d['tau_nom'].fill_(-23.)
    qp=solver(endpoint_torque_constraints=True,torque_rate_constraint_weight=0,qp_solver=backend,solver_dtype='float64')
    m=qp._build(d);x=torch.cat((d['tau_nom'],d['force_pred_world'].flatten(1)),1)
    assert ((m.physical_G[:,68:]@x[...,None]).squeeze(-1)-m.physical_h[:,68:]).max()>1
    d['tau_nom'].requires_grad_()
    out=qp.solve(**d,differentiable=True)
    assert out.differentiated_mask.all()
    solved=torch.cat((out.tau_safe,out.force_world.flatten(1)),1)
    assert ((m.physical_G[:,68:]@solved[...,None]).squeeze(-1)-m.physical_h[:,68:]).max()<1e-5
    out.tau_safe.sum().backward();assert d['tau_nom'].grad.isfinite().all()


@pytest.mark.parametrize('alpha',[0.,.5,1.])
def test_execution_scaled_reconstruction_and_gradients(alpha):
    z=torch.zeros(2,12,dtype=torch.float64)
    p={k:torch.full_like(z,v) for k,v in dict(control_kp=30.,control_kd=2.,
        control_motor_strength=.8,control_feedback_weight=.6,control_feedforward_weight=.4).items()}
    qd=(z+.2).requires_grad_();selected=(z+5*alpha).requires_grad_()
    total,command,physical,available=execution_feedforward(selected,qd,z,z+.3,p)
    torch.testing.assert_close(total,selected);assert available.all()
    total.sum().backward();assert selected.grad.isfinite().all()
    torch.testing.assert_close(effective_feedback_gains(p)['effective_kp'],z+14.4)
    p['control_feedforward_weight'].zero_()
    total,command,_,available=execution_feedforward(selected,qd,z,z,p)
    torch.testing.assert_close(total,selected);assert not available.any() and command.eq(0).all()


@pytest.mark.parametrize('mode,count',[('every_substep',4),('random_one_substep',1)])
def test_endpoint_control_replay_anchor_and_backward(mode,count):
    from dataclasses import replace
    from test_hard_pact_qp_modes import fixture
    task,heads,qp,solve,counts,q,v,quat,_=fixture(mode)
    qp.cfg=replace(qp.cfg,endpoint_torque_constraints=True)
    p={k:torch.full_like(q,value) for k,value in dict(control_kp=3.,control_kd=1.,
        control_motor_strength=.8,control_feedback_weight=.6,control_feedforward_weight=.4,
        control_torque_limits=23.5).items()}
    task._hard_pact_control_parameters=p
    for _ in range(2):
        task._begin_qp_interval()
        for k in range(4):
            q.fill_(k*.005);v.fill_(k*.001)
            task._hard_pact_bounded_nominal_torque=bounded_nominal_torque(task._hard_pact_q_d,task._hard_pact_tau_ff,q,v,p)
            task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(8,6))
        assert task._qp_interval_solve_count.eq(count).all()
        packet=task._qp_sampled_transition
        d=inputs(8,torch.float32);qd=task._hard_pact_q_d.clone().requires_grad_()
        ff=task._hard_pact_tau_ff.clone().requires_grad_()
        d.update(joint_position=packet['sampled_qp_q'][:,7:],joint_velocity=packet['sampled_qp_v'][:,6:],
            previous_torque=packet['sampled_qp_previous_torque'],
            force_pred_world=packet['sampled_qp_rollout_grf_world'].reshape(8,4,3),
            contact_probability=packet['sampled_qp_rollout_contact_probability'])
        d['tau_nom']=bounded_nominal_torque(qd,ff,d['joint_position'],d['joint_velocity'],p)
        d.update(effective_feedback_gains(p))
        out=solve(differentiable=True,**d)
        torch.testing.assert_close(out.tau_safe,packet['sampled_qp_safe_torque'],atol=1e-7,rtol=1e-6)
        out.tau_safe.sum().backward()
        assert all(t.grad.isfinite().all() and t.grad.abs().sum()>0 for t in (qd,ff))
    task.reset_idx(torch.tensor([1]));assert task._hard_pact_execution_feedforward_command[1].eq(0).all()
