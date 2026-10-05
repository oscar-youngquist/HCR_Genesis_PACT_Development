"""Two bounded command-pair checks: allocation and complete hold/replay model."""
from dataclasses import replace
import torch
from rsl_rl.modules.hard_pact_control import (allocate_command_correction,
    command_pair_gains, command_pair_inputs, held_command_model,
    requested_torque_components)
from test_hard_pact_reduced_qp import inputs, solver


def parameters(ref):
    return {key:torch.full_like(ref,value) for key,value in dict(
        control_kp=30.,control_kd=2.,control_motor_strength=.8,
        control_feedback_weight=.7,control_feedforward_weight=.6,
        control_torque_limits=23.5).items()}


def test_allocation_preserves_physical_torque_and_gradients():
    u=torch.linspace(-3,3,24,dtype=torch.float64).reshape(2,12).requires_grad_()
    p=parameters(u);kp,kd,ff=command_pair_gains(p)
    for share in (0.,.3,1.):
        dq,dff,rho=allocate_command_correction(u,p,share)
        torch.testing.assert_close(kp*dq+ff*dff,u,rtol=1e-14,atol=1e-14)
        torch.testing.assert_close(torch.autograd.grad((kp*dq+ff*dff).sum(),u)[0],torch.ones_like(u))
        if 0<share<1:
            torch.testing.assert_close((kp*dq).square()/share+(ff*dff).square()/(1-share),u.square())
    for disabled in ('control_feedback_weight','control_feedforward_weight'):
        pp={k:v.clone() for k,v in p.items()};pp[disabled].zero_()
        dq,dff,_=allocate_command_correction(u,pp,.3);k,_,f=command_pair_gains(pp)
        torch.testing.assert_close(k*dq+f*dff,u)
    p['control_feedback_weight'].zero_();p['control_feedforward_weight'].zero_()
    dq,dff,_=allocate_command_correction(u,p,.3)
    assert dq.eq(0).all() and dff.eq(0).all()
    assert torch.autograd.grad((dq+dff).sum(),u)[0].eq(0).all()


def test_hold_endpoint_bounds_replay_and_backward():
    for beta in (1.,.5):
        qp=solver(qp_update_mode='command_pair',torque_rate_constraint_weight=0.,
                  position_integration_coefficient=beta,diagnostics_level='minimal')
        d=inputs();d['dt'].fill_(.005);d['joint_velocity'].fill_(.4)
        p=parameters(d['tau_nom']);desired=torch.full_like(d['tau_nom'],.1,requires_grad=True)
        feed=torch.ones_like(desired,requires_grad=True)
        d.update(command_pair_inputs(desired,feed,d['joint_position'],d['joint_velocity'],p))
        d['tau_nom']=d['command_nominal'];m=qp._build(d)
        assert m.Q.shape==(2,24,24) and m.G.shape==(2,68,24)
        recovery=qp._soft_joint_problem(m)
        assert recovery.Q.shape==(2,36,36) and recovery.G.shape==(2,80,36)
        assert qp._cupiqp_native_pack(recovery)[0].shape==(2,44,36)
        assert (torch.linalg.eigvalsh(m.Q)>0).all()
        out=qp.solve(differentiable=True,**d)
        assert out.differentiated_mask.all()
        dq,dff,_=allocate_command_correction(out.tau_safe-d['command_nominal'],p,.3)
        k,b,_=command_pair_gains(p)
        t,c,drift,decay=held_command_model(d['dt'],beta,k,b,d['joint_velocity'])
        q=d['joint_position'].clone();v=d['joint_velocity'].clone();torques=[]
        for step in range(4):
            torque=requested_torque_components(desired+dq,feed+dff,q,v,p)[0]
            torques.append(torque)
            expected=out.tau_safe-drift[:,step]-decay[:,step]*out.qdd[:,6:]
            torch.testing.assert_close(torque,expected,rtol=1e-10,atol=1e-10)
            q=q+d['dt']*v+beta*d['dt'].square()*out.qdd[:,6:]
            v=v+d['dt']*out.qdd[:,6:]
            torch.testing.assert_close(q,d['joint_position']+t[:,step+1]*d['joint_velocity']+c[:,step+1]*out.qdd[:,6:])
            assert (q.abs()<=2).all() and (v.abs()<=30).all() and (torque.abs()<=23.5).all()
        # One-step PD dynamics use the initial application, not hold-average.
        expected=torch.cat((torch.zeros_like(v[:,:6]),torques[0]),1)
        torch.testing.assert_close(out.qdd,expected,rtol=1e-9,atol=1e-9)
        lo=torch.maximum((-30-d['joint_velocity'])/t[:,1],
            (-2-d['joint_position']-t[:,1]*d['joint_velocity'])/c[:,1])
        torch.testing.assert_close(m.qdd_lower,lo)
        replay=qp.solve(differentiable=True,**d)
        torch.testing.assert_close(replay.tau_safe,out.tau_safe)
        (dq.square().sum()+dff.square().sum()+out.tau_safe.sum()).backward()
        assert all(x.grad.isfinite().all() and x.grad.abs().sum()>0 for x in (desired,feed))
    # Recovery softens only the all-endpoint joint envelope; failed mechanics
    # retain detached actuator fallback. Both use the new u-coordinate layout.
    d=inputs();d['joint_position'][0].fill_(2.01)
    p=parameters(d['tau_nom']);desired=torch.zeros_like(d['tau_nom'],requires_grad=True)
    d.update(command_pair_inputs(desired,torch.zeros_like(desired),d['joint_position'],d['joint_velocity'],p))
    d['tau_nom']=d['command_nominal'];d['mass_matrix'][1].fill_(float('nan'))
    recovered=qp.solve(differentiable=True,**d)
    assert recovered.stage.tolist()==[1,2]
    assert recovered.recovery_slack[0].max()>0
    assert recovered.tau_safe.abs().max()<=23.5
    recovered.tau_safe.sum().backward()
    assert desired.grad.isfinite().all() and desired.grad[1].eq(0).all()
    from legged_gym.envs.go2.go2_hard_pact.deployment import qp_update_contract,validate_qp_deployment_contract
    contract=qp_update_contract('command_pair',4,qp_config=qp.cfg)
    assert contract['physics_substep_anchors']==[0,1,2,3]
    validate_qp_deployment_contract({'schema_version':17,'qp_update':contract})
    # Real control callback: reset, three intervals, one invocation and no
    # extra neural forwards per interval; fresh measured states at each PD.
    from test_hard_pact_qp_modes import fixture
    task,heads,qp,real,counts,q,v,quat,_=fixture('every_substep',n=2)
    qp.cfg=replace(qp.cfg,qp_update_mode='command_pair',torque_rate_constraint_weight=0.,command_pair_second_solve_fraction=0.)
    task.cfg.control.clip_torque_rate_without_qp=False
    task._hard_pact_control_parameters=parameters(q)
    task._hard_pact_bounded_nominal_torque=q.clone()
    for iteration in range(3):
        task._begin_qp_interval()
        for step in range(4):
            q.fill_(.001*step);v.fill_(.002*step)
            task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(2,6))
            expected=requested_torque_components(task._hard_pact_q_d+task._qp_command_delta_q,
                task._hard_pact_tau_ff+task._qp_command_delta_ff,q,v,task._hard_pact_control_parameters)[0]
            torch.testing.assert_close(task.simulator._torques,expected.clamp(-23.5,23.5))
        assert counts.eq(iteration+1).all() and task._qp_sampled_substep_index.eq(0).all()
    assert heads.calls==3
    task.reset_idx(torch.tensor([0]))
    assert task._qp_command_delta_q[0].eq(0).all() and not task._qp_command_accepted[0]


def test_sampling_horizons_adaptive_allocation_and_loss_masks():
    from rsl_rl.modules.hard_pact_control import command_pair_sample,allocation_deviation_loss
    from rsl_rl.algorithms.hard_pact_qp import qp_substep_mask
    for fraction in (0.,.5,1.):
        selected=command_pair_sample(10,fraction,2,'cpu')
        dispatch=[qp_substep_mask('command_pair',k,selected) for k in range(4)]
        assert dispatch[0].all() and dispatch[2].sum()==round(10*fraction)
        assert not dispatch[1].any() and not dispatch[3].any()
        assert torch.stack(dispatch).sum()==10+round(10*fraction)
        assert all(qp_substep_mask('command_pair_every_substep',k,selected).all() for k in range(4))
    qp=solver(qp_update_mode='command_pair',torque_rate_constraint_weight=0.,
        position_command_lower=(-.1,)*12,position_command_upper=(.1,)*12)
    d=inputs();p=parameters(d['tau_nom']);target=torch.full_like(d['tau_nom'],.099,requires_grad=True)
    ff=torch.zeros_like(target,requires_grad=True)
    bounds=qp.command_inputs(target,ff,d['joint_position'],d['joint_velocity'],p)
    u=torch.ones_like(target,requires_grad=True)
    dq,dff,_=allocate_command_correction(u,p,.2,bounds);k,_,f=command_pair_gains(p)
    torch.testing.assert_close(k*dq+f*dff,u)
    assert (target+dq<=.1+1e-12).all()
    loss=allocation_deviation_loss(u,k*dq,torch.ones(12),.2,torch.tensor([True,False]))
    loss.backward()
    assert target.grad[0].abs().sum()>0 and target.grad[1].eq(0).all()
    assert u.grad.isfinite().all() and u.grad[0].abs().sum()>0 and u.grad[1].eq(0).all()
    assert allocation_deviation_loss(u,k*dq,torch.ones(12),0.,torch.ones(2,dtype=torch.bool))==0
    assert allocation_deviation_loss(u,k*dq,torch.ones(12),.2,torch.zeros(2,dtype=torch.bool))==0
    d.update(bounds);d['tau_nom']=bounds['command_nominal']
    m=qp._build(d);dt=d['dt'].clone()
    qp.cfg=replace(qp.cfg,constraint_prediction_horizon_s=.01)
    changed=qp._build(d)
    assert not torch.equal(m.qdd_lower,changed.qdd_lower)
    assert torch.equal(m.Q,changed.Q) and torch.equal(d['dt'],dt)
    # Forced positive correction saturates the POSITION share, not an extra
    # per-branch motor limit; feedforward receives the residual. Gradients
    # include both solver u and the command-dependent allocation boundary.
    dd=inputs();dd['tau_nom'].zero_();dd['joint_position'].fill_(1.99)
    bounds=qp.command_inputs(target,ff,dd['joint_position'],dd['joint_velocity'],p)
    dd.update(bounds);dd['tau_nom']=bounds['command_nominal']
    out=qp.solve(differentiable=True,**dd)
    correction=out.tau_safe-bounds['command_nominal']
    aq,aff,_=allocate_command_correction(correction,p,.2,bounds)
    objective=allocation_deviation_loss(correction,k*aq,qp.torque_limits,.2,
        out.differentiated_mask|out.recovery_mask)
    gradients=torch.autograd.grad(objective,(target,ff),allow_unused=True)
    assert all(g is not None and g.isfinite().all() for g in gradients)
    assert gradients[0].abs().sum()>0
    # Shared later index is balanced across three intervals; later corrections
    # replace originals and the replay state belongs to the actual last solve.
    from test_hard_pact_qp_modes import fixture
    task,heads,qp,real,counts,q,v,quat,_=fixture('every_substep',n=8)
    qp.cfg=replace(qp.cfg,qp_update_mode='command_pair',torque_rate_constraint_weight=0.)
    task.cfg.control.clip_torque_rate_without_qp=False
    task._hard_pact_control_parameters=parameters(q);task._hard_pact_bounded_nominal_torque=q.clone()
    later=[]
    for iteration in range(3):
        task._begin_qp_interval();selection=task._qp_sampled_substep_index.clone()
        later.append(int(selection.max()))
        for step in range(4):
            q.fill_(.001*step);v.fill_(.002*step)
            task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(8,6))
            expected=requested_torque_components(task._hard_pact_q_d+task._qp_command_delta_q,
                task._hard_pact_tau_ff+task._qp_command_delta_ff,q,v,task._hard_pact_control_parameters)[0]
            torch.testing.assert_close(task.simulator._torques,expected.clamp(-23.5,23.5))
        assert task._qp_interval_solve_count.sum()==12
        torch.testing.assert_close(task._qp_sampled_transition['sampled_qp_q'][:,7],selection.float()*.001)
    assert sorted(later)==[1,2,3] and heads.calls==3
