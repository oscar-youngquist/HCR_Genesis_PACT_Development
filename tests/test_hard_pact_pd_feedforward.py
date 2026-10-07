"""Focused feedforward reconstruction, coupled affine prediction and recovery."""
import torch
from test_hard_pact_reduced_qp import inputs,solver
from test_hard_pact_command_pair import parameters
from rsl_rl.modules.hard_pact_control import (feedforward_for_total_torque,
    requested_torque_components,command_pair_gains)


def make_qp():
    return solver(qp_update_mode='command_pair_every_substep',
        command_correction_mode='feedforward_only',torque_rate_constraint_weight=0.)


def test_feedforward_reconstruction_blending_replacement_and_gradients():
    q=torch.zeros(2,12,dtype=torch.float64);v=q+.3;p=parameters(q)
    desired=torch.full_like(q,.1,requires_grad=True);ff=torch.full_like(q,2.,requires_grad=True)
    original=desired.detach().clone()
    nominal=requested_torque_components(desired,ff,q,v,p)[0]
    for alpha in (0.,.5,1.):
        selected=nominal+alpha*2
        command,available=feedforward_for_total_torque(selected,desired,q,v,p,ff)
        reconstructed=requested_torque_components(desired,command,q,v,p)[0]
        torch.testing.assert_close(reconstructed,selected,atol=1e-12,rtol=1e-12)
        assert available.all() and torch.equal(desired.detach(),original)
    # A later solve references the original target with NEW measured q/v;
    # substituting a previous correction does not add it a second time.
    command,_=feedforward_for_total_torque(nominal-1,desired,q+.01,v+.02,p,ff)
    reconstructed=requested_torque_components(desired,command,q+.01,v+.02,p)[0]
    torch.testing.assert_close(reconstructed,nominal-1)
    gradients=torch.autograd.grad(reconstructed.sum(),(desired,ff))
    assert all(g.isfinite().all() and g.abs().sum()>0 for g in gradients)
    p['control_feedforward_weight'].zero_()
    command,available=feedforward_for_total_torque(nominal,desired,q,v,p,ff)
    assert not available.any() and torch.equal(command,ff)


def test_affine_pd_maps_match_two_step_full_coupled_model_and_constraints():
    torch.manual_seed(14);qp=make_qp();d=inputs()
    R=torch.randn(2,18,18,dtype=torch.float64)
    d['mass_matrix']=R@R.transpose(1,2)+torch.eye(18,dtype=torch.float64)*3
    d['foot_jacobians']=torch.randn_like(d['foot_jacobians'])*.2
    d['wrench_pred_world']=torch.randn_like(d['wrench_pred_world'])
    d['bias']=torch.randn_like(d['bias']);d['joint_velocity'].fill_(.3)
    p=parameters(d['tau_nom'])
    d.update(qp.command_inputs(d['joint_position']+.1,torch.ones_like(d['tau_nom']),
        d['joint_position'],d['joint_velocity'],p));d['tau_nom']=d['command_nominal']
    m=qp._build(d);assert m.Q.shape==(2,24,24) and m.G.shape==(2,188,24)
    x=torch.randn(2,24,dtype=torch.float64)
    affine=(m.prediction_maps@x[:,None,:,None]).squeeze(-1)+m.prediction_offsets
    q=d['joint_position'].clone();v=d['joint_velocity'].clone()
    kp,kd,_=command_pair_gains(p);initial=d['command_nominal']+x[:,:12]
    ff=initial-(kp*(d['command_desired_position']-q)-kd*v)
    qs=[];vs=[];ts=[];h=.0025
    for step in range(3):
        torque=ff+kp*(d['command_desired_position']-q)-kd*v;ts.append(torque)
        if step==2:break
        g=torch.cat((torch.zeros(2,6,dtype=q.dtype),torque),1)
        g+=torch.einsum('bfkn,bfk->bn',d['foot_jacobians'],x[:,12:].reshape(2,4,3))
        g+=torch.einsum('bkn,bk->bn',d['base_jacobian'],d['wrench_pred_world'])
        a=torch.linalg.solve(d['mass_matrix'],(g-d['bias'])[...,None]).squeeze(-1)
        q=q+h*v+h*h*a[:,6:];v=v+h*a[:,6:];qs.append(q);vs.append(v)
    torch.testing.assert_close(affine,torch.stack(qs+vs+ts,1),atol=1e-11,rtol=1e-11)
    # Compare every hard row with its independently calculated physical value.
    residual=(m.physical_G@x[...,None]).squeeze(-1)-m.physical_h
    for k in range(2):
        t=(k+1)*h;start=24+48*k
        torch.testing.assert_close(residual[:,start:start+12],(qs[k]-2)/(t*t))
        torch.testing.assert_close(residual[:,start+24:start+36],(vs[k]-30)/t)
        torch.testing.assert_close(residual[:,140+24*k:152+24*k],ts[k+1]-23.5)
    recovery=qp._soft_joint_problem(m)
    assert recovery.G.shape==(2,200,36)
    assert recovery.physical_G[:,140:188,24:].eq(0).all()  # torque never softened
    assert qp._cupiqp_native_pack(recovery)[0].shape==(2,164,36)


def test_pd_recovery_torques_and_actor_gradients():
    qp=make_qp();d=inputs(3);d['joint_position'][1].fill_(2.01)
    d['mass_matrix'][2].fill_(float('nan'))
    p=parameters(d['tau_nom']);target=torch.full_like(d['tau_nom'],.01,requires_grad=True)
    ff=torch.full_like(target,.2,requires_grad=True)
    d.update(qp.command_inputs(target,ff,d['joint_position'],d['joint_velocity'],p))
    d['tau_nom']=d['command_nominal'];out=qp.solve(differentiable=True,**d)
    assert out.stage.tolist()==[0,1,2]
    m=qp._build({k:v[:2] for k,v in d.items()})
    x=torch.cat((out.tau_safe[:2]-d['command_nominal'][:2],out.force_world[:2].flatten(1)),1)
    pred=(m.prediction_maps@x[:,None,:,None]).squeeze(-1)+m.prediction_offsets
    assert pred[:,4:].abs().max()<=23.5+1e-6
    assert out.recovery_slack[1].max()>0
    assert out.diagnostics['soft_joint/original_joint_violation_max_rad_s2'][1]>0
    gradients=torch.autograd.grad(out.tau_safe.sum()+out.recovery_slack.sum(),(target,ff))
    assert all(g.isfinite().all() and g[0].abs().sum()>0 and g[2].eq(0).all() for g in gradients)


def test_execution_preserves_targets_and_replaces_blended_feedforward():
    from dataclasses import replace
    from test_hard_pact_qp_modes import fixture
    task,heads,qp,real,counts,q,v,quat,_=fixture('every_substep',n=2)
    qp.cfg=replace(qp.cfg,qp_update_mode='command_pair_every_substep',
        command_correction_mode='feedforward_only',torque_rate_constraint_weight=0.)
    task.cfg.control.clip_torque_rate_without_qp=False
    task._hard_pact_control_parameters=parameters(q);task._hard_pact_bounded_nominal_torque=q.clone()
    desired=task._hard_pact_q_d.clone();ff=task._hard_pact_tau_ff.clone()
    candidates=[]
    def capture(**kwargs):
        result=real(**kwargs);candidates.append(result);return result
    qp.solve=capture
    for alpha in (0.,.5,1.):
        task._qp_execution_alpha=alpha;task._begin_qp_interval()
        for step in range(4):
            q.fill_(.001*step);v.fill_(.002*step)
            nominal=requested_torque_components(desired,ff,q,v,task._hard_pact_control_parameters)[0].clamp(-23.5,23.5)
            task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(2,6))
            result=candidates[-1];accepted=result.differentiated_mask|result.recovery_mask
            expected=torch.where(accepted[:,None],nominal+alpha*(result.tau_safe-nominal),result.tau_safe)
            torch.testing.assert_close(task.simulator._torques,expected,atol=2e-6,rtol=1e-6)
            assert task._qp_command_delta_q.eq(0).all()
            assert torch.equal(task._hard_pact_q_d,desired)
    assert heads.calls==3 and len(candidates)==12
