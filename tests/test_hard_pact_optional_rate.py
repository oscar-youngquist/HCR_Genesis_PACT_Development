"""Independent inner horizon and structurally optional QP torque-rate limits."""
from dataclasses import replace
import pytest
import torch
from test_hard_pact_reduced_qp import inputs, solver
from test_hard_pact_velocity_objective import data
from rsl_rl.algorithms.hard_pact_qp import recovery_projection_loss
from rsl_rl.algorithms.hard_pact_qp_diagnose import QPCapture, candidate_assessment


def test_inner_outer_horizons_independent_and_constraint_dt_unchanged():
    q=solver(planar_velocity_weight=2.,yaw_rate_weight=3.)
    d=data();d['dt'][1]=.005;dt=d['dt'].clone()
    legacy=q._build(d)
    q.cfg=replace(q.cfg,qp_velocity_objective_horizon_s=.020)
    new=q._build(d)
    C,e,_=q._velocity_tracking_affine(d,new.acceleration_map,new.acceleration_offset,horizon_s=.020)
    C0,e0,_=q._velocity_tracking_affine(d,new.acceleration_map,new.acceleration_offset)
    weights=C.new_tensor([2.,2.,3.]);C=C*new.variable_scale;C0=C0*new.variable_scale
    torch.testing.assert_close(new.Q-legacy.Q,2*(C.transpose(1,2)@(weights[None,:,None]*C)-C0.transpose(1,2)@(weights[None,:,None]*C0)),atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(new.p-legacy.p,2*((C.transpose(1,2)@(weights*e)[...,None])-(C0.transpose(1,2)@(weights*e0)[...,None])).squeeze(-1))
    for name in ('G','h','tau_lower','tau_upper','qdd_lower','qdd_upper'):
        assert torch.equal(getattr(new,name),getattr(legacy,name))
    a=torch.ones(2,18,dtype=torch.float64,requires_grad=True);mask=torch.ones(2,dtype=torch.bool)
    loss=q.velocity_tracking_losses(a,d,mask,mask)[:2]
    q.cfg=replace(q.cfg,qp_velocity_loss_horizon_s=.1)
    outer=q.velocity_tracking_losses(a,d,mask,mask)[:2]
    assert any(not torch.equal(v,w) for v,w in zip(loss,outer))
    assert torch.equal(q._build(d).Q,new.Q) and torch.equal(d['dt'],dt)
    q.cfg=replace(q.cfg,qp_velocity_objective_horizon_s=None)
    assert torch.equal(q._build(d).Q,legacy.Q)
    assert torch.equal(q._soft_joint_problem(new).Q[:,:24,:24],new.Q)


@pytest.mark.parametrize('kwargs',[
    {'qp_velocity_objective_horizon_s':0}, {'qp_velocity_objective_horizon_s':float('nan')},
    {'qp_velocity_objective_horizon_s':float('inf')}, {'qp_velocity_objective_horizon_s':-1},
    {'torque_rate_constraint_weight':-1}, {'torque_rate_constraint_weight':float('nan')}])
def test_validation(kwargs):
    with pytest.raises(ValueError):solver(**kwargs)


@pytest.mark.parametrize('weight,nvar,nrows,npacked',[(0.,36,80,44),(1.,48,116,68),(2.,48,116,68)])
def test_layout_native_bounds_and_positive_enable_semantics(weight,nvar,nrows,npacked):
    q=solver(torque_rate_constraint_weight=weight);d=inputs()
    m=q._build(d);r=q._soft_joint_problem(m)
    assert r.Q.shape==(2,nvar,nvar) and r.G.shape==(2,nrows,nvar)
    assert q._cupiqp_native_pack(r)[0].shape==(2,npacked,nvar)
    assert q._cupiqp_native_pack(m)[0].shape==(2,44,24)
    assert (torch.linalg.eigvalsh(r.Q)>0).all()
    assert m.tau_upper.eq(10. if weight else 23.5).all()
    assert (m.rate_lower is None)==(weight==0)
    # Canonical and native packing represent exactly the same constraints.
    z=torch.linspace(-2,2,2*nvar,dtype=torch.float64).reshape(2,nvar)
    G,h,lo,hi=q._cupiqp_native_pack(r)
    canonical=(r.G@z[...,None]).squeeze(-1)<=r.h
    packed=((G@z[...,None]).squeeze(-1)<=h).all(-1)&(z>=lo).all(-1)&(z<=hi).all(-1)
    assert torch.equal(canonical.all(-1),packed)
    if weight:
        ref=solver(torque_rate_constraint_weight=1.)._build(d)
        assert torch.equal(ref.Q,m.Q) and torch.equal(ref.G,m.G) and torch.equal(ref.h,m.h)
    else:
        ref=solver()._build(d)
        assert torch.equal(m.physical_G[:,24:],ref.physical_G[:,24:])
        assert torch.equal(m.physical_h[:,24:],ref.physical_h[:,24:])
        assert r.acceleration_map.shape[-1]==36


@pytest.mark.parametrize('backend',['qpth','cupiqp'])
def test_no_hidden_rate_projection_recovery_gradients_and_fallback(backend):
    if backend=='cupiqp' and not torch.cuda.is_available():pytest.skip('CUDA required')
    device='cuda:0' if backend=='cupiqp' else 'cpu'
    q=solver(torque_rate_constraint_weight=0.,qp_solver=backend,soft_joint_recovery_enabled=True,
             diagnostics_level='physical',solver_dtype='float64')
    d=inputs(2,device=device);d['tau_nom'].fill_(20);d['tau_nom'].requires_grad_()
    d['joint_position'][1]=2.01  # Cannot satisfy position with hard torque: recover.
    result=q.solve(differentiable=True,**d)
    assert result.stage.tolist()==[0,1]
    assert result.tau_safe[0].min()>10 and result.tau_safe.abs().max()<=23.5
    assert result.recovery_rate_slack.eq(0).all()
    loss,_=recovery_projection_loss(result,d['tau_nom'],q.torque_limits.to(device),torch.ones(2,device=device),q.cfg)
    grad=torch.autograd.grad(loss+result.tau_safe[0].sum(),d['tau_nom'],retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    result.recovery_rate_slack=torch.full_like(result.recovery_rate_slack,float('nan'),requires_grad=True)
    no_rate,_=recovery_projection_loss(result,d['tau_nom'],q.torque_limits.to(device),torch.ones(2,device=device),q.cfg)
    torch.testing.assert_close(loss,no_rate)
    assert torch.autograd.grad(no_rate,result.recovery_rate_slack,allow_unused=True)[0] is None
    assert torch.isnan(result.diagnostics['soft_joint/original_rate_violation_max_nm']).all()
    metrics=q.iteration_metrics('ppo',d['tau_nom'])
    assert metrics['qp/ppo/torque_rate_constraints_enabled']==0
    # Failed mechanics: deterministic magnitude fallback, not 10-Nm rate clip.
    d['mass_matrix'].fill_(float('nan'));d['tau_nom']=torch.full_like(d['tau_nom'],50.)
    failed=q.solve(differentiable=False,**d)
    assert failed.stage.eq(2).all() and failed.tau_safe.eq(23.5).all()
    q.cfg=replace(q.cfg,torque_rate_constraint_weight=1.)
    failed=q.solve(differentiable=False,**d)
    assert failed.tau_safe.eq(10.).all()


def test_capture_roundtrip_both_layouts_and_old_defaults(tmp_path):
    from test_hard_pact_qp_diagnose import owner
    from scripts.diagnose_hard_pact_qp import replay_csv_row
    for weight in (0.,1.):
        q=owner();q.cfg=replace(q.cfg,torque_rate_constraint_weight=weight,qp_velocity_objective_horizon_s=.020)
        d=inputs();m=q._soft_joint_problem(q._build(d))
        packet=QPCapture(tmp_path).before(q,m,d,'recovery',torch.tensor([11,19]))
        path=tmp_path/f'layout_{weight}.pt';torch.save(packet,path)
        packet=torch.load(path,weights_only=False)
        x=torch.zeros_like(m.p);x[:,:12]=12
        if weight:x[:,36:48]=2
        report=candidate_assessment(packet,x/m.variable_scale)
        assert report['production_accepted'].all()
        assert packet['schema_version']==4 and packet['rows'].tolist()==[11,19]
        assert packet['inner_velocity_objective_horizon_s']==.020
        assert report['torque_rate_constraints_enabled']==bool(weight)
        csv=replay_csv_row({'assessment':report})
        assert csv['torque_rate_constraints_enabled']==bool(weight)
        if not weight:
            assert report['original_hard_rate_satisfied'] is None
            assert torch.isnan(report['joint']['rate_slack_nm']).all()
            assert not any('rate_soft' in k for k in report['groups'])
        else:
            packet['schema_version']=3
            packet['config'].pop('torque_rate_constraint_weight')
            packet['config'].pop('qp_velocity_objective_horizon_s')
            old=candidate_assessment(packet,x/m.variable_scale)
            assert torch.equal(old['production_accepted'],report['production_accepted'])


def test_physical_costs_use_inner_horizon_but_comparisons_keep_dt_and_outer():
    q=solver(planar_velocity_weight=2.,yaw_rate_weight=3.,torque_rate_constraint_weight=0.,
             qp_velocity_objective_horizon_s=.02,qp_velocity_loss_horizon_s=.1,diagnostics_level='physical')
    d=data();m=q._build(d)
    x=torch.cat((d['tau_nom'],d['force_pred_world'].flatten(1)),1)
    q._joint_candidate_diagnostics('primary',m,x,d,torch.ones(2,dtype=torch.bool))
    metrics=q.iteration_metrics('rollout',d['tau_nom'])
    a=(m.acceleration_map@x[...,None]).squeeze(-1)+m.acceleration_offset
    _,_,now=q._velocity_tracking_affine(d,m.acceleration_map,m.acceleration_offset)
    error=now+.02*a[:,[0,1,5]]-d['velocity_command']
    p='qp/rollout/model_velocity_tracking/primary/accepted/'
    torch.testing.assert_close(metrics[p+'planar_cost'],(2*error[:,:2].square().sum(-1)).mean().float())
    torch.testing.assert_close(metrics[p+'yaw_cost'],(3*error[:,2].square()).mean().float())
    p='qp/rollout/model_tracking_conflict/primary/accepted/'
    assert metrics[p+'inner_objective_horizon_s']==pytest.approx(.02)
    assert metrics[p+'physics_dt/horizon_s']==pytest.approx(.01)
    assert metrics[p+'outer_horizon/horizon_s']==pytest.approx(.1)
    assert not any('original_bounds/torque_rate_nm' in key for key in metrics)
