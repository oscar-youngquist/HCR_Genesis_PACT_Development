"""Outer extrapolation is independent of the physical QP integration horizon."""
from dataclasses import replace
import pytest
import torch
from test_hard_pact_velocity_objective import data
from test_hard_pact_reduced_qp import solver
from rsl_rl.algorithms.hard_pact_qp_diagnostics import QPIterationDiagnostics


@pytest.mark.parametrize('horizon', [None, .020])
def test_shared_horizon_masks_gradients_and_unchanged_assembly(horizon):
    q=solver(planar_velocity_weight=2.,yaw_rate_weight=3.)
    d=data(3);d['dt'][:,0]=torch.tensor([.0025,.005,.01])
    before=q._build(d);dt=d['dt'].clone()
    q.cfg=replace(q.cfg,qp_velocity_loss_horizon_s=horizon)
    after=q._build(d)
    for name in ('Q','p','G','h','tau_lower','tau_upper','qdd_lower','qdd_upper'):
        torch.testing.assert_close(getattr(before,name),getattr(after,name),rtol=0,atol=0)
    a=torch.ones(3,18,dtype=torch.float64,requires_grad=True)
    with torch.no_grad(): a[2].fill_(float('nan'))
    valid=torch.tensor([True,True,False]);accepted=torch.ones(3,dtype=torch.bool)
    xy,yaw,count=q.velocity_tracking_losses(a,d,valid,accepted)
    _,_,now=q._velocity_tracking_affine(d,after.acceleration_map,after.acceleration_offset)
    h=dt[:2] if horizon is None else torch.full_like(dt[:2],horizon)
    error=now[:2]+h-d['velocity_command'][:2]
    torch.testing.assert_close(xy,error[:,:2].square().sum(-1).mean())
    torch.testing.assert_close(yaw,error[:,2].square().mean())
    (xy+yaw).backward()
    assert count==2 and torch.isfinite(a.grad).all() and a.grad[:2].abs().sum()>0
    assert a.grad[2].eq(0).all() and torch.equal(dt,d['dt'])


@pytest.mark.parametrize('horizon',[0.,-1.,float('nan'),float('inf')])
def test_invalid_horizon(horizon):
    with pytest.raises(ValueError,match='horizon'):
        solver(qp_velocity_loss_horizon_s=horizon)


def diagnostics(padded=False):
    q=solver(qp_velocity_loss_horizon_s=.020)
    d=data(3)
    d['velocity_command'][:]=torch.tensor([[-1.,0,0],[0.,0,0],[1.,0,0]])
    d['base_linear_velocity_world'][:]=torch.tensor([[-.5,.2,0],[.1,0,0],[100.,0,0]])
    m=q._build(d)
    # Transparent affine mechanics: tau0 and f0 both contribute to body ax.
    amap=torch.zeros_like(m.acceleration_map);amap[:,0,0]=1;amap[:,0,12]=2
    x=torch.cat((d['tau_nom'],d['force_pred_world'].flatten(1)),1)
    x[:,0]-=2;x[:,12]-=3  # da_tau=-2, da_force=-6, total=-8.
    _,_,now=q._velocity_tracking_affine(d,amap,m.acceleration_offset)
    agg=QPIterationDiagnostics()
    real=torch.tensor([True,True,False]) if padded else torch.tensor([True,True,True])
    agg.tracking_conflict('primary',d,x,amap,m.acceleration_offset,now,
        torch.ones(3,dtype=torch.bool),q.cfg,q._limits(x),m.rate_lower,m.rate_upper,real_rows=real)
    return agg.finalize(x),q,d,m,x,amap,now


def test_direction_decomposition_padding_and_denominators():
    metrics,*_=diagnostics(padded=True)
    p='model_tracking_conflict/primary/accepted/'
    assert metrics[p+'matched_rows']==2
    assert metrics[p+'state/signed_direction_error_m_s']==pytest.approx(-.5)
    assert metrics[p+'state/signed_direction_error_m_s/samples']==1
    assert metrics[p+'state/underspeed_fraction']==1
    assert metrics[p+'outer_horizon/torque_correction/delta_velocity_along_command_m_s']==pytest.approx(.04)
    assert metrics[p+'outer_horizon/contact_force_change/delta_velocity_along_command_m_s']==pytest.approx(.12)
    assert metrics[p+'outer_horizon/full_candidate/signed_velocity_change_m_s']==pytest.approx(.152)
    assert metrics[p+'original_bounds/friction_n/near_fraction/samples']==16


def test_empty_nonfinite_and_partition_weighting():
    _,q,d,m,x,amap,now=diagnostics()
    a=QPIterationDiagnostics();b=QPIterationDiagnostics()
    x[2]=float('nan')
    accepted=torch.ones(3,dtype=torch.bool)
    a.tracking_conflict('recovery',d,x,amap,m.acceleration_offset,now,accepted,q.cfg,q._limits(x),m.rate_lower,m.rate_upper)
    for sl in (slice(0,1),slice(1,3)):
        b.tracking_conflict('recovery',{k:v[sl] for k,v in d.items()},x[sl],amap[sl],m.acceleration_offset[sl],now[sl],
            accepted[sl],q.cfg,q._limits(x),m.rate_lower[sl],m.rate_upper[sl])
    ma,mb=a.finalize(x),b.finalize(x)
    for key in ma: torch.testing.assert_close(ma[key],mb[key],equal_nan=True)
    p='model_tracking_conflict/recovery/accepted/'
    assert ma[p+'nonfinite_rows']==1
    assert ma[p+'matched_rows']==2
    c=QPIterationDiagnostics()
    c.tracking_conflict('primary',d,x,amap,m.acceleration_offset,now,~accepted,q.cfg,q._limits(x),m.rate_lower,m.rate_upper)
    assert torch.isnan(c.finalize(x)['model_tracking_conflict/primary/accepted/state/error_l2_m_s'])


def test_recovery_original_bounds_swing_exclusion_and_metadata():
    _,q,d,m,x,amap,now=diagnostics()
    d['contact_probability'][:,2:]=0
    x=torch.cat((x,torch.full((3,12),4.,dtype=x.dtype),torch.full((3,12),2.,dtype=x.dtype)),1)
    x[:,0]=12.  # Original rate bound is 10 Nm; softened accepted example.
    agg=QPIterationDiagnostics()
    agg.tracking_conflict('recovery',d,x,amap,m.acceleration_offset,now,
        torch.ones(3,dtype=torch.bool),q.cfg,q._limits(x),m.rate_lower,m.rate_upper)
    metrics=agg.finalize(x);p='model_tracking_conflict/recovery/accepted/'
    assert metrics[p+'original_bounds/friction_n/near_fraction/samples']==12
    assert metrics[p+'original_bounds/torque_rate_nm/violation_mean']==pytest.approx(2/12)
    assert metrics[p+'recovery/rate_slack_nm']==2
    assert metrics[p+'recovery/joint_slack_rad_s2']==4
    from legged_gym.envs.go2.go2_hard_pact.deployment import qp_update_contract
    contract=qp_update_contract('every_substep',4,qp_config=q.cfg)
    assert contract['velocity_tracking']['outer_loss_horizon_s']==.020
    assert 'physics_dt' in contract['velocity_tracking']['prediction']
    assert solver().cfg.qp_velocity_loss_horizon_s is None


@pytest.mark.parametrize('level,scheduled,expected',[('minimal',True,False),('physical',False,False),('physical',True,True),('full',True,True)])
def test_profile_gate_and_forward_gradient_parity(level,scheduled,expected):
    q=solver(diagnostics_level=level,qp_velocity_loss_horizon_s=.02,velocity_tracking_replay_enabled=True)
    q.diagnostics_scheduled=scheduled
    d=data(1);d['tau_nom'].requires_grad_()
    out=q.solve(differentiable=True,**d)
    grad=torch.autograd.grad(out.tau_safe.sum(),d['tau_nom'])[0]
    keys=q.iteration_metrics('ppo',d['tau_nom'])
    assert any('model_tracking_conflict' in k for k in keys)==expected
    ref=solver(diagnostics_level='minimal',velocity_tracking_replay_enabled=True)
    d2={k:v.detach().clone() for k,v in d.items()};d2['tau_nom'].requires_grad_()
    o2=ref.solve(differentiable=True,**d2)
    torch.testing.assert_close(out.tau_safe,o2.tau_safe,rtol=0,atol=0)
    assert torch.equal(out.stage,o2.stage)
    assert torch.equal(out.differentiated_mask,o2.differentiated_mask)
    torch.testing.assert_close(grad,torch.autograd.grad(o2.tau_safe.sum(),d2['tau_nom'])[0],rtol=0,atol=0)
