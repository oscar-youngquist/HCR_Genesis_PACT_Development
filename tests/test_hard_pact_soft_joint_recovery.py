"""Small cuPIQP safety/recovery check; no simulator or training."""
from unittest.mock import patch
from dataclasses import replace
import pytest
import torch
from test_hard_pact_reduced_qp import inputs, solver
from rsl_rl.algorithms.hard_pact_qp import recovery_projection_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuPIQP requires CUDA")
@pytest.mark.parametrize('failure',['empty_joint_interval','coupled_infeasible'])
def test_soft_joint_recovery_preserves_hard_constraints_and_isolates_gradients(failure):
    d = inputs(3, device="cuda")
    # Row 1 has an impossible joint envelope: q>qmax requires a<-100.
    if failure == 'empty_joint_interval':
        d['joint_position'][1,0] = 2.02
    else:
        # Joint interval [-100,100] is nonempty, but bounded torque cannot
        # offset this bias: a_joint=tau-200. Contact J is zero in this fixture.
        d['bias'][1,6] = 200
    d['contact_probability'][:,1] = 0
    d['tau_nom'].requires_grad_()
    settings = dict(qp_solver='cupiqp',solver_dtype='float64',
                    rollout_max_iter=100,ppo_max_iter=100)
    qp = solver(**settings)
    out = qp.solve(differentiable=True,**d)
    assert out.stage.tolist() == [0,1,0]
    assert out.differentiated_mask.tolist() == [True,False,True]
    assert torch.isfinite(out.tau_safe).all()
    assert out.force_world[:,1].eq(0).all()
    bounds = min(23.5,qp.cfg.torque_rate_limit_nm_s*.01)
    assert out.tau_safe.abs().max() <= bounds
    f = out.force_world
    assert f[:,:,2].min() >= -1e-6
    assert (f[:,:,:2].abs()-qp.cfg.friction_coefficient*f[:,:,2,None]).max() <= 1e-6
    assert out.diagnostics['soft_joint/slack_max_rad_s2'][1] > 0
    # Original hard certificate is not claimed, but softened problem is certified.
    assert out.diagnostics['selected/inequality_max'][1] <= qp.cfg.ppo_feasibility_tolerance
    out.tau_safe.square().sum().backward()
    assert d['tau_nom'].grad.isfinite().all()
    assert d['tau_nom'].grad[1].abs().sum() > 0
    assert d['tau_nom'].grad[[0,2]].abs().sum() > 0
    reference = solver(**settings,soft_joint_recovery_enabled=False).solve(differentiable=True,**d)
    assert reference.stage.tolist() == [0,2,0]
    torch.testing.assert_close(out.tau_safe[[0,2]],reference.tau_safe[[0,2]],atol=1e-5,rtol=1e-5)
    # Recovery-only exception still reaches deterministic bounded fallback.
    real_solve = qp._backend_solve
    def fail_recovery(m):
        if m.p.shape[1] == 36:
            raise RuntimeError('forced recovery failure')
        return real_solve(m)
    with patch.object(qp,'_backend_solve',side_effect=fail_recovery):
        failed = qp.solve(differentiable=False,**d)
    assert failed.stage.tolist() == [0,2,0]
    assert failed.diagnostics['soft_joint/solver_exception'][1]
    assert failed.tau_safe.abs().max() <= bounds


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuPIQP requires CUDA")
def test_recovery_outer_loss_heads_retained_backward_masks_and_weights():
    torch.manual_seed(41)
    d=inputs(2,device='cuda')
    d['bias'][:,6]=200
    d['joint_position'][:,0]=2.02
    d['foot_jacobians'][:,0,2,6]=.1
    d['base_jacobian'][:,0,6]=.2
    d['bias'].requires_grad_()
    d['foot_jacobians'].requires_grad_()
    d['contact_probability'].requires_grad_()
    # Minimal shared encoder and separate actor/force/wrench heads.
    trunk=torch.nn.Linear(2,3).double().cuda()
    heads=torch.nn.ModuleList([torch.nn.Linear(3,k) for k in (12,12,6)]).double().cuda()
    z=trunk(torch.ones(2,2,dtype=torch.float64,device='cuda'))
    d['tau_nom']=heads[0](z)
    d['force_pred_world']=heads[1](z).reshape(2,4,3)
    d['wrench_pred_world']=heads[2](z)
    qp=solver(qp_solver='cupiqp',solver_dtype='float64',ppo_max_iter=100)
    out=qp.solve(differentiable=True,**d)
    assert out.stage.eq(1).all()
    limits=qp.torque_limits.to(d['tau_nom'])
    valid=torch.tensor([True,False],device='cuda')
    loss,per_row=recovery_projection_loss(out,d['tau_nom'],limits,valid,qp.cfg)
    expected=((out.tau_safe[0]-d['tau_nom'][0])/limits).square().sum()
    expected+=(out.recovery_slack[0]/qp.cfg.soft_joint_recovery_scale_rad_s2).square().sum()
    torch.testing.assert_close(loss,expected)
    assert per_row[1]==0 and loss.isfinite() and loss>0
    # Repeated VJPs exercise the solver's retained-graph ownership, as PCGrad does.
    params=list(trunk.parameters())+list(heads.parameters())
    first=torch.autograd.grad(loss,params,retain_graph=True)
    second=torch.autograd.grad(loss,params,retain_graph=True)
    for a,b in zip(first,second):
        assert a.isfinite().all() and a.abs().sum()>0
        torch.testing.assert_close(a,b)
    assert torch.autograd.grad(loss,[d['bias'],d['foot_jacobians'],d['contact_probability']],allow_unused=True,retain_graph=True)==(None,None,None)
    zero,_=recovery_projection_loss(out,d['tau_nom'],limits,valid,replace(qp.cfg,recovery_projection_weight=0))
    assert zero==0
    empty,_=recovery_projection_loss(out,d['tau_nom'],limits,torch.zeros_like(valid),qp.cfg)
    assert empty.isfinite() and empty==0
    gradients=torch.autograd.grad(empty,params)
    assert all(g.eq(0).all() for g in gradients)
