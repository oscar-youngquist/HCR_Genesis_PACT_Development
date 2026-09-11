"""Fixed-mask QP versus reviewed grouped formulation, without relaxing limits."""
import runpy
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from qpth.qp import QPFunction
from test_hard_pact_reduced_qp import inputs,solver
from hard_pact_grouped_reference import _build as grouped_build
from rsl_rl.algorithms.hard_pact_qp_backends import _configure_cupiqp
from types import SimpleNamespace
from rsl_rl.algorithms.qpth_warm_start import solve_qpth_warm

def test_mixed_masks_single_dispatch_and_physical_map():
    d=inputs(16)
    d['contact_probability']=((torch.arange(16)[:,None]>>torch.arange(4))&1).double()
    d['foot_jacobians'][:,:,:,6:]=torch.eye(12).reshape(4,3,12)*.1
    qp=solver(ppo_chunk_size=32)
    m=qp._build(d)
    assert m.G.shape==(16,68,24) and m.A.shape==(16,0,24)
    assert qp._cupiqp_native_pack(m)[0].shape==(16,44,24)
    mask=d['contact_probability'].repeat_interleave(3,1)
    assert m.acceleration_map[:,:,12:][(mask==0)[:,None,:].expand(-1,18,-1)].eq(0).all()
    with patch.object(qp,'_backend_solve',wraps=qp._backend_solve) as call:
        out=qp.solve(**d)
    assert call.call_count==1
    assert out.force_world[d['contact_probability']==0].eq(0).all()
    assert out.stage.eq(0).all()
    torch.linalg.cholesky(m.Q)

@pytest.mark.parametrize('pattern',[0,1,5,10,15])
def test_grouped_forward_and_vjp_parity(pattern):
    qp=solver();d=inputs(1)
    d['contact_probability'][0]=torch.tensor([(pattern>>i)&1 for i in range(4)])
    d['foot_jacobians'][:,:,:,6:]=torch.eye(12).reshape(4,3,12)*.1
    d['base_jacobian'][:,:,6:12]=torch.eye(6)*.05
    keys=('tau_nom','force_pred_world','wrench_pred_world')
    for k in keys:d[k].requires_grad_()
    outputs=[]
    for build in (qp._build(d),grouped_build(qp,d,pattern)):
        z,_=solve_qpth_warm(build.Q,build.p,build.G,build.h,build.A,build.b,
            warm_start=None,eps=1e-9,max_iter=60,not_improved_limit=10,verbose=-1,check_q_spd=True)
        x=z*build.variable_scale
        f=x[:,12:].reshape(-1,4,3)*d['contact_probability'][...,None]
        a=(build.acceleration_map@x[...,None]).squeeze(-1)+build.acceleration_offset
        loss=x[:,:12].square().sum()+f.square().sum()*.01+a.square().sum()*.01
        outputs.append((torch.cat((x[:,:12],f.flatten(1)),1),torch.autograd.grad(loss,[d[k] for k in keys],retain_graph=True)))
    torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=1e-5,atol=1e-6)
    for a,b in zip(outputs[0][1],outputs[1][1]):torch.testing.assert_close(a,b,rtol=3e-3,atol=2e-5)

def test_default_and_override_solver_resolution_without_simulator_import():
    resolve=runpy.run_path(str(Path(__file__).parents[1]/'legged_gym/scripts/hard_pact_solver_selection.py'))['effective_solvers']
    assert resolve(['--task','go2_hard_pact_full_isaaclab'])==('cupiqp','cupiqp')
    assert resolve(['--qp_solver=qpth'])==('qpth','qpth')
    assert resolve(['--qp_solver','qpth','--ppo_qp_solver','cupiqp'])==('qpth','cupiqp')

def test_report_does_not_enable_gap_stopping_require_does():
    qp=solver();s=SimpleNamespace(settings=SimpleNamespace())
    _configure_cupiqp(s,qp.cfg,torch.float64,differentiable=False)
    assert s.settings.check_duality_gap is False
    _configure_cupiqp(s,qp.cfg,torch.float64,differentiable=True)
    assert s.settings.check_duality_gap is True

@pytest.mark.skipif(not torch.cuda.is_available(),reason='cuPIQP CUDA required')
@pytest.mark.parametrize('differentiable',[False,True])
def test_cupiqp_fixed_batch_gap_available_and_reuse(differentiable):
    qp=solver(qp_solver='cupiqp',solver_dtype='float64',ppo_chunk_size=32,rollout_chunk_size=32)
    d={k:v.cuda() for k,v in inputs(16).items()}
    d['contact_probability']=((torch.arange(16,device='cuda')[:,None]>>torch.arange(4,device='cuda'))&1).double()
    d['foot_jacobians'][:,:,:,6:]=torch.eye(12,device='cuda').reshape(4,3,12)*.1
    d['tau_nom'].requires_grad_();d['force_pred_world'].requires_grad_()
    for _ in range(2):
        r=qp.solve(differentiable=differentiable,**d)
        assert r.stage.eq(0).all()
        assert r.diagnostics['full/duality_gap'].isfinite().all()
        assert r.force_world[d['contact_probability']==0].eq(0).all()
        if differentiable:
            g=torch.autograd.grad(r.tau_safe.square().sum()+.01*r.force_world.square().sum(),
                (d['tau_nom'],d['force_pred_world']))
            assert all(v.isfinite().all() and v.abs().sum()>0 for v in g)
        else:assert not r.tau_safe.requires_grad
