from types import SimpleNamespace
from unittest.mock import patch
import torch
import pytest
from test_hard_pact_reduced_qp import solver, inputs
from rsl_rl.algorithms.hard_pact_qp import _ScaleClipRows


def test_identity_gradient_conditioner_skips_norms_when_not_auditing():
    x=torch.randn(3,12,requires_grad=True)
    sink=SimpleNamespace(diagnostics_scheduled=False,_last_gradient_metrics={})
    y=_ScaleClipRows.apply(x,1.,0.,sink,'tau')
    with patch.object(torch.Tensor,'norm',side_effect=AssertionError('diagnostic reduction')):
        y.sum().backward()
    assert x.grad.eq(1).all() and not sink._last_gradient_metrics


def test_interval_counts_and_maxima_survive_unscheduled_iterations():
    qp=solver(); d=inputs(2)
    for iteration in range(3):
        qp.diagnostics_scheduled=iteration==2
        qp.begin_iteration_diagnostics('rollout')
        d['tau_nom'].fill_(iteration+1.)
        qp.solve(**d)
    m=qp.iteration_metrics('rollout',d['tau_nom'])
    assert m['qp/rollout/real_rows']==6
    assert m['qp/rollout/reporting_iterations']==3
    assert m['qp/rollout/health/primary/accepted/torque_abs_nm/max']>2.9
    qp.begin_iteration_diagnostics('rollout');qp.solve(**d)
    assert qp.iteration_metrics('rollout',d['tau_nom'])['qp/rollout/real_rows']==2


def test_compact_logging_forward_and_gradient_parity():
    outputs=[]
    for enabled in (False,True):
        qp=solver(tensorboard_diagnostics_enabled=enabled)
        qp.diagnostics_scheduled=False
        d=inputs(2);d['tau_nom'].requires_grad_();d['force_pred_world'].requires_grad_()
        r=qp.solve(differentiable=True,**d)
        grads=torch.autograd.grad(r.tau_safe.square().sum()+r.force_world.square().sum(),
                                  [d['tau_nom'],d['force_pred_world']])
        outputs.append((r,grads))
    a,b=outputs
    assert torch.equal(a[0].stage,b[0].stage)
    torch.testing.assert_close(a[0].tau_safe,b[0].tau_safe,rtol=0,atol=0)
    for x,y in zip(a[1],b[1]):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_recovery_capacity_padding_vjp_and_exclusive_lease(monkeypatch):
    """CPU API double: exercise actual custom Function and graph lifetime."""
    import gc, sys
    from contextlib import nullcontext
    from rsl_rl.algorithms import hard_pact_qp_backends as backend_module
    from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig
    gradients=[]
    class Dense:
        def __init__(self,*args): self.settings=SimpleNamespace()
        def setup(self,**kw): self.kw=kw
        def update(self,**kw): self.kw=kw
        def solve(self):
            c=self.kw['c'];x=-c.clone()
            self.result=SimpleNamespace(x=x,info=SimpleNamespace(duality_gap=c.new_zeros(c.shape[0]),
                duality_gap_rel=c.new_zeros(c.shape[0]),iter=None))
        def backward(self,grad_x):
            gradients.append(grad_x.clone())
            return SimpleNamespace(**{key: (-grad_x if key=='c' else torch.zeros_like(self.kw[source]))
                for key,source in [('P','P'),('c','c'),('G','G'),('h_u','h_u'),('A','A'),('b','b')]})
    monkeypatch.setitem(sys.modules,'cupiqp',SimpleNamespace(DenseSolver=Dense,SparseSolver=Dense))
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace(cuda=SimpleNamespace(
        Device=lambda *_:nullcontext(),ExternalStream=lambda *_:nullcontext())))
    monkeypatch.setattr(torch.cuda,'current_stream',lambda *_:SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(backend_module,'_cupiqp_dtype',lambda dtype:dtype)
    monkeypatch.setattr(backend_module,'_configure_cupiqp',lambda *a,**k:None)
    monkeypatch.setattr(backend_module,'_as_cupy_zero_copy',lambda x:x)
    monkeypatch.setattr(backend_module,'_as_torch_zero_copy',lambda x,ref:x)
    cfg=HardPACTQPConfig(cupiqp_ppo_pool_size=1)
    backend=backend_module.SolverBackend('cupiqp',cfg)
    def forward(rows):
        p=torch.ones(rows,48,requires_grad=True)
        Q=torch.eye(48).expand(rows,-1,-1);G=torch.zeros(rows,68,48);h=torch.ones(rows,68)
        A=torch.empty(rows,0,48);b=torch.empty(rows,0)
        x,*_=backend_module.CuPIQPFunction.apply(Q,p,G,h,A,b,None,None,True,0,cfg,backend)
        return x,p
    x,p=forward(3)
    for _ in range(2):
        g=torch.autograd.grad(x.sum(),p,retain_graph=True)[0]
        assert g.eq(-1).all() and gradients[-1].shape==(4,48)
        assert gradients[-1][3].eq(0).all()
    assert not backend._ppo_pool.idle
    y,q=forward(4)  # outstanding x graph owns its lease, cannot be reused
    assert backend.stats['ppo']['pool_hits']==0
    del x,y;gc.collect()
    z,_=forward(4)
    assert backend.stats['ppo']['pool_hits']==1
    assert backend.stats['ppo']['requested_rows']==11
    assert backend.stats['ppo']['capacity_rows']==12
    assert backend.stats['ppo']['padded_rows']==1


def test_recovery_reuses_primary_including_presolver_rejections_and_owner_gradients():
    from rsl_rl.algorithms.hard_pact_qp import recovery_projection_loss
    results=[]
    for reuse in (False,True):
        torch.manual_seed(17)
        qp=solver(torque_rate_limit_nm_s=10.,soft_rate_recovery_weight=.1)
        qp._reuse_primary_assembly=reuse
        d=inputs(3)
        d['joint_position'][0,0]=2.4  # empty before the primary dispatch
        d['joint_position'][1,0]=2.0005
        d['mass_matrix'][2,0,0]=float('nan')  # stays excluded/analytic
        trunk=torch.nn.Linear(2,3).double()
        actor=torch.nn.Linear(3,12).double()
        force=torch.nn.Linear(3,12).double()
        wrench=torch.nn.Linear(3,6).double()
        z=trunk(torch.ones(3,2,dtype=torch.float64))
        d['tau_nom']=actor(z);d['force_pred_world']=force(z).reshape(3,4,3)
        d['wrench_pred_world']=wrench(z)
        d['foot_jacobians'][:,0,2,6]=.1
        d['base_jacobian'][:,0,6]=.2
        original=qp._backend_solve
        def fail_primary(m):
            if m.p.shape[1]==24:raise RuntimeError('force recovery')
            return original(m)
        with patch.object(qp,'_backend_solve',side_effect=fail_primary),patch.object(qp,'_build',wraps=qp._build) as build:
            out=qp.solve(differentiable=True,**d)
            assert build.call_count==(1 if reuse else 2)
        loss,_=recovery_projection_loss(out,d['tau_nom'],qp.torque_limits,torch.ones(3,dtype=torch.bool),qp.cfg)
        params=[p for module in (actor,trunk,force,wrench) for p in module.parameters()]
        gradients=torch.autograd.grad(loss,params,retain_graph=True)
        repeated=torch.autograd.grad(loss,params)
        for a,b in zip(gradients,repeated):torch.testing.assert_close(a,b)
        assert out.stage.tolist()==[1,1,2]
        assert all(g.isfinite().all() for g in gradients)
        results.append((out.tau_safe.detach(),gradients))
    torch.testing.assert_close(results[0][0],results[1][0],atol=1e-8,rtol=1e-6)
    for a,b in zip(results[0][1],results[1][1]):torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-6)


def test_recovery_native_slack_bounds_equal_canonical_rows():
    qp=solver();m=qp._soft_joint_problem(qp._build(inputs(3)))
    G,h,lo,hi=qp._cupiqp_native_pack(m)
    assert G.shape==(3,68,48) and m.G.shape==(3,116,48)
    assert lo[:,24:].eq(0).all() and torch.isposinf(hi[:,24:]).all()
    x=torch.zeros_like(m.p);x[:,24:]=1.;x[1,36]=-.1
    canonical=(m.G@x[...,None]).squeeze(-1).le(m.h+1e-9).all(-1)
    packed=(G@x[...,None]).squeeze(-1).le(h+1e-9).all(-1)&(x>=lo).all(-1)&(x<=hi).all(-1)
    assert torch.equal(canonical,packed)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="cuPIQP needs CUDA")
@pytest.mark.parametrize('dtype,tolerance',[(torch.float64,2e-6),(torch.float32,2e-4)])
def test_cuda_recovery_capacity_matches_exact_fresh_forward_vjp(dtype,tolerance):
    import gc
    reused=solver(qp_solver='cupiqp',cupiqp_ppo_capacity_reuse=True,ppo_max_iter=100)
    fresh=solver(qp_solver='cupiqp',cupiqp_ppo_capacity_reuse=False,cupiqp_ppo_reuse=False,ppo_max_iter=100)
    def run(qp,rows):
        d=inputs(rows,dtype=dtype,device='cuda')
        d['foot_jacobians'][:,0,2,6]=.1;d['base_jacobian'][:,0,6]=.2
        learned=[d[k].requires_grad_() for k in ('tau_nom','force_pred_world','wrench_pred_world')]
        m=qp._soft_joint_problem(qp._build(d));G,h,lo,hi=qp._cupiqp_native_pack(m)
        backend=qp._backend_instances['cupiqp']
        r=backend.solve(m.Q,m.p,G,h,m.A,m.b,native_lower=lo,native_upper=hi,differentiable=True)
        assert qp._certificate(m,r.solution,qp.cfg.ppo_feasibility_tolerance)[0].all()
        loss=(r.solution*m.variable_scale).square().sum()
        g=torch.autograd.grad(loss,learned,retain_graph=True)
        g2=torch.autograd.grad(loss,learned)
        for a,b in zip(g,g2):torch.testing.assert_close(a,b,atol=tolerance,rtol=tolerance)
        return r.solution.detach().clone(),[v.detach().clone() for v in g]
    for rows in (3,4,3):
        a,ga=run(reused,rows);b,gb=run(fresh,rows)
        torch.testing.assert_close(a,b,atol=tolerance,rtol=tolerance)
        for x,y in zip(ga,gb):
            assert torch.isfinite(x).all()
            torch.testing.assert_close(x,y,atol=tolerance,rtol=tolerance)
        gc.collect()
    assert reused._backend_instances['cupiqp'].stats['ppo']['pool_hits']==2
