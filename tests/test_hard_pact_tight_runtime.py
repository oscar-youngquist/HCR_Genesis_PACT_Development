"""Runtime-only bucket, tensor-kernel, replay-index and sampling invariants."""
import pytest
import torch
from rsl_rl.algorithms.hard_pact_qp_backends import solver_capacity
from rsl_rl.algorithms.hard_pact_qp import qp_interval_selection
from rsl_rl.algorithms.hard_pact_qp_tensor_ops import kernel
from rsl_rl.algorithms.hard_pact_qp_mechanics_cache import gather_mechanics,PREFIX


def test_capacity_chunk_limits():
    assert solver_capacity(1365,4096,'tight')==1408
    assert solver_capacity(4900,8000,'tight')==4992
    assert solver_capacity(1365,4096)==2048
    assert solver_capacity(4900,8000)==8000
    assert solver_capacity(3999,4000,'tight')==4000
    assert solver_capacity(5000,4000,'tight')==5000
    assert solver_capacity(3,8000,'tight')==4


@pytest.mark.parametrize('shared',[False,True])
def test_sampling_reproducible_and_replay_alignment(shared):
    g=torch.Generator().manual_seed(42);state=g.get_state()
    later,replay=qp_interval_selection('first_plus_random_substep',4096,4,'cpu',generator=g,shared_later=shared)
    assert ((replay==0)|(replay==later)).all()
    assert .47 < (replay==0).float().mean() < .53
    if shared: assert later.unique().numel()==1
    else: assert torch.bincount(later)[1:].max()-torch.bincount(later)[1:].min()<=1
    g.set_state(state)
    repeated=qp_interval_selection('first_plus_random_substep',4096,4,'cpu',generator=g,shared_later=shared)
    assert torch.equal(later,repeated[0]) and torch.equal(replay,repeated[1])


def test_direct_gather_matches_shuffled_then_selected():
    fields={PREFIX+'bias':torch.arange(6*8*18).reshape(6,8,18),
            PREFIX+'valid':torch.ones(6,8,1,dtype=torch.bool),'other':torch.ones(6,8)}
    shuffled=torch.tensor([47,2,13,7,21,0]);qp=torch.tensor([4,0,2])
    actual=gather_mechanics(fields,shuffled[qp])
    assert 'other' not in actual
    for name,value in actual.items():
        torch.testing.assert_close(value,fields[name].flatten(0,1)[shuffled][qp])


def test_shared_rollout_two_dispatches_and_owned_anchor():
    from dataclasses import replace
    from test_hard_pact_qp_modes import fixture
    task,heads,qp,real,counts,q,v,quat,build_rows=fixture('first_plus_random_substep')
    qp.cfg=replace(qp.cfg,qp_shared_later_substep=True)
    task._cache_qp_replay_mechanics=True
    task._begin_qp_interval()
    for k in range(4):
        q.fill_(.005*k)
        task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(8,6))
    assert counts.eq(2).all() and build_rows==[8,8]
    packet=task._qp_sampled_transition
    torch.testing.assert_close(packet['sampled_qp_q'][:,7],.005*task._qp_sampled_substep_index.float())
    assert packet[PREFIX+'valid'].all()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_compiled_kernels_dynamic_shapes_vjp_and_certificate(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    for rows in (3,7):
        torch.manual_seed(4)
        C=torch.randn(rows,12,24,device=device,dtype=torch.float64,requires_grad=True)
        e=torch.randn(rows,12,device=device,dtype=torch.float64,requires_grad=True)
        Q=torch.eye(24,device=device,dtype=torch.float64).expand(rows,-1,-1)
        p=torch.zeros(rows,24,device=device,dtype=torch.float64)
        weight=C.new_tensor(.25 if rows==3 else .75)
        results=[]
        for compiled in (False,True):
            q,v=kernel('add_residual',compiled)(Q,p,C,e,weight)
            scaled=kernel('row_scale',compiled)(C,e)
            grad=torch.autograd.grad(q.sum()+v.sum()+scaled[0].sum(),(C,e),retain_graph=True)
            A=C[:,:0];b=e[:,:0];z=p
            cert=kernel('certify',compiled)(A,b,C,e,z,C.new_tensor(1e-6))
            results.append((q,v,*scaled,*grad,*cert))
        for a,b in zip(*results): torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
        with torch.no_grad():
            for name,args in [('add_residual',(Q,p,C,e,weight)),('row_scale',(C,e)),
                             ('certify',(C[:,:0],e[:,:0],C,e,p,C.new_tensor(1e-6)))]:
                for a,b in zip(kernel(name,False)(*args),kernel(name,True)(*args)):
                    torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
