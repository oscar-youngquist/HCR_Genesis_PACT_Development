"""Owned replay mechanics, anchor alignment, and scheduled diagnostic parity."""
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
from rsl_rl.algorithms.hard_pact_qp_mechanics_cache import (
    allocate_mechanics, copy_mechanics, read_mechanics, MECHANICS_SHAPES, PREFIX,
)
from test_hard_pact_qp_modes import fixture
from test_hard_pact_reduced_qp import inputs,solver


@pytest.mark.parametrize('mode',['every_substep','random_one_substep','first_plus_random_substep'])
def test_selected_mechanics_own_storage_and_follow_shuffled_replay(mode):
    task,heads,qp,real,counts,q,v,quat,_=fixture(mode)
    task._cache_qp_replay_mechanics=True
    contexts=[]
    def build(state,velocity,**kw):
        data=inputs(state.shape[0],state.dtype)
        data['bias'][:,0]=state[:,7] # identify exact substep
        ctx=SimpleNamespace(**{k:data[k] for k in MECHANICS_SHAPES})
        contexts.append(ctx)
        return ctx
    task._hard_pact_bard_dynamics.build_context=build
    task._begin_qp_interval()
    for k in range(4):
        q.fill_(.005*k)
        task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(8,6))
    fields=task._qp_sampled_transition
    assert fields[PREFIX+'valid'].all()
    torch.testing.assert_close(fields[PREFIX+'bias'][:,0],.005*task._qp_sampled_substep_index.float())
    original=fields[PREFIX+'bias'].clone()
    for ctx in contexts: ctx.bias.fill_(99)
    assert torch.equal(fields[PREFIX+'bias'],original)
    order=torch.tensor([7,1,4,0])
    alg=PPO_HardPACT.__new__(PPO_HardPACT);alg.qp_config=qp.cfg
    with patch.object(alg,'_materialize_mechanics_cache',side_effect=AssertionError('cache miss')):
        result=alg._deployment_replay_mechanics(fields['sampled_qp_q'][order],fields['sampled_qp_v'][order],
                                               {k:value[order] for k,value in fields.items()})
    assert not result.mass_matrix.requires_grad
    torch.testing.assert_close(result.bias,original[order])
    ptr=fields[PREFIX+'bias'].data_ptr()
    task._begin_qp_interval()
    assert task._qp_sampled_transition[PREFIX+'bias'].data_ptr()==ptr
    assert not task._qp_sampled_transition[PREFIX+'valid'].any()


def test_cache_misses_version_and_disabled_recompute_only_needed_rows():
    q=torch.zeros(4,19);v=torch.zeros(4,18)
    cache=allocate_mechanics(4,q)
    ctx=SimpleNamespace(**{k:torch.ones(4,*shape) for k,shape in MECHANICS_SHAPES.items()})
    copy_mechanics(cache,ctx,torch.arange(4),torch.arange(4))
    cache[PREFIX+'valid'][1]=False;cache[PREFIX+'version'][3]=-1
    alg=PPO_HardPACT.__new__(PPO_HardPACT);alg.qp_config=solver().cfg
    def fresh(**kw):
        assert kw['parameters']=={} and kw['kind']=='deployment'
        return SimpleNamespace(**{k:torch.full((kw['q'].shape[0],)+s,2.) for k,s in MECHANICS_SHAPES.items()})
    with patch.object(alg,'_materialize_mechanics_cache',side_effect=fresh) as call:
        out=alg._deployment_replay_mechanics(q,v,cache)
        assert call.call_args.kwargs['q'].shape[0]==2
        assert out.bias[:,0].tolist()==[1,2,1,2]
        assert cache[PREFIX+'bias'][:,0].tolist()==[1]*4
        alg.qp_config=replace(alg.qp_config,cache_rollout_qp_mechanics=False)
        alg._deployment_replay_mechanics(q,v,cache)
        assert call.call_args.kwargs['q'].shape[0]==4
    assert read_mechanics(cache,q.double()) is None


def test_cache_qp_forward_and_learned_vjp_parity():
    data=inputs(2);cache=allocate_mechanics(2,data['tau_nom'])
    copy_mechanics(cache,SimpleNamespace(**data),torch.arange(2),torch.arange(2))
    results=[]
    for cached in (False,True):
        d={k:v.clone() for k,v in data.items()}
        if cached: d.update(read_mechanics(cache,d['tau_nom'])[0])
        keys=('tau_nom','force_pred_world','wrench_pred_world')
        for k in keys:d[k].requires_grad_()
        out=solver().solve(differentiable=True,**d)
        grads=torch.autograd.grad(out.tau_safe.square().sum()+out.force_world.square().sum(),[d[k] for k in keys])
        results.append((out,grads))
    torch.testing.assert_close(results[0][0].tau_safe,results[1][0].tau_safe,rtol=0,atol=0)
    for a,b in zip(results[0][1],results[1][1]):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_unscheduled_compact_skipped_but_all_attempt_counts_survive():
    qp=solver();qp.diagnostics_scheduled=False
    for phase in ('rollout','ppo'):
        qp.begin_iteration_diagnostics(phase)
        aggregate=qp.iteration_diagnostics[phase]
        with patch.object(aggregate,'compact_candidate',side_effect=AssertionError('unscheduled physical work')):
            out=qp.solve(diagnostics_phase=phase,**inputs(3))
        assert aggregate.sums['real_rows']==3
        assert sum(aggregate.sums['final/'+s+'_count'] for s in ('full','soft_joint','analytic'))==3
        assert not any(k.startswith('sampled_health') for k in aggregate.sums)


def test_cache_disabled_allocates_nothing_and_rejection_does_not_change_anchor():
    task,_,qp,_,_,q,v,quat,_=fixture('every_substep')
    task._begin_qp_interval()
    assert not any(k.startswith(PREFIX) for k in task._qp_sampled_transition)
    task._cache_qp_replay_mechanics=True
    qp.cfg=replace(qp.cfg,exception_capture_enabled=False)
    task._begin_qp_interval()
    with patch.object(qp,'_backend_solve',side_effect=RuntimeError('forced rejection')):
        for k in range(4):
            q.fill_(.005*k)
            task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(8,6))
    packet=task._qp_sampled_transition
    assert packet['sampled_qp_stage'].eq(2).all()
    assert packet[PREFIX+'valid'].all()
    torch.testing.assert_close(packet['sampled_qp_q'][:,7],.005*task._qp_sampled_substep_index.float())


def test_cache_dtype_mismatch_is_invalid_not_rounded():
    data=inputs(2,dtype=torch.float64)
    cache=allocate_mechanics(2,data['tau_nom'].float())
    copy_mechanics(cache,SimpleNamespace(**data),torch.arange(2),torch.arange(2))
    assert not cache[PREFIX+'valid'].any()
