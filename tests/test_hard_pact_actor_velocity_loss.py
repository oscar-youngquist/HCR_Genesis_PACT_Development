from dataclasses import replace
import torch
import pytest
from test_hard_pact_qp_modes import fixture
from test_hard_pact_reduced_qp import inputs, solver
from test_hard_pact_auxiliary import make_algorithm
from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT


@pytest.mark.parametrize('alpha',[0.,.5,1.])
def test_recovery_blending_execution(alpha):
    task,_,qp,_,_,q,v,quat,_=fixture('every_substep',n=3)
    task._begin_qp_interval();task._qp_execution_alpha=alpha
    def solve(**kw):
        from types import SimpleNamespace
        return SimpleNamespace(tau_safe=torch.full((3,12),.3),
            differentiated_mask=torch.tensor([True,False,False]),
            recovery_mask=torch.tensor([False,True,False]),
            qdd=torch.zeros(3,18),force_world=torch.zeros(3,4,3),
            diagnostics={'selected/equality_max':torch.zeros(3),'selected/inequality_max':torch.zeros(3)},
            stage=torch.tensor([0,1,2]))
    qp.solve=solve
    task._solve_hard_pact_rollout_qp_substep(quat,torch.zeros(3,6))
    torch.testing.assert_close(task.simulator._torques[:2],torch.full((2,12),.1+alpha*.2))
    assert task.simulator._torques[2].eq(.3).all()  # fallback untouched
    count=qp.iteration_diagnostics['rollout'].sums['execution/partially_corrected_rows']
    assert count == (2 if alpha<1 else 0)


def test_masks_normalization_and_empty_nonfinite():
    qp=solver();d=inputs(4)
    d['velocity_command']=torch.zeros(4,3);d['base_linear_velocity_world']=torch.zeros(4,3)
    a=torch.zeros(4,18,requires_grad=True)
    with torch.no_grad():a[:,0]=100;a[:,5]=200;a[2:]=float('nan')
    valid=torch.tensor([True,True,True,False]);accepted=torch.tensor([True,True,False,True])
    xy,yaw,n=qp.velocity_tracking_losses(a,d,valid,accepted)
    assert n==2
    torch.testing.assert_close(xy,torch.tensor(1.,dtype=xy.dtype))
    torch.testing.assert_close(yaw,torch.tensor(4.,dtype=yaw.dtype))
    (xy+yaw).backward();assert a.grad[2:].eq(0).all()
    xy,yaw,n=qp.velocity_tracking_losses(a,d,valid&False,accepted)
    assert n==0 and xy==0 and yaw==0
    a.grad=None;(xy+yaw).backward();assert a.grad.eq(0).all()


def test_actor_only_vjp_holds_estimator_reference_fixed():
    # Shared features feed both actor and a torque-conditioned head, but the
    # separate replay torque conversions are siblings (as in production).
    actor=torch.nn.Parameter(torch.tensor(2.))
    encoder=torch.nn.Parameter(torch.tensor(3.))
    decoder=torch.nn.Parameter(torch.tensor(4.))
    nominal=actor*encoder
    force=(actor*encoder)*decoder
    candidate=nominal*2+force*3
    loss=candidate.square()
    isolated=PPO_HardPACT._actor_only_torque_vjp(loss,nominal,[actor])
    ga,ge,gd=torch.autograd.grad(isolated,(actor,encoder,decoder),allow_unused=True)
    torch.testing.assert_close(ga,4*candidate.detach()*encoder.detach())
    assert ge is None and gd is None
    assert isolated.detach()==loss.detach()
    # Original estimator gradient graph was not detached or consumed.
    original=torch.autograd.grad(loss,(encoder,decoder))
    assert all(g.abs()>0 for g in original)
    stopped=PPO_HardPACT._actor_only_torque_vjp(loss.detach(),nominal,[actor])
    assert not stopped.requires_grad


def test_inherited_and_zero_weights():
    alg=make_algorithm(lambda_projection=.1)
    assert alg.lambda_qp_velocity_xy==alg.lambda_qp_velocity_yaw==.1
    assert alg.qp_config.velocity_tracking_replay_enabled
    zero=make_algorithm(lambda_projection=.1,lambda_qp_velocity_xy=0,lambda_qp_velocity_yaw=0)
    assert not zero.qp_config.velocity_tracking_replay_enabled
    with pytest.raises(ValueError,match='nonnegative'):
        make_algorithm(lambda_qp_velocity_xy=-1)


def test_stopgrad_skips_actor_vjp(monkeypatch):
    alg=make_algorithm(ablation_variant='stopgrad')
    monkeypatch.setattr(alg,'_actor_only_torque_vjp',lambda *a:pytest.fail('stopgrad called VJP'))
    x=torch.ones(1,requires_grad=True)
    assert alg._actor_velocity_objective(x.sum(),x.sum(),x,False) is None
    assert alg._actor_velocity_objective(x.sum(),x.sum(),x,True) is None


@pytest.mark.skipif(not torch.cuda.is_available(),reason='cuPIQP requires CUDA')
def test_real_ppo_actor_term_has_no_estimator_edges(monkeypatch):
    import test_hard_pact_contact_indexing_and_inverse_gate as live
    algorithms=[];checks=[]
    factory=live.make_algorithm
    def make(**kwargs):
        alg=factory(**kwargs);algorithms.append(alg);return alg
    monkeypatch.setattr(live,'make_algorithm',make)
    original=PPO_HardPACT._actor_only_torque_vjp
    def checked(loss,nominal,parameters):
        result=original(loss,nominal,parameters)
        alg=algorithms[0]
        grads=torch.autograd.grad(result,alg.ppo_parameters+alg.auxiliary_parameters,
                                  allow_unused=True,retain_graph=True)
        actor=grads[:len(alg.ppo_parameters)];estimator=grads[len(alg.ppo_parameters):]
        assert any(g is not None and g.abs().sum()>0 for g in actor)
        assert all(g is None for g in estimator)
        checks.append(True)
        return result
    monkeypatch.setattr(PPO_HardPACT,'_actor_only_torque_vjp',staticmethod(checked))
    with torch.device('cuda'):
        live._run_owned_update('random_one_substep','cuda')
    assert checks
