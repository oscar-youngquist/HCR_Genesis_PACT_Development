"""Estimator-label weighting and causal decoder input contract."""
import pytest
import torch
from rsl_rl.modules.hard_pact_physics import (
    DeploymentPhysicsHeads, height_sample_weights, terrain_relative_torso_height,
    normalized_grf_huber_loss, normalized_wrench_huber_loss)
from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT, replay_qp_torques
from rsl_rl.algorithms.ppo_pact_pos import PPO_PACT_Pos


def heads(**kwargs):
    return DeploymentPhysicsHeads(latent_dim=4,explicit_dim=12,
        grf_hidden_layers=(8,),wrench_hidden_layers=(8,),
        grf_scale_n=[100.]*12,wrench_scale=[100.]*6,wrench_qp_clip=[100.]*6,**kwargs)


def test_height_center_and_uniform():
    points=torch.tensor([[0.,0.,0.],[.2,0.,0.],[0.,.4,0.]])
    base=torch.tensor([[0.,0.,1.]])
    terrain=torch.tensor([[.1,.2,.8]])
    raw=torch.exp(-points[:,:2].square().sum(-1)/(.08))
    weights=height_sample_weights(points)
    torch.testing.assert_close(weights,raw/raw.sum())
    torch.testing.assert_close(terrain_relative_torso_height(base,terrain,weights),
        1-(terrain*raw).sum(-1,keepdim=True)/raw.sum())
    torch.testing.assert_close(terrain_relative_torso_height(base,terrain,
        height_sample_weights(points,'uniform')),terrain_relative_torso_height(base,terrain))
    for sigma in (0.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError):height_sample_weights(points,sigma=sigma)


@pytest.mark.parametrize('loss',[PPO_HardPACT._masked_explicit_loss,PPO_PACT_Pos._masked_explicit_loss])
def test_height_component_preserves_denominator(loss):
    prediction=torch.arange(24,dtype=torch.float32).reshape(2,12)/10
    target=torch.zeros_like(prediction);logits=torch.zeros(2,4);valid=torch.ones(2,dtype=torch.bool)
    original=torch.cat((prediction[:,:3],prediction[:,7:]),-1).square().mean()
    torch.testing.assert_close(loss(prediction,logits,target,valid,0.,1.),original)
    torch.testing.assert_close(loss(prediction,logits,target,valid,0.,4.),
        original+3*prediction[:,11].square().mean()/8)


@pytest.mark.parametrize('fn,width,repeats',[(normalized_grf_huber_loss,12,4),(normalized_wrench_huber_loss,6,1)])
def test_force_axis_weights_and_denominator(fn,width,repeats):
    pred=torch.ones(2,width,requires_grad=True);target=torch.zeros_like(pred)
    valid=torch.ones(2,dtype=torch.bool)
    unit=fn(pred,target,valid,axis_weights=[1.,1.,1.])
    torch.testing.assert_close(unit,fn(pred,target,valid))
    weighted=fn(pred,target,valid,axis_weights=[1.,1.,2.])
    torch.testing.assert_close(weighted,unit*(1+repeats/width))
    grad=torch.autograd.grad(weighted,pred)[0]
    torch.testing.assert_close(grad[:,2],2*grad[:,0])
    if width==6:torch.testing.assert_close(grad[:,3:],grad[:,:1].expand(-1,3))


def test_both_heads_detach_torque_and_explicit_not_latent():
    model=heads();z=torch.randn(3,4,requires_grad=True)
    explicit=torch.randn(3,12,requires_grad=True);previous=torch.randn(3,12,requires_grad=True)
    out=model(z,explicit,previous)
    for prediction in (out.grf_normalized,out.wrench_raw_normalized):
        grads=torch.autograd.grad(prediction.square().sum(),(z,explicit,previous),retain_graph=True,allow_unused=True)
        assert torch.isfinite(grads[0]).all() and grads[0].abs().sum()>0
        assert grads[1:]==(None,None)
    other=model(z,explicit,previous+1)
    assert not torch.equal(out.grf_normalized,other.grf_normalized)
    assert not torch.equal(out.wrench_raw_normalized,other.wrench_raw_normalized)


def test_replay_holds_snapshot_preserves_direct_actor_path():
    target=torch.ones(2,12,requires_grad=True);ff=torch.ones_like(target,requires_grad=True)
    snapshot=torch.full((2,12),7.,requires_grad=True)
    packet={'sampled_qp_q':torch.zeros(2,19),'sampled_qp_v':torch.zeros(2,18),
        'sampled_qp_grf_conditioning_torque':snapshot}
    nominal,conditioning=replay_qp_torques(target,ff,packet,lambda d,q,v:2*(d-q)-v)
    torch.testing.assert_close(conditioning,snapshot)
    assert not conditioning.requires_grad
    nominal.sum().backward()
    assert target.grad.eq(2).all() and ff.grad.eq(1).all() and snapshot.grad is None
    packet['sampled_qp_q'][:,7:]=9
    new_nominal,new_conditioning=replay_qp_torques(target,ff,packet,lambda d,q,v:2*(d-q)-v)
    assert not torch.equal(nominal,new_nominal)
    torch.testing.assert_close(new_conditioning,conditioning)


def test_semantic_checkpoint_guards():
    model=heads();state=model.state_dict();heads().load_state_dict(state)
    legacy=dict(state);del legacy['physics_input_semantics']
    with pytest.raises(RuntimeError,match='previous-executed'):model.load_state_dict(legacy,strict=False)
    with pytest.raises(RuntimeError,match='height label'):heads(height_label_mode='uniform').load_state_dict(state)
    with pytest.raises(RuntimeError,match='height label'):heads(height_label_sigma_m=.3).load_state_dict(state)
