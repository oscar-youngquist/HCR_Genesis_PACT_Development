"""CPU model/contracts only: no simulator instance, PPO trainer, or payloads."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import io
import pytest
import torch
from torch import nn
from legged_gym.envs.go2.go2_mule.go2_mule_config import Go2MULECfgPPO as Config, resolve_go2_mule_configs
from legged_gym.envs.go2.go2_mule.mule_schema import (
    MULESchema, MULEHistory, MULEAuxiliaryBatch, estimate_foot_forces)
from rsl_rl.modules.actor_critic_mule import ActorCriticMULE as Model


@pytest.fixture(autouse=True)
def seed_threads():
    old=torch.get_num_threads(); torch.set_num_threads(1); torch.manual_seed(19)
    yield
    torch.set_num_threads(old)


def setup(phase='joint_adaptation'):
    c=Config(schema=MULESchema(history_length=3), optimization_phase=phase)
    m=Model(c)
    obs=torch.randn(4,45); h=torch.randn(4,c.schema.history_dim)
    ah=torch.randn(4,c.schema.adaptive_history_dim); privileged=torch.randn(4,c.schema.critic_dim)
    return m,obs,h,ah,privileged


def has_grad(module):
    gs=[p.grad for p in module.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in gs)
    return any(g.abs().sum()>0 for g in gs)


def test_paper_widths_and_separate_owners():
    m,o,h,a,p=setup()
    widths={'nominal_actor.mean':[64,512,256,128,12],
        'adaptive_actor.mean':[3*57+19,512,256,128,12],
        'nominal_critic':[51,512,256,128,1], 'adaptive_critic':[51,512,256,128,1],
        'context_estimator.encoder':[135,128,64], 'context_estimator.decoder':[19,64,128,45]}
    for name,w in widths.items():
        layers=[x for x in m.get_submodule(name) if isinstance(x,nn.Linear)]
        assert [layers[0].in_features]+[x.out_features for x in layers]==w
    groups=m.parameter_groups(); ids={k:{id(p) for p in v} for k,v in groups.items()}
    assert not(ids['nominal'] & ids['adaptive'] or ids['nominal'] & ids['auxiliary'] or ids['adaptive'] & ids['auxiliary'])
    assert set.union(*ids.values())=={id(p) for p in m.parameters()}
    estimate=m.context_estimator(h)
    assert estimate.velocity_b.shape==(4,3) and estimate.latent.shape==(4,16)
    assert len([x for x in m.context_estimator.encoder if isinstance(x,nn.ELU)])==2


def test_live_and_mutated_shared_config_inheritance():
    from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_hard_pact_backend_configs
    E,T=make_hard_pact_backend_configs('isaaclab'); e,t=E(),T()
    ec,c=resolve_go2_mule_configs(e,t)
    assert c.schema.history_length==e.env.num_obs_hist==20
    assert c.schema.terrain_dim==143 and c.schema.critic_dim==194
    assert ec.num_actions==12 and ec.num_observations==45
    assert c.learning_rate==t.algorithm.learning_rate
    assert c.auxiliary_learning_rate==t.algorithm.auxiliary_learning_rate
    assert not hasattr(c,'hard_pact_qp') and not hasattr(c,'bard_inverse_enabled')
    e.env.num_obs_hist=7; e.env.num_envs=8; e.control.action_scale=.17
    e.terrain.measured_points_x=[-.1,.2]; e.terrain.measured_points_y=[0.]
    e.normalization.obs_scales.ang_vel=.3
    t.policy.init_noise_std=[.4]*12+[.9]*12
    t.policy.min_noise_std=.1; t.policy.max_noise_std=1.
    t.algorithm.auxiliary_learning_rate=.0007; t.algorithm.num_learning_epochs=2
    t.algorithm.hard_pact_qp={'lambda_load':10000}; t.algorithm.vae_kld_weight=10000
    ec,new=resolve_go2_mule_configs(e,t)
    assert new.schema.history_length==7 and new.schema.terrain_dim==2
    assert ec.action_scale==new.action_scale==.17 and new.num_envs==8
    assert new.schema.angular_velocity_scale==.3 and new.init_noise_std==(.4,)*12
    assert new.auxiliary_learning_rate==.0007 and new.auxiliary_epochs==2
    assert new.beta==1 and new.latent_dim==16
    assert c.schema.history_length==20 and c.schema.terrain_dim==143  # no mutable alias
    assert Config(**asdict(new)).contract()==new.contract()


def test_schema_packing_no_privileged_actor_channels():
    s=MULESchema(history_length=2,angular_velocity_scale=.2, joint_position_scale=2.)
    fields={name:torch.ones(2,w) for name,w,_,_ in s.fields('actor')}
    o=s.pack('actor',fields)
    torch.testing.assert_close(o[:,:3],torch.full((2,3),.2))
    torch.testing.assert_close(o[:,9:21],torch.full((2,12),2.))  # absolute q
    with pytest.raises(ValueError): s.pack('actor',dict(fields,payload_mass=torch.ones(2,1)))
    with pytest.raises(ValueError): MULESchema(joint_order=tuple(reversed(s.joint_order)))


def test_force_proxy_frames_signs_and_invalid_inputs():
    # Rotated non-diagonal leg maps, known signed forces; tau=J.T*f exactly.
    j=torch.randn(3,4,3,3,dtype=torch.float64)+3*torch.eye(3,dtype=torch.float64)
    f=torch.randn(3,4,3,dtype=torch.float64)
    tau=(j.transpose(-1,-2)@f[...,None]).squeeze(-1).flatten(1)
    out=estimate_foot_forces(tau,j)
    assert out.valid.all(); torch.testing.assert_close(out.force_b,f,atol=1e-9,rtol=1e-9)
    tau.requires_grad_(); j.requires_grad_()
    assert not estimate_foot_forces(tau,j).force_b.requires_grad
    j=j.detach().clone(); j[0,0]=0.; j[1,1]=float('nan')
    out=estimate_foot_forces(tau,j)
    assert not out.valid[0,0] and not out.valid[1,1]
    assert out.force_b[0,0].eq(0).all() and torch.isfinite(out.force_b).all()


def test_history_partial_reset_and_owned_snapshot():
    s=MULESchema(history_length=3); hist=MULEHistory(s,2)
    f=estimate_foot_forces(torch.ones(2,12),torch.eye(3).expand(2,4,3,3))
    hist.append(torch.ones(2,45),f,torch.zeros(2,dtype=torch.long),0)
    old=hist.snapshot()
    hist.append(torch.full((2,45),2.),f,torch.tensor([0,1]),1)
    assert hist.observations[0,-2].eq(1).all() and hist.observations[1,:-1].eq(0).all()
    assert old[0].reshape(2,3,45)[:,-1].eq(1).all()
    torch.testing.assert_close(hist.adaptive[:,-1,45:],torch.full((2,12),.01))
    with pytest.raises(ValueError): hist.append(torch.zeros(2,45),f,torch.tensor([0,1]),1)


@pytest.mark.parametrize('phase',['nominal_only','joint_adaptation'])
def test_phase_gradient_ownership_and_no_correction_exploration(phase):
    m,o,h,a,p=setup(phase)
    context=m.context_estimator(h)  # Deliberately connected: distribution boundary must detach.
    dist=m.distributions(o,a,context)
    sample=m.sample_actions(o,h,a)
    if phase=='nominal_only':
        assert dist.adaptive is None and sample.adaptive_raw.eq(0).all()
        assert sample.adaptive_log_prob.eq(0).all()
    replay=m.evaluate_actions(o,a,sample.context,sample.nominal_raw,sample.adaptive_raw,phase=phase)
    vn,va=m.values(p)
    loss=-(replay.nominal_log_prob+replay.adaptive_log_prob).mean()+vn.square().mean()+va.square().mean()
    loss.backward()
    assert has_grad(m.nominal_actor) and has_grad(m.nominal_critic)
    assert has_grad(m.adaptive_actor)==(phase=='joint_adaptation')
    assert has_grad(m.adaptive_critic)==(phase=='joint_adaptation')
    assert not has_grad(m.context_estimator)
    m.zero_grad(); dist.nominal.mean.sum().backward()
    assert not has_grad(m.context_estimator)


def test_phase_switch_preserves_weights_and_clears_inactive_gradients():
    m,o,h,a,p=setup(); before=deepcopy(m.state_dict())
    (m.values(p)[1].sum()).backward(); assert has_grad(m.adaptive_critic)
    m.set_phase('nominal_only'); assert not has_grad(m.adaptive_critic)
    assert all(not p.requires_grad for p in m.parameter_groups()['adaptive'])
    m.set_phase('joint_adaptation')
    for k,v in before.items():
        if isinstance(v,torch.Tensor): torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
    assert all(p.requires_grad for p in m.parameter_groups()['adaptive'])
    with pytest.raises(ValueError): m.evaluate_actions(o,a,None,torch.zeros(4,12),torch.zeros(4,12),phase='nominal_only')


@pytest.mark.parametrize('phase',['nominal_only','joint_adaptation'])
def test_exact_context_replay_shuffling_and_no_rng(phase):
    m,o,h,a,p=setup(phase); sample=m.sample_actions(o,h,a)
    ids=torch.tensor([3,0,2,1]); rng=torch.get_rng_state().clone()
    for _ in range(3):
        replay=m.evaluate_actions(o[ids],a[ids],sample.context.index(ids),sample.nominal_raw[ids],sample.adaptive_raw[ids],phase=phase)
        torch.testing.assert_close(replay.nominal_log_prob,sample.nominal_log_prob[ids],atol=2e-6,rtol=0)
        torch.testing.assert_close(replay.adaptive_log_prob,sample.adaptive_log_prob[ids],atol=2e-6,rtol=0)
    assert torch.equal(rng,torch.get_rng_state())
    with torch.no_grad():
        for par in m.context_estimator.parameters(): par.add_(.1)
    again=m.evaluate_actions(o,a,sample.context,sample.nominal_raw,sample.adaptive_raw,phase=phase)
    torch.testing.assert_close(again.nominal_log_prob,sample.nominal_log_prob,atol=2e-6,rtol=0)
    with pytest.raises(ValueError,match='stored rollout context'):
        m.evaluate_actions(o,a,None,sample.nominal_raw,sample.adaptive_raw,phase=phase)


def test_combination_scaling_only_once_and_inference_deployable_only():
    m,o,h,a,p=setup(); m.config=replace(m.config,action_scale=.2,clip_actions=1.)
    n=torch.full((4,12),.8,requires_grad=True); correction=torch.full_like(n,.6,requires_grad=True)
    cmd=m.compose_actions(n,correction,torch.ones(12))
    torch.testing.assert_close(cmd.combined_raw,torch.full_like(n,1.4))
    torch.testing.assert_close(cmd.position_target,torch.full_like(n,1.2))
    assert cmd.nominal_raw is n and cmd.adaptive_raw is correction
    cmd=m.compose_actions(n*.1,correction*.1,torch.ones(12)); cmd.position_target.sum().backward()
    torch.testing.assert_close(n.grad,torch.full_like(n,.02)); torch.testing.assert_close(correction.grad,n.grad)
    rng=torch.get_rng_state().clone(); one=m.inference(o,h,a); two=m.inference(o,h,a)
    assert torch.equal(rng,torch.get_rng_state())
    torch.testing.assert_close(one.nominal_raw,two.nominal_raw,rtol=0,atol=0)
    torch.testing.assert_close(one.adaptive_raw,two.adaptive_raw,rtol=0,atol=0)


def aux_batch(m,h):
    return MULEAuxiliaryBatch(h,torch.randn(h.shape[0],3),torch.randn(h.shape[0],45),
        torch.ones(h.shape[0],dtype=torch.bool),torch.zeros(h.shape[0],dtype=torch.long),torch.zeros(h.shape[0],dtype=torch.long))


def test_auxiliary_formulas_and_ownership():
    m,o,h,a,p=setup(); b=aux_batch(m,h); noise=torch.randn(4,16)
    b=replace(b,history=h.requires_grad_(),velocity_b=b.velocity_b.requires_grad_(),next_observation=b.next_observation.requires_grad_())
    result=m.context_estimator.losses(b,noise=noise)
    est=m.context_estimator(h,noise=noise); recon=m.context_estimator.decoder(m.context_estimator.features(est))
    torch.testing.assert_close(result.velocity_mse,(est.velocity_b-b.velocity_b).square().mean())
    torch.testing.assert_close(result.reconstruction_mse,(recon-b.next_observation).square().mean())
    torch.testing.assert_close(result.latent_kl,.5*(est.mu.square()+est.logvar.exp()-1-est.logvar).sum(-1).mean())
    torch.testing.assert_close(result.total,result.velocity_mse+result.reconstruction_mse+m.config.beta*result.latent_kl)
    result.total.backward()
    for part in ('encoder','velocity_head','mu_head','logvar_head','decoder'):
        assert has_grad(getattr(m.context_estimator,part))
    assert not has_grad(m.nominal_actor) and not has_grad(m.adaptive_actor)
    assert h.grad is None and b.velocity_b.grad is None and b.next_observation.grad is None


def test_auxiliary_invalid_nan_reset_rows_and_empty_optimizer_skip():
    m,o,h,a,p=setup(); b=aux_batch(m,h); noise=torch.randn(4,16)
    reference=m.context_estimator.losses(b,noise=noise); reference.total.backward()
    grads=[p.grad.clone() for p in m.context_estimator.parameters()]; m.zero_grad()
    padded=replace(b,history=torch.cat((h,torch.full_like(h,float('nan')))),
        velocity_b=torch.cat((b.velocity_b,torch.full_like(b.velocity_b,float('inf')))),
        next_observation=torch.cat((b.next_observation,torch.full_like(b.next_observation,float('nan')))),
        valid=torch.tensor([True]*4+[False]*2+[True]*2),episode_id=torch.zeros(8,dtype=torch.long),
        next_episode_id=torch.tensor([0]*6+[1,1]))
    result=m.context_estimator.losses(padded,noise=torch.cat((noise,noise)))
    torch.testing.assert_close(result.total,reference.total); result.total.backward()
    for expected,par in zip(grads,m.context_estimator.parameters()): torch.testing.assert_close(par.grad,expected)
    optimizer=torch.optim.AdamW(m.context_estimator.parameters(),lr=.01,weight_decay=.1)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    before=deepcopy(m.context_estimator.state_dict()); opt=deepcopy(optimizer.state_dict())
    empty=m.context_estimator.losses(replace(b,valid=torch.zeros(4,dtype=torch.bool)))
    if empty.should_step:
        empty.total.backward(); optimizer.step()
    assert empty.total==0 and not empty.total.requires_grad and not empty.should_step
    for k,v in before.items(): torch.testing.assert_close(v,m.context_estimator.state_dict()[k],rtol=0,atol=0)
    for k,v in opt['state'].items():
        for field,x in v.items(): torch.testing.assert_close(x,optimizer.state_dict()['state'][k][field])


@pytest.mark.parametrize('phase',['nominal_only','joint_adaptation'])
def test_serialization_and_contract_rejection_before_mutation(phase):
    m,o,h,a,p=setup(phase); sample=m.inference(o,h,a)
    file=io.BytesIO(); torch.save(m.state_dict(),file); file.seek(0)
    state=torch.load(file,weights_only=True); restored=Model(replace(m.config,optimization_phase='nominal_only'))
    restored.load_state_dict(state)
    assert restored.phase==phase
    new=restored.inference(o,h,a); torch.testing.assert_close(new.nominal_raw,sample.nominal_raw,rtol=0,atol=0)
    torch.testing.assert_close(new.adaptive_raw,sample.adaptive_raw,rtol=0,atol=0)
    before=deepcopy(restored.state_dict()); broken=deepcopy(state); broken['_extra_state']['contract']['version']=99
    with pytest.raises(ValueError,match='contract'): restored.load_state_dict(broken)
    for k,v in before.items():
        if isinstance(v,torch.Tensor): torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)


def test_env_interface_location_snapshot_and_production_registration():
    from legged_gym.envs.go2.go2_mule.go2_mule import Go2MULE, MULEObservation
    from legged_gym.envs import task_registry
    ec,c=resolve_go2_mule_configs()
    with pytest.raises(TypeError): Go2MULE(ec)  # abstract until physical integration
    assert task_registry.task_classes['go2_mule_isaaclab'].__name__=='IsaacLabMULEEnv'
    forces=estimate_foot_forces(torch.zeros(2,12),torch.eye(3).expand(2,4,3,3))
    obs=MULEObservation(torch.ones(2,45),torch.ones(2,c.schema.critic_dim),torch.ones(2,3),forces,
        torch.zeros(2,dtype=torch.long),torch.ones(2,dtype=torch.bool),7)
    snap=obs.snapshot(); obs.actor.zero_(); obs.velocity_b.zero_()
    assert snap.actor.eq(1).all() and snap.velocity_b.eq(1).all()
    root=Path(__file__).resolve().parents[1]
    assert not (root/'rsl_rl/mule_config.py').exists() and not (root/'rsl_rl/mule_schema.py').exists()
