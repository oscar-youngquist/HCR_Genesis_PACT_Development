"""Reference invariants and actor-gradient smoke tests, without simulation."""
from types import SimpleNamespace
from pathlib import Path
import math

import pytest
import torch

from legged_gym.envs.b1z1.directional_compliance import (
    ComplianceCfg,ComplianceReference,signed_response,rotate_yaw,heading,task_position,
    world_position,task_velocity,world_velocity,bound_reference,workspace_gate,
    directional_probe,spherical)
from legged_gym.envs.b1z1.compliance_geometry import RobotGeometry
from legged_gym.envs.b1z1.compliance_task import configure,configure_algorithm,attach_transition
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg


def fixture(n=4):
    cfg=ComplianceCfg(); cfg.collision_gate=False;cfg.joint_gate=False
    model=ComplianceReference(cfg,n,'cpu');model.profile.fill_(1)
    q=torch.zeros(n,3)
    kw=dict(nominal=torch.tensor([.6,0.,0.]).expand(n,-1).clone(),nominal_velocity=q.clone(),
        center=q.clone(),bounds=torch.tensor([[.3,.9],[-1.,1.],[-1.,1.]]),force=q.clone(),lever=torch.tensor([.6,0.,0.]).expand(n,-1),
        command=q.clone(),command_limits=torch.tensor([[-1.,1.],[-1.,1.],[-1.,1.]]),q=q,
        limits=torch.tensor([[-2.,2.]]).expand(n,3,2),jacobian=torch.eye(3).expand(n,-1,-1),
        collision_clearance=lambda q:q.new_ones(q.shape[:-1]),dt=.02)
    return model,kw


def test_signed_gains_and_independent_episode_profiles():
    f=torch.tensor([[10.,-10.,10.]])
    c=torch.tensor([[.1,.8,.3,.9,.4,.2]])
    torch.testing.assert_close(signed_response(f,c,[.01]*3),torch.tensor([[.08,-.03,.02]]))
    m,_=fixture(256)
    m.reset(torch.arange(256)); before=m.profile.clone()
    assert ((before>=0)&(before<=1)).all()
    assert torch.linalg.matrix_rank(before-before.mean(0))==before.shape[1]
    m.delta.fill_(1);m.rate.fill_(1);m.eta.fill_(1)
    m.reset(torch.tensor([1,3]))
    assert m.delta[[1,3]].eq(0).all() and m.rate[[1,3]].eq(0).all() and m.eta[[1,3]].eq(0).all()
    torch.testing.assert_close(m.profile[0],before[0])


def test_arm_first_stiff_handoff_and_zero_base_axes():
    m,k=fixture();k['force'][:,1]=10
    r=m.advance(**k)
    assert r.gates.eq(0).all() and r.base.eq(0).all() and r.position[:,1].gt(0).all()
    m.profile[:,:6]=0
    m.profile[:,10:]=0  # independent yaw disabled
    k['command'][:,0]=.4
    r=m.advance(**k)
    assert r.base[:,1].gt(0).all() and r.base[:,2].eq(0).all()
    torch.testing.assert_close(r.base[:,0],k['command'][:,0])
    m.profile[:,6:10]=0;m.profile[:,10:]=1
    r=m.advance(**k)
    assert r.base[:,1].eq(0).all() and r.base[:,2].gt(0).all()
    m.profile[:,6:]=0
    assert m.advance(**k).base[:,1:].eq(0).all()


def test_exact_filter_release_and_position_velocity_consistency():
    m,k=fixture();k['force'][:,1]=10
    for _ in range(100):r=m.advance(**k)
    before=r.position.clone(); k['force'].zero_()
    after=m.advance(**k)
    assert after.position[:,1].gt(0).all() and after.position[:,1].lt(before[:,1]).all()
    for _ in range(200):r=m.advance(**k)
    torch.testing.assert_close(r.position,k['nominal'],atol=1e-6,rtol=0)
    m,k=fixture();k['dt']=1e-4;k['force'][:,1]=10
    for _ in range(500):r=m.advance(**k)
    p=r.position.clone();v=r.velocity.clone();r=m.advance(**k)
    torch.testing.assert_close((r.position-p)/k['dt'],(r.velocity+v)/2,atol=2e-5,rtol=.02)


def test_frames_world_height_and_yaw_transport():
    q=torch.tensor([[.1,.2,.3,.9]]);q=q/q.norm(dim=-1,keepdim=True)
    omega=torch.tensor([[.2,-.1,.4]]);yaw,rate=heading(q,omega)
    base=torch.tensor([[1.,2.,.6]]);bv=torch.tensor([[.4,-.2,.05]])
    x=torch.tensor([[.5,.2,.9]]);v=torch.tensor([[.1,.2,-.03]])
    p=world_position(x,base,yaw);wv=world_velocity(x,v,base,bv,yaw,rate)
    torch.testing.assert_close(task_position(p,base,yaw),x)
    torch.testing.assert_close(task_velocity(p,wv,base,bv,yaw,rate),v)
    assert p[:,2]==x[:,2]
    dt=1e-4
    p2=world_position(x+dt*v,base+dt*bv,yaw+dt*rate)
    torch.testing.assert_close((p2-p)/dt,wv,atol=.003,rtol=.01)
    f=torch.tensor([[10.,2.,-3.]])
    torch.testing.assert_close(rotate_yaw(rotate_yaw(f,yaw),yaw,True),f)
    # EE orientation is intentionally absent from every compliance-frame API.


def test_workspace_directional_gate_and_combined_feasibility():
    m,k=fixture();k['nominal'][:,0]=.895;k['force'][:,0]=40
    outward=m.advance(**k)
    assert outward.workspace[:,0].min()>.9
    m,k=fixture();k['nominal'][:,0]=.895;k['force'][:,0]=-40
    assert m.advance(**k).workspace[:,0].max()<.01
    m,k=fixture();k['force'][:]=torch.tensor([300.,300.,300.])
    for _ in range(100):r=m.advance(**k)
    s=spherical(r.position-k['center'])
    assert (s<=k['bounds'][:,1]+1e-6).all() and (s>=k['bounds'][:,0]-1e-6).all()
    _,v=bound_reference(r.position,r.velocity,k['center'],k['bounds'])
    torch.testing.assert_close(v,r.velocity,atol=1e-6,rtol=0)
    assert m.delta.abs().max()<1


def test_joint_headroom_direction_and_singular_progress():
    m,k=fixture();q=k['q'].clone();q[:,0]=1.96
    d=q.new_tensor([.04,0,0]).expand_as(q)
    _,out,_=directional_probe(k['jacobian'],q,k['limits'],d,m.cfg)
    _,inside,_=directional_probe(k['jacobian'],q,k['limits'],-d,m.cfg)
    assert out.min()>.9 and inside.max()<.1
    _,bad,progress=directional_probe(k['jacobian']*0,q,k['limits'],d,m.cfg)
    assert bad.eq(1).all() and progress.eq(0).all()


def test_collision_gate_approach_vs_recovery_and_combined_probe():
    m,k=fixture();m.cfg.collision_gate=True;k['force'][:,1]=10
    k['collision_clearance']=lambda q:.02-q[...,1]
    assert m.advance(**k).collision[:,1].min()>.9
    m.reset(torch.arange(4));m.profile.fill_(1);k['force'][:,1]=-10
    assert m.advance(**k).collision[:,1].max()==0
    k['force'][:,:2]=10
    k['collision_clearance']=lambda q:.02-(q[...,:2].sum(-1)).clamp_min(0)
    assert m.advance(**k).collision[:,:2].min()>.9


def test_config_inheritance_disabled_identity_and_model_construction():
    from legged_gym.utils.helpers import class_to_dict
    from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs
    from rsl_rl.modules.actor_critic_b1z1_pact import ActorCriticB1Z1PACT
    from rsl_rl.b1z1_pact_ablations import B1Z1_PACT_ABLATIONS
    cfg=B1Z1PACTCfg();original=class_to_dict(cfg)
    cfg.compliance.enabled=False
    old=class_to_dict(cfg);configure(cfg);assert class_to_dict(cfg)==old
    assert cfg.compliance.compile_geometry is True
    for name in ('ee_velocity_reward_weight', 'ee_velocity_sigma',
                 'actor_ee_velocity_weight', 'actor_ee_velocity_scale'):
        assert not hasattr(ComplianceCfg, name)
    for variant in B1Z1_PACT_ABLATIONS:
        e,p=make_b1z1_pact_ablation_configs(variant);cfg=e();cfg.compliance.enabled=True
        assert cfg.compliance.compile_geometry is True
        assert cfg.compliance.sampling_mode == 'mixture'
        assert cfg.compliance.family_weights == [.20, .20, .15, .20, .15, .10]
        before=cfg.env.num_observations;configure(cfg);configure(cfg)
        assert cfg.env.num_observations==before+12
        algorithm=configure_algorithm(class_to_dict(p().algorithm),cfg)
        assert algorithm['actor_phys_enabled']==p.algorithm.actor_phys_enabled
        assert cfg.rewards.scales.compliance_ee_velocity_weight == 0.5
        assert cfg.rewards.compliance_ee_velocity_sigma == 0.25
        assert algorithm['compliance_ee_velocity_weight'] == 0.25
        assert algorithm['compliance_ee_velocity_scale'] == 0.5
        cfg.rewards.scales.compliance_ee_velocity_weight = 0.7
        configure(cfg)
        assert cfg.rewards.scales.compliance_ee_velocity_weight == 0.7
        algorithm.update(compliance_ee_velocity_weight=0.8, compliance_ee_velocity_scale=0.9)
        overridden = configure_algorithm(algorithm, cfg)
        assert overridden['compliance_ee_velocity_weight'] == 0.8
        assert overridden['compliance_ee_velocity_scale'] == 0.9
        model=ActorCriticB1Z1PACT(cfg.env.num_observations,cfg.env.num_privileged_obs,cfg.env.num_actions,
            cfg.env.num_observations*2,actor_layers=(16,8),critic_layers=(16,8),context_layers=(16,8),
            action_mode=p.policy.action_mode,conditioning_mode=p.policy.conditioning_mode)
        result=model.act_inference(torch.zeros(2,cfg.env.num_observations),torch.zeros(2,cfg.env.num_observations*2))
        assert result.shape==(2,cfg.env.num_policy_actions)
    assert class_to_dict(B1Z1PACTCfg())==original


def test_compliance_reward_scale_registration():
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
    for enabled, weight in ((False, .5), (True, .5), (True, .7), (True, 0.)):
        cfg = B1Z1PACTCfg()
        cfg.compliance.enabled = enabled
        env = SimpleNamespace(cfg=cfg, num_actions=cfg.env.num_policy_actions,
            reward_scales={'compliance_ee_velocity_weight': weight},
            dt=.02, num_envs=2, device='cpu',
            _reward_tracking_compliant_ee_velocity=lambda: torch.ones(2))
        B1Z1PACT._prepare_reward_function(env)
        expected = {'tracking_compliant_ee_velocity': weight * env.dt} if enabled and weight else {}
        assert env.reward_scales == expected
        assert set(env.reward_names) == set(expected)


def test_actor_gradient_and_detached_time_aligned_target():
    from test_b1z1_actor_physics import setup
    from rsl_rl.algorithms import b1z1_actor_physics as physics
    a,batch,actions,context=setup()
    target=torch.full((3,3),.2,requires_grad=True)
    batch['actor_phys_compliance_position']=target
    batch['actor_phys_compliance_velocity']=target*.1
    a.cfg.update(compliance_ee_velocity_weight=.25,compliance_ee_velocity_scale=.5)
    a.dynamics_backend.ee_pose_twist=lambda s,**kw:(s[:,:3]+s[:,19:22],s[:,26:29]+s[:,44:47])
    loss,_=physics.objective(a,batch,actions,context);loss.backward()
    assert torch.isfinite(loss) and a.actor_critic.position_head.weight.grad.abs().sum()>0
    assert a.actor_critic.torque_head.weight.grad.abs().sum()>0
    assert target.grad is None
    for mod in (a.actor_critic.context_encoder,a.actor_critic.explicit_decoder,a.actor_critic.physics_decoder):
        assert all(p.grad is None for p in mod.parameters())
    alg=SimpleNamespace(device='cpu',transition=SimpleNamespace(actor_physics={}))
    env=SimpleNamespace(compliance=SimpleNamespace(interval={'compliance_position':target}))
    attach_transition(alg,env)
    assert not alg.transition.actor_physics['compliance_position'].requires_grad
    with torch.no_grad():target.add_(10)
    assert alg.transition.actor_physics['compliance_position'].max()<1


def geometry(device='cpu',compile_geometry=False):
    cfg=B1Z1PACTCfg()
    cfg.compliance.compile_geometry=compile_geometry
    names=cfg.asset.dof_names
    ids=[i for i,n in enumerate(names) if n.startswith('z1_') and n!='z1_jointGripper']
    path=Path(__file__).parents[1]/cfg.asset.file.split('}/')[1]
    g=RobotGeometry(str(path),names,ids,cfg.asset.base_name,cfg.asset.gripper_name,cfg.asset.foot_name,cfg.compliance,device)
    q=torch.tensor([cfg.init_state.default_joint_angles[n] for n in names],device=device).repeat(3,1)
    return g,q


def test_urdf_jacobian_finite_difference_and_collision_batching():
    g,q=geometry();ee,j,_,_=g.forward(q)
    for column,index in enumerate(g.arm_ids):
        pert=q.clone();pert[:,index]+=1e-3
        minus=q.clone();minus[:,index]-=1e-3
        fd=(g.forward(pert)[0]-g.forward(minus)[0])/.002
        torch.testing.assert_close(fd,j[:,:,column],atol=1e-4,rtol=.01)
    clear=g.clearance(q[:,g.arm_ids],q)
    repeated=g.clearance(q[:,None,g.arm_ids].expand(-1,4,-1),q)
    assert torch.isfinite(clear).all()
    torch.testing.assert_close(repeated,clear[:,None].expand(-1,4))
    g.cfg.collision_batch_size=2
    torch.testing.assert_close(g.clearance(q[:,None,g.arm_ids].expand(-1,4,-1),q),repeated)


def test_cached_probe_geometry_matches_full_fk():
    torch.manual_seed(45)
    g,q=geometry();q=q+torch.randn_like(q)*.12
    ee,j,arm,legs=g.forward(q)
    torch.testing.assert_close(g.arm_points(q[:,g.arm_ids]),arm,atol=1e-6,rtol=1e-5)
    probes=q[:,None,g.arm_ids]+torch.randn(3,12,len(g.arm_ids))*.1
    full=q[:,None].expand(-1,12,-1).clone();full[...,g.arm_ids]=probes
    expected=g._clearance_flat(full.reshape(-1,q.shape[-1])).reshape(3,12)
    torch.testing.assert_close(g.clearance(probes,q,legs),expected,atol=1e-6,rtol=1e-5)


def test_measured_workspace_gate_and_base_filter_release():
    m,k=fixture();k['force'][:,0]=10
    k['measured_position']=k['nominal'].clone();k['measured_position'][:,0]=.899
    assert m.advance(**k).workspace[:,0].min()>.99
    m.profile[:,:6]=0
    for _ in range(100):r=m.advance(**k)
    assert r.base[:,0].min()>0
    before=r.base.clone();k['force'].zero_();r=m.advance(**k)
    assert (r.base[:,0]>0).all() and (r.base[:,0]<before[:,0]).all()
    for _ in range(300):r=m.advance(**k)
    torch.testing.assert_close(r.base,k['command'],atol=1e-6,rtol=0)


def test_invalid_profiles_and_critical_damping_validation():
    cfg=ComplianceCfg();cfg.ee_profile_high=[-1]*6
    with pytest.raises(ValueError):ComplianceReference(cfg,1,'cpu')
    cfg=ComplianceCfg();cfg.zeta=.5
    with pytest.raises(ValueError):ComplianceReference(cfg,1,'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_reference_and_geometry_match_cpu(monkeypatch):
    # Compare algorithms in full float32, independently of training's TF32 mode.
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',False)
    cpu,inputs=fixture()
    cpu.cfg.collision_gate=True
    inputs['force'][:]=torch.tensor([10.,-20.,5.])
    gpu=ComplianceReference(cpu.cfg,4,'cuda:0')
    gpu.profile.copy_(cpu.profile)
    gpu_inputs={k:v.to('cuda:0') if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
    for _ in range(20):
        a,b=cpu.advance(**inputs),gpu.advance(**gpu_inputs)
    for name in ('position','velocity','base','gates'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name).cpu(),atol=1e-6,rtol=1e-5)
    gc,qc=geometry();gg,qg=geometry('cuda:0')
    for a,b in zip(gc.forward(qc),gg.forward(qg)):
        torch.testing.assert_close(a,b.cpu(),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(gc.clearance(qc[:,gc.arm_ids],qc),
                               gg.clearance(qg[:,gg.arm_ids],qg).cpu(),atol=1e-6,rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_compiled_geometry_matches_eager(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',False)
    eager,q=geometry('cuda:0');compiled,_=geometry('cuda:0',True)
    q=q+torch.randn_like(q)*.05
    for a,b in zip(eager.forward(q),compiled.forward(q)):
        torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
    probes=q[:,None,eager.arm_ids]+torch.randn(3,12,len(eager.arm_ids),device=q.device)*.1
    torch.testing.assert_close(eager.clearance(probes,q),compiled.clearance(probes,q),atol=2e-5,rtol=1e-4)
