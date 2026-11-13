"""CPU physical contracts; real PhysX evidence belongs to smoke_mule_payload.py."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from legged_gym.envs.go2.go2_mule.rolling_payload import (
    RollingPayloadConfig,sphere_geometry,sphere_inertia,sample_offsets,ball_spawn_state,ball_motion,lost_balls,due_mass_changes)
from legged_gym.envs.go2.go2_mule.go2_mule_config import resolve_go2_mule
from legged_gym.envs.go2.go2_mule.go2_mule_isaaclab import IsaacLabMULEEnv
from legged_gym.simulator.isaaclab_simulator_mule import canonical_leg_jacobians
from legged_gym.envs.go2.go2_mule.mule_schema import estimate_foot_forces
from rsl_rl.runners.mule_runner import training_cuda_rng


def test_geometry_inertia_seed_and_nonoverlap():
    c=RollingPayloadConfig(); bank,ids=sphere_geometry(c,4096); radii=bank[ids]
    torch.testing.assert_close(bank,sphere_geometry(c,4096)[0])
    assert not torch.equal(bank,sphere_geometry(replace(c,seed=2),4096)[0])
    offsets=sample_offsets(c,radii,torch.Generator().manual_seed(9),1.)
    assert (offsets[...,:2].abs()+radii[...,None]+c.wall_thickness<torch.tensor(c.tray_size[:2])/2).all()
    distance=torch.cdist(offsets,offsets)+torch.eye(4)[None]*10
    assert (distance>radii[:,:,None]+radii[:,None,:]).all()
    mass=torch.ones_like(radii)*2
    inertia=sphere_inertia(mass,radii)
    torch.testing.assert_close(inertia.diagonal(dim1=-1,dim2=-2),(.4*mass*radii.square())[...,None].expand(-1,-1,3))


def test_rotating_spawn_motion_and_loss_geometry():
    c=RollingPayloadConfig(); radii=torch.full((2,4),.04)
    offsets=sample_offsets(c,radii,torch.Generator().manual_seed(9),1.)
    q=torch.tensor([[.70710678,0,0,.70710678],[1,0,0,0.]])
    p=torch.randn(2,3); v=torch.randn(2,3); w=torch.randn(2,3)
    relative=torch.randn(2,4,3); omega=torch.randn(2,4,3)
    state=ball_spawn_state(q,p,v,w,offsets,relative,omega)
    r,rdot=ball_motion(q,p,v,w,state)
    torch.testing.assert_close(r,offsets,atol=3e-7,rtol=1e-5)
    torch.testing.assert_close(rdot,relative,atol=1e-6,rtol=1e-5)
    assert not lost_balls(r,radii,c).any()
    r[0,:,2]+=2  # transient upward motion inside tray is NOT loss
    assert not lost_balls(r,radii,c).any()
    r[1,0,0]=1
    assert lost_balls(r,radii,c).tolist()==[False,True]


def test_mass_timer_floor_cap_and_partial_reset():
    c=RollingPayloadConfig()
    assert c.mass_range(1)==(.01,1.) and c.mass_range(1,periodic=True)==(.01,2.)
    assert c.mass_range(0,periodic=True)==c.light_mass_range
    elapsed=torch.tensor([3.99,0.,3.98]); due,new=due_mass_changes(elapsed,.02,4.)
    assert due.tolist()==[True,False,True]
    assert not due_mass_changes(new,.02,4.)[0].any()  # no double trigger from rounding
    old=new.clone(); new[1]=0
    torch.testing.assert_close(old[[0,2]],new[[0,2]])
    assert not due_mass_changes(elapsed,.02,0)[0].any()
    with pytest.raises(ValueError,match='2.0 kg'): RollingPayloadConfig(periodic_mass_range=(0,2.5))
    with pytest.raises(ValueError,match='floor exceeds'): RollingPayloadConfig(mass_floor_kg=.1)


@pytest.mark.parametrize('fixed',[False,True])
def test_named_jacobian_columns_force_sign_and_frame(fixed):
    # Deliberately permuted articulation joints and feet, nonidentity root yaw.
    joints=[5,7,9,0,2,4,1,3,6,8,10,11]; bodies=[4,1,3,2]
    jac=torch.zeros(2,5,6,12 if fixed else 18)
    for k,body in enumerate(bodies):
        jac[:,body-int(fixed),:3,torch.tensor(joints[3*k:3*k+3])+(0 if fixed else 6)]=torch.eye(3)
    q=torch.tensor([[.70710678,0,0,.70710678],[1,0,0,0.]])
    leg=canonical_leg_jacobians(jac,bodies,joints,q,fixed_base=fixed)
    force=torch.randn(2,4,3)
    tau=(leg.transpose(-1,-2)@force[...,None]).squeeze(-1).flatten(1)
    estimate=estimate_foot_forces(tau,leg)
    assert estimate.valid.all()
    torch.testing.assert_close(estimate.force_b,force,atol=1e-6,rtol=1e-5)


def test_physical_factory_isolated_and_requires_explicit_reward():
    a=resolve_go2_mule(environment_overrides={'env.num_envs':3,'control.dt':.04,'noise.noise_level':.23},
                       training_overrides={'seed':19},payload_overrides={'mode':'nominal'})
    b=resolve_go2_mule()
    assert a.sim_params['dt']==.01 and a.environment.control_dt==.04
    assert a.training.seed==a.payload.seed==19 and a.training.num_envs==3
    assert a.shared.noise.noise_level==.23 and b.shared.noise.noise_level!=.23
    assert a.environment.schema.critic_dim==51+143
    with pytest.raises(ValueError,match='Explicit MULE reward'): IsaacLabMULEEnv(a,'cpu')
    with pytest.raises(ValueError,match='persistent'): resolve_go2_mule(environment_overrides={'domain_rand.persistent_disturbance':True})


def test_fresh_cuda_rng_maps_training_gpu_not_visible_count():
    states=[torch.tensor([1]),torch.tensor([2])]
    data=dict(device='cuda:1',cuda_rng=states,training_cuda_index=1)
    mapped=training_cuda_rng(data,'cuda:0',device_count=1)
    assert len(mapped)==1 and mapped[0][0]==0 and mapped[0][1] is states[1]
    with pytest.raises(ValueError,match='device count'): training_cuda_rng(data,'cuda:0',exact=True,device_count=1)
    assert training_cuda_rng(data,'cpu',device_count=0)==[]


def test_physical_contract_rejected_before_network_mutation(tmp_path):
    from test_mule_learning import runner,assert_equal
    r=runner(); r.env.physical_contract=lambda: {'payload':'rolling','version':1}
    r.learn(1); path=tmp_path/'m.pt'; r.save(path)
    before=deepcopy(r.alg.model.state_dict())
    r.env.physical_contract=lambda: {'payload':'liquid','version':1}
    with pytest.raises(ValueError,match='physical contract'): r.load(path)
    assert_equal(r.alg.model.state_dict(),before)
