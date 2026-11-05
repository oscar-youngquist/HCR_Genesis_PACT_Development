"""Direct applied-wrench reference tests; no learned/disturbance estimates."""
from types import SimpleNamespace
import math

import pytest
import torch

from test_b1z1_directional_compliance import fixture
from legged_gym.envs.b1z1.directional_compliance import torso_wrench_heading, rotate_yaw
from legged_gym.envs.b1z1.compliance_task import DisturbanceCoverage, ComplianceTask, attach_transition


def test_torso_only_translation_yaw_and_vertical_exclusion():
    model, kw = fixture(3)
    wrench = torch.tensor([[10., -10., 0.], [0., 0., 4.], [0., 0., 0.]], requires_grad=True)
    result = model.advance(**kw, torso_wrench=wrench)
    factor = -math.expm1(-kw['dt']/model.cfg.base_time_constant)
    expected = wrench.detach()*torch.tensor(model.cfg.base_gain)*factor
    torch.testing.assert_close(result.base, expected)
    torch.testing.assert_close(result.position, kw['nominal'])
    torch.testing.assert_close(result.velocity, kw['nominal_velocity'])
    assert result.ee_wrench.eq(0).all() and result.gates.eq(0).all()
    assert not result.base.requires_grad and not result.combined_wrench.requires_grad
    # Pure world vertical force and roll/pitch free couples at the base origin
    # cannot become planar velocity/yaw-rate commands.
    z = torch.zeros(1,3)
    torch.testing.assert_close(torso_wrench_heading(torch.tensor([[0.,0.,30.]]),
        torch.tensor([[5.,6.,0.]]),z,z,torch.tensor([.8])),z)


def test_direct_bypasses_closed_ee_gates_and_ee_only_is_unchanged():
    model, kw = fixture(4); other, _ = fixture(4)
    kw['force'][:,1] = 10  # compliant arm in interior workspace: all EE gates zero
    original = other.advance(**kw)
    result = model.advance(**kw, torso_wrench=torch.tensor([[4.,0.,0.]]).expand(4,-1))
    assert result.gates.eq(0).all() and result.base[:,0].gt(0).all()
    torch.testing.assert_close(result.position, original.position)
    torch.testing.assert_close(result.velocity, original.velocity)
    first, kw = fixture(); second, _ = fixture()
    first.profile[:,:6]=0; second.profile.copy_(first.profile)
    kw['force'][:,1]=10
    a=first.advance(**kw); b=second.advance(**kw,torso_wrench=torch.zeros(4,3))
    for name in ('position','velocity','base','gates','combined_wrench'):
        assert torch.equal(getattr(a,name),getattr(b,name))


def test_sum_before_asymmetric_gains_opposition_and_disabled_axes():
    model, kw = fixture(4)
    model.profile[:,:6]=0  # immediate EE recruitment
    model.profile[:,6:]=torch.tensor([.2,.8,0.,0.,0.,0.])
    kw['force'][:,0]=10
    direct=torch.tensor([[10.,0.,0.],[-10.,0.,0.],[-15.,0.,0.],[0.,20.,5.]])
    result=model.advance(**kw,torso_wrench=direct)
    torch.testing.assert_close(result.combined_wrench[:,0],torch.tensor([20.,0.,-5.,10.]))
    factor=-math.expm1(-kw['dt']/model.cfg.base_time_constant)*model.cfg.base_gain[0]
    torch.testing.assert_close(result.base[:,0],torch.tensor([16.,0.,-1.,8.])*factor)
    assert result.base[:,1:].eq(0).all()
    model.profile[:,6:]=0
    kw['command'][:]=torch.tensor([.2,-.1,.1])
    torch.testing.assert_close(model.advance(**kw,torso_wrench=direct).base,kw['command'])


def test_heading_rotation_and_single_application_point_shift():
    yaw=torch.tensor([math.pi/2, -.7])
    base=torch.tensor([[2.,3.,1.],[-1.,2.,.5]])
    lever=torch.tensor([[.2,.4,0.],[.2,.4,0.]])
    f=torch.tensor([[10.,-5.,30.],[10.,-5.,30.]])
    torque=torch.tensor([[1.,2.,3.],[1.,2.,3.]])
    fw=rotate_yaw(f,yaw);tw=rotate_yaw(torque,yaw)
    point=base+rotate_yaw(lever,yaw)
    result=torso_wrench_heading(fw,tw,point,base,yaw)
    expected=torch.cat((f[:,:2],(torque+torch.cross(lever,f,dim=-1))[:,2:]),-1)
    torch.testing.assert_close(result,expected)
    # Equivalent moment already shifted to the base: no second lever addition.
    shifted=tw+torch.cross(point-base,fw,dim=-1)
    torch.testing.assert_close(torso_wrench_heading(fw,shifted,base,base,yaw),expected)


def test_release_partial_reset_and_compatibility_switch():
    model, kw = fixture(4)
    direct=torch.ones(4,3)*5
    for _ in range(10): model.advance(**kw,torso_wrench=direct)
    before=model.eta.clone()
    model.advance(**kw)
    torch.testing.assert_close(model.eta,before*math.exp(-kw['dt']/model.cfg.base_time_constant))
    before=model.eta.clone();model.reset(torch.tensor([1]))
    assert model.eta[1].eq(0).all()
    assert torch.equal(model.eta[[0,2,3]],before[[0,2,3]])
    a, kw=fixture(4);b,_=fixture(4)
    a.cfg.include_torso_disturbances=False;b.cfg.include_torso_disturbances=False
    kw['force'][:,1]=10; a.profile[:,:6]=0;b.profile.copy_(a.profile)
    one=a.advance(**kw,torso_wrench=direct);two=b.advance(**kw)
    for name in vars(one): assert torch.equal(getattr(one,name),getattr(two,name))
    assert one.torso_wrench.eq(0).all()


def test_applied_source_gates_and_application_point_without_com_gravity():
    # Importing the adapter does not create a simulator.
    from legged_gym.simulator.isaaclab_simulator_b1z1 import _IsaacLabSimulatorB1Z1
    commands=SimpleNamespace(push_robot_base=True,apply_base_external_forces=True,apply_base_external_torques=True)
    point=torch.tensor([[[1.,2.,3.]]])
    sim=SimpleNamespace(_cfg=SimpleNamespace(commands=commands),_base_link_index=0,
        _base_force_world=torch.tensor([[3.,4.,5.]]),_base_torque_world=torch.tensor([[6.,7.,8.]]),
        _robot=SimpleNamespace(data=SimpleNamespace(body_link_pos_w=point)),
        _base_com_bias=torch.ones(1,3)*99,_added_base_mass=torch.tensor([50.]))
    f,t,p=_IsaacLabSimulatorB1Z1.applied_torso_wrench_source(sim)
    assert torch.equal(f,sim._base_force_world) and torch.equal(t,sim._base_torque_world)
    assert torch.equal(p,point[:,0])
    commands.apply_base_external_forces=False
    f,t,_=_IsaacLabSimulatorB1Z1.applied_torso_wrench_source(sim)
    assert f.eq(0).all() and torch.equal(t,sim._base_torque_world)
    commands.push_robot_base=False
    f,t,_=_IsaacLabSimulatorB1Z1.applied_torso_wrench_source(sim)
    assert f.eq(0).all() and t.eq(0).all()


def test_coverage_masks_amplitude_preservation_and_partial_resets():
    torch.manual_seed(7)
    coverage=DisturbanceCoverage([1,1,1],6000,'cpu')
    fractions=torch.bincount(coverage.ids,minlength=3)/6000
    torch.testing.assert_close(fractions,torch.full((3,),1/3),atol=.025,rtol=0)
    ids=torch.arange(6000); raw=torch.rand(6000,3)*20-10
    ee=raw.clone();torso=raw.clone()
    coverage.mask(ee,ids,True);coverage.mask(torso,ids,False)
    assert ee[coverage.ids==1].eq(0).all() and torso[coverage.ids==0].eq(0).all()
    assert torch.equal(ee[coverage.ids!=1],raw[coverage.ids!=1])
    assert torch.equal(torso[coverage.ids!=0],raw[coverage.ids!=0])
    old=coverage.ids.clone();coverage.reset(torch.tensor([0,3]))
    assert torch.equal(coverage.ids[4:],old[4:])
    torch.manual_seed(3);expected=torch.rand(1)
    torch.manual_seed(3);off=DisturbanceCoverage(None,6000,'cpu');off.reset(ids)
    assert torch.equal(torch.rand(1),expected)
    unchanged=raw.clone();off.mask(unchanged,ids,True);assert torch.equal(unchanged,raw)
    for bad in ([0,0,0], [-1,2,3], [1,1], [1,float('nan'),1]):
        with pytest.raises(ValueError): DisturbanceCoverage(bad,1,'cpu')


def test_wrench_logging_replay_and_reset_do_not_advance_targets():
    model, kw=fixture(2)
    result=model.advance(**kw,torso_wrench=torch.tensor([[4.,0.,2.]]).expand(2,-1))
    task=ComplianceTask.__new__(ComplianceTask)
    task.reference=model;task.coverage=DisturbanceCoverage(None,2,'cpu')
    task.wrench_sum=torch.stack((result.torso_wrench,result.ee_wrench,result.combined_wrench),1)*3
    task.wrench_count=torch.ones(2)*3;task.target=None
    task.interval={'command':result.base.clone()}
    stats=task.wrench_statistics(torch.arange(2))
    assert stats['compliance_wrench/torso/fx_N']==4
    assert stats['compliance_wrench/combined/tz_Nm']==2
    alg=SimpleNamespace(device='cpu',transition=SimpleNamespace(actor_physics={}))
    env=SimpleNamespace(compliance=task)
    attach_transition(alg,env)
    captured=alg.transition.actor_physics['command'].clone()
    before=model.eta.clone();task.reset(torch.tensor([0]))
    assert task.wrench_count[0]==0 and task.wrench_count[1]==3
    assert torch.equal(model.eta[1],before[1])
    assert torch.equal(alg.transition.actor_physics['command'],captured)
    assert not captured.requires_grad
