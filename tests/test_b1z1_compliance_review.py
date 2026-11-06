"""Review regressions: EE-only safety, Cartesian bounds and applied moments."""
from types import SimpleNamespace
import math
import pytest
import torch

from test_b1z1_directional_compliance import fixture, geometry
from legged_gym.envs.b1z1.directional_compliance import collision_approach, world_velocity
from legged_gym.envs.b1z1.compliance_bounds import bound_reference, first_exit
from legged_gym.envs.b1z1.ee_collision import point_segment, segment_segment, segment_box
from rsl_rl.algorithms.b1z1_wrench import about_base, application_lever
from legged_gym.envs.b1z1.compliance_task import ComplianceTask


def test_default_pose_clear_and_only_ee_pairs():
    g,q=geometry();ee,_,arm,legs=g.forward(q)
    assert g.clearance(q[:,g.arm_ids],q).min()>0
    torch.testing.assert_close(g.clearance_pairs(q[:,g.arm_ids],q),g.point_clearance(ee,arm,legs))
    # Moving the sphere into each obstacle is genuinely unsafe. Its own wrist
    # segment is explicitly excluded, never excluded by a proximity heuristic.
    assert g.point_clearance(g.box_center.expand_as(ee),arm,legs)[:,0].max()<0
    assert g.point_clearance(legs[:,0].mean(1),arm,legs)[:,1].max()<0
    assert g.point_clearance(arm[:,0:2].mean(1),arm,legs)[:,-len(g.ee_obstacle_segments)].max()<0
    assert 4 not in g.ee_obstacle_segments and 5 not in g.ee_obstacle_segments


def test_swept_path_not_certified_by_short_probe_and_analytic_distances():
    g,q=geometry();ee,_,arm,legs=g.forward(q)
    start=g.box_center.expand_as(ee)+ee.new_tensor([1.,0.,0.])
    end=start-ee.new_tensor([2.,0.,0.])
    assert g.path_clearance(start,end,arm,legs)[:,0].max()<0
    p=torch.tensor([[0.,.5,0.]]);a=torch.zeros_like(p);b=a.clone();b[:,0]=1
    torch.testing.assert_close(point_segment(p,a,b),torch.tensor([.5]))
    torch.testing.assert_close(segment_segment(p,p+torch.tensor([[0.,-1.,0.]]),a,b),torch.zeros(1))
    torch.testing.assert_close(segment_box(p,p+torch.tensor([[0.,-1.,0.]]),a,torch.ones(3)*.1),torch.tensor([-.1]))
    # Both endpoints can be shallower than an intervening penetration.
    torch.testing.assert_close(segment_box(torch.tensor([[.08,0.,0.]]),torch.tensor([[-.09,0.,0.]]),a,torch.ones(3)*.1),torch.tensor([-.1]))


def test_pairwise_approach_cannot_be_hidden_by_retreat():
    start=torch.tensor([[-.01,.02]])
    assert collision_approach(start,torch.tensor([[.01,-.01]]),.04)>0.99
    assert collision_approach(start,torch.tensor([[.0,.03]]),.04)==0


def actual_urdf_task():
    g,q=geometry();ee,_,arm,_=g.forward(q)
    ref,k=fixture(3);ref.cfg.collision_gate=True;ref.cfg.include_torso_disturbances=False
    # A nominal goal exactly at the current EE isolates the requested offset.
    force=((arm[:,3]+arm[:,4])/2-ee)/ee.new_tensor(ref.cfg.ee_gain)
    base=torch.zeros_like(ee);quat=ee.new_tensor([0.,0.,0.,1.]).expand(3,-1)
    limits=ee.new_tensor([-4.,4.]).expand(q.shape[-1],-1)
    sim=SimpleNamespace(base_pos=base,base_quat=quat,ee_pos=ee,dof_pos=q,dof_pos_limits=limits)
    env=SimpleNamespace(cfg=SimpleNamespace(compliance=ref.cfg),simulator=sim,num_envs=3,
        commands=torch.zeros_like(ee),ee_force_ext_world=force,dt=.02,
        collision_lower_limits=ee.new_tensor([-2.,-2.,-2.]),
        collision_upper_limits=ee.new_tensor([-1.,-1.,-1.]),underground_limit=-2.)
    task=ComplianceTask.__new__(ComplianceTask)
    task.env=env;task.reference=ref;task.geometry=g;task.arm_ids=g.arm_ids
    task.bounds=k['bounds'];task.command_limits=k['command_limits']
    task.center=lambda:base;task.nominal=lambda advance:(ee,torch.zeros_like(ee))
    task.wrench_sum=ee.new_zeros(3,3,3);task.wrench_count=ee.new_zeros(3)
    task.workspace_clearance=segment_box
    return g,q,ee,task


@pytest.mark.parametrize('base_enabled',[True,False])
def test_adapter_yields_before_progressive_collision_handoff(base_enabled):
    g,q,ee,task=actual_urdf_task()
    if not base_enabled:task.reference.profile[:,6:]=0
    task.begin()
    assert task.target.collision.max()==0
    assert (task.target.position-ee).norm(dim=-1).min()>0
    _,_,arm,legs=g.forward(q)
    initial=task.target.position.clone()
    for _ in range(90):
        previous=task.target.position.clone();task.begin()
        assert g.path_clearance(previous,task.target.position,arm,legs).min()>=-1e-6
    assert task.target.collision.max()>.5
    assert (task.target.position-ee).norm(dim=-1).min()>(initial-ee).norm(dim=-1).max()
    if base_enabled:assert task.target.base.abs().max()>0
    else:assert task.target.base.eq(0).all()
    # Reversal and removal recover without crossing an obstacle or retaining
    # a saturated outward filter velocity.
    task.env.ee_force_ext_world.neg_();task.begin()
    assert torch.isfinite(task.target.velocity).all()
    task.env.ee_force_ext_world.zero_()
    for _ in range(20):task.begin()
    assert g.point_clearance(task.target.position,arm,legs).min()>=0


def test_workspace_center_recovery_zero_profile_and_repeated_shifts():
    from legged_gym.envs.b1z1.directional_compliance import cartesian
    from legged_gym.envs.b1z1.compliance_bounds import feasible
    m,k=fixture();m.profile.zero_()
    k['nominal'][:]=cartesian(torch.tensor([.9,-.5,0.]))
    m.advance(**k);m.rate.fill_(.3)
    k['center'][:,2]+=.01;k['nominal'][:,2]+=.01
    assert (m.accepted-k['center']).norm(dim=-1).min()>.904
    r=m.advance(**k)
    assert m.recovering.all() and feasible(r.position,k['center'],k['bounds']).all()
    torch.testing.assert_close(r.position,k['nominal'])
    assert m.delta.eq(0).all() and m.rate.eq(0).all()
    for shift in (.01,-.01,-.02,.03,0.):
        k['center'][:,2]+=shift;k['nominal'][:,2]+=shift
        r=m.advance(**k)
        torch.testing.assert_close(r.position,k['nominal'])
        assert feasible(r.position,k['center'],k['bounds']).all()
    previous=m.accepted.clone();m.reset(torch.tensor([0]))
    torch.testing.assert_close(m.accepted[1:],previous[1:])


def test_unloaded_rigid_nominal_does_not_inherit_old_offset_velocity():
    m,k=fixture();m.profile.zero_();m.advance(**k)
    m.delta[:,1]=.01;m.rate[:,1]=.3
    k['nominal_velocity'][:,0]=.1
    k['nominal'][:,0]+=.002
    r=m.advance(**k)
    torch.testing.assert_close(r.position,k['nominal'])
    torch.testing.assert_close(r.velocity,k['nominal_velocity'])
    assert m.delta.eq(0).all() and m.rate.eq(0).all()


def test_workspace_recovery_retains_only_enabled_offset_and_reinitializes_rate():
    from legged_gym.envs.b1z1.compliance_bounds import feasible
    m,k=fixture();m.profile.zero_();m.profile[:,3]=1
    m.accepted[:]=torch.tensor([.9,0.,0.]);m.initialized.fill_(True)
    m.delta[:]=torch.tensor([.2,.05,.1]);m.rate.fill_(4)
    k['center'][:,2]=.1
    r=m.advance(**k)
    assert m.recovering.all() and feasible(r.position,k['center'],k['bounds']).all()
    torch.testing.assert_close(m.delta[:,0],torch.zeros(4))
    torch.testing.assert_close(m.delta[:,2],torch.zeros(4))
    assert m.delta[:,1].min()>0 and m.rate.eq(0).all()


def test_recovery_anchor_checks_geometry_and_reports_no_safe_anchor():
    from legged_gym.envs.b1z1.compliance_bounds import feasible
    g,q,ee,task=actual_urdf_task()
    task.reference.accepted.fill_(-5);task.reference.initialized.fill_(True)
    task.reference.profile.zero_()
    # Invalid nominal inside torso cannot be used as a recovery anchor. The
    # measured EE is the first valid fallback; this is explicit reinitialization.
    task.nominal=lambda advance:(g.box_center.expand_as(ee),torch.zeros_like(ee))
    task.begin()
    torch.testing.assert_close(task.target.position,ee)
    assert feasible(task.target.position,task.center(),task.bounds).all()
    assert task.target.velocity.eq(0).all()
    task.reference.accepted.fill_(-5)
    task.env.underground_limit=100
    with pytest.raises(RuntimeError,match='No collision-free compliance recovery anchor'):
        task.begin()


def test_lateral_only_boundary_has_no_radial_redirection():
    m,k=fixture();m.profile.zero_();m.profile[:,3]=1
    k['nominal'][:,0]=.895;k['force'][:,1]=50
    for _ in range(150):r=m.advance(**k)
    torch.testing.assert_close(r.position[:,0],k['nominal'][:,0],rtol=0,atol=0)
    assert r.position[:,1].min()>0 and r.position.norm(dim=-1).max()<=.900001
    assert m.delta[:,[0,2]].eq(0).all() and m.rate[:,[0,2]].eq(0).all()


def test_first_exit_inner_hole_angles_and_combined_direction():
    bounds=torch.tensor([[.3,.9],[-.5,.5],[-math.pi,math.pi]])
    start=torch.tensor([[.6,0.,0.]]);end=-start
    alpha=first_exit(start,end,torch.zeros_like(start),bounds)
    torch.testing.assert_close(alpha,torch.tensor([.25]),atol=2e-6,rtol=0)
    for end in (torch.tensor([[.7,.8,.8]]),torch.tensor([[.6,.9,0.]])):
        narrow=bounds.clone();narrow[2]=torch.tensor([-.4,.4])
        p,v=bound_reference(end,end-start,torch.zeros_like(start),narrow,start=start)
        displacement=p-start
        torch.testing.assert_close(torch.cross(displacement,end-start,dim=-1),torch.zeros_like(start),atol=1e-7,rtol=0)
        assert v.eq(0).all()  # stop rather than rotating into disabled axes
        assert first_exit(start,p,torch.zeros_like(start),narrow).min()>.999


def test_rejection_holds_actual_reference_with_moving_nominal():
    m,k=fixture();m.advance(**k)
    previous=m.accepted.clone();m.rate[:,0]=.3
    k['nominal'][:,0]+=.1*k['dt'];k['nominal_velocity'][:,0]=.1
    k['target_safe']=lambda candidate,previous:previous
    r=m.advance(**k)
    torch.testing.assert_close(r.position,previous,rtol=0,atol=0)
    assert r.velocity.eq(0).all()
    torch.testing.assert_close(m.delta,r.position-k['nominal'])
    torch.testing.assert_close(m.rate,-k['nominal_velocity'])
    velocity=world_velocity(r.position,r.velocity,k['center'],torch.ones_like(previous)*.2,
                            torch.zeros(4),torch.ones(4)*.5)
    assert velocity[:,:2].abs().sum()>0  # held task coordinates still have transport
    assert velocity[:,2].eq(0).all()
    other=m.accepted[1].clone();m.reset(torch.tensor([0]))
    torch.testing.assert_close(m.accepted[1],other)
    assert not m.initialized[0] and m.initialized[1]


def test_application_origin_not_randomized_com_and_offcenter_gradients():
    base=torch.zeros(2,3);point=base.clone();point[1,1]=.15
    sim=SimpleNamespace(base_pos=base,_base_com_bias=torch.ones(2,3)*.15,
        applied_torso_wrench_source=lambda:(torch.zeros(2,3),torch.zeros(2,3),point))
    lever=application_lever(sim)
    wrench=torch.tensor([[35.,0.,0.,0.,0.,0.]]).repeat(2,1).requires_grad_()
    lever.requires_grad_()
    shifted=about_base(wrench,lever)
    torch.testing.assert_close(shifted[:,5],torch.tensor([0.,-5.25]))
    shifted.sum().backward()
    assert wrench.grad is not None and lever.grad is None
    # A supplied base-referenced moment uses a zero lever: do not shift twice.
    torch.testing.assert_close(about_base(shifted,torch.zeros_like(lever)),shifted)
    with pytest.raises(RuntimeError,match='verified application point'):
        application_lever(SimpleNamespace(base_pos=base))


def test_actor_wrench_shift_uses_captured_geometry(monkeypatch):
    from test_b1z1_actor_physics import setup
    from rsl_rl.algorithms import b1z1_actor_physics as physics
    from rsl_rl.algorithms import b1z1_wrench
    a,b,actions,context=setup()
    b['actor_phys_state'][:,177]=.15
    context['base_wrench']=torch.zeros(3,6);context['base_wrench'][:,0]=.35
    b['actor_phys_torso_lever_world']=torch.zeros(3,3)
    seen=[]
    def shift(w,l):
        result=about_base(w,l);seen.append(result);return result
    monkeypatch.setattr(b1z1_wrench,'about_base',shift)
    physics.objective(a,b,actions,context)
    assert len(seen)==1 and seen[0][:,3:].eq(0).all()
    b['actor_phys_torso_lever_world'][:,1]=.15
    physics.objective(a,b,actions,context)
    torch.testing.assert_close(seen[-1][:,5],torch.full((3,),-5.25))
