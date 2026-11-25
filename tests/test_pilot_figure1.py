"""Physical coupling and selection contracts without Kit/PhysX."""
import numpy as np
import pytest
import legged_gym.envs  # Existing registry import order; no simulator is constructed.
from legged_gym.utils.pilot_figure1 import checkpoint_path, RUN, CONFIG, PROFILES, partner_motion, spring_force


def test_exact_run_numeric_checkpoint_selection(tmp_path):
    run=tmp_path/RUN;run.mkdir()
    for name in ('model_9.pt','model_100.pt','model_20.pt','model_bad.pt','.model_999.pt'):
        (run/name).touch()
    assert checkpoint_path(run)[1]==100
    assert checkpoint_path(run,20)[0].name=='model_20.pt'
    with pytest.raises(FileNotFoundError): checkpoint_path(run,500)
    with pytest.raises(ValueError): checkpoint_path(tmp_path)


def test_force_sign_cap_rotation_and_equal_opposite_moments():
    p=np.array([.2,.1,.3]);h=np.array([.3,.5,.1]);z=np.zeros(3)
    force=spring_force(p,z,h,z,150.,10.,45.)
    assert force[0]>0 and force[1]>0 and force[2]<0
    assert np.linalg.norm(force)==pytest.approx(45.)
    angle=.9; c,s=np.cos(angle),np.sin(angle)
    R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    rotated=spring_force(R@p,z,R@h,z,150.,10.,45.)
    np.testing.assert_allclose(rotated,R@force)
    np.testing.assert_allclose(force+(-force),0)
    np.testing.assert_allclose(np.cross(R@h,-rotated),R@np.cross(h,-force))


def test_profiles_separate_base_response_without_stiffness_shortcut():
    assert min(PROFILES['support-follow'][:4])>.15
    assert PROFILES['support-follow'][4]<PROFILES['support-follow'][5]
    assert min(PROFILES['support-follow'][6:10])>0
    assert PROFILES['adjust-hold'][6:]==[0.]*6


def test_lift_removal_and_planar_guidance():
    p,v=partner_motion(0,'support-follow');np.testing.assert_array_equal(p,0)
    p,v=partner_motion(7,'support-follow');assert p[2]==pytest.approx(CONFIG['lift_m'])
    p,v=partner_motion(9,'support-follow');assert p[2]==pytest.approx(0)
    p,v=partner_motion(17,'support-follow');assert p[0]==pytest.approx(CONFIG['guidance_m'])
    assert p[2]==0
    p,v=partner_motion(17,'adjust-hold');assert p[1]==pytest.approx(CONFIG['guidance_m'])
    np.testing.assert_allclose(v,0)


def test_zero_separation_rest_force_and_velocity_damping():
    z=np.zeros(3)
    np.testing.assert_array_equal(spring_force(z,z,z,z,150,10,45),z)
    force=spring_force(z,np.array([1.,0,0]),z,z,150,10,45)
    np.testing.assert_array_equal(force,[-10.,0,0])


def test_substep_wrench_has_single_equal_opposite_force_and_lever_moment():
    import torch
    from types import SimpleNamespace as NS
    from legged_gym.scripts.pilot_figure1_scene import InteractionScene
    scene=InteractionScene.__new__(InteractionScene)
    scene.active=True;scene.substep_records=[]
    scene.robot_force=torch.tensor([[0.,0.,-10.]])
    scene.partner_force=torch.tensor([[0.,0.,-5.]])
    scene.handles=torch.tensor([[[-.3,0,1.],[.3,0,1.]]])
    offsets=scene.handles.clone();offsets[:,:,2]=0
    scene._geometry=lambda:(None,offsets,None)
    scene.sim=NS(_gripper_index=0,_robot=NS(data=NS(body_link_pos_w=torch.zeros(1,1,3))))
    calls=[];wrenches=[]
    scene.payload=NS(permanent_wrench_composer=NS(reset=lambda:calls.append('refresh'),
        set_forces_and_torques=lambda **kw:wrenches.append(kw)),write_data_to_sim=lambda:calls.append('write'))
    scene.before_substep()
    assert calls==['refresh','write']
    torch.testing.assert_close(wrenches[0]['forces'],torch.tensor([[[0.,0.,15.]]]))
    torch.testing.assert_close(wrenches[0]['torques'],torch.tensor([[[0.,1.5,0.]]]))
    assert wrenches[0]['is_global']
    assert len(scene.substep_records)==1
def test_summary_separates_settling_from_guidance_and_missing_phases():
    from legged_gym.utils.pilot_figure1 import summarize
    t=np.arange(.02,20.01,.02)
    zeros=np.zeros((len(t),3))
    base=zeros.copy();base[:,0]=.3
    rows=dict(time=t,ee_world=zeros,nominal_world=zeros,base_world=base,
              base_quat_xyzw=np.tile([0.,0.,0.,1.],(len(t),1)),
              joint_gate=np.ones_like(zeros),robot_force=zeros,reference_world=zeros,
              ee_task=zeros,clipped=np.zeros(len(t)))
    result=summarize(rows,'support-follow')
    assert abs(result['guidance_base_progress_m'])<1e-10
    assert result['joint_gate_increase']==0.
    short={k:v[:10] for k,v in rows.items()}
    assert summarize(short,'support-follow')['upward_yield_m'] is None
    assert summarize(rows,'nominal')['behavior_checks']['capability_and_base_motion'] is None


def test_coupling_audit_uses_handle_levers_and_excludes_unattached_rows():
    from legged_gym.utils.pilot_figure1 import audit_coupling
    x=np.zeros((2,1,21));x[0]=np.nan
    x[1,0,3:9]=[-.3,0,0,.3,0,0]
    x[1,0,15:18]=[0,0,-10];x[1,0,18:21]=[0,0,-5]
    x[1,0,9:12]=[0,0,15];x[1,0,12:15]=[0,1.5,0]
    result=audit_coupling({'substep_interaction':x})
    assert result==dict(active_substeps=1,force_balance_max_n=0.,moment_balance_max_nm=0.)

