from types import SimpleNamespace
import torch
from scripts.hard_pact_wrench_stress import heavier_waveform,Tapes


def test_heavy_event_caps_preserve_waveform_and_direction():
    target=torch.tensor([[3.,4.,-8.,2.,-3.,5.],[0.,0.,0.,0.,0.,0.]])
    caps=torch.tensor([7.,10.,4.,6.])
    peak=heavier_waveform(target,target,caps)
    assert peak[0,:2].norm()<=7.00001 and peak[0,2]==-10
    assert peak[0,3:5].abs().max()==4 and peak[0,5]==6
    for amplitude in (0.,.25,.5,1.):
        torch.testing.assert_close(heavier_waveform(target*amplitude,target,caps),peak*amplitude)
    torch.testing.assert_close(peak[1],torch.zeros(6))
    torch.testing.assert_close(heavier_waveform(target,target,caps*100),target*2)


def test_tape_index_ignores_episode_reset_and_qp_sampling():
    e=SimpleNamespace(_current_sustained_wrench_world=torch.zeros(2,6),_current_sustained_active_mask=torch.zeros(2,1,dtype=torch.bool))
    o=SimpleNamespace(e=e,t=2);t=Tapes.__new__(Tapes);t.o=o
    t.wrench=torch.arange(48.).reshape(4,2,6);t.active=torch.ones(4,2,1,dtype=torch.bool)
    t.wrench_update(9999);expected=e._current_sustained_wrench_world.clone()
    e._current_sustained_wrench_world.zero_();torch.rand(100)
    t.wrench_update(-5)
    torch.testing.assert_close(e._current_sustained_wrench_world,expected)


def test_selection_prioritizes_survival_then_torso_without_tracking_veto():
    from scripts.run_hard_pact_wrench_stress import rank
    def row(fail,tilt,tracking):
        return dict(physical_resets=fail,stability={k:{'rms':tilt} for k in ('roll_rad','pitch_rad','height_error_m')},tracking_rmse=[tracking]*3)
    assert rank(row(1,1.,10.))<rank(row(2,.1,.1))
    assert rank(row(1,.1,10.))<rank(row(1,.2,.1))


def test_velocity_stress_override_matrix_and_offline_unavailable():
    import numpy as np
    from scripts.eval_hard_pact_tradeoff import INTERVENTION_VARIANTS
    from scripts.run_hard_pact_velocity_stress import stats
    assert INTERVENTION_VARIANTS['B_planar10']==({'planar_velocity_weight':10.},.5)
    assert INTERVENTION_VARIANTS['B_yaw5']==({'yaw_rate_weight':5.},.5)
    assert INTERVENTION_VARIANTS['B_both_lower']==({'planar_velocity_weight':10.,'yaw_rate_weight':5.},.5)
    assert stats([float('nan')])['rms'] is None
    assert stats(np.array([True,False]))['mean']==.5


def test_velocity_stress_masks_reset_and_teleport_without_losing_failures():
    import numpy as np
    from scripts.run_hard_pact_velocity_stress import stats
    reset=np.array([[False,True],[False,False]])
    teleport=np.array([[False,False],[True,False]])
    mask=np.repeat(~reset&~teleport,4,axis=0)
    values=np.where(mask,2.,1000.)
    assert stats(values[mask])['rms']==2.
    assert mask.sum()==8 and reset.sum()==1


def test_oracle_only_changes_wrench_with_single_origin_shift_and_held_local_frame():
    from scripts.hard_pact_wrench_stress import oracle_input
    angle=torch.tensor(0.)
    def rotate(v,q):
        c,s=q.cos(),q.sin();out=v.clone()
        out[...,0]=c*v[...,0]-s*v[...,1];out[...,1]=s*v[...,0]+c*v[...,1]
        return out
    def local(w,q):return torch.cat([rotate(w[:,None,sl],-q)[:,0] for sl in (slice(0,3),slice(3,6))],-1)
    force=torch.tensor([[2.,3.,-10.,1.,2.,3.]])
    calls=[]
    def origin(w):
        calls.append(True);out=w.clone();out[:,3:]+=torch.cross(torch.tensor([[.1,0.,0.]]),w[:,:3],dim=-1);return out
    e=SimpleNamespace(_current_base_quat_xyzw=lambda:angle,_current_sustained_wrench_world=force,_yaw_local_to_world=rotate)
    o=SimpleNamespace(e=e,s=SimpleNamespace(hard_pact_wrench_at_base_origin_world=origin),k=0,record=lambda *a:None)
    data=dict(wrench_pred_world=torch.ones(1,6)*99,environment_ids=torch.tensor([0]),mass_matrix=torch.eye(18),force_pred_world=torch.ones(1,4,3))
    r=oracle_input(o,data,local);expected=torch.tensor([[2.,3.,-10.,1.,3.,3.3]])
    torch.testing.assert_close(r['wrench_pred_world'],expected)
    assert len(calls)==1 and r['mass_matrix'] is data['mass_matrix'] and r['force_pred_world'] is data['force_pred_world']
    assert (data['wrench_pred_world']==99).all()
    angle=torch.tensor(torch.pi/2);o.k=1;force.fill_(100.)
    r=oracle_input(o,data,local)
    torch.testing.assert_close(r['wrench_pred_world'],torch.cat([rotate(expected[:,None,sl],angle)[:,0] for sl in (slice(0,3),slice(3,6))],-1))
    assert len(calls)==1  # No future measurement, second origin shift or inertia addition.
