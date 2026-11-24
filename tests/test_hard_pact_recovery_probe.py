"""Forward-only diagnostic state-machine checks; no optimizer or simulator."""
import torch
from scripts.hard_pact_recovery_probe import RecoveryProbe


def call(p,t,bad=True,reset=True,hard=False,timeout=False):
    v=lambda x:torch.tensor([x])
    return p.update(v(reset),v(reset),v(bad),v(hard),v(timeout),t)


def test_disabled_exact_reset_parity_and_original_event():
    p=RecoveryProbe(1,'cpu',.02,False)
    for t in range(20):
        r=call(p,t,reset=t%2==0)
        assert bool(r['actual_reset'])==(t%2==0)
        assert bool(r['ordinary']) and not bool(r['probe_active'])


def test_latch_full_extension_then_fail_and_reset():
    p=RecoveryProbe(1,'cpu',.02,True)
    for t in range(10):
        r=call(p,t)
        assert bool(r['original_event'])==(t==0)
        assert not bool(r['actual_reset'])
    r=call(p,10)
    assert r['probe_outcome'].item()==2 and bool(r['actual_reset'])
    assert p.start.item()==-1
    assert bool(call(p,11,bad=False,reset=False)['ordinary'])


def test_return_then_sustain_500ms_not_just_deferred_reset():
    p=RecoveryProbe(1,'cpu',.02,True)
    call(p,0)
    for t in range(1,30):
        r=call(p,t,bad=t<5,reset=t<5)
        assert not bool(r['actual_reset'])
    r=call(p,30,bad=False,reset=False)
    assert r['probe_outcome'].item()==1 and bool(r['actual_reset'])
    assert r['probe_first_healthy_step'].item()==5


def test_contact_remains_active_and_relapse_fails_confirmation():
    for hard,bad in [(True,False),(False,True)]:
        p=RecoveryProbe(1,'cpu',.02,True)
        call(p,0);call(p,1,bad=False,reset=False)
        r=call(p,2,bad=bad,hard=hard)
        assert r['probe_outcome'].item()==2 and bool(r['actual_reset'])
    p=RecoveryProbe(1,'cpu',.02,True)
    r=call(p,0,hard=True)
    assert not bool(r['probe_start']) and bool(r['actual_reset'])


def test_timeout_unresolved_partial_reset_isolation():
    p=RecoveryProbe(2,'cpu',.02,True)
    p.update(torch.tensor([True,False]),torch.tensor([True,False]),torch.tensor([True,False]),
             torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),0)
    r=p.update(torch.tensor([True,False]),torch.tensor([True,False]),torch.tensor([True,False]),
               torch.zeros(2,dtype=torch.bool),torch.tensor([True,False]),1)
    assert r['probe_outcome'].tolist()==[3,0]
    assert r['ordinary'].tolist()==[False,True]
    assert p.start.tolist()==[-1,-1]


def test_capture_alignment_exposure_and_failure_denominators(tmp_path):
    import json
    import numpy as np
    from types import SimpleNamespace as NS
    from scripts.hard_pact_recovery_probe import save_probe
    shape=(6,2)
    a={f'control_{key}':np.zeros(shape,dtype=bool) for key in
       ('ordinary','original_event','original_reset','eligible','reset','probe_start','probe_confirming')}
    a['control_ordinary'][:]=True;a['control_ordinary'][2:5,0]=False
    for key in ('original_event','original_reset','eligible','probe_start'):a['control_'+key][1,0]=True
    a['control_reset'][4,0]=True
    a['control_probe_outcome']=np.zeros(shape,dtype=int);a['control_probe_outcome'][4,0]=1
    a['control_probe_first_healthy_step']=np.full(shape,-1);a['control_probe_first_healthy_step'][2:5,0]=2
    a['control_probe_confirming'][2:5,0]=True
    a['control_nonreset_error']=np.ones(shape+(3,));a['control_reasons']=np.zeros(shape+(7,),dtype=bool)
    a['control_episode_steps']=np.ones(shape);a['control_terrain_level']=np.full(shape,5)
    for key in ('joint_position_violation','joint_velocity_violation','torque_violation','correction',
                'contact_probability','contact_truth','velocity_before','stance_slip'):
        a[key]=np.arange(24)[:,None,None]*np.ones((24,2,3))
    observer=NS(e=NS(dt=.02),args=NS(capture_cap=1),out=tmp_path,
        manifest={'target_plan':[dict(family='stairs_down',level=5)]*2})
    result={};save_probe(observer,a,result);p=result['probe']
    assert abs(p['ordinary_env_s']-.18)<1e-9 and abs(p['probe_env_s']-.06)<1e-9
    assert p['original_failures']==1 and p['probe_starts']==1 and p['recovery_fraction']==1
    saved=np.load(tmp_path/'probe_histories.npz')
    assert saved['event0_identity'].tolist()==[0,0,1,4]
    assert saved['event0_correction'].shape==(20,3)
    assert saved['event0_control_ordinary'].shape==(5,)
    assert json.loads((tmp_path/'result.json').read_text())['probe']['captured_events']==1
