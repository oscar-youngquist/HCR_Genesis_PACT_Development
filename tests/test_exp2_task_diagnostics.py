"""Small offline checks for fixed-window amplification and failure-aware stages."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
import exp2_task_diagnostics as a
from test_exp3_film_oscillation import fixture


def test_component_detrending_bias_and_ratios():
    t=np.arange(200)*.02
    x=np.column_stack([2+3*t,-1+2*t])
    assert a.amplitude(x)<1e-12
    assert a.amplitude(x,'constant')>1
    wave=np.column_stack([np.sin(2*np.pi*3*t),np.cos(2*np.pi*3*t)])
    assert a.ratio(a.amplitude(2*wave+x),a.amplitude(wave),1e-6)==pytest.approx(2.)
    assert np.isnan(a.ratio(1.,0.,1e-6))


def test_stage_failure_denominators_and_joint_base_tolerance(tmp_path):
    m,d,_=fixture(tmp_path)
    d.update(base_velocity_body=np.zeros((500,3)),base_angular_velocity_body=np.zeros((500,3)),command_nominal=np.zeros((500,3)),
             ee_force_applied_world_interval_mean=np.zeros((500,3)))
    m['evaluation'].update(move_start_s=3.,stop_s=7.,loads_n=[25,55,35,0],load_times_s=[4.,5.,6.,8.],
                           ee_tolerance_m=.1,velocity_tolerance_m_s=.2,yaw_tolerance_rad_s=.3)
    d['ee_position_world']=d['ee_reference_world']+np.array([.01,-.02,.03])
    d['physical_failure'][49]=True; d['alive_for_metrics'][49:]=False
    d['base_angular_velocity_body'][:,2]=.31
    first=a.stage_statistics(d,m)[0]
    assert first['prescribed_steps']==150
    assert first['ee_adherence']==49/150
    assert first['base_adherence']==0  # Planar success alone is insufficient.
    assert first['ee_z_bias_m']==pytest.approx(.03)
    after=a.stage_statistics(d,m)[-1]
    assert after['ee_adherence']==0 and np.isnan(after['ee_z_bias_m'])
    assert sum(r['prescribed_steps'] for r in a.stage_statistics(d,m))==500
    assert a.stages(m)[-1]['phase']=='release'


def test_equal_seed_aggregation_and_selection():
    rows=[dict(task='P',checkpoint='a',evaluation_seed=s,trial_id=t,run=str(s),value=v)
          for s,t,v in [(0,0,0.),(0,1,2.),(1,0,5.)]]
    seeds,summary=a.aggregate(pd.DataFrame(rows),['task','checkpoint'],['value'])
    assert summary['mean'].item()==3.
    assert summary['sd'].item()==pytest.approx(np.std([1.,5.],ddof=1))
    frame=pd.DataFrame([dict(evaluation_seed=0,trial_id=i,run='r',window_id=i,ee_z_rms_m=v,ff_ratio=1.) for i,v in enumerate([1.,2.,10.])])
    assert a.representative(frame).trial_id==1
    assert a.representative(frame.iloc[::-1]).window_id==1


def test_offline_pipeline_smoke_and_action_verification(tmp_path,monkeypatch):
    m,d,root=fixture(tmp_path)
    m['evaluation'].update(move_start_s=1.,stop_s=7.,loads_n=[25.,55.,35.,0.],load_times_s=[2.,4.,6.,8.],
                           ee_tolerance_m=.1,velocity_tolerance_m_s=.2,yaw_tolerance_rad_s=.3)
    m['task_stages']={'0':'deliberately stale, not used'}
    d.update(base_velocity_body=np.zeros((500,3)),base_angular_velocity_body=np.zeros((500,3)),command_nominal=np.zeros((500,3)),
             horizon_complete=d['step']==499,ee_force_applied_world_interval_mean=np.zeros((500,3)))
    d['ee_force_applied_world_interval_mean'][:,0]=np.select([d['time_seconds']<2,d['time_seconds']<4,d['time_seconds']<6,d['time_seconds']<8],[0,25,55,35],default=0)
    d['ee_position_t_world'][:,2]+=.02*np.sin(2*np.pi*2*d['time_seconds'])
    weight=np.array([[.1,0,0,0],[0,.1,.1,0]]); heads={k:(weight,np.zeros(2)) for k in ('position','torque')}
    d['action_raw']=np.tile(d['film_features_post_t']@weight.T,(1,2))
    for key in (*a.shared.REQUIRED,*a.exp2.EXTRA):
        if key not in d: d[key]=np.zeros(500)
    np.savez_compressed(root/'numeric.npz',**d)
    (root/'records.json').write_text(json.dumps(dict(chunks=[dict(file='numeric.npz',rows=500)],fields={k:dict(shape=list(v.shape[1:]),dtype=str(v.dtype)) for k,v in d.items()})))
    (root/'metadata.json').write_text(json.dumps(m))
    monkeypatch.setattr(a.film,'load_heads',lambda *args:(heads,None))
    cfg=dict(recording_roots=[str(root)],output_dir=str(tmp_path/'output'),evaluation_seeds=[0],loading=['loaded'],
             transient_margin_seconds=0.,comparison_window_seconds=2.,figure_arm_joints=['z1_arm'])
    result=a.run_analysis(cfg)
    output=Path(cfg['output_dir'])
    assert (output/'exp2_film_oscillation.pdf').is_file()
    assert (output/'exp2_film_oscillation.png').is_file()
    assert (output/'figure_selection.json').is_file()
    assert (output/'vertical_rms.tex').is_file()
    assert (output/'COMPLETE').is_file()
    windows=pd.read_csv(output/'figure_windows.csv')
    assert set(windows.preprocessing)=={'linear','constant'}
    assert windows.ff_ratio.notna().any()
    assert any(r['mean'] is not None for r in result['stage_summary'])
    assert any(r.get('task')=='b1z1_ppo_pos' for r in result['diagnostics'])
    cache={m['checkpoint_sha256']:(heads,None)}
    small_drift=dict(d,action_raw=d['action_raw']+0.002)
    predictions,_,errors=a.action_reference(small_drift,m,cfg,cache,[])
    assert set(predictions)=={'position','torque'}
    assert errors['position_max_error']>0 and errors['torque_max_error']>0
    strict=dict(cfg,action_reference_atol=1e-6,action_reference_rtol=0.)
    assert a.action_reference(small_drift,m,strict,cache,[])[0]=={}
    bad=dict(d,action_raw=np.ones_like(d['action_raw'])*99)
    predictions,_,errors=a.action_reference(bad,m,cfg,cache,[])
    assert predictions=={} and 'torque' in errors
    assert 'isaaclab' not in sys.modules and 'torch' not in sys.modules
