"""CPU-only checks for portable fixed-window diagnostics and offline probes."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
import exp2_film_package as p
from test_exp3_film_oscillation import fixture


def data(tmp_path):
    m,d,root=fixture(tmp_path); n=len(d['step'])
    m['resolved_training']['policy']['activation']='elu'
    m['evaluation'].update(loads_n=[25,55,35,0],ee_tolerance_m=.1,velocity_tolerance_m_s=.2,yaw_tolerance_rad_s=.3)
    d.update(base_position_t=np.zeros((n,3)),base_position_world=np.zeros((n,3)),
             command_nominal_t=np.zeros((n,3)),joint_position_t=np.zeros((n,2)),joint_velocity_t=np.ones((n,2))*.1,
             nominal_kp=np.ones((n,2))*4,nominal_kd=np.ones((n,2)),mechanics_kp_scale=np.ones((n,2))*1.2,
             mechanics_kd_scale=np.ones((n,2)),mechanics_motor_strength=np.ones((n,2))*1.5,
             mechanics_feedback_tau_weight=np.ones((n,1))*.6,mechanics_feedforward_tau_weight=np.ones((n,1))*1.3,
             actual_clipping_limits=np.ones((n,2))*2,source_action_id=np.arange(n)-2)
    weights={'0.weight':np.arange(48).reshape(8,6)*.01,'0.bias':np.zeros(8)}
    g,b=p.conditioning_network(d['film_condition_t'],weights,'elu')
    d['film_gamma_t'],d['film_beta_t']=g,b
    d['film_features_post_t']=(1+g)*d['film_features_pre_t']+b
    h=np.array([[.1,0,0,0],[0,.1,.1,0]])
    heads={k:(h,np.zeros(2)) for k in ('position','torque')}; heads['_conditioning']=weights
    d['action_raw']=p.head_actions(d['film_features_post_t'],heads)
    return m,d,root,heads


def test_bias_drift_residual_and_near_zero():
    t=np.arange(100)*.02; x=3+2*t
    fit=p.detrended(x,t)
    assert fit['slope'].item()==pytest.approx(2)
    assert fit['bias'].item()==pytest.approx(x.mean())
    assert fit['rms'].item()<1e-12
    measured=np.array([[1.,2,3],[0,0,0]])
    agree=p.input_agreement(.5*measured,measured,1e-3)
    assert agree['magnitude_ratio']==pytest.approx(.5)
    assert agree['direction_cosine']==pytest.approx(1)
    assert agree['measured_near_zero']==1 and agree['direction_samples']==1
    assert agree['axis1_gain']==pytest.approx(.5)


def test_same_state_torques_delay_and_cancellation(tmp_path):
    m,d,_,heads=data(tmp_path); a=p.action_variants(d,heads)
    t=p.torque_channels(a['original'],d,m)
    np.testing.assert_allclose(t['pd']+t['ff'],t['total'])
    np.testing.assert_allclose(t['pd'],((a['original'][:,:2]*.25)*4*1.2-.1)*.6*1.5)
    assert np.max(abs(t['clipped']))<=2
    delayed=p.torque_channels(a['original'],d,m,True)
    assert not delayed['valid'][:2].any() and delayed['valid'][2:].all()
    assert np.isnan(delayed['total'][:2]).all()
    np.testing.assert_allclose(delayed['total'][2:],t['total'][:-2])
    wave=np.sin(np.arange(100)*.2)[:,None]
    c=p.channel_coupling(wave,-wave,np.arange(100))
    assert c['total_variance']==pytest.approx(0) and c['reinforcement']==pytest.approx(-1)
    assert abs(c['variance_closure'])<1e-12
    c=p.channel_coupling(wave,wave,np.arange(100)); assert c['reinforcement']==pytest.approx(1)
    position=p.torque_channels(a['original'][:,:2],d,m)
    np.testing.assert_allclose(position['pd'],t['pd']/.6)
    assert not position['ff'].any()


def test_endpoint_motion_identity_and_input_alignment(tmp_path):
    m,d,_,_=data(tmp_path)
    d['base_position_world'][:,0]=.01
    theta=.02; d['base_quaternion_xyzw']=np.tile([0,0,np.sin(theta/2),np.cos(theta/2)],(len(d['step']),1))
    split=p.motion_split(d,m['dt'])
    assert abs(split['closure']).max()<1e-12
    measured,_=p.film.measured_condition(d,m)
    np.testing.assert_allclose(measured[:,3],-.02*np.sin(2*np.pi*2*d['time_seconds']),atol=1e-12)


def test_fixed_film_and_local_probes(tmp_path):
    m,d,_,heads=data(tmp_path); variants=p.action_variants(d,heads)
    expected=(1+d['film_gamma_t'].mean(0))*d['film_features_pre_t']+d['film_beta_t'].mean(0)
    np.testing.assert_allclose(variants['fixed'],p.head_actions(expected,heads))
    cfg=p.film.default_config(dict(nperseg=32,joint_groups={'arm':['z1_arm'],'leg':['leg']}))
    before={k:v.copy() for k,v in d.items()}
    metrics,arrays,curves=p.analyze_window(d,m,cfg,heads)
    assert metrics['conditioning_reconstruction_max']<1e-12
    assert metrics['sensitivity.base.coordinate0.arm.ff_rms']>0
    assert all('near-zero' in reason for reason in json.loads(metrics['missing']))
    assert curves['cross_spectrum'].shape[1]==len(curves['frequency_hz'])
    assert np.isfinite(curves['psd']).all()
    assert 'motion_reference_relative' in arrays
    for key in d: np.testing.assert_array_equal(d[key],before[key])
    assert abs(metrics['ee.energy_closure_m2'])<1e-12
    position_heads={k:v for k,v in heads.items() if k!='torque'}
    pd_data=dict(d,action_raw=d['action_raw'][:,:2])
    pm,_,_=p.analyze_window(pd_data,m,cfg,position_heads)
    assert all('near-zero' in reason for reason in json.loads(pm['missing']))
    assert pm['torque.original.arm.ff.rms_nm']==0.
    for name,plus,minus,_ in p.probes(d['film_condition_t'],.01,.05,.05):
        if name=='ee.direction': np.testing.assert_allclose(np.linalg.norm(plus[:,3:],axis=1),np.linalg.norm(d['film_condition_t'][:,3:],axis=1),atol=1e-12)


def test_package_only_regeneration_and_aggregation(tmp_path,monkeypatch):
    m,d,root,heads=data(tmp_path)
    np.savez_compressed(root/'numeric.npz',**d)
    records=dict(chunks=[dict(file='numeric.npz',rows=500)],fields={k:dict(shape=list(x.shape[1:]),dtype=str(x.dtype)) for k,x in d.items()})
    (root/'records.json').write_text(json.dumps(records)); (root/'metadata.json').write_text(json.dumps(m))
    cfg=p.film.default_config(dict(output_dir=str(tmp_path/'out'),comparison_window_seconds=2.,
        nperseg=32,feature_basis_samples=24,package_example_windows=2,joint_groups={'arm':['z1_arm'],'leg':['leg']}))
    builder=p.PackageBuilder(cfg,tmp_path/'out')
    monkeypatch.setattr(p.film,'load_heads',lambda *args,**kwargs:(heads,None))
    run=dict(path=root,meta=m,records=records,trial_ids={0}); builder.register(run,True)
    for wid,start in enumerate([3.,5.]):
        mask=(d['time_seconds']>=start)&(d['time_seconds']<start+2)
        builder.accept({k:v[mask] for k,v in d.items()},m,
            dict(task='b1z1_pact',checkpoint='abc',training_seed=123,evaluation_seed=0,trial_id=0,run=str(root),window_id=wid),
            dict(start=start,end=start+2,stage=wid+2,motion_phase='moving',scheduled_load_n=25+30*wid))
    result=builder.finish(); package=Path(result['directory'])
    with tarfile.open(result['archive']) as tar:
        names=tar.getnames(); assert not any(n.endswith('.pt') or 'numeric.npz' in n for n in names)
    frame=pd.read_csv(package/'windows.csv.gz'); trials,seeds,summary=p.aggregate_windows(frame)
    assert summary.sd.isna().all()  # One evaluation seed, not independent policies.
    assert len(trials)==2 and frame.window_id.is_unique
    manifest=json.loads((package/'manifest.json').read_text())
    assert len(manifest['examples'])<=2 and manifest['spectra_parts']
    assert manifest['complete'] and (package/'COMPLETE').exists()
    assert manifest['dictionary']['torque_original_total']['units']=='Nm'
    assert manifest['dictionary']['ee_error']['units']=='m'
    assert manifest['metric_units']['ee.z.rms_m']=='m'
    assert 'NaN' not in (package/'summary.json').read_text()
    with np.load(package/'paired_features.npz') as pairs:
        for s in manifest['examples']:
            prefix=f'w{int(s["window_id"]):06d}__'
            assert len(pairs[prefix+'step'])<=cfg['feature_pairs_per_window']
            assert np.all(pairs[prefix+'time_seconds']>=s['start_s'])
    # The script must run after the original recordings are inaccessible.
    root.rename(root.with_name('not_accessible'))
    subprocess.run([sys.executable,str(package/'plot_package.py'),str(package),'--output',str(tmp_path/'rebuilt_again')],check=True,cwd=tmp_path,capture_output=True)
    rebuilt=pd.read_csv(tmp_path/'rebuilt_again'/'summary.csv')
    np.testing.assert_allclose(rebuilt['mean'],summary['mean'],equal_nan=True)
    assert json.loads((tmp_path/'rebuilt_again'/'regeneration.json').read_text())['summary_matches']
    assert 'torch' not in sys.modules and 'isaaclab' not in sys.modules


def test_equal_seed_weights_and_deterministic_selection():
    rows=[]
    for s,tid,v in [(0,0,1.),(0,1,3.),(1,0,8.)]:
        for w in range(1+tid):
            rows.append(dict(task='P',checkpoint='abc',profile='pulling',loading='loaded',stage=2,motion='moving',scheduled_load_n=55.,
                evaluation_seed=s,training_seed=10,trial_id=tid,run=str(s),window_id=len(rows),start_s=w,end_s=w+1,samples=10,**{'ee.z.rms_m':v}))
    frame=pd.DataFrame(rows); trial,seed,summary=p.aggregate_windows(frame)
    assert summary['mean'].item()==5. and summary.sd.item()==pytest.approx(np.std([2,8],ddof=1))
    assert [s['window_id'] for s in p.select_examples(frame,3)]==[s['window_id'] for s in p.select_examples(frame.iloc[::-1],3)]
