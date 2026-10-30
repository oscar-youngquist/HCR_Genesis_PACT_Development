"""Recorded-output-only plot checks; no checkpoint, Torch, or simulator required."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
import exp3_film_figures as plots
import exp3_film_oscillation as analysis


def fixture(output):
    cfg=analysis.default_config(dict(output_dir=str(output),nperseg=32,min_nperseg=16,
        oscillation_selection=dict(profile='pushing',loading='loaded',motion_phase='moving'),
        paired_pca_points_per_facet=8,feature_bases={'abc':dict(center=[1,2,3],axes=np.eye(3).tolist(),explained_variance=[.6,.3,.1])}))
    windows=[]
    for stage in (3,4):
        for seed in (0,1):
            for trial in range(3):
                for part in range(2):
                    i=len(windows); t=7+part*2+np.arange(100)*.02; x=np.sin(2*np.pi*2*t)
                    amplitude=(trial+1)*(.01+.001*seed)
                    pre=np.column_stack([x,np.cos(2*np.pi*2*t),.2*x]); post=pre*1.4
                    actions=np.column_stack([x,x*.5,x*.2])*amplitude
                    traces=dict(pre_projection=pre*amplitude,post_projection=post*amplitude,
                        position_recorded=actions,torque_recorded=actions*100,position_unmodulated=actions*.6,
                        torque_unmodulated=actions*60,ee_cartesian_error=actions)
                    arrays=dict(time_seconds=t,successor_time_seconds=t+.02)
                    for name,value in traces.items():
                        sp=analysis.spectrum(value,.02,cfg)
                        arrays['trace_'+name]=value
                        arrays['frequency_'+name]=sp['frequency']; arrays['psd_components_'+name]=sp['psd']
                    np.savez_compressed(output/f'window-{i:05d}.npz',**arrays)
                    windows.append(dict(window_id=i,checkpoint='abc',task='b1z1_pact',profile='pushing',loading='loaded',
                        stage=stage,load_stage=1,motion_phase='moving',analysis_mode='stationary',kind='sustained',
                        evaluation_seed=seed,trial_id=trial,run=f'seed-{seed}',actual_start=t[0],
                        spectral_plan=analysis.spectral_plan(100,.02,cfg),
                        units=dict(signal_components=dict(position_recorded=['z1_joint1','z1_joint2','FR_hip'],position_recorded_arm=['z1_joint1','z1_joint2']))))
    return windows,cfg


def test_paired_projection_uses_same_center_and_axes():
    basis=dict(center=[1.,2.,3.],axes=[[0.,1.,0.],[1.,0.,0.]])
    pre=np.array([[1.,3.,3.],[2.,2.,3.]])
    post=pre+[2.,-1.,0.]
    a,b=plots.paired_projection(pre,post,basis)
    np.testing.assert_allclose(a,[[1,0],[0,1]])
    np.testing.assert_allclose(b-a,[[-1,2],[-1,2]])


def test_alignment_and_trial_weighting(tmp_path):
    windows,cfg=fixture(tmp_path)
    d=plots.read_window(tmp_path,windows[0]); rows=[plots.window_statistics(windows[0],d,cfg)]
    rows.append(dict(rows[0],position_rms=3.,window_id=99))
    rows[0]['position_rms']=1.
    assert plots.trial_statistics(rows).position_rms.item()==2.
    d['successor_time_seconds']=d['time_seconds'].copy()
    np.savez_compressed(tmp_path/'window-00000.npz',**d)
    with pytest.raises(ValueError,match='timestamp'): plots.read_window(tmp_path,windows[0])


def test_smoke_exports_median_selection_and_determinism(tmp_path):
    windows,cfg=fixture(tmp_path)
    report=plots.export_figures(windows,cfg,tmp_path)
    assert report['selection']['trial_id']==1  # Middle-amplitude trial, not the largest.
    assert report['selection']['stage']==3
    selected=report['selection']['window_id']
    trials=pd.read_csv(tmp_path/'oscillation_plotted_trials.csv')
    assert len(trials)==6 and set(trials.stage)=={3}
    assert all(c['trials']==6 for c in report['correlations'])
    before=json.loads((tmp_path/'paired_pca_points.json').read_text())
    assert len(before['points'])==16
    again=plots.export_figures(list(reversed(windows)),cfg,tmp_path)
    assert again['selection']['window_id']==selected
    assert json.loads((tmp_path/'paired_pca_points.json').read_text())==before
    for stem in report['figures']:
        for extension in ('png','pdf'): assert (tmp_path/(stem+'.'+extension)).stat().st_size>1000
    chosen=[plots.window_statistics(w,plots.read_window(tmp_path,w),cfg) for w in windows if w['stage']==3]
    assert plots.select_example(chosen,plots.trial_statistics(chosen),'x',{'window_id':0})==0
    assert 'torch' not in sys.modules and 'isaaclab' not in sys.modules


def test_all_axes_share_preparation_and_preserve_separate_exports(tmp_path,monkeypatch):
    windows,cfg=fixture(tmp_path)
    original=plots.read_window; reads=[]
    def counted(source,window):
        reads.append(window['window_id'])
        return original(source,window)
    monkeypatch.setattr(plots,'read_window',counted)
    reports=plots.export_all_axes(windows,cfg,tmp_path)
    assert list(reports)==list('xyz')
    assert len(reads)==len(windows)+3  # One shared pass plus one example per axis.
    samples=[]
    for axis,report in reports.items():
        output=tmp_path/('axis-'+axis)
        assert report['selection']['ee_axis']==axis
        assert any(row['y']==f'ee_{axis}_rms' for row in report['correlations'])
        saved=json.loads((output/'paired_pca_points.json').read_text())
        assert saved['color_axis']==axis
        samples.append(saved['points'])
        for stem in report['figures']:
            assert (output/(stem+'.png')).is_file()
            assert (output/(stem+'.pdf')).is_file()
    assert samples[0]==samples[1]==samples[2]
    with pytest.raises(ValueError,match='unique'):
        plots.export_all_axes(windows,dict(cfg,oscillation_axes=['x','x']),tmp_path)
