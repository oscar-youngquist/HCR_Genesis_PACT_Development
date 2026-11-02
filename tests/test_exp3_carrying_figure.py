"""Portable measured-support figure arithmetic; no simulator or inference."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
import exp3_carrying_render as r
import exp3_carrying_figure as export


def fixture():
    t=np.arange(30)/10
    bounds=[0.,1.,2.,2.5,3.]
    meta=dict(stage_boundaries_seconds=bounds,baseline_interval_seconds=[0.,1.],
              events=[dict(time_seconds=1.,kind='load'),dict(time_seconds=2.,kind='stop'),dict(time_seconds=2.5,kind='release')])
    d=dict(trial=np.zeros(30,int),step=np.arange(30),time_seconds=t,successor_time_seconds=t+.1,
           stage_interval=np.searchsorted(bounds[1:-1],t,side='right'),
           stage_successor=np.searchsorted(bounds[1:-1],np.round(t+.1,10),side='right'),
           valid=np.ones(30,bool),alive_for_metrics=np.ones(30,bool),
           physical_failure=np.zeros(30,bool),artificial_reset=np.zeros(30,bool),
           front_normal_n=np.where(t<1,100.,110.),rear_normal_n=np.where(t<1,200.,215.),
           load_n=np.where(t<1,0.,25.),ee_error_m=t.copy(),geometry_error_m=t.copy(),
           arm_ff_sumsq=np.full(30,2.),arm_ff_count=np.full(30,2,int),
           leg_ff_sumsq=np.full(30,8.),leg_ff_count=np.full(30,2,int))
    identity=dict(method='PILOT',checkpoint='a',evaluation_seed=0,trial=0)
    cfg={**r.DEFAULT_CONFIG,'bin_width_seconds':.2}
    return d,meta,cfg,identity


def test_baseline_units_rms_and_timestamp_alignment():
    d,meta,cfg,identity=fixture()
    d['arm_ff_sumsq'][10:12]=[0.,32.]  # 0 and 4 Nm, two joint-substeps each.
    rows,b,_=r.reduce_trial(d,meta,cfg,identity); frame=pd.DataFrame(rows)
    assert b['front_baseline_n']==100 and b['rear_baseline_n']==200
    assert b['arm_baseline_nm']==1 and b['leg_baseline_nm']==2
    assert b['arm_baseline_joint_substeps']==20
    selected=frame[(frame.metric=='arm_ff_change_nm')&np.isclose(frame.bin_start_s,1.)]
    assert selected.value.item()==pytest.approx(np.sqrt(8)-1)
    assert selected.denominator.item()==4
    assert frame[(frame.metric=='front_change_n')&(frame.bin_start_s>=1)].value.eq(10).all()
    cfg['bin_width_seconds']=.5
    frame=pd.DataFrame(r.reduce_trial(d,meta,cfg,identity)[0])
    get=lambda m: frame[(frame.metric==m)&np.isclose(frame.bin_start_s,1)].value.item()
    assert get('ee_error_m')==pytest.approx(1.1)  # t+dt assignment, not action timestamp.
    assert get('geometry_error_m')==pytest.approx(1.2)


def test_merge_and_smoothing_repool_squares_not_rms():
    d,meta,cfg,identity=fixture()
    d['arm_ff_sumsq'][10:14]=[0,0,32,32]
    cfg['bin_width_seconds']=.4
    frame=pd.DataFrame(r.reduce_trial(d,meta,cfg,identity)[0])
    # Stage boundary 1 splits the global .4 grid at 1 and 1.2.
    assert not any((a<1<b) for a,b in zip(r.edges(meta,cfg)[:-1],r.edges(meta,cfg)[1:]))
    cfg['bin_width_seconds']=.2;cfg['smoothing_bins']=3
    frame=pd.DataFrame(r.reduce_trial(d,meta,cfg,identity)[0])
    first=frame[(frame.metric=='arm_ff_change_nm')&np.isclose(frame.bin_start_s,1)].iloc[0]
    assert first.value==pytest.approx(np.sqrt(8)-1)
    assert first.denominator==8


@pytest.mark.parametrize('boundary',['gap','invalid','failure','reset'])
def test_no_gap_or_terminal_bridging(boundary):
    d,meta,cfg,identity=fixture();cfg['bin_width_seconds']=.5
    if boundary=='gap': d={k:np.delete(v,12,axis=0) for k,v in d.items()}
    else: d[{'invalid':'valid','failure':'physical_failure','reset':'artificial_reset'}[boundary]][12]=boundary!='invalid'
    rows,_,dropped=r.reduce_trial(d,meta,cfg,identity); frame=pd.DataFrame(rows)
    if boundary in ('gap','invalid'):
        assert dropped>0
        assert not ((frame.metric=='front_change_n')&np.isclose(frame.bin_start_s,1)).any()
    else:
        assert frame.last_sample_s.max()<1.3
        assert not (frame.bin_start_s>=1.5).any()
        assert not frame.loc[np.isclose(frame.bin_start_s,1),'surviving_at_bin_end'].any()


def test_missing_baseline_keeps_absolute_errors():
    d,meta,cfg,identity=fixture();d['valid'][5]=False
    rows,b,_=r.reduce_trial(d,meta,cfg,identity)
    assert not b['baseline_valid']
    assert np.isnan(b['arm_baseline_nm'])
    assert set(pd.DataFrame(rows).metric)=={'load_n','ee_error_m','geometry_error_m'}
    cfg['baseline_interval_seconds']=[.5,1.5]
    with pytest.raises(ValueError,match='unloaded'): r.reduce_trial(d,meta,cfg,identity)


def test_equal_seed_weights_and_sample_sd():
    rows=[]
    for seed,values in [(0,[0,0,0]),(1,[4])]:
        for trial,value in enumerate(values):
            rows.append(dict(method='PILOT',checkpoint='a',metric='front_change_n',stage=0,bin=0,
                bin_start_s=0,bin_end_s=1,time_seconds=.5,evaluation_seed=seed,trial=trial,
                value=value,surviving_at_bin_end=True,samples=trial+1,denominator=trial+1))
    seeds,summary=r.aggregate(pd.DataFrame(rows)); s=summary.iloc[0]
    assert s['mean']==2 and s.sd==pytest.approx(np.sqrt(8))
    assert s.contributing_seeds==2 and s.contributing_trials==4
    assert s.valid_samples==7
    assert np.isnan(r.aggregate(pd.DataFrame(rows[:3]))[1].sd.item())


def test_native_float32_values_reduce_in_float64():
    d,meta,cfg,identity=fixture()
    d['front_normal_n']=(200+np.arange(30)*.12345).astype(np.float32)
    rows,b,_=r.reduce_trial(d,meta,cfg,identity)
    assert b['front_baseline_n']==d['front_normal_n'][:10].mean(dtype=np.float64)
    frame=pd.DataFrame(rows)
    actual=frame[(frame.metric=='front_change_n')&np.isclose(frame.bin_start_s,1.)].value.item()
    expected=d['front_normal_n'][10:12].mean(dtype=np.float64)-b['front_baseline_n']
    assert actual==expected


def test_metadata_schedule_ignores_stale_names():
    cfg=dict(move_start_s=3.,velocity_ramp_s=.5,load_times_s=[8.,13.,18.,23.],
             stop_s=20.,finish_s=28.,loads_n=[25.,55.,35.,0.])
    meta=dict(evaluation=cfg,task_stages={str(i):'stale_load65' for i in range(9)})
    s=export.schedule(meta)
    assert s['baseline_interval_seconds']==[3.5,8.]
    assert [v['load_n'] for v in s['stages']][3:8]==[25,55,35,35,35]


def test_package_only_regeneration(tmp_path):
    package=tmp_path/'package';package.mkdir();(package/'data').mkdir()
    d,meta,cfg,identity=fixture();meta['sources']=[]
    for method in ('PILOT','A6'):
        for seed in (0,1):
            f=f'data/{method}-{seed}.npz';np.savez_compressed(package/f,**d)
            meta['sources'].append(dict(method=method,checkpoint_sha256=method,evaluation_seed=seed,data_file=f))
    (package/'metadata.json').write_text(json.dumps(meta))
    (package/'plot_config.json').write_text(json.dumps(cfg))
    shutil.copyfile(r.__file__,package/'render_figure.py')
    for name in ('README.md','dependency_versions.json','build_report.json','COMPLETE'):
        (package/name).write_text('{}')
    r.render(package)
    before=pd.read_csv(package/'plotted_summary.csv')
    archive=export.archive_package(package)
    isolated=tmp_path/'isolated'
    with zipfile.ZipFile(archive) as z:
        assert not any('checkpoint.pt' in x or 'transitions-' in x for x in z.namelist())
        z.extractall(isolated)
    # Isolated Python disables repository/PYTHONPATH imports. Only packaged data used.
    subprocess.run([sys.executable,'-I',str(isolated/'render_figure.py')],cwd=isolated,check=True,capture_output=True)
    pd.testing.assert_frame_equal(before,pd.read_csv(isolated/'plotted_summary.csv'))
    for extension in ('pdf','svg','png'):
        assert (isolated/f'exp3_carrying_compact.{extension}').stat().st_size>1000
    assert 'torch' not in sys.modules and 'isaaclab' not in sys.modules
