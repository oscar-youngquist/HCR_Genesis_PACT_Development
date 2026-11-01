"""Fast-path parity, bounded cache layout, and paired RMSE tests."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'notebooks'))
import exp3_representation as a
import exp3_representation_batch as b
from test_exp3_representation import fixture,config


def test_batch_solve_matches_scalar_and_isolates_singular():
    rng=np.random.default_rng(17); n=9; nv=8
    meta=dict(runner='B1Z1PACTRunner',dt=.02,dof_names=['leg','z1_arm'],resolved_environment=dict(
        env=dict(num_policy_actions=4,num_actions=2),asset=dict(foot_name=['a','b','c','d']),
        normalization=dict(obs_scales=dict(ee_force=.1,base_force=.01,grf=.001))))
    rows=[]; fixed=[]; predictions=[]
    for i in range(n):
        row={k:rng.normal(size=2 if 'joint' in k else 3) for k in b.STATE_KEYS}
        row.update(base_quaternion_t=np.array([0.,0.,0.,1.]),base_quaternion_xyzw=np.array([0.,0.,0.,1.]),
            executed_torque_interval_mean=rng.normal(size=2),grf_raw_world_interval_mean=rng.normal(size=(4,3)),
            entry=dict(run='r',source='P',trial_id=i%2,step=i,phase=0,motion='locomotion',condition='nominal',load_range='in',evaluation_seed=0,transition_id=str(i)))
        for field in a.LOADS: row[field]=rng.normal(size=3); row['history_'+field]=row[field].copy()
        rows.append(row); m=rng.normal(size=(nv,nv))
        fixed.append(dict(mass_matrix=m@m.T+np.eye(nv),bias=rng.normal(size=nv),foot_jacobians=rng.normal(size=(4,6,nv)),
            ee_jacobian=rng.normal(size=(6,nv)),base_jacobian=rng.normal(size=(6,nv)),ee_lever=rng.normal(size=3),base_lever=rng.normal(size=3)))
        predictions.append(dict(prediction_ee_force=rng.normal(size=3),prediction_base_wrench=rng.normal(size=6),prediction_grf=rng.normal(size=12)))
    result=b.process_batch((rows,predictions,fixed,meta,'P',True))
    for i,row in enumerate(rows):
        predicted=a.physical_predictions(predictions[i],meta,row)
        for variant,forces in [('decoded',predicted),('measured_loading',dict(ee_force=row['ee_force_world'],base_force=row['base_force_world'],base_moment=row['base_moment_world'],grf=row['grf_raw_world_interval_mean']))]:
            expected=a.velocity_errors(row,forces,fixed[i],meta['dt'])
            np.testing.assert_allclose(result['velocity']['predictions'][variant][i]-result['velocity']['post'][i],expected,atol=1e-12)
    packed=b.process_batch(b.pack_job((rows,predictions,fixed,meta,'P',True)))
    np.testing.assert_array_equal(packed['velocity']['predictions']['decoded'],result['velocity']['predictions']['decoded'])
    fixed[0]['mass_matrix'][:]=0
    isolated=b.process_batch((rows,predictions,fixed,meta,'P',True))
    assert np.isnan(isolated['velocity']['predictions']['decoded'][0]).all()
    np.testing.assert_allclose(isolated['velocity']['predictions']['decoded'][1:],result['velocity']['predictions']['decoded'][1:])


def test_recorded_representation_pinn_flags():
    meta=dict(resolved_training=dict(runner=dict(algorithm_class_name='PPO_B1Z1PACT'),
        policy=dict(pinn_loss_weight=-1.),algorithm={}))
    assert b.representation_pinn_status(meta)==(True,'legacy_runner_default')
    meta['resolved_training']['algorithm']['representation_pinn_enabled']=False
    assert b.representation_pinn_status(meta)==(False,'explicit')
    meta['resolved_training']['algorithm']['representation_pinn_enabled']=True
    meta['resolved_training']['policy']['pinn_loss_weight']=0.
    assert b.representation_pinn_status(meta)==(False,'explicit')
    assert b.representation_pinn_status({})==(None,'unknown')


def test_compact_cache_and_complete_recording_reuse(tmp_path,monkeypatch):
    terms=[dict(mass_matrix=np.eye(8),bias=np.zeros(8),foot_jacobians=np.zeros((4,6,8)),ee_jacobian=np.zeros((6,8)),
        base_jacobian=np.zeros((6,8)),ee_lever=np.zeros(3),base_lever=np.zeros(3),approximation=['test']) for _ in range(10)]
    cache=a.NumericCache(dict(cache_enabled=True,cache_dir=tmp_path/'cache'),signature='test')
    key,_=cache.lookup('terms',0); cache.save(key,b.pack_terms(terms))
    with zipfile.ZipFile(key) as z: assert len(z.namelist())==8
    _,saved=cache.lookup('terms',0)
    np.testing.assert_array_equal(b.unpack_terms(saved)[5]['mass_matrix'],terms[5]['mass_matrix'])
    fixture(tmp_path/'raw',seeds=(0,)); cfg=config(tmp_path)
    cfg.update(evaluation_seeds=[0],cache_enabled=True,cache_dir=str(tmp_path/'cache'))
    cold=a.run_analysis(cfg)
    def unavailable(*args,**kwargs): pytest.fail('Cached recording read raw transitions')
    original=a.chunks; monkeypatch.setattr(a,'chunks',unavailable)
    warm=a.run_analysis(cfg)
    assert warm['timings']['completed_run_hits']==6
    assert cold['selection']['transitions_sha256']==warm['selection']['transitions_sha256']
    assert cold['table_summary']==warm['table_summary']
    assert cold['prediction_source']=='recorded_first'
    assert cold['paired_rollout']['summary']==warm['paired_rollout']['summary']
    pd.testing.assert_frame_equal(pd.DataFrame(cold['paired_rollout']['summary']),pd.DataFrame(warm['paired_rollout']['summary']))
    monkeypatch.setattr(a,'chunks',original)
    # A real protocol/recording edit invalidates only its recording cache.
    path=next((tmp_path/'raw').rglob('metadata.json')); meta=json.loads(path.read_text()); meta['extra_note']='new'; path.write_text(json.dumps(meta))
    assert a.run_analysis(cfg)['timings']['completed_run_hits']==5


def test_metric_process_workers_match_serial(tmp_path):
    fixture(tmp_path/'raw',seeds=(0,)); cfg=config(tmp_path)
    cfg.update(evaluation_seeds=[0],inference_batch_size=4)
    serial=a.run_analysis(cfg); cfg['metric_workers']=2
    parallel=a.run_analysis(cfg)
    assert serial['table_summary']==parallel['table_summary']
    pd.testing.assert_frame_equal(pd.DataFrame(serial['paired_rollout']['summary']),pd.DataFrame(parallel['paired_rollout']['summary']))
