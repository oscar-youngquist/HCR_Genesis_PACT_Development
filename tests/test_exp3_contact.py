"""CPU momentum/normal-GRF contracts; no simulator, Torch, policy or dynamics solve."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
import exp3_contact as c
import exp3_support as s


def fixture():
    meta=dict(dt=.1, conventions={'forces':'world N'}, resolved_environment={
        'sim':{'gravity':[0,0,-9.81]},'terrain':{'mesh_type':'plane'},'domain_rand':{'push_robots':False}})
    initial={i:{'body_masses_kg':[10.]} for i in (0,1)}
    step=np.repeat(np.arange(6),2); n=len(step)
    data=dict(trial_id=np.tile([0,1],6),step=step,time_seconds=step*.1,successor_time_seconds=(step+1)*.1,
        valid=np.ones(n,bool),alive_for_metrics=np.ones(n,bool),physical_failure=np.zeros(n,bool),
        artificial_reset=np.zeros(n,bool),phase=np.zeros(n,int),robot_com_velocity_world=np.zeros((n,3)),
        ee_force_world=np.zeros((n,3)),base_force_world=np.zeros((n,3)),nonfoot_contact=np.zeros(n,bool))
    return meta,initial,data


def test_static_acceleration_load_alignment_and_chunks():
    meta,initial,data=fixture()
    data['robot_com_velocity_world'][:,0]=2*data['successor_time_seconds']
    data['ee_force_world'][:,0]=np.where(data['step']<3,5,-7)
    one=c.MomentumContact(meta,initial,{'contact_window_seconds':.2}).process(data)
    stream=c.MomentumContact(meta,initial,{'contact_window_seconds':.2})
    parts=[stream.process({k:v[a:b] for k,v in data.items()}) for a,b in [(0,5),(5,9),(9,12)]]
    for k in one: np.testing.assert_allclose(one[k],np.concatenate([p[k] for p in parts]),equal_nan=True)
    np.testing.assert_allclose(one['momentum_contact_world'][2:,0],20-data['ee_force_world'][2:,0])
    np.testing.assert_allclose(one['momentum_contact_world'][2:,2],98.1)
    assert np.isnan(one['momentum_contact_world'][:2]).all()
    valid=np.isfinite(one['momentum_window_duration'])
    assert not ((one['momentum_window_start'][valid]<.3)&(data['successor_time_seconds'][valid]>.3+1e-8)).any()
    # Static equilibrium has gravity support but no horizontal reaction.
    data['robot_com_velocity_world'][:]=0; data['ee_force_world'][:]=0
    np.testing.assert_allclose(c.MomentumContact(meta,initial).process(data)['momentum_contact_world'][2:],np.tile([0,0,98.1],(10,1)))


@pytest.mark.parametrize('field',['physical_failure','artificial_reset','valid','alive_for_metrics'])
def test_invalid_endpoint_not_crossed(field):
    meta,initial,data=fixture(); data[field][4]=field in ('physical_failure','artificial_reset')
    out=c.MomentumContact(meta,initial).process(data)
    assert np.isnan(out['momentum_contact_world'][[4,6]]).all()
    assert np.isfinite(out['momentum_contact_world'][8]).all()


def test_missing_gap_nonfoot_and_unknown_loads():
    meta,initial,data=fixture(); data['nonfoot_contact'][4]=True
    out=c.MomentumContact(meta,initial).process(data)
    assert np.isfinite(out['momentum_contact_world'][4]).all() and not out['momentum_foot_eligible'][4]
    missing={k:np.delete(v,4,axis=0) for k,v in data.items()}
    out=c.MomentumContact(meta,initial).process(missing)
    assert np.isnan(out['momentum_contact_world'][(missing['trial_id']==0)&(missing['step']==3)]).all()
    for field in ('base_force_world','robot_com_velocity_world'):
        bad=copy.deepcopy(data);del bad[field]
        assert np.isnan(c.MomentumContact(meta,initial).process(bad)['momentum_contact_world']).all()
    assert np.isnan(c.MomentumContact(meta,{}).process(data)['momentum_contact_world']).all()
    del data['nonfoot_contact']
    out=c.MomentumContact(meta,initial).process(data)
    assert not out['momentum_foot_eligible'].any()  # Unknown != confirmed foot-only.
    assert np.isfinite(out['momentum_contact_world'][2:]).all()


def test_windows_do_not_mix_equal_resultant_load_stages():
    meta,initial,data=fixture()
    data['ee_force_world'][data['step']>=2,0]=3
    data['base_force_world'][data['step']>=2,0]=-3
    out=c.MomentumContact(meta,initial,{'contact_window_seconds':.3}).process(data)
    complete=np.isfinite(out['momentum_window_duration'])
    assert complete.any()
    assert (out['momentum_window_start'][complete]>=.2).all()


def test_explicit_foot_only_assumption_and_observed_contacts():
    meta,initial,data=fixture(); del data['nonfoot_contact']
    cfg={'contact_assume_foot_only':True,'contact_window_seconds':.2}
    model=c.MomentumContact(meta,initial,cfg)
    predicted=np.tile([0.,0.,98.1],(len(data['trial_id']),1))
    out=model.process(data,predicted)
    assert out['momentum_foot_eligible'][2:].all()
    complete=out['momentum_window_foot_eligible']
    assert complete.any()
    np.testing.assert_allclose(out['momentum_window_prediction'][complete],out['momentum_window_reference'][complete])
    assert model.counts['assumed_foot_only_rows']==len(data['trial_id'])
    assert any('assumed from experiment protocol' in n for n in model.notes)
    data['nonfoot_contact']=np.zeros(len(data['trial_id']),bool); data['nonfoot_contact'][4]=True
    out=c.MomentumContact(meta,initial,cfg).process(data,predicted)
    assert not out['momentum_foot_eligible'][4]  # Assumption never overrides actual contact evidence.
    data.pop('base_force_world')
    assert np.isnan(c.MomentumContact(meta,initial,cfg).process(data)['momentum_contact_world']).all()


def test_representation_assumption_primary_table_and_cache(tmp_path,monkeypatch):
    from test_exp3_representation import fixture as make, config
    import exp3_representation as a
    make(tmp_path/'raw',seeds=(0,))
    path=tmp_path/'raw/b1z1_pact/0-nominal-in'
    meta=json.loads((path/'metadata.json').read_text())
    meta['resolved_environment']['sim']={'gravity':[0,0,-9.81]}
    meta['resolved_environment'].setdefault('domain_rand',{})['push_robots']=False
    meta.setdefault('conventions',{})['forces']='world N'
    (path/'metadata.json').write_text(json.dumps(meta))
    manifest=json.loads((path/'records.json').read_text())
    manifest['fields']['robot_com_velocity_world']=dict(shape=[3],dtype='float64')
    for chunk in manifest['chunks']:
        f=path/chunk['file']
        with np.load(f) as saved: arrays=dict(saved)
        arrays['robot_com_velocity_world']=np.zeros((chunk['rows'],3))
        np.savez_compressed(f,**arrays)
    (path/'records.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(a,'build_model',lambda *a,**k: pytest.fail('Unneeded inference'))
    monkeypatch.setattr(a,'OfflineBard',lambda *a,**k: pytest.fail('Unneeded dynamics'))
    cfg=config(tmp_path);cfg.update(recording_roots=[str(path)],evaluation_seeds=[0],
        cache_enabled=True,cache_dir=str(tmp_path/'cache'),fallback_predictions=False)
    enabled=a.run_analysis(cfg)
    assert enabled['momentum_reference_policy']['assume_foot_only_when_unrecorded']
    assert all(any(r['metric']==m and np.isfinite(r['mean']) for r in enabled['table_summary'])
               for m in ('grf_net_rmse','grf_net_horizontal_rmse','grf_net_window_rmse'))
    out=Path(enabled['output_dir'])
    table=pd.read_csv(out/'native_in_table.csv',header=[0,1],index_col=0)
    assert pd.notna(table.loc['PILOT',('nominal','Momentum Net GRF (xyz) [N]')])
    assert 'foot-only experimental assumption' in (out/'native_in_table.tex').read_text()
    reference=json.loads(next((out/'contact_intervals').glob('*/metadata.json')).read_text())
    assert reference['assume_foot_only_when_unrecorded']
    disabled=a.run_analysis(dict(cfg,representation_assume_foot_only_contacts=False))
    assert not any(r['metric'].startswith('grf_net') for r in disabled['table_summary'])
    assert disabled['timings']['completed_run_hits']==0  # Changed eligibility invalidates old reductions.
    unchanged=lambda result: [r for r in result['table_summary'] if not r['metric'].startswith('grf_net')]
    assert unchanged(enabled)==unchanged(disabled)
    monkeypatch.setattr(a,'chunks',lambda *a,**k: pytest.fail('Warm cache reread'))
    assert a.run_analysis(cfg)['timings']['completed_run_hits']==1


def test_batched_kinematic_com_and_fk(tmp_path):
    path=tmp_path/'test.urdf'
    path.write_text('<robot name="r"><link name="base"/><link name="arm"/>'
        '<joint name="z1_joint" type="revolute"><parent link="base"/><child link="arm"/>'
        '<origin xyz="0 0 1"/><axis xyz="0 0 1"/></joint></robot>')
    tree=c.KinematicTree(path,['z1_joint']); q=np.random.default_rng(5).normal(size=(8,1))
    data=dict(joint_position=q,joint_velocity=np.ones_like(q),base_quaternion_xyzw=np.tile([0,0,0,1.],(8,1)),
              base_velocity_world=np.tile([.2,0,0],(8,1)),base_angular_velocity_world=np.zeros((8,3)))
    com=np.zeros((8,2,7)); com[:,1,0]=1
    v=tree.com_velocity(data,np.ones((8,2)),com,['base','arm'])
    expected=np.column_stack((.2-.5*np.sin(q[:,0]),.5*np.cos(q[:,0]),np.zeros(8)))
    np.testing.assert_allclose(v,expected,atol=1e-13)
    fk=s.RecordedFK(path,'base','arm')
    np.testing.assert_allclose(fk.batch(q,['z1_joint']),np.stack([fk(row,['z1_joint']) for row in q]),atol=1e-13)


def test_normal_only_and_net_metric_contracts():
    meta,_,_=fixture(); sensor=np.array([[[0.,0,10.],[0,0,20.]]])
    pred=sensor+np.array([[[12.,0,1.],[0,8.,-2.]]])
    np.testing.assert_allclose(c.normal_errors(pred,{'grf_raw_world_interval_mean':sensor},meta),[[1,-2]])
    meta['resolved_environment']['terrain']['mesh_type']='trimesh'
    assert c.normal_errors(pred,{'grf_raw_world_interval_mean':sensor},meta) is None
    normals=np.tile([0.,1.,0.],(1,2,1))
    np.testing.assert_allclose(c.normal_errors(pred,dict(grf_raw_world_interval_mean=sensor,surface_normals_world=normals),meta),[[0,8]])


def test_representation_grf_metrics_and_missing_head():
    from exp3_representation_batch import process_batch
    import exp3_representation as a
    meta=dict(runner='B1Z1PACTRunner',dt=.1,dof_names=['leg','z1_arm'],resolved_environment=dict(
        env=dict(num_policy_actions=4,num_actions=2),terrain=dict(mesh_type='plane'),
        asset=dict(foot_name=['FR','FL','RR','RL']),normalization=dict(obs_scales=dict(ee_force=.1,base_force=.01,grf=1.))))
    grf=np.array([[1.,0,10],[0,2,20],[-1,0,30],[0,-2,40]])
    sensor=grf.copy(); sensor[:,:2]=0
    row=dict(entry=dict(run='r',source='P',trial_id=0,step=1,phase=0,motion='stationary',condition='nominal',load_range='in',evaluation_seed=0),
        base_quaternion_t=np.array([0.,0,0,1]),base_quaternion_xyzw=np.array([0.,0,0,1]),
        grf_raw_world_interval_mean=sensor,momentum_contact_world=np.array([2.,-4,100]),
        momentum_foot_eligible=True,momentum_window_reference=np.array([2.,-4,100]),momentum_window_prediction=grf.sum(0))
    def summary(pred):
        result=process_batch(([row],[pred],[None],meta,'P',False))
        stats=a.TrialStatistics(); stats.merge(result['stats'])
        return {r['metric']:r['mean'] for r in stats.seeds() if r['axis']=='all'}
    metrics=summary(dict(prediction_grf=grf.flatten()))
    assert metrics['grf_normal_rmse']==0
    assert metrics['grf_rmse']==pytest.approx(np.sqrt(10/12))
    assert metrics['grf_net_rmse']==pytest.approx(np.sqrt(20/3))
    assert metrics['grf_net_horizontal_rmse']==pytest.approx(np.sqrt(10))
    assert metrics['grf_net_window_rmse']==metrics['grf_net_rmse']
    assert not any(k.startswith('grf_') for k in summary({}))
    row['momentum_foot_eligible']=False; row['momentum_window_prediction']=np.full(3,np.nan)
    metrics=summary(dict(prediction_grf=grf.flatten()))
    assert 'grf_net_rmse' not in metrics and 'grf_normal_rmse' in metrics


def test_vector_reductions_match_scalar_channel_semantics():
    rng=np.random.default_rng(1); names=['hip','z1_joint']; n=40
    data={k:rng.normal(size=(n,2)).astype(np.float32) for k in
          ('pd_weighted_motor_mean_square','ff_weighted_motor_mean_square','executed_torque_mean_square','opposing_channel_cancellation_mean')}
    data.update(ee_position_world=np.ones((n,3)),ee_reference_world=np.zeros((n,3)),
                base_velocity_body=np.ones((n,3)),command_nominal=np.zeros((n,3)))
    values=s._values_batch(data,{'arm':np.array([1]),'leg':np.array([0])},[])
    expected=[s.channel_values({k:v[i] for k,v in data.items()},names) for i in range(n)]
    for key in expected[0]: np.testing.assert_array_equal(values[key],[r[key] for r in expected])
    acc=s._reduce(None,values,[np.arange(n)%3]); out=s._long(acc,['trial'])
    for row in out.itertuples():
        assert row.value==pytest.approx(np.asarray(values[row.metric],float)[np.arange(n)%3==row.trial].mean(),abs=1e-13)


def test_support_cache_and_numerical_invalidation(tmp_path,monkeypatch):
    from test_exp3_support import fixture as make
    import exp3_representation as r
    make(tmp_path/'raw'); cfg=dict(output_dir=str(tmp_path/'out'),exp2_roots=[str(tmp_path/'raw')],
        evaluation_seeds=[0],cache_enabled=True,support_plots=False)
    cold=s.analyze(cfg); before=pd.read_csv(tmp_path/'out/action_support/summary.csv')
    original=r.chunks
    monkeypatch.setattr(r,'chunks',lambda *a: (_ for _ in ()).throw(AssertionError('warm reread')))
    warm=s.analyze(cfg); pd.testing.assert_frame_equal(before,pd.read_csv(tmp_path/'out/action_support/summary.csv'))
    assert warm['timings']['cache_hits']==len(cold['timings']['runs'])
    monkeypatch.setattr(r,'chunks',original)
    changed=s.analyze(dict(cfg,contact_window_seconds=.4))
    assert changed['timings']['cache_misses']==len(cold['timings']['runs'])
