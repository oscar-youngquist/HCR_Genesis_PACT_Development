"""Paired trial inference and explicit-current-state metrics, without a simulator."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
import exp3_rollout_statistics as s
import exp3_representation as a
import exp3_representation_batch as b
from test_exp3_representation import fixture,config


def pair_runs():
    runs=[]
    for name,task,enabled in [('PILOT','b1z1_pact',True),('A7','b1z1_pact_ab10_actor',False)]:
        meta=dict(dt=.1,prescribed_steps=10,checkpoint_sha256=task,evaluation_seed=0,
            arguments=dict(condition='combined-disturbance',load_range='in'),resolved_environment={},
            resolved_training=dict(policy=dict(pinn_loss_weight=1.,action_mode='coupled'),
                algorithm=dict(representation_pinn_enabled=enabled,actor_phys_enabled=True)))
        trials={i:dict(seed=0,motion_mode='locomotion',condition='combined-disturbance',load_range='in',
            command=[.4,0,0],events=[dict(time=3,force=[-25,0,0])],initial_seed=i,mechanics_seed=i+2,
            scenario_hash=str(i),prescribed_steps=10,physical_failure=False) for i in range(2)}
        runs.append(dict(name=name,path=Path(name),meta=meta,trials=trials,
            initial={i:dict(initial_state_hash=str(i)) for i in trials}))
    methods={'PILOT':dict(task='b1z1_pact'),'A7':dict(task='b1z1_pact_ab10_actor')}
    return runs,methods


def test_scenarios_not_just_seeds_and_initial_state():
    runs,_=pair_runs(); on,off=runs
    assert s.scenario_match(on,off,0)[0]
    for field in ('scenario_hash','initial_seed','command','events'):
        changed=copy.deepcopy(off); changed['trials'][0][field]='different'
        assert not s.scenario_match(on,changed,0)[0]
    off['initial'][0]['initial_state_hash']='changed'
    assert not s.scenario_match(on,off,0)[0]
    off['initial'][0].clear()
    assert not s.scenario_match(on,off,0)[0]


@pytest.mark.parametrize('on_task,off_task,isolated',s.PAIRS)
def test_pair_matrix_keeps_actor_status_and_labels_joint_ablation(on_task,off_task,isolated):
    runs,methods=pair_runs()
    methods['PILOT']['task']=on_task; methods['A7']['task']=off_task
    for run,task in zip(runs,(on_task,off_task)):
        train=run['meta']['resolved_training']
        train['policy']['action_mode']='position' if not isolated else 'coupled'
        train['algorithm']['actor_phys_enabled']=True if on_task=='b1z1_pact' else (run['name']=='PILOT' if not isolated else False)
    plan=s.plan_pairs(runs,methods,lambda r:{i:np.arange(1,10) for i in r['trials']})
    assert all(r['configuration_matched'] and r['representation_only_contrast']==isolated for r in plan['audit'])
    assert any('both representation and actor' in d for d in plan['diagnostics'])== (not isolated)


def test_common_support_excludes_failed_step_gaps_and_sparse_sampling():
    runs,methods=pair_runs(); runs[1]['trials'][0].update(physical_failure=True,failure_time_seconds=.6)
    assert s.end_step(runs[1],0)==5
    supports={'PILOT':{0:np.array([1,2,4,5,6]),1:np.arange(1,10)},
              'A7':{0:np.array([1,3,4,5]),1:np.arange(1,10)}}
    plan=s.plan_pairs(runs,methods,lambda r:supports[r['name']])
    row=plan['audit'][0]; assert row['configuration_matched'] and row['common_selected_steps']==2
    np.testing.assert_array_equal(plan['windows']['PILOT'][row['pair_id']][0],[1,4])
    entries=[dict(trial_id=0,step=i) for i in range(10)]
    np.testing.assert_array_equal(np.flatnonzero(s.batch_windows(plan['windows']['PILOT'],entries)[row['pair_id']]),[1,4])
    runs[1]['trials'][0]['failure_time_seconds']=1.
    assert s.end_step(runs[1],0)==9  # Physical failure at the horizon is still failure.
    runs[1]['trials'][0]['failure_time_seconds']=None
    assert s.end_step(runs[1],0) is None


def test_rmse_sufficient_statistics_are_pooled_then_rooted_and_batch_invariant():
    runs,_=pair_runs(); errors=np.zeros((4,8)); errors[:2,:3]=[3,0,0]; errors[2:,:3]=[0,4,0]
    velocity=dict(post=np.zeros_like(errors),predictions={'decoded':errors},entries=[dict(trial_id=0) for _ in errors])
    collector=s.TrialRMSE(); collector.add(velocity,['leg','z1_arm'],{'pair':np.array([True,True,False,False])})
    records=collector.records(runs[0]); full=next(r for r in records if r['block']=='base_linear' and not r['pair_id'])
    common=next(r for r in records if r['block']=='base_linear' and r['pair_id'])
    assert full['rmse']==pytest.approx(np.sqrt((9+9+16+16)/12))
    assert common['rmse']==pytest.approx(np.sqrt(3)) and common['samples']==2
    split=s.TrialRMSE()
    for idx in (slice(0,1),slice(1,None)):
        split.add(dict(post=velocity['post'][idx],predictions={'decoded':errors[idx]},entries=velocity['entries'][idx]),['leg','z1_arm'],{'pair':np.array([True,True,False,False])[idx]})
    assert split.sums==collector.sums


def test_bootstrap_seed_weighting_ci_exact_permutation_and_determinism():
    groups=[np.tile([.5,1.],(100,1)),np.tile([2.,3.],(2,1))]
    settings=s.options(dict(rmse_bootstrap_resamples=300))
    result=s.paired_intervals(groups,settings,'x')
    assert result['difference']==pytest.approx(-.75)  # Not dominated by the longer seed.
    assert result['ci_low']==result['ci_high']==-.75
    again=s.paired_intervals(groups,settings,'x')
    assert result['ci_low']==again['ci_low'] and np.isnan(result['p_value'])
    exact=s.paired_intervals([np.tile([1.,2.],(3,1))],s.options(dict(rmse_permutation_resamples=8,rmse_bootstrap_resamples=30)),'x')
    assert exact['p_value']==pytest.approx(.25)
    single=s.paired_intervals([np.array([[1.,0.]])],settings,'x')
    assert np.isnan(single['ci_low']) and np.isnan(single['reduction_percent'])


def test_nonfinite_common_support_and_configuration_mismatch_are_explicit():
    runs,methods=pair_runs(); runs[1]['meta']['resolved_training']['policy']['action_mode']='position'
    plan=s.plan_pairs(runs,methods,lambda r:{i:np.arange(1,10) for i in r['trials']})
    records=[]
    for run in runs:
        collector=s.TrialRMSE(); value=np.ones((18,8))*(.5 if run['name']=='PILOT' else 1.)
        if run['name']=='PILOT': value[0,0]=np.nan
        entries=[dict(trial_id=i,step=j) for i in (0,1) for j in range(1,10)]
        collector.add(dict(post=np.zeros_like(value),predictions={'decoded':value},entries=entries),['leg','z1_arm'],
            s.batch_windows(plan['windows'][str(run['path'])],entries)); records+=collector.records(run)
    result=s.paired_results(records,plan,dict(rmse_bootstrap_resamples=20))
    primary=next(r for r in result['summary'] if r['scope']=='own_pre_failure' and r['block']=='base_linear')
    common=next(r for r in result['summary'] if r['scope']=='common_pre_failure' and r['block']=='base_linear')
    assert primary['trial_pairs']==2 and common['trial_pairs']==1
    assert not primary['configuration_matched']
    assert all(r['trial_pairs']==0 for r in result['summary'] if r['block'].startswith('explicit'))


@pytest.mark.parametrize('rpy',[(0,0,np.pi/2),(.35,-.2,1.1)])
def test_explicit_prediction_scales_body_frame_and_cartesian_workspace_at_t(rpy):
    from scipy.spatial.transform import Rotation
    orientation=Rotation.from_euler('xyz',rpy); q=orientation.as_quat()[None]
    meta=dict(resolved_environment=dict(normalization=dict(obs_scales=dict(lin_vel=2.,
        ee_sphe_radius_cmd=2.,ee_sphe_pitch_cmd=3.,ee_sphe_yaw_cmd=4.))))
    reference=np.array([[1,2,3.]])
    row=dict(base_quaternion_t=q,base_velocity_world_t=orientation.apply([[1,0,0.]]),
        commands_t=np.array([[0,0,0,.5,0,0]]),ee_reference_t_world=reference,
        ee_position_t_world=reference+.6*np.array([[np.cos(rpy[2]),np.sin(rpy[2]),0.]]))
    pred=dict(prediction_base_velocity=np.array([[2.,0.,999.]]),prediction_ee_position=np.array([[2.2,0.,0.]]))
    errors=b.explicit_errors(pred,meta,row)
    np.testing.assert_allclose(errors['explicit_base_velocity'],0,atol=1e-14)
    np.testing.assert_allclose(errors['explicit_ee_position'],0,atol=1e-14)
    pred['prediction_ee_position'][0,0]+= .2
    assert b.explicit_errors(pred,meta,row)['explicit_ee_position'][0,0]==pytest.approx(.1)
    alias=dict(prediction_explicit=np.concatenate([pred['prediction_base_velocity'],pred['prediction_ee_position'],np.zeros((1,6))],axis=1))
    np.testing.assert_allclose(b.explicit_errors(alias,meta,row)['explicit_ee_position'],[ [.1,0,0] ],atol=1e-14)
    row['base_velocity_world']=np.ones((1,3))*1000; row['ee_position_world']=np.ones((1,3))*1000
    np.testing.assert_allclose(b.explicit_errors(pred,meta,row)['explicit_base_velocity'],0,atol=1e-14)


def matched_fixture(root):
    tasks=('b1z1_pact','b1z1_pact_ab10_actor'); fixture(root,tasks=tasks,seeds=(0,))
    for path in root.rglob('metadata.json'):
        meta=json.loads(path.read_text()); enabled=meta['task']=='b1z1_pact'
        meta['resolved_training']['algorithm']=dict(representation_pinn_enabled=enabled,actor_phys_enabled=True)
        meta['resolved_training']['policy'].update(pinn_loss_weight=1.,action_mode='coupled')
        meta['resolved_environment']['normalization']['obs_scales'].update(lin_vel=2.,ee_sphe_radius_cmd=1.,ee_sphe_pitch_cmd=1.,ee_sphe_yaw_cmd=1.)
        path.write_text(json.dumps(meta)); directory=path.parent
        trials=json.loads((directory/'trials.json').read_text())
        for trial in trials:
            trial.update(command=[.4,0,0],events=[dict(time=3,load=[-25,0,0])],initial_seed=trial['trial_id'],
                mechanics_seed=trial['trial_id'],scenario_hash=str(trial['trial_id']),load_range=meta['arguments']['load_range'],
                prescribed_steps=8,failure_time_seconds=5. if trial['physical_failure'] else None)
        (directory/'trials.json').write_text(json.dumps(trials))
        initial=[json.loads(line) for line in (directory/'initial_states.jsonl').read_text().splitlines()]
        for state in initial: state['initial_state_hash']=str(state['trial_id'])
        (directory/'initial_states.jsonl').write_text('\n'.join(json.dumps(state) for state in initial))
        manifest=json.loads((directory/'records.json').read_text())
        for chunk in manifest['chunks']:
            file=directory/chunk['file']
            with np.load(file) as saved: data=dict(saved)
            n=len(data['step']); factor=.5 if enabled else 1.
            data.update(prediction_next_velocity_decoded=np.full((n,8),factor),prediction_next_velocity_measured_loading=np.zeros((n,8)),
                prediction_base_velocity=np.full((n,3),factor),prediction_ee_position=np.tile([.5+factor,0,0],(n,1)),
                commands_t=np.tile([0,0,0,.5,0,0],(n,1)),ee_reference_t_world=np.tile([.5,0,1.],(n,1)),
                ee_position_t_world=np.tile([.5,0,1.],(n,1)))
            np.savez_compressed(file,**data)
        manifest['fields'].update({k:dict(shape=list(v.shape[1:]),dtype=str(v.dtype)) for k,v in data.items()})
        (directory/'records.json').write_text(json.dumps(manifest))


def test_end_to_end_paired_stored_metrics_and_resampling_cache(tmp_path,monkeypatch):
    matched_fixture(tmp_path/'raw'); cfg=config(tmp_path)
    cfg.update(methods={'PILOT':dict(task='b1z1_pact'),'A7':dict(task='b1z1_pact_ab10_actor')},
        evaluation_seeds=[0],dynamics=True,cache_enabled=True,cache_dir=str(tmp_path/'cache'),rmse_bootstrap_resamples=30)
    monkeypatch.setattr(a,'OfflineBard',lambda *args,**kwargs:pytest.fail('Stored successors need no BARD'))
    monkeypatch.setattr(a,'build_model',lambda *args,**kwargs:pytest.fail('Stored predictions need no checkpoint'))
    result=a.run_analysis(cfg)
    paired=result['paired_rollout']; assert paired['summary']
    known=[r for r in paired['summary'] if r['block']=='base_linear' and r['trial_pairs']]
    assert known and all(r['difference']==pytest.approx(-.5) for r in known)
    assert not any('correlation' in str(k) for k in result)
    metrics={r['metric'] for r in result['table_summary']}
    assert {'explicit_base_velocity_rmse','explicit_ee_position_rmse'}<=metrics
    tables=list(Path(result['output_dir']).glob('native_*_table.csv')); assert len(tables)==2
    for path in tables:
        assert 'explicit base velocity' in path.read_text() and 'explicit ee position' in path.read_text()
    cfg['rmse_bootstrap_resamples']=50
    warm=a.run_analysis(cfg); assert warm['timings']['completed_run_hits']==12
    assert result['table_summary']==warm['table_summary']
    assert all(r['bootstrap_resamples']==50 for r in warm['paired_rollout']['summary'])
    assert (Path(result['output_dir'])/'paired_rollout_matching.csv').is_file()


def test_notebook_paired_statistics_and_explicit_columns(tmp_path,monkeypatch):
    nbformat=pytest.importorskip('nbformat'); nbclient=pytest.importorskip('nbclient')
    matched_fixture(tmp_path/'raw'); cfg=config(tmp_path)
    cfg.update(methods={'PILOT':dict(task='b1z1_pact'),'A7':dict(task='b1z1_pact_ab10_actor')},
        evaluation_seeds=[0],dynamics=True,rmse_bootstrap_resamples=50)
    monkeypatch.setenv('EXP3_CONFIG',json.dumps(cfg))
    root=Path(__file__).resolve().parents[1]
    nb=nbformat.read(root/'notebooks/exp3_representation_analysis.ipynb',as_version=4)
    nb.cells=[cell for cell in nb.cells if not cell.id.startswith('exp3-action')]
    nb.cells.append(nbformat.v4.new_code_cell("assert len(comparison) > 0\nassert not any(k.startswith(('torch', 'isaaclab', 'genesis')) for k in sys.modules)\nassert {'explicit_base_velocity_rmse','explicit_ee_position_rmse'} <= {r['metric'] for r in result['table_summary']}"))
    nbclient.NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
    nbformat.write(nb,tmp_path/'executed_paired_rmse.ipynb')
