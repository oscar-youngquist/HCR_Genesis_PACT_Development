"""Checkpoint cohorts, unchanged native metrics, and fixed-policy uncertainty."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'notebooks'))
import pilot_v02_comparison as c
import exp3_representation as rep
import exp3_rollout_statistics as stats
from test_exp1_analysis import make_fixture
from test_exp2_analysis import fixture as make_exp2
from test_exp3_representation import fixture as make_rep, config as rep_config
from test_exp3_rollout_statistics import pair_runs


def config(tmp):
    cfg=c.defaults(ROOT)
    cfg.update(recording_roots=[str(tmp/'raw')],output_dir=str(tmp/'out'),
        cache_dir=str(tmp/'cache'),rmse_bootstrap_resamples=50,progress=False,
        dynamics=False,fallback_predictions=False,device='cpu',metric_workers=1,num_workers=1)
    return cfg


def stamp(root,spec,key):
    for path in root.rglob('metadata.json'):
        m=json.loads(path.read_text())
        m.update(checkpoint=f'/logs/{spec["run_name"]}/model_1.pt',checkpoint_sha256=key)
        m.setdefault('resolved_training',{}).setdefault('policy',{})['film_v02_enabled']=spec.get('film_v02',False)
        path.write_text(json.dumps(m))


def fixture(tmp,experiment):
    cfg=config(tmp)
    for key,spec in cfg['methods'].items():
        root=tmp/'raw'/key
        if experiment==1: make_fixture(root,task=spec['task'])
        else: make_exp2(root,task=spec['task'])
        stamp(root,spec,key+'-hash')
    return cfg


def test_exact_checkpoint_cohorts_and_architecture(tmp_path):
    cfg=fixture(tmp_path,1)
    methods,runs,inventory=c.resolve_cohorts(cfg,1)
    assert len(methods)==4 and all(r['recordings']==9 for r in inventory)
    assert {m['checkpoint_sha256'] for m in methods.values()}=={k+'-hash' for k in methods}
    assert all('pilot_v02' in str(p) for p,_ in runs['pilot_v02'])
    cfg['methods']['pilot_v02']['film_v02']=False
    with pytest.raises(ValueError,match='architecture'): c.resolve_cohorts(cfg,1)


def test_seed_ci_preserves_units_pairing_and_missing_metrics(tmp_path):
    cfg=config(tmp_path); methods=cfg['methods']
    rows=[dict(method=k,evaluation_seed=s,metric='S_track',unit='fraction',value=.4+.1*s+(0 if k=='pilot_v02' else .2))
          for k in methods for s in range(3)]
    rows += [dict(method='pilot_v02',evaluation_seed=s,metric='grf',unit='N',value=2.) for s in range(3)]
    result=c.seed_confidence_intervals(rows,methods,cfg,tmp_path/'ci.csv')
    track=result[result.metric=='S_track']
    np.testing.assert_allclose(track.difference,-.2)
    np.testing.assert_allclose(track.ci_low,-.2)
    np.testing.assert_allclose(track.ci_high,-.2)
    assert (track.evaluation_seeds==3).all()
    assert result[result.metric=='grf'].difference.isna().all()
    again=c.seed_confidence_intervals(rows[::-1],methods,cfg,tmp_path/'again.csv')
    pd.testing.assert_frame_equal(result,again)
    one=c.seed_confidence_intervals(rows[:1],methods,cfg,tmp_path/'single.csv')
    assert one.ci_low.isna().all()
    with pytest.raises(ValueError,match='Duplicate'): c.seed_confidence_intervals(rows+[rows[0]],methods,cfg,tmp_path/'dup.csv')


@pytest.mark.parametrize('experiment',[1,2])
def test_native_summaries_aliases_cache_and_comparison(tmp_path,experiment,monkeypatch):
    cfg=fixture(tmp_path,experiment)
    summaries,inventory,ci=c.summarize_tasks(cfg,experiment)
    assert set(summaries)==set(cfg['methods'])
    assert summaries['pilot_v02']['native_task']==summaries['pilot_original']['native_task']=='b1z1_pact'
    assert summaries['pilot_v02']['task']!=summaries['pilot_original']['task']
    # Identical fixtures: every available difference is zero, including pooled limits.
    assert (ci.difference.dropna().abs()<1e-14).all()
    assert ci[ci.metric=='actuator_saturation_pct'].unit.eq('%').all()
    engine=c.exp1 if experiment==1 else c.exp2
    comp=c.comparison_config(cfg,summaries,experiment)
    tables,fig,diagnostics=engine.compare(comp)
    assert len(tables['main'])==4
    assert list(Path(comp['output_dir']).glob('*main*.tex'))
    import matplotlib.pyplot as plt
    plt.close('all')
    monkeypatch.setattr(engine,'summarize',lambda *a,**k:pytest.fail('Unchanged summaries recomputed'))
    cfg.update(rmse_bootstrap_resamples=25,title_font_size=14)
    assert set(c.summarize_tasks(cfg,experiment)[0])==set(summaries)


def test_method_id_pairs_keep_scenario_audit_and_common_time(tmp_path):
    runs,methods=pair_runs()
    runs[0]['name']='v02'; runs[1]['name']='old'
    methods={'v02':dict(task='b1z1_pact',display_name='PILOT V02'),
             'old':dict(task='b1z1_pact',display_name='PILOT Original')}
    plan=stats.plan_pairs(runs,methods,lambda r:{i:np.arange(10) for i in r['trials']},[['v02','old']])
    assert len(plan['audit'])==2
    assert all(r['matched'] and not r['representation_only_contrast'] for r in plan['audit'])
    assert plan['audit'][0]['pair']=='PILOT V02 vs PILOT Original'
    runs[1]['trials'][0]['scenario_hash']='changed'
    plan=stats.plan_pairs(runs,methods,lambda r:{i:np.arange(10) for i in r['trials']},[['v02','old']])
    assert not plan['audit'][0]['matched']
    result=stats.paired_results([],plan,dict(rmse_pairs=[['v02','old']],rmse_bootstrap_resamples=20))
    assert result['summary'] and all(r['attempted_pairs']==2 and r['trial_pairs']==0 for r in result['summary'])
    assert all(np.isnan(r['ci_low']) for r in result['summary'])
    with pytest.raises(ValueError,match='rmse_pairs'): stats.plan_pairs(runs,methods,lambda r:{},[['v02','missing']])


def test_representation_hash_selection_needs_no_checkpoint(tmp_path):
    for name in ('v02','old'):
        make_rep(tmp_path/'raw'/name,tasks=('b1z1_pact',),seeds=(0,))
        for path in (tmp_path/'raw'/name).rglob('metadata.json'):
            m=json.loads(path.read_text()); m['checkpoint_sha256']=name; path.write_text(json.dumps(m))
    cfg=rep_config(tmp_path)
    cfg['methods']={k:dict(task='b1z1_pact',checkpoint_sha256=k) for k in ('v02','old')}
    runs,_=rep.discover(cfg)
    assert len(runs)==12 and all(r['meta']['checkpoint_sha256']==r['name'] for r in runs)


@pytest.mark.parametrize('separate',[False,True])
def test_representation_table_uses_aggregation_identity(tmp_path,separate):
    cfg=config(tmp_path)
    cfg['separate_method_ids']=separate
    methods=cfg['methods']
    keys=list(dict.fromkeys(name if separate else spec['task'] for name,spec in methods.items()))
    values={key:float(i+1) for i,key in enumerate(keys)}
    summary=[dict(task=key,benchmark='native',variant='decoded',load_range='in',
        condition=condition,metric='ee_force_rmse',axis='all',mean=value,sd=.1)
        for key,value in values.items() for condition in ('nominal','disturbed')]
    result=dict(methods=methods,coverage=[dict(stratum=['in'])],comparative_summary=summary)
    rep.export_tables(result,cfg,tmp_path)
    table=pd.read_csv(tmp_path/'native_in_table.csv',header=[0,1],index_col=0)
    assert len(table)==(4 if separate else 3)
    for name,spec in methods.items():
        if not separate and name=='pilot_v02':
            assert spec['display_name'] not in table.index
            continue
        key=name if separate else spec['task']
        expected=c.exp1.formatted(dict(mean=values[key],sd=.1,unit='N'))
        assert table.loc[spec['display_name'],('nominal','ee force [N]')]==expected
        assert table.loc[spec['display_name'],('disturbed','ee force [N]')]==expected
    # Unsupported outputs stay unavailable, not zero or a copied metric.
    assert table.loc['UniFP',('nominal','per-foot normal GRF [N]')]=='\u2014'
    tex=(tmp_path/'native_in_table.tex').read_text()
    assert ('PILOT V02' in tex)==separate


def test_support_aliases_do_not_merge_pilot_versions(tmp_path):
    from test_exp3_support import fixture as support_fixture
    import exp3_support as support
    cfg=config(tmp_path)
    cfg['methods']={k:v for k,v in cfg['methods'].items() if k.startswith('pilot')}
    for k,spec in cfg['methods'].items():
        support_fixture(tmp_path/'raw'/k)
        stamp(tmp_path/'raw'/k,spec,k)
    settings,_=c.support_config(cfg)
    settings.update(support_plots=False,cache_enabled=False,evaluation_seeds=[0])
    result=support.analyze(settings)
    assert {r['task'] for r in result['summary']}==set(cfg['methods'])
    ci=c.support_confidence_intervals(result,cfg)
    assert ci.evaluation_seeds.eq(1).all() and ci.ci_low.isna().all()
    assert (ci.difference.dropna().abs()<1e-12).all()


@pytest.mark.parametrize('experiment',[1,2,3])
def test_new_notebook_cpu_smoke(tmp_path,experiment):
    nbformat=pytest.importorskip('nbformat'); nbclient=pytest.importorskip('nbclient')
    if experiment<3:
        cfg=fixture(tmp_path,experiment)
    else:
        cfg=config(tmp_path)
        cfg['methods']={k:v for k,v in cfg['methods'].items() if k.startswith('pilot')}
        from test_exp3_support import fixture as support_fixture
        for key,spec in cfg['methods'].items():
            make_rep(tmp_path/'raw'/key/'exp1',tasks=('b1z1_pact',),seeds=(0,))
            support_fixture(tmp_path/'raw'/key/'exp2')
            stamp(tmp_path/'raw'/key,spec,key)
        cfg.update(evaluation_seeds=[0],cache_enabled=False,support_plots=True,
                   support_legacy_figures=False,acceleration_metrics=False)
    conf=tmp_path/'config.json'; conf.write_text(json.dumps(cfg))
    notebook=nbformat.read(ROOT/f'notebooks/exp{experiment}_pilot_v02_comparison.ipynb',as_version=4)
    notebook.cells.insert(0,nbformat.v4.new_code_cell(f'import os; os.environ["PILOT_COMPARISON_CONFIG"]={str(conf)!r}'))
    nbclient.NotebookClient(notebook,timeout=180,kernel_name='python3',
        resources={'metadata':{'path':str(ROOT)}}).execute()
    outputs=list((tmp_path/'out').rglob('*confidence_intervals.csv'))
    assert len(outputs)>=(3 if experiment==3 else 1)
    assert all(p.is_file() for p in outputs)
    if experiment==3:
        tables=list((tmp_path/'out').rglob('native_*_table.csv'))
        assert tables
        for path in tables:
            table=pd.read_csv(path,header=[0,1],index_col=0)
            assert set(table.index)=={v['display_name'] for v in cfg['methods'].values()}
            assert table.ne('\u2014').any(axis=1).all()
