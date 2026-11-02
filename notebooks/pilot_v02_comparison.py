"""Four fixed-checkpoint comparisons using the existing offline metric engines.

Aliases distinguish two b1z1_pact checkpoints; recorded task IDs are never edited.
Broad CIs resample paired evaluation-seed summaries, not timesteps. The separate
rollout CI engine retains its verified trial pairing and common-time sensitivity.
"""
from pathlib import Path
import copy
import hashlib

import numpy as np
import pandas as pd

import exp1_analysis as exp1
import exp2_analysis as exp2
from exp3_rollout_statistics import options, paired_intervals


def defaults(root):
    root = Path(root)
    return dict(
        recording_roots=[str(root/'evaluation/exp1'), str(root/'evaluation/exp2'),
                         str(root/'evaluation/pilot_v02')],
        output_dir=str(root/'analysis/pilot_v02_comparison'),
        methods={
            'unifp': dict(task='b1z1_unifp_original', display_name='UniFP',
                          run_name='Oct23_15-25-20_upstream_architecture'),
            'pos': dict(task='b1z1_ppo_pos', display_name='Pos', run_name='Oct23_16-58-40_position_only'),
            'pilot_original': dict(task='b1z1_pact', display_name='PILOT (Original)',
                                   run_name='Oct21_00-46-21_b1z1_pact_improved', film_v02=False),
            'pilot_v02': dict(task='b1z1_pact', display_name='PILOT V02',
                              run_name='Oct29_15-14-37_b1z1_pact_improved', film_v02=True)},
        reference_method='pilot_v02', evaluation_seeds=[0,1,2],
        conditions=list(exp1.CONDITIONS), loading=['loaded','unloaded'], load_range='in',
        motion='combined', comparison_loading='loaded',
        ignore_evaluator_source_hash=True, reuse_summaries=True,
        histogram_edges=np.linspace(-.5,.5,101).tolist(), histogram_frame='yaw',
        include_combined=True, acceleration_metrics=True,
        rmse_bootstrap_resamples=10000, rmse_confidence_level=.95,
        rmse_resampling_seed=0, rmse_permutation_resamples=0,
        num_workers=4, metric_workers=2, cpu_blas_threads=1,
        inference_batch_size=1024, dynamics_batch_size=1024,
        cache_enabled=True, cache_completed_runs=True,
        cache_dir=str(root/'analysis/exp3_numeric_cache_02'),
        balanced_resampling=False, sampling_budget=None, samples_per_stratum=32,
        minimum_per_stratum=8, sampling_seed=0, progress=True,
        device='cuda:0', dynamics=True, fallback_predictions=True,
        contact_window_seconds=.1, representation_assume_foot_only_contacts=True,
        support_plots=True, support_legacy_figures=True,
        figure_title_font_size=12, title_font_size=11, axis_label_font_size=10,
        tick_font_size=9, legend_font_size=9, annotation_font_size=8,
        colors=['#D55E00','#009E73','#777777','#0072B2'])


def resolve_cohorts(config, experiment):
    """Discover by recorded task/run/hash, not by evaluation folder names."""
    cohorts={name:[] for name in config['methods']}
    for path in exp1.discover(config['recording_roots'],'metadata.json'):
        if not (path.parent/'COMPLETE').is_file(): continue
        meta=exp1.read_json(path)
        if meta.get('experiment',1)!=experiment or meta.get('evaluation_seed') not in config['evaluation_seeds']:
            continue
        args=meta['arguments']
        if experiment==1 and args['condition'] not in config['conditions']: continue
        if experiment==2 and args['loading'] not in config['loading']: continue
        matches=[]
        for name,spec in config['methods'].items():
            if spec['task']!=meta['task']: continue
            if spec.get('run_name') and spec['run_name']!=Path(meta['checkpoint']).parent.name: continue
            if spec.get('checkpoint_sha256') and spec['checkpoint_sha256']!=meta['checkpoint_sha256']: continue
            if 'film_v02' in spec:
                actual=meta.get('resolved_training',{}).get('policy',{}).get('film_v02_enabled',False)
                if actual!=spec['film_v02']: raise ValueError(f'{path}: unexpected FiLM architecture for {name}')
            matches.append(name)
        if len(matches)>1: raise ValueError(f'Ambiguous method mapping: {path}: {matches}')
        if matches: cohorts[matches[0]].append((path.parent,meta))
    resolved,inventory={},[]
    protocols=set()
    for name,items in cohorts.items():
        hashes={m['checkpoint_sha256'] for _,m in items}
        if len(hashes)>1: raise ValueError(f'{name}: select checkpoint_sha256; multiple checkpoints in selected run')
        spec=dict(config['methods'][name])
        if hashes: spec['checkpoint_sha256']=next(iter(hashes))
        resolved[name]=spec
        identities=set()
        for path,m in items:
            args=m['arguments']; group=(args['condition'],args['load_range']) if experiment==1 else (args['profile'],args['loading'])
            identity=(m['evaluation_seed'],*group)
            if identity in identities: raise ValueError(f'{name}: duplicate recording identity {identity}')
            identities.add(identity)
            p=exp1.protocol(m,config.get('ignore_evaluator_source_hash',False)) if experiment==1 else exp2.protocol(m)
            protocols.add(exp1.fingerprint(p))
        inventory.append(dict(method=name,task=spec['task'],checkpoint_sha256=spec.get('checkpoint_sha256'),
            recordings=len(items),evaluation_seeds=sorted({m['evaluation_seed'] for _,m in items}),
            training_seeds=sorted({m['checkpoint_training_seed'] for _,m in items if m.get('checkpoint_training_seed') is not None}),
            status='available' if items else 'no completed matching recordings'))
    if len(protocols)>1: raise ValueError(f'Experiment {experiment}: incompatible recorded protocols')
    return resolved,cohorts,inventory


def comparison_config(config, summaries, experiment):
    cfg=copy.deepcopy(config)
    cfg.update(summary_roots=[str(Path(config['output_dir'])/f'exp{experiment}/summaries'/name) for name in summaries],
        output_dir=str(Path(config['output_dir'])/f'exp{experiment}/comparison'),
        method_order=list(config['methods']), focused_methods=list(config['methods']),
        display_names={k:v['display_name'] for k,v in config['methods'].items()},
        latex_method_names={k:r'\textbf{'+v['display_name']+'}' for k,v in config['methods'].items()},
        motion_modes_table=True,motion_modes_condition='combined-disturbance',response_table=True,
        additional_plots=['step','ecdf','box'], optional_metrics=(list(exp2.SMOOTH) if config.get('acceleration_metrics') else []),
        loading=config.get('comparison_loading','loaded'),band='seed_sd')
    return cfg


def summarize_tasks(config, experiment):
    """Run native summarizers unchanged; only portable output identities are aliased."""
    from exp3_representation import recording_signature
    methods,cohorts,inventory=resolve_cohorts(config,experiment)
    engine=exp1 if experiment==1 else exp2
    output=Path(config['output_dir'])/f'exp{experiment}'
    output.mkdir(parents=True,exist_ok=True)
    exp1.write_json(output/'cohorts.json',inventory)
    summaries={}
    for name,items in cohorts.items():
        if not items: continue
        spec=methods[name]; dest=output/'summaries'/name
        settings=dict(config,task=spec['task'],display_name=spec['display_name'],
            checkpoint_sha256=spec['checkpoint_sha256'],input_roots=[str(p) for p,_ in items],
            expected_seeds=config['evaluation_seeds'],output_dir=str(dest))
        # Reuse numeric exports only when settings, code, schemas and chunk stamps match.
        numerical_keys=('task','display_name','checkpoint_sha256','input_roots','expected_seeds',
            'histogram_edges','histogram_frame','include_combined','acceleration_metrics',
            'ignore_evaluator_source_hash','loading')
        # Changing bootstrap counts or figure styling must not reread trajectories.
        signature=exp1.fingerprint(dict(config={k:settings.get(k) for k in numerical_keys},
            code=[hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (exp1,engine)],
            recordings=[recording_signature(dict(path=p,records=exp1.read_json(p/'records.json'))) for p,_ in items]))
        cached=exp1.read_json(dest/'summary.json') if (dest/'COMPLETE').is_file() and (dest/'summary.json').is_file() else {}
        if config.get('reuse_summaries',True) and cached.get('cohort_signature')==signature:
            summary=cached
        else:
            print(f'Exp{experiment}: {spec["display_name"]} ({len(items)} recordings)',flush=True)
            summary=engine.summarize(settings)
            (dest/'COMPLETE').unlink(missing_ok=True)
            summary.update(task=name,native_task=spec['task'],cohort_signature=signature)
            exp1.write_json(dest/'summary.json',summary)
            (dest/'COMPLETE').write_text('Checkpoint-specific portable summary complete\n')
        summaries[name]=summary
    if not summaries: raise ValueError('No matching completed recordings')
    seed_rows=[]
    for name,s in summaries.items():
        for r in s['per_seed']:
            labels={k:r[k] for k in (('condition','load_range','motion') if experiment==1 else ('profile','loading'))}
            for metric,value in r['values' if experiment==1 else 'metrics'].items():
                seed_rows.append(dict(method=name,evaluation_seed=r['seed'],stage='all',metric=metric,
                    value=value,unit=s['units'][metric],**labels))
            if experiment==2:
                for stage,metrics in r['stages'].items():
                    for metric,value in metrics.items():
                        seed_rows.append(dict(method=name,evaluation_seed=r['seed'],stage=stage,metric=metric,
                            value=value,unit=s['units'][metric],**labels))
    ci=seed_confidence_intervals(seed_rows,methods,config,output/f'exp{experiment}_metric_confidence_intervals.csv')
    return summaries,inventory,ci


def seed_confidence_intervals(rows,methods,config,path):
    """Paired seed-block percentile CIs; preserve each metric's upstream reduction.

    E.g. saturation remains sum(count)/sum(joint-substeps) within each seed.
    We never bootstrap timesteps or re-average per-trial physical-limit ratios.
    These pointwise CIs describe repeated evaluation, not independent training.
    """
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    settings=options(config); reference=config['reference_method']; results=[]
    frame=pd.DataFrame(rows)
    if not frame.empty:
        keys=[k for k in frame if k not in ('method','evaluation_seed','value')]
        if frame.duplicated(['method','evaluation_seed',*keys]).any(): raise ValueError('Duplicate seed metric')
        for key,g in frame.groupby(keys,dropna=False,sort=True):
            labels=dict(zip(keys,key if isinstance(key,tuple) else (key,)))
            left=g[g.method==reference].set_index('evaluation_seed').value
            for other in methods:
                if other==reference: continue
                right=g[g.method==other].set_index('evaluation_seed').value
                paired=pd.concat([left.rename('reference'),right.rename('comparator')],axis=1)
                paired=paired.replace([np.inf,-np.inf],np.nan).dropna().sort_index()
                result=dict(labels,reference_method=reference,comparator_method=other,
                    reference_checkpoint=methods[reference].get('checkpoint_sha256'),
                    comparator_checkpoint=methods[other].get('checkpoint_sha256'),
                    evaluation_seeds=len(paired),seed_ids=','.join(map(str,paired.index)),
                    reference_available_seeds=int(np.isfinite(left.to_numpy(float)).sum()),
                    comparator_available_seeds=int(np.isfinite(right.to_numpy(float)).sum()),
                    confidence_level=settings['confidence'],bootstrap_resamples=settings['resamples'],
                    resampling_unit='paired evaluation seed',
                    status='available' if len(paired)>1 else 'insufficient paired seeds' if len(paired) else 'metric unavailable in one or both methods')
                values=paired.to_numpy(float)
                if len(values):
                    stat=paired_intervals([values],dict(settings,permutations=0),[reference,other,labels])
                    result.update(reference_mean=stat['enabled_mean'],comparator_mean=stat['baseline_mean'],
                                  difference=stat['difference'],ci_low=stat['ci_low'],ci_high=stat['ci_high'])
                else: result.update({k:None for k in ('reference_mean','comparator_mean','difference','ci_low','ci_high')})
                results.append(result)
    result=pd.DataFrame(results)
    result.to_csv(path,index=False)
    exp1.write_json(path.with_suffix('.json'),dict(results=results,
        convention='Difference = PILOT V02 minus comparator, in the stated unit. Equal evaluation-seed means; paired seed-block percentile bootstrap. Pointwise exploratory intervals, no multiplicity correction. Three seeds give limited CI precision; fixed checkpoints, not independent training seeds. No same-state or trial pairing is implied.',
        configuration={k:config[k] for k in config if k.startswith('rmse_')}))
    return result


def representation_config(config):
    methods,cohorts,inventory=resolve_cohorts(config,1)
    # Missing cohorts keep a nonmatching hash, never accidentally select another version.
    for spec in methods.values(): spec.setdefault('checkpoint_sha256','unavailable')
    cfg=dict(config,methods=methods,tasks=None,separate_method_ids=True,
        recording_roots=[str(p) for items in cohorts.values() for p,_ in items],
        output_dir=str(Path(config['output_dir'])/'exp3'),
        latex_method_names={k:r'\textbf{'+v['display_name']+'}' for k,v in methods.items()},
        rmse_pairs=[[config['reference_method'],k] for k in methods if k!=config['reference_method']])
    return cfg,inventory


def representation_confidence_intervals(result,config):
    rows=[]
    for scope,records in (('primary',result['table_per_seed']),('breakdown',result['per_seed'])):
        for r in records:
            labels={k:r[k] for k in ('method','evaluation_seed','variant','load_range','condition','motion','phase','load_timing','metric','axis')}
            for statistic in ('mean','bias'):
                rows.append(dict(labels,scope=scope,statistic=statistic,value=r.get(statistic),unit=result['units'].get(r['metric'],'unavailable')))
    path=Path(result['output_dir'])
    frame=seed_confidence_intervals(rows,result['methods'],config,path/'representation_metric_confidence_intervals.csv')
    # Dedicated trial-bootstrap export retains exact shared pre-failure support.
    pd.DataFrame(result['paired_rollout']['summary']).to_csv(path/'representation_rollout_confidence_intervals.csv',index=False)
    return frame


def support_config(config):
    methods,cohorts,inventory=resolve_cohorts(config,2)
    for spec in methods.values(): spec.setdefault('checkpoint_sha256','unavailable')
    cfg=dict(config,support_methods=methods,
        exp2_roots=[str(p) for items in cohorts.values() for p,_ in items], exp2_loading=config['loading'],
        output_dir=str(Path(config['output_dir'])/'exp3'),support_plot_methods=list(methods),support_pairs=[],
        display_names={k:v['display_name'] for k,v in methods.items()},
        support_colors=dict(zip(methods,config['colors'])))
    return cfg,inventory


def support_confidence_intervals(result,config):
    path=Path(result['configuration']['output_dir'])/'action_support'
    source=path/'per_seed_stages.csv'
    rows=[]
    if result['summary'] and source.exists():
        for r in pd.read_csv(source).to_dict('records'):
            m=r['metric']
            unit=('Nm^2' if m.endswith('_ms') else 'Nm' if m.endswith('_nm') else
                  'N' if m.endswith('_n') else 'm/s' if m.endswith('_m_s') else 'm' if m.endswith('_m') else 'unknown')
            rows.append(dict(method=r['task'],evaluation_seed=r['seed'],profile=r['profile'],loading=r['loading'],
                             stage=r['stage'],metric=m,unit=unit,value=r['value']))
    return seed_confidence_intervals(rows,result['configuration']['support_methods'],config,
        path/'support_metric_confidence_intervals.csv')
