"""Failure-aware Exp2 stage diagnostics and offline FiLM action amplification."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import signal, stats
import exp1_analysis as shared
import exp2_analysis as exp2
import exp3_film_oscillation as film
from exp3_oscillation_physical import comparable_config, REQUIRED
from exp3_representation import file_hash
from analysis_runtime import optional_thread_limits
from b1z1_paper_names import DISPLAY_NAMES, LATEX_METHOD_NAMES


def stages(meta):
    e=meta['evaluation']; end=meta['prescribed_steps']*meta['dt']
    cuts=sorted({0.,end,*[float(t) for t in [e['move_start_s'],e['move_start_s']+e['velocity_ramp_s'],
                  e['stop_s'],e['stop_s']+e['velocity_ramp_s'],*e['load_times_s']] if 0<t<end]})
    result=[]
    for i,(left,right) in enumerate(zip(cuts,cuts[1:])):
        mid=(left+right)/2
        motion=('starting' if e['move_start_s']<=mid<e['move_start_s']+e['velocity_ramp_s'] else
                'stopping' if e['stop_s']<=mid<e['stop_s']+e['velocity_ramp_s'] else
                'moving' if e['move_start_s']<=mid<e['stop_s'] else 'standing')
        load_index=int(np.searchsorted(e['load_times_s'],mid,side='right'))-1
        loads=e.get('loads_n',[])
        nominal=loads[load_index] if 0<=load_index<len(loads) and meta['arguments']['loading']=='loaded' else 0. if load_index<0 or meta['arguments']['loading']=='unloaded' else None
        phase='release' if load_index>=0 and nominal==0 else 'loaded' if nominal and nominal>0 else 'unloaded'
        result.append(dict(stage=i,start_s=left,end_s=right,motion=motion,phase=phase,scheduled_load_n=nominal))
    return result


def stage_mask(time, stage, horizon):
    return (time>=stage['start_s']) & ((time<stage['end_s']) if stage['end_s']<horizon else (time<=horizon))


def stage_statistics(d,meta):
    """Successor errors/references, attributed to the executed interval's stage.

    Denominators count prescribed interval STARTS: a 3 s stage has 3/dt steps,
    not an extra endpoint at release or one fewer step in initial standing.
    """
    e=meta['evaluation']; dt=meta['dt']; horizon=meta['prescribed_steps']*dt
    if np.any(np.diff(d['step'])<=0) or not np.allclose(d['time_seconds'],d['step']*dt,atol=1e-6) or not np.allclose(d['successor_time_seconds'],d['time_seconds']+dt,atol=1e-6):
        raise ValueError('Invalid/duplicate transition timestamps')
    valid=exp2.valid_rows(d)
    ee=film.inverse_rotate(d['ee_position_world']-d['ee_reference_world'],d['base_quaternion_xyzw'],True)
    v=d['base_velocity_body'][:,:2]-d['command_nominal'][:,:2]
    yaw=d['base_angular_velocity_body'][:,2]-d['command_nominal'][:,2]
    ee_valid=valid & np.isfinite(ee).all(1)
    base_valid=valid & np.isfinite(v).all(1) & np.isfinite(yaw)
    grid=np.arange(meta['prescribed_steps'])*dt; rows=[]
    for stage in stages(meta):
        selected=stage_mask(d['time_seconds'],stage,horizon)
        a=selected & ee_valid; b=selected & base_valid
        den=int(stage_mask(grid,stage,horizon).sum())
        ep=int((a & (np.linalg.norm(ee,axis=1)<=e['ee_tolerance_m'])).sum())
        bp=int((b & (np.linalg.norm(v,axis=1)<=e['velocity_tolerance_m_s']) & (abs(yaw)<=e['yaw_tolerance_rad_s'])).sum())
        row=dict(stage,ee_valid_steps=int(a.sum()),base_valid_steps=int(b.sum()),prescribed_steps=den,
                 ee_pass_steps=ep,base_pass_steps=bp,ee_adherence=ep/den if den else np.nan,
                 base_adherence=bp/den if den else np.nan,forward_error_m_s=float(v[b,0].mean()) if b.any() else np.nan,
                 forward_command_m_s=float(d['command_nominal'][b,0].mean()) if b.any() else np.nan)
        row.update({f'ee_{axis}_bias_m':float(ee[a,i].mean()) if a.any() else np.nan for i,axis in enumerate('xyz')})
        loads=d.get('ee_force_applied_world_interval_mean')
        use=valid & stage_mask(d['time_seconds'],stage,horizon)
        use=use & np.isfinite(loads).all(1) if loads is not None else np.zeros(len(valid),bool)
        row['actual_load_n']=float(np.linalg.norm(loads[use],axis=1).mean()) if use.any() else np.nan
        row['load_intervals']=int(use.sum()); rows.append(row)
    return rows


def amplitude(x,preprocessing='linear'):
    """Detrend each physical output joint/axis BEFORE taking its aggregate RMS."""
    x=np.asarray(x,float)
    if len(x)<3 or not np.isfinite(x).all(): return np.nan
    return float(np.sqrt(np.mean(signal.detrend(x,axis=0,type=preprocessing)**2)))


def ratio(modulated,unmodulated,epsilon):
    return modulated/unmodulated if np.isfinite(modulated) and np.isfinite(unmodulated) and unmodulated>epsilon else np.nan


def aggregate(frame,keys,metrics):
    """Input is one row/trial; trials -> equal seeds, retaining checkpoint identity."""
    if frame.empty: return pd.DataFrame(),pd.DataFrame()
    long=frame.melt(id_vars=keys+['evaluation_seed','trial_id','run'],value_vars=metrics,var_name='metric',value_name='_metric_value')
    seeds=long.groupby(keys+['evaluation_seed','metric'],dropna=False).agg(value=('_metric_value','mean'),trials=('_metric_value','count')).reset_index()
    summary=seeds.groupby(keys+['metric'],dropna=False).agg(mean=('value','mean'),sd=('value','std'),
        seeds=('value','count'),trials=('trials','sum')).reset_index()
    return seeds,summary


def action_reference(d,meta,cfg,head_cache,references):
    """Reuse exported references first; otherwise load only hash-matched linear heads."""
    scales=film.action_scales(meta); n=len(scales[0]); output={}; sources={}; errors={}
    for channel,scale,offset in [('position',scales[0],0),('torque',scales[1],n)]:
        if d['action_raw'].shape[1]<offset+n: continue
        saved=d['action_raw'][:,offset:offset+n]*scale
        pair=None
        for ref in references:
            if ref['checkpoint']!=meta['checkpoint_sha256'] or ref['evaluation_seed']!=meta['evaluation_seed'] or ref['trial_id']!=int(d['trial_id'][0]): continue
            if ref['profile']!=meta['arguments']['profile'] or ref['loading']!=meta['arguments']['loading']: continue
            with np.load(ref['file'],allow_pickle=False) as a:
                t=a['time_seconds']; ids=np.searchsorted(t,d['time_seconds'])
                if np.any(ids>=len(t)) or not np.allclose(t[ids],d['time_seconds'],atol=1e-6,rtol=0): continue
                keys=['trace_'+channel+'_actual','trace_'+channel+'_unmodulated']
                if all(k in a for k in keys): pair=(a[keys[0]][ids],a[keys[1]][ids]); sources[channel]='stored action reference'; break
        if pair is None:
            key=meta['checkpoint_sha256']
            if key not in head_cache: head_cache[key]=film.load_heads(meta,cfg)
            heads,reason=head_cache[key]
            if heads is None or channel not in heads: errors[channel]=reason or 'Action head unavailable'; continue
            weight,bias=heads[channel]
            pair=tuple((d[k]@weight.T+bias)*scale for k in ('film_features_post_t','film_features_pre_t'))
            sources[channel]='frozen action head'
        mod,unmod=pair
        errors[channel+'_max_error']=float(np.max(abs(mod-saved))) if mod.shape==saved.shape else np.nan
        # Permissive tolerance in NORMALIZED action units, scaled per joint so
        # radian and Nm channels receive the same check. Always report errors.
        atol=cfg.get('action_reference_atol',5e-3); rtol=cfg.get('action_reference_rtol',1e-2)
        matches=mod.shape==saved.shape and np.allclose(mod,saved,atol=atol*np.abs(scale),rtol=rtol)
        if not matches or unmod.shape!=saved.shape or not np.isfinite(mod).all() or not np.isfinite(unmod).all():
            errors[channel]='Action reconstruction mismatch/nonfinite'; continue
        output[channel]=(mod,unmod)
    return output,sources,errors


def representative(frame):
    """Median trial amplitude first, representative window second, deterministic ties."""
    eligible=frame[np.isfinite(frame.ee_z_rms_m)&np.isfinite(frame.ff_ratio)]
    if eligible.empty: return None
    trials=eligible.groupby(['evaluation_seed','trial_id','run']).ee_z_rms_m.mean().reset_index()
    trials['distance']=abs(trials.ee_z_rms_m-trials.ee_z_rms_m.median())
    trial=trials.sort_values(['distance','evaluation_seed','trial_id','run']).iloc[0]
    group=eligible[(eligible.evaluation_seed==trial.evaluation_seed)&(eligible.trial_id==trial.trial_id)&(eligible.run==trial.run)]
    return group.assign(distance=abs(group.ee_z_rms_m-trial.ee_z_rms_m)).sort_values(['distance','window_id']).iloc[0]


def run_analysis(config):
    with optional_thread_limits(config.get('cpu_blas_threads',1),user_api='blas'):
        return _run_analysis(config)


def _run_analysis(config):
    cfg=film.default_config(config); cfg.setdefault('comparison_window_seconds',2.); cfg=comparable_config(cfg)
    output=Path(cfg['output_dir']); output.mkdir(parents=True,exist_ok=True)
    (output/'COMPLETE').unlink(missing_ok=True)
    methods=cfg.get('diagnostic_tasks',['b1z1_pact','b1z1_unifp_original','b1z1_ppo_pos'])
    package=None
    if cfg.get('export_package',False):
        from exp2_film_package import PackageBuilder, FIELDS as package_fields
        package=PackageBuilder(cfg,output)
    stage_rows=[]; window_rows=[]; diagnostics=[]; provenance=[]; head_cache={}; references=[]; seen=set(); protocols={}
    for root in cfg.get('action_reference_roots',[]):
        p=Path(root)/'summary.json'
        if not p.exists() or not (p.parent/'COMPLETE').exists(): continue
        saved=shared.read_json(p)
        references.extend(dict(w,file=p.parent/f"window-{w['window_id']:05d}.npz") for w in saved.get('windows',[]))
        for key,value in saved.get('config',{}).get('feature_bases',{}).items():
            cfg.setdefault('feature_bases',{}).setdefault(key,value)
    for path in shared.discover(cfg['recording_roots'],'metadata.json'):
        if not (path.parent/'COMPLETE').exists(): continue
        m=shared.read_json(path); args=m.get('arguments',{})
        if m.get('experiment')!=2 or m['task'] not in methods or args.get('profile') not in cfg.get('diagnostic_profiles',['pushing','pulling','carrying']) or args.get('loading') not in cfg['loading'] or m['evaluation_seed'] not in cfg['evaluation_seeds']: continue
        policy=m['resolved_training']['policy']
        if package and m['task']=='b1z1_pact' and (policy.get('film_v02_enabled',False) or policy.get('conditioning_mode','film')!='film'):
            diagnostics.append(dict(run=str(path.parent),reason='Package selects original actor-feature FiLM only')); continue
        chosen=cfg.get('comparison_checkpoints',{}).get(m['task'])
        if chosen and m['checkpoint_sha256'] not in chosen: continue
        identity=dict(task=m['task'],checkpoint=m['checkpoint_sha256'],training_seed=m.get('checkpoint_training_seed'),
            evaluation_seed=m['evaluation_seed'],profile=args['profile'],loading=args['loading'],run=str(path.parent))
        key=(identity['task'],identity['checkpoint'],identity['evaluation_seed'],identity['profile'],identity['loading'])
        if key in seen: raise ValueError('Duplicate diagnostic run: '+str(key))
        seen.add(key)
        protocol_key=(identity['task'],identity['checkpoint'],identity['profile'],identity['loading'])
        protocol_hash=shared.fingerprint(exp2.protocol(m))
        if protocol_key in protocols and protocols[protocol_key]!=protocol_hash:
            raise ValueError('Incompatible protocols across evaluation seeds: '+str(protocol_key))
        protocols[protocol_key]=protocol_hash
        trials={int(t['trial_id']):t for t in shared.read_json(path.parent/'trials.json')}
        run=dict(path=path.parent,meta=m,records=shared.read_json(path.parent/'records.json'),trial_ids=set(trials))
        provenance.append(dict(identity,metadata_sha256=file_hash(path),protocol=exp2.protocol(m),stages=stages(m)))
        print('Task diagnostics:',path.parent,flush=True)
        try:
            remaining=set(cfg['trial_ids']) & set(trials) if cfg.get('trial_ids') is not None else None
            for trial,d in exp2.stream_trials(path.parent,m,trials):
                tid=int(trial['trial_id'])
                if cfg.get('trial_ids') is None or tid in cfg['trial_ids']:
                    stage_rows.extend(dict(identity,trial_id=tid,**r) for r in stage_statistics(d,m))
                if remaining is not None:
                    remaining.discard(tid)
                    if not remaining: break
        except (KeyError,ValueError,OSError) as exc:
            diagnostics.append(dict(run=str(path.parent),component='stage statistics',reason=str(exc)))
        is_film=m['task']=='b1z1_pact' and m['resolved_training']['policy'].get('conditioning_mode','film')=='film' and not m['resolved_training']['policy'].get('film_v02_enabled',False)
        fields=(*REQUIRED,'ee_force_applied_world_interval_mean','action_raw',*film.FILM[1:])
        if package:
            fields=tuple(dict.fromkeys((*fields,*package_fields)))
            package.register(run,is_film)
        local={**cfg,'_fields':fields,'_window_specs':[s for s in film.window_specs(m,cfg) if s['kind']=='sustained']}
        if not set(REQUIRED)<=set(run['records']['fields']):
            diagnostics.append(dict(run=str(path.parent),reason='Actor-time physical fields missing')); continue
        for tid,spec,raw in film.stream_windows(run,local):
            for segment in film.contiguous_segments(raw,m['dt'],REQUIRED):
                if len(segment)!=round(cfg['comparison_window_seconds']/m['dt']): continue
                d={k:np.stack([r[k] for r in segment]) for k in segment[0]}
                err=film.inverse_rotate(d['ee_position_t_world']-d['ee_reference_t_world'],d['base_quaternion_t'],True)
                force=d.get('ee_force_applied_world_interval_mean')
                magnitudes=np.linalg.norm(force,axis=1) if force is not None else np.full(len(segment),np.nan)
                load=float(magnitudes.mean()); stable=np.isfinite(load) and np.ptp(magnitudes)<=cfg.get('load_tolerance_n',1.)
                pair={}; sources={}; errors={}
                if is_film and all(k in d for k in ('film_features_pre_t','film_features_post_t','action_raw')):
                    try: pair,sources,errors=action_reference(d,m,cfg,head_cache,references)
                    except (ValueError,KeyError,OSError,RuntimeError) as exc: errors={'action_reference':str(exc)}
                indices=film.joint_groups(m,cfg)['arm']
                wid=len(window_rows)//2
                if package: package.accept(d,m,dict(identity,trial_id=tid,window_id=wid),spec)
                for prep in ('linear','constant'):
                    row=dict(identity,trial_id=tid,stage=spec['stage'],start_s=spec['start'],window_id=wid,
                        motion=spec['motion_phase'],preprocessing=prep,actual_load_n=load,stable_load=stable,
                        position_ratio=np.nan,ff_ratio=np.nan,position_mod_rms_rad=np.nan,position_unmod_rms_rad=np.nan,
                        ff_mod_rms_nm=np.nan,ff_unmod_rms_nm=np.nan)
                    row.update({f'ee_{a}_rms_m':amplitude(err[:,i:i+1],prep) for i,a in enumerate('xyz')})
                    for channel,(mod,unmod) in pair.items():
                        ids=[i for i in indices if i<mod.shape[1]]
                        if not ids: continue
                        a,b=amplitude(mod[:,ids],prep),amplitude(unmod[:,ids],prep)
                        prefix,unit=('position','rad') if channel=='position' else ('ff','nm')
                        row[prefix+'_mod_rms_'+unit]=a; row[prefix+'_unmod_rms_'+unit]=b
                        row[prefix+'_ratio']=ratio(a,b,cfg.get('ratio_epsilon',1e-6))
                        if not np.isfinite(row[prefix+'_ratio']): errors[prefix+'_ratio']='Near-zero/nonfinite denominator'
                    window_rows.append(row)
                if errors: diagnostics.append(dict(run=str(path.parent),window_id=wid,action_reference=errors,sources=sources))
                if is_film and args['profile']=='pushing' and stable and abs(load-55)<=cfg.get('load_tolerance_n',1.) and 'torque' in pair:
                    basis=cfg.get('feature_bases',{}).get(m['checkpoint_sha256'])
                    if basis is None:
                        basis=film.fit_feature_bases([run],cfg).get(m['checkpoint_sha256'])
                        cfg.setdefault('feature_bases',{})[m['checkpoint_sha256']]=basis
                    if basis:
                        center,pc=np.asarray(basis['center']),np.asarray(basis['axes'])[0]
                        np.savez_compressed(output/f'example-{wid:06d}.npz',time=d['time_seconds'],pre=(d['film_features_pre_t']-center)@pc,
                            post=(d['film_features_post_t']-center)@pc,ff_mod=pair['torque'][0],ff_unmod=pair['torque'][1],ee=err,
                            names=np.asarray(m['dof_names']),arm_indices=np.asarray(indices),explained=basis['explained_variance'][0])
    for task in methods:
        if not any(p['task']==task for p in provenance): diagnostics.append(dict(task=task,reason='No selected completed recordings'))
    result=finish(stage_rows,window_rows,provenance,diagnostics,cfg,output)
    if package:
        (output/'COMPLETE').unlink(missing_ok=True)
        result['package']=package.finish()
        shared.write_json(output/'summary.json',result)
        (output/'COMPLETE').write_text('complete\n')
    return result


def finish(stage_rows,window_rows,provenance,diagnostics,cfg,output):
    stages_frame=pd.DataFrame(stage_rows); windows=pd.DataFrame(window_rows)
    keys=['task','checkpoint','profile','loading','stage']
    if len(windows):
        metrics=['ee_x_rms_m','ee_y_rms_m','ee_z_rms_m','position_ratio','ff_ratio','position_mod_rms_rad','position_unmod_rms_rad','ff_mod_rms_nm','ff_unmod_rms_nm']
        trial=windows.groupby(keys+['evaluation_seed','trial_id','run','preprocessing'],dropna=False)[metrics+['actual_load_n']].mean().reset_index()
        counts=windows.groupby(keys+['evaluation_seed','trial_id','run','preprocessing']).size().rename('windows').reset_index()
        trial=trial.merge(counts)
        if len(stages_frame):
            rms=trial[trial.preprocessing=='linear'][keys+['evaluation_seed','trial_id','run','ee_x_rms_m','ee_y_rms_m','ee_z_rms_m','windows']]
            stages_frame=stages_frame.merge(rms,on=keys+['evaluation_seed','trial_id','run'],how='left')
    else: trial=pd.DataFrame()
    stage_metrics=['ee_x_bias_m','ee_y_bias_m','ee_z_bias_m','forward_error_m_s','forward_command_m_s','actual_load_n',
                   'ee_adherence','base_adherence','ee_x_rms_m','ee_y_rms_m','ee_z_rms_m']
    for name in stage_metrics:
        if name not in stages_frame: stages_frame[name]=np.nan
    seeds,summary=aggregate(stages_frame,keys+['start_s','end_s','motion','phase'],stage_metrics)
    for name,frame in [('stage_trials',stages_frame),('stage_seeds',seeds),('stage_summary',summary),('figure_windows',windows),('figure_trials',trial)]:
        frame.to_csv(output/(name+'.csv'),index=False)
    correlations=render(windows,trial,cfg,output) if len(windows) else []
    # Explicit moving/sustained 25/55 N comparison, never merge different loads or checkpoints.
    comparison=[]
    if len(windows):
        for load in (25.,55.):
            use=windows[(windows.preprocessing=='linear')&(windows.motion=='moving')&windows.stable_load & (abs(windows.actual_load_n-load)<=cfg.get('load_tolerance_n',1.))]
            group=['task','checkpoint','profile','loading','evaluation_seed','trial_id','run']
            if len(use):
                per_trial=use.groupby(group).ee_z_rms_m.mean().reset_index()
                _,table=aggregate(per_trial,['task','checkpoint','profile','loading'],['ee_z_rms_m'])
                comparison.extend(dict(r,load_n=load) for r in table.to_dict('records'))
    table=pd.DataFrame(comparison); table.to_csv(output/'vertical_rms.csv',index=False)
    if len(table):
        table['Method']=table.task.map(lambda t:DISPLAY_NAMES.get(t,t)); table['checkpoint']=table.checkpoint.str[:10]
        table['RMS [m]']=table.apply(lambda r:f'{r["mean"]:.4f} ± {r.sd:.4f}' if pd.notna(r.sd) else f'{r["mean"]:.4f} ± —',axis=1)
        pivot=table.pivot(index=['Method','checkpoint','loading'],columns=['profile','load_n'],values='RMS [m]').fillna('—')
        tex=shared.paper_latex(pivot,DISPLAY_NAMES,LATEX_METHOD_NAMES).replace('±',r'$\pm$').replace('—','---')
        (output/'vertical_rms.tex').write_text('\\begin{table*}\n\\centering\n\\caption{Moving-stage vertical EE RMS [m], linearly detrended fixed windows; equal evaluation-seed means and sample SD for fixed checkpoints.}\n'+tex+'\\end{table*}\n')
    notes=['Signed bias describes sustained offsets; detrended RMS describes oscillation, not total tracking error.',
           'Forward error is measured minus commanded BODY-frame velocity. Stage adherence includes zero credit after failure.',
           'Unmodulated actions are offline action-head evaluations, not executed trajectories. Evaluation seeds are repeats of fixed checkpoints.']
    if len(summary):
        for r in summary[(summary.profile=='pulling') & summary.metric.isin(['forward_error_m_s','ee_z_bias_m','ee_z_rms_m'])].to_dict('records'):
            notes.append(f"{r['task']} [{r['checkpoint'][:10]}] pulling {r['phase']}/{r['motion']} {r['start_s']:g}-{r['end_s']:g}s: {r['metric']}={r['mean']:.5g}, SD={r['sd']:.5g}, seeds={r['seeds']}, trials={r['trials']}.")
    (output/'numerical_summary.txt').write_text('\n'.join(notes))
    result=dict(configuration=cfg,provenance=provenance,diagnostics=diagnostics,stage_summary=summary.to_dict('records'),stage_seeds=seeds.to_dict('records'),
                vertical_rms=comparison,correlations=correlations,notes=notes,
                timing='Stage scores: successor error/reference attributed to executed interval-start stage; denominators count prescribed interval starts. Oscillation windows: actor time/reference, no event/reset/gap/failure crossing.',
                preprocessing='Componentwise linear detrending; mean-centered (constant) results separately exported.')
    shared.write_json(output/'summary.json',result); (output/'COMPLETE').write_text('complete\n')
    return result


def render(windows,trial,cfg,output):
    import matplotlib.pyplot as plt
    use=windows[(windows.task=='b1z1_pact')&(windows.profile=='pushing')&(windows.loading=='loaded')&
                (windows.motion=='moving')&windows.stable_load&(windows.preprocessing=='linear')]
    checkpoint=cfg.get('figure_checkpoint') or (sorted(use.checkpoint.unique())[0] if len(use) else None)
    use=use[use.checkpoint==checkpoint]; high=use[abs(use.actual_load_n-55)<=cfg.get('load_tolerance_n',1.)]
    example=representative(high); correlations=[]
    fig=plt.figure(figsize=cfg.get('task_figure_size',(13,6))); grid=fig.add_gridspec(3,3,width_ratios=[1.4,1,1.1]); aa=[fig.add_subplot(grid[i,0]) for i in range(3)]
    for ax in aa[1:]: ax.sharex(aa[0])
    b=fig.add_subplot(grid[:,1]); c=fig.add_subplot(grid[:,2])
    if example is not None and (output/f'example-{int(example.window_id):06d}.npz').exists():
        with np.load(output/f'example-{int(example.window_id):06d}.npz') as d:
            aa[0].plot(d['time'],d['pre'],'--',label='Pre PC1'); aa[0].plot(d['time'],d['post'],label='Post PC1')
            names=d['names'].tolist(); joints=cfg.get('figure_arm_joints') or [names[i] for i in d['arm_indices'][:2]]
            for j,name in enumerate(joints):
                i=names.index(name)
                if i not in d['arm_indices']: raise ValueError('Panel A requires an active arm joint: '+name)
                for key,ls,label in [('ff_mod','-','mod.'),('ff_unmod','--','unmod.')]:
                    aa[1].plot(d['time'],d[key][:,i],ls=ls,color=f'C{j}',label=f'{name} {label}')
            aa[2].plot(d['time'],d['ee'][:,2],color='C2'); aa[0].set_ylabel(f'PC1 ({100*d["explained"]:.1f}%)')
            shared.write_json(output/'figure_selection.json',dict(example.to_dict(),joints=joints,criterion='Median trial vertical linear-detrended RMS; representative contiguous window'))
    else: aa[0].text(.5,.5,'Verified action references unavailable',ha='center',wrap=True,transform=aa[0].transAxes)
    for ax,label in zip(aa,['Features [a.u.]','FF command [Nm]','Signed EE z error [m]']):
        ax.set_ylabel(ax.get_ylabel() or label)
        if ax.lines and ax!=aa[2]: ax.legend()
    aa[-1].set_xlabel('Actor time [s]')
    aa[0].set_title('A  55 N pushing example' if example is None else
                    f'A  55 N pushing; seed {example.evaluation_seed}, trial {example.trial_id}')
    plotted=[]
    for i,(load,channel) in enumerate([(25,'position_ratio'),(25,'ff_ratio'),(55,'position_ratio'),(55,'ff_ratio')]):
        part=use[abs(use.actual_load_n-load)<=cfg.get('load_tolerance_n',1.)]
        vals=part.groupby(['evaluation_seed','trial_id','run'])[channel].mean().dropna()
        for identity,value in vals.items():
            plotted.append(dict(checkpoint=checkpoint,load_n=load,channel=channel,preprocessing='linear',
                evaluation_seed=identity[0],trial_id=identity[1],run=identity[2],ratio=value,
                windows=int(part[(part.evaluation_seed==identity[0])&(part.trial_id==identity[1])&(part.run==identity[2])][channel].count())))
        if len(vals):
            b.boxplot([vals.to_numpy()],positions=[i],widths=.5,showfliers=False)
            b.scatter(i+np.linspace(-.12,.12,len(vals)),vals,s=10,alpha=.6)
    b.axhline(1,color='gray',ls='--'); b.set_xticks(range(4),['25 N\nposition','25 N\nFF','55 N\nposition','55 N\nFF'])
    b.set(ylabel='Modulated / unmodulated RMS',title='B  All-arm action amplification')
    # Both coordinates must come from the SAME eligible paired windows.
    means=high.dropna(subset=['ff_ratio','ee_z_rms_m']).groupby(['evaluation_seed','trial_id','run'])[['ff_ratio','ee_z_rms_m']].mean().reset_index()
    for seed,g in means.groupby('evaluation_seed'):
        rho=float(stats.spearmanr(g.ff_ratio,g.ee_z_rms_m).statistic) if len(g)>=3 and g.ff_ratio.nunique()>1 and g.ee_z_rms_m.nunique()>1 else np.nan
        correlations.append(dict(checkpoint=checkpoint,evaluation_seed=seed,spearman=rho,trials=len(g)))
        correlation_label=f'{rho:.2f}' if np.isfinite(rho) else 'unavailable'
        c.scatter(g.ff_ratio,g.ee_z_rms_m,label=f'Seed {seed}: r={correlation_label}, n={len(g)}')
    finite=[r['spearman'] for r in correlations if np.isfinite(r['spearman'])]
    c.set(xlabel='All-arm FF RMS ratio',ylabel='Vertical EE RMS [m]',title='C  55 N trial associations')
    if len(means): c.legend()
    c.text(.03,.98,f'Mean within-seed r: {np.mean(finite):.2f}' if finite else 'Correlation unavailable',va='top',transform=c.transAxes)
    pd.DataFrame(correlations).to_csv(output/'within_seed_correlations.csv',index=False)
    pd.DataFrame(plotted).to_csv(output/'panel_b_trials.csv',index=False)
    means.assign(checkpoint=checkpoint,load_n=55.,preprocessing='linear').to_csv(output/'panel_c_trials.csv',index=False)
    for ax in [*aa,b,c]: ax.grid(alpha=.15)
    fig.suptitle(f'PILOT pushing; checkpoint {str(checkpoint)[:10]}')
    fig.text(.5,.01,'B/C: componentwise linear detrending, fixed-duration windows averaged within trials. Unmodulated commands are offline, not executed.',ha='center')
    shared.style_figure(fig,cfg); fig.tight_layout(rect=(0,.05,1,.94))
    for ext in ('pdf','png'): fig.savefig(output/('exp2_film_oscillation.'+ext),dpi=250,bbox_inches='tight')
    plt.close(fig)
    return correlations
