"""Publication views of existing FiLM window exports; no model or simulator imports."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from exp1_analysis import style_figure, write_json
from b1z1_paper_names import DISPLAY_NAMES

STAGE = ['checkpoint','task','profile','loading','stage','motion_phase','analysis_mode']
TRIAL = STAGE+['evaluation_seed','trial_id','run']


def paired_projection(pre, post, basis):
    """Identical centering/axes for both halves; never refit a facet's basis."""
    center, axes = np.asarray(basis['center']), np.asarray(basis['axes'])[:2]
    return (pre-center)@axes.T, (post-center)@axes.T


def matches(item, selection):
    return all(value is None or item.get(key)==value for key,value in selection.items())


def read_window(output, window):
    with np.load(output/f"window-{window['window_id']:05d}.npz", allow_pickle=False) as saved:
        data = dict(saved)
    t = data['time_seconds']
    if len(t)<3 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Invalid actor timestamps')
    dt = np.median(np.diff(t))
    if not np.allclose(np.diff(t),dt,rtol=1e-4,atol=1e-7):
        raise ValueError('Noncontiguous window timestamps')
    if not np.allclose(data['successor_time_seconds']-t,dt,rtol=1e-4,atol=1e-7):
        raise ValueError('Actor/successor timestamp mismatch')
    for name in ('pre_projection','post_projection','position_recorded','ee_cartesian_error'):
        x=data['trace_'+name]
        if len(x)!=len(t) or not np.isfinite(x).all():
            raise ValueError('Invalid aligned trace: '+name)
    return data


def joint_selection(window, cfg):
    names=window['units']['signal_components']['position_recorded']
    arm=cfg.get('oscillation_arm_joints')
    if arm is None:
        arm=window['units']['signal_components'].get('position_recorded_arm',[])[:2]
    leg=cfg.get('oscillation_leg_joints',[])
    if not arm or any(name not in names for name in [*arm,*leg]):
        raise ValueError('Select available named arm/leg joints in oscillation_arm_joints/oscillation_leg_joints')
    return names, list(arm), list(leg)


def rms(x):
    # Same constant detrending as the simplified recorded-data amplitude metrics.
    return float(np.sqrt(np.mean((x-x.mean(axis=0))**2)))


def window_statistics(window, data, cfg):
    names,arm,_=joint_selection(window,cfg)
    pcs=np.asarray(cfg.get('oscillation_pcs',[1,2]),dtype=int)-1
    if not len(pcs) or np.any(pcs<0) or np.max(pcs)>=data['trace_pre_projection'].shape[1]:
        raise ValueError('Requested PCs are absent from the saved fixed basis')
    row={key:window[key] for key in TRIAL}
    row.update(window_id=window['window_id'],arm_joints=json.dumps(arm),pcs=json.dumps((pcs+1).tolist()))
    for name in ('pre','post'):
        row[name+'_rms']=rms(data['trace_'+name+'_projection'][:,pcs])
    for name in ('position','torque'):
        field='trace_'+name+'_recorded'
        row[name+'_rms']=rms(data[field][:,[names.index(j) for j in arm]]) if field in data else np.nan
    for i,axis in enumerate('xyz'): row['ee_'+axis+'_rms']=rms(data['trace_ee_cartesian_error'][:,i:i+1])
    return row


def trial_statistics(rows):
    # Equal windows within each trial, not time samples as independent replicates.
    frame=pd.DataFrame(rows)
    values=['pre_rms','post_rms','position_rms','torque_rms',*[f'ee_{a}_rms' for a in 'xyz']]
    return frame.groupby(TRIAL+['arm_joints','pcs'],dropna=False,sort=True)[values].mean().reset_index()


def select_example(rows, trials, axis, override=None):
    candidates=pd.DataFrame(rows)
    for key,value in (override or {}).items():
        if key not in candidates: raise ValueError('Unknown example selector: '+key)
        candidates=candidates[candidates[key]==value]
    if candidates.empty: raise ValueError('Example override has no eligible window in the selected stage')
    metric='ee_'+axis+'_rms'
    # Pick a median-amplitude TRIAL first, then its most representative window.
    eligible=trials.merge(candidates[TRIAL].drop_duplicates(),on=TRIAL)
    eligible=eligible.assign(distance=abs(eligible[metric]-trials[metric].median()))
    chosen=eligible.sort_values(['distance','evaluation_seed','trial_id','run']).iloc[0]
    subset=candidates
    for key in TRIAL: subset=subset[subset[key]==chosen[key]]
    selected=subset.assign(distance=abs(subset[metric]-chosen[metric])).sort_values(['distance','window_id']).iloc[0]
    return int(selected.window_id)


def save_figure(fig, output, stem, cfg):
    style_figure(fig,cfg)
    fig.tight_layout(rect=(0,.035,1,.94),w_pad=2.0,h_pad=1.5)
    for extension in ('pdf','png'):
        fig.savefig(output/(stem+'.'+extension),dpi=220,bbox_inches='tight')


def oscillation_figure(window, data, trials, cfg, output, axis):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=cfg.get('oscillation_figsize',[17,11]))
    for col in (0,1):
        for row in range(1,4): axes[row,col].sharex(axes[0,col])
    names,arm,leg=joint_selection(window,cfg)
    pcs=[i-1 for i in cfg.get('oscillation_pcs',[1,2])]
    basis=cfg['feature_bases'][window['checkpoint']]
    variance=np.asarray(basis['explained_variance'])*100
    t=data['time_seconds']
    signals=[]
    for pc in pcs:
        for name,style in [('pre','--'),('post','-')]:
            signals.append((0,name+'_projection',pc,f'{name} PC{pc+1} ({variance[pc]:.1f}%)',f'C{pc}',style))
    for row,channel in [(1,'position'),(2,'torque')]:
        for index,joint in enumerate([*arm,*leg]):
            for suffix,style in [('recorded','-'),('unmodulated',':')]:
                key=channel+'_'+suffix
                if 'trace_'+key in data:
                    signals.append((row,key,names.index(joint),joint+(' (unmod.)' if suffix=='unmodulated' else ''),f'C{index}',style))
    for i,label in enumerate('xyz'):
        signals.append((3,'ee_cartesian_error',i,label,f'C{i}','-' if label==axis else '--'))
    peak=None
    f=data.get('frequency_ee_cartesian_error'); psd=data.get('psd_components_ee_cartesian_error')
    plan=window['spectral_plan']
    if f is not None and psd is not None:
        keep=(f>=plan['usable_low_hz'])&(f<=plan['usable_high_hz'])
        column=psd[keep,'xyz'.index(axis)]
        if len(column) and column.max()>1e-20: peak=float(f[keep][np.argmax(column)])
    frequency=None
    for row,key,j,label,color,style in signals:
        lw=1.6 if row==3 and label==axis else 1.
        axes[row,0].plot(t,data['trace_'+key][:,j],color=color,ls=style,lw=lw,label=label)
        f=data.get('frequency_'+key); psd=data.get('psd_components_'+key)
        if f is None or psd is None: continue
        if frequency is not None and not np.array_equal(f,frequency):
            raise ValueError('Signals have incompatible spectral frequency bins')
        frequency=f
        keep=(f>=plan['usable_low_hz'])&(f<=plan['usable_high_hz'])
        axes[row,1].plot(f[keep],psd[keep,j],color=color,ls=style,lw=lw,label=label)
    for row,(unit,power) in enumerate(zip(['Feature [a.u.]','Position offset [rad]','FF request [Nm]','EE error [m]'],
                                          ['Feature PSD [a.u.²/Hz]','Position PSD [rad²/Hz]','FF PSD [Nm²/Hz]','EE PSD [m²/Hz]'])):
        axes[row,0].set_ylabel(unit); axes[row,1].set_ylabel(power)
        if axes[row,0].lines: axes[row,0].legend(ncol=2,loc='best')
        if not axes[row,1].lines: axes[row,1].text(.5,.5,'Insufficient spectral window',ha='center',transform=axes[row,1].transAxes)
        if peak is not None: axes[row,1].axvline(peak,color='0.35',ls=':',lw=.8)
    axes[-1,0].set_xlabel('Actor time t [s]'); axes[-1,1].set_xlabel('Frequency [Hz]')
    pairs=[('post_rms','position_rms','Post-PC RMS [a.u.]','Position RMS [rad]'),
           ('post_rms','torque_rms','Post-PC RMS [a.u.]','FF RMS [Nm]'),
           ('position_rms','ee_'+axis+'_rms','Position RMS [rad]',f'EE {axis} RMS [m]'),
           ('torque_rms','ee_'+axis+'_rms','FF RMS [Nm]',f'EE {axis} RMS [m]')]
    correlations=[]
    for ax,(x,y,xlabel,ylabel) in zip(axes[:,2],pairs):
        finite=trials[np.isfinite(trials[x])&np.isfinite(trials[y])]
        rho=float(spearmanr(finite[x],finite[y]).statistic) if len(finite)>=3 and finite[x].nunique()>1 and finite[y].nunique()>1 else None
        for i,(seed,g) in enumerate(finite.groupby('evaluation_seed')):
            ax.scatter(g[x],g[y],s=24,marker=['o','s','^','D'][i%4],label=f'Seed {seed}',alpha=.8)
        ax.set(xlabel=xlabel,ylabel=ylabel)
        text=f"Spearman r={rho:.2f}" if rho is not None else 'Spearman r unavailable'
        ax.text(.03,.96,text+f'; n={len(finite)}',va='top',transform=ax.transAxes)
        correlations.append(dict(x=x,y=y,spearman=rho,trials=len(finite),seeds=finite.evaluation_seed.nunique()))
    if axes[0,2].collections: axes[0,2].legend(loc='lower right')
    for ax in axes.flat: ax.grid(alpha=.18)
    for col,title in enumerate(['A  Aligned traces','B  Frequency alignment'+(f' ({axis}: {peak:.2f} Hz)' if peak is not None else ''),'C  Across-trial amplitudes']):
        axes[0,col].set_title(title,pad=14)
    load=window.get('actual_load_mean_n')
    load_text=f' / mean |F|={load:.1f} N' if load is not None else ''
    fig.suptitle(f"{DISPLAY_NAMES.get(window['task'],window['task'])}: {window['profile']} / {window['loading']}{load_text} / stage {window['stage']} / {window['motion_phase']}\n"
                 f"Checkpoint {window['checkpoint'][:10]}; seed {window['evaluation_seed']}, trial {window['trial_id']}; arm: {', '.join(arm)}")
    fig.text(.5,.015,'Recorded actions are primary; dotted action traces, when present, are unmodulated algebraic outputs. '
             'C: selected arm joints, equal-window trial means; no post-failure samples.',ha='center')
    save_figure(fig,output,'feature-action-ee',cfg); plt.close(fig)
    return correlations


def paired_pca_figure(facets, basis, cfg, output, checkpoint):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    axis=cfg.get('paired_pca_error_axis','x')
    if axis not in 'xyz' or len(axis)!=1: raise ValueError('paired_pca_error_axis must be x, y or z')
    points=[point for _,pool in facets for point in pool]
    xy=np.concatenate([np.asarray([p['pre'] for p in points]),np.asarray([p['post'] for p in points])])
    limit=cfg.get('paired_pca_error_limit_m') or max(abs(p['error']['xyz'.index(axis)]) for p in points)
    norm=Normalize(-max(limit,1e-9),max(limit,1e-9)); cmap=plt.get_cmap('coolwarm')
    ncols=2 if len(facets)==4 else min(3,len(facets)); nrows=int(np.ceil(len(facets)/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(5*ncols,4*nrows+1.1),squeeze=False,sharex=True,sharey=True)
    lo,hi=xy.min(0),xy.max(0); pad=np.maximum((hi-lo)*.08,.01)
    variance=np.asarray(basis['explained_variance'][:2])*100
    for ax,(key,pool) in zip(axes.flat,facets):
        pre=np.asarray([p['pre'] for p in pool]); post=np.asarray([p['post'] for p in pool])
        colors=cmap(norm([p['error']['xyz'.index(axis)] for p in pool]))
        for p,q in zip(pre,post):
            ax.annotate('',q,p,arrowprops=dict(arrowstyle='->',color='0.5',alpha=.25,lw=.5))
        ax.scatter(pre[:,0],pre[:,1],facecolors='none',edgecolors=colors,s=25)
        ax.scatter(post[:,0],post[:,1],c=colors,s=20)
        ax.set(xlabel=f'PC1 ({variance[0]:.1f}%)',ylabel=f'PC2 ({variance[1]:.1f}%)',
               xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]))
        ax.set_title(f'{DISPLAY_NAMES.get(key[1],key[1])}: {key[2]} / {key[3]}\nstage {key[4]} / {key[5]}; {len(pool)} pairs')
        ax.tick_params(labelbottom=True,labelleft=True)
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(facets):]: ax.set_visible(False)
    fig.suptitle(f'Paired FiLM projections: {DISPLAY_NAMES.get(facets[0][0][1],facets[0][0][1])}, checkpoint {checkpoint[:10]}')
    fig.legend(handles=[Line2D([],[],marker='o',mfc='none',color='0.3',ls='',label='Pre-FiLM'),
                        Line2D([],[],marker='o',color='0.3',ls='',label='Post-FiLM')],loc='upper center',bbox_to_anchor=(.5,.945),ncol=2)
    fig.text(.5,.02,'Arrows connect pre/post features at the same actor timestep; colors are the same signed EE error for each pair.',ha='center')
    style_figure(fig,cfg); fig.tight_layout(rect=(0,.06,.91,.87))
    cax=fig.add_axes([.93,.22,.012,.5])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,label=f'Signed EE {axis} error [m] (yaw frame)',extend='both')
    style_figure(fig,cfg)
    stem='paired-pca-'+checkpoint[:12]
    for ext in ('pdf','png'): fig.savefig(output/(stem+'.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    return stem


def export_figures(windows, cfg, output, *, source=None, _prepared=None):
    """Stream compact NPZ windows; keep only trial statistics and bounded paired points."""
    output=Path(output); rows=[]; pools={}; diagnostics=[]
    source=Path(source) if source is not None else output
    output.mkdir(parents=True,exist_ok=True)
    budget=int(cfg.get('paired_pca_points_per_facet',30))
    if budget<1: raise ValueError('paired_pca_points_per_facet must be positive')
    selection=cfg.get('oscillation_selection',dict(profile='pushing',loading='loaded',motion_phase='moving'))
    # Axis changes do not change these statistics or paired samples. The all-axis
    # exporter shares this compact preparation, reading each NPZ only once.
    for w in ([] if _prepared else windows):
        if w['kind']!='sustained': continue
        try:
            data=read_window(source,w)
            if matches(w,selection) and w.get('load_stage',-1)>=0:
                rows.append(window_statistics(w,data,cfg))
            if not matches(w,cfg.get('paired_pca_selection',dict(loading='loaded'))): continue
            if data['trace_pre_projection'].shape[1]<2: raise ValueError('Two fixed PCs required for paired plot')
            key=tuple(w[k] for k in STAGE)
            seed=int.from_bytes(hashlib.sha256(f"{cfg.get('sampling_seed',0)}:{w['run']}:{w['trial_id']}:{w['actual_start']}".encode()).digest()[:8],'little')
            scores=np.random.default_rng(seed).random(len(data['time_seconds']))
            # Small deterministic reservoir per facet, never retain raw feature histories.
            pool=pools.setdefault(key,[])
            for i in np.argsort(scores)[:budget]:
                pool.append(dict(priority=float(scores[i]),pre=data['trace_pre_projection'][i,:2].tolist(),
                    post=data['trace_post_projection'][i,:2].tolist(),error=data['trace_ee_cartesian_error'][i].tolist(),
                    time_seconds=float(data['time_seconds'][i]),window_id=w['window_id'],trial_id=w['trial_id'],evaluation_seed=w['evaluation_seed']))
            pools[key]=sorted(pool,key=lambda p:(p['priority'],p['window_id'],p['time_seconds']))[:budget]
        except (ValueError,KeyError,OSError) as exc:
            diagnostics.append(dict(window=w['window_id'],reason=str(exc)))
    if _prepared is not None:
        if _prepared:
            rows,pools,diagnostics=_prepared['rows'],_prepared['pools'],list(_prepared['diagnostics'])
        else:
            _prepared.update(rows=rows,pools=pools,diagnostics=list(diagnostics))
    report=dict(diagnostics=diagnostics,figures=[],selection=None,
                configuration={k:v for k,v in cfg.items() if k.startswith(('oscillation_','paired_pca_')) or k.endswith('_font_size')},
                aggregation='Equal windows within trial; C correlations across trial means, evaluation seeds distinguished, not independent training checkpoints.')
    if cfg.get('max_windows'):
        diagnostics.append(dict(reason='Coverage limited to analyzed windows by max_windows',max_windows=cfg['max_windows']))
    if rows:
        # Never pool checkpoints, motion phases, selected joints, or action channels.
        eligible=[r for r in rows if matches(r,cfg.get('oscillation_example') or {})]
        if not eligible: raise ValueError('Example override has no eligible window in the selected stages')
        first=min(tuple(r[k] for k in STAGE) for r in eligible)
        chosen=[r for r in rows if tuple(r[k] for k in STAGE)==first]
        trials=trial_statistics(chosen)
        if trials[['arm_joints','pcs']].drop_duplicates().shape[0]!=1:
            raise ValueError('Selected stage mixes different joint/PC selections')
        axis=cfg.get('oscillation_ee_axis','x')
        if axis=='dominant': axis='xyz'[np.argmax([trials[f'ee_{a}_rms'].mean() for a in 'xyz'])]
        if axis not in ('x','y','z'): raise ValueError('oscillation_ee_axis must be x, y, z, or dominant')
        example=select_example(chosen,trials,axis,cfg.get('oscillation_example'))
        w=next(w for w in windows if w['window_id']==example)
        example_data=read_window(source,w)
        correlations=oscillation_figure(w,example_data,trials,cfg,output,axis)
        trials.to_csv(output/'oscillation_plotted_trials.csv',index=False)
        pd.DataFrame(chosen).to_csv(output/'oscillation_plotted_windows.csv',index=False)
        pd.DataFrame(correlations).to_csv(output/'oscillation_correlations.csv',index=False)
        report.update(selection=dict(window_id=example,ee_axis=axis,**{k:w[k] for k in STAGE},
                      evaluation_seed=w['evaluation_seed'],trial_id=w['trial_id'],criterion='Nearest median trial EE oscillatory RMS, then representative window',
                      run=w['run'],spectral_plan=w['spectral_plan'],actual_start=w['actual_start'],
                      plotted_actor_start=float(example_data['time_seconds'][0]),plotted_actor_end=float(example_data['time_seconds'][-1]),
                      actual_load_mean_n=w.get('actual_load_mean_n'),
                      override=cfg.get('oscillation_example'),pcs=cfg.get('oscillation_pcs',[1,2]),arm_joints=joint_selection(w,cfg)[1],leg_joints=joint_selection(w,cfg)[2]),
                      correlations=correlations)
        report['figures'].append('feature-action-ee')
        if len(chosen)<len(rows): diagnostics.append(dict(reason='Other stages not pooled; select them with oscillation_selection',windows=len(rows)-len(chosen)))
    else: diagnostics.append(dict(reason='No eligible sustained-load stage for the three-panel figure'))
    saved_points=[]
    for checkpoint in sorted({k[0] for k in pools}):
        facets=sorted((k,v) for k,v in pools.items() if k[0]==checkpoint)
        maximum=int(cfg.get('paired_pca_max_facets',6))
        if maximum<1: raise ValueError('paired_pca_max_facets must be positive')
        if len(facets)>maximum: diagnostics.append(dict(checkpoint=checkpoint,reason='PCA facet budget; narrow paired_pca_selection',omitted_facets=len(facets)-maximum))
        facets=facets[:maximum]
        basis=cfg.get('feature_bases',{}).get(checkpoint)
        if not basis: diagnostics.append(dict(checkpoint=checkpoint,reason='Missing saved fixed PCA basis')); continue
        report['figures'].append(paired_pca_figure(facets,basis,cfg,output,checkpoint))
        for key,pool in facets:
            saved_points.extend(dict(zip(STAGE,key),**point) for point in pool)
    write_json(output/'paired_pca_points.json',dict(points=saved_points,bases=cfg.get('feature_bases'),color_axis=cfg.get('paired_pca_error_axis','x')))
    write_json(output/'oscillation_figure_selection.json',report)
    return report


def export_all_axes(windows, cfg, output):
    """Separate axis outputs; fixed PCA pairs/basis, axis-specific median examples."""
    output=Path(output)
    axes=cfg.get('oscillation_axes',['x','y','z'])
    if not axes or len(set(axes))!=len(axes) or any(a not in ('x','y','z') for a in axes):
        raise ValueError('oscillation_axes must contain unique x/y/z axes')
    prepared={}; reports={}
    for axis in axes:
        reports[axis]=export_figures(windows,dict(cfg,oscillation_ee_axis=axis,paired_pca_error_axis=axis),
            output/('axis-'+axis),source=output,_prepared=prepared)
    write_json(output/'oscillation_axes.json',dict(axes=list(axes),
        outputs={axis:'axis-'+axis for axis in axes},
        convention='Shared PCA basis/pairs; example median and EE association/peak use each selected axis. Explicit example overrides remain fixed.'))
    return reports
