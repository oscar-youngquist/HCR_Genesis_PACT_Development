"""Rebuild offline FiLM package tables/figures using numeric exports ONLY.

This file is copied into the shareable archive and intentionally has no repository,
PyTorch, checkpoint, or simulator imports. --config overrides plotting settings.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GROUP=['task','checkpoint','profile','loading','stage','motion','scheduled_load_n']
IDENTITY=GROUP+['evaluation_seed','training_seed','trial_id','run','window_id','start_s','end_s']


def aggregate_windows(frame):
    metrics=[k for k in frame.select_dtypes(include=np.number) if k not in set(IDENTITY+['samples','spectra_available'])]
    keys=GROUP+['evaluation_seed','trial_id','run']
    numeric=pd.concat([frame[keys],frame[metrics].astype(float)],axis=1).copy()
    trial=numeric.groupby(keys,dropna=False)[metrics].mean().copy().reset_index()
    trial['windows']=frame.groupby(keys,dropna=False).size().to_numpy()
    trial['total_samples']=frame.groupby(keys,dropna=False)['samples'].sum().to_numpy() if 'samples' in frame else 0
    counts=numeric.groupby(keys,dropna=False)[metrics].count().copy().add_suffix('.window_count').reset_index()
    trial=trial.merge(counts,on=keys)
    long=trial.melt(id_vars=keys+['windows','total_samples'],value_vars=metrics,var_name='metric',value_name='value')
    valid_counts=counts.melt(id_vars=keys,var_name='metric',value_name='valid_windows')
    valid_counts['metric']=valid_counts.metric.str.removesuffix('.window_count')
    long=long.merge(valid_counts,on=keys+['metric'],validate='one_to_one')
    seed=long.groupby(GROUP+['evaluation_seed','metric'],dropna=False).agg(value=('value','mean'),trials=('value','count'),windows=('valid_windows','sum'),total_windows=('windows','sum'),total_samples=('total_samples','sum')).reset_index()
    final=seed.groupby(GROUP+['metric'],dropna=False).agg(mean=('value','mean'),sd=('value','std'),seeds=('value','count'),trials=('trials','sum'),windows=('windows','sum'),total_windows=('total_windows','sum'),total_samples=('total_samples','sum')).reset_index()
    return trial,seed,final


def configure(config):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    available={f.name for f in font_manager.fontManager.ttflist}
    font='Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    matplotlib.rcParams.update({'font.family':font,'font.size':config.get('font_size',10),
        'axes.titlesize':config.get('title_font_size',12),'axes.labelsize':config.get('axis_label_font_size',11),
        'xtick.labelsize':config.get('tick_font_size',9),'ytick.labelsize':config.get('tick_font_size',9),
        'legend.fontsize':config.get('legend_font_size',9),'pdf.fonttype':42,'ps.fonttype':42})
    return font


def save(fig,output,name):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    for ext in ('pdf','png'): fig.savefig(output/(name+'.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)


def rebuild(root,output=None,config=None):
    import matplotlib.pyplot as plt
    root=Path(root); output=Path(output or root/'rebuilt'); output.mkdir(parents=True,exist_ok=True)
    cfg=dict(config or {}); font=configure(cfg); m=json.loads((root/'manifest.json').read_text())
    figures=[]
    def emit(fig,name):
        save(fig,output,name); figures.extend([name+'.pdf',name+'.png'])
    names={**m.get('display_names',{}),**cfg.get('labels',{})}
    frame=pd.read_csv(root/'windows.csv.gz',float_precision='round_trip')
    if frame.window_id.duplicated().any(): raise ValueError('Duplicate window identifiers')
    trial,seed,summary=aggregate_windows(frame)
    original=pd.read_csv(root/'summary.csv.gz',float_precision='round_trip')
    # Validate keys/counts as well as values: matching numbers alone can hide misalignment.
    cols=GROUP+['metric','mean','sd','seeds','trials','windows','total_windows','total_samples']
    pd.testing.assert_frame_equal(summary[cols].reset_index(drop=True),original[cols].reset_index(drop=True),check_dtype=False,rtol=1e-10,atol=1e-12)
    for name,table in [('trials',trial),('seeds',seed),('summary',summary)]: table.to_csv(output/(name+'.csv'),index=False)
    metrics=['ee.x.bias_m','ee.y.bias_m','ee.z.bias_m','ee.x.rms_m','ee.y.rms_m','ee.z.rms_m','ee.mean_error_m','ee.adherence','base.forward_bias_m_s','base.adherence']
    compact=summary[summary.metric.isin(metrics)].copy()
    compact['method']=compact.task.map(lambda x:names.get(x,x))
    compact['mean_sd']=compact.apply(lambda r:'---' if pd.isna(r['mean']) else f'{r["mean"]:.4g} +/- {r.sd:.3g}' if pd.notna(r.sd) else f'{r["mean"]:.4g} +/- ---',axis=1)
    compact.to_csv(output/'tracking_table.csv',index=False)
    # Long form avoids a falsely shared stage label between different protocols.
    compact.pivot(index=['method','checkpoint','profile','loading','stage','motion','scheduled_load_n'],columns='metric',values='mean_sd').to_latex(output/'tracking_table.tex',escape=True)
    associations=pd.read_csv(root/'associations.csv.gz')
    associations.to_csv(output/'associations.csv',index=False)
    associations.to_latex(output/'associations.tex',index=False,escape=True)
    sensitivity=summary[summary.metric.str.startswith('sensitivity.')]
    sensitivity.to_csv(output/'sensitivity.csv',index=False)
    sensitivity.to_latex(output/'sensitivity.tex',index=False,escape=True)
    findings=['# Numerical diagnostics',
              'Bias is a sustained signed offset; drift is a fitted slope; residual RMS excludes both.',
              'Elevation transfer gain is a through-origin projection in matched normalized coordinates, not a claim of pure amplitude attenuation. Signed bias/sign agreement remain in the numeric tables.',
              'Values below describe repeated evaluations of fixed checkpoints; sample/seed counts are retained.']
    selected_metrics=['ee.z.bias_m','ee.z.rms_m','base.forward_bias_m_s','input.ee.axis1.gain','input.ee.axis1.bias']
    for _,r in summary[(summary.profile=='pulling')&summary.metric.isin(selected_metrics)].iterrows():
        value=f'{r["mean"]:.6g}' if pd.notna(r['mean']) else 'unavailable'
        sd=f'{r.sd:.6g}' if pd.notna(r.sd) else 'undefined (one/no seed)'
        findings.append(f'- {names.get(r.task,r.task)} {r.checkpoint[:10]}, {r.loading}, stage {r.stage} ({r.motion}, {r.scheduled_load_n:g} N): {r.metric} = {value}; SD {sd}; seeds {r.seeds}, trials {r.trials}, windows {r.windows}.')
    (output/'findings.md').write_text('\n'.join(findings)+'\n')
    colors=cfg.get('colors',{'b1z1_pact':'#1976b5','b1z1_unifp_original':'#d45b32','b1z1_ppo_pos':'#319254'})
    fig,axes=plt.subplots(1,3,figsize=cfg.get('summary_figsize',[12,3.6]))
    for ax,metric in zip(axes,['ee.z.bias_m','ee.z.rms_m','base.forward_bias_m_s']):
        part=summary[summary.metric==metric]
        for (task,checkpoint,profile),g in part.groupby(['task','checkpoint','profile']):
            g=g.sort_values('stage'); ax.errorbar(g.stage,g['mean'],yerr=g.sd.fillna(0),marker='.',color=colors.get(task),
                label=f'{names.get(task,task)} {profile} {checkpoint[:6]}')
        ax.set(xlabel='Metadata-derived stage index',ylabel=cfg.get('metric_labels',{}).get(metric,metric)); ax.grid(alpha=.2)
    if axes[0].lines: axes[0].legend()
    emit(fig,'tracking_summary')
    # All trials retained; axes correspond to matched old normalized inputs.
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    agreement=trial[trial.task==cfg.get('agreement_task','b1z1_pact')]
    if len(agreement):
        ckpt=cfg.get('agreement_checkpoint',sorted(agreement.checkpoint.unique())[0])
        agreement=agreement[agreement.checkpoint==ckpt]
        fig.suptitle(f'{names.get(cfg.get("agreement_task","b1z1_pact"))} inputs; checkpoint {ckpt[:10]}')
    for profile,g in agreement.groupby('profile'):
        if 'input.measured4.mean' in g:
            axes[0].scatter(g['input.measured4.mean'],g['input.estimated4.mean'],s=10,alpha=.65,label=profile)
        if 'input.ee.axis1.gain' in g:
            axes[1].scatter(g.stage,g['input.ee.axis1.gain'],s=10,alpha=.65,label=profile)
    axes[0].axline((0,0),slope=1,color='gray',ls='--')
    axes[0].set(xlabel='Measured elevation error [normalized]',ylabel='FiLM elevation error [normalized]')
    axes[1].axhline(1,color='gray',ls='--'); axes[1].set(xlabel='Stage index',ylabel='Elevation error gain (through origin)')
    if axes[0].collections: axes[0].legend()
    emit(fig,'conditioning_agreement')
    # Do not mix stages or average unlike output units in sensitivity heatmaps.
    chosen=summary[(summary.task=='b1z1_pact')&(summary.profile==cfg.get('sensitivity_profile','pulling'))&
                   (summary.motion==cfg.get('sensitivity_motion','moving'))&
                   np.isclose(summary.scheduled_load_n,cfg.get('sensitivity_load_n',55.))]
    if len(chosen):
        checkpoint=cfg.get('sensitivity_checkpoint',sorted(chosen.checkpoint.unique())[0])
        chosen=chosen[(chosen.checkpoint==checkpoint)&(chosen.stage==chosen.stage.min())]
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    if len(chosen):
        r=chosen.iloc[0]
        fig.suptitle(cfg.get('sensitivity_title',f'PILOT: {r.profile}, {r.scheduled_load_n:g} N, {r.motion}; {r.checkpoint[:8]}'),fontsize=cfg.get('title_font_size',12))
    for ax,channel,unit in zip(axes,['position','ff','total_nm'],['rad','Nm','Nm']):
        values=np.full((2,2),np.nan)
        for i,source in enumerate(('base','ee')):
            for j,group in enumerate(('arm','leg')):
                value=chosen[chosen.metric==f'sensitivity.{source}.jacobian.{group}.{channel}_rms']['mean']
                if len(value)==1: values[i,j]=value.iloc[0]
        im=ax.imshow(values,cmap=cfg.get('sensitivity_cmap','viridis'),aspect='auto')
        label=cfg.get('channel_labels',{}).get(channel,{'position':'Position','ff':'Feedforward','total_nm':'Combined request'}[channel])
        ax.set(xticks=[0,1],xticklabels=['Arm','Leg'],yticks=[0,1],yticklabels=['Base input','EE input'],title=f'{label} [{unit}/norm. input]')
        fig.colorbar(im,ax=ax)
        for i in range(2):
            for j in range(2): ax.text(j,i,f'{values[i,j]:.3g}' if np.isfinite(values[i,j]) else 'N/A',ha='center',va='center',fontsize=cfg.get('annotation_font_size',10),color='black' if np.isfinite(values[i,j]) and im.norm(values[i,j])>.5 else 'white')
    emit(fig,'conditioning_sensitivity')
    examples=m['examples']
    if cfg.get('window_ids') is not None: examples=[s for s in examples if s['window_id'] in cfg['window_ids']]
    for s in examples[:cfg.get('max_trace_figures',3)]:
        with np.load(root/s['file'],allow_pickle=False) as a:
            t=a['time_seconds']; fig,axes=plt.subplots(5,1,sharex=True,figsize=cfg.get('trace_figsize',[8,9]))
            if 'ee_force_applied_world_interval_mean' in a:
                for i,axis in enumerate('xyz'): axes[0].plot(t,a['ee_force_applied_world_interval_mean'][:,i],label=axis)
            for i,axis in enumerate('xyz'):
                color=cfg.get('ee_axis_colors',['#1976b5','#d45b32','#319254'])[i]
                axes[1].plot(t,a['ee_error'][:,i],label=axis,color=color); axes[1].plot(t,a['ee_trend'][:,i],ls='--',alpha=.5,color=color)
                axes[2].plot(t,a['ee_residual'][:,i],label=axis,color=color)
            if 'feature_pre_pc' in a:
                axes[3].plot(t,a['feature_pre_pc'][:,0],label='Pre PC1'); axes[3].plot(t,a['feature_post_pc'][:,0],label='Post PC1')
            run=next(r for r in m['records'] if r['run']==s['run'])
            joints=run['joint_names']; name=cfg.get('torque_joint') or next((j for j in joints if j.startswith('z1_') and 'Gripper' not in j),joints[0])
            for v in ('original','fixed','bypass'):
                key='torque_'+v+'_total'
                if key in a:
                    # Named configured arm joint, never selected for its correlation.
                    axes[4].plot(t,a[key][:,joints.index(name)],label=f'{v}: {name}')
            if 'executed_torque_interval_mean' in a:
                midpoint=(a['time_seconds']+a['successor_time_seconds'])/2
                axes[4].plot(midpoint,a['executed_torque_interval_mean'][:,joints.index(name)],':',color='black',label='Recorded interval execution')
            for ax,label in zip(axes,['EE force [N]','Signed EE error [m]','Detrended EE [m]','Feature PC1 [a.u.]','Torque [Nm]']):
                ax.set_ylabel(label); ax.grid(alpha=.15)
                if ax.lines: ax.legend(ncol=3)
                else: ax.text(.5,.5,'Not available/applicable',ha='center',transform=ax.transAxes)
            axes[-1].set_xlabel('Time [s]; executed interval means at midpoints')
            axes[0].set_title(f'{names.get(s["task"],s["task"])}: {s["profile"]}, {s["scheduled_load_n"]:g} N, {s["motion"]}; seed {s["evaluation_seed"]}, trial {s["trial_id"]}')
            emit(fig,f'trace-{int(s["window_id"]):06d}')
        prefix=f'w{int(s["window_id"]):06d}__'
        for part in m['spectra_parts']:
            with np.load(root/part,allow_pickle=False) as a:
                if prefix+'psd' not in a: continue
                fig,axes=plt.subplots(1,2,figsize=(9,3.5)); f=a[prefix+'frequency_hz']
                for j,name in enumerate(a[prefix+'signal_names']):
                    if name.startswith('ee_'): axes[0].semilogy(f,a[prefix+'psd'][:,j],label=name)
                    elif name.startswith('ff_arm'): axes[1].semilogy(f,a[prefix+'psd'][:,j],label=name)
                for ax,label in zip(axes,['EE PSD [m^2/Hz]','Named arm FF PSD [Nm^2/Hz]']):
                    ax.set(xlabel='Frequency [Hz]',ylabel=label); ax.grid(alpha=.2)
                    if ax.lines: ax.legend()
                emit(fig,f'spectrum-{int(s["window_id"]):06d}'); break
    report={'font':font,'windows':len(frame),'trials':len(frame[['run','trial_id']].drop_duplicates()),
            'trial_stage_rows':len(trial),'examples':len(m['examples']),
            'summary_matches':True,'configuration':cfg,'warning':'Evaluation-seed variability of fixed checkpoints, not training-seed variability.'}
    report['outputs']=figures+['trials.csv','seeds.csv','summary.csv','tracking_table.csv','tracking_table.tex',
                               'associations.csv','associations.tex','sensitivity.csv','sensitivity.tex','findings.md']
    (output/'regeneration.json').write_text(json.dumps(report,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',type=Path); parser.add_argument('--output',type=Path); parser.add_argument('--config',type=Path)
    args=parser.parse_args(); settings=json.loads(args.config.read_text()) if args.config else {}
    rebuild(args.package,args.output,settings)
