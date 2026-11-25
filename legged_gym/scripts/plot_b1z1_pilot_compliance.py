"""Regenerate RSS plots/tables from completed trials; no policy or simulator imports."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('rss_util',ROOT/'legged_gym/utils/pilot_compliance_eval.py')
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    args=p.parse_args();root=args.directory
    if not (root/'COMPLETE').exists(): raise ValueError('Collection is not complete')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42})
    meta=json.loads((root/'metadata.json').read_text());c=meta['evaluation_config'];dt=meta['control_dt']
    episodes=[e for e in json.loads((root/'episodes.json').read_text()) if e['condition'] in meta['conditions']]
    summary=u.aggregate(episodes)  # Calibration is exported separately, never pooled into the main eight.
    curves=[];representatives=[];steps=round(c['horizon_s']/dt);t=(np.arange(steps)+1)*dt
    for condition in meta['conditions']:
        by_seed=[]
        relevant=[e for e in episodes if e['condition']==condition]
        eligible=[e for e in relevant if e['signed_displacement_m'] is not None]
        if eligible:
            median=np.median([abs(e['signed_displacement_m']) for e in eligible])
            chosen=min(eligible,key=lambda e:(abs(abs(e['signed_displacement_m'])-median),e['seed'],e['trial_id']))
            representatives.append(dict(condition=condition,seed=chosen['seed'],trial_id=chosen['trial_id'],
                rule='Nearest median absolute measured loaded displacement; ties seed/trial ID; failures eligible when window exists'))
        for seed in meta['evaluation_seeds']:
            rows=[e for e in relevant if e['seed']==seed];sums=np.zeros((steps,4));counts=np.zeros((steps,4),int)
            for e in rows:
                with np.load(root/f'seed_{seed}'/condition/f'trial_{e["trial_id"]:04d}.npz') as d:
                    n=len(d['time']);valid=d['alive_for_metrics'].astype(bool)
                    axis=2 if condition in u.CONDITIONS[:4] or condition=='nominal' else 0
                    signals=np.column_stack((d['ee_task'][:,axis]-d['nominal_task'][:,axis],
                        d['reference_task'][:,axis]-d['nominal_task'][:,axis],d['base_measured'][:,0],d['base_reference'][:,0]))
                    baseline=valid&(d['time']>=c['force_on_s']-1)&(d['time']<=c['force_on_s'])
                    if baseline.any(): signals[:,:2]-=signals[baseline,:2].mean(0)
                    else: signals[:,:2]=np.nan
                    ok=valid[:,None]&np.isfinite(signals)
                    sums[:n]+=np.where(ok,signals,0);counts[:n]+=ok
            values=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
            by_seed.append(values)
            for j,name in enumerate(('ee_measured_m','ee_reference_m','base_measured_m_s','base_reference_m_s')):
                curves.extend(dict(condition=condition,seed=seed,time=float(t[i]),signal=name,
                    value=float(values[i,j]) if np.isfinite(values[i,j]) else None,n_trials=int(counts[i,j])) for i in range(steps))
        stack=np.stack(by_seed);finite=np.isfinite(stack);count=finite.sum(0)
        mean=np.divide(np.nansum(stack,axis=0),count,out=np.full((steps,4),np.nan),where=count>0)
        var=np.divide(np.nansum((stack-mean)**2,axis=0),count-1,out=np.full((steps,4),np.nan),where=count>1)
        if condition==meta['conditions'][0]: all_curves={}
        all_curves[condition]=(mean,np.sqrt(var),count)
    for filename,names,channels,ylabel in (
        ('signed_ee_response',u.CONDITIONS[:4],(0,1),'Baseline-Corrected EE Shift [m]'),
        ('base_recruitment_response',u.CONDITIONS[4:],(2,3),'Heading-Forward Velocity [m/s]')):
        fig,axes=plt.subplots(2,2,figsize=(8,5),sharex=True,sharey=True)
        for ax,name in zip(axes.flat,names):
            ax.set_title(name.replace('_',' ').title())
            if name not in all_curves: ax.text(.5,.5,'Unavailable',transform=ax.transAxes,ha='center');continue
            mean,sd,count=all_curves[name]
            for j,label,style in zip(channels,('Measured','Reference'),('-', '--')):
                ax.plot(t,mean[:,j],style,label=label,color='C0' if label=='Measured' else 'C1')
                ax.fill_between(t,mean[:,j]-sd[:,j],mean[:,j]+sd[:,j],alpha=.15)
            for event in (c['force_on_s'],c['force_on_s']+c['ramp_up_s'],c['force_off_s'],c['force_off_s']+c['ramp_down_s']):
                ax.axvline(event,color='gray',lw=.6,alpha=.6)
            ax.axhline(0,color='gray',lw=.5);ax.grid(alpha=.15)
        for ax in axes[-1]: ax.set_xlabel('Time [s]')
        for ax in axes[:,0]: ax.set_ylabel(ylabel)
        axes[0,0].legend();fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(root/f'{filename}.{ext}',dpi=200)
        plt.close(fig)
    with (root/'plot_curves.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=curves[0]);w.writeheader();w.writerows(curves)
    (root/'representative_trials.json').write_text(json.dumps(representatives,indent=2))
    rows=[]
    for group in summary['per_condition']:
        row=dict(condition=group['condition'],failure_rate=group['failure_rate']['mean'])
        for metric,stats in group['metrics'].items():
            row.update({metric:stats['mean'],metric+'_seed_sd':stats['sd'],metric+'_seeds':stats['n']})
        rows.append(row)
    with (root/'results_table.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(f'Exported two response plots, plot-ready seed curves/counts, representatives and results table to {root}')


if __name__=='__main__': main()
