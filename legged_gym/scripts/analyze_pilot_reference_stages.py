"""CPU-only stage attribution from opt-in snapshots, never legacy aliased fields."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np

THRESHOLDS=(.10,.15,.20)
PREFIX='reference_stage_'


def measurements(d,axis,dt):
    if PREFIX+'raw_displacement' not in d: return None
    nominal=d[PREFIX+'nominal_position'];force=d[PREFIX+'force_task'][:,axis]
    coefficient=d[PREFIX+'selected_coefficients'][:,axis]
    valid=d['alive_for_metrics'].astype(bool)&np.isfinite(nominal).all(1)
    active=np.abs(force)>1e-6
    # Exact zero is disabled; even arbitrarily small positive coefficients yield.
    masks=dict(all=valid,yielding_active=valid&active&(coefficient>0),
        yielding_sustained=valid&active&(coefficient>0)&(d['phase']==2),
        protected_active=valid&active&(coefficient==0))
    offsets={'raw':d[PREFIX+'raw_displacement']}
    for stage,key in [('workspace','equilibrium_workspace'),('geometry','equilibrium_geometry'),
                      ('filtered','filtered_candidate'),('accepted','accepted_position')]:
        offsets[stage]=d[PREFIX+key]-nominal
    offsets['measured']=d['ee_task']-nominal
    offsets['tracking']=d['ee_task']-d[PREFIX+'accepted_position']
    values={f'{stage}_{kind}':v for stage,x in offsets.items() for kind,v in
        [('norm',np.linalg.norm(x,axis=1)),('axis_abs',np.abs(x[:,axis])),('axis_signed',x[:,axis])]}
    for name,a,b in [('workspace_change','workspace','raw'),('combined_geometry_change','geometry','workspace'),
                     ('filter_lag','filtered','geometry'),('final_guard_change','accepted','filtered')]:
        values[name]=np.linalg.norm(offsets[a]-offsets[b],axis=1)
    return values,masks,int(np.sum(valid&active&(coefficient>0)&(coefficient<=.05)))


def describe(x,dt):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    result=dict(samples=len(x),observed_seconds=len(x)*dt,
        mean=float(x.mean()) if len(x) else None,median=float(np.median(x)) if len(x) else None,
        p90=float(np.quantile(x,.9)) if len(x) else None,p95=float(np.quantile(x,.95)) if len(x) else None,
        maximum=float(x.max()) if len(x) else None,
        fraction_nonzero=float(np.mean(np.abs(x)>1e-7)) if len(x) else None)
    # 1 um threshold tolerance prevents exact 15 cm calibration requests being
    # called greater than 15 cm solely due to float32 representation.
    for threshold in THRESHOLDS:
        result[f'fraction_gt_{round(threshold*100)}cm']=float(np.mean(np.abs(x)>threshold+1e-6)) if len(x) else None
    return result


def write_csv(path,rows):
    if not rows: return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def run(root):
    meta=json.loads((root/'metadata.json').read_text());episodes=json.loads((root/'episodes.json').read_text())
    if not (root/'COMPLETE').exists(): raise ValueError('Collection not complete')
    dt=meta['control_dt'];seed_rows=[];exposure=[];missing=[];representatives=[]
    conditions=sorted({e['condition'] for e in episodes});writer=None
    with gzip.open(root/'reference_stage_trials.csv.gz','wt') as stream:
        for condition in conditions:
            for seed in sorted({e['seed'] for e in episodes}):
                group=[e for e in episodes if e['condition']==condition and e['seed']==seed]
                samples={};maxima={};means={};counts={};survival={};small=0
                for e in group:
                    source=root/f'seed_{seed}'/condition/f'trial_{e["trial_id"]:04d}.npz'
                    axis=int(np.argmax(np.abs(e['force_heading_direction'])))
                    with np.load(source) as d:
                        result=measurements(d,axis,dt)
                        if result is None:
                            missing.append(dict(condition=condition,seed=seed,trial_id=e['trial_id'],reason='Intermediate stages were not recorded'));continue
                        values,masks,small_count=result;small+=small_count
                        for mask_name,mask in masks.items():
                            for metric,value in values.items():
                                x=value[mask];key=(mask_name,metric)
                                stats=describe(x,dt)
                                row=dict(condition=condition,seed=seed,trial_id=e['trial_id'],pair_hash=e['pair_hash'],
                                    failed=e['failed'],mask=mask_name,metric=metric,units='m',**stats)
                                if writer is None: writer=csv.DictWriter(stream,fieldnames=row);writer.writeheader()
                                writer.writerow(row)
                                samples.setdefault(key,[]).append(x)
                                if len(x):
                                    maxima.setdefault(key,[]).append(stats['maximum'])
                                    means.setdefault(key,[]).append(stats['mean'])
                                counts[key]=counts.get(key,0)+bool(len(x))
                            for kind in ('norm','axis_abs'):
                                for threshold in THRESHOLDS:
                                    raw=values['raw_'+kind][mask]>threshold+1e-6
                                    accepted=values['accepted_'+kind][mask]>threshold+1e-6
                                    key=(mask_name,kind,threshold)
                                    s=survival.setdefault(key,np.zeros(4,dtype=np.int64))
                                    s+=np.array([len(raw),raw.sum(),accepted.sum(),(raw&accepted).sum()])
                        if condition.startswith('calibration') and masks['yielding_sustained'].any():
                            representatives.append((condition,seed,e['trial_id'],float(np.mean(values['accepted_norm'][masks['yielding_sustained']]))))
                for (mask,metric),chunks in samples.items():
                    x=np.concatenate(chunks);stats=describe(x,dt)
                    seed_rows.append(dict(condition=condition,seed=seed,mask=mask,metric=metric,units='m',
                        episodes=counts[(mask,metric)],attempted_episodes=len(group),condition_seed_small_nonzero_coefficient_samples=small,
                        mean_episode_maximum=float(np.mean(maxima[(mask,metric)])) if maxima.get((mask,metric)) else None,
                        equal_episode_mean=float(np.mean(means[(mask,metric)])) if means.get((mask,metric)) else None,**stats))
                for (mask,kind,threshold),s in survival.items():
                    exposure.append(dict(condition=condition,seed=seed,mask=mask,kind=kind,threshold_m=threshold,
                        samples=int(s[0]),raw_exceedances=int(s[1]),accepted_exceedances=int(s[2]),
                        surviving_exceedances=int(s[3]),raw_large_survival_fraction=float(s[3]/s[1]) if s[1] else None))
    # Exact sample distributions above; inference summaries retain trial -> seed
    # weighting. A maximum is a true maximum, never an average of episode maxima.
    aggregate=[]
    keys=sorted({(r['condition'],r['mask'],r['metric']) for r in seed_rows})
    for condition,mask,metric in keys:
        rows=[r for r in seed_rows if (r['condition'],r['mask'],r['metric'])==(condition,mask,metric)]
        available=[r for r in rows if r['episodes']]
        values=[r['equal_episode_mean'] for r in available]
        aggregate.append(dict(condition=condition,mask=mask,metric=metric,units='m',
            mean=float(np.mean(values)) if values else None,seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
            seeds=len(values),episodes=sum(r['episodes'] for r in rows),samples=sum(r['samples'] for r in rows),
            observed_seconds=sum(r['observed_seconds'] for r in rows),
            **{f'mean_seed_{k}':float(np.mean([r[k] for r in available])) if available else None
                for k in ('median','p90','p95','fraction_gt_10cm','fraction_gt_15cm','fraction_gt_20cm','fraction_nonzero')},
            true_maximum=max((r['maximum'] for r in available),default=None),
            mean_episode_maximum=float(np.mean([r['mean_episode_maximum'] for r in available])) if available else None))
    write_csv(root/'reference_stage_seeds.csv',seed_rows);write_csv(root/'reference_stage_summary.csv',aggregate)
    write_csv(root/'reference_exposure.csv',exposure)
    exposure_summary=[]
    for condition,mask,kind,threshold in sorted({(r['condition'],r['mask'],r['kind'],r['threshold_m']) for r in exposure}):
        rows=[r for r in exposure if (r['condition'],r['mask'],r['kind'],r['threshold_m'])==(condition,mask,kind,threshold)]
        present=[r for r in rows if r['samples']];raw=[r for r in rows if r['raw_exceedances']]
        exposure_summary.append(dict(condition=condition,mask=mask,kind=kind,threshold_m=threshold,
            samples=sum(r['samples'] for r in rows),raw_exceedances=sum(r['raw_exceedances'] for r in rows),
            accepted_exceedances=sum(r['accepted_exceedances'] for r in rows),
            surviving_exceedances=sum(r['surviving_exceedances'] for r in rows),
            raw_fraction=float(np.mean([r['raw_exceedances']/r['samples'] for r in present])) if present else None,
            accepted_fraction=float(np.mean([r['accepted_exceedances']/r['samples'] for r in present])) if present else None,
            raw_large_survival_fraction=float(np.mean([r['raw_large_survival_fraction'] for r in raw])) if raw else None,
            seeds=len(present),raw_large_seeds=len(raw)))
    write_csv(root/'reference_exposure_summary.csv',exposure_summary)
    selected=[]
    for condition in sorted({r[0] for r in representatives}):
        choices=[r for r in representatives if r[0]==condition];median=np.median([r[3] for r in choices])
        chosen=min(choices,key=lambda r:(abs(r[3]-median),r[1],r[2]))
        selected.append(dict(condition=condition,seed=chosen[1],trial_id=chosen[2]))
    (root/'reference_diagnostics.json').write_text(json.dumps(dict(schema_version=1,checkpoint_sha256=meta['checkpoint_sha256'],
        masks={'all':'finite pre-failure samples','yielding_active':'all AND abs(axis force)>1e-6 N AND sign-selected coefficient>0',
            'yielding_sustained':'yielding_active AND phase==sustained','protected_active':'all AND active force AND coefficient==0'},
        conventions={'displacements':'relative to matching interval accepted-reference nominal; not legacy offsets; task coordinates, meters',
            'statistics':'Per-seed sample quantiles; per-trial means -> equal-seed means; true maximum separate from mean episode maximum',
            'thresholds':'absolute magnitude strictly above threshold + 1e-6 m float tolerance',
            'geometry':'combined native target_safe predicate, no invented joint/collision attribution',
            'filter_lag':'vector difference, not necessarily attenuation; no zero-denominator stage ratios',
            'selection':'nearest median sustained accepted norm across calibration trials, ties seed/ID'},
        missing=missing,summary=aggregate,calibration_examples=selected),indent=2,allow_nan=False))
    plot(root,aggregate,selected,meta)
    print(f'Stage analysis: {len(seed_rows)} seed rows; {len(missing)} trials lack stage instrumentation')


def plot(root,summary,selected,meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9,'pdf.fonttype':42})
    names=sorted({r['condition'] for r in summary if r['mask']=='yielding_sustained' and r['episodes']})
    fig,ax=plt.subplots(figsize=(8,4));stages=['raw','workspace','geometry','accepted','measured']
    for name in names:
        values=[next(r['mean'] for r in summary if r['condition']==name and r['mask']=='yielding_sustained' and r['metric']==s+'_norm') for s in stages]
        ax.plot(stages,np.array(values)*100,'o-',label=name.replace('_',' '))
    ax.set_ylabel('Mean Offset Norm [cm]');ax.set_title('Sustained Yielding: Reference Stages and Measured Response')
    if names: ax.legend(fontsize=7,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.1))
    ax.grid(alpha=.2);fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(root/f'reference_stage_reduction.{ext}',dpi=200)
    plt.close(fig)
    if not selected:return
    fig,axes=plt.subplots(4,len(selected),figsize=(9,8),sharex=True,squeeze=False)
    for col,e in enumerate(selected):
        with np.load(root/f'seed_{e["seed"]}'/e['condition']/f'trial_{e["trial_id"]:04d}.npz') as d:
            t=d['time'];valid=d['alive_for_metrics'].astype(bool)
            axes[0,col].plot(d['interval_start'][valid],d[PREFIX+'force_task'][valid,2]);axes[0,col].set_title(f'{e["condition"]}: seed {e["seed"]}, trial {e["trial_id"]}')
            for field,label,style in [(PREFIX+'nominal_position','Nominal',':'),(PREFIX+'accepted_position','Reference','--'),('ee_task','Measured','-')]:
                axes[1,col].plot(t[valid],d[field][valid,2],style,label=label)
            for stage,key in [('Raw','raw_displacement'),('Workspace','equilibrium_workspace'),('Geometry','equilibrium_geometry'),('Accepted','accepted_position')]:
                x=d[PREFIX+key] if stage=='Raw' else d[PREFIX+key]-d[PREFIX+'nominal_position']
                axes[2,col].plot(t[valid],np.linalg.norm(x[valid],axis=1),label=stage)
            axes[3,col].plot(t[valid],d['gates'][valid,2],label='Recruitment gate')
            axes[3,col].plot(t[valid],d['clipped'][valid],label='Final clipping')
            axes[3,col].set_ylim(-.05,1.05)
        for row in range(4):
            axes[row,col].grid(alpha=.2)
            for event in ('force_on_s','force_off_s'):axes[row,col].axvline(meta['evaluation_config'][event],color='gray',lw=.6)
        for row in (1,2,3):axes[row,col].legend(fontsize=7)
        axes[3,col].set_xlabel('Time [s]')
    for row,label in enumerate(('Applied Fz [N]','World EE Height [m]','Offset Norm [m]','Gate / Clipping')):axes[row,0].set_ylabel(label)
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(root/f'calibration_traces.{ext}',dpi=200)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    run(parser.parse_args().directory)
