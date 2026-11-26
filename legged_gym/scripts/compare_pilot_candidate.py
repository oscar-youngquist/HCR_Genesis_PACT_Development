"""Paired setting/training contrasts; trials -> evaluation seeds, no training-seed claim."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def trials(root):
    df=pd.read_csv(root/'episodes.csv')
    extra=root/'candidate_trials.csv'
    if extra.exists():
        d=pd.read_csv(extra)
        return df.merge(d[['condition','seed','trial_id','drift_corrected_vx','unloaded_swing_fraction']],on=['condition','seed','trial_id'],validate='one_to_one')
    c=json.loads((root/'metadata.json').read_text())['evaluation_config']
    values=[]
    for row in df.itertuples():
        with np.load(root/f'seed_{row.seed}'/row.condition/f'trial_{row.trial_id:04d}.npz') as a:
            t=a['time'];alive=a['alive_for_metrics'].astype(bool);v=a['base_measured'][:,0]
            pre=alive&(t>=c['force_on_s']-1)&(t<=c['force_on_s'])
            hold=alive&(t>=c['force_on_s']+c['ramp_up_s']+1)&(t<=c['force_off_s'])
            values.append(float(v[hold].mean()-v[pre].mean()) if pre.any() and hold.any() else np.nan)
    df['drift_corrected_vx']=values;df['unloaded_swing_fraction']=np.nan
    return df


def compare(baseline,start,output,final=None):
    output.mkdir(parents=True,exist_ok=True)
    original_meta=json.loads((baseline/'metadata.json').read_text())
    start_meta=json.loads((start/'metadata.json').read_text())
    for key in ('force_on_s','force_off_s','ramp_up_s','ramp_down_s','trajectory_s','horizon_s',
                'yielding_coefficient','base_coefficient','force_n','amplitude_fraction',
                'vertical_sphere','ample_sphere','restricted_sphere'):
        if original_meta['evaluation_config'][key]!=start_meta['evaluation_config'][key]:
            raise ValueError('Unmatched evaluation protocol: '+key)
    sources={'original_settings':trials(baseline),'candidate_start':trials(start)}
    sources['original_settings']=sources['original_settings'].merge(
        sources['candidate_start'][['condition','seed','trial_id']],on=['condition','seed','trial_id'],validate='one_to_one')
    if final is not None:sources['candidate_final']=trials(final)
    metrics=['ee_tracking_error_m','signed_displacement_m','reference_signed_displacement_m',
             'base_reference_velocity_m_s','drift_corrected_vx','recovery_error_m','failed','unloaded_swing_fraction']
    allrows=pd.concat([v.assign(stage=k) for k,v in sources.items()],ignore_index=True)
    allrows.to_csv(output/'all_trial_metrics.csv.gz',index=False)
    perseed=allrows.groupby(['stage','condition','seed'])[metrics].mean()
    perseed.to_csv(output/'per_seed.csv')
    perseed.groupby(['stage','condition']).agg(['mean','std','count']).to_csv(output/'summary.csv')
    contrasts=[];paired=[]
    for before,after in [('original_settings','candidate_start'),('candidate_start','candidate_final')]:
        if after not in sources:continue
        pair=sources[before].merge(sources[after],on=['condition','seed','trial_id'],suffixes=('_before','_after'),validate='one_to_one')
        if not (pair.pair_hash_before==pair.pair_hash_after).all():raise ValueError('Mismatched force/initialization manifests')
        # The pair hash contains initialization and force schedule, not goals.
        for k in ('nominal_sphere','profile'):
            if k+'_before' in pair and not (pair[k+'_before']==pair[k+'_after']).all():raise ValueError('Mismatched '+k)
        for condition,group in pair.groupby('condition'):
            for metric in metrics:
                d=group[metric+'_after'].astype(float)-group[metric+'_before'].astype(float)
                valid=d.notna();seeds=[];rng=np.random.default_rng(9173)
                for seed,g in group[valid].groupby('seed'):
                    v=d.loc[g.index].to_numpy();seeds.append(v)
                    for index,value in zip(g.index,v):
                        paired.append(dict(contrast=after+' minus '+before,condition=condition,metric=metric,
                            seed=int(seed),trial_id=int(group.loc[index,'trial_id']),difference=float(value)))
                if not seeds:continue
                bootstrap=np.mean([rng.choice(v,(2000,len(v))).mean(1) for v in seeds],axis=0)
                contrasts.append(dict(contrast=after+' minus '+before,condition=condition,metric=metric,
                    mean_difference=float(np.mean([v.mean() for v in seeds])),
                    ci_low=float(np.quantile(bootstrap,.025)),ci_high=float(np.quantile(bootstrap,.975)),
                    trial_pairs=sum(map(len,seeds)),evaluation_seeds=len(seeds)))
    pd.DataFrame(contrasts).to_csv(output/'paired_contrasts.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired_trial_differences.csv.gz',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9})
    conditions=['up_yield_up_force','up_yield_down_force','restricted_base_on','restricted_base_off']
    labels=['Upward\nyielding','Downward\nprotected','Restricted\nbase on','Restricted\nbase off']
    fig,axes=plt.subplots(1,3,figsize=(10,3.2))
    for ax,metric,label in zip(axes,['ee_tracking_error_m','signed_displacement_m','drift_corrected_vx'],
            ['Mean EE Tracking Error [m]','Signed EE Displacement [m]','Drift-Corrected Base vx [m/s]']):
        for i,stage in enumerate(sources):
            g=perseed.loc[stage][metric].groupby('condition').agg(['mean','std']).reindex(conditions)
            x=np.arange(len(conditions))+(i-(len(sources)-1)/2)*.23
            ax.bar(x,g['mean'],.22,yerr=g['std'],capsize=2,label=stage.replace('_',' '))
        ax.set_xticks(range(4),labels);ax.set_ylabel(label);ax.axhline(0,color='.5',lw=.5)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8);fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(output/f'comparison.{ext}',dpi=200)
    plt.close(fig)
    (output/'comparison_metadata.json').write_text(json.dumps(dict(
        baseline=str(baseline),start=str(start),final=str(final) if final else None,
        aggregation='trial means within each evaluation seed, then equal seeds; SD ddof=1',
        confidence_intervals='2000 paired trial bootstrap resamples within evaluation seed, then equal seed means; conditional on one checkpoint',
        training_comparison_available=final is not None,manifest_matching=True,
        caveat='Settings-only differences are not learning improvements. No independently trained policy replicates.'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--start',type=Path,required=True)
    p.add_argument('--final',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.baseline,a.start,a.output,a.final)
