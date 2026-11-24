#!/usr/bin/env python3
"""Offline descriptive comparisons; no simulator, training or significance tests."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.eval_hard_pact_tradeoff import VARIANTS, report, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    root=p.parse_args().output
    results=[json.loads(f.read_text()) for f in sorted(root.glob('*/result.json'))]
    if results and all('probe' in r for r in results):
        from scripts.hard_pact_recovery_probe import report_probe
        return report_probe(root,results)
    for r in results:
        a=np.load(root/f"{r['condition']}_seed{r['seed']}"/'trace.npz')
        failure=a['control_reset']&a['control_failure']
        terrain=[]
        for level in np.unique(a['control_terrain_level']):
            for kind in np.unique(a['control_terrain_type']):
                mask=(a['control_terrain_level']==level)&(a['control_terrain_type']==kind)
                if not mask.any():continue
                exposure=float(mask.sum()*.02);events=failure&mask
                terrain.append(dict(level=int(level),terrain_column=int(kind),env_seconds=exposure,
                    failures=int(events.sum()),failure_rate=float(events.sum()/exposure),
                    overlapping_reasons={name:int((events&a['control_reasons'][...,i]).sum())
                        for i,name in enumerate(('contact','roll','pitch','height_low','height_high'))}))
        r['terrain_failure_exposure']=terrain
        r['base_height_m']=report.finite_stats(a['control_base_height'])
        r['base_tilt_abs_rad']=report.finite_stats(abs(a['control_tilt']))
        r['diagnostic_conventions']={
            'wrench_error':'QP-clamped control-rate wrench reference minus applied base-origin world wrench, not raw decoder error',
            'height_estimate_error':'Control-rate terrain/base-height truth (cached within four substeps)',
            'stance_slip':'Control-rate cached world foot speed times physics-rate contact; not an instantaneous substep velocity estimate',
            'objective_costs':'Accepted primary/recovery together; nominal may be infeasible; recovery slack and numerical SPD regularization excluded from named task costs',
            'model_motion':'Solved accepted rows only; whole reset intervals excluded; evolving body-frame differences match canonical body velocity derivatives',
            'correction_quantiles':'Pooled absolute joint-coordinate values in Nm; RMS is across joint coordinates',
            'reason_counts':'Overlapping threshold conditions at failure, not mutually exclusive causal attribution'}
    paired=[]
    for seed in sorted({r['seed'] for r in results}):
        base=next((r for r in results if r['seed']==seed and r['condition']=='saved'),None)
        if base is None:continue
        a=np.load(root/f'saved_seed{seed}'/'trace.npz')
        for r in results:
            if r['seed']!=seed or r['condition']=='saved':continue
            b=np.load(root/f"{r['condition']}_seed{seed}"/'trace.npz')
            if r['initial_state_sha256']!=base['initial_state_sha256']:
                raise RuntimeError('Initial state mismatch; do not present unmatched comparison as paired')
            valid=~(a['control_reset']|b['control_reset']).astype(bool)
            # Non-overlapping one-second windows with no reset in EITHER run.
            n=(len(valid)//50)*50
            window=valid[:n].reshape(-1,50,valid.shape[1]).all(1)
            mask=np.repeat(window,50,axis=0)
            ea=a['control_nonreset_error'][:n][mask]
            eb=b['control_nonreset_error'][:n][mask]
            paired.append(dict(seed=seed,condition=r['condition'],reference='saved',
                failures_delta=r['physical_resets']-base['physical_resets'],
                failure_rate_delta=r['failure_per_env_s']-base['failure_per_env_s'],
                ordinary_rmse_delta=np.asarray(r['tracking_rmse'])-base['tracking_rmse'],
                matched_window_seconds=float(mask.sum()*.02),coverage=float(mask.mean()),
                matched_reference_rmse=np.sqrt(np.mean(ea**2,axis=0)) if len(ea) else None,
                matched_variant_rmse=np.sqrt(np.mean(eb**2,axis=0)) if len(eb) else None,
                note='Common time windows, not identical physical states after earlier resets; correlated samples.'))
    write_json(root/'paired.json',paired)
    write_json(root/'summary.json',results)
    rows=[]
    for r in results:
        rows.append(dict(variant=r['condition'],seed=r['seed'],failures=r['physical_resets'],
            failures_per_env_s=r['failure_per_env_s'],timeouts=r['timeout_resets'],
            completed_episode_mean_s=r['completed_episode_length_s']['mean'],
            rmse_x=r['tracking_rmse'][0],rmse_y=r['tracking_rmse'][1],rmse_yaw=r['tracking_rmse'][2],
            primary=r['primary_rows'],recovery=r['recovery_rows'],fallback=r['fallback_rows'],
            correction_rms_nm=r['correction_coordinate_abs_nm']['rms'],
            correction_p95_nm=r['correction_coordinate_abs_nm']['p95'],
            correction_max_nm=r['correction_coordinate_abs_nm']['max'],contact_brier=r['contact_brier'],
            collection_wall_s=r['timing']['wall_s'],torch_peak_allocated_mib=r['timing']['torch_peak_allocated_bytes']/2**20))
    if rows:
        with (root/'comparison.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(VARIANTS,['black','tab:blue','tab:orange','tab:green']))
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    for name in VARIANTS:
        rr=[r for r in results if r['condition']==name]
        for i,r in enumerate(rr):
            x,y=report.survival(r['episode_records'])
            axes[0].step(x,y,where='post',color=colors[name],alpha=.55,label=name if i==0 else None)
        if rr:
            axes[1].plot(range(3),np.mean([r['tracking_rmse'] for r in rr],axis=0),'o-',color=colors[name],label=name)
            for r in rr:axes[2].scatter(r['seed'],r['contact_brier'],color=colors[name])
    axes[0].set(xlabel='Episode age [s]',ylabel='Descriptive survival (seeds separate)',ylim=(0,1.02));axes[0].legend(fontsize=7)
    axes[1].set(xticks=range(3),xticklabels=['vx [m/s]','vy [m/s]','yaw [rad/s]'],ylabel='Non-reset RMSE (seed mean)')
    axes[2].set(xlabel='Seed',ylabel='Contact probability Brier error')
    fig.tight_layout();fig.savefig(root/'survival_tracking_contacts.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for name in VARIANTS:
        rr=[r for r in results if r['condition']==name]
        for r in rr:
            for i,event in enumerate(('touchdown','liftoff')):
                value=r['contact_lags'][event]
                mean=value['lag_seconds']['mean']
                if mean is not None:axes[0].scatter(i,1000*mean,color=colors[name],label=name if i==0 and r is rr[0] else None)
                if value['matched_fraction'] is not None:axes[1].scatter(i,value['matched_fraction'],color=colors[name])
    for ax in axes:ax.set(xticks=[0,1],xticklabels=['touchdown','liftoff'])
    axes[0].set(ylabel='Mean signed lag [ms]; positive = late');axes[0].legend(fontsize=7)
    axes[1].set(ylabel='Matched within ±100 ms',ylim=(0,1))
    fig.tight_layout();fig.savefig(root/'contact_transitions.png',dpi=160);plt.close(fig)
    # First recorded failure per variant/seed: no cherry-picking severity.
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    for name in VARIANTS:
        rr=next((r for r in results if r['condition']==name and r['physical_resets']>0),None)
        if rr is None:continue
        a=np.load(root/f"{name}_seed{rr['seed']}"/'trace.npz')
        fail=a['control_reset']&a['control_failure']
        t,e=np.argwhere(fail)[0];start=max(0,t-49)
        previous=np.where(a['control_reset'][:t,e])[0]
        if len(previous):start=max(start,previous[-1]+1)
        sl=slice(start*4,(t+1)*4);times=(np.arange(start*4,(t+1)*4)-(t+1)*4)*.005
        axes[0].plot(times,np.sqrt(np.mean(a['correction'][sl,e]**2,axis=-1)),label=name,color=colors[name])
        axes[1].plot(times,np.mean(abs(a['contact_probability'][sl,e]-a['contact_truth'][sl,e]),axis=-1),color=colors[name])
        command=a['command'][sl,e,:2];speed=np.linalg.norm(command,axis=-1)
        direction=command/np.maximum(speed[:,None],1e-8)
        lateral=np.stack((-direction[:,1],direction[:,0]),axis=-1)
        acceleration=(a['candidate_delta'][sl,e,:2]-a['nominal_model_delta'][sl,e,:2])/.005
        projected=np.sum(acceleration*lateral,-1);projected[speed<=.1]=np.nan
        axes[2].plot(times,projected,color=colors[name])
    axes[0].set(ylabel='Applied correction RMS [Nm]');axes[0].legend(fontsize=8)
    axes[1].set(ylabel='Mean foot contact error',xlabel='Time before first failure [s]')
    axes[2].set(ylabel='Predicted lateral correction\nacceleration [m/s²]',xlabel='Time before first failure [s]')
    fig.tight_layout();fig.savefig(root/'first_failure_history.png',dpi=160);plt.close(fig)
    print(f'{len(results)} completed trials; {len(paired)} paired comparisons in {root}')


if __name__=='__main__':main()
