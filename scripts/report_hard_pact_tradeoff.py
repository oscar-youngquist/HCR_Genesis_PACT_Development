#!/usr/bin/env python3
"""Offline descriptive comparisons; no simulator, training or significance tests."""
import argparse
import csv
import json
import hashlib
import shutil
from pathlib import Path
import numpy as np
from scripts.eval_hard_pact_tradeoff import VARIANTS, report, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    root=p.parse_args().output
    archive=root/'sources'
    if archive.exists():shutil.copy2(Path(__file__),archive/Path(__file__).name)
    write_json(root/'analysis_provenance.json',dict(report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        memory_scope='PyTorch peak allocated/reserved only; excludes PhysX/renderer allocations. Collection includes diagnostics and shared-host contention, not hardware latency.'))
    results=[json.loads(f.read_text()) for f in sorted(root.glob('*/result.json'))]
    variants=list(dict.fromkeys(r['condition'] for r in results))
    reference='A_reference' if 'A_reference' in variants else 'saved'
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
            'aligned_next_substep_error_xyz':'Legacy key: components are body vx, vy, wz (NOT xyz linear velocity). m/s, m/s, rad/s.',
            'stance_slip':'Control-rate cached world foot speed times physics-rate contact; not an instantaneous substep velocity estimate',
            'objective_costs':'Accepted primary/recovery together; nominal may be infeasible; recovery slack and numerical SPD regularization excluded from named task costs',
            'model_motion':'Solved accepted rows only; whole reset intervals excluded; evolving body-frame differences match canonical body velocity derivatives',
            'correction_quantiles':'Pooled absolute joint-coordinate values in Nm; RMS is across joint coordinates',
            'reason_counts':'Overlapping threshold conditions at failure, not mutually exclusive causal attribution'}
    paired=[]
    for seed in sorted({r['seed'] for r in results}):
        base=next((r for r in results if r['seed']==seed and r['condition']==reference),None)
        if base is None:continue
        a=np.load(root/f'{reference}_seed{seed}'/'trace.npz')
        for r in results:
            if r['seed']!=seed or r['condition']==reference:continue
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
            prefix=np.cumsum((a['control_reset']|b['control_reset']).astype(bool),axis=0)==0
            prefix_window=prefix[:n].reshape(-1,50,prefix.shape[1]).all(1)
            prefix_mask=np.repeat(prefix_window,50,axis=0)
            pa=a['control_nonreset_error'][:n][prefix_mask];pb=b['control_nonreset_error'][:n][prefix_mask]
            paired.append(dict(seed=seed,condition=r['condition'],reference=reference,
                failures_delta=r['physical_resets']-base['physical_resets'],
                failure_rate_delta=r['failure_per_env_s']-base['failure_per_env_s'],
                ordinary_rmse_delta=np.asarray(r['tracking_rmse'])-base['tracking_rmse'],
                matched_window_seconds=float(mask.sum()*.02),coverage=float(mask.mean()),
                matched_reference_rmse=np.sqrt(np.mean(ea**2,axis=0)) if len(ea) else None,
                matched_variant_rmse=np.sqrt(np.mean(eb**2,axis=0)) if len(eb) else None,
                shared_pre_reset_env_s=float(prefix_mask.sum()*.02),
                shared_pre_reset_reference_rmse=np.sqrt(np.mean(pa**2,axis=0)) if len(pa) else None,
                shared_pre_reset_variant_rmse=np.sqrt(np.mean(pb**2,axis=0)) if len(pb) else None,
                note='Common time windows, not identical physical states after earlier resets; correlated samples.'))
    write_json(root/'paired.json',paired)
    write_json(root/'summary.json',results)
    if reference=='A_reference':
        objective_rows=[];decisions=[]
        for r in results:
            a=np.load(root/f"{r['condition']}_seed{r['seed']}"/'trace.npz')
            mask=np.repeat(~a['control_reset'].astype(bool)&~a['control_reasons'][...,6].astype(bool),4,axis=0)
            # Offline controls: same environment/terrain and closest mean
            # physical command among episode-safe, non-failure half-seconds.
            folder=root/f"{r['condition']}_seed{r['seed']}"
            matches=[];healthy={};reset=a['control_reset'].astype(bool)
            failures=reset&a['control_failure'].astype(bool)
            cap=json.loads((folder/'manifest.json').read_text()).get('capture_cap',24)
            for j,(t,e) in enumerate(np.argwhere(failures)[:cap]):
                target=a['control_command'][max(0,t-24):t+1,e].mean(0)
                ends=[end for end in range(24,len(reset)-25,25) if not reset[end-24:end+26,e].any()]
                if not ends:continue
                end=min(ends,key=lambda end:np.linalg.norm(a['control_command'][end-24:end+1,e].mean(0)-target))
                matches.append(dict(failure_id=j,environment=int(e),failure_step=int(t),healthy_start=end-24,healthy_end=end,
                    command_distance=float(np.linalg.norm(a['control_command'][end-24:end+1,e].mean(0)-target))))
                for field in ('command','velocity_before','orientation_xyzw','contact_probability','contact_truth','correction','stage','executed_model_delta','realized_delta'):
                    healthy[f'{j:02d}_{field}']=a[field][(end-24)*4:(end+1)*4,e]
            write_json(folder/'matched_healthy_index.json',matches)
            np.savez_compressed(folder/'matched_healthy_histories.npz',**healthy)
            for component in ('torque','grf','stance','attitude','height','planar','yaw'):
                for state in ('nominal','candidate','execution'):
                    value=a[f'objective_{state}_{component}'];delta=value-a[f'objective_nominal_{component}']
                    stats=report.finite_stats(value[mask]);changes=report.finite_stats(delta[mask])
                    objective_rows.append(dict(condition=r['condition'],seed=r['seed'],component=component,state=state,
                        valid_count=stats['count'],mean_weighted_cost=stats['mean'],mean_change=changes['mean'],max_cost=stats['max']))
            conflict={}
            for state in ('candidate','execution'):
                delta=a[f'objective_{state}_planar_error']-a['objective_nominal_planar_error']
                valid=mask&np.isfinite(delta);worse=delta>1e-4
                conflict[state]=dict(valid_rows=int(valid.sum()),worsened_fraction=float(worse[valid].mean()))
                for component in ('stance','attitude','height'):
                    improved=a[f'objective_{state}_{component}']<a[f'objective_nominal_{component}']-1e-6
                    conflict[state]['tracking_worse_'+component+'_better_fraction']=float((worse&improved)[valid].mean())
            r['objective_associations']=conflict
            baseline=next((v for v in results if v['condition']==reference and v['seed']==r['seed']),None)
            if baseline is not None and r['condition']!=reference:
                tracking=all(x<=y*1.10+.01 for x,y in zip(r['tracking_rmse'],baseline['tracking_rmse']))
                joint=all(r[k]['coordinate_fraction_above_tolerance']<=baseline[k]['coordinate_fraction_above_tolerance']+.001
                    for k in ('joint_position_violation','joint_velocity_violation'))
                torque=r['torque_violation']['max']<=baseline['torque_violation']['max']+.01
                decisions.append(dict(condition=r['condition'],seed=r['seed'],survival_improved=r['physical_resets']<baseline['physical_resets'],
                    tracking_within_predeclared_margin=tracking,joint_within_margin=joint,torque_within_margin=torque,
                    promising=r['physical_resets']<baseline['physical_resets'] and tracking and joint and torque))
        if objective_rows:
            with (root/'objective_costs.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(objective_rows[0]));writer.writeheader();writer.writerows(objective_rows)
        write_json(root/'followup_decisions.json',dict(preliminary=True,decisions=decisions,
            caveat='Descriptive screen, not statistical confirmation. Shared seeds diverge after resets; continuation requires reviewing terrain exposure and paired tracking.'))
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
    colors=dict(zip(variants,['black','tab:blue','tab:orange','tab:green','tab:purple']))
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    for name in variants:
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
    for name in variants:
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
    for name in variants:
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
    if reference=='A_reference':
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        components=('torque','grf','stance','attitude','height','planar','yaw')
        for name in variants:
            rows=[r for r in results if r['condition']==name]
            if not rows:continue
            changes=[np.mean([r['diagnostics']['objective_execution_change_'+k]['mean'] for r in rows]) for k in components]
            axes[0].plot(components,changes,'o-',label=name,color=colors[name])
            rms=[np.mean([r['intervention_groups'][g]['applied_nm']['rms'] for r in rows]) for g in ('baseline','additional')]
            axes[1].plot(['baseline','additional'],rms,'o-',label=name,color=colors[name])
        axes[0].axhline(0,color='gray',lw=.5)
        axes[0].set(ylabel='Mean executed-model weighted cost minus nominal',yscale='symlog')
        axes[1].set(ylabel='Applied correction RMS [Nm]')
        axes[0].legend(fontsize=7);axes[1].legend(fontsize=7)
        fig.tight_layout();fig.savefig(root/'objective_changes_corrections.png',dpi=160);plt.close(fig)
    print(f'{len(results)} completed trials; {len(paired)} paired comparisons in {root}')


if __name__=='__main__':main()
