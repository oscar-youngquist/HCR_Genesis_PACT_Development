"""Offline paired reports; no simulator imports. All uncertainty is descriptive."""
import csv
import json
from pathlib import Path
import numpy as np


def finite_stats(x):
    a = np.asarray(x).reshape(-1)
    f = a[np.isfinite(a)]
    return dict(count=int(f.size),nonfinite=int(a.size-f.size),
        mean=float(f.mean()) if f.size else None,
        rms=float(np.sqrt(np.mean(f*f))) if f.size else None,
        p95=float(np.quantile(f,.95)) if f.size else None,
        p99=float(np.quantile(f,.99)) if f.size else None,
        max=float(f.max()) if f.size else None)


def contact_lags(a):
    truth = a['contact_truth'].astype(bool)
    pred = a['contact_probability'] >= .5
    reset = a['control_reset'].astype(bool)
    result = {}
    # Match nearest transition of the same type within +/-100ms, one-to-one.
    # Search never crosses an episode boundary; signed lag>0 means late.
    for up,name in [(True,'touchdown'),(False,'liftoff')]:
        lags=[]; unmatched=0; total=0
        for env in range(truth.shape[1]):
            ends=list((np.where(reset[:,env])[0]+1)*4)+[truth.shape[0]]
            start=0
            for end in sorted(set(ends)):
                for foot in range(4):
                    t=truth[start:end,env,foot]; p=pred[start:end,env,foot]
                    te=np.where((t[1:]!=t[:-1]) & (t[1:]==up))[0]+1
                    pe=list(np.where((p[1:]!=p[:-1]) & (p[1:]==up))[0]+1)
                    for event in te:
                        total+=1
                        if pe:
                            j=int(np.argmin(np.abs(np.asarray(pe)-event)))
                            delta=pe[j]-event
                            if abs(delta)<=20:
                                lags.append(delta*.005);pe.pop(j);continue
                        unmatched+=1
                start=end
        result[name]=dict(lag_seconds=finite_stats(lags),truth_transitions=total,
            unmatched_within_100ms=unmatched,matched_fraction=(total-unmatched)/total if total else None)
    return result


def summarize(folder):
    folder=Path(folder)
    a=np.load(folder/'trace.npz')
    p=json.loads((folder/'progress.json').read_text())
    m=json.loads((folder/'manifest.json').read_text())
    reset=a['control_reset'].astype(bool); failure=a['control_failure'].astype(bool)&reset
    valid=~reset
    error=a['control_nonreset_error']
    rmse=[finite_stats(error[...,i][valid])['rms'] for i in range(3)]
    first=[]; episode_records=[]
    for env in range(reset.shape[1]):
        ids=np.where(failure[:,env])[0]
        first.append(float((ids[0]+1)*.02) if len(ids) else None)
        start=0
        for end in np.where(reset[:,env])[0]:
            episode_records.append([env,(end-start+1)*.02,bool(failure[end,env])]);start=end+1
        if start<reset.shape[0]: episode_records.append([env,(reset.shape[0]-start)*.02,False])
    stage=a['stage']; accepted=(stage==0)|(stage==1)
    residual=a['constraint_residual']
    correction=a['correction']
    result=dict(condition=p['condition'],seed=p['seed'],steps=p['completed_steps'],
        tracking_rmse=rmse,tracking_rows=int(valid.sum()),physical_resets=int(failure.sum()),
        non_timeout_resets=int((reset&~a['control_reasons'][...,5].astype(bool)).sum()),
        timeout_resets=int((reset&a['control_reasons'][...,5].astype(bool)&~failure).sum()),
        cause_counts={name:int((failure&a['control_reasons'][...,i]).sum()) for i,name in enumerate(
            ['contact','roll','pitch','height_low','height_high'])},
        time_to_first_failure_s=first,episode_records=episode_records,
        primary_rows=int((stage==0).sum()),recovery_rows=int((stage==1).sum()),
        fallback_rows=int((stage==2).sum()),unsolved_rows=int((stage==-1).sum()),
        correction_coordinate_abs_nm=finite_stats(abs(correction)),
        correction_row_norm_nm=finite_stats(np.linalg.norm(correction,axis=-1)),
        constraint_residual_normalized=finite_stats(residual[accepted]),
        contact_brier=float(np.mean((a['contact_probability']-a['contact_truth'])**2)),
        contact_lags=contact_lags(a),timing=p,
        initial_state_sha256=m['initial_state_hash'],scenario_sha256=m['scenario_sha256'])
    for field,tolerance in [('torque_violation',1e-4),('joint_position_violation',1e-4),('joint_velocity_violation',1e-3),
        ('candidate_joint_position_violation',1e-4),('candidate_joint_velocity_violation',1e-3)]:
        if field not in a: continue
        data=a[field];finite=np.isfinite(data)
        result[field]=finite_stats(data);result[field]['coordinate_fraction_above_tolerance']=float(np.mean(data[finite]>tolerance)) if finite.any() else None
        result[field]['count_threshold']=tolerance
    result['one_substep_model']={}
    for st,name in [(0,'primary'),(1,'recovery')]:
        mask=stage==st
        result['one_substep_model'][name]={}
        for field in ('candidate_delta','executed_model_delta','nominal_model_delta'):
            err=a[field]-a['realized_delta']
            result['one_substep_model'][name][field]=[finite_stats(err[...,j][mask]) for j in range(3)]
        cmd=a['command'][...,:2]; length=np.linalg.norm(cmd,axis=-1)
        direction=cmd/np.maximum(length[...,None],1e-8)
        lateral=np.stack((-direction[...,1],direction[...,0]),-1)
        for basis,label in [(direction,'along_command'),(lateral,'lateral_to_command')]:
            for field in ('torque_correction_delta','candidate_delta'):
                delta=a[field][...,:2]
                if field=='candidate_delta': delta=delta-a['nominal_model_delta'][...,:2]
                value=np.sum(delta*basis,-1)/.005
                result['one_substep_model'][name][field+'_'+label+'_m_s2']=finite_stats(value[mask&(length>.1)])
    # Bounded 0.5-second pre-failure windows: at most 20 episodes/run. Full
    # compact traces remain available for paired offline analysis.
    snippets={}; events=np.argwhere(failure)
    for i,(step,env) in enumerate(events[:20]):
        end=(step+1)*4;start=max(0,end-100)
        previous=np.where(reset[:step,env])[0]
        if len(previous): start=max(start,(previous[-1]+1)*4)
        for field in ('command','velocity_before','orientation_xyzw','contact_probability','contact_truth',
            'nominal_torque','executed_torque','correction','stage','candidate_delta','realized_delta','torque_correction_delta'):
            snippets[f'event{i}_{field}']=a[field][start:end,env]
        snippets[f'event{i}_identity']=np.array([step,env,start,end])
    np.savez_compressed(folder/'prefailure.npz',**snippets)
    result['prefailure_windows']=min(20,len(events));result['prefailure_windows_dropped']=max(0,len(events)-20)
    return result


def survival(records):
    times=sorted(set(float(r[1]) for r in records))
    x=[0.]; y=[1.];s=1.
    for t in times:
        risk=sum(r[1]>=t for r in records);d=sum(r[1]==t and r[2] for r in records)
        if risk: s*=1-d/risk
        x.append(t);y.append(s)
    return x,y


def report(root):
    from evaluate import write_json
    root=Path(root)
    results=[json.loads(p.read_text()) for p in sorted(root.glob('*/result.json'))]
    with (root/'comparison.csv').open('w') as f:
        keys=['condition','seed','forward_rmse_m_s','lateral_rmse_m_s','yaw_rmse_rad_s','resets',
            'primary','recovery','fallback','correction_rms_nm','contact_brier','collection_wall_s']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in results:
            vals=[r['condition'],r['seed'],*r['tracking_rmse'],r['physical_resets'],r['primary_rows'],
                r['recovery_rows'],r['fallback_rows'],r['correction_coordinate_abs_nm']['rms'],
                r['contact_brier'],r['timing']['wall_s']]
            w.writerow(dict(zip(keys,vals)))
    paired=[]
    for seed in sorted({r['seed'] for r in results}):
        rows=[r for r in results if r['seed']==seed]
        for i,left in enumerate(rows):
            for right in rows[i+1:]:
                a=np.load(root/f"{left['condition']}_seed{seed}"/'trace.npz')
                b=np.load(root/f"{right['condition']}_seed{seed}"/'trace.npz')
                # Shared 1-second windows, initial episodes only. No window
                # contains or follows either condition's first reset.
                reset=a['control_reset']|b['control_reset']
                safe=np.cumsum(reset,axis=0)==0
                mask=np.zeros_like(safe)
                for start in range(0,len(safe)-49,50):
                    mask[start:start+50]=safe[start:start+50].all(axis=0)
                ea=a['control_terminal_velocity']-a['control_command']
                eb=b['control_terminal_velocity']-b['control_command']
                ra=[finite_stats(ea[...,j][mask])['rms'] for j in range(3)]
                rb=[finite_stats(eb[...,j][mask])['rms'] for j in range(3)]
                paired.append(dict(seed=seed,left=left['condition'],right=right['condition'],
                    identical_initial_hash=left['initial_state_sha256']==right['initial_state_sha256'],
                    identical_scenario_hash=left['scenario_sha256']==right['scenario_sha256'],
                    shared_window_seconds_per_env=mask.sum(0)*.02,coverage_fraction=float(mask.mean()),
                    shared_rmse_left=ra,shared_rmse_right=rb,
                    shared_rmse_right_minus_left=[y-x if x is not None and y is not None else None for x,y in zip(ra,rb)],
                    ordinary_rmse_right_minus_left=np.array(right['tracking_rmse'])-left['tracking_rmse'],
                    physical_resets_right_minus_left=right['physical_resets']-left['physical_resets']))
    write_json(root/'paired.json',paired)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for r in results:
        label=f"{r['condition']} s{r['seed']}"
        axs[0].step(*survival(r['episode_records']),where='post',label=label)
        axs[1].plot(['vx','vy','yaw'],r['tracking_rmse'],marker='.',label=label)
    axs[0].set(xlabel='Episode duration [s]',ylabel='KM survival (timeouts censored)',ylim=(0,1.02))
    axs[1].set(ylabel='RMSE [m/s; yaw rad/s]')
    axs[0].legend(fontsize=5);fig.tight_layout();fig.savefig(root/'survival_tracking.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for j,r in enumerate(results):
        for k,name in enumerate(('touchdown','liftoff')):
            data=r['contact_lags'][name]
            val=data['lag_seconds']['mean']
            if val is not None: axs[k].scatter(j,val*1000,s=25,label=f"{r['condition']} s{r['seed']}")
            axs[k].set(title=name,xlabel='Run (comparison.csv order)',ylabel='Matched signed lag [ms]')
    axs[0].legend(fontsize=5);fig.tight_layout();fig.savefig(root/'contact_lags.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for r in results:
        p=np.load(root/f"{r['condition']}_seed{r['seed']}"/'prefailure.npz')
        curves=[];lateral=[]
        for name in p.files:
            if name.endswith('_correction'):
                c=p[name];buf=np.full(100,np.nan);buf[-len(c):]=np.sqrt(np.mean(c*c,-1));curves.append(buf)
            if name.endswith('_torque_correction_delta'):
                c=p[name];buf=np.full(100,np.nan);buf[-len(c):]=c[:,1]/.005;lateral.append(buf)
        if curves:
            label=f"{r['condition']} s{r['seed']}"
            axs[0].plot(np.arange(-99,1)*.005,np.nanmean(curves,axis=0),label=label)
            axs[1].plot(np.arange(-99,1)*.005,np.nanmean(lateral,axis=0),label=label)
    axs[0].set(xlabel='Seconds before failure',ylabel='Torque correction RMS [Nm]')
    axs[1].set(xlabel='Seconds before failure',ylabel='Predicted torque-only lateral acceleration [m/s²]')
    axs[0].legend(fontsize=5);fig.tight_layout();fig.savefig(root/'prefailure.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    import sys
    report(Path(sys.argv[1]))
