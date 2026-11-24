"""Evaluation-only termination probe and bounded event reporting. Never imported by training."""
import math
import json
from pathlib import Path


class RecoveryProbe:
    # Outcome: 0=none/pending, 1=sustained recovery, 2=delayed failure,
    # 3=unresolved (timeout/safety relocation or end-of-evaluation).
    def __init__(self,n,device,dt,enabled,extension=.2,confirmation=.5):
        import torch
        if not all(math.isfinite(v) and v>0 for v in (dt,extension,confirmation)):
            raise ValueError('Probe durations must be finite and positive')
        self.enabled=enabled
        self.extension=math.ceil(extension/dt-1e-9)
        self.confirmation=math.ceil(confirmation/dt-1e-9)
        self.start=torch.full((n,),-1,device=device,dtype=torch.long)
        self.healthy=self.start.clone()

    def update(self,reset,failure,geometry_bad,hard,timeout,step):
        import torch
        active=self.start>=0
        ordinary=~active
        event=ordinary & reset & failure
        eligible=event & geometry_bad & ~hard & ~timeout
        start=eligible if self.enabled else torch.zeros_like(eligible)
        self.start[start]=step
        active=self.start>=0
        # First healthy control observation starts continuous confirmation.
        first_good=active & ~geometry_bad & (self.healthy<0)
        self.healthy[first_good]=step
        elapsed=step-self.start
        confirming=active & (self.healthy>=0)
        failed=active & (hard | (confirming & geometry_bad) |
                        (~confirming & (elapsed>=self.extension)))
        recovered=confirming & ~geometry_bad & ~hard & (step-self.healthy>=self.confirmation)
        unresolved=active & timeout & ~failed & ~recovered
        done=failed|recovered|unresolved
        outcome=torch.zeros_like(self.start)
        outcome[recovered]=1;outcome[failed]=2;outcome[unresolved]=3
        actual=reset.clone()
        actual[active]=done[active]
        record=dict(ordinary=ordinary,original_event=event,eligible=eligible,probe_start=start,
            probe_active=active,probe_outcome=outcome,probe_age_steps=torch.where(active,elapsed,-1),
            probe_first_healthy_step=self.healthy.clone(),probe_confirming=confirming,
            actual_reset=actual)
        self.start[done]=-1;self.healthy[done]=-1
        return record


def targeted_indices(source,terrain,seed,n=10):
    """Equal coverage of observed failing family/level cells, balanced by family."""
    import numpy as np
    import torch
    from collections import Counter
    source=Path(source)
    protocol=json.loads((source/'protocol.json').read_text())
    config=json.loads(Path(protocol['config']).read_text())['environment']['resolved']['terrain']
    proportions=np.cumsum(config['terrain_proportions']);cols=config['num_cols']
    failures=Counter()
    for path in sorted(source.glob('*/trace.npz')):
        a=np.load(path)
        for t,e in np.argwhere(a['control_reset']&a['control_failure']):
            col=int(a['control_terrain_type'][t,e]);level=int(a['control_terrain_level'][t,e])
            family=int(np.searchsorted(proportions,col/cols+.001,side='right'))
            if family in (2,3,4):failures[(family,level,col)]+=1
    rng=np.random.default_rng(seed); selected=[];plan=[]
    for family,budget in ((2,3),(3,3),(4,4)):
        levels=sorted({level for f,level,col in failures if f==family})
        if not levels or len(levels)>budget:raise ValueError('Insufficient balanced target slots; resolve explicitly')
        counts={level:sum(v for (f,l,c),v in failures.items() if f==family and l==level) for level in levels}
        allocation=levels+[max(levels,key=counts.get)]*(budget-len(levels))
        for level in allocation:
            columns=sorted(c for f,l,c in failures if f==family and l==level)
            col=int(rng.choice(columns))
            ids=((terrain['terrain_levels']==level)&(terrain['terrain_types']==col)).nonzero().flatten()
            if not len(ids):raise ValueError(f'No saved terrain row at {family}/{level}/{col}')
            selected.append(int(ids[int(rng.integers(len(ids)))]))
            plan.append(dict(family={2:'stairs_down',3:'stairs_up',4:'discrete_obstacles'}[family],
                             level=level,column=col,prior_failures=counts[level]))
    assert len(selected)==n
    return torch.tensor(selected),plan


def save_probe(observer,arrays,result):
    import numpy as np
    from scripts.eval_hard_pact_tradeoff import report,write_json
    ordinary=arrays['control_ordinary'].astype(bool)
    original=arrays['control_original_event'].astype(bool)
    original_reset=arrays['control_original_reset'].astype(bool)
    dt=observer.e.dt
    valid=ordinary & ~original_reset
    summary=dict(ordinary_env_s=float(ordinary.sum()*dt),probe_env_s=float((~ordinary).sum()*dt),
        original_failures=int(original.sum()),original_failure_rate=float(original.sum()/(ordinary.sum()*dt)),
        ordinary_tracking_rmse=[report.finite_stats(arrays['control_nonreset_error'][...,i][valid])['rms'] for i in range(3)],
        eligible_original_events=int(arrays['control_eligible'].sum()),
        ordinary_timeouts=int((ordinary&original_reset&arrays['control_reasons'][...,5]&~original).sum()),
        ordinary_completed_lengths_s=report.finite_stats(arrays['control_episode_steps'][ordinary&original_reset]*dt))
    events=[];captures={};cap=observer.args.capture_cap
    resets=arrays['control_reset'].astype(bool)
    outcomes=arrays['control_probe_outcome']
    for number,(t,env_id) in enumerate(np.argwhere(original)):
        started=bool(arrays['control_probe_start'][t,env_id])
        eligible=bool(arrays['control_eligible'][t,env_id])
        ends=np.flatnonzero(outcomes[t:,env_id]) if started else []
        end=t+int(ends[0]) if len(ends) else (len(original)-1 if started else t)
        status=int(outcomes[end,env_id]) if started and len(ends) else (3 if started else 0)
        healthy=int(arrays['control_probe_first_healthy_step'][end,env_id])
        confirm=arrays['control_probe_confirming'][t:end+1,env_id].astype(bool)
        err=arrays['control_nonreset_error'][t:end+1,env_id]
        event=dict(event_id=number,env=int(env_id),step=int(t),end_step=int(end),
            family=observer.manifest['target_plan'][env_id]['family'],level=int(arrays['control_terrain_level'][t,env_id]),
            eligible=eligible,probe_started=started,outcome={0:'normal_reset',1:'recovered',2:'delayed_failure',3:'unresolved'}[status],
            first_return_s=(healthy-t)*dt if healthy>=t else None,continuation_s=(end-t)*dt,
            reasons=arrays['control_reasons'][t,env_id].tolist(),end_reasons=arrays['control_reasons'][end,env_id].tolist(),
            confirmation_tracking_rmse=[report.finite_stats(err[:,i][confirm])['rms'] for i in range(3)],
            confirmation_samples=int(confirm.sum()))
        events.append(event)
        if number<cap:
            start=max(0,int(t)-math.ceil(.5/dt))
            prev=np.flatnonzero(resets[:t,env_id])
            if len(prev):start=max(start,int(prev[-1])+1)
            for name,a in arrays.items():
                captures[f'event{number}_{name}']=a[start:end+1,env_id] if name.startswith('control_') else a[start*4:(end+1)*4,env_id]
            captures[f'event{number}_identity']=np.array([env_id,start,t,end])
            # Same-clock non-failure reference, where an uninterrupted ordinary
            # interval exists. Not asserted to share identical physical state.
            candidates=np.flatnonzero((ordinary[start:end+1]&~original_reset[start:end+1]).all(0))
            if len(candidates):
                other=int(candidates[0]);event['reference_env']=other
                for name in ('correction','contact_probability','contact_truth','velocity_before','stance_slip','joint_position_violation'):
                    captures[f'event{number}_reference_{name}']=arrays[name][start*4:(end+1)*4,other]
    summary['outcomes']={s:sum(e['outcome']==s for e in events) for s in ('normal_reset','recovered','delayed_failure','unresolved')}
    starts=sum(e['probe_started'] for e in events)
    summary['probe_starts']=starts
    summary['recovery_fraction']=summary['outcomes']['recovered']/starts if starts else None
    summary['events']=events
    summary['captured_events']=min(len(events),cap);summary['capture_drops']=max(0,len(events)-cap)
    summary['terrain']=[]
    for family in ('stairs_down','stairs_up','discrete_obstacles'):
        for level in sorted({p['level'] for p in observer.manifest['target_plan'] if p['family']==family}):
            selected=np.array([p['family']==family and p['level']==level for p in observer.manifest['target_plan']])
            exposure=float(ordinary[:,selected].sum()*dt);failures=int(original[:,selected].sum())
            summary['terrain'].append(dict(family=family,level=level,ordinary_env_s=exposure,original_failures=failures,
                failure_rate=failures/exposure if exposure else None))
    physical_mask=np.repeat(valid,4,axis=0)
    for name in ('joint_position_violation','joint_velocity_violation','torque_violation','correction'):
        summary[name]=report.finite_stats(abs(arrays[name][physical_mask]))
    result['probe']=summary
    result['statistics_warning']='Legacy top-level metrics include diagnostic tails. Use probe.ordinary_* and original_failure_rate for comparisons.'
    write_json(observer.out/'result.json',result)
    write_json(observer.out/'probe_events.json',summary)
    np.savez_compressed(observer.out/'probe_histories.npz',**captures)


def report_probe(root,results):
    """Offline summaries exclude diagnostic tails; all event plots are descriptive."""
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.eval_hard_pact_tradeoff import report,write_json
    rows=[];terrain=[];parity=[];timing=[];contracts=[]
    for r in results:
        name=r['condition'];directory=root/f'{name}_seed1';p=r['probe']
        a=np.load(directory/'trace.npz');manifest=json.loads((directory/'manifest.json').read_text())
        contract={k:manifest[k] for k in ('checkpoint_sha256','initial_state_hash','terrain_source_rows',
            'execution_alpha','extra_strength','substep_probabilities','environment')}
        contract['qp_except_attitude']={k:v for k,v in manifest['effective_qp'].items() if k!='attitude_weight'}
        contracts.append(contract)
        if contract!=contracts[0]:raise RuntimeError('Matched-condition contract changed beyond attitude/probe override')
        ordinary=a['control_ordinary'].astype(bool)
        valid=np.repeat(ordinary&~a['control_original_reset'],4,axis=0)
        physical=np.repeat(ordinary,4,axis=0)  # post-physics measurements precede reset
        r['ordinary_physical_including_failure_step']={k:report.finite_stats(abs(a[k][physical])) for k in
            ('correction','joint_position_violation','joint_velocity_violation','torque_violation')}
        for key,tolerance in (('joint_position_violation',1e-4),('joint_velocity_violation',1e-3),('torque_violation',1e-4)):
            data=a[key][physical];finite=np.isfinite(data)
            r['ordinary_physical_including_failure_step'][key].update(count_threshold=tolerance,
                coordinate_fraction_above_threshold=float((data[finite]>tolerance).mean()) if finite.any() else None)
        r['ordinary_diagnostics']={k:report.finite_stats(a[k][valid]) for k in a.files
            if k.startswith('objective_') or k in ('stance_slip','grf_error','wrench_error','velocity_estimate_error')}
        r['verified_assembly_weights']=np.unique(a['assembly_weights'][np.isfinite(a['assembly_weights']).all(-1)],axis=0).tolist()
        r['frozen_schedule']={k:manifest[k] for k in ('execution_alpha','extra_strength','substep_probabilities')}
        expected=[.25,20.,.5 if '050' in name else .1]
        if not np.allclose(r['verified_assembly_weights'],[expected]):raise RuntimeError('Wrong effective objective weights')
        if not r['timing']['weights_unchanged']:raise RuntimeError('Network weights changed during evaluation')
        row=dict(condition=name,ordinary_env_s=p['ordinary_env_s'],probe_env_s=p['probe_env_s'],
            failures=p['original_failures'],failure_rate=p['original_failure_rate'],
            recovered=p['outcomes']['recovered'],delayed_failure=p['outcomes']['delayed_failure'],
            unresolved=p['outcomes']['unresolved'],probe_starts=p['probe_starts'],recovery_fraction=p['recovery_fraction'],
            rmse_x=p['ordinary_tracking_rmse'][0],rmse_y=p['ordinary_tracking_rmse'][1],rmse_yaw=p['ordinary_tracking_rmse'][2],
            correction_rms_nm=p['correction']['rms'],correction_p95_nm=p['correction']['p95'],correction_max_nm=p['correction']['max'],
            joint_position_max_rad=p['joint_position_violation']['max'],joint_velocity_max_rad_s=p['joint_velocity_violation']['max'],
            torque_max_nm=p['torque_violation']['max'])
        measurements=r['ordinary_physical_including_failure_step']
        for stat in ('rms','p95','max'):row[f'correction_{stat}_nm']=measurements['correction'][stat]
        for field,tag in (('joint_position_violation','joint_position_max_rad'),('joint_velocity_violation','joint_velocity_max_rad_s'),('torque_violation','torque_max_nm')):
            row[tag]=measurements[field]['max']
        rows.append(row)
        for t in p['terrain']:terrain.append(dict(condition=name,**t))
        # Counts and correction summaries have explicitly ordinary exposure only.
        r['ordinary_solver_stage_counts']={str(int(s)):int(((a['stage']==s)&physical).sum()) for s in np.unique(a['stage'])}
        r['ordinary_contact_brier']=report.finite_stats(((a['contact_probability']-a['contact_truth'])**2)[physical])
        for cost in ('torque','grf','stance','attitude','height','planar','yaw'):
            delta=a['objective_candidate_'+cost]-a['objective_nominal_'+cost]
            r['ordinary_diagnostics']['objective_change_'+cost]=report.finite_stats(delta[valid])
        r['ordinary_corrections_by_substep_group']={}
        for k in range(4):
            for base in (True,False):
                mask=valid[k::4]&(a['baseline'][k::4]==base)&(a['stage'][k::4]>=0)
                r['ordinary_corrections_by_substep_group'][f'{k}_{"baseline" if base else "extra"}']=report.finite_stats(abs(a['correction'][k::4][mask]))
        r['overlapping_original_reasons']={label:int((a['control_original_event']&a['control_reasons'][...,i]).sum())
            for i,label in enumerate(('contact','roll','pitch','height_low','height_high','timeout','terrain_exit'))}
        histories=np.load(directory/'probe_histories.npz')
        eligible_plots=[e['event_id'] for e in p['events'][:p['captured_events']] if e['eligible']]
        plot_ids=set((eligible_plots or list(range(p['captured_events'])))[:3])
        for e in p['events'][:p['captured_events']]:
            prefix=f"event{e['event_id']}_";env,start,event,end=histories[prefix+'identity']
            # Original decision is AFTER the fourth physics step of event.
            # Commands/predictions are at step start, measured contacts at end.
            x=(np.arange(start*4,(end+1)*4)-(event+1)*4)*.005
            xc=(np.arange(start,end+1)-event)*.02
            correction=np.sqrt(np.mean(histories[prefix+'correction']**2,-1))
            bad=histories[prefix+'control_reasons'][:,1:5].any(-1)
            first=np.flatnonzero(bad)
            peak=int(np.argmax(correction));cross=float(xc[first[0]]) if len(first) else None
            timing.append(dict(condition=name,event_id=e['event_id'],outcome=e['outcome'],
                peak_correction_time_s=float(x[peak]),first_observed_bounds_violation_time_s=cross,
                crossing_left_censored=bool(len(first) and first[0]==0),peak_rms_nm=float(correction[peak])))
            if e['event_id'] not in plot_ids:continue  # prioritize eligible geometry events
            fig,ax=plt.subplots(4,1,figsize=(10,10),sharex=True)
            for foot in range(4):
                line=ax[0].plot(x,histories[prefix+'contact_probability'][:,foot],label=f'foot {foot}')[0]
                ax[0].plot(x+.005,histories[prefix+'contact_truth'][:,foot],':',color=line.get_color(),alpha=.5)
            ax[0].set(ylabel='Contact p / measured (dotted)');ax[0].legend(ncol=4,fontsize=8)
            for index,label in enumerate(('roll [rad]','pitch [rad]')):
                ax[1].plot(xc,histories[prefix+'control_tilt'][:,index],label=label)
            ax[1].plot(xc,histories[prefix+'control_base_height'],label='height [m]')
            ax[1].set(ylabel='Roll/pitch [rad], height [m]');ax[1].legend()
            ax[2].plot(x,correction,label='executed correction RMS [Nm]')
            ax[2].plot(x,histories[prefix+'execution_strength'],label='execution alpha');ax[2].legend()
            for label in ('nominal','optimized'):
                values=histories[prefix+label+'_angular_acceleration_world'][:,:2]
                ax[3].plot(x,np.linalg.norm(values,axis=-1),label=label+' world horizontal angular acceleration')
            ax[3].plot(x,np.linalg.norm(histories[prefix+'desired_angular_acceleration_world'][:,:2],axis=-1),':',label='desired horizontal acceleration')
            ax[3].set(xlabel='Time relative to ORIGINAL failure decision [s]',ylabel='[rad/s²]');ax[3].legend(fontsize=8)
            for axis in ax:axis.axvline(0,color='black',linestyle='--');axis.grid(alpha=.2)
            fig.suptitle(f"{name}: {e['family']} level {e['level']}, env {env}, {e['outcome']}")
            fig.tight_layout();fig.savefig(root/f"{name}_event{e['event_id']}.png",dpi=130);plt.close(fig)
    for attitude in ('050','010'):
        paths=[root/f'attitude{attitude}_{suffix}_seed1' for suffix in ('normal','probe')]
        if not all((d/'trace.npz').exists() for d in paths):continue
        traces=[np.load(d/'trace.npz') for d in paths]
        manifests=[json.loads((d/'manifest.json').read_text()) for d in paths]
        first=min([int(np.flatnonzero(t['control_eligible'].any(1))[0]) if t['control_eligible'].any() else len(t['control_eligible']) for t in traces])
        delta=traces[0]['executed_torque'][:first*4]-traces[1]['executed_torque'][:first*4]
        parity.append(dict(attitude=attitude,initial_hash_equal=manifests[0]['initial_state_hash']==manifests[1]['initial_state_hash'],
            pre_probe_eligible_steps=first,max_torque_difference_nm=float(np.max(abs(delta))) if delta.size else None,
            note='Separate GPU runs can differ numerically; comparison ends before either eligible probe event.'))
    for filename,data in (('comparison.csv',rows),('summary.csv',rows),('terrain_comparison.csv',terrain),('event_timing.csv',timing)):
        if data:
            with (root/filename).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write_json(root/'summary.json',results);write_json(root/'baseline_parity.json',parity)
    write_json(root/'verification.json',dict(completed_conditions=len(rows),matched_initial_contracts=True,
        weights_unchanged=True,assembly_weights_verified=True,checkpoint_sha256=contracts[0]['checkpoint_sha256']))
    differences=[]
    for probe in ('normal','probe'):
        reference=next((r for r in rows if r['condition']==f'attitude050_{probe}'),None)
        candidate=next((r for r in rows if r['condition']==f'attitude010_{probe}'),None)
        if reference and candidate:
            differences.append(dict(reference=reference['condition'],candidate=candidate['condition'],
                **{k:candidate[k]-reference[k] for k in ('failures','failure_rate','rmse_x','rmse_y','rmse_yaw','correction_rms_nm')},
                note='Single matched initial seed, not matched states after resets; differences are descriptive.'))
    write_json(root/'paired_differences.json',differences)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for row in rows:
        axes[0].bar(row['condition'],row['failure_rate'])
        axes[1].plot(range(3),[row['rmse_x'],row['rmse_y'],row['rmse_yaw']],'o-',label=row['condition'])
    axes[0].set(ylabel='ORIGINAL failures / ordinary env-second');axes[0].tick_params(axis='x',rotation=35)
    axes[1].set(xticks=range(3),xticklabels=['vx [m/s]','vy [m/s]','yaw [rad/s]'],ylabel='Ordinary non-reset RMSE');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(rows,indent=2))
