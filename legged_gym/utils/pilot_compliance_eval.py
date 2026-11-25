"""Simulator-independent RSS trial manifests and episode-level statistics."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np

DEFAULT_RUN = 'Nov19_04-02-42_b1z1_pact_improved'
CONDITIONS = ('up_yield_up_force', 'up_yield_down_force', 'down_yield_up_force',
              'down_yield_down_force', 'ample_base_on', 'ample_base_off',
              'restricted_base_on', 'restricted_base_off')
CALIBRATIONS=('calibration_up_30n','calibration_up_50n')
EVAL_CONFIG = dict(horizon_s=16., force_on_s=5., ramp_up_s=1., force_off_s=10.,
    ramp_down_s=1., trajectory_s=3., yielding_coefficient=.5, base_coefficient=.5,
    force_n=20., amplitude_fraction=[.95,1.05], initial_yaw_rad=[-.2,.2],
    initial_xy_m=.02, initial_joint_rad=.02, vertical_sphere=[.55,.2,0.],
    ample_sphere=[.55,0.,0.], restricted_sphere=[.85,0.,0.],
    witness_grid=16, compile_geometry=True, settle_tracking_tolerance_m=.15,
    recovery_tolerance_m=.03, bootstrap_samples=2000, reference_diagnostics=True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def resolve_checkpoint(run, checkpoint=None):
    run=Path(run).expanduser().resolve()
    if not run.is_dir(): raise FileNotFoundError(f'Exact run directory missing: {run}')
    files={int(m[1]):p for p in run.iterdir() if p.is_file()
           and (m:=re.fullmatch(r'model_(\d+)\.pt',p.name))}
    if not files: raise FileNotFoundError(f'No numeric checkpoint in {run}')
    if checkpoint is None or str(checkpoint)=='-1': number=max(files)
    else:
        match=re.fullmatch(r'(?:model_)?(\d+)(?:\.pt)?',str(checkpoint))
        if not match: raise ValueError('Checkpoint must be a numeric iteration or model_<iteration>.pt inside this run')
        number=int(match[1])
    if number not in files: raise FileNotFoundError(f'model_{number}.pt missing from {run}')
    return files[number],number


def validate_config(c):
    if not 0<c['trajectory_s']<c['force_on_s']<c['force_on_s']+c['ramp_up_s']<c['force_off_s']<c['force_off_s']+c['ramp_down_s']<c['horizon_s']:
        raise ValueError('Need ordered trajectory, settling, ramp, sustained, removal, recovery intervals')
    for name in ('yielding_coefficient','base_coefficient'):
        if not 0<c[name]<=1: raise ValueError(name)
    if c['force_n']<=0 or c['amplitude_fraction'][0]<=0 or c['amplitude_fraction'][0]>c['amplitude_fraction'][1]:
        raise ValueError('Invalid force range')


def condition(name,c):
    profile=np.zeros(12);force=np.zeros(3)
    if name=='nominal': sphere=c['vertical_sphere']
    elif name in CALIBRATIONS:
        profile[5]=1.;force[2]=1.;sphere=c['vertical_sphere']
    elif name in CONDITIONS[:4]:
        profile[5 if name.startswith('up_yield') else 4]=c['yielding_coefficient']
        force[2]=1 if name.endswith('_up_force') else -1
        sphere=c['vertical_sphere']
    elif name in CONDITIONS[4:]:
        profile[1]=c['yielding_coefficient'];force[0]=1
        if name.endswith('_on'): profile[7]=c['base_coefficient']
        sphere=c['restricted_sphere'] if name.startswith('restricted') else c['ample_sphere']
    else: raise ValueError(name)
    return dict(condition=name,profile=profile.tolist(),force_heading_direction=force.tolist(),nominal_sphere=list(sphere))


def manifests(seed,episodes,conditions,c):
    """One immutable draw per pair ID; condition/reset/batch order cannot change it."""
    rows=[]
    for trial in range(episodes):
        rng=np.random.default_rng(np.random.SeedSequence([seed,trial,9173]))
        draw=dict(initial_yaw_rad=float(rng.uniform(*c['initial_yaw_rad'])),
            initial_xy_m=rng.uniform(-c['initial_xy_m'],c['initial_xy_m'],2).tolist(),
            joint_delta_rad=rng.uniform(-c['initial_joint_rad'],c['initial_joint_rad'],17).tolist(),
            force_amplitude_n=float(c['force_n']*rng.uniform(*c['amplitude_fraction'])),
            force_on_s=c['force_on_s'],force_off_s=c['force_off_s'],
            ramp_up_s=c['ramp_up_s'],ramp_down_s=c['ramp_down_s'])
        for name in conditions:
            # Calibration uses exact forces, not the main paired amplitude jitter.
            actual=draw if name not in CALIBRATIONS else dict(draw,force_amplitude_n=30. if name==CALIBRATIONS[0] else 50.)
            rows.append(dict(seed=seed,trial_id=trial,pair_id=f'{seed}:{trial}',
                pair_hash=fingerprint(actual),**actual,**condition(name,c)))
    return rows


def batches(rows,n):
    if n<1: raise ValueError('num-envs must be positive')
    for i in range(0,len(rows),n): yield rows[i:i+n]


def force_scale(t,c):
    """Held force for [t,t+dt), continuous linear ramps, no extra physics force."""
    return np.minimum(np.clip((np.asarray(t)-c['force_on_s'])/c['ramp_up_s'],0,1),
                      1-np.clip((np.asarray(t)-c['force_off_s'])/c['ramp_down_s'],0,1))


def phase(t,c):
    return np.select([t<c['force_on_s'],t<c['force_on_s']+c['ramp_up_s'],
                      t<c['force_off_s'],t<c['force_off_s']+c['ramp_down_s']],
                     [0,1,2,3],default=4)


def summarize(data,manifest,c,dt):
    """Tracking uses finite pre-failure successors; failures retain attempted counts."""
    t=data['time'];alive=data['alive_for_metrics'].astype(bool)
    valid=alive & np.isfinite(data['ee_task']).all(-1)
    mean=lambda a,mask: float(np.mean(a[mask])) if np.any(mask) else None
    pre=valid&(t>=c['force_on_s']-1)&(t<=c['force_on_s'])
    loaded=valid&(t>=c['force_on_s']+c['ramp_up_s']+1)&(t<=c['force_off_s'])
    recovered=valid&(t>=c['horizon_s']-2)
    axis=2 if manifest['condition'] in (*CONDITIONS[:4],*CALIBRATIONS,'nominal') else 0
    error=data['ee_task']-data['reference_task']
    deviation=data['ee_task']-data['nominal_task']
    refdev=data['reference_task']-data['nominal_task']
    b0=mean(deviation[:,axis],pre);r0=mean(refdev[:,axis],pre)
    displacement=None if b0 is None else mean(deviation[:,axis]-b0,loaded)
    reference_displacement=None if r0 is None else mean(refdev[:,axis]-r0,loaded)
    protected=manifest['profile'][2*axis+(manifest['force_heading_direction'][axis]>0)]==0
    failed=bool(data['failure'].any())
    reasons=[k for i,k in enumerate(('contact','roll','pitch','height_min','height_max','reference_recovery','nonfinite','native_timeout')) if data['failure_reasons'][:,i].any()]
    base_delta=data['base_world'][:,:2]-data['base_world'][0,:2]
    yaw=manifest['initial_yaw_rad'];direction=np.array([np.cos(yaw),np.sin(yaw)])
    base_error=data['base_measured']-data['base_reference']
    settle_error=mean(np.linalg.norm(deviation,axis=1),pre)
    return dict(**manifest,failed=failed,failure_reasons=reasons,
        setup_tracking_failed=settle_error is None or settle_error>c['settle_tracking_tolerance_m'],
        observed_seconds=len(t)*dt,valid_pre_failure_seconds=int(valid.sum())*dt,
        valid_samples=int(valid.sum()),prescribed_seconds=c['horizon_s'],
        ee_tracking_error_m=mean(np.linalg.norm(error,axis=1),valid),
        planar_reference_error_m_s=mean(np.linalg.norm(base_error[:,:2],axis=1),valid),
        yaw_reference_error_rad_s=mean(np.abs(base_error[:,2]),valid),
        signed_displacement_m=displacement,reference_signed_displacement_m=reference_displacement,
        protected_deviation_m=abs(displacement) if protected and displacement is not None else None,
        recovery_error_m=None if b0 is None else mean(np.abs(deviation[:,axis]-b0),recovered),
        base_guidance_velocity_m_s=mean(data['base_measured'][:,0],loaded),
        base_reference_velocity_m_s=mean(data['base_reference'][:,0],loaded),
        base_guidance_displacement_m=mean(base_delta@direction,loaded),
        settled_nominal_error_m=settle_error,
        settled_workspace_margin=([mean(data['workspace_margin'][:,i],pre) for i in range(3)]),
        settled_joint_margin_rad=mean(data['joint_margin'],pre),
        loaded_gate=mean(data['gates'][:,axis],loaded),
        loaded_joint_gate=mean(data['joint_gate'][:,axis],loaded),
        loaded_workspace_gate=mean(data['workspace_gate'][:,axis],loaded),
        loaded_collision_gate=mean(data['collision_gate'][:,axis],loaded),
        clipping_fraction=mean(data['clipped'].astype(float),valid),
        actual_peak_force_n=float(np.linalg.norm(data['applied_force_world'],axis=1).max()),
        baseline_samples=int(pre.sum()),loaded_samples=int(loaded.sum()),recovery_samples=int(recovered.sum()))


METRICS=('ee_tracking_error_m','protected_deviation_m','planar_reference_error_m_s',
    'yaw_reference_error_rad_s','signed_displacement_m','reference_signed_displacement_m',
    'recovery_error_m','base_guidance_velocity_m_s','base_reference_velocity_m_s',
    'base_guidance_displacement_m','clipping_fraction','valid_pre_failure_seconds',
    'settled_nominal_error_m','settled_joint_margin_rad','loaded_gate',
    'loaded_joint_gate','loaded_workspace_gate','loaded_collision_gate','actual_peak_force_n')


def statistics(values):
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    return dict(mean=float(x.mean()) if len(x) else None,sd=float(x.std(ddof=1)) if len(x)>1 else None,n=len(x))


def aggregate(episodes):
    groups=[]
    for seed in sorted({e['seed'] for e in episodes}):
        for name in sorted({e['condition'] for e in episodes}):
            rows=[e for e in episodes if e['seed']==seed and e['condition']==name]
            if not rows: continue
            groups.append(dict(seed=seed,condition=name,attempted=len(rows),
                failure_rate=sum(e['failed'] for e in rows)/len(rows),
                failure_reasons={reason:sum(reason in e.get('failure_reasons',[]) for e in rows)
                    for reason in sorted({r for e in rows for r in e.get('failure_reasons',[])})},
                setup_tracking_failure_rate=sum(e['setup_tracking_failed'] for e in rows)/len(rows),
                metrics={k:statistics([e[k] for e in rows]) for k in METRICS}))
    per_condition=[]
    for name in sorted({g['condition'] for g in groups}):
        g=[x for x in groups if x['condition']==name]
        per_condition.append(dict(condition=name,failure_rate=statistics([x['failure_rate'] for x in g]),
            metrics={k:statistics([x['metrics'][k]['mean'] for x in g]) for k in METRICS}))
    return dict(per_seed_condition=groups,per_condition=per_condition,
        equal_condition_mean={k:statistics([x['metrics'][k]['mean'] for x in per_condition]) for k in METRICS},
        equal_condition_failure_rate=statistics([x['failure_rate']['mean'] for x in per_condition]),
        interpretation='Evaluation variability of ONE checkpoint, not independent training seeds. SD at each hierarchy is sample SD.')


def paired_contrasts(episodes,repeats=2000):
    """Paired episode draws within each seed, then equal-seed bootstrap means."""
    contrasts=[('up_yield_up_force','down_yield_up_force','signed_displacement_m'),
        ('down_yield_down_force','up_yield_down_force','signed_displacement_m'),
        ('restricted_base_on','restricted_base_off','base_guidance_velocity_m_s'),
        ('ample_base_on','ample_base_off','base_guidance_velocity_m_s'),
        ('restricted_base_on','ample_base_on','base_guidance_velocity_m_s'),
        ('restricted_base_off','ample_base_off','base_guidance_velocity_m_s')]
    lookup={(e['seed'],e['trial_id'],e['condition']):e for e in episodes};out=[]
    rng=np.random.default_rng(1701)
    for a,b,metric in contrasts:
        groups={};pairs=[]
        for (seed,trial,cond),x in lookup.items():
            if cond!=a: continue
            y=lookup.get((seed,trial,b))
            if y is None: continue
            if x['pair_hash']!=y['pair_hash']: raise ValueError('Mismatched paired scenarios')
            if x[metric] is None or y[metric] is None: continue
            delta=x[metric]-y[metric];groups.setdefault(seed,[]).append(delta)
            pairs.append(dict(seed=seed,trial_id=trial,a=x[metric],b=y[metric],difference=delta))
        boot=[]
        for values in groups.values():
            x=np.asarray(values);boot.append(x[rng.integers(0,len(x),(repeats,len(x)))].mean(1))
        interval=np.quantile(np.mean(boot,axis=0),[.025,.975]).tolist() if boot else [None,None]
        out.append(dict(a=a,b=b,metric=metric,mean_difference=float(np.mean([np.mean(x) for x in groups.values()])) if groups else None,
            ci95=interval,n_pairs=len(pairs),n_seeds=len(groups),paired_episodes=pairs,
            convention='A minus B; complete paired measurements only; failures remain in main outcome denominator'))
    return out
