"""Paired rollout RMSE inference for fixed checkpoints; CPU-only, no model imports."""
from collections import defaultdict
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

PAIRS=(('b1z1_pact','b1z1_pact_ab10_actor',True),
       ('b1z1_pact_ab9_representation','b1z1_pact_ab8_coupled_film',True),
       ('b1z1_pact_ab7_position_full','b1z1_pact_ab6_position_film',False))


def options(config):
    result=dict(resamples=config.get('rmse_bootstrap_resamples',10000),
                permutations=config.get('rmse_permutation_resamples',0),
                seed=config.get('rmse_resampling_seed',0),confidence=config.get('rmse_confidence_level',.95))
    for key in ('resamples','permutations','seed'):
        value=result[key]
        if isinstance(value,bool) or not isinstance(value,int) or value < (1 if key=='resamples' else 0):
            raise ValueError('Invalid RMSE resampling option: '+key)
    if not 0 < result['confidence'] < 1: raise ValueError('RMSE confidence level must lie in (0,1)')
    return result


def model_signature(meta):
    train=meta.get('resolved_training',{}); policy=train.get('policy',{})
    env=meta.get('resolved_environment',{}); dims=env.get('env',{})
    return dict(action_mode=policy.get('action_mode','position' if dims.get('num_policy_actions')==dims.get('num_actions') else 'coupled'),
        conditioning_mode=policy.get('conditioning_mode','film'),film_v02=policy.get('film_v02_enabled',False),
        dimensions={k:v for k,v in policy.items() if 'dim' in k},
        observations={k:v for k,v in dims.items() if k.startswith('num_') and ('obs' in k or 'action' in k)},
        gains={k:env.get('control',{}).get(k) for k in ('stiffness','damping')})


def scenario_match(on,off,tid):
    """Same prescribed experiment, not same resulting trajectory. Never infer pairing from seeds alone."""
    a,b=on['trials'][tid],off['trials'][tid]
    if not a.get('scenario_hash') or not b.get('scenario_hash'): return False,'missing scenario hash'
    if a['scenario_hash']!=b['scenario_hash']: return False,'scenario hash mismatch'
    for key in ('seed','motion_mode','condition','load_range','command','events','initial_seed','mechanics_seed'):
        if key not in a or key not in b: return False,'missing scenario field: '+key
        if a[key]!=b[key]: return False,'scenario field mismatch: '+key
    if on['meta']['dt']!=off['meta']['dt'] or a.get('prescribed_steps')!=b.get('prescribed_steps'):
        return False,'time grid/horizon mismatch'
    x,y=on['initial'].get(tid,{}),off['initial'].get(tid,{})
    if x.get('initial_state_hash') and x.get('initial_state_hash')==y.get('initial_state_hash'):
        return True,'scenario and initial-state hashes match'
    # Environment-grid offsets and tiny float roundoff must not reject equivalent starts.
    keys=('joint_position','base_quaternion_xyzw','body_masses_kg','body_inertias_kg_m2','body_local_com_pose')
    try:
        for key in keys:
            u,v=np.asarray(x[key]),np.asarray(y[key])
            if u.shape!=v.shape or not np.allclose(u,v,rtol=1e-5,atol=1e-6): return False,'initial mechanics/state mismatch: '+key
        u=np.asarray(x['base_position_world'])-x['origin_world']
        v=np.asarray(y['base_position_world'])-y['origin_world']
        if not np.allclose(u,v,rtol=1e-5,atol=1e-5): return False,'initial relative base position mismatch'
    except (KeyError,ValueError,TypeError): return False,'initial-state verification unavailable'
    return True,'scenario hash and numerical initial state match'


def end_step(run,tid):
    """Exclusive control-step bound: a failed successor itself earns no error sample."""
    trial=run['trials'][tid]; horizon=int(trial.get('prescribed_steps',run['meta'].get('prescribed_steps')))
    if not trial.get('physical_failure',False): return horizon
    time=trial.get('failure_time_seconds')
    if time is None or not np.isfinite(time): return None
    return max(0,min(horizon,int(round(time/run['meta']['dt']))-1))


def plan_pairs(runs,methods,support):
    """Resolve hashes and exact shared eligible steps once, before any BARD work.

    support(run) loads only compact index fields (or the already selected IDs).
    No histories, forces or predictions are read here.
    """
    from b1z1_paper_names import DISPLAY_NAMES
    from exp3_representation_batch import representation_pinn_status
    windows=defaultdict(dict); audit=[]; diagnostics=[]; supports={}
    by_method=defaultdict(list)
    for r in runs: by_method[r['name']].append(r)
    for on_task,off_task,actor_matched in PAIRS:
        on_ids=[k for k,v in methods.items() if v['task']==on_task and k in by_method]
        off_ids=[k for k,v in methods.items() if v['task']==off_task and k in by_method]
        label=DISPLAY_NAMES[on_task]+' vs '+DISPLAY_NAMES[off_task]
        if not on_ids and not off_ids: continue
        if not on_ids or not off_ids:
            diagnostics.append(label+': counterpart unavailable.'); continue
        if not actor_matched: diagnostics.append(label+': both representation and actor PINNs change.')
        for on_id,off_id in itertools.product(on_ids,off_ids):
            pair_id=json.dumps([on_id,off_id],separators=(',',':'))
            def run_key(r):
                m=r['meta']; return m['evaluation_seed'],m['arguments']['condition'],m['arguments']['load_range']
            off_runs={run_key(r):r for r in by_method[off_id]}
            for on in by_method[on_id]:
                off=off_runs.get(run_key(on))
                if off is None:
                    diagnostics.append(label+': missing counterpart recording for '+str(run_key(on))); continue
                a,b=on['meta'],off['meta']
                aa=a.get('resolved_training',{}).get('algorithm',{}).get('actor_phys_enabled')
                ba=b.get('resolved_training',{}).get('algorithm',{}).get('actor_phys_enabled')
                verified=(representation_pinn_status(a)[0] is True and representation_pinn_status(b)[0] is False
                    and model_signature(a)==model_signature(b) and isinstance(aa,bool) and isinstance(ba,bool)
                    and (aa==ba if actor_matched else aa and not ba))
                for tid in sorted(set(on['trials'])|set(off['trials'])):
                    row=dict(pair_id=pair_id,pair=label,enabled_method=on_id,baseline_method=off_id,
                        enabled_run=str(on['path']),baseline_run=str(off['path']),trial_id=tid,
                        enabled_checkpoint=a['checkpoint_sha256'],baseline_checkpoint=b['checkpoint_sha256'],
                        enabled_training_seed=a.get('checkpoint_training_seed'),baseline_training_seed=b.get('checkpoint_training_seed'),
                        evaluation_seed=a['evaluation_seed'],condition=a['arguments']['condition'],load_range=a['arguments']['load_range'],
                        representation_only_contrast=actor_matched,configuration_matched=bool(verified))
                    if tid not in on['trials'] or tid not in off['trials']:
                        audit.append(dict(row,matched=False,reason='missing counterpart trial')); continue
                    matched,reason=scenario_match(on,off,tid)
                    row.update(matched=matched,reason=reason,motion=on['trials'][tid]['motion_mode'],
                        scenario_hash=on['trials'][tid].get('scenario_hash'),
                        enabled_failed=on['trials'][tid].get('physical_failure'),baseline_failed=off['trials'][tid].get('physical_failure'))
                    common=np.array([],dtype=np.int64); stop=None
                    if matched:
                        limits=[end_step(r,tid) for r in (on,off)]
                        if all(v is not None for v in limits):
                            stop=min(limits)
                            for r in (on,off):
                                path=str(r['path'])
                                if path not in supports: supports[path]=support(r)
                            common=np.intersect1d(supports[str(on['path'])].get(tid,[]),supports[str(off['path'])].get(tid,[])).astype(np.int64)
                            common=common[common<stop]
                        for r in (on,off): windows[str(r['path'])].setdefault(pair_id,{})[tid]=common
                    row.update(common_end_step_exclusive=stop,common_end_seconds=None if stop is None else stop*a['dt'],
                        common_selected_steps=len(common),common_steps_sha256=hashlib.sha256(common.tobytes()).hexdigest())
                    audit.append(row)
    rejected=defaultdict(int)
    for row in audit:
        if not row['matched']: rejected[row['reason']]+=1
    diagnostics.extend(f'{n} trial pairs excluded: {reason}' for reason,n in sorted(rejected.items()))
    return dict(windows=dict(windows),audit=audit,diagnostics=diagnostics)


def batch_windows(plan,entries):
    """Only tiny boolean masks travel to workers, not full-trial index maps."""
    tids=np.array([e['trial_id'] for e in entries]); steps=np.array([e['step'] for e in entries])
    masks={}
    for pair,trials in plan.items():
        mask=np.zeros(len(entries),bool)
        for tid in np.unique(tids):
            ids=np.flatnonzero(tids==tid); mask[ids]=np.isin(steps[ids],trials.get(int(tid),[]),assume_unique=False)
        masks[pair]=mask
    return masks


class TrialRMSE:
    """SSE/count sufficient statistics; no full trajectories or rank sorting."""
    def __init__(self): self.sums={}
    def merge(self,sums):
        for key,(sse,n) in sums.items():
            old=self.sums.setdefault(key,[0.,0]); old[0]+=sse; old[1]+=n
    def add(self,velocity,names,windows):
        from exp3_representation_batch import velocity_indices
        if velocity is None: return
        tids=np.array([e['trial_id'] for e in velocity['entries']]); blocks=velocity_indices(names)
        scopes={'':np.ones(len(tids),bool),**windows}
        variants={variant:{block:(pred-velocity['post'])[:,ix] for block,ix in blocks.items() if len(ix)}
                  for variant,pred in velocity['predictions'].items()}
        variants.setdefault('decoded',{}).update(velocity.get('explicit_errors',{}))
        for variant,errors in variants.items():
            for block,e in errors.items():
                finite=np.isfinite(e).all(1); mse=np.mean(np.square(e),axis=1)
                for pair,mask in scopes.items():
                    ids=np.flatnonzero(mask&finite)
                    if not len(ids): continue
                    unique,inverse=np.unique(tids[ids],return_inverse=True)
                    sums=np.bincount(inverse,weights=mse[ids]); counts=np.bincount(inverse)
                    self.merge({(pair,int(t),variant,block):(float(s),int(n)) for t,s,n in zip(unique,sums,counts)})
    def records(self,run):
        m=run['meta']; rows=[]
        for (pair,tid,variant,block),(sse,n) in self.sums.items():
            rows.append(dict(method=run['name'],run=str(run['path']),checkpoint=m['checkpoint_sha256'],
                training_seed=m.get('checkpoint_training_seed'),evaluation_seed=m['evaluation_seed'],
                trial_id=tid,condition=m['arguments']['condition'],load_range=m['arguments']['load_range'],
                motion=run['trials'][tid]['motion_mode'],scope='common_pre_failure' if pair else 'own_pre_failure',
                pair_id=pair,variant=variant,block=block,unit=unit_for(block),
                sum_squared_error=sse,samples=n,rmse=float(np.sqrt(sse/n))))
        return rows


def paired_intervals(groups,settings,key):
    """Stratified paired trial bootstrap, conditional on these checkpoints/seeds.

    Draws are chunked (<=256) and vectorized. Equal seed weighting is retained;
    timesteps are never treated as independent replicates. Swaps test paired
    exchangeability, not just a zero mean under arbitrary unequal distributions.
    """
    entropy=int.from_bytes(hashlib.sha256(json.dumps([settings['seed'],key],sort_keys=True,default=str).encode()).digest()[:8],'little')
    rng=np.random.default_rng(entropy); n=sum(len(g) for g in groups); seeds=len(groups)
    observed=np.mean([g.mean(0) for g in groups],axis=0); delta=observed[0]-observed[1]
    result=dict(enabled_mean=observed[0],baseline_mean=observed[1],difference=delta,
        reduction_percent=100*(observed[1]-observed[0])/observed[1] if observed[1]>0 else np.nan,
        ci_low=np.nan,ci_high=np.nan,reduction_ci_low=np.nan,reduction_ci_high=np.nan,p_value=np.nan)
    if n>seeds:  # No within-stratum replication cannot support this bootstrap.
        draws=np.zeros((settings['resamples'],2))
        for start in range(0,len(draws),256):
            stop=min(start+256,len(draws))
            for g in groups:
                ids=rng.integers(len(g),size=(stop-start,len(g)))
                draws[start:stop]+=g[ids].mean(1)/seeds
        q=[(1-settings['confidence'])/2,(1+settings['confidence'])/2]
        result['ci_low'],result['ci_high']=np.quantile(draws[:,0]-draws[:,1],q)
        valid=draws[:,1]>0
        if valid.any(): result['reduction_ci_low'],result['reduction_ci_high']=np.quantile(100*(draws[valid,1]-draws[valid,0])/draws[valid,1],q)
    if settings['permutations'] and n>1:
        weighted=np.concatenate([(g[:,0]-g[:,1])/(seeds*len(g)) for g in groups])
        exact=n<24 and 2**n<=settings['permutations']; draws=2**n if exact else settings['permutations']; extreme=0
        for start in range(0,draws,256):
            size=min(256,draws-start)
            signs=(2*((np.arange(start,start+size)[:,None]>>np.arange(n))&1)-1 if exact else 2*rng.integers(2,size=(size,n))-1)
            extreme+=np.count_nonzero(np.abs(signs@weighted)>=abs(delta)-1e-12*max(1.,abs(delta)))
        result['p_value']=extreme/draws if exact else (extreme+1)/(draws+1)
    return result


def unit_for(block):
    return 'm' if block=='explicit_ee_position' else 'm/s' if block in ('base_linear','explicit_base_velocity') else 'rad/s'


def paired_results(records,plan,config):
    """Primary: paired trial RMSE difference; sensitivity: exactly shared valid steps."""
    settings=options(config); lookup={}; pairs=[]; diagnostics=list(plan['diagnostics'])
    for r in records:
        if r['variant']=='decoded': lookup[(r['run'],r['trial_id'],r['scope'],r['pair_id'],r['block'])]=r
    blocks=('base_linear','base_angular','joint_velocity','arm_velocity','leg_velocity','explicit_base_velocity','explicit_ee_position')
    for match in plan['audit']:
        if not match['matched']: continue
        for scope in ('own_pre_failure','common_pre_failure'):
            for block in blocks:
                pair=match['pair_id'] if scope=='common_pre_failure' else ''
                on=lookup.get((match['enabled_run'],match['trial_id'],scope,pair,block))
                off=lookup.get((match['baseline_run'],match['trial_id'],scope,pair,block))
                row=dict(match,scope=scope,variant='decoded',block=block,unit=unit_for(block))
                row.update(enabled_rmse=on['rmse'] if on else np.nan,baseline_rmse=off['rmse'] if off else np.nan,
                    enabled_samples=on['samples'] if on else 0,baseline_samples=off['samples'] if off else 0)
                # Nonfinite decoded rows must not create unequal sensitivity supports.
                usable=on is not None and off is not None
                if scope=='common_pre_failure': usable=usable and on['samples']==off['samples']==match['common_selected_steps']
                row.update(usable=usable,difference=on['rmse']-off['rmse'] if usable else np.nan)
                pairs.append(row)
    if not pairs: return dict(summary=[],per_seed=[],trials=[],diagnostics=diagnostics+['No verified trial pairs with velocity data.'])
    frame=pd.DataFrame(pairs); results=[]; seed_rows=[]
    keys=['pair_id','pair','enabled_method','baseline_method','enabled_checkpoint','baseline_checkpoint',
          'condition','load_range','motion','scope','variant','block','unit']
    for key,g in frame.groupby(keys,dropna=False,sort=True):
        result=dict(zip(keys,key)); valid=g[g.usable]; groups=[]
        for seed,s in valid.groupby('evaluation_seed',sort=True):
            values=s[['enabled_rmse','baseline_rmse']].to_numpy(); groups.append(values)
            seed_rows.append(dict(result,evaluation_seed=int(seed),enabled_mean=values[:,0].mean(),baseline_mean=values[:,1].mean(),
                difference=(values[:,0]-values[:,1]).mean(),trial_pairs=len(values)))
        result.update(trial_pairs=len(valid),attempted_pairs=len(g),evaluation_seeds=len(groups),
            enabled_samples=int(valid.enabled_samples.sum()),baseline_samples=int(valid.baseline_samples.sum()),
            representation_only_contrast=bool(g.representation_only_contrast.all()),configuration_matched=bool(g.configuration_matched.all()),
            enabled_training_seed=g.enabled_training_seed.iloc[0],baseline_training_seed=g.baseline_training_seed.iloc[0],
            confidence_level=settings['confidence'],bootstrap_resamples=settings['resamples'],permutation_resamples=settings['permutations'])
        if groups: result.update(paired_intervals(groups,settings,key))
        else: result.update({k:np.nan for k in ('enabled_mean','baseline_mean','difference','reduction_percent','ci_low','ci_high','reduction_ci_low','reduction_ci_high','p_value')})
        result['ci_excludes_zero']=bool(np.isfinite(result['ci_high']) and (result['ci_high']<0 or result['ci_low']>0))
        results.append(result)
    # One declared family: all available reported decoded contrasts, blocks and
    # both windows/conditions/motions in this analysis. CIs remain pointwise.
    ordered=sorted((i for i,r in enumerate(results) if np.isfinite(r['p_value'])),key=lambda i:results[i]['p_value'])
    previous=0.
    for rank,i in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-rank)*results[i]['p_value'])); results[i]['p_holm']=previous
    for r in results: r.setdefault('p_holm',np.nan)
    if any(not r['usable'] for r in pairs): diagnostics.append('Missing/nonfinite velocity data or empty shared samples excluded per metric; attempted pairs retained.')
    return dict(summary=results,per_seed=seed_rows,trials=pairs,diagnostics=diagnostics,
        convention='Difference = PINN minus control RMSE (negative favors PINNs). Equal trials within fixed evaluation-seed strata, then equal seeds; paired trial percentile bootstrap. CIs are pointwise; p_holm adjusts all reported tests. Fixed checkpoint inference, not independent training replication. Shared-time sensitivity does not make physical states identical.')
