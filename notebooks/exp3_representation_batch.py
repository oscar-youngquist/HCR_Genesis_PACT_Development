"""Batched offline reductions and trial-level rollout agreement; no model imports."""
from collections import defaultdict
import numpy as np

TERM_KEYS=('mass_matrix','bias','foot_jacobians','ee_jacobian','base_jacobian','ee_lever','base_lever')
STATE_KEYS=('base_velocity_world_t','base_angular_velocity_world_t','joint_velocity_t',
            'base_velocity_world','base_angular_velocity_world','joint_velocity')
PREDICTIONS=('prediction_ee_force','prediction_base_wrench','prediction_grf','prediction_explicit',
             'prediction_next_velocity_decoded','prediction_next_velocity_measured_loading',
             'prediction_base_velocity','prediction_ee_position','prediction_explicit_condition','prediction_explicit_prediction')
EXPLICIT_STATE_FIELDS=('ee_position_t_world','ee_reference_t_world','ee_reference_sphere_t','commands_t')


def missing_predictions(meta,available):
    import exp3_representation as a
    if not a.family(meta).startswith('pact'): return [] if 'prediction_explicit' in available else ['prediction_explicit']
    required=['prediction_ee_force','prediction_base_wrench','prediction_grf']
    if not any(k in available for k in ('prediction_explicit_condition','prediction_explicit_prediction','prediction_explicit')):
        required+=['prediction_base_velocity','prediction_ee_position']
    return [k for k in required if k not in available]


def explicit_errors(pred,meta,row):
    """Current-state estimates, never successor predictions or command tracking errors.

    PACT predicts body [vx, vy, wz], UniFP predicts [vx, vy, vz]; compare their
    common planar velocity components in m/s, never mix yaw rate into linear RMSE.
    EE spherical estimates become Cartesian meters in the yaw-aligned workspace.
    Its origin cancels using the recorded reference at t; no guessed terrain height.
    """
    import exp3_representation as a
    scales=meta['resolved_environment']['normalization']['obs_scales']; result={}
    explicit=next((pred[k] for k in ('prediction_explicit_condition','prediction_explicit_prediction','prediction_explicit') if k in pred),None)
    velocity=pred.get('prediction_base_velocity',None if explicit is None else explicit[...,:3])
    sphere=pred.get('prediction_ee_position',None if explicit is None else explicit[...,3:6])
    q=np.asarray(row['base_quaternion_t'],float).copy(); q[...,:3]*=-1
    if velocity is not None and 'base_velocity_world_t' in row:
        measured=a.rotate(row['base_velocity_world_t'],q)
        result['explicit_base_velocity']=velocity[...,:2]/scales['lin_vel']-measured[...,:2]
    reference=row.get('ee_reference_sphere_t')
    if reference is None and 'commands_t' in row: reference=row['commands_t'][...,3:6]
    if sphere is not None and reference is not None and all(k in row for k in ('ee_position_t_world','ee_reference_t_world')):
        scale=np.array([scales[k] for k in ('ee_sphe_radius_cmd','ee_sphe_pitch_cmd','ee_sphe_yaw_cmd')])
        def cart(s):
            r,p,y=np.moveaxis(s,-1,0)
            return np.stack((r*np.cos(p)*np.cos(y),r*np.cos(p)*np.sin(y),r*np.sin(p)),axis=-1)
        # Extract forward yaw before inversion: yaw(conjugate(q)) != -yaw(q)
        # when roll and pitch are nonzero. The workspace ignores both tilts.
        orientation=np.asarray(row['base_quaternion_t'],float)
        x,y,z,w=np.moveaxis(orientation/np.linalg.norm(orientation,axis=-1,keepdims=True),-1,0)
        yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        inverse_yaw=np.stack((np.zeros_like(yaw),np.zeros_like(yaw),-np.sin(yaw/2),np.cos(yaw/2)),axis=-1)
        measured=cart(reference)+a.rotate(row['ee_position_t_world']-row['ee_reference_t_world'],inverse_yaw)
        result['explicit_ee_position']=cart(sphere/scale)-measured
    return result


def representation_pinn_status(meta):
    """Match the recorded PACT runner's legacy default and policy-weight override."""
    train=meta.get('resolved_training',{}); algorithm=train.get('algorithm',{}); policy=train.get('policy',{})
    weight=policy.get('pinn_loss_weight',algorithm.get('pinn_loss_weight'))
    flag=algorithm.get('representation_pinn_enabled')
    source='explicit'
    if flag is None:
        if train.get('runner',{}).get('algorithm_class_name')=='PPO_B1Z1PACT' and weight is not None:
            flag=True; source='legacy_runner_default'
        else: return None,'unknown'
    return bool(flag and (weight is None or weight!=0)),source


def pack_terms(terms):
    """Seven dense arrays per batch, instead of seven ZIP members per transition."""
    return dict(arrays={k:np.stack([r[k] for r in terms]) for k in TERM_KEYS},
                approximation=[r.get('approximation',[]) for r in terms])


def unpack_terms(packed):
    return [dict({k:v[i] for k,v in packed['arrays'].items()},approximation=notes)
            for i,notes in enumerate(packed['approximation'])]


def pack_job(job):
    """Avoid pickling thousands of tiny NumPy arrays between CPU workers."""
    rows,predictions,fixed,*rest=job
    def columns(records):
        keys=set(records[0])
        if any(set(r)!=keys for r in records): return records
        return {'_columns':{k:np.stack([r[k] for r in records]) for k in keys if k!='entry'},
                '_entries':[r.get('entry') for r in records]}
    return columns(rows),columns(predictions),({'_terms':pack_terms(fixed)} if all(isinstance(f,dict) for f in fixed) else fixed),*rest


def process_batch(job):
    """Pure CPU worker: same SI/frame equations, two RHS sharing each mass solve.

    Error arrays remain per transition until trial-group reduction. Invalid fields
    are NaN internally and excluded, never treated as zero-error observations.
    """
    import exp3_representation as a
    rows,predictions,fixed,meta,name,dynamics,*extra=job
    def restore(records):
        if not isinstance(records,dict): return records
        return [dict({k:v[i] for k,v in records['_columns'].items()},**({'entry':e} if e else {}))
                for i,e in enumerate(records['_entries'])]
    rows,predictions=restore(rows),restore(predictions)
    if isinstance(fixed,dict): fixed=unpack_terms(fixed['_terms'])
    n=len(rows); entries=[r['entry'] for r in rows]; issues=[]
    def issue(component,reason,indices):
        if len(indices): issues.append((component,reason,len(indices),entries[int(indices[0])]))
    def stack(key): return np.stack([r[key] for r in rows])
    timing=np.full(n,'sustained',dtype=object)
    for k in a.LOADS:
        if all(k in r and 'history_'+k in r for r in rows):
            timing[np.any(~np.isclose(stack(k),stack('history_'+k),atol=1e-6),axis=1)]='new_load'
    physical={}
    for field in PREDICTIONS[:4]:
        ids=[i for i,p in enumerate(predictions) if field in p]
        if not ids: continue
        try:
            context={k:np.stack([rows[i][k] for i in ids]) for k in ('base_quaternion_t','base_quaternion_xyzw')}
            values=a.physical_predictions({field:np.stack([predictions[i][field] for i in ids])},meta,context)
            for k,v in values.items():
                physical.setdefault(k,np.full((n,*v.shape[1:]),np.nan))[ids]=v
        except (KeyError,ValueError):
            # Unusual malformed rows do not discard valid neighbours.
            for i in ids:
                try:
                    for k,v in a.physical_predictions({field:predictions[i][field]},meta,rows[i]).items():
                        physical.setdefault(k,np.full((n,*v.shape),np.nan))[i]=v
                except (KeyError,ValueError) as exc: issue(field,str(exc),[i])
    errors={}
    targets=dict(ee_force='history_ee_force_world',base_force='history_base_force_world',
                 base_moment='history_base_moment_world',grf='grf_raw_world_interval_mean')
    for component,key in targets.items():
        if not all(key in r for r in rows): continue
        if component not in physical:
            issue(component,'Decoder output unavailable; no measured substitution',np.arange(n))
        else: errors[component]=physical[component]-stack(key)
    stats,pooled=a.TrialStatistics(),a.TrialStatistics(pooled=True)
    def accumulate(variant,values):
        for component,value in values.items():
            bad=~np.isfinite(value).reshape(n,-1).all(1)
            issue(component,'Nonfinite/empty error; excluded from this metric',np.flatnonzero(bad))
        for target in (stats,pooled):
            target.add_batch(entries,name,variant,values,meta['resolved_environment']['asset']['foot_name'],meta['dof_names'],timing)
    accumulate('decoded',errors)
    explicit={}
    for fields in (('prediction_base_velocity',),('prediction_ee_position',),('prediction_explicit_condition',),('prediction_explicit_prediction',),('prediction_explicit',)):
        ids=[i for i,p in enumerate(predictions) if any(k in p for k in fields)]
        if not ids: continue
        try:
            available=set.intersection(*(set(rows[i]) for i in ids))
            context={k:np.stack([rows[i][k] for i in ids]) for k in ('base_quaternion_t','base_velocity_world_t',*EXPLICIT_STATE_FIELDS) if k in available}
            present=set.intersection(*(set(predictions[i]) for i in ids))
            values=explicit_errors({k:np.stack([predictions[i][k] for i in ids]) for k in fields if k in present},meta,context)
            for k,v in values.items():
                array=explicit.setdefault(k,np.full((n,*v.shape[1:]),np.nan)); keep=np.isnan(array[ids]).all(1); array[np.asarray(ids)[keep]]=v[keep]
        except (KeyError,ValueError) as exc: issue('explicit state',str(exc),ids)
    for key in ('explicit_base_velocity','explicit_ee_position'):
        if key not in explicit: issue(key,'Current-state prediction or corresponding recorded measurement unavailable',np.arange(n))
    accumulate('decoded',explicit)
    if not all(all(k in r for k in STATE_KEYS) for r in rows):
        issue('velocity dynamics','Required velocity states unavailable',np.arange(n))
        sums={}
        if extra and extra[0] is not None:
            from exp3_rollout_statistics import TrialRMSE
            accumulator=TrialRMSE(); accumulator.add(dict(entries=entries,predictions={},explicit_errors=explicit),meta['dof_names'],extra[0]); sums=accumulator.sums
        return dict(stats=stats.sums,pooled=pooled.sums,issues=issues,velocity=None,rmse_sums=sums,sources={})
    pre=np.concatenate([stack(k) for k in STATE_KEYS[:3]],axis=1).astype(float)
    post=np.concatenate([stack(k) for k in STATE_KEYS[3:]],axis=1).astype(float)
    successor={'constant_velocity':pre.copy()}; sources=defaultdict(int)
    ids=[i for i,r in enumerate(rows) if isinstance(fixed[i],dict)
         and all(k in r for k in (*a.LOADS,'grf_raw_world_interval_mean','executed_torque_interval_mean'))]
    if ids:
        terms={k:np.stack([fixed[i][k] for i in ids]) for k in TERM_KEYS}
        m=len(ids); rhs=np.zeros((m,pre.shape[1],2)); valid=np.zeros((m,2),bool)
        for j,variant in enumerate(('measured_loading','decoded')):
            if variant=='decoded':
                if set(physical)!={'ee_force','base_force','base_moment','grf'}: continue
                forces={k:v[ids] for k,v in physical.items()}
            else:
                forces=dict(zip(('ee_force','base_force','base_moment','grf'),
                    [np.stack([rows[i][k] for i in ids]) for k in (*a.LOADS,'grf_raw_world_interval_mean')]))
            ee=np.concatenate((forces['ee_force'],np.cross(terms['ee_lever'],forces['ee_force'])),axis=1)
            base=np.concatenate((forces['base_force'],forces['base_moment']+np.cross(terms['base_lever'],forces['base_force'])),axis=1)
            generalized=np.pad(np.stack([rows[i]['executed_torque_interval_mean'] for i in ids]),((0,0),(6,0))).astype(float)
            generalized=generalized+np.einsum('bfkn,bfk->bn',terms['foot_jacobians'][:,:,:3],forces['grf'])
            generalized+=np.einsum('bkn,bk->bn',terms['ee_jacobian'],ee)+np.einsum('bkn,bk->bn',terms['base_jacobian'],base)
            rhs[:,:,j]=generalized-terms['bias']; valid[:,j]=np.isfinite(rhs[:,:,j]).all(1)
        # A missing decoded head must not contaminate measured-loading solves.
        rhs=np.where(valid[:,None,:],rhs,0.)
        try: acceleration=np.linalg.solve(terms['mass_matrix'],rhs)
        except np.linalg.LinAlgError:
            acceleration=np.full_like(rhs,np.nan)
            for k in range(m):
                try: acceleration[k]=np.linalg.solve(terms['mass_matrix'][k],rhs[k])
                except np.linalg.LinAlgError: issue('rollout dynamics','Singular mass matrix',[ids[k]])
        for j,variant in enumerate(('measured_loading','decoded')):
            keep=np.flatnonzero(valid[:,j])
            if len(keep):
                value=successor.setdefault(variant,np.full_like(pre,np.nan))
                dest=np.asarray(ids)[keep]; value[dest]=pre[dest]+meta['dt']*acceleration[keep,:,j]
    if dynamics:
        for variant in ('decoded','measured_loading'):
            for i,row in enumerate(rows):
                error=a.stored_successor_error(row,variant)
                if error is not None:
                    successor.setdefault(variant,np.full_like(pre,np.nan))[i]=post[i]+error
                    sources['stored:next_velocity_'+variant]+=1
    blocks=velocity_indices(meta['dof_names'])
    for variant,pred in successor.items():
        # Missing predictions are excluded per block, rather than reported as real zeros.
        values={key:(pred-post)[:,ix] for key,ix in blocks.items()}
        for target in (stats,pooled):
            target.add_batch(entries,name,variant,values,meta['resolved_environment']['asset']['foot_name'],meta['dof_names'],timing)
    velocity=dict(pre=pre,post=post,predictions=successor,entries=entries,timing=timing,explicit_errors=explicit)
    rmse_sums={}
    if extra:
        if extra[0] is not None:
            from exp3_rollout_statistics import TrialRMSE
            accumulator=TrialRMSE(); accumulator.add(velocity,meta['dof_names'],extra[0]); rmse_sums=accumulator.sums
        velocity=None  # Production IPC carries sufficient statistics, not velocity trajectories.
    return dict(stats=stats.sums,pooled=pooled.sums,issues=issues,velocity=velocity,rmse_sums=rmse_sums,sources=dict(sources))


def velocity_indices(names):
    return dict(base_linear=np.arange(3),base_angular=np.arange(3,6),joint_velocity=np.arange(6,6+len(names)),
                arm_velocity=np.array([6+i for i,n in enumerate(names) if n.startswith('z1_')],int),
                leg_velocity=np.array([6+i for i,n in enumerate(names) if not n.startswith('z1_')],int))


