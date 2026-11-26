"""Causal offline shadow inverse dynamics; never imports a policy or simulator."""
import argparse,csv,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from scripts.hard_pact_shadow_estimation import acceleration,estimate,rotate


def valid_intervals(a):
    clean=~a['control_reset'].astype(bool)&~a['control_reasons'][...,6].astype(bool)
    clean[0]=False
    # Any unannounced world-position discontinuity invalidates the interval.
    jump=np.linalg.norm(a['shadow_position_after']-a['shadow_position_before'],axis=-1)>.1
    return np.repeat(clean,4,axis=0)&~jump


def stats(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if not x.size:return dict(n=0,bias=None,mae=None,rmse=None,p95=None,p99=None,max=None)
    return dict(n=int(x.size),bias=float(x.mean()),mae=float(abs(x).mean()),
        rmse=float(np.sqrt((x*x).mean())),p95=float(np.quantile(abs(x),.95)),
        p99=float(np.quantile(abs(x),.99)),max=float(abs(x).max()))


def calculate(a,damping):
    outputs={};n,e=a['executed_torque'].shape[:2]
    def tensor(x):return torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float64)
    q=tensor(a['orientation_xyzw']);v=tensor(a['shadow_velocity_before']);end=tensor(a['shadow_velocity_after'])
    dt=tensor(a['shadow_dt'])
    outputs['physics_acceleration']=acceleration(q,v,end,dt).numpy()
    good=valid_intervals(a)
    control_good=good.reshape(-1,4,e).all(1)[:-1]&good[4::4]
    # Instantaneous pushes/resets between the last physics sample and the next
    # policy tick are not a continuous 20ms inverse-dynamics interval.
    boundary_jump=np.max(abs(a['shadow_velocity_before'][4::4]-a['shadow_velocity_after'][3:-1:4]),axis=-1)>1e-4
    control_good &= ~boundary_jump
    # Endpoint t+1 estimates become available only at the next 20ms policy tick.
    cv=v[::4].clone()
    cv[...,:3]=rotate(q[::4],tensor(a['shadow_estimated_linear_body'][::4]))
    ca=acceleration(q[::4][:-1],cv[:-1],cv[1:],dt[::4][:-1]*4)
    outputs['control_acceleration']=ca.numpy()
    outputs['control_boundary_jump']=boundary_jump
    outputs['physics_interval_end_s']=(np.arange(n)+1)*float(a['shadow_dt'][0,0,0])
    outputs['control_interval_end_s']=(np.arange(n//4-1)+1)*float(a['shadow_dt'][0,0,0])*4
    controls=(n//4)-1
    streams={}
    for cadence,acc,ix,torque,mask in (
        ('physics',outputs['physics_acceleration'],slice(None),a['shadow_applied_torque_readback'],good),
        ('control',ca.numpy(),slice(0,controls*4,4),a['shadow_applied_torque_readback'].reshape(-1,4,e,12).mean(1)[:-1],control_good)):
        dims=acc.shape[:2]
        mechanics=[tensor(a['shadow_'+k][ix]).flatten(0,1) for k in ('mass_matrix','bias','foot_jacobians','base_jacobian')]
        contact=a['contact_truth'][ix] if cadence=='physics' else a['contact_truth'].reshape(-1,4,e,4).mean(1)[:-1]>.5
        gates={'ungated':np.ones(dims+(4,)), 'predicted':a['qp_stance_mask'][ix], 'oracle_gate':contact}
        for gate,g in gates.items():
            out=estimate(*mechanics,tensor(acc).flatten(0,1),tensor(torque).flatten(0,1),tensor(g).flatten(0,1),damping)
            name=cadence+'_'+gate
            streams[name]={k:x.numpy().reshape(dims+x.shape[1:]) for k,x in out.items()}
            streams[name]['valid'] &= mask
    return outputs,streams,good,control_good


def lag(pred,truth,valid,dt):
    # Positive lag: prediction best matches an earlier truth (delayed estimate).
    scores=[]
    for k in range(-5,6):
        p=pred[max(k,0):len(pred)+min(k,0)]
        t=truth[max(-k,0):len(truth)-max(k,0)]
        m=valid[max(k,0):len(valid)+min(k,0)]&valid[max(-k,0):len(valid)-max(k,0)]
        for offset in range(1,abs(k)):
            m &= valid[offset:len(valid)-abs(k)+offset]
        scores.append(stats((p-t)[m])['rmse'])
    return float((int(np.nanargmin([np.inf if x is None else x for x in scores]))-5)*dt)


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--damping',type=float,default=1e-4)
    p.add_argument('--force-residual-bound-n',type=float)
    p.add_argument('--moment-residual-bound-nm',type=float);args=p.parse_args()
    for bound in (args.force_residual_bound_n,args.moment_residual_bound_nm):
        if bound is not None and (not np.isfinite(bound) or bound<=0):p.error('Residual bounds must be positive and finite')
    torch.set_num_threads(1);started=time.monotonic();out=args.output;out.mkdir(parents=True,exist_ok=True)
    a=dict(np.load(args.run/'trace.npz'));m=json.loads((args.run/'manifest.json').read_text())
    arrays,streams,good,cgood=calculate(a,args.damping)
    rows=[];lags=[];health=[];exceed=[]
    e=good.shape[1];n=good.shape[0];c=n//4-1
    failure=a['control_reset'].astype(bool)&a['control_failure'].astype(bool)
    prefail=np.zeros_like(failure)
    for t,env in np.argwhere(failure):
        previous=np.flatnonzero(a['control_reset'][:t,env])
        start=max(0,t-25,int(previous[-1])+1 if len(previous) else 0)
        prefail[start:t+1,env]=True
    active=np.linalg.norm(a['shadow_applied_wrench'],axis=-1)>1e-6
    trans=np.zeros_like(a['contact_truth'],dtype=bool)
    trans[1:]=a['contact_truth'][1:]!=a['contact_truth'][:-1]
    for shift in (1,2,3,4):trans[shift:]|=(a['contact_truth'][shift:]!=a['contact_truth'][:-shift])
    family=np.array([x['family'] for x in m['target_plan']])
    truth={}
    for cadence,valid,sl,dt in [('physics',good,slice(None),.005),('control',cgood,slice(0,c*4,4),.020)]:
        mean=lambda x:x.reshape((-1,4)+x.shape[1:]).mean(1)[:-1]
        truth[cadence]={'force':a['shadow_force_truth'] if cadence=='physics' else mean(a['shadow_force_truth']),
                        'wrench':a['shadow_applied_wrench'] if cadence=='physics' else mean(a['shadow_applied_wrench'])}
        neural={'force':a['pred_force'] if cadence=='physics' else mean(a['pred_force']),
                'wrench':a.get('shadow_predicted_wrench',a['predicted_wrench']) if cadence=='physics' else mean(a.get('shadow_predicted_wrench',a['predicted_wrench'])),'valid':valid}
        models={'neural':neural,**{k:v for k,v in streams.items() if k.startswith(cadence)}}
        if cadence=='control':
            for name,model in streams.items():
                if name.startswith('physics'):
                    models[name+'_interval_mean']={key:mean(model[key]) for key in ('force','wrench')}
                    models[name+'_interval_mean']['valid']=model['valid'].reshape(-1,4,e).all(1)[:-1]&valid
        basegroups={'all':np.ones_like(valid),'active':active[sl],'inactive':~active[sl],
                    'pre_failure':np.repeat(prefail,4,axis=0)[sl]}
        for f in np.unique(family):basegroups['terrain_'+f]=np.broadcast_to(family==f,valid.shape)
        capture_path=args.run/'shadow_capture.json'
        if capture_path.exists() and 'shadow_ground_contacts' in a:
            semantics=json.loads(capture_path.read_text())['ground_contact_semantics']
            nonfoot=[i for i,b in enumerate(semantics['body_order']) if b not in semantics['foot_order']]
            nonfoot_contact=(np.linalg.norm(a['shadow_ground_contacts'][:,:,nonfoot],axis=-1)>1).any(-1)
            basegroups['nonfoot_contact']=nonfoot_contact[sl]
        for level in np.unique(a['control_terrain_level']):basegroups['level_'+str(level)]=np.repeat(a['control_terrain_level'],4,axis=0)[sl]==level
        for name,model in models.items():
            for quantity,axes in [('force','xyz'),('wrench',('Fx','Fy','Fz','Mx','My','Mz'))]:
                err=model[quantity]-truth[cadence][quantity]
                groups=dict(basegroups)
                if quantity=='force':
                    groups={k:np.broadcast_to(v[...,None],err.shape[:-1]) for k,v in groups.items()}
                    if cadence=='control':
                        contact=a['contact_truth'].reshape(-1,4,e,4)[:-1]
                        groups.update(stance=contact.all(1),swing=(~contact).all(1),
                            transition=(contact.any(1)!=contact.all(1))|trans[sl])
                    else:groups.update(stance=a['contact_truth'][sl],swing=~a['contact_truth'][sl],transition=trans[sl])
                else:groups['transition']=trans[sl].any(-1)
                validrows=valid&model['valid']
                for group,mask in groups.items():
                    mask=mask&(validrows[...,None] if quantity=='force' else validrows)
                    for i,axis in enumerate(axes):
                        rows.append(dict(cadence=cadence,model=name,quantity=quantity,axis=axis,group=group,**stats(err[...,i][mask])))
                        bound=args.moment_residual_bound_nm if quantity=='wrench' and i>=3 else args.force_residual_bound_n
                        if bound is not None:
                            values=err[...,i][mask];values=values[np.isfinite(values)]
                            exceed.append(dict(cadence=cadence,model=name,quantity=quantity,axis=axis,group=group,
                                bound=bound,n=int(values.size),exceeded=int((abs(values)>bound).sum()),
                                fraction=float((abs(values)>bound).mean()) if values.size else None))
                for i,axis in enumerate(axes):
                    lags.append(dict(cadence=cadence,model=name,quantity=quantity,axis=axis,
                        best_lag_s=lag(model[quantity][...,i],truth[cadence][quantity][...,i],validrows,dt)))
                arrays[cadence+'_'+name+'_'+quantity]=model[quantity]
                arrays[cadence+'_'+name+'_'+quantity+'_required_residual']= -err
            if 'regularized_condition' in model:
                for key in ('regularized_condition','rank','joint_residual','base_residual'):
                    health.append(dict(cadence=cadence,model=name,metric=key,**stats(model[key][validrows])))
                    arrays[name+'_'+key]=model[key]
            arrays[cadence+'_'+name+'_valid']=validrows
        # Training force supervision uses deadbanded/clipped per-substep mean,
        # separately from raw physical contact accuracy reported above.
        if cadence=='control':
            label=mean(a['shadow_force_label'])
            for name,model in models.items():
                for i,axis in enumerate('xyz'):
                    rows.append(dict(cadence=cadence,model=name,quantity='force_training_label',axis=axis,group='all',
                        **stats((model['force']-label)[...,i][valid&model['valid']])))
    def csvwrite(name,values):
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    csvwrite('errors.csv',rows);csvwrite('lag.csv',lags);csvwrite('dynamics_health.csv',health)
    if exceed:csvwrite('residual_bound_exceedance.csv',exceed)
    # Independent dynamics check using measured contact/external labels, rather
    # than the algebraic residual minimized by each estimator.
    td=lambda x:torch.as_tensor(x,dtype=torch.float64)
    mass=td(a['shadow_mass_matrix']);bias=td(a['shadow_bias'])
    jt=td(a['shadow_foot_jacobians']).flatten(-3,-2).transpose(-1,-2)
    bt=td(a['shadow_base_jacobian']).transpose(-1,-2)
    torque=td(a['shadow_applied_torque_readback']);tau=torch.zeros_like(bias);tau[...,6:]=torque
    dyn=[]
    for name,f,w in [('truth',a['shadow_force_truth'],a['shadow_applied_wrench']),
                     ('neural',a['pred_force'],a.get('shadow_predicted_wrench',a['predicted_wrench']))]:
        force=(jt@td(f).flatten(-2)[...,None]).squeeze(-1)+(bt@td(w)[...,None]).squeeze(-1)+tau
        residual=(mass@td(arrays['physics_acceleration'])[...,None]).squeeze(-1)+bias-force
        arrays[name+'_independent_dynamics_residual']=residual.numpy()
        for j in range(18):dyn.append(dict(model=name,metric='generalized_residual',axis=j,**stats(residual.numpy()[...,j][good])))
        predicted=torch.linalg.solve(mass,(force-bias)[...,None]).squeeze(-1)
        q=td(a['orientation_xyzw']);v=td(a['shadow_velocity_before'])
        from scripts.hard_pact_shadow_estimation import inverse_rotate
        transport=torch.cross(inverse_rotate(q,v[...,3:6]),inverse_rotate(q,v[...,:3]),dim=-1)
        dp=torch.cat((rotate(q,predicted[...,:3]+transport),rotate(q,predicted[...,3:6])),-1)*td(a['shadow_dt'])
        err=dp.numpy()-(a['shadow_velocity_after']-a['shadow_velocity_before'])[...,:6]
        arrays[name+'_world_motion_error']=err
        for j in range(6):dyn.append(dict(model=name,metric='world_velocity_increment_error',axis=j,**stats(err[...,j][good])))
    csvwrite('independent_dynamics.csv',dyn)
    # Compact traces omit bulky mechanics but keep aligned states, labels and all estimates.
    arrays.update({k:v for k,v in a.items() if not any(s in k for s in ('mass_matrix','jacobian','assembled_','objective_'))})
    compact=lambda values:{k:v.astype(np.float32) if v.dtype==np.float64 else v for k,v in values.items()}
    np.savez_compressed(out/'traces.npz',**compact(arrays))
    events=[]
    for t,env in np.argwhere(failure)[:24]:events.append(('failure',int((t+1)*4-1),int(env),None))
    changes=np.argwhere(a['contact_truth'][1:]!=a['contact_truth'][:-1])
    if len(changes):
        for idx in np.linspace(0,len(changes)-1,min(12,len(changes)),dtype=int):
            t,env,foot=changes[idx];events.append(('contact_transition',int(t+1),int(env),int(foot)))
    for env in range(min(6,e)):
        eligible=[t for t in range(100,n-50,100) if good[t-100:t+1,env].all()]
        if eligible:events.append(('healthy',eligible[len(eligible)//2],env,None))
    captures={};index=[]
    for idx,(kind,t,env,foot) in enumerate(events):
        lo=max(0,t-(50 if kind=='contact_transition' else 99));hi=min(n,t+(51 if kind=='contact_transition' else 1))
        if kind=='failure':
            previous=np.flatnonzero(a['control_reset'][:t//4,env])
            if len(previous):lo=max(lo,4*(int(previous[-1])+1))
        index.append(dict(id=idx,kind=kind,substep=t,env=env,foot=foot,start=lo,end_exclusive=hi))
        for key,value in arrays.items():
            if value.shape[:2]==(n,e):captures[f'{idx:02d}_{key}']=value[lo:hi,env]
        captures[f'{idx:02d}_valid']=good[lo:hi,env]
    np.savez_compressed(out/'bounded_histories.npz',**compact(captures))
    (out/'history_index.json').write_text(json.dumps(index,indent=2))
    metadata=dict(damping=args.damping,checkpoint=m['checkpoint'],checkpoint_sha256=m['checkpoint_sha256'],
        code_sha=m['code_sha'],source=str(args.run),offline_seconds=time.monotonic()-started,
        valid_physics_rows=int(good.sum()),valid_control_rows=int(cgood.sum()),
        failure_events=int(failure.sum()),failure_capture_drops=max(0,int(failure.sum())-24),
        contact_transition_events=int(len(changes)),transition_capture_drops=max(0,len(changes)-12),
        force_residual_bound_n=args.force_residual_bound_n,moment_residual_bound_nm=args.moment_residual_bound_nm,
        base_jacobian_actuated_columns_max=float(abs(a['shadow_base_jacobian'][...,6:]).max()),
        latency='physics: available at end of 5ms interval; deployment: end of 20ms interval, 20ms estimated linear-velocity update',
        semantics='world forces; world moments about base-link origin. Body-twist acceleration includes -omega cross v. Full nominal floating-base/leg coupling.',
        limitations=['Control input uses estimated linear velocity and measured gyro/joint rates; orientation/geometry remain the existing QP inputs, not an independent hardware state-estimator validation.',
        'Applied actuator torque excludes unmodeled passive friction/damping and armature. Nominal mass/CoM mismatch remains in residual, not ground-truth wrench.',
        'Contact sensor impulses/forces and continuous nominal mechanics differ at impact; start-of-interval mechanics are frozen over each backward difference.',
        'Control intervals crossing an instantaneous boundary velocity jump >1e-4 are excluded; physics intervals with position jumps >0.1m are excluded.',
        'Small solved base residual is algebraic self-consistency, NOT independent wrench accuracy.',
        'Scalar residual bounds are reported only when explicitly requested; companion momentum reporting supplies separate per-axis base-frame bounds.',
        'Lag search is descriptive pooled +/-5 samples; correlated observations, single seed, not a significance test.'],
        torque_readback_error=stats(a['shadow_applied_torque_readback']-a['executed_torque']))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    (out/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for quantity in ('force','wrench'):
        selected=[r for r in rows if r['group']=='all' and r['quantity']==quantity]
        models=list(dict.fromkeys((r['cadence'],r['model']) for r in selected))
        fig,axes=plt.subplots(1,2 if quantity=='wrench' else 1,figsize=(12,5));axes=np.atleast_1d(axes);labels=[];values=[]
        for cadence,model in models:
            labels.append(cadence+'/'+model)
        for ax,unit,prefix in zip(axes,('N','Nm'),('F','M')):
            values=[np.mean([r['rmse'] for r in selected if r['cadence']==cadence and r['model']==model and (quantity=='force' or r['axis'].startswith(prefix))]) for cadence,model in models]
            ax.bar(labels,values);ax.tick_params(axis='x',labelrotation=80);ax.set_ylabel('mean axis RMSE ('+unit+')')
        fig.tight_layout();fig.savefig(out/(quantity+'_rmse.png'));plt.close(fig)
    # First three failures and two healthy controls, bounded half-second windows.
    events=list(np.argwhere(failure)[:3])+[(min(100,n//4-2),0),(min(200,n//4-2),1)]
    for idx,(t,env) in enumerate(events):
        lo=max(0,int(t)*4-100);hi=min(n,(int(t)+1)*4);x=np.arange(lo,hi)*.005
        fig,axs=plt.subplots(3,1,figsize=(9,7),sharex=True)
        axs[0].plot(x,a['shadow_force_truth'][lo:hi,env,:,2].sum(-1),label='truth total Fz')
        axs[0].plot(x,streams['physics_predicted']['force'][lo:hi,env,:,2].sum(-1),label='oracle-input/predicted gate')
        axs[0].plot(x,a['pred_force'][lo:hi,env,:,2].sum(-1),label='neural')
        axs[1].plot(x,a['shadow_applied_wrench'][lo:hi,env,2],label='truth wrench Fz')
        axs[1].plot(x,streams['physics_predicted']['wrench'][lo:hi,env,2],label='analytical Fz')
        axs[2].plot(x,a['correction'][lo:hi,env].std(-1),label='correction joint std Nm')
        for ax in axs:ax.legend();ax.grid(alpha=.2)
        axs[-1].set_xlabel('simulation seconds');fig.tight_layout();fig.savefig(out/f'event_{idx}.png');plt.close(fig)
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
