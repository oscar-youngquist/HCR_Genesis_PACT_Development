"""Offline whole-body momentum validation/observer replay on bounded captures."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from scripts.hard_pact_shadow_estimation import inverse_rotate,rotate
from scripts.hard_pact_momentum_observer import MomentumObserver,DEFAULT_GAINS,contact_input,base_origin_wrench
from scripts.report_hard_pact_shadow import calculate,stats,valid_intervals


def nominal_kinematics(a,urdf):
    """Independent Pinocchio link sum, compared with captured BARD M*v.

    Every moving body's linear and spin/orbital angular momentum is included.
    Pinocchio merges fixed children; BARD retains them. No torso-only inertia.
    """
    import pinocchio as pin
    from legged_gym.dynamics.bard_go2_dynamics import SIMULATOR_JOINT_ORDER
    model=pin.buildModelFromUrdf(str(urdf),pin.JointModelFreeFlyer());data=model.createData()
    names=list(SIMULATOR_JOINT_ORDER);order=[names.index(str(n)) for n in model.names[2:]]
    foot_ids=[model.getFrameId(n) for n in ('FR_foot','FL_foot','RR_foot','RL_foot')]
    qraw=a['shadow_configuration'];world=a['shadow_velocity_before'];shape=qraw.shape[:2]
    com=np.empty(shape+(3,));feet=np.empty(shape+(4,3));link_h=np.empty(shape+(6,));centroidal=np.empty_like(link_h)
    masses=np.array([x.mass for x in model.inertias]);mass=masses.sum()
    qxy=torch.as_tensor(qraw[...,3:7],dtype=torch.float64)
    vw=torch.as_tensor(world,dtype=torch.float64)
    vbody=torch.cat((inverse_rotate(qxy,vw[...,:3]),inverse_rotate(qxy,vw[...,3:6]),vw[...,6:]),-1).numpy()
    for t in range(shape[0]):
        for e in range(shape[1]):
            q=np.r_[qraw[t,e,:7],qraw[t,e,7:][order]].astype(float)
            v=np.r_[vbody[t,e,:6],vbody[t,e,6:][order]]
            pin.forwardKinematics(model,data,q,v);pin.updateFramePlacements(model,data)
            positions=[];momenta=[];spins=[]
            for j in range(1,model.njoints):
                inertia=model.inertias[j];pose=data.oMi[j];twist=data.v[j]
                positions.append(pose.translation+pose.rotation@inertia.lever)
                momenta.append(inertia.mass*(pose.rotation@(twist.linear+np.cross(twist.angular,inertia.lever))))
                spins.append(pose.rotation@(inertia.inertia@twist.angular))
            positions=np.array(positions);momenta=np.array(momenta)
            c=(positions*masses[1:,None]).sum(0)/mass;com[t,e]=c
            link_h[t,e]=np.r_[momenta.sum(0),(np.array(spins)+np.cross(positions-c,momenta)).sum(0)]
            feet[t,e]=np.array([data.oMf[f].translation for f in foot_ids])
            pin.computeCentroidalMomentum(model,data,q,v)
            centroidal[t,e]=np.r_[data.hg.linear,data.hg.angular]
    generalized=np.einsum('teij,tej->tei',a['shadow_mass_matrix'],vbody)[...,:6]
    linear=rotate(qxy,torch.as_tensor(generalized[...,:3])).numpy()
    angular=rotate(qxy,torch.as_tensor(generalized[...,3:])).numpy()-np.cross(com-qraw[...,:3],linear)
    h=np.concatenate((linear,angular),-1)
    estimated=vbody.copy();estimated[...,:3]=a['shadow_estimated_linear_body']
    gm=np.einsum('teij,tej->tei',a['shadow_mass_matrix'],estimated)[...,:6]
    el=rotate(qxy,torch.as_tensor(gm[...,:3])).numpy()
    ea=rotate(qxy,torch.as_tensor(gm[...,3:])).numpy()-np.cross(com-qraw[...,:3],el)
    return dict(momentum=h,estimated_momentum=np.concatenate((el,ea),-1),com=com,feet=feet,mass=mass,
        bard_vs_link_sum=h-link_h,pin_centroidal_vs_link_sum=centroidal-link_h)


def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def property_validation(readback,urdf):
    """Aggregate fixed URDF children into each actual PhysX body for readback checks."""
    import xml.etree.ElementTree as ET
    import pinocchio as pin
    root=ET.parse(urdf).getroot();links={x.attrib['name']:x for x in root.findall('link')}
    parents={j.find('child').attrib['link']:j for j in root.findall('joint')}
    names=readback['body_names'];parts={n:[] for n in names}
    def pose(origin):
        xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ');rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
        return pin.SE3(pin.rpy.rpyToMatrix(rpy),xyz)
    for name,link in links.items():
        inertial=link.find('inertial')
        if inertial is None:continue
        node=name;transform=pin.SE3.Identity()
        while node not in parts:
            joint=parents[node];transform=pose(joint.find('origin'))*transform;node=joint.find('parent').attrib['link']
        inertia_pose=transform*pose(inertial.find('origin'));m=float(inertial.find('mass').get('value'))
        x=inertial.find('inertia');I=np.array([[float(x.get('i'+a+b,x.get('i'+b+a,'0'))) for b in 'xyz'] for a in 'xyz'])
        parts[node].append((m,inertia_pose.translation,inertia_pose.rotation@I@inertia_pose.rotation.T))
    expected=[]
    for n in names:
        p=parts[n];mass=sum(x[0] for x in p);c=sum(x[0]*x[1] for x in p)/mass
        I=sum(x[2]+x[0]*(np.dot(x[1]-c,x[1]-c)*np.eye(3)-np.outer(x[1]-c,x[1]-c)) for x in p)
        expected.append((mass,c,I))
    return dict(mass_max_abs_kg=float(abs(np.asarray(readback['masses'])-np.array([x[0] for x in expected])).max()),
        com_max_abs_m=float(abs(np.asarray(readback['coms'])[...,:3]-np.array([x[1] for x in expected])).max()),
        inertia_eigenvalue_max_abs_kg_m2=float(abs(np.linalg.eigvalsh(np.asarray(readback['inertias']).reshape(-1,len(names),3,3))-
            np.linalg.eigvalsh(np.array([x[2] for x in expected]))).max()),
        note='Fixed-child aggregation; inertia eigenvalues avoid assuming the PhysX inertia-frame rotation convention.')


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gains',nargs=6,type=float,default=DEFAULT_GAINS)
    p.add_argument('--multipliers',nargs='+',type=float,default=[.5,1.,2.]);args=p.parse_args()
    torch.set_num_threads(1);out=args.output;out.mkdir(parents=True,exist_ok=True)
    a=dict(np.load(args.run/'trace.npz'));m=json.loads((args.run/'manifest.json').read_text())
    start=time.perf_counter();kin=nominal_kinematics(a,Path(__file__).resolve().parents[1]/'resources/robots/go2/urdf/go2.urdf')
    kinematics_s=time.perf_counter()-start
    _,fd,good,_=calculate(a,1e-4);T,E=good.shape;dt=float(a['shadow_dt'][0,0,0])
    truth=a['shadow_applied_wrench'];q=a['orientation_xyzw'];base=a['shadow_position_before']
    delivered=a.get('shadow_composer_wrench')
    delivery_source='Runtime composer buffer, rotated to world axes about link origin'
    if delivered is None and not m.get('momentum_sanity_correct_wrench',False):
        tape=np.load(args.run/'tapes.npz')['heavy' if m.get('load')=='heavier' else 'light']
        delivered=truth.copy();delivered[...,3:]-=np.repeat(tape[...,3:],4,axis=0)[:len(truth)]
        delivery_source='INFERRED from audited installed composer overwrite kernel and tape; not per-step readback'
    boundary=np.zeros((T,E),bool);boundary[0]=True
    boundary[1:]=np.max(abs(a['shadow_velocity_before'][1:]-a['shadow_velocity_after'][:-1]),axis=-1)>1e-4
    boundary[1:] |= np.linalg.norm(base[1:]-a['shadow_position_after'][:-1],axis=-1)>.01
    for t in range(1,T):
        if t%4==0:boundary[t]|=a['control_reset'][t//4-1].astype(bool)
    rows=[];exceed=[];traces={};timing=[];lags=[];delivery_errors=[]
    failure=a['control_reset'].astype(bool)&a['control_failure'].astype(bool)
    prefail=np.zeros_like(good)
    for t,e in np.argwhere(failure):
        prev=np.flatnonzero(a['control_reset'][:t,e]);lo=max(0,t-25,int(prev[-1])+1 if len(prev) else 0)
        prefail[lo*4:(t+1)*4,e]=True
    transition=np.zeros_like(good)
    for lag in range(1,5):transition[lag:]|=(a['contact_truth'][lag:]!=a['contact_truth'][:-lag]).any(-1)
    active=np.linalg.norm(truth,axis=-1)>1e-6
    groups=dict(all=np.ones_like(good),active=active,inactive=~active,transition=transition,pre_failure=prefail)
    if m.get('momentum_sanity'):
        groups['zero_load_env0']=np.broadcast_to(np.arange(E)==0,good.shape)
    for family in {r['family'] for r in m['target_plan']}:
        groups['terrain_'+family]=np.broadcast_to(np.array([r['family']==family for r in m['target_plan']]),good.shape)
    axes=('Fx','Fy','Fz','Mx','My','Mz');bounds=np.array([30,30,40,10,15,10])
    def assess(name,pred,valid,indices,phase):
        # Same availability timestamp and base origin for both estimates and truth.
        if delivered is not None:
            mask=valid&good[indices];error=pred-delivered[indices]
            qr=torch.as_tensor(q[indices],dtype=torch.float64);er=torch.as_tensor(error,dtype=torch.float64)
            error=torch.cat((inverse_rotate(qr,er[...,:3]),inverse_rotate(qr,er[...,3:])),dim=-1).numpy()
            for j,axis in enumerate(axes):
                delivery_errors.append(dict(model=name,phase=phase,axis=axis,frame='body',target=delivery_source,**stats(error[...,j][mask])))
        if phase in ('available','available_after_burnin'):
            selected=np.arange(T)[indices];step=dt*(selected[1]-selected[0]);mask=valid&good[indices]
            target=truth[indices]
            for j,axis in enumerate(axes):
                choices=[]
                for shift in range(-int(.15/step),int(.15/step)+1):
                    lo=max(0,shift);end=len(pred)+min(0,shift)
                    clean=mask[lo:end]&mask[lo-shift:end-shift]
                    for offset in range(1,abs(shift)):
                        clean &= mask[offset:len(pred)-abs(shift)+offset]
                    x=pred[lo:end,:,j][clean];y=target[lo-shift:end-shift,:,j][clean]
                    finite=np.isfinite(x)&np.isfinite(y);x=x[finite];y=y[finite]
                    corr=float(np.corrcoef(x,y)[0,1]) if len(x)>20 and np.std(x)>1e-8 and np.std(y)>1e-8 else None
                    if corr is not None and np.isfinite(corr):choices.append((corr,shift*step,len(x)))
                best=max(choices) if choices else (None,None,0)
                lags.append(dict(model=name,axis=axis,lag_s=best[1],correlation=best[0],n=best[2],
                    interpretation='Exploratory pooled correlation in +/-150ms; positive means estimate lags truth, not calibrated latency.'))
        for group,selection in groups.items():
            mask=valid&good[indices]&selection[indices]
            for frame in ('world','body'):
                error=pred-truth[indices]
                if frame=='body':
                    qr=torch.as_tensor(q[indices],dtype=torch.float64);er=torch.as_tensor(error,dtype=torch.float64)
                    error=torch.cat((inverse_rotate(qr,er[...,:3]),inverse_rotate(qr,er[...,3:])),dim=-1).numpy()
                for j,axis in enumerate(axes):
                    rows.append(dict(model=name,phase=phase,frame=frame,group=group,axis=axis,**stats(error[...,j][mask])))
                    if frame=='body':
                        values=error[...,j][mask];values=values[np.isfinite(values)]
                        exceed.append(dict(model=name,phase=phase,group=group,axis=axis,bound=bounds[j],n=int(len(values)),
                            fraction=float((abs(values)>bounds[j]).mean()) if len(values) else None))
    assess('neural',a['shadow_predicted_wrench'],np.ones_like(good),slice(None),'available')
    # FD is retrospective: interval i finishes at i+1, not at its start.
    for label in ('physics_predicted','physics_ungated','physics_oracle_gate'):
        pred=fd[label]['wrench'][:-1].copy()
        pred[...,3:]+=np.cross(base[:-1]-base[1:],pred[...,:3])
        assess('FD_'+label,pred,fd[label]['valid'][:-1]&~boundary[1:],slice(1,None),'available')
        assess('FD_'+label,fd[label]['wrench'],fd[label]['valid'],slice(None),'retrospective_interval')
    control_ix=np.arange(4,T,4)
    cp=fd['control_predicted']['wrench'].copy()
    cp[...,3:]+=np.cross(base[control_ix-4]-base[control_ix],cp[...,:3])
    assess('FD_control_predicted',cp,fd['control_predicted']['valid']&~boundary[control_ix],control_ix,'available')
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float64)
    for cadence,stride,h in [('oracle_5ms',1,kin['momentum']),('estimated_20ms',4,kin['estimated_momentum'])]:
        ix=np.arange(0,T,stride);N=len(ix)
        for source in ('measured','neural','analytical','all_contacts_measured'):
            force={'measured':a['shadow_force_truth'],'neural':a['pred_force'],
                   'analytical':fd['physics_predicted']['force'],'all_contacts_measured':a['shadow_force_truth']}[source]
            if source=='analytical':
                force=np.where(fd['physics_predicted']['valid'][...,None,None],force,np.nan)
            if stride==4 and source=='analytical':
                control_force=np.where(fd['control_predicted']['valid'][...,None,None],fd['control_predicted']['force'],np.nan)
                force=np.concatenate((np.repeat(control_force,4,axis=0),np.full((4,E,4,3),np.nan)))
            # Complete-interval inputs: forces already measured at completed physics step.
            u=contact_input(kin['mass'],tensor(kin['com']),tensor(kin['feet']),tensor(force)).numpy()
            if source=='all_contacts_measured':
                total=a['shadow_ground_contacts'].sum(-2)
                moment=a['shadow_ground_moment_world_origin'].sum(-2)-np.cross(kin['com'],total)
                u=np.concatenate((total+np.array([0,0,-kin['mass']*9.81]),moment),-1)
            for mult in args.multipliers:
                observer=MomentumObserver(E,np.array(args.gains)*mult)
                estimates=[];available=[];burnin=[];initial=[];beg=time.perf_counter();update_seconds=0.
                for k,t in enumerate(ix):
                    lo=ix[k-1] if k else t
                    reset=boundary[lo+1:t+1].any(0) if k else np.ones(E,bool)
                    inp=u[lo:t].mean(0) if k else np.zeros((E,6))
                    ht=tensor(h[t]);ut=tensor(inp);rt=torch.as_tensor(reset);tic=time.perf_counter()
                    r=observer.update(ht,ut,t*dt,rt)
                    if k>=20:update_seconds+=time.perf_counter()-tic
                    estimates.append(base_origin_wrench(r['wrench'],tensor(kin['com'][t]),tensor(base[t])).numpy())
                    available.append(r['available'].numpy());burnin.append(r['burnin'].numpy());initial.append(r['initialized'].numpy())
                elapsed=time.perf_counter()-beg
                name=f'{cadence}_{source}_gain{mult:g}';pred=np.array(estimates);valid=np.array(available);burn=np.array(burnin)
                timing.append(dict(model=name,seconds=elapsed,batch_updates=N,us_per_env_update=elapsed/(N*E)*1e6,
                    warmed_update_seconds=update_seconds,warmed_update_us_per_env=update_seconds/(max(1,N-20)*E)*1e6,
                    scope='CPU float64 batched observer only; loop total additionally includes input reduction/frame conversion. Not hardware/GPU latency.'))
                assess(name,pred,valid&~burn,ix,'available_after_burnin')
                assess(name,pred,valid&burn,ix,'burnin')
                # Compare held current output with the following sample, origin-shifted.
                forecast=pred[:-1].copy();forecast[...,3:]+=np.cross(base[ix[:-1]]-base[ix[1:]],forecast[...,:3])
                assess(name,forecast,valid[:-1]&~burn[:-1]&~np.array(initial)[1:],ix[1:],'one_sample_ahead')
                traces[name]=pred.astype(np.float32);traces[name+'_valid']=valid;traces[name+'_burnin']=burn
                traces[name+'_initialized']=np.array(initial)
    # Like-for-like table at actual control-rate availability, not pooled
    # statistics with different burn-in/missing-input populations.
    common=good[control_ix]&fd['control_predicted']['valid']&~boundary[control_ix]
    matched={'neural':a['shadow_predicted_wrench'][control_ix],'FD_control_predicted':cp}
    for cadence in ('oracle_5ms','estimated_20ms'):
        for source in ('measured','neural','analytical','all_contacts_measured'):
            name=f'{cadence}_{source}_gain1'
            if name not in traces:continue
            ix=control_ix if cadence=='oracle_5ms' else np.arange(1,len(control_ix)+1)
            common &= traces[name+'_valid'][ix]&~traces[name+'_burnin'][ix]
            matched[name]=traces[name][ix]
    for name,pred in matched.items():assess(name,pred,common,control_ix,'common_20ms_after_burnin')
    write_csv(out/'errors.csv',rows);write_csv(out/'wrench_bound_exceedance.csv',exceed);write_csv(out/'observer_timing.csv',timing)
    write_csv(out/'lag.csv',lags)
    if delivery_errors:write_csv(out/'composer_delivery_errors.csv',delivery_errors)
    if m.get('momentum_sanity'):
        # Observable response, not a fitted sensor latency: report bias/noise too.
        responses=[]
        for name,pred in [('neural',a['shadow_predicted_wrench'])]+[(k,v) for k,v in traces.items() if v.ndim==3]:
            stride=T//len(pred);tt=np.arange(len(pred))*dt*stride
            for axis in range(6):
                e=axis+1;before=(tt>=1.5)&(tt<2.);after=(tt>=2.)&(tt<3.)
                steady=(tt>=3.)&(tt<3.8);target=5. if axis<3 else .5
                baseline=np.nanmean(pred[before,e,axis]);values=pred[after,e,axis]-baseline
                crossing=np.flatnonzero(values>=.63212056*target)
                responses.append(dict(model=name,axis=axes[axis],baseline_bias=float(baseline),
                    baseline_std=float(np.nanstd(pred[before,e,axis])),
                    first_63pct_s=float(tt[after][crossing[0]]-2.) if len(crossing) else None,
                    steady_mean=float(np.nanmean(pred[steady,e,axis])),target=target,
                    note='First crossing after subtracting measured pre-step bias; noisy crossings are not proof of correct response.'))
        write_csv(out/'step_response.csv',responses)
    # Force residual bounds also evaluated in the actual base frame.
    grf=[]
    for name,force in [('neural',a['pred_force']),('analytical',fd['physics_predicted']['force'])]:
        qr=tensor(q)[:,:,None,:].expand(-1,-1,4,-1)
        er=inverse_rotate(qr,tensor(force-a['shadow_force_truth'])).numpy()
        for j,b in enumerate([45,45,85]):
            x=er[...,j][good];x=x[np.isfinite(x)]
            grf.append(dict(model=name,axis='xyz'[j],bound=b,fraction=float((abs(x)>b).mean()),**stats(x)))
    write_csv(out/'grf_bounds.csv',grf)
    validation={key:[stats(kin[key][...,j]) for j in range(6)] for key in ('bard_vs_link_sum','pin_centroidal_vs_link_sum')}
    validation['matched_available_20ms_rows']=int(common.sum())
    validation['delivery_source']=delivery_source
    if delivered is not None:
        validation['label_minus_composer_delivery']=[stats((truth-delivered)[...,j][good]) for j in range(6)]
    capture=json.loads((args.run/'shadow_capture.json').read_text())
    body_names=capture['ground_contact_semantics']['body_order'];nonfoot=[i for i,n in enumerate(body_names) if n not in ('FR_foot','FL_foot','RR_foot','RL_foot')]
    nonfoot_force=a['shadow_ground_contacts'][...,nonfoot,:].sum(-2)
    point_moment=np.cross(kin['feet'],a['shadow_force_truth']).sum(-2)
    omitted_moment=a['shadow_ground_moment_world_origin'].sum(-2)-point_moment-np.cross(base,nonfoot_force)
    validation['omitted_nonfoot_force_world']=[stats(nonfoot_force[...,j][good]) for j in range(3)]
    validation['omitted_contact_moment_base_origin_world_axes']=[stats(omitted_moment[...,j][good]) for j in range(3)]
    validation['contact_note']='Moment difference includes contact point/patch offsets plus non-foot contacts, about base origin. Self-contact availability is not asserted.'
    readback=m.get('nominal_property_readback',{})
    validation.update(mass_kg=float(kin['mass']),kinematics_seconds=kinematics_s,
        realized_mass_sum=np.array(readback['masses']).sum(-1).tolist() if readback.get('masses') else None,
        realized_armature_max=float(abs(np.array(readback['armatures'])).max()) if readback.get('armatures') else None,
        realized_damping_max=float(abs(np.array(readback['damping'])).max()) if readback.get('damping') else None,
        realized_friction_max=float(abs(np.array(readback['friction'])).max()) if readback.get('friction') else None)
    if readback:
        validation['urdf_property_comparison']=property_validation(readback,Path(__file__).resolve().parents[1]/'resources/robots/go2/urdf/go2.urdf')
    (out/'validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False))
    traces.update({k:v.astype(np.float32) if isinstance(v,np.ndarray) else np.asarray(v) for k,v in kin.items()})
    traces.update(truth=truth,neural=a['shadow_predicted_wrench'],reset=boundary,valid=good,
                  time=np.arange(T)*dt,orientation=q,base=base,contact=a['contact_truth'])
    np.savez_compressed(out/'observer_traces.npz',**traces)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(12,8),sharex=True);env=1
    for j,ax in enumerate(axs.flat):
        ax.plot(np.arange(T)*dt,truth[:,env,j],label='applied truth')
        ax.plot(np.arange(T)*dt,a['shadow_predicted_wrench'][:,env,j],label='neural',alpha=.7)
        for source in ('measured','neural','analytical'):
            key=f'oracle_5ms_{source}_gain1';ax.plot(np.arange(T)*dt,traces[key][:,env,j],label=source,alpha=.7)
        ax.set_ylabel(axes[j]);ax.grid(alpha=.2)
    axs[0,0].legend();fig.tight_layout();fig.savefig(out/'wrench_comparison.png');plt.close(fig)
    events=[((int(t)+1)*4,int(e),'failure') for t,e in np.argwhere(failure)[:3]]
    events.append((min(T-1,600),0,'healthy_reference'))
    for number,(end,e,label) in enumerate(events):
        lo=max(0,end-100);hi=min(T,end if label=='failure' else end+20);tt=(np.arange(lo,hi)-end)*dt
        fig,axs=plt.subplots(4,1,figsize=(10,9),sharex=True)
        axs[0].plot(tt,a['control_tilt'][np.arange(lo,hi)//4,e,:2]);axs[0].set_ylabel('roll/pitch [rad]')
        axs[1].plot(tt,np.linalg.norm(truth[lo:hi,e,:3],axis=-1),label='original label |F|')
        axs[2].plot(tt,np.linalg.norm(truth[lo:hi,e,3:],axis=-1),label='original label |M|')
        for source in ('measured','neural','analytical'):
            key=f'oracle_5ms_{source}_gain1'
            axs[1].plot(tt,np.linalg.norm(traces[key][lo:hi,e,:3],axis=-1),label=source)
            axs[2].plot(tt,np.linalg.norm(traces[key][lo:hi,e,3:],axis=-1),label=source)
        axs[1].set_ylabel('force [N]');axs[2].set_ylabel('moment [Nm]')
        axs[3].plot(tt,a['contact_truth'][lo:hi,e].sum(-1),label='measured contacts')
        axs[3].plot(tt,a['qp_stance_mask'][lo:hi,e].sum(-1),label='QP contacts')
        for ax in axs:ax.axvline(0,color='k',ls=':');ax.grid(alpha=.2)
        axs[1].legend(fontsize=8);axs[3].legend(fontsize=8);axs[3].set_xlabel('seconds relative to event')
        fig.suptitle(f'{label}, env {e}; failure curves stop before reset; raw pre-event evidence')
        fig.tight_layout();fig.savefig(out/f'event_{number}.png');plt.close(fig)
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
