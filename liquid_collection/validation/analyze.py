"""Offline same-trajectory diagnostics; independent compact and native objects."""
import argparse
import json
from pathlib import Path
import time
import shutil
import sys
import shlex
import numpy as np
import torch
from liquid_collection.audit import stream_rows, audit, interior_mask
from liquid_collection.storage import atomic_json, digest
from .replay import replay, accounting, META
from .reconstruction import Reconstruction, tensor


def segments(path,env):
    rows=sorted(list(stream_rows(path))+list(stream_rows(Path(path)/'boundaries')),
                key=lambda r:(float(r['time_s']),int(r['record_sequence'])))
    events=list(stream_rows(Path(path)/'events')) or rows
    def single(r): return {k:(v if k in META else v[env]) for k,v in r.items()}
    result=[]
    for episode in sorted(set(int(r['episode_id'][env]) for r in rows)):
        rr=[single(r) for r in rows if r['episode_id'][env]==episode]
        unique={float(r['time_s']):r for r in rr}
        # Support is sticky until reset, so each eligible episode has one contiguous prefix.
        states=[r for r in unique.values() if all(bool(r[k]) for k in ('state_valid','held_valid','episode_supported','episode_selected'))]
        if len(states)<3: continue
        start,end=states[0]['time_s'],states[-1]['time_s']
        ints=[r for r in rr if r['interval_duration_s']>0 and start<=r['time_s']<=end]
        ev=[single(r) for r in events if r['episode_id'][env]==episode and r['time_s']<=end]
        ev=list({float(r['time_s']):r for r in ev}.values())
        if len(ints)>1 and ev: result.append((episode,states,ev,ints))
    return result


def stats(error,scale,reference=None,times=None,estimate=None):
    e=np.asarray(error); n=np.linalg.norm(e/np.asarray(scale),axis=-1)
    out=dict(count=len(e),bias=e.mean(0).tolist(),rmse=float(np.sqrt(np.mean(e*e))),
        p95=float(np.percentile(np.linalg.norm(e,axis=-1),95)),max=float(np.linalg.norm(e,axis=-1).max()),
        normalized_rmse=float(np.sqrt(np.mean(n*n))), normalized_p95=float(np.percentile(n,95)))
    if reference is not None:
        ref=np.asarray(reference); dynamic=float(np.sqrt(np.mean(np.sum((ref-ref.mean(0))**2,axis=-1))))
        out['changing_component_rms']=dynamic
        out['changing_component_relative_rmse']=float(np.sqrt(np.mean(np.sum(e*e,axis=-1)))/dynamic) if dynamic>1e-10 else None
    if times is not None and estimate is not None:
        a=np.linalg.norm(estimate,axis=-1); b=np.linalg.norm(reference,axis=-1)
        out.update(peak_amplitude_error=float(a.max()-b.max()),peak_timing_error_s=float(times[a.argmax()]-times[b.argmax()]))
    return out



def native_velocity_reference(intervals,t,pose_reference):
    """Sensitivity convention: native endpoint velocities and interval accelerations.

    This is an independent discrete reference, not instantaneous truth. It avoids
    taking two derivatives of float32 native positions at 0.25 ms spacing.
    """
    index=np.searchsorted([r['time_s'] for r in intervals],t).clip(0,len(intervals)-1)
    velocity=tensor([intervals[i]['velocity_W'] for i in index])
    accel=tensor([intervals[i]['base_acceleration_interval_average_W'] for i in index])
    from .reconstruction import matvec,cross
    Rt=pose_reference['R'].transpose(-1,-2)
    uB=matvec(Rt,velocity[:,:3]); omega=matvec(Rt,velocity[:,3:])
    aB=matvec(Rt,accel[:,:3]); alpha=matvec(Rt,accel[:,3:])
    gB=pose_reference['z'][:,:3]; hlin=pose_reference['z'][:,9:12];hang=pose_reference['z'][:,12:15]
    return dict(u=velocity[:,:3],omega=omega,a=accel[:,:3],alpha=alpha,
        z=torch.cat((gB,uB,omega,hlin,hang),-1),
        zdot=torch.cat((-cross(omega,gB),aB-cross(omega,uB),alpha,-cross(omega,hlin),-cross(omega,hang)),-1))


def integrated(model,t):
    d=model.evaluate(t)
    def quad(key): return torch.trapezoid(d[key],tensor(t),dim=0)
    return {k:v.detach().numpy() for k,v in dict(r=d['r'][-1]-d['r'][0]-quad('v'),
        P=d['P'][-1]-d['P'][0]-quad('dP'),S=d['SW'][-1]-d['SW'][0]-quad('dS')).items()}


def collocation_times(c,n,events,count,rng,margin):
    lo=max(c.start,n.start)+1.24; hi=min(c.end,n.end)
    if hi<=lo: return np.array([]),0.
    # Union all shifted forbidden knots: exact connected complement, then
    # duration-weighted sampling. No preference for the middle of a native step.
    boundaries=np.unique(np.r_[c.times,n.times,events])
    shifted=np.unique(np.concatenate([boundaries+i*.005 for i in range(249)]))
    shifted=shifted[(shifted>=lo-margin)&(shifted<=hi+margin)]
    left=np.maximum(np.r_[lo,shifted+margin],lo)
    right=np.minimum(np.r_[shifted-margin,hi],hi)
    good=right>left; left,right=left[good],right[good]
    length=right-left
    if not len(length): return np.array([]),0.
    idx=rng.choice(len(length),count,p=length/length.sum())
    return left[idx]+rng.random(count)*length[idx],float(length.sum())


def jvp_check(model,points,steps):
    errors={str(h):[] for h in steps}; auto=[]; vector_errors=[]
    for point in points[:12]:
        def mapping(t):
            z=model.evaluate(t-torch.arange(249,dtype=torch.float64)*.005)['z']
            weights=torch.linspace(.3,1.,249*15,dtype=torch.float64).reshape(249,15)
            return (torch.sin(z*.1)*weights).sum()/(249*15)
        t=torch.tensor(point,dtype=torch.float64,requires_grad=True)
        d=model.evaluate(t-torch.arange(249,dtype=torch.float64)*.005)
        _,vector_ad=torch.autograd.functional.jvp(
            lambda time: model.evaluate(time-torch.arange(249,dtype=torch.float64)*.005)['z'],t,torch.ones_like(t))
        vector_errors.append(float(((vector_ad-d['zdot']).abs()/(1+d['zdot'].abs())).max().detach()))
        weights=torch.linspace(.3,1.,249*15,dtype=torch.float64).reshape(249,15)
        analytic=(.1*torch.cos(d['z']*.1)*d['zdot']*weights).sum()/(249*15)
        automatic=torch.autograd.grad(mapping(t),t)[0]
        denom=max(abs(float(analytic.detach())),1.)
        auto.append(abs(float((analytic-automatic).detach()))/denom)
        for h in steps:
            fd=(mapping(t+h)-mapping(t-h))/(2*h)
            errors[str(h)].append(abs(float((analytic-fd).detach()))/denom)
    return dict(autodiff_max=max(auto+vector_errors,default=None),history_vector_autodiff_max=max(vector_errors,default=None),finite_difference_max={k:max(v,default=None) for k,v in errors.items()},tested_points=len(auto))


def analyze_case(path,criteria):
    resolved=json.loads((path/'resolved.json').read_text()); dt=resolved['native_dt']
    compact=path/'compact_200hz'
    if (compact/'source.json').exists():
        identity=json.loads((compact/'source.json').read_text())
        if identity['chunks_sha256'] != digest(path/'chunks.json') or identity.get('replay_sha256')!=digest(Path(__file__).with_name('replay.py')):
            compact.rename(path/('compact_prior_'+digest(compact/'source.json')[:12]))
    if not (compact/'result.json').exists(): replay(path,compact,dt)
    result=dict(recording=accounting(path,compact,criteria['reduction_atol'],criteria['reduction_rtol']),
        coverage=audit(compact), environments=[],status={},
        runtime=json.loads((path/'metrics.json').read_text()) if (path/'metrics.json').exists() else None,
        native_bytes=sum(p.stat().st_size for p in path.glob('chunk-*.npz')),
        compact_bytes=sum(p.stat().st_size for p in compact.rglob('chunk-*.npz')))
    L=float(np.linalg.norm(resolved['geometry']['interior'])); g=9.81
    rng=np.random.default_rng(2718); curves=[]; diagnostics=[]
    for env in range(2):
        natives={s[0]:s for s in segments(path,env)}
        for ep,states,events,intervals in segments(compact,env):
            if ep not in natives: continue
            c=Reconstruction(states,events,intervals); _,ns,ne,ni=natives[ep]; n=Reconstruction(ns,ne,ni)
            m=c.mass; F0=m*g; tau0=F0*L; t0=np.sqrt(L/g); u0=np.sqrt(g*L); w0=1/t0
            scales=dict(F=F0,tau=tau0,u=u0,omega=w0,a=g,alpha=g/L,
                z=np.repeat([g,u0,w0,g,g/L],3),zdot=np.repeat([g,u0,w0,g,g/L],3)/t0,
                e_kin=u0,e_Newton=F0,e_Euler=tau0)
            lo=max(c.start,n.start); hi=min(c.end,n.end)
            grid=c.times[(c.times>=lo)&(c.times<=hi)]
            # Reproducible random phase within each compact interval; a fixed
            # midpoint can alias the native coupling oscillations.
            interior=grid[:-1]+(grid[1:]-grid[:-1])*rng.random(len(grid)-1)
            er=dict(env=env,episode=int(ep),valid_start=float(lo),valid_end=float(hi),mass=m,L=L,metrics={},
                    histories=max(0,int(np.floor((hi-lo-1.24)/.005))+1),windows=0)
            for label,ts in (('grid',grid),('interior',interior)):
                if not len(ts): continue
                cd=c.evaluate(ts); nd=n.evaluate(ts); smooth=n.evaluate(ts,'midpoint_linear')
                er['metrics'][label]={k:stats((cd[k]-nd[k]).detach().numpy(),scale,nd[k].detach().numpy(),ts,cd[k].detach().numpy()) for k,scale in scales.items()}
                discrete=native_velocity_reference(ni,ts,nd)
                er['metrics'][label]['kinematic_reference_sensitivity']={k:stats((nd[k]-discrete[k]).numpy(),scales[k]) for k in discrete}
                er['metrics'][label]['against_native_interval_kinematics']={k:stats((cd[k]-discrete[k]).numpy(),scales[k]) for k in discrete}
                er['metrics'][label]['reference_sensitivity']={k:stats((smooth[k]-nd[k]).numpy(),scales[k]) for k in ('F','tau')}
                er['metrics'][label]['native_floor']={k:stats(nd[k].numpy(),scales[k]) for k in ('e_kin','e_Newton','e_Euler')}
                er['metrics'][label]['compact_conservation']={k:stats(cd[k].numpy(),scales[k]) for k in ('e_kin','e_Newton','e_Euler')}
                if label=='grid':
                    curves.append((env,ep,ts,cd['F'].numpy(),nd['F'].numpy()))
                    diagnostics.append((env,ep,ts,cd,nd))
            ev=np.array([e['time_s'] for e in events if any(e[k] for k in ('feature_update','reset','terminal','command_update'))])
            points,measure=collocation_times(c,n,ev,criteria['minimum_collocations'],rng,criteria['interior_margin_s'])
            if len(points):
                assert interior_mask(points,np.r_[c.times,n.times,ev],lo,hi,margin=criteria['interior_margin_s']).all()
            er.update(collocations=len(points),collocation_available_duration_s=measure,
                jvp=jvp_check(c,points,criteria['finite_difference_steps_s']))
            if len(points):
                taps=(points[:,None]-np.arange(249)[None,:]*.005).ravel()
                ct,nt=c.evaluate(taps),n.evaluate(taps)
                er['collocation_history_metrics']={k:stats((ct[k]-nt[k]).numpy(),scales[k]) for k in ('z','zdot')}
            er['maneuvers']={name:int(sum(bool(r['policy_update']) and a<=float(r['episode_native_step'])*dt<b for r in ne))
                for name,a,b in [('forward',1.6,3.),('stop',3.,4.2),('turn',4.2,6.)]}
            window_errors={k:[] for k in ('r','P','S')}; floors={k:[] for k in window_errors}; coarse={k:[] for k in window_errors}; quadrature={k:[] for k in window_errors}
            for start in grid[::4]:
                if start<lo+1.24 or start+.2>hi+1e-9: continue
                t=start+np.arange(41)*.005
                dense=start+np.arange(round(.2/dt)+1)*dt
                cr=integrated(c,t); nr=integrated(n,dense); nq=integrated(n,t)
                for k in cr:
                    window_errors[k].append(cr[k]-nr[k]); floors[k].append(nr[k]); coarse[k].append(cr[k]);quadrature[k].append(nq[k]-nr[k])
                er['windows']+=1
            er['integrated']={k:dict(excess=stats(window_errors[k],scale),native_floor=stats(floors[k],scale),compact=stats(coarse[k],scale),native_5ms_quadrature_excess=stats(quadrature[k],scale))
                for k,scale in dict(r=L,P=F0*.2,S=tau0*.2).items() if window_errors[k]}
            result['environments'].append(er)
    # Independent world-origin closure, excluding any interval crossing a reset.
    closure=[]; previous=None
    first_escape=[None,None]; termination_times=[[],[]]
    selected_policy_commands=[[],[]]
    for r in stream_rows(path):
        for env in range(2):
            if not r['contained'][env] and r['episode_selected'][env] and first_escape[env] is None: first_escape[env]=float(r['time_s'])
            if r['terminal'][env] and r['episode_selected'][env]: termination_times[env].append(float(r['episode_native_step'][env])*dt)
            if r['policy_update'][env] and r['episode_selected'][env]: selected_policy_commands[env].append(dict(time_s=float(r['time_s']),episode_age_s=float(r['episode_native_step'][env])*dt,commands=r['commands'][env].tolist()))
        if previous is not None and np.any(r['interval_native_steps']):
            duration=r['interval_duration_s'][:,None]; mg=r['mass'][:,None]*r['gravity_W']
            force=mg*duration-(r['momentum_W']-previous['momentum_W'])-r['coupling_force_impulse_W']
            moment=np.cross(previous['com_W'],mg)*duration-(r['angular_momentum_world_origin_W']-previous['angular_momentum_world_origin_W'])-r['coupling_moment_impulse_world_origin_W']
            # Internal angular momentum removes moving CoM orbital momentum at BOTH endpoints.
            S0=previous['angular_momentum_world_origin_W']-np.cross(previous['com_W'],previous['momentum_W'])
            S1=r['angular_momentum_world_origin_W']-np.cross(r['com_W'],r['momentum_W'])
            internal=(S1-S0)-(np.cross(previous['com_W'],mg)*duration-r['coupling_moment_impulse_world_origin_W']-
                (np.cross(r['com_W'],r['momentum_W'])-np.cross(previous['com_W'],previous['momentum_W'])))
            mask=r['interval_valid'] & (r['episode_id']==previous['episode_id'])
            if mask.any(): closure.extend(np.c_[force[mask],moment[mask],internal[mask]].tolist())
        previous=r
    result['first_escape_s_by_env']=first_escape
    result['selected_episode_durations_s']=termination_times
    atomic_json(path/'consumed_commands.json',selected_policy_commands)
    if (path/'terrain.npz').exists():
        with np.load(path/'terrain.npz') as terrain:
            heights=terrain['heights']*terrain['vertical_scale']
            result['terrain']=dict(min_height_m=float(heights.min()),max_height_m=float(heights.max()),origins=terrain['origins'].tolist())
    result['closure']=dict(valid_intervals=len(closure),max_absolute_by_component=np.abs(closure).max(0).tolist() if closure else None,
        sum_over_valid_intervals=np.sum(closure,axis=0).tolist() if closure else None,
        columns='force closure 3, fixed world angular closure 3, moving CoM internal angular closure 3')
    result['rejected_native_intervals']=sum(int((~r['interval_valid']).sum()) for r in stream_rows(path) if np.any(r['interval_native_steps']))
    enough=len(result['environments'])>=2 and all(e['histories']>=criteria['minimum_histories'] and e['collocations']>=criteria['minimum_collocations'] and e['windows']>=criteria['minimum_windows'] and min(e['maneuvers'].values())>=criteria['minimum_maneuver_samples'] for e in result['environments'])
    def pass_metric(metric, rms=None): return metric['normalized_rmse']<=(criteria['normalized_rmse'] if rms is None else rms) and metric['normalized_p95']<=criteria['normalized_p95']
    result['observed_threshold_status']={}
    for category,keys in [('wrench',('F','tau')),('input_derivative',('u','omega','a','alpha','z','zdot')),('local_conservation',('e_kin','e_Newton','e_Euler'))]:
        vals=[e['metrics'][loc][k] for e in result['environments'] for loc in ('grid','interior') for k in keys]
        passed=bool(vals) and all(pass_metric(v,criteria['excess_conservation_rms'] if category=='local_conservation' else None) for v in vals)
        if category=='local_conservation':
            passed &= all(pass_metric(e['metrics'][loc]['compact_conservation'][k],criteria['conservation_rms']) for e in result['environments'] for loc in ('grid','interior') for k in keys)
        if category=='input_derivative':
            passed &= all(pass_metric(v) for e in result['environments'] for v in e.get('collocation_history_metrics',{}).values())
        result['observed_threshold_status'][category]='pass' if passed else 'fail'
        result['status'][category]='inconclusive' if not enough else 'pass' if passed else 'fail'
    instantaneous=any(not pass_metric(e['metrics'][loc]['reference_sensitivity'][k],criteria['native_reference_sensitivity_rms']) for e in result['environments'] for loc in ('grid','interior') for k in ('F','tau'))
    result['native_instantaneous_reference']='inconclusive' if instantaneous or not enough else 'conventions_agree_only_not_ground_truth'
    if instantaneous: result['status']['wrench']='inconclusive'
    uncertain_kinematics=any(not pass_metric(v,criteria['native_reference_sensitivity_rms']) for e in result['environments'] for loc in ('grid','interior') for v in e['metrics'][loc]['kinematic_reference_sensitivity'].values())
    result['kinematic_reference_sensitive']=uncertain_kinematics
    if uncertain_kinematics: result['status']['input_derivative']='inconclusive'
    jvps=[e['jvp'] for e in result['environments']]
    result['status']['jvp_plumbing']='inconclusive' if not enough else 'pass' if all(j['autodiff_max'] is not None and j['autodiff_max']<=criteria['jvp_relative_error'] and all(v<=criteria['jvp_relative_error'] for v in j['finite_difference_max'].values()) for j in jvps) else 'fail'
    integ=[v for e in result['environments'] for v in e['integrated'].values()]
    result['status']['integrated']='inconclusive' if not enough else 'pass' if all(pass_metric(v['excess'],criteria['excess_conservation_rms']) and pass_metric(v['compact'],criteria['conservation_rms']) for v in integ) else 'fail'
    result['averaged_wrench_accounting']=result['recording']['status']
    result['sufficient_coverage']=enough; result['full_collection_ready']=False
    if curves:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(3,1,figsize=(10,7),sharex=True)
        for env,ep,ts,c,n in curves:
            for i in range(3):
                ax[i].plot(ts,n[:,i],alpha=.4,label=f'native env{env} ep{ep}')
                ax[i].plot(ts,c[:,i],lw=.8,label=f'200Hz env{env} ep{ep}')
                ax[i].set_ylabel('F'+ 'xyz'[i]+' (N)')
        ax[0].legend(fontsize=6);ax[-1].set_xlabel('time (s)');fig.tight_layout();fig.savefig(path/'comparison.png');plt.close(fig)
    if diagnostics:
        fig,axes=plt.subplots(3,2,figsize=(11,8),sharex=True)
        for env,ep,ts,cd,nd in diagnostics:
            for ax,key in zip(axes.flat,('u','a','zdot','e_kin','e_Newton','e_Euler')):
                ax.plot(ts,np.linalg.norm(nd[key].numpy(),axis=-1),alpha=.5,label=f'native env{env}')
                ax.plot(ts,np.linalg.norm(cd[key].numpy(),axis=-1),lw=.8,label=f'compact env{env}')
                ax.set_ylabel(key+' norm');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(path/'diagnostics.png');plt.close(fig)
    if any(selected_policy_commands):
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True)
        for env,commands in enumerate(selected_policy_commands):
            if not commands: continue
            ts=[r['time_s'] for r in commands];values=np.array([r['commands'] for r in commands])
            for i,ax in enumerate(axes):
                ax.step(ts,values[:,i],where='post',label=f'env {env}')
                if first_escape[env] is not None: ax.axvline(first_escape[env],color='red',alpha=.4,linestyle='--')
                ax.set_ylabel(('vx (m/s)','vy (m/s)','yaw (rad/s)')[i]);ax.grid(alpha=.2)
        axes[0].legend();axes[-1].set_xlabel('time (s); dashed red = first particle escape')
        fig.tight_layout();fig.savefig(path/'commands.png');plt.close(fig)
    result['attempt_path']=str(path.resolve())
    result['input_checksums']={name:digest(path/name) for name in ('job.json','resolved.json','provenance.json','chunks.json') if (path/name).exists()}
    atomic_json(path/'validation.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);args=p.parse_args()
    started=time.monotonic()
    torch.set_num_threads(1)
    source_snapshot=args.output/'analysis_sources';source_snapshot.mkdir(exist_ok=True)
    sources={}
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(source,source_snapshot/source.name);sources[source.name]=digest(source)
    criteria=json.loads((args.output/'criteria.json').read_text()); report={}
    for name in json.loads((args.output/'experiment.json').read_text())['plans']:
        paths=sorted((args.output/name).glob('round-*/collection/attempt-*'))
        if not paths or not (paths[-1]/'chunks.json').exists(): report[name]=dict(status='inconclusive',reason='No durable native chunks');continue
        if (paths[-1]/'status.json').exists() and json.loads((paths[-1]/'status.json').read_text()).get('status')=='running':
            report[name]=dict(status='inconclusive',reason='Worker still running; no snapshot analyzed');continue
        try: report[name]=analyze_case(paths[-1],criteria)
        except Exception:
            import traceback
            report[name]=dict(status='analysis_failed',error=traceback.format_exc())
        atomic_json(args.output/'report.json',dict(cases=report,criteria=criteria,full_collection_ready=False))
        print(name,report[name].get('status'),flush=True)
    atomic_json(args.output/'report.json',dict(cases=report,criteria=criteria,full_collection_ready=False))
    atomic_json(args.output/'analysis_provenance.json',dict(sources=sources,criteria_sha256=digest(args.output/'criteria.json'),
        command=shlex.join([sys.executable,'-m','liquid_collection.validation.analyze',*sys.argv[1:]]),wall_seconds=time.monotonic()-started))
    lines=['# Liquid format validation','', 'Production readiness remains **false**. Thresholds are provisional and frozen before simulation.','',
        '| Case | Recording | Coverage | Wrench | Input derivative | Integrated |','|---|---|---|---|---|---|']
    for name,r in report.items():
        status=r.get('status',{})
        if not isinstance(status,dict): lines.append(f'| {name} | {status} | insufficient | inconclusive | inconclusive | inconclusive |');continue
        lines.append(f"| {name} | {r['recording']['status']} | {r['sufficient_coverage']} | {status['wrench']} | {status['input_derivative']} | {status['integrated']} |")
    lines+=['','| Case / env | Valid end (s) | Histories | Collocations | 200 ms windows | Forward / stop / turn observations |','|---|---|---|---|---|---|']
    for name,r in report.items():
        for e in r.get('environments',[]):
            lines.append(f"| {name} / {e['env']} | {e['valid_end']:.3f} | {e['histories']} | {e['collocations']} | {e['windows']} | {list(e['maneuvers'].values())} |")
    lines+=['','See report.json for per-environment metrics, rejected samples, scales, counts, runtimes, storage, native floors and sensitivity. Each attempt contains comparison.png and checksummed native/compact streams.', '',
        'Impulse conservation establishes interval-average bookkeeping only. Instantaneous force is not directly observed. Local JVP adequacy requires both input reconstruction and plumbing checks; integrated adequacy requires measured native floors and excess error. No pooled pass and no automatic production-readiness promotion.', '',
        'Smallest next adjustments: correct any coverage/configuration blocker first; if derivative accuracy fails, revise pose interpolation or retain selected native pose/velocity channels. If wrench sensitivity fails, test an impulse-preserving smooth reconstruction or selected native wrench channels. Do not change the training objectives.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    return 2 if any(r.get('status')=='analysis_failed' for r in report.values()) else 0

if __name__=='__main__': raise SystemExit(main())
