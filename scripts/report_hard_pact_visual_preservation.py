"""Offline bounded-event analysis and synchronized videos. No simulator imports."""
import csv,json,subprocess
from pathlib import Path
import numpy as np
from scripts.run_hard_pact_velocity_stress import read,write,stats


def motion_shape(values,prominence=.05):
    from scipy.signal import find_peaks
    x=np.asarray(values);x=np.unwrap(x[np.isfinite(x)])
    if len(x)<3:return dict(available=False)
    maxima=find_peaks(x,prominence=prominence)[0];minima=find_peaks(-x,prominence=prominence)[0]
    return dict(available=True,peak_abs=float(abs(x).max()),peak_to_peak=float(np.ptp(x)),
        prominent_reversals=int(len(maxima)+len(minima)),net_change=float(x[-1]-x[0]),
        total_variation=float(abs(np.diff(x)).sum()),
        description='repeated reversals' if len(maxima)+len(minima)>=2 else 'no repeated prominent reversals in captured window',
        prominence_rad=prominence,angle_convention='Unwrapped Euler angle; pi branch crossings are not reversals.')


def report_events(root,inventory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot=root/'events';plot.mkdir(exist_ok=True);summaries=[];event_rows=[];descent=[];repeats=[]
    for p in sorted((root/'video_seed2').glob('*_seed2')):
        old=root/'reused'/p.name/'trace.npz'
        if not old.exists() or not (p/'trace.npz').exists():continue
        a=np.load(p/'trace.npz');b=np.load(old);comparison=dict(condition=p.name,threshold=1e-5)
        for k in ('velocity_before','orientation_xyzw','command','executed_torque'):
            err=abs(a[k]-b[k]).reshape(len(a[k]),-1).max(1);different=np.flatnonzero(err>1e-5)
            comparison[k]=dict(initial_max=float(err[0]),first_different_time_s=float(different[0]*.005) if len(different) else None)
        repeats.append(comparison)
    write(root/'video_repeat_parity.json',repeats)
    # Standard-event descriptions remain separate from probe outcomes.
    for item in inventory:
        p=Path(item['path']);a=np.load(p/'trace.npz')
        failures=np.argwhere(a['control_reset'].astype(bool)&a['control_failure'].astype(bool))
        for t,e in failures:
            start=max(0,int(t)-25)
            resets=np.flatnonzero(a['control_reset'][:t,e])
            if len(resets):start=max(start,int(resets[-1])+1)
            event_rows.append(dict(condition=item['condition'],seed=item['seed'],env=int(e),step=int(t),
                roll=motion_shape(a['control_tilt'][start:t+1,e,0]),pitch=motion_shape(a['control_tilt'][start:t+1,e,1]),
                height=stats(a['control_base_height'][start:t+1,e]),
                reasons=a['control_reasons'][t,e].tolist(),window='0.5s pre-failure, truncated at reset'))
    write(root/'standard_event_motion.json',event_rows)
    for p in sorted((root/'probes_seed2').glob('*_seed2')):
        if not (p/'probe_events.json').exists():continue
        result=read(p/'probe_events.json');manifest=read(p/'manifest.json');a=np.load(p/'trace.npz')
        result['condition']=manifest['condition'];summaries.append(result)
        selected=[e for e in result['events'] if e['probe_started']]
        for event in selected:
            env,t,end=event['env'],event['step'],event['end_step'];start=max(0,t-25)
            resets=np.flatnonzero(a['control_reset'][:t,env])
            if len(resets):start=max(start,int(resets[-1])+1)
            sl=slice(start,end+1);physical=slice(start*4,(end+1)*4)
            event['roll_motion']=motion_shape(a['control_tilt'][sl,env,0])
            event['pitch_motion']=motion_shape(a['control_tilt'][sl,env,1])
            event['height_min']=float(a['control_base_height'][sl,env].min())
            event['height_peak_to_peak']=float(np.ptp(a['control_base_height'][sl,env]))
            event['angular_rates']=stats(a['body_velocity_after'][physical,env,3:6])
            if 'foot_clearance' in a:
                event['foot_clearance']=stats(a['foot_clearance'][physical,env])
                event['clearance_convention']='Foot z minus local mean sampled terrain minus configured foot-height offset; not a torso collision clearance.'
            # Matched healthy comparison: same environment, no reset or trigger,
            # nearest mean command among fixed half-second windows.
            target=a['control_command'][start:t+1,env].mean(0)
            candidates=[j for j in range(25,len(a['control_reset'])-25,25)
                if not a['control_original_reset'][j-25:j+1,env].any()
                and not a['control_reasons'][j-25:j+1,env,:5].any()
                and a['control_ordinary'][j-25:j+1,env].all()]
            healthy=min(candidates,key=lambda j:np.linalg.norm(a['control_command'][j-25:j+1,env].mean(0)-target)) if candidates else None
            event['healthy_control_end_step']=healthy
            fig,axes=plt.subplots(5,1,figsize=(10,11),sharex=True,constrained_layout=True)
            x=(np.arange(start,end+1)-t)*.02;xp=(np.arange(start*4,(end+1)*4)+1-(t+1)*4)*.005
            thresholds=manifest['environment']['termination']
            for i,name in enumerate(('roll','pitch')):
                axes[0].plot(x,np.unwrap(a['control_tilt'][sl,env,i]),label=name+' (unwrapped)')
                axes[0].axhline(thresholds[name+'_threshold'],ls=':',alpha=.4)
                axes[0].axhline(-thresholds[name+'_threshold'],ls=':',alpha=.4)
            axes[1].plot(x,a['control_base_height'][sl,env],label='termination height')
            axes[1].axhline(thresholds['height_min'],ls=':',label='original low bound')
            if 'foot_clearance' in a:axes[1].plot(xp,a['foot_clearance'][physical,env].min(-1),alpha=.6,label='minimum foot clearance')
            if 'control_terrain_height_reference' in a:axes[1].plot(x,a['control_terrain_height_reference'][sl,env]-a['control_terrain_height_reference'][start,env],label='terrain reference change')
            for i in range(3):axes[2].plot(xp,a['body_velocity_after'][physical,env,3+i],label=['wx','wy','wz'][i])
            for i in range(4):
                line=axes[3].plot(xp,a['contact_probability'][physical,env,i],label=f'foot{i}')[0]
                axes[3].plot(xp,a['contact_truth'][physical,env,i],':',color=line.get_color(),alpha=.5)
            axes[4].plot(xp,np.sqrt(np.mean(a['correction'][physical,env]**2,-1)),label='correction RMS')
            if healthy is not None:
                hx=np.arange(-25,1)*.02
                axes[0].plot(hx,a['control_tilt'][healthy-25:healthy+1,env,0],'--',alpha=.5,label='healthy roll')
                axes[1].plot(hx,a['control_base_height'][healthy-25:healthy+1,env],'--',alpha=.5,label='healthy height')
            for ax,label in zip(axes,['angles [rad]','height [m]','angular rate [rad/s]','contact p / truth dotted','correction [Nm]']):
                ax.set_ylabel(label);ax.axvline(0,color='red',ls='--');ax.legend(fontsize=8,ncol=4);ax.grid(alpha=.2)
            axes[-1].set_xlabel('Seconds relative to original failure, not delayed reset')
            fig.suptitle(f"{manifest['condition']} env{env} {event['family']} L{event['level']}: {event['outcome']}")
            fig.savefig(plot/f"{manifest['condition']}_event{event['event_id']}.png",dpi=130);plt.close(fig)
    write(root/'probe_summary.json',summaries)
    with (root/'probe_summary.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','original_failures','ordinary_env_s','probe_env_s','probe_starts','recovered','hard_stop','unresolved','recovery_fraction'])
        for r in summaries:w.writerow([r['condition'],r['original_failures'],r['ordinary_env_s'],r['probe_env_s'],r['probe_starts'],*[r['outcomes'][k] for k in ('recovered','hard_stop','unresolved')],r['recovery_fraction']])
    for p in sorted((root/'video_seed2').glob('*_seed2')):
        if not (p/'video_manifest.json').exists():continue
        m=read(p/'video_manifest.json');a=np.load(p/'trace.npz')
        for e in m['env_ids']:
            # Descent requires a lowering terrain reference along horizontal
            # travel within ONE uninterrupted episode, not a reset jump/fall.
            reset=a['control_reset'][:,e];starts=np.r_[0,np.flatnonzero(reset)+1];ends=np.r_[np.flatnonzero(reset),len(reset)-1]
            segments=[]
            for lo,hi in zip(starts,ends):
                if hi<=lo:continue
                z=a['control_terrain_height_reference'][lo:hi+1,e]
                pos=a['control_base_position_world'][lo:hi+1,e]
                travel=np.linalg.norm(pos[:,:2]-pos[0,:2],axis=-1)
                drop=z[0]-z
                j=int(np.argmax(np.where(travel>.25,drop,-np.inf)))
                segments.append(dict(start_step=int(lo),end_step=int(hi),terrain_drop_m=float(drop[j]),horizontal_travel_m=float(travel[j]),
                    descent_verified=bool(travel[j]>.25 and drop[j]>.08)))
            descent.append(dict(condition=p.name,env=e,verified=any(s['descent_verified'] for s in segments),episodes=segments,
                                criterion='terrain-reference drop >8cm with >25cm horizontal travel, same episode; inspect video too'))
    write(root/'descent_verification.json',descent)


def comparison_videos(root,variants):
    import imageio_ffmpeg
    out=root/'comparison_videos';out.mkdir(exist_ok=True)
    manifest=root/'video_seed2'/f'{variants[0]}_seed2'/'video_manifest.json'
    if not manifest.exists():return
    for env in read(manifest)['env_ids']:
        inputs=[root/'video_seed2'/f'{v}_seed2'/f'stairs_down_env{env}.mp4' for v in variants]
        if not all(p.exists() for p in inputs):continue
        target=out/f'stairs_down_env{env}_all_conditions.mp4'
        if target.exists():continue
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-filter_complex_threads','1']
        for p in inputs:cmd+=['-i',str(p)]
        cmd+=['-filter_complex','[0:v][1:v][2:v][3:v][4:v]xstack=inputs=5:layout=0_0|640_0|1280_0|0_480|640_480:fill=black[v]',
              '-map','[v]','-c:v','libx264','-threads','2','-crf','22','-pix_fmt','yuv420p',str(target)]
        subprocess.run(cmd,check=True)
