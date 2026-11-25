#!/usr/bin/env python3
"""Bounded eight-run evaluation protocol and offline report; never trains."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EVENT='events.out.tfevents.1795413624.CICSUNIX-US-003.133079.0'
CKPT=ROOT/'logs/hardpact_iclr/go2_hard_pact_rss_full/Nov23_01-00-22_hard_pact_rss_full_isaaclab/model_10500.pt'
VARIANTS=['B_extra050','C_attitude005','E_extra075','F_extra075_attitude005']


def rank(r):
    s=r['stability']
    return (r['physical_resets'],s['roll_rad']['rms']+s['pitch_rad']['rms'],
        s['height_error_m']['rms'],sum(r['tracking_rmse']))


def load(folder):return [json.loads(p.read_text()) for p in folder.glob('*/result.json')]


def check_matching(folder):
    manifests=[json.loads(p.read_text()) for p in folder.glob('*/manifest.json')]
    for key in ('initial_state_hash','terrain_levels','terrain_types','terrain_source_rows'):
        assert all(m[key]==manifests[0][key] for m in manifests),key
    for key in ('commands','light_wrench','heavy_wrench','applied','mechanics'):
        assert len({m['tapes'][key] for m in manifests})==1,key


def report(root,selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shutil
    archive=root/'report_sources';archive.mkdir(exist_ok=True)
    sources=[Path(__file__),ROOT/'scripts/hard_pact_wrench_stress.py',ROOT/'tests/test_hard_pact_wrench_stress.py']
    for source in sources:shutil.copy2(source,archive/source.name)
    (root/'analysis_provenance.json').write_text(json.dumps(dict(
        code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        argv=sys.argv,limitations=['Single seed for selection, separate single seed for confirmation; correlated environments/time samples.',
            'Fixed target-heading tapes do not imply identical realized yaw commands after trajectory divergence.',
            'Wrench stress only, not moving-liquid validation. Execution predictions assume candidate GRFs.']),indent=2))
    records=[]
    for label in ('stage1_light','stage2_light','stage2_heavier'):
        for r in load(root/label):r['comparison_group']=label;records.append(r)
    records.sort(key=lambda r:(('stage1_light','stage2_light','stage2_heavier').index(r['comparison_group']),
        (VARIANTS+['off']).index(r['condition'])))
    # At the next pre-physics hook orientation equals the preceding endpoint.
    # Exclude all reset/teleport intervals and the final missing endpoint.
    for r in records:
        folder=root/r['comparison_group']/f"{r['condition']}_seed{r['seed']}"
        with np.load(folder/'trace.npz') as a:
            v=a['body_velocity_after'][:-1,:,:3];q=a['orientation_xyzw'][1:]
            cross=2*np.cross(q[...,:3],v);world=v+q[...,3:]*cross+np.cross(q[...,:3],cross)
            valid=np.repeat(~a['control_reset'].astype(bool)&~a['control_reasons'][...,6].astype(bool),4,axis=0)[:-1]
            z=world[...,2][valid]
            r['stability']['body_z_velocity_m_s']=r['stability'].pop('vertical_velocity_m_s')
            r['stability']['world_vertical_velocity_m_s']={'count':len(z),'rms':float(np.sqrt(np.mean(z*z))),
                'abs_p95':float(np.quantile(abs(z),.95)),'abs_max':float(abs(z).max())}
    (root/'comparison.json').write_text(json.dumps(records,indent=2,allow_nan=False))
    columns=['group','condition','seed','failures','failure_per_env_s','failure_free_fraction',
        'roll_rms_rad','pitch_rms_rad','height_error_rms_m','angular_rate_rms','rmse_x','rmse_y','rmse_yaw',
        'joint_position_violation_fraction','joint_velocity_violation_fraction','torque_violation_max_nm']
    with (root/'comparison.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(columns)
        for r in records:
            s=r['stability'];writer.writerow([r['comparison_group'],r['condition'],r['seed'],r['physical_resets'],r['failure_per_env_s'],
                r['failure_free_fraction'],s['roll_rad']['rms'],s['pitch_rad']['rms'],s['height_error_m']['rms'],s['angular_velocity_rad_s']['rms'],
                *r['tracking_rmse'],r['joint_position_violation']['coordinate_fraction_above_tolerance'],
                r['joint_velocity_violation']['coordinate_fraction_above_tolerance'],r['torque_violation']['max']])
    changes={}
    for variant in (selected,'off'):
        light=next(r for r in records if r['comparison_group']=='stage2_light' and r['condition']==variant)
        heavy=next(r for r in records if r['comparison_group']=='stage2_heavier' and r['condition']==variant)
        changes[variant]={'tracking_rmse_light_to_heavier':(np.array(heavy['tracking_rmse'])-light['tracking_rmse']).tolist(),
            'failure_rate_change':heavy['failure_per_env_s']-light['failure_per_env_s'],
            'roll_rms_change':heavy['stability']['roll_rad']['rms']-light['stability']['roll_rad']['rms'],
            'pitch_rms_change':heavy['stability']['pitch_rad']['rms']-light['stability']['pitch_rad']['rms']}
    (root/'load_deterioration.json').write_text(json.dumps(changes,indent=2))
    with (root/'objective_costs.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['group','condition','stage','component','count','mean','p95','max'])
        for r in records:
            for k,v in r['diagnostics'].items():
                if k.startswith(('objective_nominal_','objective_candidate_','objective_execution_')) and '_change_' not in k:
                    _,stage,component=k.split('_',2)
                    writer.writerow([r['comparison_group'],r['condition'],stage,component,v['count'],v['mean'],v['p95'],v['max']])
    # Save per-axis estimator errors and censoring explicitly, from already collected traces.
    details=[]
    for r in records:
        folder=root/r['comparison_group']/f"{r['condition']}_seed{r['seed']}"
        with np.load(folder/'trace.npz') as a:
            mask=np.repeat(~a['control_reset'].astype(bool)&~a['control_reasons'][...,6].astype(bool),4,axis=0)
            err=(a['predicted_wrench']-a['applied_wrench'])[mask]
            details.append(dict(group=r['comparison_group'],condition=r['condition'],wrench_rmse_axes=np.sqrt(np.mean(err**2,axis=0)).tolist(),
                first_failure_or_censor_s=[20. if v is None else v for v in r['time_to_first_failure_s']],
                first_failure_censored=[v is None for v in r['time_to_first_failure_s']],
                valid_nonreset_substep_rows=int(mask.sum()),candidate_primary=r['primary_rows'],candidate_recovery=r['recovery_rows'],fallback=r['fallback_rows']))
            details[-1]['model_violation_summaries']={}
            for stage in ('candidate','execution'):
                for field,unit,tolerance in (('joint_position_violation','rad',1e-4),('joint_velocity_violation','rad/s',1e-3),('endpoint_torque_violation','Nm',1e-4)):
                    key=stage+'_'+field
                    if key not in a:continue
                    value=a[key][mask];valid=np.isfinite(value);values=value[valid]
                    details[-1]['model_violation_summaries'][key]=dict(units=unit,threshold=tolerance,count=len(values),
                        fraction=float((values>tolerance).mean()) if len(values) else None,max=float(values.max()) if len(values) else None)
    (root/'health_estimators_censoring.json').write_text(json.dumps(details,indent=2,allow_nan=False))
    labels=[r['comparison_group'].replace('stage','S')+'\n'+r['condition'] for r in records]
    fig,axes=plt.subplots(3,1,figsize=(13,10),constrained_layout=True)
    axes[0].bar(labels,[r['failure_per_env_s'] for r in records]);axes[0].set_ylabel('Failures / env-second')
    for name in ('roll_rad','pitch_rad'):axes[1].plot(labels,[r['stability'][name]['rms'] for r in records],'o-',label=name)
    axes[1].set_ylabel('RMS [rad]');axes[1].legend()
    for j,name in enumerate(('vx [m/s]','vy [m/s]','yaw [rad/s]')):axes[2].plot(labels,[r['tracking_rmse'][j] for r in records],'o-',label=name)
    axes[2].set_ylabel('Tracking RMSE');axes[2].legend()
    for ax in axes:ax.tick_params(axis='x',labelsize=7)
    fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    for group in ('stage2_light','stage2_heavier'):
        for variant in (selected,'off'):
            r=next(x for x in records if x['comparison_group']==group and x['condition']==variant)
            first=r['time_to_first_failure_s'];times=np.linspace(0,20,101)
            axes[0].step(times,[np.mean([v is None or v>t for v in first]) for t in times],where='post',label=group+' '+variant)
            axes[1].plot([0,1],[r['contact_lags'][k]['matched_fraction'] or 0 for k in ('touchdown','liftoff')],'o-',label=group+' '+variant)
    axes[0].set(xlabel='Time [s]',ylabel='Initial cohort failure-free fraction');axes[0].legend(fontsize=7)
    axes[1].set(xticks=[0,1],xticklabels=['touchdown','liftoff'],ylabel='Matched transitions within 100 ms');axes[1].legend(fontsize=7)
    fig.savefig(root/'survival_contacts.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,8),constrained_layout=True)
    for group in ('stage2_light','stage2_heavier'):
        for variant in (selected,'off'):
            folder=root/group/f'{variant}_seed2';index=json.loads((folder/'history_index.json').read_text())
            events=[v for v in index if v['kind']=='failure']
            if not events:continue
            v=events[0];prefix=f"{v['id']:02d}_";label=group+' '+variant
            with np.load(folder/'bounded_histories.npz') as h:
                tilt=np.linalg.norm(h[prefix+'control_tilt'],axis=-1);x=np.arange(1-len(tilt),1)*.02
                axes[0].plot(x,tilt,label=label)
                torque=h[prefix+'correction'];x=np.arange(1-len(torque),1)*.005
                axes[1].plot(x,np.sqrt(np.mean(torque**2,axis=-1)),label=label)
                contact=abs(h[prefix+'contact_probability']-h[prefix+'contact_truth']).mean(-1)
                axes[2].plot(x,contact,label=label)
    for ax,y in zip(axes,('Tilt norm [rad]','Correction RMS [Nm]','Mean contact probability error')):
        ax.set(xlabel='Time before first captured failure [s]',ylabel=y);ax.legend(fontsize=7)
    fig.savefig(root/'prefailure_examples.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--report-only',action='store_true')
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    def run(folder,seed,variants,load_name):
        cmd=[sys.executable,str(ROOT/'scripts/eval_hard_pact_tradeoff.py'),'--wrench-stress','--event',EVENT,
            '--checkpoint',str(CKPT),'--output',str(root/folder),'--steps','1000','--seeds',str(seed),'--load',load_name,'--variants',*variants]
        print('RUN', ' '.join(cmd),flush=True)
        subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',SIMULATOR='isaaclab',PYTHONPATH=str(ROOT)),check=True)
        check_matching(root/folder)
    if not args.report_only:run('stage1_light',1,VARIANTS,'light')
    results=load(root/'stage1_light');assert len(results)==4
    selected=min(results,key=rank)['condition']
    (root/'selection.json').write_text(json.dumps({'selected':selected,'rule':'Failures, then sum roll/pitch RMS, then height RMS, then tracking sum; no tracking veto.',
        'ranking':[{'condition':r['condition'],'score':rank(r)} for r in sorted(results,key=rank)]},indent=2))
    if not args.report_only:
        run('stage2_light',2,[selected,'off'],'light');run('stage2_heavier',2,[selected,'off'],'heavier')
    for name in ('commands','light_wrench','heavy_wrench','mechanics'):
        a=load(root/'stage2_light')[0]['tapes'][name];b=load(root/'stage2_heavier')[0]['tapes'][name];assert a==b,name
    report(root,selected)


if __name__=='__main__':main()
