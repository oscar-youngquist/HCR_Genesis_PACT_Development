#!/usr/bin/env python3
"""Frozen velocity-weight comparison; reuses absolute-time wrench tapes, no training."""
import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_hard_pact_wrench_stress import EVENT,CKPT
NEW=['B_planar10','B_yaw5','B_both_lower']
REF=['B_extra050','off']

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return dict(count=int(x.size),rms=float(np.sqrt(np.mean(x*x))) if x.size else None,
        mean=float(x.mean()) if x.size else None,p95=float(np.quantile(abs(x),.95)) if x.size else None,
        max=float(abs(x).max()) if x.size else None)
def axes(x):return [stats(x[...,i]) for i in range(x.shape[-1])]

def metrics(folder,load):
    r=read(folder/'result.json');m=read(folder/'manifest.json')
    with np.load(folder/'trace.npz') as a:
        clean=~a['control_reset'].astype(bool)&~a['control_reasons'][...,6].astype(bool)
        valid=np.repeat(clean,4,axis=0)
        active=np.linalg.norm(a['applied_wrench'],axis=-1)>1e-6
        # Use pre-substep state and its active command; no next-episode or teleport samples.
        err=a['velocity_before']-a['command']
        tracking={k:axes(err[mask]) for k,mask in [('all',valid),('active',valid&active)]}
        cmd=a['command'][...,:2];speed=np.linalg.norm(cmd,axis=-1);direction=cmd/np.maximum(speed[...,None],1e-12)
        along=np.sum(a['velocity_before'][...,:2]*direction,axis=-1)
        motion={k:dict(achieved=stats(along[mask&(speed>.1)]),commanded=stats(speed[mask&(speed>.1)]),
            signed_error=stats((along-speed)[mask&(speed>.1)])) for k,mask in [('all',valid),('active',valid&active)]}
        wrench={k:axes(a[k][valid&active]) for k in ('applied_wrench','predicted_wrench')}
        wrench['prediction_error']=axes((a['predicted_wrench']-a['applied_wrench'])[valid&active])
        stability=dict(roll=stats(a['control_tilt'][...,0][clean]),pitch=stats(a['control_tilt'][...,1][clean]),
            height=stats((a['control_base_height']-m['environment']['rewards']['base_height_target'])[clean]),
            angular_velocity=axes(a['body_velocity_after'][...,3:][valid]))
        objectives={k:stats(a[k][valid]) for k in a.files if k.startswith('objective_')}
        violations={k:dict(**stats(a[k][valid]),fraction=float(np.mean(a[k][valid]>tol)))
            for k,tol in [('joint_position_violation',1e-4),('joint_velocity_violation',1e-3),('torque_violation',1e-4)]}
        for stage in ('candidate','execution'):
            for field,tol in [('joint_position_violation',1e-4),('joint_velocity_violation',1e-3),('endpoint_torque_violation',1e-4)]:
                key=stage+'_'+field
                if key in a:
                    x=a[key][valid];x=x[np.isfinite(x)]
                    violations[key]=dict(**stats(x),fraction=float(np.mean(x>tol)) if x.size else None)
        corrections={}
        if 'execution_strength' in a:
            for name,mask in [('baseline',a['baseline'].astype(bool)),('extra',~a['baseline'].astype(bool))]:
                mask=valid&mask
                corrections[name]=dict(magnitude=stats(a['correction'][mask]),clipped=stats(a['clipped'][mask]),alpha=stats(a['execution_strength'][mask]))
            expected=[m['effective_qp']['contact_acceleration_weight'],m['effective_qp']['torque_tracking_weight'],m['effective_qp']['attitude_weight'],m['effective_qp']['planar_velocity_weight'],m['effective_qp']['yaw_rate_weight']]
            weights=a['assembly_weights'];assert np.allclose(weights[...,:4],expected[:4])
            if weights.shape[-1]==5:assert np.allclose(weights,expected)
            assert np.max(a['alpha_error'])<1e-6 and np.max(a['execution_reconstruction_error'])<2e-4
        first=r['time_to_first_failure_s']
        return dict(condition=r['condition'],seed=r['seed'],load=load,path=str(folder),failures=r['physical_resets'],
            failure_rate=r['failure_per_env_s'],failure_free=r['failure_free_fraction'],
            first_failure_or_censor_s=[20. if t is None else t for t in first],censored=[t is None for t in first],
            tracking=tracking,motion=motion,wrench=wrench,stability=stability,objectives=objectives,violations=violations,
            corrections=corrections,primary=r['primary_rows'],recovery=r['recovery_rows'],fallback=r['fallback_rows'],
            valid_substeps=int(valid.sum()),active_substeps=int((valid&active).sum()),
            original_result=r)

def audit(folders):
    manifests=[read(p/'manifest.json') for p in folders]
    first=manifests[0]
    for key in ('checkpoint_sha256','config_sha256','code_sha','initial_state_hash','terrain_levels','terrain_types','terrain_source_rows'):
        assert all(m[key]==first[key] for m in manifests),key
    for key in ('commands','light_wrench','heavy_wrench','applied','mechanics'):
        assert len({m['tapes'][key] for m in manifests})==1,key
    # Execution/diagnostic differences are explicitly allowed, no other QP changes.
    ignore={'planar_velocity_weight','yaw_rate_weight'}
    q={k:v for k,v in first['effective_qp'].items() if k not in ignore}
    assert all({k:v for k,v in m['effective_qp'].items() if k not in ignore}==q for m in manifests)
    assert first['checkpoint_sha256']==sha(CKPT)
    assert first['config_sha256']==sha(Path(first['config']))
    return dict(paths=[str(p) for p in folders],matched_fields=list(first['tapes']),
        checkpoint_sha256=first['checkpoint_sha256'],config_sha256=first['config_sha256'],code_sha=first['code_sha'],
        note='Evaluator changes limited to three explicit overrides, assertion allowlist, and an additional recorded yaw weight; tape generator unchanged. Archived original sources retained.')

def source_audit(reference):
    old=reference/'stage2_light/sources';checks={}
    for filename,names in [('hard_pact_wrench_stress.py',('Tapes','heavier_waveform')),('eval_hard_pact_tradeoff.py',('worker','freeze_settings'))]:
        sources=[old/filename,ROOT/'scripts'/filename]
        trees=[ast.parse(p.read_text()) for p in sources]
        for name in names:
            dumps=[ast.dump(next(n for n in tree.body if getattr(n,'name',None)==name)) for tree in trees]
            assert dumps[0]==dumps[1],(filename,name)
            checks[filename+':'+name]=hashlib.sha256(dumps[0].encode()).hexdigest()
    helper='hard_pact_intervention_eval.py'
    assert sha(old/helper)==sha(ROOT/'scripts'/helper)
    changes=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
    assert not any(p.startswith(('rsl_rl/','legged_gym/')) for p in changes),changes
    return dict(unchanged_execution_ast_hashes=checks,unchanged_diagnostics_hash=sha(old/helper),
        current_sources={p:sha(ROOT/'scripts'/p) for p in ('eval_hard_pact_tradeoff.py','hard_pact_wrench_stress.py','run_hard_pact_velocity_stress.py')},
        modified_tracked_files=changes)

def report(root,reference,seed,variants):
    rows=[];audits=[]
    for load in ('light','heavier'):
        folders=[root/f'seed{seed}_{load}'/f'{v}_seed{seed}' for v in variants]
        folders += [(reference/f'stage2_{load}' if seed==2 else root/f'seed{seed}_{load}')/f'{v}_seed{seed}' for v in REF]
        audits.append(audit(folders));rows.extend(metrics(p,load) for p in folders)
    write(root/f'comparison_seed{seed}.json',rows);write(root/f'parity_seed{seed}.json',audits)
    fields=['condition','seed','load','failures','failure_rate','failure_free','roll','pitch','height','vx','vy','yaw','active_vx','active_vy','active_yaw','achieved_motion']
    with (root/f'comparison_seed{seed}.csv').open('w') as f:
        w=csv.writer(f);w.writerow(fields)
        for r in rows:w.writerow([r[k] for k in fields[:6]]+[r['stability'][k]['rms'] for k in ('roll','pitch','height')]+[x['rms'] for x in r['tracking']['all']]+[x['rms'] for x in r['tracking']['active']]+[r['motion']['active']['achieved']['mean']])
    with (root/f'objectives_seed{seed}.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','load','metric','count','mean','rms','abs_p95','abs_max'])
        for r in rows:
            for k,v in r['objectives'].items():w.writerow([r['condition'],r['load'],k,v['count'],v['mean'],v['rms'],v['p95'],v['max']])
    changes={}
    for v in variants+REF:
        l,h=[next(r for r in rows if r['condition']==v and r['load']==load) for load in ('light','heavier')]
        changes[v]={k:(np.array([x['rms'] for x in h['tracking'][k]])-np.array([x['rms'] for x in l['tracking'][k]])).tolist() for k in ('all','active')}
    write(root/f'load_changes_seed{seed}.json',changes)
    paired=[]
    for r in rows:
        if r['condition'] not in variants:continue
        for ref in REF:
            b=next(x for x in rows if x['condition']==ref and x['load']==r['load'])
            paired.append(dict(condition=r['condition'],reference=ref,load=r['load'],
                failure_rate_difference=r['failure_rate']-b['failure_rate'],
                tracking_rmse_difference={k:[x['rms']-y['rms'] if x['rms'] is not None and y['rms'] is not None else None
                    for x,y in zip(r['tracking'][k],b['tracking'][k])] for k in ('all','active')},
                stability_rms_difference={k:r['stability'][k]['rms']-b['stability'][k]['rms'] for k in ('roll','pitch','height')}))
    write(root/f'paired_seed{seed}.json',paired)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for r in rows:
        label=r['condition']+' '+r['load'];rmse=np.linalg.norm([x['rms'] for x in r['tracking']['all'][:2]])
        axs[0].scatter(rmse,r['failure_rate'],label=label)
        axs[1].scatter(r['stability']['roll']['rms']+r['stability']['pitch']['rms'],r['failure_rate'],label=label)
        t=np.linspace(0,20,101);first=r['first_failure_or_censor_s'];c=r['censored']
        axs[2].step(t,[np.mean([cc or f>x for f,cc in zip(first,c)]) for x in t],where='post',label=label)
    axs[0].set(xlabel='Planar tracking RMSE norm [m/s]',ylabel='Failures / env-second')
    axs[1].set(xlabel='Roll + pitch RMS [rad]',ylabel='Failures / env-second')
    axs[2].set(xlabel='Time [s]',ylabel='Initial-cohort failure-free fraction');axs[2].legend(fontsize=6)
    fig.savefig(root/f'tracking_survival_seed{seed}.png',dpi=160);plt.close(fig)
    combined=[]
    for path in sorted(root.glob('comparison_seed*.json')):combined.extend(read(path))
    write(root/'comparison_all.json',combined)
    with (root/'comparison_all.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(fields)
        for path in sorted(root.glob('comparison_seed*.csv')):
            with path.open() as source:
                reader=csv.reader(source);next(reader);writer.writerows(reader)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path,default=ROOT/'diagnostics/wrench_stress10500/full')
    p.add_argument('--seed',type=int,choices=(2,3),default=2);p.add_argument('--candidate',choices=NEW)
    p.add_argument('--report-only',action='store_true');args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    write(root/'reference_source_audit.json',source_audit(args.reference.resolve()))
    archive=root/'report_sources';archive.mkdir(exist_ok=True)
    for source in (Path(__file__),ROOT/'tests/test_hard_pact_wrench_stress.py',ROOT/'tests/test_eval_hard_pact_tradeoff.py'):
        shutil.copy2(source,archive/source.name)
    write(root/'metric_conventions.json',dict(
        tracking='Pre-substep body vx,vy,wz versus matching physical command. Entire reset/teleport intervals excluded.',
        active='Applied world/base-origin wrench L2 norm > 1e-6 (N and Nm components used only as a nonzero test).',
        motion='Signed body-planar velocity along command, excluding command norm <= 0.1 m/s; no progress inferred from speed alone.',
        censoring='Initial cohort first-failure times, right-censored at 20 s; repeated failures remain in rate numerator.',
        objectives='Production-normalized weighted costs and physical residuals; compare residuals, not reduced coefficients alone. Execution reconstruction retains optimized candidate GRFs.',
        failures='All original failures retained; cause counts can overlap. Denominator 200 simulated environment-seconds/run.',
        matching='Absolute command/heading targets, wrench tapes, noise/push RNG and mechanics matched. Yaw feedback and reset histories remain trajectory-dependent.',
        limits='No independent samples/significance claims; 10 correlated environments and short trials. Wrench stress, not moving-liquid validation.',
        timing='Shared-machine simulator collection, not deployment latency. PyTorch allocator memory excludes PhysX/renderer allocations.'))
    variants=NEW if args.seed==2 else [args.candidate]
    if args.seed==3 and args.candidate is None:p.error('Seed 3 requires explicit promising candidate selection')
    if not args.report_only:
        commands=[]
        for load in ('light','heavier'):
            cmd=[sys.executable,str(ROOT/'scripts/eval_hard_pact_tradeoff.py'),'--wrench-stress','--event',EVENT,
                '--checkpoint',str(CKPT),'--output',str(root/f'seed{args.seed}_{load}'),'--steps','1000','--seeds',str(args.seed),
                '--load',load,'--variants',*(variants if args.seed==2 else variants+REF)]
            commands.append(cmd);write(root/f'commands_seed{args.seed}.json',commands)
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',SIMULATOR='isaaclab',PYTHONPATH=str(ROOT)),check=True)
    report(root,args.reference.resolve(),args.seed,variants)

if __name__=='__main__':main()
