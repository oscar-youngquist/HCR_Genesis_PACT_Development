#!/usr/bin/env python3
"""Frozen 15000-checkpoint nominal-preservation/oracle diagnostic; never trains."""
import argparse,csv,json,os,subprocess,sys,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_hard_pact_velocity_stress import metrics,read,write,sha,stats
from scripts.run_hard_pact_wrench_stress import EVENT
CKPT=ROOT/'logs/hardpact_iclr/go2_hard_pact_rss_full/Nov23_01-00-22_hard_pact_rss_full_isaaclab/model_15000.pt'
VARIANTS=['off','A_reference','C_torque30','D_true_wrench']

def summarize(root,seed,variants):
    folders=[root/f'seed{seed}'/f'{v}_seed{seed}' for v in variants]
    manifests=[read(p/'manifest.json') for p in folders];first=manifests[0]
    for k in ('checkpoint_sha256','config_sha256','code_sha','initial_state_hash','terrain_levels','terrain_types','terrain_source_rows','restored_curriculum'):
        assert all(m[k]==first[k] for m in manifests),k
    for k in ('commands','light_wrench','applied','mechanics'):
        assert len({m['tapes'][k] for m in manifests})==1,k
    q=first['effective_qp']
    for m in manifests:
        expected=dict(q,torque_tracking_weight=30. if m['condition']=='C_torque30' else 20.)
        assert m['effective_qp']==expected
        assert m['execution_alpha']==m['extra_strength']==1.
        assert m['substep_probabilities']==[0.,0.,1.]
    assert first['checkpoint_sha256']==sha(CKPT)
    write(root/f'parity_seed{seed}.json',dict(passed=True,manifests=manifests))
    rows=[]
    for folder in folders:
        r=metrics(folder,'checkpoint_envelope')
        with np.load(folder/'trace.npz') as a,np.load(folder/'tapes.npz') as t:
            mask=np.repeat(~a['control_reset'].astype(bool)&~a['control_reasons'][...,6].astype(bool),4,axis=0)
            # Align post-step body velocity with the next pre-step orientation.
            v=a['body_velocity_after'][:-1,:,:3];q=a['orientation_xyzw'][1:]
            c=2*np.cross(q[...,:3],v);world=v+q[...,3:]*c+np.cross(q[...,:3],c)
            r['stability']['world_vertical_velocity']=stats(world[...,2][mask[:-1]])
            r['disturbance']=dict(active_env_seconds=float((np.linalg.norm(a['applied_wrench'],axis=-1)>1e-6).sum()*.005),
                valid_active_env_seconds=r['active_substeps']*.005,
                tape_peak_axes=np.max(abs(t['light']),axis=(0,1)).tolist(),
                tape_planar_peak=float(np.linalg.norm(t['light'][...,:2],axis=-1).max()))
            if r['condition']!='off':
                assert np.all(a['execution_strength']==1.)
                assert r['original_result']['execution_checks']['solves_per_env_control']==4
            if r['condition']=='D_true_wrench':
                assert np.max(a['oracle_initial_frame_error'])<1e-4
                assert np.all(a['oracle_world_at_sample'][::4]==a['oracle_world_at_sample'][1::4])
                r['oracle']=dict(initial_frame_max_error=float(np.max(a['oracle_initial_frame_error'])),
                    qp_input_vs_current_truth=[stats(x) for x in (a['qp_wrench_input_world']-a['applied_wrench'])[mask].T],
                    learned_prediction_preserved=True,hold='yaw-local, control rate; no future state or inertial/gravity residual')
        rows.append(r)
    write(root/f'comparison_seed{seed}.json',rows)
    with (root/f'comparison_seed{seed}.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','failures','failure_rate','failure_free','roll_rms','pitch_rms','height_rms','vertical_v_rms','vx_rmse','vy_rmse','yaw_rmse','active_vx','active_vy','active_yaw','active_progress','active_env_seconds'])
        for r in rows:w.writerow([r['condition'],r['failures'],r['failure_rate'],r['failure_free'],
            *[r['stability'][k]['rms'] for k in ('roll','pitch','height','world_vertical_velocity')],
            *[v['rms'] for v in r['tracking']['all']],*[v['rms'] for v in r['tracking']['active']],
            r['motion']['active']['achieved']['mean'],r['disturbance']['active_env_seconds']])
    with (root/f'objectives_seed{seed}.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','metric','count','mean','rms','p95','max'])
        for r in rows:
            for k,v in r['objectives'].items():w.writerow([r['condition'],k,*[v[n] for n in ('count','mean','rms','p95','max')]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for r in rows:
        label=r['condition'];ax[0].scatter(np.linalg.norm([v['rms'] for v in r['tracking']['all'][:2]]),r['failure_rate'],label=label)
        ax[1].scatter(r['stability']['roll']['rms']+r['stability']['pitch']['rms'],r['failure_rate'],label=label)
        ts=np.linspace(0,20,101);ax[2].step(ts,[np.mean([c or t>x for t,c in zip(r['first_failure_or_censor_s'],r['censored'])]) for x in ts],where='post',label=label)
    ax[0].set(xlabel='Planar tracking RMSE norm [m/s]',ylabel='Failures / env-second');ax[1].set(xlabel='Roll + pitch RMS [rad]',ylabel='Failures / env-second')
    ax[2].set(xlabel='Time [s]',ylabel='Initial-cohort failure-free fraction');ax[2].legend(fontsize=8)
    fig.savefig(root/f'comparison_seed{seed}.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=(2,3),default=2);p.add_argument('--smoke',action='store_true');p.add_argument('--report-only',action='store_true')
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    variants=VARIANTS if a.seed==2 else ['A_reference','C_torque30']
    pin=root/'checkpoint_pin.json';identity=dict(path=str(CKPT),sha256=sha(CKPT),iteration=15000)
    if pin.exists():assert read(pin)==identity
    else:write(pin,identity)
    write(root/'experiment.json',dict(checkpoint=identity,
        conditions={'off':'A: no intervention','A_reference':'B: four full-strength solves',
                    'C_torque30':'C: B with torque weight 30','D_true_wrench':'D: B with held applied external wrench'},
        confirmation='Only B and C, seed 3, if seed-2 C offers promising survival/stability with useful tracking.',
        envelope='Checkpoint-15000 restored current ranges; light is only the legacy tape-selector name.',
        physics='Fresh episodes; no optimizer updates. Oracle is diagnostic only.'))
    cmd=[sys.executable,str(ROOT/'scripts/eval_hard_pact_tradeoff.py'),'--wrench-stress','--event',EVENT,'--checkpoint',str(CKPT),
        '--output',str(root/f'seed{a.seed}'),'--steps','12' if a.smoke else '1000','--seeds',str(a.seed),'--load','light','--variants',*variants]
    write(root/f'command_seed{a.seed}.json',cmd)
    archive=root/'report_sources';archive.mkdir(exist_ok=True)
    for src in (Path(__file__),ROOT/'scripts/run_hard_pact_velocity_stress.py'):shutil.copy2(src,archive/src.name)
    if not a.report_only:subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',SIMULATOR='isaaclab',PYTHONPATH=str(ROOT)),check=True)
    if not a.smoke:summarize(root,a.seed,variants)

if __name__=='__main__':main()
