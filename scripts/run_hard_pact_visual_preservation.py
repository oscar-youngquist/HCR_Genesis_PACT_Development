#!/usr/bin/env python3
"""Bounded assessment/video/probe orchestration; no training. Physical GPU1 only."""
import argparse,json,os,subprocess,sys,shutil,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_hard_pact_velocity_stress import read,write,sha,metrics,stats
from scripts.run_hard_pact_preservation_eval import CKPT
from scripts.eval_hard_pact_tradeoff import INTERVENTION_VARIANTS
VARIANTS=['off','A_reference','C_torque30','D_true_wrench','E_rebalanced']
OLD=ROOT/'diagnostics/preservation15000/full'
EVENT='events.out.tfevents.1795413624.CICSUNIX-US-003.133079.0'


def run(root,folder,seed,variants,*,video=False,probe=False,steps=1000):
    cmd=[sys.executable,str(ROOT/'scripts/eval_hard_pact_tradeoff.py'),'--wrench-stress',
         '--preservation-visual','--event',EVENT,'--checkpoint',str(CKPT),'--output',str(root/folder),
         '--steps',str(steps),'--seeds',str(seed),'--variants',*variants]
    if video:cmd+=['--video']
    if probe:cmd+=['--recovery-diagnostic','--probe-duration','1','--probe-confirmation','.25','--probe-cap','3']
    write(root/(folder+'_command.json'),cmd)
    subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',SIMULATOR='isaaclab',PYTHONPATH=str(ROOT)),check=True)


def audit(rows):
    for seed in (2,3):
        group=[read(Path(r['path'])/'manifest.json') for r in rows if r['seed']==seed]
        first=group[0]
        for m in group:
            for k in ('checkpoint_sha256','config_sha256','initial_state_hash','terrain_source_rows','terrain_levels','terrain_types','restored_curriculum'):
                assert m[k]==first[k],(seed,k)
            for k in ('commands','applied','mechanics'):assert m['tapes'][k]==first['tapes'][k],(seed,k)
            expected=dict(first['effective_qp']);expected.update(INTERVENTION_VARIANTS[m['condition']][0])
            assert m['effective_qp']==expected,(seed,m['condition'],'non-allowlisted config')
            assert m['execution_alpha']==m['extra_strength']==1. and m['substep_probabilities']==[0.,0.,1.]
    return dict(passed=True,matching='checkpoint/config, initial states, terrain, mechanics, absolute input tapes',
                note='Heading feedback, physical trajectories and delay queues diverge after resets.')


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.report_hard_pact_visual_preservation import report_events,comparison_videos
    inventory=read(root/'inventory.json');rows=[]
    for item in inventory:
        p=Path(item['path']);r=metrics(p,'checkpoint15000');r['reused']=item['reused'];rows.append(r)
        a=np.load(p/'trace.npz')
        if 'terrain_metrics' not in r['original_result']:
            plan=read(p/'manifest.json')['target_plan'];cells=[]
            for family in sorted({x['family'] for x in plan}):
                envs=np.array([x['family']==family for x in plan])
                for level in np.unique(a['control_terrain_level'][:,envs]):
                    mask=envs[None,:]&(a['control_terrain_level']==level)
                    events=mask&a['control_reasons'][...,:5].any(-1)&a['control_reset'].astype(bool)
                    exposure=float(mask.sum()*.02)
                    cells.append(dict(family=family,level=int(level),env_s=exposure,failures=int(events.sum()),
                        failure_per_env_s=float(events.sum()/exposure),overlapping_causes={name:int((events&a['control_reasons'][...,i]).sum())
                        for i,name in enumerate(('contact','roll','pitch','height_low','height_high'))}))
            r['original_result']['terrain_metrics']=cells
        clean=np.repeat(~a['control_reset'].astype(bool)&~a['control_reasons'][...,6].astype(bool),4,axis=0)
        v=a['body_velocity_after'][:-1,:,:3];q=a['orientation_xyzw'][1:]
        cross=2*np.cross(q[...,:3],v);world=v+q[...,3:]*cross+np.cross(q[...,:3],cross)
        r['stability']['world_vertical_velocity']=stats(world[...,2][clean[:-1]])
        r['disturbance_active_env_s']=float((np.linalg.norm(a['applied_wrench'],axis=-1)>1e-6).sum()*.005)
        r['valid_disturbance_active_env_s']=r['active_substeps']*.005
        if r['condition']!='off':
            assert (a['execution_strength']==1).all()
            assert r['original_result']['execution_checks']['solves_per_env_control']==4
            if 'assembled_seven_weights' in a:
                m=read(p/'manifest.json')['effective_qp']
                expected=[m[k] for k in ('torque_tracking_weight','force_tracking_weight','contact_acceleration_weight','attitude_weight','height_weight','planar_velocity_weight','yaw_rate_weight')]
                assert np.allclose(a['assembled_seven_weights'],expected)
        if r['condition']=='D_true_wrench':
            error=float(np.max(a['oracle_initial_frame_error']));assert error<1e-4
            assert np.array_equal(a['oracle_world_at_sample'][::4],a['oracle_world_at_sample'][1::4])
            r['oracle_check']=dict(initial_frame_max_error=error,control_rate_hold_verified=True,
                qp_input_vs_current_label=[stats(x) for x in (a['qp_wrench_input_world']-a['applied_wrench'])[clean].T])
    write(root/'parity.json',audit(rows));write(root/'assessment.json',rows)
    with (root/'objectives.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','seed','metric','count','mean','rms','p95_abs','max_abs'])
        for r in rows:
            for name,value in r['objectives'].items():w.writerow([r['condition'],r['seed'],name,*[value[k] for k in ('count','mean','rms','p95','max')]])
    with (root/'terrain_failures.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','seed','family','level','env_s','failures','failure_rate','overlapping_causes'])
        for r in rows:
            for cell in r['original_result']['terrain_metrics']:
                w.writerow([r['condition'],r['seed'],*[cell[k] for k in ('family','level','env_s','failures','failure_per_env_s')],json.dumps(cell['overlapping_causes'])])
    with (root/'assessment.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','seed','reused','failures','failure/env_s','failure_free','roll_rms','pitch_rms','height_rms','vertical_velocity_rms','vx','vy','yaw','active_vx','active_vy','active_yaw','active_progress'])
        for r in rows:w.writerow([r['condition'],r['seed'],r['reused'],r['failures'],r['failure_rate'],r['failure_free'],
            *[r['stability'][k]['rms'] for k in ('roll','pitch','height','world_vertical_velocity')],*[v['rms'] for v in r['tracking']['all']],
            *[v['rms'] for v in r['tracking']['active']],r['motion']['active']['achieved']['mean']])
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for v in VARIANTS:
        group=[r for r in rows if r['condition']==v]
        axes[0].plot([r['seed'] for r in group],[r['failure_rate'] for r in group],'o-',label=v)
        axes[1].scatter([np.linalg.norm([x['rms'] for x in r['tracking']['all'][:2]]) for r in group],
                        [r['failure_rate'] for r in group],label=v)
    axes[0].set(xlabel='Seed',ylabel='Failures/env-second',xticks=[2,3]);axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Planar tracking RMSE norm [m/s]',ylabel='Failures/env-second')
    fig.savefig(root/'assessment.png',dpi=160);plt.close(fig)
    # Rebuild probe reports from owned traces, including runs collected before
    # the combined wrench/probe writer's early-return fix. No simulation rerun.
    from types import SimpleNamespace
    from scripts.hard_pact_recovery_probe import save_probe
    for directory in (root/'probes_seed2').glob('*_seed2'):
        if not (directory/'result.json').exists():continue
        manifest=read(directory/'manifest.json')
        observer=SimpleNamespace(e=SimpleNamespace(dt=.02),out=directory,manifest=manifest,
            args=SimpleNamespace(capture_cap=24,recovery_diagnostic=True))
        with np.load(directory/'trace.npz') as trace:
            save_probe(observer,trace,read(directory/'result.json'))
    report_events(root,inventory)
    comparison_videos(root,VARIANTS)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('collect','report'),default='collect');a=p.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    pin=read(OLD/'checkpoint_pin.json');assert sha(CKPT)==pin['sha256'];write(root/'checkpoint_pin.json',pin)
    if a.phase=='report':return report(root)
    # Only production change since prior collection is an outer-loss coefficient
    # in the live config; frozen evaluation loads the pinned saved config.
    changed=subprocess.check_output(['git','diff','--name-only','d46f8294ede731df18ce5d30d8e9e7064ef25270','HEAD','--','rsl_rl','legged_gym'],cwd=ROOT,text=True).splitlines()
    assert set(changed)<={'legged_gym/envs/go2/go2_hard_pact/go2_hard_pact_config.py'},changed
    write(root/'reuse_source_audit.json',dict(prior_commit='d46f8294ede731df18ce5d30d8e9e7064ef25270',
        current_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),changed_production_paths=changed,
        resolved_config_source='Pinned saved config, not live task defaults. Final input/config parity asserted in report.'))
    archive=root/'sources';archive.mkdir(exist_ok=True)
    for name in ('run_hard_pact_visual_preservation.py','report_hard_pact_visual_preservation.py','hard_pact_eval_video.py','hard_pact_recovery_probe.py','eval_hard_pact_tradeoff.py','hard_pact_wrench_stress.py'):
        shutil.copy2(ROOT/'scripts'/name,archive/name)
    inventory=[]
    for seed in (2,3):
        for v in VARIANTS:
            old=OLD/f'seed{seed}'/f'{v}_seed{seed}'
            if seed==3 and (old/'result.json').exists():
                target=root/'reused'/f'{v}_seed{seed}'
                if not target.exists():shutil.copytree(old,target)
                inventory.append(dict(condition=v,seed=seed,path=str(target),reused=True,source=str(old)))
            else:
                folder='video_seed2' if seed==2 else 'assessment_seed3'
                inventory.append(dict(condition=v,seed=seed,path=str(root/folder/f'{v}_seed{seed}'),reused=False))
    write(root/'inventory.json',inventory)
    # Seed2 recordings include all three descending-terrain environments, not
    # selected successes. Rebalanced also supplies its new standard assessment.
    run(root,'video_seed2',2,VARIANTS,video=True)
    run(root,'assessment_seed3',3,['off','D_true_wrench','E_rebalanced'])
    run(root,'probes_seed2',2,VARIANTS,probe=True)
    report(root)

if __name__=='__main__':main()
