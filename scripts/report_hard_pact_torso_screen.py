#!/usr/bin/env python3
"""Offline torso-objective screen; no simulation, optimization or training imports."""
import argparse,ast,csv,json,shutil
from pathlib import Path
import numpy as np
from scripts.run_hard_pact_velocity_stress import read,write,sha,stats,metrics
from scripts.eval_hard_pact_tradeoff import INTERVENTION_VARIANTS
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'diagnostics/preservation_visual15000/full'
KEYS=('torque_tracking_weight','force_tracking_weight','contact_acceleration_weight',
      'attitude_weight','height_weight','planar_velocity_weight','yaw_rate_weight')


def summarize(a,m,r):
    clean=~a['control_reset'].astype(bool)&~a['control_reasons'][...,6].astype(bool)
    valid=np.repeat(clean,4,axis=0)
    active=np.linalg.norm(a['applied_wrench'],axis=-1)>1e-6
    failure=a['control_reset'].astype(bool)&a['control_failure'].astype(bool)
    err=a['velocity_before']-a['command']
    speed=np.linalg.norm(a['command'][...,:2],axis=-1)
    along=(a['velocity_before'][...,:2]*a['command'][...,:2]).sum(-1)/np.maximum(speed,1e-12)
    # Post-step body velocity uses the next pre-step orientation at the same timestamp.
    v=a['body_velocity_after'][:-1,:,:3];q=a['orientation_xyzw'][1:]
    cross=2*np.cross(q[...,:3],v);world=v+q[...,3:]*cross+np.cross(q[...,:3],cross)
    groups={}
    for name,exposure in [('overall',np.ones_like(active)),('active',active),('inactive',~active)]:
        mask=valid&exposure;controls=clean&exposure[3::4]
        events=failure&exposure[3::4];seconds=float(exposure.sum()*.005)
        row=dict(env_s=seconds,valid_env_s=float(mask.sum()*.005),failures=int(events.sum()),
            failure_rate=float(events.sum()/seconds) if seconds else None,
            tracking=[stats(err[...,i][mask]) for i in range(3)],
            achieved_command_aligned_speed=stats(along[mask&(speed>.1)]),
            commanded_speed=stats(speed[mask&(speed>.1)]),
            roll=stats(a['control_tilt'][...,0][controls]),pitch=stats(a['control_tilt'][...,1][controls]),
            angular_rate=[stats(a['body_velocity_after'][...,3+i][mask]) for i in range(2)],
            height_error=stats((a['control_base_height']-m['environment']['rewards']['base_height_target'])[controls]),
            vertical_velocity=stats(world[...,2][mask[:-1]]),
            correction=stats(a['correction'][mask]),
            causes={key:int((events&a['control_reasons'][...,i]).sum()) for i,key in enumerate(('contact','roll','pitch','height_low','height_high'))},
            solver_counts={key:int(((a['stage']==i)&exposure).sum()) for i,key in enumerate(('primary','recovery','fallback'))})
        x=abs(a['correction'][mask]);row['correction']['p99']=float(np.quantile(x[np.isfinite(x)],.99)) if np.isfinite(x).any() else None
        attempts=sum(row['solver_counts'].values())
        row['solver_rates']={k:v/attempts if attempts else None for k,v in row['solver_counts'].items()}
        accepted=mask&((a['stage']==0)|(a['stage']==1))
        row['clipped_fraction']=float(a['clipped'][accepted].mean()) if 'clipped' in a and accepted.any() else None
        row['violations']={key:dict(**stats(a[key][mask]),fraction=float((a[key][mask]>tol).mean()) if mask.any() else None)
            for key,tol in [('joint_position_violation',1e-4),('joint_velocity_violation',1e-3),('torque_violation',1e-4)]}
        row['objectives']={k:stats(a[k][mask]) for k in a.files if k.startswith('objective_')}
        groups[name]=row
    terrain=[]
    for family in sorted({p['family'] for p in m['target_plan']}):
        envs=np.array([p['family']==family for p in m['target_plan']])
        for level in np.unique(a['control_terrain_level'][:,envs]):
            select=envs[None,:]&(a['control_terrain_level']==level);events=select&failure
            terrain.append(dict(family=family,level=int(level),env_s=float(select.sum()*.02),failures=int(events.sum()),
                causes={key:int((events&a['control_reasons'][...,i]).sum()) for i,key in enumerate(('contact','roll','pitch','height_low','height_high'))}))
    return dict(groups=groups,terrain=terrain,failure_free_fraction=r['failure_free_fraction'],
        episode_lengths=r['completed_episode_length_s'],time_to_first_failure=r['time_to_first_failure_s'],
        capture_count=r.get('failure_captures',r.get('prefailure_windows')),capture_drops=r.get('failure_capture_drops',r.get('prefailure_windows_dropped')))


def event_plots(a,m,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    failure=a['control_reset'].astype(bool)&a['control_failure'].astype(bool)
    index=[]
    for number,(t,e) in enumerate(np.argwhere(failure)[:3]):
        start=max(0,t-25);prev=np.flatnonzero(a['control_reset'][:t,e])
        if len(prev):start=max(start,int(prev[-1])+1)
        sl=slice(start,t+1);ph=slice(start*4,(t+1)*4)
        x=(np.arange(start,t+1)-t)*.02;xp=(np.arange(start*4,(t+1)*4)+1-(t+1)*4)*.005
        fig,axes=plt.subplots(4,1,figsize=(9,9),sharex=True,constrained_layout=True)
        for i,key in enumerate(('roll','pitch')):axes[0].plot(x,np.unwrap(a['control_tilt'][sl,e,i]),label=key)
        axes[0].axhline(.7,ls=':',color='grey');axes[0].axhline(-.7,ls=':',color='grey')
        axes[1].plot(x,a['control_base_height'][sl,e],label='exact termination height');axes[1].axhline(.2,ls=':',color='grey')
        axes[2].plot(xp,np.sqrt((a['correction'][ph,e]**2).mean(-1)),label='applied correction RMS [Nm]')
        for i,key in enumerate(('vx','vy','yaw')):axes[3].plot(xp,(a['velocity_before']-a['command'])[ph,e,i],label=key+' error')
        for ax in axes:ax.axvline(0,color='red',ls='--');ax.legend(fontsize=8);ax.grid(alpha=.2)
        axes[3].set_xlabel('Seconds before original termination; reset sample retained only for event illustration')
        fig.suptitle(f"{m['condition']} env{e} {m['target_plan'][e]['family']} L{a['control_terrain_level'][t,e]}")
        fig.savefig(out/f"{m['condition']}_event{number}.png",dpi=130);plt.close(fig)
        index.append(dict(condition=m['condition'],env=int(e),step=int(t),reasons=a['control_reasons'][t,e].tolist()))
    return index


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.output.resolve();bundle=root/'report';bundle.mkdir(exist_ok=True);plots=bundle/'events';plots.mkdir(exist_ok=True)
    paths=[REFERENCE/'video_seed2'/f'{v}_seed2' for v in ('off','E_rebalanced')]+[root/f'{v}_seed2' for v in ('F_no_torso','G_equal_stability')]
    manifests=[read(p/'manifest.json') for p in paths];ref=manifests[1]
    assert sha(Path(ref['checkpoint']))==ref['checkpoint_sha256']
    assert sha(Path(ref['config']))==ref['config_sha256']
    # The taped exogenous process must be byte-for-byte the same implementation.
    old=REFERENCE/'final_sources/hard_pact_wrench_stress.py';new=ROOT/'scripts/hard_pact_wrench_stress.py'
    trees=[ast.parse(p.read_text()) for p in (old,new)]
    assert ast.dump(next(n for n in trees[0].body if getattr(n,'name',None)=='Tapes'))==ast.dump(next(n for n in trees[1].body if getattr(n,'name',None)=='Tapes'))
    rows=[];events=[]
    for path,m in zip(paths,manifests):
        for key in ('checkpoint_sha256','config_sha256','code_sha','initial_state_hash','terrain_levels','terrain_types','terrain_source_rows','dependencies','restored_curriculum','environment'):
            assert m[key]==ref[key],(m['condition'],key)
        for key in ('commands','light_wrench','heavy_wrench','applied','mechanics'):assert m['tapes'][key]==ref['tapes'][key],key
        assert m['video'] and m['execution_alpha']==m['extra_strength']==1. and m['substep_probabilities']==[0.,0.,1.]
        expected=dict(ref['effective_qp']);expected.update(INTERVENTION_VARIANTS[m['condition']][0])
        if m['condition']=='off':
            # QP-off's unused candidate config is Current; no QP intervention occurs.
            expected=dict(ref['effective_qp']);expected.update(dict(zip(KEYS,[20.,1.,.1,.1,.5,20.,10.])))
        assert m['effective_qp']==expected,m['condition']
        r=read(path/'result.json');assert r['timing']['weights_unchanged']
        with np.load(path/'trace.npz') as a:
            if m['condition']!='off':
                assert np.allclose(a['assembled_seven_weights'],[m['effective_qp'][k] for k in KEYS])
                assert (a['execution_strength']==1).all() and r['execution_checks']['solves_per_env_control']==4
                if m['condition']=='F_no_torso':
                    for phase in ('nominal','candidate','execution'):
                        for key in ('attitude','height'):
                            cost=a[f'objective_{phase}_{key}'];assert np.isfinite(cost).any() and (cost[np.isfinite(cost)]==0).all()
            row=summarize(a,m,r);events+=event_plots(a,m,plots)
        row.update(condition=m['condition'],reused=path in paths[:2],source=str(path),timing=r['timing'])
        rows.append(row)
        for file in ('manifest.json','result.json','video_manifest.json','video_timing.json','history_index.json'):
            if (path/file).exists():shutil.copy2(path/file,bundle/f'{m["condition"]}_{file}')
    write(bundle/'comparison.json',rows);write(bundle/'event_index.json',events)
    with (bundle/'objective_costs_residuals.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','exposure','metric','count','mean','rms','p95_abs','max_abs'])
        for r in rows:
            for group,g in r['groups'].items():
                for key,v in g['objectives'].items():w.writerow([r['condition'],group,key,*[v[k] for k in ('count','mean','rms','p95','max')]])
    with (bundle/'terrain_failures.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','family','level','env_s','failures','overlapping_causes'])
        for r in rows:
            for cell in r['terrain']:w.writerow([r['condition'],cell['family'],cell['level'],cell['env_s'],cell['failures'],json.dumps(cell['causes'])])
    with (bundle/'comparison.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['condition','exposure','env_s','valid_env_s','failures','failure_rate','failure_free','roll_rms','roll_p95','pitch_rms','pitch_p95','wx_rms','wy_rms','height_rms','vz_rms','vx_rmse','vy_rmse','yaw_rmse','aligned_speed','correction_rms','correction_p99','clipped_fraction'])
        for r in rows:
            for name,g in r['groups'].items():w.writerow([r['condition'],name,g['env_s'],g['valid_env_s'],g['failures'],g['failure_rate'],r['failure_free_fraction'] if name=='overall' else None,
                g['roll']['rms'],g['roll']['p95'],g['pitch']['rms'],g['pitch']['p95'],*[v['rms'] for v in g['angular_rate']],g['height_error']['rms'],g['vertical_velocity']['rms'],
                *[v['rms'] for v in g['tracking']],g['achieved_command_aligned_speed']['mean'],g['correction']['rms'],g['correction']['p99'],g['clipped_fraction']])
    write(bundle/'reuse_verification.json',dict(passed=True,checkpoint_sha256=ref['checkpoint_sha256'],config_sha256=ref['config_sha256'],
        code_sha=ref['code_sha'],runtime='Same NumPy/dependencies, renderer, camera calls and warmup; new frames discarded instead of encoded. No physics time advancement.',
        scope='Initial states, mechanics, terrain, tapes, frozen curriculum, QP differences allowlisted. Trajectories/heading feedback diverge after resets.',
        taped_process_ast_identical=True,zero_torso_costs_verified=True))
    shutil.copy2(Path(ref['config']),bundle/'saved_resolved_config.json')

if __name__=='__main__':main()
