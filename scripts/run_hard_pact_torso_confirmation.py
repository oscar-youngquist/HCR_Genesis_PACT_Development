#!/usr/bin/env python3
"""Six bounded, sequential GPU-0 evaluations and matched-seed offline report."""
import argparse,ast,csv,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.report_hard_pact_torso_screen import summarize,event_plots,KEYS,REFERENCE
from scripts.run_hard_pact_velocity_stress import read,write,sha
from scripts.eval_hard_pact_tradeoff import INTERVENTION_VARIANTS

def inventory(root):
    rows=[]
    for seed in (2,3,4):
        for v in ('off','E_rebalanced','H_weak_torso'):
            reused=seed<4 and v!='H_weak_torso'
            folder=REFERENCE/('video_seed2' if seed==2 else 'assessment_seed3') if reused else root/f'seed{seed}'
            rows.append((seed,v,folder/f'{v}_seed{seed}',reused))
        if seed<4:
            folder=ROOT/'diagnostics/torso_screen15000/full' if seed==2 else root/'seed3'
            rows.append((seed,'F_no_torso',folder/f'F_no_torso_seed{seed}',seed==2))
    return rows

def check(m,ref):
    for k in ('checkpoint_sha256','config_sha256','code_sha','initial_state_hash','terrain_levels','terrain_types','terrain_source_rows','dependencies','restored_curriculum','environment','video'):
        assert m[k]==ref[k],(m['condition'],m['seed'],k)
    for k in ('commands','light_wrench','heavy_wrench','applied','mechanics'):
        assert m['tapes'][k]==ref['tapes'][k],(m['condition'],k)
    assert m['steps']==1000 and not m['recovery_diagnostic']
    assert m['execution_alpha']==m['extra_strength']==1 and m['substep_probabilities']==[0,0,1]
    expected=dict(ref['effective_qp']);expected.update(INTERVENTION_VARIANTS[m['condition']][0])
    if m['condition']=='off':expected.update(dict(zip(KEYS,[20,1,.1,.1,.5,20,10])))
    assert m['effective_qp']==expected,(m['condition'],'objective allowlist')

def preflight(root):
    ref=read(REFERENCE/'video_seed2/E_rebalanced_seed2/manifest.json')
    assert sha(Path(ref['checkpoint']))==ref['checkpoint_sha256']
    assert sha(Path(ref['config']))==ref['config_sha256']
    old=REFERENCE/'final_sources/hard_pact_wrench_stress.py'
    trees=[ast.parse(p.read_text()) for p in (old,ROOT/'scripts/hard_pact_wrench_stress.py')]
    assert ast.dump(next(n for n in trees[0].body if getattr(n,'name',None)=='Tapes'))==ast.dump(next(n for n in trees[1].body if getattr(n,'name',None)=='Tapes'))
    for seed,v,path,reused in inventory(root):
        if reused:
            baseline=REFERENCE/('video_seed2' if seed==2 else 'assessment_seed3')/f'E_rebalanced_seed{seed}'
            check(read(path/'manifest.json'),read(baseline/'manifest.json'))
    return ref

def collect(root,ref):
    jobs=[(3,['F_no_torso']),(2,['H_weak_torso']),(3,['H_weak_torso']),(4,['H_weak_torso','off','E_rebalanced'])]
    commands=[]
    for seed,variants in jobs:
        cmd=[sys.executable,str(ROOT/'scripts/eval_hard_pact_tradeoff.py'),'--physical-gpu','0','--wrench-stress','--preservation-visual',
             '--event',Path(ref['event']).name,'--checkpoint',ref['checkpoint'],'--steps','1000','--seeds',str(seed),
             '--output',str(root/f'seed{seed}'),'--variants',*variants]
        if seed==2:cmd+=['--discard-video']
        commands.append(cmd)
    write(root/'commands.json',commands)
    for cmd in commands:
        subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,HARDPACT_EVAL_NUMPY_PATH='/tmp/hardpact_eval_numpy126',CUDA_VISIBLE_DEVICES='0'),check=True)

def flatten(row,group):
    g=row['groups'][group]
    return dict(failures=g['failures'],failure_rate=g['failure_rate'],env_s=g['env_s'],valid_env_s=g['valid_env_s'],
        roll_rms=g['roll']['rms'],roll_p95=g['roll']['p95'],pitch_rms=g['pitch']['rms'],pitch_p95=g['pitch']['p95'],
        wx_rms=g['angular_rate'][0]['rms'],wy_rms=g['angular_rate'][1]['rms'],height_rms=g['height_error']['rms'],vz_rms=g['vertical_velocity']['rms'],
        vx_rmse=g['tracking'][0]['rms'],vy_rmse=g['tracking'][1]['rms'],yaw_rmse=g['tracking'][2]['rms'],
        aligned_speed=g['achieved_command_aligned_speed']['mean'],correction_rms=g['correction']['rms'],correction_p99=g['correction']['p99'],clipped_fraction=g['clipped_fraction'])

def report(root):
    bundle=root/'report';bundle.mkdir(exist_ok=True);rows=[];flat=[];events=[]
    for seed,v,path,reused in inventory(root):
        baseline=next(p for s,c,p,_ in inventory(root) if s==seed and c=='E_rebalanced')
        m=read(path/'manifest.json');check(m,read(baseline/'manifest.json'));r=read(path/'result.json')
        assert r['timing']['weights_unchanged']
        plots=bundle/'events'/f'{v}_seed{seed}';plots.mkdir(parents=True,exist_ok=True)
        with np.load(path/'trace.npz') as a:
            if v!='off':
                assert np.allclose(a['assembled_seven_weights'],[m['effective_qp'][k] for k in KEYS])
                assert (a['execution_strength']==1).all() and r['execution_checks']['solves_per_env_control']==4
            row=summarize(a,m,r);events+=event_plots(a,m,plots)
        row.update(seed=seed,condition=v,source=str(path),reused=reused,timing=r['timing']);rows.append(row)
        for group in row['groups']:flat.append(dict(seed=seed,condition=v,exposure=group,**flatten(row,group)))
        for file in ('manifest.json','result.json','history_index.json'):
            if (path/file).exists():shutil.copy2(path/file,bundle/f'{v}_seed{seed}_{file}')
    write(bundle/'comparison.json',rows);write(bundle/'event_index.json',events)
    def csvfile(name,values):
        with (bundle/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    csvfile('comparison.csv',flat)
    terrain=[];objectives=[]
    for r in rows:
        for cell in r['terrain']:
            terrain.append(dict(seed=r['seed'],condition=r['condition'],**{**cell,'causes':json.dumps(cell['causes'])}))
        for group,g in r['groups'].items():
            for name,values in g['objectives'].items():
                objectives.append(dict(seed=r['seed'],condition=r['condition'],exposure=group,metric=name,**values))
    csvfile('terrain_failures.csv',terrain);csvfile('objective_costs_residuals.csv',objectives)
    pairs=[]
    for candidate,seeds in [('H_weak_torso',(2,3,4)),('F_no_torso',(2,3))]:
        for baseline in ('off','E_rebalanced'):
            for seed in seeds:
                for group in ('overall','active','inactive'):
                    x=next(r for r in flat if (r['seed'],r['condition'],r['exposure'])==(seed,candidate,group))
                    b=next(r for r in flat if (r['seed'],r['condition'],r['exposure'])==(seed,baseline,group))
                    pairs.append(dict(candidate=candidate,baseline=baseline,seed=seed,exposure=group,
                        **{k:x[k]-b[k] if x[k] is not None and b[k] is not None else None for k in x if k not in ('seed','condition','exposure')}))
    csvfile('paired_differences.csv',pairs);write(bundle/'paired_differences.json',pairs)
    write(bundle/'reuse_verification.json',dict(passed=True,scope='Checkpoint/config, initial state, terrain, mechanics, tapes, dependencies, curriculum and allowlisted weights matched within each seed.',
        runtime='Seed2 retains renderer calls (new frames discarded); seeds3/4 headless. Cached runs physical GPU1, new GPU0; GPU/PhysX numerical divergence remains possible. Metadata package version may differ from actual NumPy import override.',
        caveats='Heading feedback and trajectories diverge after asynchronous resets. Modeled execution retains optimized GRFs. No causal/significance claim from 2/3 seeds. No unmatched-seed pooling.'))
    shutil.copy2(root/'commands.json',bundle/'commands.json')
    shutil.copy2(Path(read(next(p for s,v,p,_ in inventory(root) if s==2 and v=='off')/'manifest.json')['config']),bundle/'saved_resolved_config.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,(group,key) in zip(axes,[('overall','failure_rate'),('active','pitch_rms'),('active','wy_rms')]):
        for v in ('off','E_rebalanced','H_weak_torso','F_no_torso'):
            rr=[r for r in flat if r['condition']==v and r['exposure']==group]
            ax.plot([r['seed'] for r in rr],[r[key] for r in rr],'o-',label=v)
        ax.set_title(group+' '+key);ax.set_xticks([2,3,4]);ax.set_xlabel('Seed');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.savefig(bundle/'paired_overview.png',dpi=150);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'diagnostics/torso_confirmation15000/full');p.add_argument('--report-only',action='store_true');args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True);ref=preflight(root)
    if not args.report_only:collect(root,ref)
    report(root)

if __name__=='__main__':main()
