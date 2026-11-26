"""Package bounded offline diagnostics; never runs simulation or changes models."""
import argparse,csv,hashlib,json,subprocess,zipfile
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('diagnostics/momentum15000'));args=p.parse_args()
    root=args.root;out=root/'deliverable';out.mkdir(exist_ok=True)
    cases={'standing':'standing_zero_yaw','walking':'walking','full':'full',
           'standing_corrected':'standing_correct_delivery','walking_corrected':'walking_correct_delivery'}
    rows=[];files=[];parity={}
    for case,folder in cases.items():
        run=root/folder/'H_weak_torso_seed3';report=root/(case+'_analysis')
        if not (run/'result.json').exists() or not (report/'errors.csv').exists():
            raise RuntimeError(f'Incomplete requested case: {case}')
        with (report/'errors.csv').open() as f:
            rows.extend(dict(case=case,**r) for r in csv.DictReader(f))
        a=dict(np.load(run/'trace.npz'));T,E=a['execution_strength'].shape
        pairs=set();events=[]
        failures=np.argwhere(a['control_reset'].astype(bool)&a['control_failure'].astype(bool))
        for t,e in failures[:24]:
            lo=max(0,int(t)*4-100);hi=min(T,(int(t)+1)*4+40)
            pairs.update((k,int(e)) for k in range(lo,hi));events.append(dict(kind='failure',step=int(t*4),env=int(e),lo=lo,hi=hi))
        transitions=np.argwhere((a['contact_truth'][1:]!=a['contact_truth'][:-1]).any(-1))
        for t,e in transitions[::max(1,len(transitions)//16)][:16]:
            lo=max(0,int(t)-20);hi=min(T,int(t)+21);pairs.update((k,int(e)) for k in range(lo,hi))
            events.append(dict(kind='contact_transition',step=int(t+1),env=int(e),lo=lo,hi=hi))
        for e in (0,1,2):pairs.update((k,e) for k in range(min(T,200),min(T,300)))
        ix=np.array(sorted(pairs));ti,ei=ix.T
        bounded={'physics_index':ti,'environment_index':ei}
        for k,v in a.items():
            if v.shape[:2]==(T,E):bounded[k]=v[ti,ei]
            elif v.shape[:2]==(T//4,E):bounded[k]=v[ti//4,ei]
        obs=dict(np.load(report/'observer_traces.npz'))
        for k,v in obs.items():
            if v.shape[:2]==(T,E):bounded['observer_'+k]=v[ti,ei]
            elif v.shape[:2]==(T//4,E):bounded['observer_'+k]=v[ti//4,ei]
        target=out/(case+'_bounded.npz');np.savez_compressed(target,**bounded)
        index=out/(case+'_events.json');index.write_text(json.dumps(dict(events=events,total_failures=len(failures),
            failure_capture_drops=max(0,len(failures)-24),sampling='Contact events subsampled to 16; healthy reference samples 200:300 envs0/1/2. 20ms estimates held in this view; original availability indicated by index%4==0.'),indent=2))
        files.extend([(target,target.name),(index,index.name)])
        for name in ('manifest.json','result.json','shadow_capture.json','shadow_realized_parameters.npz','tapes.npz'):
            files.append((run/name,case+'/run/'+name))
        for path in report.iterdir():
            if path.suffix in ('.csv','.json','.png'):files.append((path,case+'/analysis/'+path.name))
        if case=='full':
            oldroot=Path('diagnostics/shadow15000/full/H_weak_torso_seed3');old=np.load(oldroot/'trace.npz')
            for k in ('executed_torque','control_reset','velocity_before','command','shadow_applied_wrench','shadow_force_truth','assembled_seven_weights','execution_strength'):
                parity[k]=bool(np.array_equal(old[k],a[k],equal_nan=True))
            om=json.loads((oldroot/'manifest.json').read_text());nm=json.loads((run/'manifest.json').read_text())
            for k in ('checkpoint_sha256','config_sha256','initial_state_hash','terrain_levels','terrain_types','tapes','effective_qp'):
                parity[k]=om[k]==nm[k]
            parity['old_commit']=om['code_sha'];parity['new_commit']=nm['code_sha']
    with (out/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    compact=[r for r in rows if r['phase']=='common_20ms_after_burnin' and r['frame']=='body' and r['group']=='all']
    (out/'comparison.json').write_text(json.dumps(compact,indent=2))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    models=['neural','FD_control_predicted','oracle_5ms_measured_gain1','oracle_5ms_all_contacts_measured_gain1','estimated_20ms_neural_gain1']
    labels=['Neural','FD estimated','Momentum oracle/feet','Momentum oracle/all contacts','Momentum estimated/neural GRF']
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for j,axis in enumerate(['Fx','Fy','Fz','Mx','My','Mz']):
        lookup={r['model']:float(r['rmse']) for r in compact if r['case']=='full' and r['axis']==axis}
        axs.flat[j].bar(range(len(models)),[lookup[k] for k in models]);axs.flat[j].set_title(axis+(' RMSE [N]' if j<3 else ' RMSE [Nm]'))
        axs.flat[j].set_xticks(range(len(models)),labels,rotation=65,ha='right',fontsize=7)
    fig.suptitle('Randomized terrain: identical 8,767 availability timestamps; original labels (moment delivery caveat)')
    fig.tight_layout();fig.savefig(out/'matched_comparison.png');plt.close(fig)
    (out/'parity.json').write_text(json.dumps(parity,indent=2))
    for path in (root/'fd_analysis').iterdir():
        if path.suffix in ('.csv','.json','.png'):files.append((path,'finite_difference/'+path.name))
    source=list(Path('scripts').glob('*momentum*.py'))+[Path('scripts/hard_pact_shadow_estimation.py'),
        Path('scripts/report_hard_pact_shadow.py'),Path('scripts/eval_hard_pact_tradeoff.py'),Path('tests/test_hard_pact_momentum_observer.py'),
        Path('tests/test_hard_pact_shadow_estimation.py')]
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source}
    (out/'source.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        files=hashes,note='Working-tree evaluation-only extensions. Per-run evaluator hashes are in manifests; final source additionally captures composer readback. No production modification.'),indent=2))
    for f in source:files.append((f,'source/'+str(f)))
    for f in out.iterdir():
        if f.suffix!='.npz' and f.name not in {p.name for p,_ in files}:files.append((f,f.name))
    archive=root/'momentum15000_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file,name in files:z.write(file,name)
    print(archive.resolve(),archive.stat().st_size)


if __name__=='__main__':main()
