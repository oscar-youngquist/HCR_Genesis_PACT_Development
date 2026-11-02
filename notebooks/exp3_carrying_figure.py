"""Export the measured-support PILOT/A6 carrying figure and a portable ZIP.

Read only existing Exp2 numeric recordings. Reuse Exp3's reader and commanded
geometry FK, but never call momentum reconstruction, dynamics, or inference.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

import exp1_analysis as shared
import exp2_analysis as exp2
from b1z1_paper_names import DISPLAY_NAMES
from exp3_support import RecordedFK
from exp3_representation import chunks, file_hash, rotate
import exp3_carrying_render as renderer

TASKS = ('b1z1_pact', 'b1z1_pact_ab9_representation')
FIELDS = ('trial_id','step','time_seconds','successor_time_seconds','task_stage_interval',
          'task_stage_successor','valid','alive_for_metrics','physical_failure','artificial_reset',
          'ee_force_applied_world_interval_mean','grf_raw_world_interval_mean','ff_weighted_motor_mean_square',
          'ee_position_world','ee_reference_world','position_target_interval_mean','base_position_t',
          'base_quaternion_t','ee_reference_t_world','command_nominal_t')


def schedule(meta):
    cfg=meta['evaluation']
    boundaries=[0.,cfg['move_start_s'],cfg['move_start_s']+cfg['velocity_ramp_s'],
                *cfg['load_times_s'][:3],cfg['stop_s'],cfg['stop_s']+cfg['velocity_ramp_s'],
                cfg['load_times_s'][3],cfg['finish_s']]
    if np.any(np.diff(boundaries)<=0): raise ValueError('Invalid recorded carrying stage ordering')
    baseline=[boundaries[2],boundaries[3]]
    # Stored names may say load45/load65 even when numeric settings changed.
    loads=[0.,0.,0.,*cfg['loads_n'][:3],cfg['loads_n'][2],cfg['loads_n'][2],cfg['loads_n'][3]]
    motion=['Standing','Starting','Moving','Moving / EE Reposition','Moving','Moving','Stopping','Standing','Standing']
    stages=[dict(id=i,start_s=a,end_s=b,load_n=load,motion=move,
                 recorded_label=meta['task_stages'][str(i)])
            for i,(a,b,load,move) in enumerate(zip(boundaries[:-1],boundaries[1:],loads,motion))]
    events=[dict(time_seconds=t,kind='load' if i<3 else 'release',load_n=cfg['loads_n'][i])
            for i,t in enumerate(cfg['load_times_s'])]
    events.append(dict(time_seconds=cfg['stop_s'],kind='stop'))
    return dict(stage_boundaries_seconds=boundaries,baseline_interval_seconds=baseline,stages=stages,
                events=sorted(events,key=lambda e:e['time_seconds']))


def extract(run):
    meta=run['meta']; env=meta['resolved_environment']
    if env['terrain']['mesh_type']!='plane':
        raise ValueError('This figure requires recorded flat-ground normal support')
    missing=set(FIELDS)-set(run['records']['fields'])
    if missing: raise ValueError(f'{run["path"]}: missing required fields {sorted(missing)}')
    names=meta['dof_names']; feet=env['asset']['foot_name']
    regions={r:np.array([i for i,n in enumerate(names) if n.startswith('z1_')==(r=='arm')]) for r in ('arm','leg')}
    feet_ids={r:[i for i,n in enumerate(feet) if n.startswith(prefix)]
              for r,prefix in (('front',('FR','FL')),('rear',('RR','RL')))}
    if any(len(v)==0 for v in [*regions.values(),*feet_ids.values()]): raise ValueError('Unknown joint/foot mapping')
    model=Path(meta['urdf_source'])
    if file_hash(model)!=meta['urdf_sha256']: raise ValueError('Commanded-geometry URDF hash mismatch')
    fk=RecordedFK(model,env['asset']['base_name'],env['asset']['gripper_name'])
    substeps=round(meta['dt']/meta['physics_dt'])
    if substeps!=env['control']['decimation'] or not np.isclose(substeps*meta['physics_dt'],meta['dt']):
        raise ValueError('Cannot recover valid joint-substep denominators')
    info=schedule(meta); lo,hi=info['baseline_interval_seconds']; pieces=[]; last={}
    for _,d in chunks(run,FIELDS):
        if not np.allclose(d['successor_time_seconds']-d['time_seconds'],meta['dt'],atol=1e-7,rtol=0):
            raise ValueError('Recorded interval durations disagree with the control timestep')
        for tid in np.unique(d['trial_id']):
            steps=d['step'][d['trial_id']==tid]
            if int(tid) not in run['trials'] or steps[0]<=last.get(int(tid),-1) or np.any(np.diff(steps)<=0):
                raise ValueError('Unknown/duplicate/out-of-order trial transition')
            last[int(tid)]=steps[-1]
        for field,stage in (('time_seconds','task_stage_interval'),('successor_time_seconds','task_stage_successor')):
            if not np.array_equal(exp2.stage_ids(d[field],meta['evaluation']),d[stage]):
                raise ValueError('Recorded stages disagree with timestamps/metadata')
        good=exp2.valid_rows(d)
        base=good & (d['time_seconds']>=lo-1e-8) & (d['successor_time_seconds']<=hi+1e-8)
        if not np.allclose(d['ee_force_applied_world_interval_mean'][base],0,atol=1e-6):
            raise ValueError('Nominal unloaded baseline contains applied loading')
        command=meta['evaluation']['profiles']['carrying']['velocity_m_s']
        if not np.allclose(d['command_nominal_t'][base,0],command,atol=1e-6):
            raise ValueError('Baseline is not steady commanded locomotion')
        force=d['ee_force_applied_world_interval_mean']
        expected=np.array([s['load_n'] for s in info['stages']])[d['task_stage_interval']]
        if not np.allclose(force[good,:2],0,atol=1e-6) or not np.allclose(-force[good,2],expected[good],atol=1e-5):
            raise ValueError('Actual carrying load disagrees with numeric metadata')
        values={k:d[k] for k in ('step','time_seconds','successor_time_seconds','valid','alive_for_metrics','physical_failure','artificial_reset')}
        values.update(trial=d['trial_id'],stage_interval=d['task_stage_interval'],stage_successor=d['task_stage_successor'],load_n=-force[:,2])
        normal=d['grf_raw_world_interval_mean'][:,:,2]
        for region,ids in feet_ids.items(): values[region+'_normal_n']=normal[:,ids].sum(1)
        for region,ids in regions.items():
            squared=d['ff_weighted_motor_mean_square'][:,ids].astype(np.float64)
            finite=np.isfinite(squared)&(squared>=0)
            # Collector stores mean of per-substep squares, after branch/motor
            # scaling and BEFORE total-torque clipping; reconstruct its sum/count.
            values[region+'_ff_sumsq']=np.where(finite,squared,0.).sum(1)*substeps
            values[region+'_ff_count']=finite.sum(1).astype(np.int32)*substeps
        command_world=d['base_position_t']+rotate(fk.batch(d['position_target_interval_mean'],names),d['base_quaternion_t'])
        values['geometry_error_m']=np.linalg.norm(command_world-d['ee_reference_t_world'],axis=1)
        values['ee_error_m']=np.linalg.norm(d['ee_position_world']-d['ee_reference_world'],axis=1)
        pieces.append(values)
    if not pieces: raise ValueError('Empty completed recording')
    data={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    if set(data['trial'])!=set(run['trials']): raise ValueError('Recorded trial inventory mismatch')
    order=np.lexsort((data['step'],data['trial']))
    return {k:v[order] for k,v in data.items()},dict(
        joints=names,arm_joints=[names[i] for i in regions['arm']],leg_joints=[names[i] for i in regions['leg']],
        feet=feet,foot_groups=feet_ids,substeps_per_interval=substeps)


README = """# PILOT / A6 Carrying: Measured Support and Compensation

Run `python render_figure.py` from this directory (or `--package PATH`). Requires
Python 3.10+, NumPy, pandas, and Matplotlib; dependency_versions.json records the
tested versions. No repository, checkpoint, URDF, GPU, or simulator is required.
Edit plot_config.json to change fonts, labels, colors, limits, legends, bin width,
or baseline interval, then rerun. PDF/SVG/300-dpi PNG and CSVs are regenerated.
Times New Roman is preferred; DejaVu Serif is the documented fallback, recorded
in render_report.json. SVG text remains editable; PDF embeds subset TrueType fonts.

## Definitions and Timing

Panel A uses measured front/rear world-z normal support ONLY, minus each trial's
unloaded-locomotion baseline. The gray curve is the magnitude of recorded downward
EE loading (not a support estimate). No momentum reconstruction is used.
Panel B is weighted/motor-scaled requested feedforward torque RMS, minus the
trial's baseline RMS, in N*m. These are channel usage, NOT clipped executed torque
or a unique compensation decomposition. All named arm/leg DoFs used by existing
Exp3 support analysis are retained, including zero-FF gripper/inactive channels.
RMS = sqrt(sum(substep torque squares)/count(joint-substeps)); RMS values are
never averaged to merge time bins. Native square/count fields retain that ability.
Panel C shows actual successor EE error versus its successor reference, and
URDF FK(mean executed/delayed position target) at measured base_t versus reference_t.
This commanded-geometry error is the existing Exp3 approximation: nonlinear FK
of the mean command is not mean substep FK, and base motion during the interval
is not integrated. The robot model was verified during export, not needed to plot.

Bins are common across methods, split at every stage boundary. Interval-average
loads/forces/torques are assigned by midpoint; geometry by t; actual error by t+dt.
At 0.02 s control resolution, 0.25 s bins alternate native sample counts; no
substep interpolation/proration is invented. Every finite pre-failure sample is
retained. A bin spanning a gap is excluded; no failed tails or missing bins are
filled. A baseline must be contiguous and complete by default. Short/failed
baselines remain unavailable, not zero; absolute EE errors remain available.
Optional odd smoothing_bins re-pools neighboring bins' sufficient statistics
within contiguous stage segments BEFORE computing RMS; default 1 disables it.

Compute trial baselines and bin values first, average trials within evaluation
seed, then weight evaluation seeds equally. Bands are sample SD (ddof=1) of those
seed means, not trial SD, confidence intervals, or independently trained policies.
One seed has undefined SD. Counts include contributing and surviving-at-bin-end
trials; partial pre-failure bins may contribute without survival through bin end.

## Files

- data/*.npz: native-resolution numeric sufficient statistics; source metadata
  supplies method/checkpoint/evaluation seed for every row. No observation/history
  tensors, checkpoints, raw joint trajectories, or contact caches are included.
- metadata.json: provenance, frames/units, stages, numeric load values, verified
  baseline interval, exclusions, joint/foot ordering, data dictionary and hashes.
- baselines.csv; plotted_trials.csv.gz; plotted_seeds.csv; plotted_summary.csv:
  unrounded statistics and counts at each aggregation level.
- render_report.json: settings, font used, excluded gap bins and missing baselines.

These are repeated evaluations of one fixed checkpoint per method. Recorded stage
names are not authoritative for force values; use numeric stages in metadata.json.
The plotted normal forces omit tangential friction and are not complete GRFs.
"""


def build(roots,output,seeds=(0,1,2),workers=2,checkpoints=None):
    start=time.perf_counter(); output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    (output/'COMPLETE').unlink(missing_ok=True)
    (output/'data').mkdir(exist_ok=True)
    sources=[]; seen=set(); protocols=set(); hashes={}; info=None; diagnostics=[]
    for path in shared.discover(roots,'metadata.json'):
        if not (path.parent/'COMPLETE').is_file(): continue
        meta=shared.read_json(path); task=meta.get('task'); args=meta.get('arguments',{})
        if meta.get('experiment')!=2 or task not in TASKS or args.get('profile')!='carrying' or args.get('loading')!='loaded': continue
        if meta['evaluation_seed'] not in seeds: continue
        if checkpoints and task in checkpoints and meta['checkpoint_sha256']!=checkpoints[task]: continue
        key=(task,meta['evaluation_seed'])
        if key in seen: raise ValueError(f'Duplicate method/seed {key}; select checkpoint hashes or narrower roots')
        seen.add(key); hashes.setdefault(task,set()).add(meta['checkpoint_sha256'])
        if len(hashes[task])!=1: raise ValueError('Do not mix checkpoints within a method')
        protocols.add(shared.fingerprint(exp2.protocol(meta)))
        if len(protocols)!=1: raise ValueError('Incompatible recorded protocols')
        info=schedule(meta)
        records=shared.read_json(path.parent/'records.json')
        trials={t['trial_id']:t for t in shared.read_json(path.parent/'trials.json')}
        run=dict(meta=meta,records=records,trials=trials,path=path.parent,_num_workers=workers)
        print(f'Extracting {DISPLAY_NAMES[task]}, evaluation seed {meta["evaluation_seed"]}: {path.parent}',flush=True)
        data,mapping=extract(run)
        relative=f'data/{task}-seed-{meta["evaluation_seed"]}.npz'
        np.savez_compressed(output/relative,**data)
        sources.append(dict(method=DISPLAY_NAMES[task],task=task,evaluation_seed=meta['evaluation_seed'],
            training_seed=meta.get('checkpoint_training_seed'),checkpoint_sha256=meta['checkpoint_sha256'],
            checkpoint_recorded_path=meta['checkpoint'],recording=str(path.parent.resolve()),data_file=relative,
            data_sha256=file_hash(output/relative),metadata_sha256=file_hash(path),records_sha256=file_hash(path.parent/'records.json'),
            urdf_sha256=meta['urdf_sha256'],dt=meta['dt'],physics_dt=meta['physics_dt'],
            source_commit=meta['git_commit'],source_dirty=meta['git_dirty'],mapping=mapping,
            trials=len(trials),rows=len(data['trial']),invalid_or_failure_rows=int((~exp2.valid_rows(data)).sum()),
            trial_ids=sorted(trials)))
    if not sources or set(hashes)!=set(TASKS): raise ValueError('Completed carrying recordings for BOTH PILOT and A6 are required')
    for task in TASKS:
        missing=sorted(set(seeds)-{s for t,s in seen if t==task})
        if missing: diagnostics.append(f'{task}: missing evaluation seeds {missing}; no fabricated results')
    if {s for t,s in seen if t==TASKS[0]}!={s for t,s in seen if t==TASKS[1]}:
        raise ValueError('PILOT and A6 evaluation seed coverage differs')
    root=Path(__file__).resolve().parents[1]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip())
    dictionary={
        'trial':'Trial ID within source method/checkpoint/evaluation seed; never a training seed',
        'step':'Control interval index; discontinuities delimit gaps',
        'time_seconds':'Interval start t [s]', 'successor_time_seconds':'Physical successor t+dt [s]',
        'stage_interval':'Stage at t; numeric stage definitions below', 'stage_successor':'Stage at t+dt',
        'valid':'Collector transition-valid flag', 'alive_for_metrics':'Pre-failure tracking mask',
        'physical_failure':'Failure at this successor; excluded from statistics', 'artificial_reset':'Reset/teleport mask',
        'load_n':'Negative of actual applied interval-average world Fz [N], positive downward',
        'front_normal_n':'Sum of front-foot raw normal world-z interval means [N]',
        'rear_normal_n':'Sum of rear-foot raw normal world-z interval means [N]',
        'ee_error_m':'Euclidean measured successor EE minus nominal successor reference [m]',
        'geometry_error_m':'Existing Exp3 FK(mean delayed command), measured base_t/reference_t approximation [m]',
        'arm_ff_sumsq':'Sum of weighted/motor-scaled requested arm FF torque squared over joint-substeps [(N*m)^2]',
        'leg_ff_sumsq':'Same for leg joints [(N*m)^2]',
        'arm_ff_count':'Number of finite arm joint-substeps in sumsq',
        'leg_ff_count':'Number of finite leg joint-substeps in sumsq'}
    metadata=dict(schema_version=1,figure='Experiment 3 Carrying: PILOT / A6',**info,sources=sources,
        protocol_hash=next(iter(protocols)),code_commit=commit,code_dirty=dirty,
        code_sha256={p.name:file_hash(p) for p in (Path(__file__),Path(renderer.__file__))},
        data_dictionary=dictionary,frames='World z-up normal forces; geometry/actual EE differences in world coordinates',
        aggregation='Trial-specific baseline and bins -> equal trials within evaluation seed -> equal seeds, sample SD ddof=1',
        timing='Interval midpoint for force/torque, interval start for geometry, successor for actual EE error; bins split at stages',
        exclusions='Invalid/reset/failure and post-failure samples; no gap-spanning bins; incomplete/gapped baselines unavailable',
        interpretation='Measured normal support only, no reconstructed momentum forces. FF RMS is requested channel usage, not clipped execution.',
        baseline_verified='Actual applied EE loads zero; native forward commands at steady prescribed locomotion',
        diagnostics=diagnostics)
    shared.write_json(output/'metadata.json',metadata)
    shared.write_json(output/'plot_config.json',renderer.DEFAULT_CONFIG)
    shutil.copyfile(renderer.__file__,output/'render_figure.py')
    (output/'README.md').write_text(README)
    import matplotlib
    shared.write_json(output/'dependency_versions.json',dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__))
    renderer.render(output)
    shared.write_json(output/'build_report.json',dict(seconds=time.perf_counter()-start,rows=sum(s['rows'] for s in sources),
        recordings=len(sources),policy_inference=False,simulator=False))
    (output/'COMPLETE').write_text('Complete\n')
    archive=archive_package(output)
    print(f'Created {archive} ({archive.stat().st_size/1024**2:.2f} MiB)',flush=True)
    return archive


def archive_package(output):
    output=Path(output); archive=output.with_suffix('.zip')
    # Explicit inventory prevents old caches/unrelated results from entering ZIP.
    files=[output/name for name in ('render_figure.py','plot_config.json','metadata.json','README.md',
        'dependency_versions.json','build_report.json','COMPLETE','render_report.json','baselines.csv',
        'plotted_trials.csv.gz','plotted_seeds.csv','plotted_summary.csv',
        'exp3_carrying_compact.pdf','exp3_carrying_compact.svg','exp3_carrying_compact.png')]
    meta=shared.read_json(output/'metadata.json'); files += [output/s['data_file'] for s in meta['sources']]
    if (output/'validation.json').exists(): files.append(output/'validation.json')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files: z.write(path,path.relative_to(output))
    return archive


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,action='append')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--checkpoint',action='append',default=[],help='task=full SHA256 (optional selection)')
    args=parser.parse_args()
    build(args.root,args.output,args.seed or (0,1,2),args.workers,dict(x.split('=',1) for x in args.checkpoint))
