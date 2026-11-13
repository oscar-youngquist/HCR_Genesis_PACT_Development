"""Production MULE: total iteration budgets; fresh PhysX resume, never exact physics.

Physical payload progression follows the shared disturbance curriculum. The
optimization phase is independent; explicit forward transitions preserve actors.
"""
import argparse
from copy import deepcopy
from dataclasses import replace, asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
TASK='go2_mule_isaaclab'


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--task',choices=(TASK,),default=TASK)
    p.add_argument('--device','--gpu',default='cuda:0')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--seed',type=int)
    p.add_argument('--num-envs','--num_envs',type=int)
    b=p.add_mutually_exclusive_group()
    b.add_argument('--max-iterations','--max_iterations',type=int,help='TOTAL absolute update budget')
    b.add_argument('--additional-iterations',type=int,help='Explicit additional updates on resume')
    p.add_argument('--save-interval',type=int)
    p.add_argument('--log-dir',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--load-operation',choices=('resume','weights_only','stage_transfer'),default='resume')
    p.add_argument('--finetune-learning-rate',type=float)
    p.add_argument('--allow-legacy-checkpoint',action='store_true')
    p.add_argument('--target-stage')
    p.add_argument('--optimization-phase',choices=('nominal_only','joint_adaptation'),help='Fresh runs only')
    p.add_argument('--transition-phase',choices=('joint_adaptation',),help='Explicit forward transition after loading')
    p.add_argument('--payload-mode',choices=('nominal','rolling'),help='Creation-time scene choice, fresh runs only')
    p.add_argument('--follow-disturbance-curriculum',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--auto-joint-adaptation',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--payload-severity',type=float,help='Fresh fixed severity; requires --no-follow-disturbance-curriculum')
    p.add_argument('--smoke',action='store_true',help='Fresh: 4 envs, 4 rollout steps, 1 epoch/batch, at most 2 updates')
    p.add_argument('--smoke-curriculum',action='store_true',help='Test-local accelerated gates/reset/mass cadence; requires --smoke')
    p.add_argument('--dry-run',action='store_true',help='Validate only: no simulator or file writes')
    return p


def resolve_run(args):
    import torch
    from legged_gym.envs.go2.go2_mule.go2_mule_config import resolve_go2_mule,ResolvedGo2MULE
    from legged_gym.envs.go2.go2_mule.mule_rewards import VERSION
    if torch.device(args.device).type!='cuda': raise ValueError('Isaac Lab requires a CUDA device')
    for k in ('seed','max_iterations','additional_iterations','save_interval'):
        if getattr(args,k) is not None and getattr(args,k)<0: raise ValueError(k+' must be nonnegative')
    if args.num_envs is not None and args.num_envs<1: raise ValueError('num_envs must be positive')
    if args.smoke_curriculum and not args.smoke: raise ValueError('Accelerated curriculum is smoke-only')
    if args.finetune_learning_rate is not None and (not math.isfinite(args.finetune_learning_rate) or args.finetune_learning_rate<=0):
        raise ValueError('Fine-tuning LR must be finite and positive')
    if args.load_operation=='stage_transfer' or args.target_stage:
        raise ValueError('No liquid target registered; stage_transfer requires a validated target adapter via the runner API')
    if args.load_operation=='weights_only' and args.finetune_learning_rate is not None:
        raise ValueError('Weights-only does not restore optimizers; LR override requires resume')
    data=None; changes={}
    if args.checkpoint:
        if any(getattr(args,k) is not None for k in ('optimization_phase','payload_mode','payload_severity',
                'follow_disturbance_curriculum','auto_joint_adaptation')) or args.smoke_curriculum:
            raise ValueError('Resume constructs saved physical settings/phase; use --transition-phase for an explicit phase change')
        data=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        c=deepcopy(data.get('resolved_configuration'))
        if not isinstance(c,ResolvedGo2MULE): raise ValueError('Missing saved resolved configuration; explicit legacy migration required')
        if c.shared.mule_reward['version']!=VERSION: raise ValueError('Incompatible MULE clearance/reward contract; explicit migration required')
        c.training.schema.contract()
        if (c.environment.schema!=c.training.schema or
                c.training.schema.clip_observations!=c.shared.normalization.clip_observations):
            raise ValueError('Incompatible MULE observation clipping/schema')
        if 'training_state' not in data and not args.allow_legacy_checkpoint: raise ValueError('Legacy checkpoint requires explicit permission')
        for k in ('seed','num_envs'):
            if getattr(args,k) is not None and getattr(args,k)!=getattr(c.training,k): raise ValueError('Incompatible resume '+k)
        if data['algorithm']['config']!=c.training.resolved(): raise ValueError('Inconsistent checkpoint configuration')
        state=data.get('training_state')
        if state is not None and (state['next_iteration']!=data['algorithm']['iteration'] or state['phase']!=data['algorithm']['phase']):
            raise ValueError('Inconsistent authoritative checkpoint iteration/phase')
        start=0 if args.load_operation=='weights_only' else data['algorithm']['iteration']
        changes['load_operation']=args.load_operation
        if args.transition_phase: changes['optimization_phase']=[data['algorithm']['phase'],args.transition_phase]
    else:
        if args.additional_iterations is not None or args.transition_phase or args.load_operation!='resume' or args.finetune_learning_rate is not None:
            raise ValueError('Load operations, additional budgets and transitions require --checkpoint')
        env={}; train={}; payload={}; curriculum={}
        if args.smoke:
            env['env.num_envs']=4
            # A tiny terrain bank is appropriate only for tiny scenes. Packing
            # 4096 robots into those few spawn tiles exhausts broadphase pairs.
            if args.num_envs is None or args.num_envs<=64:
                env.update({'terrain.num_rows':2,'terrain.num_cols':2,'terrain.max_init_terrain_level':0})
            train.update(num_steps_per_env=4,num_learning_epochs=1,auxiliary_epochs=1,num_mini_batches=1,max_iterations=2)
        if args.smoke_curriculum:
            train['max_iterations']=3
            env.update({'domain_rand.use_domainrand_curriculum':True,
                'domain_rand.use_joint_dynamics_curriculum':False,'domain_rand.use_mass_com_curriculum':False,
                'domain_rand.disabled_curriculum_use_final_bounds':False,
                'domain_rand.use_disturbance_curriculum':True,'domain_rand.push_warmup':0,
                'domain_rand.step_interval':1,'domain_rand.disturbance_progress_delta':.25,
                'domain_rand.min_reward_to_step':0.,'domain_rand.recovery_ratio':0.,'env.episode_length_s':.10})
            payload['mass_change_interval_s']=.02
        if args.num_envs is not None: env['env.num_envs']=args.num_envs
        if args.seed is not None: train['seed']=args.seed
        if args.optimization_phase: train['optimization_phase']=args.optimization_phase
        if args.payload_mode: payload['mode']=args.payload_mode
        for k in ('follow_disturbance_curriculum','auto_joint_adaptation'):
            if getattr(args,k) is not None: curriculum[k]=getattr(args,k)
        if args.payload_severity is not None:
            if args.follow_disturbance_curriculum is not False: raise ValueError('Fixed severity requires disabling curriculum following explicitly')
            payload['initial_severity']=args.payload_severity
        c=resolve_go2_mule(environment_overrides=env,training_overrides=train,payload_overrides=payload,curriculum_overrides=curriculum)
        start=0
    budget=args.max_iterations
    if args.additional_iterations is not None: budget=start+args.additional_iterations
    elif args.smoke and data is not None and budget is None: budget=start+1
    overrides={}
    if budget is not None: overrides['max_iterations']=budget
    if args.save_interval is not None: overrides['save_interval']=args.save_interval
    for k,v in overrides.items(): changes[k]=[getattr(c.training,k),v]
    c.training=replace(c.training,**overrides)
    if c.training.max_iterations<start: raise ValueError('Total budget precedes checkpoint')
    if args.smoke and c.training.max_iterations-start>(3 if args.smoke_curriculum else 2):
        raise ValueError('Smoke permits two updates (three for accelerated curriculum checks)')
    c.manifest['effective_training']=c.training.resolved()
    c.manifest['approved_runtime_changes']=changes
    return c,start,changes


def archive(log_dir,c,args,changes):
    from rsl_rl.utils.training_checkpoint import plain
    def git(*cmd):
        try: return subprocess.check_output(['git','-C',str(ROOT),*cmd],text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):
            # run.sh supplies host-side identity for linked worktrees whose
            # .git administrative directory is intentionally outside the bind.
            return os.environ.get('TRAINING_GIT_SHA' if cmd[0]=='rev-parse' else 'TRAINING_GIT_STATUS','unavailable')
    with tarfile.open(log_dir/'sources.tar.gz','w:gz') as tar:
        for folder in ('rsl_rl','legged_gym','containers/isaaclab'):
            for path in sorted((ROOT/folder).rglob('*')):
                if path.is_file() and path.suffix in ('.py','.sh','.sbatch') and '__pycache__' not in path.parts:
                    tar.add(path,arcname=str(path.relative_to(ROOT)))
    versions={}
    for name in ('torch','isaaclab','isaacsim','numpy'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='unavailable'
    digest=None
    if args.checkpoint:
        h=hashlib.sha256()
        with args.checkpoint.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
        digest=h.hexdigest()
    manifest=dict(version=1,resolved=plain(c),reward=c.shared.mule_reward,schema=c.training.schema.contract(),
        cli=plain(vars(args)),git_sha=git('rev-parse','HEAD'),git_status=git('status','--short'),
        dependencies=versions,parent_sha256=digest,approved_changes=changes,
        resume_semantics='fresh physical reset; no exact PhysX continuation')
    (log_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))


def main(argv=None):
    args=parser().parse_args(argv)
    os.environ.setdefault('SIMULATOR','isaaclab')
    if os.environ['SIMULATOR']!='isaaclab': raise ValueError('MULE requires SIMULATOR=isaaclab')
    import torch
    from legged_gym.envs import task_registry
    c,start,changes=resolve_run(args)
    if args.dry_run:
        print(json.dumps(dict(task=args.task,start_iteration=start,total_iterations=c.training.max_iterations,
            training=c.training.resolved(),payload=asdict(c.payload),approved_changes=changes,simulator_initialized=False),indent=2))
        return
    log_dir=(args.log_dir or ROOT/'logs'/'mule'/datetime.now().strftime('%b%d_%H-%M-%S_mule')).resolve()
    if args.checkpoint and log_dir==args.checkpoint.resolve().parent: raise ValueError('Source run directory is protected')
    log_dir.mkdir(parents=True,exist_ok=False); archive(log_dir,c,args,changes)
    env=runner=None
    try:
        env,_=task_registry.make_env(args.task,args=args,env_cfg=c)
        runner,_=task_registry.make_alg_runner(env,name=args.task,train_cfg=c.training,log_root=log_dir)
        if args.checkpoint:
            runner.load(args.checkpoint,operation=args.load_operation,learning_rate=args.finetune_learning_rate,
                allow_legacy=args.allow_legacy_checkpoint,transition_phase=args.transition_phase)
        torch.cuda.synchronize(args.device); torch.cuda.reset_peak_memory_stats(args.device)
        begun=time.perf_counter(); count=0
        while runner.current_learning_iteration<c.training.max_iterations:
            metrics=runner.learn(1,save_interval=0,final_checkpoint=False)[0]; count+=1
            if any(not math.isfinite(float(v)) for k,v in metrics.items() if k.startswith('loss/')):
                raise FloatingPointError('Nonfinite MULE loss')
            i=runner.current_learning_iteration
            if c.training.save_interval and i%c.training.save_interval==0: runner.save(log_dir/f'model_{i}.pt')
        torch.cuda.synchronize(args.device); elapsed=time.perf_counter()-begun
        final=log_dir/f'model_{runner.current_learning_iteration}.pt'; runner.save(final,stage_end=True)
        if args.smoke_curriculum and count>=3:
            if runner.alg.model.phase!='joint_adaptation' or not env.simulator.payload_active.any():
                raise AssertionError('Accelerated curriculum did not activate at reset and transition at the update boundary')
            if not env.simulator.mass_change_count.sum(): raise AssertionError('Periodic mass resampling was not exercised')
        result=dict(start_iteration=start,end_iteration=runner.current_learning_iteration,updates=count,
            phase=runner.alg.model.phase,curriculum=env.iteration_metrics(),interactions=runner.interactions,
            training_wall_s=elapsed,transitions_per_s=count*c.training.num_envs*c.training.num_steps_per_env/max(elapsed,1e-9),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(args.device),memory_scope='PyTorch allocator, excludes PhysX/Kit',
            checkpoint=str(final),resume_report=getattr(runner,'last_transfer_report',None),
            mass_change_count=int(env.simulator.mass_change_count.sum()),
            metrics={k:float(v) if math.isfinite(float(v)) else None for k,v in runner.last_logged_metrics.items()})
        # Transfer report may contain an explicit unavailable (-inf) gate baseline;
        # it is archived separately, not included as a measurement in strict JSON.
        result['resume_report']=getattr(runner,'resume_report',None)
        (log_dir/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print('MULE_TRAIN_PASS '+json.dumps({k:v for k,v in result.items() if k not in ('metrics','curriculum')}),flush=True)
    except BaseException as error:
        # Kit's exception/shutdown hooks can otherwise hide the original failure.
        detail=traceback.format_exc()
        (log_dir/'failure.txt').write_text(detail)
        print('MULE_TRAIN_FAIL\n'+detail,flush=True)
        return 1
    finally:
        if runner is not None: runner.close()
        if env is not None:
            try: env.close()
            except SystemExit: pass # Kit shutdown must not replace a training exception.


if __name__=='__main__': raise SystemExit(main())
