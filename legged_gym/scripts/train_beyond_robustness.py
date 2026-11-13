"""Registered BR training. --max-iterations is TOTAL, --additional-iterations
is explicitly additional on resume. Saved configs construct resumed scenes.
Resume is a fresh physical reset, not exact PhysX continuation.
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))  # Direct container dispatch also works from scripts/.
TASK='go2_beyond_robustness_isaaclab'


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--task',choices=(TASK,),default=TASK)
    p.add_argument('--device','--gpu',default='cuda:0')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--seed',type=int)
    p.add_argument('--num-envs','--num_envs',type=int)
    b=p.add_mutually_exclusive_group()
    b.add_argument('--max-iterations','--max_iterations',type=int,help='TOTAL absolute iteration budget')
    b.add_argument('--additional-iterations',type=int,help='Additional updates; resume only')
    p.add_argument('--save-interval',type=int)
    p.add_argument('--log-dir',type=Path)
    p.add_argument('--checkpoint',type=Path,help='Restore model, all optimizers, RNG and curricula')
    p.add_argument('--load-operation',choices=('resume','stage_transfer','weights_only'),default='resume')
    p.add_argument('--target-stage')
    p.add_argument('--stage-local-initial',type=float,default=0.)
    p.add_argument('--finetune-learning-rate',type=float)
    p.add_argument('--allow-legacy-checkpoint',action='store_true')
    p.add_argument('--training-phase',choices=('teacher_student','student_reinforcement'),help='Fresh runs only')
    p.add_argument('--student-start-iteration',type=int,help='Absolute optimization-phase switch; fresh runs only')
    p.add_argument('--transition-phase',choices=('teacher_student','student_reinforcement'),help='Explicit phase change on resume')
    p.add_argument('--smoke',action='store_true',help='Fresh: 4 envs, 4 steps, 1 epoch/minibatch, 2 updates; resume: 1 extra update')
    p.add_argument('--dry-run',action='store_true',help='Resolve/validate only; no simulator or file writes')
    return p


def resolve_run(args):
    import torch
    from rsl_rl.beyond_robustness_config import resolve_beyond_robustness, BeyondRobustnessResolvedConfig
    if os.environ.get('SIMULATOR','isaaclab')!='isaaclab': raise ValueError('BR requires Isaac Lab')
    if torch.device(args.device).type!='cuda': raise ValueError('Isaac Lab BR requires a CUDA device')
    for key in ('seed','max_iterations','additional_iterations','save_interval','student_start_iteration'):
        if getattr(args,key) is not None and getattr(args,key)<0: raise ValueError(f'{key} must be nonnegative')
    if args.num_envs is not None and args.num_envs<1: raise ValueError('num_envs must be positive')
    if not args.checkpoint and (args.load_operation!='resume' or args.target_stage or args.finetune_learning_rate is not None):
        raise ValueError('Load operations/overrides require --checkpoint')
    if args.finetune_learning_rate is not None and (not math.isfinite(args.finetune_learning_rate) or args.finetune_learning_rate<=0):
        raise ValueError('Fine-tuning LR must be positive and finite')
    if args.load_operation=='stage_transfer':
        raise ValueError('No production liquid target adapter is registered yet. Use the validated runner transfer API with a supported target.')
    if args.target_stage: raise ValueError('Stage changes require stage_transfer')
    data=None; changes={}
    if args.checkpoint:
        if any(getattr(args,k) is not None for k in ('training_phase','student_start_iteration')):
            raise ValueError('Resume restores phase/schedule; use --transition-phase')
        data=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        c=deepcopy(data.get('resolved_configuration'))
        if not isinstance(c,BeyondRobustnessResolvedConfig): raise ValueError('Checkpoint lacks production resolved contract; migration required')
        from rsl_rl.beyond_robustness_rewards import VERSION
        if c.environment.br_reward['version']!=VERSION: raise ValueError('Incompatible BR reward contract; explicit migration required')
        if data.get('curriculum_state',{}).get('version')!=5 or c.environment.domain_rand.persistent_disturbance:
            raise ValueError('Incompatible BR physical curriculum contract; explicit migration required')
        for key,want in (('seed',args.seed),('num_envs',args.num_envs)):
            if want is not None and want!=getattr(c.training,key): raise ValueError(f'Resume {key} differs from checkpoint')
        start=0 if args.load_operation=='weights_only' else int(data['algorithm']['iteration'])
        if data['algorithm']['config']!=c.training.resolved(): raise ValueError('Inconsistent resolved checkpoint training metadata')
        if args.transition_phase is not None: changes['training_phase']=[data['algorithm']['training_phase'],args.transition_phase]
    else:
        if args.additional_iterations is not None or args.transition_phase is not None:
            raise ValueError('Additional budgets and transitions require --checkpoint')
        env={}; train={}
        if args.smoke:
            env.update({'env.num_envs':4,'terrain.num_rows':2,'terrain.num_cols':2,'terrain.max_init_terrain_level':0})
            train.update(num_steps_per_env=4,num_learning_epochs=1,num_mini_batches=1,auxiliary_epochs=1,max_iterations=2)
        if args.num_envs is not None: env['env.num_envs']=args.num_envs
        if args.seed is not None: train['seed']=args.seed
        if args.training_phase is not None: train['training_phase']=args.training_phase
        if args.student_start_iteration is not None: train['student_reinforcement_start_iteration']=args.student_start_iteration
        c=resolve_beyond_robustness(environment_overrides=env,training_overrides=train)
        start=0
    budget=args.max_iterations
    if args.additional_iterations is not None: budget=start+args.additional_iterations
    elif args.smoke and data is not None and budget is None: budget=start+1
    overrides={}
    if budget is not None: overrides['max_iterations']=budget
    if args.save_interval is not None: overrides['save_interval']=args.save_interval
    for key,value in overrides.items():
        if value!=getattr(c.training,key): changes[key]=[getattr(c.training,key),value]
    c.training=replace(c.training,**overrides)
    if c.training.max_iterations<start: raise ValueError('Total budget precedes checkpoint; use --additional-iterations')
    if args.smoke and c.training.max_iterations-start>2: raise ValueError('--smoke permits at most two updates')
    if (args.transition_phase=='teacher_student' and c.training.student_reinforcement_start_iteration is not None
            and start>=c.training.student_reinforcement_start_iteration):
        raise ValueError('Teacher transition conflicts with the saved student-phase schedule')
    c.manifest['effective_training']=c.training.resolved()
    c.manifest.setdefault('approved_runtime_changes',[]).append(changes)
    from legged_gym.envs.go2.go2_hard_pact.domain_rand_curriculum import HardPACTDomainRandCurriculum
    initial_progress=HardPACTDomainRandCurriculum(c.environment,seed=c.training.seed).progress['disturbance']
    c.manifest['start_state']=dict(iteration=start,
        environment_stage=data['curriculum_state']['stage']['stage'] if data else (2 if initial_progress>0 else 1),
        optimization_phase=(args.transition_phase or data['algorithm']['training_phase']) if data else c.training.training_phase)
    return c,start,changes


def archive(log_dir,resolved,args,changes):
    """Archive BR and shared code, without task-directory assumptions."""
    from rsl_rl.beyond_robustness_config import settings_dict
    def git(*cmd): return subprocess.check_output(['git','-C',str(ROOT),*cmd],text=True).strip()
    with tarfile.open(log_dir/'sources.tar.gz','w:gz') as tar:
        for folder in ('rsl_rl','legged_gym','containers/isaaclab'):
            for path in sorted((ROOT/folder).rglob('*')):
                if path.is_file() and path.suffix in ('.py','.sh','.sbatch') and '__pycache__' not in path.parts:
                    tar.add(path,arcname=str(path.relative_to(ROOT)))
    digest=None
    if args.checkpoint:
        h=hashlib.sha256()
        with args.checkpoint.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
        digest=h.hexdigest()
    versions={}
    for key in ('torch','isaaclab','isaacsim','numpy'):
        try: versions[key]=importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError: versions[key]='unavailable'
    manifest=dict(version=1,cli={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        git_sha=git('rev-parse','HEAD'),git_status=git('status','--short'),dependencies=versions,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,checkpoint_sha256=digest,
        approved_changes=changes,resolved=resolved.manifest,input_schema=resolved.training.input_schema.contract(),
        environment=settings_dict(resolved.environment),training=resolved.training.resolved(),
        payload=asdict(resolved.payload),stages=asdict(resolved.stages),reward=resolved.environment.br_reward,
        resume_semantics='fresh physics reset; learning/curriculum/RNG restored; no exact PhysX continuation')
    (log_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))


def main(argv=None):
    args=parser().parse_args(argv)
    os.environ.setdefault('SIMULATOR','isaaclab')
    if os.environ['SIMULATOR']!='isaaclab': raise ValueError('BR production requires SIMULATOR=isaaclab')
    import torch
    from legged_gym.envs import task_registry
    c,start,changes=resolve_run(args)
    if args.dry_run:
        print(json.dumps(dict(task=args.task,device=args.device,start_iteration=start,
            total_iterations=c.training.max_iterations,training=c.training.resolved(),stages=asdict(c.stages),
            approved_changes=changes,simulator_initialized=False),indent=2)); return
    log_dir=(args.log_dir or ROOT/'logs'/'beyond_robustness'/datetime.now().strftime('%b%d_%H-%M-%S_br')).resolve()
    if args.checkpoint and log_dir==args.checkpoint.resolve().parent:
        raise ValueError('Resume must use a new log directory')
    log_dir.mkdir(parents=True,exist_ok=False)
    archive(log_dir,c,args,changes)
    env=runner=None
    try:
        env,_=task_registry.make_env(args.task,args=args,env_cfg=c)
        runner,_=task_registry.make_alg_runner(env,name=args.task,train_cfg=c.training,log_root=log_dir)
        if args.checkpoint:
            runner.load(args.checkpoint,transition_phase=args.transition_phase,
                operation=args.load_operation,learning_rate=args.finetune_learning_rate,
                allow_legacy=args.allow_legacy_checkpoint,target_stage=args.target_stage,
                stage_local_initial=args.stage_local_initial)
        torch.cuda.synchronize(args.device); torch.cuda.reset_peak_memory_stats(args.device)
        begun=time.perf_counter(); count=0; final_metrics={}
        while runner.current_learning_iteration<c.training.max_iterations:
            final_metrics=runner.learn(1,save_interval=0,final_checkpoint=False)[0]; count+=1
            if any(not math.isfinite(float(v)) for k,v in final_metrics.items() if k.startswith('loss/')):
                raise FloatingPointError('Nonfinite BR training loss')
            iteration=runner.current_learning_iteration
            if c.training.save_interval and iteration%c.training.save_interval==0:
                runner.save(log_dir/f'model_{iteration}.pt')
        torch.cuda.synchronize(args.device); elapsed=time.perf_counter()-begun
        final=log_dir/f'model_{runner.current_learning_iteration}.pt'; runner.save(final,stage_end=True)
        clean={k:float(v) if math.isfinite(float(v)) else None for k,v in runner.last_logged_metrics.items()}
        summary=dict(start_iteration=start,end_iteration=runner.current_learning_iteration,updates=count,
            stage=env.stage_state.state_dict(),phase=runner.alg.training_phase,interactions=runner.interactions,
            training_wall_s=elapsed,transitions_per_s=count*c.training.num_envs*c.training.num_steps_per_env/max(elapsed,1e-9),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(args.device),gpu=torch.cuda.get_device_name(args.device),
            memory_scope='PyTorch allocator only, excludes PhysX/Kit',checkpoint=str(final),metrics=clean)
        (log_dir/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print('BR_TRAIN_PASS '+json.dumps({k:v for k,v in summary.items() if k!='metrics'}),flush=True)
    except Exception:
        print('BR_TRAIN_FAIL',flush=True)
        raise
    finally:
        if runner is not None: runner.close()
        if env is not None: env.close()


if __name__=='__main__': main()
