"""Bounded real rollout/PPO comparison; iteration 0 warms up, no autotuning.

Use the same checkpoint/task/seed/env count for all --runtime-variant runs.
--runtime-reference additionally disables the preceding generation's three
optimizations; leave it unset to isolate the current runtime changes.
Collection/update timing uses the runner's synchronized profiling path. CUDA
memory is Torch allocator memory (cuPIQP/renderer allocations are not included).
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--runtime-reference',action='store_true')
parser.add_argument('--runtime-report',type=Path,required=True)
parser.add_argument('--production-shape',action='store_true',help='Keep production terrain and rollout length; allow 4096 envs')
parser.add_argument('--warmup-iterations',type=int,default=1)
parser.add_argument('--measurement-iterations',type=int,default=2)
parser.add_argument('--rollout-steps',type=int)
parser.add_argument('--runtime-checkpoint',type=Path)
parser.add_argument('--runtime-variant',choices=['reference','buckets','gather','compile','combined','shared'],default='combined')
parser.add_argument('--bucket-alignment',type=int,default=128)
options,remaining=parser.parse_known_args()
sys.argv=[sys.argv[0]]+remaining
from legged_gym.scripts.train_hard_pact import prepare_solver_runtime
prepare_solver_runtime(remaining)
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry


def main():
    args=get_args()
    count=options.warmup_iterations+options.measurement_iterations
    if options.warmup_iterations<0 or options.measurement_iterations<1 or count>5:
        raise ValueError('Require 1..5 total iterations and at least one measurement')
    if args.num_envs is None or args.num_envs>(4096 if options.production_shape else 128):
        raise ValueError('Use --production-shape for up to 4096 environments')
    cfg,train=task_registry.get_cfgs(args.task,args)
    if not options.production_shape:
        cfg.terrain.mesh_type='trimesh';cfg.terrain.num_rows=2;cfg.terrain.num_cols=2
        cfg.terrain.max_init_terrain_level=1
    train.runner.resume=False;train.policy.pretrained_path=None
    if options.rollout_steps is not None or not options.production_shape:
        train.runner.num_steps_per_env=options.rollout_steps or 8
    train.algorithm.profile_bard_timing=True
    train.algorithm.hard_pact_qp.update(warmup_iterations=0,
        cache_rollout_qp_mechanics=not options.runtime_reference,
        scheduled_compact_diagnostics=not options.runtime_reference,
        cupiqp_selective_backward_copy=not options.runtime_reference)
    variant=options.runtime_variant
    train.algorithm.hard_pact_qp.update(
        cupiqp_primary_bucket_policy='tight' if variant in ('buckets','combined','shared') else 'power_of_two',
        cupiqp_bucket_alignment=options.bucket_alignment,
        cupiqp_ppo_primary_capacity_reuse=variant in ('buckets','combined','shared'),
        direct_qp_mechanics_gather=variant in ('gather','combined','shared'),
        compile_tensor_ops=variant=='compile',qp_shared_later_substep=variant=='shared')
    env,_=task_registry.make_env(args.task,args,env_cfg=cfg)
    app=env.simulator._app_launcher.app
    try:
        runner,_=task_registry.make_alg_runner(env,args.task,args,train_cfg=train,
            log_root=tempfile.mkdtemp(prefix='hard_pact_runtime_short_'))
        if options.runtime_checkpoint:
            runner.load(str(options.runtime_checkpoint),load_optimizer=True)
        records=[];original=runner.log
        def log(locs,*a,**kw):
            original(locs,*a,**kw)
            qp=runner.alg.hard_pact_qp
            stages={phase:{k:float(v) for k,v in qp.iteration_diagnostics[phase].sums.items()
                           if k=='real_rows' or k.startswith('final/')}
                    for phase in ('rollout','ppo')}
            records.append(dict(iteration=locs['it'],collection_s=locs['collection_time'],
                update_s=locs['learn_time'],stage_counts_reporting_window=stages,
                backend_counts={name:{phase:dict(values) for phase,values in backend.stats.items()}
                                for name,backend in qp._backend_instances.items()}))
        runner.log=log
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        runner.learn(num_learning_iterations=count,init_at_random_ep_len=True)
        torch.cuda.synchronize()
        report=dict(reference=variant=='reference',legacy_runtime_reference=options.runtime_reference,
            task=args.task,seed=args.seed,num_envs=env.num_envs,
            variant=variant,checkpoint=str(options.runtime_checkpoint),
            warmup_iterations=options.warmup_iterations,
            rollout_steps=train.runner.num_steps_per_env,records=records,
            warmed_mean={k:statistics.mean(r[k] for r in records[options.warmup_iterations:]) for k in ('collection_s','update_s')},
            torch_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu=torch.cuda.get_device_name(),torch_version=torch.__version__)
        options.runtime_report.parent.mkdir(parents=True,exist_ok=True)
        options.runtime_report.write_text(json.dumps(report,indent=2))
        print('RUNTIME_REPORT '+str(options.runtime_report),flush=True)
    finally:
        app.close()


if __name__=='__main__':main()
