"""Bounded real rollout/PPO comparison; iteration 0 warms up, no autotuning.

Use the same task/seed/env count for default and --runtime-reference runs.
Only benchmark startup/terrain sizes and the three optimization switches change.
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
options,remaining=parser.parse_known_args()
sys.argv=[sys.argv[0]]+remaining
from legged_gym.scripts.train_hard_pact import prepare_solver_runtime
prepare_solver_runtime(remaining)
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry


def main():
    args=get_args()
    if args.num_envs is None or args.num_envs>128 or args.max_iterations is None or not 2<=args.max_iterations<=5:
        raise ValueError('Bounded benchmark requires --num_envs <=128 and --max_iterations 2..5')
    cfg,train=task_registry.get_cfgs(args.task,args)
    cfg.terrain.mesh_type='trimesh';cfg.terrain.num_rows=2;cfg.terrain.num_cols=2
    cfg.terrain.max_init_terrain_level=1
    train.runner.resume=False;train.policy.pretrained_path=None
    train.runner.num_steps_per_env=8
    train.algorithm.profile_bard_timing=True
    train.algorithm.hard_pact_qp.update(warmup_iterations=0,
        cache_rollout_qp_mechanics=not options.runtime_reference,
        scheduled_compact_diagnostics=not options.runtime_reference,
        cupiqp_selective_backward_copy=not options.runtime_reference)
    env,_=task_registry.make_env(args.task,args,env_cfg=cfg)
    app=env.simulator._app_launcher.app
    try:
        runner,_=task_registry.make_alg_runner(env,args.task,args,train_cfg=train,
            log_root=tempfile.mkdtemp(prefix='hard_pact_runtime_short_'))
        records=[];original=runner.log
        def log(locs,*a,**kw):
            original(locs,*a,**kw)
            qp=runner.alg.hard_pact_qp
            stages={phase:{k:float(v) for k,v in qp.iteration_diagnostics[phase].sums.items()
                           if k=='real_rows' or k.startswith('final/')}
                    for phase in ('rollout','ppo')}
            records.append(dict(iteration=locs['it'],collection_s=locs['collection_time'],
                update_s=locs['learn_time'],stage_counts_reporting_window=stages))
        runner.log=log
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        runner.learn(num_learning_iterations=args.max_iterations,init_at_random_ep_len=True)
        torch.cuda.synchronize()
        report=dict(reference=options.runtime_reference,task=args.task,seed=args.seed,num_envs=env.num_envs,
            rollout_steps=8,records=records,
            warmed_mean={k:statistics.mean(r[k] for r in records[1:]) for k in ('collection_s','update_s')},
            torch_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu=torch.cuda.get_device_name(),torch_version=torch.__version__)
        options.runtime_report.parent.mkdir(parents=True,exist_ok=True)
        options.runtime_report.write_text(json.dumps(report,indent=2))
        print('RUNTIME_REPORT '+str(options.runtime_report),flush=True)
    finally:
        app.close()


if __name__=='__main__':main()
