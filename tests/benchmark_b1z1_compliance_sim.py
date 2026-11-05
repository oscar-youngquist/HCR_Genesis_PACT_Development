"""Short, matched rollout benchmark, not a training-throughput claim.

Run disabled/enabled in separate processes with the same seed, environment
count and actions. Includes current reward/config changes, excludes PPO/BARD
backward and policy inference. Prints phase timings and peak Torch allocation.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import torch


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--compliance',action='store_true')
    parser.add_argument('--steps',type=int,default=16)
    parser.add_argument('--warmup',type=int,default=4)
    parser.add_argument('--output',required=True)
    options,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0]]+remaining
    import legged_gym.envs
    from legged_gym.utils import get_args,task_registry
    from rsl_rl.utils.rollout_timing import RolloutPhaseTimer
    args=get_args();args.headless=True
    cfg,training=task_registry.get_cfgs(args.task,args)
    cfg.compliance.enabled=options.compliance
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    try:
        from legged_gym.envs.b1z1.b1z1_pact import arm_feedforward_reward
        if arm_feedforward_reward.enabled(env):
            # The disabled baseline retains its configured BARD reward. Attach
            # the same backend directly; do not allocate an optimizer/rollout.
            from legged_gym import LEGGED_GYM_ROOT_DIR
            from legged_gym.dynamics.bard_b1z1_dynamics import BardB1Z1DynamicsBackend
            backend=BardB1Z1DynamicsBackend(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
                cfg.asset.dof_names,cfg.asset.foot_name,cfg.asset.gripper_name,cfg.asset.base_name,
                device=torch.device(env.device),batch_capacity=training.algorithm.bard_batch_capacity or env.num_envs)
            arm_feedforward_reward.configure(env,backend)
        env.reset()
        action=torch.zeros(env.num_envs,cfg.env.num_policy_actions,device=env.device)
        for _ in range(options.warmup):env.step(action)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        timer=RolloutPhaseTimer(env.device);env._rollout_phase_timer=timer
        reference_events=[]
        if env.compliance is not None:
            begin=env.compliance.begin
            def measured_begin():
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();begin();end.record();reference_events.append((start,end))
            env.compliance.begin=measured_begin
        started=time.perf_counter();failures=action.new_zeros(())
        with torch.no_grad():
            for _ in range(options.steps):
                result=env.step(action);failures+=result[5].sum()
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        phases=timer.finish()
        out=dict(compliance=options.compliance,environments=env.num_envs,steps=options.steps,
            seed=args.seed,wall_ms_per_step=1000*elapsed/options.steps,
            reference_ms_per_step=sum(a.elapsed_time(b) for a,b in reference_events)/options.steps,
            phase_ms_per_step={k:v/options.steps for k,v in phases.items()},resets=int(failures.item()),
            peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            terrain=cfg.terrain.mesh_type,gpu=torch.cuda.get_device_name(0))
        Path(options.output).write_text(json.dumps(out,indent=2))
        print('ROLLOUT_BENCHMARK '+json.dumps(out),flush=True)
    except BaseException:
        import traceback
        traceback.print_exc();raise
    finally:env.simulator._app_launcher.app.close()


if __name__=='__main__':main()
