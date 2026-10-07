"""Eight-environment real Isaac Lab reset/control check; no PPO update/training."""
import sys
from legged_gym.scripts.train_hard_pact import prepare_solver_runtime
prepare_solver_runtime(sys.argv[1:])
import json
import tempfile
import time
from dataclasses import replace
import torch
from legged_gym.envs import *  # register actual tasks
from legged_gym.utils import get_args, task_registry


def main():
    args=get_args();cfg,train=task_registry.get_cfgs(args.task,args)
    cfg.terrain.mesh_type="trimesh"
    cfg.terrain.num_rows=2;cfg.terrain.num_cols=2;cfg.terrain.max_init_terrain_level=1
    train.algorithm.hard_pact_qp.update(warmup_iterations=0,diagnostics_level="minimal",cuda_event_profiling=False)
    train.runner.resume=False;train.policy.pretrained_path=None
    train.runner.num_steps_per_env=4
    command_pair = str(getattr(args,'qp_update_mode','')).startswith('command_pair')
    if command_pair:
        train.algorithm.num_learning_epochs=1
        train.algorithm.num_mini_batches=1
        train.algorithm.num_encoder_epochs=1
        train.algorithm.ppo_qp_shard_percentage=100.
        train.algorithm.hard_pact_qp.update(qp_update_mode=args.qp_update_mode,
            torque_rate_constraint_weight=0.,correction_ramp_enabled=False,
            objective_curriculum_enabled=False)
    env,_=task_registry.make_env(args.task,args,env_cfg=cfg)
    app=env.simulator._app_launcher.app
    try:
        runner,_=task_registry.make_alg_runner(env,args.task,args,train_cfg=train,
            log_root=tempfile.mkdtemp(prefix="hard_pact_modes_control_"))
        qp=runner.alg.hard_pact_qp
        qp.diagnostics_scheduled=True
        original=qp.solve;counts=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
        stages=torch.zeros(3,device=env.device,dtype=torch.long)
        def solve(**kw):
            ids=kw["environment_ids"]
            counts[ids]+=1
            if kw.get('diagnostics_phase')=='ppo':
                return original(**kw)
            assert not torch.is_grad_enabled()
            out=original(**kw)
            stages.add_(torch.bincount(out.stage,minlength=3))
            assert torch.isfinite(out.tau_safe).all()
            return out
        qp.solve=solve
        callback=env._solve_hard_pact_rollout_qp_substep
        in_substep=[False]
        def prohibit_network(*_):
            assert not in_substep[0], "Network forward executed inside physics substepping"
        hooks=[module.register_forward_pre_hook(prohibit_network)
               for module in runner.alg.actor_critic.modules()]
        simulate=env.simulator.step
        def simulation_step(*a,**kw):
            in_substep[0]=True
            try:return simulate(*a,**kw)
            finally:in_substep[0]=False
        env.simulator.step=simulation_step
        def checked(*a):
            prev=env._hard_pact_previous_substep_torque.clone()
            callback(*a)
            tau=env._hard_pact_previous_substep_torque
            assert (tau.abs()<=qp.torque_limits+1e-6).all()
            if qp.cfg.torque_rate_constraint_weight>0:
                assert ((tau-prev).abs()<=qp.cfg.torque_rate_limit_nm_s*cfg.sim.dt+1e-5).all()
        env._solve_hard_pact_rollout_qp_substep=checked
        report={}
        if command_pair:
            # Exactly one short normal-loop update exercises compact storage,
            # current-policy recomputation, solver leases and PPO backward.
            backward_rows=[]
            solve_with_count=qp.solve
            def replay_checked(**kw):
                if kw.get('diagnostics_phase')=='ppo':
                    out=original(**kw)
                    if out.tau_safe.requires_grad:
                        out.tau_safe.register_hook(lambda g: backward_rows.append(g.detach().clone()))
                    return out
                return solve_with_count(**kw)
            qp.solve=replay_checked
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            runner.learn(1)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            expected=(env.num_envs*16 if args.qp_update_mode.endswith('every_substep')
                      else 4*(env.num_envs+round(env.num_envs*qp.cfg.command_pair_second_solve_fraction)))
            assert counts.sum()==expected,counts
            assert backward_rows and all(g.isfinite().all() for g in backward_rows)
            assert any(g.abs().sum()>0 for g in backward_rows)
            assert all(p.grad is None or p.grad.isfinite().all() for p in runner.alg.actor_critic.parameters())
            print('COMMAND_PAIR_SMOKE_PASS '+json.dumps(dict(
                environments=env.num_envs,control_steps=4,primary_invocations_per_environment=counts.tolist(),
                stages=stages.tolist(),finite_nonzero_qp_backward=True,
                constraint_horizon_s=qp.cfg.constraint_prediction_horizon_s,
                inner_velocity_horizon_s=qp.cfg.qp_velocity_objective_horizon_s)),flush=True)
            print('SMOKE_RESOURCES '+json.dumps(dict(seconds=elapsed,
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())),flush=True)
            return
        for mode,expected in (("every_substep",4),("random_one_substep",1)):
            env.set_hard_pact_qp_enabled(False)
            with torch.inference_mode():env.reset()
            qp.cfg=replace(qp.cfg,qp_update_mode=mode)
            env.set_hard_pact_qp_enabled(True)
            counts.zero_();stages.zero_()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            for _ in range(3):
                obs,hist,priv,_=env.get_observations()
                with torch.inference_mode():
                    actions=runner.alg.act(obs,priv,hist,*env.get_prev_obs())
                    result=env.step(actions)
                assert all(torch.isfinite(t).all() for t in (actions,result[0],result[4],result[-1]))
                assert env._qp_interval_solve_count.eq(expected).all()
                bins=torch.bincount(env._qp_sampled_substep_index.long(),minlength=4)
                assert bins.max()-bins.min()<=1
            assert counts.eq(3*expected).all(),counts
            torch.cuda.synchronize()
            report[mode]={"problems_per_environment":counts.tolist(),"stages":stages.tolist(),
                          "three_control_intervals_wall_ms":1000*(time.perf_counter()-started),
                          "torch_cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                          "torch_cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                          "network_calls_during_substeps":0}
        print("QP_MODES_CONTROL_SMOKE_PASS "+json.dumps(report),flush=True)
    finally:
        app.close()


if __name__=="__main__":main()
