"""One four-step PPO update on two Isaac Lab robots; no checkpoints saved.

SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python tests/smoke_b1z1_local_feedforward.py --task b1z1_pact --gpu cuda:0 --headless
"""
import json
import os
# Avoid forking a compile worker pool after Kit has started many threads.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
import torch
import legged_gym.envs
from legged_gym.utils import get_args, task_registry


def main(num_envs=2, steps=4):
    args=get_args();args.num_envs=num_envs;args.headless=True
    cfg,train=task_registry.get_cfgs(args.task,args)
    cfg.compliance.enabled=True;cfg.terrain.mesh_type='plane';cfg.terrain.curriculum=False
    train.policy.pretrained_path=None;train.runner.resume=False
    train.runner.num_steps_per_env=steps
    train.algorithm.num_learning_epochs=1;train.algorithm.num_mini_batches=1
    train.algorithm.representation_pinn_enabled=False
    train.algorithm.pinn_init_steps=0;train.algorithm.pinn_warmup=1
    train.algorithm.actor_phys_arm_feedforward_enabled=True
    train.algorithm.actor_phys_arm_feedforward_compliant_local_enabled=True
    train.algorithm.actor_phys_sample_fraction=1.
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    try:
        print('Constructing smoke runner',flush=True)
        runner,_=task_registry.make_alg_runner(env,name=args.task,args=args,train_cfg=train,log_root=None)
        print('Running one smoke PPO update',flush=True)
        runner.alg.pinn_updates=1
        captured={}
        update=runner.alg.update
        def check_update(*args,**kwargs):
            packet=runner.alg.storage.actor_physics
            assert packet['local_ref_valid'].any()
            assert torch.isfinite(packet['local_ref_acceleration']).all()
            captured['valid_reference_rows']=int(packet['local_ref_valid'].sum())
            values=update(*args,**kwargs)
            captured.update({k:float(v) for k,v in values.items() if 'arm_feedforward' in k})
            return values
        runner.alg.update=check_update
        runner.learn(1)
        assert all(torch.isfinite(p).all() for p in runner.actor_critic.parameters())
        assert captured['ActorPhysics/arm_feedforward_raw'] > 0
        assert captured['ActorPhysics/arm_feedforward_valid_fraction'] > 0
        print('LOCAL_TEACHER_SMOKE '+json.dumps(captured,sort_keys=True),flush=True)
    except BaseException:
        # Kit shutdown can exit before Python unwinds: print failures first.
        import traceback
        traceback.print_exc()
        raise
    finally:
        env.simulator._app_launcher.app.close()


if __name__=='__main__':main()
