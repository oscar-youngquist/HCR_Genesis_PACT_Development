"""Two robots, four control steps, one PPO update; no checkpoint or long training.

SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python tests/smoke_b1z1_recruitment_ablations.py --task b1z1_pact_ab13_workspace --gpu cuda:0 --headless
"""
import json
import os
import time

os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
import torch
import legged_gym.envs
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    args.num_envs, args.headless = 2, True
    cfg, train = task_registry.get_cfgs(args.task, args)
    # Test-local size/schedule only; retain the selected variant's losses/gates.
    cfg.terrain.mesh_type, cfg.terrain.curriculum = 'plane', False
    cfg.compliance.compile_geometry = False
    cfg.domain_rand.push_robots = False
    cfg.commands.push_gripper_stators = cfg.commands.push_robot_base = True
    cfg.commands.apply_ee_external_forces = True
    cfg.commands.apply_base_external_forces = cfg.commands.apply_base_external_torques = True
    cfg.commands.compliance_disturbance_mode_weights = None
    train.policy.pretrained_path, train.runner.resume = None, False
    train.runner.num_steps_per_env = 4
    train.algorithm.num_learning_epochs, train.algorithm.num_mini_batches = 1, 1
    train.algorithm.pinn_init_steps, train.algorithm.pinn_warmup = 0, 1
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    try:
        start = time.perf_counter()
        runner, _ = task_registry.make_alg_runner(env, name=args.task, args=args, train_cfg=train, log_root=None)
        runner.alg.pinn_updates = 1
        # Diagnostic held disturbances bypass production sampling, not physics.
        # Keep exact source values so we can detect duplicate or gated application.
        ee = torch.tensor([0., 10., 0.], device=env.device).expand(2, -1)
        torso = torch.tensor([5., 0., 0.], device=env.device).expand(2, -1)
        couple = torch.tensor([0., 0., 1.], device=env.device).expand(2, -1)
        def apply_ee(ids):
            env.ee_force_ext_world.copy_(ee)
            env.simulator.apply_ee_force(ee)
        def apply_torso(ids):
            env.base_force_ext_world.copy_(torso)
            env.base_torque_ext_world.copy_(couple)
            env.simulator.apply_base_force(torso)
            env.simulator.apply_base_torque(couple)
        env._push_gripper, env._push_robot_base = apply_ee, apply_torso
        env.compliance.reference.profile.fill_(1)
        step = env.step
        def checked_step(actions):
            result = step(actions)
            assert not result[5].any(), 'Unexpected reset in four-step load diagnostic'
            torch.testing.assert_close(env.simulator._external_force_world.sum(1), ee + torso)
            torch.testing.assert_close(env.simulator._external_torque_world.sum(1), couple)
            if cfg.ablation_variant == 14:
                ref = env.compliance.target
                assert ref.gates.eq(0).all() and ref.ee_wrench.eq(0).all()
                torch.testing.assert_close(ref.combined_wrench, ref.torso_wrench)
                assert ref.torso_wrench.norm() > 0
            return result
        env.step = checked_step
        metrics = {}
        update = runner.alg.update
        def checked_update(*args, **kwargs):
            result = update(*args, **kwargs)
            metrics.update({k: float(v) for k, v in result.items()})
            assert all(torch.isfinite(torch.as_tensor(v)).all() for v in result.values())
            return result
        runner.alg.update = checked_update
        runner.learn(1)
        model = runner.actor_critic
        assert all(torch.isfinite(p).all() for p in model.parameters())
        with torch.no_grad():
            model.reset()
            one = model.act_inference(env.obs_buf, env.obs_history)
            model.reset()
            two = model.act_inference(env.obs_buf, env.obs_history)
            torch.testing.assert_close(one, two, rtol=0, atol=0)
        print('ABLATION_SMOKE ' + json.dumps(dict(task=args.task, device=str(env.device), applied_loads_checked=True,
              seconds=time.perf_counter()-start, metrics=metrics)), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        env.simulator._app_launcher.app.close()


if __name__ == '__main__':
    main()
