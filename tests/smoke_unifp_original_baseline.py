"""Optional one-environment, two-decision simulator check; never calls learn()."""
import torch
import legged_gym.envs
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    args.num_envs = 1
    args.headless = True
    cfg, _ = task_registry.get_cfgs(args.task, args)
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.curriculum = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    try:
        runner, _ = task_registry.make_alg_runner(env, name=args.task, args=args, log_root=None)
        policy = runner.get_inference_policy()
        for _ in range(2):
            obs, history, privileged, labels = env.get_observations()
            assert history.shape == (1,2336) and labels.shape == (1,12)
            with torch.no_grad():
                action = policy(history)
                result = env.step(action)
            assert action.shape == (1,17)
            assert all(torch.isfinite(value).all() for value in result[:6])
        print(f"SMOKE PASSED: {args.task}: two decisions, no learning")
    finally:
        env.simulator._app_launcher.app.close()


if __name__ == "__main__":
    main()
