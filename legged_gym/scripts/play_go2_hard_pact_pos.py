"""Visually inspect a Go2 HardPACTPos checkpoint with random commands."""

import os

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR
from legged_gym.envs import *  # noqa: F403 - imports task registrations
from legged_gym.utils import get_args, init_genesis, task_registry
from legged_gym.utils.helpers import get_load_path


def play(args):
    if args.load_run is None:
        raise SystemExit("--load_run RUN is required (use --ckpt N, or -1 for latest)")
    if SIMULATOR in ("genesis", "genesis_pact", "genesis_pact_pos"):
        import genesis as gs

        init_genesis(args, gs)

    env_cfg, train_cfg = task_registry.get_cfgs(args.task, args)
    env_cfg.env.num_envs = 10
    env_cfg.viewer.rendered_envs_idx = list(range(10))
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 2.0
    env_cfg.commands.ranges.lin_vel_x = [-1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [-0.5, 0.5]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]
    # Ten terrain cells keep visible startup light while retaining the task's
    # normal terrain generator and reward/observation behavior.
    if env_cfg.terrain.mesh_type in ("heightfield", "trimesh"):
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.num_subterrains = 10
        env_cfg.terrain.curriculum = False

    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env, name=args.task, args=args, log_root=None
    )
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "hardpact_iclr",
        train_cfg.runner.experiment_name,
    )
    checkpoint = get_load_path(
        log_root, load_run=args.load_run, checkpoint=args.ckpt
    )
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    obs, obs_history, _, _ = env.get_observations()
    print(f"Playing {checkpoint} with 10 robots; velocity commands resample every 2 s.")

    # Roughly ten episodes; Ctrl-C can be used to switch checkpoints sooner.
    # The environment override is useful for automated smoke checks.
    play_steps = int(os.getenv(
        "HARD_PACT_PLAY_STEPS", 10 * env.max_episode_length
    ))
    with torch.inference_mode():
        for _ in range(play_steps):
            actions = policy(obs, obs_history)
            obs, _, obs_history, _, _, _, _, _ = env.step(actions)


if __name__ == "__main__":
    play(get_args())
