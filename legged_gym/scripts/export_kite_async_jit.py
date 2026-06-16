"""Export KITE as two independent TorchScript deployment pipelines.

The depth pipeline is intended to run at the camera/preprocessing rate
(typically 10 Hz). The actor pipeline is intended to run at the control rate
(typically 50 Hz), consuming the most recent depth-sequence latent produced by
the slower depth pipeline.
"""

import os

from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR
from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.utils import get_args, get_load_path, init_genesis, task_registry
from rsl_rl.modules import export_kite_async_deployment_pipelines


def _prepare_export_env_cfg(env_cfg):
    env_cfg.env.num_envs = 1
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.rendered_envs_idx = [0]
    return env_cfg


def export_kite_async_jit(args):
    args.headless = True
    if "genesis" in SIMULATOR:
        from legged_gym import gs

        init_genesis(args, gs)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, args=args)
    env_cfg = _prepare_export_env_cfg(env_cfg)

    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.ckpt

    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        "pact_corl",
        train_cfg.runner.experiment_name,
    )
    load_path = get_load_path(
        log_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=train_cfg.runner.checkpoint,
    )

    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )

    train_cfg.runner.resume = False
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    print(f"Loading KITE checkpoint from: {load_path}")
    runner.load(load_path, load_optimizer=False)

    actor_critic = runner.alg.actor_critic.eval()
    export_dir = os.path.join(os.path.dirname(load_path), "exported_async")
    depth_path, actor_path = export_kite_async_deployment_pipelines(
        actor_critic,
        export_dir,
        device="cpu",
    )

    print("Exported separate KITE TorchScript models:")
    print(f"  10 Hz depth pipeline: {depth_path}")
    print(f"  50 Hz actor pipeline: {actor_path}")
    print("")
    print("Depth pipeline inputs:")
    print("  depth_image: B x 1 x H x W")
    print("  depth_torso_state: B x torso_state_dim")
    print("  depth_latent_history: B x (depth_sequence_length - 1) x depth_latent_dim")
    print("Depth pipeline outputs:")
    print("  depth_sequence_latent, updated_depth_latent_history, latest_depth_latent")
    print("")
    print("Actor pipeline inputs:")
    print("  obs: B x num_actor_obs")
    print("  obs_history: B x cenet_in_dim")
    print("  depth_sequence_latent: B x depth_latent_dim")
    print("Actor pipeline outputs:")
    print("  actions, context_latent, body_velo_est, feet_state_est")


if __name__ == "__main__":
    export_kite_async_jit(get_args())
