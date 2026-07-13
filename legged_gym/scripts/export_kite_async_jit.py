"""Export KITE as two independent TorchScript deployment pipelines.

The depth pipeline is intended to run at the camera/preprocessing rate
(typically 10 Hz). The actor pipeline is intended to run at the control rate
(typically 50 Hz), consuming the most recent depth-sequence latent produced by
the slower depth pipeline.
"""

import argparse
import os
import sys

from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR
from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.utils import (
    class_to_dict,
    get_args,
    get_load_path,
    init_genesis,
    task_registry,
)
from rsl_rl.modules import ActorCritic_KITE, export_kite_async_deployment_pipelines


def _parse_export_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--debug_export_untrained",
        action="store_true",
        help=(
            "construct an untrained KITE model from the task config and export "
            "the async TorchScript models next to this script"
        ),
    )
    parser.add_argument(
        "--debug_export_dir",
        type=str,
        default=None,
        help=(
            "optional output directory for --debug_export_untrained; defaults "
            "to this script's directory"
        ),
    )
    export_args, remaining_argv = parser.parse_known_args()
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining_argv
    try:
        args = get_args()
    finally:
        sys.argv = original_argv
    args.debug_export_untrained = export_args.debug_export_untrained
    args.debug_export_dir = export_args.debug_export_dir
    return args


def _prepare_export_env_cfg(env_cfg):
    env_cfg.env.num_envs = 1
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.rendered_envs_idx = [0]
    return env_cfg


def _get_depth_output_resolution(env_cfg):
    depth_cfg = env_cfg.sensor.depth_camera_config
    if hasattr(depth_cfg, "resized_resolution"):
        return tuple(depth_cfg.resized_resolution)

    height, width = depth_cfg.resolution
    decimation = getattr(depth_cfg, "horizontal_decimation", 1)
    return (height, width // decimation)


def _build_untrained_kite_actor_critic(env_cfg, train_cfg, device="cpu"):
    policy_cfg = class_to_dict(train_cfg.policy)
    latest_privileged_obs_dim = (
        env_cfg.env.num_privileged_obs
        if env_cfg.env.num_privileged_obs is not None
        else env_cfg.env.num_observations
    )
    num_critic_obs = (
        latest_privileged_obs_dim
        + policy_cfg.get("privileged_terrain_latent_dim", 32)
        + policy_cfg.get("privileged_dynamics_latent_dim", 16)
    )

    actor_critic = ActorCritic_KITE(
        env_cfg.env.num_observations,
        env_cfg.env.num_obs_hist,
        num_critic_obs,
        _get_depth_output_resolution(env_cfg),
        policy_cfg["depth_image_latent_dim"],
        policy_cfg["depth_image_norm"],
        policy_cfg.get("depth_decoder_norm", "none"),
        policy_cfg.get("depth_image_std_min", 0.01),
        policy_cfg.get("depth_image_std_max", 2.0),
        policy_cfg.get("depth_autoencoder_skip_dropout_prob", 0.25),
        policy_cfg.get("depth_sequence_length", 5),
        policy_cfg.get("depth_sequence_outdim", 16),
        policy_cfg["depth_sequence_norm"],
        policy_cfg.get("depth_sequence_std_min", 0.01),
        policy_cfg.get("depth_sequence_std_max", 1.5),
        policy_cfg.get("depth_sequence_conf_min", 0.1),
        policy_cfg.get("depth_sequence_conf_mask_scale", 0.2),
        policy_cfg["proprio_in_dim"],
        policy_cfg["proprio_latent_dim"],
        policy_cfg.get("proprio_use_norm", True),
        policy_cfg.get("proprio_num_blocks", 2),
        policy_cfg.get("proprio_hidden_dim", 128),
        policy_cfg.get("proprio_token_dim", 128),
        policy_cfg.get("proprio_channel_dim", 256),
        policy_cfg.get("proprio_std_min", 0.01),
        policy_cfg.get("proprio_std_max", 1.5),
        policy_cfg["mixer_velo_dim"],
        policy_cfg["mixer_feet_state_dim"],
        policy_cfg["mixer_latent_dim"],
        policy_cfg["mixer_use_norm"],
        policy_cfg.get("mixer_hidden_dims", [128, 64]),
        policy_cfg.get("mixer_velo_hidden", 32),
        policy_cfg.get("mixer_feet_hidden", 32),
        policy_cfg.get("mixer_std_min", 0.01),
        policy_cfg.get("mixer_std_max", 1.5),
        policy_cfg.get("privileged_terrain_latent_dim", 32),
        policy_cfg.get("privileged_dynamics_latent_dim", 16),
        env_cfg.env.num_actions,
        policy_cfg["actor_layers"],
        policy_cfg["critic_layers"],
        policy_cfg["activation"],
        policy_cfg["init_noise_std"],
        proprio_context_layer_sizes=policy_cfg.get("proprio_context_layer_sizes", [256, 128]),
    )
    return actor_critic.eval().to(device)


def _print_export_summary(depth_path, actor_path):
    print("Exported separate KITE TorchScript models:")
    print(f"  10 Hz depth pipeline: {depth_path}")
    print(f"  50 Hz actor pipeline: {actor_path}")
    print("")
    print("Depth pipeline inputs:")
    print("  depth_image: B x 1 x H x W")
    print("  depth_torso_state: B x 8")
    print("    [roll, pitch, body_velo_xyz, imu_gyro_xyz]")
    print("    body_velo_xyz is predicted when boot-gated on, otherwise simulator/estimator fallback")
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


def export_kite_async_jit(args):
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, args=args)
    env_cfg = _prepare_export_env_cfg(env_cfg)

    if args.debug_export_untrained:
        actor_critic = _build_untrained_kite_actor_critic(
            env_cfg,
            train_cfg,
            device="cpu",
        )
        export_dir = args.debug_export_dir or os.path.dirname(os.path.abspath(__file__))
        depth_path, actor_path = export_kite_async_deployment_pipelines(
            actor_critic,
            export_dir,
            device="cpu",
        )
        print("Exported untrained debug KITE model from config.")
        _print_export_summary(depth_path, actor_path)
        return

    if "genesis" in SIMULATOR:
        from legged_gym import gs

        init_genesis(args, gs)

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

    _print_export_summary(depth_path, actor_path)


if __name__ == "__main__":
    export_kite_async_jit(_parse_export_args())
