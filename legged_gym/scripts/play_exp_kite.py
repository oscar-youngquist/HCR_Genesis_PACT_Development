"""Run a trained KITE visual policy in Genesis.

This script is intentionally separate from the older generic play_exp.py
because KITE's actor consumes visual tensors in addition to proprioception:
current depth image, previous depth-frame latents, and the depth torso state.
"""

import argparse
import sys

import numpy as np
import torch

from legged_gym import SIMULATOR
from legged_gym.envs import *  # noqa: F401,F403 - registers tasks
from legged_gym.scripts.joystick import Joystick
from legged_gym.utils import get_args, init_genesis, task_registry


def parse_kite_visualization_args():
    """Parse KITE-only visualization flags before the shared parser runs."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--render-kite-debug",
        action="store_true",
        default=False,
        help="enable depth image, height samples, normal arrows, and camera markers",
    )
    parser.add_argument(
        "--render-depth-image",
        action="store_true",
        default=False,
        help="show the processed depth image for --debug-robot-id",
    )
    parser.add_argument(
        "--render-depth-reconstruction",
        action="store_true",
        default=False,
        help="show actual and reconstructed processed depth images side-by-side",
    )
    parser.add_argument(
        "--depth-reconstruction-scale",
        type=float,
        default=1.0,
        help="display scale multiplier for actual/reconstructed depth images",
    )
    parser.add_argument(
        "--render-height-field",
        action="store_true",
        default=False,
        help="draw green sampled local height-field points for --debug-robot-id",
    )
    parser.add_argument(
        "--render-surface-normals",
        action="store_true",
        default=False,
        help="draw yellow surface-normal arrows for --debug-robot-id",
    )
    parser.add_argument(
        "--draw-depth-camera",
        action="store_true",
        default=False,
        help="draw the depth camera marker and red view-direction arrow",
    )
    parser.add_argument(
        "--print-depth-stats",
        action="store_true",
        default=False,
        help="print depth image min/median/max statistics",
    )
    parser.add_argument(
        "--debug-robot-id",
        type=int,
        default=0,
        help="parallel robot id used for KITE visual debugging",
    )
    parser.add_argument(
        "--normal-length",
        type=float,
        default=0.12,
        help="rendered surface-normal arrow length in meters",
    )
    parser.add_argument(
        "--normal-refresh-steps",
        type=int,
        default=5,
        help="control steps between height/normal overlay refreshes",
    )
    parser.add_argument(
        "--viz-height-offset",
        type=float,
        default=0.02,
        help="vertical offset for height markers and surface normals in meters",
    )
    parser.add_argument(
        "--depth-stats-interval",
        type=int,
        default=25,
        help="depth-stat print interval in depth-camera updates",
    )
    kite_viz_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return kite_viz_args


def override_kite_play_configs(env_cfg, args, kite_viz_args):
    """Keep play lightweight while preserving the trained visual inputs."""
    debug_robot_id = int(kite_viz_args.debug_robot_id)
    if debug_robot_id < 0:
        raise ValueError("--debug-robot-id must be non-negative.")
    if args.num_envs is None:
        env_cfg.env.num_envs = max(1, debug_robot_id + 1)
    elif debug_robot_id >= args.num_envs:
        raise ValueError(
            "--debug-robot-id must be smaller than --num_envs "
            f"(got {debug_robot_id} >= {args.num_envs})."
        )
    else:
        env_cfg.env.num_envs = args.num_envs
    env_cfg.viewer.rendered_envs_idx = [debug_robot_id]

    if hasattr(env_cfg, "sensor"):
        env_cfg.sensor.add_depth = True
        camera_cfg = env_cfg.sensor.depth_camera_config
        enable_all_viz = kite_viz_args.render_kite_debug
        camera_cfg.debug_camera_env_id = debug_robot_id
        camera_cfg.debug_render_depth_image = (
            enable_all_viz or kite_viz_args.render_depth_image
        )
        camera_cfg.debug_draw_camera_position = (
            enable_all_viz or kite_viz_args.draw_depth_camera
        )
        camera_cfg.debug_draw_camera_direction = (
            enable_all_viz or kite_viz_args.draw_depth_camera
        )
        camera_cfg.debug_print_depth_stats = (
            enable_all_viz or kite_viz_args.print_depth_stats
        )
        camera_cfg.debug_depth_stats_interval = max(
            1,
            int(kite_viz_args.depth_stats_interval),
        )

    enable_terrain_viz = (
        kite_viz_args.render_kite_debug
        or kite_viz_args.render_height_field
        or kite_viz_args.render_surface_normals
    )
    env_cfg.terrain.debug_draw_measured_surface_normals = enable_terrain_viz
    env_cfg.terrain.debug_height_map_env_id = debug_robot_id
    env_cfg.terrain.debug_surface_normal_length = float(
        kite_viz_args.normal_length
    )
    env_cfg.terrain.debug_surface_normal_refresh_steps = max(
        1,
        int(kite_viz_args.normal_refresh_steps),
    )
    env_cfg.terrain.debug_height_visualization_offset = max(
        0.0,
        float(kite_viz_args.viz_height_offset),
    )

    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = True
        env_cfg.terrain.border_size = 1.0

        # Default to a visual terrain that exercises the depth pipeline.
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.random_uniform_terrain",
        #     "min_height": -0.05,
        #     "max_height": 0.05,
        #     "step": 0.005,
        #     "downsampled_scale": 0.2,
        # }
        # KITE training terrain presets. Uncomment one block at a time to
        # inspect a specific visual terrain in play mode.
        #
        # # Slope up/down.
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.pyramid_sloped_terrain",
        #     "slope": 0.10,
        #     "platform_size": env_cfg.terrain.platform_size,
        # }
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.pyramid_sloped_terrain",
        #     "slope": -0.35,
        #     "platform_size": env_cfg.terrain.platform_size,
        # }
        #
        # # Random rough terrain.
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.random_uniform_terrain",
        #     "min_height": -0.10,
        #     "max_height": 0.10,
        #     "step": 0.005,
        #     "downsampled_scale": 0.2,
        # }
        #
        # # Stairs down/up.
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.pyramid_stairs_terrain",
        #     "step_width": 0.4,
        #     "step_height": -0.18,
        #     "platform_size": env_cfg.terrain.platform_size,
        # }
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.pyramid_stairs_terrain",
        #     "step_width": 0.4,
        #     "step_height": 0.18,
        #     "platform_size": env_cfg.terrain.platform_size,
        # }
        #
        # # Discrete obstacles.
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.discrete_obstacles_terrain",
        #     "max_height": 0.18,
        #     "min_size": 1.0,
        #     "max_size": 2.0,
        #     "num_rects": 20,
        #     "platform_size": env_cfg.terrain.platform_size,
        # }
        #
        # # Stepping stones.
        env_cfg.terrain.terrain_kwargs = {
            "type": "terrain_utils.stepping_stones_terrain",
            "stone_length": 0.55,
            "stone_width": 0.55,
            "stone_distance_x": 0.25,
            "stone_distance_y": 0.25,
            "max_height": 0.10,
            "platform_size": env_cfg.terrain.platform_size,
            "min_stone_length": 0.20,
            "min_stone_width": 0.20,
            "stepping_stone_edge_clearance": 0.4,
        }
        #
        # # Gap terrain.
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.gap_terrain",
        #     "gap_size": 0.30,
        #     "platform_size": env_cfg.terrain.platform_size,
        # }
        #
        # # Pit terrain.
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.pit_terrain",
        #     "depth": 0.25,
        #     "platform_size": env_cfg.terrain.platform_size,
        # }
        #
        # # Multiple high platforms.
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.multiple_high_platforms_terrain",
        #     "high_platform_height": 0.25,
        #     "high_platform_length": 1.2,
        #     "high_platform_width": 7.0,
        #     "high_platform_interval": 1.4,
        #     "min_high_platform_interval": 0.8,
        #     "min_high_platform_edge_clearance": 0.8,
        #     "platform_size": env_cfg.terrain.platform_size,
        # }
        #
        # # High-platform gaps.
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.high_platform_gaps_terrain",
        #     "high_platform_height": 0.25,
        #     "high_platform_length": 1.8,
        #     "high_platform_width": 7.0,
        #     "high_platform_distance_y": 0.8,
        #     "gap_size": 0.30,
        #     "min_high_platform_track_width": 0.35,
        #     "min_high_platform_edge_clearance": 0.8,
        #     "platform_size": env_cfg.terrain.platform_size,
        # }

    if args.use_joystick:
        env_cfg.commands.heading_command = False

    env_cfg.commands.resampling_time = 5.0
    env_cfg.commands.ranges.lin_vel_x = [-0.5, 0.5]
    env_cfg.commands.ranges.lin_vel_y = [-0.3, 0.3]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]
    env_cfg.commands.ranges.heading = [-3.14, 3.14]

    env_cfg.termination.roll_threshold = 1.57
    env_cfg.termination.pitch_threshold = 1.57
    env_cfg.termination.height_min = 0.0
    env_cfg.asset.fix_base_link = False
    env_cfg.termination.reset_unrecoverable_gaps = True

    # Disable stochastic training disturbances for a cleaner visual-policy run.
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_pd_gain = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_base_mass = False

    if args.record_frames or args.follow_robot:
        env_cfg.viewer.add_camera = True


def _update_commands(env, args, joystick=None):
    """Apply either joystick commands or a simple fixed forward command."""
    if joystick is not None:
        joystick.update()
        env.commands[:, 0] = -joystick.ly
        env.commands[:, 1] = -joystick.lx
        env.commands[:, 2] = -joystick.rx
        env.commands[:, 3] = 0.0
        return

    # env.commands[:, 0] = 1.0
    # env.commands[:, 1] = 0.0
    # env.commands[:, 2] = 0.0
    # env.commands[:, 3] = 0.0


def _follow_robot_camera(env, robot_index=0):
    pos = (
        env.simulator.base_pos[robot_index].detach().cpu().numpy()
        + np.array(env.cfg.viewer.pos, dtype=np.float32)
    )
    lookat = (
        env.simulator.base_pos[robot_index].detach().cpu().numpy()
        + np.array(env.cfg.viewer.lookat, dtype=np.float32)
    )
    env.set_camera(pos, lookat)
    env.simulator._floating_camera.render()


@torch.no_grad()
def interaction_loop(train_cfg, env, policy, args, kite_viz_args):
    """Run the trained KITE visual policy with rollout-time depth memory."""
    (
        obs,
        obs_history,
        _privileged_obs,
        _explicit_labels,
        depth_obs,
        depth_torso_state,
        _terrain_map,
    ) = env.get_observations()

    depth_h, depth_w = getattr(env, "depth_output_resolution", (48, 64))
    depth_sequence_length = getattr(
        train_cfg.policy,
        "depth_sequence_length",
        5,
    )
    depth_latent_dim = train_cfg.policy.depth_image_latent_dim
    depth_latent_history = torch.zeros(
        env.num_envs,
        max(0, depth_sequence_length - 1),
        depth_latent_dim,
        device=env.device,
    )

    joystick = None
    if args.use_joystick:
        joystick = Joystick(joystick_type=args.joystick_type)

    if args.record_frames:
        env.simulator._floating_camera.start_recording()

    for _ in range(int(2.01 * env.max_episode_length)):
        _update_commands(env, args, joystick)

        if args.follow_robot:
            _follow_robot_camera(env)

        depth_image = depth_obs[:, 0:1].reshape(
            env.num_envs, 1, depth_h, depth_w
        )
        actions = policy(
            obs,
            obs_history,
            depth_image=depth_image,
            depth_latent_history=depth_latent_history,
            depth_torso_state=depth_torso_state,
        )
        latest_depth_z = env_policy_latest_depth_latent(
            policy,
            depth_image,
            depth_torso_state,
            render_reconstruction=kite_viz_args.render_depth_reconstruction,
            debug_robot_id=kite_viz_args.debug_robot_id,
            reconstruction_scale=kite_viz_args.depth_reconstruction_scale,
        )

        (
            obs,
            _privileged_obs,
            obs_history,
            _explicit_labels,
            depth_obs,
            depth_torso_state,
            _terrain_map,
            _rews,
            dones,
            _infos,
        ) = env.step(actions.detach())

        # The monolithic inference policy encodes the latest depth image
        # internally but does not return that latent. Recompute only the compact
        # frame latent from the action-time image so the next step receives the
        # same visual history structure used during training.
        if depth_latent_history.shape[1] > 0:
            if depth_latent_history.shape[1] > 1:
                depth_latent_history[:, :-1].copy_(
                    depth_latent_history[:, 1:].clone()
                )
            depth_latent_history[:, -1].copy_(latest_depth_z)
            depth_latent_history[dones.bool()] = 0.0


def env_policy_latest_depth_latent(
    policy,
    depth_image,
    depth_torso_state,
    render_reconstruction=False,
    debug_robot_id=0,
    reconstruction_scale=1.0,
):
    """Fetch the frame latent and optionally render its decoder reconstruction."""
    actor_critic = getattr(policy, "__self__", None)
    if actor_critic is None or not hasattr(actor_critic, "depth_frame_encoder"):
        raise AttributeError(
            "KITE play expects policy to be ActorCritic_KITE.act_inference."
        )
    if render_reconstruction:
        if hasattr(actor_critic, "depth_frame_autoencoder"):
            # Use the training-time autoencoder wrapper for visualization so
            # the rendered reconstruction includes the same U-Net skip path
            # used by the depth-frame reconstruction objective.
            depth_recon, _mean, _logvar, latest_depth_z, _aux = (
                actor_critic.depth_frame_autoencoder(depth_image, depth_torso_state)
            )
        else:
            # Compatibility fallback for older checkpoints created before the
            # U-Net wrapper existed.
            _mean, _logvar, latest_depth_z, depth_aux = (
                actor_critic.depth_frame_encoder(depth_image, depth_torso_state)
            )
            depth_recon, _ = actor_critic.depth_frame_decoder(
                latest_depth_z,
                depth_aux["transform_matrices"],
            )
        _show_depth_reconstruction(
            depth_image,
            depth_recon,
            int(debug_robot_id),
            float(reconstruction_scale),
        )
        return latest_depth_z.detach()

    return actor_critic.depth_frame_encoder.forward_inference(
        depth_image,
        depth_torso_state,
    ).detach()


def _show_depth_reconstruction(
    depth_image,
    depth_recon,
    debug_robot_id,
    reconstruction_scale=1.0,
):
    """Show actual processed depth next to the decoder reconstruction."""
    if not 0 <= debug_robot_id < depth_image.shape[0]:
        return

    import cv2 as cv

    actual = depth_image[debug_robot_id, 0].detach().clamp(0.0, 1.0)
    recon = depth_recon[debug_robot_id, 0].detach().clamp(0.0, 1.0)
    actual_pixels = (actual * 255.0).byte().cpu().numpy()
    recon_pixels = (recon * 255.0).byte().cpu().numpy()
    separator = np.full(
        (actual_pixels.shape[0], 4),
        127,
        dtype=actual_pixels.dtype,
    )
    comparison = np.concatenate(
        (actual_pixels, separator, recon_pixels),
        axis=1,
    )
    scale = max(1.0e-3, float(reconstruction_scale))
    if scale != 1.0:
        comparison = cv.resize(
            comparison,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv.INTER_NEAREST,
        )
    cv.imshow(
        f"KITE Depth actual | recon env {debug_robot_id}",
        comparison,
    )
    cv.waitKey(1)


def play(args, kite_viz_args):
    if args.task == "go2":
        args.task = "go2_kite"

    if "genesis" in SIMULATOR:
        from legged_gym import gs

        init_genesis(args, gs)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, args=args)
    override_kite_play_configs(env_cfg, args, kite_viz_args)

    train_cfg.runner.resume = True
    train_cfg.runner.checkpoint = args.ckpt
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    policy = runner.get_inference_policy(device=env.device)

    interaction_loop(train_cfg, env, policy, args, kite_viz_args)

    if args.record_frames:
        filename_mp4 = f"{train_cfg.runner.experiment_name}_kite_visual_play.mp4"
        env.simulator._floating_camera.stop_recording(
            save_to_filename=filename_mp4,
            fps=30,
        )
        print("Saved recording to " + filename_mp4)

    print("KITE visual play complete.")


if __name__ == "__main__":
    kite_viz_args = parse_kite_visualization_args()
    play(get_args(), kite_viz_args)
