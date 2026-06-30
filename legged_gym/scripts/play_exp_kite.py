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
from legged_gym.utils.math_utils import quat_rotate_inverse


# File-local joystick command limits for KITE play mode. The stick axes are
# normalized to [-1, 1] and then scaled into these command ranges.
JOYSTICK_LIN_VEL_X_RANGE = (-0.75, 0.75)
JOYSTICK_LIN_VEL_Y_RANGE = (-0.3, 0.3)
JOYSTICK_ANG_VEL_YAW_RANGE = (-1.0, 1.0)


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
    parser.add_argument(
        "--check-leg-jacobian",
        action="store_true",
        default=False,
        help="run finite-difference checks for the KITE analytic leg Jacobian and exit",
    )
    parser.add_argument(
        "--jacobian-check-eps",
        type=str,
        default="1e-3,1e-4,1e-5",
        help="comma-separated finite-difference epsilons for --check-leg-jacobian",
    )
    parser.add_argument(
        "--jacobian-check-random-scale",
        type=float,
        default=0.15,
        help="small-angle random perturbation scale in radians for Jacobian checks",
    )
    parser.add_argument(
        "--jacobian-check-print-matrices",
        action="store_true",
        default=False,
        help="print analytic and finite-difference Jacobian matrices for env 0",
    )
    parser.add_argument(
        "--jacobian-velocity-check-steps",
        type=int,
        default=8,
        help="control steps per dominant-motion case for simulator velocity alignment",
    )
    parser.add_argument(
        "--jacobian-velocity-check-action",
        type=float,
        default=0.35,
        help="raw action magnitude for dominant-motion Jacobian velocity checks",
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
        # Stairs down/up.
        env_cfg.terrain.terrain_kwargs = {
            "type": "terrain_utils.pyramid_stairs_terrain",
            "step_width": 0.4,
            "step_height": -0.10,
            "platform_size": env_cfg.terrain.platform_size,
        }
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
        # env_cfg.terrain.terrain_kwargs = {
        #     "type": "terrain_utils.stepping_stones_terrain",
        #     "stone_length": 0.55,
        #     "stone_width": 0.55,
        #     "stone_distance_x": 0.25,
        #     "stone_distance_y": 0.25,
        #     "max_height": 0.20,
        #     "platform_size": env_cfg.terrain.platform_size,
        #     "min_stone_length": 0.20,
        #     "min_stone_width": 0.20,
        #     "stepping_stone_edge_clearance": 0.4,
        # }
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
    env_cfg.commands.ranges.lin_vel_x = list(JOYSTICK_LIN_VEL_X_RANGE)
    env_cfg.commands.ranges.lin_vel_y = list(JOYSTICK_LIN_VEL_Y_RANGE)
    env_cfg.commands.ranges.ang_vel_yaw = list(JOYSTICK_ANG_VEL_YAW_RANGE)
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


def _scale_joystick_axis(axis_value, command_range):
    """Scale a normalized joystick axis into a signed command range."""
    min_command, max_command = command_range
    axis_value = float(np.clip(axis_value, -1.0, 1.0))
    if axis_value >= 0.0:
        return axis_value * max(0.0, float(max_command))
    return axis_value * max(0.0, abs(float(min_command)))


def _update_commands(env, args, joystick=None):
    """Apply either joystick commands or a simple fixed forward command."""
    if joystick is not None:
        joystick.update()
        env.commands[:, 0] = _scale_joystick_axis(
            -joystick.ly,
            JOYSTICK_LIN_VEL_X_RANGE,
        )
        env.commands[:, 1] = _scale_joystick_axis(
            -joystick.lx,
            JOYSTICK_LIN_VEL_Y_RANGE,
        )
        env.commands[:, 2] = _scale_joystick_axis(
            -joystick.rx,
            JOYSTICK_ANG_VEL_YAW_RANGE,
        )
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


def compute_all_foot_positions_fk(env, q, include_hip_offset=False):
    """Forward-kinematics foot positions using the raw GO2 URDF convention."""
    assert q.ndim == 3 and q.shape[1:] == (4, 3), (
        f"Expected q shape (N, 4, 3), got {q.shape}"
    )

    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    dtype = q.dtype
    device = q.device
    N = q.shape[0]

    l1 = torch.as_tensor(env.cfg.asset.abad_link_length, device=device, dtype=dtype)
    l2 = torch.as_tensor(env.cfg.asset.hip_link_length, device=device, dtype=dtype)
    l3 = torch.as_tensor(env.cfg.asset.knee_link_length, device=device, dtype=dtype)
    l4 = torch.as_tensor(env.cfg.asset.knee_link_y_offset, device=device, dtype=dtype)
    lateral_offset = l1 + l4

    side_sign = torch.as_tensor(
        env.cfg.asset.side_signs,
        device=device,
        dtype=dtype,
    ).view(1, 4).expand(N, 4)
    side_sign = torch.nan_to_num(side_sign, nan=1.0, posinf=1.0, neginf=-1.0)

    q0 = q[:, :, 0]  # abad / hip roll
    q1 = q[:, :, 1]  # thigh / hip pitch
    q2 = q[:, :, 2]  # calf / knee

    s0 = torch.sin(q0)
    c0 = torch.cos(q0)
    s1 = torch.sin(q1)
    c1 = torch.cos(q1)
    s12 = torch.sin(q1 + q2)
    c12 = torch.cos(q1 + q2)

    sagittal_x = l2 * s1 + l3 * s12
    sagittal_z = l2 * c1 + l3 * c12

    foot_pos = torch.zeros(N, 4, 3, device=device, dtype=dtype)
    foot_pos[:, :, 0] = -sagittal_x
    foot_pos[:, :, 1] = side_sign * lateral_offset * c0 + sagittal_z * s0
    foot_pos[:, :, 2] = side_sign * lateral_offset * s0 - sagittal_z * c0

    if include_hip_offset:
        hip_offsets = torch.tensor(
            [
                [0.1934, -0.0465, 0.0],
                [0.1934, 0.0465, 0.0],
                [-0.1934, -0.0465, 0.0],
                [-0.1934, 0.0465, 0.0],
            ],
            device=device,
            dtype=dtype,
        ).view(1, 4, 3)
        foot_pos = foot_pos + hip_offsets

    return torch.nan_to_num(foot_pos, nan=0.0, posinf=1e6, neginf=-1e6)


def _finite_difference_leg_jacobian(env, q, eps):
    """Finite-difference FK to estimate d foot_pos / d q for all legs."""
    p0 = compute_all_foot_positions_fk(env, q)
    J_fd = torch.zeros(q.shape[0], 4, 3, 3, device=q.device, dtype=q.dtype)

    for leg_id in range(4):
        for joint_id in range(3):
            q_eps = q.clone()
            q_eps[:, leg_id, joint_id] += eps
            p_eps = compute_all_foot_positions_fk(env, q_eps)
            J_fd[:, leg_id, :, joint_id] = (
                p_eps[:, leg_id, :] - p0[:, leg_id, :]
            ) / eps

    return J_fd


def _print_jacobian_check_case(env, case_name, q, eps_values, print_matrices):
    """Compare analytic and finite-difference leg Jacobians for one q batch."""
    leg_names = ["FR", "FL", "RR", "RL"]
    joint_names = ["abad", "thigh", "calf"]
    q = q.detach()

    print(f"\n[KITE Jacobian Check] case={case_name}, q_shape={tuple(q.shape)}")
    for eps in eps_values:
        J_analytic = env.compute_all_leg_jacobians(q)
        J_fd = _finite_difference_leg_jacobian(env, q, eps)
        err = (J_analytic - J_fd).abs()
        per_leg = err.amax(dim=(0, 2, 3))
        per_joint = err.amax(dim=(0, 1, 2))

        print(f"  eps={eps:.1e}")
        print(f"    max_err={err.max().item():.6e}")
        print(f"    mean_err={err.mean().item():.6e}")
        print(
            "    per_leg_max="
            + ", ".join(
                f"{name}:{value.item():.6e}"
                for name, value in zip(leg_names, per_leg)
            )
        )
        print(
            "    per_joint_col_max="
            + ", ".join(
                f"{name}:{value.item():.6e}"
                for name, value in zip(joint_names, per_joint)
            )
        )
        print("    env0 thigh x-column signs:")
        print(f"      analytic J[:,0,1]={J_analytic[0, :, 0, 1].detach().cpu()}")
        print(f"      fd       J[:,0,1]={J_fd[0, :, 0, 1].detach().cpu()}")
        print("    env0 calf x-column signs:")
        print(f"      analytic J[:,0,2]={J_analytic[0, :, 0, 2].detach().cpu()}")
        print(f"      fd       J[:,0,2]={J_fd[0, :, 0, 2].detach().cpu()}")

        if print_matrices:
            print("    J_analytic[0]=")
            print(J_analytic[0].detach().cpu())
            print("    J_fd[0]=")
            print(J_fd[0].detach().cpu())


def _print_single_joint_fk_sign_test(env, case_name, q, eps_values):
    """Log raw FK foot-motion direction for positive single-joint perturbations."""
    leg_names = ["FR", "FL", "RR", "RL"]
    joint_names = ["abad", "thigh", "calf"]
    q = q.detach()

    print(f"\n[KITE FK Sign Check] case={case_name}, q_shape={tuple(q.shape)}")
    l2 = float(env.cfg.asset.hip_link_length)
    l3 = float(env.cfg.asset.knee_link_length)
    print(
        "  zero-pose raw GO2 expected x-response: "
        f"thigh≈-{l2 + l3:.6f}, calf≈-{l3:.6f}"
    )

    for eps in eps_values:
        p0 = compute_all_foot_positions_fk(env, q, include_hip_offset=False)
        print(f"  eps={eps:.1e}")
        for leg_id, leg_name in enumerate(leg_names):
            for joint_id, joint_name in enumerate(joint_names):
                q_eps = q.clone()
                q_eps[:, leg_id, joint_id] += eps
                p_eps = compute_all_foot_positions_fk(
                    env,
                    q_eps,
                    include_hip_offset=False,
                )
                dp = (p_eps[:, leg_id, :] - p0[:, leg_id, :]) / eps
                print(
                    f"    {leg_name:>2} {joint_name:>6} dp[0]="
                    f"{dp[0].detach().cpu().numpy()}"
                )


def _feet_vel_world_to_base(env, feet_vel_world):
    """Rotate simulator world-frame foot velocities into the base frame."""
    num_envs, num_feet, _ = feet_vel_world.shape
    quat = env.simulator.base_quat[:, None, :].expand(num_envs, num_feet, 4)
    return quat_rotate_inverse(
        quat.reshape(num_envs * num_feet, 4),
        feet_vel_world.reshape(num_envs * num_feet, 3),
    ).reshape(num_envs, num_feet, 3)


def _print_velocity_alignment_summary(env, case_name):
    """Compare J(q) qd against simulator foot velocity in the base frame."""
    leg_names = ["FR", "FL", "RR", "RL"]
    joint_names = ["abad", "thigh", "calf"]

    q = env.simulator.dof_pos.view(env.num_envs, 4, 3).detach()
    qd = env.simulator.dof_vel.view(env.num_envs, 4, 3).detach()
    J = env.compute_all_leg_jacobians(q)
    v_pred = torch.matmul(J, qd.unsqueeze(-1)).squeeze(-1)
    v_sim_base = _feet_vel_world_to_base(env, env.simulator.feet_vel.detach())

    err = v_pred - v_sim_base
    abs_err = err.abs()
    per_leg_mean = abs_err.mean(dim=(0, 2))
    per_leg_max = abs_err.amax(dim=(0, 2))
    speed_pred = torch.linalg.norm(v_pred, dim=-1)
    speed_sim = torch.linalg.norm(v_sim_base, dim=-1)
    cos = (v_pred * v_sim_base).sum(dim=-1) / (
        speed_pred * speed_sim + 1.0e-8
    )
    speed_mask = (speed_pred > 1.0e-4) & (speed_sim > 1.0e-4)
    if speed_mask.any():
        cos_mean = cos[speed_mask].mean()
        cos_min = cos[speed_mask].min()
    else:
        cos_mean = torch.tensor(float("nan"), device=q.device)
        cos_min = torch.tensor(float("nan"), device=q.device)

    dominant_joint = qd.abs().mean(dim=(0, 1)).argmax().item()

    print(f"\n[KITE Velocity Alignment] case={case_name}")
    print(f"  mean_abs_err={abs_err.mean().item():.6e}")
    print(f"  max_abs_err={abs_err.max().item():.6e}")
    print(f"  cosine_mean_nontrivial={cos_mean.item():.6e}")
    print(f"  cosine_min_nontrivial={cos_min.item():.6e}")
    print(
        "  qd_abs_mean_per_joint="
        + ", ".join(
            f"{name}:{value.item():.6e}"
            for name, value in zip(joint_names, qd.abs().mean(dim=(0, 1)))
        )
        + f" dominant={joint_names[dominant_joint]}"
    )
    print(
        "  per_leg_mean_abs_err="
        + ", ".join(
            f"{name}:{value.item():.6e}"
            for name, value in zip(leg_names, per_leg_mean)
        )
    )
    print(
        "  per_leg_max_abs_err="
        + ", ".join(
            f"{name}:{value.item():.6e}"
            for name, value in zip(leg_names, per_leg_max)
        )
    )
    print(f"  env0 v_pred={v_pred[0].detach().cpu()}")
    print(f"  env0 v_sim_base={v_sim_base[0].detach().cpu()}")


def _run_velocity_alignment_case(env, case_name, joint_id, action_mag, steps):
    """Drive one joint column mostly, then compare analytic and sim foot velocity."""
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    actions[:, joint_id::3] = action_mag

    # A few steps lets the PD targets produce nontrivial joint velocities.
    for _ in range(max(1, int(steps))):
        env.step(actions)

    _print_velocity_alignment_summary(env, case_name)


@torch.no_grad()
def run_leg_jacobian_sanity_check(env, kite_viz_args):
    """Run opt-in Jacobian finite-difference diagnostics, then return."""
    eps_values = [
        float(value.strip())
        for value in kite_viz_args.jacobian_check_eps.split(",")
        if value.strip()
    ]
    if not eps_values:
        raise ValueError("--jacobian-check-eps must contain at least one value.")

    num_envs = env.num_envs
    device = env.device
    dtype = env.simulator.dof_pos.dtype

    q_zero = torch.zeros(num_envs, 4, 3, device=device, dtype=dtype)
    _print_single_joint_fk_sign_test(
        env,
        "zero_pose",
        q_zero,
        eps_values,
    )
    _print_jacobian_check_case(
        env,
        "zero_pose",
        q_zero,
        eps_values,
        kite_viz_args.jacobian_check_print_matrices,
    )

    q_default = env.simulator.default_dof_pos.to(device=device, dtype=dtype)
    q_default = q_default.expand(num_envs, -1).reshape(num_envs, 4, 3).clone()
    _print_single_joint_fk_sign_test(
        env,
        "default_standing_pose",
        q_default,
        eps_values,
    )
    _print_jacobian_check_case(
        env,
        "default_standing_pose",
        q_default,
        eps_values,
        kite_viz_args.jacobian_check_print_matrices,
    )

    _print_velocity_alignment_summary(env, "initial_state")
    action_mag = float(kite_viz_args.jacobian_velocity_check_action)
    steps = int(kite_viz_args.jacobian_velocity_check_steps)
    dominant_cases = [
        ("abad_dominant_positive_action", 0),
        ("thigh_dominant_positive_action", 1),
        ("calf_dominant_positive_action", 2),
    ]
    for case_name, joint_id in dominant_cases:
        env.reset_idx(torch.arange(env.num_envs, device=env.device))
        _run_velocity_alignment_case(env, case_name, joint_id, action_mag, steps)

    rand_scale = float(kite_viz_args.jacobian_check_random_scale)
    q_random = q_default + (2.0 * torch.rand_like(q_default) - 1.0) * rand_scale
    _print_jacobian_check_case(
        env,
        f"default_plus_uniform_{rand_scale:.3f}_rad",
        q_random,
        eps_values,
        kite_viz_args.jacobian_check_print_matrices,
    )


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
    if kite_viz_args.check_leg_jacobian:
        run_leg_jacobian_sanity_check(env, kite_viz_args)
        return

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
