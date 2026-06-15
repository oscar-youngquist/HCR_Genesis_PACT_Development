"""Visual smoke test for the KITE Warp depth-camera pipeline."""

import argparse
import sys
import time

import torch

from legged_gym import SIMULATOR, gs
from legged_gym.envs import *  # noqa: F403
from legged_gym.utils import get_args, init_genesis, task_registry


NUM_ROBOTS = 1
DEBUG_ROBOT_ID = 0
RESPAWN_INTERVAL_SECONDS = 5.0


def parse_visualization_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--terrain",
        choices=("flat", "rough"),
        default="rough",
        help="terrain used to inspect the camera and measured normals",
    )
    parser.add_argument(
        "--roughness",
        type=float,
        default=0.12,
        help="maximum absolute rough-terrain height in meters",
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
        help="control steps between normal-overlay refreshes",
    )
    visual_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return visual_args


def configure_test_environment(env_cfg, visual_args):
    if visual_args.roughness <= 0.0:
        raise ValueError("--roughness must be positive.")
    if visual_args.normal_length <= 0.0:
        raise ValueError("--normal-length must be positive.")
    if visual_args.normal_refresh_steps <= 0:
        raise ValueError("--normal-refresh-steps must be positive.")

    env_cfg.env.num_envs = NUM_ROBOTS
    env_cfg.viewer.rendered_envs_idx = [DEBUG_ROBOT_ID]

    env_cfg.terrain.mesh_type = "heightfield"
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.num_subterrains = 1
    if visual_args.terrain == "rough":
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = True
        env_cfg.terrain.terrain_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -visual_args.roughness,
            "max_height": visual_args.roughness,
            "step": 0.005,
            "downsampled_scale": 0.2,
        }
    else:
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.selected = False
        env_cfg.terrain.terrain_kwargs = None
        env_cfg.terrain.terrain_proportions = [
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ]
        env_cfg.terrain.terrain_curriculum_difficulty["slope"] = "0.0"

    env_cfg.terrain.debug_draw_measured_surface_normals = True
    env_cfg.terrain.debug_height_map_env_id = DEBUG_ROBOT_ID
    env_cfg.terrain.debug_surface_normal_length = visual_args.normal_length
    env_cfg.terrain.debug_surface_normal_refresh_steps = (
        visual_args.normal_refresh_steps
    )

    camera_cfg = env_cfg.sensor.depth_camera_config
    camera_cfg.debug_camera_env_id = DEBUG_ROBOT_ID
    camera_cfg.debug_render_depth_image = True
    camera_cfg.debug_draw_camera_position = True
    camera_cfg.debug_print_depth_stats = True
    camera_cfg.debug_depth_stats_interval = 5

    # Keep the base orientation fixed so mounting variation is easy to inspect.
    env_cfg.init_state.roll_random_scale = 0.0
    env_cfg.init_state.pitch_random_scale = 0.0
    env_cfg.init_state.yaw_random_scale = 0.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_camera_pos = True
    env_cfg.domain_rand.randomize_camera_euler = True


def print_camera_mount(simulator):
    position = simulator._sensor_offset_pos[DEBUG_ROBOT_ID].tolist()
    euler = simulator._sensor_offset_euler[DEBUG_ROBOT_ID].tolist()
    position_delta = (
        simulator._sensor_offset_pos[DEBUG_ROBOT_ID]
        - simulator._sensor_nominal_offset_pos[DEBUG_ROBOT_ID]
    ).tolist()
    euler_delta = (
        simulator._sensor_offset_euler[DEBUG_ROBOT_ID]
        - simulator._sensor_nominal_euler[DEBUG_ROBOT_ID]
    ).tolist()
    print(
        "Camera mount sampled: "
        f"position=({position[0]:+.4f}, {position[1]:+.4f}, "
        f"{position[2]:+.4f}) m, "
        f"rpy=({euler[0]:+.4f}, {euler[1]:+.4f}, "
        f"{euler[2]:+.4f}) rad; "
        f"delta_xyz=({position_delta[0]:+.4f}, "
        f"{position_delta[1]:+.4f}, {position_delta[2]:+.4f}) m, "
        f"delta_rpy=({euler_delta[0]:+.4f}, "
        f"{euler_delta[1]:+.4f}, {euler_delta[2]:+.4f}) rad",
        flush=True,
    )


def main():
    visual_args = parse_visualization_args()
    args = get_args()
    if SIMULATOR != "genesis_kite_depth":
        raise RuntimeError(
            "Set SIMULATOR=genesis_kite_depth before running this test."
        )
    if args.headless:
        raise RuntimeError("This visual test must run without --headless.")

    init_genesis(args, gs)
    env_cfg, _ = task_registry.get_cfgs(name=args.task, args=args)
    configure_test_environment(env_cfg, visual_args)
    env, _ = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
    )

    actions = torch.zeros(
        (NUM_ROBOTS, env.num_actions), device=env.device
    )
    print(
        "Running the one-robot KITE depth-camera test until Ctrl+C. "
        "The viewer shows the camera-position marker and the OpenCV window "
        "shows the live depth image. Green dots mark measured terrain heights "
        "and yellow arrows show their world-frame surface normals. "
        f"Terrain mode: {visual_args.terrain}. The robot and camera mount are "
        f"resampled every {RESPAWN_INTERVAL_SECONDS:.1f} seconds. "
        "Press Ctrl+C to exit."
    )
    robot_id = torch.tensor(
        [DEBUG_ROBOT_ID], device=env.device, dtype=torch.long
    )
    print_camera_mount(env.simulator)
    next_respawn = time.monotonic() + RESPAWN_INTERVAL_SECONDS

    try:
        while True:
            env.step(actions)
            now = time.monotonic()
            if now >= next_respawn:
                env.simulator.resample_camera_mount(robot_id)
                env.reset_idx(robot_id)
                print_camera_mount(env.simulator)
                next_respawn = now + RESPAWN_INTERVAL_SECONDS
    except KeyboardInterrupt:
        print("KITE depth-camera test stopped.")


if __name__ == "__main__":
    main()
