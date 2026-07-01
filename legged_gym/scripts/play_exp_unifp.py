from legged_gym import *
import argparse

from legged_gym.envs import *
from legged_gym.scripts.joystick import Joystick
from legged_gym.utils import *

import numpy as np
import torch


def override_configs(env_cfg, train_cfg, args):
    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else 1
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))

    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 2
        env_cfg.terrain.border_size = 5.0
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = True
        env_cfg.env.debug_draw_terrain_height_points = False
        env_cfg.terrain.terrain_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -0.05,
            "max_height": 0.05,
            "step": 0.005,
            "downsampled_scale": 0.2,
        }

    env_cfg.env.debug = False
    env_cfg.env.debug_viz = False
    env_cfg.env.render_ee_goal_debug = args.render_ee_goal_debug

    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [-args.max_lin_vel_x, args.max_lin_vel_x]
    env_cfg.commands.ranges.lin_vel_y = [-args.max_lin_vel_y, args.max_lin_vel_y]
    env_cfg.commands.ranges.ang_vel_yaw = [-args.max_yaw_vel, args.max_yaw_vel]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    env_cfg.commands.force_start_step = args.force_start_step
    env_cfg.commands.apply_ee_external_forces = args.apply_ee_external_forces
    env_cfg.commands.apply_base_external_forces = args.apply_base_external_forces
    env_cfg.commands.use_external_impedance_compensation = args.use_unifp_impedance_controller
    env_cfg.commands.compensate_ee_external_force = args.compensate_ee_external_force
    env_cfg.commands.compensate_base_external_force = args.compensate_base_external_force

    env_cfg.domain_rand.push_robots = False

    if args.record_frames or args.follow_robot:
        env_cfg.viewer.add_camera = True


def _clip_command(value, command_range):
    return max(command_range[0], min(command_range[1], value))


def update_base_velocity_command(env, args, joystick):
    if joystick is not None:
        joystick.update()
        cmd_x = -joystick.ly * args.max_lin_vel_x
        cmd_y = -joystick.lx * args.max_lin_vel_y
        cmd_yaw = -joystick.rx * args.max_yaw_vel
    else:
        cmd_x = args.cmd_x
        cmd_y = args.cmd_y
        cmd_yaw = args.cmd_yaw

    cmd_x = _clip_command(cmd_x, env.command_ranges["lin_vel_x"])
    cmd_y = _clip_command(cmd_y, env.command_ranges["lin_vel_y"])
    cmd_yaw = _clip_command(cmd_yaw, env.command_ranges["ang_vel_yaw"])
    env.commands[:, 0] = cmd_x
    env.commands[:, 1] = cmd_y
    env.commands[:, 2] = cmd_yaw


def follow_camera(env, robot_index):
    pos = env.simulator.base_pos[robot_index].detach().cpu().numpy() + np.array(
        env.cfg.viewer.pos,
        dtype=np.float32,
    )
    lookat = env.simulator.base_pos[robot_index].detach().cpu().numpy() + np.array(
        env.cfg.viewer.lookat,
        dtype=np.float32,
    )
    env.set_camera(pos, lookat)
    floating_camera = getattr(env.simulator, "_floating_camera", None)
    if floating_camera is not None:
        floating_camera.render()


def interaction_loop(env, policy, args):
    robot_index = 0
    joystick = Joystick(joystick_type=args.joystick_type) if args.use_joystick else None
    obs_buf, obs_history, privileged_obs_buf, explicit_labels = env.get_observations()
    num_steps = args.num_steps if args.num_steps is not None else int(5.56 * env.max_episode_length)

    if args.record_frames:
        env.simulator._floating_camera.start_recording()

    print(
        "UniFP play: "
        f"joystick={args.use_joystick}, "
        f"ee_external_forces={args.apply_ee_external_forces}, "
        f"base_external_forces={args.apply_base_external_forces}, "
        f"impedance_compensation={args.use_unifp_impedance_controller}, "
        f"ee_goal_debug={args.render_ee_goal_debug}",
        flush=True,
    )
    if args.render_ee_goal_debug:
        print("EE debug colors: yellow=nominal target, magenta=force-offset target, blue=current EE, cyan=spherical center", flush=True)

    for _ in range(num_steps):
        update_base_velocity_command(env, args, joystick)

        if args.follow_robot:
            follow_camera(env, robot_index)
        if args.render_ee_goal_debug:
            env.draw_ee_goal_debug_vis()

        policy_info = {}
        actions = policy({"obs": obs_history.detach()}, policy_info=policy_info)
        if args.use_unifp_impedance_controller:
            env.set_impedance_force_estimates(policy_info["latents"])
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, rews, dones, infos, grfs = env.step(actions.detach())


def play(args):
    if "genesis" in SIMULATOR:
        init_genesis(args, gs)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, args=args)
    override_configs(env_cfg, train_cfg, args)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    interaction_loop(env, policy, args)

    if args.record_frames:
        filename_mp4 = f"{train_cfg.runner.experiment_name}_play_exp_unifp.mp4"
        env.simulator._floating_camera.stop_recording(save_to_filename=filename_mp4, fps=30)
        print("Saved recording to " + filename_mp4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="b1z1_unifp")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("-o", "--offline", action="store_true", default=False)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--sync_wandb", action="store_true", default=False)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--use_joystick", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--joystick_type", type=str, default="xbox", choices=["xbox", "switch"])
    parser.add_argument("--follow_robot", action="store_true", default=False)
    parser.add_argument("--record_frames", action="store_true", default=False)
    parser.add_argument("--num_steps", type=int, default=None)

    parser.add_argument("--cmd_x", type=float, default=0.0)
    parser.add_argument("--cmd_y", type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--max_lin_vel_x", type=float, default=0.6)
    parser.add_argument("--max_lin_vel_y", type=float, default=0.4)
    parser.add_argument("--max_yaw_vel", type=float, default=0.6)

    parser.add_argument("--force_start_step", type=int, default=0)
    parser.add_argument("--apply_ee_external_forces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply_base_external_forces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_unifp_impedance_controller", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compensate_ee_external_force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compensate_base_external_force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render_ee_goal_debug", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    if args.gpu is None:
        args.gpu = args.device
    configure_runtime_device(args)

    play(args)
