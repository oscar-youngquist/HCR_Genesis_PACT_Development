"""Interactive/automated evaluation for the standalone Genesis B1 UniFP policy."""

import argparse
from collections import deque

import numpy as np
import torch

from legged_gym import SIMULATOR, gs
from legged_gym.envs import *  # noqa: F401,F403 - registers environments
from legged_gym.scripts.joystick import Joystick
from legged_gym.utils import configure_runtime_device, init_genesis, task_registry
from legged_gym.utils.math_utils import quat_apply, quat_from_euler_xyz


def override_configs(env_cfg, args):
    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else 1
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    env_cfg.env.debug = False
    env_cfg.env.debug_viz = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [-args.max_lin_vel_x, args.max_lin_vel_x]
    env_cfg.commands.ranges.lin_vel_y = [-args.max_lin_vel_y, args.max_lin_vel_y]
    env_cfg.commands.ranges.ang_vel_yaw = [-args.max_yaw_vel, args.max_yaw_vel]
    env_cfg.commands.force_start_step = args.force_start_step
    env_cfg.commands.push_robot_base = args.enable_torso_force_streams
    env_cfg.commands.apply_base_external_forces = args.apply_base_external_forces
    env_cfg.domain_rand.push_robots = False

    if args.terrain == "plane":
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.measure_heights = False
        env_cfg.terrain.obtain_terrain_info_around_feet = False
        env_cfg.rewards.scales.foot_clearance_terrain_aware = 0.0
    else:
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 2
        env_cfg.terrain.border_size = 5.0
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = True
        env_cfg.terrain.terrain_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -args.terrain_height,
            "max_height": args.terrain_height,
            "step": 0.005,
            "downsampled_scale": 0.2,
        }

    if args.record_frames or args.follow_robot or args.camera_preset != "none":
        env_cfg.viewer.add_camera = True


def clip_command(value, limits):
    return max(limits[0], min(limits[1], value))


class CommandScheduler:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.mode = args.command_mode
        if self.mode == "auto":
            self.mode = "joystick" if args.use_joystick else "scripted"
        self.hold_steps = max(1, int(args.command_hold_s / env.dt))
        self.command = (args.cmd_x, args.cmd_y, args.cmd_yaw)
        self.sequence_index = -1
        self.sequence = [
            (0.0, 0.0, 0.0),
            (args.max_lin_vel_x, 0.0, 0.0),
            (-args.max_lin_vel_x, 0.0, 0.0),
            (0.0, args.max_lin_vel_y, 0.0),
            (0.0, -args.max_lin_vel_y, 0.0),
            (0.0, 0.0, args.max_yaw_vel),
            (0.0, 0.0, -args.max_yaw_vel),
            (0.6 * args.max_lin_vel_x, 0.5 * args.max_lin_vel_y, 0.5 * args.max_yaw_vel),
        ]

    def update(self, step, joystick=None):
        if self.mode == "joystick":
            joystick.update()
            self.command = (
                -joystick.ly * self.args.max_lin_vel_x,
                -joystick.lx * self.args.max_lin_vel_y,
                -joystick.rx * self.args.max_yaw_vel,
            )
        elif self.mode == "fixed":
            self.command = (self.args.cmd_x, self.args.cmd_y, self.args.cmd_yaw)
        elif self.mode == "random" and step % self.hold_steps == 0:
            self.command = (
                np.random.uniform(-self.args.max_lin_vel_x, self.args.max_lin_vel_x),
                np.random.uniform(-self.args.max_lin_vel_y, self.args.max_lin_vel_y),
                np.random.uniform(-self.args.max_yaw_vel, self.args.max_yaw_vel),
            )
            self.print_command(step)
        elif self.mode == "scripted" and step % self.hold_steps == 0:
            self.sequence_index = (self.sequence_index + 1) % len(self.sequence)
            self.command = self.sequence[self.sequence_index]
            self.print_command(step)
        self.apply()

    def apply(self):
        vx = clip_command(self.command[0], self.env.command_ranges["lin_vel_x"])
        vy = clip_command(self.command[1], self.env.command_ranges["lin_vel_y"])
        yaw = clip_command(self.command[2], self.env.command_ranges["ang_vel_yaw"])
        self.command = (vx, vy, yaw)
        self.env.commands[:, 0] = vx
        self.env.commands[:, 1] = vy
        self.env.commands[:, 2] = yaw

    def print_command(self, step):
        print(
            f"[command step {step}] vx={self.command[0]:+.2f} m/s, "
            f"vy={self.command[1]:+.2f} m/s, yaw={self.command[2]:+.2f} rad/s",
            flush=True,
        )


class LocomotionMetrics:
    def __init__(self, window_steps):
        self.window = deque(maxlen=window_steps)
        self.samples = []
        self.reset_count = 0

    def update(self, env, dones):
        yaw_quat = quat_from_euler_xyz(
            torch.zeros(env.num_envs, device=env.device),
            torch.zeros(env.num_envs, device=env.device),
            env.simulator.base_euler[:, 2],
        )
        force_local = quat_apply(
            torch.cat((-yaw_quat[:, :3], yaw_quat[:, 3:4]), dim=1),
            env.base_force_ext_world,
        )
        effective_xy = env.commands[:, :2] + (
            force_local[:, :2] + env.current_Fxyz_base_cmd[:, :2]
        ) / env.base_force_kds
        sample = {
            "nominal_xy_error": torch.norm(env.commands[:, :2] - env.simulator.base_lin_vel[:, :2], dim=1).mean().item(),
            "effective_xy_error": torch.norm(effective_xy - env.simulator.base_lin_vel[:, :2], dim=1).mean().item(),
            "yaw_error": torch.abs(env.commands[:, 2] - env.simulator.base_ang_vel[:, 2]).mean().item(),
            "roll_pitch": torch.norm(env.simulator.base_euler[:, :2], dim=1).mean().item(),
            "base_height": env.simulator.base_pos[:, 2].mean().item(),
            "force_cmd": torch.norm(env.current_Fxyz_base_cmd, dim=1).mean().item(),
            "force_ext": torch.norm(env.base_force_ext_world, dim=1).mean().item(),
        }
        self.reset_count += int(dones.sum().item())
        self.window.append(sample)
        self.samples.append(sample)

    @staticmethod
    def means(samples):
        return {key: float(np.mean([sample[key] for sample in samples])) for key in samples[0]}

    def print_window(self, step):
        if not self.window:
            return
        mean = self.means(self.window)
        print(
            f"[eval step {step}] effective_xy_err={mean['effective_xy_error']:.3f} m/s, "
            f"nominal_xy_err={mean['nominal_xy_error']:.3f} m/s, "
            f"yaw_err={mean['yaw_error']:.3f} rad/s, "
            f"roll_pitch={mean['roll_pitch']:.3f} rad, height={mean['base_height']:.3f} m, "
            f"Fcmd/Fext={mean['force_cmd']:.1f}/{mean['force_ext']:.1f} N, resets={self.reset_count}",
            flush=True,
        )

    def print_summary(self):
        if not self.samples:
            return
        mean = self.means(self.samples)
        print(
            "B1 UniFP evaluation summary: "
            f"effective_xy_err={mean['effective_xy_error']:.3f} m/s, "
            f"nominal_xy_err={mean['nominal_xy_error']:.3f} m/s, "
            f"yaw_err={mean['yaw_error']:.3f} rad/s, "
            f"roll_pitch={mean['roll_pitch']:.3f} rad, resets={self.reset_count}",
            flush=True,
        )


def update_camera(env, args, robot_index=0):
    if args.camera_preset == "none" and not args.follow_robot:
        return
    base = env.simulator.base_pos[robot_index].detach().cpu().numpy()
    preset = "follow" if args.follow_robot else args.camera_preset
    offsets = {
        "follow": (np.array(env.cfg.viewer.pos), np.array(env.cfg.viewer.lookat)),
        "side": (np.array([0.0, 2.4, 0.9]), np.array([0.2, 0.0, 0.3])),
        "front": (np.array([2.0, 0.0, 0.8]), np.array([0.2, 0.0, 0.3])),
        "top": (np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, 0.0])),
    }
    if preset not in offsets:
        return
    eye, target = offsets[preset]
    env.set_camera(base + eye, base + target)
    camera = getattr(env.simulator, "_floating_camera", None)
    if camera is not None:
        camera.render()


def play(args):
    if "genesis" in SIMULATOR:
        init_genesis(args, gs)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, args=args)
    override_configs(env_cfg, args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.ckpt
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    _, obs_history, _, _ = env.get_observations()

    mode = args.command_mode
    if mode == "auto":
        mode = "joystick" if args.use_joystick else "scripted"
    joystick = Joystick(joystick_type=args.joystick_type) if mode == "joystick" else None
    scheduler = CommandScheduler(env, args)
    metrics = LocomotionMetrics(args.metrics_window_steps)
    num_steps = args.num_steps or int(5.0 * env.max_episode_length)
    camera = getattr(env.simulator, "_floating_camera", None)
    if args.record_frames:
        camera.start_recording()

    print(
        f"B1 UniFP play: mode={mode}, terrain={args.terrain}, envs={env.num_envs}, "
        f"torso_force_streams={args.enable_torso_force_streams}, "
        f"external_forces={args.apply_base_external_forces}",
        flush=True,
    )
    for step in range(num_steps):
        scheduler.update(step, joystick)
        update_camera(env, args)
        actions = policy({"obs": obs_history.detach()})
        _, _, obs_history, _, _, dones, _, _ = env.step(actions.detach())
        # Resets and the environment's periodic sampler can replace commands.
        scheduler.apply()
        metrics.update(env, dones)
        if args.metrics_interval_steps > 0 and step % args.metrics_interval_steps == 0:
            metrics.print_window(step)
    metrics.print_summary()

    if args.record_frames:
        filename = args.video_filename or f"{train_cfg.runner.experiment_name}_play.mp4"
        camera.stop_recording(save_to_filename=filename, fps=args.video_fps)
        print(f"Saved recording to {filename}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="b1_unifp")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("-c", "--cpu", action="store_true")
    parser.add_argument("-B", "--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("-o", "--offline", action="store_true")
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--sync_wandb", action="store_true")
    parser.add_argument("--load_run", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--terrain", choices=["plane", "rough"], default="rough")
    parser.add_argument("--terrain_height", type=float, default=0.05)
    parser.add_argument("--command_mode", choices=["auto", "joystick", "fixed", "random", "scripted"], default="auto")
    parser.add_argument("--command_hold_s", type=float, default=3.0)
    parser.add_argument("--cmd_x", type=float, default=0.0)
    parser.add_argument("--cmd_y", type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--max_lin_vel_x", type=float, default=0.5)
    parser.add_argument("--max_lin_vel_y", type=float, default=0.6)
    parser.add_argument("--max_yaw_vel", type=float, default=1.0)
    parser.add_argument("--use_joystick", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--joystick_type", choices=["xbox", "switch"], default="xbox")
    parser.add_argument("--force_start_step", type=int, default=0)
    parser.add_argument("--enable_torso_force_streams", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--apply_base_external_forces", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--follow_robot", action="store_true")
    parser.add_argument("--camera_preset", choices=["none", "follow", "side", "front", "top"], default="follow")
    parser.add_argument("--record_frames", action="store_true")
    parser.add_argument("--video_filename", default=None)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--metrics_interval_steps", type=int, default=50)
    parser.add_argument("--metrics_window_steps", type=int, default=100)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.gpu is None:
        args.gpu = args.device
    configure_runtime_device(args)
    play(args)
