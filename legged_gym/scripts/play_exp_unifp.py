from legged_gym import *
import argparse

from legged_gym.envs import *
from legged_gym.scripts.joystick import Joystick
from legged_gym.utils import *
from legged_gym.utils.math_utils import quat_apply, quat_from_euler_xyz

from collections import deque
import numpy as np
import torch


# Isaac Gym's Python 3.8 environment predates argparse.BooleanOptionalAction.
# Keep the same --flag/--no-flag CLI used by the Genesis play environment.
if not hasattr(argparse, "BooleanOptionalAction"):
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            expanded_options = []
            for option in option_strings:
                expanded_options.append(option)
                if option.startswith("--"):
                    expanded_options.append("--no-" + option[2:])
            super().__init__(
                option_strings=expanded_options,
                dest=dest,
                nargs=0,
                default=default,
                **kwargs,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not option_string.startswith("--no-"))

    argparse.BooleanOptionalAction = BooleanOptionalAction


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
    env_cfg.env.render_ee_frame_debug = args.render_ee_frame_debug

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

    if args.record_frames or args.follow_robot or args.camera_preset != "none":
        env_cfg.viewer.add_camera = True


def sphere_to_cart(sphere_coords):
    radius = sphere_coords[:, 0]
    pitch = sphere_coords[:, 1]
    yaw = sphere_coords[:, 2]
    return torch.stack(
        (
            radius * torch.cos(pitch) * torch.cos(yaw),
            radius * torch.cos(pitch) * torch.sin(yaw),
            radius * torch.sin(pitch),
        ),
        dim=-1,
    )


def _clip_command(value, command_range):
    return max(command_range[0], min(command_range[1], value))


class CommandScheduler:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.base_mode = args.base_command_mode
        if self.base_mode == "auto":
            self.base_mode = "joystick" if args.use_joystick else "random"
        self.ee_mode = args.ee_eval_mode
        self.base_hold_steps = max(1, int(args.base_command_hold_s / env.dt))
        self.ee_hold_steps = max(1, int(args.ee_command_hold_s / env.dt))
        self.base_idx = -1
        self.ee_idx = -1
        self.base_cmd = (args.cmd_x, args.cmd_y, args.cmd_yaw)
        self.ee_sphere = torch.tensor(args.fixed_ee_sphere, device=env.device, dtype=torch.float)
        self.base_sequence = [
            (0.0, 0.0, 0.0),
            (args.max_lin_vel_x, 0.0, 0.0),
            (-args.max_lin_vel_x, 0.0, 0.0),
            (0.0, args.max_lin_vel_y, 0.0),
            (0.0, -args.max_lin_vel_y, 0.0),
            (0.0, 0.0, args.max_yaw_vel),
            (0.0, 0.0, -args.max_yaw_vel),
            (0.6 * args.max_lin_vel_x, 0.5 * args.max_lin_vel_y, 0.5 * args.max_yaw_vel),
        ]
        self.ee_sequence = [
            [0.55, 0.00, 0.00],
            [0.70, 0.10, 0.00],
            [0.50, 0.35, 0.00],
            [0.50, -0.25, 0.00],
            [0.65, 0.05, 0.60],
            [0.65, -0.20, -0.60],
            [0.82, 0.00, 0.00],
        ]

    def update(self, step, joystick=None):
        if self.base_mode == "joystick":
            self._update_joystick_base(joystick)
        elif self.base_mode == "fixed":
            self.base_cmd = (self.args.cmd_x, self.args.cmd_y, self.args.cmd_yaw)
        elif self.base_mode == "random" and step % self.base_hold_steps == 0:
            self.base_cmd = (
                np.random.uniform(-self.args.max_lin_vel_x, self.args.max_lin_vel_x),
                np.random.uniform(-self.args.max_lin_vel_y, self.args.max_lin_vel_y),
                np.random.uniform(-self.args.max_yaw_vel, self.args.max_yaw_vel),
            )
            self._print_base_command(step)
        elif self.base_mode == "scripted" and step % self.base_hold_steps == 0:
            self.base_idx = (self.base_idx + 1) % len(self.base_sequence)
            self.base_cmd = self.base_sequence[self.base_idx]
            self._print_base_command(step)
        self._apply_base_command()

        if self.ee_mode == "env_sampled":
            return
        if self.ee_mode == "fixed_sphere":
            self.ee_sphere = torch.tensor(self.args.fixed_ee_sphere, device=self.env.device, dtype=torch.float)
            self._apply_ee_sphere()
        elif self.ee_mode == "random_sphere" and step % self.ee_hold_steps == 0:
            ranges = self.env.cfg.goal_ee.ranges
            self.ee_sphere = torch.tensor(
                [
                    np.random.uniform(*ranges.pos_l),
                    np.random.uniform(*ranges.pos_p),
                    np.random.uniform(*ranges.pos_y),
                ],
                device=self.env.device,
                dtype=torch.float,
            )
            self._apply_ee_sphere()
            self._print_ee_command(step)
        elif self.ee_mode == "scripted_sphere" and step % self.ee_hold_steps == 0:
            self.ee_idx = (self.ee_idx + 1) % len(self.ee_sequence)
            self.ee_sphere = torch.tensor(self.ee_sequence[self.ee_idx], device=self.env.device, dtype=torch.float)
            self._apply_ee_sphere()
            self._print_ee_command(step)
        elif self.ee_mode in ["random_sphere", "scripted_sphere"]:
            self._apply_ee_sphere()

    def reapply(self):
        self._apply_base_command()
        if self.ee_mode != "env_sampled":
            self._apply_ee_sphere()

    def _update_joystick_base(self, joystick):
        if joystick is None:
            return
        joystick.update()
        self.base_cmd = (
            -joystick.ly * self.args.max_lin_vel_x,
            -joystick.lx * self.args.max_lin_vel_y,
            -joystick.rx * self.args.max_yaw_vel,
        )

    def _apply_base_command(self):
        cmd_x, cmd_y, cmd_yaw = self.base_cmd
        cmd_x = _clip_command(cmd_x, self.env.command_ranges["lin_vel_x"])
        cmd_y = _clip_command(cmd_y, self.env.command_ranges["lin_vel_y"])
        cmd_yaw = _clip_command(cmd_yaw, self.env.command_ranges["ang_vel_yaw"])
        self.env.commands[:, 0] = cmd_x
        self.env.commands[:, 1] = cmd_y
        self.env.commands[:, 2] = cmd_yaw
        self.base_cmd = (cmd_x, cmd_y, cmd_yaw)

    def _apply_ee_sphere(self):
        sphere = self.ee_sphere.unsqueeze(0).repeat(self.env.num_envs, 1)
        self.env.ee_goal_sphere[:] = sphere
        self.env.ee_start_sphere[:] = sphere
        self.env.curr_ee_goal_sphere[:] = sphere
        self.env.goal_timer[:] = 0.0
        self.env.traj_timesteps[:] = max(1.0, self.args.ee_command_hold_s / self.env.dt)
        self.env.traj_total_timesteps[:] = self.env.traj_timesteps
        self.env.curr_ee_goal_cart[:] = sphere_to_cart(sphere)
        base_yaw_quat = quat_from_euler_xyz(
            torch.zeros(self.env.num_envs, device=self.env.device),
            torch.zeros(self.env.num_envs, device=self.env.device),
            self.env.simulator.base_euler[:, 2],
        )
        self.env.curr_ee_goal_cart_world[:] = self.env.get_ee_goal_spherical_center(base_yaw_quat) + quat_apply(
            base_yaw_quat,
            self.env.curr_ee_goal_cart,
        )
        self.env.commands[:, 3:6] = sphere

    def _print_base_command(self, step):
        print(
            f"[command step {step}] base vx={self.base_cmd[0]:+.2f} "
            f"vy={self.base_cmd[1]:+.2f} yaw={self.base_cmd[2]:+.2f}",
            flush=True,
        )

    def _print_ee_command(self, step):
        values = self.ee_sphere.detach().cpu().numpy()
        print(
            f"[command step {step}] ee sphere r={values[0]:.2f} pitch={values[1]:+.2f} yaw={values[2]:+.2f}",
            flush=True,
        )


class EvalMetrics:
    def __init__(self, window_steps):
        self.window_steps = window_steps
        self.window = deque(maxlen=window_steps)
        self.all_samples = []

    def update(self, env):
        nominal, effective = get_ee_targets(env)
        base_lin_error = torch.norm(env.commands[:, :2] - env.simulator.base_lin_vel[:, :2], dim=1)
        base_yaw_error = torch.abs(env.commands[:, 2] - env.simulator.base_ang_vel[:, 2])
        ee_nominal_error = torch.norm(nominal - env.simulator.ee_pos, dim=1)
        ee_effective_error = torch.norm(effective - env.simulator.ee_pos, dim=1)
        force_offset = torch.norm(effective - nominal, dim=1)
        sample = {
            "base_lin": base_lin_error.mean().item(),
            "base_yaw": base_yaw_error.mean().item(),
            "ee_nominal": ee_nominal_error.mean().item(),
            "ee_effective": ee_effective_error.mean().item(),
            "force_offset": force_offset.mean().item(),
            "ee_force_ext": torch.norm(env.ee_force_ext_world, dim=1).mean().item(),
            "ee_force_cmd": torch.norm(env.current_Fxyz_gripper_cmd, dim=1).mean().item(),
            "base_force_ext": torch.norm(env.base_force_ext_world, dim=1).mean().item(),
            "base_force_cmd": torch.norm(env.current_Fxyz_base_cmd, dim=1).mean().item(),
        }
        self.window.append(sample)
        self.all_samples.append(sample)
        return sample

    def print_window(self, step):
        if not self.window:
            return
        means = self._means(self.window)
        print(
            f"[eval step {step}] "
            f"base_xy_err={means['base_lin']:.3f} m/s, "
            f"yaw_err={means['base_yaw']:.3f} rad/s, "
            f"ee_eff_err={means['ee_effective']:.3f} m, "
            f"ee_nom_err={means['ee_nominal']:.3f} m, "
            f"target_offset={means['force_offset']:.3f} m, "
            f"F_ee_ext/cmd={means['ee_force_ext']:.1f}/{means['ee_force_cmd']:.1f} N",
            flush=True,
        )

    def print_summary(self):
        if not self.all_samples:
            return
        means = self._means(self.all_samples)
        print(
            "UniFP eval summary: "
            f"base_xy_err={means['base_lin']:.3f} m/s, "
            f"yaw_err={means['base_yaw']:.3f} rad/s, "
            f"ee_eff_err={means['ee_effective']:.3f} m, "
            f"ee_nom_err={means['ee_nominal']:.3f} m",
            flush=True,
        )

    @staticmethod
    def _means(samples):
        keys = samples[0].keys()
        return {key: float(np.mean([sample[key] for sample in samples])) for key in keys}


class EvalVisualizer:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.nominal_trail = deque(maxlen=args.trail_length)
        self.effective_trail = deque(maxlen=args.trail_length)
        self.ee_trail = deque(maxlen=args.trail_length)

    def draw(self):
        if self.env.headless:
            return
        scene = getattr(self.env.simulator, "_scene", None)
        if scene is not None and not self.args.render_ee_goal_debug:
            scene.clear_debug_objects()
        if self.args.render_ee_goal_debug:
            self.env.draw_ee_goal_debug_vis()
        if self.args.render_base_velocity_arrows:
            self._draw_base_velocity_arrows()
        if self.args.render_ee_trails:
            self._draw_ee_trails()

    def _draw_base_velocity_arrows(self):
        scene = getattr(self.env.simulator, "_scene", None)
        if scene is None:
            return
        env_id = int(self.args.debug_robot_id)
        yaw_quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.env.device),
            torch.zeros(1, device=self.env.device),
            self.env.simulator.base_euler[env_id : env_id + 1, 2],
        )
        origin = self.env.simulator.base_pos[env_id] + torch.tensor([0.0, 0.0, 0.35], device=self.env.device)
        cmd_local = torch.zeros(1, 3, device=self.env.device)
        vel_local = torch.zeros(1, 3, device=self.env.device)
        cmd_local[0, :2] = self.env.commands[env_id, :2]
        vel_local[0, :2] = self.env.simulator.base_lin_vel[env_id, :2]
        cmd_world = quat_apply(yaw_quat, cmd_local)[0] * self.args.velocity_arrow_scale
        vel_world = quat_apply(yaw_quat, vel_local)[0] * self.args.velocity_arrow_scale
        scene.draw_debug_arrow(
            origin.detach().cpu().numpy(),
            vec=cmd_world.detach().cpu().numpy(),
            radius=0.015,
            color=(0.0, 1.0, 0.0, 0.9),
        )
        scene.draw_debug_arrow(
            (origin + torch.tensor([0.0, 0.0, 0.05], device=self.env.device)).detach().cpu().numpy(),
            vec=vel_world.detach().cpu().numpy(),
            radius=0.015,
            color=(1.0, 0.0, 0.0, 0.9),
        )

    def _draw_ee_trails(self):
        scene = getattr(self.env.simulator, "_scene", None)
        if scene is None:
            return
        env_id = int(self.args.debug_robot_id)
        nominal, effective = get_ee_targets(self.env)
        self.nominal_trail.append(nominal[env_id].detach().clone())
        self.effective_trail.append(effective[env_id].detach().clone())
        self.ee_trail.append(self.env.simulator.ee_pos[env_id].detach().clone())
        if len(self.nominal_trail) > 1:
            scene.draw_debug_spheres(torch.stack(list(self.nominal_trail)), radius=0.015, color=(1.0, 1.0, 0.0, 0.45))
            scene.draw_debug_spheres(torch.stack(list(self.effective_trail)), radius=0.015, color=(1.0, 0.0, 1.0, 0.45))
            scene.draw_debug_spheres(torch.stack(list(self.ee_trail)), radius=0.015, color=(0.0, 0.0, 1.0, 0.45))


def get_ee_targets(env):
    base_yaw_quat = quat_from_euler_xyz(
        torch.zeros(env.num_envs, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
        env.simulator.base_euler[:, 2],
    )
    force_offset = (env.ee_force_ext_world + quat_apply(base_yaw_quat, env.current_Fxyz_gripper_cmd)) / env.gripper_force_kps
    nominal = env.curr_ee_goal_cart_world
    effective = nominal + force_offset
    return nominal, effective


def update_camera(env, args, robot_index):
    if args.camera_preset == "none" and not args.follow_robot:
        return
    base = env.simulator.base_pos[robot_index].detach().cpu().numpy()
    ee = env.simulator.ee_pos[robot_index].detach().cpu().numpy()
    if args.follow_robot or args.camera_preset == "follow":
        pos = base + np.array(env.cfg.viewer.pos, dtype=np.float32)
        lookat = base + np.array(env.cfg.viewer.lookat, dtype=np.float32)
    elif args.camera_preset == "side":
        pos = base + np.array([0.0, 2.4, 0.9], dtype=np.float32)
        lookat = base + np.array([0.25, 0.0, 0.35], dtype=np.float32)
    elif args.camera_preset == "front":
        pos = base + np.array([2.0, 0.0, 0.8], dtype=np.float32)
        lookat = base + np.array([0.25, 0.0, 0.35], dtype=np.float32)
    elif args.camera_preset == "top":
        pos = base + np.array([0.0, 0.0, 3.0], dtype=np.float32)
        lookat = base + np.array([0.2, 0.0, 0.0], dtype=np.float32)
    elif args.camera_preset == "arm":
        pos = base + np.array([0.75, -1.25, 0.95], dtype=np.float32)
        lookat = 0.5 * (ee + env.curr_ee_goal_cart_world[robot_index].detach().cpu().numpy())
    else:
        return
    env.set_camera(pos, lookat)
    floating_camera = getattr(env.simulator, "_floating_camera", None)
    if floating_camera is not None:
        floating_camera.render()


def interaction_loop(env, policy, args):
    robot_index = 0
    base_mode = args.base_command_mode
    if base_mode == "auto":
        base_mode = "joystick" if args.use_joystick else "random"
    joystick = Joystick(joystick_type=args.joystick_type) if base_mode == "joystick" else None
    scheduler = CommandScheduler(env, args)
    metrics = EvalMetrics(args.metrics_window_steps)
    visualizer = EvalVisualizer(env, args)
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
        f"base_mode={base_mode}, "
        f"ee_mode={args.ee_eval_mode}, "
        f"camera={args.camera_preset}, "
        f"ee_goal_debug={args.render_ee_goal_debug}",
        flush=True,
    )
    if args.render_ee_goal_debug:
        print(
            "Debug colors: yellow=nominal EE target, magenta=force-offset EE target, "
            "blue=current EE, cyan=spherical center, orange=reconstructed TCP, "
            "green=commanded base velocity, red=measured base velocity",
            flush=True,
        )

    for step in range(num_steps):
        scheduler.update(step, joystick)
        update_camera(env, args, robot_index)

        policy_info = {}
        actions = policy({"obs": obs_history.detach()}, policy_info=policy_info)
        if args.use_unifp_impedance_controller:
            env.set_impedance_force_estimates(policy_info["latents"])
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, rews, dones, infos, grfs = env.step(actions.detach())
        scheduler.reapply()
        metrics.update(env)
        if args.metrics_interval_steps > 0 and step % args.metrics_interval_steps == 0:
            metrics.print_window(step)
        visualizer.draw()

    metrics.print_summary()


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
    parser.add_argument(
        "--base_command_mode",
        type=str,
        default="auto",
        choices=["auto", "joystick", "fixed", "random", "scripted"],
    )
    parser.add_argument(
        "--ee_eval_mode",
        type=str,
        default="env_sampled",
        choices=["env_sampled", "fixed_sphere", "random_sphere", "scripted_sphere"],
    )
    parser.add_argument("--base_command_hold_s", type=float, default=3.0)
    parser.add_argument("--ee_command_hold_s", type=float, default=3.0)
    parser.add_argument("--fixed_ee_sphere", type=float, nargs=3, default=[0.55, 0.0, 0.0])

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
    parser.add_argument("--render_ee_frame_debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render_base_velocity_arrows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render_ee_trails", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trail_length", type=int, default=120)
    parser.add_argument("--velocity_arrow_scale", type=float, default=0.6)
    parser.add_argument("--debug_robot_id", type=int, default=0)
    parser.add_argument("--metrics_interval_steps", type=int, default=50)
    parser.add_argument("--metrics_window_steps", type=int, default=100)
    parser.add_argument(
        "--camera_preset",
        type=str,
        default="none",
        choices=["none", "follow", "side", "front", "top", "arm"],
    )

    args = parser.parse_args()
    if args.gpu is None:
        args.gpu = args.device
    configure_runtime_device(args)

    play(args)
