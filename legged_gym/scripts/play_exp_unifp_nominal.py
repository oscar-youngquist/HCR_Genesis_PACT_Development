"""Deterministic nominal evaluation for the Genesis B1Z1 UniFP policy.

This wrapper reuses the visualization and metrics from ``play_exp_unifp.py``
while removing training-time disturbances/randomization and preserving the
smooth EE command trajectories used during training.
"""

import argparse

import torch

import play_exp_unifp as base_play


def override_nominal_configs(env_cfg, train_cfg, args):
    """Apply the normal play overrides, then make evaluation deterministic."""
    base_play._original_override_configs(env_cfg, train_cfg, args)

    # The evaluator owns base and EE commands.  Keep the environment callback
    # from replacing them with randomly sampled commands during a rollout.
    env_cfg.commands.resampling_time = 1.0e9

    # Disable both virtual commanded-force profiles and physical disturbances.
    env_cfg.commands.force_start_step = 1_000_000_000
    env_cfg.commands.push_gripper_stators = False
    env_cfg.commands.push_robot_base = False
    env_cfg.commands.apply_ee_external_forces = False
    env_cfg.commands.apply_base_external_forces = False
    env_cfg.commands.use_external_impedance_compensation = False
    env_cfg.commands.compensate_ee_external_force = False
    env_cfg.commands.compensate_base_external_force = False

    # Nominal evaluation should test the learned controller, not a freshly
    # randomized plant.  Re-enable these explicitly in the training-style play
    # script when robustness is what is being evaluated.
    dr = env_cfg.domain_rand
    for name in (
        "use_domainrand_curriculum",
        "randomize_friction",
        "randomize_base_mass",
        "randomize_gripper_mass",
        "randomize_com_displacement",
        "randomize_ctrl_delay",
        "randomize_pd_gain",
        "randomize_motor_strength",
        "randomize_joint_armature",
        "randomize_joint_friction",
        "randomize_joint_stiffness",
        "randomize_joint_damping",
        "push_robots",
    ):
        if hasattr(dr, name):
            setattr(dr, name, False)

    # Start with a flat surface so basic command tracking is unambiguous.
    if env_cfg.terrain.mesh_type in ("heightfield", "trimesh"):
        env_cfg.terrain.terrain_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": 0.0,
            "max_height": 0.0,
            "step": 0.005,
            "downsampled_scale": 0.2,
        }


class SmoothCommandScheduler(base_play.CommandScheduler):
    """Schedule new EE goals without discontinuously teleporting the command."""

    def __init__(self, env, args):
        super().__init__(env, args)
        self._applied_ee_sphere = None

    def _apply_ee_sphere(self):
        # The parent scheduler calls this every policy step.  Only initialize a
        # new trajectory when the desired endpoint actually changes.
        sphere = self.ee_sphere.to(device=self.env.device, dtype=torch.float)
        if self._applied_ee_sphere is not None and torch.equal(sphere, self._applied_ee_sphere):
            return

        target = sphere.unsqueeze(0).expand(self.env.num_envs, -1)
        self.env.ee_start_sphere.copy_(self.env.curr_ee_goal_sphere)
        self.env.ee_goal_sphere.copy_(target)
        self.env.goal_timer.zero_()

        transition_steps = max(1.0, self.args.ee_transition_s / self.env.dt)
        hold_steps = max(transition_steps + 1.0, self.args.ee_command_hold_s / self.env.dt)
        self.env.traj_timesteps.fill_(transition_steps)
        self.env.traj_total_timesteps.fill_(hold_steps)
        self._applied_ee_sphere = sphere.clone()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Nominal deterministic evaluation of a Genesis B1Z1 UniFP checkpoint."
    )
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
    parser.add_argument("--joystick_type", choices=["xbox", "switch"], default="xbox")
    parser.add_argument("--follow_robot", action="store_true", default=False)
    parser.add_argument("--record_frames", action="store_true", default=False)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument(
        "--base_command_mode",
        choices=["auto", "joystick", "fixed", "random", "scripted"],
        default="fixed",
    )
    parser.add_argument(
        "--ee_eval_mode",
        choices=["env_sampled", "fixed_sphere", "random_sphere", "scripted_sphere"],
        default="random_sphere",
    )
    parser.add_argument("--base_command_hold_s", type=float, default=5.0)
    parser.add_argument("--ee_command_hold_s", type=float, default=5.0)
    parser.add_argument("--ee_transition_s", type=float, default=2.0)
    parser.add_argument("--fixed_ee_sphere", type=float, nargs=3, default=[0.55, 0.0, 0.0])

    parser.add_argument("--cmd_x", type=float, default=0.4)
    parser.add_argument("--cmd_y", type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--max_lin_vel_x", type=float, default=0.6)
    parser.add_argument("--max_lin_vel_y", type=float, default=0.4)
    parser.add_argument("--max_yaw_vel", type=float, default=0.6)

    # Retained for compatibility with the shared play helpers.  Nominal config
    # overrides these values unconditionally.
    parser.add_argument("--force_start_step", type=int, default=1_000_000_000)
    parser.add_argument("--apply_ee_external_forces", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--apply_base_external_forces", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_unifp_impedance_controller", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compensate_ee_external_force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compensate_base_external_force", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--render_ee_goal_debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render_ee_frame_debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render_base_velocity_arrows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render_ee_trails", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trail_length", type=int, default=120)
    parser.add_argument("--velocity_arrow_scale", type=float, default=0.6)
    parser.add_argument("--debug_robot_id", type=int, default=0)
    parser.add_argument("--metrics_interval_steps", type=int, default=50)
    parser.add_argument("--metrics_window_steps", type=int, default=100)
    parser.add_argument(
        "--camera_preset",
        choices=["none", "follow", "side", "front", "top", "arm"],
        default="none",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.gpu is None:
        args.gpu = args.device

    # Patch only the two extension points used by base_play.play(); all metrics,
    # visualization, checkpoint loading, and policy inference remain shared.
    base_play._original_override_configs = base_play.override_configs
    base_play.override_configs = override_nominal_configs
    base_play.CommandScheduler = SmoothCommandScheduler
    base_play.configure_runtime_device(args)
    base_play.play(args)
