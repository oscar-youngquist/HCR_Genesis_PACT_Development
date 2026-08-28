"""Focused checks for nominal UniFP play scheduling and force filtering."""

import unittest
from types import SimpleNamespace

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass
import torch

from legged_gym.envs.b1z1.b1z1_unifp.b1z1_unifp import B1Z1UniFP
from legged_gym.scripts.play_exp_unifp import CommandScheduler


class B1Z1UniFPPlaySmoothingTests(unittest.TestCase):
    def test_scheduler_starts_minimum_jerk_trajectory_without_teleporting(self):
        env = SimpleNamespace(
            device=torch.device("cpu"), dt=0.02, num_envs=2,
            all_env_ids=torch.arange(2),
            command_ranges={
                "lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.0, 1.0],
            },
            commands=torch.zeros(2, 15),
            curr_ee_goal_sphere=torch.tensor([[0.40, 0.10, 0.20], [0.50, 0.20, 0.30]]),
            ee_start_sphere=torch.zeros(2, 3), ee_goal_sphere=torch.zeros(2, 3),
            goal_timer=torch.ones(2), traj_timesteps=torch.zeros(2),
            traj_total_timesteps=torch.zeros(2),
        )
        args = SimpleNamespace(
            base_command_mode="fixed", use_joystick=False,
            ee_eval_mode="fixed_sphere", base_command_hold_s=10.0,
            ee_command_hold_s=5.0, ee_transition_s=2.0,
            cmd_x=0.0, cmd_y=0.0, cmd_yaw=0.0,
            fixed_ee_sphere=[0.70, -0.10, 0.40],
            max_lin_vel_x=0.6, max_lin_vel_y=0.4, max_yaw_vel=0.6,
        )
        scheduler = CommandScheduler(env, args)
        before = env.curr_ee_goal_sphere.clone()

        scheduler._apply_ee_sphere()

        self.assertTrue(torch.equal(env.curr_ee_goal_sphere, before))
        self.assertTrue(torch.equal(env.ee_start_sphere, before))
        expected = torch.tensor([[0.70, -0.10, 0.40]]).expand(2, -1)
        self.assertTrue(torch.equal(env.ee_goal_sphere, expected))
        self.assertTrue(torch.equal(env.traj_timesteps, torch.full((2,), 100.0)))
        self.assertTrue(torch.equal(env.traj_total_timesteps, torch.full((2,), 250.0)))

    def test_impedance_estimates_are_filtered_and_reset_selectively(self):
        env = B1Z1UniFP.__new__(B1Z1UniFP)
        env.dt = 0.02
        env.cfg = SimpleNamespace(commands=SimpleNamespace(
            ee_impedance_force_filter_tau=0.3,
            base_impedance_force_filter_tau=0.3,
            compensate_ee_external_force=True,
            compensate_base_external_force=True,
        ))
        env.estimated_ee_force_local = torch.full((2, 3), 16.0)
        env.estimated_base_force_local = torch.full((2, 3), 32.0)
        env.filtered_ee_force_local = torch.zeros(2, 3)
        env.filtered_base_force_local = torch.zeros(2, 3)
        env.current_Fxyz_gripper_cmd = torch.zeros(2, 3)
        env.current_Fxyz_base_cmd = torch.zeros(2, 3)
        env.commands = torch.zeros(2, 15)

        env._apply_external_impedance_compensation()

        self.assertTrue(torch.allclose(env.filtered_ee_force_local, torch.ones(2, 3)))
        self.assertTrue(torch.allclose(env.filtered_base_force_local, torch.full((2, 3), 2.0)))
        self.assertTrue(torch.equal(env.commands[:, 9:12], -env.filtered_ee_force_local))
        self.assertTrue(torch.equal(env.commands[:, 12:15], -env.filtered_base_force_local))

        env._reset_impedance_force_filters(torch.tensor([1]))
        self.assertEqual(env.filtered_ee_force_local[0, 0].item(), 1.0)
        self.assertEqual(env.filtered_ee_force_local[1].count_nonzero().item(), 0)
        self.assertEqual(env.filtered_base_force_local[1].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
