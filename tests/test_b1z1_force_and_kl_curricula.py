"""Focused regression tests for B1Z1 force masks and iteration-level KL duals."""

import types
import unittest

import torch

from legged_gym.envs.b1z1.force_task_utils import (
    force_curriculum_active,
    force_neutral_mask,
    zero_velocity_probability,
)
from legged_gym.envs.b1z1.b1z1_unifp.b1z1_unifp import B1Z1UniFP
from rsl_rl.algorithms.kl_rate_band import (
    KLRateBandController,
    update_duals_from_mean,
)


class ForceTaskTests(unittest.TestCase):
    def test_force_activation_boundary_selects_correct_zero_probability(self):
        hold = 8000
        before = force_curriculum_active(hold - 1, hold)
        at_boundary = force_curriculum_active(hold, hold)
        self.assertFalse(before)
        self.assertTrue(at_boundary)
        self.assertEqual(zero_velocity_probability(before, 0.1, 0.8), 0.1)
        self.assertEqual(zero_velocity_probability(at_boundary, 0.1, 0.8), 0.8)

        env = object.__new__(B1Z1UniFP)
        env.cfg = types.SimpleNamespace(
            commands=types.SimpleNamespace(
                command_force_hold_iterations=hold,
                command_force_ramp_iterations=5000,
                command_force_initial_scale=0.25,
                command_force_final_scale=1.0,
            )
        )
        env.training_iteration = hold - 1
        self.assertFalse(env.force_command_randomization_active)
        self.assertTrue(env.force_command_stream_enabled)
        self.assertEqual(env.command_force_scale, 0.25)
        env.training_iteration = hold
        self.assertTrue(env.force_command_randomization_active)

    def _environment(self):
        env = types.SimpleNamespace()
        env.num_envs = 5
        env.device = torch.device("cpu")
        env.cfg = types.SimpleNamespace(
            rewards=types.SimpleNamespace(force_neutral_threshold=1.0e-3)
        )
        for name in (
            "current_Fxyz_gripper_cmd", "current_Fxyz_base_cmd",
            "ee_force_ext_world", "base_force_ext_world",
        ):
            setattr(env, name, torch.zeros(env.num_envs, 3))
        return env

    def test_each_commanded_or_applied_force_disables_standing(self):
        env = self._environment()
        names = (
            "current_Fxyz_gripper_cmd", "current_Fxyz_base_cmd",
            "ee_force_ext_world", "base_force_ext_world",
        )
        for index, name in enumerate(names):
            getattr(env, name)[index, 0] = 0.01
        expected = torch.tensor([False, False, False, False, True])
        self.assertTrue(torch.equal(force_neutral_mask(env), expected))

    def test_neutral_stationary_mask_preserves_both_raw_standing_rewards(self):
        env = self._environment()
        walking = torch.tensor([False, True, False, False, False])
        env.current_Fxyz_gripper_cmd[2, 0] = 0.01
        strict_standing = (~walking) & force_neutral_mask(env)
        original_pose_reward = torch.arange(1.0, 6.0)
        original_contact_reward = torch.ones(5)
        self.assertEqual((original_pose_reward * strict_standing)[0].item(), 1.0)
        self.assertEqual((original_contact_reward * strict_standing)[0].item(), 1.0)
        self.assertEqual((original_pose_reward * strict_standing)[1].item(), 0.0)
        self.assertEqual((original_contact_reward * strict_standing)[2].item(), 0.0)

    def test_actual_standing_rewards_share_force_and_walking_mask(self):
        env = object.__new__(B1Z1UniFP)
        env.num_envs = 5
        env.device = torch.device("cpu")
        env.all_env_ids = torch.arange(env.num_envs)
        env.cfg = types.SimpleNamespace(
            rewards=types.SimpleNamespace(force_neutral_threshold=1.0e-3),
            commands=types.SimpleNamespace(
                lin_vel_x_clip=0.05, lin_vel_y_clip=0.05,
                ang_vel_yaw_clip=0.1,
            ),
        )
        env.commands = torch.zeros(env.num_envs, 15)
        env.commands[4, 0] = 0.5
        for name in (
            "current_Fxyz_gripper_cmd", "current_Fxyz_base_cmd",
            "ee_force_ext_world", "base_force_ext_world",
        ):
            setattr(env, name, torch.zeros(env.num_envs, 3))
        env.current_Fxyz_gripper_cmd[0, 0] = 0.01
        env.current_Fxyz_base_cmd[1, 0] = 0.01
        env.ee_force_ext_world[2, 0] = 0.01
        env.base_force_ext_world[3, 0] = 0.01
        env.simulator = types.SimpleNamespace(
            dof_pos=torch.ones(env.num_envs, 12),
            default_dof_pos=torch.zeros(env.num_envs, 12),
            feet_indices=torch.arange(4),
            link_contact_forces=torch.zeros(env.num_envs, 4, 3),
        )
        env.simulator.link_contact_forces[:, :, 2] = 2.0

        self.assertTrue(torch.equal(env._reward_stand_still(), torch.zeros(5)))
        self.assertTrue(torch.equal(env._reward_stand_still_contact(), torch.zeros(5)))

        env.commands.zero_()
        for name in (
            "current_Fxyz_gripper_cmd", "current_Fxyz_base_cmd",
            "ee_force_ext_world", "base_force_ext_world",
        ):
            getattr(env, name).zero_()
        self.assertTrue(torch.equal(env._reward_stand_still(), torch.full((5,), 12.0)))
        self.assertTrue(torch.equal(env._reward_stand_still_contact(), torch.ones(5)))


class KLIterationUpdateTests(unittest.TestCase):
    @staticmethod
    def _controller():
        return KLRateBandController(
            warmup_iters=0, warmup_beta_max=1.0,
            rate_min=0.1, rate_max=1.0, dual_lr=1.0e-3,
            augmented_rho=0.1, ema_decay=0.9,
        )

    def test_one_dual_update_receives_iteration_mean(self):
        controller = self._controller()
        calls = []
        original = controller.update_duals

        def counted(raw_kl, iteration):
            calls.append((raw_kl.item(), iteration))
            original(raw_kl, iteration)

        controller.update_duals = counted
        mean = update_duals_from_mean(controller, 6.0, 4, 12, "cpu")
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(mean.item(), 1.5)
        self.assertEqual(calls[0][1], 12)

    def test_duals_are_invariant_to_partitioning_with_same_mean(self):
        first, second = self._controller(), self._controller()
        update_duals_from_mean(first, sum([0.5, 1.5]), 2, 3, "cpu")
        update_duals_from_mean(second, sum([0.25, 0.75, 1.25, 1.75]), 4, 3, "cpu")
        self.assertEqual(first.state_dict(), second.state_dict())


if __name__ == "__main__":
    unittest.main()
