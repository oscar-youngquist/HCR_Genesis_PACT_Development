"""Focused tests for shared B1Z1 force-adjusted EE target projection."""

import types
import unittest

from legged_gym.envs.b1z1.force_task_utils import (
    accumulate_ee_force_target_diagnostics,
    get_force_adjusted_ee_target,
    init_ee_force_target_diagnostics,
    invalidate_force_adjusted_ee_target_cache,
    reset_ee_force_target_diagnostics,
)
import torch


class ForceTargetProjectionTests(unittest.TestCase):
    @staticmethod
    def _environment(nominal, force, *, project=True, commanded_force=None):
        nominal = torch.as_tensor(nominal, dtype=torch.float32)
        force = torch.as_tensor(force, dtype=torch.float32)
        if nominal.ndim == 1:
            nominal = nominal.unsqueeze(0)
        if force.ndim == 1:
            force = force.unsqueeze(0)
        count = nominal.shape[0]
        env = types.SimpleNamespace(
            num_envs=count,
            device=torch.device("cpu"),
            dt=0.02,
            curr_ee_goal_cart_world=nominal.clone(),
            ee_force_ext_world=force.clone(),
            gripper_force_kps=torch.ones(count, 1),
            collision_lower_limits=torch.tensor([-0.9, -0.2, -0.7]),
            collision_upper_limits=torch.tensor([0.1, 0.2, -0.05]),
            underground_limit=-0.7,
            cfg=types.SimpleNamespace(
                goal_ee=types.SimpleNamespace(
                    max_ee_force_offset=1.0,
                    project_force_adjusted_ee_target=project,
                    force_target_radius_limits=[0.30, 0.90],
                    force_target_projection_samples=21,
                    ranges=types.SimpleNamespace(
                        pos_l=[0.40, 0.90],
                        init_pos_start=[0.50, 0.0, 0.0],
                        init_pos_end=[0.70, 0.0, 0.0],
                    ),
                )
            ),
        )
        if commanded_force is not None:
            env.current_Fxyz_gripper_cmd = torch.as_tensor(
                commanded_force, dtype=torch.float32
            ).reshape(count, 3)
        env._get_base_yaw_quat = lambda env_ids=None: torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]]
        ).repeat(count if env_ids is None else len(env_ids), 1)
        env.get_ee_goal_spherical_center = (
            lambda base_yaw_quat, env_ids=None: torch.zeros(
                base_yaw_quat.shape[0], 3
            )
        )
        env.simulator = types.SimpleNamespace(
            ee_pos=nominal.clone(),
            ee_vel=torch.zeros_like(nominal),
        )
        return env

    @staticmethod
    def _assert_valid(env, target):
        local = target.effective_target
        radius = torch.linalg.vector_norm(local, dim=-1)
        inside_body = torch.all(local > env.collision_lower_limits, dim=-1) & torch.all(
            local < env.collision_upper_limits, dim=-1
        )
        assert torch.all(radius >= 0.30 - 1.0e-6)
        assert torch.all(radius <= 0.90 + 1.0e-6)
        assert not torch.any(inside_body)
        assert torch.all(local[:, 2] >= env.underground_limit)

    def test_zero_force_leaves_valid_nominal_target_unchanged(self):
        env = self._environment([0.5, 0.0, 0.0], [0.0, 0.0, 0.0])
        target = get_force_adjusted_ee_target(env)
        self.assertTrue(torch.equal(target.effective_target, env.curr_ee_goal_cart_world))
        self.assertTrue(torch.equal(target.projection_alpha, torch.ones(1)))
        self.assertFalse(target.workspace_projected.any())

    def test_large_offsets_are_projected_into_workspace(self):
        nominal = torch.tensor(
            [
                [0.80, 0.0, 0.0],  # outward
                [0.40, 0.0, 0.0],  # inward
                [0.50, 0.0, 0.0],  # downward
                [0.80, 0.0, 0.0],  # toward the body box
            ]
        )
        offsets = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [-0.8, 0.0, -0.3],
            ]
        )
        env = self._environment(nominal, offsets)
        target = get_force_adjusted_ee_target(env)
        self._assert_valid(env, target)
        # The inward endpoint at x=-0.6 is already valid; the other three
        # offsets require radial, terrain, or body-box projection.
        self.assertEqual(target.workspace_projected.tolist(), [True, False, True, True])
        self.assertEqual(target.projection_alpha[1].item(), 1.0)
        for value in target:
            if value.is_floating_point():
                self.assertTrue(torch.isfinite(value).all())

    def test_mixed_batch_preserves_unprojected_members(self):
        env = self._environment(
            [[0.5, 0.0, 0.0], [0.8, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )
        target = get_force_adjusted_ee_target(env)
        self._assert_valid(env, target)
        self.assertEqual(target.workspace_projected.tolist(), [False, True])
        self.assertAlmostEqual(target.projection_alpha[0].item(), 1.0)

    def test_disabling_projection_matches_previous_norm_cap(self):
        env = self._environment([0.8, 0.0, 0.0], [2.0, 0.0, 0.0], project=False)
        env.cfg.goal_ee.max_ee_force_offset = 0.4
        target = get_force_adjusted_ee_target(env)
        expected_offset = torch.tensor([[0.4, 0.0, 0.0]])
        self.assertTrue(torch.allclose(target.applied_offset, expected_offset))
        self.assertTrue(
            torch.allclose(target.effective_target, env.curr_ee_goal_cart_world + expected_offset)
        )
        self.assertTrue(target.clipped.all())
        self.assertFalse(target.workspace_projected.any())

    def test_commanded_and_external_forces_still_cancel(self):
        env = self._environment(
            [0.5, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            commanded_force=[[0.2, 0.0, 0.0]],
        )
        target = get_force_adjusted_ee_target(env)
        self.assertTrue(torch.allclose(target.raw_offset, torch.zeros(1, 3)))
        self.assertTrue(torch.equal(target.effective_target, env.curr_ee_goal_cart_world))

    def test_full_batch_cache_matches_uncached_and_projects_once(self):
        env = self._environment(
            [[0.5, 0.0, 0.0], [0.8, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )
        init_ee_force_target_diagnostics(env)
        cached = get_force_adjusted_ee_target(env)
        self.assertIs(cached, get_force_adjusted_ee_target(env))
        uncached = get_force_adjusted_ee_target(env, use_cache=False)
        for cached_value, uncached_value in zip(cached, uncached):
            self.assertTrue(torch.equal(cached_value, uncached_value))
        self.assertEqual(env._force_adjusted_ee_target_compute_count, 1)

    def test_invalidation_prevents_pre_reset_target_reuse(self):
        env = self._environment([0.5, 0.0, 0.0], [0.1, 0.0, 0.0])
        init_ee_force_target_diagnostics(env)
        pre_reset = get_force_adjusted_ee_target(env).effective_target.clone()
        env.curr_ee_goal_cart_world[0, 0] = 0.7
        env.ee_force_ext_world.zero_()
        invalidate_force_adjusted_ee_target_cache(env)
        post_reset = get_force_adjusted_ee_target(env).effective_target
        self.assertFalse(torch.equal(pre_reset, post_reset))
        self.assertTrue(torch.equal(post_reset, env.curr_ee_goal_cart_world))
        self.assertEqual(env._force_adjusted_ee_target_compute_count, 2)

    def test_reset_suppresses_target_velocity_spike(self):
        env = self._environment([0.5, 0.0, 0.0], [0.0, 0.0, 0.0])
        init_ee_force_target_diagnostics(env)
        reset_ee_force_target_diagnostics(env, torch.tensor([0]))
        env.curr_ee_goal_cart_world[0, 0] = 0.6
        env.simulator.ee_pos.copy_(env.curr_ee_goal_cart_world)
        accumulate_ee_force_target_diagnostics(env)
        velocity_error = env.ee_force_target_diagnostic_sums[
            "EE/relative_velocity_error_mean"
        ]
        self.assertTrue(torch.equal(velocity_error, torch.zeros(1)))
        self.assertTrue(torch.isfinite(velocity_error).all())


if __name__ == "__main__":
    unittest.main()
