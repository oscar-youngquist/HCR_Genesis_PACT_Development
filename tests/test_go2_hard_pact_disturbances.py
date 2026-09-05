"""Deterministic tests for curriculum-driven persistent HardPACT wrenches."""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("SIMULATOR", "genesis_pact")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_hard_pact_tests")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_hard_pact_tests")

import torch

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg
from legged_gym.envs.go2.go2_hard_pact.transition import (
    DISTURBANCE_CRITIC_DIM,
    DISTURBANCE_FIELD_DIMS,
    added_mass_gravity_wrench_world,
    pack_disturbance_fields,
    physics_transition_mask,
    wrench_world_to_scaled_yaw_local,
)
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import GO2HardPACTPosCfg
from legged_gym.envs.go2.go2_pact.go2_pact_config import GO2PACTCfg
from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos_config import GO2PACTPosCfg


def _persistent_task(num_envs=1, progress=1.0):
    task = Go2HardPACT.__new__(Go2HardPACT)
    task.num_envs = num_envs
    task.device = "cpu"
    task.dt = GO2HardPACTCfg.control.dt
    task.common_step_counter = 0
    task.cfg = GO2HardPACTCfg()
    task.simulator = SimpleNamespace(domain_rand_disturbance_progress=progress)
    task._persistent_wrench_target_world = torch.zeros(num_envs, 6)
    task._current_sustained_wrench_world = torch.zeros(num_envs, 6)
    task._persistent_component_active = torch.zeros(num_envs, 2, dtype=torch.bool)
    task._current_sustained_active_mask = torch.zeros(num_envs, 1, dtype=torch.bool)
    task._persistent_start_step = torch.zeros(num_envs, 2, dtype=torch.long)
    task._persistent_end_step = torch.zeros(num_envs, 2, dtype=torch.long)
    task._persistent_duration_steps = torch.zeros(num_envs, 2, dtype=torch.long)
    task._persistent_next_event_step = torch.zeros(num_envs, 2, dtype=torch.long)
    return task


class _WrenchSolver:
    def __init__(self):
        self.forces = []
        self.torques = []

    def apply_links_external_force(self, **kwargs):
        self.forces.append(kwargs["force"].clone())

    def apply_links_external_torque(self, **kwargs):
        self.torques.append(kwargs["torque"].clone())


class PersistentWrenchTests(unittest.TestCase):
    def test_curriculum_progress_scales_only_new_external_wrench(self):
        task = _persistent_task(progress=0.0)
        self.assertEqual(task._persistent_component_settings(0)[3], 10.0)
        self.assertEqual(task._persistent_component_settings(1)[3], 3.0)
        task.simulator.domain_rand_disturbance_progress = 0.5
        self.assertEqual(task._persistent_component_settings(0)[3], 35.0)
        self.assertEqual(task._persistent_component_settings(1)[3], 7.5)
        task.simulator.domain_rand_disturbance_progress = 1.0
        self.assertEqual(task._persistent_component_settings(0)[3], 60.0)
        self.assertEqual(task._persistent_component_settings(1)[3], 12.0)
        self.assertTrue(task.cfg.domain_rand.push_robots)

    def test_ramp_hold_ramp_down(self):
        task = _persistent_task()
        task._persistent_component_active[0, 0] = True
        task._persistent_start_step[0, 0] = 0
        task._persistent_end_step[0, 0] = 8
        task._persistent_duration_steps[0, 0] = 8
        task._persistent_next_event_step[:] = 1000
        task._persistent_wrench_target_world[0, :3] = torch.tensor([10., -4., 2.])
        for step, amplitude in {0: 0., 1: .5, 2: 1., 4: 1., 7: .5, 8: 0.}.items():
            task._update_persistent_wrench(step)
            torch.testing.assert_close(
                task._current_sustained_wrench_world[0, :3],
                torch.tensor([10., -4., 2.]) * amplitude,
            )
            torch.testing.assert_close(task._current_sustained_wrench_world[0, 3:], torch.zeros(3))

    def test_force_and_torque_events_are_independently_sampled(self):
        torch.manual_seed(9)
        task = _persistent_task()
        task.cfg.domain_rand.persistent_force_probability = 1.0
        task.cfg.domain_rand.persistent_torque_probability = 0.0
        task._persistent_next_event_step.zero_()
        task._update_persistent_wrench(0)
        self.assertTrue(task._persistent_component_active[0, 0])
        self.assertFalse(task._persistent_component_active[0, 1])
        self.assertTrue(task._persistent_wrench_target_world[0, :3].ne(0).any())
        torch.testing.assert_close(task._persistent_wrench_target_world[0, 3:], torch.zeros(3))

    def test_partial_reset_clears_only_selected_persistent_state(self):
        task = _persistent_task(num_envs=3)
        task._persistent_wrench_target_world[:] = 3
        task._current_sustained_wrench_world[:] = 2
        task._persistent_component_active[:] = True
        task._current_sustained_active_mask[:] = True
        task._reset_persistent_wrench_state(torch.tensor([1]))
        torch.testing.assert_close(task._persistent_wrench_target_world[1], torch.zeros(6))
        torch.testing.assert_close(task._persistent_wrench_target_world[0], torch.full((6,), 3.0))
        self.assertFalse(task._persistent_component_active[1].any())
        self.assertTrue(task._persistent_component_active[0].all())


class WrenchLabelTests(unittest.TestCase):
    def test_equivalent_gravity_wrench_identity_and_rotated_pose(self):
        mass = torch.tensor([[2.0]])
        gravity = [0.0, 0.0, -9.81]
        com = torch.tensor([[0.1, 0.0, 0.0]])
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        torch.testing.assert_close(
            added_mass_gravity_wrench_world(mass, gravity, com, identity),
            torch.tensor([[0., 0., -19.62, 0., 1.962, 0.]]), atol=1e-6, rtol=0,
        )
        half = torch.tensor(torch.pi / 4.0)
        yaw_90 = torch.tensor([[0., 0., half.sin(), half.cos()]])
        torch.testing.assert_close(
            added_mass_gravity_wrench_world(mass, gravity, com, yaw_90),
            torch.tensor([[0., 0., -19.62, -1.962, 0., 0.]]), atol=1e-5, rtol=0,
        )

    def test_frame_scaling_and_substep_application(self):
        half = torch.tensor(torch.pi / 4.0)
        yaw_90 = torch.tensor([[0., 0., half.sin(), half.cos()]])
        world = torch.tensor([[10., 0., 5., 2., 0., -1.]])
        torch.testing.assert_close(
            wrench_world_to_scaled_yaw_local(world, yaw_90, 0.01),
            torch.tensor([[0., -.1, .05, 0., -.02, -.01]]), atol=1e-6, rtol=0,
        )

        task = _persistent_task()
        task.cfg = SimpleNamespace(sim=SimpleNamespace(gravity=[0., 0., -10.]))
        task.obs_scales = SimpleNamespace(base_wrench=0.1)
        solver = _WrenchSolver()
        task.simulator = SimpleNamespace(
            _robot=SimpleNamespace(
                _solver=solver,
                get_quat=lambda: torch.tensor([[1., 0., 0., 0.]]),
            ),
            _base_link_index=0,
        )
        task._current_sustained_wrench_world[:] = torch.tensor([[3., 4., 5., .5, .6, .7]])
        task._current_sustained_active_mask[:] = True
        task._realized_added_mass = torch.tensor([[2.]])
        task._realized_com_shift_body = torch.tensor([[.1, 0., 0.]])
        for name in (
            "_disturbance_interval_sum_sustained", "_disturbance_interval_sum_mass_com",
            "_disturbance_interval_sum_total", "_disturbance_interval_sum_sustained_yaw_scaled",
            "_disturbance_interval_sum_mass_com_yaw_scaled", "_disturbance_interval_sum_yaw_scaled",
            "_disturbance_interval_sum_yaw_physical",
        ):
            setattr(task, name, torch.zeros(1, 6))
        task._disturbance_interval_count = torch.zeros(1, 1)
        task._begin_disturbance_interval()
        for _ in range(3):
            task._hard_pact_pre_physics_substep()
        interval = task._end_disturbance_interval()
        sustained = task._current_sustained_wrench_world
        mass_wrench = torch.tensor([[0., 0., -20., 0., 2., 0.]])
        torch.testing.assert_close(interval["applied_sustained_wrench_world"], sustained)
        torch.testing.assert_close(interval["equivalent_mass_com_wrench_world"], mass_wrench)
        torch.testing.assert_close(interval["total_external_wrench_label_world"], sustained + mass_wrench)
        torch.testing.assert_close(
            interval["total_external_wrench_label_yaw_normalized"],
            (sustained + mass_wrench)
            / torch.tensor([[100., 100., 100., 25., 25., 25.]]),
        )
        self.assertEqual(len(solver.forces), 3)
        self.assertEqual(len(solver.torques), 3)
        for force, torque in zip(solver.forces, solver.torques):
            torch.testing.assert_close(force[:, 0], sustained[:, :3])
            torch.testing.assert_close(torque[:, 0], sustained[:, 3:])


class TransitionTests(unittest.TestCase):
    def test_masks_and_named_critic_width(self):
        reset = torch.tensor([[False], [True], [False], [False]])
        timeout = torch.tensor([[False], [False], [True], [False]])
        teleport = torch.tensor([[False], [False], [False], [True]])
        self.assertEqual(
            physics_transition_mask(reset, timeout, teleport).flatten().tolist(),
            [True, False, False, False],
        )
        fields = {name: torch.ones(2, width) for name, width in DISTURBANCE_FIELD_DIMS}
        persistent_wrench = torch.tensor([
            [1., 2., 3., 4., 5., 6.],
            [-1., -2., -3., -4., -5., -6.],
        ])
        fields["applied_sustained_wrench_world"] = persistent_wrench
        packed = pack_disturbance_fields(fields)
        self.assertEqual(packed.shape, (2, DISTURBANCE_CRITIC_DIM))
        torch.testing.assert_close(packed[:, :6], persistent_wrench)
        self.assertEqual(DISTURBANCE_CRITIC_DIM, 33)
        self.assertEqual(GO2HardPACTCfg.env.num_privileged_obs, GO2PACTCfg.env.num_privileged_obs + 33)
        self.assertEqual(GO2HardPACTPosCfg.env.num_privileged_obs, GO2PACTPosCfg.env.num_privileged_obs + 33)


if __name__ == "__main__":
    unittest.main()
