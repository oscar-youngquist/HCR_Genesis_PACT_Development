"""Unit tests for HardPACT GRF conditioning and interval targets."""

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
from legged_gym.envs.go2.go2_hard_pact.grf import (
    GRFProcessingConfig,
    IntervalGRFProcessor,
    world_to_yaw_local,
)
from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
from legged_gym.envs.go2.go2_pact.go2_pact_config import GO2PACTCfg
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos import Go2HardPACTPos
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import (
    GO2HardPACTPosCfg,
)
from legged_gym.simulator.genesis_simulator_pact import GenesisSimulator_PACT
from legged_gym.simulator.genesis_simulator_pact_pos import GenesisSimulator_PACT_Pos


def _processor(num_envs=1, num_feet=2, **overrides):
    values = dict(
        vertical_deadband_n=3.0,
        clip_min_n=-10.0,
        clip_max_n=10.0,
        ema_alpha=0.5,
        contact_threshold_n=5.0,
    )
    values.update(overrides)
    return IntervalGRFProcessor(
        num_envs, num_feet, "cpu", torch.float32, GRFProcessingConfig(**values)
    )


class _FakeScene:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class _FakeRobot:
    def __init__(self, scene, samples):
        self.scene = scene
        self.samples = samples
        self.controlled = []

    def control_dofs_force(self, torques, indices):
        self.controlled.append(torques.clone())

    def get_dofs_position(self, indices):
        return torch.zeros(1, 12)

    def get_dofs_velocity(self, indices):
        return torch.zeros(1, 12)

    def get_links_net_contact_force(self):
        return self.samples[self.scene.steps - 1]


def _synthetic_simulator(samples, install_callback=True, sim_cls=GenesisSimulator_PACT):
    sim = sim_cls.__new__(sim_cls)
    sim._cfg = SimpleNamespace(control=SimpleNamespace(decimation=len(samples)))
    sim._scene = _FakeScene()
    sim._robot = _FakeRobot(sim._scene, samples)
    sim._dof_indices = torch.arange(12)
    sim._feet_indices = torch.arange(4)
    sim._last_base_lin_vel = torch.zeros(1, 3)
    sim._base_lin_vel = torch.zeros(1, 3)
    sim._last_base_ang_vel = torch.zeros(1, 3)
    sim._base_ang_vel = torch.zeros(1, 3)
    sim._last_feet_vel = torch.zeros(1, 4, 3)
    sim._feet_vel = torch.zeros(1, 4, 3)
    sim._last_dof_vel = torch.zeros(1, 12)
    sim._dof_vel = torch.zeros(1, 12)
    sim._dof_pos = torch.zeros(1, 12)
    sim._last_base_world_lin_vel = torch.zeros(1, 3)
    sim._base_world_lin_vel = torch.zeros(1, 3)
    sim._last_base_world_ang_vel = torch.zeros(1, 3)
    sim._base_world_ang_vel = torch.zeros(1, 3)
    sim._base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    sim._compute_torques = lambda actions: actions[:, :12]
    sim.first_loop = True
    processor = _processor(num_envs=1, num_feet=4)
    if install_callback:
        sim._hard_pact_grf_post_physics_substep = lambda: processor.update_substep(
            sim._robot.get_links_net_contact_force()[:, sim._feet_indices, :]
        )
    return sim, processor


class GRFProcessorTests(unittest.TestCase):
    def test_world_force_is_rotated_into_pre_step_yaw_frame(self):
        half = torch.tensor(torch.pi / 4.0)
        quaternion = torch.tensor([[0.0, 0.0, half.sin(), half.cos()]])
        world = torch.tensor([[[1.0, 0.0, 2.0]]])
        expected = torch.tensor([[[0.0, -1.0, 2.0]]])
        torch.testing.assert_close(
            world_to_yaw_local(world, quaternion), expected, atol=1.0e-6, rtol=0.0
        )

    def test_whole_vector_deadband_clipping_ema_and_contacts(self):
        processor = _processor()
        raw = torch.tensor([[[9.0, -8.0, 2.0], [20.0, -20.0, 8.0]]])
        processor.begin_interval()
        processor.update_substep(raw)

        torch.testing.assert_close(processor.raw, raw)
        torch.testing.assert_close(processor.complete[0, 0], torch.zeros(3))
        torch.testing.assert_close(
            processor.complete[0, 1], torch.tensor([20.0, -20.0, 8.0])
        )
        torch.testing.assert_close(
            processor.clipped[0, 1], torch.tensor([10.0, -10.0, 8.0])
        )
        torch.testing.assert_close(
            processor.ema[0, 1], torch.tensor([5.0, -5.0, 4.0])
        )
        self.assertFalse(processor.contacts[0, 0])
        self.assertTrue(processor.contacts[0, 1])

        second = torch.tensor([[[4.0, 2.0, 6.0], [0.0, 0.0, 0.0]]])
        processor.update_substep(second)
        torch.testing.assert_close(
            processor.ema[0, 0], torch.tensor([2.0, 1.0, 3.0])
        )
        torch.testing.assert_close(
            processor.ema[0, 1], torch.tensor([2.5, -2.5, 2.0])
        )
        self.assertTrue(processor.contacts[0, 0])
        self.assertFalse(processor.contacts[0, 1])

    def test_multi_substep_average_includes_swing_zeros(self):
        processor = _processor()
        samples = (
            torch.tensor([[[4.0, 2.0, 6.0], [10.0, 10.0, 2.0]]]),
            torch.tensor([[[8.0, 4.0, 10.0], [0.0, 0.0, 0.0]]]),
            torch.tensor([[[0.0, 0.0, 0.0], [6.0, -3.0, 9.0]]]),
        )
        processor.begin_interval()
        for sample in samples:
            processor.update_substep(sample)
        average = processor.end_interval()
        expected = torch.tensor([[[4.0, 2.0, 16.0 / 3.0], [2.0, -1.0, 3.0]]])
        torch.testing.assert_close(average, expected)
        torch.testing.assert_close(processor.interval_count, torch.tensor([[[3.0]]]))
        self.assertEqual(torch.count_nonzero(processor.clipped[0, 0]), 0)

    def test_partial_reset_clears_every_stage_and_accumulator(self):
        processor = _processor(num_envs=3)
        processor.begin_interval()
        processor.update_substep(torch.ones(3, 2, 3) * 8.0)
        processor.end_interval()
        before_env_zero = {
            name: value[0].clone() for name, value in processor.flattened_stages().items()
        }
        processor.reset(torch.tensor([1, 2]))

        for name, value in processor.flattened_stages().items():
            torch.testing.assert_close(value[0], before_env_zero[name])
            self.assertEqual(torch.count_nonzero(value[1:]), 0)
        self.assertEqual(torch.count_nonzero(processor.contacts[1:]), 0)
        self.assertEqual(torch.count_nonzero(processor.interval_sum[1:]), 0)
        self.assertEqual(torch.count_nonzero(processor.interval_count[1:]), 0)


class GRFIntegrationTests(unittest.TestCase):
    def test_ema_grfs_buf_switch_is_opt_in_for_both_aliases(self):
        for task_cls, cfg_cls in (
            (Go2HardPACT, GO2HardPACTCfg),
            (Go2HardPACTPos, GO2HardPACTPosCfg),
        ):
            with self.subTest(task=task_cls.__name__):
                task = task_cls.__new__(task_cls)
                task.cfg = cfg_cls()
                task.grf_processor = _processor(num_envs=1, num_feet=4)
                task.grf_processor.ema.copy_(
                    torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
                )
                legacy_raw = torch.full((1, 12), 123.0)
                task.simulator = SimpleNamespace(_grfs_buf=legacy_raw.clone())

                self.assertFalse(task.cfg.sim.grf.use_ema_grfs_buf)
                task._update_legacy_grfs_buf_input()
                torch.testing.assert_close(task.simulator._grfs_buf, legacy_raw)

                task.cfg.sim.grf.use_ema_grfs_buf = True
                task._update_legacy_grfs_buf_input()
                torch.testing.assert_close(
                    task.simulator._grfs_buf, task.grf_processor.ema.flatten(1)
                )
                task.cfg.sim.grf.use_ema_grfs_buf = False

    def test_simulator_callback_samples_every_decimation_substep(self):
        samples = []
        expected = torch.zeros(1, 4, 3)
        for index, vertical_force in enumerate((0.0, 6.0, 12.0)):
            sample = torch.zeros(1, 4, 3)
            sample[:, :, 0] = index + 1.0
            sample[:, :, 2] = vertical_force
            samples.append(sample)
            if vertical_force > 3.0:
                expected += sample.clamp(-10.0, 10.0)
        for sim_cls, action_width in (
            (GenesisSimulator_PACT, 24),
            (GenesisSimulator_PACT_Pos, 12),
        ):
            with self.subTest(simulator=sim_cls.__name__):
                sim, processor = _synthetic_simulator(samples, sim_cls=sim_cls)
                processor.begin_interval()
                actions = torch.arange(action_width, dtype=torch.float32).reshape(1, -1)
                sim.step(actions)
                average = processor.end_interval()

                self.assertEqual(sim._scene.steps, 3)
                self.assertEqual(len(sim._robot.controlled), 3)
                torch.testing.assert_close(
                    processor.interval_count, torch.tensor([[[3.0]]])
                )
                torch.testing.assert_close(average, expected / 3.0)

    def test_alias_step_returns_interval_target_and_keeps_ema_separate(self):
        samples = []
        for vertical_force in (0.0, 10.0, 10.0):
            sample = torch.zeros(1, 4, 3)
            sample[:, :, 2] = vertical_force
            samples.append(sample)
        sim, processor = _synthetic_simulator(samples)
        task = Go2HardPACT.__new__(Go2HardPACT)
        task.cfg = GO2HardPACTCfg()
        task.simulator = sim
        task.grf_processor = processor
        task._pre_sim_step = lambda actions: actions
        task.post_physics_step = lambda: None
        task.obs_buf = torch.zeros(1, task.cfg.env.num_observations)
        task.privileged_obs_buf = torch.zeros(1, task.cfg.env.num_privileged_obs)
        task.obs_history = torch.zeros(1, task.cfg.env.num_observations * task.cfg.env.num_obs_hist)
        task.explicit_labels_buf = torch.zeros(1, task.cfg.env.num_explicit_recon_obs)
        task.rew_buf = torch.zeros(1)
        task.reset_buf = torch.zeros(1, dtype=torch.bool)
        task.extras = {}
        task.obs_scales = SimpleNamespace(grf=0.01)

        result = task.step(torch.zeros(1, 24))
        expected_target = torch.zeros(1, 12)
        expected_target[:, 2::3] = (20.0 / 3.0) * 0.01
        torch.testing.assert_close(result[-1], expected_target)
        self.assertFalse(torch.equal(processor.ema.flatten(1) * 0.01, result[-1]))

    def test_legacy_simulator_path_remains_callback_free(self):
        sample = torch.ones(1, 4, 3) * 8.0
        sim, processor = _synthetic_simulator([sample, sample], install_callback=False)
        actions = torch.arange(24, dtype=torch.float32).reshape(1, 24)
        sim.step(actions)
        self.assertEqual(sim._scene.steps, 2)
        self.assertEqual(GO2PACTCfg.control.decimation, 5)
        self.assertEqual(torch.count_nonzero(processor.raw), 0)
        self.assertFalse(hasattr(Go2PACT.__new__(Go2PACT), "grf_processor"))


if __name__ == "__main__":
    unittest.main()
