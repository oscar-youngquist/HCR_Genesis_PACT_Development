"""Focused regression tests for B1Z1 force masks and iteration-level KL duals."""

import types
import unittest

import torch

from legged_gym.envs.b1z1.force_task_utils import (
    B1Z1StagedForceCurriculum,
    force_curriculum_active,
    force_neutral_mask,
    zero_velocity_probability,
)
from legged_gym.envs.b1z1.b1z1_unifp.b1z1_unifp import B1Z1UniFP
from rsl_rl.algorithms.kl_rate_band import (
    KLRateBandController,
    update_duals_from_mean,
)
from rsl_rl.algorithms.ppo_b1z1_pact import (
    FORCE_GATE_METRIC_NAMES,
    PPO_B1Z1PACT,
    _event_conditioned_force_statistics,
)


class ForceTaskTests(unittest.TestCase):
    @staticmethod
    def _curriculum_config(**overrides):
        values = {
            "force_curriculum_command_start_iteration": 8000,
            "force_curriculum_command_ramp_iterations": 4000,
            "force_curriculum_gate_start_iteration": 12000,
            "force_curriculum_external_ramp_iterations": 4000,
            "force_curriculum_ee_l1_threshold": 0.25,
            "force_curriculum_roll_termination_threshold": 0.05,
            "force_curriculum_episode_length_threshold": 950.0,
            "force_curriculum_gate_patience": 3,
            "force_curriculum_metric_ema_alpha": 1.0,
            "force_curriculum_use_latest_start_fallback": True,
            "force_curriculum_latest_start_iteration": 20000,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_force_activation_boundary_selects_correct_zero_probability(self):
        hold = 8000
        before = force_curriculum_active(hold - 1, hold)
        at_boundary = force_curriculum_active(hold, hold)
        self.assertFalse(before)
        self.assertTrue(at_boundary)
        self.assertEqual(zero_velocity_probability(before, 0.1, 0.8), 0.1)
        self.assertEqual(zero_velocity_probability(at_boundary, 0.1, 0.8), 0.8)

        env = object.__new__(B1Z1UniFP)
        env._staged_force_curriculum = B1Z1StagedForceCurriculum(
            self._curriculum_config()
        )
        env.training_iteration = hold - 1
        self.assertFalse(env.force_command_randomization_active)
        self.assertFalse(env.force_command_stream_enabled)
        self.assertEqual(env.command_force_scale, 0.0)
        env.training_iteration = hold + 1
        self.assertTrue(env.force_command_randomization_active)

    def test_staged_scales_and_per_iteration_gate(self):
        curriculum = B1Z1StagedForceCurriculum(self._curriculum_config())
        self.assertEqual(curriculum.command_scale(7999), 0.0)
        self.assertEqual(curriculum.command_scale(8000), 0.0)
        self.assertAlmostEqual(curriculum.command_scale(10000), 0.5)
        self.assertEqual(curriculum.command_scale(12000), 1.0)
        self.assertEqual(curriculum.external_scale(20000), 0.0)

        for iteration in (12000, 12001, 12002):
            curriculum.update(iteration, 0.20, 0.01, 1000.0)
            # Repeated calls in one PPO iteration must not advance patience.
            curriculum.update(iteration, 0.20, 0.01, 1000.0)
        self.assertTrue(curriculum.gate_latched)
        self.assertEqual(curriculum.trigger_iteration, 12002)
        self.assertEqual(curriculum.gate_patience, 3)
        self.assertEqual(curriculum.external_scale(12002), 0.0)
        self.assertAlmostEqual(curriculum.external_scale(14002), 0.5)
        self.assertEqual(curriculum.external_scale(16002), 1.0)

    def test_gate_state_survives_resume(self):
        original = B1Z1StagedForceCurriculum(self._curriculum_config())
        original.update(12000, 0.20, 0.01, 1000.0)
        original.update(12001, 0.20, 0.01, 1000.0)

        restored = B1Z1StagedForceCurriculum(self._curriculum_config())
        restored.load_state_dict(original.state_dict())
        restored.update(12001, 0.20, 0.01, 1000.0)
        self.assertEqual(restored.gate_patience, 2)
        restored.update(12002, 0.20, 0.01, 1000.0)
        self.assertTrue(restored.gate_latched)
        self.assertEqual(restored.trigger_iteration, 12002)

    def test_latest_start_fallback_latches_without_metrics(self):
        curriculum = B1Z1StagedForceCurriculum(self._curriculum_config())
        curriculum.update(20000)
        self.assertTrue(curriculum.gate_latched)
        self.assertEqual(curriculum.trigger_iteration, 20000)

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
            foot_contacts=torch.ones(env.num_envs, 4, dtype=torch.bool),
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


class PACTForceBlendTests(unittest.TestCase):
    @staticmethod
    def _ppo(start_ema=1.0, current_ema=1.0, active=False):
        ppo = object.__new__(PPO_B1Z1PACT)
        ppo.cfg = {"force_gate_threshold": 0.10}
        ppo.force_blend_min_alpha = 0.01
        ppo.force_blend_start_ema = start_ema
        ppo.force_ema = current_ema
        ppo.force_gate_active = active
        return ppo

    def test_blend_starts_at_configured_minimum(self):
        self.assertAlmostEqual(self._ppo()._force_prediction_blend_alpha(), 0.01)

    def test_blend_is_linear_in_reconstruction_ema(self):
        # Halfway from EMA 1.0 to threshold 0.1 gives halfway authority.
        ppo = self._ppo(current_ema=0.55)
        self.assertAlmostEqual(ppo._force_prediction_blend_alpha(), 0.505)
        ppo.force_ema = 0.10
        self.assertLess(ppo._force_prediction_blend_alpha(), 1.0)

    def test_blend_is_bounded_and_gate_forces_full_prediction(self):
        self.assertAlmostEqual(
            self._ppo(current_ema=2.0)._force_prediction_blend_alpha(), 0.01
        )
        self.assertAlmostEqual(
            self._ppo(current_ema=2.0, active=True)._force_prediction_blend_alpha(), 1.0
        )


class PACTEventConditionedForceGateTests(unittest.TestCase):
    def test_active_neutral_masks_and_terminal_exclusion(self):
        target = torch.zeros(4, 21)
        prediction = torch.zeros_like(target)
        # Stored force order remains [GRF, base wrench, EE force].
        prediction[:3, :12] = 1.0
        target[0, 18:21], prediction[0, 18:21] = 2.0, 1.0
        prediction[1:3, 18:21] = 0.5
        target[1, 12:18], prediction[1, 12:18] = 2.0, 1.0
        prediction[[0, 2], 12:18] = 0.25
        prediction[3] = 100.0
        valid = torch.tensor([[1.0], [1.0], [1.0], [0.0]])

        statistics = _event_conditioned_force_statistics(
            prediction, target, valid, 0.1, 0.1
        )
        expected = {
            "grf": (1.0, 3),
            "ee_active": (1.0, 1),
            "ee_neutral": (0.25, 2),
            "base_active": (1.0, 1),
            "base_neutral": (0.0625, 2),
        }
        for name, (expected_mse, expected_samples) in expected.items():
            numerator, elements, samples = statistics[name]
            self.assertAlmostEqual((numerator / elements).item(), expected_mse)
            self.assertEqual(samples.item(), expected_samples)
        for name in ("ee_event_fraction", "base_event_fraction"):
            active, valid_count = statistics[name]
            self.assertAlmostEqual((active / valid_count).item(), 1.0 / 3.0)

    @staticmethod
    def _ppo():
        ppo = object.__new__(PPO_B1Z1PACT)
        ppo.cfg = {
            "force_gate_ema_alpha": 1.0,
            "force_gate_patience": 2,
        }
        for name in FORCE_GATE_METRIC_NAMES:
            ppo.cfg[f"force_gate_{name}_threshold"] = 0.1
            ppo.cfg[f"force_gate_{name}_hysteresis"] = 0.2
            ppo.cfg[f"force_gate_{name}_min_samples"] = 1
        ppo.force_ema = None
        ppo.force_blend_start_ema = None
        ppo.force_metric_emas = {
            name: None for name in FORCE_GATE_METRIC_NAMES
        }
        ppo.force_gate_count = 0
        ppo.force_gate_active = False
        return ppo

    def test_patience_advances_once_per_iteration_statistic(self):
        ppo = self._ppo()
        errors = {name: 0.05 for name in FORCE_GATE_METRIC_NAMES}
        samples = {name: 4 for name in FORCE_GATE_METRIC_NAMES}

        ppo._update_event_conditioned_force_gate(errors, samples, 0.05)
        self.assertEqual(ppo.force_gate_count, 1)
        self.assertFalse(ppo.force_gate_active)
        ppo._update_event_conditioned_force_gate(errors, samples, 0.05)
        self.assertEqual(ppo.force_gate_count, 2)
        self.assertTrue(ppo.force_gate_active)

    def test_missing_active_event_cannot_advance_gate(self):
        ppo = self._ppo()
        errors = {name: 0.01 for name in FORCE_GATE_METRIC_NAMES}
        samples = {name: 4 for name in FORCE_GATE_METRIC_NAMES}
        samples["ee_active"] = 0

        ppo._update_event_conditioned_force_gate(errors, samples, 0.01)
        self.assertEqual(ppo.force_gate_count, 0)
        self.assertFalse(ppo.force_gate_active)
        self.assertIsNone(ppo.force_metric_emas["ee_active"])


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

    def test_baseline_kl_remains_active_after_warmup(self):
        controller = self._controller()
        raw_kl = torch.tensor(0.5)

        self.assertAlmostEqual(controller.loss(raw_kl, iteration=1).item(), 0.5)
        metrics = controller.metrics(raw_kl, controller.loss(raw_kl, 1), 1)
        self.assertAlmostEqual(metrics["kl_base_beta"].item(), 1.0)
        self.assertAlmostEqual(metrics["kl_effective_coef"].item(), 1.0)

    def test_standard_kl_can_use_cosine_warmup(self):
        controller = KLRateBandController(
            warmup_iters=10, warmup_beta_max=0.25,
            rate_min=0.1, rate_max=1.0, dual_lr=1.0,
            augmented_rho=1.0, ema_decay=0.0,
        )
        raw_kl = torch.tensor(2.0)

        expected_losses = ((0, 0.0), (5, 0.25), (10, 0.5), (100, 0.5))
        for iteration, expected_loss in expected_losses:
            loss = controller.loss(
                raw_kl, iteration, use_rate_band=False,
                use_cosine_warmup=True,
            )
            self.assertAlmostEqual(loss.item(), expected_loss)
            metrics = controller.metrics(
                raw_kl, loss, iteration, use_rate_band=False,
                use_cosine_warmup=True,
            )
            self.assertEqual(metrics["kl_rate_band_enabled"].item(), 0.0)
            self.assertEqual(metrics["kl_warmup_enabled"].item(), 1.0)

    def test_standard_kl_can_disable_cosine_warmup(self):
        controller = KLRateBandController(
            warmup_iters=10, warmup_beta_max=0.25,
            rate_min=0.1, rate_max=1.0, dual_lr=1.0,
            augmented_rho=1.0, ema_decay=0.0,
        )
        raw_kl = torch.tensor(2.0)

        for iteration in (0, 5, 10, 100):
            loss = controller.loss(
                raw_kl, iteration, use_rate_band=False,
                use_cosine_warmup=False,
            )
            self.assertAlmostEqual(loss.item(), 0.5)
            metrics = controller.metrics(
                raw_kl, loss, iteration, use_rate_band=False,
                use_cosine_warmup=False,
            )
            self.assertEqual(metrics["kl_rate_band_enabled"].item(), 0.0)
            self.assertEqual(metrics["kl_warmup_enabled"].item(), 0.0)
            self.assertEqual(metrics["kl_band_active"].item(), 0.0)
            self.assertAlmostEqual(metrics["kl_effective_coef"].item(), 0.25)

    def test_rate_band_without_base_warmup_starts_its_own_ramp_at_zero(self):
        controller = self._cosine_band_controller()

        self.assertTrue(controller.band_active(0, use_cosine_warmup=False))
        self.assertEqual(
            controller.band_warmup_scale(0, use_cosine_warmup=False), 0.0
        )
        self.assertAlmostEqual(
            controller.band_warmup_scale(10, use_cosine_warmup=False), 0.5
        )
        self.assertEqual(
            controller.band_warmup_scale(20, use_cosine_warmup=False), 1.0
        )

    def test_standard_kl_disables_dual_updates(self):
        controller = self._controller()
        mean = update_duals_from_mean(
            controller, 4.0, 2, 20, "cpu", enabled=False
        )

        self.assertAlmostEqual(mean.item(), 2.0)
        self.assertEqual(controller.lambda_low, 0.0)
        self.assertEqual(controller.lambda_high, 0.0)
        self.assertIsNone(controller.kl_ema)

    @staticmethod
    def _cosine_band_controller(band_warmup_iters=20):
        return KLRateBandController(
            warmup_iters=10, warmup_beta_max=1.0,
            band_warmup_iters=band_warmup_iters,
            rate_min=0.1, rate_max=1.0, dual_lr=1.0,
            augmented_rho=0.1, ema_decay=0.0,
        )

    def test_rate_band_uses_separate_cosine_warmup(self):
        controller = self._cosine_band_controller()
        self.assertEqual(controller.band_warmup_scale(9), 0.0)
        self.assertEqual(controller.band_warmup_scale(10), 0.0)
        self.assertAlmostEqual(controller.band_warmup_scale(20), 0.5)
        self.assertEqual(controller.band_warmup_scale(30), 1.0)
        self.assertEqual(controller.band_warmup_scale(40), 1.0)

        # raw KL=2 has a unit upper-band violation. The augmented penalty is
        # 0.05 at full scale and must enter continuously after base warmup.
        raw_kl = torch.tensor(2.0)
        self.assertAlmostEqual(controller.loss(raw_kl, 10).item(), 2.0)
        self.assertAlmostEqual(controller.loss(raw_kl, 20).item(), 2.025, places=6)
        self.assertAlmostEqual(controller.loss(raw_kl, 30).item(), 2.05, places=6)

    def test_dual_updates_use_same_cosine_scale(self):
        raw_kl = torch.tensor(2.0)
        expected = ((10, 0.0), (20, 0.5), (30, 1.0))
        for iteration, expected_increment in expected:
            controller = self._cosine_band_controller()
            controller.update_duals(raw_kl, iteration)
            self.assertAlmostEqual(controller.lambda_high, expected_increment)
            self.assertEqual(controller.lambda_low, 0.0)

    def test_zero_band_warmup_preserves_immediate_activation(self):
        controller = self._cosine_band_controller(band_warmup_iters=0)
        self.assertEqual(controller.band_warmup_scale(9), 0.0)
        self.assertEqual(controller.band_warmup_scale(10), 1.0)

    def test_negative_band_warmup_is_rejected(self):
        with self.assertRaises(ValueError):
            self._cosine_band_controller(band_warmup_iters=-1)


if __name__ == "__main__":
    unittest.main()
