import unittest

import torch

from legged_gym.envs.go2.go2_hard_pact.disturbances import (
    InstantaneousPushConfig,
    InstantaneousPushes,
    SustainedBaseWrench,
    SustainedWrenchConfig,
    added_mass_gravity_wrench_world,
    physics_transition_mask,
    torso_wrench_scale_from_ranges,
)
from legged_gym.envs.go2.go2_hard_pact.grf import (
    GRFProcessingConfig,
    IntervalGRFProcessor,
)


class GRFProcessingTests(unittest.TestCase):
    def test_all_stages_interval_average_and_reset(self):
        processor = IntervalGRFProcessor(
            1, 2, "cpu", torch.float32,
            GRFProcessingConfig(
                vertical_deadband_n=3.0,
                clip_min_n=-10.0,
                clip_max_n=10.0,
                ema_alpha=0.5,
                contact_threshold_n=5.0,
            ),
        )
        first = torch.tensor([[[9.0, 8.0, 2.0], [20.0, -20.0, 8.0]]])
        second = torch.tensor([[[4.0, 2.0, 6.0], [0.0, 0.0, 0.0]]])
        processor.begin_interval()
        processor.update_substep(first)
        self.assertTrue(torch.equal(processor.deadbanded[0, 0], torch.zeros(3)))
        torch.testing.assert_close(processor.clipped[0, 1], torch.tensor([10.0, -10.0, 8.0]))
        processor.update_substep(second)
        average = processor.end_interval()
        torch.testing.assert_close(average[0, 0], torch.tensor([2.0, 1.0, 3.0]))
        torch.testing.assert_close(average[0, 1], torch.tensor([5.0, -5.0, 4.0]))
        self.assertTrue(processor.contacts[0, 0])
        self.assertFalse(processor.contacts[0, 1])
        self.assertFalse(torch.equal(processor.ema, processor.interval_average))
        processor.reset(torch.tensor([0]))
        for stage in processor.flattened_stages().values():
            self.assertEqual(torch.count_nonzero(stage), 0)
        self.assertEqual(torch.count_nonzero(processor.contacts), 0)


class DisturbanceTests(unittest.TestCase):
    def test_added_mass_wrench_is_signed_weight_at_torso_com(self):
        added_mass = torch.tensor([[8.0], [-1.0], [0.0]])
        wrench = added_mass_gravity_wrench_world(
            added_mass, torch.tensor([0.0, 0.0, -9.81])
        )
        torch.testing.assert_close(
            wrench[:, :3],
            torch.tensor([
                [0.0, 0.0, -78.48],
                [0.0, 0.0, 9.81],
                [0.0, 0.0, 0.0],
            ]),
        )
        self.assertEqual(torch.count_nonzero(wrench[:, 3:]), 0)

    def test_torso_wrench_scale_tracks_configured_ranges(self):
        scale = torso_wrench_scale_from_ranges(
            (-20.0, 30.0),
            (-2.0, 3.0),
            (-2.0, 5.0),
            (1.0, -2.0, -10.0),
        )
        self.assertEqual(scale, (35.0, 40.0, 80.0, 3.0, 3.0, 3.0))
        no_disturbances = torso_wrench_scale_from_ranges(
            (-20.0, 30.0),
            (-2.0, 3.0),
            (-2.0, 5.0),
            (1.0, -2.0, -10.0),
            include_sustained_force=False,
            include_sustained_torque=False,
            include_added_mass=False,
        )
        self.assertEqual(no_disturbances, (0.0,) * 6)

    def test_instantaneous_push_is_atomic_and_zero_between_events(self):
        torch.manual_seed(4)
        pushes = InstantaneousPushes(
            4, "cpu", torch.float32,
            InstantaneousPushConfig(
                probability=1.0,
                interval_steps_min=2,
                interval_steps_max=2,
                planar_delta_v=(-1.0, 1.0),
                downward_delta_vz=(-0.5, 0.0),
                angular_delta_v=(-2.0, 2.0),
            ),
        )
        pushes.next_event_step.zero_()
        delta, mask = pushes.sample(0)
        self.assertTrue(mask.all())
        self.assertTrue((delta[:, 2] <= 0.0).all())
        self.assertTrue((delta[:, 2] >= -0.5).all())
        delta, mask = pushes.sample(1)
        self.assertFalse(mask.any())
        self.assertEqual(torch.count_nonzero(delta), 0)

    def test_sustained_wrench_has_ramp_and_normalized_yaw_label(self):
        torch.manual_seed(7)
        wrench = SustainedBaseWrench(
            1, "cpu", torch.float32,
            SustainedWrenchConfig(
                force_probability=1.0,
                torque_probability=1.0,
                interval_steps=(20, 20),
                duration_steps=(8, 8),
                ramp_fraction=0.25,
                force_bounds_n=(10.0, 10.0),
                torque_bounds_nm=(2.0, 2.0),
                force_normalizer_n=10.0,
                torque_normalizer_nm=2.0,
            ),
        )
        wrench.next_event_step.zero_()
        at_zero, active = wrench.step(0)
        self.assertTrue(active.item())
        self.assertEqual(torch.count_nonzero(at_zero), 0)
        at_one, _ = wrench.step(1)
        torch.testing.assert_close(at_one, torch.tensor([[5.0, 5.0, 5.0, 1.0, 1.0, 1.0]]))
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        torch.testing.assert_close(
            wrench.yaw_local_normalized(identity),
            torch.full((1, 6), 0.5),
        )
        at_seven, _ = wrench.step(7)
        torch.testing.assert_close(at_seven, at_one)

    def test_transition_mask_keeps_sustained_events_valid(self):
        false = torch.zeros(3, 1, dtype=torch.bool)
        push = false.clone()
        push[1] = True
        mask = physics_transition_mask(false, false, false, push)
        self.assertEqual(mask.squeeze(-1).tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
