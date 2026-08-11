"""Plumbing tests and opt-in physical diagnostics for Isaac Gym foot sensors."""

import os
import unittest

# Importing the simulator initializes Isaac Gym before PyTorch, as required by
# Isaac Gym's Python bindings.
from legged_gym.simulator.isaacgym_simulator import (
    reshape_foot_force_sensor_tensor,
    summarize_substep_tensor,
)
import torch

from legged_gym.scripts.diagnose_b1z1_isaacgym_foot_forces import (
    compare_foot_force_series,
)


class IsaacGymFootForceSensorTests(unittest.TestCase):
    def test_force_sensor_tensor_reshape_preserves_actor_and_foot_order(self):
        num_envs, num_feet = 3, 4
        raw = torch.arange(num_envs * num_feet * 6, dtype=torch.float32).reshape(-1, 6)

        wrenches = reshape_foot_force_sensor_tensor(raw, num_envs, num_feet)

        self.assertEqual(wrenches.shape, (num_envs, num_feet, 6))
        for env_id in range(num_envs):
            for foot_id in range(num_feet):
                sensor_id = env_id * num_feet + foot_id
                self.assertTrue(torch.equal(wrenches[env_id, foot_id], raw[sensor_id]))

    def test_force_sensor_tensor_reshape_rejects_wrong_sensor_count(self):
        with self.assertRaisesRegex(AssertionError, "Expected force-sensor tensor"):
            reshape_foot_force_sensor_tensor(
                torch.zeros(7, 6), num_envs=2, num_feet=4
            )

    def test_substep_statistics_do_not_sum_forces(self):
        values = torch.tensor(
            [[[1.0, -2.0]], [[3.0, 4.0]], [[-5.0, 1.0]]],
            dtype=torch.float32,
        )
        statistics = summarize_substep_tensor(values)
        self.assertTrue(torch.equal(statistics["final"], values[-1]))
        self.assertTrue(torch.equal(statistics["minimum"], values.amin(dim=0)))
        self.assertTrue(torch.equal(statistics["maximum"], values.amax(dim=0)))
        self.assertTrue(
            torch.equal(statistics["maximum_absolute"], values.abs().amax(dim=0))
        )
        self.assertTrue(torch.allclose(statistics["mean"], values.mean(dim=0)))

    def test_force_comparison_preserves_configured_foot_order(self):
        names = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
        raw = torch.tensor([[100.0, 110.0, 120.0, 130.0]]).repeat(3, 1)
        sensor = raw * 0.25
        comparison = compare_foot_force_series(sensor, raw, names)
        self.assertEqual(tuple(comparison), names)
        self.assertAlmostEqual(
            comparison["FR_foot"]["mean_ratio_sensor_to_raw"], 0.25
        )

    @unittest.skipUnless(
        os.environ.get("RUN_ISAACGYM_SENSOR_TESTS") == "1",
        "set RUN_ISAACGYM_SENSOR_TESTS=1 to launch the PhysX integration test",
    )
    def test_b1z1_force_sensor_physical_diagnostic(self):
        """Run one equilibrium-gated diagnostic without assuming sensor == GRF."""
        self.assertEqual(os.environ.get("SIMULATOR"), "isaacgym_b1z1_unifp")
        from legged_gym.scripts.diagnose_b1z1_isaacgym_foot_forces import (
            run_variant,
        )

        report = run_variant(
            "constraint_all_substeps",
            sample_count=100,
            settle_timeout_steps=4000,
            aerial_steps=10,
        )
        self.assertIn(report["status"], ("conclusive", "inconclusive"))
        configuration = report["resolved_configuration"]
        self.assertEqual(
            tuple(configuration["sensor_names"]),
            ("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
        )
        self.assertEqual(configuration["sensor_count_per_actor"], 4)
        self.assertEqual(configuration["sensor_count_in_sim"], 4)
        self.assertFalse(
            configuration["sensor_flags"]["enable_forward_dynamics_forces"]
        )
        self.assertTrue(
            configuration["sensor_flags"]["enable_constraint_solver_forces"]
        )
        self.assertTrue(configuration["sensor_flags"]["use_world_frame"])

        # Aerial raw contacts should be nearly zero. Sensor values are logged,
        # not asserted as pure contacts, because that is the issue diagnosed.
        expected_weight = report["equilibrium"]["expected_mg"]
        aerial_raw = [
            abs(sample["raw_all_body_fz"])
            for sample in report["aerial_samples"]
        ]
        self.assertLess(max(aerial_raw), 0.05 * expected_weight)

        # Physical weight validation is meaningful only after both gates pass.
        if report["status"] == "conclusive":
            equilibrium = report["equilibrium"]
            self.assertTrue(equilibrium["quasi_static"])
            self.assertTrue(equilibrium["raw_support_matches_weight"])
            self.assertGreater(equilibrium["raw_foot_fz_sum_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
