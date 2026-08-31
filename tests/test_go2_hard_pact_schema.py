import types
import unittest

import torch

from legged_gym.envs.go2.go2_hard_pact.backend import (
    ISAACLAB_CAPABILITIES,
    domain_randomization_report,
)
from legged_gym.envs.go2.go2_hard_pact.schema import (
    CANONICAL,
    QPStateEstimate,
    RECONSTRUCTION_SCHEMA,
    RandomizedDynamicsParameters,
    fixed_gravity_normal,
    permutation_by_name,
    quat_wxyz_to_xyzw,
    reconstruct_coupled_nominal_torque,
    world_to_body,
    world_to_yaw_local,
    yaw_local_to_world,
)


class ReconstructionSchemaTests(unittest.TestCase):
    def test_named_schema_is_exactly_79d_and_force_free(self):
        self.assertEqual(RECONSTRUCTION_SCHEMA.width, 79)
        self.assertEqual(RECONSTRUCTION_SCHEMA.next_state_width, 33)
        names = {field.name for field in RECONSTRUCTION_SCHEMA.fields}
        for prohibited in ("terrain", "normal", "grf", "force", "wrench"):
            self.assertFalse(any(prohibited in name for name in names))

    def test_scaling_round_trip(self):
        batch = 3
        fields = {}
        for field in RECONSTRUCTION_SCHEMA.fields:
            fields[field.name] = torch.randn(batch, field.width)
            if field.offset:
                fields[field.name] += field.offset
        encoded = RECONSTRUCTION_SCHEMA.build(fields, normalized=True)
        decoded = RECONSTRUCTION_SCHEMA.unpack(encoded, normalized=True)
        self.assertEqual(encoded.shape, (batch, 79))
        self.assertEqual(
            RECONSTRUCTION_SCHEMA.system_identification_vector(encoded).shape,
            (batch, 46),
        )
        for name, expected in fields.items():
            torch.testing.assert_close(decoded[name], expected)

    def test_realized_parameter_schema_round_trip_is_named(self):
        values = {
            name: torch.randn(2, width)
            for name, width in RandomizedDynamicsParameters.FIELD_WIDTHS
        }
        parameters = RandomizedDynamicsParameters(**values)
        restored = RandomizedDynamicsParameters.unpack(parameters.pack())
        for name, _ in RandomizedDynamicsParameters.FIELD_WIDTHS:
            torch.testing.assert_close(getattr(restored, name), values[name])
        self.assertEqual(parameters.pack().shape[-1], 46)

    def test_realized_gains_and_motor_change_reconstructed_torque(self):
        values = {}
        for name, width in RandomizedDynamicsParameters.FIELD_WIDTHS:
            values[name] = torch.zeros(1, width)
        values["kp_scale"].fill_(1.0)
        values["kd_scale"].fill_(1.0)
        values["motor_strength_scale"].fill_(1.0)
        nominal = RandomizedDynamicsParameters(**values)
        action = torch.ones(1, 24) * 0.1

        def torque(parameters):
            return reconstruct_coupled_nominal_torque(
                action, torch.zeros(1, 12), torch.ones(1, 12) * 0.2,
                torch.zeros(1, 12), torch.ones(1, 12) * 20.0,
                torch.ones(1, 12), parameters,
                torch.ones(1, 1), torch.ones(1, 1), 0.25, 5.0,
            )[0]

        reference = torque(nominal)
        for field in ("kp_scale", "kd_scale", "motor_strength_scale"):
            changed = RandomizedDynamicsParameters.unpack(nominal.pack().clone())
            getattr(changed, field).mul_(1.2)
            self.assertGreater((torque(changed) - reference).abs().max().item(), 0.0)


class CanonicalConversionTests(unittest.TestCase):
    def test_qp_state_has_no_absolute_position_and_uses_estimated_velocity(self):
        state = QPStateEstimate(
            base_linear_velocity_body=torch.tensor([[1.0, 2.0, 3.0]]),
            base_quaternion_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            base_angular_velocity_world=torch.tensor([[4.0, 5.0, 6.0]]),
            joint_position=torch.zeros(1, 12), joint_velocity=torch.ones(1, 12),
            previous_safe_torque=torch.zeros(1, 12),
            contact_probability=torch.zeros(1, 4),
            predicted_grf_yaw=torch.zeros(1, 12),
            predicted_base_wrench_yaw=torch.zeros(1, 6),
        )
        torch.testing.assert_close(state.local_q_xyzw[:, :3], torch.zeros(1, 3))
        torch.testing.assert_close(
            state.velocity_world[:, :6],
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        )

    def test_joint_names_are_verified_and_reordered(self):
        source = tuple(reversed(CANONICAL.joint_names))
        permutation = permutation_by_name(source, CANONICAL.joint_names, "joint")
        self.assertEqual(permutation, tuple(reversed(range(12))))
        with self.assertRaises(ValueError):
            permutation_by_name(source[:-1], CANONICAL.joint_names, "joint")

    def test_quaternion_and_frames(self):
        half = torch.tensor(torch.pi / 4.0)
        wxyz = torch.tensor([[torch.cos(half), 0.0, 0.0, torch.sin(half)]])
        xyzw = quat_wxyz_to_xyzw(wxyz)
        vector = torch.tensor([[1.0, 0.0, 0.0]])
        local = world_to_yaw_local(vector, xyzw)
        torch.testing.assert_close(local, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(yaw_local_to_world(local, xyzw), vector, atol=1e-6, rtol=0)
        torch.testing.assert_close(world_to_body(vector, xyzw), local, atol=1e-6, rtol=0)

    def test_gravity_normal_is_fixed(self):
        gravity = torch.tensor([[0.0, 0.0, -9.81]])
        torch.testing.assert_close(
            fixed_gravity_normal(gravity), torch.tensor([[0.0, 0.0, 1.0]])
        )


class CapabilityTests(unittest.TestCase):
    def test_isaaclab_curriculum_is_explicitly_unsupported(self):
        self.assertFalse(ISAACLAB_CAPABILITIES.supports_domain_rand_curriculum)
        cfg = types.SimpleNamespace(
            use_domainrand_curriculum=True,
            randomize_friction=True,
            friction_range=[0.2, 1.25],
        )
        report = domain_randomization_report(cfg, ISAACLAB_CAPABILITIES)
        self.assertFalse(report["domain_rand_curriculum"]["active"])
        self.assertIn("reset-time", report["domain_rand_curriculum"]["reason"])
        self.assertEqual(report["friction"]["effective_ranges"]["friction_range"], [0.2, 1.25])


if __name__ == "__main__":
    unittest.main()
