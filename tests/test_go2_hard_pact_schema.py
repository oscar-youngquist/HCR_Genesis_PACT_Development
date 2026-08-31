import types
import unittest

import torch

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import (
    ISAACLAB_CAPABILITIES,
    domain_randomization_report,
    Go2HardPACTCore,
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
from rsl_rl.modules.actor_critic_go2_hard_pact import PhysicsReferences


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
    def test_learned_force_references_are_unscaled_at_physics_boundary(self):
        core = object.__new__(Go2HardPACTCore)
        core.obs_scales = types.SimpleNamespace(grf=0.01, base_wrench=0.02)
        grf_scaled = torch.full((2, 12), 1.25)
        wrench_scaled = torch.full((2, 6), 0.8)
        grf_n, wrench_si = core._physics_references_si(
            PhysicsReferences(grf_scaled, wrench_scaled)
        )
        torch.testing.assert_close(grf_n, torch.full((2, 12), 125.0))
        torch.testing.assert_close(wrench_si, torch.full((2, 6), 40.0))

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


class StepLifecycleTests(unittest.TestCase):
    def test_step_matches_legacy_pact_lifecycle(self):
        events = []
        core = object.__new__(Go2HardPACTCore)
        core.teleport_mask = torch.ones(1, 1, dtype=torch.bool)
        safe_torque = torch.ones(1, 12)
        core._pre_sim_step = lambda *args, **kwargs: (
            events.append("pre_sim_step") or safe_torque
        )
        core.simulator = types.SimpleNamespace(
            step=lambda actions: events.append(("simulator.step", actions)),
            _grfs_buf=torch.zeros(1, 12),
        )
        core.grf_processor = types.SimpleNamespace(
            end_interval=lambda: (
                events.append("end_interval") or torch.zeros(1, 4, 3)
            ),
            ema=torch.zeros(1, 4, 3),
        )
        core.post_physics_step = lambda: events.append("post_physics_step")
        core._finish_hard_pact_step = lambda grf: (
            events.append("finish_transition") or "result"
        )

        result = Go2HardPACTCore.step(core, torch.zeros(1, 24))

        self.assertEqual(result, "result")
        self.assertEqual(
            [event if isinstance(event, str) else event[0] for event in events],
            [
                "pre_sim_step", "simulator.step", "end_interval",
                "post_physics_step", "finish_transition",
            ],
        )
        self.assertIs(events[1][1], safe_torque)

    def test_isaaclab_wrench_is_written_once_not_per_substep(self):
        calls = {"wrench": 0, "torque": 0}

        class Robot:
            def set_external_force_and_torque(self, *args, **kwargs):
                calls["wrench"] += 1

            def set_joint_effort_target(self, *args, **kwargs):
                calls["torque"] += 1

        core = object.__new__(Go2HardPACTCore)
        core.backend_name = "isaaclab"
        core.device = torch.device("cpu")
        core._step_wrench_world = torch.zeros(2, 6)
        core.simulator = types.SimpleNamespace(
            _robot=Robot(), _base_link_index=0, _dof_indices=list(range(12))
        )

        core._write_isaaclab_step_wrench()
        for _ in range(4):
            core._hard_pact_pre_physics_substep(torch.zeros(2, 12))

        self.assertEqual(calls, {"wrench": 1, "torque": 4})

    def test_genesis_wrench_is_applied_every_substep(self):
        calls = {"force": 0, "wrench_torque": 0, "joint_torque": 0}

        class Solver:
            def apply_links_external_force(self, *args, **kwargs):
                calls["force"] += 1

            def apply_links_external_torque(self, *args, **kwargs):
                calls["wrench_torque"] += 1


        class Robot:
            _solver = Solver()

            def control_dofs_force(self, *args, **kwargs):
                calls["joint_torque"] += 1

        core = object.__new__(Go2HardPACTCore)
        core.backend_name = "genesis"
        core.device = torch.device("cpu")
        core._step_wrench_world = torch.zeros(2, 6)
        core.simulator = types.SimpleNamespace(
            _robot=Robot(), _base_link_index=0, _dof_indices=list(range(12))
        )

        for _ in range(4):
            core._hard_pact_pre_physics_substep(torch.zeros(2, 12))

        self.assertEqual(
            calls, {"force": 4, "wrench_torque": 4, "joint_torque": 4}
        )


if __name__ == "__main__":
    unittest.main()
