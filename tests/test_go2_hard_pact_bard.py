import os
import unittest

os.environ.setdefault("SIMULATOR", "genesis_pact")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_hard_pact_bard_tests")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_hard_pact_bard_tests")

import numpy as np
import torch

from legged_gym.dynamics import (
    BARD_JOINT_ORDER,
    SIMULATOR_JOINT_ORDER,
    BardGo2Dynamics,
    build_linear_first_spatial_inertia,
    simulator_state_to_bard,
    wrench_at_point,
)
from rsl_rl.algorithms.hard_pact_bard import (
    corrected_bard_inverse_dynamics_loss,
    physics_valid_mask,
)
from rsl_rl.modules.hard_pact_physics import DeploymentPhysicsHeads


URDF = "resources/robots/go2/urdf/go2.urdf"
SIM_FROM_BARD = [BARD_JOINT_ORDER.index(name) for name in SIMULATOR_JOINT_ORDER]


class CorrectedInverseLossTests(unittest.TestCase):
    def inputs(self, batch=4):
        required = torch.zeros(batch, 18)
        foot_j = torch.zeros(batch, 4, 3, 18)
        base_j = torch.zeros(batch, 6, 18)
        foot_j[:, :, :, :3] = 0.25
        base_j[:, :, 3:9] = torch.eye(6).unsqueeze(0)
        return dict(
            required_generalized_force=required,
            foot_jacobians=foot_j,
            base_jacobian=base_j,
            interval_executed_torque=torch.zeros(batch, 12),
            interval_grf_world=torch.zeros(batch, 4, 3, requires_grad=True),
            total_wrench_world=torch.zeros(batch, 6, requires_grad=True),
            mass_com_wrench_world=torch.zeros(batch, 6),
            measured_generalized_contact_force=torch.ones(batch, 18),
            push_event_mask=torch.zeros(batch, 1, dtype=torch.bool),
            reset_mask=torch.zeros(batch, 1, dtype=torch.bool),
            timeout_mask=torch.zeros(batch, 1, dtype=torch.bool),
            teleport_mask=torch.zeros(batch, 1, dtype=torch.bool),
        )

    def test_exact_mask_truth_table(self):
        values = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]]).bool()
        actual = physics_valid_mask(*(values[:, i:i+1] for i in range(4)))
        torch.testing.assert_close(
            actual.flatten(), torch.tensor([True, False, False, False, False, False])
        )

    def test_exact_reduction_and_metric_blocks(self):
        data = self.inputs(2)
        data["required_generalized_force"][0] = torch.arange(1.0, 19.0)
        data["required_generalized_force"][1] = 1000.0
        data["reset_mask"][1] = True
        result = corrected_bard_inverse_dynamics_loss(**data)
        expected = torch.linalg.vector_norm(torch.arange(1.0, 19.0)) / (18.0 ** 0.5)
        torch.testing.assert_close(result.loss, expected)
        torch.testing.assert_close(
            result.metrics["inverse_residual/base_linear_mae_physical"],
            torch.tensor(2.0),
        )
        torch.testing.assert_close(
            result.metrics["inverse_residual/base_angular_mse_relative"],
            torch.tensor((16.0 + 25.0 + 36.0) / (3.0 * 18.0)),
        )
        self.assertEqual(len(result.metrics), 8)

    def test_soft_contact_weighting_and_relative_normalization(self):
        data = self.inputs(1)
        data["required_generalized_force"][0, :4] = torch.tensor([2., 4., 6., 8.])
        data["measured_generalized_contact_force"].zero_()
        data["measured_generalized_contact_force"][0, :4] = torch.tensor([1., 2., -3., 4.])
        data["interval_executed_torque"][0, 0] = 3.0
        result = corrected_bard_inverse_dynamics_loss(**data)
        residual = data["required_generalized_force"].clone()
        residual[0, 6] -= 3.0
        contact_weight = torch.tensor([[0.25, 0.5, 0.0, 1.0] + [0.0] * 14])
        expected = torch.linalg.vector_norm(residual * contact_weight, dim=1) / (
            1.0e-8 + 3.0
            + torch.linalg.vector_norm(data["measured_generalized_contact_force"], dim=1)
        )
        torch.testing.assert_close(result.loss, expected.squeeze(0))

    def test_all_invalid_is_connected_zero_with_zero_head_gradients(self):
        data = self.inputs(3)
        data["reset_mask"].fill_(True)
        data["interval_grf_world"].data.normal_()
        data["total_wrench_world"].data.normal_()
        result = corrected_bard_inverse_dynamics_loss(**data)
        self.assertEqual(result.loss.item(), 0.0)
        result.loss.backward()
        torch.testing.assert_close(
            data["interval_grf_world"].grad,
            torch.zeros_like(data["interval_grf_world"]),
        )
        torch.testing.assert_close(
            data["total_wrench_world"].grad,
            torch.zeros_like(data["total_wrench_world"]),
        )

    def test_gradient_only_reaches_force_predictions(self):
        data = self.inputs(2)
        data["required_generalized_force"] = torch.ones(2, 18, requires_grad=True)
        data["foot_jacobians"].requires_grad_()
        data["base_jacobian"].requires_grad_()
        data["interval_executed_torque"].requires_grad_()
        data["mass_com_wrench_world"].requires_grad_()
        data["measured_generalized_contact_force"].requires_grad_()
        result = corrected_bard_inverse_dynamics_loss(**data)
        result.loss.backward()
        self.assertGreater(data["interval_grf_world"].grad.abs().sum().item(), 0)
        self.assertGreater(data["total_wrench_world"].grad.abs().sum().item(), 0)
        for name in ("required_generalized_force", "foot_jacobians", "base_jacobian",
                     "interval_executed_torque", "mass_com_wrench_world",
                     "measured_generalized_contact_force"):
            self.assertIsNone(data[name].grad)

    def test_inverse_loss_routes_gradients_to_both_deployment_heads(self):
        torch.manual_seed(4)
        heads = DeploymentPhysicsHeads(torch.ones(12), torch.ones(6))
        latent = torch.randn(2, 16, requires_grad=True)
        explicit = torch.randn(2, 11, requires_grad=True)
        torque = torch.randn(2, 12, requires_grad=True)
        prediction = heads(latent, explicit, torque)
        data = self.inputs(2)
        data["required_generalized_force"].normal_()
        data["interval_grf_world"] = prediction.grf_yaw_scaled.reshape(2, 4, 3)
        data["total_wrench_world"] = prediction.base_wrench_yaw_scaled
        result = corrected_bard_inverse_dynamics_loss(**data)
        result.loss.backward()
        self.assertGreater(sum(
            parameter.grad.abs().sum().item()
            for parameter in heads.grf_head.parameters()
        ), 0.0)
        self.assertGreater(sum(
            parameter.grad.abs().sum().item()
            for parameter in heads.wrench_head.parameters()
        ), 0.0)
        self.assertGreater(latent.grad.abs().sum().item(), 0.0)
        self.assertGreater(torque.grad.abs().sum().item(), 0.0)
        self.assertIsNone(explicit.grad)

    def test_mass_com_wrench_is_subtracted_exactly_once(self):
        data = self.inputs(1)
        mass = torch.tensor([[2., 3., 4., 5., 6., 7.]])
        sustained = torch.tensor([[1., -2., 3., -4., 5., -6.]], requires_grad=True)
        data["mass_com_wrench_world"] = mass
        data["total_wrench_world"] = sustained + mass
        result_with_mass = corrected_bard_inverse_dynamics_loss(**data)
        data["mass_com_wrench_world"] = torch.zeros_like(mass)
        data["total_wrench_world"] = sustained
        result_without_mass = corrected_bard_inverse_dynamics_loss(**data)
        torch.testing.assert_close(result_with_mass.residual, result_without_mass.residual)

    def test_wrench_reference_point_conversion(self):
        wrench = torch.tensor([[2., 0., 0., 0., 0., 0.]])
        shifted = wrench_at_point(wrench, torch.tensor([[0., 1., 0.]]), torch.zeros(1, 3))
        torch.testing.assert_close(shifted, torch.tensor([[2., 0., 0., 0., 0., -2.]]))


class BARDPinocchioParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pinocchio as pin
        cls.pin = pin
        cls.dynamics = BardGo2Dynamics(URDF, device="cpu", batch_capacity=4)
        cls.pin_model = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
        cls.nominal_inertia = cls.pin_model.inertias[1].copy()

    def simulator_state(self, batch=1, rotated=False):
        q = torch.zeros(batch, 19)
        q[:, 2] = 0.42
        if rotated:
            # 90 degrees around z, simulator XYZW.
            q[:, 2:4] = torch.tensor([0.42, 0.0])
            q[:, 5] = np.sqrt(0.5)
            q[:, 6] = np.sqrt(0.5)
        else:
            q[:, 6] = 1.0
        q[:, 7:] = torch.linspace(-0.4, 0.5, 12)
        v = torch.linspace(-0.3, 0.4, 18).repeat(batch, 1)
        a = torch.linspace(0.2, -0.1, 18).repeat(batch, 1)
        return q, v, a

    def pin_result(self, q_bard, v_bard, a_bard, added=0., shift=(0., 0., 0.),
                   armature=0., friction=0., stiffness=0., damping=0.):
        pin = self.pin
        model = self.pin_model
        nominal = self.nominal_inertia
        mass = nominal.mass + added
        model.inertias[1] = pin.Inertia(
            mass,
            np.asarray(nominal.lever) + np.asarray(shift),
            np.asarray(nominal.inertia) * (mass / nominal.mass),
        )
        data = model.createData()
        q_pin = q_bard.numpy().copy()
        q_pin[3:7] = q_pin[[4, 5, 6, 3]]  # BARD WXYZ -> Pinocchio XYZW
        result = pin.rnea(
            model, data, q_pin, v_bard.numpy(), a_bard.numpy()
        )
        result[6:] += armature * a_bard.numpy()[6:]
        q_sim = q_bard.numpy()[7:][SIM_FROM_BARD]
        v_sim = v_bard.numpy()[6:][SIM_FROM_BARD]
        result_sim = result[[0,1,2,3,4,5] + [6+i for i in SIM_FROM_BARD]]
        result_sim[6:] += friction * np.tanh(v_sim / 0.01)
        result_sim[6:] += stiffness * q_sim + damping * v_sim
        return torch.tensor(result_sim, dtype=torch.float32)

    def test_quaternion_joint_and_world_to_body_twist_conversion(self):
        q, v, _ = self.simulator_state(rotated=True)
        q_bard, v_bard = simulator_state_to_bard(
            q, v, bard_joint_names=self.dynamics.bard_joint_names
        )
        torch.testing.assert_close(q_bard[:, 3:7], q[:, 3:7][:, (3,0,1,2)])
        torch.testing.assert_close(q_bard[:, 7:], q[:, 7:][:, [3,4,5,0,1,2,9,10,11,6,7,8]])
        expected_linear = torch.tensor([[v[0,1], -v[0,0], v[0,2]]])
        torch.testing.assert_close(v_bard[:, :3], expected_linear, atol=1e-6, rtol=1e-6)

    def test_spatial_inertia_parallel_axis_blocks(self):
        mass = torch.tensor([2.0])
        com = torch.tensor([[1.0, 0.0, 0.0]])
        inertia_com = torch.diag(torch.tensor([3., 4., 5.])).unsqueeze(0)
        spatial = build_linear_first_spatial_inertia(mass, com, inertia_com)
        torch.testing.assert_close(spatial[0, :3, :3], 2 * torch.eye(3))
        torch.testing.assert_close(
            spatial[0, 3:, 3:], torch.diag(torch.tensor([3., 6., 7.]))
        )

    def test_nominal_and_nonidentity_orientation_parity(self):
        for rotated in (False, True):
            q, v_world, a_world = self.simulator_state(rotated=rotated)
            q_bard, v_bard = simulator_state_to_bard(
                q, v_world, bard_joint_names=self.dynamics.bard_joint_names
            )
            _, a_bard = simulator_state_to_bard(
                q, a_world, bard_joint_names=self.dynamics.bard_joint_names
            )
            actual = self.dynamics.evaluate(q_bard, v_bard, a_bard).rnea[0]
            expected = self.pin_result(q_bard[0], v_bard[0], a_bard[0])
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)

    def test_randomized_inertia_armature_and_passive_parity(self):
        q, v_world, a_world = self.simulator_state()
        q_bard, v_bard = simulator_state_to_bard(
            q, v_world, bard_joint_names=self.dynamics.bard_joint_names
        )
        _, a_bard = simulator_state_to_bard(
            q, a_world, bard_joint_names=self.dynamics.bard_joint_names
        )
        params = {
            "added_base_mass": torch.tensor([[2.5]]),
            "base_com_shift": torch.tensor([[0.07, -0.03, 0.02]]),
            "joint_armature": torch.tensor([[0.015]]),
            "joint_friction": torch.tensor([[0.08]]),
            "joint_stiffness": torch.tensor([[0.02]]),
            "joint_damping": torch.tensor([[0.35]]),
        }
        actual = self.dynamics.evaluate(
            q_bard, v_bard, a_bard, parameters=params
        ).rnea[0]
        expected = self.pin_result(
            q_bard[0], v_bard[0], a_bard[0], 2.5, (0.07, -0.03, 0.02),
            0.015, 0.08, 0.02, 0.35,
        )
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)

    def test_world_aligned_foot_and_base_jacobian_parity(self):
        q, v_world, a_world = self.simulator_state(rotated=True)
        q_bard, v_bard = simulator_state_to_bard(
            q, v_world, bard_joint_names=self.dynamics.bard_joint_names
        )
        _, a_bard = simulator_state_to_bard(
            q, a_world, bard_joint_names=self.dynamics.bard_joint_names
        )
        actual = self.dynamics.evaluate(q_bard, v_bard, a_bard)
        q_pin = q_bard[0].numpy().copy()
        q_pin[3:7] = q_pin[[4, 5, 6, 3]]
        order = [0,1,2,3,4,5] + [6+i for i in SIM_FROM_BARD]
        data = self.pin_model.createData()
        expected_feet = []
        for name in ("FR_foot", "FL_foot", "RR_foot", "RL_foot"):
            frame = self.pin_model.getFrameId(name)
            jacobian = self.pin.computeFrameJacobian(
                self.pin_model, data, q_pin, frame,
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            expected_feet.append(torch.tensor(jacobian[:3, order], dtype=torch.float32))
        expected_feet = torch.stack(expected_feet).unsqueeze(0)
        torch.testing.assert_close(actual.foot_jacobians, expected_feet, atol=2e-5, rtol=2e-5)
        base_frame = self.pin_model.getFrameId("base")
        base = self.pin.computeFrameJacobian(
            self.pin_model, data, q_pin, base_frame,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        torch.testing.assert_close(
            actual.base_jacobian,
            torch.tensor(base[:, order], dtype=torch.float32).unsqueeze(0),
            atol=2e-5, rtol=2e-5,
        )

    def test_large_minibatch_is_capacity_chunked(self):
        dynamics = BardGo2Dynamics(URDF, device="cpu", batch_capacity=1)
        q, v_world, a_world = self.simulator_state(batch=2)
        q_bard, v_bard = simulator_state_to_bard(
            q, v_world, bard_joint_names=dynamics.bard_joint_names
        )
        _, a_bard = simulator_state_to_bard(
            q, a_world, bard_joint_names=dynamics.bard_joint_names
        )
        terms = dynamics.evaluate(q_bard, v_bard, a_bard)
        self.assertEqual(terms.rnea.shape, (2, 18))
        torch.testing.assert_close(terms.rnea[0], terms.rnea[1])


if __name__ == "__main__":
    unittest.main()
