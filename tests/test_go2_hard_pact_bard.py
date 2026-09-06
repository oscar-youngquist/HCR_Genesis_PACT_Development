import os
import unittest
from unittest import mock

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
    fixed_mechanics_forward_dynamics,
    simulator_state_to_bard,
    wrench_at_point,
)
from rsl_rl.algorithms.hard_pact_bard import (
    BARD_ROLLOUT_INCREMENT_RATE_SCALES,
    corrected_bard_inverse_dynamics_loss,
    differentiable_bard_rollout_loss,
    physics_valid_mask,
)
from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
from rsl_rl.modules.actor_critic_hard_pact import ActorCritic_HardPACT
from rsl_rl.modules.hard_pact_physics import DeploymentPhysicsHeads
from legged_gym.envs.go2.go2_hard_pact.deployment import calculate_physics_head_gains
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg


URDF = "resources/robots/go2/urdf/go2.urdf"
SIM_FROM_BARD = [BARD_JOINT_ORDER.index(name) for name in SIMULATOR_JOINT_ORDER]


class FixedMechanicsSolveTests(unittest.TestCase):
    def test_dtype_forward_residual_vjp_and_detachment(self):
        for dtype, tolerance in ((torch.float32, 2e-5), (torch.float64, 1e-11)):
            torch.manual_seed(3)
            root = torch.randn(4, 18, 18, dtype=dtype)
            matrix = (root @ root.transpose(-1, -2) + 0.5 * torch.eye(18, dtype=dtype))
            matrix.requires_grad_()
            bias = torch.randn(4, 18, dtype=dtype, requires_grad=True)
            force = torch.randn(4, 18, dtype=dtype, requires_grad=True)
            output = fixed_mechanics_forward_dynamics(matrix, bias, force)
            expected = torch.linalg.solve(matrix.detach(), (force - bias.detach()).unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)
            residual = torch.einsum("bij,bj->bi", matrix.detach(), output) + bias.detach() - force
            self.assertLess(residual.abs().max().item(), tolerance * 20)
            seed = torch.randn_like(output)
            (output * seed).sum().backward()
            expected_vjp = torch.linalg.solve(
                matrix.detach().transpose(-1, -2), seed.unsqueeze(-1)
            ).squeeze(-1)
            torch.testing.assert_close(force.grad, expected_vjp, atol=tolerance, rtol=tolerance)
            self.assertIsNone(matrix.grad)
            self.assertIsNone(bias.grad)

    def test_deterministic_factorization_fallback(self):
        # Invertible but indefinite: Cholesky must reject it and solve_ex must
        # produce the certified result used by both forward and transpose VJP.
        matrix = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float64)
        force = torch.tensor([[2.0, -3.0]], dtype=torch.float64, requires_grad=True)
        output = fixed_mechanics_forward_dynamics(
            matrix, torch.zeros_like(force), force
        )
        torch.testing.assert_close(output, torch.tensor([[-3.0, 2.0]], dtype=torch.float64))
        output.sum().backward()
        torch.testing.assert_close(force.grad, torch.ones_like(force))

    @unittest.skipUnless(
        torch.cuda.is_available() and os.environ.get("RUN_BARD_CUDA_4096") == "1",
        "explicit CUDA-4096 benchmark test",
    )
    def test_cuda_4096_real_bard_context_forward_and_backward(self):
        batch = 4096
        dynamics = BardGo2Dynamics(
            URDF, device="cuda:0", batch_capacity=batch
        )
        q = torch.zeros(batch, 19, device="cuda:0")
        q[:, 2], q[:, 6] = 0.42, 1.0
        q[:, 7:] = torch.linspace(-0.25, 0.35, 12, device="cuda:0")
        v = torch.linspace(-0.15, 0.2, 18, device="cuda:0").expand(batch, -1)
        context = dynamics.build_context(
            q, v,
            parameters={
                "added_base_mass": torch.ones(batch, 1, device="cuda:0"),
                "base_com_shift": torch.full((batch, 3), 0.01, device="cuda:0"),
                "joint_armature": torch.full((batch, 1), 0.015, device="cuda:0"),
                "joint_friction": torch.full((batch, 1), 0.03, device="cuda:0"),
                "joint_stiffness": torch.full((batch, 1), 0.02, device="cuda:0"),
                "joint_damping": torch.full((batch, 1), 0.12, device="cuda:0"),
            },
            need_forward_dynamics=True,
        )
        force = torch.randn(batch, 18, device="cuda:0", requires_grad=True)
        acceleration = context.forward_dynamics(force)
        acceleration.square().mean().backward()
        self.assertTrue(torch.isfinite(acceleration).all())
        self.assertTrue(torch.isfinite(force.grad).all())


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
        gains = calculate_physics_head_gains(GO2HardPACTCfg())
        heads = DeploymentPhysicsHeads(
            grf_scale_n=gains.grf_scale_n,
            wrench_scale=gains.wrench_scale_n_nm,
            wrench_qp_clip=gains.wrench_qp_clip_n_nm,
        )
        latent = torch.randn(2, 16, requires_grad=True)
        explicit = torch.randn(2, 11, requires_grad=True)
        torque = torch.randn(2, 12, requires_grad=True)
        prediction = heads(latent, explicit, torque)
        data = self.inputs(2)
        data["required_generalized_force"].normal_()
        data["interval_grf_world"] = heads.grf_to_physical(
            prediction.grf_normalized.reshape(2, 4, 3)
        )
        data["total_wrench_world"] = heads.wrench_to_physical(
            prediction.wrench_raw_normalized
        )
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
        # Other tests exercise Pinocchio ABA with realized armature on this
        # shared model; RNEA adds armature explicitly below, so reset it here.
        model.armature[:] = 0.0
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

    def pin_acceleration(self, context, generalized_force):
        pin, model = self.pin, self.pin_model
        parameters = context.parameters
        added = float(parameters.get("added_base_mass", torch.zeros(1)).reshape(-1)[0])
        shift = parameters.get("base_com_shift", torch.zeros(1, 3))[0].numpy()
        nominal = self.nominal_inertia
        mass = nominal.mass + added
        model.inertias[1] = pin.Inertia(
            mass, np.asarray(nominal.lever) + shift,
            np.asarray(nominal.inertia) * (mass / nominal.mass),
        )
        armature = parameters.get("joint_armature", torch.zeros(1, 1))
        armature = armature.expand(-1, 12)[0].numpy()
        model.armature[:] = 0.0
        model.armature[6:] = armature
        q = context.q_bard[0].numpy().copy()
        q[3:7] = q[[4, 5, 6, 3]]
        v = context.v_bard[0].numpy()
        canonical_to_bard = self.dynamics._bard_from_canonical.cpu().numpy()
        tau = generalized_force[0].detach().numpy()[canonical_to_bard]
        q_sim = context.q_bard[0, 7:].numpy()[SIM_FROM_BARD]
        v_sim = self.dynamics._canonical(context.v_bard)[0, 6:].numpy()
        def expanded(name):
            value = parameters.get(name, torch.zeros(1, 1)).expand(-1, 12)
            return value[0].numpy()
        passive_sim = (
            expanded("joint_friction") * np.tanh(v_sim / 0.01)
            + expanded("joint_stiffness") * q_sim
            + expanded("joint_damping") * v_sim
        )
        passive_bard = passive_sim[np.argsort(SIM_FROM_BARD)]
        tau[6:] -= passive_bard
        acceleration = pin.aba(model, model.createData(), q, v, tau)
        order = [0, 1, 2, 3, 4, 5] + [6 + i for i in SIM_FROM_BARD]
        return torch.tensor(acceleration[order], dtype=torch.float32)

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

    def test_aba_acceleration_and_velocity_pinocchio_parity(self):
        cases = (
            (False, {}),
            (True, {
                "added_base_mass": torch.tensor([[2.5]]),
                "base_com_shift": torch.tensor([[0.07, -0.03, 0.02]]),
                "joint_armature": torch.tensor([[0.015]]),
                "joint_friction": torch.tensor([[0.08]]),
                "joint_stiffness": torch.tensor([[0.02]]),
                "joint_damping": torch.tensor([[0.35]]),
            }),
        )
        for rotated, parameters in cases:
            q, v, _ = self.simulator_state(rotated=rotated)
            context = self.dynamics.build_context(
                q, v, parameters=parameters, post_v_world=v,
                need_jacobians=True,
            )
            torque = torch.linspace(-8.0, 9.0, 12).unsqueeze(0)
            grf = torch.tensor([[[10., -4., 70.], [2., 3., 65.],
                                 [-5., 2., 75.], [4., -1., 68.]]])
            wrench = torch.tensor([[7., -3., 9., 1., 2., -1.]])
            generalized = torch.cat((torch.zeros(1, 6), torque), dim=-1)
            generalized += torch.einsum("bfkn,bfk->bn", context.foot_jacobians, grf)
            generalized += torch.einsum("bkn,bk->bn", context.base_jacobian, wrench)
            actual = context.aba(generalized)
            expected = self.pin_acceleration(context, generalized)
            torch.testing.assert_close(actual[0], expected, atol=2e-3, rtol=2e-3)
            dt = 0.02
            actual_v = self.dynamics._canonical(context.v_bard) + dt * actual
            expected_v = self.dynamics._canonical(context.v_bard) + dt * expected
            torch.testing.assert_close(actual_v, expected_v, atol=4e-5, rtol=2e-3)

    def test_context_reuses_conversion_kinematics_and_jacobians(self):
        q, v, _ = self.simulator_state()
        bard = self.dynamics.bard
        with (
            mock.patch.object(
                bard, "update_kinematics", wraps=bard.update_kinematics
            ) as update,
            mock.patch.object(
                self.dynamics, "_world_jacobian",
                wraps=self.dynamics._world_jacobian,
            ) as jacobian,
            mock.patch.object(bard, "rnea", wraps=bard.rnea) as rnea,
            mock.patch.object(bard, "aba", wraps=bard.aba) as aba,
        ):
            context = self.dynamics.build_context(
                q, v, post_v_world=v, need_jacobians=True
            )
            context.rnea(torch.zeros_like(context.v_bard))
            context.aba(torch.zeros(1, 18))
        self.assertEqual(update.call_count, 1)
        self.assertEqual(jacobian.call_count, 5)  # four feet plus base
        self.assertEqual(rnea.call_count, 1)
        self.assertEqual(aba.call_count, 1)

    def test_qp_terms_share_the_single_bard_context(self):
        q, v, _ = self.simulator_state(rotated=True)
        bard = self.dynamics.bard
        with (
            mock.patch.object(
                bard, "update_kinematics", wraps=bard.update_kinematics
            ) as update,
            mock.patch.object(bard, "crba", wraps=bard.crba) as crba,
            mock.patch.object(
                bard, "spatial_acceleration", wraps=bard.spatial_acceleration
            ) as acceleration,
            mock.patch.object(bard, "rnea", wraps=bard.rnea) as rnea,
        ):
            context = self.dynamics.build_context(
                q, v,
                parameters={"joint_armature": torch.full((1, 1), 0.02)},
                need_qp=True,
            )
        self.assertEqual(update.call_count, 1)
        self.assertEqual(crba.call_count, 1)
        self.assertEqual(acceleration.call_count, 4)
        self.assertEqual(rnea.call_count, 1)
        self.assertEqual(context.mass_matrix.shape, (1, 18, 18))
        self.assertEqual(context.bias.shape, (1, 18))
        self.assertEqual(context.foot_acceleration_bias.shape, (1, 4, 3))
        torch.testing.assert_close(
            context.mass_matrix, context.mass_matrix.transpose(-1, -2)
        )
        self.assertTrue(torch.linalg.cholesky_ex(context.mass_matrix).info.eq(0).all())

    def test_forward_dynamics_context_skips_qp_only_acceleration_terms(self):
        q, v, _ = self.simulator_state(rotated=True)
        bard = self.dynamics.bard
        with (
            mock.patch.object(bard, "crba", wraps=bard.crba) as crba,
            mock.patch.object(bard, "rnea", wraps=bard.rnea) as rnea,
            mock.patch.object(
                bard, "spatial_acceleration", wraps=bard.spatial_acceleration
            ) as spatial_acceleration,
        ):
            context = self.dynamics.build_context(
                q, v, need_forward_dynamics=True
            )
        self.assertEqual(crba.call_count, 1)
        self.assertEqual(rnea.call_count, 1)
        self.assertEqual(spatial_acceleration.call_count, 0)
        self.assertIsNotNone(context.mass_matrix)
        self.assertIsNotNone(context.bias)
        self.assertIsNone(context.foot_acceleration_bias)

    def test_crba_rnea_solve_matches_official_aba_randomization_cases(self):
        cases = (
            {},
            {"added_base_mass": torch.tensor([[1.7]])},
            {"base_com_shift": torch.tensor([[0.05, -0.02, 0.03]])},
            {"joint_armature": torch.tensor([[0.018]])},
            {"joint_friction": torch.tensor([[0.04]])},
            {"joint_stiffness": torch.tensor([[0.03]])},
            {"joint_damping": torch.tensor([[0.2]])},
            {
                "added_base_mass": torch.tensor([[1.7]]),
                "base_com_shift": torch.tensor([[0.05, -0.02, 0.03]]),
                "joint_armature": torch.tensor([[0.018]]),
                "joint_friction": torch.tensor([[0.04]]),
                "joint_stiffness": torch.tensor([[0.03]]),
                "joint_damping": torch.tensor([[0.2]]),
            },
        )
        q, v, _ = self.simulator_state(rotated=True)
        generalized = torch.linspace(-7.0, 8.0, 18).unsqueeze(0)
        for parameters in cases:
            with self.subTest(parameters=tuple(parameters)):
                context = self.dynamics.build_context(
                    q, v, parameters=parameters,
                    need_forward_dynamics=True,
                )
                actual = context.forward_dynamics(generalized)
                reference = context.aba(generalized)
                torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-3)
                residual = (
                    torch.einsum("bij,bj->bi", context.mass_matrix, actual)
                    + context.bias - generalized
                )
                self.assertLess(residual.abs().max().item(), 2e-4)

    def test_cached_mass_times_acceleration_plus_bias_matches_rnea(self):
        """The optimized inverse target preserves armature/passive semantics."""
        q, v, _ = self.simulator_state(rotated=True)
        parameters = {
            "added_base_mass": torch.tensor([[1.2]]),
            "base_com_shift": torch.tensor([[0.03, -0.02, 0.01]]),
            "joint_armature": torch.full((1, 12), 0.015),
            "joint_friction": torch.full((1, 12), 0.02),
            "joint_stiffness": torch.full((1, 12), 0.04),
            "joint_damping": torch.full((1, 12), 0.1),
        }
        context = self.dynamics.build_context(
            q, v, parameters=parameters, need_forward_dynamics=True
        )
        acceleration = torch.linspace(-2.0, 3.0, 18).unsqueeze(0)
        cached = (
            torch.einsum("bij,bj->bi", context.mass_matrix, acceleration)
            + context.bias
        )
        direct = context.rnea(self.dynamics._bard_order(acceleration))
        torch.testing.assert_close(cached, direct, atol=3e-4, rtol=2e-5)

    def test_fixed_solve_and_aba_generalized_force_vjp_parity(self):
        q, v, _ = self.simulator_state(rotated=True)
        parameters = {
            "added_base_mass": torch.tensor([[1.2]]),
            "base_com_shift": torch.tensor([[0.03, -0.01, 0.02]]),
            "joint_armature": torch.tensor([[0.012]]),
            "joint_friction": torch.tensor([[0.02]]),
            "joint_stiffness": torch.tensor([[0.01]]),
            "joint_damping": torch.tensor([[0.08]]),
        }
        context = self.dynamics.build_context(
            q, v, parameters=parameters, need_forward_dynamics=True
        )
        force_fixed = torch.linspace(-2.0, 3.0, 18).unsqueeze(0).requires_grad_()
        force_aba = force_fixed.detach().clone().requires_grad_()
        seed = torch.linspace(0.3, 1.1, 18).unsqueeze(0)
        fixed = context.forward_dynamics(force_fixed)
        reference = context.aba(force_aba)
        fixed_vjp = torch.autograd.grad((fixed * seed).sum(), force_fixed)[0]
        aba_vjp = torch.autograd.grad((reference * seed).sum(), force_aba)[0]
        torch.testing.assert_close(fixed, reference, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(fixed_vjp, aba_vjp, atol=3e-3, rtol=3e-3)

    def test_rollout_loss_and_force_gradients_match_official_aba(self):
        q, v, _ = self.simulator_state(rotated=True)
        post_v = v + torch.linspace(-0.01, 0.02, 18).unsqueeze(0)
        context = self.dynamics.build_context(
            q, v, post_v_world=post_v, need_forward_dynamics=True,
        )

        class ABAReferenceContext:
            def __init__(self, source):
                self.source = source

            def __getattr__(self, name):
                return getattr(self.source, name)

            def forward_dynamics(self, generalized_force):
                return self.source.aba(generalized_force)

        common = dict(
            control_dt=torch.tensor([[0.02]]),
            push_event_mask=torch.zeros(1, 1, dtype=torch.bool),
            reset_mask=torch.zeros(1, 1, dtype=torch.bool),
            timeout_mask=torch.zeros(1, 1, dtype=torch.bool),
            teleport_mask=torch.zeros(1, 1, dtype=torch.bool),
        )
        fixed_inputs = (
            torch.linspace(-3.0, 4.0, 12).unsqueeze(0).requires_grad_(),
            torch.linspace(-5.0, 60.0, 12).reshape(1, 4, 3).requires_grad_(),
            torch.linspace(-2.0, 3.0, 6).unsqueeze(0).requires_grad_(),
        )
        aba_inputs = tuple(value.detach().clone().requires_grad_() for value in fixed_inputs)
        fixed = differentiable_bard_rollout_loss(
            context=context, control_torque=fixed_inputs[0],
            interval_grf_world=fixed_inputs[1], applied_wrench_world=fixed_inputs[2],
            **common,
        )
        reference = differentiable_bard_rollout_loss(
            context=ABAReferenceContext(context), control_torque=aba_inputs[0],
            interval_grf_world=aba_inputs[1], applied_wrench_world=aba_inputs[2],
            **common,
        )
        fixed_gradients = torch.autograd.grad(fixed.loss, fixed_inputs)
        aba_gradients = torch.autograd.grad(reference.loss, aba_inputs)
        torch.testing.assert_close(fixed.loss, reference.loss, atol=2e-4, rtol=2e-3)
        for actual, expected in zip(fixed_gradients, aba_gradients):
            torch.testing.assert_close(actual, expected, atol=3e-4, rtol=4e-3)

    def test_forward_dynamics_detaches_state_and_randomized_mechanics(self):
        q, v, _ = self.simulator_state(rotated=True)
        q.requires_grad_()
        v.requires_grad_()
        added_mass = torch.tensor([[1.1]], requires_grad=True)
        com_shift = torch.tensor([[0.03, -0.02, 0.01]], requires_grad=True)
        armature = torch.tensor([[0.015]], requires_grad=True)
        context = self.dynamics.build_context(
            q, v,
            parameters={
                "added_base_mass": added_mass,
                "base_com_shift": com_shift,
                "joint_armature": armature,
            },
            need_forward_dynamics=True,
        )
        force = torch.randn(1, 18, requires_grad=True)
        context.forward_dynamics(force).square().mean().backward()
        self.assertIsNotNone(force.grad)
        self.assertGreater(force.grad.abs().sum().item(), 0.0)
        for detached in (q, v, added_mass, com_shift, armature):
            self.assertIsNone(detached.grad)

    def test_fixed_mechanics_routes_finite_gradients_to_all_force_inputs(self):
        q, v, _ = self.simulator_state(rotated=True)
        post_v = v + 0.03
        context = self.dynamics.build_context(
            q, v,
            parameters={
                "joint_armature": torch.tensor([[0.015]]),
                "joint_friction": torch.tensor([[0.02]]),
                "joint_stiffness": torch.tensor([[0.01]]),
                "joint_damping": torch.tensor([[0.04]]),
            },
            post_v_world=post_v,
            need_jacobians=True, need_forward_dynamics=True,
        )
        torque = torch.linspace(-2.0, 2.0, 12).unsqueeze(0).requires_grad_()
        grf = torch.tensor(
            [[[3., -2., 45.], [1., 2., 40.], [-2., 1., 43.], [2., -1., 41.]]],
            requires_grad=True,
        )
        wrench = torch.tensor(
            [[4., -3., 5., 0.5, -0.2, 0.3]], requires_grad=True
        )
        result = differentiable_bard_rollout_loss(
            context=context,
            control_torque=torque,
            interval_grf_world=grf,
            applied_wrench_world=wrench,
            control_dt=torch.tensor([[0.02]]),
            push_event_mask=torch.zeros(1, 1, dtype=torch.bool),
            reset_mask=torch.zeros(1, 1, dtype=torch.bool),
            timeout_mask=torch.zeros(1, 1, dtype=torch.bool),
            teleport_mask=torch.zeros(1, 1, dtype=torch.bool),
        )
        result.loss.backward()
        for value in (torque, grf, wrench):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0.0)

    def test_aba_and_rnea_pinocchio_parity_with_replayed_stochastic_torque(self):
        torch.manual_seed(29)
        gains = calculate_physics_head_gains(GO2HardPACTCfg())
        actor = ActorCritic_HardPACT(
            num_actor_obs=57, num_critic_obs=95, num_actions=12,
            actor_layers=[32, 16], critic_layers=[32, 16],
            cenet_in_dim=57 * 20, cenet_enc_layers=[32, 16],
            cenet_explicit_layers=[16, 16],
            grf_decoder_layers=[16, 16], wrench_decoder_layers=[16, 16],
            grf_scale_n=gains.grf_scale_n,
            wrench_scale=gains.wrench_scale_n_nm,
            wrench_qp_clip=gains.wrench_qp_clip_n_nm,
        )
        algorithm = PPO_HardPACT.__new__(PPO_HardPACT)
        algorithm.actor_critic = actor
        algorithm.use_boot = True
        algorithm.action_clip = 2.0
        observation = torch.randn(1, 57)
        history = torch.randn(1, 57 * 20)
        _, _, latent, explicit = actor.cenet_enc_forward(history)
        mean_pos, mean_tau = actor.actor_forward(torch.cat((
            observation, latent, explicit
        ), dim=-1))
        mean = torch.cat((mean_pos, mean_tau), dim=-1)
        noise = torch.randn(1, 24)
        transition = {
            "standardized_action_noise": noise,
            "delayed_source_observation": observation,
            "delayed_source_history": history,
            "delayed_source_noise": noise,
            "delayed_action_source_valid": torch.ones(1, 1, dtype=torch.bool),
        }

        def transform(action):
            return action[:, :12], 1.5 * action[:, 12:]

        def feedback_fn(desired, position, velocity):
            return 2.0 * (desired - position) - 0.1 * velocity

        replay = algorithm._replay_action_path(
            mean, observation, transition, transform, feedback_fn,
            torch.zeros(12), 1.0,
        )
        torque = replay["nominal_torque"].detach()
        q, v_world, a_world = self.simulator_state(rotated=True)
        context = self.dynamics.build_context(
            q, v_world, post_v_world=v_world, need_jacobians=True
        )
        generalized = torch.cat((torch.zeros(1, 6), torque), dim=-1)
        bard_acceleration = context.aba(generalized)
        pin_acceleration = self.pin_acceleration(
            context, generalized
        ).unsqueeze(0)
        torch.testing.assert_close(
            bard_acceleration, pin_acceleration, atol=2e-3, rtol=2e-3
        )

        _, acceleration_bard = simulator_state_to_bard(
            q, a_world, bard_joint_names=self.dynamics.bard_joint_names
        )
        actuation = torch.cat((torch.zeros(1, 6), torque), dim=-1)
        bard_residual = context.rnea(acceleration_bard) - actuation
        pin_required = self.pin_result(
            context.q_bard[0], context.v_bard[0], acceleration_bard[0]
        ).unsqueeze(0)
        torch.testing.assert_close(
            bard_residual, pin_required - actuation, atol=3e-4, rtol=3e-4
        )


class RolloutLossTests(unittest.TestCase):
    class Context:
        def __init__(self, batch=3, post_value=0.0):
            self.foot_jacobians = torch.zeros(batch, 4, 3, 18)
            self.base_jacobian = torch.zeros(batch, 6, 18)
            self.base_jacobian[:, :, :6] = torch.eye(6)
            self.v_bard = torch.zeros(batch, 18)
            self.post_v_bard = torch.full((batch, 18), post_value)
            self.dynamics = self

        @staticmethod
        def _canonical(value):
            return value

        @staticmethod
        def aba(generalized_force):
            return generalized_force

        @staticmethod
        def forward_dynamics(generalized_force):
            return generalized_force

    def arguments(self, batch=3):
        return dict(
            context=self.Context(batch),
            control_torque=torch.ones(batch, 12, requires_grad=True),
            interval_grf_world=torch.zeros(batch, 4, 3, requires_grad=True),
            applied_wrench_world=torch.zeros(batch, 6, requires_grad=True),
            control_dt=torch.ones(batch, 1),
            push_event_mask=torch.zeros(batch, 1, dtype=torch.bool),
            reset_mask=torch.zeros(batch, 1, dtype=torch.bool),
            timeout_mask=torch.zeros(batch, 1, dtype=torch.bool),
            teleport_mask=torch.zeros(batch, 1, dtype=torch.bool),
        )

    def test_exact_masking_reduction_and_metrics(self):
        data = self.arguments(3)
        data["control_torque"].data.fill_(100.0)
        data["applied_wrench_world"].data[:, :3] = 10.0
        data["applied_wrench_world"].data[:, 3:] = 20.0
        data["reset_mask"][1] = True
        data["push_event_mask"][2] = True
        result = differentiable_bard_rollout_loss(**data)
        # Each block is exactly one normalized RMS unit; invalid samples do
        # not alter the valid sample's three-block mean.
        torch.testing.assert_close(result.loss, torch.tensor(1.0))
        self.assertEqual(len(result.metrics), 8)

    def test_increment_scaling_is_dt_invariant_and_blocks_are_balanced(self):
        self.assertEqual(BARD_ROLLOUT_INCREMENT_RATE_SCALES, (10.0, 20.0, 100.0))
        losses = []
        for dt in (0.01, 0.04):
            data = self.arguments(1)
            data["control_dt"].fill_(dt)
            data["control_torque"].data.fill_(100.0)
            data["applied_wrench_world"].data[:, :3] = 10.0
            data["applied_wrench_world"].data[:, 3:] = 20.0
            losses.append(differentiable_bard_rollout_loss(**data).loss)
        torch.testing.assert_close(losses[0], torch.tensor(1.0))
        torch.testing.assert_close(losses[1], torch.tensor(1.0))

        # A unit normalized residual in only one block contributes exactly
        # one third, despite the joint block containing four times more axes.
        data = self.arguments(1)
        data["control_torque"].data.fill_(100.0)
        result = differentiable_bard_rollout_loss(**data)
        torch.testing.assert_close(result.loss, torch.tensor(1.0 / 3.0))

    def test_observed_motion_softens_relative_increment_error(self):
        data = self.arguments(1)
        data["control_torque"].data.zero_()
        data["context"].post_v_bard[:, :3] = 10.0
        # Predicted base-linear increment is zero, so normalized residual and
        # normalized observed-motion RMS are both one at dt=1, rate=10.
        result = differentiable_bard_rollout_loss(**data)
        torch.testing.assert_close(result.loss, torch.tensor((1.0 / 2.0) / 3.0))

    def test_zero_motion_has_exact_zero_loss_and_finite_zero_gradients(self):
        data = self.arguments(1)
        data["control_torque"].data.zero_()
        result = differentiable_bard_rollout_loss(**data)
        torch.testing.assert_close(result.loss, torch.tensor(0.0))
        result.loss.backward()
        for name in ("control_torque", "interval_grf_world", "applied_wrench_world"):
            gradient = data[name].grad
            self.assertTrue(torch.isfinite(gradient).all())
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_all_invalid_is_graph_connected_zero(self):
        data = self.arguments(2)
        data["reset_mask"].fill_(True)
        result = differentiable_bard_rollout_loss(**data)
        self.assertEqual(result.loss.item(), 0.0)
        result.loss.backward()
        torch.testing.assert_close(
            data["control_torque"].grad,
            torch.zeros_like(data["control_torque"]),
        )

    def test_measured_targets_timestep_and_jacobians_are_detached(self):
        data = self.arguments(1)
        context = data["context"]
        context.foot_jacobians.requires_grad_()
        context.base_jacobian.requires_grad_()
        context.v_bard.requires_grad_()
        context.post_v_bard.requires_grad_()
        data["control_dt"].requires_grad_()
        differentiable_bard_rollout_loss(**data).loss.backward()
        for measured in (
            context.foot_jacobians, context.base_jacobian, context.v_bard,
            context.post_v_bard, data["control_dt"],
        ):
            self.assertIsNone(measured.grad)


class RolloutEndToEndGradientTests(unittest.TestCase):
    def test_fixed_rollout_reaches_policy_encoder_and_force_decoders(self):
        torch.manual_seed(23)
        dynamics = BardGo2Dynamics(URDF, device="cpu", batch_capacity=1)
        q = torch.zeros(1, 19)
        q[:, 2] = 0.42
        q[:, 6] = 1.0  # simulator XYZW identity quaternion
        q[:, 7:] = torch.linspace(-0.25, 0.35, 12)
        pre_v = torch.linspace(-0.1, 0.15, 18).unsqueeze(0)
        post_v = pre_v + torch.linspace(0.01, -0.02, 18).unsqueeze(0)
        context = dynamics.build_context(
            q, pre_v, post_v_world=post_v, need_jacobians=True,
            need_forward_dynamics=True,
        )
        gains = calculate_physics_head_gains(GO2HardPACTCfg())
        actor = ActorCritic_HardPACT(
            num_actor_obs=57, num_critic_obs=95, num_actions=12,
            actor_layers=[32, 16], critic_layers=[32, 16],
            cenet_in_dim=57 * 20, cenet_enc_layers=[32, 16],
            cenet_explicit_layers=[16, 16],
            grf_decoder_layers=[16, 16], wrench_decoder_layers=[16, 16],
            grf_scale_n=gains.grf_scale_n,
            wrench_scale=gains.wrench_scale_n_nm,
            wrench_qp_clip=gains.wrench_qp_clip_n_nm,
        )
        observation = torch.randn(1, 57)
        history = torch.randn(1, 57 * 20)
        _, _, latent, explicit = actor.cenet_enc_forward(history)
        policy_input = torch.cat((observation, latent, explicit), dim=-1)
        position_action, torque_action = actor.actor_forward(policy_input)
        # A compact differentiable stand-in for the legacy action/PD mapping;
        # both policy heads contribute to the torque sent through the solve.
        control_torque = position_action + torque_action
        heads = actor.physics_heads(latent, explicit, control_torque)
        result = differentiable_bard_rollout_loss(
            context=context,
            control_torque=control_torque,
            interval_grf_world=actor.physics_estimator.grf_to_physical(
                heads.grf_normalized.reshape(1, 4, 3)
            ),
            applied_wrench_world=actor.physics_estimator.wrench_to_physical(
                heads.wrench_raw_normalized
            ),
            control_dt=torch.tensor([[0.02]]),
            push_event_mask=torch.zeros(1, 1, dtype=torch.bool),
            reset_mask=torch.zeros(1, 1, dtype=torch.bool),
            timeout_mask=torch.zeros(1, 1, dtype=torch.bool),
            teleport_mask=torch.zeros(1, 1, dtype=torch.bool),
        )
        result.loss.backward()

        modules = {
            "policy": [actor.act_trunk, actor.act_pos_out, actor.act_tau_out],
            "encoder": [actor.context_encoder],
            "grf_decoder": [actor.physics_estimator.grf_head],
            "wrench_decoder": [actor.physics_estimator.wrench_head],
        }
        for name, group in modules.items():
            gradients = [
                parameter.grad
                for module in group
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients, name)
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients), name)
            self.assertGreater(sum(value.abs().sum().item() for value in gradients), 0.0, name)


if __name__ == "__main__":
    unittest.main()
