"""Canonical and loss parity for selectable HardPACT dynamics backends."""

import os
import unittest

os.environ.setdefault("SIMULATOR", "genesis_pact")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_hard_pact_backend_tests")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_hard_pact_backend_tests")

import torch

from legged_gym.dynamics import (
    BardGo2Dynamics, PinocchioGo2Dynamics, create_go2_dynamics,
)
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import (
    GO2HardPACTCfgPPO,
)
from rsl_rl.algorithms.hard_pact_bard import (
    corrected_bard_inverse_dynamics_loss,
    differentiable_bard_rollout_loss,
)


URDF = "resources/robots/go2/urdf/go2.urdf"


class DynamicsBackendParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bard = BardGo2Dynamics(URDF, device="cpu", batch_capacity=2)
        cls.pinocchio = PinocchioGo2Dynamics(
            URDF, device="cpu", batch_capacity=2, num_workers=2
        )

    @classmethod
    def tearDownClass(cls):
        cls.pinocchio.shutdown()

    @staticmethod
    def inputs():
        q = torch.zeros(2, 19)
        q[:, 2] = 0.42
        q[0, 6] = 1.0
        q[1, 5:7] = 2.0 ** -0.5
        q[:, 7:] = torch.linspace(-0.35, 0.45, 12)
        v = torch.linspace(-0.25, 0.35, 18).repeat(2, 1)
        post = v + torch.linspace(-0.01, 0.02, 18).repeat(2, 1)
        parameters = {
            "added_base_mass": torch.tensor([[0.0], [1.7]]),
            "base_com_shift": torch.tensor([[0., 0., 0.], [.04, -.02, .03]]),
            "joint_armature": torch.tensor([[0.0], [0.015]]),
            "joint_friction": torch.tensor([[0.0], [0.04]]),
            "joint_stiffness": torch.tensor([[0.0], [0.03]]),
            "joint_damping": torch.tensor([[0.0], [0.2]]),
        }
        return q, v, post, parameters

    def contexts(self, need_qp=True):
        q, v, post, parameters = self.inputs()
        kwargs = dict(
            parameters=parameters, post_v_world=post,
            need_forward_dynamics=True, need_qp=need_qp,
        )
        return (
            self.bard.build_context(q, v, **kwargs),
            self.pinocchio.build_context(q, v, **kwargs),
        )

    def test_all_consumed_context_outputs_match(self):
        bard, pin = self.contexts()
        tolerances = {
            "mass_matrix": (3e-4, 3e-4),
            "bias": (4e-4, 4e-4),
            "foot_jacobians": (3e-5, 3e-5),
            "base_jacobian": (3e-5, 3e-5),
            "foot_acceleration_bias": (5e-4, 5e-4),
        }
        for name, (atol, rtol) in tolerances.items():
            torch.testing.assert_close(
                getattr(bard, name), getattr(pin, name), atol=atol, rtol=rtol
            )
        acceleration = torch.linspace(-0.2, 0.3, 18).repeat(2, 1)
        torch.testing.assert_close(
            bard.rnea(acceleration), pin.rnea(acceleration),
            atol=5e-4, rtol=5e-4,
        )
        generalized = torch.linspace(-6.0, 8.0, 18).repeat(2, 1)
        torch.testing.assert_close(
            bard.forward_dynamics(generalized),
            pin.forward_dynamics(generalized), atol=3e-3, rtol=3e-3,
        )

    def test_default_and_factory_selection(self):
        self.assertEqual(GO2HardPACTCfgPPO.algorithm.dynamics_backend, "bard")
        temporary = create_go2_dynamics(
            "pinocchio", URDF, device="cpu", batch_capacity=1, num_workers=1
        )
        self.assertIsInstance(temporary, PinocchioGo2Dynamics)
        temporary.shutdown()
        with self.assertRaisesRegex(ValueError, "bard.*pinocchio"):
            create_go2_dynamics("unknown", URDF)

    def test_inverse_and_rollout_loss_and_gradient_parity(self):
        bard, pin = self.contexts(need_qp=False)
        dt = torch.full((2, 1), 0.02)
        acceleration = (bard.post_v_bard - bard.v_bard) / dt
        torque = torch.linspace(-2., 3., 12).repeat(2, 1)
        grf_values = torch.linspace(-4., 70., 12).reshape(1, 4, 3).repeat(2, 1, 1)
        wrench_values = torch.linspace(-2., 4., 6).repeat(2, 1)
        masks = dict(
            push_event_mask=torch.zeros(2, 1, dtype=torch.bool),
            reset_mask=torch.zeros(2, 1, dtype=torch.bool),
            timeout_mask=torch.zeros(2, 1, dtype=torch.bool),
            teleport_mask=torch.zeros(2, 1, dtype=torch.bool),
        )
        inverse_losses = []
        rollout_losses = []
        rollout_gradients = []
        for context in (bard, pin):
            grf = grf_values.clone().requires_grad_()
            wrench = wrench_values.clone().requires_grad_()
            control = torque.clone().requires_grad_()
            inverse = corrected_bard_inverse_dynamics_loss(
                required_generalized_force=context.rnea(acceleration),
                foot_jacobians=context.foot_jacobians,
                base_jacobian=context.base_jacobian,
                interval_executed_torque=torque,
                interval_grf_world=grf,
                total_wrench_world=wrench,
                mass_com_wrench_world=torch.zeros(2, 6),
                measured_generalized_contact_force=torch.ones(2, 18),
                **masks,
            )
            rollout = differentiable_bard_rollout_loss(
                context=context, control_torque=control,
                interval_grf_world=grf, applied_wrench_world=wrench,
                control_dt=dt, **masks,
            )
            inverse_losses.append(inverse.loss)
            rollout_losses.append(rollout.loss)
            rollout_gradients.append(torch.autograd.grad(
                rollout.loss, (control, grf, wrench), retain_graph=True
            ))
        torch.testing.assert_close(
            inverse_losses[0], inverse_losses[1], atol=2e-4, rtol=2e-3
        )
        torch.testing.assert_close(
            rollout_losses[0], rollout_losses[1], atol=2e-4, rtol=3e-3
        )
        for bard_gradient, pin_gradient in zip(*rollout_gradients):
            torch.testing.assert_close(
                bard_gradient, pin_gradient, atol=3e-4, rtol=4e-3
            )


if __name__ == "__main__":
    unittest.main()
