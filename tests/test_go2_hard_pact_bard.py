import importlib.util
import unittest

import torch

from legged_gym.dynamics import BardGo2Dynamics
from legged_gym.envs.go2.go2_hard_pact.schema import (
    GO2_FOOT_NAMES,
    GO2_JOINT_NAMES,
)


HAS_BARD = importlib.util.find_spec("bard") is not None


@unittest.skipUnless(HAS_BARD, "official BARD is not installed")
class Go2BARDIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = torch.tensor([
            -0.1, 0.8, -1.5, 0.1, 0.8, -1.5,
            -0.1, 0.95, -1.5, 0.1, 0.95, -1.5,
        ])
        cls.dynamics = BardGo2Dynamics(
            "resources/robots/go2/urdf/go2.urdf",
            GO2_JOINT_NAMES,
            GO2_FOOT_NAMES,
            "base",
            device="cpu",
            batch_capacity=3,
            default_joint_position=cls.default,
        )

    def state(self, batch=2):
        q = torch.zeros(batch, 19)
        q[:, 2] = 0.44
        q[:, 6] = 1.0  # canonical XYZW identity
        q[:, 7:] = self.default
        return q, torch.zeros(batch, 18)

    @staticmethod
    def parameters(batch=2):
        return {
            "added_base_mass": torch.zeros(batch, 1),
            "base_com_shift": torch.zeros(batch, 3),
            "joint_armature": torch.zeros(batch, 1),
            "joint_friction": torch.zeros(batch, 1),
            "joint_stiffness": torch.zeros(batch, 1),
            "joint_damping": torch.zeros(batch, 1),
        }

    def test_crba_bias_jacobians_and_jdot_v_shapes(self):
        q, v = self.state()
        terms = self.dynamics.terms(q, v, parameters=self.parameters())
        self.assertEqual(terms.mass.shape, (2, 18, 18))
        self.assertEqual(terms.bias.shape, (2, 18))
        self.assertEqual(terms.foot_jacobians.shape, (2, 4, 3, 18))
        self.assertEqual(terms.base_jacobian.shape, (2, 6, 18))
        self.assertEqual(terms.foot_jdot_v.shape, (2, 4, 3))
        torch.testing.assert_close(terms.mass, terms.mass.transpose(1, 2), atol=2e-5, rtol=2e-5)

    def test_batched_randomized_inertia_changes_crba_per_environment(self):
        q, v = self.state()
        parameters = self.parameters()
        parameters["added_base_mass"][1] = 3.0
        parameters["base_com_shift"][1] = torch.tensor([0.1, -0.05, 0.03])
        mass = self.dynamics.terms(q, v, parameters=parameters).mass
        self.assertGreater((mass[1] - mass[0]).abs().max().item(), 1.0e-4)

    def test_rnea_aba_consistency_and_force_gradient(self):
        q, v = self.state()
        parameters = self.parameters()
        expected_acceleration = torch.randn(2, 18) * 0.1
        force = self.dynamics.rnea(
            q, v, expected_acceleration, parameters=parameters
        ).detach().requires_grad_(True)
        recovered = self.dynamics.aba(q, v, force, parameters=parameters)
        torch.testing.assert_close(recovered, expected_acceleration, atol=2e-4, rtol=2e-4)
        recovered.square().sum().backward()
        self.assertIsNotNone(force.grad)
        self.assertTrue(torch.isfinite(force.grad).all())


if __name__ == "__main__":
    unittest.main()
