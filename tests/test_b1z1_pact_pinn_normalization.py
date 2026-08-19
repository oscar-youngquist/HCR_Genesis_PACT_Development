"""Focused checks for B1Z1's Go2-PACT relative PINN normalization."""

import unittest

import torch

from rsl_rl.algorithms.ppo_b1z1_pact import _go2_relative_pinn_loss


class B1Z1PACTPINNNormalizationTests(unittest.TestCase):
    def test_uses_sum_of_actuator_and_external_force_magnitudes(self):
        residual = torch.tensor([[3.0, 4.0], [0.0, 9.0]])
        actuator = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
        external = torch.tensor([[0.0, 12.0], [0.0, 1.0]])
        valid = torch.tensor([[1.0], [0.0]])

        loss = _go2_relative_pinn_loss(residual, actuator, external, valid)

        self.assertTrue(torch.allclose(loss, torch.tensor(5.0 / 17.0)))

    def test_zero_force_and_empty_valid_batch_remain_finite(self):
        zeros = torch.zeros(2, 25)
        valid = torch.zeros(2, 1)

        loss = _go2_relative_pinn_loss(zeros, zeros, zeros, valid)

        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
