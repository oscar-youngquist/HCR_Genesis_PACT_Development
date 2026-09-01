import os
import unittest

os.environ.setdefault("SIMULATOR", "genesis_pact")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_pcgrad_tests")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_pcgrad_tests")

import torch

from rsl_rl.algorithms.pc_grad import PCGrad


class PCGradTests(unittest.TestCase):
    def test_sum_reduction_and_diagnostics(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = PCGrad(torch.optim.SGD([parameter], lr=0.1), reduction="sum")
        optimizer.pc_backward([parameter.sum(), (2.0 * parameter).sum()])
        torch.testing.assert_close(parameter.grad, torch.tensor([3.0, 3.0]))
        self.assertEqual(len(optimizer.last_objective_grads), 2)
        torch.testing.assert_close(optimizer.last_merged_grad, torch.tensor([3.0, 3.0]))

    def test_inactive_parameter_keeps_none_gradient(self):
        active = torch.nn.Parameter(torch.tensor(1.0))
        inactive = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = PCGrad(
            torch.optim.AdamW([active, inactive], lr=0.1, weight_decay=1.0),
            reduction="sum",
        )
        optimizer.pc_backward([active.square()])
        self.assertIsNone(inactive.grad)
        before = inactive.detach().clone()
        optimizer.step()
        torch.testing.assert_close(inactive, before)

    def test_zero_norm_projection_is_finite(self):
        parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        optimizer = PCGrad(torch.optim.SGD([parameter], lr=0.1), reduction="sum")
        zero = (parameter * 0.0).sum()
        nonzero = parameter.sum()
        optimizer.pc_backward_pinn([zero, nonzero])
        self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
