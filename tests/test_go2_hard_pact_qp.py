import importlib.util
import unittest

import torch

from legged_gym.envs.go2.go2_hard_pact.qp import (
    Go2HardPACTQP,
    HardPACTQPConfig,
    HardPACTQPInputs,
)
from rsl_rl.modules.actor_critic_go2_hard_pact import ActorCriticGo2HardPACT


def qp_inputs(batch=3, *, nominal=None):
    dtype = torch.float64
    if nominal is None:
        nominal = torch.zeros(batch, 12, dtype=dtype)
    return HardPACTQPInputs(
        mass=torch.eye(18, dtype=dtype).expand(batch, -1, -1).clone(),
        bias=torch.zeros(batch, 18, dtype=dtype),
        foot_jacobian=torch.zeros(batch, 4, 3, 18, dtype=dtype),
        foot_jdot_v=torch.zeros(batch, 4, 3, dtype=dtype),
        base_jacobian=torch.zeros(batch, 6, 18, dtype=dtype),
        predicted_grf=torch.zeros(batch, 12, dtype=dtype),
        predicted_base_wrench=torch.zeros(batch, 6, dtype=dtype),
        contact_probability=torch.zeros(batch, 4, dtype=dtype),
        nominal_torque=nominal,
        previous_torque=torch.zeros(batch, 12, dtype=dtype),
        torque_limit=torch.full((12,), 100.0, dtype=dtype),
        torque_rate_limit=torch.full((12,), 1.0e6, dtype=dtype),
        joint_position=torch.zeros(batch, 12, dtype=dtype),
        joint_velocity=torch.zeros(batch, 12, dtype=dtype),
        joint_position_lower=torch.full((12,), -10.0, dtype=dtype),
        joint_position_upper=torch.full((12,), 10.0, dtype=dtype),
        joint_velocity_limit=torch.full((12,), 100.0, dtype=dtype),
        gravity_normal_force_frame=torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
        dt=0.01,
    )


class QPTests(unittest.TestCase):
    def test_fixed_shape_convex_problem(self):
        qp = Go2HardPACTQP(HardPACTQPConfig(solver="equality"))
        h, p, g, hh, a, b = qp.build(qp_inputs(2))
        self.assertEqual(h.shape, (2, 54, 54))
        self.assertEqual(a.shape, (2, 30, 54))
        self.assertEqual(g.shape, (2, 92, 54))
        self.assertTrue((torch.diagonal(h, dim1=-2, dim2=-1) > 0).all())
        self.assertEqual(p.shape, (2, 54))
        self.assertEqual(hh.shape, (2, 92))
        self.assertEqual(b.shape, (2, 30))

    def test_equality_reference_is_differentiable(self):
        nominal = torch.full((2, 12), 0.1, dtype=torch.float64, requires_grad=True)
        qp = Go2HardPACTQP(HardPACTQPConfig(solver="equality"))
        result = qp.solve(qp_inputs(2, nominal=nominal))
        self.assertTrue((result.fallback == 0).all())
        result.safe_torque.sum().backward()
        self.assertIsNotNone(nominal.grad)
        self.assertTrue(torch.isfinite(nominal.grad).all())

    @unittest.skipUnless(importlib.util.find_spec("qpth"), "qpth is not installed")
    def test_qpth_float64_reference_and_gradient(self):
        nominal = torch.full((2, 12), 0.1, dtype=torch.float64, requires_grad=True)
        inputs = qp_inputs(2, nominal=nominal)
        inputs.predicted_grf[:, 2::3] = 10.0
        result = Go2HardPACTQP(HardPACTQPConfig(
            solver="qpth", solver_dtype=torch.float64
        )).solve(inputs)
        self.assertTrue((result.fallback == 0).all())
        self.assertLess(result.equality_residual.max().item(), 1.0e-6)
        result.safe_torque.square().sum().backward()
        self.assertTrue(torch.isfinite(nominal.grad).all())

    def test_float32_chunking_matches_unchunked(self):
        inputs = qp_inputs(5)
        chunked = Go2HardPACTQP(HardPACTQPConfig(
            solver="equality", solver_dtype=torch.float32, gpu_chunk_size=2
        )).solve(inputs)
        whole = Go2HardPACTQP(HardPACTQPConfig(
            solver="equality", solver_dtype=torch.float32, gpu_chunk_size=8
        )).solve(inputs)
        torch.testing.assert_close(chunked.safe_torque, whole.safe_torque)
        self.assertEqual(chunked.safe_torque.shape, (5, 12))

    def test_deterministic_last_fallback_is_rate_and_actuator_projection(self):
        nominal = torch.full((1, 12), 50.0, dtype=torch.float64)
        inputs = qp_inputs(1, nominal=nominal)
        inputs.previous_torque.fill_(1.0)
        inputs.torque_limit.fill_(3.0)
        inputs.torque_rate_limit.fill_(100.0)
        qp = Go2HardPACTQP(HardPACTQPConfig(solver="equality"))
        qp._solve_once = lambda built: (_ for _ in ()).throw(RuntimeError("forced"))
        result = qp.solve(inputs)
        self.assertTrue((result.fallback == 2).all())
        torch.testing.assert_close(result.safe_torque, torch.full((1, 12), 2.0, dtype=torch.float64))

    def test_realized_ground_friction_changes_friction_constraints(self):
        qp = Go2HardPACTQP(HardPACTQPConfig(solver="equality"))
        low = qp_inputs(1)
        high = qp_inputs(1)
        low.friction = torch.tensor([[0.2]], dtype=torch.float64)
        high.friction = torch.tensor([[0.9]], dtype=torch.float64)
        low_g = qp.build(low)[2]
        high_g = qp.build(high)[2]
        # Torque/joint-limit rows are unchanged; the final 20 rows are the
        # four realized-friction pyramids.
        self.assertGreater((low_g[:, -20:] - high_g[:, -20:]).abs().max().item(), 0.0)

    def test_qp_gradient_reaches_both_coupled_action_heads(self):
        torch.manual_seed(13)
        policy = ActorCriticGo2HardPACT(
            actor_layers=(16,), critic_layers=(16,), encoder_layers=(16,),
            physics_head_layers=(16,),
        ).double()
        observation = torch.randn(2, 57, dtype=torch.float64)
        history = torch.randn(2, 57 * 20, dtype=torch.float64)
        policy.update_distribution(observation, history)
        source = policy.action_mean
        delayed = torch.full_like(source, 0.1) + source - source.detach()
        nominal = delayed[:, :12] + delayed[:, 12:]
        result = Go2HardPACTQP(HardPACTQPConfig(solver="equality")).solve(
            qp_inputs(2, nominal=nominal)
        )
        result.safe_torque.sum().backward()
        for head in (policy.position_head, policy.feedforward_head):
            self.assertIsNotNone(head.weight.grad)
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(head.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
