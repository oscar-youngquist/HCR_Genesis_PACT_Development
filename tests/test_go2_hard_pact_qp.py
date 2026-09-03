import unittest
from unittest import mock
from types import SimpleNamespace

import torch
from qpth.qp import QPFunction

from rsl_rl.algorithms.hard_pact_qp import (
    HardPACTDifferentiableQP,
    HardPACTQPConfig,
    _row_scale,
    balanced_substep_indices,
    projection_loss,
)
from rsl_rl.algorithms.hard_pact_qp_backends import (
    QPBackendUnavailable, backend_capability, moreau_conic_mapping,
)
from rsl_rl.modules.actor_critic_hard_pact import ActorCritic_HardPACT
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from rsl_rl.runners.pact_runner import OnPolicyRunnerPACT


def qp_data(batch=2, dtype=torch.float64):
    zeros = lambda *shape: torch.zeros(*shape, dtype=dtype)
    data = {
        "mass_matrix": torch.eye(18, dtype=dtype).expand(batch, -1, -1).clone(),
        "bias": zeros(batch, 18),
        "foot_jacobians": zeros(batch, 4, 3, 18),
        "base_jacobian": zeros(batch, 6, 18),
        "foot_acceleration_bias": zeros(batch, 4, 3),
        "tau_nom": zeros(batch, 12),
        "force_pred_world": zeros(batch, 4, 3),
        "wrench_pred_world": zeros(batch, 6),
        "contact_probability": zeros(batch, 4),
        "previous_torque": zeros(batch, 12),
        "joint_position": zeros(batch, 12),
        "joint_velocity": zeros(batch, 12),
        "dt": torch.full((batch, 1), 0.02, dtype=dtype),
    }
    data["force_pred_world"][:, :, 2] = 20.0
    return data


def make_qp(**overrides):
    config = HardPACTQPConfig(max_iter=50, not_improved_limit=10, **overrides)
    return HardPACTDifferentiableQP(
        config,
        torch.full((12,), 23.5),
        torch.full((12,), -2.0),
        torch.full((12,), 2.0),
        torch.full((12,), 30.0),
    )


class HardPACTQPTests(unittest.TestCase):
    def test_solver_registration_validation_and_capabilities(self):
        for name in ("qpth", "cupiqp", "moreau"):
            qp = make_qp(qp_solver=name)
            self.assertEqual(qp.solver_for_mode(False), name)
            self.assertEqual(qp.solver_for_mode(True), name)
        with self.assertRaisesRegex(ValueError, "allow_solver_mismatch"):
            make_qp(qp_solver="qpth", ppo_qp_solver="cupiqp")
        mixed = make_qp(
            qp_solver="qpth", ppo_qp_solver="cupiqp",
            allow_solver_mismatch=True,
        )
        self.assertEqual(mixed.solver_for_mode(False), "qpth")
        self.assertEqual(mixed.solver_for_mode(True), "cupiqp")
        self.assertTrue(backend_capability("qpth").available)

    def test_moreau_zero_and_nonnegative_cone_sign_mapping(self):
        A = torch.tensor([[[1.0, 2.0]]])
        b = torch.tensor([[3.0]])
        G = torch.tensor([[[1.0, 0.0], [0.0, -1.0]]])
        h = torch.tensor([[4.0, 5.0]])
        C, rhs, zero_count, nonnegative_count = moreau_conic_mapping(A, b, G, h)
        self.assertEqual((zero_count, nonnegative_count), (1, 2))
        x = torch.tensor([[1.0, 1.0]])
        slack = rhs - torch.einsum("bij,bj->bi", C, x)
        self.assertEqual(slack[0, 0].item(), 0.0)
        self.assertTrue((slack[0, 1:] >= 0).all())

    def test_unavailable_backend_raises_without_analytic_fallback(self):
        for name in ("cupiqp", "moreau"):
            if backend_capability(name).available:
                continue
            with self.subTest(name=name), self.assertRaises(QPBackendUnavailable):
                make_qp(qp_solver=name).solve(**qp_data(1))

    def test_dtype_aware_defaults_and_live_device_resolution(self):
        qp = make_qp(solver_dtype="auto")
        self.assertEqual(qp._solve_dtype(torch.zeros(1)), torch.float64)
        self.assertEqual(qp._eps(torch.float32), 1e-5)
        self.assertEqual(qp._eps(torch.float64), 1e-9)
        self.assertEqual(qp._normalized_tolerance(torch.float32), 1e-3)
        self.assertEqual(qp._normalized_tolerance(torch.float64), 1e-6)
        self.assertEqual(qp._chunk_size(False), 512)
        self.assertEqual(qp._chunk_size(True), 128)
        if torch.cuda.is_available():
            self.assertEqual(qp._solve_dtype(torch.zeros(1, device="cuda")), torch.float32)

    def test_constant_templates_are_reused_without_detaching_learned_entries(self):
        qp = make_qp()
        data = qp_data(1)
        first = qp._constants(data["tau_nom"])
        second = qp._constants(data["tau_nom"])
        self.assertEqual(len(qp._constant_cache), 1)
        for actual, expected in zip(first, second):
            self.assertIs(actual, expected)
            self.assertEqual(actual.data_ptr(), expected.data_ptr())

        data["base_jacobian"][:, :, :6] = torch.eye(6, dtype=torch.float64)
        data["foot_jacobians"][:, 0, :, :3] = torch.eye(3, dtype=torch.float64)
        data["tau_nom"].fill_(0.5)
        data["wrench_pred_world"].fill_(0.25)
        for name in (
            "tau_nom", "force_pred_world", "wrench_pred_world",
            "contact_probability",
        ):
            data[name].requires_grad_(True)
        built = qp._build(data)
        # Cached tensors supply only immutable structure. Learned references
        # are composed functionally into fresh matrix entries and must retain
        # their graph despite template reuse.
        objective = (
            built.p.square().sum() + built.b.square().sum()
            + built.G[:, -24:].square().sum()
        )
        objective.backward()
        for name in (
            "tau_nom", "force_pred_world", "wrench_pred_world",
            "contact_probability",
        ):
            self.assertIsNotNone(data[name].grad)
            self.assertTrue(torch.isfinite(data[name].grad).all())
            self.assertGreater(data[name].grad.abs().sum().item(), 0.0)

    def test_rhs_aware_row_scaling_preserves_constraints(self):
        matrix = torch.tensor([[[3.0, 4.0], [0.0, 0.5], [2.0, 0.0]]])
        rhs = torch.tensor([[2.0, 7.0, 0.25]])
        scaled_matrix, scaled_rhs, scale = _row_scale(matrix, rhs)
        torch.testing.assert_close(scale, torch.tensor([[5.0, 7.0, 2.0]]))
        point = torch.tensor([[1.5, -0.25]])
        physical = torch.einsum("bij,bj->bi", matrix, point) - rhs
        normalized = torch.einsum("bij,bj->bi", scaled_matrix, point) - scaled_rhs
        torch.testing.assert_close(normalized, physical / scale)

    def test_backend_position_integration_coefficients(self):
        data = qp_data(1)
        data["joint_position"].fill_(1.0)
        data["joint_velocity"].zero_()
        data["dt"].fill_(0.1)
        for coefficient, expected_bound in ((0.5, 200.0), (1.0, 100.0)):
            qp = make_qp(position_integration_coefficient=coefficient)
            built = qp._build(data, relaxed_contact=False)
            torch.testing.assert_close(
                built.qdd_upper,
                torch.full_like(built.qdd_upper, expected_bound),
            )

    def test_invalid_intersections_bypass_qpth_with_distinct_reasons(self):
        torque_bad = qp_data(1)
        torque_bad["previous_torque"].fill_(1000.0)
        with mock.patch("rsl_rl.algorithms.hard_pact_qp.QPFunction") as function:
            result = make_qp().solve(**torque_bad)
        function.assert_not_called()
        self.assertEqual(result.stage.item(), 2)
        self.assertTrue(result.diagnostics["failure/empty_torque_intersection"].item())
        self.assertTrue(torch.isfinite(result.tau_safe).all())
        self.assertTrue((result.tau_safe.abs() <= 23.5).all())

        qdd_bad = qp_data(1)
        qdd_bad["joint_position"].fill_(3.0)
        qdd_bad["joint_velocity"].fill_(30.0)
        with mock.patch("rsl_rl.algorithms.hard_pact_qp.QPFunction") as function:
            result = make_qp().solve(**qdd_bad)
        function.assert_not_called()
        self.assertTrue(result.diagnostics["failure/empty_qdd_intersection"].item())

        nonfinite = qp_data(1)
        nonfinite["mass_matrix"][0, 0, 0] = float("nan")
        with mock.patch("rsl_rl.algorithms.hard_pact_qp.QPFunction") as function:
            result = make_qp().solve(**nonfinite)
        function.assert_not_called()
        self.assertTrue(result.diagnostics["failure/nonfinite_input"].item())
        self.assertTrue(torch.isfinite(result.tau_safe).all())

    def test_contact_floor_relaxed_shape_zero_slack_and_contact_gradient(self):
        data = qp_data(1)
        data["foot_jacobians"][0, :, :, :3] = torch.eye(
            3, dtype=torch.float64
        ).unsqueeze(0).expand(4, -1, -1)
        contact = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
        data["contact_probability"] = contact
        full = make_qp()._build(data, relaxed_contact=False)
        relaxed = make_qp()._build(data, relaxed_contact=True)
        self.assertEqual(full.G.shape[1], 104)
        self.assertEqual(relaxed.G.shape[1], 68)
        self.assertEqual(relaxed.A.shape[1], 18)
        # Last 24 physical rows are +/- contact acceleration. Even at c=0,
        # the qdd block is nonzero because c_eff=c_min.
        self.assertGreater(full.physical_G[:, -24:, :18].abs().sum().item(), 0.0)
        full.physical_G[:, -24:, :18].abs().sum().backward()
        self.assertTrue(torch.isfinite(contact.grad).all())
        self.assertGreater(contact.grad.abs().sum().item(), 0.0)
        solved = make_qp().solve(**qp_data(1))
        if solved.stage.item() == 1:
            torch.testing.assert_close(
                solved.contact_slack, torch.zeros_like(solved.contact_slack),
                atol=2e-6, rtol=0.0,
            )

    def test_physical_and_normalized_certification_are_both_reported(self):
        with mock.patch("torch.linalg.svdvals") as svd, \
                mock.patch("torch.linalg.cholesky_ex") as cholesky, \
                mock.patch("torch.linalg.pinv") as pinv:
            result = make_qp(diagnostics_level="physical").solve(**qp_data(2))
        svd.assert_not_called()
        cholesky.assert_not_called()
        pinv.assert_not_called()
        for name in (
            "selected/equality_max", "selected/inequality_max",
            "selected/physical_equality_max",
            "selected/physical_inequality_max",
            "selected/physical_base_linear_equality_max",
            "selected/physical_base_angular_equality_max",
            "selected/physical_joint_equality_max",
        ):
            self.assertIn(name, result.diagnostics)
            self.assertTrue(torch.isfinite(result.diagnostics[name]).all())
        self.assertLess(result.diagnostics["selected/equality_max"].max(), 1e-6)
        self.assertLess(result.diagnostics["selected/physical_equality_max"].max(), 1e-5)

    def test_periodic_debug_audit_is_sampled_for_large_batches(self):
        qp = make_qp(
            diagnostics_level="full", full_audit_period=1,
            full_audit_sample_size=3, chunk_size=2,
        )
        original_svd = torch.linalg.svdvals
        original_cholesky = torch.linalg.cholesky_ex
        with mock.patch("torch.linalg.svdvals", wraps=original_svd) as svd, \
                mock.patch(
                    "torch.linalg.cholesky_ex", wraps=original_cholesky
                ) as cholesky:
            result = qp.solve(**qp_data(5))
        self.assertTrue(torch.isfinite(result.tau_safe).all())
        self.assertTrue(svd.called)
        self.assertTrue(cholesky.called)
        # The audit budget spans chunks: exactly three rows receive expensive
        # work even though the five-row solve is split into 2/2/1.
        self.assertEqual(svd.call_args_list[0].args[0].shape[0], 2)
        self.assertEqual(cholesky.call_args_list[0].args[0].shape[0], 2)
        self.assertEqual(sum(call.args[0].shape[0] for call in svd.call_args_list), 3)
        self.assertEqual(
            sum(call.args[0].shape[0] for call in cholesky.call_args_list), 3
        )

    def test_diagnostics_levels_preserve_primal_stages_and_gradients(self):
        records = {}
        for level in ("minimal", "physical", "full"):
            data = qp_data(2)
            data["tau_nom"].fill_(0.5).requires_grad_()
            data["force_pred_world"].requires_grad_()
            data["wrench_pred_world"].requires_grad_()
            data["contact_probability"].fill_(0.4).requires_grad_()
            result = make_qp(
                diagnostics_level=level, full_audit_period=1,
                full_audit_sample_size=1,
            ).solve(**data)
            loss = (
                result.qdd.square().mean()
                + result.force_world.square().mean()
                + result.tau_safe.square().mean()
                + result.contact_slack.square().mean()
            )
            loss.backward()
            records[level] = (
                result.qdd.detach(), result.force_world.detach(),
                result.tau_safe.detach(), result.contact_slack.detach(),
                result.stage.detach(), result.differentiated_mask.detach(),
                tuple(data[name].grad.detach().clone() for name in (
                    "tau_nom", "force_pred_world", "wrench_pred_world",
                    "contact_probability",
                )),
            )
        reference = records["minimal"]
        for level in ("physical", "full"):
            for actual, expected in zip(records[level][0:6], reference[0:6]):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            for actual, expected in zip(records[level][6], reference[6]):
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_minimal_skips_every_optional_diagnostic_path(self):
        qp = make_qp(diagnostics_level="minimal")
        with mock.patch.object(
            qp, "_physical_summary", side_effect=AssertionError
        ) as physical, mock.patch("torch.linalg.svdvals") as svd, \
                mock.patch("torch.linalg.cholesky_ex") as cholesky, \
                mock.patch("torch.linalg.pinv") as pinv, \
                mock.patch("torch.cuda.synchronize") as synchronize:
            result = qp.solve(**qp_data(3))
        physical.assert_not_called()
        svd.assert_not_called()
        cholesky.assert_not_called()
        pinv.assert_not_called()
        synchronize.assert_not_called()
        self.assertTrue(all(
            key.startswith("qp/minimal/") for key in result.metrics
        ))
        self.assertFalse(any("physical" in key for key in result.diagnostics))

    def test_physical_group_margins_match_direct_calculation(self):
        data = qp_data(2)
        data["tau_nom"].fill_(0.5)
        data["previous_torque"].fill_(0.25)
        with mock.patch("torch.cuda.synchronize") as synchronize:
            result = make_qp(diagnostics_level="physical").solve(**data)
        synchronize.assert_not_called()
        metrics = result.metrics
        torque_margin = 23.5 - result.tau_safe.abs()
        rate_margin = 1000.0 * data["dt"] - (
            result.tau_safe - data["previous_torque"]
        ).abs()
        torch.testing.assert_close(
            metrics["qp/physical/margin_min/torque"], torque_margin.min()
        )
        torch.testing.assert_close(
            metrics["qp/physical/margin_min/torque_rate"], rate_margin.min()
        )
        for group in (
            "torque", "torque_rate", "joint_position", "joint_velocity",
            "unilateral_force", "friction_pyramid", "slack",
            "contact_acceleration",
        ):
            margin = metrics[f"qp/physical/margin_min/{group}"]
            violation = metrics[f"qp/physical/violation_max/{group}"]
            self.assertTrue(torch.isfinite(margin))
            self.assertTrue(torch.isfinite(violation))
            torch.testing.assert_close(violation, (-margin).clamp_min(0.0))

    def test_full_audit_metrics_and_period_are_exact(self):
        qp = make_qp(
            diagnostics_level="full", full_audit_period=2,
            full_audit_sample_size=2,
        )
        first = qp.solve(**qp_data(5))
        second = qp.solve(**qp_data(5))
        self.assertEqual(first.metrics["qp/full/audit_ran"].item(), 0.0)
        self.assertEqual(second.metrics["qp/full/audit_ran"].item(), 1.0)
        self.assertNotIn("qp/full/q_eigen_min_mean", first.metrics)
        self.assertTrue(torch.isfinite(
            second.metrics["qp/full/q_eigen_min_mean"]
        ))
        self.assertEqual(
            second.metrics["qp/full/a_rank_min"].item(), 18.0
        )
        audited = torch.isfinite(
            second.diagnostics["full/audit_q_eigen_min"]
        ).sum()
        self.assertEqual(audited.item(), 2)

        differentiable_first_data = qp_data(1)
        differentiable_first_data["tau_nom"].requires_grad_()
        differentiable_first = qp.solve(**differentiable_first_data)
        differentiable_second_data = qp_data(1)
        differentiable_second_data["tau_nom"].requires_grad_()
        differentiable_second = qp.solve(**differentiable_second_data)
        self.assertEqual(
            differentiable_first.metrics["qp/full/audit_ran"].item(), 0.0
        )
        self.assertEqual(
            differentiable_second.metrics["qp/full/audit_ran"].item(), 1.0
        )

        disabled = make_qp(
            diagnostics_level="full", full_audit_period=0
        ).solve(**qp_data(2))
        self.assertEqual(disabled.metrics["qp/full/audit_ran"].item(), 0.0)
        self.assertFalse(any("q_eigen" in key for key in disabled.metrics))

        infeasible_data = qp_data(2)
        infeasible_data["previous_torque"].fill_(1000.0)
        infeasible = make_qp(
            diagnostics_level="full", full_audit_period=1
        ).solve(**infeasible_data)
        self.assertTrue((infeasible.stage == 2).all())
        self.assertEqual(
            infeasible.metrics[
                "qp/minimal/failure/empty_torque_intersection"
            ].item(), 1.0,
        )
        self.assertTrue(all(
            torch.isfinite(value) for value in infeasible.metrics.values()
        ))

    def test_runner_logs_only_aggregated_qp_prefixes(self):
        class Writer:
            def __init__(self): self.calls = []
            def add_scalar(self, name, value, iteration):
                self.calls.append((name, value, iteration))

        runner = OnPolicyRunnerPACT.__new__(OnPolicyRunnerPACT)
        runner.writer = Writer()
        runner.alg = SimpleNamespace(last_qp_metrics={
            "qp/minimal/full_fraction": torch.tensor(1.0),
            "qp/physical/force/max": torch.tensor(2.0),
            "qp/full/q_condition_mean": torch.tensor(3.0),
            "unscoped/internal": torch.tensor(4.0),
        })
        runner._log_qp_metrics(7)
        self.assertEqual(
            [name for name, _, _ in runner.writer.calls],
            [
                "qp/minimal/full_fraction",
                "qp/physical/force/max",
                "qp/full/q_condition_mean",
            ],
        )
        self.assertTrue(all(call[2] == 7 for call in runner.writer.calls))

    def test_successful_qpth_torque_is_post_projected_with_gradient(self):
        qp = make_qp(torque_rate_limit_nm_s=100.0)
        data = qp_data(1)
        data["tau_nom"].fill_(3.0).requires_grad_()
        data["previous_torque"].fill_(1.0)
        data["dt"].fill_(0.01)  # exact interval [0,2]
        template = {
            "equality_max": torch.zeros(1, dtype=torch.float64),
            "inequality_max": torch.zeros(1, dtype=torch.float64),
            "physical_equality_max": torch.zeros(1, dtype=torch.float64),
            "physical_inequality_max": torch.zeros(1, dtype=torch.float64),
            "physical_base_linear_equality_max": torch.zeros(1, dtype=torch.float64),
            "physical_base_angular_equality_max": torch.zeros(1, dtype=torch.float64),
            "physical_joint_equality_max": torch.zeros(1, dtype=torch.float64),
            "stationarity_max": torch.full((1,), float("nan"), dtype=torch.float64),
            "stationarity_raw_max": torch.full((1,), float("nan"), dtype=torch.float64),
            "complementarity_max": torch.full((1,), float("nan"), dtype=torch.float64),
            "kkt_valid": torch.ones(1, dtype=torch.bool),
            "output_finite": torch.ones(1, dtype=torch.bool),
            "input_finite": torch.ones(1, dtype=torch.bool),
            "equality_rank": torch.ones(1, dtype=torch.bool),
            "q_spd": torch.ones(1, dtype=torch.bool),
            "solver_exception": torch.zeros(1, dtype=torch.bool),
        }
        oversized = torch.zeros(1, 54, dtype=torch.float64)
        oversized[:, 30:42] = 2.01 + 0.1 * data["tau_nom"]
        with mock.patch.object(
            qp, "_solve_stage", return_value=(
                oversized, torch.ones(1, dtype=torch.bool), template,
            )
        ):
            result = qp.solve(**data)
        torch.testing.assert_close(result.tau_safe, torch.full_like(result.tau_safe, 2.0))
        self.assertGreater(result.diagnostics["pre_clamp_torque_violation_max"].item(), 0.0)
        result.tau_safe.sum().backward()
        self.assertTrue(torch.equal(data["tau_nom"].grad, torch.zeros_like(data["tau_nom"])))

    def test_float32_float64_primal_parity(self):
        source = qp_data(2, dtype=torch.float32)
        # A zero force reference avoids testing qpth's deliberately difficult
        # central-path degeneracy at the unilateral cone apex; nonzero torque
        # still exercises coupled qdd/tau primal coordinates.
        source["force_pred_world"].zero_()
        source["force_pred_world"][:, :, 2] = -20.0
        source["tau_nom"].fill_(0.5)
        single = make_qp(solver_dtype="float32").solve(**source)
        double = make_qp(solver_dtype="float64").solve(**source)
        torch.testing.assert_close(single.tau_safe, double.tau_safe, atol=3e-3, rtol=3e-3)
        torch.testing.assert_close(single.qdd, double.qdd, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(single.force_world, double.force_world, atol=0.15, rtol=1e-3)
        torch.testing.assert_close(
            single.contact_slack, double.contact_slack, atol=0.20, rtol=1e-3
        )

    def test_float64_well_conditioned_parity_with_previous_qp(self):
        # Frozen with the pre-refactor float64 QP on this branch. The force
        # reference is zero so the unilateral cone remains well conditioned;
        # the nonzero torque exercises the coupled dynamics equality.
        data = qp_data(1)
        data["force_pred_world"].zero_()
        data["tau_nom"].fill_(0.5)
        result = make_qp(solver_dtype="float64").solve(**data)
        self.assertEqual(result.stage.item(), 0)
        torch.testing.assert_close(
            result.tau_safe,
            torch.full_like(result.tau_safe, 0.48655974843272354),
            atol=1e-10, rtol=1e-10,
        )
        torch.testing.assert_close(
            result.qdd[:, 6:],
            torch.full_like(result.qdd[:, 6:], 0.4865597484327235),
            atol=1e-10, rtol=1e-10,
        )
        self.assertLessEqual(result.force_world.abs().max().item(), 1.1e-3)
        self.assertLessEqual(result.contact_slack.abs().max().item(), 5e-5)

    def test_substep_pd_refresh_rate_propagation_and_interval_aggregation(self):
        batch = 4

        class Robot:
            def __init__(self):
                self.q = torch.zeros(batch, 12)
                self.v = torch.zeros(batch, 12)
            def get_dofs_position(self, _): return self.q
            def get_dofs_velocity(self, _): return self.v
            def get_pos(self): return torch.zeros(batch, 3)
            def get_vel(self): return torch.zeros(batch, 3)
            def get_ang(self): return torch.zeros(batch, 3)

        class Heads:
            def predict_grf(self, _z, _e, torque): return torque

        class Dynamics:
            def build_context(self, *_args, **_kwargs):
                return SimpleNamespace(
                    mass_matrix=torch.eye(18).expand(batch, -1, -1),
                    bias=torch.zeros(batch, 18),
                    foot_jacobians=torch.zeros(batch, 4, 3, 18),
                    base_jacobian=torch.zeros(batch, 6, 18),
                    foot_acceleration_bias=torch.zeros(batch, 4, 3),
                )

        class QP:
            def __init__(self): self.previous = []; self.nominal = []
            def solve(self, **values):
                self.previous.append(values["previous_torque"].clone())
                self.nominal.append(values["tau_nom"].clone())
                safe = values["tau_nom"].clamp(-2.0, 2.0)
                zeros = torch.zeros(batch)
                diagnostics = {
                    "selected/equality_max": zeros,
                    "selected/inequality_max": zeros,
                    "selected/stationarity_max": zeros,
                    "selected/complementarity_max": zeros,
                }
                return SimpleNamespace(
                    qdd=torch.zeros(batch, 18), force_world=values["force_pred_world"],
                    tau_safe=safe, contact_slack=torch.zeros(batch, 4, 3),
                    stage=torch.zeros(batch, dtype=torch.long),
                    differentiated_mask=torch.ones(batch, dtype=torch.bool),
                    diagnostics=diagnostics,
                )

        task = Go2HardPACT.__new__(Go2HardPACT)
        task.num_envs, task.device = batch, torch.device("cpu")
        task.cfg = SimpleNamespace(
            sim=SimpleNamespace(dt=0.002), control=SimpleNamespace(decimation=2)
        )
        task.obs_scales = SimpleNamespace(grf=1.0, base_wrench=1.0)
        robot = Robot()
        task.simulator = SimpleNamespace(
            _robot=robot, _dof_indices=torch.arange(12),
            _joint_armature=torch.zeros(batch, 12),
            _joint_friction=torch.zeros(batch, 12),
            _joint_stiffness=torch.zeros(batch, 12),
            _joint_damping=torch.zeros(batch, 12), _torques=torch.zeros(batch, 12),
        )
        task._hard_pact_actor_critic = SimpleNamespace(
            physics_estimator=Heads()
        )
        task._hard_pact_bard_dynamics = Dynamics()
        qp = QP()
        task._hard_pact_rollout_qp = qp
        task._hard_pact_q_d = torch.ones(batch, 12)
        task._hard_pact_tau_ff = torch.zeros(batch, 12)
        task._hard_pact_previous_substep_torque = torch.zeros(batch, 12)
        task._hard_pact_policy_latent = torch.zeros(batch, 16)
        task._hard_pact_policy_explicit = torch.zeros(batch, 11)
        task._hard_pact_wrench_yaw_scaled = torch.zeros(batch, 6)
        task._realized_added_mass = torch.zeros(batch, 1)
        task._realized_com_shift_body = torch.zeros(batch, 3)
        task._get_pinn_feedback = lambda desired, q, qdot: 4 * (desired - q) - qdot
        task._begin_qp_interval()
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(batch, -1)
        mass_wrench = torch.zeros(batch, 6)
        task._solve_hard_pact_rollout_qp_substep(quat, mass_wrench)
        first_safe = task.simulator._torques.clone()
        robot.q.fill_(0.5)
        robot.v.fill_(0.25)
        task._solve_hard_pact_rollout_qp_substep(quat, mass_wrench)
        self.assertFalse(torch.equal(qp.nominal[0], qp.nominal[1]))
        self.assertTrue(torch.equal(qp.previous[1], first_safe))
        expected_average = (first_safe + task.simulator._torques) / 2
        self.assertTrue(torch.allclose(task._qp_interval_safe_sum / 2, expected_average))
        self.assertTrue(torch.equal(task._qp_interval_stage_counts[:, 0], torch.full((batch,), 2.0)))

    def test_balanced_stratified_substep_sampling(self):
        generator = torch.Generator().manual_seed(17)
        indices = balanced_substep_indices(
            103, 4, torch.device("cpu"), generator=generator
        )
        counts = torch.bincount(indices.long(), minlength=4)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        self.assertTrue(torch.all((indices >= 0) & (indices < 4)))
        self.assertEqual(indices.dtype, torch.int16)

    def test_decimation_one_sampling_is_exact(self):
        indices = balanced_substep_indices(31, 1, torch.device("cpu"))
        self.assertTrue(torch.equal(indices, torch.zeros_like(indices)))

    def test_rollout_qp_runs_under_inference_mode(self):
        qp = make_qp(diagnostics_level="full")
        with torch.inference_mode():
            result = qp.solve(**qp_data(2), differentiable=False)
        self.assertEqual(result.tau_safe.shape, (2, 12))
        self.assertTrue(torch.isfinite(result.tau_safe).all())
        # The same solver/cache must subsequently support the retained PPO
        # KKT graph rather than reusing inference tensors in autograd.
        training = qp_data(1)
        training["tau_nom"].fill_(0.5).requires_grad_()
        learned = qp.solve(**training, differentiable=True)
        learned.tau_safe.sum().backward()
        self.assertIsNotNone(training["tau_nom"].grad)

    def test_qpth_warning_is_quiet_by_default_and_debug_verbosity_is_preserved(self):
        # qpth prints its inaccurate-iterate banner even at verbose=0. The
        # adapter uses -1 for the public quiet setting because HardPACT's own
        # residual certification and fallback remain authoritative.
        for configured, forwarded in ((0, -1), (2, 2)):
            qp = make_qp(verbose=configured)
            with mock.patch(
                "rsl_rl.algorithms.hard_pact_qp.QPFunction"
            ) as qp_function:
                qp_function.return_value.return_value = torch.zeros(
                    1, 54, dtype=torch.float64
                )
                qp._solve_stage(qp_data(1), relaxed_contact=False)
            self.assertEqual(qp_function.call_args.kwargs["verbose"], forwarded)

    def test_qpth_warm_start_cold_parity_retry_and_reset(self):
        source = qp_data(3)
        source["force_pred_world"].zero_()
        source["tau_nom"].fill_(0.5)
        cold = make_qp(qpth_warm_start=False).solve(
            differentiable=False, **source
        )
        warm_qp = make_qp(qpth_warm_start=True)
        first = warm_qp.solve(differentiable=False, **source)
        second = warm_qp.solve(differentiable=False, **source)
        torch.testing.assert_close(first.tau_safe, cold.tau_safe, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(second.tau_safe, cold.tau_safe, atol=1e-10, rtol=1e-10)
        self.assertFalse(first.diagnostics["full/warm_start_hit"].any())
        self.assertTrue(second.diagnostics["full/warm_start_hit"].all())

        invalid_row = {name: value.clone() for name, value in source.items()}
        invalid_row["joint_position"][1].fill_(10.0)
        invalid_row["joint_velocity"][1].fill_(100.0)
        warm_qp.solve(differentiable=False, **invalid_row)
        key = next(iter(warm_qp._qpth_warm_states))
        _, row_valid = warm_qp._qpth_warm_states[key]
        self.assertEqual(row_valid.tolist(), [True, False, True])
        # Re-seed all rows for the incompatible-state retry below.
        warm_qp.solve(differentiable=False, **source)

        # A deliberately incompatible terminal state is rejected, affected
        # rows are cold-resolved, and its bad state is not retained.
        key = next(iter(warm_qp._qpth_warm_states))
        warm_qp._qpth_warm_states[key] = (
            tuple(torch.full((1, 1), float("nan")) for _ in range(4)),
            torch.ones(3, dtype=torch.bool),
        )
        retried = warm_qp.solve(differentiable=False, **source)
        self.assertEqual(retried.stage.tolist(), [0, 0, 0])
        self.assertTrue(
            retried.diagnostics["full/warm_residual_cold_retry"].all()
        )
        torch.testing.assert_close(retried.tau_safe, cold.tau_safe, atol=1e-10, rtol=1e-10)

        # A failed warm state is deliberately discarded.  A subsequent clean
        # solve may seed a new terminal state, which reset must then remove.
        self.assertFalse(warm_qp._qpth_warm_states)
        warm_qp.solve(differentiable=False, **source)
        self.assertTrue(warm_qp._qpth_warm_states)
        warm_qp.clear_warm_start(torch.tensor([1]))
        self.assertTrue(warm_qp._qpth_warm_states)
        _, valid = warm_qp._qpth_warm_states[key]
        self.assertEqual(valid.tolist(), [True, False, True])
        warm_qp.clear_warm_start(torch.tensor([0, 2]))
        self.assertFalse(next(iter(warm_qp._qpth_warm_states.values()))[1].any())

    def test_qpth_warm_flag_preserves_ppo_gradient_parity(self):
        gradients = []
        outputs = []
        for enabled in (False, True):
            source = qp_data(2)
            source["force_pred_world"].zero_()
            source["tau_nom"].fill_(0.5).requires_grad_()
            result = make_qp(qpth_warm_start=enabled).solve(
                differentiable=True, **source
            )
            result.tau_safe.square().mean().backward()
            outputs.append(result.tau_safe.detach())
            gradients.append(source["tau_nom"].grad.detach())
        torch.testing.assert_close(outputs[0], outputs[1], atol=0.0, rtol=0.0)
        torch.testing.assert_close(gradients[0], gradients[1], atol=0.0, rtol=0.0)

    def test_qpth_warm_rollout_remains_inference_only(self):
        solver = make_qp(qpth_warm_start=True)
        source = qp_data(2)
        source["force_pred_world"].zero_()
        with torch.inference_mode():
            first = solver.solve(differentiable=False, **source)
            second = solver.solve(differentiable=False, **source)
        self.assertFalse(first.tau_safe.requires_grad)
        self.assertFalse(second.tau_safe.requires_grad)
        self.assertTrue(second.diagnostics["full/warm_start_hit"].all())

    def test_sampled_projection_is_unbiased_without_decimation_multiplier(self):
        nominal = torch.zeros(4, 12, dtype=torch.float64)
        safe = torch.arange(1, 5, dtype=torch.float64)[:, None].expand(-1, 12)
        slack = torch.arange(4, dtype=torch.float64)[:, None].expand(-1, 12)
        limits = torch.full((12,), 10.0, dtype=torch.float64)
        valid = torch.ones(4, 1, dtype=torch.bool)
        differentiated = torch.ones_like(valid)
        full = projection_loss(
            safe, nominal, limits, valid, differentiated,
            contact_slack=slack, slack_scale=5.0,
        )
        sampled = [projection_loss(
            safe[k:k + 1], nominal[:1], limits, valid[:1], differentiated[:1],
            contact_slack=slack[k:k + 1], slack_scale=5.0,
        ) for k in range(4)]
        self.assertTrue(torch.allclose(torch.stack(sampled).mean(), full))

    def test_projection_slack_and_torque_gradients_are_finite_nonzero(self):
        nominal = torch.randn(3, 12, dtype=torch.float64, requires_grad=True)
        safe = 0.7 * nominal + 0.2
        slack = torch.full(
            (3, 4, 3), 0.5, dtype=torch.float64, requires_grad=True
        )
        loss = projection_loss(
            safe, nominal, torch.ones(12, dtype=torch.float64),
            torch.ones(3, 1, dtype=torch.bool),
            torch.ones(3, 1, dtype=torch.bool),
            contact_slack=slack, slack_scale=2.0,
        )
        grad_nominal, grad_slack = torch.autograd.grad(loss, (nominal, slack))
        self.assertTrue(torch.isfinite(grad_nominal).all())
        self.assertTrue(torch.isfinite(grad_slack).all())
        self.assertGreater(float(grad_nominal.abs().sum()), 0.0)
        self.assertGreater(float(grad_slack.abs().sum()), 0.0)
    def test_qpth_analytic_solution_and_gradient(self):
        # min x^2-4x with an inactive x>=-100 constraint has x*=2 and
        # dx*/dp=-Q^-1=-1/2. This directly checks qpth independently of Go2.
        p = torch.tensor([[-4.0]], dtype=torch.float64, requires_grad=True)
        solution = QPFunction(eps=1e-12, maxIter=50)(
            torch.tensor([[[2.0]]], dtype=torch.float64), p,
            torch.tensor([[[-1.0]]], dtype=torch.float64),
            torch.tensor([[100.0]], dtype=torch.float64),
            torch.empty(1, 0, 1, dtype=torch.float64),
            torch.empty(1, 0, dtype=torch.float64),
        )
        torch.testing.assert_close(solution, torch.tensor([[2.0]], dtype=torch.float64))
        solution.sum().backward()
        torch.testing.assert_close(p.grad, torch.tensor([[-0.5]], dtype=torch.float64))

    def test_fixed_shape_spd_rank_and_invalid_detection(self):
        qp = make_qp()
        matrices = qp._build(qp_data(2), relaxed_contact=False)
        q, p, g, h, a, b, scale = matrices
        self.assertEqual(q.shape, (2, 54, 54))
        self.assertEqual(a.shape, (2, 18, 54))
        self.assertEqual(g.shape[-1], 54)
        valid, finite, rank, spd = qp._validate_inputs(matrices)
        self.assertTrue(valid.all() and finite.all() and rank.all() and spd.all())

        bad_q = q.clone()
        bad_q[0, 0, 0] = -1.0
        bad_a = a.clone()
        bad_a[1, 0] = bad_a[1, 1]
        broken = qp._build(qp_data(2), relaxed_contact=False)
        broken.Q, broken.A = bad_q, bad_a
        valid, _, rank, spd = qp._validate_inputs(broken, audit_count=2)
        self.assertTrue(valid.all() and rank.all() and spd.all())
        audit = qp._diagnostics(
            torch.zeros(2, 54, dtype=torch.float64), broken, audit_count=2
        )
        self.assertEqual(audit["audit_q_cholesky_success"][0].item(), 0.0)
        self.assertLess(audit["audit_a_rank"][1].item(), 18.0)
        relaxed = qp._build(qp_data(1), relaxed_contact=True)
        self.assertEqual(relaxed.A.shape, (1, 18, 54))
        self.assertEqual(torch.linalg.matrix_rank(relaxed.A).item(), 18)

    def test_feasibility_residuals_and_fixed_order_outputs(self):
        result = make_qp().solve(**qp_data(2))
        self.assertTrue((result.stage == 0).all())
        self.assertEqual(result.qdd.shape, (2, 18))
        self.assertEqual(result.force_world.shape, (2, 4, 3))
        self.assertEqual(result.tau_safe.shape, (2, 12))
        self.assertEqual(result.contact_slack.shape, (2, 4, 3))
        self.assertLess(result.diagnostics["full/equality_max"].max().item(), 1e-7)
        self.assertLess(result.diagnostics["full/inequality_max"].max().item(), 1e-7)
        self.assertNotIn("full/stationarity_max", result.diagnostics)

    def test_torque_rate_position_and_velocity_limits(self):
        data = qp_data(1)
        data["tau_nom"].fill_(50.0)
        data["previous_torque"].fill_(1.0)
        data["dt"].fill_(0.01)
        qp = make_qp(torque_rate_limit_nm_s=100.0)
        result = qp.solve(**data)
        self.assertTrue((result.tau_safe <= 2.0 + 2e-4).all())
        self.assertTrue((result.tau_safe >= -23.5 - 2e-4).all())

        actuator = qp_data(1)
        actuator["tau_nom"].fill_(50.0)
        actuator_result = make_qp(
            torque_rate_limit_nm_s=1.0e6
        ).solve(**actuator)
        self.assertTrue((actuator_result.tau_safe <= 23.5 + 2e-4).all())
        self.assertTrue((actuator_result.tau_safe >= -23.5 - 2e-4).all())

        constrained = qp_data(1)
        constrained["tau_nom"].fill_(10.0)
        constrained["previous_torque"].fill_(-5.0)
        constrained["joint_position"].fill_(1.999)
        constrained["joint_velocity"].fill_(0.1)
        solved = qp.solve(**constrained)
        next_velocity = constrained["joint_velocity"] + constrained["dt"] * solved.qdd[:, 6:]
        next_position = (
            constrained["joint_position"]
            + constrained["dt"] * constrained["joint_velocity"]
            + constrained["dt"].square() * solved.qdd[:, 6:]
        )
        self.assertTrue((next_velocity <= 30.0 + 2e-4).all())
        self.assertTrue((next_position <= 2.0 + 2e-4).all())

        velocity_constrained = qp_data(1)
        velocity_constrained["tau_nom"].fill_(10.0)
        velocity_constrained["joint_velocity"].fill_(29.9)
        velocity_result = qp.solve(**velocity_constrained)
        next_velocity = (
            velocity_constrained["joint_velocity"]
            + velocity_constrained["dt"] * velocity_result.qdd[:, 6:]
        )
        self.assertTrue((next_velocity <= 30.0 + 2e-4).all())

    def test_friction_pyramid_and_contact_slacks(self):
        data = qp_data(1)
        data["force_pred_world"][0, 0] = torch.tensor([100.0, -80.0, 10.0])
        data["contact_probability"][0, 0] = 1.0
        data["foot_acceleration_bias"][0, 0] = torch.tensor([2.0, -3.0, 4.0])
        data["foot_jacobians"][0, 0, :, :3] = torch.eye(3, dtype=torch.float64)
        result = make_qp(friction_coefficient=0.5).solve(**data)
        force = result.force_world[0, 0]
        self.assertGreaterEqual(force[2].item(), -2e-4)
        self.assertLessEqual(abs(force[0].item()), 0.5 * force[2].item() + 2e-4)
        self.assertLessEqual(abs(force[1].item()), 0.5 * force[2].item() + 2e-4)
        acceleration = torch.einsum(
            "fkn,n->fk", data["foot_jacobians"][0], result.qdd[0]
        ) + data["foot_acceleration_bias"][0]
        self.assertTrue((acceleration.abs() <= result.contact_slack[0] + 1.1e-3).all())
        self.assertTrue((result.contact_slack >= -2e-4).all())

    def test_relaxed_then_projection_fallback(self):
        qp = make_qp(torque_rate_limit_nm_s=100.0)
        data = qp_data(1)
        data["tau_nom"].fill_(0.5)
        data["previous_torque"].fill_(1.0)
        data["dt"].fill_(0.01)

        original = qp._solve_stage
        calls = []
        def fail_full(values, relaxed, **kwargs):
            calls.append(relaxed)
            if not relaxed:
                solution, success, diagnostics = original(values, relaxed)
                return solution, torch.zeros_like(success), diagnostics
            return original(values, relaxed)
        with mock.patch.object(qp, "_solve_stage", side_effect=fail_full):
            relaxed = qp.solve(**data)
        self.assertEqual(calls, [False, True])
        self.assertEqual(relaxed.stage.item(), 1)

        data["tau_nom"].fill_(50.0)
        def fail_both(values, relaxed, **kwargs):
            solution, success, diagnostics = original(values, relaxed)
            return solution, torch.zeros_like(success), diagnostics
        with mock.patch.object(qp, "_solve_stage", side_effect=fail_both):
            projected = qp.solve(**data)
        self.assertEqual(projected.stage.item(), 2)
        torch.testing.assert_close(
            projected.tau_safe,
            torch.full((1, 12), 2.0, dtype=torch.float64),
        )

    def test_mixed_full_relaxed_analytic_stages_and_qpth_exception(self):
        qp = make_qp()
        original = qp._solve_stage
        calls = []
        def mixed(values, relaxed, **kwargs):
            solution, success, diagnostics = original(values, relaxed, **kwargs)
            calls.append((relaxed, solution.shape[0]))
            if not relaxed:
                success = torch.tensor([True, False, False])
            else:
                success = torch.tensor([True, False])
            return solution, success.to(solution.device), diagnostics
        with mock.patch.object(qp, "_solve_stage", side_effect=mixed):
            result = qp.solve(**qp_data(3))
        self.assertEqual(calls, [(False, 3), (True, 2)])
        self.assertEqual(result.stage.tolist(), [0, 1, 2])
        self.assertEqual(result.differentiated_mask.tolist(), [True, True, False])

        with mock.patch(
            "rsl_rl.algorithms.hard_pact_qp.QPFunction",
            side_effect=RuntimeError("factorization failed"),
        ):
            failed = make_qp().solve(**qp_data(2))
        self.assertEqual(failed.stage.tolist(), [2, 2])
        self.assertTrue(failed.diagnostics["full/solver_exception"].all())
        self.assertTrue(failed.diagnostics["relaxed/solver_exception"].all())

    def test_dtype_and_chunk_parity(self):
        data = qp_data(3, dtype=torch.float32)
        chunked = make_qp(solver_dtype="float64", chunk_size=1).solve(**data)
        whole = make_qp(solver_dtype="float64", chunk_size=8).solve(**data)
        torch.testing.assert_close(chunked.tau_safe, whole.tau_safe, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(chunked.force_world, whole.force_world, atol=2e-5, rtol=2e-5)
        single = make_qp(solver_dtype="float32", chunk_size=8).solve(**data)
        torch.testing.assert_close(single.tau_safe, whole.tau_safe, atol=3e-3, rtol=3e-3)

    def test_gradients_and_finite_difference_to_learned_references(self):
        qp = make_qp()
        data = qp_data(1)
        data["tau_nom"].fill_(0.5).requires_grad_()
        data["force_pred_world"].requires_grad_()
        result = qp.solve(**data)
        objective = result.tau_safe.sum() + 0.01 * result.force_world.sum()
        objective.backward()
        self.assertTrue(torch.isfinite(data["tau_nom"].grad).all())
        self.assertTrue(torch.isfinite(data["force_pred_world"].grad).all())
        self.assertGreater(data["tau_nom"].grad.abs().sum().item(), 0.0)
        self.assertGreater(data["force_pred_world"].grad.abs().sum().item(), 0.0)

        epsilon = 1e-4
        plus, minus = qp_data(1), qp_data(1)
        plus["tau_nom"].fill_(0.5)
        minus["tau_nom"].fill_(0.5)
        plus["tau_nom"][0, 0] += epsilon
        minus["tau_nom"][0, 0] -= epsilon
        fd = (
            qp.solve(**plus).tau_safe.sum()
            - qp.solve(**minus).tau_safe.sum()
        ) / (2 * epsilon)
        self.assertAlmostEqual(data["tau_nom"].grad[0, 0].item(), fd.item(), delta=2e-2)

        plus_force, minus_force = qp_data(1), qp_data(1)
        plus_force["tau_nom"].fill_(0.5)
        minus_force["tau_nom"].fill_(0.5)
        plus_force["force_pred_world"][0, 0, 2] += epsilon
        minus_force["force_pred_world"][0, 0, 2] -= epsilon
        fd_force = 0.01 * (
            qp.solve(**plus_force).force_world.sum()
            - qp.solve(**minus_force).force_world.sum()
        ) / (2 * epsilon)
        self.assertAlmostEqual(
            data["force_pred_world"].grad[0, 0, 2].item(),
            fd_force.item(), delta=2e-2,
        )

    def test_projection_loss_exact_mask_and_graph_zero(self):
        nominal = torch.zeros(3, 12, requires_grad=True)
        safe = nominal + torch.tensor([[1.0] * 12, [2.0] * 12, [3.0] * 12])
        loss = projection_loss(
            safe, nominal, torch.full((12,), 2.0),
            torch.tensor([True, False, True]),
            torch.tensor([True, True, False]),
        )
        self.assertAlmostEqual(loss.item(), 3.0)
        zero = projection_loss(
            safe, nominal, torch.ones(12),
            torch.zeros(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool),
        )
        zero.backward()
        self.assertTrue(torch.equal(nominal.grad, torch.zeros_like(nominal)))

    def test_qpth_graph_reaches_policy_encoder_and_force_heads(self):
        torch.manual_seed(31)
        actor = ActorCritic_HardPACT(
            num_actor_obs=57, num_critic_obs=95, num_actions=12,
            actor_layers=[32, 16], critic_layers=[32, 16],
            cenet_in_dim=57 * 20, cenet_enc_layers=[32, 16],
            cenet_explicit_layers=[16, 16],
            grf_decoder_layers=[16, 16], wrench_decoder_layers=[16, 16],
        ).double()
        observation = torch.randn(1, 57, dtype=torch.float64)
        history = torch.randn(1, 57 * 20, dtype=torch.float64)
        _, _, latent, explicit = actor.cenet_enc_forward(history)
        position, feedforward = actor.actor_forward(torch.cat(
            (observation, latent, explicit), dim=-1
        ))
        nominal = position + feedforward
        heads = actor.physics_heads(latent, explicit, nominal)

        data = qp_data(1)
        data["tau_nom"] = nominal
        data["force_pred_world"] = heads.grf_yaw_scaled.reshape(1, 4, 3)
        data["wrench_pred_world"] = heads.base_wrench_yaw_scaled
        data["contact_probability"] = explicit[:, 3:7]
        data["foot_acceleration_bias"].fill_(0.2)
        # Couple learned force/wrench references into floating-base dynamics.
        data["base_jacobian"][0, :, :6] = torch.eye(6, dtype=torch.float64)
        for foot in range(4):
            data["foot_jacobians"][0, foot, :, :3] = torch.eye(
                3, dtype=torch.float64
            )
        detached_inputs = (
            "mass_matrix", "bias", "foot_jacobians", "base_jacobian",
            "foot_acceleration_bias", "previous_torque", "joint_position",
            "joint_velocity", "dt",
        )
        for name in detached_inputs:
            data[name].requires_grad_()
        result = make_qp().solve(**data)
        loss = (
            result.tau_safe.square().mean()
            + result.force_world.square().mean()
            + result.qdd.square().mean()
            + result.contact_slack.square().mean()
        )
        loss.backward()
        groups = {
            "policy": (actor.act_trunk, actor.act_pos_out, actor.act_tau_out),
            "encoder": (actor.context_encoder,),
            "contact_estimator": (actor.explicit_estimator,),
            "grf": (actor.physics_estimator.grf_head,),
            "wrench": (actor.physics_estimator.wrench_head,),
        }
        for name, modules in groups.items():
            gradients = [
                parameter.grad for module in modules
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients, name)
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients), name)
            self.assertGreater(
                sum(value.abs().sum().item() for value in gradients), 0.0, name
            )
        for name in detached_inputs:
            self.assertIsNone(data[name].grad, name)


if __name__ == "__main__":
    unittest.main()
