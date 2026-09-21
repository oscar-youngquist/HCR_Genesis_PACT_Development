"""Correctness checks runnable in the isolated Python-3.12 Moreau env.

This module deliberately imports only the simulator-free benchmark loader, so
the captured canonical QP can be checked without pretending that Isaac Sim 5.1
is compatible with Python 3.12.
"""

import importlib.util

import pytest
import torch

from scripts.benchmark_hard_pact_qp import (
    load_qp_module_without_simulator,
    synthetic_data,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("moreau") is None,
    reason="requires the optional Moreau package",
)


def _qp(module, solver):
    return module.HardPACTDifferentiableQP(
        module.HardPACTQPConfig(
            qp_solver=solver,
            solver_dtype="float64",
            max_iter=50,
            not_improved_limit=10,
        ),
        torch.full((12,), 23.5, dtype=torch.float64),
        torch.full((12,), -2.0, dtype=torch.float64),
        torch.full((12,), 2.0, dtype=torch.float64),
        torch.full((12,), 30.0, dtype=torch.float64),
    )


def _nontrivial_data():
    values = synthetic_data(2, "cpu", torch.float64)
    values["tau_nom"] = torch.linspace(-0.6, 0.7, 12, dtype=torch.float64).repeat(2, 1)
    values["force_pred_world"][:, :, 2] = torch.tensor((4.0, 6.0, 5.0, 7.0))
    values["wrench_pred_world"][:] = torch.tensor((0.2, -0.1, 0.5, 0.05, 0.0, -0.03))
    values["contact_probability"][:] = torch.tensor((0.2, 0.5, 0.8, 0.35))
    return values


def _objective(build, result):
    physical = torch.cat(
        (result.qdd, result.force_world.reshape(2, 12), result.tau_safe,
         result.contact_slack.reshape(2, 12)), dim=-1,
    )
    scaled = physical / build.variable_scale
    return (
        0.5 * torch.einsum("bi,bij,bj->b", scaled, build.Q, scaled)
        + torch.einsum("bi,bi->b", build.p, scaled)
    )


def test_moreau_cpu_canonical_forward_objective_residual_and_gradient_parity():
    module = load_qp_module_without_simulator()
    reference_data = _nontrivial_data()
    candidate_data = {name: value.detach().clone() for name, value in reference_data.items()}
    learned = ("tau_nom", "force_pred_world", "wrench_pred_world", "contact_probability")
    for name in learned:
        reference_data[name].requires_grad_(True)
        candidate_data[name].requires_grad_(True)

    qpth_qp = _qp(module, "qpth")
    moreau_qp = _qp(module, "moreau")
    reference = qpth_qp.solve(differentiable=True, **reference_data)
    candidate = moreau_qp.solve(differentiable=True, **candidate_data)
    assert torch.equal(candidate.stage, reference.stage)
    assert (candidate.stage == 0).all()

    # Both solvers operate in float64 here. These are fixed solver-specific
    # tolerances, not an adaptive escape hatch for a failed solve.
    for actual, expected in (
        (candidate.qdd, reference.qdd),
        (candidate.force_world, reference.force_world),
        (candidate.tau_safe, reference.tau_safe),
        (candidate.contact_slack, reference.contact_slack),
    ):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for key in ("selected/equality_max", "selected/inequality_max"):
        torch.testing.assert_close(
            candidate.diagnostics[key], reference.diagnostics[key],
            atol=2e-6, rtol=2e-5,
        )

    reference_build = qpth_qp._build(reference_data)
    candidate_build = moreau_qp._build(candidate_data)
    reference_objective = _objective(reference_build, reference)
    candidate_objective = _objective(candidate_build, candidate)
    torch.testing.assert_close(
        candidate_objective, reference_objective, atol=2e-5, rtol=2e-5
    )

    reference_objective.mean().backward()
    candidate_objective.mean().backward()
    for name in learned:
        assert candidate_data[name].grad is not None
        assert torch.isfinite(candidate_data[name].grad).all()
        torch.testing.assert_close(
            candidate_data[name].grad, reference_data[name].grad,
            atol=3e-4, rtol=3e-3,
        )


def test_moreau_cuda_without_license_is_an_explicit_capability_failure():
    module = load_qp_module_without_simulator()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    capability = module.backend_capability(
        "moreau", device="cuda:0", dtype=torch.float64
    )
    # A provisioned CI/user machine may possess a real license; in that case
    # this test has nothing to assert about the missing-key path.
    if capability.available:
        pytest.skip("Moreau CUDA license is configured")
    assert "license" in capability.reason.lower()


def test_moreau_rollout_and_differentiable_instances_are_separate():
    module = load_qp_module_without_simulator()
    solver = _qp(module, "moreau")
    rollout_data = _nontrivial_data()
    with torch.inference_mode():
        rollout = solver.solve(differentiable=False, **rollout_data)
    assert not rollout.tau_safe.requires_grad

    ppo_data = {name: value.detach().clone() for name, value in rollout_data.items()}
    ppo_data["tau_nom"].requires_grad_(True)
    ppo = solver.solve(differentiable=True, **ppo_data)
    ppo.tau_safe.square().mean().backward()
    assert ppo_data["tau_nom"].grad is not None
    assert torch.isfinite(ppo_data["tau_nom"].grad).all()
    backend = solver._backend_instances["moreau"]
    assert len(backend._rollout_cache) == 2


def test_moreau_reused_solver_preserves_two_outstanding_vjps():
    module = load_qp_module_without_simulator()

    def gradients(combined):
        solver = _qp(module, "moreau")
        sources = []
        losses = []
        for offset in (0.0, 0.15):
            source = _nontrivial_data()
            source["tau_nom"] = (
                source["tau_nom"] + offset
            ).detach().requires_grad_(True)
            result = solver.solve(differentiable=True, **source)
            losses.append(result.tau_safe.square().mean())
            sources.append(source)
            if not combined:
                losses[-1].backward()
        if combined:
            sum(losses).backward()
        return [source["tau_nom"].grad.clone() for source in sources]

    separate = gradients(False)
    outstanding = gradients(True)
    for actual, expected in zip(outstanding, separate):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)


def test_moreau_zero_and_nonnegative_cone_mapping_has_exact_signs():
    module = load_qp_module_without_simulator()
    from rsl_rl.algorithms.hard_pact_qp_backends import moreau_conic_mapping
    A = torch.tensor([[[1.0, -2.0]]], dtype=torch.float64)
    b = torch.tensor([[0.25]], dtype=torch.float64)
    G = torch.tensor([[[1.0, 0.0], [0.0, -1.0]]], dtype=torch.float64)
    h = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    C, rhs, nzero, nnonnegative = moreau_conic_mapping(A, b, G, h)
    assert (nzero, nnonnegative) == (1, 2)
    x = torch.tensor([[0.5, 0.125]], dtype=torch.float64)
    slack = rhs - torch.einsum("bmn,bn->bm", C, x)
    # Equality is represented by the zero cone; Gx<=h is represented by a
    # nonnegative slack in Cx+s=rhs. A sign reversal would fail either check.
    torch.testing.assert_close(
        slack[:, :nzero], torch.zeros_like(slack[:, :nzero])
    )
    assert (slack[:, nzero:] >= 0.0).all()


def test_moreau_learned_reference_gradients_match_finite_differences():
    """Check Moreau's native implicit derivative, not only qpth agreement.

    The probe is deliberately kept away from actuator/rate box kinks.  The
    fixed tolerance below is for Moreau's float64 CPU factorization and is not
    adjusted from solver output at runtime.
    """
    module = load_qp_module_without_simulator()
    source = _nontrivial_data()
    source["tau_nom"].mul_(0.1)
    source["foot_acceleration_bias"].fill_(3.0)
    source["base_jacobian"][:, :, :6] = torch.eye(6, dtype=torch.float64)
    for foot in range(4):
        source["foot_jacobians"][:, foot, :, :3] = torch.eye(
            3, dtype=torch.float64
        )
    learned = (
        "tau_nom", "force_pred_world", "wrench_pred_world",
        "contact_probability",
    )
    for name in learned:
        source[name].requires_grad_(True)
    solver = _qp(module, "moreau")

    def scalar(result):
        return (
            result.tau_safe.square().mean()
            + 1.0e-3 * result.force_world.square().mean()
            + 1.0e-2 * result.qdd.square().mean()
            + 1.0e-2 * result.contact_slack.square().mean()
        )

    scalar(solver.solve(differentiable=True, **source)).backward()
    probes = {
        "tau_nom": (0, 0),
        "force_pred_world": (0, 0, 2),
        "wrench_pred_world": (0, 0),
        "contact_probability": (0, 0),
    }
    epsilon = 1.0e-4
    for name, index in probes.items():
        analytic = source[name].grad[index]
        assert torch.isfinite(analytic)
        plus = {key: value.detach().clone() for key, value in source.items()}
        minus = {key: value.detach().clone() for key, value in source.items()}
        plus[name][index] += epsilon
        minus[name][index] -= epsilon
        with torch.inference_mode():
            finite_difference = (
                scalar(solver.solve(differentiable=False, **plus))
                - scalar(solver.solve(differentiable=False, **minus))
            ) / (2.0 * epsilon)
        torch.testing.assert_close(
            analytic, finite_difference, atol=2.0e-3, rtol=2.0e-2,
            msg=lambda message: f"{name}: {message}",
        )


def test_moreau_and_qpth_share_invalid_precheck_and_analytic_fallback():
    """Invalid physical intersections bypass either numerical backend."""
    module = load_qp_module_without_simulator()
    source = _nontrivial_data()
    source["joint_position"].fill_(10.0)
    source["joint_velocity"].fill_(100.0)
    reference = _qp(module, "qpth").solve(differentiable=False, **source)
    candidate = _qp(module, "moreau").solve(differentiable=False, **source)
    assert reference.stage.tolist() == candidate.stage.tolist() == [2, 2]
    torch.testing.assert_close(
        candidate.tau_safe, reference.tau_safe, atol=1.0e-12, rtol=1.0e-12
    )
    assert candidate.diagnostics["failure/empty_qdd_intersection"].all()
