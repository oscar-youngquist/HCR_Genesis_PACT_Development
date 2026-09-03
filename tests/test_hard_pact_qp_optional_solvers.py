"""Real optional-backend correctness tests (GPU/dependency gated)."""

import importlib.util

import pytest
import torch

from rsl_rl.algorithms.hard_pact_qp import HardPACTDifferentiableQP, HardPACTQPConfig
from rsl_rl.algorithms.hard_pact_qp_backends import _as_cupy_zero_copy


def data(batch, device, dtype):
    zeros = lambda *shape: torch.zeros(*shape, device=device, dtype=dtype)
    return {
        "mass_matrix": torch.eye(18, device=device, dtype=dtype).expand(batch, -1, -1).clone(),
        "bias": zeros(batch, 18),
        "foot_jacobians": zeros(batch, 4, 3, 18),
        "base_jacobian": zeros(batch, 6, 18),
        "foot_acceleration_bias": zeros(batch, 4, 3),
        "tau_nom": torch.full((batch, 12), 0.5, device=device, dtype=dtype),
        "force_pred_world": torch.zeros(batch, 4, 3, device=device, dtype=dtype),
        "wrench_pred_world": zeros(batch, 6),
        "contact_probability": torch.full((batch, 4), 0.4, device=device, dtype=dtype),
        "previous_torque": zeros(batch, 12),
        "joint_position": zeros(batch, 12),
        "joint_velocity": zeros(batch, 12),
        "dt": torch.full((batch, 1), 0.002, device=device, dtype=dtype),
    }


def qp(solver, dtype):
    return HardPACTDifferentiableQP(
        HardPACTQPConfig(
            qp_solver=solver, solver_dtype=dtype, max_iter=50,
            not_improved_limit=10,
        ),
        torch.full((12,), 23.5), torch.full((12,), -2.0),
        torch.full((12,), 2.0), torch.full((12,), 30.0),
    )


def canonical_objective(qp_instance, source, result):
    build = qp_instance._build(source)
    physical = torch.cat(
        (result.qdd, result.force_world.reshape(result.qdd.shape[0], 12),
         result.tau_safe,
         result.contact_slack.reshape(result.qdd.shape[0], 12)), dim=-1,
    )
    scaled = physical / build.variable_scale
    return (
        0.5 * torch.einsum("bi,bij,bj->b", scaled, build.Q, scaled)
        + torch.einsum("bi,bi->b", build.p, scaled)
    )


requires_cupiqp_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("cupiqp") is None,
    reason="requires a real CUDA device and cuPIQP",
)


@requires_cupiqp_gpu
def test_cupiqp_dlpack_storage_is_shared_without_host_copy():
    tensor = torch.arange(8, device="cuda", dtype=torch.float32)
    cupy_view = _as_cupy_zero_copy(tensor)
    cupy_view += 2.0
    torch.testing.assert_close(tensor, torch.arange(8, device="cuda") + 2.0)
    assert int(cupy_view.data.ptr) == tensor.data_ptr()


@requires_cupiqp_gpu
def test_cupiqp_qpth_canonical_forward_and_gradient_parity():
    # Fixed tolerances are intentionally solver-specific. qpth float64 is the
    # reference; cuPIQP runs the production CUDA float32 path.
    reference_data = data(4, "cuda", torch.float64)
    candidate_data = {
        key: value.float().detach() for key, value in reference_data.items()
    }
    learned = (
        "tau_nom", "force_pred_world", "wrench_pred_world",
        "contact_probability",
    )
    for name in learned:
        reference_data[name].requires_grad_(True)
        candidate_data[name].requires_grad_(True)
    reference_qp = qp("qpth", "float64")
    candidate_qp = qp("cupiqp", "float32")
    reference = reference_qp.solve(
        differentiable=True, **reference_data
    )
    candidate = candidate_qp.solve(
        differentiable=True, **candidate_data
    )
    assert torch.equal(reference.stage, candidate.stage)
    assert (reference.stage == 0).all()
    for actual, expected, atol, rtol in (
        (candidate.tau_safe.double(), reference.tau_safe, 3e-3, 3e-3),
        (candidate.qdd.double(), reference.qdd, 3e-3, 3e-3),
        (candidate.force_world.double(), reference.force_world, 2e-2, 2e-3),
        (candidate.contact_slack.double(), reference.contact_slack, 2e-2, 2e-3),
    ):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    reference_loss = canonical_objective(
        reference_qp, reference_data, reference
    ).mean()
    candidate_loss = canonical_objective(
        candidate_qp, candidate_data, candidate
    ).mean()
    torch.testing.assert_close(
        candidate_loss.double(), reference_loss, atol=2e-2, rtol=3e-3
    )
    for key in ("selected/equality_max", "selected/inequality_max"):
        torch.testing.assert_close(
            candidate.diagnostics[key].double(), reference.diagnostics[key],
            atol=3e-3, rtol=3e-3,
        )
    reference_loss.backward()
    candidate_loss.backward()
    for name in learned:
        assert torch.isfinite(candidate_data[name].grad).all()
        torch.testing.assert_close(
            candidate_data[name].grad.double(), reference_data[name].grad,
            atol=2e-2, rtol=5e-2,
        )


@requires_cupiqp_gpu
def test_cupiqp_and_qpth_share_invalid_precheck_and_analytic_fallback():
    # A contradictory one-step position/velocity intersection is rejected by
    # the solver-neutral precheck, so neither numerical backend is entered.
    source64 = data(2, "cuda", torch.float64)
    source64["joint_position"].fill_(10.0)
    source64["joint_velocity"].fill_(100.0)
    source32 = {name: value.float() for name, value in source64.items()}
    reference = qp("qpth", "float64").solve(
        differentiable=False, **source64
    )
    candidate = qp("cupiqp", "float32").solve(
        differentiable=False, **source32
    )
    assert reference.stage.tolist() == candidate.stage.tolist() == [2, 2]
    torch.testing.assert_close(
        candidate.tau_safe.double(), reference.tau_safe,
        atol=2e-6, rtol=2e-6,
    )
    assert candidate.diagnostics["failure/empty_qdd_intersection"].all()


@requires_cupiqp_gpu
def test_cupiqp_implicit_backward_supports_retained_pcgrad_vjps():
    source = data(2, "cuda", torch.float32)
    source["tau_nom"].requires_grad_(True)
    result = qp("cupiqp", "float32").solve(
        differentiable=True, **source
    )
    first = torch.autograd.grad(
        result.tau_safe.square().mean(), source["tau_nom"],
        retain_graph=True,
    )[0]
    second = torch.autograd.grad(
        result.qdd.square().mean() + result.force_world.square().mean(),
        source["tau_nom"], retain_graph=False,
    )[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert first.abs().sum() > 0


@requires_cupiqp_gpu
def test_cupiqp_repeated_rollout_ppo_interleave_has_no_graph_or_lifetime_leak():
    solver = qp("cupiqp", "float32")
    for _ in range(3):
        rollout_data = data(2, "cuda", torch.float32)
        with torch.inference_mode():
            rollout = solver.solve(differentiable=False, **rollout_data)
        assert not rollout.tau_safe.requires_grad
        ppo_data = data(2, "cuda", torch.float32)
        ppo_data["tau_nom"].requires_grad_(True)
        result = solver.solve(differentiable=True, **ppo_data)
        result.tau_safe.square().mean().backward()
        assert torch.isfinite(ppo_data["tau_nom"].grad).all()


@requires_cupiqp_gpu
def test_cupiqp_sparse_rollout_reuses_setup_and_updates_zero_copy_vectors():
    config = HardPACTQPConfig(
        qp_solver="cupiqp", solver_dtype="float32", cupiqp_mode="sparse",
        max_iter=50, not_improved_limit=10,
    )
    solver = HardPACTDifferentiableQP(
        config, torch.full((12,), 23.5), torch.full((12,), -2.0),
        torch.full((12,), 2.0), torch.full((12,), 30.0),
    )
    try:
        first = solver.solve(
            differentiable=False, **data(2, "cuda", torch.float32)
        )
        second = solver.solve(
            differentiable=False, **data(2, "cuda", torch.float32)
        )
    except (ImportError, OSError, RuntimeError) as error:
        # Sparse cuPIQP additionally requires a compatible nvmath/cuDSS
        # runtime. Dense cuPIQP remains independently testable and supported.
        pytest.skip(f"cuPIQP sparse runtime unavailable: {error}")
    backend = solver._backend_instances["cupiqp"]
    assert backend.setup_count == 1
    assert backend.update_count >= 1
    assert (first.stage == 0).all() and (second.stage == 0).all()
    torch.testing.assert_close(first.tau_safe, second.tau_safe)


@requires_cupiqp_gpu
def test_cupiqp_coupled_reference_gradients_match_finite_differences():
    source = data(1, "cuda", torch.float32)
    # Keep the torque tracking center strictly inside its magnitude/rate box;
    # finite differences at an active box kink are not a derivative test.
    source["tau_nom"].fill_(0.01)
    source["foot_acceleration_bias"].fill_(3.0)
    source["base_jacobian"][:, :, :6] = torch.eye(6, device="cuda")
    for foot in range(4):
        source["foot_jacobians"][:, foot, :, :3] = torch.eye(
            3, device="cuda"
        )
    source["force_pred_world"][:, :, 2] = 5.0
    source["wrench_pred_world"][:] = torch.tensor(
        (0.2, -0.1, 0.5, 0.05, 0.02, -0.03), device="cuda"
    )
    learned = (
        "tau_nom", "force_pred_world", "wrench_pred_world",
        "contact_probability",
    )
    for name in learned:
        source[name].requires_grad_(True)
    solver = qp("cupiqp", "float32")

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
    epsilon = 2.0e-2
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
            analytic, finite_difference, atol=2.5e-2, rtol=7.5e-2,
            msg=lambda message: f"{name}: {message}",
        )
