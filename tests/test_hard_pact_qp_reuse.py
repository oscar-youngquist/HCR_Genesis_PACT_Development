"""Allocation/synchronization regressions without changing the physical QP."""
import inspect
from unittest import mock

import pytest
import torch

from test_go2_hard_pact_qp import make_qp, qp_data
from test_hard_pact_qp_optional_solvers import requires_cupiqp_gpu
from rsl_rl.algorithms.hard_pact_qp import HardPACTDifferentiableQP


LEARNED = ("tau_nom", "force_pred_world", "wrench_pred_world", "contact_probability")


def coupled_data(batch, dtype, device="cpu"):
    data = {key: value.to(device) for key, value in qp_data(batch, dtype).items()}
    data["tau_nom"].fill_(0.3)
    data["contact_probability"].fill_(0.4)
    data["foot_acceleration_bias"].fill_(0.15)
    data["base_jacobian"][:, :, :6] = torch.eye(6, device=device, dtype=dtype)
    data["wrench_pred_world"][:, 2] = 0.7
    for foot in range(4):
        data["foot_jacobians"][:, foot, :, :3] = torch.eye(3, device=device, dtype=dtype)
        data["foot_jacobians"][:, foot, :, 6 + 3 * foot:9 + 3 * foot] = (
            0.1 * torch.eye(3, device=device, dtype=dtype)
        )
    return data


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("relaxed,elastic", [(False, False), (True, False), (True, True)])
def test_cached_blocks_match_physical_equations_and_are_not_mutated(dtype, relaxed, elastic):
    qp = make_qp()
    data = coupled_data(3, dtype)
    first = qp._build(data, relaxed, elastic)
    templates = qp._assembly_templates(data["tau_nom"], relaxed, elastic)
    preserved = [value.clone() for value in templates]
    data["mass_matrix"] *= 1.2
    data["bias"] += 0.03
    data["contact_probability"] *= 0.8
    data["wrench_pred_world"] += 0.05
    second = qp._build(data, relaxed, elastic)
    for value, original in zip(templates, preserved):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    if not elastic:
        assert first.Q.data_ptr() == second.Q.data_ptr()
        assert second.Q.stride(0) == 0  # one Hessian, not B dense copies
        torch.testing.assert_close(second.physical_A[:, :, :18], data["mass_matrix"])
        torch.testing.assert_close(second.physical_A[:, :, 18:30],
                                   -data["foot_jacobians"].reshape(3, 12, 18).transpose(1, 2))
        torch.testing.assert_close(second.physical_b,
                                   torch.einsum("bkn,bk->bn", data["base_jacobian"],
                                                data["wrench_pred_world"]) - data["bias"])
    else:
        assert not torch.equal(first.Q, second.Q)
    start = 24 if elastic else 48
    friction = qp._constants(data["tau_nom"])[1]
    torch.testing.assert_close(second.physical_G[:, start:start + 20], friction.expand(3, -1, -1))
    if not relaxed:
        contact_j = (data["foot_jacobians"] * data["contact_probability"][:, :, None, None]).reshape(3, 12, 18)
        torch.testing.assert_close(second.physical_G[:, 80:92, :18], contact_j)
        torch.testing.assert_close(second.physical_G[:, 92:104, :18], -contact_j)
    # Keep the exact max(||physical row||, |rhs|, 1) scaling convention.
    row_scale = torch.maximum(second.physical_G.norm(dim=-1),
                              second.physical_h.abs()).clamp_min(1)
    torch.testing.assert_close(second.G, second.physical_G / row_scale[..., None] * second.variable_scale)


def test_inference_templates_reused_by_ppo_with_all_input_gradients():
    qp = make_qp()
    data = coupled_data(3, torch.float64)
    with torch.inference_mode():
        qp._build(data)
    cached = qp._assembly_templates(data["tau_nom"], False, False)
    for value in cached:
        assert not torch.is_inference(value)
        assert not value.requires_grad
    for key in LEARNED:
        data[key].requires_grad_(True)
    for key in ("mass_matrix", "foot_jacobians", "base_jacobian", "dt"):
        data[key].requires_grad_(True)
    matrix = qp._build(data)
    first = torch.autograd.grad(matrix.p.square().sum() + matrix.b.square().sum()
                                + matrix.G.square().sum(), [data[k] for k in LEARNED])
    # A fresh cache is algebraically and differentially identical; no cached
    # graph may leak from one optimizer step into the next minibatch.
    qp._assembly_cache.clear()
    matrix = qp._build(data)
    second = torch.autograd.grad(matrix.p.square().sum() + matrix.b.square().sum()
                                 + matrix.G.square().sum(), [data[k] for k in LEARNED])
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.isfinite(a).all() and a.abs().sum() > 0
    for key in ("mass_matrix", "foot_jacobians", "base_jacobian", "dt"):
        assert data[key].grad is None


def test_cached_assembly_has_no_repeated_eye_or_device_limit_copy():
    qp = make_qp()
    data = qp_data()
    qp._build(data)
    original_limits = qp._limits(data["tau_nom"])
    with mock.patch.object(torch, "eye", side_effect=AssertionError("repeated identity allocation")):
        qp._build(data)
    assert all(a.data_ptr() == b.data_ptr()
               for a, b in zip(original_limits, qp._limits(data["tau_nom"])))


def test_chunk_decisions_do_not_read_device_scalars():
    # nonzero remains the explicit boundary for variable-size compaction;
    # tensor truth-value, sum->int, and boolean gathers must not re-sync for
    # every input/output tensor. Test our orchestration, not solver internals.
    for method in (HardPACTDifferentiableQP.solve, HardPACTDifferentiableQP._solve_stage,
                   HardPACTDifferentiableQP._cupiqp_native_pack):
        source = inspect.getsource(method)
        for forbidden in ("if valid.any()", "if failed.any()", "if invalid.any()",
                          "bool(valid.all())", "int(valid.sum())", ".item()",
                          "value[failed]", "value[valid]", "[:, keep]"):
            assert forbidden not in source


def test_orchestration_uses_no_tensor_truth_values_or_item_calls():
    qp = make_qp(chunk_size=2)
    data = qp_data(5)
    data["tau_nom"][1] = float("nan")
    # The backend library has its own convergence loop. Isolate our wrapper
    # to check scalar reads in precheck, assembly, certification and fallback.
    def numerical_solver(**kwargs):
        return lambda Q, p, G, h, A, b: torch.zeros_like(p)
    with mock.patch("rsl_rl.algorithms.hard_pact_qp.QPFunction", numerical_solver), \
         mock.patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("scalar bool read")), \
         mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar item read")):
        result = qp.solve(differentiable=False, **data)
    assert torch.isfinite(result.tau_safe).all()


def test_mixed_compaction_preserves_order_fallback_and_safety():
    qp = make_qp(chunk_size=3)
    data = coupled_data(7, torch.float64)
    data["previous_torque"][1] = 200
    data["joint_position"][4] = 200
    data["tau_nom"][6] = float("nan")
    result = qp.solve(**data)
    assert result.stage[[1, 4, 6]].tolist() == [2, 2, 2]
    assert result.diagnostics["failure/empty_torque_intersection"][1]
    assert result.diagnostics["failure/empty_qdd_intersection"][4]
    assert result.diagnostics["failure/nonfinite_input"][6]
    assert torch.isfinite(result.tau_safe).all()
    assert (result.tau_safe.abs() <= 23.5).all()
    for row in (0, 2, 3, 5):
        single = qp.solve(**{name: value[row:row + 1] for name, value in data.items()})
        torch.testing.assert_close(result.tau_safe[row], single.tau_safe[0], rtol=1e-7, atol=1e-7)


@requires_cupiqp_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("relaxed,elastic", [(False, False), (True, False), (True, True)])
def test_capacity_reuse_forward_parity_and_bounded_setups(dtype, relaxed, elastic):
    # Exercise the original tracking/slack objective, including the existing
    # SPD ridge. Primal feasibility alone does not certify solution parity.
    options = dict(qp_solver="cupiqp", solver_dtype=str(dtype).split(".")[-1],
                   rollout_eps_abs=1e-7 if dtype == torch.float64 else 1e-6,
                   rollout_eps_rel=1e-7 if dtype == torch.float64 else 1e-6,
                   rollout_max_iter=30)
    reused = make_qp(**options)
    exact = make_qp(**options, cupiqp_rollout_capacity_reuse=False)
    # Fixed, declared tolerances, not chosen per observed result. Identical
    # independent padded QPs can differ at the iterative solver tolerance.
    tolerance = 2e-4 if dtype == torch.float32 else 2e-6
    with torch.inference_mode():
        for batch in (7, 5, 3, 8, 6):
            data = coupled_data(batch, dtype, "cuda")
            got, accepted, diagnostics = reused._solve_stage(data, relaxed, elastic=elastic)
            expected, reference_accepted, _ = exact._solve_stage(data, relaxed, elastic=elastic)
            assert not diagnostics["solver_exception"].any()
            assert torch.equal(accepted, reference_accepted)
            assert accepted.all(), "parity may not pass by comparing rejected solves"
            torch.testing.assert_close(got, expected, rtol=tolerance, atol=tolerance)
    backend = reused._backend_instances["cupiqp"]
    assert backend.setup_count == 2 and backend.update_count == 3
    assert exact._backend_instances["cupiqp"].setup_count == 5
    assert len(backend._rollout_cache) == 2


@requires_cupiqp_gpu
def test_capacity_growth_output_ownership_and_elastic_hessian_refresh():
    qp = make_qp(qp_solver="cupiqp", solver_dtype="float64", rollout_eps_abs=1e-8,
                 rollout_eps_rel=1e-8, rollout_max_iter=30)
    backend = qp._backend_instances["cupiqp"]
    with torch.inference_mode():
        for batch in (3, 2, 5, 3):
            data = coupled_data(batch, torch.float64, "cuda")
            build = qp._build(data, True, True)
            G, h, lower, upper = qp._cupiqp_native_pack(build, True, True)
            first = backend.solve(build.Q, build.p, G, h, build.A, build.b,
                                  differentiable=False, native_lower=lower, native_upper=upper)
            retained = first.solution.clone()
            # A changed mass changes elastic A^T*A; compare reuse with a fresh
            # solve so stale cached P can never pass unnoticed.
            data["mass_matrix"] *= 1.8
            changed = qp._build(data, True, True)
            inputs = dict(differentiable=False, native_lower=lower, native_upper=upper)
            result = backend.solve(changed.Q, changed.p, G, h, changed.A, changed.b, **inputs)
            from rsl_rl.algorithms.hard_pact_qp_backends import create_backend
            fresh = create_backend("cupiqp", qp.cfg).solve(
                changed.Q, changed.p, G, h, changed.A, changed.b, **inputs)
            torch.testing.assert_close(result.solution, fresh.solution, rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(first.solution, retained, rtol=0, atol=0)
    assert backend.setup_count == 3
    assert len(backend._rollout_cache) == 3


@requires_cupiqp_gpu
def test_capacity_never_pads_beyond_rollout_chunk_budget():
    qp = make_qp(qp_solver="cupiqp", rollout_chunk_size=6)
    with torch.inference_mode():
        for batch in (5, 3):
            qp._solve_stage(coupled_data(batch, torch.float32, "cuda"), False)
    backend = qp._backend_instances["cupiqp"]
    assert backend.setup_count == 2
    assert {key[3] for key in backend._rollout_cache} == {4, 6}


@requires_cupiqp_gpu
def test_bucketing_does_not_change_differentiable_ppo_or_retain_graphs():
    outputs, gradients = [], []
    for reuse in (False, True):
        qp = make_qp(qp_solver="cupiqp", solver_dtype="float64",
                     cupiqp_rollout_capacity_reuse=reuse)
        data = coupled_data(3, torch.float64, "cuda")
        with torch.inference_mode():
            rollout = qp.solve(differentiable=False, **data)
        assert not rollout.tau_safe.requires_grad
        for key in LEARNED:
            data[key].requires_grad_(True)
        result = qp.solve(differentiable=True, **data)
        assert result.differentiated_mask.all()
        output = torch.cat((result.tau_safe, result.force_world.flatten(1),
                            result.contact_slack.flatten(1)), -1)
        outputs.append(output.detach())
        gradients.append(torch.autograd.grad(output.square().sum(), [data[k] for k in LEARNED]))
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.isfinite(a).all() and a.abs().sum() > 0
