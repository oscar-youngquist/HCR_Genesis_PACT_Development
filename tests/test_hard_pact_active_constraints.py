"""Bounded active-update correctness; CUDA cases use real cuPIQP, not mocks."""
from dataclasses import replace
from unittest import mock

import pytest
import torch

from rsl_rl.algorithms.hard_pact_active_constraints import (
    ActiveConstraintCache, canonical_snapshot, certify, equality_candidate,
)
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig, qp_substep_anchors, projection_loss
from legged_gym.envs.go2.go2_hard_pact.deployment import qp_update_contract
from test_go2_hard_pact_qp import make_qp, qp_data
from test_hard_pact_qp_reuse import coupled_data, LEARNED


def make_active(dtype=torch.float64, **kwargs):
    # Tight *reference* solves expose forward errors independently of the
    # production rollout gap-report profile. Production is benchmarked too.
    eps = 1.e-9 if dtype == torch.float64 else 1.e-6
    return make_qp(qp_solver="cupiqp", qp_update_mode="active_constraint_update",
                   solver_dtype=str(dtype), rollout_eps_abs=eps, rollout_eps_rel=eps,
                   rollout_duality_gap_abs=eps, rollout_duality_gap_rel=eps,
                   rollout_max_iter=60, exception_capture_enabled=False, **kwargs)


def primal(r):
    return torch.cat((r.qdd, r.force_world.flatten(1), r.tau_safe, r.contact_slack.flatten(1)), -1)


def call(qp, data, k, ids=None, capacity=None):
    n = data["tau_nom"].shape[0]
    return qp.solve(differentiable=False, substep_index=k,
                    environment_ids=torch.arange(n, device=data["tau_nom"].device) if ids is None else ids,
                    environment_count=n if capacity is None else capacity, **data)


def test_mode_contract_and_default_are_unchanged():
    assert HardPACTQPConfig().qp_update_mode == "every_substep"
    assert qp_substep_anchors("active_constraint_update", 4) == (0, 1, 2, 3)
    contract = qp_update_contract("active_constraint_update", 4)
    assert contract["physics_substep_anchors"] == [0, 1, 2, 3]
    assert contract["ppo_projection_loss_multiplier"] == 1
    assert "isolated full differentiable" in contract["ppo_execution"]
    with pytest.raises(ValueError, match="requires cuPIQP"):
        make_qp(qp_update_mode="active_constraint_update")


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 2.e-5), (torch.float64, 1.e-11)])
def test_ecqp_primal_stationarity_and_native_torque_bounds(dtype, atol):
    qp = make_qp()
    data = qp_data(3, dtype)
    data["tau_nom"].fill_(100.)
    m = qp._build(data)
    binding = torch.zeros_like(m.h, dtype=torch.bool)
    binding[:, :12] = True  # native upper torque/rate box
    solution, ok, metrics = equality_candidate(m, binding, qp.cfg)
    assert ok.all(), metrics
    x = solution["primal"] * m.variable_scale
    torch.testing.assert_close(x[:, 30:42], m.tau_upper, rtol=0, atol=atol)
    assert (solution["dual"][:, :12] > 0).all()
    assert metrics["stationarity"].max() < atol
    # Solve explicitly with the same independent equality rows.
    C = torch.cat((m.A, m.G[:, :12]), dim=1)
    rhs = torch.cat((m.b, m.h[:, :12]), dim=1)
    n, nc = m.p.shape[1], C.shape[1]
    kkt = torch.cat((torch.cat((m.Q, C.transpose(1, 2)), 2),
                     torch.cat((C, m.p.new_zeros(3, nc, nc)), 2)), 1)
    expected = torch.linalg.solve(kkt, torch.cat((-m.p, rhs), 1))[:, :n]
    torch.testing.assert_close(solution["primal"], expected, atol=atol, rtol=atol)


def test_redundant_constraints_and_wrong_dual_sign_are_rejected():
    qp = make_qp()
    m = qp._build(qp_data(2))
    binding = torch.zeros_like(m.h, dtype=torch.bool)
    # With J=0, both contact sides and slack lower bound select the same
    # slack normal. Never regularize away this rank deficiency.
    binding[0, [68, 80, 92]] = True
    # Lower torque bound is not optimal for tau_nom=0: multiplier is negative.
    binding[1, 12:24] = True
    _, ok, metrics = equality_candidate(m, binding, qp.cfg)
    assert not ok.any()
    assert metrics["rank_rejected"][0]
    assert metrics["multiplier_violation_rejected"][1]


def test_cache_is_owned_resettable_identity_indexed_and_settings_sensitive():
    qp = make_qp()
    m = qp._build(qp_data(3))
    s, ok, _ = equality_candidate(m, torch.zeros_like(m.h, dtype=torch.bool), qp.cfg)
    c = ActiveConstraintCache()
    c.prepare(m, 4, qp.cfg)
    ids = torch.tensor([3, 0, 2])
    c.commit(m, ids, s, ok, qp.cfg)
    old = c.state["primal"].clone()
    s["primal"].fill_(99.)
    torch.testing.assert_close(c.state["primal"], old, rtol=0, atol=0)
    c.clear(torch.tensor([0]))
    assert c.state["valid"].tolist() == [False, False, True, True]
    c.prepare(m, 4, replace(qp.cfg, friction_coefficient=.7))
    assert not c.state["valid"].any()
    c.clear()
    assert not c.state


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA/cuPIQP required")


@gpu
@pytest.mark.parametrize("dtype,atol", [(torch.float64, .003), (torch.float32, .03)])
def test_gpu_full_parity_changing_mechanics_rate_bounds_and_owned_native_snapshots(dtype, atol):
    qp = make_active(dtype, torque_rate_limit_nm_s=10.)
    reference = make_active(dtype, torque_rate_limit_nm_s=10.)
    data = {k: v.cuda() for k, v in qp_data(4, dtype).items()}
    data["tau_nom"].fill_(2.)
    previous = data["previous_torque"].clone()
    for interval in range(2):
        for k in range(4):
            data["mass_matrix"] *= 1.005
            data["tau_nom"] += .01
            data["previous_torque"] = previous
            actual = call(qp, data, k)
            expected = reference.solve(differentiable=False, **data)
            assert (actual.stage == 0).all()
            assert not primal(actual).requires_grad
            torch.testing.assert_close(primal(actual), primal(expected), rtol=1.e-4, atol=atol)
            assert ((actual.tau_safe - previous).abs() <= .2 + 1.e-6).all()
            assert (actual.tau_safe.abs() <= 23.5).all()
            m = qp._build(data)
            a, b = primal(actual) / m.variable_scale, primal(expected) / m.variable_scale
            objective = lambda z: .5 * (z * (m.Q @ z[..., None]).squeeze(-1)).sum(-1) + (m.p * z).sum(-1)
            torch.testing.assert_close(objective(a), objective(b), rtol=1.e-4, atol=1.e-4)
            if k == 0:
                assert actual.diagnostics["full/active/full_solve"].all()
                # Mapped native duals reproduce the original solver-space
                # stationarity and retain *owned* native slack snapshots.
                s = {name: qp._active_constraint_cache.state[name][:4] for name in
                     ("primal", "dual", "slack", "equality_dual")}
                _, metrics = certify(m, s, qp.cfg)
                assert metrics["stationarity"].max() < 1.e-4
                torch.testing.assert_close(s["slack"], m.h - (m.G @ s["primal"][..., None]).squeeze(-1), atol=1.e-4, rtol=1.e-4)
            else:
                assert actual.diagnostics["full/active/accepted"].all()
            previous = actual.tau_safe


@gpu
def test_gpu_contact_transition_reordering_reset_and_mixed_recovery_exceptions():
    qp = make_active()
    data = {k: v.cuda() for k, v in qp_data(4).items()}
    call(qp, data, 0)
    qp.clear_warm_start(torch.tensor([1], device="cuda"))
    ids = torch.tensor([3, 1, 0, 2], device="cuda")
    reordered = {k: v[ids].clone() for k, v in data.items()}
    result = call(qp, reordered, 1, ids)
    assert result.diagnostics["full/active/full_solve"].tolist() == [False, True, False, False]
    # Rank-degenerate cache rejects one row; another row's contact pattern
    # changes. Other environments must retain their accepted candidates.
    qp._active_constraint_cache.state["binding"][3, [68, 80, 92]] = True
    reordered["foot_jacobians"][1, 0, 0, 6] = .1
    reordered["contact_probability"][1, 0] = .9
    reordered["foot_acceleration_bias"][1, 0, 0] = .4
    backend = qp._backend_instances["cupiqp"]
    original = backend.solve
    calls = []
    def solve(*args, **kwargs):
        calls.append(args[0].shape[0])
        if len(calls) == 1:
            raise RuntimeError("forced rejected-batch full solve failure")
        return original(*args, **kwargs)
    with mock.patch.object(backend, "solve", side_effect=solve):
        result = call(qp, reordered, 2, ids)
    assert calls[0] == 2
    assert result.diagnostics["full/active/accepted"].tolist() == [False, False, True, True]
    assert (result.stage[:2] != 0).all() and (result.stage[2:] == 0).all()
    assert not qp._active_constraint_cache.state["valid"][[3, 1]].any()
    assert torch.isfinite(primal(result)).all()
    # If every recovery also raises, only rejected rows become analytic.
    with mock.patch.object(backend, "solve", side_effect=RuntimeError("all recovery failed")):
        result = call(qp, reordered, 3, ids)
    assert result.stage.tolist() == [2, 2, 0, 0]
    assert not result.differentiated_mask[:2].any()


@gpu
def test_gpu_sampled_ppo_isolated_forward_backward_matches_full_cupiqp():
    qp = make_active()
    data = coupled_data(4, torch.float64, "cuda")
    call(qp, data, 0)
    call(qp, data, 1)
    # PPO selects/shuffles compact substep packets, not the rollout solver's
    # result or KKT factor. Both paths use the PPO profile and same raw data.
    a = {k: v[[3, 0]].clone().requires_grad_(k in LEARNED) for k, v in data.items()}
    b = {k: v.detach().clone().requires_grad_(k in LEARNED) for k, v in a.items()}
    reference = make_active()
    with mock.patch.object(qp._active_constraint_cache, "solve", side_effect=AssertionError("PPO used rollout factors")):
        ra = qp.solve(differentiable=True, **a)
    rb = reference.solve(differentiable=True, **b)
    torch.testing.assert_close(primal(ra), primal(rb), rtol=1.e-9, atol=1.e-9)
    for data, result in ((a, ra), (b, rb)):
        loss = projection_loss(result.tau_safe, data["tau_nom"], qp.torque_limits.cuda(),
                               torch.ones(2, 1, device="cuda", dtype=torch.bool), result.differentiated_mask[:, None],
                               result.contact_slack, qp.cfg.slack_scale_m_s2)
        loss.backward()
    for name in LEARNED:
        assert a[name].grad.isfinite().all() and a[name].grad.abs().sum() > 0
        torch.testing.assert_close(a[name].grad, b[name].grad, rtol=1.e-7, atol=1.e-9)


@gpu
def test_gpu_chunked_identity_cache_and_factor_exception_retry_full_before_recovery():
    qp = make_active(chunk_size=2)
    data = {k: v.cuda() for k, v in qp_data(5).items()}
    call(qp, data, 0)
    ids = torch.tensor([4, 0, 3, 1, 2], device="cuda")
    data = {k: v[ids] for k, v in data.items()}
    result = call(qp, data, 1, ids)
    assert result.diagnostics["full/active/accepted"].all()
    with mock.patch("rsl_rl.algorithms.hard_pact_active_constraints.equality_candidate", side_effect=RuntimeError("factor failed")):
        result = call(qp, data, 2, ids)
    assert (result.stage == 0).all()
    assert result.diagnostics["full/active/factor_exception"].all()
    assert result.diagnostics["full/active/full_solve"].all()
    # A no-grad inference allocation can be invalidated by an outer runner.
    qp.clear_warm_start()
    with torch.inference_mode():
        call(qp, data, 0, ids)
    qp.clear_warm_start(torch.tensor([2], device="cuda"))
    assert not qp._active_constraint_cache.state["valid"][2]
