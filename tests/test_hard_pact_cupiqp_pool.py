"""Bounded cuPIQP ownership and iteration diagnostics regressions."""
import gc
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from rsl_rl.algorithms.hard_pact_qp_backends import SolverPool, CUDAEventProfile
from rsl_rl.algorithms.hard_pact_qp_diagnostics import QPIterationDiagnostics
from rsl_rl.runners.pact_runner import OnPolicyRunnerPACT
from rsl_rl.algorithms.pc_grad import PCGrad
from test_go2_hard_pact_qp import make_qp, qp_data
from test_hard_pact_qp_reuse import coupled_data, LEARNED
from test_hard_pact_qp_optional_solvers import requires_cupiqp_gpu


def test_pool_never_evicts_busy_and_discards_unhealthy_leases():
    pool = SolverPool(1)
    a, hit = pool.acquire("a", object)
    assert not hit and a.pooled
    b, hit = pool.acquire("a", object)
    assert not hit and not b.pooled and a.solver is not b.solver
    b.release()
    a.release()
    c, hit = pool.acquire("a", object)
    assert hit
    c.healthy = False
    c.release()
    assert pool.size == 0 and not pool.idle
    d, _ = pool.acquire("a", object)
    d.release()
    e, hit = pool.acquire("new_structure", object)
    assert not hit and pool.size == 1 and e.pooled


def test_disabled_profiling_does_not_create_events_or_synchronize():
    profile = CUDAEventProfile(False)
    with mock.patch.object(torch.cuda, "Event", side_effect=AssertionError), \
         mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError):
        with profile.measure("solve", SimpleNamespace(device=torch.device("cuda"))):
            pass
        assert profile.finalize() == {}


def test_final_stages_counts_and_fraction_partition_invariance():
    qp = make_qp(elastic_recovery_enabled=True)
    # Exercise mixed full/relaxed/elastic/analytic results without relying on
    # numerical failure of a particular iterative solver on this platform.
    def stage_solver(data, relaxed, **kwargs):
        n = data["tau_nom"].shape[0]
        row_id = data["tau_nom"][:, 0].long()
        code = 2 if kwargs.get("elastic") else int(relaxed)
        ok = row_id == code
        diag = {"attempted": torch.ones(n, dtype=torch.bool),
                "solver_exception": torch.zeros(n, dtype=torch.bool),
                "output_finite": torch.ones(n, dtype=torch.bool),
                "equality_max": torch.zeros(n), "inequality_max": torch.zeros(n)}
        return torch.zeros(n, 54, dtype=torch.float64), ok, diag
    data = qp_data(4)
    data["tau_nom"][:, 0] = torch.arange(4)
    with mock.patch.object(qp, "_solve_stage", side_effect=stage_solver):
        result = qp.solve(differentiable=True, diagnostics_phase="ppo", **data)
    metrics = qp.iteration_metrics("ppo", data["tau_nom"])
    for name in ("full", "relaxed", "elastic", "analytic"):
        assert metrics[f"qp/ppo/final/{name}_count"] == 1
        assert metrics[f"qp/ppo/final/{name}_fraction"] == 0.25
    assert metrics["qp/ppo/attempt/full_count"] == 4
    assert metrics["qp/ppo/attempt/relaxed_count"] == 3
    assert metrics["qp/ppo/attempt/elastic_count"] == 2
    assert metrics["qp/ppo/differentiated_fraction"] == .75
    assert metrics["qp/ppo/selected/equality_max"] == 0
    assert torch.isnan(metrics["qp/ppo/backend/solver_iterations_mean"])
    split = QPIterationDiagnostics()
    for ids in (slice(0, 1), slice(1, 4)):
        split.add_result(SimpleNamespace(
            stage=result.stage[ids], differentiated_mask=result.differentiated_mask[ids],
            diagnostics={k: v[ids] for k, v in result.diagnostics.items()}, metrics={},
        ), True, True)
    for key, value in split.finalize(data["tau_nom"]).items():
        if key != "solve_calls":
            torch.testing.assert_close(value, metrics[f"qp/ppo/{key}"], equal_nan=True)


def test_runner_emits_disjoint_scalar_tags_once_and_held_rows_are_separate():
    qp = make_qp()
    data = qp_data(2)
    qp.solve(differentiable=False, **data)
    qp.iteration_diagnostics["rollout"].add_sum("held/real_rows", torch.tensor(2))
    qp.solve(differentiable=False, diagnostics_phase="ppo", **data)  # stopgrad
    runner = OnPolicyRunnerPACT.__new__(OnPolicyRunnerPACT)
    runner.alg = SimpleNamespace(hard_pact_qp=qp, last_qp_metrics={
        "qp/minimal/full_fraction": torch.tensor(99.),  # stale minibatch ignored
    })
    calls = []
    runner.writer = SimpleNamespace(add_scalar=lambda *args: calls.append(args))
    runner._log_qp_metrics(5)
    keys = [key for key, _, _ in calls]
    assert len(keys) == len(set(keys))
    assert all(key.startswith(("qp/rollout/", "qp/ppo/")) for key in keys)
    values = {key: value for key, value, _ in calls}
    assert values["qp/rollout/real_rows"] == 2
    assert values["qp/rollout/held/real_rows"] == 2
    assert values["qp/ppo/differentiated_fraction"] == 0
    assert values["qp/ppo/certified_fraction"] == 1


def raw_solve(qp, data, relaxed=False, elastic=False):
    build = qp._build(data, relaxed, elastic)
    G, h, lo, hi = qp._cupiqp_native_pack(build, relaxed, elastic)
    return qp._backend_instances["cupiqp"].solve(
        build.Q, build.p, G, h, build.A, build.b, differentiable=True,
        native_lower=lo, native_upper=hi,
    ).solution


@requires_cupiqp_gpu
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 2e-6), (torch.float32, 2e-4)])
def test_reused_ppo_matches_fresh_changed_mechanics_and_all_vjps(dtype, tol):
    options = dict(qp_solver="cupiqp", cupiqp_ppo_pool_size=2)
    reused, fresh = make_qp(**options), make_qp(**options, cupiqp_ppo_reuse=False)
    def evaluate(qp, iteration):
        data = coupled_data(3, dtype, "cuda")
        data["mass_matrix"] *= 1 + .1 * iteration
        data["bias"] += .02 * iteration
        data["wrench_pred_world"] += .03 * iteration
        data["contact_probability"] += .02 * iteration
        data["tau_nom"] += .04 * iteration
        data["force_pred_world"] += .05 * iteration
        for key in LEARNED:
            data[key].requires_grad_()
        out = raw_solve(qp, data)
        grads = torch.autograd.grad(out.square().mean(), [data[k] for k in LEARNED])
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
        return out.detach().clone(), grads
    for iteration in range(3):
        a, ga = evaluate(reused, iteration)
        b, gb = evaluate(fresh, iteration)
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, rtol=tol, atol=tol)
        gc.collect()
    stats = reused._backend_instances["cupiqp"].stats["ppo"]
    assert stats["setup_count"] == 1 and stats["update_count"] == 2


@requires_cupiqp_gpu
def test_retained_graph_pcgrad_and_disposal_never_recycle_early():
    qp = make_qp(qp_solver="cupiqp", cupiqp_ppo_pool_size=1)
    backend = qp._backend_instances["cupiqp"]
    source = coupled_data(2, torch.float64, "cuda")
    source["tau_nom"].requires_grad_()
    a = raw_solve(qp, source)
    first = torch.autograd.grad(a.square().sum(), source["tau_nom"], retain_graph=True)[0]
    saved = first.clone()
    # A second forward must use an overflow private instance, not overwrite A.
    b = raw_solve(qp, source)
    assert backend.stats["ppo"]["pool_hits"] == 0
    again = torch.autograd.grad(a.square().sum(), source["tau_nom"], retain_graph=True)[0]
    torch.testing.assert_close(again, first, rtol=0, atol=0)
    torch.autograd.grad(b.sum(), source["tau_nom"])
    torch.testing.assert_close(first, saved, rtol=0, atol=0)
    del b
    gc.collect()
    assert not backend._ppo_pool.idle  # A still owns the pooled instance.
    del a
    gc.collect()
    assert len(backend._ppo_pool.idle) == 1
    c = raw_solve(qp, source)
    assert backend.stats["ppo"]["pool_hits"] == 1
    del c  # abandoned graph (no backward) must release too
    gc.collect()
    assert len(backend._ppo_pool.idle) == 1


@requires_cupiqp_gpu
@pytest.mark.parametrize("failure_method", ["update", "solve"])
def test_reuse_update_exception_retries_fresh_and_discards_poisoned_solver(failure_method):
    qp = make_qp(qp_solver="cupiqp", cupiqp_ppo_pool_size=1)
    backend = qp._backend_instances["cupiqp"]
    data = coupled_data(2, torch.float64, "cuda")
    data["tau_nom"].requires_grad_()
    result = raw_solve(qp, data)
    del result
    gc.collect()
    solver = next(iter(backend._ppo_pool.idle.values()))[1]
    with mock.patch.object(solver, failure_method, side_effect=RuntimeError("forced reuse failure")):
        out = raw_solve(qp, data)
        torch.autograd.grad(out.sum(), data["tau_nom"])
    assert backend.stats["ppo"]["reuse_exception_fresh_retry"] == 1
    assert backend._ppo_pool.size == 0


@requires_cupiqp_gpu
def test_real_pcgrad_repeated_backwards_and_exception_disposal():
    qp = make_qp(qp_solver="cupiqp", cupiqp_ppo_pool_size=1)
    backend = qp._backend_instances["cupiqp"]
    data = coupled_data(2, torch.float64, "cuda")
    param = torch.nn.Parameter(data["tau_nom"])
    data["tau_nom"] = param
    optimizer = PCGrad(torch.optim.SGD([param], lr=.01))
    out = raw_solve(qp, data)
    objectives = [param.square().mean() + out[:, 30:42].square().mean(),
                  out.square().mean()]
    optimizer.pc_backward_pinn(objectives, record_diagnostics=False)
    assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    assert not backend._ppo_pool.idle
    del objectives, out
    gc.collect()
    assert len(backend._ppo_pool.idle) == 1
    solver = next(iter(backend._ppo_pool.idle.values()))[1]
    out = raw_solve(qp, data)
    with mock.patch.object(solver, "backward", side_effect=RuntimeError("forced VJP failure")):
        with pytest.raises(RuntimeError, match="forced VJP failure"):
            torch.autograd.grad(out.sum(), param)
    del out
    gc.collect()
    assert backend._ppo_pool.size == 0


@requires_cupiqp_gpu
def test_rollout_buckets_shrink_and_lru_cache_is_bounded():
    qp = make_qp(qp_solver="cupiqp", cupiqp_rollout_cache_size=2)
    backend = qp._backend_instances["cupiqp"]
    with torch.inference_mode():
        for batch, expected in ((17, 32), (2, 2), (3, 4), (2, 2)):
            result = qp.solve(differentiable=False, **coupled_data(batch, torch.float64, "cuda"))
            assert result.stage.shape == (batch,)
            assert not result.tau_safe.requires_grad
            assert len(backend._rollout_cache) <= 2
            assert list(backend._rollout_cache)[-1][3] == expected
    assert {key[3] for key in backend._rollout_cache} == {2, 4}
    assert backend.setup_count == 3 and backend.update_count == 1
    stats = backend.stats["rollout"]
    assert stats["requested_rows"] == 24
    assert stats["capacity_rows"] == 40
    assert stats["padded_rows"] == 16


@requires_cupiqp_gpu
def test_profiled_forward_vjp_parity_and_event_timings():
    outputs = []
    for enabled in (False, True):
        qp = make_qp(qp_solver="cupiqp", cuda_event_profiling=enabled)
        data = coupled_data(2, torch.float64, "cuda")
        data["tau_nom"].requires_grad_()
        out = qp.solve(differentiable=True, **data)
        grad = torch.autograd.grad(out.tau_safe.sum(), data["tau_nom"])[0]
        metrics = qp.iteration_metrics("ppo", data["tau_nom"])
        if enabled:
            for name in ("assembly", "packing", "setup_update", "solve", "certification_recovery", "backward"):
                assert metrics[f"qp/ppo/profiling/{name}_ms"] > 0
        outputs.append((out.tau_safe.detach(), grad))
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@requires_cupiqp_gpu
def test_ppo_elastic_pool_updates_state_dependent_hessian():
    options = dict(qp_solver="cupiqp", proximal_rho=.1,
                   ppo_eps_abs=1e-8, ppo_eps_rel=1e-8)
    reused = make_qp(**options)
    fresh = make_qp(**options, cupiqp_ppo_reuse=False)
    def evaluate(qp, mass):
        data = coupled_data(2, torch.float64, "cuda")
        data["mass_matrix"] *= mass
        data["tau_nom"].requires_grad_()
        out = raw_solve(qp, data, relaxed=True, elastic=True)
        grad = torch.autograd.grad(out.square().mean(), data["tau_nom"])[0]
        return out.detach().clone(), grad
    for mass in (1., 1.8, .9):
        for a, b in zip(evaluate(reused, mass), evaluate(fresh, mass)):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)
        gc.collect()
    assert reused._backend_instances["cupiqp"].stats["ppo"]["update_count"] == 2
