"""Rollout-only equality-constrained updates of the canonical HardPACT QP.

No interior-point factor or autograd context is reused. All calculations use
the current scaled problem z=D^-1 x; canonical G includes cuPIQP native bounds.
PPO never calls this module. A failed certificate always goes back to cuPIQP.
"""
from dataclasses import fields, replace

import torch

from .hard_pact_qp_backends import QPBackendResult


def select_problem(m, rows):
    """Index environment-dependent data, retaining the shared D vector."""
    return replace(m, **{f.name: getattr(m, f.name).index_select(0, rows)
                         for f in fields(m) if f.name != "variable_scale"})


def canonical_snapshot(m, result):
    """Map cuPIQP 0.1 native multipliers/slacks to canonical Gz<=h.

    cuPIQP returns unpreconditioned y, z_u, z_bl, z_bu and s_* in the input
    z-coordinate problem (verified in Solver.solve/unscale_solution). Native
    lower/upper arrays have n columns, including inactive infinite entries.
    For a canonical bound a*z_j<=h, lambda_canonical=lambda_native/|a|
    and slack_canonical=|a|*slack_native. Equality rows are unchanged.
    """
    s = result.snapshot
    dual, slack = torch.zeros_like(m.h), torch.zeros_like(m.h)
    dual[:,24:], slack[:,24:] = s["z_u"], s["s_u"]
    for rows, side in ((slice(0,12),"bu"),(slice(12,24),"bl")):
        coefficient = m.G[:,rows].abs().amax(-1)
        dual[:,rows] = s["z_"+side][:,:12] / coefficient
        slack[:,rows] = s["s_"+side][:,:12] * coefficient
    return {"primal": result.solution.detach().clone(), "dual": dual,
            "slack": slack, "equality_dual": s["y"].detach().clone()}


def equality_candidate(m, binding, cfg):
    r"""Solve one ECQP, then certify against *all* original constraints.

    C=[A;G_I], d=[b;h_I], Qz+p+C^T lambda=0, Cz=d.
    Refactor Q=L L^T and H=C Q^-1 C^T for each new state. Then
      lambda = H^-1(-d-C Q^-1 p), z=-Q^-1(p+C^T lambda).
    Inactive padded rows get a unit diagonal in H and zero RHS; they are
    not physical equalities. More than n-rank(A) binding rows is rejected,
    never silently truncated. Failed/tiny normalized Cholesky pivots reject
    dependent or degenerate rows; no pseudoinverse or regularized constraint solve
    can turn a rank failure into an accepted candidate.
    """
    batch, n = m.p.shape
    ne = m.A.shape[1]
    slots = n - ne
    count = binding.sum(-1)
    # Stable canonical row order, without variable-size CPU lists or RNG.
    ids = torch.argsort((~binding).to(torch.int32), dim=-1, stable=True)[:, :slots]
    used = torch.arange(slots, device=m.p.device)[None] < count[:, None]
    rows = m.G.gather(1, ids[..., None].expand(-1, -1, n)) * used[..., None]
    rhs = m.h.gather(1, ids) * used
    C = torch.cat((m.A, rows), dim=1)
    d = torch.cat((m.b, rhs), dim=1)
    L, qi = torch.linalg.cholesky_ex(m.Q, check_errors=False)
    inv_ct = torch.cholesky_solve(C.transpose(1, 2), L)
    inv_p = torch.cholesky_solve(m.p[..., None], L).squeeze(-1)
    H = C @ inv_ct
    padding = torch.cat((torch.zeros_like(m.b, dtype=torch.bool), ~used), dim=1)
    H = H + torch.diag_embed(padding.to(H.dtype))
    # Unit-diagonal congruence improves rank decisions without changing ECQP.
    scale = H.diagonal(dim1=-2, dim2=-1).clamp_min(torch.finfo(H.dtype).tiny).sqrt()
    H = H / scale[:, :, None] / scale[:, None, :]
    H = 0.5 * (H + H.transpose(1, 2))
    finite = torch.isfinite(H).all(dim=(-1, -2))
    safe_H = torch.where(finite[:, None, None], H, torch.eye(n, device=H.device, dtype=H.dtype))
    LH, hi = torch.linalg.cholesky_ex(safe_H, check_errors=False)
    pivots = LH.diagonal(dim1=-2, dim2=-1).square()
    rank_ok = ((hi == 0) & (qi == 0) & finite & (count <= slots)
               & (pivots.amin(-1) > cfg.active_rank_tolerance * pivots.amax(-1)))
    # Failed rows still run a finite placeholder factorization; they are
    # rejected below and only their owning environments enter full cuPIQP.
    LH = torch.where(rank_ok[:, None, None], LH, torch.eye(n, device=H.device, dtype=H.dtype))
    dual = torch.cholesky_solve(((-d - (C @ inv_p[..., None]).squeeze(-1)) / scale)[..., None], LH).squeeze(-1) / scale
    z = -inv_p - (inv_ct @ dual[..., None]).squeeze(-1)
    # Certify the actual post-projected command, not a slightly out-of-box
    # floating-point optimum. The normal solver repeats this idempotent clamp.
    z[:, 0:12] = z[:, 0:12].clamp(m.native_lower[:, 0:12], m.native_upper[:, 0:12])
    inequality_dual = torch.zeros_like(m.h).scatter(1, ids, dual[:, ne:] * used)
    snapshot = {"primal": z, "equality_dual": dual[:, :ne],
                "dual": inequality_dual, "slack": m.h - (m.G @ z[..., None]).squeeze(-1)}
    accepted, metrics = certify(m, snapshot, cfg)
    metrics["rank_rejected"] = ~rank_ok | (hi != 0)
    accepted &= ~metrics["rank_rejected"]
    return snapshot, accepted, metrics


def certify(m, s, cfg):
    """Scaled KKT certificate plus physical checks on the executed x=Dz.

    nu unrestricted; lambda>=0, h-Gz>=0, lambda*(h-Gz)=0.
    gap=|z^T Qz+p^Tz+b^Tnu+h^Tlambda|. Relative dual residual
    uses 1+max(||Qz||inf,||p||inf,||A^Tnu+G^Tlambda||inf).
    Physical tolerances are exactly the existing row scales times primal
    tolerance (not an unrelated loose physical tolerance).
    """
    z, nu, lam = s["primal"], s["equality_dual"], s["dual"]
    qz = (m.Q @ z[..., None]).squeeze(-1)
    dual_force = (m.A.transpose(1, 2) @ nu[..., None] + m.G.transpose(1, 2) @ lam[..., None]).squeeze(-1)
    eq = (m.A @ z[..., None]).squeeze(-1) - m.b
    ineq = (m.G @ z[..., None]).squeeze(-1) - m.h
    objective = .5 * (z * qz).sum(-1) + (m.p * z).sum(-1)
    dual_objective = -.5 * (z * qz).sum(-1) - (m.b * nu).sum(-1) - (m.h * lam).sum(-1)
    gap = (objective - dual_objective).abs()
    norm = 1 + torch.stack((qz.abs().amax(-1), m.p.abs().amax(-1), dual_force.abs().amax(-1))).amax(0)
    x = z * m.variable_scale
    peq = ((m.physical_A @ x[..., None]).squeeze(-1) - m.physical_b).abs()
    pineq = ((m.physical_G @ x[..., None]).squeeze(-1) - m.physical_h).clamp_min(0)
    finite = torch.stack([v.isfinite().flatten(1).all(-1) for v in
                          (z, nu, lam, eq, ineq, peq, pineq, qz, dual_force,
                           objective[:, None], dual_objective[:, None], gap[:, None])]).all(0)
    tolerance = cfg.rollout_feasibility_tolerance
    metrics = {
        "nonfinite_rejected": ~finite,
        "primal_rejected": (_max(eq.abs()) > tolerance) | (ineq.amax(-1) > tolerance),
        "physical_rejected": (peq > tolerance * m.equality_row_scale).any(-1) | (pineq > tolerance * m.inequality_row_scale).any(-1),
        "stationarity": (qz + m.p + dual_force).abs().amax(-1) / norm,
        "multiplier_violation": (-lam).clamp_min(0).amax(-1) / norm,
        "complementarity": (lam * ineq).abs().amax(-1) / (1 + objective.abs()),
        "gap": gap, "gap_relative": gap / (1 + torch.maximum(objective.abs(), dual_objective.abs())),
        "physical_equality_max": _max(peq), "physical_inequality_max": pineq.amax(-1),
    }
    for key in ("stationarity", "multiplier_violation", "complementarity"):
        metrics[key + "_rejected"] = metrics[key] > cfg.active_kkt_tolerance
    metrics["gap_rejected"] = (gap > cfg.rollout_duality_gap_abs) & (metrics["gap_relative"] > cfg.rollout_duality_gap_rel)
    accepted = ~torch.stack([v for k, v in metrics.items() if k.endswith("_rejected")]).any(0)
    return accepted, metrics


def _max(value):
    return value.amax(-1) if value.shape[-1] else value.new_zeros(value.shape[0])


class ActiveConstraintCache:
    """GPU snapshots indexed by stable environment IDs, not compact slots."""
    def __init__(self):
        self.key, self.state = None, {}

    def clear(self, ids=None):
        if ids is None:
            self.key, self.state = None, {}
        elif self.state:
            ids = ids.to(self.state["valid"].device)
            ids = ids[(ids >= 0) & (ids < self.state["valid"].numel())]
            # Runner warmup/reset gates may run outside the inference-mode
            # region that allocated these rollout-only buffers.
            with torch.inference_mode():
                self.state["valid"][ids] = False

    def prepare(self, m, capacity, cfg):
        key = (capacity, m.p.device, m.p.dtype, m.Q.shape[1:], m.G.shape[1:], m.A.shape[1:], cfg)
        if key != self.key:
            self.key = key
            self.state = {"valid": torch.zeros(capacity, device=m.p.device, dtype=torch.bool)}
            for name, template in (("primal", m.p), ("dual", m.h), ("slack", m.h), ("equality_dual", m.b),
                                   ("lower_pattern", m.native_lower.isfinite()), ("upper_pattern", m.native_upper.isfinite())):
                self.state[name] = template.new_zeros((capacity, *template.shape[1:]))
            self.state["binding"] = torch.zeros(capacity, m.h.shape[1], dtype=torch.bool, device=m.p.device)

    def commit(self, m, ids, snapshot, valid, cfg):
        valid = valid & torch.stack([v.isfinite().flatten(1).all(-1) for v in snapshot.values()]).all(0)
        self.state["valid"][ids] = valid
        for name in ("primal", "dual", "slack", "equality_dual"):
            self.state[name][ids] = snapshot[name].detach()
        # Snapshot slacks are retained, but recompute margins from the owned
        # primal against canonical current rows to identify binding IDs.
        margin = m.h - (m.G @ snapshot["primal"][..., None]).squeeze(-1)
        binding = (margin <= cfg.active_binding_tolerance) & (snapshot["dual"] > cfg.active_dual_tolerance)
        self.state["binding"][ids] = binding
        for name, value in (("lower_pattern", m.native_lower.isfinite()), ("upper_pattern", m.native_upper.isfinite())):
            self.state[name][ids] = value

    @torch.no_grad()
    def solve(self, qp, m, ids, reuse, capacity):
        """Only rejected rows enter the normal cuPIQP full-stage backend."""
        self.prepare(m, capacity, qp.cfg)
        cfg = qp.cfg
        valid = self.state["valid"][ids].clone() & reuse
        # The builder has a fixed canonical row/column topology; dense and
        # sparse cuPIQP both use that full stored pattern. Numerical zeros in
        # changing Jacobians are VALUES, not structural changes. Shape/config
        # keys above and native finite-bound masks identify actual changes.
        for name, value in (("lower_pattern", m.native_lower.isfinite()), ("upper_pattern", m.native_upper.isfinite())):
            valid &= (self.state[name][ids] == value).flatten(1).all(-1)
        accepted = torch.zeros_like(valid)
        s = {name: torch.zeros_like(template) for name, template in
             (("primal", m.p), ("dual", m.h), ("slack", m.h), ("equality_dual", m.b))}
        metrics = {"cache_miss": ~valid, "attempted": valid.clone()}
        rows = valid.nonzero(as_tuple=True)[0]
        if rows.numel():
            try:
                with qp.profiles[qp._diagnostics_phase].measure("active_factor_certification", m.p):
                    candidate, ok, diagnostics = equality_candidate(select_problem(m, rows), self.state["binding"][ids[rows]], cfg)
                for name, value in candidate.items():
                    s[name][rows] = value
                accepted[rows] = ok
            except RuntimeError:
                # A cuSOLVER factor failure rejects the candidate, NOT
                # the original full QP. Still try ordinary cuPIQP before the
                # actuator-only analytic fallback if that full solve fails.
                diagnostics = {"factor_exception": torch.ones_like(rows, dtype=torch.bool),
                               "gap": m.p.new_full(rows.shape, float("nan")),
                               "gap_relative": m.p.new_full(rows.shape, float("nan"))}
            for name, value in diagnostics.items():
                metrics[name] = torch.zeros_like(valid) if value.dtype == torch.bool else m.p.new_full(valid.shape, float("nan"))
                metrics[name][rows] = value
        rejected = (~accepted).nonzero(as_tuple=True)[0]
        gap = m.p.new_full(valid.shape, float("nan"))
        gap_rel = gap.clone()
        if rows.numel():
            gap[rows], gap_rel[rows] = diagnostics["gap"], diagnostics["gap_relative"]
        if rejected.numel():
            part = select_problem(m, rejected)
            G, h, lower, upper = qp._cupiqp_native_pack(part)
            # Ordinary full solve and owned snapshots, not a warm start. An
            # exception is handled row-locally here so certified candidates
            # are not lost when a disjoint rejected batch fails in cuPIQP.
            try:
                result = qp._backend_instances["cupiqp"].solve(
                    part.Q, part.p, G, h, part.A, part.b, differentiable=False,
                    native_lower=lower, native_upper=upper, constant_hessian=False)
                fresh = canonical_snapshot(part, result)
                for name, value in fresh.items():
                    s[name][rejected] = value
                gap[rejected], gap_rel[rejected] = result.duality_gap, result.duality_gap_rel
            except Exception as error:
                from .hard_pact_qp_capture import capture_failure
                capture_failure(qp, error, {"Q": part.Q, "p": part.p, "G": G, "h": h,
                                "A": part.A, "b": part.b, "native_lower": lower, "native_upper": upper},
                                relaxed_contact=False, elastic=False)
                s["primal"][rejected] = float("nan")
                metrics["solver_exception"] = ~accepted
        metrics["accepted"] = accepted
        metrics["full_solve"] = ~accepted
        # Differences from the previous snapshot are not claimed as
        # same-state full-solver parity; the benchmark measures that separately.
        old = self.state["primal"][ids]
        metrics["torque_change_nm"] = ((s["primal"][:, 0:12] - old[:, 0:12]) * m.variable_scale[0:12]).abs().mean(-1)
        def objective(z):
            return .5 * (z * (m.Q @ z[..., None]).squeeze(-1)).sum(-1) + (m.p * z).sum(-1)
        metrics["objective_change_current_data"] = objective(s["primal"]) - objective(old)
        return QPBackendResult(s["primal"], gap, gap_rel), s, metrics
