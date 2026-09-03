"""Numerical backends for the solver-neutral HardPACT QP.

Every backend receives the *same* already-scaled ``(Q,p,G,h,A,b)`` tensors.
It may only solve that problem; physical certification and the three-stage
fallback cascade remain in :mod:`hard_pact_qp`.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
import warnings

import torch


class QPBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested optional backend cannot run."""


@dataclass(frozen=True)
class QPBackendCapability:
    name: str
    available: bool
    reason: str = ""


@dataclass
class QPBackendResult:
    solution: torch.Tensor
    duality_gap: torch.Tensor | None = None
    duality_gap_rel: torch.Tensor | None = None


def backend_capability(name: str, *, device=None, dtype=None):
    """Inspect a backend without importing optional CUDA libraries eagerly."""
    name = name.lower()
    if name == "qpth":
        available = importlib.util.find_spec("qpth") is not None
        return QPBackendCapability(name, available, "" if available else "qpth is not installed")
    if name == "cupiqp":
        if importlib.util.find_spec("cupiqp") is None:
            return QPBackendCapability(name, False, "cuPIQP is not installed")
        if device is not None and torch.device(device).type != "cuda":
            return QPBackendCapability(name, False, "cuPIQP is GPU-only")
        return QPBackendCapability(name, True)
    if name == "moreau":
        if importlib.util.find_spec("moreau") is None:
            return QPBackendCapability(name, False, "Moreau is not installed")
        if device is not None and torch.device(device).type == "cuda" and sys.version_info < (3, 12):
            return QPBackendCapability(
                name, False, "Moreau CUDA wheels require Python 3.12+"
            )
        if (
            device is not None
            and torch.device(device).type == "cuda"
            and not os.environ.get("MOREAU_LICENSE_KEY")
            and not (Path.home() / ".moreau" / "key").is_file()
        ):
            # Moreau's CPU backend is unlicensed, but its CUDA backend aborts
            # at solve time without a key.  Treat that as a capability failure
            # before constructing a QP so it can never masquerade as a
            # numerical failure and enter HardPACT's analytic fallback.
            return QPBackendCapability(
                name, False,
                "Moreau CUDA requires MOREAU_LICENSE_KEY or ~/.moreau/key",
            )
        if dtype is not None and dtype != torch.float64:
            return QPBackendCapability(name, False, "Moreau requires float64")
        return QPBackendCapability(name, True)
    raise ValueError(f"Unknown HardPACT QP solver {name!r}; expected qpth, cupiqp, or moreau")


def require_backend(name: str, *, device, dtype):
    capability = backend_capability(name, device=device, dtype=dtype)
    if not capability.available:
        raise QPBackendUnavailable(
            f"HardPACT QP backend {name!r} is unavailable on {device}/{dtype}: "
            f"{capability.reason}. No solver or CPU fallback was used."
        )
    return capability


def moreau_conic_mapping(A, b, G, h):
    r"""Map ``Az=b, Gz<=h`` to ``Cz+s=d``.

    The first ``A.shape[1]`` rows belong to Moreau's zero cone, forcing
    ``s_eq=0`` and therefore ``Az=b``. Remaining rows use its nonnegative
    cone: ``s_ineq>=0`` and ``Gz+s_ineq=h``, exactly equivalent to ``Gz<=h``.
    """
    return torch.cat((A, G), dim=1), torch.cat((b, h), dim=1), A.shape[1], G.shape[1]


def _as_torch_zero_copy(value, reference):
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    if hasattr(value, "__dlpack__"):
        result = torch.utils.dlpack.from_dlpack(value)
        if result.device != reference.device or result.dtype != reference.dtype:
            raise RuntimeError("optional QP backend returned a mismatched device/dtype")
        return result
    raise TypeError("optional QP backend result does not expose DLPack/Torch storage")


def _as_cupy_zero_copy(value):
    """Expose a contiguous CUDA Torch tensor to CuPy without a host copy."""
    import cupy as cp
    if isinstance(value, torch.Tensor):
        if value.device.type != "cuda":
            raise QPBackendUnavailable("cuPIQP inputs must remain on CUDA")
        # CUDA array-interface access is intentionally blocked by Torch for
        # requires-grad tensors. DLPack on a detached alias is safe here:
        # CuPIQP supplies the corresponding implicit data gradients through
        # CuPIQPFunction.backward, while storage remains shared on the GPU.
        return cp.from_dlpack(value.detach())
    return value


def _cupiqp_dtype(dtype):
    """Translate Torch's dtype without allowing cuPIQP's float64 default."""
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise TypeError(f"cuPIQP supports float32/float64, not {dtype}")


def _configure_cupiqp(solver, config, dtype, *, differentiable):
    """Apply the HardPACT numerical policy to a newly-created cuPIQP solver."""
    prefix = "ppo" if differentiable else "rollout"
    eps_abs = float(getattr(config, f"{prefix}_eps_abs"))
    eps_rel = float(getattr(config, f"{prefix}_eps_rel"))
    gap_abs = float(getattr(config, f"{prefix}_duality_gap_abs"))
    gap_rel = float(getattr(config, f"{prefix}_duality_gap_rel"))
    # cuPIQP warns when HardPACT's configured tolerance is tighter than its
    # generic float32 recommendation. Honor QP verbosity: the shared primal
    # certification remains mandatory and is the actual acceptance gate.
    warning_context = warnings.catch_warnings()
    with warning_context:
        if not config.verbose:
            warnings.simplefilter("ignore", UserWarning)
        solver.settings.eps_abs = eps_abs
        solver.settings.eps_rel = eps_rel
        solver.settings.eps_duality_gap_abs = gap_abs
        solver.settings.eps_duality_gap_rel = gap_rel
    solver.settings.check_duality_gap = (
        getattr(config, f"{prefix}_duality_gap_policy") != "ignore"
    )
    solver.settings.max_iter = int(getattr(config, f"{prefix}_max_iter"))
    # Keep Ruiz/preconditioner state across rollout updates. Mechanics values
    # are still updated below, so this cannot stale M/J-dependent matrices.
    solver.settings.preconditioner_reuse_on_update = not differentiable
    solver.settings.enable_grad = bool(differentiable)
    solver.settings.verbose = bool(config.verbose)
    # cuPIQP 0.1 can capture stable standalone streams, but its captured
    # rollout solver segfaults when interleaved with implicit-backward solver
    # lifetimes. Production alternates those paths every iteration, so graph
    # capture is explicit opt-in until upstream makes that combination safe.
    solver.settings.enable_cuda_graph = bool(config.cupiqp_cuda_graph)


class CuPIQPFunction(torch.autograd.Function):
    """cuPIQP implicit VJP wrapper; one solver owns each outstanding graph."""

    @staticmethod
    def forward(ctx, Q, p, G, h, A, b, x_l, x_u, dense, verbose, config):
        from cupiqp import DenseSolver, SparseSolver
        if not dense:
            # Sparse values require a uniform CSR pack. Keep this explicit
            # instead of silently executing the dense backend under its name.
            raise QPBackendUnavailable(
                "cuPIQP sparse mode requires UniformBatchedCsrMatrix packing; "
                "select cupiqp_dense until a native sparse installation is available"
            )
        solver = DenseSolver(_cupiqp_dtype(Q.dtype))
        _configure_cupiqp(solver, config, Q.dtype, differentiable=True)
        solver.settings.verbose = bool(verbose)
        # Learned objective/constraint entries are often produced by cat or
        # einsum and therefore need not expose dense CUDA strides. cuPIQP's
        # public API explicitly requires dense CUDA arrays; contiguous() is a
        # GPU-to-GPU operation and custom backward returns gradients to the
        # corresponding original arguments in the same logical ordering.
        packed = tuple(
            value.contiguous() for value in (Q, p, G, h, A, b, x_l, x_u)
        )
        Qc, pc, Gc, hc, Ac, bc, xlc, xuc = packed
        Qc, pc, Gc, hc, Ac, bc, xlc, xuc = (
            _as_cupy_zero_copy(value)
            for value in (Qc, pc, Gc, hc, Ac, bc, xlc, xuc)
        )
        solver.setup(
            P=Qc, c=pc, A=Ac, b=bc, G=Gc, h_u=hc,
            x_l=xlc, x_u=xuc,
        )
        solver.solve()
        solution = _as_torch_zero_copy(solver.result.x, Q)
        gap = _as_torch_zero_copy(solver.result.info.duality_gap, p).clone()
        gap_rel = _as_torch_zero_copy(
            solver.result.info.duality_gap_rel, p
        ).clone()
        # cuPIQP reuses backward buffers. Retaining a private solver per graph
        # prevents a later PPO solve from overwriting this factorization.
        ctx.solver = solver
        ctx.inputs = (Q, p, G, h, A, b)
        return solution, gap, gap_rel

    @staticmethod
    def backward(ctx, grad_x, _grad_gap, _grad_gap_rel):
        gradients = ctx.solver.backward(
            grad_x=_as_cupy_zero_copy(grad_x.contiguous())
        )
        Q, p, G, h, A, b = ctx.inputs
        mapped = (
            _as_torch_zero_copy(gradients.P, Q),
            _as_torch_zero_copy(gradients.c, p),
            _as_torch_zero_copy(gradients.G, G),
            _as_torch_zero_copy(gradients.h_u, h),
            _as_torch_zero_copy(gradients.A, A),
            _as_torch_zero_copy(gradients.b, b),
        )
        # Native bounds contain measured state/limits only. Deliberately do
        # not expose their cuPIQP data gradients to the learned graph.
        return (*mapped, None, None, None, None, None)


class SolverBackend:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self._rollout_cache = {}
        self.setup_count = 0
        self.update_count = 0

    def solve(self, Q, p, G, h, A, b, *, differentiable,
              native_lower=None, native_upper=None):
        require_backend(self.name, device=Q.device, dtype=Q.dtype)
        if self.name == "qpth":
            # Import at call time so tests and downstream users can replace the
            # repository-visible qpth function without rebuilding this object.
            from qpth.qp import QPFunction
            eps = self.config.eps
            if eps is None:
                eps = (self.config.eps_float32 if Q.dtype == torch.float32
                       else self.config.eps_float64)
            return QPBackendResult(QPFunction(
                eps=eps,
                verbose=self.config.verbose,
                notImprovedLim=self.config.not_improved_limit,
                maxIter=self.config.max_iter,
                check_Q_spd=self.config.check_q_spd,
            )(Q, p, G, h, A, b))
        if self.name == "cupiqp":
            if differentiable:
                if self.config.cupiqp_mode == "sparse":
                    raise QPBackendUnavailable(
                        "cuPIQP sparse implicit matrix-gradient unpacking is "
                        "not available; use dense for differentiable PPO"
                    )
                solution, gap, gap_rel = CuPIQPFunction.apply(
                    Q, p, G, h, A, b, native_lower, native_upper,
                    self.config.cupiqp_mode == "dense", self.config.verbose,
                    self.config,
                )
                return QPBackendResult(solution, gap, gap_rel)
            return self._solve_cupiqp_rollout(
                Q, p, G, h, A, b, native_lower, native_upper
            )
        return QPBackendResult(self._solve_moreau(
            Q, p, G, h, A, b, differentiable=differentiable,
        ))

    def _solve_cupiqp_rollout(self, Q, p, G, h, A, b, x_l, x_u):
        from cupiqp import DenseSolver, SparseSolver
        from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
        sparse = self.config.cupiqp_mode == "sparse"
        key = (Q.device, Q.dtype, Q.shape[0], Q.shape[1], G.shape[1], A.shape[1])
        profile = (
            self.config.rollout_eps_abs, self.config.rollout_eps_rel,
            self.config.rollout_duality_gap_abs,
            self.config.rollout_duality_gap_rel,
            self.config.rollout_duality_gap_policy,
            self.config.rollout_max_iter,
        )
        key = (sparse,) + key + profile
        solver = self._rollout_cache.get(key)
        packed = None
        if sparse:
            def full_csr(matrix):
                import cupy as cp
                rows, cols = matrix.shape[1:]
                values = _as_cupy_zero_copy(matrix.contiguous()).reshape(
                    matrix.shape[0], rows * cols
                )
                indices = cp.tile(cp.arange(cols, dtype=cp.int32), rows)
                indptr = cp.arange(
                    0, (rows + 1) * cols, cols, dtype=cp.int32
                )
                return UniformBatchedCsrMatrix(
                    matrix.shape[0], indices, indptr, values,
                    shape=(rows, cols), dtype=values.dtype,
                )
            packed = (full_csr(Q), full_csr(A), full_csr(G))
            # SparseData.update assigns into CuPy-owned buffers directly and,
            # unlike the initial setup validator, does not coerce Torch
            # tensors. Keep all vectors zero-copy on CUDA for both setup and
            # cached updates. Holding this tuple through solve also makes the
            # DLPack producer lifetime explicit.
        else:
            # Use the same explicit DLPack path as differentiable cuPIQP.
            # Relying on implicit Torch coercion during DenseSolver.update can
            # leave stale numerical values on some Torch/CuPy combinations.
            packed = tuple(
                _as_cupy_zero_copy(value.contiguous())
                for value in (Q, A, G)
            )
        vector_packed = tuple(
            _as_cupy_zero_copy(value.contiguous())
            for value in (p, b, h, x_l, x_u)
        )
        if solver is None:
            solver = (SparseSolver if sparse else DenseSolver)(
                _cupiqp_dtype(Q.dtype)
            )
            _configure_cupiqp(
                solver, self.config, Q.dtype, differentiable=False,
            )
            P, AA, GG = packed
            pc, bc, hc, xlc, xuc = vector_packed
            solver.setup(
                P=P, c=pc, A=AA, b=bc, G=GG, h_u=hc,
                x_l=xlc, x_u=xuc,
            )
            self._rollout_cache[key] = solver
            self.setup_count += 1
        else:
            _P, AA, GG = packed
            pc, bc, hc, xlc, xuc = vector_packed
            # Q and its sparsity are immutable for a fixed config/cache key.
            # State-dependent A/G and every vector are refreshed each solve.
            solver.update(
                c=pc, A=AA, b=bc, G=GG, h_u=hc,
                x_l=xlc, x_u=xuc,
            )
            self.update_count += 1
        solver.solve()
        return QPBackendResult(
            _as_torch_zero_copy(solver.result.x, Q),
            _as_torch_zero_copy(solver.result.info.duality_gap, p).clone(),
            _as_torch_zero_copy(
                solver.result.info.duality_gap_rel, p
            ).clone(),
        )

    def _solve_moreau(self, Q, p, G, h, A, b, *, differentiable):
        import moreau
        from moreau.torch import Solver
        C, rhs, neq, nineq = moreau_conic_mapping(A, b, G, h)
        batch, n, _ = Q.shape
        m = C.shape[1]
        # Moreau allocates different internal buffers when autograd is
        # enabled. Never reuse a no-grad rollout object for a PPO graph (or
        # vice versa), even when every matrix dimension is identical.
        key = (Q.device, Q.dtype, batch, n, m, bool(differentiable))
        solver = self._rollout_cache.get(key)
        if solver is None:
            # Full fixed CSR patterns are intentional: values vary per state,
            # while their structure never changes across HardPACT solves.
            p_rows = torch.arange(0, n * n + 1, n, device=Q.device)
            p_cols = torch.arange(n, device=Q.device).repeat(n)
            c_rows = torch.arange(0, m * n + 1, n, device=Q.device)
            c_cols = torch.arange(n, device=Q.device).repeat(m)
            cones = moreau.Cones(
                num_zero_cones=neq, num_nonneg_cones=nineq
            )
            settings = moreau.Settings(
                batch_size=batch,
                device=Q.device.type,
                # Moreau 0.3 selects the current CUDA device when device_id is
                # -1. Supplying zero reaches cuDSS's unsupported
                # UBATCH_INDEX configuration on some wheel/driver pairs.
                device_id=-1,
                enable_grad=bool(differentiable),
                max_iter=int(self.config.max_iter),
                verbose=bool(self.config.verbose),
            )
            solver = Solver(
                n=n, m=m, P_row_offsets=p_rows, P_col_indices=p_cols,
                A_row_offsets=c_rows, A_col_indices=c_cols,
                cones=cones, settings=settings,
            )
            self._rollout_cache[key] = solver
            self.setup_count += 1
        else:
            self.update_count += 1
        solution = solver.solve(Q.reshape(batch, -1), C.reshape(batch, -1), p, rhs)
        return solution.x


def create_backend(name, config):
    return SolverBackend(name.lower(), config)
