"""Numerical backends for the solver-neutral HardPACT QP.

Every backend receives the *same* already-scaled ``(Q,p,G,h,A,b)`` tensors.
It may only solve that problem; physical certification and the three-stage
fallback cascade remain in :mod:`hard_pact_qp`.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict, Counter
from contextlib import contextmanager
import weakref
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
    # Owned, unpreconditioned cuPIQP variables. Rollout active-set seeding
    # only: these must never be reused as an implicit-backward context.
    snapshot: dict[str, torch.Tensor] | None = None


class SolverLease:
    """Exclusive solver ownership until its autograd context is destroyed.

    Never release from backward: PCGrad/retain_graph can call it repeatedly.
    The weak finalizer does not reference ctx, so abandoned graphs also return
    their leases. Failed solvers are discarded rather than returned to idle.
    """
    def __init__(self, pool, key, solver, pooled):
        self.pool, self.key, self.solver = pool, key, solver
        self.pooled, self.healthy = pooled, True

    def release(self):
        if self.solver is not None:
            self.pool.release(self)
            self.solver = None


class SolverPool:
    """Bound total pooled instances; overflow graphs use fresh private solvers."""
    def __init__(self, limit):
        self.limit = max(0, int(limit))
        self.idle = OrderedDict()
        self.size = 0
        self.serial = 0

    def acquire(self, key, factory):
        for token, (cached_key, solver) in list(self.idle.items()):
            if cached_key == key:
                del self.idle[token]
                return SolverLease(self, key, solver, True), True
        if self.size >= self.limit and self.idle:
            self.idle.popitem(last=False)  # Only idle leases may be evicted.
            self.size -= 1
        pooled = self.size < self.limit
        solver = factory()
        if pooled:
            self.size += 1
        return SolverLease(self, key, solver, pooled), False

    def release(self, lease):
        if lease.pooled:
            if lease.healthy:
                self.serial += 1
                self.idle[self.serial] = (lease.key, lease.solver)
            else:
                self.size -= 1


class CUDAEventProfile:
    """Opt-in queued events; synchronization happens only at iteration export."""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.events = []

    @contextmanager
    def measure(self, name, reference):
        token = self.begin(reference)
        try:
            yield
        finally:
            self.end(name, token)

    def begin(self, reference):
        if not self.enabled or reference.device.type != "cuda":
            return None
        stream = torch.cuda.current_stream(reference.device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        return start, end, stream

    def end(self, name, token):
        if token is not None:
            start, end, stream = token
            end.record(stream)
            self.events.append((name, start, end))

    def finalize(self):
        result = Counter()
        # Only opt-in callers pay for completion. Never synchronize per chunk.
        for name, start, end in self.events:
            end.synchronize()
            result[name] += start.elapsed_time(end)
        self.events.clear()
        return dict(result)


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
        getattr(config, f"{prefix}_duality_gap_policy") == "require"
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
    def forward(ctx, Q, p, G, h, A, b, x_l, x_u, dense, verbose, config, backend=None):
        from cupiqp import DenseSolver, SparseSolver
        if not dense:
            # Sparse values require a uniform CSR pack. Keep this explicit
            # instead of silently executing the dense backend under its name.
            raise QPBackendUnavailable(
                "cuPIQP sparse mode requires UniformBatchedCsrMatrix packing; "
                "select cupiqp_dense until a native sparse installation is available"
            )
        # Learned objective/constraint entries are often produced by cat or
        # einsum and therefore need not expose dense CUDA strides. cuPIQP's
        # public API explicitly requires dense CUDA arrays; contiguous() is a
        # GPU-to-GPU operation and custom backward returns gradients to the
        # corresponding original arguments in the same logical ordering.
        backend = backend or SolverBackend("cupiqp", config)
        phase = backend.diagnostics_phase
        profile = backend.profiles[phase]
        def factory():
            solver = DenseSolver(_cupiqp_dtype(Q.dtype))
            _configure_cupiqp(solver, config, Q.dtype, differentiable=True)
            solver.settings.verbose = bool(verbose)
            return solver
        key = (Q.device, Q.dtype, tuple(v.shape for v in (Q, p, G, h, A, b)),
               tuple(None if v is None else v.shape for v in (x_l, x_u)),
               dense, verbose, repr(config), torch.cuda.current_stream(Q.device).cuda_stream)
        pool = backend._ppo_pool
        if not getattr(config, "cupiqp_ppo_reuse", True):
            pool = SolverPool(0)  # fresh-instance numerical reference
        lease, hit = pool.acquire(key, factory)
        backend.record(phase, pool_hits=int(hit), pool_misses=int(not hit),
                       requested_rows=Q.shape[0], capacity_rows=Q.shape[0])
        # One finalizer for the whole ctx lifetime, not one per backward use.
        weakref.finalize(ctx, lease.release)
        ctx.lease, ctx.profile = lease, profile
        ctx.inputs = tuple(v.detach() for v in (Q, p, G, h, A, b))
        # Put cuPIQP/Warp work on Torch's stream, also making CUDA event
        # profiling and asynchronous lease reuse correctly ordered.
        import cupy as cp
        with cp.cuda.Device(Q.device.index), cp.cuda.ExternalStream(torch.cuda.current_stream(Q.device).cuda_stream):
            try:
                with profile.measure("packing", Q):
                    packed = tuple(None if v is None else _as_cupy_zero_copy(v.contiguous())
                                   for v in (Q, p, G, h, A, b, x_l, x_u))
                    Qc, pc, Gc, hc, Ac, bc, xlc, xuc = packed
                kwargs = dict(P=Qc, c=pc, A=Ac, b=bc, G=Gc, h_u=hc, x_l=xlc, x_u=xuc)
                try:
                    with profile.measure("setup_update", Q):
                        if hit:
                            # Fresh Ruiz scaling, exactly the PPO numerical
                            # profile. ALL changing matrices/bounds refresh.
                            lease.solver.update(**kwargs)
                            backend.record(phase, update_count=1)
                        else:
                            lease.solver.setup(**kwargs)
                            backend.record(phase, setup_count=1)
                    with profile.measure("solve", Q):
                        lease.solver.solve()
                except Exception:
                    if not hit:
                        raise
                    # An update/solve failure invalidates this instance; retry
                    # the identical QP once with a private fresh solver.
                    lease.healthy = False
                    lease.release()
                    lease, _ = SolverPool(0).acquire(key, factory)
                    ctx.lease = lease
                    weakref.finalize(ctx, lease.release)
                    with profile.measure("setup_update", Q):
                        lease.solver.setup(**kwargs)
                    backend.record(phase, setup_count=1, reuse_exception_fresh_retry=1)
                    with profile.measure("solve", Q):
                        lease.solver.solve()
                solution = _as_torch_zero_copy(lease.solver.result.x, Q).clone()
                gap = _as_torch_zero_copy(lease.solver.result.info.duality_gap, p).clone()
                gap_rel = _as_torch_zero_copy(lease.solver.result.info.duality_gap_rel, p).clone()
                backend.record_iterations(phase, lease.solver, p, Q.shape[0])
            except Exception:
                ctx.lease.healthy = False
                ctx.lease.release()
                raise
        ctx.mark_non_differentiable(gap, gap_rel)
        return solution, gap, gap_rel

    @staticmethod
    def backward(ctx, grad_x, _grad_gap, _grad_gap_rel):
        Q, p, G, h, A, b = ctx.inputs
        import cupy as cp
        with cp.cuda.Device(Q.device.index), cp.cuda.ExternalStream(torch.cuda.current_stream(Q.device).cuda_stream):
            try:
                with ctx.profile.measure("backward", Q):
                    gradients = ctx.lease.solver.backward(
                        grad_x=_as_cupy_zero_copy(grad_x.contiguous())
                    )
                    # Each VJP must own its output: PCGrad can retain these
                    # while another backward overwrites cuPIQP's scratch space.
                    mapped = tuple(_as_torch_zero_copy(value, ref).clone() for value, ref in zip(
                        (gradients.P, gradients.c, gradients.G, gradients.h_u, gradients.A, gradients.b),
                        (Q, p, G, h, A, b),
                    ))
            except Exception:
                ctx.lease.healthy = False
                raise
        # Native bounds contain measured state/limits only. Deliberately do
        # not expose their cuPIQP data gradients to the learned graph.
        return (*mapped, None, None, None, None, None, None)[:len(ctx.needs_input_grad)]


class SolverBackend:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self._rollout_cache = OrderedDict()
        self._rollout_hessians = {}
        self.setup_count = 0
        self.update_count = 0
        self._ppo_pool = SolverPool(getattr(config, "cupiqp_ppo_pool_size", 8))
        self.diagnostics_phase = "ppo"
        self.stats = {phase: Counter() for phase in ("rollout", "ppo")}
        self.profiles = {phase: CUDAEventProfile(getattr(config, "cuda_event_profiling", False))
                         for phase in self.stats}

    def record(self, phase, **counts):
        self.stats[phase].update(counts)

    def record_iterations(self, phase, solver, reference, count):
        iterations = getattr(solver.result.info, "iter", None)
        if iterations is not None:
            # cuPIQP 0.1 exposes iteration counts as an already-host-resident
            # NumPy array. Reading it adds no GPU transfer/synchronization.
            if not isinstance(iterations, torch.Tensor) and not hasattr(iterations, "__cuda_array_interface__"):
                self.record(phase, iteration_sum=float(iterations[:count].sum()), iteration_rows=count)
                return
            values = _as_torch_zero_copy(iterations, reference).reshape(-1)[:count]
            self.record(phase, iteration_sum=values.detach().sum(), iteration_rows=values.numel())

    def solve(self, Q, p, G, h, A, b, *, differentiable,
              native_lower=None, native_upper=None, constant_hessian=False):
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
                    self.config, self,
                )
                return QPBackendResult(solution, gap, gap_rel)
            return self._solve_cupiqp_rollout(
                Q, p, G, h, A, b, native_lower, native_upper,
                constant_hessian=constant_hessian,
            )
        return QPBackendResult(self._solve_moreau(
            Q, p, G, h, A, b, differentiable=differentiable,
        ))

    def _solve_cupiqp_rollout(self, Q, p, G, h, A, b, x_l, x_u,
                             *, constant_hessian=False):
        import cupy as cp
        with cp.cuda.Device(Q.device.index), cp.cuda.ExternalStream(torch.cuda.current_stream(Q.device).cuda_stream):
            return self._solve_cupiqp_rollout_bucket(
                Q, p, G, h, A, b, x_l, x_u, constant_hessian=constant_hessian,
            )

    def _solve_cupiqp_rollout_bucket(self, Q, p, G, h, A, b, x_l, x_u,
                                    *, constant_hessian=False):
        from cupiqp import DenseSolver, SparseSolver
        from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
        sparse = self.config.cupiqp_mode == "sparse"
        batch = Q.shape[0]
        phase = self.diagnostics_phase
        profile_timer = self.profiles[phase]
        source_Q = Q.detach()  # cache ownership never retains an autograd graph
        capacity = (1 << (batch - 1).bit_length()
                    if self.config.cupiqp_rollout_capacity_reuse else batch)
        # Never pad beyond the configured rollout chunk budget (important
        # for non-power-of-two VRAM-tuned limits such as 8000). Direct backend
        # callers may supply larger B; keep those exact rather than shrinking.
        chunk_limit = self.config.chunk_size or self.config.rollout_chunk_size
        capacity = max(batch, min(capacity, int(chunk_limit)))
        key = (Q.device, Q.dtype, capacity, Q.shape[1], G.shape[1], A.shape[1])
        profile = (
            self.config.rollout_eps_abs, self.config.rollout_eps_rel,
            self.config.rollout_duality_gap_abs,
            self.config.rollout_duality_gap_rel,
            self.config.rollout_duality_gap_policy,
            self.config.rollout_max_iter,
            self.config.cupiqp_cuda_graph, self.config.verbose,
            torch.cuda.current_stream(Q.device).cuda_stream,
        )
        key = (sparse,) + key + profile
        # Smallest fitting power-of-two bucket, capped by chunk size. Never
        # promote a small request to a previously seen large batch capacity.
        solver = self._rollout_cache.get(key)
        self.record(phase, pool_hits=int(solver is not None), pool_misses=int(solver is None),
                    requested_rows=batch, capacity_rows=capacity, padded_rows=capacity-batch)
        packing_token = profile_timer.begin(Q)
        if capacity != batch:
            def pad(value):
                if value is None:
                    return None
                value = value.detach()
                if value.stride(0) == 0:
                    return value[:1].expand(capacity, *value.shape[1:])
                # Independent duplicate QPs introduce no new worst residual
                # or infeasibility. No CPU handoff or training RNG is used.
                return torch.cat((value, value[-1:].expand(
                    capacity - batch, *value.shape[1:]
                )), dim=0)
            Q, p, G, h, A, b, x_l, x_u = map(pad, (Q, p, G, h, A, b, x_l, x_u))
        previous_Q = self._rollout_hessians.get(key)
        # Only the builder may promise an immutable Q. Elastic recovery has
        # Q += 2*w*A^T*A, so its Hessian MUST be updated with the new mechanics.
        # Retain the previous tensor to prevent storage-address recycling.
        reuse_Q = (constant_hessian and previous_Q is not None
                   and source_Q.data_ptr() == previous_Q.data_ptr()
                   and source_Q.stride()[1:] == previous_Q.stride()[1:])
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
            packed = (None if reuse_Q else full_csr(Q), full_csr(A), full_csr(G))
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
                None if index == 0 and reuse_Q else
                _as_cupy_zero_copy(value.contiguous())
                for index, value in enumerate((Q, A, G))
            )
        vector_packed = tuple(
            _as_cupy_zero_copy(value.contiguous())
            for value in (p, b, h, x_l, x_u)
        )
        profile_timer.end("packing", packing_token)
        if solver is None:
            # Rollout results own copies and have no graph. Only completed
            # host calls are cached, so LRU entries here are all idle. Same-
            # stream keys keep queued GPU work ordered without host waits.
            limit = max(1, int(getattr(self.config, "cupiqp_rollout_cache_size", 4)))
            while len(self._rollout_cache) >= limit:
                cached, evicted_solver = self._rollout_cache.popitem(last=False)
                self._rollout_hessians.pop(cached, None)
                del evicted_solver  # release its capacity before allocating another
            solver = (SparseSolver if sparse else DenseSolver)(
                _cupiqp_dtype(Q.dtype)
            )
            _configure_cupiqp(
                solver, self.config, Q.dtype, differentiable=False,
            )
            P, AA, GG = packed
            pc, bc, hc, xlc, xuc = vector_packed
            with profile_timer.measure("setup_update", Q):
                solver.setup(
                    P=P, c=pc, A=AA, b=bc, G=GG, h_u=hc,
                    x_l=xlc, x_u=xuc,
                )
            self.setup_count += 1
            self.record(phase, setup_count=1)
        else:
            P, AA, GG = packed
            pc, bc, hc, xlc, xuc = vector_packed
            # Elastic A^T*A can change the Hessian's conditioning by orders
            # of magnitude. Reuse allocations, but recompute Ruiz scaling
            # for a changing Hessian. Constant full/relaxed Q keeps the fast
            # preconditioner-reuse path. This avoids history-dependent early
            # termination from an obsolete elastic preconditioner.
            solver.settings.preconditioner_reuse_on_update = bool(constant_hessian)
            # A/G and bounds always refresh. P=None skips copying/reprocessing
            # the immutable shared Hessian, not a state-dependent elastic Q.
            # Remove while in use; an update/solve exception cannot leave a
            # poisoned entry available for the next request.
            self._rollout_cache.pop(key)
            self._rollout_hessians.pop(key, None)
            with profile_timer.measure("setup_update", Q):
                solver.update(
                    P=P, c=pc, A=AA, b=bc, G=GG, h_u=hc,
                    x_l=xlc, x_u=xuc,
                )
            self.update_count += 1
            self.record(phase, update_count=1)
        with profile_timer.measure("solve", Q):
            solver.solve()
        self._rollout_cache[key] = solver
        self._rollout_hessians[key] = source_Q if constant_hessian else None
        self.record_iterations(phase, solver, p, batch)
        return QPBackendResult(
            # The solver owns mutable buffers. Consumers may hold an earlier
            # result while this same capacity is reused; return owned slices.
            _as_torch_zero_copy(solver.result.x, Q)[:batch].clone(),
            _as_torch_zero_copy(solver.result.info.duality_gap, p)[:batch].clone(),
            _as_torch_zero_copy(
                solver.result.info.duality_gap_rel, p
            )[:batch].clone(),
            None,  # No rollout active-set snapshots/factors are retained.
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
