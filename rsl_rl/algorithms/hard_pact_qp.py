"""HardPACT 24-variable torque/world-force QP with eliminated acceleration."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Mapping

import torch
from qpth.qp import QPFunction
from .hard_pact_qp_backends import (
    QPBackendUnavailable,
    backend_capability,
    create_backend,
    require_backend,
)
from .qpth_warm_start import solve_qpth_warm
from .hard_pact_qp_capture import capture_failure


# Fixed slices make every Q/P/G/A block visibly correspond to one physical
# variable group. Keeping these compile-time fixed also avoids ragged QPs.
TORQUE = slice(0, 12)
FORCE = slice(12, 24)
NUM_VARIABLES = 24


def qp_substep_anchors(mode, decimation):
    """Possible dispatch times, not per-environment solve counts."""
    if mode not in ("every_substep", "random_one_substep"):
        raise ValueError(f"Unsupported qp_update_mode: {mode}")
    if decimation != 4:
        raise ValueError("HardPACT QP execution requires exactly four physics substeps")
    return (0,1,2,3)


def qp_substep_mask(mode, substep, selected):
    qp_substep_anchors(mode,4)
    return torch.ones_like(selected,dtype=torch.bool) if mode=="every_substep" else selected==substep


def project_torque_interval(torque, lower, upper):
    """Ordinary exact actuator/rate clamp, not a joint/contact certificate."""
    return torch.maximum(torch.minimum(torque, upper), lower)


def project_nominal_torque(tau_nom, previous_torque, torque_limit, torque_rate_limit, dt):
    """No held correction: project sanitized fresh total PD/feedforward torque."""
    previous = torch.nan_to_num(previous_torque,nan=0.,posinf=0.,neginf=0.)
    lower = torch.maximum(-torque_limit,previous-torque_rate_limit*dt)
    upper = torch.minimum(torque_limit,previous+torque_rate_limit*dt)
    empty = lower>upper
    lower,upper = torch.where(empty,-torque_limit,lower),torch.where(empty,torque_limit,upper)
    return project_torque_interval(torch.nan_to_num(tau_nom,nan=0.,posinf=0.,neginf=0.),lower,upper)


@dataclass(frozen=True)
class HardPACTQPConfig:
    """Numerics and physical weights exposed by the HardPACT config.

    Scales define ``D`` in ``x=Dz``. Tracking/regularization values define
    terms in the physical objective before that substitution. Solver controls
    are passed directly to :class:`qpth.qp.QPFunction`.
    """

    enabled: bool = True  # Master rollout/PPO projection switch.
    # Train normally without rollout/replay QPs until this absolute PPO
    # iteration. Zero preserves immediate projection; independent of PINN.
    warmup_iterations: int = 0
    soft_joint_recovery_enabled: bool = True
    soft_joint_recovery_weight: float = 200.0
    soft_joint_recovery_scale_rad_s2: float = 100.0
    qp_update_mode: str = "random_one_substep"
    # One canonical QP and fallback cascade can be solved by any registered
    # numerical backend. Overrides are diagnostics-only and require an
    # explicit mismatch opt-in so rollout/training cannot diverge silently.
    qp_solver: str = "qpth"
    rollout_qp_solver: str | None = None
    ppo_qp_solver: str | None = None
    allow_solver_mismatch: bool = False
    cupiqp_mode: str = "dense"  # dense | sparse
    # cuPIQP 0.1's CUDA graph is not safe when a cached rollout solver is
    # interleaved with short-lived implicit-backward solvers. Keep the stable
    # path as default; users may benchmark graph capture explicitly.
    cupiqp_cuda_graph: bool = False
    # Bounded reuse; fresh PPO instances remain an explicit reference mode.
    cupiqp_rollout_capacity_reuse: bool = True
    cupiqp_rollout_cache_size: int = 4
    cupiqp_ppo_reuse: bool = True
    cupiqp_ppo_pool_size: int = 8
    cuda_event_profiling: bool = False
    # Independent numerical policies. ``None`` preserves the legacy scalar
    # setting, while the defaults below encode the measured training intent:
    # rollout favours throughput and reports its gap; PPO is stricter and
    # requires a trustworthy gap before retaining an implicit VJP.
    rollout_eps_abs: float = 1.0e-4
    rollout_eps_rel: float = 1.0e-4
    rollout_max_iter: int = 20
    rollout_feasibility_tolerance: float = 1.0e-3
    rollout_duality_gap_abs: float = 1.0e-3
    rollout_duality_gap_rel: float = 1.0e-3
    rollout_duality_gap_policy: str = "report"
    ppo_eps_abs: float = 3.0e-6
    ppo_eps_rel: float = 3.0e-6
    ppo_max_iter: int = 30
    ppo_feasibility_tolerance: float = 1.0e-3
    ppo_duality_gap_abs: float = 3.0e-6
    ppo_duality_gap_rel: float = 3.0e-6
    ppo_duality_gap_policy: str = "require"
    qpth_warm_start: bool = False
    friction_coefficient: float = 0.6  # mu in |fx|,|fy|<=mu*fz.
    torque_rate_limit_nm_s: float = 1000.0  # dot(tau)_lim [Nm/s].
    contact_threshold: float = 0.5
    contact_acceleration_weight: float = 1.0
    contact_acceleration_scale_m_s2: float = 50.0
    attitude_weight: float = 1.0
    attitude_acceleration_scale_rad_s2: float = 20.0
    attitude_kp: float = 20.0
    attitude_kd: float = 5.0
    joint_acceleration_limits_rad_s2: tuple = (100.0,) * 12
    interior_margin: float = 1.0e-3  # Strict-feasibility epsilon for contact rows.
    force_scale_n: float = 250.0  # D diagonal/reference normalization for GRF.
    torque_scale_nm: float = 40.0  # D diagonal for safe torque variables.
    torque_tracking_weight: float = 20.0  # w_tau.
    force_tracking_weight: float = 1.0  # w_f.
    q_regularization: float = 1.0e-7  # r_Q and final solver-space SPD ridge.
    # Deprecated scalar overrides remain accepted for checkpoint/config
    # compatibility; None selects the dtype-aware values below.
    feasibility_tolerance: float | None = None
    normalized_feasibility_tolerance_float32: float = 1.0e-3
    normalized_feasibility_tolerance_float64: float = 1.0e-6
    kkt_tolerance: float = 1.0e-1  # Relative dual/complementarity threshold.
    # qpth returns the best finite interior iterate; 0.1 is used only to infer
    # its numerical active set for post-solve KKT diagnostics.
    active_tolerance: float = 1.0e-1  # -r_ineq cutoff for diagnostic active set.
    eps: float | None = None  # Deprecated all-dtype override.
    eps_float32: float = 1.0e-5
    eps_float64: float = 1.0e-9
    max_iter: int = 30  # qpth interior-point iteration cap.
    not_improved_limit: int = 6  # qpth stagnation cap.
    check_q_spd: bool = True  # Validate Q>0 and ask qpth to do likewise.
    check_equality_rank: bool = True  # Require full-row-rank A.
    # "auto" uses float32 on CUDA to halve the retained KKT graph and float64
    # on CPU for reference tests; either concrete dtype can be forced.
    solver_dtype: str = "auto"
    # User-facing verbosity: 0 is quiet; positive values expose qpth's solver
    # diagnostics. qpth itself treats verbose=0 as permission to print its
    # large inaccurate-solution warning, so the call adapter maps 0 to -1.
    verbose: int = 0
    exception_capture_enabled: bool = True
    exception_capture_limit: int = 1  # Per solver instance, not per substep.
    exception_capture_dir: str = "/tmp/hard_pact_qp_failures"
    # Diagnostics never participate in acceptance or fallback decisions.
    # ``minimal`` is the production default; ``physical`` adds detached
    # physical-unit summaries; ``full`` periodically enables sampled matrix,
    # KKT, timing, memory, and gradient audits.
    diagnostics_level: str = "minimal"
    tensorboard_diagnostics_enabled: bool = True
    tensorboard_diagnostics_interval: int = 50  # Absolute PPO iterations.
    full_audit_period: int = 1000
    full_audit_sample_size: int = 8
    # Legacy chunk_size overrides both paths when not None.
    chunk_size: int | None = None
    rollout_chunk_size: int = 512
    ppo_chunk_size: int = 128
    # Deprecated aliases are accepted so older experiment configs still load.
    # When set, they override the corresponding full-audit value above.
    debug_audit_period: int | None = None
    debug_audit_sample_size: int | None = None
    # Genesis and PhysX use semi-implicit Euler: q+=dt*v+dt^2*qdd. A backend
    # with constant-acceleration position integration may configure 0.5.
    position_integration_coefficient: float = 1.0
    gradient_scale_tau: float = 1.0
    gradient_scale_grf: float = 1.0
    gradient_scale_wrench: float = 1.0
    gradient_scale_contact: float = 1.0
    gradient_clip_tau: float = 0.0
    gradient_clip_grf: float = 0.0
    gradient_clip_wrench: float = 0.0
    gradient_clip_contact: float = 0.0

    @classmethod
    def from_dict(cls, values):
        """Reject incompatible formulation settings rather than silently migrate."""
        unknown = set(values)-{field.name for field in fields(cls)}
        if unknown:
            raise ValueError("Incompatible HardPACT QP metadata; re-export current 24-D settings. Unknown keys: "
                             + ", ".join(sorted(unknown)))
        return cls(**dict(values))

    def __post_init__(self):
        qp_substep_anchors(self.qp_update_mode,4)
        if self.cupiqp_rollout_cache_size < 1 or self.cupiqp_ppo_pool_size < 0:
            raise ValueError("cuPIQP cache size must be positive and pool size nonnegative")
        if (self.warmup_iterations < 0
                or int(self.warmup_iterations) != self.warmup_iterations):
            raise ValueError("QP warmup_iterations must be a nonnegative integer")


@dataclass
class HardPACTQPResult:
    """Physical torque/force solution and derived acceleration; stage 2 is uncertified fallback."""
    qdd: torch.Tensor  # [B,18], canonical generalized acceleration.
    force_world: torch.Tensor  # [B,4,3], FR/FL/RR/RL world XYZ [N].
    tau_safe: torch.Tensor  # [B,12], actuator command [Nm].
    stage: torch.Tensor  # [B], 0=hard certified, 1=soft-joint recovery, 2=analytic.
    differentiated_mask: torch.Tensor  # [B], true exactly for certified rows.
    diagnostics: Mapping[str, torch.Tensor]  # Per-stage primal/KKT metrics.
    metrics: Mapping[str, torch.Tensor] | None = None  # Aggregated GPU scalars.


@dataclass
class _QPBuild:
    """Scaled qpth problem plus its exact physical-space counterpart."""

    Q: torch.Tensor
    p: torch.Tensor
    G: torch.Tensor
    h: torch.Tensor
    A: torch.Tensor
    b: torch.Tensor
    variable_scale: torch.Tensor
    physical_G: torch.Tensor
    physical_h: torch.Tensor
    physical_A: torch.Tensor
    physical_b: torch.Tensor
    equality_row_scale: torch.Tensor
    inequality_row_scale: torch.Tensor
    tau_lower: torch.Tensor
    tau_upper: torch.Tensor
    qdd_lower: torch.Tensor
    qdd_upper: torch.Tensor
    native_lower: torch.Tensor
    native_upper: torch.Tensor
    acceleration_map: torch.Tensor
    acceleration_offset: torch.Tensor
    mechanics_valid: torch.Tensor

    def __iter__(self):
        # Preserve the legacy seven-value private test/debug unpacking API.
        return iter((self.Q, self.p, self.G, self.h, self.A, self.b,
                     self.variable_scale))


def select_problem(m, rows):
    """Index only batch-dependent mechanics/matrices; share immutable scales."""
    return replace(m, **{f.name:getattr(m,f.name).index_select(0,rows)
                         for f in fields(m) if f.name!="variable_scale"})


def _dtype_from_name(name: str, reference=None):
    """Resolve solver precision without changing the public tensor precision."""
    normalized = str(name).lower()
    if normalized == "auto":
        device = torch.as_tensor(reference).device
        return torch.float32 if device.type == "cuda" else torch.float64
    if normalized in ("float64", "double", "torch.float64"):
        return torch.float64
    if normalized in ("float32", "float", "torch.float32"):
        return torch.float32
    raise ValueError("QP solver_dtype must be auto, float32, or float64")


def _row_scale(matrix, rhs):
    r"""Scale each row by ``max(||row||_2, |rhs|, 1)`` without changing it."""
    scale = torch.maximum(
        matrix.detach().square().sum(dim=-1).sqrt(), rhs.detach().abs()
    ).clamp_min(1.0)
    return matrix / scale.unsqueeze(-1), rhs / scale, scale


class _CertifiedRows(torch.autograd.Function):
    """Identity in forward; prevent failed rows entering an implicit VJP."""

    @staticmethod
    def forward(ctx, value, certified):
        ctx.save_for_backward(certified)
        return value

    @staticmethod
    def backward(ctx, gradient):
        (certified,) = ctx.saved_tensors
        return gradient * certified.to(gradient.dtype).unsqueeze(-1), None


class _ScaleClipRows(torch.autograd.Function):
    """Value-preserving per-environment physical-input gradient conditioner."""

    @staticmethod
    def forward(ctx, value, scale, maximum_norm, sink=None, name="input"):
        ctx.scale = float(scale)
        ctx.maximum_norm = float(maximum_norm)
        ctx.sink = sink
        ctx.name = str(name)
        return value

    @staticmethod
    def backward(ctx, gradient):
        raw_norm = gradient.reshape(gradient.shape[0], -1).norm(dim=-1)
        scaled = gradient * ctx.scale
        scaled_norm = scaled.reshape(scaled.shape[0], -1).norm(dim=-1)
        clipped_rows = torch.zeros_like(raw_norm, dtype=torch.bool)
        if ctx.maximum_norm > 0.0:
            flat = scaled.reshape(scaled.shape[0], -1)
            factor = (ctx.maximum_norm / flat.norm(dim=-1).clamp_min(1.0e-12)).clamp_max(1.0)
            clipped_rows = factor < 1.0
            scaled = scaled * factor.reshape((-1,) + (1,) * (scaled.ndim - 1))
        if ctx.sink is not None:
            clipped_norm = scaled.reshape(scaled.shape[0], -1).norm(dim=-1)
            prefix = f"qp/gradient/{ctx.name}"
            ctx.sink._last_gradient_metrics.update({
                f"{prefix}/raw_norm": raw_norm.detach().mean(),
                f"{prefix}/scaled_norm": scaled_norm.detach().mean(),
                f"{prefix}/clipped_norm": clipped_norm.detach().mean(),
                f"{prefix}/clipped_fraction": clipped_rows.float().mean().detach(),
            })
        return scaled, None, None, None, None


class HardPACTDifferentiableQP:
    """Shared 24-variable solver. Mechanics are detached; learned references retain VJPs."""

    def __init__(self, config: HardPACTQPConfig, torque_limits,
                 position_lower, position_upper, velocity_limits):
        # Store the immutable numerical/weight specification used by every
        # rollout and PPO solve in this training run.
        self.cfg = config
        # Backend limits are labels, never optimization variables. Detaching
        # prevents an accidental gradient edge if a caller supplies tensors.
        self.torque_limits = torch.as_tensor(torque_limits).reshape(12).detach()
        # "auto" is intentionally not resolved here: limits may be created on
        # CPU while the live learned torque arrives on CUDA. solve() resolves
        # from tau_nom on every call, preventing an accidental float64 GPU QP.
        self.solver_dtype = None
        # Joint boxes are canonical simulator order: FR, FL, RR, RL, three
        # actuated coordinates per leg.
        self.position_lower = torch.as_tensor(position_lower).reshape(12).detach()
        self.position_upper = torch.as_tensor(position_upper).reshape(12).detach()
        self.velocity_limits = torch.as_tensor(velocity_limits).reshape(12).detach()
        if min(config.soft_joint_recovery_weight, config.soft_joint_recovery_scale_rad_s2) <= 0:
            raise ValueError("soft-joint recovery weight and scale must be positive")
        # D must be invertible, hence every variable scale is strictly positive.
        if min(config.force_scale_n, config.torque_scale_nm,
               config.contact_acceleration_scale_m_s2, config.attitude_acceleration_scale_rad_s2) <= 0:
            raise ValueError("QP physical scales must be positive")
        if min(config.torque_tracking_weight, config.force_tracking_weight) <= 0:
            raise ValueError("torque and force tracking weights must be positive")
        if min(config.contact_acceleration_weight, config.attitude_weight,
               config.attitude_kp, config.attitude_kd) < 0:
            raise ValueError("soft objective weights and gains must be nonnegative")
        if len(config.joint_acceleration_limits_rad_s2) != 12 or min(config.joint_acceleration_limits_rad_s2) <= 0:
            raise ValueError("12 positive canonical joint acceleration limits required")
        if not 0 <= config.contact_threshold <= 1:
            raise ValueError("contact_threshold must be in [0,1]")
        if min(config.friction_coefficient, config.torque_rate_limit_nm_s, config.q_regularization) <= 0:
            raise ValueError("friction, torque rate and SPD ridge must be positive")
        if ((config.eps is not None and config.eps <= 0.0)
                or config.eps_float32 <= 0.0 or config.eps_float64 <= 0.0
                or config.max_iter <= 0):
            raise ValueError("QP eps and max_iter must be positive")
        chunk_sizes = (config.rollout_chunk_size, config.ppo_chunk_size)
        if config.chunk_size is not None:
            chunk_sizes = (config.chunk_size, config.chunk_size)
        if config.not_improved_limit <= 0 or min(chunk_sizes) <= 0:
            raise ValueError("QP iteration and chunk limits must be positive")
        if config.position_integration_coefficient not in (0.5, 1.0):
            raise ValueError("position integration coefficient must be 0.5 or 1.0")
        if config.diagnostics_level not in ("minimal", "physical", "full"):
            raise ValueError(
                "QP diagnostics_level must be minimal, physical, or full"
            )
        solvers = {
            config.qp_solver,
            config.rollout_qp_solver or config.qp_solver,
            config.ppo_qp_solver or config.qp_solver,
        }
        if not solvers <= {"qpth", "cupiqp", "moreau"}:
            raise ValueError("QP solver must be qpth, cupiqp, or moreau")
        if config.cupiqp_mode not in ("dense", "sparse"):
            raise ValueError("cupiqp_mode must be dense or sparse")
        qp_substep_anchors(config.qp_update_mode,4)
        for policy in (
            config.rollout_duality_gap_policy, config.ppo_duality_gap_policy,
        ):
            if policy not in ("ignore", "report", "require"):
                raise ValueError("duality_gap_policy must be ignore, report, or require")
        if len(solvers) > 1 and not config.allow_solver_mismatch:
            raise ValueError(
                "different rollout/PPO QP solvers require "
                "allow_solver_mismatch=True"
            )
        if self._full_audit_period(config) < 0:
            raise ValueError("QP full_audit_period must be nonnegative")
        if config.tensorboard_diagnostics_interval < 0:
            raise ValueError("QP tensorboard_diagnostics_interval must be nonnegative")
        if self._full_audit_sample_size(config) <= 0:
            raise ValueError("QP full_audit_sample_size must be positive")
        self._constant_cache = {}
        self._limit_cache = {}
        self._solve_count = 0
        self._backend_config = config
        self._backend_instances = {
            name: create_backend(name, config) for name in solvers
            if name != "qpth"
        }
        self._active_solver = config.qp_solver
        self._active_differentiable = False
        # Rollout-only qpth terminal (primal, equality dual, inequality dual,
        # slack) states plus a validity bit for every environment row. Keys
        # identify stable chunks, while the mask prevents a reset in one row
        # from discarding or contaminating any other environment's state.
        # PPO remains cold so no rollout iterate enters an unrelated minibatch.
        self._qpth_warm_states = {}
        self._last_gradient_metrics = {}

        from .hard_pact_qp_diagnostics import QPIterationDiagnostics
        from .hard_pact_qp_backends import CUDAEventProfile
        self.iteration_diagnostics = {phase: QPIterationDiagnostics() for phase in ("rollout", "ppo")}
        self.profiles = {phase: CUDAEventProfile(config.cuda_event_profiling) for phase in self.iteration_diagnostics}
        self._diagnostics_phase = "rollout"

    def begin_iteration_diagnostics(self, phase):
        from .hard_pact_qp_diagnostics import QPIterationDiagnostics
        self.iteration_diagnostics[phase] = QPIterationDiagnostics()
        self.profiles[phase].events.clear()
        for backend in self._backend_instances.values():
            backend.stats[phase].clear()
            backend.profiles[phase].events.clear()

    def iteration_metrics(self, phase, reference):
        metrics = self.iteration_diagnostics[phase].finalize(reference)
        zero = reference.new_zeros((), dtype=torch.float32)
        counts = {}
        for backend in self._backend_instances.values():
            for name, value in backend.stats[phase].items():
                counts[name] = counts.get(name, 0) + value
        for name in ("setup_count", "update_count", "pool_hits", "pool_misses", "requested_rows", "capacity_rows", "padded_rows", "reuse_exception_fresh_retry"):
            metrics[f"backend/{name}"] = zero + counts.get(name, 0) if "cupiqp" in self._backend_instances else zero + float("nan")
        metrics["backend/counters_available"] = zero + float("cupiqp" in self._backend_instances)
        metrics["backend/solver_iterations_mean"] = (
            (zero + counts.get("iteration_sum", 0)) / (zero + counts["iteration_rows"]).clamp_min(1)
            if "iteration_rows" in counts else zero + float("nan")
        )
        metrics["profiling/enabled"] = zero + float(self.cfg.cuda_event_profiling)
        if self.cfg.cuda_event_profiling:
            times = self.profiles[phase].finalize()
            for backend in self._backend_instances.values():
                for name, value in backend.profiles[phase].finalize().items():
                    times[name] = times.get(name, 0) + value
            for name, value in times.items():
                metrics[f"profiling/{name}_ms"] = zero + value
        for name in ("assembly", "packing", "setup_update", "solve", "certification_recovery", "backward"):
            metrics.setdefault(f"profiling/{name}_ms", zero + float("nan"))
        return {f"qp/{phase}/{name}": value for name, value in metrics.items()}

    def clear_warm_start(self, env_ids=None):
        """Clear qpth rollout state globally or for chunks touching env_ids."""
        if env_ids is None:
            self._qpth_warm_states.clear()
            return
        # Keep reset ownership updates on the warm-state device. A reset is a
        # hot rollout path and must not introduce GPU->CPU->GPU synchronization.
        ids = torch.as_tensor(env_ids).detach()
        for key, (state, owners, valid) in self._qpth_warm_states.items():
            self._qpth_warm_states[key] = (state, owners,
                valid & ~torch.isin(owners, ids.to(owners.device)))

    def solver_for_mode(self, differentiable):
        """Return the explicitly configured rollout or PPO backend."""
        override = (
            self.cfg.ppo_qp_solver if differentiable
            else self.cfg.rollout_qp_solver
        )
        return override or self.cfg.qp_solver

    def solver_capabilities(self, reference):
        """Report all registered paths without importing unavailable solvers."""
        dtype = self._solve_dtype(reference)
        return {
            name: backend_capability(name, device=reference.device, dtype=dtype)
            for name in ("qpth", "cupiqp", "moreau")
        }

    @staticmethod
    def _full_audit_period(config):
        value = config.debug_audit_period
        return int(config.full_audit_period if value is None else value)

    @staticmethod
    def _full_audit_sample_size(config):
        value = config.debug_audit_sample_size
        return int(config.full_audit_sample_size if value is None else value)

    @property
    def diagnostics_level(self):
        return self.cfg.diagnostics_level

    def _physical_enabled(self):
        return (self.diagnostics_level in ("physical", "full")
                and getattr(self, "diagnostics_scheduled", True))

    def _solve_dtype(self, reference):
        """Resolve auto precision from the live learned input's device."""
        return _dtype_from_name(self.cfg.solver_dtype, reference)

    def _eps(self, dtype):
        if self.cfg.eps is not None:
            return float(self.cfg.eps)
        return (self.cfg.eps_float32 if dtype == torch.float32
                else self.cfg.eps_float64)

    def _normalized_tolerance(self, dtype):
        if self.cfg.feasibility_tolerance is not None:
            return float(self.cfg.feasibility_tolerance)
        return (self.cfg.normalized_feasibility_tolerance_float32
                if dtype == torch.float32
                else self.cfg.normalized_feasibility_tolerance_float64)

    def _profile(self, differentiable):
        """Return the immutable rollout/PPO numerical profile."""
        prefix = "ppo" if differentiable else "rollout"
        return {
            "eps_abs": float(getattr(self.cfg, f"{prefix}_eps_abs")),
            "eps_rel": float(getattr(self.cfg, f"{prefix}_eps_rel")),
            "max_iter": int(getattr(self.cfg, f"{prefix}_max_iter")),
            "feasibility": float(getattr(
                self.cfg, f"{prefix}_feasibility_tolerance"
            )),
            "gap_abs": float(getattr(self.cfg, f"{prefix}_duality_gap_abs")),
            "gap_rel": float(getattr(self.cfg, f"{prefix}_duality_gap_rel")),
            "gap_policy": getattr(self.cfg, f"{prefix}_duality_gap_policy"),
        }

    def _chunk_size(self, differentiable):
        if self.cfg.chunk_size is not None:
            return int(self.cfg.chunk_size)
        return int(self.cfg.ppo_chunk_size if differentiable
                   else self.cfg.rollout_chunk_size)

    @torch.inference_mode(False)
    def _limits(self, reference):
        """Copy immutable actuator limits once per device/dtype, not per chunk."""
        key = (reference.device, reference.dtype)
        if key not in self._limit_cache:
            self._limit_cache[key] = tuple(
                value.detach().to(device=reference.device, dtype=reference.dtype).clone()
                for value in (self.torque_limits, self.position_lower,
                              self.position_upper, self.velocity_limits)
            )
        return self._limit_cache[key]


    @torch.inference_mode(False)
    def _constants(self, reference):
        """Only immutable selectors and physical scales are cached, never Q."""
        key = (reference.device, reference.dtype, self.cfg.torque_scale_nm, self.cfg.force_scale_n)
        if key not in self._constant_cache:
            eye = torch.eye(24, device=reference.device, dtype=reference.dtype)
            scale = reference.new_tensor([self.cfg.torque_scale_nm] * 12
                                         + [self.cfg.force_scale_n] * 12)
            self._constant_cache[key] = (eye, scale)
        return self._constant_cache[key]

    def _build(self, data):
        r"""Assemble x=[tau_12; f_FR,FL,RR,RL_world_12], and substitute x=D z.

        x=[tau,tilde_f]; physical f=D_m tilde_f. All contact patterns share
        68 inequalities (24 actuator, 24 joint, 20 friction), zero equalities.
        Positive force curvature fixes otherwise unused swing variables.
        """
        ref = data["tau_nom"]
        batch = ref.shape[0]
        eye, scale = self._constants(ref)
        mass = data["mass_matrix"].detach()
        J = data["foot_jacobians"].detach().reshape(batch, 12, 18)
        stance = data["contact_probability"].detach() >= self.cfg.contact_threshold
        force_mask = stance.repeat_interleave(3,dim=1).to(ref.dtype)
        Jb = data["base_jacobian"].detach()
        bias = data["bias"].detach()
        # M a = [S^T J^T]x + Jb^T W - h. solve_ex reports a singular
        # mechanics row without poisoning all other environments in its batch.
        selector = ref.new_zeros(18, 12)
        selector[6:] = torch.eye(12, device=ref.device, dtype=ref.dtype)
        rhs = torch.cat((selector.expand(batch, -1, -1), J.transpose(1, 2)*force_mask[:,None,:],
                         Jb.transpose(1, 2), bias[..., None]), -1)
        solved, info = torch.linalg.solve_ex(mass, rhs, check_errors=False)
        mechanics_valid = (info == 0) & torch.isfinite(solved).all(dim=(1,2))
        # Fixed mechanics have no gradient responsibility. Sanitizing a failed
        # factor before multiplying learned W also prevents 0*NaN VJPs.
        solved = torch.nan_to_num(solved.detach(), nan=0., posinf=0., neginf=0.)
        mechanics_map = solved[:,:,:24]
        wrench = _ScaleClipRows.apply(data["wrench_pred_world"],
            self.cfg.gradient_scale_wrench, self.cfg.gradient_clip_wrench, self, "wrench")
        offset = (solved[:,:,24:30] @ wrench[...,None]).squeeze(-1)-solved[:,:,30]
        tau = _ScaleClipRows.apply(ref, self.cfg.gradient_scale_tau,
                                  self.cfg.gradient_clip_tau, self, "tau_nom")
        force = _ScaleClipRows.apply(data["force_pred_world"],
            self.cfg.gradient_scale_grf, self.cfg.gradient_clip_grf, self, "grf")
        # Mask the tilde-f tracking reference; D_m already masks mechanics.
        # Raw supervised predictions remain unbounded and unchanged.
        force = torch.where(stance[..., None], force, torch.zeros_like(force)).flatten(1)
        target = torch.cat((tau, force), -1)
        weights = ref.new_tensor([self.cfg.torque_tracking_weight] * 12
                                 + [self.cfg.force_tracking_weight] * 12)
        diagonal = 2 * weights / scale.square()
        Q = torch.diag(diagonal).expand(batch, -1, -1).clone()
        p = -diagonal * target

        def add_residual(C, e, weight):
            # w||C x+e||^2 -> Q += 2w C^T C, p += 2w C^T e.
            nonlocal Q, p
            Q = Q + 2 * weight * C.transpose(1, 2) @ C
            p = p + 2 * weight * (C.transpose(1, 2) @ e[..., None]).squeeze(-1)

        if self.cfg.contact_acceleration_weight:
            contact_J = J*force_mask[:,:,None]
            C = contact_J @ mechanics_map / self.cfg.contact_acceleration_scale_m_s2
            e = ((contact_J @ offset[..., None]).squeeze(-1)
                 + data["foot_acceleration_bias"].detach().flatten(1)*force_mask)
            add_residual(C, e / self.cfg.contact_acceleration_scale_m_s2,
                         self.cfg.contact_acceleration_weight)

        if self.cfg.attitude_weight:
            # H maps canonical acceleration to yaw-local PHYSICAL angular
            # acceleration (not Euler-angle second derivatives). In a free
            # flyer Jb_angular*v = R_WB*w_B and Jdotb_angular*v =
            # R_WB*(w_B cross w_B)=0. Project into instantaneous yaw axes;
            # we do not differentiate the yaw coordinate frame.
            q = data["base_quaternion"].detach()  # canonical xyzw
            yaw = torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),
                              1-2*(q[:,1].square()+q[:,2].square()))
            c, s = yaw.cos(), yaw.sin()
            R = ref.new_zeros(batch, 2, 3)
            R[:,0,0], R[:,0,1] = c, s
            R[:,1,0], R[:,1,1] = -s, c
            up = torch.stack((2*(q[:,0]*q[:,2]+q[:,3]*q[:,1]),
                              2*(q[:,1]*q[:,2]-q[:,3]*q[:,0]),
                              1-2*(q[:,0].square()+q[:,1].square())), -1)
            # z_world cross z_body is a restoring tilt-error rotation vector
            # near upright: [roll,pitch] in yaw-local horizontal axes.
            tilt_world = torch.stack((-up[:,1], up[:,0], torch.zeros_like(up[:,0])), -1)
            tilt = (R @ tilt_world[...,None]).squeeze(-1)
            omega = (R @ data["base_angular_velocity_world"].detach()[...,None]).squeeze(-1)
            desired = -self.cfg.attitude_kp * tilt - self.cfg.attitude_kd * omega
            H = R @ Jb[:,3:6]
            C = H @ mechanics_map / self.cfg.attitude_acceleration_scale_rad_s2
            e = ((H @ offset[...,None]).squeeze(-1) - desired)
            add_residual(C, e / self.cfg.attitude_acceleration_scale_rad_s2,
                         self.cfg.attitude_weight)

        limits, qmin, qmax, vmax = self._limits(ref)
        dt = data["dt"].detach().reshape(-1, 1)
        previous = data["previous_torque"].detach()
        lower = torch.maximum(-limits, previous - self.cfg.torque_rate_limit_nm_s * dt)
        upper = torch.minimum(limits, previous + self.cfg.torque_rate_limit_nm_s * dt)
        q, v = data["joint_position"].detach(), data["joint_velocity"].detach()
        amax = ref.new_tensor(self.cfg.joint_acceleration_limits_rad_s2).reshape(1,12)
        beta = self.cfg.position_integration_coefficient
        alower = torch.maximum(-amax, torch.maximum((-vmax-v)/dt, (qmin-q-dt*v)/(beta*dt.square())))
        aupper = torch.minimum(amax, torch.minimum((vmax-v)/dt, (qmax-q-dt*v)/(beta*dt.square())))
        joint_map, joint_offset = mechanics_map[:,6:], offset[:,6:]
        # Torque 24 rows, acceleration intersection 24 rows. beta=1 matches
        # semi-implicit Genesis/PhysX; beta=.5 is the constant-a convention.
        G = [eye[:12].expand(batch,-1,-1), -eye[:12].expand(batch,-1,-1),
             joint_map, -joint_map]
        h = [upper, -lower, aupper-joint_offset, joint_offset-alower]
        for foot in range(4):
            block = ref.new_zeros(5,24)
            col = 12+3*foot
            block[0,col+2] = -1
            block[1,col], block[2,col] = 1, -1
            block[3,col+1], block[4,col+1] = 1, -1
            block[1:,col+2] = -self.cfg.friction_coefficient
            # In swing these become 0<=1 after row normalization, not active
            # zero equalities. Stance rows retain the physical friction cone.
            G.append(block.expand(batch,-1,-1)*stance[:,foot,None,None])
            h.append((~stance[:,foot,None]).to(ref.dtype).expand(-1,5))
        physical_G, physical_h = torch.cat(G,1), torch.cat(h,1)
        physical_A = ref.new_empty(batch,0,24)
        physical_b = ref.new_empty(batch,0)
        G, h, gs = _row_scale(physical_G * scale, physical_h)
        A, b, es = _row_scale(physical_A * scale, physical_b)
        Q = Q * scale[:,None] * scale[None,:]
        Q = .5 * (Q + Q.transpose(1,2))
        Q = Q + self.cfg.q_regularization * eye
        p = p * scale
        native_lower = ref.new_full((batch,24), -torch.inf)
        native_upper = ref.new_full((batch,24), torch.inf)
        native_lower[:,:12], native_upper[:,:12] = lower/scale[:12], upper/scale[:12]
        return _QPBuild(Q,p,G,h,A,b,scale,physical_G,physical_h,physical_A,physical_b,
                        es,gs,lower,upper,alower,aupper,native_lower,native_upper,
                        mechanics_map,offset,mechanics_valid)

    @staticmethod
    def _cupiqp_native_pack(m):
        # cuPIQP 0.1 setup/update support x_l/x_u. Only the first 24 canonical
        # inequalities are true coordinate bounds. Joint acceleration bounds
        # are coupled affine rows and MUST remain general inequalities.
        return m.G[:,24:], m.h[:,24:], m.native_lower, m.native_upper

    def _soft_joint_problem(self, m):
        """Recovery only: y=[tau, f_tilde, s], s>=0 in rad/s².

        Relax the combined joint envelope to lower-s <= a_joint <= upper+s.
        This softens acceleration and predicted position/velocity limits, NOT
        actuator/rate, friction or exact physical swing-force elimination.
        Add w*||s/scale||²; the original 24-D objective remains unchanged.
        """
        batch = m.p.shape[0]
        scale = m.variable_scale.new_full((12,), self.cfg.soft_joint_recovery_scale_rad_s2)
        variable_scale = torch.cat((m.variable_scale, scale))
        Q = m.Q.new_zeros(batch,36,36)
        Q[:,:24,:24] = m.Q
        eye = torch.eye(12,device=Q.device,dtype=Q.dtype)
        Q[:,24:,24:] = (2*self.cfg.soft_joint_recovery_weight+self.cfg.q_regularization)*eye
        physical_G = m.G.new_zeros(batch,80,36)
        physical_G[:,:68,:24] = m.physical_G
        physical_G[:,24:36,24:] = -eye
        physical_G[:,36:48,24:] = -eye
        physical_G[:,68:,24:] = -eye
        physical_h = torch.cat((m.physical_h,m.p.new_zeros(batch,12)),1)
        G,h,row_scale = _row_scale(physical_G*variable_scale,physical_h)
        empty = m.A.new_zeros(batch,0,36)
        return replace(m,Q=Q,p=torch.cat((m.p,m.p.new_zeros(batch,12)),1),
            G=G,h=h,A=empty,physical_A=empty,physical_G=physical_G,physical_h=physical_h,
            variable_scale=variable_scale,inequality_row_scale=row_scale,
            native_lower=torch.cat((m.native_lower,m.p.new_full((batch,12),-torch.inf)),1),
            native_upper=torch.cat((m.native_upper,m.p.new_full((batch,12),torch.inf)),1),
            acceleration_map=torch.cat((m.acceleration_map,m.p.new_zeros(batch,18,12)),2))

    @staticmethod
    def _maximum(value):
        return value.amax(-1) if value.shape[-1] else value.new_zeros(value.shape[0])

    @torch.no_grad()
    def _certificate(self, m, z, tolerance):
        # Certify the EXACT final command after torque projection and exact
        # swing equality enforcement; never certify a different pre-clamp x.
        eq = (m.A @ z[...,None]).squeeze(-1)-m.b
        iq = (m.G @ z[...,None]).squeeze(-1)-m.h
        finite = torch.isfinite(z).all(-1)
        er, ir = self._maximum(eq.abs()), self._maximum(iq.clamp_min(0))
        return finite & (er<=tolerance) & (ir<=tolerance), er, ir

    @torch.no_grad()
    def _physical_diagnostics(self, m, x, data):
        """Physical-unit checks remain separate from normalized acceptance."""
        residual = (m.physical_G @ x[...,None]).squeeze(-1)-m.physical_h
        equality = (m.physical_A @ x[...,None]).squeeze(-1)-m.physical_b
        a = (m.acceleration_map @ x[...,None]).squeeze(-1)+m.acceleration_offset
        dyn = (data["mass_matrix"] @ a[...,None]).squeeze(-1)+data["bias"]
        dyn[:,6:] -= x[:,:12]
        dyn -= torch.einsum("bfkn,bfk->bn",data["foot_jacobians"],x[:,12:].reshape(-1,4,3))
        dyn -= torch.einsum("bkn,bk->bn",data["base_jacobian"],data["wrench_pred_world"])
        metrics = {"equality_max":self._maximum(equality.abs()),
                   "inequality_max":self._maximum(residual.clamp_min(0))}
        for name,sl in (("base_linear",slice(0,3)),("base_angular",slice(3,6)),("joint",slice(6,18))):
            metrics["dynamics/"+name+"_mae"] = dyn[:,sl].abs().mean(-1)
        for name,sl in (("torque_rate_intersection",slice(0,24)),
                        ("joint_acceleration_intersection",slice(24,48)),
                        ("friction_unilateral",slice(48,None))):
            metrics[name+"/violation_max"] = self._maximum(residual[:,sl].clamp_min(0))
            metrics[name+"/margin_min"] = (-residual[:,sl]).amin(-1) if residual[:,sl].shape[1] else x.new_full((x.shape[0],),float("nan"))
            metrics[name+"/active_fraction"] = (residual[:,sl].abs()<=self.cfg.active_tolerance).float().mean(-1) if residual[:,sl].shape[1] else x.new_full((x.shape[0],),float("nan"))
        stance = (data["contact_probability"] >= self.cfg.contact_threshold).detach()
        ca = torch.einsum("bfkn,bn->bfk",data["foot_jacobians"],a)+data["foot_acceleration_bias"]
        stance_sq = ca.square().sum(-1).mul(stance).sum(-1)
        metrics["model_stance_acceleration_rms_m_s2"] = (stance_sq/(3*stance.sum(-1)).clamp_min(1)).sqrt()
        torque_error = x[:,:12]-data["tau_nom"]
        force_error = x[:,12:].reshape(-1,4,3)-torch.where(stance[...,None],data["force_pred_world"],0.)
        metrics["torque_correction_rms_nm"] = torque_error.square().mean(-1).sqrt()
        metrics["force_reference_error_rms_n"] = force_error.square().mean((1,2)).sqrt()
        metrics["objective/torque_dimensionless"] = self.cfg.torque_tracking_weight*(torque_error/self.cfg.torque_scale_nm).square().sum(-1)
        metrics["objective/grf_dimensionless"] = self.cfg.force_tracking_weight*(force_error/self.cfg.force_scale_n).square().sum((1,2))
        metrics["objective/stance_dimensionless"] = self.cfg.contact_acceleration_weight*stance_sq/self.cfg.contact_acceleration_scale_m_s2**2
        # Same instantaneous yaw-local physical angular acceleration used by
        # the soft attitude objective (not Euler-angle second derivatives).
        q=data["base_quaternion"]
        yaw=torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1].square()+q[:,2].square()))
        R=x.new_zeros(x.shape[0],2,3)
        R[:,0,0],R[:,0,1]=yaw.cos(),yaw.sin()
        R[:,1,0],R[:,1,1]=-yaw.sin(),yaw.cos()
        tilt_world=torch.stack((-2*(q[:,1]*q[:,2]-q[:,3]*q[:,0]),
            2*(q[:,0]*q[:,2]+q[:,3]*q[:,1]),torch.zeros_like(yaw)),-1)
        angular=(data["base_jacobian"][:,3:6]@a[...,None]).squeeze(-1)
        error=(R@(angular+self.cfg.attitude_kp*tilt_world+
            self.cfg.attitude_kd*data["base_angular_velocity_world"])[...,None]).squeeze(-1)
        metrics["objective/attitude_dimensionless"]=self.cfg.attitude_weight*(error/self.cfg.attitude_acceleration_scale_rad_s2).square().sum(-1)
        for name,value in (("qdd",a),("force",x[:,12:]),("torque",x[:,:12])):
            metrics[name+"/mean"] = value.abs().mean(-1)
            metrics[name+"/max"] = value.abs().amax(-1)
        return metrics

    @torch.no_grad()
    def _audit(self, m, z):
        """Periodic bounded audits only; never used to reject a solution."""
        eig = torch.linalg.eigvalsh(m.Q)
        rank = torch.linalg.matrix_rank(m.A) if m.A.shape[1] else z.new_zeros(z.shape[0])
        slack = m.h-(m.G @ z[...,None]).squeeze(-1)
        active = slack.abs() <= self.cfg.active_tolerance
        C = torch.cat((m.A,m.G*active[...,None]),1)
        grad = (m.Q @ z[...,None]).squeeze(-1)+m.p
        dual = (torch.linalg.pinv(C.transpose(1,2)) @ -grad[...,None]).squeeze(-1)
        stationarity = grad+(C.transpose(1,2)@dual[...,None]).squeeze(-1)
        lam = dual[:,m.A.shape[1]:]
        return {"q_min_eigenvalue":eig[:,0],"q_max_eigenvalue":eig[:,-1],
                "q_condition":eig[:,-1]/eig[:,0],"equality_rank":rank,
                "stationarity_max":stationarity.abs().amax(-1),
                "complementarity_max":(lam*slack).abs().amax(-1),
                "active_constraint_fraction":active.float().mean(-1)}

    def _backend_solve(self, m):
        from .hard_pact_qp_backends import QPBackendResult
        if self._active_solver == "qpth":
            if self.cfg.qpth_warm_start and not self._active_differentiable:
                key, owners = self._qpth_context
                entry = self._qpth_warm_states.get(key)
                warm = entry[0] if entry is not None else None
                mask = (entry[2] & (entry[1]==owners)) if entry is not None else None
                z, terminal = solve_qpth_warm(m.Q,m.p,m.G,m.h,m.A,m.b,
                    warm_start=warm,warm_mask=mask,eps=self._eps(m.p.dtype),
                    verbose=-1 if self.cfg.verbose==0 else self.cfg.verbose,
                    not_improved_limit=self.cfg.not_improved_limit,
                    max_iter=self.cfg.max_iter,check_q_spd=self.cfg.check_q_spd)
                valid,_,_ = self._certificate(m,z,self._normalized_tolerance(m.p.dtype))
                self._qpth_warm_states[key]=(terminal,owners.detach().clone(),valid.detach().clone())
                retry=(~valid).nonzero(as_tuple=True)[0]
                if retry.numel():
                    # A bad initializer gets the same unmodified cold QP,
                    # never a relaxed/elastic recovery problem.
                    cold=QPFunction(eps=self._eps(m.p.dtype),verbose=-1,
                        notImprovedLim=self.cfg.not_improved_limit,maxIter=self.cfg.max_iter,
                        check_Q_spd=self.cfg.check_q_spd)(m.Q[retry],m.p[retry],m.G[retry],m.h[retry],m.A[retry],m.b[retry])
                    z=z.index_copy(0,retry,cold)
                return QPBackendResult(z)
            z, _ = solve_qpth_warm(m.Q,m.p,m.G,m.h,m.A,m.b,
                warm_start=None,eps=self._eps(m.p.dtype),
                verbose=-1 if self.cfg.verbose==0 else self.cfg.verbose,
                not_improved_limit=self.cfg.not_improved_limit,
                max_iter=self.cfg.max_iter, check_q_spd=self.cfg.check_q_spd)
            return QPBackendResult(z)
        G,h,lo,hi = self._cupiqp_native_pack(m) if self._active_solver=="cupiqp" else (
            m.G,m.h,None,None)
        return self._backend_instances[self._active_solver].solve(
            m.Q,m.p,G,h,m.A,m.b,differentiable=self._active_differentiable,
            native_lower=lo,native_upper=hi,
            # Contact/attitude residuals refresh Q at EVERY state. Never
            # reuse a Hessian/preconditioner as if the mechanics were constant.
            constant_hessian=False)

    def solve(self, *, differentiable=None, diagnostics_phase=None,
              environment_ids=None, substep_index=None, environment_count=None, **data):
        """Hard QP, optional soft-joint recovery, then actuator/rate projection.

        A fixed stance mask eliminates physical swing forces without equality
        padding. Backend caches are keyed by actual matrix shape;
        PPO graphs retain exclusive backend leases through all backward uses.
        """
        reference = data["tau_nom"]
        if differentiable is None:
            differentiable = torch.is_grad_enabled() and any(
                data[k].requires_grad for k in ("tau_nom","force_pred_world","wrench_pred_world"))
        if not differentiable and torch.is_grad_enabled():
            with torch.no_grad():
                return self.solve(differentiable=False, diagnostics_phase=diagnostics_phase,
                    environment_ids=environment_ids,substep_index=substep_index,
                    environment_count=environment_count,**data)
        dtype = self._solve_dtype(reference)
        if self._backend_config != self.cfg:
            # Never carry solver allocations/settings or active/warm snapshots
            # across a runtime settings change.
            self.clear_warm_start()
            names={self.solver_for_mode(False),self.solver_for_mode(True)}
            self._backend_instances={name:create_backend(name,self.cfg) for name in names if name!="qpth"}
            self._backend_config=self.cfg
        self.solver_dtype = dtype
        self._active_solver = self.solver_for_mode(differentiable)
        self._active_differentiable = bool(differentiable)
        self._diagnostics_phase = diagnostics_phase or ("ppo" if differentiable else "rollout")
        for backend in self._backend_instances.values():
            backend.diagnostics_phase = self._diagnostics_phase
        require_backend(self._active_solver,device=reference.device,dtype=dtype)
        self._solve_count += 1
        audit_remaining = (self._full_audit_sample_size(self.cfg)
            if self.diagnostics_level=="full" and getattr(self,"diagnostics_scheduled",True) and self._full_audit_period(self.cfg)>0
            and self._solve_count % self._full_audit_period(self.cfg)==0 else 0)
        event_profile = self.profiles[self._diagnostics_phase]
        self._last_gradient_metrics = {}
        values = {k: v.to(dtype=dtype) for k,v in data.items()}
        n = reference.shape[0]
        ref = values["tau_nom"]
        limits = self._limits(ref)[0]
        dt = values["dt"].detach().reshape(-1,1)
        # Invalid previous state has no feasible rate claim. Reset/nonfinite
        # history to zero for deterministic actuator-only fallback and log it.
        prev = torch.nan_to_num(values["previous_torque"].detach(),nan=0.,posinf=0.,neginf=0.)
        safe_dt = torch.where(torch.isfinite(dt)&(dt>0),dt,torch.zeros_like(dt))
        lower = torch.maximum(-limits,prev-self.cfg.torque_rate_limit_nm_s*safe_dt)
        upper = torch.minimum(limits,prev+self.cfg.torque_rate_limit_nm_s*safe_dt)
        empty_tau = (lower>upper).any(-1)
        # Empty intersection cannot satisfy both claims: actuator limits win,
        # with a distinct failure reason, NEVER a joint/contact certificate.
        lower = torch.where(empty_tau[:,None],-limits,lower)
        upper = torch.where(empty_tau[:,None],limits,upper)
        fallback = torch.nan_to_num(ref.detach(),nan=0.,posinf=0.,neginf=0.).clamp(lower,upper)
        primal = torch.cat((fallback,ref.new_zeros(n,12)),1)
        # Graph-connected zero (also for an all-invalid update), no learned
        # VJP through analytic fallback and no invalid NaN arithmetic.
        primal = primal + sum(torch.nan_to_num(values[k],nan=0.,posinf=0.,neginf=0.).sum()*0
                              for k in ("tau_nom","force_pred_world","wrench_pred_world"))
        qdd = ref.new_zeros(n,18)
        ok = torch.zeros(n,device=ref.device,dtype=torch.bool)
        finite_input = torch.stack([torch.isfinite(v).reshape(n,-1).all(-1)
                                    for v in values.values()]).all(0) & (dt[:,0]>0)
        diag = {"failure/nonfinite_input":~finite_input,
                "failure/empty_torque_intersection":empty_tau,
                "failure/empty_qdd_intersection":torch.zeros_like(ok),
                "failure/mechanics":torch.zeros_like(ok),
                "full/attempted":torch.zeros_like(ok),
                "full/solver_exception":torch.zeros_like(ok),
                "full/input_finite":finite_input,
                "full/output_finite":torch.zeros_like(ok),
                "full/duality_gap":ref.new_full((n,),float("nan")),
                "full/duality_gap_rel":ref.new_full((n,),float("nan")),
                "selected/equality_max":ref.new_full((n,),float("nan")),
                "selected/inequality_max":ref.new_full((n,),float("nan")),
                "pre_clamp_torque_violation_max":ref.new_zeros(n)}
        stance = values["contact_probability"].detach() >= self.cfg.contact_threshold
        profile = self._profile(differentiable)
        tolerance = (profile["feasibility"] if self._active_solver!="qpth"
                     else self._normalized_tolerance(dtype))
        ids = (finite_input & ~empty_tau).nonzero(as_tuple=True)[0]
        for chunk_index, rows in enumerate(ids.split(self._chunk_size(differentiable))):
            if not rows.numel():
                continue
            part = {k:v.index_select(0,rows) for k,v in values.items()}
            with event_profile.measure("assembly",ref):
                m = self._build(part)
            finite = torch.stack([torch.isfinite(t).flatten(1).all(-1)
                                  for t in (m.Q,m.p,m.G,m.h,m.A,m.b)]).all(0)
            empty_a = (m.qdd_lower>m.qdd_upper).any(-1)
            diag["failure/empty_qdd_intersection"][rows] = empty_a
            diag["failure/mechanics"][rows] = ~m.mechanics_valid
            local = (finite & m.mechanics_valid & ~empty_a).nonzero(as_tuple=True)[0]
            if not local.numel():
                continue
            rows = rows[local]
            m = select_problem(m,local)
            owners=rows if environment_ids is None else environment_ids[rows]
            self._qpth_context=((chunk_index,m.p.shape,m.G.shape,m.A.shape,ref.device,ref.dtype),owners)
            diag["full/attempted"][rows] = True
            aggregate=self.iteration_diagnostics[self._diagnostics_phase]
            aggregate.add_sum("backend/dispatch_count",rows.new_tensor(1))
            aggregate.add_sum("backend/dispatched_rows",rows.new_tensor(rows.numel()))
            try:
                with event_profile.measure("solve",ref):
                    result = self._backend_solve(m)
                z = result.solution
            except QPBackendUnavailable:
                raise
            except Exception as error:
                G,h,lo,hi = self._cupiqp_native_pack(m) if self._active_solver=="cupiqp" else (m.G,m.h,None,None)
                capture_failure(self,error,dict(Q=m.Q,p=m.p,G=G,h=h,A=m.A,b=m.b,
                                                native_lower=lo,native_upper=hi))
                diag["full/solver_exception"][rows] = True
                continue
            diag["full/output_finite"][rows] = torch.isfinite(z.detach()).all(-1)
            x = z * m.variable_scale
            pre = torch.maximum((m.tau_lower-x[:,:12]).clamp_min(0),
                                (x[:,:12]-m.tau_upper).clamp_min(0)).amax(-1)
            diag["pre_clamp_torque_violation_max"][rows] = pre.detach()
            torque = x[:,:12].clamp(m.tau_lower,m.tau_upper)
            forces = x[:,12:].reshape(-1,4,3)
            forces = torch.where(stance[rows,:,None],forces,torch.zeros_like(forces))
            x = torch.cat((torque,forces.flatten(1)),1)
            with event_profile.measure("certification_recovery",ref):
                accepted,er,ir = self._certificate(m,x/m.variable_scale,tolerance)
            accepted &= torch.isfinite(z.detach()).all(-1)
            for key,value in (("duality_gap",result.duality_gap),
                              ("duality_gap_rel",result.duality_gap_rel)):
                if value is not None:
                    diag["full/"+key][rows] = value.detach()
            if self._active_solver=="cupiqp" and profile["gap_policy"]=="require":
                gap,rel = result.duality_gap,result.duality_gap_rel
                accepted &= (torch.isfinite(gap)&torch.isfinite(rel)
                             & ((gap<=profile["gap_abs"])|(rel<=profile["gap_rel"]))) if gap is not None and rel is not None else False
            diag["selected/equality_max"][rows],diag["selected/inequality_max"][rows]=er,ir
            if self._physical_enabled():
                physical = self._physical_diagnostics(m,x,{k:v.index_select(0,local) for k,v in part.items()})
                for key,value in physical.items():
                    name="physical/"+key
                    if name not in diag: diag[name]=ref.new_full((n,),float("nan"))
                    diag[name][rows]=value.to(ref.dtype)
            count=min(audit_remaining,rows.numel())
            if count:
                sampled=torch.arange(count,device=ref.device)
                for key,value in self._audit(select_problem(m,sampled),(x/m.variable_scale)[:count]).items():
                    name="full/audit/"+key
                    if name not in diag: diag[name]=ref.new_full((n,),float("nan"))
                    diag[name][rows[:count]]=value.to(ref.dtype)
                audit_remaining-=count
            # The backend gets a zero adjoint on each failed row before
            # invoking its implicit solve. Forward NaNs are removed too.
            x = _CertifiedRows.apply(torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.),accepted)
            primal = primal.index_copy(0,rows,torch.where(accepted[:,None],x,primal[rows]))
            derived = (m.acceleration_map @ x[...,None]).squeeze(-1)+m.acceleration_offset
            qdd = qdd.index_copy(0,rows,torch.where(accepted[:,None],derived,torch.zeros_like(derived)))
            ok[rows]=accepted
        # Recovery is execution-only: never provide an implicit VJP or claim
        # hard joint certification for a softened solution. Preserve the hard
        # solution graph outside no_grad, including mixed accepted/failed rows.
        soft_ok = torch.zeros_like(ok)
        recovered = primal.detach().clone()
        recovered_qdd = qdd.detach().clone()
        diag["soft_joint/attempted"] = torch.zeros_like(ok)
        diag["soft_joint/solver_exception"] = torch.zeros_like(ok)
        diag["soft_joint/slack_max_rad_s2"] = ref.new_full((n,),float("nan"))
        if self.cfg.soft_joint_recovery_enabled:
            try:
                self._active_differentiable = False
                with torch.no_grad():
                    recovery_ids = (finite_input & ~empty_tau & ~ok).nonzero(as_tuple=True)[0]
                    for rows in recovery_ids.split(self._chunk_size(False)):
                        if not rows.numel():
                            continue
                        m = self._soft_joint_problem(self._build({k:v[rows].detach() for k,v in values.items()}))
                        finite = m.mechanics_valid & torch.stack([
                            torch.isfinite(t).flatten(1).all(-1) for t in (m.Q,m.p,m.G,m.h)]).all(0)
                        local = finite.nonzero(as_tuple=True)[0]
                        if not local.numel():
                            continue
                        rows,m = rows[local],select_problem(m,local)
                        diag["soft_joint/attempted"][rows] = True
                        owners=rows if environment_ids is None else environment_ids[rows]
                        self._qpth_context=(("soft_joint",m.p.shape,m.G.shape,m.A.shape,ref.device,ref.dtype),owners)
                        aggregate=self.iteration_diagnostics[self._diagnostics_phase]
                        aggregate.add_sum("backend/dispatch_count",rows.new_tensor(1))
                        aggregate.add_sum("backend/dispatched_rows",rows.new_tensor(rows.numel()))
                        try:
                            result = self._backend_solve(m)
                        except QPBackendUnavailable:
                            raise
                        except Exception as error:
                            capture_failure(self,error,dict(Q=m.Q,p=m.p,G=m.G,h=m.h,A=m.A,b=m.b))
                            diag["soft_joint/solver_exception"][rows] = True
                            continue
                        x = result.solution*m.variable_scale
                        pre = torch.maximum((m.tau_lower-x[:,:12]).clamp_min(0),
                                            (x[:,:12]-m.tau_upper).clamp_min(0)).amax(-1)
                        diag["pre_clamp_torque_violation_max"][rows] = pre
                        x[:,:12] = x[:,:12].clamp(m.tau_lower,m.tau_upper)
                        x[:,12:24] *= stance[rows].repeat_interleave(3,1)
                        accepted,er,ir = self._certificate(m,x/m.variable_scale,tolerance)
                        accepted &= torch.isfinite(result.solution).all(-1)
                        # Execution-only recovery uses the rollout numerical
                        # profile, including its configured gap policy.
                        recovery_profile = self._profile(False)
                        if self._active_solver=="cupiqp" and recovery_profile["gap_policy"]=="require":
                            gap,rel = result.duality_gap,result.duality_gap_rel
                            accepted &= (torch.isfinite(gap)&torch.isfinite(rel)&
                                ((gap<=recovery_profile["gap_abs"])|(rel<=recovery_profile["gap_rel"]))) if gap is not None and rel is not None else False
                        soft_ok[rows] = accepted
                        recovered[rows] = torch.where(accepted[:,None],x[:,:24],recovered[rows])
                        a = (m.acceleration_map@x[...,None]).squeeze(-1)+m.acceleration_offset
                        recovered_qdd[rows] = torch.where(accepted[:,None],a,recovered_qdd[rows])
                        diag["soft_joint/slack_max_rad_s2"][rows] = x[:,24:].amax(-1)
                        diag["selected/equality_max"][rows] = er
                        diag["selected/inequality_max"][rows] = ir
            finally:
                self._active_differentiable = bool(differentiable)
        primal = torch.where(soft_ok[:,None],recovered,primal)
        qdd = torch.where(soft_ok[:,None],recovered_qdd,qdd)
        stage = torch.where(ok,0,2)
        stage = torch.where(soft_ok,1,stage)
        metrics = {"qp/minimal/full_fraction":ok.float().mean(),
                   "qp/minimal/soft_joint_fraction":soft_ok.float().mean(),
                   "qp/minimal/fallback_fraction":(stage==2).float().mean(),
                   "qp/minimal/differentiated_fraction":ok.float().mean()*int(differentiable)}
        # Physical diagnostics are optional, never part of the objective.
        if self._physical_enabled():
            metrics["qp/physical/torque_correction_mean"]=(primal[:,:12].detach()-ref.detach()).abs().mean()
        for key,value in diag.items():
            if key.startswith(("physical/","full/audit/")):
                finite=torch.isfinite(value)
                metrics["qp/"+key]=value.where(finite,0).sum()/finite.sum().clamp_min(1)
        result = HardPACTQPResult(qdd.to(reference.dtype),
            primal[:,12:].reshape(n,4,3).to(reference.dtype),
            primal[:,:12].to(reference.dtype),stage,ok,diag,metrics)
        self.iteration_diagnostics[self._diagnostics_phase].add_result(result,differentiable)
        return result


def projection_loss(tau_safe, tau_nom, torque_limit, physics_valid, differentiated,
                    *, qdd=None, foot_jacobians=None, foot_acceleration_bias=None,
                    stance_mask=None, contact_weight=0.0, contact_scale=1.0,
                    return_per_row=False):
    """Certified outer loss: normalized torque correction plus soft stance acceleration.

    Mechanics and discrete stance are constants; the certified solution retains
    its implicit derivative. Select valid rows before arithmetic (NaN * 0 is
    not safe). The solver independently masks failed-row VJPs inside autograd.
    The existing unit torque coefficient and outer lambda_projection are retained.
    """
    valid = physics_valid.reshape(-1).bool() & differentiated.reshape(-1).bool()
    per_valid = ((tau_safe[valid]-tau_nom[valid])/torque_limit).square().sum(-1)
    if contact_weight:
        acceleration = (torch.einsum("bfkn,bn->bfk", foot_jacobians.detach()[valid],
                                     qdd[valid]) + foot_acceleration_bias.detach()[valid])
        stance = stance_mask.detach()[valid].bool()
        acceleration = torch.where(stance[..., None], acceleration, 0.0)
        per_valid = per_valid + contact_weight * (acceleration/contact_scale).square().sum((1,2))
    per_row = tau_nom.new_zeros(tau_nom.shape[0]).masked_scatter(valid, per_valid)
    loss = per_valid.sum()/valid.sum().clamp_min(1)
    return (loss, per_row) if return_per_row else loss


def balanced_substep_indices(num_samples, decimation, device, *, generator=None):
    """Stratified-uniform substep samples with counts differing by at most one.

    A random cyclic offset makes every substep marginally uniform, while a
    random assignment prevents environment identity from becoming correlated
    with phase.  Only the resulting int16 index is retained in rollout storage.

    Let ``D=decimation`` and draw ``o~Uniform({0,...,D-1})``. Before the final
    permutation, sample ``i`` receives ``k_i=(i+o) mod D``. Thus every
    ``k_i`` is marginally uniform, while bin counts differ by at most one.
    """
    if decimation < 1:
        raise ValueError("decimation must be positive")
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative")
    if num_samples == 0:
        return torch.empty(0, device=device, dtype=torch.int16)
    # Random cyclic offset supplies marginal uniformity across rollout steps.
    offset = torch.randint(
        decimation, (1,), device=device, generator=generator
    )
    # Deterministic balanced strata after conditioning on the offset.
    strata = (torch.arange(num_samples, device=device) + offset) % decimation
    # Randomly associate those strata with environment identities.
    assignment = torch.randperm(num_samples, device=device, generator=generator)
    # Scatter preserves one sample per environment and balanced bin counts.
    result = torch.empty_like(strata)
    result[assignment] = strata
    # int16 is sufficient for practical decimation and minimizes rollout VRAM.
    return result.to(torch.int16)


def balanced_anchor_indices(num_samples, anchors, device, *, generator=None):
    """Balanced uniform sampling over an explicit ordered anchor set."""
    anchors = torch.as_tensor(anchors, device=device, dtype=torch.int16)
    if anchors.ndim != 1 or anchors.numel() == 0:
        raise ValueError("anchors must be a nonempty one-dimensional sequence")
    bins = balanced_substep_indices(
        num_samples, int(anchors.numel()), device, generator=generator
    ).long()
    return anchors[bins]
