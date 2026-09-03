r"""Differentiable fixed-shape safety QP for HardPACT.

The public solver operates in physical units and canonical Go2 order.  It
internally scales variables and constraint rows before calling qpth's OptNet
layer; returned accelerations, forces, torques, and slacks are physical again.

For each environment the layer solves the standard convex QP

.. math::

   \min_x \; \tfrac12 x^TQx+p^Tx
   \quad\text{s.t.}\quad Gx\le h,\;Ax=b,

with physical decision vector

.. math::

   x=[\ddot q_{18},\;f_{FR,FL,RR,RL}^{W}\!{}_{12},\;
      \tau_{safe,12},\;s_{contact,12}].

The code constructs that physical problem first and then substitutes
``x = D z``. qpth solves for the dimensionless variable ``z``; the public
result is recovered as ``x = D z``. Comments beside each matrix assignment
below state the exact scalar/vector constraint represented by that assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import time

import torch
from qpth.qp import QPFunction
from .hard_pact_qp_backends import (
    QPBackendUnavailable,
    backend_capability,
    create_backend,
    require_backend,
)
from .qpth_warm_start import solve_qpth_warm


# Fixed slices make every Q/P/G/A block visibly correspond to one physical
# variable group. Keeping these compile-time fixed also avoids ragged QPs.
QDD = slice(0, 18)       # x[0:18]  = generalized acceleration, [m/s^2, rad/s^2].
FORCE = slice(18, 30)    # x[18:30] = four world-frame XYZ GRFs, [N].
TORQUE = slice(30, 42)   # x[30:42] = executed actuator torque, [Nm].
SLACK = slice(42, 54)    # x[42:54] = contact-acceleration slack, [m/s^2].
NUM_VARIABLES = 54


@dataclass(frozen=True)
class HardPACTQPConfig:
    """Numerics and physical weights exposed by the HardPACT config.

    Scales define ``D`` in ``x=Dz``. Tracking/regularization values define
    terms in the physical objective before that substitution. Solver controls
    are passed directly to :class:`qpth.qp.QPFunction`.
    """

    enabled: bool = True  # Master rollout/PPO projection switch.
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
    qpth_warm_start: bool = False
    friction_coefficient: float = 0.6  # mu in |fx|,|fy|<=mu*fz.
    torque_rate_limit_nm_s: float = 1000.0  # dot(tau)_lim [Nm/s].
    contact_acceleration_limit_m_s2: float = 0.0  # a_tol [m/s^2].
    interior_margin: float = 1.0e-3  # Strict-feasibility epsilon for contact rows.
    contact_probability_floor: float = 1.0e-2  # c_min prevents zero contact rows.
    qdd_scale: float = 50.0  # D diagonal for generalized acceleration.
    force_scale_n: float = 250.0  # D diagonal/reference normalization for GRF.
    torque_scale_nm: float = 40.0  # D diagonal for safe torque variables.
    slack_scale_m_s2: float = 50.0  # D/objective scale for contact slack.
    torque_tracking_weight: float = 20.0  # w_tau.
    force_tracking_weight: float = 5.0  # w_f.
    slack_weight: float = 200.0  # w_s.
    qdd_regularization: float = 1.0e-3  # r_qdd.
    force_regularization: float = 1.0e-4  # r_f.
    torque_regularization: float = 1.0e-4  # r_tau.
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
    # Diagnostics never participate in acceptance or fallback decisions.
    # ``minimal`` is the production default; ``physical`` adds detached
    # physical-unit summaries; ``full`` periodically enables sampled matrix,
    # KKT, timing, memory, and gradient audits.
    diagnostics_level: str = "minimal"
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


@dataclass
class HardPACTQPResult:
    """Physical-space primal result plus fallback/KKT metadata.

    ``stage`` is 0 for the full QP, 1 for the relaxed-contact QP, and 2 for
    analytic actuator/rate projection. ``differentiated_mask`` is true only
    for stages 0/1, whose outputs came from qpth's implicit KKT backward.
    """
    qdd: torch.Tensor  # [B,18], canonical generalized acceleration.
    force_world: torch.Tensor  # [B,4,3], FR/FL/RR/RL world XYZ [N].
    tau_safe: torch.Tensor  # [B,12], actuator command [Nm].
    contact_slack: torch.Tensor  # [B,4,3], acceleration slack [m/s^2].
    stage: torch.Tensor  # [B], 0=full, 1=relaxed, 2=analytic projection.
    differentiated_mask: torch.Tensor  # [B], true exactly for stages 0/1.
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

    def __iter__(self):
        # Preserve the legacy seven-value private test/debug unpacking API.
        return iter((self.Q, self.p, self.G, self.h, self.A, self.b,
                     self.variable_scale))


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


class HardPACTDifferentiableQP:
    r"""Construct and solve the fixed 54-variable HardPACT OptNet problem.

    Physical variable ordering is

    .. math:: x=[\ddot q_{18}, f_{12}, \tau_{safe,12}, s_{contact,12}].

    Floating-base ``qdd[:3]`` is m/s^2, angular ``qdd[3:6]`` and joint
    ``qdd[6:]`` are rad/s^2. ``f`` is Newtons in world-frame XYZ and
    FR, FL, RR, RL order.  Slack has foot linear-acceleration units (m/s^2).
    The wrench is a
    world-aligned ``[Fx,Fy,Fz,Tx,Ty,Tz]`` wrench about the base-Jacobian point.
    Dynamics therefore use the same sign convention as BARD:

    .. math:: M\ddot q+b-S^T\tau-J_f^Tf=J_b^TW_{pred}.

    The physical objective is

    .. math::

       w_\tau\left\|\frac{\tau-\tau_{nom}}{\tau_{lim}}\right\|_2^2
       +w_f\left\|\frac{f-\hat f}{f_s}\right\|_2^2
       +w_s\left\|\frac{s}{s_s}\right\|_2^2
       +r_{\ddot q}\|\ddot q\|_2^2
       +r_f\left\|\frac f{f_s}\right\|_2^2
       +r_\tau\left\|\frac\tau{\tau_{lim}}\right\|_2^2
       +r_Q\|x\|_2^2.

    Its inequalities enforce, componentwise,

    .. math::

       \max(-\tau_{lim},\tau_{k-1}-\dot\tau_{lim}\Delta t)
       \le\tau_k\le
       \min(\tau_{lim},\tau_{k-1}+\dot\tau_{lim}\Delta t),

    one-step joint position/velocity bounds, ``f_z>=0``, the fixed-world-Z
    pyramid ``|f_x|,|f_y|<=mu*f_z``, ``s>=0``, and

    .. math::

       |c_i(J_i\ddot q+\dot J_i v)|\le s_i+a_{tol}.

    qpth implicitly differentiates the KKT system. Measured state, matrices,
    limits, and acceleration bias are detached; ``tau_nom``, predicted GRF,
    predicted wrench, and contact probabilities retain gradients.

    For optimum ``(x*,lambda*,nu*)``, qpth differentiates the residual system

    .. math::

       Qx^*+p+G^T\lambda^*+A^T\nu^*=0,\quad
       \operatorname{diag}(\lambda^*)(Gx^*-h)=0,\quad
       Ax^*-b=0,

    rather than differentiating solver iterations. Consequently gradients of
    ``tau_safe``, ``f``, and ``s`` propagate to the learned entries of ``p``,
    ``G``, and ``b`` whenever stage 0 or 1 passes certification.
    """

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
        # D must be invertible, hence every variable scale is strictly positive.
        if min(config.qdd_scale, config.force_scale_n, config.torque_scale_nm,
               config.slack_scale_m_s2) <= 0:
            raise ValueError("all QP variable scales must be positive")
        if config.friction_coefficient <= 0 or config.torque_rate_limit_nm_s <= 0:
            raise ValueError("friction and torque-rate limits must be positive")
        if config.q_regularization <= 0.0:
            raise ValueError("QP q_regularization must be strictly positive")
        if min(
            config.torque_tracking_weight, config.force_tracking_weight,
            config.slack_weight, config.qdd_regularization,
            config.force_regularization, config.torque_regularization,
        ) < 0.0:
            raise ValueError("QP weights and regularizers must be nonnegative")
        if ((config.eps is not None and config.eps <= 0.0)
                or config.eps_float32 <= 0.0 or config.eps_float64 <= 0.0
                or config.max_iter <= 0):
            raise ValueError("QP eps and max_iter must be positive")
        chunk_sizes = (config.rollout_chunk_size, config.ppo_chunk_size)
        if config.chunk_size is not None:
            chunk_sizes = (config.chunk_size, config.chunk_size)
        if config.not_improved_limit <= 0 or min(chunk_sizes) <= 0:
            raise ValueError("QP iteration and chunk limits must be positive")
        if not (0.0 < config.contact_probability_floor <= 1.0):
            raise ValueError("contact_probability_floor must lie in (0,1]")
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
        if len(solvers) > 1 and not config.allow_solver_mismatch:
            raise ValueError(
                "different rollout/PPO QP solvers require "
                "allow_solver_mismatch=True"
            )
        if self._full_audit_period(config) < 0:
            raise ValueError("QP full_audit_period must be nonnegative")
        if self._full_audit_sample_size(config) <= 0:
            raise ValueError("QP full_audit_sample_size must be positive")
        self._constant_cache = {}
        self._solve_count = 0
        self._solve_count_by_mode = {False: 0, True: 0}
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

    def clear_warm_start(self, env_ids=None):
        """Clear qpth rollout state globally or for chunks touching env_ids."""
        if env_ids is None:
            self._qpth_warm_states.clear()
            return
        # Keep reset ownership updates on the warm-state device. A reset is a
        # hot rollout path and must not introduce GPU->CPU->GPU synchronization.
        ids = torch.as_tensor(env_ids).detach()
        for key in list(self._qpth_warm_states):
            start, stop, _ = key
            state, valid = self._qpth_warm_states[key]
            device_ids = ids.to(valid.device)
            local = device_ids[
                (device_ids >= start) & (device_ids < stop)
            ] - start
            if local.numel():
                valid = valid.clone()
                valid[local] = False
                self._qpth_warm_states[key] = (state, valid)

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
        return self.diagnostics_level in ("physical", "full")

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

    def _chunk_size(self, differentiable):
        if self.cfg.chunk_size is not None:
            return int(self.cfg.chunk_size)
        return int(self.cfg.ppo_chunk_size if differentiable
                   else self.cfg.rollout_chunk_size)

    def _constants(self, reference):
        """Cache immutable 54-D selectors/scales/shared Hessian by device/dtype."""
        key = (reference.device.type, reference.device.index, reference.dtype)
        cached = self._constant_cache.get(key)
        # Rollout may populate the cache under torch.inference_mode(). Such a
        # tensor cannot later be saved by qpth's autograd graph. Rebuild this
        # key once when a differentiable reference first arrives; ordinary
        # cached tensors are safe to read from subsequent inference solves.
        differentiable_reference = torch.is_grad_enabled() and reference.requires_grad
        cached_is_inference = (
            cached is not None and torch.is_inference(cached[2])
        )
        if cached is not None and not (
            differentiable_reference and cached_is_inference
        ):
            return cached
        device, dtype = reference.device, reference.dtype
        eye12 = torch.eye(12, device=device, dtype=dtype)
        selectors = {}
        for name, columns in (("tau", TORQUE), ("qdd", slice(6, 18)),
                              ("slack", SLACK)):
            value = torch.zeros(12, NUM_VARIABLES, device=device, dtype=dtype)
            value[:, columns] = eye12
            selectors[name] = value
        variable_scale = reference.new_tensor(
            [self.cfg.qdd_scale] * 18 + [self.cfg.force_scale_n] * 12
            + [self.cfg.torque_scale_nm] * 12
            + [self.cfg.slack_scale_m_s2] * 12
        )
        torque_limit = self._limits(reference)[0]
        diagonal = torch.full(
            (NUM_VARIABLES,), 2.0 * self.cfg.q_regularization,
            device=device, dtype=dtype,
        )
        diagonal[QDD] += 2.0 * self.cfg.qdd_regularization
        diagonal[FORCE] += 2.0 * (
            self.cfg.force_tracking_weight + self.cfg.force_regularization
        ) / self.cfg.force_scale_n ** 2
        diagonal[TORQUE] += 2.0 * (
            self.cfg.torque_tracking_weight + self.cfg.torque_regularization
        ) / torque_limit.square()
        diagonal[SLACK] += 2.0 * self.cfg.slack_weight / (
            self.cfg.slack_scale_m_s2 ** 2
        )
        shared_q = torch.diag(diagonal)
        friction = torch.zeros(20, NUM_VARIABLES, device=device, dtype=dtype)
        for foot in range(4):
            col, row = FORCE.start + 3 * foot, 5 * foot
            friction[row, col + 2] = -1.0
            friction[row + 1, col] = 1.0
            friction[row + 1, col + 2] = -self.cfg.friction_coefficient
            friction[row + 2, col] = -1.0
            friction[row + 2, col + 2] = -self.cfg.friction_coefficient
            friction[row + 3, col + 1] = 1.0
            friction[row + 3, col + 2] = -self.cfg.friction_coefficient
            friction[row + 4, col + 1] = -1.0
            friction[row + 4, col + 2] = -self.cfg.friction_coefficient
        cached = selectors, friction, variable_scale, shared_q
        self._constant_cache[key] = cached
        return cached

    def _limits(self, reference):
        """Materialize fixed limits beside a minibatch without persistent copies."""
        return tuple(value.to(device=reference.device, dtype=reference.dtype)
                     for value in (self.torque_limits, self.position_lower,
                                   self.position_upper, self.velocity_limits))

    def _build(self, data, relaxed_contact=False):
        r"""Build physical blocks, then apply ``x=D z`` and row scaling.

        Every inequality is written ``G x <= h``. The fixed-gravity-normal
        friction pyramid uses world Z: ``|fx|,|fy| <= mu*fz`` and ``fz>=0``.
        Contact acceleration is softened symmetrically as
        ``|c_i*(J_i*qdd+jdotv_i)| <= slack_i + a_tol`` with ``slack>=0``.

        Input shapes use ``B`` environments, ``n_v=18``, four feet, and three
        Cartesian axes: ``M:[B,18,18]``, ``J_f:[B,4,3,18]``,
        ``J_b:[B,6,18]``. All Jacobians are world-axis and all generalized
        coordinates are canonical ``[base-linear, base-angular, FR,FL,RR,RL]``.
        """
        # Measured mechanics are constants in the OptNet graph. Only learned
        # references below remain attached to autograd.
        mass = data["mass_matrix"].detach()
        bias = data["bias"].detach()
        foot_jac = data["foot_jacobians"].detach()
        base_jac = data["base_jacobian"].detach()
        foot_bias = data["foot_acceleration_bias"].detach()
        # tau_nom, f_hat, and W_hat are learned QP inputs. In the equality,
        # W_hat enters b and therefore retains a qpth implicit-gradient path.
        tau_nom = data["tau_nom"]
        force_pred = data["force_pred_world"].reshape(-1, 12)
        wrench_pred = data["wrench_pred_world"]
        # c_i in [0,1] continuously gates the contact-acceleration rows.
        contact = data["contact_probability"].clamp(0.0, 1.0)
        # Previous execution/state/dt define hard boxes and are measurements.
        previous_tau = data["previous_torque"].detach()
        joint_position = data["joint_position"].detach()
        joint_velocity = data["joint_velocity"].detach()
        dt = data["dt"].detach().reshape(-1, 1).clamp_min(1.0e-8)
        # Derive allocation metadata from the live learned torque reference.
        batch = tau_nom.shape[0]
        device, dtype = tau_nom.device, tau_nom.dtype
        # Copy fixed backend limits to the current solver device/dtype only.
        torque_limit, q_lower, q_upper, velocity_limit = self._limits(tau_nom)
        selectors, friction_template, variable_scale, shared_q = self._constants(tau_nom)

        # Objective is assembled in physical x. For w||y-y*||^2, qpth's
        # 1/2*x'Q*x+p'x convention requires Q=2wI and p=-2w*y*.
        # Every weight is nonnegative and q_regularization is strictly
        # positive, so the symmetrized Q below is positive definite and the
        # OptNet problem is a strictly convex QP with a unique primal result.
        # Start Q and p for 1/2*x^T Q x + p^T x in physical coordinates.
        Q = shared_q.unsqueeze(0).expand(batch, -1, -1)
        p = torch.zeros(batch, NUM_VARIABLES, device=device, dtype=dtype)
        # Expanding w_f||(f-f_hat)/f_s||^2 gives
        # p_f=-2*w_f*f_hat/f_s^2 (the constant ||f_hat|| term is discarded).
        p[:, FORCE] = -2.0 * self.cfg.force_tracking_weight * force_pred / (
            self.cfg.force_scale_n ** 2
        )
        # Likewise p_tau=-2*w_tau*tau_nom/tau_lim^2.
        p[:, TORQUE] = -2.0 * self.cfg.torque_tracking_weight * tau_nom / (
            torque_limit.square().unsqueeze(0)
        )

        # Dynamics equality A*x=b. M and all Jacobians use canonical
        # [base linear, base angular, FR,FL,RR,RL joints] generalized order.
        # Starting from M*qdd+b=S^T*tau+J_f^T*f+J_b^T*W, move all decision
        # variables left and the fixed learned wrench/bias right:
        # [M, -J_f^T, -S^T, 0] x = J_b^T*W - b.
        A = torch.zeros(batch, 18, NUM_VARIABLES, device=device, dtype=dtype)
        # Coefficient of qdd is the realized generalized mass matrix M.
        A[:, :, QDD] = mass
        # Flatten feet in FR,FL,RR,RL XYZ order, then transpose J_f to J_f^T.
        A[:, :, FORCE] = -foot_jac.reshape(batch, 12, 18).transpose(1, 2)
        # S^T=[0_{6x12};I_12], hence only actuated rows 6:18 are -I.
        A[:, 6:, TORQUE] = -torch.eye(12, device=device, dtype=dtype)
        # einsum computes J_b^T W because base_jac is stored [B,6,18].
        b = torch.einsum("bkn,bk->bn", base_jac, wrench_pred) - bias

        # Inequality blocks are collected independently, then concatenated
        # into one fixed G/h pair. `add(C,d)` always means C*x<=d.
        rows, bounds = [], []
        def add(block, bound):
            rows.append(block)
            bounds.append(bound)

        # Reusable I_12 and selection matrices extract tau, actuated qdd, or s
        # from x without changing the fixed 54-D variable layout.
        eye12 = torch.eye(12, device=device, dtype=dtype).expand(batch, -1, -1)
        selector_tau = selectors["tau"].unsqueeze(0).expand(batch, -1, -1)
        selector_qdd = selectors["qdd"].unsqueeze(0).expand(batch, -1, -1)
        selector_slack = selectors["slack"].unsqueeze(0).expand(batch, -1, -1)

        # Intersect magnitude and rate boxes before adding rows.  Two pairs of
        # parallel rows are mathematically valid but redundant; qpth's primal-
        # dual factorization is substantially better conditioned with the
        # exact same feasible set represented by one lower/upper pair.
        # Delta_tau_max = dot_tau_lim*dt at this *physics* substep.
        rate_delta = self.cfg.torque_rate_limit_nm_s * dt
        # Upper intersection: min(tau_lim, tau_previous+Delta_tau_max).
        tau_upper = torch.minimum(
            torque_limit.expand(batch, -1), previous_tau + rate_delta
        )
        # Lower intersection: max(-tau_lim, tau_previous-Delta_tau_max).
        tau_lower = torch.maximum(
            -torque_limit.expand(batch, -1), previous_tau - rate_delta
        )
        # +I*tau<=tau_upper.
        add(selector_tau, tau_upper)
        # -I*tau<=-tau_lower, equivalent to tau>=tau_lower.
        add(-selector_tau, -tau_lower)

        # One-step joint limits use the configured backend coefficient alpha:
        # v+ = v + dt*qdd; q+ = q + dt*v + alpha*dt^2*qdd.
        # Isolate qdd in q_lower<=q+dt*v+alpha*dt^2*qdd<=q_upper.
        position_coefficient = (
            self.cfg.position_integration_coefficient * dt.square()
        )
        # Isolate qdd in -v_lim<=v+dt*qdd<=v_lim.
        velocity_coefficient = dt
        # Position-derived componentwise qdd upper/lower bounds.
        qdd_upper_position = (
            q_upper - joint_position - dt * joint_velocity
        ) / position_coefficient
        qdd_lower_position = (
            q_lower - joint_position - dt * joint_velocity
        ) / position_coefficient
        # Velocity-derived componentwise qdd upper/lower bounds.
        qdd_upper_velocity = (
            velocity_limit - joint_velocity
        ) / velocity_coefficient
        qdd_lower_velocity = (
            -velocity_limit - joint_velocity
        ) / velocity_coefficient
        # Their intersection enforces position and velocity simultaneously.
        qdd_upper = torch.minimum(qdd_upper_position, qdd_upper_velocity)
        qdd_lower = torch.maximum(qdd_lower_position, qdd_lower_velocity)
        # Select generalized coordinates 6:18 because the floating base has
        # no actuator joint position/velocity box in this QP.
        add(selector_qdd, qdd_upper)
        add(-selector_qdd, -qdd_lower)

        # World-Z unilateral/friction pyramid, five rows per foot.
        friction_rows = friction_template.unsqueeze(0).expand(batch, -1, -1)
        # All five pyramid bounds have a zero right-hand side.
        add(friction_rows, torch.zeros(batch, 20, device=device, dtype=dtype))
        if not relaxed_contact:
            # -s<=0 <=> s>=0 for the full problem.
            add(-selector_slack, torch.zeros(
                batch, 12, device=device, dtype=dtype
            ))

        if not relaxed_contact:
            # c is scalar per foot and is repeated over XYZ. Scaling the whole
            # acceleration makes the constraint vanish continuously in swing.
            # Repeat each scalar c_i over its foot's XYZ acceleration axes.
            c_min = self.cfg.contact_probability_floor
            contact_effective = c_min + (1.0 - c_min) * contact
            contact_xyz = contact_effective.unsqueeze(-1).expand(
                -1, -1, 3
            ).reshape(batch, 12)
            # c_i*J_i is the qdd coefficient in c_i*a_i, where
            # a_i=J_i*qdd+(Jdot_i*v).
            weighted_jacobian = foot_jac.reshape(batch, 12, 18) * contact_xyz.unsqueeze(-1)
            # c_i*(Jdot_i*v) is the known affine acceleration term.
            weighted_bias = foot_bias.reshape(batch, 12) * contact_xyz
            # Positive side: cJ*qdd-s <= a_tol-c*Jdot*v.
            contact_rows = torch.zeros(batch, 12, NUM_VARIABLES,
                                       device=device, dtype=dtype)
            contact_rows[:, :, QDD] = weighted_jacobian
            contact_rows[:, :, SLACK] = -eye12
            # A tiny margin makes s=0 a strict interior point when c=0,
            # avoiding duplicate active rows with the independent s>=0 box.
            tolerance = torch.full_like(
                weighted_bias,
                self.cfg.contact_acceleration_limit_m_s2
                + self.cfg.interior_margin,
            )
            add(contact_rows, tolerance - weighted_bias)
            # Negative side: -cJ*qdd-s <= a_tol+c*Jdot*v. Together the two
            # rows encode |c_i*a_i|<=s_i+a_tol independently for XYZ.
            opposite = contact_rows.clone()
            opposite[:, :, QDD].neg_()
            add(opposite, tolerance + weighted_bias)

        # Concatenation order is torque box, joint-step box, friction pyramid,
        # slack nonnegativity, positive contact acceleration, negative contact
        # acceleration. This order is fixed across every full solve.
        G = torch.cat(rows, dim=1)
        h = torch.cat(bounds, dim=1)

        # x=D*z makes all solver variables O(1). Constraint-row scaling then
        # balances Newton-system rows without changing equality/inequality sets.
        # D=diag([qdd_scale]*18,[force_scale]*12,[torque_scale]*12,
        #        [slack_scale]*12), with x=Dz. z is dimensionless/O(1).
        # Keep exact physical matrices for post-solve certification before
        # applying the variable substitution and numerical row scaling.
        physical_G, physical_h = G, h
        physical_A, physical_b = A, b
        # 1/2*(Dz)^T Q (Dz) = 1/2*z^T(DQD)z.
        Q = Q * variable_scale.view(1, -1, 1) * variable_scale.view(1, 1, -1)
        # p^T(Dz)=(Dp)^Tz because D is diagonal.
        p = p * variable_scale
        # Normalize physical rows by max(||row||,|rhs|,1), then apply x=Dz.
        # The order matters numerically: including D in the row norm would
        # erase the useful relative scaling between constraint families and
        # caused qpth to return poor but primal-feasible force iterates.
        A, b, equality_row_scale = _row_scale(A, b)
        G, h, inequality_row_scale = _row_scale(G, h)
        # A(Dz)=b and G(Dz)<=h scale matrix columns, not right-hand sides.
        A = A * variable_scale
        G = G * variable_scale
        # Remove roundoff asymmetry before Cholesky and add the final strictly
        # positive diagonal in solver space, following qpth's SPD guidance.
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        # qpth requires Q strictly positive definite, not merely semidefinite.
        # This solver-space ridge ensures lambda_min(Q)>0 after roundoff.
        Q = Q.clone()
        Q.diagonal(dim1=-2, dim2=-1).add_(self.cfg.q_regularization)
        return _QPBuild(
            Q, p, G, h, A, b, variable_scale,
            physical_G, physical_h, physical_A, physical_b,
            equality_row_scale, inequality_row_scale,
            tau_lower, tau_upper, qdd_lower, qdd_upper,
        )

    def _validate_inputs(self, matrices, audit_count=0):
        r"""Check mandatory finiteness and optional sampled matrix health.

        Only finiteness controls whether a row may enter qpth. Rank and SPD
        values are detached observations in ``full`` diagnostics; enabling an
        audit must never change a QP solution or fallback decision.
        """
        Q, p, G, h, A, b, _ = matrices
        # Reduce each coefficient tensor to one finite/nonfinite bit per env.
        finite = torch.stack([
            tensor.reshape(tensor.shape[0], -1).isfinite().all(dim=-1)
            for tensor in (Q, p, G, h, A, b)
        ]).all(dim=0)
        # Matrix health is calculated once in _diagnostics for sampled full
        # audits. These compatibility fields are deliberately observational.
        rank_ok = torch.ones_like(finite)
        spd = torch.ones_like(finite)
        return finite, finite, rank_ok, spd

    @staticmethod
    def _audit_inequality_groups(relaxed_contact, device):
        """Return fixed row indices for active-set summaries."""
        groups = {
            "torque_box": torch.arange(0, 24, device=device),
            "joint_step": torch.arange(24, 48, device=device),
            "unilateral_force": torch.tensor(
                [48, 53, 58, 63], device=device
            ),
            "friction_pyramid": torch.tensor(
                [row for foot in range(4)
                 for row in range(49 + 5 * foot, 53 + 5 * foot)],
                device=device,
            ),
        }
        if not relaxed_contact:
            groups.update({
                "slack": torch.arange(68, 80, device=device),
                "contact_acceleration": torch.arange(80, 104, device=device),
            })
        return groups

    def _diagnostics(
        self, x_scaled, matrices, audit_count=0,
        relaxed_contact=False, data=None,
    ):
        r"""Evaluate primal and approximate dual KKT residuals in z-space.

        With equality multiplier ``nu`` and inequality multiplier
        ``lambda>=0``, the KKT conditions are

        .. math::

           Az-b=0,\quad Gz-h\le0,\quad
           Qz+p+A^T\nu+G^T\lambda=0,\quad
           \lambda\odot(Gz-h)=0.

        qpth owns the actual backward multipliers. The least-squares values
        computed here are detached diagnostics only.
        """
        Q, p, G, h, A, b, variable_scale = matrices
        with torch.no_grad():
            equality = torch.einsum("bij,bj->bi", A, x_scaled) - b
            inequality = torch.einsum("bij,bj->bi", G, x_scaled) - h
            finite_output = torch.isfinite(x_scaled).all(dim=-1)
        result = {
            # Mandatory normalized primal certification. These values are
            # computed at every diagnostics level because they decide whether
            # qpth's candidate advances or enters the fallback cascade.
            "equality_max": equality.detach().abs().amax(dim=-1),
            "inequality_max": inequality.detach().clamp_min(0.0).amax(dim=-1),
            "output_finite": finite_output,
        }
        if self._physical_enabled():
            # Physical matrix products are reporting-only and intentionally
            # absent from minimal training.
            with torch.no_grad():
                physical_x = x_scaled * variable_scale
                physical_equality = (
                    torch.einsum(
                        "bij,bj->bi", matrices.physical_A, physical_x
                    ) - matrices.physical_b
                )
                physical_inequality = (
                    torch.einsum(
                        "bij,bj->bi", matrices.physical_G, physical_x
                    ) - matrices.physical_h
                )
            result.update({
                "physical_equality_max": physical_equality.abs().amax(dim=-1),
                "physical_inequality_max": physical_inequality.clamp_min(0.0).amax(dim=-1),
                "physical_base_linear_equality_max": physical_equality[:, :3].abs().amax(dim=-1),
                "physical_base_angular_equality_max": physical_equality[:, 3:6].abs().amax(dim=-1),
                "physical_joint_equality_max": physical_equality[:, 6:18].abs().amax(dim=-1),
            })
        # Full multiplier reconstruction is sampled debug instrumentation;
        # normal acceptance uses exact primal residuals only.
        stationarity = x_scaled.new_full((x_scaled.shape[0],), float("nan"))
        stationarity_raw = stationarity.clone()
        complementarity = stationarity.clone()
        audit_count = min(int(audit_count), x_scaled.shape[0])
        if audit_count:
            gradient = torch.einsum("bij,bj->bi", Q, x_scaled).detach() + p.detach()
            active = inequality > -self.cfg.active_tolerance
            stationarity_values, raw_values, complementarity_values = [], [], []
            for index in range(audit_count):
            # G_A contains the active inequality rows for one environment.
                active_g = G[index, active[index]]
            # Diagnostic multipliers are detached; implicit gradients are
            # provided exclusively by qpth's backward KKT solve. Alternate
            # equality least-squares with a projected (lambda>=0) active-set
            # solve; this validates dual feasibility without retaining another
            # differentiable KKT graph.
            # Stationarity matrix columns are [A^T,G_A^T].
                a_t = A[index].detach().transpose(0, 1)
                g_t = active_g.detach().transpose(0, 1)
                grad = gradient[index]
                combined = torch.cat((a_t, g_t), dim=-1)
            # Minimum-norm initial multipliers solve
            # [A^T,G_A^T][nu,lambda] approximately equal to -grad.
                initial = torch.linalg.pinv(combined) @ (-grad)
                equality_multiplier = initial[:A.shape[1]]
            # Inequality dual feasibility requires lambda>=0.
                inequality_multiplier = initial[A.shape[1]:].clamp_min(0.0)
            # A few alternating least-squares projections improve the
            # detached multiplier estimate without another KKT factorization.
                for _ in range(3):
                    if active_g.shape[0]:
                        inequality_multiplier = (
                            torch.linalg.pinv(g_t)
                            @ (-grad - a_t @ equality_multiplier)
                        ).clamp_min(0.0)
                    equality_multiplier = (
                        torch.linalg.pinv(a_t)
                        @ (-grad - g_t @ inequality_multiplier)
                    )
            # r_dual=Qz+p+A^T*nu+G_A^T*lambda.
                residual = grad + a_t @ equality_multiplier + g_t @ inequality_multiplier
                raw_stationarity = residual.abs().max()
                raw_values.append(raw_stationarity)
            # Q and p are dimensionless scaled-space coefficients whose
            # magnitude changes with configured tracking weights. Report and
            # gate on a relative KKT stationarity residual while retaining the
            # raw value for diagnosis.
                stationarity_values.append(
                    raw_stationarity / (1.0 + grad.abs().max())
                )
                if active_g.shape[0]:
                # Complementarity residual max|lambda_i*(G_i z-h_i)|.
                    complementarity_values.append((
                        inequality_multiplier * inequality[index, active[index]].detach()
                    ).abs().max())
                else:
                    complementarity_values.append(residual.new_zeros(()))
            stationarity[:audit_count] = torch.stack(stationarity_values)
            stationarity_raw[:audit_count] = torch.stack(raw_values)
            complementarity[:audit_count] = torch.stack(complementarity_values)
            # Matrix-health audits operate on only the configured sample.
            eigenvalues = torch.linalg.eigvalsh(Q[:audit_count].detach())
            singular_values = torch.linalg.svdvals(A[:audit_count].detach())
            _, cholesky_info = torch.linalg.cholesky_ex(Q[:audit_count].detach())

            def sampled(values):
                aligned = x_scaled.new_full(
                    (x_scaled.shape[0],), float("nan")
                )
                aligned[:audit_count] = values
                return aligned

            q_min = eigenvalues[:, 0]
            q_max = eigenvalues[:, -1]
            a_min = singular_values[:, -1]
            a_max = singular_values[:, 0]
            result.update({
                "stationarity_max": stationarity,
                "stationarity_raw_max": stationarity_raw,
                "complementarity_max": complementarity,
                "audit_q_eigen_min": sampled(q_min),
                "audit_q_eigen_max": sampled(q_max),
                "audit_q_condition": sampled(q_max / q_min.clamp_min(1.0e-30)),
                "audit_q_cholesky_success": sampled(
                    (cholesky_info == 0).to(x_scaled.dtype)
                ),
                "audit_a_singular_min": sampled(a_min),
                "audit_a_singular_max": sampled(a_max),
                "audit_a_rank": sampled(
                    (singular_values > 1.0e-9).sum(dim=-1).to(x_scaled.dtype)
                ),
                "audit_equality_row_scale_min": sampled(
                    matrices.equality_row_scale[:audit_count].amin(dim=-1)
                ),
                "audit_equality_row_scale_max": sampled(
                    matrices.equality_row_scale[:audit_count].amax(dim=-1)
                ),
                "audit_equality_row_scale_mean": sampled(
                    matrices.equality_row_scale[:audit_count].mean(dim=-1)
                ),
                "audit_inequality_row_scale_min": sampled(
                    matrices.inequality_row_scale[:audit_count].amin(dim=-1)
                ),
                "audit_inequality_row_scale_max": sampled(
                    matrices.inequality_row_scale[:audit_count].amax(dim=-1)
                ),
                "audit_inequality_row_scale_mean": sampled(
                    matrices.inequality_row_scale[:audit_count].mean(dim=-1)
                ),
            })
            for name, indices in self._audit_inequality_groups(
                relaxed_contact, inequality.device
            ).items():
                result[f"audit_active_fraction_{name}"] = sampled(
                    active[:audit_count, indices].float().mean(dim=-1)
                )

            if data is not None:
                # Decompose the configured physical objective on the audited
                # candidates. Constant target-only offsets are omitted, just
                # as they are from qpth's p/Q representation.
                physical = x_scaled[:audit_count] * variable_scale
                tau = physical[:, TORQUE]
                force = physical[:, FORCE]
                qdd = physical[:, QDD]
                slack = physical[:, SLACK]
                tau_ref = data["tau_nom"][:audit_count].detach()
                force_ref = data["force_pred_world"][:audit_count].reshape(-1, 12).detach()
                tau_limit = self._limits(tau)[0]
                objective_terms = {
                    "torque_tracking": self.cfg.torque_tracking_weight
                    * ((tau - tau_ref) / tau_limit).square().sum(dim=-1),
                    "force_tracking": self.cfg.force_tracking_weight
                    * ((force - force_ref) / self.cfg.force_scale_n).square().sum(dim=-1),
                    "slack": self.cfg.slack_weight
                    * (slack / self.cfg.slack_scale_m_s2).square().sum(dim=-1),
                    "qdd_regularization": self.cfg.qdd_regularization
                    * qdd.square().sum(dim=-1),
                    "force_regularization": self.cfg.force_regularization
                    * (force / self.cfg.force_scale_n).square().sum(dim=-1),
                    "torque_regularization": self.cfg.torque_regularization
                    * (tau / tau_limit).square().sum(dim=-1),
                    "ridge": self.cfg.q_regularization
                    * physical.square().sum(dim=-1),
                    # The final solver-space diagonal ridge contributes
                    # 1/2*r_Q*||z||^2 in qpth's 1/2*z^TQz convention.
                    "solver_space_spd_ridge": 0.5
                    * self.cfg.q_regularization
                    * x_scaled[:audit_count].square().sum(dim=-1),
                }
                for name, values in objective_terms.items():
                    result[f"audit_objective_{name}"] = sampled(values)
                result["audit_objective_total"] = sampled(
                    sum(objective_terms.values())
                )
        return result

    def _solve_stage(self, data, relaxed_contact, audit_count=0, warm_key=None):
        r"""Build, validate, solve, and certify one QP fallback stage.

        qpth returns ``z*`` and its backward differentiates the KKT system;
        this method returns physical ``x*=D z*`` only after residual checks.
        """
        # Construct either the full contact-softened problem or stage one with
        # contact/slack inequalities removed. No ``s=0`` equality is added:
        # the strictly positive quadratic slack cost has its minimum at zero.
        matrices = self._build(data, relaxed_contact=relaxed_contact)
        # Validate coefficients before entering qpth's batched factorization.
        valid, finite, rank_ok, spd = self._validate_inputs(
            matrices, audit_count=audit_count
        )
        Q, p, G, h, A, b, variable_scale = matrices
        # A bad row cannot be sent through a batched factorization. Replace it
        # by a finite, known-SPD placeholder; its output is rejected below and
        # deterministically falls through to the next stage.
        invalid = ~valid
        if invalid.any():
            # Q=I is finite/SPD and cannot poison other environments' batch.
            eye = torch.eye(NUM_VARIABLES, device=Q.device, dtype=Q.dtype)
            Q = torch.where(invalid[:, None, None], eye, Q)
            # p=0 makes z=0 the placeholder objective minimizer.
            p = torch.where(invalid[:, None], torch.zeros_like(p), p)
            # 0*z<=1 gives a strictly feasible placeholder inequality set.
            G = torch.where(invalid[:, None, None], torch.zeros_like(G), G)
            h = torch.where(invalid[:, None], torch.ones_like(h), h)
            # [I_18,0]z=0 is full-row-rank and feasible at z=0.
            placeholder_a = torch.zeros_like(A)
            placeholder_a[:, :, :18] = torch.eye(
                18, device=A.device, dtype=A.dtype
            )
            A = torch.where(invalid[:, None, None], placeholder_a, A)
            b = torch.where(invalid[:, None], torch.zeros_like(b), b)
            # Diagnostics must inspect exactly the sanitized matrices qpth saw.
            matrices = Q, p, G, h, A, b, variable_scale
        # This bit is separate from coefficient validity: qpth can reject an
        # otherwise valid numerical factorization at runtime.
        solver_exception = torch.zeros_like(valid)
        warm_hit_mask = torch.zeros_like(valid)
        terminal_state = None
        try:
            # Every backend consumes these exact canonical scaled matrices.
            # QP construction, certification, and fallback never live in an
            # adapter, so changing solvers cannot change physical semantics.
            if (
                self._active_solver == "qpth"
                and self.cfg.qpth_warm_start
                and not self._active_differentiable
                and warm_key is not None
            ):
                warm_entry = self._qpth_warm_states.get(warm_key)
                if warm_entry is None:
                    warm_state, warm_mask = None, None
                else:
                    warm_state, warm_mask = warm_entry
                    warm_hit_mask = warm_mask.to(valid.device).clone()
                solution_scaled, terminal_state = solve_qpth_warm(
                    Q, p, G, h, A, b, warm_start=warm_state,
                    warm_mask=warm_mask,
                    eps=self._eps(Q.dtype),
                    verbose=(-1 if self.cfg.verbose == 0 else self.cfg.verbose),
                    not_improved_limit=self.cfg.not_improved_limit,
                    max_iter=self.cfg.max_iter,
                    check_q_spd=self.cfg.check_q_spd,
                )
            elif self._active_solver == "qpth":
                solution_scaled = QPFunction(
                    eps=self._eps(Q.dtype),
                    # qpth gates its inaccurate-solution banner on verbose>=0.
                    # Our mandatory primal checks classify the candidate, so
                    # public quiet mode maps to -1 without weakening safety.
                    verbose=(-1 if self.cfg.verbose == 0 else self.cfg.verbose),
                    notImprovedLim=self.cfg.not_improved_limit,
                    maxIter=self.cfg.max_iter,
                    check_Q_spd=self.cfg.check_q_spd,
                )(Q, p, G, h, A, b)
            else:
                solution_scaled = self._backend_instances[
                    self._active_solver
                ].solve(
                    Q, p, G, h, A, b,
                    differentiable=self._active_differentiable,
                )
        except QPBackendUnavailable:
            # An explicitly selected unavailable GPU solver is a configuration
            # error, never a reason to execute another backend or CPU path.
            raise
        except (RuntimeError, ValueError):
            # qpth factorizes a complete batch. A numerical failure therefore
            # rejects this stage for the chunk; solve() proceeds to the less
            # constrained stage, then the analytic actuator/rate projection.
            solution_scaled = torch.zeros_like(p)
            solver_exception = torch.ones_like(valid)
        if solution_scaled is None:
            # Defensive handling for solver versions that return None instead
            # of throwing when their factorization fails.
            solution_scaled = torch.zeros_like(p)
            solver_exception = torch.ones_like(valid)
        # Certify the returned z against the precise scaled problem.
        diagnostics = self._diagnostics(
            solution_scaled, matrices, audit_count=audit_count,
            relaxed_contact=relaxed_contact, data=data,
        )
        # Dual KKT validity requires both relative stationarity and
        # complementarity below their configured tolerance.
        if audit_count:
            diagnostics["kkt_valid"] = (
                (diagnostics["stationarity_max"] <= self.cfg.kkt_tolerance)
                & (diagnostics["complementarity_max"] <= self.cfg.kkt_tolerance)
            )
        # A stage is accepted only if all input, solver, finite-output, primal,
        # and dual checks pass for that environment.
        success = (
            valid & ~solver_exception
            & diagnostics["output_finite"]
            & (diagnostics["equality_max"] <= self._normalized_tolerance(Q.dtype))
            & (diagnostics["inequality_max"] <= self._normalized_tolerance(Q.dtype))
        )
        if warm_key is not None and terminal_state is not None:
            # Install only certified terminal rows. Reset/failed rows remain
            # cold next substep, while every unaffected environment keeps its
            # own primal/dual/slack iterate.
            self._qpth_warm_states[warm_key] = (
                terminal_state, success.detach().clone()
            )
        elif warm_key is not None and warm_key in self._qpth_warm_states:
            # No terminal state means the batched warm call raised before a
            # candidate existed; every row is cold-retried below, so discard
            # the structurally bad cached entry and let the next solve seed it.
            self._qpth_warm_states.pop(warm_key, None)
        if warm_key is not None and not success.all():
            # Any failed warm candidate must be cold-resolved before entering
            # the relaxed-contact stage. Successful rows retain their states;
            # failed rows stay invalid even if this one-off cold retry passes.
            failed = ~success
            cold_data = {name: value[failed] for name, value in data.items()}
            cold_solution, cold_ok, cold_diag = self._solve_stage(
                cold_data, relaxed_contact, audit_count=0, warm_key=None,
            )
            solution_scaled = solution_scaled.clone()
            solution_scaled[failed] = (
                cold_solution / variable_scale
            )
            success = success.clone()
            success[failed] = cold_ok
            for name, values in cold_diag.items():
                if name in diagnostics and diagnostics[name].shape == success.shape:
                    diagnostics[name] = diagnostics[name].clone()
                    diagnostics[name][failed] = values
            diagnostics["warm_residual_cold_retry"] = failed
        else:
            diagnostics["warm_residual_cold_retry"] = torch.zeros_like(success)
        diagnostics["warm_start_hit"] = warm_hit_mask
        diagnostics.update({"input_finite": finite, "equality_rank": rank_ok,
                            "q_spd": spd,
                            "solver_exception": solver_exception})
        # Undo x=Dz. This multiplication retains qpth's implicit gradient.
        return solution_scaled * variable_scale, success, diagnostics

    def _precheck(self, data):
        """Return cheap validity masks and exact bound intersections."""
        reference = data["tau_nom"]
        torque_limit, q_lower, q_upper, velocity_limit = self._limits(reference)
        previous = data["previous_torque"].detach()
        joint_position = data["joint_position"].detach()
        joint_velocity = data["joint_velocity"].detach()
        dt = data["dt"].detach().reshape(-1, 1)
        rate = self.cfg.torque_rate_limit_nm_s * dt
        tau_lower = torch.maximum(-torque_limit, previous - rate)
        tau_upper = torch.minimum(torque_limit, previous + rate)
        position_coefficient = (
            self.cfg.position_integration_coefficient * dt.square()
        ).clamp_min(torch.finfo(reference.dtype).tiny)
        velocity_coefficient = dt.clamp_min(torch.finfo(reference.dtype).tiny)
        qdd_upper = torch.minimum(
            (q_upper - joint_position - dt * joint_velocity)
            / position_coefficient,
            (velocity_limit - joint_velocity) / velocity_coefficient,
        )
        qdd_lower = torch.maximum(
            (q_lower - joint_position - dt * joint_velocity)
            / position_coefficient,
            (-velocity_limit - joint_velocity) / velocity_coefficient,
        )
        finite = torch.stack([
            value.detach().reshape(value.shape[0], -1).isfinite().all(dim=-1)
            for value in data.values()
        ]).all(dim=0)
        torque_ok = (tau_lower <= tau_upper).all(dim=-1)
        qdd_ok = (qdd_lower <= qdd_upper).all(dim=-1)
        return finite, torque_ok, qdd_ok, tau_lower, tau_upper

    def _full_audit_due(self, differentiable):
        period = self._full_audit_period(self.cfg)
        return (
            self.diagnostics_level == "full"
            and period > 0
            and self._solve_count_by_mode[bool(differentiable)] % period == 0
        )

    @staticmethod
    def _finite_mean(values):
        finite = torch.isfinite(values)
        return torch.where(finite, values, torch.zeros_like(values)).sum() / (
            finite.sum().clamp_min(1).to(values.dtype)
        )

    @staticmethod
    def _absolute_stats(metrics, prefix, values):
        """Add detached scalar mean/p95/max statistics on the live device."""
        flat = values.detach().abs().reshape(-1)
        metrics[f"{prefix}/mean"] = flat.mean()
        metrics[f"{prefix}/p95"] = torch.quantile(flat, 0.95)
        metrics[f"{prefix}/max"] = flat.max()

    def _minimal_summary(self, stage, differentiated, diagnostics):
        """Aggregate mandatory certification/fallback values on-device."""
        prefix = "qp/minimal"
        metrics = {
            f"{prefix}/full_count": (stage == 0).sum(),
            f"{prefix}/relaxed_count": (stage == 1).sum(),
            f"{prefix}/fallback_count": (stage == 2).sum(),
            f"{prefix}/full_fraction": (stage == 0).float().mean(),
            f"{prefix}/relaxed_fraction": (stage == 1).float().mean(),
            f"{prefix}/fallback_fraction": (stage == 2).float().mean(),
            f"{prefix}/differentiated_fraction": differentiated.float().mean(),
            f"{prefix}/pre_clamp_torque_violation_max": diagnostics[
                "pre_clamp_torque_violation_max"
            ].max(),
            f"{prefix}/solver_qpth": stage.new_tensor(
                float(self._active_solver == "qpth"), dtype=torch.float32
            ),
            f"{prefix}/solver_cupiqp": stage.new_tensor(
                float(self._active_solver == "cupiqp"), dtype=torch.float32
            ),
            f"{prefix}/solver_moreau": stage.new_tensor(
                float(self._active_solver == "moreau"), dtype=torch.float32
            ),
            f"{prefix}/rollout_ppo_solver_mismatch": stage.new_tensor(
                float(self.solver_for_mode(False) != self.solver_for_mode(True)),
                dtype=torch.float32,
            ),
        }
        for source, name in (
            ("selected/equality_max", "normalized_equality_residual_max"),
            ("selected/inequality_max", "normalized_inequality_violation_max"),
        ):
            metrics[f"{prefix}/{name}"] = diagnostics[source].nan_to_num().max()
        for source in (
            "failure/nonfinite_input",
            "failure/empty_torque_intersection",
            "failure/empty_qdd_intersection",
        ):
            metrics[f"{prefix}/{source}"] = diagnostics[source].float().mean()
        for stage_name in ("full", "relaxed"):
            for source in ("input_finite", "output_finite", "solver_exception"):
                key = f"{stage_name}/{source}"
                if key in diagnostics:
                    metrics[f"{prefix}/{key}_fraction"] = (
                        diagnostics[key].float().mean()
                    )
        for name in ("warm_start_hit", "warm_residual_cold_retry"):
            key = f"full/{name}"
            metrics[f"{prefix}/{name}_fraction"] = (
                diagnostics[key].float().mean()
                if key in diagnostics else stage.new_tensor(0.0, dtype=torch.float32)
            )
        return metrics

    def _physical_summary(self, solution, data):
        """Compute physical-unit summaries without host transfer or sync."""
        prefix = "qp/physical"
        qdd = solution[:, QDD]
        force = solution[:, FORCE].reshape(-1, 4, 3)
        tau = solution[:, TORQUE]
        slack = solution[:, SLACK].reshape(-1, 4, 3)
        dtype, device = solution.dtype, solution.device

        def measured(name):
            return data[name].detach().to(device=device, dtype=dtype)

        mass = measured("mass_matrix")
        bias = measured("bias")
        foot_jac = measured("foot_jacobians")
        base_jac = measured("base_jacobian")
        wrench = measured("wrench_pred_world")
        dynamics = (
            torch.einsum("bij,bj->bi", mass, qdd) + bias
            - torch.cat((torch.zeros_like(tau[:, :6]), tau), dim=-1)
            - torch.einsum("bfkn,bfk->bn", foot_jac, force)
            - torch.einsum("bkn,bk->bn", base_jac, wrench)
        )
        metrics = {}
        for name, block in (
            ("base_linear", dynamics[:, :3]),
            ("base_angular", dynamics[:, 3:6]),
            ("joint", dynamics[:, 6:18]),
            ("all", dynamics),
        ):
            metrics[f"{prefix}/dynamics_residual/{name}_mae"] = block.abs().mean()
            metrics[f"{prefix}/dynamics_residual/{name}_max"] = block.abs().max()

        torque_limit, q_lower, q_upper, velocity_limit = self._limits(solution)
        previous = measured("previous_torque")
        joint_position = measured("joint_position")
        joint_velocity = measured("joint_velocity")
        dt = measured("dt").reshape(-1, 1)
        next_velocity = joint_velocity + dt * qdd[:, 6:]
        next_position = (
            joint_position + dt * joint_velocity
            + self.cfg.position_integration_coefficient * dt.square() * qdd[:, 6:]
        )
        c = measured("contact_probability").clamp(0.0, 1.0)
        c_eff = self.cfg.contact_probability_floor + (
            1.0 - self.cfg.contact_probability_floor
        ) * c
        foot_acceleration = (
            torch.einsum("bfkn,bn->bfk", foot_jac, qdd)
            + measured("foot_acceleration_bias")
        )
        margins = {
            "torque": torque_limit - tau.abs(),
            "torque_rate": self.cfg.torque_rate_limit_nm_s * dt
            - (tau - previous).abs(),
            "joint_position": torch.minimum(
                next_position - q_lower, q_upper - next_position
            ),
            "joint_velocity": velocity_limit - next_velocity.abs(),
            "unilateral_force": force[..., 2],
            "friction_pyramid": torch.stack((
                self.cfg.friction_coefficient * force[..., 2] - force[..., 0],
                self.cfg.friction_coefficient * force[..., 2] + force[..., 0],
                self.cfg.friction_coefficient * force[..., 2] - force[..., 1],
                self.cfg.friction_coefficient * force[..., 2] + force[..., 1],
            ), dim=-1),
            "slack": slack,
            "contact_acceleration": (
                self.cfg.contact_acceleration_limit_m_s2
                + self.cfg.interior_margin + slack
                - (c_eff.unsqueeze(-1) * foot_acceleration).abs()
            ),
        }
        for name, margin in margins.items():
            metrics[f"{prefix}/margin_min/{name}"] = margin.min()
            metrics[f"{prefix}/violation_max/{name}"] = (
                -margin
            ).clamp_min(0.0).max()

        force_prediction = measured("force_pred_world").reshape_as(force)
        self._absolute_stats(metrics, f"{prefix}/torque_correction", tau - measured("tau_nom"))
        self._absolute_stats(metrics, f"{prefix}/grf_tracking_error", force - force_prediction)
        self._absolute_stats(metrics, f"{prefix}/slack", slack)
        self._absolute_stats(metrics, f"{prefix}/qdd", qdd)
        self._absolute_stats(metrics, f"{prefix}/force", force)
        self._absolute_stats(metrics, f"{prefix}/torque", tau)
        return metrics

    def _full_summary(self, diagnostics, audit_solve, forward_metrics):
        """Aggregate sampled matrix/KKT values and full-stage causes."""
        prefix = "qp/full"
        metrics = {
            f"{prefix}/audit_ran": diagnostics["stage"].new_tensor(
                float(audit_solve), dtype=torch.float32
            ),
            f"{prefix}/full_solver_success_fraction": (
                diagnostics["stage"] == 0
            ).float().mean(),
            f"{prefix}/relaxed_solver_success_fraction": (
                diagnostics["stage"] == 1
            ).float().mean(),
            f"{prefix}/analytic_fallback_fraction": (
                diagnostics["stage"] == 2
            ).float().mean(),
        }
        precheck_failure = (
            diagnostics["failure/nonfinite_input"]
            | diagnostics["failure/empty_torque_intersection"]
            | diagnostics["failure/empty_qdd_intersection"]
        )
        metrics[f"{prefix}/fallback_precheck_fraction"] = (
            ((diagnostics["stage"] == 2) & precheck_failure).float().mean()
        )
        metrics[f"{prefix}/fallback_solver_fraction"] = (
            ((diagnostics["stage"] == 2) & ~precheck_failure).float().mean()
        )
        for stage_name in ("full", "relaxed"):
            exception = diagnostics.get(f"{stage_name}/solver_exception")
            if exception is not None:
                metrics[f"{prefix}/{stage_name}_solver_exception_fraction"] = (
                    exception.float().mean()
                )
        if audit_solve:
            for key, values in diagnostics.items():
                marker = "/audit_"
                if not key.startswith("full/audit_"):
                    continue
                name = key[len("full/audit_"):]
                metrics[f"{prefix}/{name}_mean"] = self._finite_mean(values)
                finite = values[torch.isfinite(values)]
                metrics[f"{prefix}/{name}_min"] = finite.min()
                metrics[f"{prefix}/{name}_max"] = finite.max()
            metrics.update(forward_metrics)
        return metrics

    def solve(self, *, differentiable=None, **data):
        r"""Solve full QP, relaxed-contact QP, then actuator/rate projection.

        Per environment the deterministic cascade is

        ``stage 0``: all dynamics/hard-limit/friction/contact-softening rows;
        ``stage 1``: remove contact/slack inequalities (the positive quadratic
        slack cost makes the unconstrained slack optimum exactly zero);
        ``stage 2``: return
        ``clamp(tau_nom, max(-tau_lim,tau_prev-rate*dt),
        min(tau_lim,tau_prev+rate*dt))`` with other variables zero.

        Only failed rows enter stage 1. This avoids retaining a second qpth
        KKT graph for the common successful case and materially reduces VRAM.
        """
        reference = data["tau_nom"]
        original_dtype = reference.dtype
        solver_dtype = self._solve_dtype(reference)
        self.solver_dtype = solver_dtype  # observable diagnostic, not a cache key
        if differentiable is None:
            differentiable = torch.is_grad_enabled() and any(
                data[name].requires_grad for name in (
                    "tau_nom", "force_pred_world", "wrench_pred_world",
                    "contact_probability",
                )
            )
        solver_name = self.solver_for_mode(bool(differentiable))
        # Fail before constructing/falling back if a requested optional
        # backend is unavailable. This prevents a missing package from being
        # misreported as an ordinary numerical stage-2 fallback.
        require_backend(
            solver_name, device=reference.device, dtype=solver_dtype
        )
        self._active_solver = solver_name
        self._active_differentiable = bool(differentiable)
        chunk_size = self._chunk_size(bool(differentiable))
        self._solve_count += 1
        self._solve_count_by_mode[bool(differentiable)] += 1
        audit_solve = self._full_audit_due(bool(differentiable))
        audit_remaining = (
            self._full_audit_sample_size(self.cfg) if audit_solve else 0
        )
        # CUDA synchronization is forbidden in minimal/physical. A periodic
        # full audit explicitly pays for synchronization so its forward time
        # and peak allocation are meaningful rather than queue-submit times.
        forward_start = None
        cuda_memory_start = None
        if audit_solve:
            if reference.device.type == "cuda":
                torch.cuda.synchronize(reference.device)
                # Torch 2.8's reset API rejects ``torch.device`` even though
                # synchronize accepts it; use the live tensor's ordinal.
                cuda_ordinal = reference.device.index
                if cuda_ordinal is None:
                    cuda_ordinal = torch.cuda.current_device()
                torch.cuda.reset_peak_memory_stats(cuda_ordinal)
                cuda_memory_start = torch.cuda.memory_allocated(cuda_ordinal)
            forward_start = time.perf_counter()
        outputs = []

        for start in range(0, reference.shape[0], chunk_size):
            stop = min(start + chunk_size, reference.shape[0])
            chunk = {
                name: value[start:stop].to(dtype=solver_dtype)
                for name, value in data.items()
            }
            count = stop - start
            finite, torque_ok, qdd_ok, lower, upper = self._precheck(chunk)
            valid = finite & torque_ok & qdd_ok

            # Invalid rows never enter qpth. Initialize all rows as analytic
            # fallbacks, then scatter certified full/relaxed solutions back.
            chosen = chunk["tau_nom"].new_zeros(count, NUM_VARIABLES)
            stage = torch.full(
                (count,), 2, device=reference.device, dtype=torch.long
            )
            differentiated_mask = torch.zeros(
                count, device=reference.device, dtype=torch.bool
            )
            stage_diagnostics = {}

            if valid.any():
                compact = {name: value[valid] for name, value in chunk.items()}
                # Expensive factorization audits are periodic and bounded even
                # for large rollout batches. The leading compact rows are a
                # deterministic sample; ordinary primal certification still
                # covers every environment below.
                audit_count = min(int(valid.sum()), audit_remaining)
                audit_remaining -= audit_count
                warm_key = None
                if (
                    self.cfg.qpth_warm_start
                    and solver_name == "qpth"
                    and not differentiable
                    and bool(valid.all())
                ):
                    warm_key = (start, stop, False)
                elif self.cfg.qpth_warm_start:
                    # Invalid coefficients/bounds clear only their owning
                    # environments. Other rows must retain independent warm
                    # iterates even though this compact solve cannot use the
                    # fixed full-chunk structure on this call.
                    invalid_ids = torch.arange(
                        start, stop, device=valid.device
                    )[~valid]
                    self.clear_warm_start(invalid_ids)
                full, full_ok, full_diag = self._solve_stage(
                    compact, False, audit_count=audit_count, warm_key=warm_key,
                )
                failed = ~full_ok
                relaxed = torch.zeros_like(full)
                relaxed_ok = torch.zeros_like(full_ok)
                relaxed_diag = {
                    key: torch.full_like(value, False if value.dtype == torch.bool
                                         else float("nan"))
                    for key, value in full_diag.items()
                }
                if failed.any():
                    failed_compact = {
                        name: value[failed] for name, value in compact.items()
                    }
                    relaxed_solution, failed_ok, failed_diag = self._solve_stage(
                        failed_compact, True, audit_count=0,
                    )
                    relaxed[failed] = relaxed_solution
                    relaxed_ok[failed] = failed_ok
                    for key, value in failed_diag.items():
                        relaxed_diag[key][failed] = value
                selected_compact = torch.where(
                    (failed & relaxed_ok)[:, None], relaxed, full
                )
                differentiated_compact = full_ok | relaxed_ok
                valid_indices = valid.nonzero(as_tuple=False).squeeze(-1)
                chosen[valid_indices] = selected_compact
                differentiated_mask[valid_indices] = differentiated_compact
                stage[valid_indices] = torch.where(
                    full_ok, torch.zeros_like(full_ok, dtype=torch.long),
                    torch.where(relaxed_ok,
                                torch.ones_like(full_ok, dtype=torch.long),
                                torch.full_like(full_ok, 2, dtype=torch.long)),
                )
                stage_diagnostics = {
                    "full": (valid_indices, full_diag),
                    "relaxed": (valid_indices, relaxed_diag),
                }

            # An empty raw intersection can only arise from corrupted/reset
            # previous torque. Project that center into actuator limits first;
            # normal valid rows use the exact original intersection unchanged.
            torque_limit = self._limits(chunk["tau_nom"])[0]
            dt = chunk["dt"].detach().reshape(-1, 1)
            rate = self.cfg.torque_rate_limit_nm_s * dt
            repaired_previous = chunk["previous_torque"].detach().nan_to_num().clamp(
                -torque_limit, torque_limit
            )
            repaired_lower = torch.maximum(-torque_limit, repaired_previous - rate)
            repaired_upper = torch.minimum(torque_limit, repaired_previous + rate)
            project_lower = torch.where(torque_ok[:, None], lower, repaired_lower)
            project_upper = torch.where(torque_ok[:, None], upper, repaired_upper)
            safe_nominal = chunk["tau_nom"].nan_to_num()
            projected = torch.maximum(
                torch.minimum(safe_nominal, project_upper), project_lower
            )
            fallback = torch.zeros_like(chosen)
            fallback[:, TORQUE] = projected
            chosen = torch.where(differentiated_mask[:, None], chosen, fallback)

            # Even a certified interior-point result may exceed a hard box by
            # roundoff. Preserve the clamp's piecewise gradient while ensuring
            # the command sent to the simulator is exactly in the interval.
            pre_clamp_tau = chosen[:, TORQUE]
            pre_clamp_violation = torch.maximum(
                (pre_clamp_tau - project_upper).clamp_min(0.0),
                (project_lower - pre_clamp_tau).clamp_min(0.0),
            ).amax(dim=-1)
            chosen_tau = torch.maximum(
                torch.minimum(pre_clamp_tau, project_upper), project_lower
            )
            chosen = torch.cat((chosen[:, :TORQUE.start], chosen_tau,
                                chosen[:, TORQUE.stop:]), dim=-1)

            # Scatter diagnostics into fixed batch order. Missing/analytic
            # entries are NaN, while explicit failure reasons are booleans.
            diagnostics = {
                "failure/nonfinite_input": ~finite,
                "failure/empty_torque_intersection": finite & ~torque_ok,
                "failure/empty_qdd_intersection": finite & torque_ok & ~qdd_ok,
                "pre_clamp_torque_violation_max": pre_clamp_violation.detach(),
            }
            diagnostic_keys = set()
            for _, (_, values) in stage_diagnostics.items():
                diagnostic_keys.update(values)
            for prefix in ("full", "relaxed"):
                indices_values = stage_diagnostics.get(prefix)
                for key in diagnostic_keys:
                    template = None
                    if indices_values is not None:
                        template = indices_values[1].get(key)
                    if template is None:
                        template = next(
                            values[key] for _, (_, values) in stage_diagnostics.items()
                            if key in values
                        )
                    fill = False if template.dtype == torch.bool else float("nan")
                    aligned = torch.full(
                        (count,), fill, device=reference.device,
                        dtype=template.dtype,
                    )
                    if indices_values is not None and key in indices_values[1]:
                        aligned[indices_values[0]] = indices_values[1][key]
                    diagnostics[f"{prefix}/{key}"] = aligned
            outputs.append((chosen, stage, differentiated_mask, diagnostics))

        solution = torch.cat([item[0] for item in outputs]).to(original_dtype)
        # ``differentiable=False`` is an explicit API guarantee, not merely a
        # chunk-size hint. This is used by the stop-gradient ablation outside
        # rollout inference mode, so sever every accidental construction path
        # (including the exact post-clamp) before returning.
        if not differentiable:
            solution = solution.detach()
        stage = torch.cat([item[1] for item in outputs])
        differentiated_mask = torch.cat([item[2] for item in outputs])
        all_diagnostic_keys = set().union(*(item[3].keys() for item in outputs))
        diagnostics = {}
        for key in all_diagnostic_keys:
            template = next(item[3][key] for item in outputs if key in item[3])
            fill = False if template.dtype == torch.bool else float("nan")
            diagnostics[key] = torch.cat([
                item[3].get(key, torch.full(
                    (item[1].shape[0],), fill, device=stage.device,
                    dtype=template.dtype,
                ))
                for item in outputs
            ])
        diagnostics["stage"] = stage
        diagnostics["differentiated"] = differentiated_mask
        selected_keys = ["equality_max", "inequality_max", "output_finite"]
        if self._physical_enabled():
            selected_keys.extend((
                "physical_equality_max", "physical_inequality_max",
                "physical_base_linear_equality_max",
                "physical_base_angular_equality_max",
                "physical_joint_equality_max",
            ))
        if audit_solve:
            selected_keys.extend((
                "stationarity_max", "stationarity_raw_max",
                "complementarity_max", "kkt_valid",
            ))
        for key in selected_keys:
            full_value = diagnostics.get(f"full/{key}")
            relaxed_value = diagnostics.get(f"relaxed/{key}")
            if full_value is None:
                dtype = torch.bool if key in ("kkt_valid", "output_finite") else solver_dtype
                fill = False if dtype == torch.bool else float("nan")
                full_value = torch.full(
                    stage.shape, fill, device=stage.device, dtype=dtype
                )
                relaxed_value = full_value.clone()
            fallback_value = (
                torch.zeros_like(full_value) if full_value.dtype == torch.bool
                else torch.full_like(full_value, float("nan"))
            )
            diagnostics[f"selected/{key}"] = torch.where(
                stage == 0, full_value,
                torch.where(stage == 1, relaxed_value, fallback_value),
            )
        forward_metrics = {}
        if audit_solve:
            if reference.device.type == "cuda":
                torch.cuda.synchronize(reference.device)
            elapsed_ms = (time.perf_counter() - forward_start) * 1000.0
            forward_metrics["qp/full/forward_time_ms"] = reference.new_tensor(
                elapsed_ms, dtype=torch.float32
            )
            if reference.device.type == "cuda":
                peak = (
                    torch.cuda.max_memory_allocated(cuda_ordinal)
                    - cuda_memory_start
                ) / (1024.0 ** 2)
                forward_metrics["qp/full/forward_peak_cuda_mib"] = (
                    reference.new_tensor(peak, dtype=torch.float32)
                )

        metrics = self._minimal_summary(stage, differentiated_mask, diagnostics)
        if self._physical_enabled():
            metrics.update(self._physical_summary(solution, data))
        if self.diagnostics_level == "full":
            metrics.update(
                self._full_summary(diagnostics, audit_solve, forward_metrics)
            )
        return HardPACTQPResult(
            solution[:, QDD], solution[:, FORCE].reshape(-1, 4, 3),
            solution[:, TORQUE], solution[:, SLACK].reshape(-1, 4, 3),
            stage, differentiated_mask, diagnostics, metrics,
        )


def projection_loss(tau_safe, tau_nom, torque_limit, physics_valid,
                    differentiated, contact_slack=None, slack_scale=1.0):
    r"""Unbiased sampled-substep projection objective.

    The torque term is ``||(tau_safe-tau_nom)/tau_limit||^2``.  When supplied,
    the nonnegative contact slack contributes
    ``mean((slack/slack_scale)^2)``.  There is deliberately no decimation
    multiplier: one uniformly selected substep per transition is already an
    unbiased estimator of the interval mean.

    ``physics_valid`` is the exact no-push/no-reset/no-timeout/no-teleport
    transition mask. ``differentiated`` additionally removes actuator-only
    fallbacks, while sustained-wrench and randomized-inertia samples remain.
    A zero denominator leaves a graph-connected zero with zero gradients.

    In notation, for selected substep ``K_e`` of transition ``e``,

    .. math::

       \ell_e=\left\|\frac{\tau_{safe,e,K_e}-\tau_{nom,e,K_e}}
       {\tau_{lim}}\right\|_2^2
       +\frac1{12}\left\|\frac{s_{e,K_e}}{s_{scale}}\right\|_2^2,

       L_{proj}=\frac{\sum_e m_e\ell_e}{\max(1,\sum_e m_e)}.

    Since ``K_e`` is uniform on ``{0,...,D-1}``,
    ``E[ell_e,K_e]=(1/D)sum_k ell_e,k``. Multiplication by ``D`` would bias
    the estimator and is intentionally absent.
    """
    # First term: squared normalized Euclidean torque correction.
    per_sample = ((tau_safe - tau_nom) / torque_limit).square().sum(dim=-1)
    if contact_slack is not None:
        # Second term: mean squared normalized XYZ slack over four feet.
        per_sample = per_sample + (
            contact_slack.reshape(contact_slack.shape[0], -1)
            / float(slack_scale)
        ).square().mean(dim=-1)
    # m_e excludes discontinuous transitions and non-qpth stage-2 fallbacks.
    valid = physics_valid.reshape(-1).bool() & differentiated.reshape(-1).bool()
    # Cast the boolean mask only at the final reduction.
    weights = valid.to(per_sample.dtype)
    # clamp_min gives a graph-connected zero when every mask entry is false.
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


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
