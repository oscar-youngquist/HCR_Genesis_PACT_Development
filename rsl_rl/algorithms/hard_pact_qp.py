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

import torch
from qpth.qp import QPFunction


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
    friction_coefficient: float = 0.6  # mu in |fx|,|fy|<=mu*fz.
    torque_rate_limit_nm_s: float = 1000.0  # dot(tau)_lim [Nm/s].
    contact_acceleration_limit_m_s2: float = 0.0  # a_tol [m/s^2].
    interior_margin: float = 1.0e-5  # Strict-feasibility epsilon for swing rows.
    qdd_scale: float = 50.0  # D diagonal for generalized acceleration.
    force_scale_n: float = 250.0  # D diagonal/reference normalization for GRF.
    torque_scale_nm: float = 40.0  # D diagonal for safe torque variables.
    slack_scale_m_s2: float = 50.0  # D/objective scale for contact slack.
    torque_tracking_weight: float = 20.0  # w_tau.
    force_tracking_weight: float = 5.0  # w_f.
    slack_weight: float = 1000.0  # w_s.
    qdd_regularization: float = 1.0e-3  # r_qdd.
    force_regularization: float = 1.0e-4  # r_f.
    torque_regularization: float = 1.0e-4  # r_tau.
    q_regularization: float = 1.0e-7  # r_Q and final solver-space SPD ridge.
    feasibility_tolerance: float = 2.0e-4  # Primal infinity-norm threshold.
    kkt_tolerance: float = 1.0e-1  # Relative dual/complementarity threshold.
    # qpth returns the best finite interior iterate; 0.1 is used only to infer
    # its numerical active set for post-solve KKT diagnostics.
    active_tolerance: float = 1.0e-1  # -r_ineq cutoff for diagnostic active set.
    eps: float = 1.0e-12  # qpth primal-dual termination epsilon.
    max_iter: int = 20  # qpth interior-point iteration cap.
    not_improved_limit: int = 3  # qpth stagnation cap.
    check_q_spd: bool = True  # Validate Q>0 and ask qpth to do likewise.
    check_equality_rank: bool = True  # Require full-row-rank A.
    # "auto" uses float32 on CUDA to halve the retained KKT graph and float64
    # on CPU for reference tests; either concrete dtype can be forced.
    solver_dtype: str = "auto"
    # User-facing verbosity: 0 is quiet; positive values expose qpth's solver
    # diagnostics. qpth itself treats verbose=0 as permission to print its
    # large inaccurate-solution warning, so the call adapter maps 0 to -1.
    verbose: int = 0
    chunk_size: int = 128  # Maximum simultaneous retained KKT systems.


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
    r"""Condition rows using the equivalence ``a/c*x <= b/c`` for ``c>0``."""
    # Variable substitution has already scaled every constraint column. Only
    # pathological rows above 1e3 are reduced further. Aggressively forcing
    # every row to unit norm is counterproductive for qpth because it changes
    # relative primal/dual initialization across parallel box constraints.
    norm = matrix.detach().square().sum(dim=-1).sqrt()
    scale = (norm / 1.0e3).clamp_min(1.0)
    return matrix / scale.unsqueeze(-1), rhs / scale


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
        # qpth is substantially more stable in float64 on CPU. CUDA float32
        # halves the retained KKT graph, so "auto" chooses by device.
        self.solver_dtype = _dtype_from_name(
            config.solver_dtype, self.torque_limits
        )
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
        if config.eps <= 0.0 or config.max_iter <= 0:
            raise ValueError("QP eps and max_iter must be positive")
        if config.not_improved_limit <= 0 or config.chunk_size <= 0:
            raise ValueError("QP iteration and chunk limits must be positive")

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

        # Objective is assembled in physical x. For w||y-y*||^2, qpth's
        # 1/2*x'Q*x+p'x convention requires Q=2wI and p=-2w*y*.
        # Every weight is nonnegative and q_regularization is strictly
        # positive, so the symmetrized Q below is positive definite and the
        # OptNet problem is a strictly convex QP with a unique primal result.
        # Start Q and p for 1/2*x^T Q x + p^T x in physical coordinates.
        Q = torch.zeros(batch, NUM_VARIABLES, NUM_VARIABLES,
                        device=device, dtype=dtype)
        p = torch.zeros(batch, NUM_VARIABLES, device=device, dtype=dtype)
        # r_Q||x||^2 contributes 2*r_Q to every Q diagonal entry.
        diagonal = torch.full((batch, NUM_VARIABLES), 2.0 * self.cfg.q_regularization,
                              device=device, dtype=dtype)
        # r_qdd||qdd||^2.
        diagonal[:, QDD] += 2.0 * self.cfg.qdd_regularization
        # (w_f+r_f)||f/f_s||^2. The reference cross term is added to p below.
        diagonal[:, FORCE] += 2.0 * (
            self.cfg.force_tracking_weight + self.cfg.force_regularization
        ) / self.cfg.force_scale_n ** 2
        # (w_tau+r_tau)||tau/tau_lim||^2, with a per-joint physical limit.
        diagonal[:, TORQUE] += 2.0 * (
            self.cfg.torque_tracking_weight + self.cfg.torque_regularization
        ) / torque_limit.square().unsqueeze(0)
        # w_s||s/s_s||^2. Slack has zero desired value.
        diagonal[:, SLACK] += 2.0 * self.cfg.slack_weight / (
            self.cfg.slack_scale_m_s2 ** 2
        )
        # Q is diagonal here; copy_ avoids constructing a batched diag matrix.
        Q.diagonal(dim1=-2, dim2=-1).copy_(diagonal)
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
        selector_tau = torch.zeros(batch, 12, NUM_VARIABLES, device=device, dtype=dtype)
        selector_tau[:, :, TORQUE] = eye12
        selector_qdd = torch.zeros_like(selector_tau)
        selector_qdd[:, :, 6:18] = eye12
        selector_slack = torch.zeros_like(selector_tau)
        selector_slack[:, :, SLACK] = eye12

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

        # Semi-implicit one-step joint limits:
        # v+ = v + dt*qdd; q+ = q + dt*v + 0.5*dt^2*qdd.
        # Isolate qdd in q_lower<=q+dt*v+1/2*dt^2*qdd<=q_upper.
        position_coefficient = 0.5 * dt.square()
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
        friction_rows = torch.zeros(batch, 20, NUM_VARIABLES,
                                    device=device, dtype=dtype)
        for foot in range(4):
            # First force column for this foot in packed [fx,fy,fz] order.
            col = FORCE.start + 3 * foot
            # First of the five inequality rows assigned to this foot.
            row = 5 * foot
            # -fz<=0  <=>  fz>=0 (unilateral ground reaction).
            friction_rows[:, row, col + 2] = -1.0
            # +fx-mu*fz<=0  <=>  fx<=mu*fz.
            friction_rows[:, row + 1, col] = 1.0
            friction_rows[:, row + 1, col + 2] = -self.cfg.friction_coefficient
            # -fx-mu*fz<=0  <=> -fx<=mu*fz.
            friction_rows[:, row + 2, col] = -1.0
            friction_rows[:, row + 2, col + 2] = -self.cfg.friction_coefficient
            # +fy-mu*fz<=0  <=>  fy<=mu*fz.
            friction_rows[:, row + 3, col + 1] = 1.0
            friction_rows[:, row + 3, col + 2] = -self.cfg.friction_coefficient
            # -fy-mu*fz<=0  <=> -fy<=mu*fz.
            friction_rows[:, row + 4, col + 1] = -1.0
            friction_rows[:, row + 4, col + 2] = -self.cfg.friction_coefficient
        # All five pyramid bounds have a zero right-hand side.
        add(friction_rows, torch.zeros(batch, 20, device=device, dtype=dtype))
        if relaxed_contact:
            # Contact rows are absent in stage two, so their slack variables
            # would otherwise sit at an inequality-bound optimum and degrade
            # the interior-point system. Fix them to zero with independent
            # equalities; decision-vector shape remains exactly 54.
            # [0,0,0,I]x=0 fixes s=0 without adding another inequality pair.
            A = torch.cat((A, selector_slack), dim=1)
            b = torch.cat((b, torch.zeros(
                batch, 12, device=device, dtype=dtype
            )), dim=1)
        else:
            # -s<=0 <=> s>=0 for the full problem.
            add(-selector_slack, torch.zeros(
                batch, 12, device=device, dtype=dtype
            ))

        if not relaxed_contact:
            # c is scalar per foot and is repeated over XYZ. Scaling the whole
            # acceleration makes the constraint vanish continuously in swing.
            # Repeat each scalar c_i over its foot's XYZ acceleration axes.
            contact_xyz = contact.unsqueeze(-1).expand(-1, -1, 3).reshape(batch, 12)
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
        variable_scale = tau_nom.new_tensor(
            [self.cfg.qdd_scale] * 18
            + [self.cfg.force_scale_n] * 12
            + [self.cfg.torque_scale_nm] * 12
            + [self.cfg.slack_scale_m_s2] * 12
        )
        # 1/2*(Dz)^T Q (Dz) = 1/2*z^T(DQD)z.
        Q = Q * variable_scale.view(1, -1, 1) * variable_scale.view(1, 1, -1)
        # p^T(Dz)=(Dp)^Tz because D is diagonal.
        p = p * variable_scale
        # A(Dz)=b and G(Dz)<=h scale matrix columns, not right-hand sides.
        A = A * variable_scale
        G = G * variable_scale
        # Positive row divisors leave both feasible sets mathematically equal.
        A, b = _row_scale(A, b)
        G, h = _row_scale(G, h)
        # Remove roundoff asymmetry before Cholesky and add the final strictly
        # positive diagonal in solver space, following qpth's SPD guidance.
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        # qpth requires Q strictly positive definite, not merely semidefinite.
        # This solver-space ridge ensures lambda_min(Q)>0 after roundoff.
        Q.diagonal(dim1=-2, dim2=-1).add_(self.cfg.q_regularization)
        # variable_scale is returned so z*D can be mapped back to physical x.
        return Q, p, G, h, A, b, variable_scale

    def _validate_inputs(self, matrices):
        r"""Check the prerequisites for qpth's primal-dual factorization.

        For every environment this verifies finite coefficients,
        ``rank(A)=number_of_equalities``, and ``Q\succ0``. Invalid rows are
        never trusted even if a placeholder solve later returns finite output.
        """
        Q, p, G, h, A, b, _ = matrices
        # Reduce each coefficient tensor to one finite/nonfinite bit per env.
        finite = torch.stack([
            tensor.reshape(tensor.shape[0], -1).isfinite().all(dim=-1)
            for tensor in (Q, p, G, h, A, b)
        ]).all(dim=0)
        if self.cfg.check_equality_rank:
            # A has full row rank iff its smallest singular value is nonzero.
            singular = torch.linalg.svdvals(A.detach())
            rank_ok = singular[:, -1] > 1.0e-9
        else:
            rank_ok = torch.ones_like(finite)
        if self.cfg.check_q_spd:
            # Successful Cholesky factorization is the direct Q>0 test.
            _, info = torch.linalg.cholesky_ex(Q.detach())
            spd = info == 0
        else:
            spd = torch.ones_like(finite)
        return finite & rank_ok & spd, finite, rank_ok, spd

    def _diagnostics(self, x_scaled, matrices):
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
        Q, p, G, h, A, b, _ = matrices
        # Equality primal residual r_eq=A*z-b.
        equality = torch.einsum("bij,bj->bi", A, x_scaled) - b
        # Inequality signed residual r_ineq=G*z-h; positive means violation.
        inequality = torch.einsum("bij,bj->bi", G, x_scaled) - h
        # Objective gradient grad_z L_objective=Q*z+p.
        gradient = torch.einsum("bij,bj->bi", Q, x_scaled) + p
        # Only near-binding rows participate in the diagnostic dual solve.
        active = inequality > -self.cfg.active_tolerance
        stationarity = []
        stationarity_raw = []
        complementarity = []
        for index in range(x_scaled.shape[0]):
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
            grad = gradient[index].detach()
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
            stationarity_raw.append(raw_stationarity)
            # Q and p are dimensionless scaled-space coefficients whose
            # magnitude changes with configured tracking weights. Report and
            # gate on a relative KKT stationarity residual while retaining the
            # raw value for diagnosis.
            stationarity.append(
                raw_stationarity / (1.0 + grad.abs().max())
            )
            if active_g.shape[0]:
                # Complementarity residual max|lambda_i*(G_i z-h_i)|.
                complementarity.append((
                    inequality_multiplier * inequality[index, active[index]].detach()
                ).abs().max())
            else:
                complementarity.append(residual.new_zeros(()))
        return {
            # ||Az-b||_infinity.
            "equality_max": equality.detach().abs().amax(dim=-1),
            # ||max(Gz-h,0)||_infinity.
            "inequality_max": inequality.detach().clamp_min(0.0).amax(dim=-1),
            "stationarity_max": torch.stack(stationarity),
            "stationarity_raw_max": torch.stack(stationarity_raw),
            "complementarity_max": torch.stack(complementarity),
        }

    def _solve_stage(self, data, relaxed_contact):
        r"""Build, validate, solve, and certify one QP fallback stage.

        qpth returns ``z*`` and its backward differentiates the KKT system;
        this method returns physical ``x*=D z*`` only after residual checks.
        """
        # Construct either the full contact-softened problem or stage-two
        # problem with contact-acceleration rows removed and slack fixed zero.
        matrices = self._build(data, relaxed_contact=relaxed_contact)
        # Validate coefficients before entering qpth's batched factorization.
        valid, finite, rank_ok, spd = self._validate_inputs(matrices)
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
        try:
            # Solve min_z 1/2*z^TQz+p^Tz s.t. Gz<=h, Az=b. QPFunction's
            # autograd backward implicitly solves the differentiated KKT
            # equations; no hand-written derivative or dense inverse is used.
            solution_scaled = QPFunction(
                eps=self.cfg.eps,
                # qpth gates its unconditional inaccurate-solution banner on
                # ``verbose >= 0``.  Our own finite/SPD/rank, primal, and KKT
                # checks below already classify that iterate and drive the
                # relaxed/analytic fallback, so printing the same multi-line
                # banner at every physics substep adds no diagnostic value.
                # Preserve positive verbosity for explicit solver debugging.
                verbose=(-1 if self.cfg.verbose == 0 else self.cfg.verbose),
                notImprovedLim=self.cfg.not_improved_limit,
                maxIter=self.cfg.max_iter,
                check_Q_spd=self.cfg.check_q_spd,
            )(Q, p, G, h, A, b)
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
        diagnostics = self._diagnostics(solution_scaled, matrices)
        # Dual KKT validity requires both relative stationarity and
        # complementarity below their configured tolerance.
        diagnostics["kkt_valid"] = (
            diagnostics["stationarity_max"] <= self.cfg.kkt_tolerance
        ) & (
            diagnostics["complementarity_max"] <= self.cfg.kkt_tolerance
        )
        # A stage is accepted only if all input, solver, finite-output, primal,
        # and dual checks pass for that environment.
        success = (
            valid & ~solver_exception
            & torch.isfinite(solution_scaled).all(dim=-1)
            & (diagnostics["equality_max"] <= self.cfg.feasibility_tolerance)
            & (diagnostics["inequality_max"] <= self.cfg.feasibility_tolerance)
            & diagnostics["kkt_valid"]
        )
        diagnostics.update({"input_finite": finite, "equality_rank": rank_ok,
                            "q_spd": spd,
                            "solver_exception": solver_exception})
        # Undo x=Dz. This multiplication retains qpth's implicit gradient.
        return solution_scaled * variable_scale, success, diagnostics

    def solve(self, **data):
        r"""Solve full QP, relaxed-contact QP, then actuator/rate projection.

        Per environment the deterministic cascade is

        ``stage 0``: all dynamics/hard-limit/friction/contact-softening rows;
        ``stage 1``: remove contact-acceleration rows and impose ``s=0``;
        ``stage 2``: return
        ``clamp(tau_nom, max(-tau_lim,tau_prev-rate*dt),
        min(tau_lim,tau_prev+rate*dt))`` with other variables zero.

        Only failed rows enter stage 1. This avoids retaining a second qpth
        KKT graph for the common successful case and materially reduces VRAM.
        """
        # Public outputs preserve the caller's dtype even if qpth uses another.
        reference = data["tau_nom"]
        original_dtype = reference.dtype
        batch = reference.shape[0]
        # Each tuple below owns one memory-bounded chunk's chosen solution and
        # stage-specific diagnostics.
        outputs = []
        # Chunking changes memory scheduling only; every environment QP is
        # independent, so it cannot change the mathematical solution.
        for start in range(0, batch, self.cfg.chunk_size):
            stop = min(start + self.cfg.chunk_size, batch)
            # Convert every compact QP input to the configured solver dtype on
            # the same device. There is no GPU->CPU->GPU handoff.
            chunk = {
                name: value[start:stop].to(dtype=self.solver_dtype)
                for name, value in data.items()
            }
            # Stage 0: full contact-softened QP for every row in this chunk.
            full, full_ok, full_diag = self._solve_stage(chunk, False)

            # Most minibatches stop here.  Restricting the second qpth graph
            # to failed rows avoids retaining a second 54-variable KKT system
            # for successful environments and is the main VRAM optimization.
            failed = ~full_ok
            # Allocate aligned placeholders so stage selection remains one
            # vectorized torch.where operation in original chunk order.
            relaxed = torch.zeros_like(full)
            relaxed_ok = torch.zeros_like(full_ok)
            relaxed_diag = {
                key: torch.zeros_like(value) for key, value in full_diag.items()
            }
            if failed.any():
                # Compact only failed environments before building stage 1.
                failed_chunk = {
                    name: value[failed] for name, value in chunk.items()
                }
                failed_solution, failed_ok, failed_diag = self._solve_stage(
                    failed_chunk, True
                )
                # Scatter stage-1 outputs/diagnostics back to chunk indices.
                relaxed[failed] = failed_solution
                relaxed_ok[failed] = failed_ok
                for key, value in failed_diag.items():
                    relaxed_diag[key][failed] = value
            # Use stage 1 precisely where stage 0 failed and stage 1 certified.
            use_relaxed = failed & relaxed_ok
            # Stages 0/1 have qpth KKT graphs; stage 2 does not.
            differentiated = full_ok | relaxed_ok
            # Tentatively choose full or relaxed physical x.
            chosen = torch.where(use_relaxed[:, None], relaxed, full)

            # Reconstruct the same hard actuator/rate interval used in G so a
            # deterministic safe torque remains available if both QPs fail.
            tau_nom = chunk["tau_nom"]
            previous = chunk["previous_torque"].detach()
            dt = chunk["dt"].detach().reshape(-1, 1)
            torque_limit = self._limits(tau_nom)[0]
            rate = self.cfg.torque_rate_limit_nm_s * dt
            lower = torch.maximum(-torque_limit, previous - rate)
            upper = torch.minimum(torque_limit, previous + rate)
            # Last-resort actuator/rate projection is deterministic and
            # piecewise differentiable: unsaturated nominal torques retain an
            # identity gradient and saturated entries receive the clamp's zero
            # gradient. It intentionally supplies no invented qdd/GRF/slack.
            # L_proj excludes this stage because no qpth KKT graph succeeded;
            # rollout may still use the honest clamp gradient.
            # Componentwise projection Pi_[lower,upper](tau_nom).
            projected = torch.maximum(torch.minimum(tau_nom, upper), lower)
            # No dynamically meaningful qdd/f/slack is invented in stage 2.
            fallback = torch.zeros_like(chosen)
            fallback[:, TORQUE] = projected
            # Keep qpth results for successful rows and analytic values only
            # for rows where both QPs failed certification.
            chosen = torch.where(differentiated[:, None], chosen, fallback)
            # Encode cascade status: 0=full, 1=relaxed, 2=analytic projection.
            stage = torch.where(full_ok, torch.zeros_like(full_ok, dtype=torch.long),
                                torch.where(relaxed_ok,
                                            torch.ones_like(full_ok, dtype=torch.long),
                                            torch.full_like(full_ok, 2, dtype=torch.long)))
            outputs.append((chosen, stage, differentiated, full_diag, relaxed_diag))

        # Restore original environment order and caller dtype.
        solution = torch.cat([item[0] for item in outputs]).to(original_dtype)
        stage = torch.cat([item[1] for item in outputs])
        differentiated = torch.cat([item[2] for item in outputs])
        # Merge per-chunk diagnostics under explicit full/relaxed namespaces.
        diagnostics = {}
        for prefix, offset in (("full", 3), ("relaxed", 4)):
            keys = outputs[0][offset].keys()
            for key in keys:
                diagnostics[f"{prefix}/{key}"] = torch.cat([
                    item[offset][key] for item in outputs
                ])
        diagnostics["stage"] = stage
        diagnostics["differentiated"] = differentiated
        # Expose the diagnostics belonging to the actually selected stage.
        # Analytic fallback has no KKT system, hence NaN numeric KKT fields.
        for key in (
            "equality_max", "inequality_max", "stationarity_max",
            "stationarity_raw_max", "complementarity_max", "kkt_valid",
        ):
            full_value = diagnostics[f"full/{key}"]
            relaxed_value = diagnostics[f"relaxed/{key}"]
            if full_value.dtype == torch.bool:
                fallback_value = torch.zeros_like(full_value)
            else:
                fallback_value = torch.full_like(full_value, float("nan"))
            diagnostics[f"selected/{key}"] = torch.where(
                stage == 0, full_value,
                torch.where(stage == 1, relaxed_value, fallback_value),
            )
        # Finally unpack physical x into named tensors and restore [foot,XYZ].
        return HardPACTQPResult(
            solution[:, QDD], solution[:, FORCE].reshape(-1, 4, 3),
            solution[:, TORQUE], solution[:, SLACK].reshape(-1, 4, 3),
            stage, differentiated, diagnostics,
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
