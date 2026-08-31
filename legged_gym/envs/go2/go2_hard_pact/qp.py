"""Solver-neutral fixed-shape OptNet QP for Go2 HardPACT."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import torch


@dataclass(frozen=True)
class HardPACTQPConfig:
    enabled: bool = True
    friction: float = 0.45
    acceleration_weight: float = 1.0e-3
    grf_tracking_weight: float = 1.0
    torque_tracking_weight: float = 4.0
    contact_slack_weight: float = 2.0e3
    hessian_regularization: float = 1.0e-7
    active_tolerance: float = 1.0e-5
    solver: str = "auto"
    solver_dtype: torch.dtype = torch.float64
    gpu_chunk_size: int = 1024


@dataclass
class HardPACTQPInputs:
    mass: torch.Tensor
    bias: torch.Tensor
    foot_jacobian: torch.Tensor
    foot_jdot_v: torch.Tensor
    base_jacobian: torch.Tensor
    predicted_grf: torch.Tensor
    predicted_base_wrench: torch.Tensor
    contact_probability: torch.Tensor
    nominal_torque: torch.Tensor
    previous_torque: torch.Tensor
    torque_limit: torch.Tensor
    torque_rate_limit: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    joint_position_lower: torch.Tensor
    joint_position_upper: torch.Tensor
    joint_velocity_limit: torch.Tensor
    gravity_normal_force_frame: torch.Tensor
    dt: float


@dataclass
class HardPACTQPResult:
    safe_torque: torch.Tensor
    acceleration: torch.Tensor
    grf: torch.Tensor
    contact_slack: torch.Tensor
    correction: torch.Tensor
    equality_residual: torch.Tensor
    inequality_violation: torch.Tensor
    minimum_margin: torch.Tensor
    active_constraints: torch.Tensor
    fallback: torch.Tensor
    status: torch.Tensor
    forward_time_ms: torch.Tensor


class Go2HardPACTQP:
    """Build and solve the 54-variable convex QP.

    Variables are ``[ddq_18, f_12, tau_safe_12, contact_slack_12]``. The
    predicted GRF is an objective reference; the GRF remains a decision
    variable. All network outputs enter linearly, so the per-step QP is affine
    and convex.
    """

    nv = 18
    nf = 12
    ntau = 12
    nslack = 12
    nvar = nv + nf + ntau + nslack
    ddq_slice = slice(0, 18)
    force_slice = slice(18, 30)
    torque_slice = slice(30, 42)
    slack_slice = slice(42, 54)

    def __init__(self, config: HardPACTQPConfig):
        if config.friction <= 0.0:
            raise ValueError("QP friction must be positive")
        if config.hessian_regularization <= 0.0:
            raise ValueError("QP Hessian regularization must be strictly positive")
        if config.gpu_chunk_size <= 0:
            raise ValueError("QP chunk size must be positive")
        self.config = config

    @staticmethod
    def _expand_batch(value, batch, width, like):
        value = torch.as_tensor(value, device=like.device, dtype=like.dtype)
        if value.ndim == 0:
            value = value.expand(batch, width)
        elif value.ndim == 1:
            value = value.unsqueeze(0).expand(batch, -1)
        return value

    @staticmethod
    def _tangent_basis(normal: torch.Tensor):
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        x_axis = torch.zeros_like(normal)
        x_axis[..., 0] = 1.0
        y_axis = torch.zeros_like(normal)
        y_axis[..., 1] = 1.0
        use_y = normal[..., 0].abs() > 0.90
        seed = torch.where(use_y.unsqueeze(-1), y_axis, x_axis)
        tangent_1 = torch.cross(normal, seed, dim=-1)
        tangent_1 /= tangent_1.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        tangent_2 = torch.cross(normal, tangent_1, dim=-1)
        return tangent_1, tangent_2

    def build(self, inputs: HardPACTQPInputs, *, relaxed_contact=False):
        mass = inputs.mass
        batch = mass.shape[0]
        device, dtype = mass.device, mass.dtype
        if mass.shape != (batch, 18, 18):
            raise ValueError("QP mass matrix must be (batch,18,18)")
        if inputs.foot_jacobian.shape != (batch, 4, 3, 18):
            raise ValueError("QP foot Jacobian must be (batch,4,3,18)")

        diagonal = mass.new_full((batch, self.nvar), self.config.hessian_regularization)
        diagonal[:, self.ddq_slice] += self.config.acceleration_weight
        diagonal[:, self.force_slice] += self.config.grf_tracking_weight
        diagonal[:, self.torque_slice] += self.config.torque_tracking_weight
        slack_weight = self.config.contact_slack_weight * (0.01 if relaxed_contact else 1.0)
        diagonal[:, self.slack_slice] += slack_weight
        hessian = torch.diag_embed(diagonal)
        linear = mass.new_zeros(batch, self.nvar)
        linear[:, self.force_slice] = -self.config.grf_tracking_weight * inputs.predicted_grf
        linear[:, self.torque_slice] = -self.config.torque_tracking_weight * inputs.nominal_torque

        # M ddq - J_f^T f - S^T tau = J_b^T W - h.
        dynamics = mass.new_zeros(batch, 18, self.nvar)
        dynamics[:, :, self.ddq_slice] = mass
        foot_j = inputs.foot_jacobian.reshape(batch, 12, 18)
        dynamics[:, :, self.force_slice] = -foot_j.transpose(1, 2)
        selector = torch.eye(12, device=device, dtype=dtype).expand(batch, -1, -1)
        dynamics[:, 6:, self.torque_slice] = -selector
        dynamics_rhs = torch.einsum(
            "bki,bk->bi", inputs.base_jacobian, inputs.predicted_base_wrench
        ) - inputs.bias

        # Contact-probability weighted foot acceleration with signed slacks.
        contact_weight = inputs.contact_probability.clamp(0.0, 1.0)
        if relaxed_contact:
            contact_weight = 0.1 * contact_weight
        weighted_j = inputs.foot_jacobian * contact_weight[:, :, None, None]
        contact = mass.new_zeros(batch, 12, self.nvar)
        contact[:, :, self.ddq_slice] = weighted_j.reshape(batch, 12, 18)
        contact[:, :, self.slack_slice] = -torch.eye(12, device=device, dtype=dtype)
        contact_rhs = -(
            inputs.foot_jdot_v * contact_weight.unsqueeze(-1)
        ).reshape(batch, 12)
        equality = torch.cat((dynamics, contact), dim=1)
        equality_rhs = torch.cat((dynamics_rhs, contact_rhs), dim=1)

        rows = []
        bounds = []

        def add_box(variable_slice, lower, upper):
            width = variable_slice.stop - variable_slice.start
            eye = torch.eye(width, device=device, dtype=dtype).expand(batch, -1, -1)
            positive = mass.new_zeros(batch, width, self.nvar)
            positive[:, :, variable_slice] = eye
            rows.extend((positive, -positive))
            bounds.extend((upper, -lower))

        torque_limit = self._expand_batch(inputs.torque_limit, batch, 12, mass)
        rate = self._expand_batch(inputs.torque_rate_limit, batch, 12, mass) * float(inputs.dt)
        lower_tau = torch.maximum(-torque_limit, inputs.previous_torque - rate)
        upper_tau = torch.minimum(torque_limit, inputs.previous_torque + rate)
        add_box(self.torque_slice, lower_tau, upper_tau)

        q = inputs.joint_position
        qd = inputs.joint_velocity
        qdd_lower_v = (-inputs.joint_velocity_limit - qd) / float(inputs.dt)
        qdd_upper_v = (inputs.joint_velocity_limit - qd) / float(inputs.dt)
        half_dt_sq = 0.5 * float(inputs.dt) ** 2
        qdd_lower_q = (inputs.joint_position_lower - q - float(inputs.dt) * qd) / half_dt_sq
        qdd_upper_q = (inputs.joint_position_upper - q - float(inputs.dt) * qd) / half_dt_sq
        add_box(slice(6, 18), qdd_lower_v, qdd_upper_v)
        add_box(slice(6, 18), qdd_lower_q, qdd_upper_q)

        normal = inputs.gravity_normal_force_frame
        if normal.ndim == 1:
            normal = normal.unsqueeze(0).expand(batch, -1)
        tangent_1, tangent_2 = self._tangent_basis(normal)
        friction_rows = mass.new_zeros(batch, 20, self.nvar)
        friction_bounds = mass.new_zeros(batch, 20)
        mu = self.config.friction
        for foot in range(4):
            force_start = self.force_slice.start + 3 * foot
            vectors = torch.stack((
                -normal,
                tangent_1 - mu * normal,
                -tangent_1 - mu * normal,
                tangent_2 - mu * normal,
                -tangent_2 - mu * normal,
            ), dim=1)
            friction_rows[:, 5 * foot:5 * foot + 5, force_start:force_start + 3] = vectors
        rows.append(friction_rows)
        bounds.append(friction_bounds)
        inequality = torch.cat(rows, dim=1)
        inequality_rhs = torch.cat(bounds, dim=1)
        return hessian, linear, inequality, inequality_rhs, equality, equality_rhs

    @staticmethod
    def _equality_solution(hessian, linear, equality, rhs):
        diagonal_inv = torch.diagonal(hessian, dim1=-2, dim2=-1).reciprocal()
        ahinv = equality * diagonal_inv.unsqueeze(1)
        schur = ahinv @ equality.transpose(1, 2)
        schur = schur + 1.0e-9 * torch.eye(
            schur.shape[-1], device=schur.device, dtype=schur.dtype
        )
        multiplier_rhs = -rhs - torch.einsum("bij,bj->bi", ahinv, linear)
        multiplier = torch.linalg.solve(schur, multiplier_rhs.unsqueeze(-1)).squeeze(-1)
        return -diagonal_inv * (
            linear + torch.einsum("bji,bj->bi", equality, multiplier)
        )

    def _solve_once(self, built):
        hessian, linear, inequality, inequality_rhs, equality, equality_rhs = built
        solver = self.config.solver
        if solver not in {"auto", "qpth", "equality"}:
            raise ValueError(f"unknown QP solver {solver!r}")
        use_qpth = solver in {"auto", "qpth"}
        if use_qpth:
            try:
                from qpth.qp import QPFunction
            except (ImportError, ModuleNotFoundError):
                if solver == "qpth":
                    raise
            else:
                return QPFunction(verbose=-1, check_Q_spd=False)(
                    hessian, linear, inequality, inequality_rhs, equality, equality_rhs
                )
        solution = self._equality_solution(hessian, linear, equality, equality_rhs)
        violation = torch.einsum("bij,bj->bi", inequality, solution) - inequality_rhs
        if torch.any(violation > self.config.active_tolerance):
            raise RuntimeError("equality QP candidate violates inequalities")
        return solution

    def _candidate_is_feasible(self, built, solution) -> bool:
        if solution is None or not torch.isfinite(solution).all():
            return False
        _, _, inequality, inequality_rhs, equality, equality_rhs = built
        equality_error = (
            torch.einsum("bij,bj->bi", equality, solution) - equality_rhs
        ).abs().amax()
        inequality_error = (
            torch.einsum("bij,bj->bi", inequality, solution) - inequality_rhs
        ).clamp_min(0.0).amax()
        tolerance = max(10.0 * self.config.active_tolerance, 1.0e-5)
        return bool(
            (equality_error <= tolerance).detach().item()
            and (inequality_error <= tolerance).detach().item()
        )

    @staticmethod
    def actuator_rate_projection(nominal, previous, torque_limit, torque_rate_limit, dt):
        limit = torch.as_tensor(torque_limit, device=nominal.device, dtype=nominal.dtype)
        if limit.ndim == 1:
            limit = limit.unsqueeze(0)
        rate = torch.as_tensor(torque_rate_limit, device=nominal.device, dtype=nominal.dtype)
        if rate.ndim == 1:
            rate = rate.unsqueeze(0)
        lower = torch.maximum(-limit, previous - rate * float(dt))
        upper = torch.minimum(limit, previous + rate * float(dt))
        return torch.maximum(torch.minimum(nominal, upper), lower)

    def solve(self, inputs: HardPACTQPInputs) -> HardPACTQPResult:
        batch = inputs.mass.shape[0]
        chunk_size = self.config.gpu_chunk_size
        if batch > chunk_size:
            chunks = []
            for start_index in range(0, batch, chunk_size):
                stop_index = min(start_index + chunk_size, batch)
                sliced = {}
                for name in HardPACTQPInputs.__dataclass_fields__:
                    value = getattr(inputs, name)
                    if (
                        torch.is_tensor(value)
                        and value.ndim >= 2
                        and value.shape[0] == batch
                    ):
                        value = value[start_index:stop_index]
                    sliced[name] = value
                chunks.append(self.solve(HardPACTQPInputs(**sliced)))
            return HardPACTQPResult(**{
                name: torch.cat([getattr(chunk, name) for chunk in chunks], dim=0)
                for name in HardPACTQPResult.__dataclass_fields__
            })

        start = perf_counter()
        original_dtype = inputs.mass.dtype
        solver_dtype = self.config.solver_dtype

        def cast(value):
            return value.to(solver_dtype) if torch.is_tensor(value) and value.is_floating_point() else value

        work = HardPACTQPInputs(**{
            name: cast(getattr(inputs, name))
            for name in HardPACTQPInputs.__dataclass_fields__
        })
        batch = work.mass.shape[0]
        solution = None
        fallback_code = 0
        status_code = 0
        built = self.build(work, relaxed_contact=False)
        try:
            solution = self._solve_once(built)
            if not self._candidate_is_feasible(built, solution):
                raise RuntimeError("full QP returned an infeasible candidate")
        except (RuntimeError, ValueError):
            fallback_code = 1
            status_code = 1
            built = self.build(work, relaxed_contact=True)
            try:
                solution = self._solve_once(built)
                if not self._candidate_is_feasible(built, solution):
                    raise RuntimeError("relaxed QP returned an infeasible candidate")
            except (RuntimeError, ValueError):
                fallback_code = 2
                status_code = 2
                solution = None

        if solution is None or not torch.isfinite(solution).all():
            safe = self.actuator_rate_projection(
                work.nominal_torque, work.previous_torque, work.torque_limit,
                work.torque_rate_limit, work.dt,
            )
            acceleration = work.mass.new_zeros(batch, 18)
            # No force decision exists in the final projection fallback; do
            # not relabel the learned reference as solved/true GRF.
            grf = work.mass.new_zeros(batch, 12)
            slack = work.mass.new_zeros(batch, 12)
            equality_residual = work.mass.new_full((batch, 1), float("nan"))
            inequality_violation = work.mass.new_full((batch, 1), float("nan"))
            minimum_margin = work.mass.new_full((batch, 1), float("nan"))
            active = work.mass.new_zeros(batch, 1)
        else:
            acceleration = solution[:, self.ddq_slice]
            grf = solution[:, self.force_slice]
            safe = solution[:, self.torque_slice]
            slack = solution[:, self.slack_slice]
            h, p, g, hh, a, b = built
            eq = torch.einsum("bij,bj->bi", a, solution) - b
            margin = hh - torch.einsum("bij,bj->bi", g, solution)
            equality_residual = eq.abs().amax(dim=-1, keepdim=True)
            inequality_violation = (-margin).clamp_min(0.0).amax(dim=-1, keepdim=True)
            minimum_margin = margin.amin(dim=-1, keepdim=True)
            active = (margin.abs() < self.config.active_tolerance).sum(dim=-1, keepdim=True).to(h.dtype)

        elapsed_ms = (perf_counter() - start) * 1000.0
        elapsed = safe.new_full((batch, 1), elapsed_ms)
        fallback = torch.full((batch, 1), fallback_code, device=safe.device, dtype=torch.long)
        status = torch.full((batch, 1), status_code, device=safe.device, dtype=torch.long)
        safe = safe.to(original_dtype)
        return HardPACTQPResult(
            safe_torque=safe,
            acceleration=acceleration.to(original_dtype),
            grf=grf.to(original_dtype),
            contact_slack=slack.to(original_dtype),
            correction=safe - inputs.nominal_torque,
            equality_residual=equality_residual.to(original_dtype),
            inequality_violation=inequality_violation.to(original_dtype),
            minimum_margin=minimum_margin.to(original_dtype),
            active_constraints=active.to(original_dtype),
            fallback=fallback,
            status=status,
            forward_time_ms=elapsed.to(original_dtype),
        )
