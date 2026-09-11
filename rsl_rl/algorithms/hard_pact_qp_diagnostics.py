"""Iteration QP summaries: disjoint phases, real-row counts, no host reads."""

import torch


class QPMeasuredMotion:
    """Detached physics-rate samples; reset/gap invalidates derivative stencils.

    Means use finite environment-coordinate samples. Acceleration is delta-v/dt,
    jerk delta-acceleration/dt; torque slew is command delta/dt (not sensed torque).
    These are measured state summaries, never one-step QP certificates.
    """
    def __init__(self):
        self.velocity = self.acceleration = self.torque = self.valid = None
        self.acceleration_valid = None

    def reset(self, rows):
        if self.valid is not None:
            self.valid[rows] = False
            self.acceleration_valid[rows] = False

    @torch.no_grad()
    def update(self, aggregate, q, v, torque, dt, lower, upper, velocity_limit, solved):
        if self.valid is None or self.velocity.shape != v.shape:
            self.velocity, self.acceleration, self.torque = v.clone(), torch.zeros_like(v), torque.clone()
            self.valid = torch.zeros(v.shape[0],device=v.device,dtype=torch.bool)
            self.acceleration_valid = self.valid.clone()
        acceleration = (v-self.velocity)/dt
        jerk = (acceleration-self.acceleration)/dt
        slew = (torque-self.torque)/dt
        exceed_q = torch.maximum(lower-q,q-upper).clamp_min(0)
        exceed_v = (v.abs()-velocity_limit).clamp_min(0)
        for phase, phase_mask in (("all",torch.ones_like(solved)),("certified_command",solved),
                                  ("unsolved_or_fallback_command",~solved)):
            for name, value, valid in (
                ("velocity_abs_rad_s",v.abs(),phase_mask),
                ("acceleration_abs_rad_s2",acceleration.abs(),phase_mask & self.valid),
                ("jerk_abs_rad_s3",jerk.abs(),phase_mask & self.valid & self.acceleration_valid),
                ("applied_command_slew_abs_nm_s",slew.abs(),phase_mask & self.valid),
                ("position_exceedance_rad",exceed_q,phase_mask),
                ("position_exceedance_fraction",(exceed_q>0).float(),phase_mask),
                ("velocity_exceedance_rad_s",exceed_v,phase_mask),
                ("velocity_exceedance_fraction",(exceed_v>0).float(),phase_mask)):
                aggregate.add_values(f"measured/{phase}/{name}",value,valid[:,None])
        self.acceleration_valid.copy_(self.valid)
        self.valid.fill_(True)
        self.velocity.copy_(v);self.acceleration.copy_(acceleration);self.torque.copy_(torque)


class QPIterationDiagnostics:
    def __init__(self):
        self.sums, self.extrema, self.weights = {}, {}, {}

    def add_sum(self, key, value):
        value = value.detach()
        self.sums[key] = self.sums.get(key, 0) + value

    def add_values(self, key, values, mask=None):
        values = values.detach().float()
        finite = torch.isfinite(values)
        if mask is not None:
            finite = finite & mask
        self.add_sum(key, values.where(finite, 0).sum())
        self.weights[key] = self.weights.get(key, 0) + finite.sum()

    def add_result(self, result, differentiable):
        stage, diag = result.stage, result.diagnostics
        self.add_sum("real_rows", stage.new_tensor(stage.numel()))
        self.add_sum("solve_calls", stage.new_tensor(1))
        final_codes = {"full": 0, "soft_joint": 1, "analytic": 2}
        for name, code in final_codes.items():
            self.add_sum(f"final/{name}_count", (stage == code).sum())
        self.add_sum("certified_count", result.differentiated_mask.sum())
        recovery = getattr(result,"recovery_mask",None)
        differentiated = result.differentiated_mask if recovery is None else result.differentiated_mask | recovery
        self.add_sum("differentiated_count", differentiated.sum() * int(differentiable))
        for name in ("full", "soft_joint"):
            attempted = diag.get(f"{name}/attempted", torch.zeros_like(stage, dtype=torch.bool))
            self.add_sum(f"attempt/{name}_count", attempted.sum())
            failed = diag.get(f"{name}/solver_exception", torch.zeros_like(attempted))
            self.add_sum(f"attempt/{name}_exception_count", (failed & attempted).sum())
            for flag in ("input_finite", "output_finite"):
                if f"{name}/{flag}" in diag:
                    self.add_sum(f"attempt/{name}/{flag}_count", (diag[f"{name}/{flag}"] & attempted).sum())
            for gap in ("duality_gap", "duality_gap_rel"):
                values = diag.get(f"{name}/{gap}", torch.full_like(stage, float("nan"), dtype=torch.float32))
                self.add_values(f"attempt/{name}/{gap}_mean", values, attempted)
        # Reduce per-row physical values directly, not means of chunk means.
        for key, value in diag.items():
            if key == "soft_joint/slack_max_rad_s2":
                self.add_values(key + "_mean", value, stage == 1)
            if key.startswith("physical/"):
                for status, mask in (("certified", result.differentiated_mask),
                                     ("rejected", ~result.differentiated_mask)):
                    self.add_values(f"{key}/{status}_mean", value, mask)
        for name in ("nonfinite_input", "empty_torque_intersection", "empty_qdd_intersection", "mechanics"):
            if f"failure/{name}" not in diag:
                continue
            self.add_sum(f"failure/{name}_count", diag[f"failure/{name}"].sum())
        for key, value in (result.metrics or {}).items():
            # Counts/fractions above are authoritative. Physical/full metrics
            # remain level-gated. Quantiles cannot be merged from quantiles;
            # mark them unavailable rather than average p95s misleadingly.
            if not key.startswith("qp/full/"):
                continue
            name = key.removeprefix("qp/")
            if name.endswith("/p95"):
                self.extrema[name] = value.new_tensor(float("nan"))
            elif name.endswith(("_max", "/max")):
                self.extrema[name] = torch.maximum(self.extrema.get(name, value), value.detach())
            elif name.endswith(("_min", "/min")):
                self.extrema[name] = torch.minimum(self.extrema.get(name, value), value.detach())
            else:
                self.add_values(name, value.reshape(1).expand(stage.numel()))
        for key in ("selected/equality_max", "selected/inequality_max", "pre_clamp_torque_violation_max"):
            values = diag[key].detach()
            finite = torch.isfinite(values)
            maximum = values.where(finite, -torch.inf).max()
            self.extrema[key] = torch.maximum(self.extrema.get(key, maximum), maximum)

    def finalize(self, reference):
        zero = reference.new_zeros((), dtype=torch.float32)
        result = dict(self.sums)
        rows = self.sums.get("real_rows", zero)
        result["real_rows"] = rows
        result["solve_calls"] = self.sums.get("solve_calls", zero)
        for name in ("full", "soft_joint"):
            for suffix in ("count", "exception_count"):
                key = f"attempt/{name}_{suffix}"
                result[key] = self.sums.get(key, zero)
            for flag in ("input_finite", "output_finite"):
                key = f"attempt/{name}/{flag}_count"
                result.setdefault(key, zero + float("nan"))
        for name in ("nonfinite_input", "empty_torque_intersection", "empty_qdd_intersection"):
            key = f"failure/{name}_count"
            result[key] = self.sums.get(key, zero)
        for key in ("selected/equality_max", "selected/inequality_max", "pre_clamp_torque_violation_max", "projection_loss", "recovery_projection_loss"):
            result.setdefault(key, zero + float("nan"))
        for name in ("full", "soft_joint", "analytic"):
            count = self.sums.get(f"final/{name}_count", zero)
            result[f"final/{name}_count"] = count
            result[f"final/{name}_fraction"] = count / rows.clamp_min(1)
        for name in ("certified", "differentiated"):
            result[f"{name}_fraction"] = self.sums.get(f"{name}_count", zero) / rows.clamp_min(1)
        # Recovery frequency is per real primary-QP row (not dispatch, padded
        # capacity or unsolved physics substep). Form ratios after accumulating
        # counts across chunks. Conditional rates are zero when never invoked.
        recovery_attempts = result["attempt/soft_joint_count"]
        recovery_successes = result["final/soft_joint_count"]
        result["attempt/soft_joint_fraction"] = recovery_attempts / rows.clamp_min(1)
        result["attempt/soft_joint_success_fraction"] = recovery_successes / recovery_attempts.clamp_min(1)
        result["attempt/soft_joint_failure_fraction"] = (recovery_attempts-recovery_successes) / recovery_attempts.clamp_min(1)
        result["attempt/soft_joint_exception_fraction"] = result["attempt/soft_joint_exception_count"] / recovery_attempts.clamp_min(1)
        attempted = self.sums.get("attempt/full_count",zero)
        finite_outputs = self.sums.get("attempt/full/output_finite_count",zero)
        exceptions = self.sums.get("attempt/full_exception_count",zero)
        result["nonfinite_output_fraction_of_returned_solves"] = ((attempted-exceptions-finite_outputs).clamp_min(0)
            /(attempted-exceptions).clamp_min(1))
        for key, weight in self.weights.items():
            result[key] = torch.where(weight > 0, self.sums[key] / weight.clamp_min(1), zero + float("nan"))
        for key, value in self.extrema.items():
            result[key] = torch.where(torch.isfinite(value), value, zero + float("nan"))
        for name in ("nonfinite_input", "empty_torque_intersection", "empty_qdd_intersection", "mechanics"):
            result[f"failure/{name}_fraction"] = self.sums.get(f"failure/{name}_count", zero) / rows.clamp_min(1)
        intervals = self.sums.get("environment_control_intervals", zero)
        result["problems_per_environment_control_interval"] = rows / intervals.clamp_min(1)
        result["solved_substep_coverage"] = rows / (4*intervals).clamp_min(1)
        for k in range(4):
            result[f"sampled_substep/{k}_fraction"] = self.sums.get(f"sampled_substep/{k}_count", zero) / intervals.clamp_min(1)
        # No inference from unavailable temporal alignment or allocator APIs.
        for key in ("measured/optimized_grf_error_n", "measured/raw_grf_error_n",
                    "backend/solver_iterations_mean"):
            result.setdefault(key,zero+float("nan"))
        return result
