"""Iteration QP summaries: disjoint phases, real-row counts, no host reads."""

import torch


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

    def add_result(self, result, elastic, differentiable):
        stage, diag = result.stage, result.diagnostics
        self.add_sum("real_rows", stage.new_tensor(stage.numel()))
        self.add_sum("solve_calls", stage.new_tensor(1))
        final_codes = {"full": 0, "relaxed": 1, "elastic": 2, "analytic": 3 if elastic else 2}
        for name, code in final_codes.items():
            self.add_sum(f"final/{name}_count", (stage == code).sum() if name != "elastic" or elastic else stage.new_zeros(()))
        self.add_sum("certified_count", result.differentiated_mask.sum())
        self.add_sum("differentiated_count", result.differentiated_mask.sum() * int(differentiable))
        for name in ("full", "relaxed", "elastic"):
            attempted = diag.get(f"{name}/attempted", torch.zeros_like(stage, dtype=torch.bool))
            if name == "full" and "full/active/full_solve" in diag:
                attempted = attempted & diag["full/active/full_solve"]
            self.add_sum(f"attempt/{name}_count", attempted.sum())
            failed = diag.get(f"{name}/solver_exception", torch.zeros_like(attempted))
            self.add_sum(f"attempt/{name}_exception_count", (failed & attempted).sum())
            for flag in ("input_finite", "output_finite"):
                if f"{name}/{flag}" in diag:
                    self.add_sum(f"attempt/{name}/{flag}_count", (diag[f"{name}/{flag}"] & attempted).sum())
            for gap in ("duality_gap", "duality_gap_rel"):
                values = diag.get(f"{name}/{gap}", torch.full_like(stage, float("nan"), dtype=torch.float32))
                self.add_values(f"attempt/{name}/{gap}_mean", values, attempted)
        for key, value in diag.items():
            if key.startswith("full/active/"):
                name = key.removeprefix("full/")
                self.add_values(name + ("_fraction" if value.dtype == torch.bool else "_mean"), value)
                if name in ("active/attempted", "active/accepted", "active/full_solve"):
                    self.add_sum(name + "_count", value.sum())
        for name in ("nonfinite_input", "empty_torque_intersection", "empty_qdd_intersection", "mechanics"):
            if f"failure/{name}" not in diag:
                continue
            self.add_sum(f"failure/{name}_count", diag[f"failure/{name}"].sum())
        for key, value in (result.metrics or {}).items():
            # Counts/fractions above are authoritative. Physical/full metrics
            # remain level-gated. Quantiles cannot be merged from quantiles;
            # mark them unavailable rather than average p95s misleadingly.
            if not key.startswith(("qp/physical/", "qp/full/")):
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
        for name in ("full", "relaxed", "elastic"):
            for suffix in ("count", "exception_count"):
                key = f"attempt/{name}_{suffix}"
                result[key] = self.sums.get(key, zero)
            for flag in ("input_finite", "output_finite"):
                key = f"attempt/{name}/{flag}_count"
                result.setdefault(key, zero + float("nan"))
        for name in ("nonfinite_input", "empty_torque_intersection", "empty_qdd_intersection"):
            key = f"failure/{name}_count"
            result[key] = self.sums.get(key, zero)
        for key in ("selected/equality_max", "selected/inequality_max", "pre_clamp_torque_violation_max", "projection_loss"):
            result.setdefault(key, zero + float("nan"))
        for name in ("full", "relaxed", "elastic", "analytic"):
            count = self.sums.get(f"final/{name}_count", zero)
            result[f"final/{name}_count"] = count
            result[f"final/{name}_fraction"] = count / rows.clamp_min(1)
        for name in ("certified", "differentiated"):
            result[f"{name}_fraction"] = self.sums.get(f"{name}_count", zero) / rows.clamp_min(1)
        for key, weight in self.weights.items():
            result[key] = torch.where(weight > 0, self.sums[key] / weight.clamp_min(1), zero + float("nan"))
        for key, value in self.extrema.items():
            result[key] = torch.where(torch.isfinite(value), value, zero + float("nan"))
        if "active/attempted_count" in self.sums:
            attempted = self.sums["active/attempted_count"]
            accepted = self.sums["active/accepted_count"]
            result["active/acceptance_given_attempt"] = accepted / attempted.clamp_min(1)
            result["active/fallback_given_attempt"] = (attempted - accepted) / attempted.clamp_min(1)
        return result
