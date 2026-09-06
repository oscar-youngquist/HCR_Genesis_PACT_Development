"""Deployment physics heads and shared force-scaling helpers for HardPACT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class PhysicsHeadOutput:
    grf_normalized: torch.Tensor
    wrench_raw_normalized: torch.Tensor


def _broadcast_grf_scale(reference, grf_scale_n):
    scale = torch.as_tensor(
        grf_scale_n, device=reference.device, dtype=reference.dtype
    )
    if reference.shape[-1] == 3 and scale.numel() == 12:
        scale = scale[:3]
    if reference.shape[-1] != scale.numel():
        raise ValueError("GRF tensor must end in XYZ (3) or flattened feet XYZ (12)")
    return scale


def normalize_grf_target(grf_physical_n, grf_scale_n):
    """Normalize yaw-local FR/FL/RR/RL XYZ forces without observation scaling."""
    return grf_physical_n / _broadcast_grf_scale(
        grf_physical_n, grf_scale_n
    )


def grf_normalized_to_physical(grf_normalized, grf_scale_n):
    """Reconstruct yaw-local physical Newtons from the dedicated GRF head."""
    return grf_normalized * _broadcast_grf_scale(
        grf_normalized, grf_scale_n
    )


def normalized_grf_huber_loss(prediction, target, mask=None, delta=1.0):
    """Huber loss directly in the dedicated GRF decoder's normalized space."""
    loss = F.huber_loss(
        prediction, target, delta=delta, reduction="none"
    ).mean(dim=-1)
    if mask is None:
        return loss.mean()
    weights = mask.reshape(-1).to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


class GRFDecoderMetricsAccumulator:
    """Accumulate canonical GRF decoder diagnostics without retaining graphs.

    Inputs are normalized decoder tensors in flattened FR/FL/RR/RL XYZ order.
    All reported force errors are computed after conversion to Newtons. Sums,
    squared sums, and sample counts are retained so unequal minibatches and
    transition/contact masks are reduced exactly rather than averaging means.
    """

    FOOT_NAMES = ("FR", "FL", "RR", "RL")
    AXIS_NAMES = ("Fx", "Fy", "Fz")

    def __init__(self, grf_scale_n):
        self.grf_scale_n = tuple(float(value) for value in grf_scale_n)
        if len(self.grf_scale_n) != 12:
            raise ValueError("grf_scale_n must contain FR/FL/RR/RL XYZ scales")
        self._totals = {}

    def _add(self, name, value):
        value = value.detach()
        self._totals[name] = self._totals.get(name, torch.zeros_like(value)) + value

    @torch.no_grad()
    def update(self, prediction_normalized, target_normalized, contact_labels,
               transition_valid_mask=None):
        prediction = prediction_normalized.detach().reshape(-1, 4, 3)
        target = target_normalized.detach().reshape(-1, 4, 3)
        contact = contact_labels.detach().reshape(-1, 4).bool()
        valid_row = torch.ones(
            prediction.shape[0], device=prediction.device, dtype=torch.bool
        )
        if transition_valid_mask is not None:
            valid_row = transition_valid_mask.detach().reshape(-1).bool()
        valid_foot = valid_row[:, None].expand(-1, 4)
        valid_component = valid_foot[:, :, None].expand(-1, -1, 3)

        prediction_finite = torch.isfinite(prediction)
        target_finite = torch.isfinite(target)
        finite_pair = prediction_finite & target_finite
        regression_mask = valid_component & finite_pair
        safe_prediction = torch.where(
            prediction_finite, prediction, torch.zeros_like(prediction)
        )
        safe_target = torch.where(target_finite, target, torch.zeros_like(target))
        scale = prediction.new_tensor(self.grf_scale_n).reshape(4, 3)
        prediction_n = safe_prediction * scale
        target_n = safe_target * scale
        error_n = prediction_n - target_n
        absolute_n = error_n.abs()
        squared_n = error_n.square()
        mask_f = regression_mask.to(prediction.dtype)

        self._add("valid_component_count", mask_f.sum())
        self._add("absolute_sum", (absolute_n * mask_f).sum())
        self._add("squared_sum", (squared_n * mask_f).sum())
        self._add("axis_absolute_sum", (absolute_n * mask_f).sum(dim=(0, 1)))
        self._add("axis_squared_sum", (squared_n * mask_f).sum(dim=(0, 1)))
        self._add("axis_signed_sum", (error_n * mask_f).sum(dim=(0, 1)))
        self._add("axis_count", mask_f.sum(dim=(0, 1)))
        self._add("foot_absolute_sum", (absolute_n * mask_f).sum(dim=(0, 2)))
        self._add("foot_squared_sum", (squared_n * mask_f).sum(dim=(0, 2)))
        self._add("foot_count", mask_f.sum(dim=(0, 2)))

        normalized_huber = F.huber_loss(
            safe_prediction, safe_target, reduction="none"
        )
        self._add("normalized_huber_sum", (normalized_huber * mask_f).sum())

        valid_component_count = valid_component.to(prediction.dtype).sum()
        self._add("finite_fraction_count", valid_component_count)
        self._add(
            "prediction_nonfinite_count",
            ((~prediction_finite) & valid_component).to(prediction.dtype).sum(),
        )
        self._add(
            "target_nonfinite_count",
            ((~target_finite) & valid_component).to(prediction.dtype).sum(),
        )

        # Norms are per-foot XYZ magnitudes. Only fully finite feet contribute.
        finite_foot = finite_pair.all(dim=-1) & valid_foot
        prediction_norm = prediction_n.norm(dim=-1)
        target_norm = target_n.norm(dim=-1)
        for state_name, state_mask in (
            ("overall", finite_foot),
            ("stance", finite_foot & contact),
            ("swing", finite_foot & ~contact),
        ):
            state_f = state_mask.to(prediction.dtype)
            self._add(f"{state_name}_foot_count", state_f.sum())
            self._add(
                f"{state_name}_predicted_norm_sum",
                (prediction_norm * state_f).sum(),
            )
            self._add(
                f"{state_name}_target_norm_sum", (target_norm * state_f).sum()
            )
            state_components = state_mask[:, :, None] & finite_pair
            state_component_f = state_components.to(prediction.dtype)
            self._add(
                f"{state_name}_component_count", state_component_f.sum()
            )
            self._add(
                f"{state_name}_absolute_sum",
                (absolute_n * state_component_f).sum(),
            )
            self._add(
                f"{state_name}_squared_sum",
                (squared_n * state_component_f).sum(),
            )

    @torch.no_grad()
    def finalize(self):
        if not self._totals:
            return {}
        totals = self._totals

        def mean(sum_name, count_name):
            return totals[sum_name] / totals[count_name].clamp_min(1.0)

        metrics = {
            "grf_decoder_loss_normalized": mean(
                "normalized_huber_sum", "valid_component_count"
            ),
            "grf_mae_physical": mean("absolute_sum", "valid_component_count"),
            "grf_rmse_physical": mean(
                "squared_sum", "valid_component_count"
            ).sqrt(),
            "grf_prediction_nonfinite_fraction": mean(
                "prediction_nonfinite_count", "finite_fraction_count"
            ),
            "grf_target_nonfinite_fraction": mean(
                "target_nonfinite_count", "finite_fraction_count"
            ),
        }
        for index, name in enumerate(self.AXIS_NAMES):
            count = totals["axis_count"][index].clamp_min(1.0)
            metrics[f"grf_mae_{name}_physical"] = (
                totals["axis_absolute_sum"][index] / count
            )
            metrics[f"grf_rmse_{name}_physical"] = (
                totals["axis_squared_sum"][index] / count
            ).sqrt()
            metrics[f"grf_signed_error_{name}_mean_physical"] = (
                totals["axis_signed_sum"][index] / count
            )
        for index, name in enumerate(self.FOOT_NAMES):
            count = totals["foot_count"][index].clamp_min(1.0)
            metrics[f"grf_mae_{name}_physical"] = (
                totals["foot_absolute_sum"][index] / count
            )
            metrics[f"grf_rmse_{name}_physical"] = (
                totals["foot_squared_sum"][index] / count
            ).sqrt()
        for state_name in ("overall", "stance", "swing"):
            component_count = totals[
                f"{state_name}_component_count"
            ].clamp_min(1.0)
            foot_count = totals[f"{state_name}_foot_count"].clamp_min(1.0)
            if state_name != "overall":
                metrics[f"grf_{state_name}_mae_physical"] = (
                    totals[f"{state_name}_absolute_sum"] / component_count
                )
                metrics[f"grf_{state_name}_rmse_physical"] = (
                    totals[f"{state_name}_squared_sum"] / component_count
                ).sqrt()
            metrics[f"grf_predicted_norm_{state_name}_physical"] = (
                totals[f"{state_name}_predicted_norm_sum"] / foot_count
            )
            metrics[f"grf_target_norm_{state_name}_physical"] = (
                totals[f"{state_name}_target_norm_sum"] / foot_count
            )
        return metrics


def normalize_wrench_target(wrench_physical, wrench_scale):
    """Normalize yaw-local [force, moment] labels without observation scaling."""
    scale = torch.as_tensor(
        wrench_scale, device=wrench_physical.device, dtype=wrench_physical.dtype
    )
    return wrench_physical / scale


def wrench_normalized_to_physical(
    wrench_raw_normalized, wrench_scale
):
    """Convert the unbounded decoder output to physical N/Nm exactly once."""
    scale = torch.as_tensor(
        wrench_scale,
        device=wrench_raw_normalized.device,
        dtype=wrench_raw_normalized.dtype,
    )
    return wrench_raw_normalized * scale


def sanitize_and_clip_wrench_for_qp(
    wrench_raw_physical, wrench_qp_clip
):
    """Apply the sole deployment safety boundary in physical yaw-local units.

    This helper is intentionally downstream of physical reconstruction.  Its
    ordinary clamp has zero gradient outside the QP-safe box; decoder targets,
    supervised losses, and BARD predictions must use the unclipped value.
    """
    limit = torch.as_tensor(
        wrench_qp_clip,
        device=wrench_raw_physical.device,
        dtype=wrench_raw_physical.dtype,
    )
    finite = torch.nan_to_num(wrench_raw_physical)
    return finite.clamp(min=-limit, max=limit)


def normalized_wrench_huber_loss(prediction, target, mask=None, delta=1.0):
    """Huber loss directly between unbounded normalized wrench tensors."""
    loss = F.huber_loss(
        prediction, target, delta=delta, reduction="none"
    ).mean(dim=-1)
    if mask is None:
        return loss.mean()
    weights = mask.reshape(-1).to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def wrench_regression_metrics(
    prediction_normalized,
    target_normalized,
    mask,
    wrench_scale,
    wrench_qp_clip,
):
    """Return scalar diagnostics in physical units without affecting training."""
    raw = wrench_normalized_to_physical(prediction_normalized, wrench_scale)
    target = wrench_normalized_to_physical(target_normalized, wrench_scale)
    clipped = sanitize_and_clip_wrench_for_qp(raw, wrench_qp_clip)
    finite = torch.isfinite(raw)
    safe_raw = torch.where(finite, raw, torch.zeros_like(raw))
    valid = torch.ones(raw.shape[0], device=raw.device, dtype=raw.dtype)
    if mask is not None:
        valid = mask.reshape(-1).to(raw.dtype)
    denom = valid.sum().clamp_min(1.0)

    def row_mean(value):
        return (value * valid).sum() / denom

    def component_mean(value):
        return (value * valid[:, None]).sum(dim=0) / denom

    error = safe_raw - target
    absolute = error.abs()
    squared = error.square()
    limit = torch.as_tensor(
        wrench_qp_clip, device=raw.device, dtype=raw.dtype
    )
    exceedance = torch.where(
        finite, (raw.abs() - limit).clamp_min(0.0), torch.zeros_like(raw)
    )
    clipped_component = (~finite) | (safe_raw.abs() > limit)
    names = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
    mae = component_mean(absolute)
    rmse = component_mean(squared).sqrt()
    metrics = {
        "wrench_raw_mae_physical": row_mean(absolute.mean(dim=-1)),
        "wrench_raw_rmse_physical": row_mean(squared.mean(dim=-1)).sqrt(),
        "wrench_clipping_any_row_fraction": row_mean(
            clipped_component.any(dim=-1).to(raw.dtype)
        ),
        "wrench_nonfinite_fraction": (
            finite.logical_not().to(raw.dtype) * valid[:, None]
        ).sum() / (denom * raw.shape[-1]),
        "wrench_bound_exceedance_mean": row_mean(exceedance.mean(dim=-1)),
        "wrench_bound_exceedance_max": torch.where(
            valid[:, None].bool(), exceedance, torch.zeros_like(exceedance)
        ).max(),
        "wrench_raw_norm_mean": row_mean(safe_raw.norm(dim=-1)),
        "wrench_raw_norm_max": torch.where(
            valid.bool(), safe_raw.norm(dim=-1), torch.zeros_like(valid)
        ).max(),
        "wrench_clipped_norm_mean": row_mean(clipped.norm(dim=-1)),
        "wrench_clipped_norm_max": torch.where(
            valid.bool(), clipped.norm(dim=-1), torch.zeros_like(valid)
        ).max(),
        "wrench_raw_clipped_abs_difference_mean": row_mean(
            (safe_raw - clipped).abs().mean(dim=-1)
        ),
        "wrench_target_outside_qp_bound_fraction": row_mean(
            (target.abs() > limit).any(dim=-1).to(raw.dtype)
        ),
    }
    clipping_fraction = component_mean(clipped_component.to(raw.dtype))
    for index, name in enumerate(names):
        metrics[f"wrench_raw_mae_{name}_physical"] = mae[index]
        metrics[f"wrench_raw_rmse_{name}_physical"] = rmse[index]
        metrics[f"wrench_clipping_{name}_fraction"] = clipping_fraction[index]
    return metrics


def compose_explicit_estimator_target(
    scaled_body_linear_velocity, contact_probabilities, clipped_foot_clearances
):
    """Compose the ordered 11-D HardPACT explicit-estimator target."""
    if scaled_body_linear_velocity.shape[-1] != 3:
        raise ValueError("body linear velocity must be 3-D")
    if contact_probabilities.shape[-1] != 4:
        raise ValueError("contact probabilities must be 4-D")
    if clipped_foot_clearances.shape[-1] != 4:
        raise ValueError("foot clearances must be 4-D")
    return torch.cat(
        (
            scaled_body_linear_velocity,
            contact_probabilities.clamp(0.0, 1.0),
            clipped_foot_clearances,
        ),
        dim=-1,
    )


class ExplicitEstimatorOutput(NamedTuple):
    """The supervised logits and the sole runtime explicit representation."""

    contact_logits: torch.Tensor
    contact_probability: torch.Tensor
    explicit_for_policy: torch.Tensor


def _explicit_estimator_output(raw, epsilon):
    """Apply the canonical contact-logit transform exactly once."""
    epsilon = float(epsilon)
    if not 0.0 <= epsilon < 0.5:
        raise ValueError("contact epsilon must lie in [0, 0.5)")
    contact_logits = raw[:, 3:7]
    contact_probability = epsilon + (
        1.0 - 2.0 * epsilon
    ) * torch.sigmoid(contact_logits)
    explicit_for_policy = torch.cat((
        raw[:, :3],
        contact_probability,
        raw[:, 7:11],
    ), dim=-1)
    return ExplicitEstimatorOutput(
        contact_logits, contact_probability, explicit_for_policy
    )


class ExplicitEstimatorDecoder(nn.Module):
    """Decode shared features into velocity, contact logits, and clearance."""

    def __init__(self, latent_dim=16, hidden_layers=(128, 128), output_dim=11,
                 contact_epsilon=1.0e-2):
        super().__init__()
        if output_dim != 11:
            raise ValueError("HardPACT explicit decoding requires 11 outputs")
        if not hidden_layers:
            raise ValueError("explicit estimator requires at least one hidden layer")
        layers = []
        input_dim = latent_dim
        for hidden_dim in hidden_layers:
            layers.extend((nn.Linear(input_dim, int(hidden_dim)), nn.ELU()))
            input_dim = int(hidden_dim)
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.input_dim = int(latent_dim)
        self.contact_epsilon = float(contact_epsilon)
        if not 0.0 <= self.contact_epsilon < 0.5:
            raise ValueError("contact epsilon must lie in [0, 0.5)")
        # This persistent marker makes checkpoints from the former
        # logits-in-policy convention fail strict loading instead of silently
        # changing the policy's conditioning semantics.
        self.register_buffer(
            "contact_probability_semantics", torch.ones((), dtype=torch.int64)
        )

    def forward(self, encoder_features):
        if encoder_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"explicit estimator input must be {self.input_dim}-D"
            )
        return _explicit_estimator_output(
            self.network(encoder_features), self.contact_epsilon
        )


class ContactEstimatorMetricsAccumulator:
    """Exactly aggregate masked contact diagnostics across minibatches."""

    def __init__(self, epsilon, saturation_fraction=0.01):
        self.epsilon = float(epsilon)
        if not 0.0 <= self.epsilon < 0.5:
            raise ValueError("contact epsilon must lie in [0, 0.5)")
        self.saturation_width = (
            float(saturation_fraction) * (1.0 - 2.0 * self.epsilon)
        )
        self._totals = {}

    def _add(self, name, value, reduce="sum"):
        value = value.detach()
        if name not in self._totals:
            self._totals[name] = value
        elif reduce == "max":
            self._totals[name] = torch.maximum(self._totals[name], value)
        elif reduce == "min":
            self._totals[name] = torch.minimum(self._totals[name], value)
        else:
            self._totals[name] = self._totals[name] + value

    @torch.no_grad()
    def update(self, estimator, labels, mask=None):
        logits = estimator.contact_logits.detach()
        probability = estimator.contact_probability.detach()
        labels = labels.detach().to(logits.dtype)
        valid = torch.ones_like(labels, dtype=torch.bool)
        if mask is not None:
            valid &= mask.detach().reshape(-1, 1).bool()
        weights = valid.to(logits.dtype)
        self._add("count", weights.sum())
        self._add("bce", (
            F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            * weights
        ).sum())
        self._add("logit", (logits * weights).sum())
        self._add("probability", (probability * weights).sum())
        self._add("brier", ((probability - labels).square() * weights).sum())
        self._add("correct", (
            ((probability >= 0.5) == labels.bool()).to(logits.dtype) * weights
        ).sum())
        self._add("lower_saturated", (
            ((probability - self.epsilon) <= self.saturation_width)
            .to(logits.dtype) * weights
        ).sum())
        self._add("upper_saturated", (
            (((1.0 - self.epsilon) - probability) <= self.saturation_width)
            .to(logits.dtype) * weights
        ).sum())
        self._add(
            "logit_abs_max",
            torch.where(valid, logits.abs(), logits.new_full((), -torch.inf)).max(),
            "max",
        )
        self._add(
            "probability_min",
            torch.where(valid, probability, probability.new_full((), torch.inf)).min(),
            "min",
        )
        self._add(
            "probability_max",
            torch.where(valid, probability, probability.new_full((), -torch.inf)).max(),
            "max",
        )

    @torch.no_grad()
    def finalize(self):
        if not self._totals:
            return {}
        total = self._totals
        denominator = total["count"].clamp_min(1.0)
        zero = denominator.new_zeros(())
        finite_or_zero = lambda value: torch.where(
            torch.isfinite(value), value, zero
        )
        return {
            "contact_bce": total["bce"] / denominator,
            "contact_logit_mean": total["logit"] / denominator,
            "contact_logit_abs_max": finite_or_zero(total["logit_abs_max"]),
            "contact_probability_mean": total["probability"] / denominator,
            "contact_probability_min": finite_or_zero(total["probability_min"]),
            "contact_probability_max": finite_or_zero(total["probability_max"]),
            "contact_probability_lower_saturation_fraction": (
                total["lower_saturated"] / denominator
            ),
            "contact_probability_upper_saturation_fraction": (
                total["upper_saturated"] / denominator
            ),
            "contact_classification_accuracy": total["correct"] / denominator,
            "contact_brier_score": total["brier"] / denominator,
        }


def contact_estimator_metrics(estimator, labels, epsilon, mask=None):
    """One-batch convenience wrapper around the shared exact accumulator."""
    accumulator = ContactEstimatorMetricsAccumulator(epsilon)
    accumulator.update(estimator, labels, mask)
    return accumulator.finalize()


def _physics_head(input_dim, hidden_layers, output_dim):
    if not hidden_layers:
        raise ValueError("physics decoder requires at least one hidden layer")
    layers = []
    for hidden_dim in hidden_layers:
        hidden_dim = int(hidden_dim)
        layers.extend((nn.Linear(input_dim, hidden_dim), nn.ELU()))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class DeploymentPhysicsHeads(nn.Module):
    """Yaw-local interval-GRF and sustained-base-wrench predictors."""

    def __init__(
        self,
        latent_dim=16,
        explicit_dim=11,
        grf_hidden_layers=(128, 128),
        wrench_hidden_layers=(128, 128),
        grf_scale_n=None,
        wrench_scale=None,
        wrench_qp_clip=None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.explicit_dim = int(explicit_dim)
        self.grf_head = _physics_head(
            latent_dim + explicit_dim + 12, grf_hidden_layers, 12
        )
        self.wrench_head = _physics_head(
            latent_dim + explicit_dim, wrench_hidden_layers, 6
        )
        if grf_scale_n is None or wrench_scale is None or wrench_qp_clip is None:
            raise ValueError(
                "GRF/wrench scales and QP clip must come from the task config"
            )
        # Config-resolved values travel with each deployed checkpoint.
        self.register_buffer(
            "grf_scale_n", torch.as_tensor(grf_scale_n, dtype=torch.float32)
        )
        self.register_buffer(
            "wrench_scale", torch.as_tensor(wrench_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "wrench_qp_clip",
            torch.as_tensor(wrench_qp_clip, dtype=torch.float32),
        )
        if self.grf_scale_n.shape != (12,):
            raise ValueError("grf_scale_n must contain 12 values")
        if self.wrench_scale.shape != (6,):
            raise ValueError("wrench_scale must contain 6 values")
        if self.wrench_qp_clip.shape != (6,):
            raise ValueError("wrench_qp_clip must contain 6 values")

    def forward(self, latent_sample, explicit, nominal_torque):
        """Evaluate both heads from the reparameterized training sample.

        ``explicit`` conditions the predictions by value only.  Detaching it
        at each decoder boundary prevents either physics loss from optimizing
        the explicit-estimator branch, while gradients through
        ``latent_sample`` continue into the shared context encoder.
        """
        if (latent_sample.shape[-1] != self.latent_dim
                or explicit.shape[-1] != self.explicit_dim):
            raise ValueError(
                "HardPACT physics-head input dimensions do not match configuration"
            )
        if nominal_torque.shape[-1] != 12:
            raise ValueError("nominal torque must be 12-D")
        grf = self.predict_grf(latent_sample, explicit, nominal_torque)
        wrench = self.predict_wrench(latent_sample, explicit)
        return PhysicsHeadOutput(
            grf, wrench,
        )

    def predict_grf(self, latent_sample, explicit, nominal_torque):
        """Evaluate only the torque-conditioned head.

        Rollout calls this once per physics substep, so keeping it separate
        avoids redundantly evaluating the control-rate wrench head.
        """
        if (latent_sample.shape[-1] != self.latent_dim
                or explicit.shape[-1] != self.explicit_dim):
            raise ValueError(
                "HardPACT physics-head input dimensions do not match configuration"
            )
        if nominal_torque.shape[-1] != 12:
            raise ValueError("nominal torque must be 12-D")
        stopped_explicit = explicit.detach()
        return self.grf_head(
            torch.cat((latent_sample, stopped_explicit, nominal_torque), dim=-1)
        )

    def grf_to_physical(self, prediction_normalized):
        """Convert a normalized decoder result to yaw-local Newtons once."""
        return grf_normalized_to_physical(
            prediction_normalized, self.grf_scale_n
        )

    def predict_wrench(self, latent_sample, explicit):
        """Evaluate the unbounded wrench head from a sampled latent."""
        if (latent_sample.shape[-1] != self.latent_dim
                or explicit.shape[-1] != self.explicit_dim):
            raise ValueError(
                "HardPACT physics-head input dimensions do not match configuration"
            )
        stopped_explicit = explicit.detach()
        return self.wrench_head(
            torch.cat((latent_sample, stopped_explicit), dim=-1)
        )

    def wrench_to_physical(self, prediction_normalized):
        return wrench_normalized_to_physical(
            prediction_normalized, self.wrench_scale
        )

    def wrench_to_qp_physical(self, prediction_normalized):
        """Reconstruct then sanitize/clip at the unique QP boundary."""
        return sanitize_and_clip_wrench_for_qp(
            self.wrench_to_physical(prediction_normalized), self.wrench_qp_clip
        )
