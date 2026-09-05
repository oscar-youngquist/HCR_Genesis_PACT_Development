"""Deployment physics heads and shared force-scaling helpers for HardPACT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

GRF_NORMALIZATION_VERSION = 2
GRF_COMPONENT_SCALE_N = (250.0, 250.0, 250.0)
GRF_SCALE_N = GRF_COMPONENT_SCALE_N * 4
WRENCH_NORMALIZATION_VERSION = 2
WRENCH_SCALE_N_NM = (100.0, 100.0, 100.0, 25.0, 25.0, 25.0)
WRENCH_QP_CLIP_N_NM = (150.0, 150.0, 150.0, 40.0, 40.0, 40.0)


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


def normalize_grf_target(grf_physical_n, grf_scale_n=GRF_SCALE_N):
    """Normalize yaw-local FR/FL/RR/RL XYZ forces without observation scaling."""
    return grf_physical_n / _broadcast_grf_scale(
        grf_physical_n, grf_scale_n
    )


def grf_normalized_to_physical(grf_normalized, grf_scale_n=GRF_SCALE_N):
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


def normalize_wrench_target(wrench_physical, wrench_scale=WRENCH_SCALE_N_NM):
    """Normalize yaw-local [force, moment] labels without observation scaling."""
    scale = torch.as_tensor(
        wrench_scale, device=wrench_physical.device, dtype=wrench_physical.dtype
    )
    return wrench_physical / scale


def wrench_normalized_to_physical(
    wrench_raw_normalized, wrench_scale=WRENCH_SCALE_N_NM
):
    """Convert the unbounded decoder output to physical N/Nm exactly once."""
    scale = torch.as_tensor(
        wrench_scale,
        device=wrench_raw_normalized.device,
        dtype=wrench_raw_normalized.dtype,
    )
    return wrench_raw_normalized * scale


def sanitize_and_clip_wrench_for_qp(
    wrench_raw_physical, wrench_qp_clip=WRENCH_QP_CLIP_N_NM
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
    mask=None,
    wrench_scale=WRENCH_SCALE_N_NM,
    wrench_qp_clip=WRENCH_QP_CLIP_N_NM,
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
            clipped_foot_clearances.clamp(-1.0, 1.0),
        ),
        dim=-1,
    )


def contact_logits_to_qp_probability(contact_logits, epsilon=1.0e-2):
    """Map raw contact logits once at the QP boundary."""
    if not 0.0 <= float(epsilon) < 0.5:
        raise ValueError("contact epsilon must lie in [0, 0.5)")
    return float(epsilon) + (
        1.0 - 2.0 * float(epsilon)
    ) * torch.sigmoid(contact_logits)


def transform_explicit_estimator_output(raw):
    # Contacts remain raw logits. Only the continuous clearance branch retains
    # its established runtime clipping; QP probability conversion is separate.
    return torch.cat((raw[:, :7], raw[:, 7:11].clamp(-1.0, 1.0)), dim=-1)


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

    def forward(self, encoder_features):
        if encoder_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"explicit estimator input must be {self.input_dim}-D"
            )
        return transform_explicit_estimator_output(self.network(encoder_features))


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
        grf_scale_n=GRF_SCALE_N,
        wrench_scale=WRENCH_SCALE_N_NM,
        wrench_qp_clip=WRENCH_QP_CLIP_N_NM,
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
        # Persistent constants make the deployed numerical contract part of
        # every checkpoint and available without an external config object.
        self.register_buffer(
            "grf_scale_n", torch.as_tensor(grf_scale_n, dtype=torch.float32)
        )
        self.register_buffer(
            "grf_normalization_version",
            torch.tensor(GRF_NORMALIZATION_VERSION, dtype=torch.int64),
        )
        # The fixed scale, QP safety boundary, and contract version travel
        # with the weights used by both HardPACT training pipelines.
        self.register_buffer(
            "wrench_scale", torch.as_tensor(wrench_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "wrench_qp_clip",
            torch.as_tensor(wrench_qp_clip, dtype=torch.float32),
        )
        self.register_buffer(
            "wrench_normalization_version",
            torch.tensor(WRENCH_NORMALIZATION_VERSION, dtype=torch.int64),
        )
        if self.grf_scale_n.shape != (12,):
            raise ValueError("grf_scale_n must contain 12 values")
        if self.wrench_scale.shape != (6,):
            raise ValueError("wrench_scale must contain 6 values")
        if self.wrench_qp_clip.shape != (6,):
            raise ValueError("wrench_qp_clip must contain 6 values")

    def forward(self, latent, explicit, nominal_torque):
        if (latent.shape[-1] != self.latent_dim
                or explicit.shape[-1] != self.explicit_dim):
            raise ValueError(
                "HardPACT physics-head input dimensions do not match configuration"
            )
        if nominal_torque.shape[-1] != 12:
            raise ValueError("nominal torque must be 12-D")
        grf = self.predict_grf(latent, explicit, nominal_torque)
        wrench = self.predict_wrench(latent, explicit)
        return PhysicsHeadOutput(
            grf, wrench,
        )

    def predict_grf(self, latent, explicit, nominal_torque):
        """Evaluate only the torque-conditioned head.

        Rollout calls this once per physics substep, so keeping it separate
        avoids redundantly evaluating the control-rate wrench head.
        """
        if (latent.shape[-1] != self.latent_dim
                or explicit.shape[-1] != self.explicit_dim):
            raise ValueError(
                "HardPACT physics-head input dimensions do not match configuration"
            )
        if nominal_torque.shape[-1] != 12:
            raise ValueError("nominal torque must be 12-D")
        stopped_explicit = explicit.detach()
        return self.grf_head(
            torch.cat((latent, stopped_explicit, nominal_torque), dim=-1)
        )

    def grf_to_physical(self, prediction_normalized):
        """Convert a normalized decoder result to yaw-local Newtons once."""
        return grf_normalized_to_physical(
            prediction_normalized, self.grf_scale_n
        )

    def predict_wrench(self, latent, explicit):
        """Evaluate the unbounded, normalized control-rate wrench head."""
        if (latent.shape[-1] != self.latent_dim
                or explicit.shape[-1] != self.explicit_dim):
            raise ValueError(
                "HardPACT physics-head input dimensions do not match configuration"
            )
        stopped_explicit = explicit.detach()
        return self.wrench_head(torch.cat((latent, stopped_explicit), dim=-1))

    def wrench_to_physical(self, prediction_normalized):
        return wrench_normalized_to_physical(
            prediction_normalized, self.wrench_scale
        )

    def wrench_to_qp_physical(self, prediction_normalized):
        """Reconstruct then sanitize/clip at the unique QP boundary."""
        return sanitize_and_clip_wrench_for_qp(
            self.wrench_to_physical(prediction_normalized), self.wrench_qp_clip
        )
