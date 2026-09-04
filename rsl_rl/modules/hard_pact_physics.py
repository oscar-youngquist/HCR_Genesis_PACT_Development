"""Deployment physics heads and shared force-scaling helpers for HardPACT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class PhysicsHeadOutput:
    grf_yaw_scaled: torch.Tensor
    base_wrench_yaw_scaled: torch.Tensor


def scale_head_output(normalized_output, model_gain):
    """Map a normalized head output into model/observation-scaled units."""
    return normalized_output * torch.as_tensor(
        model_gain, device=normalized_output.device, dtype=normalized_output.dtype
    )


def scale_physical_target(physical_target, observation_scale):
    """Map a physical N/Nm target into model-space observation units."""
    return physical_target * float(observation_scale)


def model_output_to_physical(model_output, observation_scale):
    """Invert observation scaling, returning physical N/Nm values."""
    return model_output / float(observation_scale)


def normalized_huber_loss(prediction, target, model_gain, mask=None, delta=1.0):
    """Huber loss normalized componentwise by the frozen head gain."""
    gain = torch.as_tensor(
        model_gain, device=prediction.device, dtype=prediction.dtype
    ).clamp_min(1.0e-8)
    error = (prediction - target) / gain
    loss = F.huber_loss(
        error, torch.zeros_like(error), delta=delta, reduction="none"
    ).mean(dim=-1)
    if mask is None:
        return loss.mean()
    mask = mask.reshape(-1).to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def physical_unit_mae(prediction, target, observation_scale, mask=None):
    """Mean absolute error after converting model outputs to N/Nm."""
    error = model_output_to_physical(
        (prediction - target).abs(), observation_scale
    ).mean(dim=-1)
    if mask is None:
        return error.mean()
    mask = mask.reshape(-1).to(error.dtype)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


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


def smooth_contact_probability(logit, epsilon=1.0e-2):
    """Map logits once into the open deployment interval [eps, 1-eps]."""
    if not 0.0 <= float(epsilon) < 0.5:
        raise ValueError("contact epsilon must lie in [0, 0.5)")
    return float(epsilon) + (1.0 - 2.0 * float(epsilon)) * logit.sigmoid()


def transform_explicit_estimator_output(raw, contact_epsilon=1.0e-2):
    return compose_explicit_estimator_target(
        raw[:, :3], smooth_contact_probability(raw[:, 3:7], contact_epsilon),
        raw[:, 7:11].clamp(-1.0, 1.0)
    )


class ExplicitEstimatorDecoder(nn.Module):
    """Decode a configurable history latent into the fixed 11-D estimate."""

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
        self.latent_dim = int(latent_dim)
        self.contact_epsilon = float(contact_epsilon)

    def forward(self, latent_mean):
        if latent_mean.shape[-1] != self.latent_dim:
            raise ValueError(
                f"explicit estimator input must be {self.latent_dim}-D"
            )
        return transform_explicit_estimator_output(
            self.network(latent_mean), self.contact_epsilon
        )


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
        grf_scale,
        wrench_scale,
        latent_dim=16,
        explicit_dim=11,
        grf_hidden_layers=(128, 128),
        wrench_hidden_layers=(128, 128),
        wrench_center=None,
        wrench_radius=None,
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
        self.register_buffer("grf_scale", torch.as_tensor(grf_scale, dtype=torch.float32))
        self.register_buffer(
            "wrench_scale", torch.as_tensor(wrench_scale, dtype=torch.float32)
        )
        center = torch.zeros(6) if wrench_center is None else torch.as_tensor(
            wrench_center, dtype=torch.float32
        )
        radius = self.wrench_scale.detach().clone() if wrench_radius is None else torch.as_tensor(
            wrench_radius, dtype=torch.float32
        )
        # Nonpersistent deployment constants preserve strict legacy checkpoint
        # keys; the JSON contract is their authoritative serialized form.
        self.register_buffer("wrench_center", center, persistent=False)
        self.register_buffer("wrench_radius", radius, persistent=False)
        if self.grf_scale.shape != (12,):
            raise ValueError("grf_scale must contain 12 values")
        if self.wrench_scale.shape != (6,):
            raise ValueError("wrench_scale must contain 6 values")

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
        value = self.grf_head(
            torch.cat((latent, stopped_explicit, nominal_torque), dim=-1)
        )
        return scale_head_output(value, self.grf_scale)

    def predict_wrench(self, latent, explicit):
        """Evaluate the control-rate base-wrench prediction."""
        if (latent.shape[-1] != self.latent_dim
                or explicit.shape[-1] != self.explicit_dim):
            raise ValueError(
                "HardPACT physics-head input dimensions do not match configuration"
            )
        stopped_explicit = explicit.detach()
        raw = self.wrench_head(torch.cat((latent, stopped_explicit), dim=-1))
        return self.wrench_center + self.wrench_radius * torch.tanh(raw)
