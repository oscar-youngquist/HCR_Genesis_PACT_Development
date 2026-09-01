"""Corrected normalized BARD inverse-dynamics objective for HardPACT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass
class InverseDynamicsResult:
    loss: torch.Tensor
    residual: torch.Tensor
    metrics: Mapping[str, torch.Tensor]
    physics_valid_mask: torch.Tensor


def physics_valid_mask(push_event_mask, reset_mask, timeout_mask, teleport_mask):
    # m_t = ¬(push_t ∨ reset_t ∨ timeout_t ∨ teleport_t).
    # Persistent external wrenches and mass/CoM randomization are deliberately
    # absent from this expression, so those transitions remain valid.
    return ~(
        push_event_mask.bool()
        | reset_mask.bool()
        | timeout_mask.bool()
        | teleport_mask.bool()
    )


def _masked_coordinate_metric(values, valid, coordinate_slice):
    block = values[:, coordinate_slice]
    weights = valid[:, None].to(block.dtype)
    denominator = (valid.sum() * block.shape[-1]).clamp_min(1).to(block.dtype)
    return (block * weights).sum() / denominator


def corrected_bard_inverse_dynamics_loss(
    *,
    required_generalized_force,
    foot_jacobians,
    base_jacobian,
    interval_executed_torque,
    interval_grf_world,
    total_wrench_world,
    mass_com_wrench_world,
    measured_generalized_contact_force,
    push_event_mask,
    reset_mask,
    timeout_mask,
    teleport_mask,
):
    r"""Form and reduce the corrected inverse-dynamics residual.

    For each transition, BARD supplies the required generalized force

    .. math::

        \tau_{\mathrm{RNEA}}
        = \operatorname{RNEA}(q_t, v_t, \dot v_t^{\mathrm{obs}};
          \theta_{\mathrm{rand}}).

    The learned wrench is the *total* label, so its actually applied component
    is recovered without counting randomized mass/CoM gravity twice:

    .. math::

        \hat W_{\mathrm{applied}}^W
        = \hat W_{\mathrm{total}}^W - W_{\mathrm{mass/CoM}}^W.

    Generalized actuation, contact, and base-wrench forces are

    .. math::

        \tau_a=[0_6,\bar\tau_{\mathrm{exec}}],\qquad
        \tau_f=\sum_{i=1}^{4}(J_{f_i}^{W})^T\hat F_i^W,\qquad
        \tau_W=(J_b^W)^T\hat W_{\mathrm{applied}}^W,

    and the 18-D inverse-dynamics residual is

    .. math::

        r_{\mathrm{ID}}
        = \tau_{\mathrm{RNEA}}-\tau_a-\tau_f-\tau_W.

    Let :math:`g_t` be the measured generalized contact force retained from
    the legacy Pinocchio rollout. Its positive component defines a continuous
    contact weight

    .. math::

        w_{t,j}=\frac{\max(g_{t,j},0)}
        {\max_k\max(g_{t,k},0)+10^{-8}}.

    The per-transition relative error and masked loss are

    .. math::

        e_t=\frac{\lVert w_t\odot r_t\rVert_2}
        {10^{-8}+\lVert\tau_{a,t}\rVert_2+\lVert g_t\rVert_2},\qquad
        \mathcal L_{\mathrm{ID}}=\frac{\sum_t m_t e_t}
        {\max(1,\sum_t m_t)}.

    Logged relative residual blocks use this same contact weighting and scalar
    denominator, so their normalization agrees exactly with the objective.

    All measured/BARD inputs are detached here as a final guard.  Only the two
    learned physical force predictions retain gradients.
    """
    required = required_generalized_force.detach()
    torque = interval_executed_torque.detach()
    foot_jacobians = foot_jacobians.detach()
    base_jacobian = base_jacobian.detach()
    mass_com = mass_com_wrench_world.detach()
    measured_contact = measured_generalized_contact_force.detach()
    if measured_contact.shape != required.shape:
        raise ValueError("measured generalized contact force must be 18-D")
    # Ŵ_applied^W = Ŵ_total^W - W_mass/CoM^W. The second term is a
    # detached label because its inertial effect is already inside RNEA.
    applied_wrench = total_wrench_world - mass_com
    # τ_a = [0₆, τ̄_exec] ∈ ℝ¹⁸.
    actuation = torch.cat((torch.zeros_like(torque[:, :6]), torque), dim=-1)
    # τ_f = Σᵢ (J_fᵢ^W)ᵀ F̂ᵢ^W and τ_W = (J_b^W)ᵀ Ŵ_applied^W.
    contact = torch.einsum("bfkn,bfk->bn", foot_jacobians, interval_grf_world)
    wrench = torch.einsum("bkn,bk->bn", base_jacobian, applied_wrench)
    # r_ID = RNEA(q_t,v_t,v̇_obs;θ_rand) - τ_a - τ_f - τ_W.
    residual = required - actuation - contact - wrench
    valid = physics_valid_mask(
        push_event_mask, reset_mask, timeout_mask, teleport_mask
    ).reshape(-1)
    # Soft contact weighting is label-derived. Keeping it detached prevents a
    # force head from reducing the objective by shrinking its own gate.
    contact_magnitude = torch.clamp(measured_contact, min=0.0)
    contact_max = torch.max(contact_magnitude, dim=1, keepdim=True)[0]
    contact_weight = contact_magnitude / (contact_max + 1.0e-8)
    weighted_residual = residual * contact_weight

    # Match the legacy relative-force objective while retaining gradients
    # through r_ID's learned GRF and wrench terms.
    denominator = (
        1.0e-8
        + torch.linalg.vector_norm(actuation.detach(), dim=1)
        + torch.linalg.vector_norm(measured_contact, dim=1)
    )
    per_sample = torch.linalg.vector_norm(weighted_residual, dim=1) / denominator
    relative_residual = weighted_residual / denominator.unsqueeze(-1)
    relative_square = relative_residual.square()
    weights = valid.to(per_sample.dtype)
    loss = (per_sample * weights).sum() / weights.sum().clamp_min(1.0)
    # This expression is already graph-connected when all samples are invalid;
    # multiplying by zero therefore produces exact zero gradients to both heads.
    absolute = residual.abs()
    blocks = {
        "base_linear": slice(0, 3),
        "base_angular": slice(3, 6),
        "joints": slice(6, 18),
        "all": slice(0, 18),
    }
    metrics = {}
    for name, block in blocks.items():
        metrics[f"inverse_residual/{name}_mae_physical"] = (
            _masked_coordinate_metric(absolute, valid, block).detach()
        )
        metrics[f"inverse_residual/{name}_mse_relative"] = (
            _masked_coordinate_metric(relative_square, valid, block).detach()
        )
    return InverseDynamicsResult(loss, residual, metrics, valid)
