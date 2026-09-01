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


@dataclass
class RolloutDynamicsResult:
    loss: torch.Tensor
    predicted_velocity: torch.Tensor
    target_velocity: torch.Tensor
    acceleration: torch.Tensor
    generalized_force: torch.Tensor
    metrics: Mapping[str, torch.Tensor]
    physics_valid_mask: torch.Tensor


BARD_ROLLOUT_INCREMENT_RATE_SCALES = (10.0, 20.0, 100.0)


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


def compose_generalized_force(
    control_torque, foot_jacobians, interval_grf_world,
    base_jacobian, applied_wrench_world,
):
    r"""Compose the 18-D force passed to BARD ABA.

    All inputs are already expressed at the cached transition state and in
    canonical ``[base linear, base angular, FR, FL, RR, RL joints]`` order:

    .. math::

        g=S^T\tau_{\rm control}
          +\sum_{i=1}^{4}(J_{f_i}^{W})^T\hat F_i^W
          +(J_b^W)^T\hat W_{\rm applied}^W.

    The Jacobians are measured geometry, not learned quantities, so they are
    detached.  Predicted force/wrench and control torque deliberately are not:
    their gradients must pass through this composition and through ABA.
    """
    # A floating base is unactuated: S^T tau = [0_6, tau_control].
    actuation = torch.cat((
        torch.zeros_like(control_torque[:, :6]), control_torque,
    ), dim=-1)
    # Each foot Jacobian has shape [3,18], hence
    # sum_i J_f_i^T F_i -> one 18-D generalized contact force.
    contact = torch.einsum(
        "bfkn,bfk->bn", foot_jacobians.detach(), interval_grf_world
    )
    # J_b and W use the same world-aligned axes and base reference point.
    wrench = torch.einsum(
        "bkn,bk->bn", base_jacobian.detach(), applied_wrench_world
    )
    return actuation + contact + wrench


def differentiable_bard_rollout_loss(
    *, context, control_torque, interval_grf_world, applied_wrench_world,
    control_dt, push_event_mask, reset_mask, timeout_mask, teleport_mask,
    increment_rate_scales=BARD_ROLLOUT_INCREMENT_RATE_SCALES,
):
    r"""Integrate one differentiable BARD ABA step and score its velocity.

    This is the HardPACT rollout/PINN objective.  The reusable ``context`` has
    already converted and detached the measured state, installed each
    environment's realized mass/CoM and passive parameters, updated BARD
    kinematics once, and cached world-aligned Jacobians.  Consequently this
    function contains only differentiable force construction, forward
    dynamics, integration, and reduction.

    First form the generalized force

    .. math::

        g=S^T\tau_{\rm control}+J_f^T\hat F+J_b^T\hat W_{\rm applied}.

    ``context.aba`` evaluates the official articulated-body algorithm with
    the realized inertias, armature, and passive-force convention shared by
    the corrected inverse loss:

    .. math::

        \hat{\dot v}_t=\operatorname{ABA}(q_t,v_t,g;\theta_{\rm rand}).

    Compare velocity *increments*, rather than absolute endpoint velocities:

    .. math::

        \Delta\hat v_t=\Delta t\,\hat{\dot v}_t,\qquad
        \Delta v_t^{\rm obs}=v_{t+1}^{\rm BARD}-v_t^{\rm BARD}.

    The base-linear, base-angular, and joint blocks use rate envelopes
    :math:`c_b\in[10,20,100]`, respectively.  For block :math:`b`, define

    .. math::

        R_{t,b}=\operatorname{RMS}\!\left(
          \frac{\Delta\hat v_{t,b}-\Delta v_{t,b}^{\rm obs}}
               {\Delta t\,c_b}\right),\qquad
        O_{t,b}=\operatorname{stopgrad}\!\left[
          \operatorname{RMS}\!\left(
          \frac{\Delta v_{t,b}^{\rm obs}}{\Delta t\,c_b}\right)\right].

    The three coordinate groups contribute equally, independent of their
    widths and typical velocity magnitudes:

    .. math::

        \ell_t=\frac13\sum_b\frac{R_{t,b}}{1+O_{t,b}},\qquad
        \mathcal L_{\rm rollout}
          =\frac{\sum_t m_t\ell_t}{\max(1,\sum_t m_t)},

    where

    .. math::

        m_t=\neg(\text{push}\lor\text{reset}\lor
                  \text{timeout}\lor\text{teleport}).

    Sustained-wrench and randomized mass/CoM samples therefore remain valid.
    Measured state, targets, timestep, randomized parameters, and Jacobians
    are detached; gradients remain connected to control torque, predicted GRF,
    and predicted applied wrench through the official ABA computation.
    """
    if context.foot_jacobians is None or context.base_jacobian is None:
        raise ValueError("rollout context requires cached force Jacobians")
    if context.post_v_bard is None:
        raise ValueError("rollout context requires a cached post-step velocity")
    # g = S^T tau_control + J_f^T F_hat + J_b^T W_hat_applied.
    generalized_force = compose_generalized_force(
        control_torque, context.foot_jacobians, interval_grf_world,
        context.base_jacobian, applied_wrench_world,
    )
    # q_t, v_t, theta_rand, passive terms, and armature live in the cached
    # context.  Only g is differentiable at this call boundary.
    acceleration = context.aba(generalized_force)

    # Work in velocity increments. BARD stores URDF joint order; `_canonical`
    # maps both measured endpoints to simulator FR/FL/RR/RL joint order.
    dt = control_dt.detach().reshape(-1, 1).clamp_min(1.0e-8)
    pre_velocity = context.dynamics._canonical(context.v_bard).detach()
    target_velocity = context.dynamics._canonical(context.post_v_bard).detach()
    predicted_increment = dt * acceleration
    observed_increment = target_velocity - pre_velocity
    increment_residual = predicted_increment - observed_increment
    predicted_velocity = pre_velocity + predicted_increment

    # Each rate envelope is converted to an increment envelope by dt.  This
    # makes the objective invariant to the chosen control timestep for an
    # equivalent acceleration error.
    rate_scales = torch.as_tensor(
        increment_rate_scales, device=predicted_velocity.device,
        dtype=predicted_velocity.dtype,
    )
    if rate_scales.shape != (3,) or torch.any(rate_scales <= 0):
        raise ValueError("increment_rate_scales must be three positive values")
    objective_blocks = {
        "base_linear": slice(0, 3),
        "base_angular": slice(3, 6),
        "joints": slice(6, 18),
    }
    block_scores = {}
    normalized_residual_blocks = []
    for index, (name, block) in enumerate(objective_blocks.items()):
        increment_scale = dt * rate_scales[index]
        normalized_residual = increment_residual[:, block] / increment_scale
        normalized_observed = observed_increment[:, block] / increment_scale
        normalized_residual_blocks.append(normalized_residual)
        # vector_norm/sqrt(width) is RMS and has a finite zero subgradient.
        residual_rms = torch.linalg.vector_norm(
            normalized_residual, dim=-1
        ) / normalized_residual.shape[-1] ** 0.5
        observed_rms = (
            torch.linalg.vector_norm(normalized_observed, dim=-1)
            / normalized_observed.shape[-1] ** 0.5
        ).detach()
        block_scores[name] = residual_rms / (1.0 + observed_rms)
    per_sample = torch.stack(tuple(block_scores.values()), dim=-1).mean(dim=-1)
    # Discontinuous transitions invalidate finite-difference dynamics.
    # Persistent wrench and inertial randomization are intentionally omitted.
    valid = physics_valid_mask(
        push_event_mask, reset_mask, timeout_mask, teleport_mask
    ).reshape(-1)
    weights = valid.to(per_sample.dtype)
    # clamp_min makes an all-invalid minibatch a graph-connected exact zero;
    # backward then yields zero (rather than missing or NaN) head gradients.
    loss = (per_sample * weights).sum() / weights.sum().clamp_min(1.0)
    # Preserve the established logging interface. Endpoint velocity error and
    # increment error are algebraically identical, so physical MAE retains its
    # old meaning. Normalized MSE now uses the new dt-scaled block envelopes;
    # the optimized relative-RMS value is available as the aggregate loss.
    normalized_square = torch.cat(normalized_residual_blocks, dim=-1).square()
    metric_blocks = {**objective_blocks, "all": slice(0, 18)}
    metrics = {}
    for name, block in metric_blocks.items():
        metrics[f"rollout_velocity/{name}_mae_physical"] = (
            _masked_coordinate_metric(
                increment_residual.abs(), valid, block
            ).detach()
        )
        metrics[f"rollout_velocity/{name}_mse_normalized"] = (
            _masked_coordinate_metric(normalized_square, valid, block).detach()
        )
    return RolloutDynamicsResult(
        loss, predicted_velocity, target_velocity, acceleration,
        generalized_force, metrics, valid,
    )


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
