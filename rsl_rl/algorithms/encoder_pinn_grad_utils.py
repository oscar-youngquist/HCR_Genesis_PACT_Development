"""
Utilities for adding PINN-loss gradients to a context encoder update while
keeping actor/critic and encoder optimizers separate.

Typical use inside PPO_PACT.update(...):

    # After computing weighted_pinn_loss, before actor PCGrad backward:
    pinn_encoder_grads, pinn_grad_info = compute_encoder_grads_from_loss(
        weighted_pinn_loss,
        self.actor_critic.context_encoder,
    )

    # ... run actor/critic PCGrad update ...

    # Clear stale encoder grads caused by backward through the actor graph:
    zero_module_grads(self.actor_critic.context_encoder)

    # During the encoder update:
    self.enc_optimizer.zero_grad()
    vae_loss.backward()
    add_projected_aux_grads_to_module(
        module=self.actor_critic.context_encoder,
        aux_grads=pinn_encoder_grads,
        scale=self.pinn_encoder_grad_weight,
        primary_name="vae",
        aux_name="pinn",
    )
    self.enc_optimizer.step()

The projection treats the existing .grad fields on the encoder as the primary
encoder objective gradients, usually from the VAE/reconstruction loss, and the
saved auxiliary gradients as PINN gradients. If the PINN gradient conflicts
with the VAE gradient, only the conflicting component is removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


OptionalGradList = List[Optional[torch.Tensor]]


@dataclass
class EncoderGradInfo:
    """Diagnostics for saved auxiliary gradients."""

    total_params: int
    non_none_grads: int
    total_grad_norm: float
    has_any_grad: bool

    def as_dict(self):
        return {
            "total_params": self.total_params,
            "non_none_grads": self.non_none_grads,
            "total_grad_norm": self.total_grad_norm,
            "has_any_grad": self.has_any_grad,
        }


@dataclass
class ProjectedGradInfo:
    """Diagnostics for PCGrad-style auxiliary-gradient projection."""

    applied: bool
    reason: str
    primary_grad_norm: float = 0.0
    aux_grad_norm: float = 0.0
    projected_aux_grad_norm: float = 0.0
    dot_before: float = 0.0
    dot_after: float = 0.0
    cosine_before: float = 0.0
    cosine_after: float = 0.0
    shared_numel: int = 0

    def as_dict(self):
        return {
            "applied": self.applied,
            "reason": self.reason,
            "primary_grad_norm": self.primary_grad_norm,
            "aux_grad_norm": self.aux_grad_norm,
            "projected_aux_grad_norm": self.projected_aux_grad_norm,
            "dot_before": self.dot_before,
            "dot_after": self.dot_after,
            "cosine_before": self.cosine_before,
            "cosine_after": self.cosine_after,
            "shared_numel": self.shared_numel,
        }


def trainable_parameters(module: nn.Module) -> List[nn.Parameter]:
    """Return trainable parameters from a module in a stable order."""
    return [p for p in module.parameters() if p.requires_grad]


def zero_module_grads(module: nn.Module, set_to_none: bool = True) -> None:
    """Clear gradients for all parameters in a module."""
    for p in module.parameters():
        if set_to_none:
            p.grad = None
        elif p.grad is not None:
            p.grad.zero_()


def grad_list_info(grads: Sequence[Optional[torch.Tensor]]) -> EncoderGradInfo:
    """Summarize a list of optional gradients."""
    non_none = 0
    total_sq_norm = 0.0

    for g in grads:
        if g is None:
            continue
        non_none += 1
        total_sq_norm += float(g.detach().pow(2).sum().item())

    total_norm = total_sq_norm ** 0.5
    return EncoderGradInfo(
        total_params=len(grads),
        non_none_grads=non_none,
        total_grad_norm=total_norm,
        has_any_grad=non_none > 0 and total_norm > 0.0,
    )


def compute_module_grads_from_loss(
    loss: Optional[torch.Tensor],
    module: nn.Module,
    retain_graph: bool = True,
    allow_unused: bool = True,
    detach: bool = True,
) -> Tuple[OptionalGradList, EncoderGradInfo]:
    """
    Compute gradients of a loss with respect to a module's trainable parameters.

    This function does not write gradients into parameter .grad fields. It is
    therefore safe to call before a separate optimizer/backward pass.

    If `loss` is None, or if the loss is disconnected from the module, the
    returned gradient list contains only None entries. This is the expected
    behavior when, for example, a bootmasked actor path detaches or bypasses the
    context encoder.

    Returns:
        grads:
            List aligned with trainable_parameters(module). Each item is either
            a Tensor gradient or None.

        info:
            Diagnostic summary indicating whether any useful gradient exists.
    """
    params = trainable_parameters(module)

    if loss is None:
        grads: OptionalGradList = [None for _ in params]
        return grads, grad_list_info(grads)

    if not loss.requires_grad:
        grads = [None for _ in params]
        return grads, grad_list_info(grads)

    raw_grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )

    if detach:
        grads = [None if g is None else g.detach().clone() for g in raw_grads]
    else:
        grads = list(raw_grads)

    return grads, grad_list_info(grads)


def _flatten_optional_grads(
    grads: Sequence[Optional[torch.Tensor]],
    params: Sequence[nn.Parameter],
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Size]]:
    """
    Flatten optional gradients into one vector.

    None gradients are replaced by zeros and marked False in the returned mask.
    """
    if len(grads) != len(params):
        raise ValueError(
            f"Gradient/parameter length mismatch: {len(grads)} grads vs "
            f"{len(params)} params."
        )

    flat_grads: List[torch.Tensor] = []
    flat_masks: List[torch.Tensor] = []
    shapes: List[torch.Size] = []

    for g, p in zip(grads, params):
        shapes.append(p.shape)

        if g is None:
            flat_grads.append(torch.zeros_like(p).flatten())
            flat_masks.append(torch.zeros_like(p, dtype=torch.bool).flatten())
        else:
            flat_grads.append(g.flatten())
            flat_masks.append(torch.ones_like(g, dtype=torch.bool).flatten())

    if not flat_grads:
        # Use CPU tensors for an empty module. This should rarely occur, but it
        # makes downstream checks robust.
        return torch.empty(0), torch.empty(0, dtype=torch.bool), shapes

    return torch.cat(flat_grads), torch.cat(flat_masks), shapes


def _unflatten_grad(
    flat_grad: torch.Tensor,
    shapes: Sequence[torch.Size],
) -> List[torch.Tensor]:
    """Unflatten a single flat gradient vector into tensors with given shapes."""
    grads: List[torch.Tensor] = []
    idx = 0

    for shape in shapes:
        numel = 1
        for dim in shape:
            numel *= int(dim)
        grads.append(flat_grad[idx : idx + numel].view(shape).clone())
        idx += numel

    return grads


def _safe_cosine(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    denom = a.norm() * b.norm()
    if denom <= eps:
        return 0.0
    return float(torch.dot(a, b).item() / (float(denom.item()) + eps))


def project_aux_against_primary_pcgrad(
    primary_grad: torch.Tensor,
    aux_grad: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, bool]:
    """
    Project an auxiliary gradient using standard PCGrad logic.

    If dot(aux, primary) < 0, remove from aux the component that conflicts with
    primary. If dot(aux, primary) >= 0, keep aux unchanged.

    Args:
        primary_grad:
            Primary objective gradient, e.g. VAE encoder gradient.

        aux_grad:
            Auxiliary objective gradient, e.g. PINN encoder gradient.

    Returns:
        projected_aux_grad:
            Auxiliary gradient after optional projection.

        was_projected:
            True if a conflicting component was removed.
    """
    dot = torch.dot(aux_grad, primary_grad)

    if dot < 0.0:
        projected = aux_grad - dot * primary_grad / (primary_grad.norm() ** 2 + eps)
        return projected, True

    return aux_grad, False


def add_projected_aux_grads_to_module(
    module: nn.Module,
    aux_grads: Optional[Sequence[Optional[torch.Tensor]]],
    scale: float = 1.0,
    eps: float = 1e-12,
    project_only_on_shared: bool = True,
) -> ProjectedGradInfo:
    """
    Add saved auxiliary gradients to a module's current gradients with PCGrad.

    Expected call order:
        optimizer.zero_grad()
        primary_loss.backward()   # fills p.grad with primary gradients
        add_projected_aux_grads_to_module(module, aux_grads, scale)
        optimizer.step()

    The current .grad fields on `module` are treated as the primary objective
    gradients. `aux_grads` are treated as the auxiliary gradients.

    If all auxiliary gradients are None, this function does nothing and returns
    a diagnostic object with applied=False. This safely handles bootmask paths
    where the PINN loss is disconnected from the encoder.

    Args:
        module:
            Module receiving the combined update, e.g. context_encoder.

        aux_grads:
            Saved auxiliary gradients aligned with trainable_parameters(module),
            e.g. output of compute_module_grads_from_loss(...).

        scale:
            Multiplicative coefficient for the projected auxiliary gradient.

        eps:
            Numerical stability constant.

        project_only_on_shared:
            If True, compute conflict/projection only on entries where both the
            primary and auxiliary losses produced gradients. Non-shared aux
            entries are still added normally.

    Returns:
        ProjectedGradInfo diagnostics.
    """
    params = trainable_parameters(module)

    if aux_grads is None:
        return ProjectedGradInfo(applied=False, reason="aux_grads_is_none")

    aux_info = grad_list_info(aux_grads)
    if not aux_info.has_any_grad:
        return ProjectedGradInfo(applied=False, reason="aux_grads_all_none_or_zero")

    primary_grads: OptionalGradList = [
        None if p.grad is None else p.grad.detach().clone()
        for p in params
    ]
    primary_info = grad_list_info(primary_grads)

    flat_primary, primary_mask, shapes = _flatten_optional_grads(primary_grads, params)
    flat_aux, aux_mask, _ = _flatten_optional_grads(aux_grads, params)

    if flat_aux.numel() == 0:
        return ProjectedGradInfo(applied=False, reason="module_has_no_trainable_params")

    if project_only_on_shared:
        compare_mask = primary_mask & aux_mask
    else:
        compare_mask = aux_mask

    projected_aux = flat_aux.clone()
    was_projected = False

    if compare_mask.any() and primary_info.has_any_grad:
        primary_cmp = flat_primary[compare_mask]
        aux_cmp = flat_aux[compare_mask]

        dot_before_tensor = torch.dot(aux_cmp, primary_cmp)
        dot_before = float(dot_before_tensor.item())
        cosine_before = _safe_cosine(aux_cmp, primary_cmp, eps=eps)

        projected_cmp, was_projected = project_aux_against_primary_pcgrad(
            primary_grad=primary_cmp,
            aux_grad=aux_cmp,
            eps=eps,
        )

        projected_aux[compare_mask] = projected_cmp

        dot_after = float(torch.dot(projected_cmp, primary_cmp).item())
        cosine_after = _safe_cosine(projected_cmp, primary_cmp, eps=eps)
    else:
        dot_before = 0.0
        dot_after = 0.0
        cosine_before = 0.0
        cosine_after = 0.0

    projected_aux_grads = _unflatten_grad(projected_aux, shapes)

    # Add projected auxiliary gradients into .grad, preserving None entries for
    # parameters where aux_grads was None.
    for p, g_aux_projected, g_aux_original in zip(params, projected_aux_grads, aux_grads):
        if g_aux_original is None:
            continue

        if p.grad is None:
            p.grad = scale * g_aux_projected
        else:
            p.grad.add_(g_aux_projected, alpha=scale)

    projected_norm = float(projected_aux.norm().item())

    return ProjectedGradInfo(
        applied=True,
        reason="projected" if was_projected else "no_conflict",
        primary_grad_norm=primary_info.total_grad_norm,
        aux_grad_norm=aux_info.total_grad_norm,
        projected_aux_grad_norm=projected_norm,
        dot_before=dot_before,
        dot_after=dot_after,
        cosine_before=cosine_before,
        cosine_after=cosine_after,
        shared_numel=int(compare_mask.sum().item()) if compare_mask.numel() > 0 else 0,
    )


# Backwards-compatible aliases with names matching the earlier PPO snippets.
def compute_encoder_grads_from_loss(
    loss: Optional[torch.Tensor],
    context_encoder: nn.Module,
    retain_graph: bool = True,
    allow_unused: bool = True,
) -> Tuple[OptionalGradList, EncoderGradInfo]:
    return compute_module_grads_from_loss(
        loss=loss,
        module=context_encoder,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )


def add_projected_pinn_grads_to_encoder(
    context_encoder: nn.Module,
    pinn_encoder_grads: Optional[Sequence[Optional[torch.Tensor]]],
    scale: float = 1.0,
    eps: float = 1e-12,
) -> ProjectedGradInfo:
    return add_projected_aux_grads_to_module(
        module=context_encoder,
        aux_grads=pinn_encoder_grads,
        scale=scale,
        eps=eps,
    )
