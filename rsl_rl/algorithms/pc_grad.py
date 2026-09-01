"""Gradient projection wrapper used by PACT and B1Z1-style physics losses."""

import copy
import random

import numpy as np
import torch


class PCGrad:
    """Wrap an optimizer and merge gradients from multiple objectives.

    The two PINN entry points preserve the legacy B1Z1 projection rules. The
    packing code additionally records which parameters each objective touched;
    parameters untouched by every objective receive ``grad=None`` so AdamW
    cannot change them through momentum or weight decay.
    """

    def __init__(self, optimizer, reduction="mean"):
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self._optim = optimizer
        self._reduction = reduction
        self.last_objective_grads = None
        self.last_merged_grad = None
        self.last_has_grads = None

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives):
        """Apply ordinary symmetric PCGrad to one or more objectives."""
        self._backward(objectives, self._project_conflicting)

    def pc_backward_pinn(self, objectives):
        """Apply B1Z1's PPO-primary, orthogonal-PINN projection."""
        self._backward(objectives, self._project_conflicting_pinn)

    def pc_backward_ppgrad(self, objectives):
        """Apply the norm-balanced form of B1Z1's PINN projection."""
        self._backward(objectives, self._project_conflicting_pinn_balanced)

    def _backward(self, objectives, projector):
        if not objectives:
            raise ValueError("PCGrad requires at least one objective")
        grads, shapes, has_grads, has_any_grad = self._pack_grad(objectives)
        merged = projector(grads, has_grads)
        self._record_backward(grads, merged, has_grads)
        self._set_grad(self._unflatten_grad(merged, shapes[0]), has_any_grad)

    def _merge(self, projected, shared):
        merged = torch.zeros_like(projected[0])
        shared_values = torch.stack([gradient[shared] for gradient in projected])
        if self._reduction == "mean":
            merged[shared] = shared_values.mean(dim=0)
        else:
            merged[shared] = shared_values.sum(dim=0)
        # A parameter used by only a subset of objectives is not a conflict;
        # sum its available task gradients exactly as upstream PCGrad does.
        merged[~shared] = torch.stack(
            [gradient[~shared] for gradient in projected]
        ).sum(dim=0)
        return merged

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        projected = copy.deepcopy(grads)
        for gradient in projected:
            comparison_order = list(range(len(grads)))
            random.shuffle(comparison_order)
            for index in comparison_order:
                other = grads[index]
                dot = torch.dot(gradient, other)
                if dot < 0:
                    gradient -= dot * other / other.square().sum().clamp_min(1.0e-12)
        return self._merge(projected, shared)

    def _project_conflicting_pinn(self, grads, has_grads, shapes=None):
        if len(grads) != 2:
            raise ValueError("B1Z1 PINN projection requires [PPO, physics]")
        reward, physics = copy.deepcopy(grads)
        coefficient = torch.dot(reward, physics) / reward.square().sum().clamp_min(1.0e-12)
        physics_orthogonal = physics - coefficient * reward
        shared = torch.stack(has_grads).prod(0).bool()
        return self._merge([grads[0], physics_orthogonal], shared)

    def _project_conflicting_pinn_balanced(self, grads, has_grads, shapes=None):
        if len(grads) != 2:
            raise ValueError("balanced B1Z1 projection requires [PPO, physics]")
        reward, physics = copy.deepcopy(grads)
        coefficient = torch.dot(reward, physics) / reward.square().sum().clamp_min(1.0e-12)
        physics_orthogonal = physics - coefficient * reward
        reward_norm = reward.norm()
        physics_norm = physics_orthogonal.norm()
        beta = (
            reward_norm / physics_norm.clamp_min(1.0e-12)
            if physics_norm > reward_norm else 1.0
        )
        shared = torch.stack(has_grads).prod(0).bool()
        return self._merge([grads[0], beta * physics_orthogonal], shared)

    def _record_backward(self, objective_grads, merged_grad, has_grads):
        self.last_objective_grads = tuple(
            gradient.detach().clone() for gradient in objective_grads
        )
        self.last_merged_grad = merged_grad.detach().clone()
        self.last_has_grads = tuple(mask.detach().clone() for mask in has_grads)

    def _set_grad(self, grads, has_any_grad):
        index = 0
        for group in self._optim.param_groups:
            for parameter in group["params"]:
                parameter.grad = grads[index] if has_any_grad[index] else None
                index += 1

    def _pack_grad(self, objectives):
        grads, shapes, has_grads, parameter_masks = [], [], [], []
        for objective in objectives:
            self._optim.zero_grad(set_to_none=True)
            objective.backward(retain_graph=True)
            grad, shape, has_grad, parameter_has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad))
            shapes.append(shape)
            has_grads.append(self._flatten_grad(has_grad))
            parameter_masks.append(parameter_has_grad)
        has_any_grad = [any(flags) for flags in zip(*parameter_masks)]
        return grads, shapes, has_grads, has_any_grad

    @staticmethod
    def _unflatten_grad(flat_grad, shapes):
        gradients, index = [], 0
        for shape in shapes:
            # np.prod(Size([])) is floating-point for scalar parameters.
            length = int(np.prod(shape))
            gradients.append(flat_grad[index:index + length].view(shape).clone())
            index += length
        return gradients

    @staticmethod
    def _flatten_grad(grads):
        return torch.cat([gradient.flatten() for gradient in grads])

    def _retrieve_grad(self):
        grads, shapes, has_grads, parameter_has_grad = [], [], [], []
        for group in self._optim.param_groups:
            for parameter in group["params"]:
                active = parameter.grad is not None
                parameter_has_grad.append(active)
                shapes.append(parameter.shape)
                if active:
                    grads.append(parameter.grad.detach().clone())
                    has_grads.append(torch.ones_like(parameter))
                else:
                    grads.append(torch.zeros_like(parameter))
                    has_grads.append(torch.zeros_like(parameter))
        return grads, shapes, has_grads, parameter_has_grad
