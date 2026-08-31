"""HardPACT physics objectives, reliability tracking, and corrected PCGrad."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


def normalized_huber(prediction, target, scale, mask=None, delta=1.0):
    scale = torch.as_tensor(scale, device=prediction.device, dtype=prediction.dtype)
    error = (prediction - target) / scale.clamp_min(1.0e-8)
    loss = F.huber_loss(error, torch.zeros_like(error), delta=delta, reduction="none").mean(dim=-1)
    if mask is None:
        return loss.mean()
    mask = mask.reshape(-1).to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def supervised_physics_head_losses(
    predicted_grf,
    target_grf,
    predicted_wrench,
    target_wrench,
    wrench_active_mask,
):
    """Train every foot/transition and split active/neutral wrench events."""
    grf_loss = normalized_huber(
        predicted_grf, target_grf, predicted_grf.new_tensor([120.0, 120.0, 250.0] * 4)
    )
    wrench_scale = predicted_wrench.new_tensor([60.0, 60.0, 60.0, 12.0, 12.0, 12.0])
    active = wrench_active_mask.bool().reshape(-1)
    active_loss = normalized_huber(
        predicted_wrench, target_wrench, wrench_scale, active
    )
    neutral_loss = normalized_huber(
        predicted_wrench, target_wrench, wrench_scale, ~active
    )
    return {
        "grf": grf_loss,
        "wrench_active": active_loss,
        "wrench_neutral": neutral_loss,
    }


class ReliabilityEMA:
    """One update per PPO iteration; repeated minibatches cannot bias it."""

    def __init__(self, alpha=0.05):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EMA alpha must lie in (0,1]")
        self.alpha = float(alpha)
        self.values: Dict[str, float] = {}
        self.last_iteration = -1

    def update(self, iteration: int, metrics: Mapping[str, float]):
        iteration = int(iteration)
        if iteration == self.last_iteration:
            return False
        if iteration < self.last_iteration:
            raise ValueError("PPO iterations must be monotonic")
        for name, sample in metrics.items():
            sample = float(sample)
            old = self.values.get(name, sample)
            self.values[name] = (1.0 - self.alpha) * old + self.alpha * sample
        self.last_iteration = iteration
        return True


def generalized_actuator_force(torque: torch.Tensor) -> torch.Tensor:
    return torch.cat((torque.new_zeros(torque.shape[0], 6), torque), dim=-1)


def inverse_dynamics_loss(
    dynamics,
    pre_q,
    pre_v,
    post_v,
    executed_torque,
    predicted_grf,
    predicted_wrench,
    valid_mask,
    dt,
    *,
    parameters=None,
):
    observed_acceleration = ((post_v - pre_v) / float(dt)).detach()
    required = dynamics.rnea(
        pre_q.detach(), pre_v.detach(), observed_acceleration, parameters=parameters
    ).detach()
    terms = dynamics.terms(pre_q.detach(), pre_v.detach(), parameters=parameters)
    foot_j = terms.foot_jacobians.detach()
    base_j = terms.base_jacobian.detach()
    grf_force = torch.einsum(
        "bfkn,bfk->bn", foot_j, predicted_grf.reshape(-1, 4, 3)
    )
    wrench_force = torch.einsum("bkn,bk->bn", base_j, predicted_wrench)
    actuator = generalized_actuator_force(executed_torque.detach())
    residual = required - actuator - grf_force - wrench_force
    denominator = (
        actuator.norm(dim=-1) + grf_force.norm(dim=-1) + wrench_force.norm(dim=-1)
    ).detach().clamp_min(1.0e-8)
    per_sample = residual.norm(dim=-1) / denominator
    mask = valid_mask.reshape(-1).to(per_sample.dtype)
    loss = (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, {
        "base_linear": residual[:, :3].abs().mean(),
        "base_angular": residual[:, 3:6].abs().mean(),
        "joint": residual[:, 6:].abs().mean(),
    }


def rollout_loss(
    dynamics,
    pre_q,
    pre_v,
    post_v,
    safe_torque,
    predicted_grf,
    predicted_wrench,
    valid_mask,
    dt,
    *,
    parameters=None,
):
    terms = dynamics.terms(pre_q.detach(), pre_v.detach(), parameters=parameters)
    foot_force = torch.einsum(
        "bfkn,bfk->bn", terms.foot_jacobians.detach(), predicted_grf.reshape(-1, 4, 3)
    )
    wrench_force = torch.einsum(
        "bkn,bk->bn", terms.base_jacobian.detach(), predicted_wrench
    )
    generalized = generalized_actuator_force(safe_torque) + foot_force + wrench_force
    acceleration = dynamics.aba(
        pre_q.detach(), pre_v.detach(), generalized, parameters=parameters
    )
    prediction = pre_v.detach() + float(dt) * acceleration
    scales = prediction.new_tensor([1.0] * 3 + [2.0] * 3 + [10.0] * 12)
    squared = ((prediction - post_v.detach()) / scales).square()
    mask = valid_mask.reshape(-1, 1).to(squared.dtype)
    loss = (squared.mean(dim=-1, keepdim=True) * mask).sum() / mask.sum().clamp_min(1.0)
    blocks = {
        "base_linear": (squared[:, :3] * mask).sum() / (3.0 * mask.sum().clamp_min(1.0)),
        "base_angular": (squared[:, 3:6] * mask).sum() / (3.0 * mask.sum().clamp_min(1.0)),
        "joint": (squared[:, 6:] * mask).sum() / (12.0 * mask.sum().clamp_min(1.0)),
        "base_linear_mae_physical": (
            (prediction[:, :3] - post_v[:, :3].detach()).abs() * mask
        ).sum() / (3.0 * mask.sum().clamp_min(1.0)),
        "base_angular_mae_physical": (
            (prediction[:, 3:6] - post_v[:, 3:6].detach()).abs() * mask
        ).sum() / (3.0 * mask.sum().clamp_min(1.0)),
        "joint_mae_physical": (
            (prediction[:, 6:] - post_v[:, 6:].detach()).abs() * mask
        ).sum() / (12.0 * mask.sum().clamp_min(1.0)),
    }
    return loss, blocks


@dataclass
class PhysicsLosses:
    total: torch.Tensor
    inverse: torch.Tensor
    rollout: torch.Tensor
    projection: torch.Tensor
    metrics: Dict[str, torch.Tensor]


def combine_physics_losses(inverse, rollout, projection, lambda_inverse, lambda_rollout, lambda_projection, metrics=None):
    return PhysicsLosses(
        total=lambda_inverse * inverse + lambda_rollout * rollout + lambda_projection * projection,
        inverse=inverse,
        rollout=rollout,
        projection=projection,
        metrics={} if metrics is None else metrics,
    )


@dataclass
class PCGradDiagnostics:
    primary_norm: float
    auxiliary_norm: float
    cosine: float
    conflict: bool
    zero_fraction: float
    nonfinite_fraction: float
    module_metrics: Dict[str, float]


def pcgrad_backward_two_objectives(
    primary_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    parameter_names: Sequence[str] | None = None,
):
    """Backpropagate ``[L_PPO,L_physics]`` and project only negative dots."""
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if parameter_names is None:
        parameter_names = ["all"] * len(parameters)
    else:
        parameter_names = list(parameter_names)
        if len(parameter_names) != len(parameters):
            raise ValueError("parameter_names must align with trainable parameters")
    primary = torch.autograd.grad(
        primary_loss, parameters, retain_graph=True, allow_unused=True
    )
    auxiliary = torch.autograd.grad(
        auxiliary_loss, parameters, retain_graph=True, allow_unused=True
    )
    primary_flat = torch.cat([
        torch.zeros_like(parameter).flatten() if grad is None else grad.flatten()
        for parameter, grad in zip(parameters, primary)
    ])
    auxiliary_flat = torch.cat([
        torch.zeros_like(parameter).flatten() if grad is None else grad.flatten()
        for parameter, grad in zip(parameters, auxiliary)
    ])
    finite = torch.isfinite(primary_flat) & torch.isfinite(auxiliary_flat)
    p = torch.where(finite, primary_flat, torch.zeros_like(primary_flat))
    a = torch.where(finite, auxiliary_flat, torch.zeros_like(auxiliary_flat))
    unprojected_a = a.clone()
    dot = torch.dot(p, a)
    conflict = bool(dot.detach().item() < 0.0)
    if conflict:
        a = a - dot * p / p.square().sum().clamp_min(1.0e-12)
    merged = p + a
    index = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = merged[index:index + count].view_as(parameter).clone()
        index += count
    denom = p.norm() * unprojected_a.norm()
    cosine = 0.0 if denom.item() == 0.0 else float((dot / denom).detach().item())
    module_metrics: Dict[str, float] = {}
    groups: Dict[str, list[int]] = {}
    for index, name in enumerate(parameter_names):
        groups.setdefault(name.split(".", 1)[0], []).append(index)
    offsets = [0]
    for parameter in parameters:
        offsets.append(offsets[-1] + parameter.numel())
    for module, indices in groups.items():
        slices = [slice(offsets[index], offsets[index + 1]) for index in indices]
        module_p = torch.cat([p[item] for item in slices])
        module_a = torch.cat([unprojected_a[item] for item in slices])
        module_merged = torch.cat([merged[item] for item in slices])
        module_finite = torch.cat([finite[item] for item in slices])
        module_dot = torch.dot(module_p, module_a)
        module_denom = module_p.norm() * module_a.norm()
        prefix = f"gradient/module/{module}"
        module_metrics[f"{prefix}/ppo_norm"] = float(module_p.norm().detach().item())
        module_metrics[f"{prefix}/physics_norm"] = float(
            module_a.norm().detach().item()
        )
        module_metrics[f"{prefix}/cosine"] = (
            0.0 if module_denom.item() == 0.0
            else float((module_dot / module_denom).detach().item())
        )
        module_metrics[f"{prefix}/conflict"] = float(module_dot.detach().item() < 0.0)
        module_metrics[f"{prefix}/zero_fraction"] = float(
            (module_merged == 0).float().mean().detach().item()
        )
        module_metrics[f"{prefix}/nonfinite_fraction"] = float(
            (~module_finite).float().mean().detach().item()
        )
    return PCGradDiagnostics(
        primary_norm=float(p.norm().detach().item()),
        auxiliary_norm=float(unprojected_a.norm().detach().item()),
        cosine=cosine,
        conflict=conflict,
        zero_fraction=float((merged == 0).float().mean().detach().item()),
        nonfinite_fraction=float((~finite).float().mean().detach().item()),
        module_metrics=module_metrics,
    )


def delayed_action_for_update(raw_sampled_action, stored_delayed_action, delay_steps):
    """Use sampled actions for zero-delay QPs and the executed replay otherwise."""
    no_delay = delay_steps.reshape(-1, 1) == 0
    return torch.where(no_delay, raw_sampled_action, stored_delayed_action.detach())


class PPOGo2HardPACT:
    """PPO with one optimizer step and two-objective corrected PCGrad."""

    def __init__(
        self,
        actor_critic,
        learning_rate=3.0e-4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        reliability_ema_alpha=0.05,
        device="cpu",
        **unused,
    ):
        self.actor_critic = actor_critic.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(actor_critic.parameters(), lr=learning_rate)
        self.clip_param = float(clip_param)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.reliability = ReliabilityEMA(reliability_ema_alpha)
        self.storage = None
        self._pending_core = None

    def init_storage(self, num_envs, num_steps, observation_dim, critic_dim, history_dim):
        from rsl_rl.storage import RolloutStorageGo2HardPACT

        self.storage = RolloutStorageGo2HardPACT(
            num_envs,
            num_steps,
            observation_dim,
            critic_dim,
            history_dim,
            position_pretraining=self.actor_critic.position_pretraining,
            device=self.device,
        )

    def act(self, observation, critic_observation, history):
        with torch.no_grad():
            action = self.actor_critic.act(observation, history)
            value = self.actor_critic.evaluate(critic_observation)
            log_probability = self.actor_critic.get_actions_log_prob(action)
            mean = self.actor_critic.action_mean
            std = self.actor_critic.action_std
        self._pending_core = {
            "observation": observation,
            "critic_observation": critic_observation,
            "history": history,
            "value": value,
            "raw_action_log_probability": log_probability,
            "action_mean": mean,
            "action_std": std,
        }
        return action, action - mean

    def process_env_step(self, reward, done, transition):
        if self._pending_core is None:
            raise RuntimeError("act must be called before process_env_step")
        core = dict(self._pending_core)
        core["reward"] = reward
        core["done"] = done
        self.storage.add(core, transition)
        self._pending_core = None

    def compute_returns(self, last_critic_observation):
        with torch.no_grad():
            last_value = self.actor_critic.evaluate(last_critic_observation)
        self.storage.compute_returns(last_value, self.gamma, self.lam)

    def update(self, recompute_objectives, iteration):
        totals: Dict[str, float] = {}
        updates = 0
        named_parameters = [
            (name, parameter)
            for name, parameter in self.actor_critic.named_parameters()
            if parameter.requires_grad
        ]
        parameter_names = [name for name, _ in named_parameters]
        parameters = [parameter for _, parameter in named_parameters]
        for batch in self.storage.mini_batches(
            self.num_mini_batches, self.num_learning_epochs
        ):
            log_probability, entropy = self.actor_critic.evaluate_actions(
                batch["observation"], batch["history"], batch["raw_action"]
            )
            value = self.actor_critic.evaluate(batch["critic_observation"])
            ratio = torch.exp(
                log_probability - batch["raw_action_log_probability"].squeeze(-1)
            )
            advantage = batch["advantage"].squeeze(-1)
            surrogate = torch.maximum(
                -advantage * ratio,
                -advantage * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param),
            ).mean()
            value_loss = (value - batch["return"]).square().mean()
            ppo_loss = (
                surrogate + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )
            recompute_wall_start = perf_counter()
            recompute_start = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            recompute_end = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            if recompute_start is not None:
                recompute_start.record()
            recomputed = recompute_objectives(
                batch, batch["raw_action"], self.actor_critic
            )
            if recompute_end is not None:
                recompute_end.record()
            physics_loss = recomputed["physics"].total
            auxiliary_loss = recomputed["auxiliary"]
            self.optimizer.zero_grad(set_to_none=True)
            backward_wall_start = perf_counter()
            backward_start = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            backward_end = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            if backward_start is not None:
                backward_start.record()
            diagnostics = pcgrad_backward_two_objectives(
                ppo_loss, physics_loss, parameters, parameter_names
            )
            auxiliary_grads = torch.autograd.grad(
                auxiliary_loss, parameters, allow_unused=True
            )
            for parameter, gradient in zip(parameters, auxiliary_grads):
                if gradient is not None:
                    if parameter.grad is None:
                        parameter.grad = gradient
                    else:
                        parameter.grad.add_(gradient)
            if backward_end is not None:
                backward_end.record()
            backward_wall_ms = (perf_counter() - backward_wall_start) * 1000.0
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            self.optimizer.step()

            metrics = {
                "loss/ppo": ppo_loss,
                "loss/surrogate": surrogate,
                "loss/value": value_loss,
                "loss/physics": physics_loss,
                "loss/inverse": recomputed["physics"].inverse,
                "loss/rollout": recomputed["physics"].rollout,
                "loss/projection": recomputed["physics"].projection,
                "loss/auxiliary": auxiliary_loss,
                "gradient/ppo_norm": diagnostics.primary_norm,
                "gradient/physics_norm": diagnostics.auxiliary_norm,
                "gradient/cosine": diagnostics.cosine,
                "gradient/conflict": float(diagnostics.conflict),
                "gradient/zero_fraction": diagnostics.zero_fraction,
                "gradient/nonfinite_fraction": diagnostics.nonfinite_fraction,
            }
            metrics.update(diagnostics.module_metrics)
            if recompute_end is not None:
                recompute_end.synchronize()
                metrics["qp/recompute_forward_ms"] = recompute_start.elapsed_time(recompute_end)
                backward_end.synchronize()
                metrics["qp/recompute_backward_ms"] = backward_start.elapsed_time(backward_end)
                metrics["qp/gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(self.device) / (1024.0 ** 2)
                metrics["qp/gpu_memory_peak_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
            else:
                metrics["qp/recompute_forward_ms"] = (
                    backward_wall_start - recompute_wall_start
                ) * 1000.0
                metrics["qp/recompute_backward_ms"] = backward_wall_ms
            metrics.update(recomputed["metrics"])
            for name, value in metrics.items():
                scalar = float(value.detach().item()) if torch.is_tensor(value) else float(value)
                totals[name] = totals.get(name, 0.0) + scalar
            updates += 1
        means = {name: value / max(updates, 1) for name, value in totals.items()}
        reliability_metrics = {
            name: value
            for name, value in means.items()
            if name in {
                "grf_mae_n", "wrench_mae", "qp/fallback",
                "supervised/grf", "supervised/wrench_active",
                "supervised/wrench_neutral",
            }
        }
        self.reliability.update(iteration, reliability_metrics)
        means.update({f"reliability/{name}": value for name, value in self.reliability.values.items()})
        self.storage.clear()
        return means

    def train_mode(self):
        self.actor_critic.train()

    def test_mode(self):
        self.actor_critic.eval()
