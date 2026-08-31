"""HardPACT physics objectives, reliability tracking, and corrected PCGrad."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from .pc_grad import PCGrad


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


def _diagnostics_from_pcgrad(pcgrad, parameters, parameter_names):
    primary, auxiliary = pcgrad.last_objective_grads
    merged = pcgrad.last_merged_grad
    finite = torch.isfinite(primary) & torch.isfinite(auxiliary)
    primary = torch.where(finite, primary, torch.zeros_like(primary))
    auxiliary = torch.where(finite, auxiliary, torch.zeros_like(auxiliary))
    dot = torch.dot(primary, auxiliary)
    denominator = primary.norm() * auxiliary.norm()
    module_metrics: Dict[str, float] = {}
    groups: Dict[str, list[int]] = {}
    for index, name in enumerate(parameter_names):
        groups.setdefault(name.split(".", 1)[0], []).append(index)
    offsets = [0]
    for parameter in parameters:
        offsets.append(offsets[-1] + parameter.numel())
    for module, indices in groups.items():
        slices = [slice(offsets[index], offsets[index + 1]) for index in indices]
        module_primary = torch.cat([primary[item] for item in slices])
        module_auxiliary = torch.cat([auxiliary[item] for item in slices])
        module_merged = torch.cat([merged[item] for item in slices])
        module_finite = torch.cat([finite[item] for item in slices])
        module_dot = torch.dot(module_primary, module_auxiliary)
        module_denominator = module_primary.norm() * module_auxiliary.norm()
        prefix = f"gradient/module/{module}"
        module_metrics[f"{prefix}/ppo_norm"] = float(module_primary.norm().item())
        module_metrics[f"{prefix}/physics_norm"] = float(
            module_auxiliary.norm().item()
        )
        module_metrics[f"{prefix}/cosine"] = (
            0.0 if module_denominator.item() == 0.0
            else float((module_dot / module_denominator).item())
        )
        module_metrics[f"{prefix}/conflict"] = float(module_dot.item() < 0.0)
        module_metrics[f"{prefix}/zero_fraction"] = float(
            (module_merged == 0).float().mean().item()
        )
        module_metrics[f"{prefix}/nonfinite_fraction"] = float(
            (~module_finite).float().mean().item()
        )
    return PCGradDiagnostics(
        primary_norm=float(primary.norm().item()),
        auxiliary_norm=float(auxiliary.norm().item()),
        cosine=(
            0.0 if denominator.item() == 0.0
            else float((dot / denominator).item())
        ),
        conflict=bool(dot.item() < 0.0),
        zero_fraction=float((merged == 0).float().mean().item()),
        nonfinite_fraction=float((~finite).float().mean().item()),
        module_metrics=module_metrics,
    )


class PPOGo2HardPACT:
    """B1Z1-style PCGrad actor update followed by an auxiliary update."""

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
        adaptation_learning_rate=1.0e-5,
        desired_kl=None,
        schedule="fixed",
        use_clipped_value_loss=True,
        device="cpu",
        **unused,
    ):
        self.actor_critic = actor_critic.to(device)
        self.device = torch.device(device)
        self.learning_rate = float(learning_rate)
        self.actor_optimizer = PCGrad(
            torch.optim.Adam(actor_critic.parameters(), lr=self.learning_rate),
            reduction="sum",
        )
        self.ppo_parameters = list(actor_critic.parameters())
        auxiliary_groups = actor_critic.get_auxiliary_optim_groups()
        self.auxiliary_optimizer = torch.optim.Adam(
            auxiliary_groups, lr=float(adaptation_learning_rate)
        )
        self.auxiliary_parameters = [
            parameter
            for group in self.auxiliary_optimizer.param_groups
            for parameter in group["params"]
        ]
        self.clip_param = float(clip_param)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.desired_kl = desired_kl
        self.schedule = str(schedule)
        self.use_clipped_value_loss = bool(use_clipped_value_loss)
        self.reliability = ReliabilityEMA(reliability_ema_alpha)
        self.storage = None
        from rsl_rl.storage import RolloutStorageGo2HardPACT
        self.transition = RolloutStorageGo2HardPACT.Transition()

    def init_storage(
        self, num_envs, num_steps, observation_dim, critic_dim, history_dim,
    ):
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
        self.transition.observations = observation
        self.transition.critic_observations = critic_observation
        self.transition.observation_history = history
        self.transition.actions = action
        self.transition.values = value
        self.transition.actions_log_prob = log_probability
        self.transition.action_mean = mean
        self.transition.action_sigma = std
        return action

    def process_env_step(self, reward, done, infos, transition):
        if self.transition.actions is None:
            raise RuntimeError("act must be called before process_env_step")
        self.transition.rewards = reward.clone()
        self.transition.dones = done
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        self.transition.hard_pact = transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(done)

    def compute_returns(self, last_critic_observation):
        with torch.no_grad():
            last_value = self.actor_critic.evaluate(last_critic_observation)
        self.storage.compute_returns(last_value, self.gamma, self.lam)

    def _compute_ppo_loss(self, batch):
        """Legacy PACT PPO objective evaluated on the exact raw sample."""
        log_probability, entropy = self.actor_critic.evaluate_actions(
            batch["observation"], batch["history"], batch["raw_action"]
        )
        value = self.actor_critic.evaluate(batch["critic_observation"])
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                mean, std = (
                    self.actor_critic.action_mean,
                    self.actor_critic.action_std,
                )
                old_mean, old_std = batch["action_mean"], batch["action_std"]
                kl = torch.sum(
                    torch.log(std / old_std + 1.0e-5)
                    + (old_std.square() + (old_mean - mean).square())
                    / (2.0 * std.square())
                    - 0.5,
                    dim=-1,
                ).mean()
                if kl > 2.0 * self.desired_kl:
                    self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                elif 0.0 < kl < self.desired_kl / 2.0:
                    self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                for group in self.actor_optimizer.optimizer.param_groups:
                    group["lr"] = self.learning_rate
        ratio = torch.exp(
            log_probability - batch["raw_action_log_probability"].squeeze(-1)
        )
        advantage = batch["advantage"].squeeze(-1)
        surrogate = torch.maximum(
            -advantage * ratio,
            -advantage * ratio.clamp(
                1.0 - self.clip_param, 1.0 + self.clip_param
            ),
        ).mean()
        if self.use_clipped_value_loss:
            old_value = batch["value"]
            clipped = old_value + (value - old_value).clamp(
                -self.clip_param, self.clip_param
            )
            value_loss = torch.maximum(
                (value - batch["return"]).square(),
                (clipped - batch["return"]).square(),
            ).mean()
        else:
            value_loss = (value - batch["return"]).square().mean()
        total = (
            surrogate + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy.mean()
        )
        return total, surrogate, value_loss

    def update(self, recompute_objectives, recompute_auxiliary, iteration):
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
            ppo_loss, surrogate, value_loss = self._compute_ppo_loss(batch)
            recompute_wall_start = perf_counter()
            recompute_start = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            recompute_end = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            if recompute_start is not None:
                recompute_start.record()
            recomputed = recompute_objectives(batch, self.actor_critic)
            if recompute_end is not None:
                recompute_end.record()
            physics_loss = recomputed["physics"].total
            actor_auxiliary_loss = recomputed.get("actor_auxiliary", physics_loss)
            self.actor_optimizer.zero_grad()
            backward_wall_start = perf_counter()
            backward_start = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            backward_end = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            if backward_start is not None:
                backward_start.record()
            self.actor_optimizer.pc_backward_pinn(
                [ppo_loss, actor_auxiliary_loss]
            )
            diagnostics = _diagnostics_from_pcgrad(
                self.actor_optimizer, parameters, parameter_names
            )
            if backward_end is not None:
                backward_end.record()
            backward_wall_ms = (perf_counter() - backward_wall_start) * 1000.0
            torch.nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm)
            self.actor_optimizer.step()

            # Match B1Z1 PACT: rebuild the auxiliary graph after the complete
            # actor update, then step only encoder/decoder/estimators.
            auxiliary = recompute_auxiliary(batch, self.actor_critic)
            auxiliary_loss = auxiliary["loss"]
            self.auxiliary_optimizer.zero_grad(set_to_none=True)
            auxiliary_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.auxiliary_parameters, self.max_grad_norm
            )
            self.auxiliary_optimizer.step()

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
            metrics.update(recomputed["metrics"])
            metrics.update(diagnostics.module_metrics)
            metrics.update(auxiliary.get("metrics", {}))
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
