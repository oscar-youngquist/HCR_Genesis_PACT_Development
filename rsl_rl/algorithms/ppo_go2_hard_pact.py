"""HardPACT physics objectives, reliability tracking, and corrected PCGrad."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from .pc_grad import PCGrad


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


@dataclass
class PhysicsLosses:
    total: torch.Tensor
    inverse: torch.Tensor
    rollout: torch.Tensor
    projection: torch.Tensor
    metrics: Dict[str, torch.Tensor]


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
        use_adaptive_entropy=True,
        adaptive_ent_bounds=(0.01, 0.001),
        adaptive_ent_lin_threshold=0.75,
        adaptive_ent_ang_threshold=0.35,
        adaptive_ent_ter_threshold=5.0,
        adaptive_ent_softmax_temp=2.0,
        max_grad_norm=1.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        reliability_ema_alpha=0.05,
        adaptation_learning_rate=1.0e-5,
        supervised_physics_head_pretraining=True,
        use_bard_inverse_loss=True,
        use_bard_rollout_loss=True,
        use_qp=True,
        differentiate_qp=True,
        stop_gradient_qp=False,
        use_soft_projection_penalty=False,
        lambda_inverse=0.01,
        lambda_rollout=0.01,
        lambda_projection=0.001,
        grf_loss_weight=1.0,
        active_wrench_loss_weight=1.0,
        neutral_wrench_loss_weight=0.25,
        feedforward_clone_weight=1.0,
        reconstruction_loss_weight=1.0,
        explicit_loss_weight=1.0,
        kl_loss_weight=1.0e-3,
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
        self.use_adaptive_entropy = bool(use_adaptive_entropy)
        self.entropy_coef_bounds = tuple(float(value) for value in adaptive_ent_bounds)
        if len(self.entropy_coef_bounds) != 2:
            raise ValueError("adaptive_ent_bounds must contain exactly two values")
        self.ent_linvelo_threshold = float(adaptive_ent_lin_threshold)
        self.ent_angvelo_threshold = float(adaptive_ent_ang_threshold)
        self.ent_terrain_threshold = float(adaptive_ent_ter_threshold)
        self.ent_softmax_temperature = float(adaptive_ent_softmax_temp)
        if self.ent_softmax_temperature <= 0.0:
            raise ValueError("adaptive_ent_softmax_temp must be positive")
        self.current_entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.desired_kl = desired_kl
        self.schedule = str(schedule)
        self.use_clipped_value_loss = bool(use_clipped_value_loss)
        self.reliability = ReliabilityEMA(reliability_ema_alpha)
        self.supervised_physics_head_pretraining = bool(
            supervised_physics_head_pretraining
        )
        self.use_bard_inverse_loss = bool(use_bard_inverse_loss)
        self.use_bard_rollout_loss = bool(use_bard_rollout_loss)
        self.use_qp = bool(use_qp)
        self.differentiate_qp = bool(differentiate_qp)
        self.stop_gradient_qp = bool(stop_gradient_qp)
        self.use_soft_projection_penalty = bool(use_soft_projection_penalty)
        self.lambda_inverse = float(lambda_inverse)
        self.lambda_rollout = float(lambda_rollout)
        self.lambda_projection = float(lambda_projection)
        self.grf_loss_weight = float(grf_loss_weight)
        self.active_wrench_loss_weight = float(active_wrench_loss_weight)
        self.neutral_wrench_loss_weight = float(neutral_wrench_loss_weight)
        self.feedforward_clone_weight = float(feedforward_clone_weight)
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.explicit_loss_weight = float(explicit_loss_weight)
        self.kl_loss_weight = float(kl_loss_weight)
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

    def set_entropy_coef(self, coef=1.0e-3):
        if self.use_adaptive_entropy:
            self.current_entropy_coef = float(coef)
        else:
            self.entropy_coef = float(coef)

    def update_adaptive_entropy_coef(self, performance_metrics):
        """Match Go2 PACT's performance-conditioned entropy schedule."""
        lin_vel_tracking = float(performance_metrics.get("lin_vel_tracking", 0.0))
        ang_vel_tracking = float(performance_metrics.get("ang_vel_tracking", 0.0))
        terrain_level = float(performance_metrics.get("terrain_level", 0.0))

        lin_vel_gap = max(0.0, self.ent_linvelo_threshold - lin_vel_tracking)
        ang_vel_gap = max(0.0, self.ent_angvelo_threshold - ang_vel_tracking)
        terrain_gap = max(0.0, self.ent_terrain_threshold - terrain_level)
        normalized_gaps = torch.tensor(
            [
                lin_vel_gap / self.ent_linvelo_threshold
                if self.ent_linvelo_threshold > 0.0 else 0.0,
                ang_vel_gap / self.ent_angvelo_threshold
                if self.ent_angvelo_threshold > 0.0 else 0.0,
                terrain_gap / self.ent_terrain_threshold
                if self.ent_terrain_threshold > 0.0 else 0.0,
            ],
            dtype=torch.float32,
        )
        weights = F.softmax(
            normalized_gaps / self.ent_softmax_temperature, dim=0
        )
        weighted_gap = torch.sum(weights * normalized_gaps).item()
        lower, upper = self.entropy_coef_bounds
        self.current_entropy_coef = lower + weighted_gap * (upper - lower)
        return self.current_entropy_coef

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
        entropy_coefficient = (
            self.current_entropy_coef
            if self.use_adaptive_entropy else self.entropy_coef
        )
        total = (
            surrogate + self.value_loss_coef * value_loss
            - entropy_coefficient * entropy.mean()
        )
        return total, surrogate, value_loss

    @staticmethod
    def _normalized_huber(prediction, target, scale, mask=None, delta=1.0):
        scale = torch.as_tensor(
            scale, device=prediction.device, dtype=prediction.dtype
        )
        error = (prediction - target) / scale.clamp_min(1.0e-8)
        loss = F.huber_loss(
            error, torch.zeros_like(error), delta=delta, reduction="none"
        ).mean(dim=-1)
        if mask is None:
            return loss.mean()
        mask = mask.reshape(-1).to(loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def _supervised_physics_losses(self, batch, references):
        wrench_active = batch["sustained_wrench_active_mask"].bool().reshape(-1)
        return {
            "grf": self._normalized_huber(
                references.grf_yaw_n,
                batch["interval_grf_yaw"],
                references.grf_yaw_n.new_tensor([120.0, 120.0, 250.0] * 4),
            ),
            "wrench_active": self._normalized_huber(
                references.base_wrench_yaw,
                batch["interval_wrench_yaw"],
                references.base_wrench_yaw.new_tensor(
                    [60.0, 60.0, 60.0, 12.0, 12.0, 12.0]
                ),
                wrench_active,
            ),
            "wrench_neutral": self._normalized_huber(
                references.base_wrench_yaw,
                batch["interval_wrench_yaw"],
                references.base_wrench_yaw.new_tensor(
                    [60.0, 60.0, 60.0, 12.0, 12.0, 12.0]
                ),
                ~wrench_active,
            ),
        }

    @staticmethod
    def _inverse_dynamics_loss(batch, outputs):
        dynamics = outputs["dynamics"]
        dt = outputs["dt"]
        parameters = outputs["dynamics_parameters"]
        observed_acceleration = (
            (batch["post_v"] - batch["pre_v"]) / dt
        ).detach()
        required = dynamics.rnea(
            batch["pre_q"].detach(), batch["pre_v"].detach(),
            observed_acceleration, parameters=parameters,
        ).detach()
        terms = dynamics.terms(
            batch["pre_q"].detach(), batch["pre_v"].detach(),
            parameters=parameters,
        )
        grf_force = torch.einsum(
            "bfkn,bfk->bn", terms.foot_jacobians.detach(),
            outputs["grf_world"].reshape(-1, 4, 3),
        )
        wrench_force = torch.einsum(
            "bkn,bk->bn", terms.base_jacobian.detach(), outputs["wrench_world"]
        )
        actuator = generalized_actuator_force(batch["average_torque"].detach())
        residual = required - actuator - grf_force - wrench_force
        denominator = (
            actuator.norm(dim=-1)
            + grf_force.norm(dim=-1)
            + wrench_force.norm(dim=-1)
        ).detach().clamp_min(1.0e-8)
        per_sample = residual.norm(dim=-1) / denominator
        mask = batch["physics_valid_mask"].reshape(-1).to(per_sample.dtype)
        loss = (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, {
            "base_linear": residual[:, :3].abs().mean(),
            "base_angular": residual[:, 3:6].abs().mean(),
            "joint": residual[:, 6:].abs().mean(),
        }

    @staticmethod
    def _rollout_loss(batch, outputs, safe_torque):
        dynamics = outputs["dynamics"]
        parameters = outputs["dynamics_parameters"]
        terms = dynamics.terms(
            batch["pre_q"].detach(), batch["pre_v"].detach(),
            parameters=parameters,
        )
        foot_force = torch.einsum(
            "bfkn,bfk->bn", terms.foot_jacobians.detach(),
            outputs["grf_world"].reshape(-1, 4, 3),
        )
        wrench_force = torch.einsum(
            "bkn,bk->bn", terms.base_jacobian.detach(), outputs["wrench_world"]
        )
        generalized = (
            generalized_actuator_force(safe_torque) + foot_force + wrench_force
        )
        acceleration = dynamics.aba(
            batch["pre_q"].detach(), batch["pre_v"].detach(), generalized,
            parameters=parameters,
        )
        prediction = batch["pre_v"].detach() + outputs["dt"] * acceleration
        scales = prediction.new_tensor([1.0] * 3 + [2.0] * 3 + [10.0] * 12)
        squared = ((prediction - batch["post_v"].detach()) / scales).square()
        mask = batch["physics_valid_mask"].reshape(-1, 1).to(squared.dtype)
        samples = mask.sum().clamp_min(1.0)
        loss = (squared.mean(dim=-1, keepdim=True) * mask).sum() / samples
        return loss, {
            "base_linear": (squared[:, :3] * mask).sum() / (3.0 * samples),
            "base_angular": (squared[:, 3:6] * mask).sum() / (3.0 * samples),
            "joint": (squared[:, 6:] * mask).sum() / (12.0 * samples),
            "base_linear_mae_physical": (
                (prediction[:, :3] - batch["post_v"][:, :3].detach()).abs()
                * mask
            ).sum() / (3.0 * samples),
            "base_angular_mae_physical": (
                (prediction[:, 3:6] - batch["post_v"][:, 3:6].detach()).abs()
                * mask
            ).sum() / (3.0 * samples),
            "joint_mae_physical": (
                (prediction[:, 6:] - batch["post_v"][:, 6:].detach()).abs()
                * mask
            ).sum() / (12.0 * samples),
        }

    def _compute_physics_objective(self, batch, outputs):
        references = outputs["references"]
        qp_result = outputs["qp_result"]
        nominal = outputs["nominal_torque"]
        zero = (
            references.grf_yaw_n.sum() + references.base_wrench_yaw.sum()
        ) * 0.0
        safe_torque = qp_result.safe_torque
        if not self.differentiate_qp or self.stop_gradient_qp:
            safe_torque = safe_torque.detach()

        inverse, rollout = zero, zero
        metrics = {}
        if outputs["dynamics"] is not None and self.use_bard_inverse_loss:
            inverse, inverse_metrics = self._inverse_dynamics_loss(batch, outputs)
            metrics.update(
                {f"inverse/{name}": value for name, value in inverse_metrics.items()}
            )
        if outputs["dynamics"] is not None and self.use_bard_rollout_loss:
            rollout, rollout_metrics = self._rollout_loss(
                batch, outputs, safe_torque
            )
            metrics.update(
                {f"rollout/{name}": value for name, value in rollout_metrics.items()}
            )

        if not self.differentiate_qp or self.stop_gradient_qp:
            correction = qp_result.safe_torque.detach() - nominal
        else:
            correction = qp_result.correction
        projection = correction.square().mean()
        if not self.use_qp and not self.use_soft_projection_penalty:
            projection = zero
        physics = PhysicsLosses(
            total=(
                self.lambda_inverse * inverse
                + self.lambda_rollout * rollout
                + self.lambda_projection * projection
            ),
            inverse=inverse,
            rollout=rollout,
            projection=projection,
            metrics=metrics,
        )

        clone = zero
        if self.actor_critic.position_pretraining:
            clone = F.smooth_l1_loss(
                outputs["feedforward_prediction"], outputs["feedforward_target"]
            )
        actor_auxiliary = physics.total + self.feedforward_clone_weight * clone

        grf_error = (
            references.grf_yaw_n - batch["interval_grf_yaw"]
        ).abs().reshape(-1, 4, 3)
        wrench_error = (
            references.base_wrench_yaw - batch["interval_wrench_yaw"]
        ).abs()
        metrics.update({
            "clone": clone,
            "grf_mae_n": grf_error.mean(),
            "wrench_mae": wrench_error.mean(),
            "qp/correction_l2_nm": qp_result.correction.norm(dim=-1).mean(),
            "qp/correction_max_nm": qp_result.correction.abs().amax(dim=-1).mean(),
            "qp/contact_slack_l2": qp_result.contact_slack.norm(dim=-1).mean(),
            "qp/contact_slack_max": qp_result.contact_slack.abs().amax(dim=-1).mean(),
            "qp/equality_residual_max": torch.nan_to_num(
                qp_result.equality_residual, nan=0.0
            ).mean(),
            "qp/inequality_violation_max": torch.nan_to_num(
                qp_result.inequality_violation, nan=0.0
            ).mean(),
            "qp/minimum_margin": torch.nan_to_num(
                qp_result.minimum_margin, nan=0.0
            ).mean(),
            "qp/active_constraints": qp_result.active_constraints.float().mean(),
            "qp/fallback": (qp_result.fallback > 0).float().mean(),
            "qp/actuator_projection_fallback": (
                qp_result.fallback == 2
            ).float().mean(),
            "qp/infeasible": (qp_result.status > 0).float().mean(),
            "qp/forward_time_ms": qp_result.forward_time_ms.mean(),
            "transition/physics_valid_fraction": batch["physics_valid_mask"].float().mean(),
            "transition/push_fraction": batch["instantaneous_push_mask"].float().mean(),
            "transition/wrench_active_fraction": batch["sustained_wrench_active_mask"].float().mean(),
            "transition/reset_fraction": batch["reset_mask"].float().mean(),
            "transition/timeout_fraction": batch["timeout_mask"].float().mean(),
            "transition/teleport_fraction": batch["teleport_mask"].float().mean(),
        })
        for foot_index, foot_name in enumerate(("FR", "FL", "RR", "RL")):
            for axis_index, axis in enumerate(("x", "y", "z")):
                metrics[f"grf/{foot_name}_{axis}_mae_n"] = grf_error[
                    :, foot_index, axis_index
                ].mean()
        for axis_index, axis in enumerate(("fx", "fy", "fz", "tx", "ty", "tz")):
            units = "n" if axis_index < 3 else "nm"
            metrics[f"wrench/{axis}_mae_{units}"] = wrench_error[:, axis_index].mean()
        intervention = qp_result.correction.norm(dim=-1) > 1.0e-5
        metrics["qp/intervention_fraction"] = intervention.float().mean()
        for label, mask in (("intervened", intervention), ("not_intervened", ~intervention)):
            denominator = mask.float().sum().clamp_min(1.0)
            metrics[f"qp_conditioned/{label}_grf_mae_n"] = (
                grf_error.mean(dim=(1, 2)) * mask.float()
            ).sum() / denominator
            metrics[f"qp_conditioned/{label}_wrench_mae"] = (
                wrench_error.mean(dim=-1) * mask.float()
            ).sum() / denominator
        for status_code in range(3):
            metrics[f"qp/status_{status_code}_fraction"] = (
                qp_result.status == status_code
            ).float().mean()
        return {
            "physics": physics,
            "actor_auxiliary": actor_auxiliary,
            "metrics": metrics,
        }

    def _compute_auxiliary_objective(self, batch, outputs):
        references = outputs["references"]
        reconstruction = outputs["reconstruction"]
        encoded = outputs["encoded"]
        supervised = self._supervised_physics_losses(batch, references)
        reconstruction_loss = F.smooth_l1_loss(
            reconstruction, batch["reconstruction_target"]
        )
        explicit_loss = F.smooth_l1_loss(
            encoded.explicit, batch["explicit_estimator_target"]
        )
        kl = -0.5 * torch.mean(
            1.0 + encoded.latent_log_variance
            - encoded.latent_mean.square()
            - encoded.latent_log_variance.exp()
        )
        loss = (
            self.reconstruction_loss_weight * reconstruction_loss
            + self.explicit_loss_weight * explicit_loss
            + self.kl_loss_weight * kl
        )
        if self.supervised_physics_head_pretraining:
            loss = (
                loss
                + self.grf_loss_weight * supervised["grf"]
                + self.active_wrench_loss_weight * supervised["wrench_active"]
                + self.neutral_wrench_loss_weight * supervised["wrench_neutral"]
            )

        schema = outputs["reconstruction_schema"]
        physical = schema.unpack(batch["reconstruction_target"], normalized=True)
        reconstructed_physical = schema.unpack(reconstruction, normalized=True)
        metrics = {
            "supervised/grf": supervised["grf"],
            "supervised/wrench_active": supervised["wrench_active"],
            "supervised/wrench_neutral": supervised["wrench_neutral"],
            "reconstruction": reconstruction_loss,
            "explicit": explicit_loss,
            "kl": kl,
        }
        for field in schema.fields:
            field_slice = schema.slices[field.name]
            metrics[f"reconstruction/{field.name}_huber_normalized"] = (
                F.smooth_l1_loss(
                    reconstruction[:, field_slice],
                    batch["reconstruction_target"][:, field_slice],
                )
            )
            metrics[f"reconstruction/{field.name}_mae_physical"] = (
                reconstructed_physical[field.name] - physical[field.name]
            ).abs().mean()
        return {"loss": loss, "metrics": metrics}

    def update(self, recompute_outputs, recompute_auxiliary_outputs, iteration):
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
            outputs = recompute_outputs(batch, self.actor_critic)
            recomputed = self._compute_physics_objective(batch, outputs)
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
            auxiliary_outputs = recompute_auxiliary_outputs(
                batch, self.actor_critic
            )
            auxiliary = self._compute_auxiliary_objective(
                batch, auxiliary_outputs
            )
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
                "policy/entropy_coefficient": (
                    self.current_entropy_coef
                    if self.use_adaptive_entropy else self.entropy_coef
                ),
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
