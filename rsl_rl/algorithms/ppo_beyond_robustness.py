"""Standalone BR learning core, using legacy PPO algebra and TS-style two passes.

No environment, curriculum stage, QP, PCGrad or simulator initialization is owned here.
Collection snapshots t inputs before env.step; labels never come from the next/reset state.
"""
from dataclasses import replace
import math
import torch
from torch import nn

from .ppo import PPO
from rsl_rl.beyond_robustness_config import PHASES
from rsl_rl.utils.adaptive_entropy import performance_gap_entropy
from rsl_rl.modules.hard_pact_action_std import clip_action_std
from rsl_rl.storage.rollout_storage_beyond_robustness import (
    ProprioceptionHistory, RolloutStorageBeyondRobustness,
)


class PPOBeyondRobustness(PPO):
    checkpoint_version = 2

    def __init__(self, actor_critic, config=None, device="cpu"):
        self.config = config or actor_critic.config
        if self.config != actor_critic.config:
            raise ValueError("Model and algorithm must use the same resolved BR config")
        cfg = self.config
        names = ("num_learning_epochs", "num_mini_batches", "clip_param", "gamma", "lam",
                 "value_loss_coef", "entropy_coef", "learning_rate", "max_grad_norm",
                 "use_clipped_value_loss", "schedule", "desired_kl")
        super().__init__(actor_critic, device=device, **{key: getattr(cfg, key) for key in names})
        # One PPO optimizer preserves actor/critic moments across phase changes. The
        # inactive encoder has grad=None and receives no Adam update in that phase.
        self.rl_parameters = [p for name, p in actor_critic.named_parameters()
                              if not name.startswith("load_estimator.")]
        self.optimizer = torch.optim.Adam(self.rl_parameters, lr=cfg.learning_rate)
        self.history_optimizer = torch.optim.Adam(actor_critic.history_encoder.parameters(),
                                                  lr=cfg.auxiliary_learning_rate)
        self.load_optimizer = torch.optim.Adam(actor_critic.load_estimator.parameters(),
                                               lr=cfg.auxiliary_learning_rate)
        self.training_phase = cfg.training_phase
        self.iteration = 0
        self.pending = None
        self.returns_ready = False
        self.init_storage(cfg.num_envs, cfg.num_steps_per_env)

    def init_storage(self, num_envs, num_transitions_per_env):
        self.storage = RolloutStorageBeyondRobustness(num_envs, num_transitions_per_env,
                                                     self.config.history_length, self.device, self.config.input_schema)
        self.history = ProprioceptionHistory(num_envs, self.config.history_length, self.device,
                                            self.config.input_schema.observation_dim)
        self.reset_pending = torch.ones(num_envs, device=self.device, dtype=torch.bool)

    def set_training_phase(self, phase):
        if phase not in PHASES:
            raise ValueError(f"training_phase must be one of {PHASES}")
        if self.pending is not None or self.storage.step:
            raise RuntimeError("Switch learning phase only between completed rollout/update iterations")
        self.actor_critic.zero_grad(set_to_none=True)
        self.training_phase = phase

    def test_mode(self):
        self.actor_critic.eval()

    def apply_phase_schedule(self):
        """Call only at completed iteration boundaries, before the next rollout."""
        start = self.config.student_reinforcement_start_iteration
        if start is not None and self.iteration >= start and self.training_phase == "teacher_student":
            self.set_training_phase("student_reinforcement")

    @torch.no_grad()
    def clear_history(self):
        if self.pending is not None or self.storage.step:
            raise RuntimeError("Clear history only between complete iterations")
        self.history.frames.zero_()
        self.reset_pending.fill_(True)
        self.returns_ready = False

    def update_adaptive_entropy_coef(self, performance_metrics):
        """Missing/nonfinite evidence holds the current value; no curriculum coupling."""
        keys = ("lin_vel_tracking", "ang_vel_tracking", "terrain_level")
        if not self.config.use_adaptive_entropy or not all(
                key in performance_metrics and math.isfinite(performance_metrics[key]) for key in keys):
            return False
        cfg = self.config
        self.entropy_coef = performance_gap_entropy(performance_metrics, cfg.adaptive_ent_bounds,
            (cfg.adaptive_ent_lin_threshold, cfg.adaptive_ent_ang_threshold, cfg.adaptive_ent_ter_threshold),
            cfg.adaptive_ent_softmax_temp)
        return True

    @torch.no_grad()
    def act(self, observations, state, parameters, load_labels, *, reset_mask=None):
        if self.pending is not None:
            raise RuntimeError("process_env_step must follow each act")
        if self.storage.step == self.storage.num_transitions_per_env:
            raise RuntimeError("Update full rollout before collecting more actions")
        n = self.storage.num_envs
        for name, value in dict(observations=observations, state=state, parameters=parameters,
                                load_labels=load_labels).items():
            self.config.input_schema.validate(name, value, n, self.history.frames.device)
        reset = self.reset_pending if reset_mask is None else self.reset_pending | reset_mask.bool()
        history = self.history.append(observations, reset)
        # Own inputs NOW: simulator tensors and history mutate in place before process_env_step.
        batch = {name: value.detach().clone() for name, value in dict(
            observations=observations, history=history, state=state,
            parameters=parameters, load_labels=load_labels).items()}
        dist, values = self.actor_critic.evaluate_batch(batch, self.training_phase)
        actions = dist.sample()
        batch.update(actions=actions.clone(), values=values, log_prob=dist.log_prob(actions).sum(-1, keepdim=True),
                     mu=dist.mean, sigma=dist.stddev.clone())
        self.pending = batch
        self.reset_pending.zero_()
        return actions

    @torch.no_grad()
    def process_env_step(self, rewards, dones, *, timeouts=None, terminal_values=None, supervision_valid=None):
        """Timeout values MUST be V(s_terminal), evaluated before automatic reset.

        Terminal-value evaluation uses the active phase and terminal episode history.
        A timeout is still a done for GAE/history. Genuine terminations are not bootstrapped.
        Pre-action supervision remains valid on a transition that subsequently terminates.
        """
        if self.pending is None:
            raise RuntimeError("act must precede process_env_step")
        dones = dones.reshape(-1).bool()
        timeouts = torch.zeros_like(dones) if timeouts is None else timeouts.reshape(-1).bool()
        if (timeouts & ~dones).any():
            raise ValueError("Every timeout must also mark done")
        rewards = rewards.reshape(-1).clone()
        if timeouts.any():
            if terminal_values is None:
                raise ValueError("Timeout bootstrap requires pre-reset terminal_values, never reset values")
            terminal = terminal_values.reshape(-1)[timeouts]
            if not torch.isfinite(terminal).all():
                raise ValueError("Nonfinite timeout terminal values")
            rewards[timeouts] += self.gamma * terminal
        valid = torch.ones_like(dones) if supervision_valid is None else supervision_valid.reshape(-1).bool()
        self.storage.add(self.pending, rewards, dones, timeouts, valid)
        self.pending = None
        self.reset_pending.copy_(dones)

    @torch.no_grad()
    def bootstrap_values(self, observations, state, parameters, load_labels, *, env_ids=None):
        """Value at the next observation without changing history or sampling actions.

        For timeouts call BEFORE process_env_step, with pre-reset terminal inputs;
        pass its output as terminal_values. After process_env_step this instead
        respects its pending-reset mask (appropriate for rollout-end bootstrapping).
        """
        reset = self.reset_pending if env_ids is None else self.reset_pending[env_ids]
        history = self.history.preview(observations, reset, env_ids).flatten(1)
        z = self.actor_critic.latent(history, self.training_phase, state, parameters)
        return self.actor_critic.evaluate(state, parameters, load_labels, z)

    @torch.no_grad()
    def compute_returns(self, observations, state, parameters, load_labels):
        if self.pending is not None:
            raise RuntimeError("process_env_step must complete before compute_returns")
        values = self.bootstrap_values(observations, state, parameters, load_labels)
        self.storage.compute_returns(values, self.gamma, self.lam)
        self.returns_ready = True

    def ppo_loss(self, batch, *, adjust_learning_rate=False):
        # No sample() during replay: ratio, adaptive KL and entropy share one distribution.
        dist, value = self.actor_critic.evaluate_batch(batch, self.training_phase)
        if adjust_learning_rate:
            self._adjust_learning_rate(dist.stddev, batch["sigma"], dist.mean, batch["mu"])
        log_prob = dist.log_prob(batch["actions"]).sum(-1)
        ratio = (log_prob - batch["log_prob"].squeeze(-1)).exp()
        surrogate = self._compute_surrogate_loss(ratio, batch["advantages"])
        value_loss = self._compute_value_function_loss(value, batch["returns"], batch["values"])
        return surrogate + self.value_loss_coef * value_loss - self.entropy_coef * dist.entropy().sum(-1).mean(), {
            "surrogate": surrogate.detach(), "value": value_loss.detach(),
            "ratio": ratio.detach().mean(),
            "kl": torch.distributions.kl_divergence(
                torch.distributions.Normal(batch["mu"], batch["sigma"]), dist).sum(-1).mean().detach()}

    def auxiliary_losses(self, batch, *, return_metrics=False):
        valid = batch["valid"]
        # Filter BEFORE arithmetic; empty batches must not step Adam's existing momentum.
        if not valid.any():
            return (None, None, {}) if return_metrics else (None, None)
        h = batch["history"][valid].detach()
        target = batch["load_labels"][valid].detach()
        weights = target.new_tensor((3., 3., 3., 1., 1., 1., 10.))
        error = self.actor_critic.load_estimator(h) - target
        load_loss = (error * weights).square().sum(-1).mean()
        matching = None
        if self.training_phase == "teacher_student":
            with torch.no_grad():
                z = self.actor_critic.teacher_latent(batch["state"][valid], batch["parameters"][valid])
            matching = (self.actor_critic.history_encoder(h) - z).square().sum(-1).mean()
        if return_metrics:
            # Raw physical-unit component errors; reuse the supervised forward before its step.
            error = error.detach()
            metrics = {"load_mae": error.abs().mean(0), "load_mse": error.square().mean(0)}
            return matching, load_loss, metrics
        return matching, load_loss

    def update(self):
        if not self.returns_ready or self.pending is not None:
            raise RuntimeError("Complete rollout and compute_returns before update")
        totals, counts = {}, {}
        def record(key, value, count):
            totals[key] = totals.get(key, 0.) + value.detach() * count
            counts[key] = counts.get(key, 0) + count
        for batch in self.storage.batches(self.num_mini_batches, self.num_learning_epochs):
            self.actor_critic.zero_grad(set_to_none=True)
            loss, metrics = self.ppo_loss(batch, adjust_learning_rate=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.rl_parameters, self.max_grad_norm)
            self.optimizer.step()
            clip_action_std(self.actor_critic)
            for key, value in metrics.items():
                record(key, value, len(batch["actions"]))
        # Complete all PPO epochs before changing load estimates (same as concurrent TS).
        # PPO ratios in this update therefore never see auxiliary-mutated estimates.
        for batch in self.storage.batches(self.num_mini_batches, self.config.auxiliary_epochs):
            self.actor_critic.zero_grad(set_to_none=True)
            matching, load, load_metrics = self.auxiliary_losses(batch, return_metrics=True)
            count = batch["valid"].sum()
            if matching is not None:
                matching.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.history_encoder.parameters(), self.max_grad_norm)
                self.history_optimizer.step()
                record("latent_matching", matching, count)
            if load is not None:
                load.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.load_estimator.parameters(), self.max_grad_norm)
                self.load_optimizer.step()
                record("load_estimation", load, count)
                for key, value in load_metrics.items():
                    record(key, value, count)
        self.storage.clear()
        self.returns_ready = False
        self.iteration += 1
        result = {key: (value / counts[key]).item() for key, value in totals.items() if not key.startswith("load_m")}
        components = ("com_x_m", "com_y_m", "com_z_m", "velocity_x_m_s", "velocity_y_m_s", "velocity_z_m_s", "mass_kg")
        for metric in ("load_mae", "load_mse"):
            values = totals[metric] / counts[metric] if metric in totals else torch.full((7,), float("nan"))
            if metric == "load_mse": values = values.sqrt()
            for name, value in zip(components, values.tolist()):
                result[f"{metric.replace('mse', 'rmse')}/{name}"] = value
        for key in ("latent_matching", "load_estimation"):
            result.setdefault(key, 0.)
        return result

    def checkpoint_state(self):
        if self.pending is not None or self.storage.step:
            raise RuntimeError("Checkpoint only at completed update boundaries")
        return dict(schema_version=self.checkpoint_version, config=self.config.resolved(),
                    input_contract=self.actor_critic.input_contract(),
                    training_phase=self.training_phase, iteration=self.iteration,
                    model_state_dict=self.actor_critic.state_dict(),
                    optimizer=self.optimizer.state_dict(), history_optimizer=self.history_optimizer.state_dict(),
                    load_optimizer=self.load_optimizer.state_dict(), learning_rate=self.learning_rate,
                    entropy_coef=self.entropy_coef,
                    history=self.history.frames.clone(), reset_pending=self.reset_pending.clone(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(self.device) if torch.device(self.device).type == "cuda" else None)

    def save(self, path):
        torch.save(self.checkpoint_state(), path)

    def load(self, path, *, restore_rng=True, restore_history=True):
        data = torch.load(path, map_location=self.device, weights_only=False)
        return self.load_checkpoint_state(data, restore_rng=restore_rng, restore_history=restore_history)

    def load_checkpoint_state(self, data, *, restore_rng=True, restore_history=True):
        if self.pending is not None or self.storage.step:
            raise RuntimeError("Load only at update boundaries")
        if data.get("schema_version") != self.checkpoint_version:
            raise ValueError("Incompatible BR checkpoint schema: Stage-1 opaque packing requires explicit conversion")
        if data.get("input_contract") != self.actor_critic.input_contract():
            raise ValueError("Incompatible BR input/history/latent/action contract")
        config = type(self.config)(**data["config"])
        if replace(config, training_phase=self.config.training_phase) != self.config:
            raise ValueError("BR resolved config mismatch; reconstruct model/algorithm with checkpoint config")
        self.set_training_phase(data["training_phase"])
        self.actor_critic.load_state_dict(data["model_state_dict"], strict=True)
        for name in ("optimizer", "history_optimizer", "load_optimizer"):
            getattr(self, name).load_state_dict(data[name])
        self.iteration, self.learning_rate = data["iteration"], data["learning_rate"]
        self.entropy_coef = data["entropy_coef"]
        if restore_history:
            self.history.frames.copy_(data["history"])
            self.reset_pending.copy_(data["reset_pending"])
        else:
            self.clear_history()
        if restore_rng:
            torch.set_rng_state(data["torch_rng"].cpu())
            if data["cuda_rng"] is not None and torch.device(self.device).type == "cuda":
                torch.cuda.set_rng_state(data["cuda_rng"].cpu(), self.device)
        # Caller must also restore simulator state, or mark all reset on the next act.
        return self.iteration
