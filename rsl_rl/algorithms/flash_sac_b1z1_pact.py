"""Off-policy FlashSAC optimizer with the existing B1Z1 PACT physics objectives."""
import copy
import math
from types import SimpleNamespace
import torch
from torch import nn
from rsl_rl.modules.b1z1_flashsac import (
    FlashSACDoubleCritic, FlashSACTemperature, RewardNormalizer, RepeatedNoise,
    normalize_weights, _select_min_q_log_probs, _compute_categorical_td_target,
)
from rsl_rl.storage.replay_buffer_b1z1_pact import ReplayBufferB1Z1PACT
from .ppo_b1z1_pact import PPO_B1Z1PACT, FORCE_GATE_METRIC_NAMES, _event_conditioned_force_statistics
from . import b1z1_actor_physics as actor_physics
from .b1z1_bard_pinn import auxiliary_step, combined_pinn_loss, frozen

from rsl_rl.algorithms.kl_rate_band import KLRateBandController, update_duals_from_mean

class FlashSAC_B1Z1PACT:
    # Reuse only the established auxiliary objectives and reliability controller.
    # No PPO rollout, likelihood, GAE, value or surrogate update runs here.
    _coupled_torque = PPO_B1Z1PACT._coupled_torque
    _force_prediction_blend_alpha = PPO_B1Z1PACT._force_prediction_blend_alpha
    _update_event_conditioned_force_gate = PPO_B1Z1PACT._update_event_conditioned_force_gate
    _resolve_pinn_forces = PPO_B1Z1PACT._resolve_pinn_forces
    _compute_vae_loss = PPO_B1Z1PACT._compute_vae_loss
    _masked_force_slice_mse = PPO_B1Z1PACT._masked_force_slice_mse

    def __init__(self, actor_critic, privileged_decoder, dynamics_backend, cfg, device):
        self.actor_critic, self.privileged_decoder = actor_critic, privileged_decoder
        self.dynamics_backend, self.cfg, self.device = dynamics_backend, cfg, device
        self.enable_additional_diagnostics = True
        self.gamma = cfg.get("sac_gamma", .95)
        self._set_action_ranges({name: cfg.get(f"sac_{name}_action_range", 1.)
                                 for name in ("position", "leg_torque", "arm_torque")})
        self.learning_rate = cfg.get("sac_learning_rate", 3e-4)
        self.max_grad_norm = cfg["max_grad_norm"]
        self.film_identity_loss_weight = cfg.get("film_identity_loss_weight", 0.0)
        self.film_identity_error_scale = cfg.get("film_identity_error_scale", 1.0)
        if self.film_identity_loss_weight < 0.0:
            raise ValueError("film_identity_loss_weight must be nonnegative")
        if self.film_identity_error_scale <= 0.0:
            raise ValueError("film_identity_error_scale must be positive")

        # ``get_optim_groups`` is the actor-critic's source of truth for
        # actor/critic/context partitioning and its weight-decay conventions.
        actor_groups, context_groups = actor_critic.get_optim_groups()

        encoder_weight_decay = context_groups[0].get("weight_decay", 0.0)
        auxiliary_groups = list(context_groups) + [
            {
                "params": list(privileged_decoder.parameters()),
                "weight_decay": encoder_weight_decay,
                "name": "privileged_decoder",
            },
        ]

        # Three disjoint owners in every backend, not a BARD-only override.
        from .b1z1_bard_pinn import configure_optimizers
        configure_optimizers(self, actor_groups, auxiliary_groups)

        self.kl_controller = KLRateBandController(
            warmup_iters=cfg.get("kl_warmup_env_steps", cfg.get("kl_warmup_iters", 500)),
            warmup_beta_max=cfg.get("kl_warmup_beta_max", cfg["vae_kld_weight"]),
            band_warmup_iters=cfg.get("kl_band_warmup_env_steps", cfg.get("kl_band_warmup_iters", 500)),
            rate_min=cfg.get("kl_r_min", 0.10), rate_max=cfg.get("kl_r_max", 1.00),
            dual_lr=cfg.get("kl_dual_lr", 1.0e-3),
            augmented_rho=cfg.get("kl_aug_rho", 0.1),
            ema_decay=cfg.get("kl_ema_decay", 0.99),
        )
        self.bard_auxiliary = cfg.get("dynamics_backend", "pinocchio").lower() == "bard"
        from .b1z1_actor_physics import configure
        configure(self)
        self.bard_phase_metrics = {}
        self.use_kl_rate_band = bool(cfg.get("use_kl_rate_band", True))
        self.use_cosine_kl_warmup = bool(
            cfg.get("use_cosine_kl_warmup", True)
        )

        self.transition = SimpleNamespace()

        self.pinn_weight, self.pinn_updates = 0.0, 0
        # The first decoder measurement initializes the EMA; starting at
        # infinity would keep the reliability gate permanently closed.
        self.force_ema = None
        self.force_gate_active, self.force_gate_count = False, 0
        self.force_metric_emas = {name: None for name in FORCE_GATE_METRIC_NAMES}
        self.force_blend_min_alpha = float(cfg.get("force_blend_min_alpha", 0.01))
        if not 0.0 <= self.force_blend_min_alpha <= 1.0:
            raise ValueError("force_blend_min_alpha must lie in [0, 1]")
        # The first reconstruction EMA defines alpha_min. As reconstruction
        # improves toward the gate threshold, predicted-force authority ramps
        # linearly to one. Persist this reference across checkpoint resumes.
        self.force_blend_start_ema = None

        if not self.bard_auxiliary:
            raise ValueError("The minimal B1Z1 FlashSAC path requires dynamics_backend='bard'")
        self.replay = ReplayBufferB1Z1PACT(cfg.get("replay_capacity", 32768),
            n_step=cfg.get("n_step", 1), device=cfg.get("replay_device", "cpu"),
            pin_memory=cfg.get("replay_pin_memory", True))
        self.batch_size = cfg.get("sac_batch_size", 2048)
        self.warmup = cfg.get("replay_warmup", 10000)
        if not 1 <= self.warmup <= self.replay.capacity:
            raise ValueError("Replay warm-up must lie within replay capacity")
        self.updates_per_step = cfg.get("sac_updates_per_step", 2)
        self.actor_period = cfg.get("sac_actor_period", 2)
        if self.batch_size < 2 or self.actor_period < 1 or self.updates_per_step < 1:
            raise ValueError("Invalid FlashSAC batch size or update periods")
        self.bins = cfg.get("sac_num_bins", 101)
        self.minimum, self.maximum = cfg.get("sac_min_v", -5.), cfg.get("sac_max_v", 5.)
        self.tau = cfg.get("sac_tau", .01)
        critic_dim = actor_critic.critic[0].in_features
        self.q = FlashSACDoubleCritic(cfg.get("sac_critic_blocks", 2),
            critic_dim + 2 * actor_critic.num_actions, cfg.get("sac_critic_width", 256),
            self.bins, self.minimum, self.maximum).to(device)
        normalize_weights(self.q)
        self.target_q = copy.deepcopy(self.q).requires_grad_(False)
        self.temperature = FlashSACTemperature(cfg.get("sac_initial_temperature", .01)).to(device)
        self.target_entropy = actor_critic.num_actions * math.log(2 * math.pi * math.e * cfg.get("sac_target_sigma", .15)**2)
        # Preserve the PACT parameter grouping/PCGrad wrapper with FlashSAC Adam.
        self.actor_optimizer._optim = torch.optim.Adam(actor_groups, lr=self.learning_rate)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=self.learning_rate)
        self.temperature_optimizer = torch.optim.Adam(self.temperature.parameters(), lr=self.learning_rate)
        self.reward_normalizer = RewardNormalizer(self.gamma, self.maximum).to(device)
        self.noise = RepeatedNoise()
        self.update_step = self.target_updates = self.env_steps = self.pending_updates = 0
        self.schedule_env_steps = self.schedule_step_delta = 0
        self.force_statistics = {}
        self.use_amp = cfg.get("sac_use_amp", False)
        if self.use_amp and torch.device(device).type != "cuda":
            raise ValueError("FlashSAC AMP requires CUDA")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.q_forward = self.q.forward
        self.target_forward = self.target_q.forward
        if cfg.get("sac_use_compile", False):
            self.q_forward = torch.compile(self.q_forward)
            self.target_forward = torch.compile(self.target_forward)
        duration = cfg.get("sac_lr_decay_updates", 1000000)
        ratio = cfg.get("sac_lr_end", 1.5e-4) / self.learning_rate
        schedule = lambda step: ratio + (1 - ratio) * .5 * (1 + math.cos(math.pi * min(step / duration, 1.)))
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, schedule) for opt in
            (self.actor_optimizer.optimizer, self.q_optimizer, self.temperature_optimizer)]
        owners = [set(map(id, group)) for group in (self.ppo_parameters, self.enc_parameters,
            self.decoder_parameters, list(self.q.parameters()), list(self.temperature.parameters()))]
        assert all(not owners[i] & owners[j] for i in range(len(owners)) for j in range(i))

    def _set_action_ranges(self, ranges):
        validated = {}
        for name in ("position", "leg_torque", "arm_torque"):
            value = float(ranges[name])
            key = f"sac_{name}_action_range"
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
            if value > self.cfg["clip_actions"]:
                raise ValueError(f"{key} must not exceed the environment's clip_actions")
            validated[name] = value
        self.action_ranges = validated
        self.cfg.update({f"sac_{name}_action_range": value for name, value in validated.items()})

    def to_env_actions(self, normalized_actions):
        """Expand the 17 position, 12 leg-torque and 5 arm-torque commands once."""
        return torch.cat((
            normalized_actions[..., :17] * self.action_ranges["position"],
            normalized_actions[..., 17:29] * self.action_ranges["leg_torque"],
            normalized_actions[..., 29:34] * self.action_ranges["arm_torque"],
        ), dim=-1)

    @torch.no_grad()
    def act_inference(self, observations, histories):
        return self.to_env_actions(self.actor_critic.act_inference(observations, histories))

    @property
    def current_entropy_coef(self):
        return self.temperature().detach().item()

    def init_storage(self, *args, **kwargs):
        # Replay schema is inferred from the first complete transition.
        pass

    @torch.no_grad()
    def act(self, obs, critic_obs, history, explicit_labels):
        mean, std = self.actor_critic.get_mean_and_std(obs, history)
        actions = (mean + std * self.noise.sample(mean)).tanh()
        self.transition = SimpleNamespace(**{k: v.detach().clone() for k, v in dict(
            observations=obs, critic_observations=critic_obs, histories=history,
            explicit_targets=explicit_labels, actions=actions).items()})
        self.transition.actor_physics = None
        return self.to_env_actions(actions)

    def process_env_step(self, rewards, dones, infos, next_privileged, dynamics_state,
                         rollout_initial_state, *, next_observations, next_histories,
                         next_critic_observations):
        truncated = infos.get("time_outs", torch.zeros_like(dones)).to(self.device).bool().flatten()
        terminated = dones.bool().flatten() & ~truncated
        successor = dict(next_observations=next_observations.clone(), next_histories=next_histories.clone(),
                         next_critic_observations=next_critic_observations.clone())
        if dones.any():
            final = infos.get("flash_sac_final")
            if final is None:
                raise RuntimeError("Episode end requires pre-reset FlashSAC observations/history")
            for name in successor:
                successor[name][dones.flatten().bool()] = final[name].to(self.device)[dones.flatten().bool()]
        fields = vars(self.transition).copy()
        physics = fields.pop("actor_physics")
        if physics:
            fields.update({"actor_phys_" + k: v for k, v in physics.items()})
        fields.update(successor)
        fields.update(rewards=rewards.reshape(-1, 1), terminated=terminated[:, None],
            truncated=truncated[:, None], dones=dones.bool().reshape(-1, 1),
            next_privileged=next_privileged, dynamics_state=dynamics_state,
            rollout_initial_state=rollout_initial_state)
        self.replay.add(fields)
        self.reward_normalizer.update(rewards, terminated, truncated)
        self.noise.reset(dones)
        self.env_steps += 1
        if self.replay.size >= self.warmup:
            self.pending_updates += self.updates_per_step

    def _prepare_batch(self, batch):
        actor_physics.refresh_replay_target(self, batch)
        # Adapt existing bounded mechanics caches to a single sampled batch.
        self.storage = SimpleNamespace(
            rollout_initial_state=batch["rollout_initial_state"][None],
            dynamics_state=batch["dynamics_state"][None], dones=batch["dones"][None],
            physics_invalid=batch["physics_invalid"][None],
            actor_physics={k[len("actor_phys_"):]: v[None] for k, v in batch.items() if k.startswith("actor_phys_")})
        actor_physics.prepare(self)
        if self.pinn_weight > 0:
            from .b1z1_bard_pinn import cache_rollout
            self.bard_mechanics_cache = cache_rollout(self)

    def _actor_update(self, batch):
        # Physics solves and signed PCGrad stay FP32 even with critic AMP.
        self.actor_optimizer.zero_grad()
        with frozen(list(self.q.parameters())):
            action, log_prob = self.actor_critic.sample_squashed(batch["observations"], batch["histories"])
            mean = self.actor_critic.action_mean
            context = self.actor_critic.last_context
            q, _ = self.q_forward(batch["critic_observations"], action, training=False)
            sac = (self.temperature().detach() * log_prob - q.min(0).values).mean()
            film = self.actor_critic.last_film_identity_deviation.mean()
            loss = sac + self.film_identity_loss_weight * film
            actor_physics.backward(self, batch, loss, self.to_env_actions(mean.tanh()), context)
        nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm)
        self.actor_optimizer.step()
        self.schedulers[0].step()
        entropy = -log_prob.detach().mean()
        temp_loss = self.temperature() * (entropy - self.target_entropy)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temp_loss.backward()
        self.temperature_optimizer.step()
        self.schedulers[2].step()
        return {"SAC/actor_loss": sac.detach(), "SAC/entropy": entropy,
                "SAC/log_probability": -entropy, "SAC/temperature_loss": temp_loss.detach(),
                "film_identity": film.detach(),
                "FiLM/magnitude": self.actor_critic.last_film_magnitude.detach().mean()}

    def _critic_update(self, batch):
        with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.float16, enabled=self.use_amp):
            with torch.no_grad():
                actions, log_prob = self.actor_critic.sample_squashed(batch["next_observations"], batch["next_histories"])
                obs_all = torch.cat((batch["critic_observations"], batch["next_critic_observations"]))
                act_all = torch.cat((batch["actions"], actions))
                qs, info = self.target_forward(obs_all, act_all, training=True)
                selected = _select_min_q_log_probs(qs.chunk(2, 1)[1], info["log_prob"].chunk(2, 1)[1])
                target = _compute_categorical_td_target(selected.float(),
                    self.reward_normalizer(batch["rewards"]).flatten(), batch["terminated"].float(),
                    self.temperature() * log_prob.float(), self.gamma, self.bins, self.minimum, self.maximum)
            q, info = self.q_forward(obs_all, act_all, training=True)
            loss = -(target[None] * info["log_prob"].chunk(2, 1)[0].float()).sum(-1).mean()
        self.q_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.q_optimizer)
        self.scaler.update()
        self.schedulers[1].step()
        normalize_weights(self.q)
        with torch.no_grad():
            # Upstream EMA is parameter-only: target BN maintains its own stats
            # from the same concatenated current/next training batches.
            for target_p, source in zip(self.target_q.parameters(), self.q.parameters()):
                target_p.lerp_(source, self.tau)
        self.target_updates += 1
        return {"SAC/critic_loss": loss.detach(), "SAC/q": q.detach().mean()}

    def update_batch(self, batch, iteration):
        self.pinn_metric_sums = {}
        self._prepare_batch(batch)
        metrics = {}
        actor_updated = self.update_step % self.actor_period == 0
        if actor_updated:
            metrics.update(self._actor_update(batch))
        metrics.update(self._critic_update(batch))
        valid = (~batch["dones"].bool()).float()
        with torch.no_grad():
            context = self.actor_critic.decode_context(self.actor_critic.context_encoder(batch["histories"], sample=True))
            prediction = torch.cat((self.actor_critic.predict_grf(context, batch["nominal_torque"]),
                                    context["base_wrench"], context["ee_force"]), -1)
            start = self.cfg["privileged_force_start"]
            target = torch.cat((batch["next_privileged"][:, start:start+12], batch["explicit_targets"][:, 6:15]), -1)
            stats = _event_conditioned_force_statistics(prediction, target, valid,
                self.cfg["force_gate_ee_event_norm_threshold"], self.cfg["force_gate_base_event_norm_threshold"])
            for name, values in stats.items():
                packed = torch.stack(values).detach()
                self.force_statistics[name] = self.force_statistics.get(name, torch.zeros_like(packed)) + packed
            overall = torch.stack((((prediction-target).square()*valid).sum(), valid.sum()*prediction.shape[-1]))
            self.force_statistics["overall"] = self.force_statistics.get("overall", torch.zeros_like(overall)) + overall
        aux, inverse, rollout = auxiliary_step(self, batch, valid, iteration)
        metrics.update({k: v.detach() for k, v in aux.items() if isinstance(v, torch.Tensor) and v.numel() == 1})
        metrics.pop("loss", None)
        metrics["base_velo"] = metrics.pop("base_velocity")
        metrics.update(pinn=combined_pinn_loss(inverse, rollout, self.cfg),
                       pinn_inverse_dynamics=inverse, pinn_rollout=rollout)
        metrics.update(self.bard_phase_metrics)
        metrics.update({k: total / max(count, 1) for k, (total, count) in self.pinn_metric_sums.items()})
        raw = aux["kl_raw"].detach()
        metrics.update(self.kl_controller.metrics(raw, aux["kl_reg_loss"], iteration,
            self.use_kl_rate_band, self.use_cosine_kl_warmup))
        actor_physics.finish(self, metrics, 1)
        if "ActorPhysics/ppo_gradient_cosine" in metrics:
            metrics["ActorPhysics/sac_gradient_cosine"] = metrics.pop("ActorPhysics/ppo_gradient_cosine")
        self.bard_mechanics_cache = None
        self.storage = None
        self.update_step += 1
        return {k: float(v) for k, v in metrics.items()}

    def update(self, completed_env_steps):
        from .b1z1_training_clock import prepare_step_schedules
        prepare_step_schedules(self, completed_env_steps)
        sums, counts = {}, {}
        self.force_statistics = {}
        for _ in range(self.pending_updates):
            metrics = self.update_batch(self.replay.sample(self.batch_size, self.device), completed_env_steps)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.) + value
                counts[key] = counts.get(key, 0) + 1
        self.pending_updates = 0
        result = {k: v / counts[k] for k, v in sums.items()}
        if "kl_raw" in result:
            raw = update_duals_from_mean(self.kl_controller, result["kl_raw"], 1, completed_env_steps, self.device,
                enabled=self.use_kl_rate_band, use_cosine_warmup=self.use_cosine_kl_warmup)
            result.update({k: float(v) for k, v in self.kl_controller.metrics(raw,
                raw.new_tensor(result["kl_reg_loss"]), completed_env_steps, self.use_kl_rate_band,
                self.use_cosine_kl_warmup).items()})
        if self.force_statistics:
            errors = {k: (self.force_statistics[k][0] / self.force_statistics[k][1].clamp_min(1)).item()
                      for k in FORCE_GATE_METRIC_NAMES}
            samples = {k: self.force_statistics[k][2].item() for k in FORCE_GATE_METRIC_NAMES}
            numerator, denominator = self.force_statistics["overall"].tolist()
            if denominator > 0:
                self._update_event_conditioned_force_gate(errors, samples, numerator / denominator)
            result.update({f"force_gate_{k}_mse": v for k, v in errors.items()})
            for name in ("ee_event_fraction", "base_event_fraction"):
                active, valid = self.force_statistics[name].tolist()
                result["force_gate_" + name] = active / max(valid, 1)

        result.update({f"SAC/{name}_action_range": value for name, value in self.action_ranges.items()})
        result.update({ "SAC/temperature": self.current_entropy_coef, "SAC/replay_size": self.replay.size,
            "SAC/reward_scale": self.reward_normalizer(torch.ones(1, device=self.device)).item(),
            "SAC/target_updates": self.target_updates, "SAC/updates": self.update_step,
            "SAC/actor_lr": self.actor_optimizer.optimizer.param_groups[0]["lr"],
            "SAC/critic_lr": self.q_optimizer.param_groups[0]["lr"],
            "SAC/temperature_lr": self.temperature_optimizer.param_groups[0]["lr"],
            "Auxiliary/encoder_lr": self.auxiliary_optimizer.param_groups[0]["lr"],
            "Auxiliary/decoder_lr": self.decoder_optimizer.param_groups[0]["lr"],
            "PINN/scheduled_weight": self.pinn_weight,
            "PINN/inverse/component_weight": self.cfg.get("pinn_inverse_weight", 1.),
            "PINN/rollout/component_weight": self.cfg.get("pinn_rollout_weight", 1.),
            "PINN/rollout/enabled": float(self.cfg["use_pinn_rollout_loss"]),
            "PINN/combined_loss": result.get("pinn", 0.),
            "PINN/weighted_combined_loss": self.pinn_weight * result.get("pinn", 0.),
            "force_gate_active": float(self.force_gate_active), "force_gate_ema": self.force_ema or 0.,
            "force_gate_count": self.force_gate_count, "force_prediction_alpha": self._force_prediction_blend_alpha()})
        result.update({f"force_gate_{k}_ema": v or 0. for k, v in self.force_metric_emas.items()})
        return result

    def state_dict(self):
        state = {name: getattr(self, name).state_dict() for name in
            ("q", "target_q", "temperature", "q_optimizer", "temperature_optimizer", "reward_normalizer", "scaler")}
        state["action_ranges"] = dict(self.action_ranges)
        state["schedulers"] = [s.state_dict() for s in self.schedulers]
        state["counters"] = {k: getattr(self, k) for k in ("update_step", "target_updates", "env_steps", "pending_updates",
            "pinn_updates", "pinn_weight", "schedule_env_steps")}
        if self.cfg.get("replay_persistence", False):
            state["replay"] = self.replay.state_dict()
        return state

    def load_state_dict(self, state, load_optimizer=True):
        # Checkpoints predating configurable rescaling used an implicit range of one.
        self._set_action_ranges(state.get("action_ranges", {
            name: state.get("action_range", 1.)
            for name in ("position", "leg_torque", "arm_torque")}))
        for name in ("q", "target_q", "temperature", "reward_normalizer"):
            getattr(self, name).load_state_dict(state[name])
        if load_optimizer:
            for name in ("q_optimizer", "temperature_optimizer", "scaler"):
                getattr(self, name).load_state_dict(state[name])
            for scheduler, saved in zip(self.schedulers, state["schedulers"]):
                scheduler.load_state_dict(saved)
            for name, value in state["counters"].items():
                setattr(self, name, value)
        if "replay" in state:
            self.replay.load_state_dict(state["replay"])
        if not self.replay.size:
            self.pending_updates = 0
