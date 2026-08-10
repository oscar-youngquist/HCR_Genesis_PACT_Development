"""PPO plus temporal context and one Pinocchio consistency objective for B1/Z1."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import random
import warnings

from rsl_rl.algorithms.pc_grad import PCGrad
from rsl_rl.algorithms.kl_rate_band import KLRateBandController
from rsl_rl.storage.rollout_storage_b1z1_pact import RolloutStorageB1Z1PACT


class PPO_B1Z1PACTPos:
    def __init__(self, actor_critic, privileged_decoder, cfg, device):
        self.actor_critic, self.privileged_decoder = actor_critic, privileged_decoder
        self.enable_additional_diagnostics = True
        self.cfg, self.device = cfg, device
        self.clip_param, self.gamma, self.lam = cfg["clip_param"], cfg["gamma"], cfg["lam"]
        self.value_loss_coef, self.entropy_coef = cfg["value_loss_coef"], cfg["entropy_coef"]
        self.use_adaptive_entropy = cfg.get("use_adaptive_entropy", False)
        self.entropy_coef_bounds = tuple(float(value) for value in cfg.get("adaptive_ent_bounds", (self.entropy_coef, self.entropy_coef)))
        self.ent_linvelo_threshold = float(cfg.get("adaptive_ent_lin_threshold", 0.0))
        self.ent_angvelo_threshold = float(cfg.get("adaptive_ent_ang_threshold", 0.0))
        self.ent_terrain_threshold = float(cfg.get("adaptive_ent_ter_threshold", 0.0))
        self.ent_softmax_temperature = float(cfg.get("adaptive_ent_softmax_temp", 1.0))
        self.current_entropy_coef = float(self.entropy_coef)
        if len(self.entropy_coef_bounds) != 2:
            raise ValueError("adaptive_ent_bounds must contain [low, high]")
        if self.entropy_coef_bounds[0] < 0.0 or self.entropy_coef_bounds[1] < self.entropy_coef_bounds[0]:
            raise ValueError("adaptive_ent_bounds must satisfy 0 <= low <= high")
        if self.ent_softmax_temperature <= 0.0:
            raise ValueError("adaptive_ent_softmax_temp must be greater than zero")
        self.max_grad_norm, self.epochs, self.mini_batches = cfg["max_grad_norm"], cfg["num_learning_epochs"], cfg["num_mini_batches"]
        self.learning_rate = cfg["learning_rate"]
        self.desired_kl, self.schedule = cfg.get("desired_kl"), cfg.get("schedule", "fixed")
        self.use_clipped_value_loss = cfg.get("use_clipped_value_loss", True)
        self.film_identity_loss_weight = cfg.get("film_identity_loss_weight", 0.0)
        self.film_identity_error_scale = cfg.get("film_identity_error_scale", 1.0)
        if self.film_identity_loss_weight < 0.0:
            raise ValueError("film_identity_loss_weight must be nonnegative")
        if self.film_identity_error_scale <= 0.0:
            raise ValueError("film_identity_error_scale must be positive")

        # Match PPO_PACT's stochastic bootstrap gate. The decoder's
        # reconstruction quality determines the probability of using the
        # learned encoder dynamics on the following rollout/update.
        self.boot_mult = 1.0
        # UniFP always exposes the encoder's deterministic history latent to
        # the actor. Keep the old flag for checkpoint/log compatibility, but
        # latent masking is disabled at every policy call below.
        self.use_boot_latent = True

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

        # The encoder receives PPO and representation-learning gradients;
        # reconstruction also owns
        # both decoder heads through the shared auxiliary optimizer below.
        ppo_enc_groups = [
            {
                "params": list(group["params"]),
                "weight_decay": group.get("weight_decay", 0.0),
                "name": f"ppo_{group['name']}",
            }
            for group in auxiliary_groups
        ]
        auxiliary_enc_groups = [
            {
                "params": list(group["params"]),
                "weight_decay": group.get("weight_decay", 0.0),
                "name": f"auxiliary_{group['name']}",
            }
            for group in auxiliary_groups
        ]

        self.actor_optimizer = PCGrad(optim.Adam([*actor_groups,*ppo_enc_groups], lr=cfg["learning_rate"]), reduction="sum")
        # Clip the same ownership boundary that is stepped by PPO, as UniFP
        # does, including all PACT-position PPO parameter groups.
        seen_ppo_parameters = set()
        self.ppo_parameters = []
        for group in self.actor_optimizer.optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) not in seen_ppo_parameters:
                    seen_ppo_parameters.add(id(parameter))
                    self.ppo_parameters.append(parameter)

        # # # We want to reduce the LR of the critic
        # for param_group in self.actor_optimizer.optimizer.param_groups:
        #     # specifically modifies the learning rate of the crtic specific parameters
        #     if "name" in param_group.keys():
        #         if "critic" in param_group["name"]:
        #             param_group['lr'] = (cfg["learning_rate"] / 3.0)

        self.auxiliary_optimizer = optim.Adam(
            [parameter for group in auxiliary_enc_groups for parameter in group["params"]],
            lr=cfg.get("adaptation_learning_rate", 1.0e-5),
        )
        self.kl_controller = KLRateBandController(
            warmup_iters=cfg.get("kl_warmup_iters", 500),
            warmup_beta_max=cfg.get("kl_warmup_beta_max", cfg["vae_kld_weight"]),
            rate_min=cfg.get("kl_r_min", 0.10), rate_max=cfg.get("kl_r_max", 1.00),
            dual_lr=cfg.get("kl_dual_lr", 1.0e-3),
            augmented_rho=cfg.get("kl_aug_rho", 0.1),
            ema_decay=cfg.get("kl_ema_decay", 0.99),
        )

        self.transition = RolloutStorageB1Z1PACT.Transition()
        self.storage = None
        # The first decoder measurement initializes the EMA; starting at
        # infinity would keep the reliability gate permanently closed.

    def init_storage(self, *args):
        self.storage = RolloutStorageB1Z1PACT(*args, device=self.device)

    def update_adaptive_entropy_coef(self, performance_metrics):
        """Increase exploration when tracking or terrain progress is below target."""
        def normalized_gap(name, threshold):
            if threshold <= 0.0:
                return 0.0
            value = float(performance_metrics.get(name, 0.0))
            return max(0.0, threshold - value) / threshold

        gaps = torch.tensor((
            normalized_gap("lin_vel_tracking", self.ent_linvelo_threshold),
            normalized_gap("ang_vel_tracking", self.ent_angvelo_threshold),
            normalized_gap("terrain_level", self.ent_terrain_threshold),
        ), dtype=torch.float32, device=self.device)
        weights = F.softmax(gaps / self.ent_softmax_temperature, dim=0)
        weighted_gap = torch.sum(weights * gaps).item()
        low, high = self.entropy_coef_bounds
        self.current_entropy_coef = low + weighted_gap * (high - low)
        return self.current_entropy_coef

    def act(self, obs, critic_obs, history, explicit_labels):
        actions = self.actor_critic.act(
            obs, history,
            mask_latent=False,
        ).detach()
        self.transition.observations = obs.detach().clone()
        self.transition.critic_observations = critic_obs.detach().clone()
        self.transition.histories = history.detach().clone()
        self.transition.actions = actions
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.log_probs = self.actor_critic.get_actions_log_prob(actions).detach().unsqueeze(-1)
        self.transition.mu, self.transition.sigma = self.actor_critic.action_mean.detach(), self.actor_critic.action_std.detach()
        self.transition.explicit_targets = explicit_labels.detach().clone()
        return actions

    def process_env_step(self, rewards, dones, infos, next_privileged, dynamics_state):
        # ``dynamics_state`` is the post-step state collected by the runner.
        # Storing it beside this transition keeps action_t, v_t, and v_(t+1)
        # together until the shuffled PPO update.
        self.transition.rewards =  rewards.clone()
        self.transition.dones = dones
        self.transition.next_privileged = next_privileged.detach().clone()
        self.transition.dynamics_state = dynamics_state.detach().clone()
        if "time_outs" in infos:
            # Match UniFP's [N] reward convention and squeeze the value
            # bootstrap correction before adding it in place.
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def spectral_normalization(
        self,
        model: nn.Module,
        sigma_max: float = 1.0,
        n_power_iters: int = 1,
    ):
        """
        Spectral-norm clip all Linear layers except selected output layers.

        Args:
            model: network to normalize in-place
            sigma_max: maximum allowed spectral norm
            n_power_iters: number of power iterations for sigma estimate
        """

        whitelist = (nn.Linear,)

        # lazily create persistent power-iteration vectors
        if not hasattr(self, "_spec_u"):
            self._spec_u = {}

        for module_name, module in model.named_modules():
            if not isinstance(module, whitelist):
                continue

            # skip known output layers
            if module_name.endswith("out") or module_name.endswith("mean") or module_name.endswith("var") or "critic" in module_name:
                continue

            for param_name, param in module.named_parameters(recurse=False):
                if param_name != "weight" or param.ndim != 2:
                    continue

                full_name = f"{module_name}.{param_name}" if module_name else param_name
                W = param.data  # [out_dim, in_dim]

                # initialize persistent u vector once per parameter
                if full_name not in self._spec_u or self._spec_u[full_name].shape[0] != W.shape[0]:
                    u = torch.randn(W.shape[0], device=W.device, dtype=W.dtype)
                    u = u / (u.norm() + 1e-12)
                    self._spec_u[full_name] = u

                u = self._spec_u[full_name]

                with torch.no_grad():
                    # power iteration
                    for _ in range(n_power_iters):
                        v = W.t().mv(u)
                        v = v / (v.norm() + 1e-12)

                        u = W.mv(v)
                        u = u / (u.norm() + 1e-12)

                    # sigma ~= u^T W v
                    sigma = torch.dot(u, W.mv(v))

                    # save updated u for next call
                    self._spec_u[full_name] = u

                    # clip only if above threshold
                    if sigma > sigma_max:
                        param.data.mul_(sigma_max / (sigma + 1e-12))

    def compute_returns(self, critic_obs):
        self.storage.compute_returns(self.actor_critic.evaluate(critic_obs).detach(), self.gamma, self.lam)

    def _compute_vae_loss(self, obs_hist_batch, obs_target, labels, valid, iteration):
        # Recompute the auxiliary graph after the actor update. The PPO
        # graph was consumed by PCGrad and sharing it here would either
        # fail on a second backward pass or retain an unnecessarily large
        # rollout graph.
        aux_context = self.actor_critic.decode_context(
            self.actor_critic.context_encoder(obs_hist_batch, sample=True)
        )
        # Match UniFP's z-only next-frame decoder. Its target now includes the
        # normalized force block that previously had a dedicated decoder.
        aux_privileged_prediction = self.privileged_decoder(aux_context["z"])

        pred_velo_loss = F.mse_loss(aux_context["base_velocity"], labels[:, :3])
        pred_ee_position_loss = F.mse_loss(aux_context["ee_position"], labels[:, 3:6])
        pred_base_wrench_loss = F.mse_loss(aux_context["base_wrench"], labels[:, 6:12])
        pred_ee_force_loss = F.mse_loss(aux_context["ee_force"], labels[:, 12:15])
        # BCE-with-logits is the stable binary-state reconstruction loss.
        pred_foot_contact_loss = F.binary_cross_entropy_with_logits(
            aux_context["foot_contact_logits"], labels[:, 15:19],
        )
        pred_foot_height_loss = F.mse_loss(
            aux_context["foot_height"], labels[:, 19:23],
        )

        # Loss for explicit current-state-estimation
        aux_explicit = (
            self.cfg["explicit_base_vel_weight"] * pred_velo_loss
            + self.cfg["explicit_ee_position_weight"] * pred_ee_position_loss
            + self.cfg["explicit_base_wrench_weight"] * pred_base_wrench_loss
            + self.cfg["explicit_ee_force_weight"] * pred_ee_force_loss
            + self.cfg["explicit_foot_contact_weight"] * pred_foot_contact_loss
            + self.cfg["explicit_foot_height_weight"] * pred_foot_height_loss
        )

        # VAE recon + KL losses
        privileged_error = (aux_privileged_prediction - obs_target).square() * valid
        aux_privileged_loss = privileged_error.sum() / (
            valid.sum().clamp_min(1.0) * aux_privileged_prediction.shape[-1]
        )
        kl_per_sample = -0.5 * (
            1 + aux_context["logvar"] - aux_context["mean"].square() - aux_context["logvar"].exp()
        ).sum(dim=-1, keepdim=True)
        aux_kl = (kl_per_sample * valid).sum() / valid.sum().clamp_min(1.0)
        kl_reg_loss = self.kl_controller.loss(aux_kl, iteration)

        # Total loss
        aux = (
            aux_explicit
            + self.cfg["privileged_decoder_weight"] * aux_privileged_loss
            + kl_reg_loss
        )

        self.auxiliary_optimizer.zero_grad()

        aux.backward()

        # Match UniFP adaptation training: the auxiliary optimizer steps its
        # raw reconstruction gradient without a separate clipping pass.
        self.auxiliary_optimizer.step()
        self.kl_controller.update_duals(aux_kl, iteration)

        # Return unweighted predictions/targets for the reliability gate.  The
        # gate compares raw MSEs, not the task-specific loss weights above, to
        # answer the simple question: does each decoder beat a constant mean?
        return {
            "base_velocity": pred_velo_loss,
            "ee_position": pred_ee_position_loss,
            "base_wrench": pred_base_wrench_loss,
            "ee_force": pred_ee_force_loss,
            "foot_contact": pred_foot_contact_loss,
            "foot_height": pred_foot_height_loss,
            "privileged_force": self._masked_force_slice_mse(aux_privileged_prediction, obs_target, valid),
            "privileged_decoder": aux_privileged_loss,
            **self.kl_controller.metrics(aux_kl, kl_reg_loss, iteration),
            # Sigmoid probabilities provide a bounded contact prediction for
            # the original PACT MSE-based bootstrap statistic; BCE above is
            # still the optimization objective for these binary labels.
            "explicit_prediction": torch.cat((
                aux_context["base_velocity"], aux_context["ee_position"],
                aux_context["base_wrench"], aux_context["ee_force"],
                torch.sigmoid(aux_context["foot_contact_logits"]),
                aux_context["foot_height"],
            ), dim=-1).detach(),
            "explicit_target": labels.detach(),
            "privileged_prediction": aux_privileged_prediction.detach(),
            "privileged_target": obs_target.detach(),
            "valid": valid.detach(),
        }

    def _masked_force_slice_mse(self, prediction, target, valid):
        """Report force accuracy from the shared privileged decoder."""
        start = self.cfg["privileged_force_start"]
        end = start + self.cfg["privileged_force_dim"]
        error = (prediction[:, start:end] - target[:, start:end]).square() * valid
        return error.sum() / (valid.sum().clamp_min(1.0) * (end - start))


    def _compute_rl_loss(self, batch):
        """Compute the PPO objective with the same organization as PACT PPO.

        The deterministic predicted explicit context is used for both actor and FiLM.
        """
        self.actor_critic.update_distribution(
            batch["observations"], batch["histories"], sample_context=False,
            mask_latent=False,
        )
        actions_log_prob = self.actor_critic.get_actions_log_prob(batch["actions"])
        values = self.actor_critic.evaluate(batch["critic_observations"])
        mu, sigma = self.actor_critic.action_mean, self.actor_critic.action_std

        kl_mean = values.new_zeros(())
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma / batch["sigma"] + 1.0e-5)
                    + (batch["sigma"].square() + (batch["mu"] - mu).square()) / (2.0 * sigma.square()) - 0.5,
                    dim=-1,
                )
                kl_mean = kl.mean()
                if kl_mean > 2.0 * self.desired_kl:
                    self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                elif 0.0 < kl_mean < self.desired_kl / 2.0:
                    self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                for group in self.actor_optimizer.optimizer.param_groups:
                    if group.get("name") in ("actor", "film"):
                        group["lr"] = self.learning_rate

        ratio = torch.exp(actions_log_prob - batch["log_probs"].squeeze(-1))
        advantage = batch["advantages"].squeeze(-1)
        surrogate_loss = torch.maximum(
            -advantage * ratio,
            -advantage * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param),
        ).mean()
        if self.use_clipped_value_loss:
            value_clipped = batch["values"] + (values - batch["values"]).clamp(-self.clip_param, self.clip_param)
            value_loss = torch.maximum((values - batch["returns"]).square(), (value_clipped - batch["returns"]).square()).mean()
        else:
            value_loss = (values - batch["returns"]).square().mean()
        # Near a well-tracked command, FiLM should reduce to the identity
        # transform: gamma=0 and beta=0. The exponential gate relaxes this
        # constraint as base/EE tracking error grows, leaving FiLM free to
        # produce stronger corrective modulation. Detaching the gate prevents
        # the estimator from inflating its error merely to evade the penalty.
        tracking_gate = torch.exp(
            -self.actor_critic.last_tracking_error_sq.detach()
            / self.film_identity_error_scale**2
        )
        film_identity_loss = (
            tracking_gate * self.actor_critic.last_film_identity_deviation
        ).mean()
        ppo_loss = (
            surrogate_loss
            + self.value_loss_coef * value_loss
            - (self.current_entropy_coef if self.use_adaptive_entropy else self.entropy_coef)
            * self.actor_critic.entropy.mean()
            + self.film_identity_loss_weight * film_identity_loss
        )

        return (
            ppo_loss, surrogate_loss, value_loss, film_identity_loss, kl_mean,
            self.actor_critic.action_mean, self.actor_critic.last_context,
        )

    @torch.no_grad()
    def _pre_update_diagnostics(self, batch):
        """Compare the untouched rollout policy with its stored distribution."""
        self.actor_critic.update_distribution(
            batch["observations"], batch["histories"], sample_context=False,
            mask_latent=False,
        )
        mu, sigma = self.actor_critic.action_mean, self.actor_critic.action_std
        old_mu, old_sigma = batch["mu"], batch["sigma"]
        new_logprob = self.actor_critic.get_actions_log_prob(batch["actions"])
        old_logprob = batch["log_probs"].squeeze(-1)
        mu_error, sigma_error = mu - old_mu, sigma - old_sigma
        logprob_error = new_logprob - old_logprob
        ratio = torch.exp(logprob_error)
        eps = 1.0e-8
        old_sigma_safe, sigma_safe = old_sigma.clamp_min(eps), sigma.clamp_min(eps)
        kl = torch.sum(
            torch.log(sigma_safe / old_sigma_safe)
            + (old_sigma_safe.square() + (old_mu - mu).square()) / (2.0 * sigma_safe.square())
            - 0.5,
            dim=-1,
        ).mean()
        diagnostics = {
            "pre_update_mu_rms": mu_error.square().mean().sqrt().item(),
            "pre_update_mu_abs_max": mu_error.abs().max().item(),
            "pre_update_sigma_rms": sigma_error.square().mean().sqrt().item(),
            "pre_update_sigma_abs_max": sigma_error.abs().max().item(),
            "pre_update_logprob_rms": logprob_error.square().mean().sqrt().item(),
            "pre_update_logprob_abs_max": logprob_error.abs().max().item(),
            "pre_update_ratio_mean": ratio.mean().item(),
            "pre_update_ratio_std": ratio.std(unbiased=False).item(),
            "pre_update_kl": kl.item(),
        }
        if (diagnostics["pre_update_mu_abs_max"] > 1.0e-6
                or diagnostics["pre_update_sigma_abs_max"] > 1.0e-6
                or diagnostics["pre_update_logprob_abs_max"] > 1.0e-5):
            warnings.warn(f"PACT-pos pre-update PPO inconsistency: {diagnostics}", RuntimeWarning)
        return diagnostics

    def _torque_clone_loss(self, observations, dynamics_state):
        """Train the dormant torque head toward scaled executed PD torque.

        As in go2_pact_pos, PPO executes only the position head. The auxiliary
        target prepares the torque head for later coupled fine-tuning:

          tau_head * torque_scale ~= clone_scale * tau_PD(q_des, q, qdot).
        """
        # Compact PACT-pos packet: [motor(19), Kp(19), Kd(19), default q(19)].
        motor = dynamics_state[:, 0:19]
        kp = dynamics_state[:, 19:38]
        kd = dynamics_state[:, 38:57]
        default = dynamics_state[:, 57:76]
        # Actor observation layout is [orientation(2), omega(3), q-error(17),
        # qdot(17), ...]. These values precede action_t and therefore match the
        # action mean being supervised, unlike the post-step state packet.
        q = default[:, :17] + observations[:, 5:22] / self.cfg["dof_pos_obs_scale"]
        qd = observations[:, 22:39] / self.cfg["dof_vel_obs_scale"]
        q_target = default[:, :17] + self.cfg["position_action_scale"] * self.actor_critic.last_position_mean
        pd_torque = (
            kp[:, :17] * (q_target - q[:, :17]) - kd[:, :17] * qd[:, :17]
        ) * motor[:, :17]
        predicted_torque = self.cfg["torque_action_scale"] * self.actor_critic.last_torque_mean
        return F.mse_loss(
            predicted_torque,
            self.cfg["torque_clone_target_scale"] * pd_torque,
        )

    def update(self, iteration):
        metrics = {name: 0.0 for name in (
            "value", "surrogate", "base_velo", "ee_position", "base_wrench", "ee_force", "foot_contact", "foot_height",
            "privileged_force", "privileged_decoder", "torque_clone", "film_identity",
            *KLRateBandController.METRIC_NAMES,
        )}
        # PPO_PACT's float64 sufficient statistics. Keep one set per maskable
        # signal: next-privileged reconstruction for z and explicit-state
        # reconstruction for the base/EE estimates.
        boot_stats = {"latent": [None, None, 0.0, 0], "explicit": [None, None, 0.0, 0]}
        updates = 0
        diagnostics = {"lr_before_update": self.learning_rate}
        for batch in self.storage.mini_batches(self.mini_batches, self.epochs):
            if updates == 0:
                if self.enable_additional_diagnostics:
                    diagnostics.update(self._pre_update_diagnostics(batch))
            ppo_loss, surrogate, value, film_identity, _, mean_actions, context = self._compute_rl_loss(batch)
            
            valid = (~batch["dones"].squeeze(-1)).float().unsqueeze(-1)
            torque_clone = self._torque_clone_loss(batch["observations"], batch["dynamics_state"])

            self.actor_optimizer.zero_grad()

            # Match go2_pact_pos: PCGrad reconciles PPO with the auxiliary
            # torque-cloning objective while no dynamics residual is present.
            self.actor_optimizer.pc_backward_ppgrad([
                ppo_loss,
                self.cfg["torque_clone_loss_weight"] * torque_clone,
            ])

            nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm)
            self.actor_optimizer.step()

            # Recompute the auxiliary graph after PCGrad consumes the PPO
            # graph. The shared optimizer updates encoder and both decoders
            # exactly once from their combined objective.
            aux = self._compute_vae_loss(
                obs_hist_batch=batch["histories"],
                obs_target=batch["next_privileged"], labels=batch["explicit_targets"], valid=valid,
                iteration=iteration,
            )

            # Exact PPO_PACT boot-stat accumulation, applied separately to the
            # latent decoder target and to the explicit estimator target.
            for prefix, prediction_key, target_key in (
                ("latent", "privileged_prediction", "privileged_target"),
                ("explicit", "explicit_prediction", "explicit_target"),
            ):
                target, prediction = aux[target_key], aux[prediction_key]
                x, r = target * aux["valid"], prediction * aux["valid"]
                sum_x, sum_x2, sum_recon_sqerr, count = boot_stats[prefix]
                if sum_x is None:
                    sum_x = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float64)
                    sum_x2 = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float64)
                x64, r64 = x.to(torch.float64), r.to(torch.float64)
                sum_x += x64.sum(dim=0)
                sum_x2 += (x64 * x64).sum(dim=0)
                sum_recon_sqerr += ((r64 - x64) ** 2).sum().item()
                boot_stats[prefix] = [sum_x, sum_x2, sum_recon_sqerr, count + x.shape[0]]

            # Log metrics
            for name, val in (("value", value), ("surrogate", surrogate), ("base_velo", aux["base_velocity"]),
                              ("ee_position", aux["ee_position"]),
                              ("base_wrench", aux["base_wrench"]), ("ee_force", aux["ee_force"]),
                              ("foot_contact", aux["foot_contact"]),
                              ("foot_height", aux["foot_height"]),
                              ("privileged_force", aux["privileged_force"]), ("privileged_decoder", aux["privileged_decoder"]),
                              *((name, aux[name]) for name in KLRateBandController.METRIC_NAMES),
                              ("torque_clone", torque_clone),
                              ("film_identity", film_identity)):
                metrics[name] += val.detach().item()
            updates += 1

            self.spectral_normalization(self.actor_critic, sigma_max=10.0)

        def sample_boot_flag(stats):
            """Verbatim PPO_PACT mean-baseline and Bernoulli calculation."""
            boot_sum_x, boot_sum_x2, boot_sum_recon_sqerr, boot_count = stats
            feat_dim = boot_sum_x.shape[0]
            mean_pred = boot_sum_x / boot_count
            ex2 = boot_sum_x2 / boot_count
            var = torch.clamp(ex2 - mean_pred**2, min=0.0)
            mean_pred_error = var.mean().item()
            actual_pred_error = boot_sum_recon_sqerr / (boot_count * feat_dim)
            ratio = mean_pred_error / (actual_pred_error * self.boot_mult + 1e-8)
            pboot = np.tanh(ratio)
            return random.random() < pboot, pboot, mean_pred_error, actual_pred_error

        def reconstruction_errors(stats):
            """Return the same baseline/reconstruction MSE without RNG use."""
            sum_x, sum_x2, sum_recon_sqerr, count = stats
            feat_dim = sum_x.shape[0]
            mean_pred = sum_x / count
            variance = torch.clamp(sum_x2 / count - mean_pred.square(), min=0.0)
            return variance.mean().item(), sum_recon_sqerr / (count * feat_dim)

        explicit_baseline, explicit_recon = reconstruction_errors(boot_stats["explicit"])

        # Preserve PPO_PACT's latent Bernoulli bootstrap and its RNG timing.
        # Latent bootstrap quality remains diagnostic-only. Do not restore the
        # old Bernoulli assignment: the actor always uses its encoded history z.
        # self.use_boot_latent, latent_pboot, latent_baseline, latent_recon = sample_boot_flag(boot_stats["latent"])
        _, latent_pboot, latent_baseline, latent_recon = sample_boot_flag(boot_stats["latent"])
        self.use_boot_latent = True
        self.storage.clear()
        diagnostics["lr_after_update"] = self.learning_rate
        return {key: value / max(1, updates) for key, value in metrics.items()} | diagnostics | {
            "latent_boot_probability": latent_pboot,
            "latent_boot_active": float(self.use_boot_latent),
            "latent_recon_mse": latent_recon, "latent_naive_mse": latent_baseline,
            "explicit_recon_mse": explicit_recon, "explicit_naive_mse": explicit_baseline,
        }
