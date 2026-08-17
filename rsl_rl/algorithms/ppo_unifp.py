import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from rsl_rl.algorithms.kl_rate_band import KLRateBandController, update_duals_from_mean
from rsl_rl.modules.actor_critic_unifp import ActorCriticUniFP
from rsl_rl.storage.rollout_storage_unifp import RolloutStorageUniFP


class Adaptation_Args():
    adaptation_module_learning_rate = 1.e-5


class PPO_UniFP:
    actor_critic: ActorCriticUniFP

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 use_adaptive_entropy=False,
                 adaptive_ent_bounds=(0.005, 0.01),
                 adaptive_ent_lin_threshold=0.75,
                 adaptive_ent_ang_threshold=0.35,
                 adaptive_ent_ter_threshold=6.0,
                 adaptive_ent_softmax_temp=2.0,
                 adaptation_privileged_weight=1.0,
                 adaptation_kl_weight=1.0e-3,
                 use_kl_rate_band=True,
                 use_cosine_kl_warmup=True,
                 kl_warmup_iters=500,
                 kl_warmup_beta_max=None,
                 kl_band_warmup_iters=500,
                 kl_r_min=0.10,
                 kl_r_max=1.00,
                 kl_dual_lr=1.0e-3,
                 kl_aug_rho=0.1,
                 kl_ema_decay=0.99,
                 num_encoder_epochs=1,
                 enable_additional_diagnostics=True,
                 device='cpu',
                 **kwargs,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.enable_additional_diagnostics = bool(enable_additional_diagnostics)
        if self.enable_additional_diagnostics:
            print("UniFP PPO - Learning Rate - ", self.learning_rate)
            print("UniFP PPO - Desired KL - ", self.desired_kl)
            print("UniFP PPO - LR Schedule - ", self.schedule)

        # Adaptive entropy coefficient, ported from PPO_PACT. Bounds are
        # ordered [coefficient at target performance, coefficient at maximum
        # performance gap], so weaker policies receive more exploration.
        if len(adaptive_ent_bounds) != 2:
            raise ValueError("adaptive_ent_bounds must contain [low, high]")
        if adaptive_ent_bounds[0] < 0.0 or adaptive_ent_bounds[1] < adaptive_ent_bounds[0]:
            raise ValueError("adaptive_ent_bounds must satisfy 0 <= low <= high")
        if adaptive_ent_softmax_temp <= 0.0:
            raise ValueError("adaptive_ent_softmax_temp must be greater than zero")
        self.use_adaptive_entropy = use_adaptive_entropy
        self.entropy_coef_bounds = tuple(float(value) for value in adaptive_ent_bounds)
        self.ent_linvelo_threshold = float(adaptive_ent_lin_threshold)
        self.ent_angvelo_threshold = float(adaptive_ent_ang_threshold)
        self.ent_terrain_threshold = float(adaptive_ent_ter_threshold)
        self.ent_softmax_temperature = float(adaptive_ent_softmax_temp)
        self.current_entropy_coef = float(entropy_coef)
        self.adaptation_privileged_weight = float(adaptation_privileged_weight)
        self.num_enc_epochs = int(num_encoder_epochs)
        if self.num_enc_epochs < 1:
            raise ValueError("num_encoder_epochs must be at least 1")


        self.adaptation_kl_weight = float(adaptation_kl_weight)
        self.use_kl_rate_band = bool(use_kl_rate_band)
        self.use_cosine_kl_warmup = bool(use_cosine_kl_warmup)
        self.kl_controller = KLRateBandController(
            warmup_iters=kl_warmup_iters,
            warmup_beta_max=(
                self.adaptation_kl_weight
                if kl_warmup_beta_max is None else kl_warmup_beta_max
            ),
            band_warmup_iters=kl_band_warmup_iters,
            rate_min=kl_r_min, rate_max=kl_r_max, dual_lr=kl_dual_lr,
            augmented_rho=kl_aug_rho, ema_decay=kl_ema_decay,
        )

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)

        self.adaptation_labels = self.actor_critic.adaptation_labels
        self.adaptation_dims = self.actor_critic.adaptation_dims
        self.adaptation_weights = self.actor_critic.adaptation_weights

        self.storage = None # initialized later

        # Keep PPO and CSE/adaptation optimization separated. PPO should only
        # step the policy mean, value function, and learned action std; the
        # supervised adaptation update owns the encoder/decoder latent path.
        self.ppo_parameters = [
            *self.actor_critic.actor_body.parameters(),
            *self.actor_critic.critic_body.parameters(),
            *self.actor_critic.adaptation_encoder_module.parameters(),
            *self.actor_critic.adaptation_mean_module.parameters(),
            *self.actor_critic.adaptation_logvar_module.parameters(),
            self.actor_critic.std,
        ]
        self.adaptation_module_parameters = [
            *self.actor_critic.adaptation_encoder_module.parameters(),
            *self.actor_critic.adaptation_mean_module.parameters(),
            *self.actor_critic.adaptation_logvar_module.parameters(),
            *self.actor_critic.adaptation_decoder_module.parameters(),
            *self.actor_critic.privileged_decoder_module.parameters(),
        ]

        self.optimizer = optim.Adam(self.ppo_parameters, lr=learning_rate)
        
        self.adaptation_module_optimizer = optim.Adam(
            self.adaptation_module_parameters,
            lr=Adaptation_Args.adaptation_module_learning_rate,
        )
        self.diagnostic_parameter_groups = {
            "actor": list(self.actor_critic.actor_body.parameters()),
            "critic": list(self.actor_critic.critic_body.parameters()),
            "encoder": [
                *self.actor_critic.adaptation_encoder_module.parameters(),
                *self.actor_critic.adaptation_mean_module.parameters(),
                *self.actor_critic.adaptation_logvar_module.parameters(),
            ],
            "std": [self.actor_critic.std],
        }
        self.transition = RolloutStorageUniFP.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    @staticmethod
    @torch.no_grad()
    def _gradient_norm(parameters):
        squared_norm = None
        for parameter in parameters:
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().square().sum()
            squared_norm = value if squared_norm is None else squared_norm + value
        return 0.0 if squared_norm is None else torch.sqrt(squared_norm).item()

    @staticmethod
    @torch.no_grad()
    def _relative_update_norm(parameters, before):
        update_sq = before[0].new_zeros(()) if before else torch.tensor(0.0)
        parameter_sq = update_sq.clone()
        for parameter, previous in zip(parameters, before):
            update_sq += (parameter.detach() - previous).float().square().sum()
            parameter_sq += previous.float().square().sum()
        return (torch.sqrt(update_sq) / (torch.sqrt(parameter_sq) + 1.0e-12)).item()

    def _shared_optimizer_metrics(self):
        optimizer_parameters = []
        for optimizer in (self.optimizer, self.adaptation_module_optimizer):
            optimizer_parameters.append({
                id(parameter): parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            })
        shared_ids = set(optimizer_parameters[0]).intersection(optimizer_parameters[1])
        return {
            "Debug/shared_optimizer_parameter_tensors": float(len(shared_ids)),
            "Debug/shared_optimizer_parameter_count": float(sum(
                optimizer_parameters[0][identifier].numel() for identifier in shared_ids
            )),
        }

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_pred_shape, next_privileged_shape, action_shape):
        self.storage = RolloutStorageUniFP(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
            obs_pred_shape, next_privileged_shape, action_shape, self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def set_entropy_coef(self, coef=1.0e-3):
        if self.use_adaptive_entropy:
            self.current_entropy_coef = float(coef)
        else:
            self.entropy_coef = float(coef)

    def update_adaptive_entropy_coef(self, performance_metrics):
        """Update exploration strength from B1 locomotion performance."""
        lin_vel_tracking = float(performance_metrics.get("lin_vel_tracking", 0.0))
        ang_vel_tracking = float(performance_metrics.get("ang_vel_tracking", 0.0))
        terrain_level = float(performance_metrics.get("terrain_level", 0.0))

        def normalized_gap(value, threshold):
            if threshold <= 0.0:
                return 0.0
            return max(0.0, threshold - value) / threshold

        gaps = torch.tensor(
            [
                normalized_gap(lin_vel_tracking, self.ent_linvelo_threshold),
                normalized_gap(ang_vel_tracking, self.ent_angvelo_threshold),
                normalized_gap(terrain_level, self.ent_terrain_threshold),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        weights = F.softmax(gaps / self.ent_softmax_temperature, dim=0)
        weighted_gap = torch.sum(weights * gaps).item()
        low, high = self.entropy_coef_bounds
        self.current_entropy_coef = low + weighted_gap * (high - low)
        return self.current_entropy_coef

    def act(self, obs, critic_obs, obs_pred):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs.detach().clone()
        self.transition.critic_observations = critic_obs.detach().clone()
        self.transition.observation_preds = obs_pred.detach().clone()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_privileged_observations):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_privileged_observations = next_privileged_observations
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
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

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, iteration):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0

        mean_adaptation_losses = {}
        label_start_end = {}
        si = 0
        for idx, (label, length) in enumerate(zip(self.adaptation_labels, self.adaptation_dims)):
            label_start_end[label] = (si, si + length)
            si = si + length
            mean_adaptation_losses[label] = 0
        mean_adaptation_losses["next_privileged_loss"] = 0
        for name in KLRateBandController.METRIC_NAMES:
            mean_adaptation_losses[name] = 0

        ppo_diagnostics = {}
        parameters_before = {}
        if self.enable_additional_diagnostics:
            # Diagnostic rollout statistics and parameter snapshots are optional.
            with torch.no_grad():
                advantages = self.storage.advantages.detach().float()
                returns = self.storage.returns.detach().float()
                values = self.storage.values.detach().float()
                return_variance = returns.var(unbiased=False)
                explained_variance = (
                    1.0 - (returns - values).var(unbiased=False) / return_variance
                    if return_variance > 1.0e-12 else returns.new_zeros(())
                )
                ppo_diagnostics = {
                    "PPO/advantage_mean": advantages.mean().item(),
                    "PPO/advantage_std": advantages.std(unbiased=False).item(),
                    "PPO/return_mean": returns.mean().item(),
                    "PPO/value_mean": values.mean().item(),
                    "PPO/explained_variance": explained_variance.item(),
                }
                parameters_before = {
                    name: [parameter.detach().clone() for parameter in parameters]
                    for name, parameters in self.diagnostic_parameter_groups.items()
                }

        minibatch_sums = {
            "PPO/approx_kl": 0.0,
            "PPO/ratio_mean": 0.0,
            "PPO/ratio_std": 0.0,
            "PPO/clip_fraction": 0.0,
        }
        gradient_sums = {name: 0.0 for name in self.diagnostic_parameter_groups}
        gradient_counts = {name: 0 for name in self.diagnostic_parameter_groups}
        raw_kl_sum = 0.0
        raw_kl_count = 0

        if self.enable_additional_diagnostics:
            ppo_diagnostics["PPO/lr_before_update"] = self.learning_rate

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        first_epoch = True
        for obs_batch, critic_obs_batch, obs_pred_batch, next_privileged_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, dones_batch in generator:


                self.actor_critic.act(obs_batch, masks=dones_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=dones_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy


                if first_epoch and self.enable_additional_diagnostics:
                    mu_abs_diff = (mu_batch - old_mu_batch).abs()
                    sigma_abs_diff = (sigma_batch - old_sigma_batch).abs()
                    logprob_abs_diff = (
                        actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1)
                    ).abs()

                    ppo_diagnostics["PPO/mu_abs_diff"] = mu_abs_diff.mean().item()
                    ppo_diagnostics["PPO/sigma_abs_diff"] = sigma_abs_diff.mean().item()
                    ppo_diagnostics["PPO/logprob_abs_diff"] = logprob_abs_diff.mean().item()


                # KL
                approximate_kl = None
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        approximate_kl = kl_mean

                        if first_epoch and self.enable_additional_diagnostics:
                            ppo_diagnostics["PPO/pre_update_kl"] = kl_mean.item()
                        if first_epoch:
                            first_epoch = False
                        
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate



                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                if self.enable_additional_diagnostics:
                    with torch.no_grad():
                        minibatch_sums["PPO/approx_kl"] += approximate_kl.item()
                        minibatch_sums["PPO/ratio_mean"] += ratio.mean().item()
                        minibatch_sums["PPO/ratio_std"] += ratio.std(unbiased=False).item()
                        minibatch_sums["PPO/clip_fraction"] += (
                            (ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param)
                        ).float().mean().item()
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                entropy_coef = self.current_entropy_coef if self.use_adaptive_entropy else self.entropy_coef
                loss = surrogate_loss + self.value_loss_coef * value_loss - entropy_coef * entropy_batch.mean()

                # Gradient step. Clip only PPO-owned parameters so encoder
                # gradients produced by the policy latent path cannot shrink the
                # actor/critic/std update norm.
                self.optimizer.zero_grad()
                loss.backward()
                if self.enable_additional_diagnostics:
                    for name, parameters in self.diagnostic_parameter_groups.items():
                        gradient_sums[name] += self._gradient_norm(parameters)
                        gradient_counts[name] += 1
                nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm)
                self.optimizer.step()
                # PPO backprop still traverses the adaptation encoder to build
                # actor latents; clear those non-stepped gradients immediately.
                self.adaptation_module_optimizer.zero_grad()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

                # Adaptation module gradient step: each encoder epoch reuses
                # this complete PPO minibatch without adaptation sub-batching.
                if len(self.adaptation_labels) > 0:

                    for _ in range(self.num_enc_epochs):

                        adaptation_output = self.actor_critic.get_student_latent(obs_batch)
                        with torch.no_grad():
                            adaptation_target = obs_pred_batch
                            next_privileged_target = next_privileged_batch
                        mean, logvar, _, adaptation_pred, privileged_pred = adaptation_output
                        adaptation_loss = 0
                        for idx, (label, length, weight) in enumerate(zip(self.adaptation_labels, self.adaptation_dims, self.adaptation_weights)):

                            start, end = label_start_end[label]
                            if label == "foot_contact_loss":
                                # Decoder outputs logits; BCE supplies the
                                # correct binary contact-state objective.
                                idx_adaptation_loss = weight * F.binary_cross_entropy_with_logits(
                                    adaptation_pred[:, start:end], adaptation_target[:, start:end]
                                )
                            else:
                                idx_adaptation_loss = weight * F.mse_loss(
                                    adaptation_pred[:, start:end], adaptation_target[:, start:end]
                                )
                            mean_adaptation_losses[label] += idx_adaptation_loss.item()

                            adaptation_loss += idx_adaptation_loss

                        # The target is one post-action privileged frame, not
                        # the critic's full temporal stack. Terminal transitions
                        # are excluded because reset state is not the successor
                        # of the pre-reset observation encoded above.
                        valid = (~dones_batch.bool()).float()
                        privileged_error = (privileged_pred - next_privileged_target).square() * valid
                        privileged_loss = privileged_error.sum() / (
                            valid.sum().clamp_min(1.0) * privileged_pred.shape[-1]
                        )
                        kl_loss = -0.5 * (
                            1.0 + logvar - mean.square() - logvar.exp()
                        ).sum(dim=-1, keepdim=True)
                        kl_loss = (kl_loss * valid).sum() / valid.sum().clamp_min(1.0)

                        kl_reg_loss = self.kl_controller.loss(
                            kl_loss, iteration, self.use_kl_rate_band,
                            self.use_cosine_kl_warmup,
                        )

                        adaptation_loss += (
                            self.adaptation_privileged_weight * privileged_loss
                            + kl_reg_loss
                        )
                        mean_adaptation_losses["next_privileged_loss"] += privileged_loss.item()

                        self.adaptation_module_optimizer.zero_grad()
                        adaptation_loss.backward()
                        if self.enable_additional_diagnostics:
                            gradient_sums["encoder"] += self._gradient_norm(
                                self.diagnostic_parameter_groups["encoder"]
                            )
                            gradient_counts["encoder"] += 1
                        self.adaptation_module_optimizer.step()
                        mean_adaptation_losses["kl_raw"] += kl_loss.detach().item()
                        mean_adaptation_losses["kl_reg_loss"] += kl_reg_loss.detach().item()
                        raw_kl_sum += kl_loss.detach().item()
                        raw_kl_count += 1

                        mean_adaptation_module_loss += adaptation_loss.item()

                # Keeps the interaction of incoming data with layer wieghts below the threashold that 
                #     saturates the tanh activation function.
                self.spectral_normalization(self.actor_critic, sigma_max=10.0)

        if self.enable_additional_diagnostics:
            ppo_diagnostics["PPO/lr_after_update"] = self.learning_rate

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= (num_updates * self.num_enc_epochs)
        for label in mean_adaptation_losses:
            mean_adaptation_losses[label] /= (num_updates * self.num_enc_epochs)

        mean_raw_kl = update_duals_from_mean(
            self.kl_controller, raw_kl_sum, raw_kl_count, iteration, self.device,
            enabled=self.use_kl_rate_band,
            use_cosine_warmup=self.use_cosine_kl_warmup,
        )
        if mean_raw_kl is not None:
            controller_metrics = self.kl_controller.metrics(
                mean_raw_kl,
                torch.tensor(mean_adaptation_losses["kl_reg_loss"], device=self.device),
                iteration, self.use_kl_rate_band,
                self.use_cosine_kl_warmup,
            )
            for name in KLRateBandController.METRIC_NAMES:
                if name not in ("kl_raw", "kl_reg_loss"):
                    mean_adaptation_losses[name] = controller_metrics[name].item()

        if self.enable_additional_diagnostics:
            ppo_diagnostics["PPO/param_learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
            ppo_diagnostics["PPO/class_learning_rate"] = float(self.learning_rate)
            for name, value in minibatch_sums.items():
                ppo_diagnostics[name] = value / max(1, num_updates)
            for name, parameters in self.diagnostic_parameter_groups.items():
                ppo_diagnostics[f"PPO/{name}_grad_norm"] = (
                    gradient_sums[name] / max(1, gradient_counts[name])
                )
                ppo_diagnostics[f"PPO/{name}_relative_update_norm"] = self._relative_update_norm(
                    parameters, parameters_before[name]
                )
            ppo_diagnostics.update(self._shared_optimizer_metrics())
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_adaptation_module_loss,
            mean_adaptation_losses,
            ppo_diagnostics,
        )
