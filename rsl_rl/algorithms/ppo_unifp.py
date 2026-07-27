import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from rsl_rl.modules.actor_critic_unifp import ActorCriticUniFP
from rsl_rl.storage.rollout_storage_unifp import RolloutStorageUniFP


class Adaptation_Args():
    adaptation_module_learning_rate = 1.e-5
    num_adaptation_module_substeps = 1
    adaptation_batch_size = 64


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
                 device='cpu',
                 **kwargs,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

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
        # self.ppo_parameters = [
        #     *self.actor_critic.actor_body.parameters(),
        #     *self.actor_critic.critic_body.parameters(),
        #     self.actor_critic.std,
        # ]
        self.adaptation_module_parameters = [
            *self.actor_critic.adaptation_encoder_module.parameters(),
            *self.actor_critic.adaptation_decoder_module.parameters(),
        ]

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.adaptation_module_optimizer = optim.Adam(
            self.adaptation_module_parameters,
            lr=Adaptation_Args.adaptation_module_learning_rate,
        )
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

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_pred_shape, action_shape):
        self.storage = RolloutStorageUniFP(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_pred_shape, action_shape, self.device)

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
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.observation_preds = obs_pred
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
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

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, obs_pred_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
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
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                # PPO backprop still traverses the adaptation encoder to build
                # actor latents; clear those non-stepped gradients immediately.
                self.adaptation_module_optimizer.zero_grad()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

                data_size = critic_obs_batch.shape[0]
                num_train = int(data_size // 5 * 4)

                # Adaptation module gradient step, only update concurrent state estimation module, not policy network
                if len(self.adaptation_labels) > 0:

                    for epoch in range(Adaptation_Args.num_adaptation_module_substeps):

                        adaptation_pred = self.actor_critic.get_student_latent(obs_batch)
                        with torch.no_grad():
                            adaptation_target = obs_pred_batch
                        adaptation_loss = 0
                        for idx, (label, length, weight) in enumerate(zip(self.adaptation_labels, self.adaptation_dims, self.adaptation_weights)):

                            start, end = label_start_end[label]
                            selection_indices = torch.linspace(start, end - 1, steps=end - start, dtype=torch.long)

                            idx_adaptation_loss = weight*F.mse_loss(adaptation_pred[:, selection_indices],
                                                            adaptation_target[:, selection_indices])
                            mean_adaptation_losses[label] += idx_adaptation_loss.item()

                            adaptation_loss += idx_adaptation_loss

                        self.adaptation_module_optimizer.zero_grad()
                        adaptation_loss.backward()
                        self.adaptation_module_optimizer.step()

                        mean_adaptation_module_loss += adaptation_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= (num_updates * Adaptation_Args.num_adaptation_module_substeps)
        for label in self.adaptation_labels:
            mean_adaptation_losses[label] /= (num_updates * Adaptation_Args.num_adaptation_module_substeps)
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_adaptation_losses
