# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCriticDreamWaQ
from rsl_rl.storage import RolloutStorageDreamWaQ

'''
PPO with DreamWaQ
'''

class PPO_DreamWaQ(PPO):
    actor_critic: ActorCriticDreamWaQ

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
                 use_spo=False,
                 device='cpu',
                 encoder_lr=1e-3,      # learning rate for hybrid encoder
                 num_encoder_epochs=1, # number of epochs for hybrid encoder via supervised learning
                 explicit_loss_weight=1.0,
                 contact_probability_loss_weight=1.0,
                 vae_kld_weight=1.0,   # weight of KL divergence loss in VAE
                 ):

        super().__init__(
            actor_critic,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            use_spo,
            device,
        )

        self.encoder_lr = encoder_lr
        self.num_encoder_epochs = num_encoder_epochs
        self.vae_kld_weight = vae_kld_weight
        self.explicit_loss_weight = explicit_loss_weight
        self.contact_probability_loss_weight = contact_probability_loss_weight
        self.estimation_metrics = {}
        if num_encoder_epochs < 1:
            raise ValueError("num_encoder_epochs must be positive")

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.rl_parameters = list(self.actor_critic.actor.parameters()) + \
                             list(self.actor_critic.critic.parameters()) + \
                            [self.actor_critic.std]
        self.optimizer = optim.Adam(
            self.rl_parameters, lr=learning_rate)
        self.vae_optimizer = optim.Adam(
            self.actor_critic.vae.parameters(), lr=encoder_lr)
        rl_ids = {id(p) for p in self.rl_parameters}
        vae_ids = {id(p) for p in self.actor_critic.vae.parameters()}
        if rl_ids & vae_ids or rl_ids | vae_ids != {id(p) for p in actor_critic.parameters()}:
            raise ValueError("PPO and VAE must exclusively own every model parameter")
        self.transition = RolloutStorageDreamWaQ.Transition()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, 
                     privileged_obs_shape, obs_history_shape, explicit_info_shape, next_states_shape, action_shape):
        self.storage = RolloutStorageDreamWaQ(
            num_envs, num_transitions_per_env, actor_obs_shape, 
            privileged_obs_shape, obs_history_shape, explicit_info_shape, next_states_shape, action_shape, self.device)

    def process_env_step(self, rewards, dones, infos, next_state):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_states = next_state.clone()
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * \
                torch.squeeze(self.transition.values *
                              infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def act(self, obs, privileged_obs, obs_history, explicit_info_labels):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, obs_history).detach()
        self.transition.values = self.actor_critic.evaluate(privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs.detach().clone()
        self.transition.privileged_observations = privileged_obs.detach().clone()
        self.transition.observation_histories = obs_history.detach().clone()
        self.transition.explicit_info_labels = explicit_info_labels.detach().clone()
        return self.transition.actions

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_explicit_estimation_loss = 0
        mean_reconstruction_loss = 0
        mean_kld_loss = 0
        metrics = {key: 0.0 for key in ("velocity", "contact", "height")}
        generator = self._get_data_generator()
        for obs_batch, privileged_obs_batch, obs_histories_batch, explicit_info_labels_batch, next_state_batch, \
            terminated_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            loss, surrogate_loss, value_loss = self._compute_rl_loss(
                obs_batch, privileged_obs_batch, obs_histories_batch, actions_batch,
                target_values_batch, advantages_batch, returns_batch,
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch)

            # Gradient step
            self.vae_optimizer.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.rl_parameters, self.max_grad_norm)
            self.optimizer.step()
            
            self.optimizer.zero_grad(set_to_none=True)
            # vae gradient step
            for _ in range(self.num_encoder_epochs):
                
                vae_loss, explicit_estimation_loss, reconstruction_loss, kld_loss = self._compute_vae_loss(
                    obs_histories_batch, terminated_batch, explicit_info_labels_batch, next_state_batch)
                
                self.vae_optimizer.zero_grad()
                vae_loss.backward()
                nn.utils.clip_grad_norm_(
                self.actor_critic.vae.parameters(), self.max_grad_norm)
                if terminated_batch.any():
                    self.vae_optimizer.step()  # Empty batches must not apply Adam momentum.
                
                for key in metrics:
                    metrics[key] += self.estimation_metrics[key]
                mean_explicit_estimation_loss += explicit_estimation_loss.item()
                mean_reconstruction_loss += reconstruction_loss.item()
                mean_kld_loss += kld_loss.item()
                
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_explicit_estimation_loss /= (num_updates * self.num_encoder_epochs)
        mean_reconstruction_loss /= (num_updates * self.num_encoder_epochs)
        mean_kld_loss /= (num_updates * self.num_encoder_epochs)
        self.estimation_metrics = {key: value / (num_updates * self.num_encoder_epochs)
                                   for key, value in metrics.items()}
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_explicit_estimation_loss, \
            mean_reconstruction_loss, mean_kld_loss

    def _compute_rl_loss(self, obs_batch, privileged_obs_batch, obs_histories_batch, actions_batch,
                         target_values_batch, advantages_batch, returns_batch,
                         old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch):
        self.actor_critic.act(
                obs_batch, obs_histories_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch)
        value_batch = self.actor_critic.evaluate(
                privileged_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        entropy_batch = self.actor_critic.entropy
        
        self._adjust_learning_rate(sigma_batch, old_sigma_batch, mu_batch, old_mu_batch)
        
        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch -
                          torch.squeeze(old_actions_log_prob_batch))
        surrogate_loss = self._compute_surrogate_loss(ratio, advantages_batch)
        
        # Value function loss
        value_loss = self._compute_value_function_loss(value_batch, returns_batch, target_values_batch)
        
        loss = surrogate_loss + self.value_loss_coef * \
                value_loss - self.entropy_coef * entropy_batch.mean()
        
        return loss, surrogate_loss, value_loss
    
    def _compute_vae_loss(self, obs_histories_batch, terminated_batch, explicit_info_labels_batch, next_state_batch):
        # Reset observations are not successors of the previous episode. Select
        # first (NaN * 0 is still NaN), and normalize by the valid sample count.
        rows = terminated_batch.reshape(-1).bool()
        history = obs_histories_batch[rows]
        targets = explicit_info_labels_batch.detach()[rows]
        next_states = next_state_batch.detach()[rows]
        (z, explicit), (mu, logvar, estimator) = self.actor_critic.vae(history)
        reconstructed = self.actor_critic.vae.decode(z, explicit)
        denominator = max(history.shape[0], 1)
        velocity = (explicit[:, :3] - targets[:, :3]).square()
        height = (explicit[:, 7:11] - targets[:, 7:11]).square()
        continuous_loss = torch.cat((velocity, height), dim=-1).mean(-1).sum() / denominator
        contact_loss = nn.functional.binary_cross_entropy_with_logits(
            estimator.contact_logits, targets[:, 3:7], reduction="none"
        ).mean(-1).sum() / denominator
        explicit_loss = continuous_loss + self.contact_probability_loss_weight * contact_loss
        reconstruction_loss = (reconstructed - next_states).square().mean(-1).sum() / denominator
        kld_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).sum() / denominator
        self.estimation_metrics = {
            "velocity": (velocity.sum() / (denominator * 3)).item(),
            "height": (height.sum() / (denominator * 4)).item(),
            "contact": contact_loss.item(),
        }
        loss = (reconstruction_loss + self.vae_kld_weight * kld_loss
                + self.explicit_loss_weight * explicit_loss)
        return loss, explicit_loss, reconstruction_loss, kld_loss
