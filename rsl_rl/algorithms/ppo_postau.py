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
import torch.nn.functional as F
from torch import linalg as LA
import time

import numpy as np
import random
import gc

from rsl_rl.modules import ActorCritic_PosTau, ContextDecoder
from rsl_rl.storage import RolloutStoragePosTau

from .pc_grad import PCGrad

class PPO_PosTau:
    actor_critic: ActorCritic_PosTau
    decoder_network: ContextDecoder
    def __init__(self,
                 actor_critic,
                 decoder_network,
                 num_priv_obs,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.99,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 use_spo=False,
                 num_encoder_epochs=1, # number of epochs for hybrid encoder via supervised learning
                 vae_kld_weight=2.0,   # weight of KL divergence loss in VAE
                 ):
        
        self.device = device

        self.num_priv_obs = num_priv_obs

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.num_enc_epochs = num_encoder_epochs
        self.vae_beta = vae_kld_weight

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later

        self.act_optimizer, self.enc_optimizer = actor_critic.configure_optimizers(learning_rate)
        self.transition = RolloutStoragePosTau.Transition()

        # # We want to reduce the LR of the critic
        for param_group in self.act_optimizer.param_groups:
        # for param_group in self.act_optimizer.param_groups:
            # specifically modifies the learning rate of the position-control specific parameters
            if "name" in param_group.keys():
                if "critic" in param_group["name"]:
                    param_group['lr'] = (learning_rate / 3.0)

        self.decoder = decoder_network
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        self.boot_mult = 1.0
        self.use_boot = False

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
        
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, priv_obs_shape, obs_hist_shape, action_shape, torso_velo_shape):
        self.storage = RolloutStoragePosTau(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, priv_obs_shape, obs_hist_shape, \
                                              action_shape, torso_velo_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, obs_history):
        # if self.actor_critic.is_recurrent:
        #     self.transition.hidden_states = self.actor_critic.get_hidden_states()
        if self.use_boot:
            all_actions = self.actor_critic.act(obs,obs_history).detach()
        else:
            all_actions = self.actor_critic.act_bootmask(obs,obs_history).detach()

        # Compute the actions and values
        #  - Position Control
        self.transition.actions =  all_actions
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.observation_history = obs_history
        self.transition.critic_observations = critic_obs

        return all_actions
    
    def process_env_step(self, rewards, dones, infos, obs_labels, explicit_labels):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones


        # This is now the stack of critic observations, we want to prune off the last one
        self.transition.obs_targets = obs_labels[:, -self.num_priv_obs:]
        # self.transition.obs_targets = obs_labels

        self.transition.explicit_labels = explicit_labels
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)


    def set_entropy_coef(self, coef=1e-3):
        self.entropy_coef = coef
        
    def _set_std_clip_lwr(self, clip_val=0.1):
        self.actor_critic._set_std_clip_lwr(clip_val)

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

    def update(self, action_func, fb_func, dt, itr, default_pose, qvel_scale):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_autoenc_loss = 0
        mean_vel_loss = 0
        mean_recon_loss = 0
        mean_kld_loss = 0
        mean_decoder_loss = 0
        mean_pinn_loss = 0

        timers = {
            "rl_loss": 0.0,
            "pc_backward": 0.0,
            "act_step": 0.0,
            "vae_loss": 0.0,
            "enc_step": 0.0,
            "dec_step": 0.0,
            "boot_stats": 0.0,
            "boot_prob": 0.0,
            "spec_norm" : 0.0}

        boot_count = 0
        boot_sum_x = None
        boot_sum_x2 = None
        boot_sum_recon_sqerr = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for terminated_batch, obs_batch, critic_obs_batch, obs_hist_batch, explicit_labels_batch, \
            obs_target, actions_batch, target_values_batch, \
            advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, \
            old_sigma_batch  in generator:
            
            self.actor_critic.train()
            self.act_optimizer.zero_grad()
            self.enc_optimizer.zero_grad()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            # Perform RL update
            ppo_loss, surrogate_loss, value_loss, current_actions = self._compute_rl_loss(obs_batch, obs_hist_batch, actions_batch,
                                                                                          critic_obs_batch, old_sigma_batch, old_mu_batch,
                                                                                          old_actions_log_prob_batch,
                                                                                          advantages_batch, target_values_batch, returns_batch,
                                                                                          action_func, fb_func, default_pose, dt, qvel_scale)
            
            torch.cuda.synchronize()
            timers["rl_loss"] += time.perf_counter() - t0
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            ppo_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.act_optimizer.step()

            torch.cuda.synchronize()
            timers["act_step"] += time.perf_counter() - t0
            
            # Perform some logging
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            # Calculate the encoder update n-times
            for _ in range(self.num_enc_epochs):
                ###
                #  Update encoder with frozen decoder
                ###
                self.actor_critic.train()
                self.decoder.eval()

                torch.cuda.synchronize()
                t0 = time.perf_counter()

                # Calculate the DreamWaQ-style VAE update
                vae_loss, kl_div, recon_error, vel_pred_error, dec_input, decode_targets, recons = self._compute_vae_loss(obs_hist_batch, 
                                                                                                                          obs_target, 
                                                                                                                          explicit_labels_batch, 
                                                                                                                          terminated_batch)
                
                timers["vae_loss"] += time.perf_counter() - t0
                

                torch.cuda.synchronize()
                t0 = time.perf_counter()

                # Update paramaters of encoder
                self.enc_optimizer.zero_grad()
                vae_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.context_encoder.parameters(), self.max_grad_norm)
                self.enc_optimizer.step()
                
                timers["enc_step"] += time.perf_counter() - t0

                ###
                #  Update decoder with frozen encoder
                ###
                self.actor_critic.eval()
                self.decoder.train()
                self.decoder_optimizer.zero_grad()

                torch.cuda.synchronize()
                t0 = time.perf_counter()

                dec_recon = self.decoder(dec_input)
                dec_loss = F.mse_loss(dec_recon, decode_targets)
                dec_loss.backward()
                nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)
                self.decoder_optimizer.step()

                timers["dec_step"] += time.perf_counter() - t0

                torch.cuda.synchronize()
                t0 = time.perf_counter()

                # Log the decode targets and recons for computing boot-probability
                with torch.no_grad():
                    x = decode_targets * terminated_batch
                    r = recons * terminated_batch

                    # flatten batch dimension only; keep feature dim
                    # assumes x shape [B, D]
                    if boot_sum_x is None:
                        boot_sum_x = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float64)
                        boot_sum_x2 = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float64)

                    x64 = x.to(torch.float64)
                    r64 = r.to(torch.float64)

                    boot_sum_x += x64.sum(dim=0)
                    boot_sum_x2 += (x64 * x64).sum(dim=0)

                    # scalar sum over all elements
                    boot_sum_recon_sqerr += ((r64 - x64) ** 2).sum().item()

                    boot_count += x.shape[0]


                timers["boot_stats"] += time.perf_counter() - t0

                # Log losses
                mean_autoenc_loss += vae_loss.item()
                mean_vel_loss += vel_pred_error.item()
                mean_recon_loss += recon_error.item()
                mean_kld_loss += kl_div.item()
                mean_decoder_loss += dec_loss.item()

            # Keeps the interaction of incoming data with layer wieghts below the threashold that 
            #     saturates the tanh activation function.
            self.spectral_normalization(self.actor_critic, sigma_max=6.0)

            timers["spec_norm"] += time.perf_counter() - t0

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        mean_autoenc_loss /= (num_updates * self.num_enc_epochs)
        mean_decoder_loss /= (num_updates * self.num_enc_epochs)
        mean_kld_loss /= (num_updates * self.num_enc_epochs)
        mean_vel_loss /= (num_updates * self.num_enc_epochs)
        mean_recon_loss /= (num_updates * self.num_enc_epochs)


        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # # Calculate the total bootstrapping probability over the performance of the autoencoder on all of the above
        # mean_pred = np.mean(all_enc_obs_targets, axis=0)
        # mean_pred_error = np.mean(np.square(mean_pred - all_enc_obs_targets))
        # actual_pred_error = np.mean(np.square(np.array(all_enc_recons) - np.array(all_enc_obs_targets)))
        # ratio = mean_pred_error / (actual_pred_error * self.boot_mult)
        # pboot = np.tanh(ratio)

        # total number of scalar elements per sample vector
        feat_dim = boot_sum_x.shape[0]

        mean_pred = boot_sum_x / boot_count                     # [D]
        ex2 = boot_sum_x2 / boot_count                          # [D]
        var = torch.clamp(ex2 - mean_pred**2, min=0.0)          # [D]

        mean_pred_error = var.mean().item()

        actual_pred_error = boot_sum_recon_sqerr / (boot_count * feat_dim)

        ratio = mean_pred_error / (actual_pred_error * self.boot_mult + 1e-8)
        pboot = np.tanh(ratio)

        timers["boot_prob"] += time.perf_counter() - t0

        # Use the (scaled) ratio of mean-prediction performance to actual prediction performance
        #     to determine if encoder bootstrapping is performed.
        self.use_boot = random.random() < pboot
        print("Use bootstrapped Encoder Dynamics: ", self.use_boot)

        self.storage.clear()


        # Get the average time for the various tracked timers
        for key in timers.keys():
            if "boot_prob" not in key:
                timers[key] /= num_updates


        print("update timers:", {k: round(v, 4) for k, v in timers.items()})

        return mean_value_loss, mean_surrogate_loss, mean_autoenc_loss, mean_decoder_loss, \
               mean_vel_loss, mean_recon_loss, mean_kld_loss

    def _compute_rl_loss(self, obs_batch, obs_hist_batch,
                         actions_batch, critic_obs_batch,
                         old_sigma_batch, old_mu_batch,
                         old_actions_log_prob_batch,
                         advantages_batch, target_values_batch, returns_batch,
                         action_func, fb_func, default_pose, dt, qvel_scale):
        if self.use_boot:
            self.actor_critic.act(obs_batch, obs_hist_batch)
        else:
            self.actor_critic.act_bootmask(obs_batch, obs_hist_batch)

        # Pull out the current actions for use later
        current_actions = torch.cat([self.actor_critic.mean_pos, self.actor_critic.mean_tau], dim=-1)

        # PPO stuff
        #    - Position Control
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        value_batch            = self.actor_critic.evaluate(critic_obs_batch)
        mu_batch               = self.actor_critic.action_mean
        sigma_batch            = self.actor_critic.action_std
        entropy_batch          = self.actor_critic.entropy

        # Now calculate the PPO/SPO losses
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
                
                for param_group in self.act_optimizer.param_groups:
                    # specifically modifies the learning rate of the actor-control specific parameters
                    if "name" in param_group.keys():
                        if "actor" in param_group["name"]:
                            param_group['lr'] = self.learning_rate

        # PPO Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # SPO loss
        # surrogate_loss = -(torch.squeeze(advantages_batch) * ratio - torch.abs(torch.squeeze(advantages_batch)) * torch.pow(ratio - 1.0, 2) / (2.0 * self.clip_param)).mean()

        # PPO stuff
        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

        return ppo_loss, surrogate_loss, value_loss, current_actions

    def _compute_vae_loss(self, obs_hist_batch, grf_target, 
                          obs_target, explicit_labels_batch, terminated_batch):
        vae_loss = None
        
        mean_latent, logvar_latent, cenet_latent, cenet_torso_velo = self.actor_critic.context_encoder(obs_hist_batch)
        
        dec_input = torch.cat((cenet_latent, cenet_torso_velo), dim=-1)
        enc_update_obs_decode = None
        
        # with torch.no_grad():
        enc_update_obs_decode = self.decoder(dec_input)
        
        grf_target.requires_grad = False
        obs_target.requires_grad = False
        
        # decode_target = torch.cat((obs_target, grf_target), dim=-1)
        decode_target = obs_target
        explicit_labels_batch.requires_grad = False

        vel_pred_error = F.mse_loss(cenet_torso_velo*terminated_batch,explicit_labels_batch*terminated_batch)
        recon_error    = F.mse_loss(enc_update_obs_decode*terminated_batch,decode_target*terminated_batch)
        kl_div         = -0.5*torch.mean(torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp(), dim=-1)*terminated_batch.squeeze(-1).float())
        # kl_div         = -0.5*torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp())
        vae_loss = vel_pred_error + recon_error + self.vae_beta*kl_div
        
        return vae_loss, kl_div, recon_error, vel_pred_error, dec_input.clone().detach(), decode_target.detach(), enc_update_obs_decode.detach()