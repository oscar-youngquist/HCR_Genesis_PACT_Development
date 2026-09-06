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
import math

import numpy as np
import random
import gc

from rsl_rl.modules import ActorCritic_PACT_Pos, ContextDecoder
from rsl_rl.modules.hard_pact_physics import (
    GRFDecoderMetricsAccumulator,
    normalized_grf_huber_loss,
    normalized_wrench_huber_loss,
    wrench_regression_metrics,
)
from rsl_rl.storage import RolloutStoragePACTPos

from .pc_grad import PCGrad

class PPO_PACT_Pos:
    actor_critic: ActorCritic_PACT_Pos
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
                 auxiliary_learning_rate=2.0e-4,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 use_spo=False,
                 num_encoder_epochs=1, # number of epochs for hybrid encoder via supervised learning
                 vae_kld_weight=1.0,   # weight of KL divergence loss in VAE
                 vae_kl_initial_weight=0.0,
                 vae_kl_warmup_start=0,
                 vae_kl_warmup_iterations=0,
                 use_adaptive_entropy=True,
                 adaptive_ent_bounds=[0.01, 0.001],
                 adaptive_ent_lin_threshold=0.75,
                 adaptive_ent_ang_threshold=0.35,
                 adaptive_ent_ter_threshold=5.0,
                 adaptive_ent_softmax_temp=2.0,
                 reconstruction_indices=None,
                 privileged_loss_weight=1.0,
                 explicit_loss_weight=1.0,
                 grf_loss_weight=1.0,
                 active_wrench_loss_weight=1.0,
                 neutral_wrench_loss_weight=0.25,
                 force_decoder_diagnostics_enabled=False,
                 ):
        
        self.device = device

        self.num_priv_obs = num_priv_obs
        self.reconstruction_indices = reconstruction_indices
        self.is_hard_pact_pos = hasattr(actor_critic, "physics_estimator")
        self.privileged_loss_weight = float(privileged_loss_weight)
        self.explicit_loss_weight = float(explicit_loss_weight)
        self.grf_loss_weight = float(grf_loss_weight)
        self.active_wrench_loss_weight = float(active_wrench_loss_weight)
        self.neutral_wrench_loss_weight = float(neutral_wrench_loss_weight)
        self.force_decoder_diagnostics_enabled = bool(
            force_decoder_diagnostics_enabled
        )
        self._grf_diagnostics = None
        self.last_auxiliary_metrics = {}

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.auxiliary_learning_rate = float(auxiliary_learning_rate)
        if self.auxiliary_learning_rate <= 0.0:
            raise ValueError("auxiliary_learning_rate must be positive")

        self.num_enc_epochs = num_encoder_epochs
        self.vae_beta = float(vae_kld_weight)
        self.vae_kl_initial_weight = float(vae_kl_initial_weight)
        self.vae_kl_warmup_start = int(vae_kl_warmup_start)
        self.vae_kl_warmup_iterations = int(vae_kl_warmup_iterations)
        if self.vae_kl_warmup_iterations < 0:
            raise ValueError("vae_kl_warmup_iterations must be nonnegative")
        # Before the first explicit iteration boundary, preserve the legacy
        # constant final weight. ``update`` freezes the scheduled value once
        # and every auxiliary minibatch in that PPO iteration reuses it.
        self.current_vae_beta = self.vae_beta

        # Adaptive entropy coefficent algorithm values
        self.use_adaptive_entropy = use_adaptive_entropy
        self.entropy_coef_bounds = adaptive_ent_bounds
        self.ent_linvelo_threshold = adaptive_ent_lin_threshold
        self.ent_angvelo_threshold = adaptive_ent_ang_threshold
        self.ent_terrain_threshold = adaptive_ent_ter_threshold
        self.ent_softmax_temperature = adaptive_ent_softmax_temp
        
        self.current_entropy_coef = entropy_coef
        self.entropy_coef = entropy_coef

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later

        self.act_optimizer, self.enc_optimizer = actor_critic.configure_optimizers(learning_rate)
        if self.is_hard_pact_pos:
            # HardPACTPos trains the context encoder, explicit estimator, GRF
            # decoder, wrench decoder, and privileged decoder as one auxiliary
            # system.  Keep every part on the same configured learning rate.
            for param_group in self.enc_optimizer.param_groups:
                param_group["lr"] = self.auxiliary_learning_rate
        self.transition = RolloutStoragePACTPos.Transition()

        self.act_optimizer = PCGrad(self.act_optimizer, reduction='sum')

        # # We want to reduce the LR of the critic
        for param_group in self.act_optimizer.optimizer.param_groups:
        # for param_group in self.act_optimizer.param_groups:
            # specifically modifies the learning rate of the position-control specific parameters
            if "name" in param_group.keys():
                if "critic" in param_group["name"]:
                    param_group['lr'] = (learning_rate / 3.0)

        self.decoder = decoder_network
        decoder_learning_rate = (
            self.auxiliary_learning_rate if self.is_hard_pact_pos else learning_rate
        )
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=decoder_learning_rate
        )

        self.boot_mult = 1.0
        self.use_boot = False

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, priv_obs_shape, obs_hist_shape, action_shape, torso_velo_shape, grf_shape):
        self.storage = RolloutStoragePACTPos(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, priv_obs_shape, obs_hist_shape, \
                                                               action_shape, torso_velo_shape, grf_shape, self.device,
                                                               hard_pact_auxiliary=self.is_hard_pact_pos)

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
    
    def process_env_step(self, rewards, dones, infos, grf_labels, obs_labels, explicit_labels):
        self.transition.rewards = rewards.clone()
        
        self.transition.dones = dones
        # Values from the next-time step used as labels for the decoder network
        self.transition.grf_targets = grf_labels

        # This is now the stack of critic observations, we want to prune off the last one
        reconstruction_target = obs_labels[:, -self.num_priv_obs:]
        if self.reconstruction_indices is not None:
            reconstruction_target = reconstruction_target[:, self.reconstruction_indices]
        self.transition.obs_targets = reconstruction_target
        # self.transition.obs_targets = obs_labels

        self.transition.explicit_labels = explicit_labels
        if self.is_hard_pact_pos:
            hard_pact = infos.get("hard_pact_transition")
            if hard_pact is None:
                raise RuntimeError(
                    "HardPACTPos auxiliary heads require hard_pact_transition labels"
                )
            self.transition.executed_torque_targets = hard_pact[
                "interval_executed_torque"
            ]
            self.transition.wrench_targets = hard_pact[
                "total_external_wrench_label_yaw_normalized"
            ]
            self.transition.wrench_active_masks = hard_pact[
                "sustained_wrench_active_mask"
            ].bool()
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def set_entropy_coef(self, coef=1e-3):
        if self.use_adaptive_entropy: 
            self.current_entropy_coef = coef
        else:
            self.entropy_coef = coef        
        
    def update_adaptive_entropy_coef(self, performance_metrics):
        lin_vel_tracking = performance_metrics.get('lin_vel_tracking', 0.0)
        ang_vel_tracking = performance_metrics.get('ang_vel_tracking', 0.0)
        terrain_level = performance_metrics.get('terrain_level', 0)
        
        lin_vel_gap = max(0, self.ent_linvelo_threshold - lin_vel_tracking)
        ang_vel_gap = max(0, self.ent_angvelo_threshold - ang_vel_tracking)
        terrain_gap = max(0, self.ent_terrain_threshold - terrain_level)
        
        norm_lin_gap = lin_vel_gap /  self.ent_linvelo_threshold if self.ent_linvelo_threshold > 0 else 0
        norm_ang_gap = ang_vel_gap / self.ent_angvelo_threshold if self.ent_angvelo_threshold > 0 else 0
        norm_terrain_gap = terrain_gap / self.ent_terrain_threshold if self.ent_terrain_threshold > 0 else 0
        
        gaps = torch.tensor([norm_lin_gap, norm_ang_gap, norm_terrain_gap], dtype=torch.float32)
        
        weights = F.softmax(gaps / self.ent_softmax_temperature, dim=0)
        
        weighted_gap = torch.sum(weights * gaps).item()
        
        self.current_entropy_coef = self.entropy_coef_bounds[0] + weighted_gap * (self.entropy_coef_bounds[1] - self.entropy_coef_bounds[0])
        
        return self.current_entropy_coef
        
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
        self.current_vae_beta = self._vae_beta_for_iteration(itr)
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_autoenc_loss = 0
        mean_vel_loss = 0
        mean_recon_loss = 0
        mean_kld_loss = 0
        mean_decoder_loss = 0
        mean_tau_loss = 0
        auxiliary_metric_sums = {}
        self._grf_diagnostics = (
            GRFDecoderMetricsAccumulator(
                self.actor_critic.physics_estimator.grf_scale_n
            )
            if self.is_hard_pact_pos and self.force_decoder_diagnostics_enabled
            else None
        )

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
            grf_target, obs_target, actions_batch, target_values_batch, \
            advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, \
            old_sigma_batch, executed_torque_target, wrench_target, \
            wrench_active_mask in generator:
            
            self.actor_critic.train()
            self.act_optimizer.zero_grad()
            self.enc_optimizer.zero_grad()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            # Perform RL update
            ppo_loss, surrogate_loss, value_loss, _, tau_clone_loss = self._compute_rl_loss(obs_batch, obs_hist_batch, actions_batch,
                                                                                          critic_obs_batch, old_sigma_batch, old_mu_batch,
                                                                                          old_actions_log_prob_batch,
                                                                                          advantages_batch, target_values_batch, returns_batch,
                                                                                          action_func, fb_func, default_pose, dt, qvel_scale)
            
            torch.cuda.synchronize()
            timers["rl_loss"] += time.perf_counter() - t0
            
            ppo_losses = [ppo_loss, tau_clone_loss]
            

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            self.act_optimizer.pc_backward_ppgrad(ppo_losses)

            torch.cuda.synchronize()
            timers["pc_backward"] += time.perf_counter() - t0
            

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.act_optimizer.step()

            torch.cuda.synchronize()
            timers["act_step"] += time.perf_counter() - t0
            
            # Perform some logging
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_tau_loss += tau_clone_loss.item()

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
                vae_loss, kl_div, recon_error, vel_pred_error, dec_input, \
                    decode_targets, recons, auxiliary_metrics = self._compute_vae_loss(
                        obs_hist_batch, grf_target, obs_target,
                        explicit_labels_batch, terminated_batch,
                        executed_torque_target, wrench_target,
                        wrench_active_mask,
                    )
                
                timers["vae_loss"] += time.perf_counter() - t0
                

                torch.cuda.synchronize()
                t0 = time.perf_counter()

                # Update paramaters of encoder
                self.enc_optimizer.zero_grad()
                vae_loss.backward()
                auxiliary_parameters = self.actor_critic.context_encoder.parameters()
                if self.is_hard_pact_pos:
                    auxiliary_parameters = (
                        parameter
                        for module in (
                            self.actor_critic.context_encoder,
                            self.actor_critic.explicit_estimator,
                            self.actor_critic.physics_estimator,
                        )
                        for parameter in module.parameters()
                    )
                nn.utils.clip_grad_norm_(auxiliary_parameters, self.max_grad_norm)
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
                for name, value in auxiliary_metrics.items():
                    auxiliary_metric_sums[name] = (
                        auxiliary_metric_sums.get(
                            name, torch.zeros_like(value.detach())
                        ) + value.detach()
                    )

            # Keeps the interaction of incoming data with layer wieghts below the threashold that 
            #     saturates the tanh activation function.
            self.spectral_normalization(self.actor_critic, sigma_max=6.0)

            timers["spec_norm"] += time.perf_counter() - t0

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_tau_loss /= num_updates

        mean_autoenc_loss /= (num_updates * self.num_enc_epochs)
        mean_decoder_loss /= (num_updates * self.num_enc_epochs)
        mean_kld_loss /= (num_updates * self.num_enc_epochs)
        mean_vel_loss /= (num_updates * self.num_enc_epochs)
        mean_recon_loss /= (num_updates * self.num_enc_epochs)
        auxiliary_denominator = num_updates * self.num_enc_epochs
        self.last_auxiliary_metrics = {
            name: value / auxiliary_denominator
            for name, value in auxiliary_metric_sums.items()
        }
        if self._grf_diagnostics is not None:
            self.last_auxiliary_metrics.update(self._grf_diagnostics.finalize())
        self._grf_diagnostics = None


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
               mean_vel_loss, mean_recon_loss, mean_kld_loss, mean_tau_loss

    def _compute_rl_loss(self, obs_batch, obs_hist_batch,
                         actions_batch, critic_obs_batch,
                         old_sigma_batch, old_mu_batch,
                         old_actions_log_prob_batch,
                         advantages_batch, target_values_batch, returns_batch,
                         action_func, fb_func, default_pose, dt, qvel_scale,
                         latent_noise=None):
        if self.use_boot:
            self.actor_critic.act(
                obs_batch, obs_hist_batch, latent_noise=latent_noise
            )
        else:
            self.actor_critic.act_bootmask(
                obs_batch, obs_hist_batch, latent_noise=latent_noise
            )

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
                
                for param_group in self.act_optimizer.optimizer.param_groups:
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

        if self.use_adaptive_entropy: 
            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.current_entropy_coef * entropy_batch.mean()
        else:
            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()        
        # Update PPO loss with tau-branch tracking loss (scaled magnitude of *first* feedback torques)
        #     Process current and previous actions into the action-space
        q_des_curr, tau_des_curr = action_func(current_actions)

        #     Extract joint pose and velocity data
        #     Obs - cmd (3) [0,1,2], proj_grav (3) [3,4,5], ang_vel (3) [6,7,8], qpose (12) [9-20], qvel (12) [21-32]
        q_pos_curr,  q_velo_curr  = obs_batch[:,9:21].detach().clone(),   obs_batch[:,21:33].detach().clone()
        q_pos_curr,  q_velo_curr  = (q_pos_curr + default_pose).float(),  (q_velo_curr / qvel_scale).float()
        
        # Calculate feedback torques
        pd_tau_curr  = fb_func(q_des_curr,  q_pos_curr,  q_velo_curr)

        tau_clone_loss = F.mse_loss(tau_des_curr, 0.1*pd_tau_curr) # scale down to a tenth in order to not 
                                                                   # have torque overpower in early stages of subsequent coupled training

        return ppo_loss, surrogate_loss, value_loss, current_actions, tau_clone_loss

    @staticmethod
    def _masked_mse(prediction, target, valid):
        per_sample = (prediction - target).square().mean(dim=-1)
        weights = valid.reshape(-1).to(per_sample.dtype)
        return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)

    @staticmethod
    def _masked_explicit_loss(prediction, target, valid):
        """MSE for continuous estimates and BCE-with-logits for contacts."""
        if prediction.shape[-1] != 11 or target.shape[-1] != 11:
            raise ValueError("HardPACTPos explicit estimates and labels must be 11-D")
        target = target.detach()
        continuous_loss = torch.cat((
            (prediction[:, :3] - target[:, :3]).square(),
            (prediction[:, 7:11] - target[:, 7:11]).square(),
        ), dim=-1).mean(dim=-1)
        contact_loss = F.binary_cross_entropy_with_logits(
            prediction[:, 3:7], target[:, 3:7], reduction="none"
        ).mean(dim=-1)
        weights = valid.reshape(-1).to(continuous_loss.dtype)
        denominator = weights.sum().clamp_min(1.0)
        return (
            ((continuous_loss + contact_loss) * weights).sum() / denominator
        )

    @staticmethod
    def _masked_bce(prediction, target, valid):
        per_sample = F.binary_cross_entropy_with_logits(
            prediction, target.detach(), reduction="none"
        ).mean(dim=-1)
        weights = valid.reshape(-1).to(per_sample.dtype)
        return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)

    def _vae_beta_for_iteration(self, iteration):
        """Return the cosine-warmed KL weight for an absolute PPO iteration."""
        if self.vae_kl_warmup_iterations == 0:
            return self.vae_beta
        progress = (
            (float(iteration) - self.vae_kl_warmup_start)
            / self.vae_kl_warmup_iterations
        )
        progress = min(max(progress, 0.0), 1.0)
        return self.vae_kl_initial_weight + 0.5 * (
            self.vae_beta - self.vae_kl_initial_weight
        ) * (1.0 - math.cos(math.pi * progress))

    def _compute_vae_loss(self, obs_hist_batch, grf_target,
                          obs_target, explicit_labels_batch, terminated_batch,
                          executed_torque_target=None, wrench_target=None,
                          wrench_active_mask=None):
        vae_loss = None
        
        mean_latent, logvar_latent, cenet_latent, cenet_torso_velo = self.actor_critic.cenet_enc_forward(obs_hist_batch)
        
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
        auxiliary_metrics = {
            "total": vel_pred_error + recon_error + self.current_vae_beta * kl_div,
            "privileged_reconstruction": recon_error,
            "kl": kl_div,
            "explicit": vel_pred_error,
        }

        if self.is_hard_pact_pos:
            if any(value is None for value in (
                    executed_torque_target, wrench_target,
                    wrench_active_mask)):
                raise RuntimeError("HardPACTPos auxiliary physics labels are missing")
            valid = terminated_batch.bool()
            vel_pred_error = self._masked_explicit_loss(
                cenet_torso_velo, explicit_labels_batch, valid
            )
            heads = self.actor_critic.physics_heads(
                cenet_latent, cenet_torso_velo,
                executed_torque_target.detach(),
            )
            grf_loss = normalized_grf_huber_loss(
                heads.grf_normalized, grf_target.detach(), valid,
            )
            if self._grf_diagnostics is not None:
                self._grf_diagnostics.update(
                    heads.grf_normalized,
                    grf_target,
                    explicit_labels_batch[:, 3:7],
                    valid,
                )
            active = valid & wrench_active_mask.bool()
            neutral = valid & ~wrench_active_mask.bool()
            wrench_active_loss = normalized_wrench_huber_loss(
                heads.wrench_raw_normalized, wrench_target.detach(), active,
            )
            wrench_neutral_loss = normalized_wrench_huber_loss(
                heads.wrench_raw_normalized, wrench_target.detach(), neutral,
            )
            explicit_linear = self._masked_mse(
                cenet_torso_velo[:, :3], explicit_labels_batch[:, :3], valid
            )
            explicit_contact = self._masked_bce(
                cenet_torso_velo[:, 3:7], explicit_labels_batch[:, 3:7], valid
            )
            explicit_clearance = self._masked_mse(
                cenet_torso_velo[:, 7:11], explicit_labels_batch[:, 7:11], valid
            )
            vae_loss = (
                self.privileged_loss_weight * recon_error
                + self.current_vae_beta * kl_div
                + self.explicit_loss_weight * vel_pred_error
                + self.grf_loss_weight * grf_loss
                + self.active_wrench_loss_weight * wrench_active_loss
                + self.neutral_wrench_loss_weight * wrench_neutral_loss
            )
            auxiliary_metrics.update({
                "total": vae_loss,
                "explicit": vel_pred_error,
                "grf": grf_loss,
                "wrench_active": wrench_active_loss,
                "wrench_neutral": wrench_neutral_loss,
                "explicit_base_linear_velocity": explicit_linear,
                "explicit_contact_probabilities": explicit_contact,
                "explicit_foot_clearance": explicit_clearance,
            })
            if self.force_decoder_diagnostics_enabled:
                auxiliary_metrics.update(wrench_regression_metrics(
                    heads.wrench_raw_normalized, wrench_target.detach(), valid,
                    self.actor_critic.physics_estimator.wrench_scale,
                    self.actor_critic.physics_estimator.wrench_qp_clip,
                ))
                active_diagnostics = wrench_regression_metrics(
                    heads.wrench_raw_normalized, wrench_target.detach(), active,
                    self.actor_critic.physics_estimator.wrench_scale,
                    self.actor_critic.physics_estimator.wrench_qp_clip,
                )
                neutral_diagnostics = wrench_regression_metrics(
                    heads.wrench_raw_normalized, wrench_target.detach(), neutral,
                    self.actor_critic.physics_estimator.wrench_scale,
                    self.actor_critic.physics_estimator.wrench_qp_clip,
                )
                auxiliary_metrics.update({
                    "wrench_active_mae_physical": active_diagnostics[
                        "wrench_raw_mae_physical"
                    ],
                    "wrench_active_rmse_physical": active_diagnostics[
                        "wrench_raw_rmse_physical"
                    ],
                    "wrench_neutral_mae_physical": neutral_diagnostics[
                        "wrench_raw_mae_physical"
                    ],
                    "wrench_neutral_rmse_physical": neutral_diagnostics[
                        "wrench_raw_rmse_physical"
                    ],
                })
        else:
            vae_loss = auxiliary_metrics["total"]
        
        return (
            vae_loss, kl_div, recon_error, vel_pred_error,
            dec_input.clone().detach(), decode_target.detach(),
            enc_update_obs_decode.detach(), auxiliary_metrics,
        )
