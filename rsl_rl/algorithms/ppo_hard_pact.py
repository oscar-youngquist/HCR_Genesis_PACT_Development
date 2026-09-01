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

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random

from rsl_rl.utils import print_class_attributes

from rsl_rl.storage import RolloutStoragePACT
from rsl_rl.modules.hard_pact_physics import normalized_huber_loss

from legged_gym.dynamics import (
    BardGo2Dynamics,
    simulator_state_to_bard,
    wrench_at_point,
)

from .pc_grad import PCGrad
from .hard_pact_bard import corrected_bard_inverse_dynamics_loss

def _yaw_local_to_world(values, quaternion_xyzw):
    """Rotate batched ``[..., xyz]`` vectors by base yaw only."""
    x, y, z, w = quaternion_xyzw.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    result = values.clone()
    result[..., 0] = cosine[:, None] * values[..., 0] - sine[:, None] * values[..., 1]
    result[..., 1] = sine[:, None] * values[..., 0] + cosine[:, None] * values[..., 1]
    return result


def _body_point_to_world(point_body, q_xyzw):
    """Rotate a body-frame point offset into world-aligned coordinates."""
    x, y, z, w = q_xyzw[:, 3:7].unbind(-1)
    rotation = torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(-1, 3, 3)
    return torch.einsum("bij,bj->bi", rotation, point_body)

class PPO_HardPACT:
    r"""Self-contained PACT PPO for the HardPACT aliases.

    The legacy PPO rollout, clipped surrogate/value losses, entropy schedule,
    bootstrapping decision, spectral clipping, and checkpoint-facing optimizer
    attributes are retained locally. HardPACT adds two training phases per
    minibatch:

    1. PPO and the corrected BARD objective share one PCGrad backward pass.
       The B1Z1 PACT ownership boundary is used, so this optimizer contains the
       policy, critic, history pathway, privileged decoder, and physics heads.
    2. A newly recomputed graph forms the single auxiliary objective

       ``L_aux = lambda_priv L_priv + beta L_KL + lambda_e L_e``
       ``        + lambda_F L_F + lambda_Wa L_W_active``
       ``        + lambda_Wn L_W_neutral``.

       One auxiliary optimizer step updates the shared history/decoder
       boundary. Actor, critic, and action-noise parameters are excluded.

    Recomputing phase two is important: PCGrad consumes the first autograd
    graph, and retaining it across optimizer steps would both waste memory and
    make the gradients depend on stale parameters.
    """
    actor_critic: nn.Module
    decoder_network: nn.Module
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
                 pinn_lambda=0.001,
                 pinn_encoder_weight=0.05,
                 pinn_warmup=1000,
                 pinn_init_steps=500,
                 num_encoder_epochs=1, # number of epochs for hybrid encoder via supervised learning
                 vae_kld_weight=1.0,   # weight of KL divergence loss in VAE
                 use_adaptive_entropy=True,
                 adaptive_ent_bounds=[0.01, 0.001],
                 adaptive_ent_lin_threshold=0.75,
                 adaptive_ent_ang_threshold=0.35,
                 adaptive_ent_ter_threshold=5.0,
                 adaptive_ent_softmax_temp=2.0,
                 reconstruction_indices=None,
                 auxiliary_learning_rate=2.0e-4,
                 privileged_loss_weight=1.0,
                 explicit_loss_weight=1.0,
                 grf_loss_weight=1.0,
                 active_wrench_loss_weight=1.0,
                 neutral_wrench_loss_weight=0.25,
                 bard_enabled=True,
                 bard_randomize_base_inertia=True,
                 bard_scale_rotational_inertia=True,
                 bard_urdf_path="resources/robots/go2/urdf/go2.urdf",
                 bard_batch_capacity=4096,
                 grf_observation_scale=0.01,
                 base_wrench_observation_scale=0.01,
                 ):
        
        self.device = device

        self.num_priv_obs = num_priv_obs
        self.reconstruction_indices = reconstruction_indices

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.num_enc_epochs = num_encoder_epochs
        self.vae_beta = vae_kld_weight
        self.privileged_loss_weight = float(privileged_loss_weight)
        self.explicit_loss_weight = float(explicit_loss_weight)
        self.grf_loss_weight = float(grf_loss_weight)
        self.active_wrench_loss_weight = float(active_wrench_loss_weight)
        self.neutral_wrench_loss_weight = float(neutral_wrench_loss_weight)

        # ``pinn_encoder_weight`` remains in the constructor solely so legacy
        # configs instantiate unchanged. B1Z1 PCGrad replaced that historical
        # manual gradient-injection mechanism.

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

        # Match the B1Z1 PACT ownership plan. PPO/PINN PCGrad owns the complete
        # differentiable policy path, including context and deployment
        # decoders. A second optimizer performs the temporally separate,
        # combined auxiliary update on that shared context/decoder boundary.
        actor_groups, context_groups = actor_critic.get_optim_groups()
        decoder_group = {
            "params": list(decoder_network.parameters()),
            "weight_decay": context_groups[0].get("weight_decay", 0.0),
            "name": "privileged_decoder",
        }
        shared_groups = [*context_groups, decoder_group]
        ppo_shared_groups = [
            {**group, "params": list(group["params"]), "name": f"ppo_{group['name']}"}
            for group in shared_groups
        ]
        auxiliary_groups = [
            {**group, "params": list(group["params"]), "name": f"auxiliary_{group['name']}"}
            for group in shared_groups
        ]
        self.act_optimizer = PCGrad(
            optim.AdamW([*actor_groups, *ppo_shared_groups], lr=learning_rate),
            reduction="sum",
        )
        self.auxiliary_optimizer = optim.AdamW(
            auxiliary_groups, lr=auxiliary_learning_rate
        )
        self.enc_optimizer = self.auxiliary_optimizer
        self.decoder_optimizer = self.auxiliary_optimizer
        self.transition = RolloutStoragePACT.Transition()

        # # We want to reduce the LR of the critic
        for param_group in self.act_optimizer.optimizer.param_groups:
            # specifically modifies the learning rate of the crtic specific parameters
            if "name" in param_group.keys():
                if "critic" in param_group["name"]:
                    param_group['lr'] = (learning_rate / 3.0)

        self.decoder = decoder_network

        self.ppo_parameters = list(dict.fromkeys(
            parameter for group in self.act_optimizer.optimizer.param_groups
            for parameter in group["params"]
        ))
        self.auxiliary_parameters = list(dict.fromkeys(
            parameter for group in self.auxiliary_optimizer.param_groups
            for parameter in group["params"]
        ))

        self.boot_mult = 1.0
        self.use_boot = False

        self.pinn_weight_final = pinn_lambda
        self.pinn_weight = 0.0
        self.pinn_warmup_steps = pinn_warmup
        self.pinn_init = pinn_init_steps

        self.num_pinn_updates = 0

        self.bard_enabled = bool(bard_enabled)
        self.grf_observation_scale = float(grf_observation_scale)
        self.base_wrench_observation_scale = float(base_wrench_observation_scale)
        self.last_inverse_dynamics_metrics = {}
        self.last_auxiliary_metrics = {}
        self.bard_dynamics = None
        if self.bard_enabled:
            self.bard_dynamics = BardGo2Dynamics(
                os.path.abspath(bard_urdf_path),
                device=self.device,
                batch_capacity=bard_batch_capacity,
                randomize_base_inertia=bard_randomize_base_inertia,
                scale_rotational_inertia=bard_scale_rotational_inertia,
            )

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        print_class_attributes(self)
        
        
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, priv_obs_shape, obs_hist_shape, action_shape, torso_velo_shape, grf_shape, wb_shape):
        self.storage = RolloutStoragePACT(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, priv_obs_shape, obs_hist_shape, \
                                              action_shape, torso_velo_shape, grf_shape, wb_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def _set_std_clip_lwr(self, clip_val=0.1):
        self.actor_critic._set_std_clip_lwr(clip_val)

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
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, obs_history, prev_obs, prev_obs_hist, pprev_obs, pprev_obs_hist):
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

        # PINN stuff
        self.transition.prev_obs      = prev_obs
        self.transition.prev_obs_hist = prev_obs_hist
        self.transition.pprev_obs      = pprev_obs
        self.transition.pprev_obs_hist = pprev_obs_hist
        
        return all_actions
    
    def process_env_step(self, rewards, dones, infos, grf_labels, obs_labels, explicit_labels, gt_forces, mass_mats, bias_vecs, torso_acc):
        self.transition.rewards = rewards.clone()
        
        self.transition.dones = dones
        # Values from the next-time step used as labels for the decoder network
        self.transition.grf_targets = grf_labels

        # This is now the stack of critic observations, we want to prune off the last one
        reconstruction_target = obs_labels[:, -self.num_priv_obs:]
        if self.reconstruction_indices is not None:
            reconstruction_target = reconstruction_target[:, self.reconstruction_indices]
        self.transition.obs_targets = reconstruction_target

        self.transition.explicit_labels = explicit_labels
        self.transition.hard_pact = infos.get("hard_pact_transition")
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # PINN stuff
        self.transition.wb_contact_forces = gt_forces
        self.transition.wb_mass_mat = mass_mats
        self.transition.wb_bias_vec = bias_vecs
        self.transition.torso_acc = torso_acc
        
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

    def update(self, action_func, fb_func, dt, itr, default_pose, qvel_scale):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_autoenc_loss = 0
        mean_vel_loss = 0
        mean_recon_loss = 0
        mean_kld_loss = 0
        mean_decoder_loss = 0
        mean_pinn_loss = 0

        boot_count = 0
        boot_sum_x = None
        boot_sum_x2 = None
        boot_sum_recon_sqerr = 0.0

        if itr > self.pinn_init and self.num_pinn_updates < (self.pinn_warmup_steps+1):
            if self.pinn_weight_final < 0:
                self.pinn_weight = 1.0
            else:
                self.pinn_weight = (float(self.num_pinn_updates)/float(self.pinn_warmup_steps))*self.pinn_weight_final

            print(self.pinn_weight)

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for terminated_batch, obs_batch, critic_obs_batch, obs_hist_batch, explicit_labels_batch, \
            grf_target, obs_target, actions_batch, target_values_batch, \
            advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, \
            old_sigma_batch, _prev_obs_batch, _prev_obs_hist_batch, gt_forces_batch, _mass_mat_batch, \
            _bias_vec_batch, _torso_accs_batch, _pprev_obs_batch, _pprev_obs_hist_batch in generator:
            
            self.actor_critic.train()
            self.act_optimizer.zero_grad()

            # Phase 1: reproduce the legacy PPO objective and retain its mean
            # action, which supplies the differentiable nominal-torque input
            # to the BARD force prediction path.
            ppo_loss, surrogate_loss, value_loss, current_actions = self._compute_rl_loss(obs_batch, obs_hist_batch, actions_batch,
                                                                                          critic_obs_batch, old_sigma_batch, old_mu_batch,
                                                                                          old_actions_log_prob_batch,
                                                                                          advantages_batch, target_values_batch, returns_batch)

            
            # BARD is warmed up using the same schedule as the legacy PINN.
            pinn_loss = None
            if self.pinn_weight > 0.0:
                pinn_loss = self._compute_bard_loss(
                    current_actions, obs_batch, obs_hist_batch, gt_forces_batch,
                    action_func, fb_func, default_pose, qvel_scale,
                )
                
            if self.pinn_weight > 0.0 and self.pinn_weight_final > 0:
                ppo_losses = [ppo_loss, self.pinn_weight * pinn_loss]
            elif self.pinn_weight > 0.0 and self.pinn_weight_final < 0:
                ppo_losses = [ppo_loss, pinn_loss]
            else:
                ppo_losses = [ppo_loss]
            # PCGrad treats reward learning as the primary objective and
            # removes the reward-parallel component of the BARD gradient.
            if self.pinn_weight > 0 and self.pinn_weight_final > 0 and pinn_loss is not None:    # just being extra cautious
                self.act_optimizer.pc_backward_pinn(ppo_losses)
            elif self.pinn_weight_final < 0 and pinn_loss is not None:
                self.act_optimizer.pc_backward_ppgrad(ppo_losses)
            else:
                self.act_optimizer.pc_backward(ppo_losses)
            
            nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm)
            self.act_optimizer.step()

            # Perform some logging
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            if self.pinn_weight > 0.0:
                mean_pinn_loss += pinn_loss.item()
            else:
                mean_pinn_loss += 0.0


            # Phase 2: recompute and aggregate every auxiliary term before a
            # single backward/step, following the B1Z1 adaptation phase.
            for enc_epoch in range(self.num_enc_epochs):
                self.actor_critic.train()
                self.decoder.train()

                nominal_torque = self._nominal_torque(
                    current_actions.detach(), obs_batch, action_func, fb_func,
                    default_pose, qvel_scale,
                )
                aux = self._compute_auxiliary_loss(
                    obs_hist_batch, obs_target, explicit_labels_batch, grf_target,
                    terminated_batch, nominal_torque,
                    self.storage.current_hard_pact_batch,
                )
                self.auxiliary_optimizer.zero_grad(set_to_none=True)
                aux["loss"].backward()
                nn.utils.clip_grad_norm_(self.auxiliary_parameters, self.max_grad_norm)
                self.auxiliary_optimizer.step()
                vae_loss, kl_div = aux["loss"], aux["kl"]
                recon_error, vel_pred_error = aux["privileged"], aux["explicit"]
                dec_loss = aux["privileged"]
                decode_targets, recons = obs_target, aux["reconstruction"]
                self.last_auxiliary_metrics = {
                    name: value.detach() for name, value in aux.items()
                    if name not in ("loss", "reconstruction")
                }

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

                # Log losses
                mean_autoenc_loss += vae_loss.item()
                mean_vel_loss += vel_pred_error.item()
                mean_recon_loss += recon_error.item()
                mean_kld_loss += kl_div.item()
                mean_decoder_loss += dec_loss.item()

            # Keeps the interaction of incoming data with layer wieghts below the threashold that 
            #     saturates the tanh activation function.
            self.spectral_normalization(self.actor_critic, sigma_max=6.0)

        if itr > self.pinn_init:
            self.num_pinn_updates += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_pinn_loss /= num_updates

        mean_autoenc_loss /= (num_updates * self.num_enc_epochs)
        mean_decoder_loss /= (num_updates * self.num_enc_epochs)
        mean_kld_loss /= (num_updates * self.num_enc_epochs)
        mean_vel_loss /= (num_updates * self.num_enc_epochs)
        mean_recon_loss /= (num_updates * self.num_enc_epochs)

        # Calculate the total bootstrapping probability over the performance of the autoencoder on all of the above
        #      total number of scalar elements per sample vector
        feat_dim = boot_sum_x.shape[0]

        mean_pred = boot_sum_x / boot_count                     # [D]
        ex2 = boot_sum_x2 / boot_count                          # [D]
        var = torch.clamp(ex2 - mean_pred**2, min=0.0)          # [D]
        mean_pred_error = var.mean().item()
        actual_pred_error = boot_sum_recon_sqerr / (boot_count * feat_dim)
        ratio = mean_pred_error / (actual_pred_error * self.boot_mult + 1e-8)
        pboot = np.tanh(ratio)

        # Use the (scaled) ratio of mean-prediction performance to actual prediction performance
        #     to determine if encoder bootstrapping is performed.
        self.use_boot = random.random() < pboot
        print("Use bootstrapped Encoder Dynamics: ", self.use_boot)

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_autoenc_loss, mean_decoder_loss, \
               mean_vel_loss, mean_recon_loss, mean_kld_loss, mean_pinn_loss

    def _compute_rl_loss(self, obs_batch, obs_hist_batch,
                         actions_batch, critic_obs_batch,
                         old_sigma_batch, old_mu_batch,
                         old_actions_log_prob_batch,
                         advantages_batch, target_values_batch, returns_batch):
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
                    self.learning_rate = max(1e-6, self.learning_rate / 1.5)
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

        return ppo_loss, surrogate_loss, value_loss, current_actions

    @staticmethod
    def _masked_mse(prediction, target, mask):
        per_sample = (prediction - target).square().mean(dim=-1)
        weights = mask.reshape(-1).to(per_sample.dtype)
        return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)

    def _compute_auxiliary_loss(
        self, history, privileged_target, explicit_target, grf_target,
        valid, nominal_torque, transition,
    ):
        r"""Compute every decoder term on one shared stochastic VAE graph.

        ``z = mu + exp(logvar/2) eps`` is used only for privileged
        reconstruction. Runtime actor conditioning and both deployment heads
        use the deterministic mean ``mu``. The heads internally stop gradients
        through ``e = D_e(mu)`` while retaining gradients through ``mu`` (and
        through nominal torque for the GRF head).
        """
        mean, logvar = self.actor_critic.context_encoder(history)
        sample = self.actor_critic.context_encoder.reparameterization_trick(
            mean, logvar
        )
        explicit = self.actor_critic.explicit_estimator(mean)
        reconstruction = self.decoder(torch.cat((sample, explicit), dim=-1))
        heads = self.actor_critic.physics_heads(mean, explicit, nominal_torque)

        privileged = self._masked_mse(
            reconstruction, privileged_target.detach(), valid
        )
        explicit_loss = self._masked_mse(
            explicit, explicit_target.detach(), valid
        )
        per_sample_kl = -0.5 * torch.sum(
            1 + logvar - mean.square() - logvar.exp(), dim=-1
        )
        weights = valid.reshape(-1).to(per_sample_kl.dtype)
        kl = (per_sample_kl * weights).sum() / weights.sum().clamp_min(1.0)

        grf = normalized_huber_loss(
            heads.grf_yaw_scaled, grf_target.detach(),
            self.actor_critic.physics_estimator.grf_scale, valid,
        )
        if transition is None:
            wrench_target = heads.base_wrench_yaw_scaled.detach().new_zeros(
                heads.base_wrench_yaw_scaled.shape
            )
            active = torch.zeros_like(valid, dtype=torch.bool)
        else:
            wrench_target = transition[
                "total_external_wrench_label_yaw_scaled"
            ].detach()
            active = transition["sustained_wrench_active_mask"].bool()
        active_mask = valid.bool() & active
        neutral_mask = valid.bool() & ~active
        wrench_active = normalized_huber_loss(
            heads.base_wrench_yaw_scaled, wrench_target,
            self.actor_critic.physics_estimator.wrench_scale, active_mask,
        )
        wrench_neutral = normalized_huber_loss(
            heads.base_wrench_yaw_scaled, wrench_target,
            self.actor_critic.physics_estimator.wrench_scale, neutral_mask,
        )
        loss = (
            self.privileged_loss_weight * privileged
            + self.vae_beta * kl
            + self.explicit_loss_weight * explicit_loss
            + self.grf_loss_weight * grf
            + self.active_wrench_loss_weight * wrench_active
            + self.neutral_wrench_loss_weight * wrench_neutral
        )
        return {
            "loss": loss,
            "privileged": privileged,
            "kl": kl,
            "explicit": explicit_loss,
            "grf": grf,
            "wrench_active": wrench_active,
            "wrench_neutral": wrench_neutral,
            "reconstruction": reconstruction,
        }

    @staticmethod
    def _nominal_torque(actions, observations, action_func, fb_func,
                        default_pose, qvel_scale):
        desired_position, feedforward_torque = action_func(actions)
        joint_position = observations[:, 9:21] + default_pose
        joint_velocity = observations[:, 21:33] / qvel_scale
        return feedforward_torque + fb_func(
            desired_position, joint_position, joint_velocity
        )

    def _compute_bard_loss(
        self, current_actions, obs_batch, obs_hist_batch,
        measured_generalized_contact_force,
        action_func, fb_func, default_pose, qvel_scale,
    ):
        r"""Evaluate the corrected BARD inverse-dynamics objective.

        Simulator states are first mapped to canonical BARD coordinates. The
        observed acceleration is

            vdot_obs = (v_{t+1}^BARD - v_t^BARD) / Delta t.

        BARD then evaluates RNEA with the realized inertial/passive parameters.
        With actuation ``tau_a=[0_6,tau_exec]``, learned contact force ``F``
        and applied wrench ``W_applied=W_total-W_mass/CoM``, the residual is

            r_ID = RNEA(q_t,v_t,vdot_obs;theta_rand)
                   - tau_a - J_f^T F - J_b^T W_applied.

        Measured quantities and BARD terms are detached; gradients reach only
        the learned GRF and total-wrench outputs.
        """
        if not self.bard_enabled:
            return current_actions.sum() * 0.0
        batch = self.storage.current_hard_pact_batch
        if batch is None:
            raise RuntimeError("HardPACT BARD loss requires named transition fields")
        pre_q, pre_v = simulator_state_to_bard(
            batch["pre_q"].detach(), batch["pre_v"].detach()
        )
        _, post_v = simulator_state_to_bard(
            batch["pre_q"].detach(), batch["post_v"].detach()
        )
        control_dt = batch["control_dt"].detach().clamp_min(1.0e-8)
        acceleration = (post_v - pre_v) / control_dt
        self.bard_dynamics.default_joint_position = torch.as_tensor(
            default_pose, device=pre_q.device, dtype=pre_q.dtype
        ).detach()
        parameters = {
            "added_base_mass": batch["realized_added_mass"].detach(),
            "base_com_shift": batch["realized_com_shift_body"].detach(),
            "joint_armature": batch["joint_armature"].detach(),
            "joint_friction": batch["joint_friction"].detach(),
            "joint_stiffness": batch["joint_stiffness"].detach(),
            "joint_damping": batch["joint_damping"].detach(),
        }
        terms = self.bard_dynamics.evaluate(
            pre_q.detach(), pre_v.detach(), acceleration.detach(),
            parameters=parameters,
        )
        nominal_torque = self._nominal_torque(
            current_actions, obs_batch.detach(), action_func, fb_func,
            default_pose, qvel_scale,
        )
        # The force targets cover the same control interval as action_t, so
        # their conditioning is the current history h_t—not h_{t-1}.
        _, _, latent, explicit = self.actor_critic.cenet_enc_forward(obs_hist_batch)
        heads = self.actor_critic.physics_heads(
            latent, explicit, nominal_torque
        )
        grf_yaw = heads.grf_yaw_scaled.reshape(-1, 4, 3)
        grf_world = _yaw_local_to_world(
            grf_yaw / self.grf_observation_scale, batch["pre_q"][:, 3:7]
        )
        total_wrench_yaw = (
            heads.base_wrench_yaw_scaled / self.base_wrench_observation_scale
        )
        total_wrench_world = torch.cat((
            _yaw_local_to_world(
                total_wrench_yaw[:, :3].unsqueeze(1), batch["pre_q"][:, 3:7]
            ).squeeze(1),
            _yaw_local_to_world(
                total_wrench_yaw[:, 3:].unsqueeze(1), batch["pre_q"][:, 3:7]
            ).squeeze(1),
        ), dim=-1)
        mass_wrench = batch["equivalent_mass_com_wrench_world"].detach()
        base_point_world = batch["pre_q"][:, :3].detach()
        com_world = base_point_world + _body_point_to_world(
            batch["realized_com_shift_body"].detach(), batch["pre_q"].detach()
        )
        # Move the learned applied wrench from the realized CoM to the base
        # frame point used by J_b, then restore the label-only mass term so the
        # reducer's mandatory subtraction still occurs exactly once.
        applied_at_base = wrench_at_point(
            total_wrench_world - mass_wrench, com_world, base_point_world
        )
        total_at_base = applied_at_base + mass_wrench
        result = corrected_bard_inverse_dynamics_loss(
            required_generalized_force=terms.rnea,
            foot_jacobians=terms.foot_jacobians,
            base_jacobian=terms.base_jacobian,
            interval_executed_torque=batch["interval_executed_torque"],
            interval_grf_world=grf_world,
            total_wrench_world=total_at_base,
            mass_com_wrench_world=mass_wrench,
            measured_generalized_contact_force=measured_generalized_contact_force,
            push_event_mask=batch["push_event_mask"],
            reset_mask=batch["reset_mask"],
            timeout_mask=batch["timeout_mask"],
            teleport_mask=batch["teleport_mask"],
        )
        self.last_inverse_dynamics_metrics = result.metrics
        return result.loss
