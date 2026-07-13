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
import time
import math
from contextlib import contextmanager, nullcontext

import numpy as np
import random

from rsl_rl.modules import ActorCritic_KITE
from rsl_rl.modules.kite_privileged_encoders import (
    PrivDynamicsDecoder,
    # PrviDynamicsMLPMixerKITE,
    PrivilegedProprioceptiveContextEncoder,
    TerrainAttentionEncoder,
    TerrainTwoHeadDecoder,
)
from rsl_rl.storage import RolloutStorageKITE

class DepthGradientLoss(nn.Module):
    """
    Penalizes differences in spatial gradients (X and Y directions) 
    to preserve sharp edges and geometric boundaries.
    """
    def __init__(self):
        super().__init__()
        # Sobel/Prewitt style kernels for edge extraction
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        kernel_x = kernel_x / 8.0
        kernel_y = kernel_y / 8.0

        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def _masked_sample_mean(self, values, mask):
        sample_values = values.reshape(values.shape[0], -1).mean(dim=1)
        if mask is None:
            return sample_values.mean()
        sample_mask = mask.squeeze(-1).float()
        denom = torch.clamp(sample_mask.sum(), min=1.0)
        return torch.sum(sample_values * sample_mask) / denom

    def forward(self, pred, target, mask=None):
        # F.conv2d does not accept padding_mode directly, so pad explicitly
        # before applying the fixed Sobel kernels.
        pred_padded = F.pad(pred, (1, 1, 1, 1), mode="replicate")
        target_padded = F.pad(target, (1, 1, 1, 1), mode="replicate")

        # Calculate gradients using 2D convolutions
        pred_grad_x = F.conv2d(pred_padded, self.kernel_x)
        pred_grad_y = F.conv2d(pred_padded, self.kernel_y)
        
        target_grad_x = F.conv2d(target_padded, self.kernel_x)
        target_grad_y = F.conv2d(target_padded, self.kernel_y)
        
        # Calculate L1 distance of the gradients while respecting terminated
        # episode masks at the sample level.
        loss_x = self._masked_sample_mean(
            torch.abs(pred_grad_x - target_grad_x),
            mask,
        )
        loss_y = self._masked_sample_mean(
            torch.abs(pred_grad_y - target_grad_y),
            mask,
        )
        
        return loss_x + loss_y
class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Measure (SSIM) Loss adapted for deep learning.
    Measures luminance, contrast, and structure similarity.
    """
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _masked_sample_mean(self, values, mask):
        sample_values = values.reshape(values.shape[0], -1).mean(dim=1)
        if mask is None:
            return sample_values.mean()
        sample_mask = mask.squeeze(-1).float()
        denom = torch.clamp(sample_mask.sum(), min=1.0)
        return torch.sum(sample_values * sample_mask) / denom

    def forward(self, pred, target, mask=None):
        # Continuous average using a uniform or Gaussian kernel (uniform used here for simplicity)
        mu_p = F.avg_pool2d(pred, self.window_size, stride=1, padding=self.window_size//2)
        mu_t = F.avg_pool2d(target, self.window_size, stride=1, padding=self.window_size//2)
        
        mu_p_sq = mu_p.pow(2)
        mu_t_sq = mu_t.pow(2)
        mu_p_t = mu_p * mu_t
        
        sigma_p_sq = F.avg_pool2d(pred * pred, self.window_size, stride=1, padding=self.window_size//2) - mu_p_sq
        sigma_t_sq = F.avg_pool2d(target * target, self.window_size, stride=1, padding=self.window_size//2) - mu_t_sq
        sigma_pt = F.avg_pool2d(pred * target, self.window_size, stride=1, padding=self.window_size//2) - mu_p_t
        
        ssim_idx = ((2 * mu_p_t + self.C1) * (2 * sigma_pt + self.C2)) / \
                   ((mu_p_sq + mu_t_sq + self.C1) * (sigma_p_sq + sigma_t_sq + self.C2))
        
        # Return 1 - SSIM bounded within [0, 1] for optimization, averaged
        # only across non-terminated rollout samples when a mask is provided.
        return self._masked_sample_mean((1.0 - ssim_idx) / 2.0, mask)

class CompositeDepthLoss(nn.Module):
    """
    Combines pixel-wise L1 loss, gradient loss, and SSIM loss.
    """
    def __init__(self, w_l1=0.80, w_grad=1.00, w_ssim=0.20):
        super().__init__()
        self.grad_loss = DepthGradientLoss()
        self.ssim_loss = SSIMLoss()
        
        # Hyperparameters weights to balance the three criteria
        self.w_l1 = w_l1
        self.w_grad = w_grad
        self.w_ssim = w_ssim

    def _masked_l1_loss(self, pred, target, mask):
        sample_values = torch.abs(pred - target).reshape(
            pred.shape[0],
            -1,
        ).mean(dim=1)
        if mask is None:
            return sample_values.mean()
        sample_mask = mask.squeeze(-1).float()
        denom = torch.clamp(sample_mask.sum(), min=1.0)
        return torch.sum(sample_values * sample_mask) / denom

    def forward(self, pred, target, mask=None):
        l1 = self._masked_l1_loss(pred, target, mask)
        l_grad = self.grad_loss(pred, target, mask)
        l_ssim = self.ssim_loss(pred, target, mask)
        
        # Weighted summation
        total_loss = (
            self.w_l1 * l1
            + self.w_grad * l_grad
            + self.w_ssim * l_ssim
        )
        return total_loss

class PPO_KITE:
    actor_critic: ActorCritic_KITE
    def __init__(self,
                 actor_critic,
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
                 num_encoder_epochs=1, # number of epochs for hybrid encoder via supervised learning
                 use_adaptive_entropy=True,
                 adaptive_ent_bounds=[0.01, 0.001],
                 adaptive_ent_lin_threshold=0.75,
                 adaptive_ent_ang_threshold=0.35,
                 adaptive_ent_ter_threshold=5.0,
                 adaptive_ent_softmax_temp=2.0,
                 terrain_map_shape=None,
                 num_priv_obs_history=3,
                 privileged_terrain_latent_dim=32,
                 privileged_dynamics_latent_dim=16,
                 priv_activation_func='elu',
                 cnn_norm_type="layer",
                 terrain_encoder_attention_dim=128,
                 terrain_encoder_n_heads=4,
                 privileged_terrain_std_min=0.01,
                 privileged_terrain_std_max=1.5,
                 terrain_decoder_hidden_dim=128,
                 terrain_decoder_encoded_spatial_dim=(3,4),
                 terrain_decoder_channels=64,
                 privileged_dynamics_decoder_layers=[32,64,128,256],
                 priv_mixer_num_blocks=2,
                 priv_mixer_hidden_dim=128,
                 priv_mixer_token_dim=32,
                 priv_mixer_channel_dim=64,
                 priv_mixer_use_layer_norm=True,
                 privileged_dynamics_std_min=0.01,
                 privileged_dynamics_std_max=1.5,

                 depth_frame_recon_weight=1.0,
                 depth_frame_kl_weight=1.0e-3,
                 depth_transform_identity_weight=1.0e-3,

                 depth_sequence_kl_weight=1.0,
                 proprio_kl_weight=1.0,
                 
                 depth_sequence_terrain_weight=1.0,
                 proprio_dynamics_weight=1.0,
                 modality_explicit_weight=1.0,
                 
                 contrastive_weight=0.1,
                 contrastive_lambda=0.5,
                 contrastive_margin=1.0,
                 
                 versatility_weight=0.01,
                 versatility_lambda_e=0.1,
                 use_adaptive_kl_beta=True,
                 adaptive_kl_beta_delta=0.05,
                 adaptive_kl_beta_ema_alpha=0.05,
                 depth_frame_kl_recon_target=0.15,
                 depth_frame_kl_beta_min=1.0e-5,
                 depth_frame_kl_beta_max=1.0e-1,
                 depth_sequence_kl_recon_target=0.5,
                 depth_sequence_kl_beta_min=1.0e-5,
                 depth_sequence_kl_beta_max=1.0,
                 proprio_kl_recon_target=0.5,
                 proprio_kl_beta_min=1.0e-5,
                 proprio_kl_beta_max=1.0,
                 privileged_terrain_kl_weight=0.1,
                 privileged_terrain_kl_recon_target=0.5,
                 privileged_terrain_kl_beta_min=1.0e-5,
                 privileged_terrain_kl_beta_max=1.0,
                 privileged_dynamics_kl_weight=0.1,
                 privileged_dynamics_kl_recon_target=0.5,
                 privileged_dynamics_kl_beta_min=1.0e-5,
                 privileged_dynamics_kl_beta_max=1.0,
                 mixer_kl_weight=0.1,
                 mixer_kl_recon_target=0.5,
                 mixer_kl_beta_min=1.0e-5,
                 mixer_kl_beta_max=1.0,
                 gpu_debugging=False,
                 log_detailed_encoder_losses=False,
                 profile_learning=False,
                 privileged_dynamics_context_layer_sizes=[256, 128],
                 ):
        
        self.device = device

        self.num_priv_obs = num_priv_obs

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.num_enc_epochs = num_encoder_epochs

        # Entropy coefficent adapation parameters
        self.use_adaptive_entropy = use_adaptive_entropy
        self.entropy_coef_bounds = adaptive_ent_bounds
        self.ent_linvelo_threshold = adaptive_ent_lin_threshold
        self.ent_angvelo_threshold = adaptive_ent_ang_threshold
        self.ent_terrain_threshold = adaptive_ent_ter_threshold
        self.ent_softmax_temperature = adaptive_ent_softmax_temp
        self.current_entropy_coef = entropy_coef
       
        # Data shapes
        self.terrain_map_shape = terrain_map_shape
        self.num_priv_obs_history = num_priv_obs_history
        self.privileged_terrain_latent_dim = privileged_terrain_latent_dim
        self.privileged_dynamics_latent_dim = privileged_dynamics_latent_dim

        # loss weights for depth-image processing
        self.depth_frame_recon_weight = depth_frame_recon_weight              
        self.depth_frame_kl_weight = depth_frame_kl_weight
        self.depth_transform_identity_weight = depth_transform_identity_weight
        self.depth_sequence_kl_weight = depth_sequence_kl_weight
        self.depth_frame_reconstruction_loss = CompositeDepthLoss().to(self.device)
        
        # loss weights for proprioceptive history encoder losses
        self.proprio_kl_weight = proprio_kl_weight
        
        # Student encoder reconstruction weights. Terrain reconstruction trains
        #   the depth-sequence encoder; privileged-dynamics reconstruction trains
        #   the proprioceptive encoder.
        self.depth_sequence_terrain_weight = depth_sequence_terrain_weight
        self.proprio_dynamics_weight = proprio_dynamics_weight
        self.modality_explicit_weight = modality_explicit_weight
        self.versatility_weight = versatility_weight
        self.versatility_lambda_e = versatility_lambda_e
        self.use_adaptive_kl_beta = use_adaptive_kl_beta
        self.adaptive_kl_beta_delta = adaptive_kl_beta_delta
        self.adaptive_kl_beta_ema_alpha = adaptive_kl_beta_ema_alpha
        self.depth_frame_kl_recon_target = depth_frame_kl_recon_target
        self.depth_frame_kl_beta_min = depth_frame_kl_beta_min
        self.depth_frame_kl_beta_max = depth_frame_kl_beta_max
        self.depth_sequence_kl_recon_target = depth_sequence_kl_recon_target
        self.depth_sequence_kl_beta_min = depth_sequence_kl_beta_min
        self.depth_sequence_kl_beta_max = depth_sequence_kl_beta_max
        self.proprio_kl_recon_target = proprio_kl_recon_target
        self.proprio_kl_beta_min = proprio_kl_beta_min
        self.proprio_kl_beta_max = proprio_kl_beta_max
        self.privileged_terrain_kl_weight = privileged_terrain_kl_weight
        self.privileged_terrain_kl_recon_target = privileged_terrain_kl_recon_target
        self.privileged_terrain_kl_beta_min = privileged_terrain_kl_beta_min
        self.privileged_terrain_kl_beta_max = privileged_terrain_kl_beta_max
        self.privileged_dynamics_kl_weight = privileged_dynamics_kl_weight
        self.privileged_dynamics_kl_recon_target = privileged_dynamics_kl_recon_target
        self.privileged_dynamics_kl_beta_min = privileged_dynamics_kl_beta_min
        self.privileged_dynamics_kl_beta_max = privileged_dynamics_kl_beta_max
        self.mixer_kl_recon_target = mixer_kl_recon_target
        self.mixer_kl_beta_min = mixer_kl_beta_min
        self.mixer_kl_beta_max = mixer_kl_beta_max
        self.depth_frame_kl_recon_ema = None
        self.depth_sequence_kl_recon_ema = None
        self.proprio_kl_recon_ema = None
        self.privileged_terrain_kl_recon_ema = None
        self.privileged_dynamics_kl_recon_ema = None
        self.mixer_kl_recon_ema = None
        if self.use_adaptive_kl_beta:
            self.versatility_lambda_e = mixer_kl_weight
        self.gpu_debugging = gpu_debugging
        self.log_detailed_encoder_losses = log_detailed_encoder_losses
        self.profile_learning = profile_learning
        self.last_update_timers = {}
        
        # Contrastive loss weights/margin shared across all contrastive updates
        self.contrastive_weight = contrastive_weight
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_margin = contrastive_margin

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later

        # Actor-critic owns the non-privileged trainable modules and returns
        #   three optimizers: RL actor/critic, depth-frame autoencoder, and the
        #   merged sequence/proprio/mixer auxiliary path.
        self.act_optimizer, self.depth_frame_optimizer, self.enc_optimizer = actor_critic.configure_optimizers(learning_rate)
        
        # Transition data structure from storage class
        self.transition = RolloutStorageKITE.Transition()

        # Reduce the LR of the critic
        for param_group in self.act_optimizer.param_groups:
            if "name" in param_group.keys():
                if "critic" in param_group["name"]:
                    param_group['lr'] = (learning_rate / 3.0)


        ##
        ##  Construct privileged encoder-decoder pairs
        terrain_h, terrain_w, _ = terrain_map_shape or (1, 1, 4)

        # Privileged Terrain encoder network
        self.priv_terrain_encoder = TerrainAttentionEncoder(
            height=terrain_h,
            width=terrain_w,
            latent_dim=privileged_terrain_latent_dim,
            cnn_activation=priv_activation_func,
            norm_type=cnn_norm_type,
            attention_dim=terrain_encoder_attention_dim,
            n_heads=terrain_encoder_n_heads,
            std_min=privileged_terrain_std_min,
            std_max=privileged_terrain_std_max,
        ).to(self.device)

        # Two-headed (height, surface normals) privileged terrain decoder network
        self.priv_terrain_decoder = TerrainTwoHeadDecoder(
            height=terrain_h,
            width=terrain_w,
            latent_dim=privileged_terrain_latent_dim,
            cnn_activation=priv_activation_func,
            decoder_hidden_dim=terrain_decoder_hidden_dim,
            encoded_spatial_shape=terrain_decoder_encoded_spatial_dim,
            decoder_channels=terrain_decoder_channels,
            norm_type=cnn_norm_type,
        ).to(self.device)
        
        
        # Old privileged dynamics MLP-Mixer encoder.
        # self.priv_dynamics_encoder = PrviDynamicsMLPMixerKITE(
        #     context_input_dim=self.num_priv_obs_history * self.num_priv_obs,
        #     num_tokens=self.num_priv_obs,
        #     input_dim_per_token=self.num_priv_obs_history,
        #     context_latent_size=privileged_dynamics_latent_dim,
        #     activation=priv_activation_func,
        #     num_mixer_blocks=priv_mixer_num_blocks,
        #     hidden_dim=priv_mixer_hidden_dim,
        #     token_mlp_dim=priv_mixer_token_dim,
        #     channel_mlp_dim=priv_mixer_channel_dim,
        #     use_layer_norm=priv_mixer_use_layer_norm,
        #     std_min=privileged_dynamics_std_min,
        #     std_max=privileged_dynamics_std_max,
        #     device=device,
        # ).to(self.device)
        self.priv_dynamics_encoder = PrivilegedProprioceptiveContextEncoder(
            context_input_dim=self.num_priv_obs_history * self.num_priv_obs,
            context_layer_sizes=list(privileged_dynamics_context_layer_sizes),
            context_latent_size=privileged_dynamics_latent_dim,
            activation=priv_activation_func,
            std_min=privileged_dynamics_std_min,
            std_max=privileged_dynamics_std_max,
            device=device,
        ).to(self.device)

        # Privileged dynamics decoder network
        self.priv_dynamics_decoder = PrivDynamicsDecoder(
            input_dim=privileged_dynamics_latent_dim,
            decode_dim=self.num_priv_obs,
            layers=privileged_dynamics_decoder_layers,
        ).to(self.device)

        # Optimizer for the privileged encoder-decoder pairs
        self.privileged_optimizer = optim.Adam(
            list(self.priv_terrain_encoder.parameters())
            + list(self.priv_terrain_decoder.parameters())
            + list(self.priv_dynamics_encoder.parameters())
            + list(self.priv_dynamics_decoder.parameters()),
            lr=learning_rate,
        )

        self.boot_mult = 1.0
        self.use_boot = False
        self.use_depth_vel_boot = False

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

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, priv_obs_shape, obs_hist_shape, action_shape,
        explicit_shape, depth_image_shape, depth_latent_history_shape, depth_torso_state_shape, terrain_map_shape, priv_obs_history_shape,
        contrastive_anchor_shape,
    ):
        self.storage = RolloutStorageKITE(num_envs, num_transitions_per_env, actor_obs_shape, priv_obs_shape, obs_hist_shape, \
                                              action_shape, explicit_shape, depth_image_shape,
                                              depth_latent_history_shape, depth_torso_state_shape,
                                              terrain_map_shape, priv_obs_history_shape,
                                              contrastive_anchor_shape, self.device,
                                              store_action_distribution=self.schedule == "adaptive")

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    @contextmanager
    def _frozen_module_params(self, *modules):
        """Temporarily stop teacher modules from storing parameter gradients."""
        old_requires_grad = []
        for module in modules:
            old_requires_grad.append([p.requires_grad for p in module.parameters()])
            for p in module.parameters():
                p.requires_grad_(False)
        try:
            yield
        finally:
            for module, flags in zip(modules, old_requires_grad):
                for p, requires_grad in zip(module.parameters(), flags):
                    p.requires_grad_(requires_grad)

    def _latest_privileged_obs(self, privileged_obs_history):
        return privileged_obs_history[:, -self.num_priv_obs:]

    def build_critic_obs(self, privileged_obs_history, terrain_map, detach_latents=True):
        """Build critic input from privileged dynamics and learned terrain latents."""
        latest_privileged_obs = self._latest_privileged_obs(privileged_obs_history)
        # The critic uses privileged latents as inputs, but the RL optimizer
        #   does not train the privileged encoders. Avoid building graphs when
        #   those latents will be detached immediately.
        context = torch.no_grad() if detach_latents else nullcontext()
        with context:
            if detach_latents:
                terrain_latent = self.priv_terrain_encoder.forward_inference(terrain_map)
                dynamics_latent = self.priv_dynamics_encoder.forward_inference(privileged_obs_history)
            else:
                terrain_latent, _ = self.priv_terrain_encoder.encode(terrain_map)
                dynamics_latent, _ = self.priv_dynamics_encoder.encode(privileged_obs_history)
        if detach_latents:
            terrain_latent = terrain_latent.detach()
            dynamics_latent = dynamics_latent.detach()
        return torch.cat(
            [latest_privileged_obs, terrain_latent, dynamics_latent],
            dim=-1,
        )

    def _masked_sample_mean(self, values, mask):
        """Mean over non-terminated samples, with each sample weighted equally."""
        sample_values = values.reshape(values.shape[0], -1).mean(dim=1)
        sample_mask = mask.squeeze(-1).float()
        denom = torch.clamp(sample_mask.sum(), min=1.0)
        return torch.sum(sample_values * sample_mask) / denom

    def _masked_mse_loss(self, prediction, target, mask):
        """MSE averaged over valid samples only."""
        return self._masked_sample_mean((prediction - target).pow(2), mask)

    def _masked_bce_with_logits_loss(self, prediction, target, mask):
        """Binary cross-entropy averaged over valid samples only."""
        return self._masked_sample_mean(
            F.binary_cross_entropy_with_logits(
                prediction,
                target,
                reduction="none",
            ),
            mask,
        )

    def _cpu_float(self, value):
        """Detach a scalar tensor immediately so logging does not hold CUDA refs."""
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _detach_scalar(self, value):
        """Keep scalar logs graph-free without forcing an immediate GPU sync."""
        if isinstance(value, torch.Tensor):
            return value.detach()
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def _finish_log_scalar(self, value):
        """Convert accumulated scalar logs to Python floats once per update."""
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _sync_if_debugging(self):
        """Synchronize CUDA only when detailed GPU timings are requested."""
        if (self.gpu_debugging or self.profile_learning) and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _empty_cache_if_debugging(self):
        """Clear CUDA cache only in debugging mode because it is slow."""
        if self.gpu_debugging and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _aux_autocast(self):
        """Use bf16 autocast for auxiliary learning to reduce activation VRAM."""
        if self.device != "cpu" and torch.cuda.is_available():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _boot_feature_summary(self, target, recon, sample_mask):
        """Reduce a reconstruction batch to small CPU stats for boot-probability."""
        flat_target = target.flatten(start_dim=1)
        flat_recon = recon.flatten(start_dim=1)
        sample_weight = sample_mask.unsqueeze(-1)

        return {
            # Store only reduced CPU statistics; the batch-sized target/recon
            #   tensors can be released before the next auxiliary backward pass.
            "sum_x": (flat_target * sample_weight).sum(dim=0).detach().cpu(),
            "sum_x2": (flat_target * flat_target * sample_weight).sum(dim=0).detach().cpu(),
            "sse": self._cpu_float(
                ((flat_recon - flat_target).pow(2) * sample_weight).sum()
            ),
            "dim": flat_target.shape[-1],
        }

    def _make_boot_summary(
        self,
        dynamics_target,
        dynamics_recon,
        terrain_target,
        terrain_recon,
        explicit_labels,
        body_velo_est,
        mask,
    ):
        """Create CPU-only summaries used to update boot probabilities later."""
        sample_mask = mask.squeeze(-1).float()
        count = self._cpu_float(sample_mask.sum())
        body_velo_dim = self.actor_critic.body_velo_dim

        return {
            "count": count,
            "dynamics": self._boot_feature_summary(
                dynamics_target, dynamics_recon, sample_mask
            ),
            "terrain": self._boot_feature_summary(
                terrain_target, terrain_recon, sample_mask
            ),
            "velocity": self._boot_feature_summary(
                explicit_labels[:, :body_velo_dim],
                body_velo_est,
                sample_mask,
            ),
        }

    def _accumulate_boot_summary(self, accumulator, summary, feature_names):
        """Accumulate compact CPU boot summaries across mini-batches."""
        if summary["count"] <= 0:
            return accumulator
        if accumulator is None:
            accumulator = {
                "count": 0.0,
                "features": {
                    name: {
                        "sum_x": torch.zeros_like(summary[name]["sum_x"]),
                        "sum_x2": torch.zeros_like(summary[name]["sum_x2"]),
                        "sse": 0.0,
                        "dim": summary[name]["dim"],
                    }
                    for name in feature_names
                },
            }

        accumulator["count"] += summary["count"]
        for name in feature_names:
            accumulator["features"][name]["sum_x"] += summary[name]["sum_x"]
            accumulator["features"][name]["sum_x2"] += summary[name]["sum_x2"]
            accumulator["features"][name]["sse"] += summary[name]["sse"]
        return accumulator

    def _boot_probability_from_summary(self, accumulator):
        """Compute boot probability from CPU summary stats."""
        if accumulator is None or accumulator["count"] <= 0:
            return 0.0

        count = accumulator["count"]
        total_dim = 0
        total_var_sum = 0.0
        total_sse = 0.0

        for stats in accumulator["features"].values():
            mean_pred = stats["sum_x"] / count
            ex2 = stats["sum_x2"] / count
            var = torch.clamp(ex2 - mean_pred**2, min=0.0)
            total_var_sum += float(var.sum().item())
            total_sse += stats["sse"]
            total_dim += stats["dim"]

        mean_pred_error = total_var_sum / max(total_dim, 1)
        actual_pred_error = total_sse / (count * max(total_dim, 1))
        ratio = mean_pred_error / (actual_pred_error * self.boot_mult + 1e-8)
        return float(np.tanh(ratio))

    def _contrastive_loss(self, z, positive, negative, mask):
        """PBRS-style latent alignment: pull toward privileged anchor, push from noise."""
        positive = positive.detach()
        negative = negative.detach()

        if negative.shape[-1] < z.shape[-1]:
            repeat_count = int(np.ceil(z.shape[-1] / negative.shape[-1]))
            negative = negative.repeat(1, repeat_count)

        if negative.shape[-1] != z.shape[-1]:
            negative = negative[:, :z.shape[-1]]

        positive_loss = self._masked_mse_loss(z, positive, mask)

        negative_distance = torch.linalg.norm(z - negative, dim=-1)

        negative_loss = self._masked_sample_mean(
            torch.clamp(self.contrastive_margin - negative_distance, min=0.0).pow(2),
            mask,
        )

        return (
            self.contrastive_lambda * positive_loss
            + (1.0 - self.contrastive_lambda) * negative_loss
        )

    def _normalized_contrastive_loss(self, z, positive, negative, mask):
        """Apply contrastive loss after normalizing student and anchor latents."""
        positive = F.normalize(positive.detach(), p=2, dim=-1, eps=1e-6)
        negative = F.normalize(negative.detach(), p=2, dim=-1, eps=1e-6)
        return self._contrastive_loss(z, positive, negative, mask)

    def _terrain_recon_loss(self, terrain_recon, terrain_target, mask):
        """Terrain decoder loss: MSE on heights plus cosine loss on normals."""
        height_loss = self._masked_mse_loss(
            terrain_recon[..., 0:1],
            terrain_target[..., 0:1],
            mask,
        )
        normal_cos = F.cosine_similarity(
            terrain_recon[..., 1:4],
            terrain_target[..., 1:4],
            dim=-1,
            eps=1e-6,
        )
        normal_loss = self._masked_sample_mean(1.0 - normal_cos, mask)

        terrain_recon_log = {"height_loss":height_loss.detach(),
                             "normal_cos":normal_loss.detach()}
        
        total_loss = height_loss + normal_loss

        return total_loss, terrain_recon_log

    def _kl_loss(self, mean, logvar, mask):
        _mask = mask.squeeze(-1).float()                           # (B,)
        _denom = torch.clamp(_mask.sum(), min=1.0)                 # (1,)

        # KL(q(z|o) || N(0, I))
        kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mean.pow(2) - torch.exp(logvar), dim=-1)

        # Average over vaid samples
        kl_loss = torch.sum(kl_per_sample * _mask) / _denom

        return kl_loss

    def _transform_identity_loss(self, transform_matrices, mask):
        identity = torch.eye(
            2,
            device=transform_matrices.device,
            dtype=transform_matrices.dtype,
        ).view(1, 2, 2)
        return self._masked_sample_mean((transform_matrices - identity).pow(2), mask)

    # "We incorporate an unsupervised RL objective through mu-
    #    tual information (MI) maximization for promoting skill dis-
    #    covery. This objective allows the emergence of novel behaviors
    #    while preserving stable behaviors induced by the handcrafted
    #    reward functions. Specifically, we maximize the MI between
    #    visited states and the latent variable inferred by the multi-
    #    modal context encoder."
    # This loss is borrowed from DreamWaQ++ - https://dreamwaqpp.github.io/static/paper.pdf
    def _versatility_kl_metric(self, mean, logvar, mask, kl_weight=None):
        vers_loss, vers_log = self._versatility_metric(mean, logvar, mask)

        # Calculate the VAE encoder's KL-divergence
        kl_loss = self._kl_loss(mean, logvar, mask)

        if kl_weight is None:
            kl_weight = self.versatility_lambda_e

        # Maximize: J = MI - beta * KL
        # Minimize: loss = -J
        vers_loss += kl_weight * kl_loss

        # populate the log-dict
        vers_log["kl"] = kl_loss.detach()

        return vers_loss, vers_log


    # "We incorporate an unsupervised RL objective through mu-
    #    tual information (MI) maximization for promoting skill dis-
    #    covery. This objective allows the emergence of novel behaviors
    #    while preserving stable behaviors induced by the handcrafted
    #    reward functions. Specifically, we maximize the MI between
    #    visited states and the latent variable inferred by the multi-
    #    modal context encoder."
    # This loss is borrowed from DreamWaQ++ - https://dreamwaqpp.github.io/static/paper.pdf
    def _versatility_metric(self, mean, logvar, mask):
        # Using torch.sum() and dividing by _denom so "masked" samples are not counted torwards
        #     the mean calculations
        _mask = mask.squeeze(-1).float()                           # (B,)
        _denom = torch.clamp(_mask.sum(), min=1.0)                 # (1,)

        _mask_col = _mask.unsqueeze(-1)                            # (B, 1)

        # Find the mean "mean" preidction,
        #     used to calculate var of means, masking out terminated samples
        mean_pred_mean = torch.sum(mean * _mask_col, dim=0) / _denom   # (latent_dim, )

        # # Var_mu = E[(mu_i - E[mu])^2]
        # #     masking out terminated examples
        mean_pred_var = torch.sum(
            (mean - mean_pred_mean).pow(2) * _mask_col,
            dim=0) / _denom                                        # (latent_dim, )

        # # #
        # #  Technically, the below is a more accurate calculation of the marginal variance
        # #       but it runs the risk of not as strictly enforcing diversity in the mean samples
        # #       which is what we use during deployment, so ignoring this for now
        # # #
        # E[sigma_i^2]
        # mean_conditional_var = torch.sum(torch.exp(logvar) * _mask_col,dim=0,) / _denom  # (latent_dim,)

        # # Law of total variance:
        # # Var(z) = Var(E[z|o]) + E[Var(z|o)]
        # marginal_var = mean_pred_var + mean_conditional_var + 1e-6

        # adding small value for numerical stability
        marginal_var = mean_pred_var + 1e-6                              # (1,)

        # H(z), approximating the marginal q(z) as diagonal Gaussian
        #     H(z) is large when the means are spread out over the batch.
        #     Maximizing this keeps the latent space diverse across different observations.
        #         torch.log is natural log (ln)
        marginal_entropy = 0.5 * torch.sum(torch.log(2.0 * np.pi * np.e * marginal_var))   # (1, )

        # H(z|o), exact entropy of diagonal Gaussian q(z|o_i),
        #         Induces a denoising/clustering effect - conditional entropy encourages intrinsically
        #         similar observations to map to similar latent representations.
        sample_conditional_entropy = 0.5 * torch.sum(np.log(2.0 * np.pi * np.e) + logvar, dim=-1)  # (B, 1)

        #         averaged over non-masked batch
        conditional_entropy = torch.sum(sample_conditional_entropy * _mask) / _denom               # (1, )


        # I(o;z) = H(z) - H(z|o)
        #     Diversify latent representations while keeping them
        #     observation-conditioned and not purely noisy.
        mutual_info = marginal_entropy - conditional_entropy                                       # (1, )

        # Maximize: J = MI - beta * KL
        # Minimize: loss = -J
        vers_loss = -mutual_info

        # populate the log-dict
        vers_log = {"marginal_entropy":marginal_entropy.detach(),
                    "conditional_entropy":conditional_entropy.detach(),
                    "mutual_info":mutual_info.detach(),
                    }

        return vers_loss, vers_log
    
    def _update_kl_weight(self, beta, target_recon_loss, recon_loss, beta_min, beta_max):
        """Apply beta <- clamp(exp(delta * (tau - L_recon)) * beta)."""
        k = math.exp(
            self.adaptive_kl_beta_delta
            * (float(target_recon_loss) - float(recon_loss))
        )
        return float(np.clip(k * float(beta), beta_min, beta_max))

    def _ema_update(self, old_value, new_value, alpha):
        if old_value is None:
            return float(new_value)
        return (1.0 - alpha) * old_value + alpha * float(new_value)

    def _update_adaptive_kl_betas(
        self,
        depth_recon_loss,
        depth_sequence_recon_loss,
        proprio_recon_loss,
        privileged_terrain_recon_loss,
        privileged_dynamics_recon_loss,
        mixer_recon_loss,
    ):
        """Update KL betas from EMA reconstruction losses after each PPO update."""
        if not self.use_adaptive_kl_beta:
            return

        alpha = float(self.adaptive_kl_beta_ema_alpha)
        self.depth_frame_kl_recon_ema = self._ema_update(
            self.depth_frame_kl_recon_ema,
            depth_recon_loss,
            alpha,
        )
        self.depth_sequence_kl_recon_ema = self._ema_update(
            self.depth_sequence_kl_recon_ema,
            depth_sequence_recon_loss,
            alpha,
        )
        self.proprio_kl_recon_ema = self._ema_update(
            self.proprio_kl_recon_ema,
            proprio_recon_loss,
            alpha,
        )
        self.privileged_terrain_kl_recon_ema = self._ema_update(
            self.privileged_terrain_kl_recon_ema,
            privileged_terrain_recon_loss,
            alpha,
        )
        self.privileged_dynamics_kl_recon_ema = self._ema_update(
            self.privileged_dynamics_kl_recon_ema,
            privileged_dynamics_recon_loss,
            alpha,
        )
        self.mixer_kl_recon_ema = self._ema_update(
            self.mixer_kl_recon_ema,
            mixer_recon_loss,
            alpha,
        )

        self.depth_frame_kl_weight = self._update_kl_weight(
            self.depth_frame_kl_weight,
            self.depth_frame_kl_recon_target,
            self.depth_frame_kl_recon_ema,
            self.depth_frame_kl_beta_min,
            self.depth_frame_kl_beta_max,
        )

        self.depth_sequence_kl_weight = self._update_kl_weight(
            self.depth_sequence_kl_weight,
            self.depth_sequence_kl_recon_target,
            self.depth_sequence_kl_recon_ema,
            self.depth_sequence_kl_beta_min,
            self.depth_sequence_kl_beta_max,
        )

        self.proprio_kl_weight = self._update_kl_weight(
            self.proprio_kl_weight,
            self.proprio_kl_recon_target,
            self.proprio_kl_recon_ema,
            self.proprio_kl_beta_min,
            self.proprio_kl_beta_max,
        )

        self.privileged_terrain_kl_weight = self._update_kl_weight(
            self.privileged_terrain_kl_weight,
            self.privileged_terrain_kl_recon_target,
            self.privileged_terrain_kl_recon_ema,
            self.privileged_terrain_kl_beta_min,
            self.privileged_terrain_kl_beta_max,
        )

        self.privileged_dynamics_kl_weight = self._update_kl_weight(
            self.privileged_dynamics_kl_weight,
            self.privileged_dynamics_kl_recon_target,
            self.privileged_dynamics_kl_recon_ema,
            self.privileged_dynamics_kl_beta_min,
            self.privileged_dynamics_kl_beta_max,
        )

        self.versatility_lambda_e = self._update_kl_weight(
            self.versatility_lambda_e,
            self.mixer_kl_recon_target,
            self.mixer_kl_recon_ema,
            self.mixer_kl_beta_min,
            self.mixer_kl_beta_max,
        )

    def act(self, obs, critic_obs, obs_history, privileged_obs_history, depth_image, depth_latent_history, depth_torso_state, terrain_map):
        # if self.actor_critic.is_recurrent:
        #     self.transition.hidden_states = self.actor_critic.get_hidden_states()
        if self.use_boot:
            all_actions, body_velo_est, _, latest_depth_latent = self.actor_critic.act_with_estimates_and_depth_latent(
                obs,
                obs_history,
                depth_image=depth_image,
                depth_latent_history=depth_latent_history,
                depth_torso_state=depth_torso_state,
            )
        else:
            all_actions, body_velo_est, _, latest_depth_latent = (
                self.actor_critic.act_bootmask_with_estimates_and_depth_latent(
                    obs,
                    obs_history,
                    depth_image=depth_image,
                    depth_latent_history=depth_latent_history,
                    depth_torso_state=depth_torso_state,
                )
            )
        all_actions = all_actions.detach()
        body_velo_est = body_velo_est.detach()
        latest_depth_latent = latest_depth_latent.detach()

        # Compute the action distribution statistics and value for storage.
        self.transition.actions =  all_actions
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        # Only adaptive KL scheduling needs old action distribution moments.
        if self.storage.store_action_distribution:
            self.transition.action_mean = self.actor_critic.action_mean.detach()
            self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.observation_history = obs_history
        # Store the pre-step tensors that generated the sampled action.
        self.transition.privileged_observation_history = privileged_obs_history
        self.transition.depth_images = depth_image
        self.transition.depth_latent_history = depth_latent_history
        self.transition.depth_torso_state = depth_torso_state
        self.transition.terrain_map = terrain_map
        return all_actions, latest_depth_latent, body_velo_est
    
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
        if self.use_adaptive_entropy:
            self.current_entropy_coef = coef
        else:
            self.entropy_coef = coef

    def update_adaptive_entropy_coef(self, performance_metrics):
        # Prefer range-invariant command-tracking percentages from the KITE
        # command curriculum. These are normalized by episode length and the
        # active reward scale, so 1.0 means the policy attained the maximum
        # possible tracking reward for the sampled command range.
        lin_vel_tracking = performance_metrics.get(
            "lin_vel_tracking_pct",
            performance_metrics.get("lin_vel_tracking", 0.0),
        )
        ang_vel_tracking = performance_metrics.get(
            "ang_vel_tracking_pct",
            performance_metrics.get("ang_vel_tracking", 0.0),
        )
        terrain_level = performance_metrics.get("terrain_level", 0.0)

        gaps = torch.tensor(
            [
                max(0.0, self.ent_linvelo_threshold - lin_vel_tracking)
                / self.ent_linvelo_threshold
                if self.ent_linvelo_threshold > 0
                else 0.0,
                max(0.0, self.ent_angvelo_threshold - ang_vel_tracking)
                / self.ent_angvelo_threshold
                if self.ent_angvelo_threshold > 0
                else 0.0,
                max(0.0, self.ent_terrain_threshold - terrain_level)
                / self.ent_terrain_threshold
                if self.ent_terrain_threshold > 0
                else 0.0,
            ],
            dtype=torch.float32,
        )
        weights = F.softmax(gaps / self.ent_softmax_temperature, dim=0)
        weighted_gap = torch.sum(weights * gaps).item()
        low, high = self.entropy_coef_bounds
        self.current_entropy_coef = low + weighted_gap * (high - low)
        
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

    def _zero_all_encoder_optimizers(self):
        """Zero every non-privileged encoder optimizer for merged aux updates."""
        self.enc_optimizer.zero_grad(set_to_none=True)

    def _step_all_encoder_optimizers(self):
        """Step every non-privileged encoder optimizer after one merged backward."""
        self.enc_optimizer.step()

    def _depth_frame_autoencoder_update(self, depth_images_batch, depth_torso_state_batch, mask):
        self.actor_critic.depth_frame_autoencoder.train()
        self.depth_frame_optimizer.zero_grad(set_to_none=True)
        
        # Keep the depth-frame autoencoder in fp32. Its decoder computes a
        # reciprocal transform, and mixing bf16 activations with fp32 geometry
        # outputs can fail during backward. The training-only wrapper exposes
        # U-Net reconstruction skips while preserving z as the policy bottleneck.
        depth_recon, depth_mean, depth_logvar, latest_depth_z, depth_aux = self.actor_critic.depth_frame_autoencoder(
            depth_images_batch,
            depth_torso_state_batch,
        )
        transform_matrices = depth_aux["transform_matrices"].float()
        
        # Composite reconstruction loss: pixel L1 + depth-gradient loss +
        # SSIM loss. Each component averages only across non-terminated
        # rollout samples according to the provided mask.
        depth_recon_loss = self.depth_frame_reconstruction_loss(
            depth_recon,
            depth_images_batch,
            mask,
        )
        
        # Calculate the depth-encoder KL-divergency loss
        depth_kl = self._kl_loss(depth_mean, depth_logvar, mask)

        # Regularize the shared encoder/decoder transform toward the identity.
        transform_identity_loss = self._transform_identity_loss(transform_matrices, mask)
        
        # Calculate the total depth-frame reconstruction loss
        depth_frame_loss = (
            self.depth_frame_recon_weight * depth_recon_loss
            + self.depth_frame_kl_weight * depth_kl
            + self.depth_transform_identity_weight * transform_identity_loss
        )
        # The reconstruction loss keeps the needed graph references; drop the
        # large image tensor before building the other auxiliary branches.
        del depth_recon

        # Step, norm-clip, and step optimizers for the joint autoencoder.
        depth_frame_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.depth_frame_encoder.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor_critic.depth_frame_decoder.parameters(), self.max_grad_norm)
        self.depth_frame_optimizer.step()

        return (
            depth_frame_loss,
            depth_recon_loss,
            depth_kl,
            transform_identity_loss,
            latest_depth_z.detach(),
            depth_logvar.detach(),
        )

    def _privileged_encoder_decoder_updates(self, terrain_maps_batch, privileged_obs_history_batch, obs_target, mask):
        # Terrain and dynamics teacher updates are intentionally separated.
        # The terrain decoder has the large map-shaped activation graph, so we
        # step and release it before building the dynamics reconstruction graph.
        self.privileged_optimizer.zero_grad(set_to_none=True)

        with self._aux_autocast():
            # Encode -> Decode the height + surface normal map
            terrain_mean, terrain_logvar, terrain_priv_z = self.priv_terrain_encoder(terrain_maps_batch)
            terrain_priv_recon = self.priv_terrain_decoder(terrain_priv_z)
            
            # terrain recon log
            privileged_terrain_loss, _ = self._terrain_recon_loss(
                terrain_priv_recon,
                terrain_maps_batch,
                mask,
            )
            privileged_terrain_kl = self._kl_loss(terrain_mean, terrain_logvar, mask)
            privileged_terrain_total = (
                privileged_terrain_loss
                + self.privileged_terrain_kl_weight * privileged_terrain_kl
            )
        privileged_terrain_total.backward()
        nn.utils.clip_grad_norm_(
            list(self.priv_terrain_encoder.parameters())
            + list(self.priv_terrain_decoder.parameters()),
            self.max_grad_norm,
        )
        self.privileged_optimizer.step()
        privileged_terrain_log = privileged_terrain_loss.detach()
        privileged_terrain_kl_log = privileged_terrain_kl.detach()
        del terrain_mean, terrain_logvar, terrain_priv_z, terrain_priv_recon
        del privileged_terrain_loss, privileged_terrain_kl, privileged_terrain_total
        self._empty_cache_if_debugging()

        self.privileged_optimizer.zero_grad(set_to_none=True)
        with self._aux_autocast():
            # Encode -> decode the privileged observation history -> next time-step privileged obs
            dyn_mean, dyn_logvar, dyn_priv_z = self.priv_dynamics_encoder(privileged_obs_history_batch)
            dyn_priv_recon = self.priv_dynamics_decoder(dyn_priv_z)
            privileged_dynamics_loss = self._masked_mse_loss(
                dyn_priv_recon,
                obs_target,
                mask,
            )
            privileged_dynamics_kl = self._kl_loss(dyn_mean, dyn_logvar, mask)
            privileged_dynamics_total = (
                privileged_dynamics_loss
                + self.privileged_dynamics_kl_weight * privileged_dynamics_kl
            )
        privileged_dynamics_total.backward()
        nn.utils.clip_grad_norm_(
            list(self.priv_dynamics_encoder.parameters())
            + list(self.priv_dynamics_decoder.parameters()),
            self.max_grad_norm,
        )
        self.privileged_optimizer.step()
        privileged_dynamics_log = privileged_dynamics_loss.detach()
        privileged_dynamics_kl_log = privileged_dynamics_kl.detach()
        del dyn_mean, dyn_logvar, dyn_priv_z, dyn_priv_recon
        del privileged_dynamics_loss, privileged_dynamics_kl, privileged_dynamics_total

        privileged_loss = (
            privileged_terrain_log
            + privileged_dynamics_log
            + self.privileged_terrain_kl_weight * privileged_terrain_kl_log
            + self.privileged_dynamics_kl_weight * privileged_dynamics_kl_log
        )
        return (
            privileged_loss,
            privileged_dynamics_log,
            privileged_terrain_log,
            privileged_dynamics_kl_log,
            privileged_terrain_kl_log,
        )

    def _non_privileged_auxiliary_update(
        self,
        obs_hist_batch,
        depth_latent_history_batch,
        latest_depth_z,
        latest_depth_logvar,
        terrain_maps_batch,
        contrastive_negative_anchor_batch,
        explicit_labels_batch,
        obs_target,
        terrain_positive,
        dynamics_positive,
        mask,
        depth_frame_loss,
        depth_recon_loss,
        depth_kl,
        transform_identity_loss,
    ):
        """Merge sequence, proprioceptive, and mixer auxiliary updates into one step."""
        self.actor_critic.depth_sequence_encoder.train()
        self.actor_critic.proprio_context_encoder.train()
        self.actor_critic.context_encoder.train()
        self.priv_terrain_decoder.eval()
        self.priv_dynamics_decoder.eval()

        # The depth-frame autoencoder is updated immediately before this
        # function. Zero the shared encoder optimizer again so this backward
        # only contributes sequence/proprio/mixer gradients.
        self._zero_all_encoder_optimizers()

        with self._aux_autocast():
            # 1. Depth-sequence student path. The latest frame latent comes
            # from the standalone frame update as a detached tensor, so this
            # graph starts at the sequence encoder.
            depth_sequence = torch.cat(
                [depth_latent_history_batch, latest_depth_z.unsqueeze(1)],
                dim=1,
            )
            seq_mean, seq_logvar, depth_seq_z = (
                self.actor_critic.depth_sequence_encoder(
                    depth_sequence,
                    latest_depth_logvar,
                )
            )

            seq_kl = self._kl_loss(seq_mean, seq_logvar, mask)
            depth_seq_contrast_z = self.actor_critic.depth_sequence_contrastive_head(
                depth_seq_z
            )
            depth_seq_recon_z = self.actor_critic.depth_sequence_recon_head(
                depth_seq_z
            )
            with self._frozen_module_params(self.priv_terrain_decoder):
                terrain_recon_from_depth_seq = self.priv_terrain_decoder(depth_seq_recon_z)
            seq_terrain_loss, seq_terrain_recon_log = self._terrain_recon_loss(
                terrain_recon_from_depth_seq,
                terrain_maps_batch,
                mask,
            )
            seq_terrain_contrast_loss = self._normalized_contrastive_loss(
                depth_seq_contrast_z,
                terrain_positive,
                contrastive_negative_anchor_batch,
                mask,
            )
            depth_sequence_loss = (
                self.depth_sequence_kl_weight * seq_kl
                + self.depth_sequence_terrain_weight * seq_terrain_loss
                + self.contrastive_weight * seq_terrain_contrast_loss
            )

            # 2. Proprioceptive student path.
            prop_mean, prop_logvar, proprio_z = (
                self.actor_critic.proprio_context_encoder(obs_hist_batch)
            )

            prop_kl = self._kl_loss(prop_mean, prop_logvar, mask)
            proprio_contrast_z = self.actor_critic.proprio_contrastive_head(
                proprio_z
            )
            proprio_recon_z = self.actor_critic.proprio_recon_head(
                proprio_z
            )
            with self._frozen_module_params(self.priv_dynamics_decoder):
                dyn_recon_from_proprio = self.priv_dynamics_decoder(proprio_recon_z)
            
            prop_dyn_loss = self._masked_mse_loss(
                dyn_recon_from_proprio,
                obs_target,
                mask,
            )
            
            prop_dyn_contrast_loss = self._normalized_contrastive_loss(
                proprio_contrast_z,
                dynamics_positive,
                contrastive_negative_anchor_batch,
                mask,
            )
            
            proprio_loss = (
                self.proprio_kl_weight * prop_kl
                + self.proprio_dynamics_weight * prop_dyn_loss
                + self.contrastive_weight * prop_dyn_contrast_loss
            )

            # 3. Modality mixer path.
            mix_mean, mix_logvar, mix_z, _, _ = (
                self.actor_critic.context_encoder(
                    depth_seq_z.detach().clone(),
                    proprio_z.detach().clone(),
                )
            )
            body_velo_est = self.actor_critic.context_encoder.velo_est_out(
                self.actor_critic.context_encoder.activation(
                    self.actor_critic.context_encoder.velo_est_hidden(mix_z)
                )
            )

            feet_state_est = self.actor_critic.context_encoder.est_feet(mix_z)

            body_velo_dim = self.actor_critic.body_velo_dim
            feet_state_dim = self.actor_critic.feet_state_dim
            
            torso_velo_explicit_loss = self._masked_mse_loss(
                body_velo_est,
                explicit_labels_batch[:, :body_velo_dim],
                mask,
            )
            
            if feet_state_dim > 0:
                feet_loss_terms = []

                contact_dim = min(4, feet_state_dim)
                if contact_dim > 0:
                    feet_contact_loss = self._masked_bce_with_logits_loss(
                        feet_state_est[:, :contact_dim],
                        explicit_labels_batch[
                            :, body_velo_dim:body_velo_dim + contact_dim
                        ].float(),
                        mask,
                    )
                    feet_loss_terms.append(feet_contact_loss)

                height_dim = min(4, max(feet_state_dim - 4, 0))
                if height_dim > 0:
                    height_start = body_velo_dim + 4
                    feet_height_loss = self._masked_mse_loss(
                        feet_state_est[:, 4:4 + height_dim],
                        explicit_labels_batch[
                            :, height_start:height_start + height_dim
                        ],
                        mask,
                    )
                    feet_loss_terms.append(feet_height_loss)

                normal_dim = max(feet_state_dim - 8, 0)
                if normal_dim > 0:
                    normal_start = body_velo_dim + 8
                    pred_surface_norm_under_feet = feet_state_est[:, 8:8 + normal_dim]
                    gt_surface_norm_under_feet = explicit_labels_batch[
                        :, normal_start:normal_start + normal_dim
                    ]
                    feet_normal_loss = self._masked_sample_mean(
                        1.0 - F.cosine_similarity(
                            pred_surface_norm_under_feet,
                            gt_surface_norm_under_feet,
                            dim=-1,
                            eps=1e-6,
                        ),
                        mask,
                    )
                    feet_loss_terms.append(feet_normal_loss)
                    del pred_surface_norm_under_feet, gt_surface_norm_under_feet

                feet_state_explicit_loss = sum(feet_loss_terms)
            else:
                feet_state_explicit_loss = torso_velo_explicit_loss.new_zeros(())

            explicit_loss = feet_state_explicit_loss + torso_velo_explicit_loss
            
            modality_kl = self._kl_loss(mix_mean, mix_logvar, mask)

            # versatility_loss, versatility_log = self._versatility_metric(
            #     mix_mean,
            #     mix_logvar,
            #     mask,
            # )
            # modality_loss = (
            #     self.modality_explicit_weight * explicit_loss
            #     + self.versatility_weight * versatility_loss
            # )

            modality_loss = self.modality_explicit_weight * explicit_loss + self.versatility_lambda_e * modality_kl

        non_privileged_loss = (
            depth_sequence_loss
            + proprio_loss
            + modality_loss
        )
        non_privileged_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor_critic.depth_sequence_encoder.parameters(),
            self.max_grad_norm,
        )
        nn.utils.clip_grad_norm_(
            self.actor_critic.depth_sequence_recon_head.parameters(),
            self.max_grad_norm,
        )
        nn.utils.clip_grad_norm_(
            self.actor_critic.proprio_context_encoder.parameters(),
            self.max_grad_norm,
        )
        nn.utils.clip_grad_norm_(
            self.actor_critic.proprio_recon_head.parameters(),
            self.max_grad_norm,
        )
        nn.utils.clip_grad_norm_(
            self.actor_critic.context_encoder.parameters(),
            self.max_grad_norm,
        )
        self._step_all_encoder_optimizers()

        return {
            "non_privileged_loss": depth_frame_loss + non_privileged_loss,
            "depth_frame_loss": depth_frame_loss,
            "depth_recon_loss": depth_recon_loss,
            "depth_kl": depth_kl,
            "transform_identity_loss": transform_identity_loss,
            "depth_sequence_loss": depth_sequence_loss,
            "seq_kl": seq_kl,
            "seq_terrain_loss": seq_terrain_loss,
            "seq_terrain_recon_log": seq_terrain_recon_log,
            "seq_terrain_contrast_loss": seq_terrain_contrast_loss,
            "terrain_recon_from_depth_seq": terrain_recon_from_depth_seq,
            "proprio_loss": proprio_loss,
            "prop_kl": prop_kl,
            "prop_dyn_loss": prop_dyn_loss,
            "prop_dyn_contrast_loss": prop_dyn_contrast_loss,
            "dyn_recon_from_proprio": dyn_recon_from_proprio,
            "modality_loss": modality_loss,
            "explicit_loss": explicit_loss,
            "torso_velo_explicit_loss": torso_velo_explicit_loss,
            "feet_state_explicit_loss": feet_state_explicit_loss,
            # "versatility_loss": versatility_loss,
            # "versatility_log": versatility_log,
            "modality_kl":modality_kl,
            "body_velo_est": body_velo_est,
        }

    def _update_auxiliary_encoders(self, obs_hist_batch, privileged_obs_history_batch, depth_images_batch, 
                                   depth_latent_history_batch, depth_torso_state_batch, terrain_maps_batch, 
                                   contrastive_negative_anchor_batch, explicit_labels_batch, obs_target, 
                                   terminated_batch,
    ):
        mask = terminated_batch.float()
        losses = {}
        profile = {} if self.profile_learning else None

        def profile_mark(name, start_time):
            if profile is None:
                return None
            # CUDA work is asynchronous; synchronize only during profiling so
            # per-phase timings reflect actual GPU completion time.
            self._sync_if_debugging()
            profile[name] = profile.get(name, 0.0) + (time.perf_counter() - start_time)
            return time.perf_counter()

        # Shared privileged anchors for contrastive alignment. These are
        # detached because the student-side encoders are updated by the merged
        # non-privileged auxiliary step below.
        if profile is not None:
            self._sync_if_debugging()
            t_profile = time.perf_counter()
        terrain_positive = dynamics_positive = None
        with torch.no_grad():
            _, _, terrain_positive = self.priv_terrain_encoder(terrain_maps_batch)
            _, _, dynamics_positive = self.priv_dynamics_encoder(privileged_obs_history_batch)
        if profile is not None:
            t_profile = profile_mark("aux_positive_anchors", t_profile)

        # 1. Update the single-frame depth autoencoder in its own graph. This
        # releases depth-image decoder activations before building the larger
        # sequence/proprio/mixer graph.
        (
            depth_frame_loss,
            depth_recon_loss,
            depth_kl,
            transform_identity_loss,
            latest_depth_z,
            latest_depth_logvar,
        ) = self._depth_frame_autoencoder_update(
            depth_images_batch,
            depth_torso_state_batch,
            mask,
        )
        self._empty_cache_if_debugging()
        if profile is not None:
            t_profile = profile_mark("aux_depth_frame", t_profile)

        # 2-4. Merge the depth-sequence, proprioceptive, and modality-mixer
        # student updates into one forward/backward/step.
        aux = self._non_privileged_auxiliary_update(
            obs_hist_batch,
            depth_latent_history_batch,
            latest_depth_z,
            latest_depth_logvar,
            terrain_maps_batch,
            contrastive_negative_anchor_batch,
            explicit_labels_batch,
            obs_target,
            terrain_positive,
            dynamics_positive,
            mask,
            depth_frame_loss.detach(),
            depth_recon_loss.detach(),
            depth_kl.detach(),
            transform_identity_loss.detach(),
        )
        del latest_depth_z, latest_depth_logvar
        if profile is not None:
            t_profile = profile_mark("aux_sequence_proprio_mixer", t_profile)

        # Boot probability only needs compact aggregate statistics. Compute
        # them immediately after the mixer update, move them to CPU, and drop
        # the large CUDA reconstructions before the privileged teacher update.
        boot_summary = self._make_boot_summary(
            obs_target,
            aux["dyn_recon_from_proprio"].detach(),
            terrain_maps_batch,
            aux["terrain_recon_from_depth_seq"].detach(),
            explicit_labels_batch,
            aux["body_velo_est"].detach(),
            mask,
        )
        del aux["dyn_recon_from_proprio"]
        del aux["terrain_recon_from_depth_seq"]
        del aux["body_velo_est"]
        self._empty_cache_if_debugging()
        if profile is not None:
            t_profile = profile_mark("aux_boot_summary", t_profile)
        
        # 5. Privileged terrain/dynamics encoder-decoder update:
        #    refresh the teacher-side representations used as positive anchors.
        (
            privileged_loss,
            privileged_dynamics_loss,
            privileged_terrain_loss,
            privileged_dynamics_kl,
            privileged_terrain_kl,
        ) = self._privileged_encoder_decoder_updates(
            terrain_maps_batch,
            privileged_obs_history_batch,
            obs_target,
            mask,
        )
        if profile is not None:
            t_profile = profile_mark("aux_privileged_teacher", t_profile)

        # Total Encoder(s) update loss
        losses["total"] = self._detach_scalar(aux["non_privileged_loss"] + privileged_loss)

        # Explicity Estimation Loss
        losses["explicit"] = self._detach_scalar(aux["explicit_loss"])
        
        # Total reconstructed loss
        losses["recon"] = self._detach_scalar(
            aux["depth_recon_loss"]
            + aux["seq_terrain_loss"]
            + aux["prop_dyn_loss"]
        )

        # Total KL loss across all encoders
        losses["total_kl"] = self._detach_scalar(
            aux["depth_kl"]
            + aux["seq_kl"]
            + aux["prop_kl"]
            + privileged_terrain_kl
            + privileged_dynamics_kl
            # + aux["versatility_log"]["kl"]
            + aux["modality_kl"]
        )
        losses["decoder"] = self._detach_scalar(aux["depth_recon_loss"] + privileged_loss)
        losses["depth_transform_identity"] = self._detach_scalar(aux["transform_identity_loss"])
        losses["kl_scheduler"] = {
            "depth_frame_recon": self._detach_scalar(aux["depth_recon_loss"]),
            "depth_sequence_recon": self._detach_scalar(aux["seq_terrain_loss"]),
            "proprio_recon": self._detach_scalar(aux["prop_dyn_loss"]),
            "privileged_terrain_recon": self._detach_scalar(privileged_terrain_loss),
            "privileged_dynamics_recon": self._detach_scalar(privileged_dynamics_loss),
            "mixer_recon": self._detach_scalar(aux["explicit_loss"]),
        }
        
        losses["boot_summary"] = boot_summary
        # The positive anchors and detached student latents are no longer
        # needed after the auxiliary losses and CPU boot summaries are built.
        del terrain_positive, dynamics_positive
        del contrastive_negative_anchor_batch
        self._empty_cache_if_debugging()
        if profile is not None:
            t_profile = profile_mark("aux_loss_aggregation", t_profile)

        # Loss details for logging. The compact path logs per-model totals and
        # a small number of aggregate losses; detailed mode restores all leaves.
        losses["detail"] = {
            "depth_frame_total": self._detach_scalar(aux["depth_frame_loss"]),
            "depth_sequence_total": self._detach_scalar(aux["depth_sequence_loss"]),
            "proprio_total": self._detach_scalar(aux["proprio_loss"]),
            "modality_total": self._detach_scalar(aux["modality_loss"]),
            "depth_sequence_terrain_recon": self._detach_scalar(aux["seq_terrain_loss"]),
            "proprio_dynamics_recon": self._detach_scalar(aux["prop_dyn_loss"]),
            "privileged_total": self._detach_scalar(privileged_loss),
            "privileged_terrain_kl": self._detach_scalar(privileged_terrain_kl),
            "privileged_dynamics_kl": self._detach_scalar(privileged_dynamics_kl),
        }
        if self.log_detailed_encoder_losses:
            losses["detail"].update({
                "depth_frame_recon": self._detach_scalar(aux["depth_recon_loss"]),
                "depth_frame_kl": self._detach_scalar(aux["depth_kl"]),
                "depth_frame_transform_identity": self._detach_scalar(aux["transform_identity_loss"]),
                "depth_autoencoder_recon": self._detach_scalar(aux["depth_recon_loss"]),
                
                "depth_sequence_kl": self._detach_scalar(aux["seq_kl"]),
                "depth_sequence_terrain_recon": self._detach_scalar(aux["seq_terrain_loss"]),
                "depth_sequence_terrain_height": self._detach_scalar(
                    aux["seq_terrain_recon_log"]["height_loss"]
                ),
                "depth_sequence_terrain_normals": self._detach_scalar(
                    aux["seq_terrain_recon_log"]["normal_cos"]
                ),
                "depth_sequence_terrain_contrastive": self._detach_scalar(
                    aux["seq_terrain_contrast_loss"]
                ),
                
                "proprio_kl": self._detach_scalar(aux["prop_kl"]),
                "proprio_dynamics_recon": self._detach_scalar(aux["prop_dyn_loss"]),
                "proprio_dynamics_contrastive": self._detach_scalar(
                    aux["prop_dyn_contrast_loss"]
                ),
                # "modality_versatility": self._detach_scalar(aux["versatility_loss"]),
                # "modality_kl": self._detach_scalar(aux["versatility_log"]["kl"]),
                "modality_kl": self._detach_scalar(aux["modality_kl"]),
                # "modality_marginal_entropy": self._detach_scalar(
                #     aux["versatility_log"]["marginal_entropy"]
                # ),
                # "modality_conditional_entropy": self._detach_scalar(
                #     aux["versatility_log"]["conditional_entropy"]
                # ),
                # "modality_mutual_info": self._detach_scalar(
                #     aux["versatility_log"]["mutual_info"]
                # ),
                "modality_explicit": self._detach_scalar(aux["explicit_loss"]),
                "modality_explicit_torso_velo": self._detach_scalar(
                    aux["torso_velo_explicit_loss"]
                ),
                "modality_explicit_feet_state": self._detach_scalar(
                    aux["feet_state_explicit_loss"]
                ),
                "privileged_terrain_recon": self._detach_scalar(privileged_terrain_loss),
                "privileged_dynamics_recon": self._detach_scalar(privileged_dynamics_loss),
                "privileged_terrain_kl": self._detach_scalar(privileged_terrain_kl),
                "privileged_dynamics_kl": self._detach_scalar(privileged_dynamics_kl),
            })
        del aux
        if profile is not None:
            losses["profile"] = profile

        return losses

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_aux_loss = 0
        mean_explicit_loss = 0
        mean_reconstruction_loss = 0
        mean_kl_loss = 0
        mean_aux_decoder_loss = 0
        mean_aux_loss_details = {}
        mean_depth_kl_recon_signal = 0
        mean_depth_sequence_kl_recon_signal = 0
        mean_proprio_kl_recon_signal = 0
        mean_privileged_terrain_kl_recon_signal = 0
        mean_privileged_dynamics_kl_recon_signal = 0
        mean_mixer_kl_recon_signal = 0

        profile_update = self.profile_learning
        timers = None
        if profile_update:
            timers = {
                "update_wall": 0.0,
                "minibatch_wall": 0.0,
                "rl_loss": 0.0,
                "act_step": 0.0,
                "aux_update": 0.0,
                "boot_stats": 0.0,
                "boot_prob": 0.0,
                "spec_norm": 0.0,
            }

        boot_summary = None
        vel_boot_summary = None

        if profile_update:
            self._sync_if_debugging()
            update_wall_start = time.perf_counter()
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for terminated_batch, obs_batch, obs_hist_batch, privileged_obs_history_batch, depth_images_batch, depth_latent_history_batch, \
            depth_torso_state_batch, terrain_maps_batch, explicit_labels_batch, obs_target, actions_batch, \
                target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch in generator:
            if profile_update:
                self._sync_if_debugging()
                minibatch_wall_start = time.perf_counter()
            
            self.actor_critic.train()
            self.act_optimizer.zero_grad(set_to_none=True)
            self.enc_optimizer.zero_grad(set_to_none=True)

            if profile_update:
                self._sync_if_debugging()
                t0 = time.perf_counter()
            
            # Perform RL update
            ppo_loss, surrogate_loss, value_loss = self._compute_rl_loss(obs_batch, obs_hist_batch, actions_batch, privileged_obs_history_batch, depth_images_batch,
                                                                         depth_latent_history_batch, depth_torso_state_batch, terrain_maps_batch, old_sigma_batch,
                                                                         old_mu_batch, old_actions_log_prob_batch, advantages_batch, target_values_batch, returns_batch)
            
            if profile_update:
                self._sync_if_debugging()
                timers["rl_loss"] += time.perf_counter() - t0

            if profile_update:
                self._sync_if_debugging()
                t0 = time.perf_counter()

            ppo_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.act_optimizer.step()

            if profile_update:
                self._sync_if_debugging()
                timers["act_step"] += time.perf_counter() - t0
            
            # Accumulate detached scalar tensors and convert to Python floats
            # once after the update to avoid many per-minibatch GPU syncs.
            mean_value_loss += self._detach_scalar(value_loss)
            mean_surrogate_loss += self._detach_scalar(surrogate_loss)
            del ppo_loss, surrogate_loss, value_loss

            # Calculate the encoder update n-times
            for _ in range(self.num_enc_epochs):
                self.actor_critic.train()

                if profile_update:
                    self._sync_if_debugging()
                    t0 = time.perf_counter()

                contrastive_negative_anchor_batch = torch.empty(
                    depth_images_batch.shape[0],
                    self.actor_critic.depth_latent_dim,
                    device=depth_images_batch.device,
                    dtype=depth_images_batch.dtype,
                ).uniform_(-1.0, 1.0)

                aux_losses = self._update_auxiliary_encoders(obs_hist_batch, privileged_obs_history_batch, depth_images_batch, depth_latent_history_batch,
                                                             depth_torso_state_batch, terrain_maps_batch, contrastive_negative_anchor_batch,
                                                             explicit_labels_batch, obs_target, terminated_batch)
                
                if profile_update:
                    self._sync_if_debugging()
                    timers["aux_update"] += time.perf_counter() - t0
                    for profile_name, profile_value in aux_losses.get("profile", {}).items():
                        timers[profile_name] = timers.get(profile_name, 0.0) + profile_value

                if profile_update:
                    self._sync_if_debugging()
                    t0 = time.perf_counter()

                # Boot summaries are already reduced and moved to CPU inside
                #   the auxiliary update, so the PPO loop never holds large
                #   reconstruction tensors just for boot-probability bookkeeping.
                boot_summary = self._accumulate_boot_summary(
                    boot_summary,
                    aux_losses["boot_summary"],
                    ("dynamics", "terrain"),
                )
                vel_boot_summary = self._accumulate_boot_summary(
                    vel_boot_summary,
                    aux_losses["boot_summary"],
                    ("velocity",),
                )

                if profile_update:
                    timers["boot_stats"] += time.perf_counter() - t0

                # Log losses
                mean_aux_loss += aux_losses["total"]
                mean_explicit_loss += aux_losses["explicit"]
                mean_reconstruction_loss += aux_losses["recon"]
                mean_kl_loss += aux_losses["total_kl"]
                mean_aux_decoder_loss += aux_losses["decoder"]
                mean_depth_kl_recon_signal += aux_losses["kl_scheduler"][
                    "depth_frame_recon"
                ]
                mean_depth_sequence_kl_recon_signal += aux_losses["kl_scheduler"][
                    "depth_sequence_recon"
                ]
                mean_proprio_kl_recon_signal += aux_losses["kl_scheduler"][
                    "proprio_recon"
                ]
                mean_privileged_terrain_kl_recon_signal += aux_losses["kl_scheduler"][
                    "privileged_terrain_recon"
                ]
                mean_privileged_dynamics_kl_recon_signal += aux_losses["kl_scheduler"][
                    "privileged_dynamics_recon"
                ]
                mean_mixer_kl_recon_signal += aux_losses["kl_scheduler"][
                    "mixer_recon"
                ]
                
                for name, value in aux_losses["detail"].items():
                    mean_aux_loss_details[name] = (mean_aux_loss_details.get(name, 0.0) + value)
                del aux_losses, contrastive_negative_anchor_batch

            # Keeps the interaction of incoming data with layer wieghts below the threashold that 
            #     saturates the tanh activation function.
            if profile_update:
                self._sync_if_debugging()
                t0 = time.perf_counter()
            self.spectral_normalization(self.actor_critic, sigma_max=6.0)

            if profile_update:
                self._sync_if_debugging()
                timers["spec_norm"] += time.perf_counter() - t0
                timers["minibatch_wall"] += time.perf_counter() - minibatch_wall_start

            # Release large minibatch references before the generator yields
            #   the next batch. This is mainly a peak-VRAM guard for the depth
            #   image and terrain-map tensors.
            del terminated_batch, obs_batch, obs_hist_batch
            del privileged_obs_history_batch, depth_images_batch
            del depth_latent_history_batch, depth_torso_state_batch
            del terrain_maps_batch, explicit_labels_batch, obs_target
            del actions_batch, target_values_batch, advantages_batch
            del returns_batch, old_actions_log_prob_batch
            del old_mu_batch, old_sigma_batch
            self._empty_cache_if_debugging()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss = self._finish_log_scalar(mean_value_loss / num_updates)
        mean_surrogate_loss = self._finish_log_scalar(mean_surrogate_loss / num_updates)

        mean_aux_loss = self._finish_log_scalar(mean_aux_loss / (num_updates * self.num_enc_epochs))
        mean_aux_decoder_loss = self._finish_log_scalar(mean_aux_decoder_loss / (num_updates * self.num_enc_epochs))
        mean_kl_loss = self._finish_log_scalar(mean_kl_loss / (num_updates * self.num_enc_epochs))
        mean_explicit_loss = self._finish_log_scalar(mean_explicit_loss / (num_updates * self.num_enc_epochs))
        mean_reconstruction_loss = self._finish_log_scalar(mean_reconstruction_loss / (num_updates * self.num_enc_epochs))
        mean_depth_kl_recon_signal = self._finish_log_scalar(
            mean_depth_kl_recon_signal / (num_updates * self.num_enc_epochs)
        )
        mean_depth_sequence_kl_recon_signal = self._finish_log_scalar(
            mean_depth_sequence_kl_recon_signal / (num_updates * self.num_enc_epochs)
        )
        mean_proprio_kl_recon_signal = self._finish_log_scalar(
            mean_proprio_kl_recon_signal / (num_updates * self.num_enc_epochs)
        )
        mean_privileged_terrain_kl_recon_signal = self._finish_log_scalar(
            mean_privileged_terrain_kl_recon_signal / (num_updates * self.num_enc_epochs)
        )
        mean_privileged_dynamics_kl_recon_signal = self._finish_log_scalar(
            mean_privileged_dynamics_kl_recon_signal / (num_updates * self.num_enc_epochs)
        )
        mean_mixer_kl_recon_signal = self._finish_log_scalar(
            mean_mixer_kl_recon_signal / (num_updates * self.num_enc_epochs)
        )
        for name in mean_aux_loss_details:
            mean_aux_loss_details[name] = self._finish_log_scalar(
                mean_aux_loss_details[name] / (num_updates * self.num_enc_epochs)
            )

        self._update_adaptive_kl_betas(
            mean_depth_kl_recon_signal,
            mean_depth_sequence_kl_recon_signal,
            mean_proprio_kl_recon_signal,
            mean_privileged_terrain_kl_recon_signal,
            mean_privileged_dynamics_kl_recon_signal,
            mean_mixer_kl_recon_signal,
        )
        mean_aux_loss_details["kl_beta_depth_frame"] = self.depth_frame_kl_weight
        mean_aux_loss_details["kl_beta_depth_sequence"] = self.depth_sequence_kl_weight
        mean_aux_loss_details["kl_beta_proprio"] = self.proprio_kl_weight
        mean_aux_loss_details["kl_beta_privileged_terrain"] = self.privileged_terrain_kl_weight
        mean_aux_loss_details["kl_beta_privileged_dynamics"] = self.privileged_dynamics_kl_weight
        mean_aux_loss_details["kl_beta_mixer"] = self.versatility_lambda_e
        if self.depth_frame_kl_recon_ema is not None:
            mean_aux_loss_details["kl_recon_ema_depth_frame"] = self.depth_frame_kl_recon_ema
        if self.depth_sequence_kl_recon_ema is not None:
            mean_aux_loss_details["kl_recon_ema_depth_sequence"] = self.depth_sequence_kl_recon_ema
        if self.proprio_kl_recon_ema is not None:
            mean_aux_loss_details["kl_recon_ema_proprio"] = self.proprio_kl_recon_ema
        if self.privileged_terrain_kl_recon_ema is not None:
            mean_aux_loss_details["kl_recon_ema_privileged_terrain"] = self.privileged_terrain_kl_recon_ema
        if self.privileged_dynamics_kl_recon_ema is not None:
            mean_aux_loss_details["kl_recon_ema_privileged_dynamics"] = self.privileged_dynamics_kl_recon_ema
        if self.mixer_kl_recon_ema is not None:
            mean_aux_loss_details["kl_recon_ema_mixer"] = self.mixer_kl_recon_ema


        if profile_update:
            self._sync_if_debugging()
            t0 = time.perf_counter()

        # Estimate whether the modality mixer reconstructs privileged dynamics
        #   plus terrain better than a mean predictor. All statistics live on CPU.
        pboot = self._boot_probability_from_summary(boot_summary)
        vel_pboot = self._boot_probability_from_summary(vel_boot_summary)

        if profile_update:
            timers["boot_prob"] += time.perf_counter() - t0
            timers["update_wall"] = time.perf_counter() - update_wall_start

        # Use the (scaled) ratio of mean-prediction performance to actual prediction performance
        #     to determine if encoder bootstrapping is performed.
        self.use_boot = random.random() < pboot
        self.use_depth_vel_boot = random.random() < vel_pboot
        print("Use bootstrapped Encoder Dynamics: ", self.use_boot)
        print("Use bootstrapped Depth Torso Velocity: ", self.use_depth_vel_boot)

        self.storage.clear()

        if profile_update:
            # Get the average time for the various tracked timers
            for key in timers.keys():
                if key not in ("boot_prob", "update_wall"):
                    timers[key] /= num_updates
            self.last_update_timers = {k: float(v) for k, v in timers.items()}
            print("update timers:", {k: round(v, 4) for k, v in timers.items()})
        else:
            self.last_update_timers = {}

        return mean_value_loss, mean_surrogate_loss, mean_aux_loss, mean_aux_decoder_loss, \
            mean_explicit_loss, mean_reconstruction_loss, mean_kl_loss, mean_aux_loss_details

    def _compute_rl_loss(self, obs_batch, obs_hist_batch,
                         actions_batch, privileged_obs_history_batch,
                         depth_images_batch, depth_latent_history_batch,
                         depth_torso_state_batch, terrain_maps_batch,
                         old_sigma_batch, old_mu_batch,
                         old_actions_log_prob_batch,
                         advantages_batch, target_values_batch, returns_batch):
        # The RL optimizer only owns actor/critic parameters. Build the actor
        #   conditioning from frozen encoder outputs so PPO does not spend time
        #   or VRAM constructing encoder graphs that auxiliary losses train later.
        with torch.no_grad():
            _, _, z, body_velo_est, feet_state_est = self.actor_critic.cenet_enc_forward(
                obs_hist_batch,
                obs=obs_batch,
                depth_image=depth_images_batch,
                depth_latent_history=depth_latent_history_batch,
                depth_torso_state=depth_torso_state_batch,
            )
            context_state = torch.cat([body_velo_est, feet_state_est], dim=-1)
            if self.use_boot:
                actor_context = torch.cat((z, context_state), dim=-1)
            else:
                actor_context = torch.zeros(
                    (obs_batch.shape[0], z.shape[1] + context_state.shape[1]),
                    device=obs_batch.device,
                    dtype=obs_batch.dtype,
                )
            current_obs = torch.cat((obs_batch, actor_context), dim=-1)

        self.actor_critic.update_distribution(current_obs)

        # PPO action log-probability and value estimates for the mini-batch.
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        critic_obs_batch = self.build_critic_obs(
            privileged_obs_history_batch,
            terrain_maps_batch,
        )
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

        entropy_coef = (
            self.current_entropy_coef
            if self.use_adaptive_entropy
            else self.entropy_coef
        )
        ppo_loss = (
            surrogate_loss
            + self.value_loss_coef * value_loss
            - entropy_coef * entropy_batch.mean()
        )

        return ppo_loss, surrogate_loss, value_loss
