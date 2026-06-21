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

import numpy as np
import random

from rsl_rl.modules import ActorCritic_KITE
from rsl_rl.modules.module_utils import ContrastiveProjectionHead, ReconDimensionProjectionHead
from rsl_rl.modules.kite_privileged_encoders import (
    PrivDynamicsDecoder,
    PrviDynamicsMLPMixerKITE,
    TerrainAttentionEncoder,
    TerrainTwoHeadDecoder,
)
from rsl_rl.modules.kite_visual_encoder import MotionRobustDepthDecoder
from rsl_rl.storage import RolloutStorageKITE

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
                 terrain_decoder_hidden_dim=128,
                 terrain_decoder_encoded_spatial_dim=(3,4),
                 terrain_decoder_channels=64,
                 privileged_dynamics_decoder_layers=[32,64,128,256],
                 priv_mixer_num_blocks=2,
                 priv_mixer_hidden_dim=128,
                 priv_mixer_token_dim=32,
                 priv_mixer_channel_dim=64,
                 priv_mixer_use_layer_norm=True,

                 depth_frame_recon_weight=1.0,
                 depth_frame_kl_weight=1.0e-3,
                 depth_transform_identity_weight=1.0e-3,

                 depth_sequence_terrain_weight=1.0,
                 depth_sequence_kl_weight=1.0,

                 proprio_dynamics_weight=1.0,
                 proprio_kl_weight=1.0,
                 
                 modality_terrain_weight=1.0,
                 modality_dynamics_weight=1.0,
                 modality_explicit_weight=1.0,
                 
                 contrastive_weight=0.1,
                 contrastive_lambda=0.5,
                 contrastive_margin=1.0,
                 
                 versatility_weight=0.01,
                 versatility_lambda_e=0.1,
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
        self.depth_sequence_terrain_weight = depth_sequence_terrain_weight
        self.depth_sequence_kl_weight = depth_sequence_kl_weight
        
        # loss weights for proprioceptive history encoder losses
        self.proprio_dynamics_weight = proprio_dynamics_weight
        self.proprio_kl_weight = proprio_kl_weight
        
        # modality-mixer network loss component weights
        self.modality_terrain_weight = modality_terrain_weight
        self.modality_dynamics_weight = modality_dynamics_weight
        self.modality_explicit_weight = modality_explicit_weight
        self.versatility_weight = versatility_weight
        self.versatility_lambda_e = versatility_lambda_e
        
        # Contrastive loss weights/margin shared across all contrastive updates
        self.contrastive_weight = contrastive_weight
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_margin = contrastive_margin

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later

        # Optimizer for actor network and all non-privileged encoders. Currently, no parameter overlap between encoder and actor
        self.act_optimizer, self.enc_optimizer = actor_critic.configure_optimizers(learning_rate)
        
        # Transition data structure from storage class
        self.transition = RolloutStorageKITE.Transition()

        # Reduce the LR of the critic
        for param_group in self.act_optimizer.param_groups:
            if "name" in param_group.keys():
                if "critic" in param_group["name"]:
                    param_group['lr'] = (learning_rate / 3.0)


        ##
        ##  Construct depth-image decoder and privileged encoder-decoder pairs
        terrain_h, terrain_w, _ = terrain_map_shape or (1, 1, 4)

        # Single-depth image decoder network
        self.depth_decoder = MotionRobustDepthDecoder(
            depth_image_resolution=actor_critic.depth_image_resolution,
            target_latent_dim=actor_critic.depth_latent_dim,
        ).to(self.device)

        # Privileged Terrain encoder network
        self.priv_terrain_encoder = TerrainAttentionEncoder(
            height=terrain_h,
            width=terrain_w,
            latent_dim=privileged_terrain_latent_dim,
            cnn_activation=priv_activation_func,
            norm_type=cnn_norm_type,
            attention_dim=terrain_encoder_attention_dim,
            n_heads=terrain_encoder_n_heads,
        ).to(self.device)

        # Two-headed (height, surface normals) privileged terrain decoder network
        self.priv_terrain_decoder = TerrainTwoHeadDecoder(
            height=terrain_h,
            width=terrain_w,
            latent_dim=privileged_terrain_latent_dim,
            cnn_activation=priv_activation_func,
            decoder_hidden_dim=terrain_decoder_hidden_dim,
            encoded_spatial_dim=terrain_decoder_encoded_spatial_dim,
            decoder_channels=terrain_decoder_channels,
            norm_type=cnn_norm_type,
        ).to(self.device)
        
        
        # Privileged dynamics MLP-Mixer encoder (treats time-step obs features as tokens, time-steps as channels)        
        self.priv_dynamics_encoder = PrviDynamicsMLPMixerKITE(
            context_input_dim=self.num_priv_obs_history * self.num_priv_obs,
            num_tokens=self.num_priv_obs,
            input_dim_per_token=self.num_priv_obs_history,
            context_latent_size=privileged_dynamics_latent_dim,
            activation=priv_activation_func,
            num_mixer_blocks=priv_mixer_num_blocks,
            hidden_dim=priv_mixer_hidden_dim,
            token_dim=priv_mixer_token_dim,
            channel_dim=priv_mixer_channel_dim,
            use_layer_norm=priv_mixer_use_layer_norm,
            device=device,
        ).to(self.device)

        # Privileged dynamics decoder network
        self.priv_dynamics_decoder = PrivDynamicsDecoder(
            input_dim=privileged_dynamics_latent_dim,
            decode_dim=self.num_priv_obs,
            layers=privileged_dynamics_decoder_layers,
        ).to(self.device)

        ## Layers used to match history encoder latent dimension to corrispodning privileged latent dimensions:
        ##     concretely -> reparamaterized encoder VAE-sample (mu + logvar*eps) -> matching layer -> input to privileged decoders

        #     depth-image sequence latent -> privileged terrain decoder input
        self.depth_to_terrain_latent = ReconDimensionProjectionHead(
            input_dim=actor_critic.depth_latent_dim,
            recon_dim=privileged_terrain_latent_dim,
        ).to(self.device)

        #     proprioceptive latent -> privileged dynamics decoder input 
        self.proprio_to_dynamics_latent = ReconDimensionProjectionHead(
            input_dim=actor_critic.proprio_latent_dim,
            recon_dim=privileged_dynamics_latent_dim,
        ).to(self.device)

        #     modality mixer latent -> privileged terrain decoder input 
        self.mixer_to_terrain_latent = ReconDimensionProjectionHead(
            input_dim=actor_critic.mixer_latent_dim,
            recon_dim=privileged_terrain_latent_dim,
        ).to(self.device)

        #     modality mixer latent -> privileged dynamics decoder input
        self.mixer_to_dynamics_latent = ReconDimensionProjectionHead(
            input_dim=actor_critic.mixer_latent_dim,
            recon_dim=privileged_dynamics_latent_dim,
        ).to(self.device)

        # Sacrificial projection heads for contrastive alignment. These keep
        # the policy/reconstruction latents one step removed from the direct
        # contrastive loss while still allowing useful alignment gradients.
        self.terrain_contrastive_head_depth = ContrastiveProjectionHead(
            input_dim=actor_critic.depth_latent_dim,
            projection_dim=privileged_terrain_latent_dim,
            hidden_dim=max(64, privileged_terrain_latent_dim),
            activation="elu",
        ).to(self.device)

        self.dynamics_contrastive_head_proprio = ContrastiveProjectionHead(
            input_dim=actor_critic.proprio_latent_dim,
            projection_dim=privileged_dynamics_latent_dim,
            hidden_dim=max(64, privileged_dynamics_latent_dim),
            activation="elu",
        ).to(self.device)

        self.terrain_contrastive_head_mixer = ContrastiveProjectionHead(
            input_dim=actor_critic.mixer_latent_dim,
            projection_dim=privileged_terrain_latent_dim,
            hidden_dim=max(64, privileged_terrain_latent_dim),
            activation="elu",
        ).to(self.device)

        self.dynamics_contrastive_head_mixer = ContrastiveProjectionHead(
            input_dim=actor_critic.mixer_latent_dim,
            projection_dim=privileged_dynamics_latent_dim,
            hidden_dim=max(64, privileged_dynamics_latent_dim),
            activation="elu",
        ).to(self.device)


        # Optimizer for the depth-image decoder
        self.depth_decoder_optimizer = optim.Adam(self.depth_decoder.parameters(), lr=learning_rate)

        # Optimizer for the privileged encoder-decoder pairs
        self.privileged_optimizer = optim.Adam(
            list(self.priv_terrain_encoder.parameters())
            + list(self.priv_terrain_decoder.parameters())
            + list(self.priv_dynamics_encoder.parameters())
            + list(self.priv_dynamics_decoder.parameters()),
            lr=learning_rate,
        )

        # Optimizer for the stand-alone reconstruction projection layers / the contrastive loss layers
        self.aux_projection_optimizer = optim.Adam(
            list(self.depth_to_terrain_latent.parameters())
            + list(self.proprio_to_dynamics_latent.parameters())
            + list(self.mixer_to_terrain_latent.parameters())
            + list(self.mixer_to_dynamics_latent.parameters())
            + list(self.terrain_contrastive_head_depth.parameters())
            + list(self.dynamics_contrastive_head_proprio.parameters())
            + list(self.terrain_contrastive_head_mixer.parameters())
            + list(self.dynamics_contrastive_head_mixer.parameters()),
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
                                              contrastive_anchor_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def _latest_privileged_obs(self, privileged_obs_history):
        return privileged_obs_history[:, -self.num_priv_obs:]

    def build_critic_obs(self, privileged_obs_history, terrain_map, detach_latents=True):
        """Build critic input from privileged dynamics and learned terrain latents."""
        latest_privileged_obs = self._latest_privileged_obs(privileged_obs_history)
        terrain_latent = self.priv_terrain_encoder(terrain_map)
        dynamics_latent = self.priv_dynamics_encoder(privileged_obs_history)
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

    def _projected_contrastive_loss(self, projection_head, z, positive, negative, mask):
        """Apply contrastive loss through a sacrificial projection head."""
        projected_z = projection_head(z)
        positive = F.normalize(positive.detach(), p=2, dim=-1, eps=1e-6)
        negative = F.normalize(negative.detach(), p=2, dim=-1, eps=1e-6)
        return self._contrastive_loss(projected_z, positive, negative, mask)

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
                             "normal_cos":normal_cos.detach()}
        
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
    def _versatility_kl_metric(self, mean, logvar, mask):
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
        # # E[sigma_i^2]
        # mean_conditional_var = torch.sum(torch.exp(logvar) * _mask,dim=0,) / _denom  # (latent_dim,)

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

        # Calculate the VAE encoder's KL-divergence
        kl_loss = self._kl_loss(mean, logvar, mask)

        # Maximize: J = MI - lambda_e * KL
        # Minimize: loss = -J
        vers_loss = -(mutual_info - self.versatility_lambda_e * kl_loss)

        # populate the log-dict
        vers_log = {"marginal_entropy":marginal_entropy.detach(),
                    "conditional_entropy":conditional_entropy.detach(),
                    "mutual_info":mutual_info.detach(),
                    "kl":kl_loss.detach()
                    }

        return vers_loss, vers_log

    def act(self, obs, critic_obs, obs_history, privileged_obs_history, depth_image, depth_latent_history, depth_torso_state, terrain_map):
        # if self.actor_critic.is_recurrent:
        #     self.transition.hidden_states = self.actor_critic.get_hidden_states()
        if self.use_boot:
            all_actions, body_velo_est, _ = self.actor_critic.act_with_estimates(
                obs,
                obs_history,
                depth_image=depth_image,
                depth_latent_history=depth_latent_history,
                depth_torso_state=depth_torso_state,
            )
        else:
            all_actions, body_velo_est, _ = (
                self.actor_critic.act_bootmask_with_estimates(
                    obs,
                    obs_history,
                    depth_image=depth_image,
                    depth_latent_history=depth_latent_history,
                    depth_torso_state=depth_torso_state,
                )
            )
        all_actions = all_actions.detach()
        body_velo_est = body_velo_est.detach()

        # Save the latest frame latent so the runner can roll the visual sequence state without storing older raw depth images.
        latest_depth_latent = self.actor_critic.depth_frame_encoder.forward_inference(depth_image, depth_torso_state)

        # Compute the action distribution statistics and value for storage.
        self.transition.actions =  all_actions
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
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
        self.transition.contrastive_negative_anchor = torch.empty_like(
            latest_depth_latent
        ).uniform_(-1.0, 1.0)

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

    def _set_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def _zero_encoder_optimizer(self, name):
        if hasattr(self.enc_optimizer, "optimizers"):
            self.enc_optimizer.optimizers[name].zero_grad()
        else:
            self.enc_optimizer.zero_grad()

    def _step_encoder_optimizer(self, name):
        if hasattr(self.enc_optimizer, "optimizers"):
            self.enc_optimizer.optimizers[name].step()
        else:
            self.enc_optimizer.step()

    def _depth_frame_autoencoder_update(self, depth_images_batch, depth_torso_state_batch, mask):
        self._set_requires_grad(self.depth_decoder, True)
        self.actor_critic.depth_frame_encoder.train()
        self.depth_decoder.train()
        self._zero_encoder_optimizer("visual_frame")
        self.depth_decoder_optimizer.zero_grad()
        
        # Encode the current depth image
        depth_mean, depth_logvar, latest_depth_z, depth_aux = self.actor_critic.depth_frame_encoder(
            depth_images_batch,
            depth_torso_state_batch,
        )

        transform_matrices = depth_aux["transform_matrices"]
        depth_recon, _ = self.depth_decoder(latest_depth_z, transform_matrices)
        
        # Calculate the depth-image reconstruction loss
        depth_recon_loss = self._masked_mse_loss(depth_recon, depth_images_batch, mask)
        
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

        # Step, norm-clip, and step optimizers for the joint autoencoder.
        depth_frame_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.depth_frame_encoder.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.depth_decoder.parameters(), self.max_grad_norm)
        self._step_encoder_optimizer("visual_frame")
        self.depth_decoder_optimizer.step()

        return (
            depth_frame_loss,
            depth_recon_loss,
            depth_kl,
            transform_identity_loss,
            latest_depth_z.detach(),
        )

    def _depth_sequence_encoder_update(self, depth_latent_history_batch, latest_depth_z_for_seq, 
                                       terrain_maps_batch, terrain_positive, contrastive_negative_anchor_batch, 
                                       mask):
        self.actor_critic.depth_sequence_encoder.train()
        self.priv_terrain_decoder.eval()
        self._zero_encoder_optimizer("visual_sequence")
        self.aux_projection_optimizer.zero_grad()

        # history-batch from storage, latest_depth_z from depth-decoder update step
        depth_sequence = torch.cat(
            [depth_latent_history_batch, latest_depth_z_for_seq.unsqueeze(1)],
            dim=1,
        )
        
        # encode the depth-latent sequence
        seq_mean, seq_logvar, depth_seq_z = self.actor_critic.depth_sequence_encoder(depth_sequence)
        
        # use the reconstruction projection head
        depth_terrain_z = self.depth_to_terrain_latent(depth_seq_z)
        
        # reconstruct terrain (height-map + surface normals) from depth-sequence latent
        terrain_recon_from_depth = self.priv_terrain_decoder(depth_terrain_z)
        
        # Calculate the terrain reconstruction loss
        terrain_recon_loss, _ = self._terrain_recon_loss(
            terrain_recon_from_depth,
            terrain_maps_batch,
            mask,
        )

        # Calculate the contrastive loss through the contrastive loss projection head
        seq_contrastive = self._projected_contrastive_loss(
            self.terrain_contrastive_head_depth,
            depth_terrain_z,
            terrain_positive,
            contrastive_negative_anchor_batch,
            mask,
        )

        # Calculate the kl-loss of the depth-sequence encoder
        seq_kl = self._kl_loss(seq_mean, seq_logvar, mask)
        
        # Calculate the total depth-sequence loss
        depth_sequence_loss = (
            self.depth_sequence_terrain_weight * terrain_recon_loss
            + self.contrastive_weight * seq_contrastive
            + self.depth_sequence_kl_weight * seq_kl
        )

        # backwards, norm-clip, step-sequence encoder optimizer, step projection-head optimizer
        depth_sequence_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.depth_sequence_encoder.parameters(), self.max_grad_norm)
        self._step_encoder_optimizer("visual_sequence")
        self.aux_projection_optimizer.step()

        return depth_sequence_loss, terrain_recon_loss, seq_contrastive, seq_kl, depth_seq_z.detach()

    def _proprioceptive_encoder_update(self, obs_hist_batch, obs_target, dynamics_positive, 
                                       contrastive_negative_anchor_batch, mask):
        self.actor_critic.proprio_context_encoder.train()
        self.priv_dynamics_decoder.eval()
        self._zero_encoder_optimizer("proprioceptive")
        self.aux_projection_optimizer.zero_grad()
        
        # Encode the current properioceptive observation history
        prop_mean, prop_logvar, proprio_z = self.actor_critic.proprio_context_encoder(obs_hist_batch)
        
        # Use the proprioceptive latent -> privileged dynamics latent projection head
        proprio_dyn_z = self.proprio_to_dynamics_latent(proprio_z)
        
        # reconstruct the next time-steps privileged observation
        priv_dyn_recon = self.priv_dynamics_decoder(proprio_dyn_z)
        
        # Calculate the privileged reconstruction loss
        dynamics_recon_loss = self._masked_mse_loss(priv_dyn_recon, obs_target, mask)
        
        # Calculate the contrastive loss w.r.t. the privileged dynamics encoder
        #     using the contrastive projection head
        prop_contrastive = self._projected_contrastive_loss(
            self.dynamics_contrastive_head_proprio,
            proprio_dyn_z,
            dynamics_positive,
            contrastive_negative_anchor_batch,
            mask,
        )
        
        # Calculate the kl-loss of the proprioceptive encoder
        prop_kl = self._kl_loss(prop_mean, prop_logvar, mask)
        
        # Calculate the total proprioceptive encoder loss
        proprio_loss = (
            self.proprio_dynamics_weight * dynamics_recon_loss
            + self.contrastive_weight * prop_contrastive
            + self.proprio_kl_weight * prop_kl
        )
        
        # Backwards, norm-clip, step proprioceptive encoder optimizer, step projection heads optimizer
        proprio_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.proprio_context_encoder.parameters(), self.max_grad_norm)
        self._step_encoder_optimizer("proprioceptive")
        self.aux_projection_optimizer.step()

        return proprio_loss, dynamics_recon_loss, prop_contrastive, prop_kl, proprio_z.detach()

    def _mixer_encoder_update(self, depth_seq_z_detached, proprio_z_detached, obs_target, 
                              explicit_labels_batch, terrain_maps_batch, 
                              terrain_positive, dynamics_positive, contrastive_negative_anchor_batch, 
                              mask
    ):    
        self.actor_critic.context_encoder.train()
        self._zero_encoder_optimizer("modality_mixer")
        self.aux_projection_optimizer.zero_grad()
        
        # Use previously computed (now detached) depth sequence latent and obs-history latent
        #     to produce mizer latent and explicit estimation values
        mix_mean, mix_logvar, mix_z, body_velo_est, feet_state_est = self.actor_critic.context_encoder(
            depth_seq_z_detached,
            proprio_z_detached,
        )

        # Use the reconstruction projection heads
        mix_terrain_z = self.mixer_to_terrain_latent(mix_z)
        mix_dynamics_z = self.mixer_to_dynamics_latent(mix_z)

        # Reconstruct the terrain
        terrain_recon_from_mix = self.priv_terrain_decoder(mix_terrain_z)
        
        # Reconstruct the next time-steps privileged observation 
        dyn_recon_from_mix = self.priv_dynamics_decoder(mix_dynamics_z)
        
        # Create the full explicit estimation vector
        explicit_est = torch.cat([body_velo_est, feet_state_est], dim=-1)

        # Calculate recon losses
        mix_terrain_loss, mix_terrain_recon_log = self._terrain_recon_loss(terrain_recon_from_mix, terrain_maps_batch, mask)  # Calculate terrain reconstruction loss (with log returned)
        mix_dyn_loss = self._masked_mse_loss(dyn_recon_from_mix, obs_target, mask)                                            # Calculate the next time-step privileged observation reconstruction loss
        explicit_loss = self._masked_mse_loss(explicit_est, explicit_labels_batch, mask)                                      # Calculate the explicit estimation (torso velocity, feet-state) reconstruction loss

        # Pull out the dimensions of the body velocity prediction and feet-state prediction
        body_velo_dim = self.actor_critic.body_velo_dim
        feet_state_dim = self.actor_critic.feet_state_dim

        # Calculate the torso velocity estimation loss specifically for logging
        torso_velo_explicit_loss = self._masked_mse_loss(
            body_velo_est,
            explicit_labels_batch[:, :body_velo_dim],
            mask,
        )

        # Calculate the feet-state estimation loss specifically for logging
        if feet_state_dim > 0:
            feet_state_explicit_loss = self._masked_mse_loss(
                feet_state_est,
                explicit_labels_batch[
                    :, body_velo_dim:body_velo_dim + feet_state_dim
                ],
                mask,
            )
        else:
            feet_state_explicit_loss = explicit_loss.new_zeros(())

        # Calculate the terrain contrastive loss
        mix_terrain_contrast_loss = self._projected_contrastive_loss(self.terrain_contrastive_head,
                                                                     mix_terrain_z,
                                                                     terrain_positive,
                                                                     contrastive_negative_anchor_batch,
                                                                     mask)

        # Calculate the next time-step privlieged obs contrastive loss
        mix_dyn_contrast_loss = self.self._projected_contrastive_loss(self.dynamics_contrastive_head,
                                                                      mix_dynamics_z,
                                                                      dynamics_positive,
                                                                      contrastive_negative_anchor_batch,
                                                                      mask)
        
        # Calculate the total contrastive loss
        mix_contrastive = mix_terrain_contrast_loss + mix_dyn_contrast_loss 

        # No standalone KL-loss for the modality mixer, it is included in the versatility metric
        versatility_loss, versatility_log = self._versatility_kl_metric(mix_mean,
                                                                        mix_logvar,
                                                                        mask)
        
        # Calculate total loss
        modality_loss = (
            self.modality_terrain_weight * mix_terrain_loss
            + self.modality_dynamics_weight * mix_dyn_loss
            + self.modality_explicit_weight * explicit_loss
            + self.contrastive_weight * mix_contrastive
            + self.versatility_weight * versatility_loss)
        
        # Backwards, norm-clip, modality-mixer encoder optimizer step, projection/contrative head optimizer step
        modality_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.context_encoder.parameters(), self.max_grad_norm)
        self._step_encoder_optimizer("modality_mixer")
        self.aux_projection_optimizer.step()
        

        return modality_loss, mix_terrain_loss, mix_terrain_recon_log, mix_dyn_loss, explicit_loss, torso_velo_explicit_loss, \
                feet_state_explicit_loss, mix_terrain_contrast_loss, mix_dyn_contrast_loss, mix_contrastive, versatility_loss, versatility_log, \
                dyn_recon_from_mix, terrain_recon_from_mix, body_velo_est

    def _privileged_encoder_decoder_updates(self, terrain_maps_batch, privileged_obs_history_batch, obs_target, mask):
        self.privileged_optimizer.zero_grad()

        # Encode -> Decode the height + surface normal map
        terrain_priv_z = self.priv_terrain_encoder(terrain_maps_batch)
        terrain_priv_recon = self.priv_terrain_decoder(terrain_priv_z)
        
        # Encode -> decode the privileged observation history -> next time-step privileged obs
        dyn_priv_z = self.priv_dynamics_encoder(privileged_obs_history_batch)
        dyn_priv_recon = self.priv_dynamics_decoder(dyn_priv_z)
        
        # terrain recon log
        privileged_terrain_loss, _ = self._terrain_recon_loss(
            terrain_priv_recon,
            terrain_maps_batch,
            mask,
        )
        privileged_dynamics_loss = self._masked_mse_loss(
            dyn_priv_recon,
            obs_target,
            mask,
        )
        privileged_loss = (
            privileged_terrain_loss
            + privileged_dynamics_loss
        )
        privileged_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.priv_terrain_encoder.parameters())
            + list(self.priv_terrain_decoder.parameters())
            + list(self.priv_dynamics_encoder.parameters())
            + list(self.priv_dynamics_decoder.parameters()),
            self.max_grad_norm,
        )
        self.privileged_optimizer.step()

        return privileged_loss, privileged_dynamics_loss, privileged_terrain_loss

    def _update_auxiliary_encoders(self, obs_hist_batch, privileged_obs_history_batch, depth_images_batch, 
                                   depth_latent_history_batch, depth_torso_state_batch, terrain_maps_batch, 
                                   contrastive_negative_anchor_batch, explicit_labels_batch, obs_target, 
                                   terminated_batch,
    ):
        mask = terminated_batch.float()
        losses = {}

        # 1. Depth-image autoencoder update:
        #    optimize the visual frame encoder and decoder together so the
        #    encoder's IMU transform conditions both halves of reconstruction.
        depth_frame_loss, depth_recon_loss, depth_kl, transform_identity_loss, latest_depth_z_detached = (
            self._depth_frame_autoencoder_update(
                depth_images_batch,
                depth_torso_state_batch,
                mask,
            )
        )

        # Shared privileged anchors for contrastive alignment. These are
        #     detached because the student-side encoders are updated in steps 3-5.
        terrain_positive = dynamics_positive = None
        with torch.no_grad():
            terrain_positive = self.priv_terrain_encoder(terrain_maps_batch)
            dynamics_positive = self.priv_dynamics_encoder(privileged_obs_history_batch)

        # 2. Depth-sequence encoder update:
        #    reconstruct privileged terrain and align the visual sequence latent
        #    to the privileged terrain encoder while pushing from a random anchor.
        depth_sequence_loss, terrain_recon_loss, seq_contrastive, seq_kl, depth_z_detach = self._depth_sequence_encoder_update(depth_latent_history_batch, latest_depth_z_detached, 
                                                                                                                               terrain_maps_batch, terrain_positive, 
                                                                                                                               contrastive_negative_anchor_batch, mask)

        # 3. Proprioceptive encoder update:
        #    reconstruct next privileged dynamics and align proprio latent to
        #    the privileged dynamics encoder using the same negative anchor.
        proprio_loss, dynamics_recon_loss, prop_contrastive, prop_kl, obs_z_detach = self._proprioceptive_encoder_update(obs_hist_batch, obs_target, dynamics_positive,
                                                                                                                         contrastive_negative_anchor_batch, mask)

        # 4. Modality mixer update:
        #    fuse frozen depth/proprio latents, then supervise the fused latent
        #    with terrain, dynamics, explicit-state, contrastive, and
        #    versatility/KL losses.
        modality_loss, mix_terrain_loss, mix_terrain_recon_log, mix_dyn_loss, explicit_loss, \
            torso_velo_explicit_loss, feet_state_explicit_loss, mix_terrain_contrast_loss, \
                mix_dyn_contrast_loss, mix_contrastive, versatility_loss, versatility_log, \
                    dyn_recon_from_mix, terrain_recon_from_mix, body_velo_est = self._mixer_encoder_update(depth_z_detach, obs_z_detach, obs_target, 
                                                                                                           explicit_labels_batch, terrain_maps_batch, 
                                                                                                           terrain_positive, dynamics_positive, 
                                                                                                           contrastive_negative_anchor_batch, mask)
        
        # 5. Privileged terrain/dynamics encoder-decoder update:
        #    refresh the teacher-side representations used as positive anchors.
        privileged_loss, privileged_dynamics_loss, privileged_terrain_loss = self._privileged_encoder_decoder_updates(terrain_maps_batch, privileged_obs_history_batch, 
                                                                                                                      obs_target, mask)

        # Total Encoder(s) update loss
        losses["total"] = depth_frame_loss.detach() + depth_sequence_loss.detach() + proprio_loss.detach() + modality_loss.detach() + privileged_loss.detach()

        # Explicity Estimation Loss
        losses["explicit"] = explicit_loss.detach()
        
        # Total reconstructed loss
        losses["recon"] = depth_recon_loss.detach() + terrain_recon_loss.detach() + dynamics_recon_loss.detach() + mix_terrain_loss.detach() + mix_dyn_loss.detach()

        # Total KL loss across all encoders
        losses["total_kl"] = (depth_kl.detach() + seq_kl.detach() + prop_kl.detach() + versatility_log["kl"])
        losses["decoder"] = depth_recon_loss.detach() + privileged_loss.detach()
        losses["depth_transform_identity"] = transform_identity_loss.detach()
        
        # Reconstructed items, used by boot-strapping prob logic
        losses["priv_dynamics_recon"] = dyn_recon_from_mix.detach()
        losses["priv_terrain_recon"] = terrain_recon_from_mix.detach()
        losses["body_velo_est"] = body_velo_est.detach()

        # Loss details for logging
        losses["detail"] = {
            "depth_frame_total": depth_frame_loss.detach(),
            "depth_frame_recon": depth_recon_loss.detach(),
            "depth_frame_kl": depth_kl.detach(),
            "depth_frame_transform_identity": transform_identity_loss.detach(),
            "depth_autoencoder_recon": depth_recon_loss.detach(),
            
            "depth_sequence_total": depth_sequence_loss.detach(),
            "depth_sequence_terrain_recon": terrain_recon_loss.detach(),
            "depth_sequence_contrastive": seq_contrastive.detach(),
            "depth_sequence_kl": seq_kl.detach(),
            
            "proprio_total": proprio_loss.detach(),
            "proprio_dynamics_recon": dynamics_recon_loss.detach(),
            "proprio_contrastive": prop_contrastive.detach(),
            "proprio_kl": prop_kl.detach(),
            
            "modality_total": modality_loss.detach(),
            "modality_terrain_recon": mix_terrain_loss.detach(),
            "modality_terrain_height": mix_terrain_recon_log["height_loss"].detach(),
            "modality_terrain_normals": mix_terrain_recon_log["normal_cos"].detach(),
            "modality_dynamics_recon": mix_dyn_loss.detach(),
            "modality_versatility": versatility_loss.detach(),
            "modality_kl": versatility_log["kl"].detach(),
            "modality_marginal_entropy": versatility_log["marginal_entropy"].detach(),
            "modality_conditional_entropy": versatility_log["conditional_entropy"].detach(),
            "modality_mutual_info": versatility_log["mutual_info"].detach(),
            "modality_explicit": explicit_loss.detach(),
            "modality_explicit_torso_velo": torso_velo_explicit_loss.detach(),
            "modality_explicit_feet_state": feet_state_explicit_loss.detach(),
            "modality_contrastive": mix_contrastive.detach(),
            "modality_terrain_contrastive": mix_terrain_contrast_loss.detach(),
            "modality_dynamics_contrastive": mix_dyn_contrast_loss.detach(),
            
            "privileged_total": privileged_loss.detach(),
            "privileged_terrain_recon": privileged_terrain_loss.detach(),
            "privileged_dynamics_recon": privileged_dynamics_loss.detach(),
        }

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

        timers = {
            "rl_loss": 0.0,
            "act_step": 0.0,
            "aux_update": 0.0,
            "boot_stats": 0.0,
            "boot_prob": 0.0,
            "spec_norm" : 0.0}

        boot_count = 0
        boot_sum_x = None
        boot_sum_x2 = None
        boot_sum_recon_sqerr = 0.0
        vel_boot_count = 0
        vel_boot_sum_x = None
        vel_boot_sum_x2 = None
        vel_boot_sum_recon_sqerr = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for terminated_batch, obs_batch, obs_hist_batch, privileged_obs_history_batch, depth_images_batch, depth_latent_history_batch, \
            depth_torso_state_batch, terrain_maps_batch, contrastive_negative_anchor_batch, explicit_labels_batch, obs_target, actions_batch, \
                target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch in generator:
            
            self.actor_critic.train()
            self.act_optimizer.zero_grad()
            self.enc_optimizer.zero_grad()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            # Perform RL update
            ppo_loss, surrogate_loss, value_loss = self._compute_rl_loss(obs_batch, obs_hist_batch, actions_batch, privileged_obs_history_batch, depth_images_batch,
                                                                         depth_latent_history_batch, depth_torso_state_batch, terrain_maps_batch, old_sigma_batch,
                                                                         old_mu_batch, old_actions_log_prob_batch, advantages_batch, target_values_batch, returns_batch)
            
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
                self.actor_critic.train()

                torch.cuda.synchronize()
                t0 = time.perf_counter()

                aux_losses = self._update_auxiliary_encoders(obs_hist_batch, privileged_obs_history_batch, depth_images_batch, depth_latent_history_batch,
                                                             depth_torso_state_batch, terrain_maps_batch, contrastive_negative_anchor_batch,
                                                             explicit_labels_batch, obs_target, terminated_batch)
                
                timers["aux_update"] += time.perf_counter() - t0

                torch.cuda.synchronize()
                t0 = time.perf_counter()

                # Log the modality-mixer terrain and dynamics reconstructions
                # for computing boot probability. Bootstrapping should only
                # depend on the mixer being better than a mean predictor across
                # both privileged dynamics and privileged terrain targets.
                with torch.no_grad():
                    sample_mask64 = terminated_batch.squeeze(-1).to(torch.float64)
                    sample_weight64 = sample_mask64.unsqueeze(-1)
                    valid_count = sample_mask64.sum().item()
                    dynamics_target = obs_target
                    dynamics_recon = aux_losses["priv_dynamics_recon"].detach()
                    terrain_target = terrain_maps_batch
                    terrain_recon = aux_losses["priv_terrain_recon"].detach()
                    
                    x = torch.cat([dynamics_target.flatten(start_dim=1),terrain_target.flatten(start_dim=1)], dim=-1)
                    r = torch.cat([dynamics_recon.flatten(start_dim=1), terrain_recon.flatten(start_dim=1)], dim=-1)

                    # flatten batch dimension only; keep feature dim
                    # x/r shape is [B, dynamics_dim + terrain_map_dim]
                    if boot_sum_x is None:
                        boot_sum_x = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float64)
                        boot_sum_x2 = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float64)

                    x64 = x.to(torch.float64)
                    r64 = r.to(torch.float64)

                    boot_sum_x += (x64 * sample_weight64).sum(dim=0)
                    boot_sum_x2 += (x64 * x64 * sample_weight64).sum(dim=0)

                    # scalar sum over all elements
                    boot_sum_recon_sqerr += ((r64 - x64).pow(2) * sample_weight64).sum().item()

                    boot_count += valid_count

                    # Use a separate boot probability for the depth-torso
                    # velocity slot, based only on torso linear velocity
                    # prediction accuracy.
                    body_velo_dim = self.actor_critic.body_velo_dim
                    vel_target = explicit_labels_batch[:, :body_velo_dim]
                    vel_recon = aux_losses["body_velo_est"].detach()
                    if vel_boot_sum_x is None:
                        vel_boot_sum_x = torch.zeros(vel_target.shape[-1],
                                                     device=vel_target.device,
                                                     dtype=torch.float64)
                        
                        vel_boot_sum_x2 = torch.zeros(vel_target.shape[-1],
                                                      device=vel_target.device,
                                                      dtype=torch.float64)

                    vel_x64 = vel_target.to(torch.float64)
                    vel_r64 = vel_recon.to(torch.float64)
                    vel_boot_sum_x += (vel_x64 * sample_weight64).sum(dim=0)
                    vel_boot_sum_x2 += (vel_x64 * vel_x64 * sample_weight64).sum(dim=0)
                    vel_boot_sum_recon_sqerr += ((vel_r64 - vel_x64).pow(2) * sample_weight64).sum().item()
                    vel_boot_count += valid_count

                timers["boot_stats"] += time.perf_counter() - t0

                # Log losses
                mean_aux_loss += aux_losses["total"].item()
                mean_explicit_loss += aux_losses["explicit"].item()
                mean_reconstruction_loss += aux_losses["recon"].item()
                mean_kl_loss += aux_losses["total_kl"].item()
                mean_aux_decoder_loss += aux_losses["decoder"].item()
                
                for name, value in aux_losses["detail"].items():
                    mean_aux_loss_details[name] = (mean_aux_loss_details.get(name, 0.0) + value.item())

            # Keeps the interaction of incoming data with layer wieghts below the threashold that 
            #     saturates the tanh activation function.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.spectral_normalization(self.actor_critic, sigma_max=6.0)

            timers["spec_norm"] += time.perf_counter() - t0

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        mean_aux_loss /= (num_updates * self.num_enc_epochs)
        mean_aux_decoder_loss /= (num_updates * self.num_enc_epochs)
        mean_kl_loss /= (num_updates * self.num_enc_epochs)
        mean_explicit_loss /= (num_updates * self.num_enc_epochs)
        mean_reconstruction_loss /= (num_updates * self.num_enc_epochs)
        for name in mean_aux_loss_details:
            mean_aux_loss_details[name] /= (num_updates * self.num_enc_epochs)


        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Estimate whether the modality mixer reconstructs privileged dynamics
        # plus terrain better than a mean predictor, then use that ratio to
        # decide bootstrapping.
        if boot_count > 0:
            feat_dim = boot_sum_x.shape[0]

            mean_pred = boot_sum_x / boot_count                 # [D]
            ex2 = boot_sum_x2 / boot_count                      # [D]
            var = torch.clamp(ex2 - mean_pred**2, min=0.0)      # [D]

            mean_pred_error = var.mean().item()

            actual_pred_error = boot_sum_recon_sqerr / (boot_count * feat_dim)

            ratio = mean_pred_error / (actual_pred_error * self.boot_mult + 1e-8)
            pboot = np.tanh(ratio)
        else:
            pboot = 0.0

        if vel_boot_count > 0:
            vel_feat_dim = vel_boot_sum_x.shape[0]
            vel_mean_pred = vel_boot_sum_x / vel_boot_count
            vel_ex2 = vel_boot_sum_x2 / vel_boot_count
            vel_var = torch.clamp(vel_ex2 - vel_mean_pred**2, min=0.0)
            vel_mean_pred_error = vel_var.mean().item()
            vel_actual_pred_error = vel_boot_sum_recon_sqerr / (
                vel_boot_count * vel_feat_dim
            )
            vel_ratio = vel_mean_pred_error / (
                vel_actual_pred_error * self.boot_mult + 1e-8
            )
            vel_pboot = np.tanh(vel_ratio)
        else:
            vel_pboot = 0.0

        timers["boot_prob"] += time.perf_counter() - t0

        # Use the (scaled) ratio of mean-prediction performance to actual prediction performance
        #     to determine if encoder bootstrapping is performed.
        self.use_boot = random.random() < pboot
        self.use_depth_vel_boot = random.random() < vel_pboot
        print("Use bootstrapped Encoder Dynamics: ", self.use_boot)
        print("Use bootstrapped Depth Torso Velocity: ", self.use_depth_vel_boot)

        self.storage.clear()

        # Get the average time for the various tracked timers
        for key in timers.keys():
            if "boot_prob" not in key:
                timers[key] /= num_updates


        print("update timers:", {k: round(v, 4) for k, v in timers.items()})

        return mean_value_loss, mean_surrogate_loss, mean_aux_loss, mean_aux_decoder_loss, \
            mean_explicit_loss, mean_reconstruction_loss, mean_kl_loss, mean_aux_loss_details

    def _compute_rl_loss(self, obs_batch, obs_hist_batch,
                         actions_batch, privileged_obs_history_batch,
                         depth_images_batch, depth_latent_history_batch,
                         depth_torso_state_batch, terrain_maps_batch,
                         old_sigma_batch, old_mu_batch,
                         old_actions_log_prob_batch,
                         advantages_batch, target_values_batch, returns_batch):
        if self.use_boot:
            self.actor_critic.act(
                obs_batch,
                obs_hist_batch,
                depth_image=depth_images_batch,
                depth_latent_history=depth_latent_history_batch,
                depth_torso_state=depth_torso_state_batch,
            )
        else:
            self.actor_critic.act_bootmask(
                obs_batch,
                obs_hist_batch,
                depth_image=depth_images_batch,
                depth_latent_history=depth_latent_history_batch,
                depth_torso_state=depth_torso_state_batch,
            )

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
