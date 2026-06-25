from __future__ import annotations

import copy
from typing import Tuple
from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kite_modality_mixer_encoder import MultimodalGatedFusionVAE
from .kite_proprio_encoder import ProprioContextMLPMixerKITE
from .kite_visual_encoder import (
    ConvDepthSequenceEncoder,
    MotionRobustDepthEncoder,
    MotionRobustDepthDecoder,
    MotionRobustDepthAutoencoderUNet,
)
from .module_utils import ContrastiveProjectionHead, get_activation


class KITEDepthAsyncPipeline(nn.Module):
    """10 Hz depth-only deployment graph.

    This module owns independent copies of the depth frame encoder and depth
    sequence encoder so it can be exported as a completely separate TorchScript
    model from the actor pipeline.
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.depth_frame_encoder = copy.deepcopy(actor_critic.depth_frame_encoder)
        self.depth_sequence_encoder = copy.deepcopy(actor_critic.depth_sequence_encoder)
        self.depth_sequence_length = actor_critic.depth_sequence_length
        self.depth_latent_dim = actor_critic.depth_latent_dim

    def forward(
        self,
        depth_image: torch.Tensor,
        depth_torso_state: torch.Tensor,
        depth_latent_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latest_depth_latent, latest_depth_logvar = self.depth_frame_encoder.encode_inf(
            depth_image,
            depth_torso_state,
        )
        depth_latent_sequence = torch.cat(
            [depth_latent_history, latest_depth_latent.unsqueeze(1)],
            dim=1,
        )
        depth_sequence_latent = self.depth_sequence_encoder.forward_inference(
            depth_latent_sequence,
            latest_depth_logvar,
        )
        updated_depth_latent_history = depth_latent_sequence[:, 1:, :]

        return (
            depth_sequence_latent,
            updated_depth_latent_history,
            latest_depth_latent,
        )


class KITEActorAsyncPipeline(nn.Module):
    """50 Hz proprioception + modality mixer + actor deployment graph."""

    def __init__(self, actor_critic):
        super().__init__()
        self.proprio_context_encoder = copy.deepcopy(
            actor_critic.proprio_context_encoder
        )
        self.context_encoder = copy.deepcopy(actor_critic.context_encoder)
        self.actor = copy.deepcopy(actor_critic.actor)

    def forward(
        self,
        obs: torch.Tensor,
        obs_history: torch.Tensor,
        depth_sequence_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        proprio_latent = self.proprio_context_encoder.forward_inference(obs_history)
        context_latent, body_velo_est, feet_state_est = (
            self.context_encoder.forward_inference(depth_sequence_latent, proprio_latent)
        )
        actor_input = torch.cat([obs, context_latent, body_velo_est, feet_state_est], dim=-1)
        actions = self.actor.actor_forward(actor_input)

        return actions


def build_kite_async_deployment_pipelines(
    actor_critic,
    device="cpu",
) -> Tuple[KITEDepthAsyncPipeline, KITEActorAsyncPipeline]:
    """Build separate 10 Hz and 50 Hz KITE deployment modules."""
    depth_pipeline = actor_critic.build_depth_async_pipeline().eval().to(device)
    actor_pipeline = actor_critic.build_actor_async_pipeline().eval().to(device)
    return depth_pipeline, actor_pipeline


def script_kite_async_deployment_pipelines(
    actor_critic,
    device="cpu",
) -> Tuple[torch.jit.ScriptModule, torch.jit.ScriptModule]:
    """Compile the separated KITE deployment modules with TorchScript."""
    depth_pipeline, actor_pipeline = build_kite_async_deployment_pipelines(
        actor_critic,
        device=device,
    )
    return torch.jit.script(depth_pipeline), torch.jit.script(actor_pipeline)


def export_kite_async_deployment_pipelines(
    actor_critic,
    path: str,
    device="cpu",
) -> Tuple[str, str]:
    """Save two independent TorchScript models for async KITE deployment."""
    import os

    os.makedirs(path, exist_ok=True)
    depth_script, actor_script = script_kite_async_deployment_pipelines(
        actor_critic,
        device=device,
    )

    depth_path = os.path.join(path, "kite_depth_10hz_pipeline.pt")
    actor_path = os.path.join(path, "kite_actor_50hz_pipeline.pt")
    depth_script.save(depth_path)
    actor_script.save(actor_path)

    return depth_path, actor_path


class ActorCritic_KITE(nn.Module):
    def __init__(self,
                 num_actor_obs=45,
                 num_act_hist=10,
                 num_critic_obs=132,
                 
                 depth_image_resolution=(48, 64),
                 depth_image_latent_dim=32,
                 depth_image_norm="layer",
                 depth_decoder_norm="none",
                 depth_image_std_min=0.01,
                 depth_image_std_max=2.0,
                 
                 depth_sequence_length=5,
                 depth_sequence_outdim=16,
                 depth_sequence_norm="layer",
                 depth_sequence_std_min=0.01,
                 depth_sequence_std_max=1.5,
                 depth_sequence_conf_min=0.1,
                 depth_sequence_conf_mask_scale=0.2,
                 
                 proprio_in_dim=450,
                 proprio_latent_dim=16,
                 proprio_use_norm=True,
                 proprio_mixer_blocks=2,
                 proprio_hidden_dim=128,
                 proprio_token_dim=128,
                 proprio_channel_dim=256,
                 proprio_std_min=0.01,
                 proprio_std_max=1.5,
                 
                 mixer_velo_dim=3,                   # torso velocity state [v_x, v_y, v_z]
                 mixer_feet_state_dim=20,            # [feet-contact-state (4), feet-height (4), surface-normal under feet (12)]
                 mixer_latent_dim=32,
                 mixer_use_norm=True,
                 mixer_hidden_dims=(128, 64),
                 mixer_velo_hidden=32,
                 mixer_feet_hidden=32,
                 mixer_std_min=0.01,
                 mixer_std_max=1.5,
                 privileged_terrain_latent_dim=32,
                 privileged_dynamics_latent_dim=16,
                 
                 num_actions=12,
                 actor_layers=[512,256,128],
                 critic_layers=[128,256,128,64],
                 activation="elu", 
                 init_noise_std=1.0,
                 ):
        super().__init__()

        
        # some generic paramaters
        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.init_noise_std = init_noise_std
        
        self.body_velo_dim = mixer_velo_dim
        self.feet_state_dim = mixer_feet_state_dim
        
        # Depth-image encoder paramaters
        self.depth_image_resolution = depth_image_resolution
        self.depth_latent_dim = depth_image_latent_dim
        
        # Number of depth-frame latents consumed by the sequence encoder. The
        #      runner stores depth_sequence_length - 1 previous latents plus the
        #      current image encoded on demand.
        self.depth_sequence_length = depth_sequence_length

        # Proprioceptive Encoder paramaters
        self.proprio_latent_dim = proprio_latent_dim
        
        
        self.mixer_latent_dim = mixer_latent_dim
        
        # Create proprioceptive context encoder
        self.proprio_context_encoder = ProprioContextMLPMixerKITE(
            context_input_dim=proprio_in_dim,
            num_tokens=num_actor_obs,
            input_dim_per_token=num_act_hist,
            hidden_dim=proprio_hidden_dim,
            num_mixer_blocks=proprio_mixer_blocks,
            token_mlp_dim=proprio_token_dim,
            channel_mlp_dim=proprio_channel_dim,
            context_latent_size=proprio_latent_dim,
            activation=activation,
            use_layer_norm=proprio_use_norm,
            std_min=proprio_std_min,
            std_max=proprio_std_max,
        )

        # Single-depth image encoder
        self.depth_frame_encoder = MotionRobustDepthEncoder(
            depth_image_resolution=self.depth_image_resolution,
            target_latent_dim=self.depth_latent_dim,
            cnn_activation=activation,
            norm_type=depth_image_norm,
            vae_std_min=depth_image_std_min,
            vae_std_max=depth_image_std_max,
        )

        # Single-depth image decoder. The frame encoder and decoder are trained
        # together as a standalone depth autoencoder update in PPO.
        self.depth_frame_decoder = MotionRobustDepthDecoder(
            depth_image_resolution=self.depth_image_resolution,
            target_latent_dim=self.depth_latent_dim,
            cnn_activation=activation,
            norm_type=depth_decoder_norm,
            use_unet_skips=True,
        )

        # Training-only reconstruction wrapper. It reuses the standalone
        # encoder/decoder modules above, so deployment can still export only
        # the encoder while PPO can train with U-Net-style reconstruction skips.
        self.depth_frame_autoencoder = MotionRobustDepthAutoencoderUNet(
            self.depth_frame_encoder,
            self.depth_frame_decoder,
        )
        
        # Depth-image latent sequence encoder
        self.depth_sequence_encoder = ConvDepthSequenceEncoder(
            feature_dim=self.depth_latent_dim,
            sequence_length=self.depth_sequence_length,
            output_dim=depth_sequence_outdim,
            activation=activation,
            norm_type=depth_sequence_norm,
            std_min=depth_sequence_std_min,
            std_max=depth_sequence_std_max,
            conf_min=depth_sequence_conf_min,
            conf_mask_scale=depth_sequence_conf_mask_scale,
        )
        
        # Modality mixer encoder
        self.context_encoder = MultimodalGatedFusionVAE(
            depth_latent_dim=depth_sequence_outdim,
            proprio_latent_dim=proprio_latent_dim,
            hidden_dims=list(mixer_hidden_dims),
            output_dim=mixer_latent_dim,
            velo_dim=mixer_velo_dim,
            feet_state_dim=mixer_feet_state_dim,
            velo_hidden=mixer_velo_hidden,
            feet_hidden=mixer_feet_hidden,
            activation=activation,
            use_layer_norm=mixer_use_norm,
            std_min=mixer_std_min,
            std_max=mixer_std_max,
        )

        # Contrastive alignment heads are used only during auxiliary training.
        # Reconstruction decoders consume the VAE samples directly; these heads
        # let the contrastive objective act through a small sacrificial layer.
        self.depth_sequence_contrastive_head = ContrastiveProjectionHead(
            input_dim=depth_sequence_outdim,
            projection_dim=privileged_terrain_latent_dim,
            activation=activation,
        )
        self.proprio_contrastive_head = ContrastiveProjectionHead(
            input_dim=proprio_latent_dim,
            projection_dim=privileged_dynamics_latent_dim,
            activation=activation,
        )
        
        # Get the activation function used by the actor and critic networks
        activation = get_activation(activation)

        ###
        #  Construct the layers for the actor network
        ###
        # Shared layer between output branches
        actor_input_dim = num_actor_obs + mixer_latent_dim + mixer_velo_dim +  mixer_feet_state_dim # current obs o_t, force-aware latent dynamics z_t, explicit esitmation v_t 

        shared_trunk_layers = []
        shared_trunk_layers.append(nn.Linear(actor_input_dim, actor_layers[0]))
        shared_trunk_layers.append(activation)
        
        for l in range(len(actor_layers)-1):
            shared_trunk_layers.append(nn.Linear(actor_layers[l], actor_layers[l+1]))
            shared_trunk_layers.append(activation)
        
        shared_trunk_layers.append(nn.Linear(actor_layers[-1], num_actions))
        
        self.actor = nn.Sequential(*shared_trunk_layers)
        
        ###
        #  Construct layers for the critic network
        ###
        _critic_layers = []
        _critic_layers.append(nn.Linear(num_critic_obs, critic_layers[0]))
        _critic_layers.append(activation)
        for l in range(len(critic_layers)):
            if l == len(critic_layers) - 1:
                _critic_layers.append(nn.Linear(critic_layers[l], 1))
            else:
                _critic_layers.append(nn.Linear(critic_layers[l], critic_layers[l + 1]))
                _critic_layers.append(activation)
        self.critic = nn.Sequential(*_critic_layers)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        
        self.distribution = None
        
        self.current_obs = None
        
        self._std_clip_lwr = 0.1
        
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _init_std(self, std_val=1.00):                
        self.std.data.fill_(std_val)

    def get_optim_groups(self, weight_decay: float = 1e-4, strong_decay: float = 1e-1):
        """Separate actor/critic, depth-frame, and merged encoder params.
        
        Args:
            weight_decay (float): Weight decay value for regularization. Default: 1e-4.
            
        Returns:
            Actor/critic parameter groups, depth-frame autoencoder groups, and
            merged sequence/proprio/mixer encoder groups.
        """
        critic_set = set()
        actor_set = set()
        no_decay  = set()
        depth_frame_set = set()
        encoder_sets = {
            "proprioceptive": set(),
            "visual_sequence": set(),
            "modality_mixer": set(),
        }
        whitelist = (nn.Linear, nn.MultiheadAttention)
        blacklist = (nn.LayerNorm, nn.Embedding, nn.Parameter)
        encoder_prefix_to_group = {
            "proprio_context_encoder.": "proprioceptive",
            "proprio_contrastive_head.": "proprioceptive",
            "depth_sequence_encoder.": "visual_sequence",
            "depth_sequence_contrastive_head.": "visual_sequence",
            "context_encoder.": "modality_mixer",
        }
        depth_frame_prefixes = (
            "depth_frame_encoder.",
            "depth_frame_decoder.",
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                if fpn.startswith(depth_frame_prefixes):
                    depth_frame_set.add(fpn)
                    continue

                encoder_group = None
                for prefix, group_name in encoder_prefix_to_group.items():
                    if fpn.startswith(prefix):
                        encoder_group = group_name
                        break

                if encoder_group is not None:
                    encoder_sets[encoder_group].add(fpn)
                elif isinstance(m, blacklist):
                    no_decay.add(fpn)
                elif isinstance(m, whitelist):
                    if "critic" in fpn:
                        critic_set.add(fpn)
                    elif "act" in fpn:
                        actor_set.add(fpn)
                    else:
                        raise ValueError(f"Parameters not categorized: {fpn}")

        # for i in range(self.options["action_net"]["num_layers"]-1):
        #     no_decay.update([f"noise_decoder.cross_field_scales_pos.{i}", f"noise_decoder.cross_field_scales_tau.{i}"])
        no_decay.update([f"std"])
        # shared_decay.update([f"std"])

        # Validate parameter separation
        param_dict   = {pn: p for pn, p in self.named_parameters()}
        encoder_set = set().union(*encoder_sets.values())
        inter_params = actor_set & no_decay & encoder_set & critic_set & depth_frame_set
        if inter_params:
            raise ValueError(f"Parameters in all sets: {inter_params}")
        missing_params = param_dict.keys() - (
            actor_set | no_decay | encoder_set | critic_set | depth_frame_set
        )
        if missing_params:
            raise ValueError(f"Parameters not categorized: {missing_params}")
        # print(f"Parameters with extra strong weight decay{special_decay}")

        params_act = [{"params": [param_dict[pn] for pn in sorted(actor_set)],  "weight_decay":weight_decay, "name":"actor"},
                      {"params": [param_dict[pn] for pn in sorted(critic_set)], "weight_decay":weight_decay, "name":"critic"},
                      {"params": [param_dict[pn] for pn in sorted(no_decay)],   "weight_decay": 0.0}]
        params_depth_frame = [
            {
                "params": [param_dict[pn] for pn in sorted(depth_frame_set)],
                "weight_decay": weight_decay,
                "name": "depth_frame_autoencoder",
            }
        ]
        
        params_enc = {
            name: [
                {
                    "params": [param_dict[pn] for pn in sorted(param_names)],
                    "weight_decay": 1e-3,
                    "name": name,
                }
            ]
            for name, param_names in encoder_sets.items()
        }

        return params_act, params_depth_frame, params_enc

    def configure_optimizers(self,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 1e-6,
                             strong_decay: float = 1e-1,
                             betas: Tuple[float, float] = (0.9, 0.999)) -> torch.optim.Optimizer:
        """Configure the AdamW optimizer with parameter groups.

        Actor and critic share one AdamW optimizer. The merged KITE auxiliary
        update uses one Adam optimizer over all non-privileged encoder groups.
            
        Returns:
            Configured AdamW optimizer.
        """
        opt_groups_act, opt_groups_depth_frame, opt_groups_enc = self.get_optim_groups(weight_decay=weight_decay, strong_decay=strong_decay)
        act_opt = torch.optim.AdamW(opt_groups_act, lr=learning_rate)
        depth_frame_opt = torch.optim.Adam(
            opt_groups_depth_frame,
            lr=2.0e-4,
            betas=betas,
        )
        enc_groups = [
            group
            for groups in opt_groups_enc.values()
            for group in groups
        ]
        enc_opt = torch.optim.Adam(enc_groups, lr=2.0e-4, betas=betas)
        
        return act_opt, depth_frame_opt, enc_opt

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def build_depth_async_pipeline(self) -> KITEDepthAsyncPipeline:
        # Export/deployment helper for the 10 Hz visual pipeline.
        return KITEDepthAsyncPipeline(self)

    def build_actor_async_pipeline(self) -> KITEActorAsyncPipeline:
        # Export/deployment helper for the 50 Hz policy pipeline.
        return KITEActorAsyncPipeline(self)

    def _make_default_depth_image(self, obs):
        return torch.zeros(
            obs.shape[0],
            1,
            *self.depth_image_resolution,
            device=obs.device,
            dtype=obs.dtype,
        )

    def _format_depth_image(self, depth_image, obs):
        if depth_image is None:
            return self._make_default_depth_image(obs)
        if depth_image.dim() == 3:
            depth_image = depth_image.unsqueeze(1)
        if depth_image.dim() != 4:
            raise ValueError(
                "Expected depth_image shape BxHxW or Bx1xHxW, "
                f"got {tuple(depth_image.shape)}."
            )
        return depth_image

    def _build_depth_latent_sequence(self, latest_depth_z, depth_latent_history):
        if depth_latent_history is None:
            return latest_depth_z.unsqueeze(1).repeat(
                1, self.depth_sequence_length, 1
            )

        if depth_latent_history.dim() == 2:
            depth_latent_history = depth_latent_history.unsqueeze(1)
        if depth_latent_history.dim() != 3:
            raise ValueError(
                "Expected depth_latent_history shape BxTxd or Bxd, "
                f"got {tuple(depth_latent_history.shape)}."
            )

        sequence = torch.cat(
            [depth_latent_history, latest_depth_z.unsqueeze(1)],
            dim=1,
        )
        
        if sequence.shape[1] < self.depth_sequence_length:
            pad_count = self.depth_sequence_length - sequence.shape[1]
            padding = latest_depth_z.unsqueeze(1).repeat(1, pad_count, 1)
            sequence = torch.cat([padding, sequence], dim=1)
        
        return sequence[:, -self.depth_sequence_length:, :]
    
    # forward methods for the histroical context VAE
    def cenet_enc_forward(
        self,
        obs_history,
        obs=None,
        depth_image=None,
        depth_latent_history=None,
        depth_torso_state=None,
        return_latest_depth_z=False,
    ):
        if obs is None:
            obs = obs_history[:, -self.num_actor_obs:]

        # Encode the proprioceptive history
        proprio_mean, proprio_logvar, proprio_z = (
            self.proprio_context_encoder(obs_history)
        )

        ###
        #   Depth-image processing
        ###

        # encode the most recent depth image (always done during simulated training)
        depth_image = self._format_depth_image(depth_image, obs)
        _, latest_depth_logvar, latest_depth_z, _ = self.depth_frame_encoder(
            depth_image,
            depth_torso_state,
        )

        # add the most recent depth image encoding into the depth-sequence
        depth_sequence = self._build_depth_latent_sequence(
            latest_depth_z,
            depth_latent_history,
        )
        # Encode the latent depth-image sequence
        _, _, depth_seq_z = (
            self.depth_sequence_encoder(depth_sequence, latest_depth_logvar)
        )

        mean, logvar, z, body_velo, feet_state = self.context_encoder(
            depth_seq_z,
            proprio_z,
        )

        # Rollout collection also needs the latest frame latent to advance the
        # asynchronous visual history. Returning it here avoids a second depth
        # frame encoder pass in PPO_KITE.act().
        if return_latest_depth_z:
            return mean, logvar, z, body_velo, feet_state, latest_depth_z

        return mean, logvar, z, body_velo, feet_state
    
    def cenet_enc_inference(
        self,
        obs_history,
        obs=None,
        depth_image=None,
        depth_latent_history=None,
        depth_torso_state=None,
    ):
        if obs is None:
            obs = obs_history[:, -self.num_actor_obs:]

        proprio_z = self.proprio_context_encoder.forward_inference(obs_history)

        depth_image = self._format_depth_image(depth_image, obs)
        latest_depth_z, latest_depth_logvar = self.depth_frame_encoder.encode_inf(
            depth_image,
            depth_torso_state,
        )
        depth_sequence = self._build_depth_latent_sequence(
            latest_depth_z,
            depth_latent_history,
        )
        depth_seq_z = self.depth_sequence_encoder.forward_inference(
            depth_sequence,
            latest_depth_logvar,
        )

        z, body_velo, feet_state = self.context_encoder.forward_inference(
            depth_seq_z,
            proprio_z,
        )

        return z, body_velo, feet_state

    # Method for the forward method of the actor network, used mostly as an internal method
    def actor_forward(self, current_obs):
        # We are assuming "current_obs" includes all of the components used in the dreamwaq policy input
        action = self.actor(current_obs)

        if torch.isnan(action).any():
            with torch.no_grad():
                action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

        return action

    # Functions that are specific to PPO training
    @property
    @torch.jit.ignore
    def action_mean(self):
        return self.distribution.mean

    @property
    @torch.jit.ignore
    def action_std(self):
        return self.distribution.stddev

    @property
    @torch.jit.ignore
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @torch.jit.ignore
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @torch.no_grad
    @torch.jit.ignore
    def _clip_std(self,):
        self.std.data.clamp_(self._std_clip_lwr, 5.0)
        
    def _set_std_clip_lwr(self, clip_val=0.1):
        self._std_clip_lwr = clip_val

    @torch.jit.ignore
    def update_distribution(self, curr_obs):
        mean = self.actor_forward(curr_obs)
        self._clip_std()
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    # method used during simulated training
    @torch.jit.ignore
    def act_with_estimates(
        self,
        obs,
        obs_history,
        depth_image=None,
        depth_latent_history=None,
        depth_torso_state=None,
        **kwargs,
    ):
        _, _, z, body_velo_est, feet_state_est = self.cenet_enc_forward(
            obs_history,
            obs=obs,
            depth_image=depth_image,
            depth_latent_history=depth_latent_history,
            depth_torso_state=depth_torso_state,
        )

        context_state = torch.cat([body_velo_est, feet_state_est], dim=-1)
        
        # create the actors observation
        current_obs = torch.cat((obs,z,context_state), dim=-1)
        
        # Upated the PPO training distribution
        self.update_distribution(current_obs)

        sample = self.distribution.sample()

        return sample, body_velo_est, feet_state_est

    @torch.jit.ignore
    def act_with_estimates_and_depth_latent(
        self,
        obs,
        obs_history,
        depth_image=None,
        depth_latent_history=None,
        depth_torso_state=None,
        **kwargs,
    ):
        _, _, z, body_velo_est, feet_state_est, latest_depth_z = self.cenet_enc_forward(
            obs_history,
            obs=obs,
            depth_image=depth_image,
            depth_latent_history=depth_latent_history,
            depth_torso_state=depth_torso_state,
            return_latest_depth_z=True,
        )

        context_state = torch.cat([body_velo_est, feet_state_est], dim=-1)
        current_obs = torch.cat((obs, z, context_state), dim=-1)
        self.update_distribution(current_obs)
        sample = self.distribution.sample()

        return sample, body_velo_est, feet_state_est, latest_depth_z

    @torch.jit.ignore
    def act(
        self,
        obs,
        obs_history,
        depth_image=None,
        depth_latent_history=None,
        depth_torso_state=None,
        **kwargs,
    ):
        sample, _, _ = self.act_with_estimates(
            obs,
            obs_history,
            depth_image=depth_image,
            depth_latent_history=depth_latent_history,
            depth_torso_state=depth_torso_state,
            **kwargs,
        )
        # return a sample from the distribution to be executed in simulation
        return sample


    # method used during simulated training
    @torch.jit.ignore
    def act_bootmask_with_estimates(
        self,
        obs,
        obs_history,
        depth_image=None,
        depth_latent_history=None,
        depth_torso_state=None,
        **kwargs,
    ):
        _, _, z, body_velo_est, feet_state_est = self.cenet_enc_forward(
            obs_history,
            obs=obs,
            depth_image=depth_image,
            depth_latent_history=depth_latent_history,
            depth_torso_state=depth_torso_state,
        )

        context_state = torch.cat([body_velo_est, feet_state_est], dim=-1)
        
        # create the actors observation
        current_obs = torch.cat((obs,z,context_state), dim=-1)

        # Mask the latent/velo from the encoder with zeros
        boot_mask = torch.zeros((z.shape[0], (z.shape[1] + context_state.shape[1])), device=obs.device)

        # create the actors observation
        current_obs = torch.cat((obs,boot_mask), dim=-1)   
        
        # Upated the PPO training distribution
        self.update_distribution(current_obs)

        sample = self.distribution.sample()

        return sample, body_velo_est, feet_state_est

    @torch.jit.ignore
    def act_bootmask_with_estimates_and_depth_latent(
        self,
        obs,
        obs_history,
        depth_image=None,
        depth_latent_history=None,
        depth_torso_state=None,
        **kwargs,
    ):
        _, _, z, body_velo_est, feet_state_est, latest_depth_z = self.cenet_enc_forward(
            obs_history,
            obs=obs,
            depth_image=depth_image,
            depth_latent_history=depth_latent_history,
            depth_torso_state=depth_torso_state,
            return_latest_depth_z=True,
        )

        context_state = torch.cat([body_velo_est, feet_state_est], dim=-1)
        boot_mask = torch.zeros(
            (z.shape[0], z.shape[1] + context_state.shape[1]),
            device=obs.device,
        )
        current_obs = torch.cat((obs, boot_mask), dim=-1)
        self.update_distribution(current_obs)
        sample = self.distribution.sample()

        return sample, body_velo_est, feet_state_est, latest_depth_z

    @torch.jit.ignore
    def act_bootmask(
        self,
        obs,
        obs_history,
        depth_image=None,
        depth_latent_history=None,
        depth_torso_state=None,
        **kwargs,
    ):
        sample, _, _ = self.act_bootmask_with_estimates(
            obs,
            obs_history,
            depth_image=depth_image,
            depth_latent_history=depth_latent_history,
            depth_torso_state=depth_torso_state,
            **kwargs,
        )
        # return a sample from the distribution to be executed in simulation
        return sample

    # Method using during simulated inference
    @torch.jit.export
    def act_inference(
        self,
        obs,
        obs_history,
        depth_image=None,
        depth_latent_history=None,
        depth_torso_state=None,
    ):
        # Call the forward method of the context encoder
        z, body_velo_est, feet_state_est = self.cenet_enc_inference(
            obs_history,
            obs=obs,
            depth_image=depth_image,
            depth_latent_history=depth_latent_history,
            depth_torso_state=depth_torso_state,
        )

        context_state = torch.cat([body_velo_est, feet_state_est], dim=-1)
        
        # create the actors observation
        current_obs = torch.cat((obs,z,context_state), dim=-1)
                
        # call the actors forward method and return it's results
        actions = self.actor_forward(current_obs)

        # total_sample = torch.cat([actions_pos, actions_tau], dim=1)

        return actions

    # Forward method for calculating the value of the current state
    #     using the privilged critic observation
    def evaluate(self, critic_observations, **kwargs):
        val = self.critic(critic_observations)
        return val
