from __future__ import annotations

import copy
from typing import Tuple, List, Dict, Any
from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kite_modality_mixer_encoder import MultimodalMixerVAE
from .kite_proprio_encoder import ProprioContextMLPMixerKITE
from .kite_visual_encoder import (
    ConvDepthSequenceEncoder,
    MotionRobustDepthEncoder,
)
from .module_utils import get_activation


class OptimizerBundle:
    """Small adapter so multiple optimizers behave like one optimizer handle."""

    def __init__(self, optimizers: Dict[str, torch.optim.Optimizer]):
        self.optimizers = optimizers

    def zero_grad(self, *args, **kwargs):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        for optimizer in self.optimizers.values():
            optimizer.step(*args, **kwargs)

    def state_dict(self):
        return {
            name: optimizer.state_dict()
            for name, optimizer in self.optimizers.items()
        }

    def load_state_dict(self, state_dict):
        for name, optimizer in self.optimizers.items():
            if name not in state_dict:
                raise KeyError(
                    f"Missing optimizer state for encoder sub-optimizer {name!r}."
                )
            optimizer.load_state_dict(state_dict[name])


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
        _, _, latest_depth_latent, _ = self.depth_frame_encoder(
            depth_image,
            depth_torso_state,
        )
        depth_latent_sequence = torch.cat(
            [depth_latent_history, latest_depth_latent.unsqueeze(1)],
            dim=1,
        )
        _, _, depth_sequence_latent = self.depth_sequence_encoder(
            depth_latent_sequence
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
        _, _, proprio_latent = self.proprio_context_encoder(obs_history)
        _, _, context_latent, body_velo_est, feet_state_est = (
            self.context_encoder(depth_sequence_latent, proprio_latent)
        )
        context_state = torch.cat([body_velo_est, feet_state_est], dim=-1)
        actor_input = torch.cat([obs, context_latent, context_state], dim=-1)
        actions = self.actor(actor_input)

        return actions, context_latent, body_velo_est, feet_state_est


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
                 num_critic_obs=131,
                 num_actions=12,
                 actor_layers=[512,256,128],
                 critic_layers=[1024,256,128,64],
                 cenet_in_dim=450,
                 cenet_latent_dim=29,
                 cenet_velo_dim=3, 
                 cenet_enc_layers=[256,128,64],
                 activation="tanh", 
                 init_noise_std=1.0,):
        super().__init__()

        self.num_actor_obs = num_actor_obs
        self.cenet_latent_dim = cenet_latent_dim
        self.cenet_velo_dim = cenet_velo_dim
        self.depth_latent_dim = cenet_latent_dim
        self.depth_sequence_length = 5
        self.depth_image_resolution = (48, 64)
        self.body_velo_dim = min(3, cenet_velo_dim)
        self.feet_state_dim = max(cenet_velo_dim - self.body_velo_dim, 0)

        num_proprio_tokens = num_actor_obs
        input_dim_per_token = cenet_in_dim // max(num_proprio_tokens, 1)
        if num_proprio_tokens * input_dim_per_token != cenet_in_dim:
            raise ValueError(
                "KITE proprio mixer expects cenet_in_dim to be an integer "
                f"multiple of num_actor_obs. Got cenet_in_dim={cenet_in_dim}, "
                f"num_actor_obs={num_actor_obs}."
            )

        self.proprio_context_encoder = ProprioContextMLPMixerKITE(
            context_input_dim=cenet_in_dim,
            num_tokens=num_proprio_tokens,
            input_dim_per_token=input_dim_per_token,
            hidden_dim=cenet_enc_layers[0],
            num_mixer_blocks=2,
            token_mlp_dim=cenet_enc_layers[1],
            channel_mlp_dim=cenet_enc_layers[0],
            context_latent_size=cenet_latent_dim,
            activation=activation,
        )
        self.depth_frame_encoder = MotionRobustDepthEncoder(
            depth_image_resolution=self.depth_image_resolution,
            target_latent_dim=self.depth_latent_dim,
            cnn_activation=activation,
        )
        self.depth_sequence_encoder = ConvDepthSequenceEncoder(
            feature_dim=self.depth_latent_dim,
            sequence_length=self.depth_sequence_length,
            output_dim=self.depth_latent_dim,
            activation=activation,
        )
        self.context_encoder = MultimodalMixerVAE(
            depth_latent_dim=self.depth_latent_dim,
            proprio_latent_dim=cenet_latent_dim,
            output_dim=cenet_latent_dim,
            velo_dim=self.body_velo_dim,
            feet_state_dim=self.feet_state_dim,
            activation=activation,
        )
        
        # Get the activation function used by the actor and critic networks
        activation = get_activation(activation)

        self.init_noise_std = init_noise_std

        ###
        #  Construct the layers for the actor network
        ###
        # Shared layer between output branches
        actor_input_dim = num_actor_obs + cenet_latent_dim + cenet_velo_dim  # current obs o_t, force-aware latent dynamics z_t, explicit esitmation v_t 

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
        self.num_actions = num_actions
        
        self.distribution = None
        
        self.current_obs = None
        
        self._std_clip_lwr = 0.1
        
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _init_std(self, std_val=1.00):                
        self.std.data.fill_(std_val)

    def get_optim_groups(self, weight_decay: float = 1e-4, strong_decay: float = 1e-1):
        """Separate actor/critic params and each KITE encoder submodule.
        
        Args:
            weight_decay (float): Weight decay value for regularization. Default: 1e-4.
            
        Returns:
            Actor/critic parameter groups and a dict of encoder parameter groups.
        """
        critic_set = set()
        actor_set = set()
        no_decay  = set()
        encoder_sets = {
            "proprioceptive": set(),
            "visual_frame": set(),
            "visual_sequence": set(),
            "modality_mixer": set(),
        }
        whitelist = (nn.Linear, nn.MultiheadAttention)
        blacklist = (nn.LayerNorm, nn.Embedding, nn.Parameter)
        encoder_prefix_to_group = {
            "proprio_context_encoder.": "proprioceptive",
            "depth_frame_encoder.": "visual_frame",
            "depth_sequence_encoder.": "visual_sequence",
            "context_encoder.": "modality_mixer",
        }

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
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
        inter_params = actor_set & no_decay & encoder_set & critic_set
        if inter_params:
            raise ValueError(f"Parameters in all sets: {inter_params}")
        missing_params = param_dict.keys() - (
            actor_set | no_decay | encoder_set | critic_set
        )
        if missing_params:
            raise ValueError(f"Parameters not categorized: {missing_params}")
        # print(f"Parameters with extra strong weight decay{special_decay}")

        params_act = [{"params": [param_dict[pn] for pn in sorted(actor_set)],  "weight_decay": 0.0, "name":"actor"},
                      {"params": [param_dict[pn] for pn in sorted(critic_set)], "weight_decay": 0.0, "name":"critic"},
                      {"params": [param_dict[pn] for pn in sorted(no_decay)],   "weight_decay": 0.0}]
        
        params_enc = {
            name: [
                {
                    "params": [param_dict[pn] for pn in sorted(param_names)],
                    "weight_decay": weight_decay,
                    "name": name,
                }
            ]
            for name, param_names in encoder_sets.items()
        }

        return params_act, params_enc

    def configure_optimizers(self,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 1e-6,
                             strong_decay: float = 1e-1,
                             betas: Tuple[float, float] = (0.9, 0.999)) -> torch.optim.Optimizer:
        """Configure the AdamW optimizer with parameter groups.

        Actor and critic share one AdamW optimizer. The KITE encoder stack uses
        one Adam optimizer per sub-encoder:
            proprioceptive, visual_frame, visual_sequence, modality_mixer.
            
        Returns:
            Configured AdamW optimizer.
        """
        opt_groups_act, opt_groups_enc = self.get_optim_groups(weight_decay=weight_decay, strong_decay=strong_decay)
        act_opt = torch.optim.AdamW(opt_groups_act, lr=learning_rate)
        enc_opt = OptimizerBundle(
            {
                name: torch.optim.Adam(
                    groups,
                    lr=2.0e-4,
                    betas=betas,
                )
                for name, groups in opt_groups_enc.items()
            }
        )
        return act_opt, enc_opt

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def build_depth_async_pipeline(self) -> KITEDepthAsyncPipeline:
        return KITEDepthAsyncPipeline(self)

    def build_actor_async_pipeline(self) -> KITEActorAsyncPipeline:
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
        _, _, latest_depth_z, _ = self.depth_frame_encoder(
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
            self.depth_sequence_encoder(depth_sequence)
        )

        mean, logvar, z, body_velo, feet_state = self.context_encoder(
            depth_seq_z,
            proprio_z,
        )

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
        latest_depth_z = self.depth_frame_encoder.forward_inference(
            depth_image,
            depth_torso_state,
        )
        depth_sequence = self._build_depth_latent_sequence(
            latest_depth_z,
            depth_latent_history,
        )
        depth_seq_z = self.depth_sequence_encoder.forward_inference(
            depth_sequence
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
    def act(
        self,
        obs,
        obs_history,
        depth_image=None,
        depth_latent_history=None,
        depth_torso_state=None,
        **kwargs,
    ):
        # Call the forward method of the context encoder
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
        
        # return a sample from the distribution to be executed in simulation
        return sample
    

    # method used during simulated training
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
        # Call the forward method of the context encoder
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
