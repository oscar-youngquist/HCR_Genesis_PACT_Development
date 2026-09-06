"""Standalone HardPACT actor-critic copied from the legacy PACT implementation."""

from __future__ import annotations

from typing import Tuple, List, Dict, Any
from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hard_pact_physics import (
    DeploymentPhysicsHeads,
    ExplicitEstimatorDecoder,
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Kaiming uniform initialization for weights
        torch.nn.init.xavier_uniform_(m.weight)
        # Initialize biases to zero if they exist
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

###
#
#   Context Encoder/Decoder models used to provide conditioning input
#
###
class ContextEncoder(nn.Module):
    """VAE-style shared history trunk and latent distribution encoder.

    Mean, log-variance, and the explicit estimate are separate branches of
    the same shared history feature tensor.

    Args:
        context_input_dim (int): Dimension of input context features. Default: 230.
        context_layer_sizes (List[int]): Sizes of hidden layers. Default: [128, 64, 32].
        context_latent_size (int): Dimension of latent space. Default: 16.
        context_torso_velo_size (int): Dimension of velocity output. Default: 3.
        device (str): Device for tensor operations. Default: "cpu".
    """
    def __init__(
        self,
        context_input_dim: int = 230,
        context_layer_sizes: List[int] = [128, 64],
        context_latent_size: int = 16,
        context_torso_velo_size: int = 3,
        activation: str = 'swish',
        device: str = "cpu"
    ) -> None:
        super().__init__()

        output_hdim = 2*(context_latent_size + context_torso_velo_size)
        self.feature_dim = output_hdim

        # VAE-style context encoder layers
        # Input Layer
        self.ce_in = nn.Linear(context_input_dim, context_layer_sizes[0])
        # Hidden Layers
        self.ce_h1 = nn.Linear(context_layer_sizes[0], context_layer_sizes[1])
        self.ce_h2 = nn.Linear(context_layer_sizes[1], output_hdim)
        # Output Layers

        self.ce_latmean_h = nn.Linear(output_hdim, output_hdim)
        self.ce_latvar_h  = nn.Linear(output_hdim, output_hdim)

        self.ce_out_mean = nn.Linear(output_hdim, context_latent_size)
        self.ce_out_var = nn.Sequential(
            nn.Linear(output_hdim, context_latent_size),
            nn.Hardtanh(min_val=-5., max_val=5.))

        self.activation = get_activation(activation)

        # self.ce_timestep = nn.Linear(context_layer_sizes[2], 1)
        self.device = device
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize all linear layers with Xavier uniform distribution."""
        for layer in [self.ce_in, self.ce_h1,
                     self.ce_out_mean, self.ce_h2,
                     self.ce_latmean_h, self.ce_latvar_h]:
            
            nn.init.xavier_uniform_(layer.weight)
            
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        self.ce_out_var.apply(init_weights)

    def encode_features(self, X_C: torch.Tensor) -> torch.Tensor:
        """Return the shared history features consumed by every output branch."""
        x = self.activation(self.ce_in(X_C))
        x = self.activation(self.ce_h1(x))
        return self.activation(self.ce_h2(x))

    def distribution_from_features(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lat_mean = self.activation(self.ce_latmean_h(features))
        lat_var = self.activation(self.ce_latvar_h(features))
        return self.ce_out_mean(lat_mean), self.ce_out_var(lat_var)

    def encode_with_features(self, X_C: torch.Tensor):
        features = self.encode_features(X_C)
        mean, logvar = self.distribution_from_features(features)
        return mean, logvar, features

    def encode(self, X_C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logvar, _ = self.encode_with_features(X_C)
        return mean, logvar

    def reparameterization_trick(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        epsilon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from latent space using reparameterization trick.

        Args:
            mean: Latent space mean
            logvar: Latent space log variance

        Returns:
            Sampled latent vector
        """
        if epsilon is None:
            epsilon = torch.randn_like(logvar)
        return mean + torch.exp(0.5 * logvar) * epsilon

    def forward(self, X_C: torch.Tensor):
        """Return the latent-distribution parameters for compatibility.

        Args:
            X_C: Input context tensor

        Returns:
            Tuple containing the latent mean and log variance.
        """
        return self.encode(X_C)
    
    def forward_inference(self, X_C: torch.Tensor):
        mean, _ = self.encode(X_C)
        return mean

class ContextDecoder(nn.Module):
    """Decoder network for reconstructing next state from latent representation and velocity.
    
    Takes a latent vector and torso velocity as input, processes through two ELU-activated
    hidden layers, and outputs a predicted next state. Uses Xavier uniform initialization.
    """

    def __init__(
            self,
            input_dim: int = 19,
            layers: List[int] = [64,128],
            decode_dim: int = 57) -> None:
        super().__init__()


        # Network architecture
        self.dec_in = nn.Linear(input_dim, layers[0])
        self.dec_h1 = nn.Linear(layers[0], layers[1])
        self.dec_h2 = nn.Linear(layers[1], layers[2])
        self.dec_out = nn.Linear(layers[2], decode_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize all linear layers with Xavier uniform distribution."""
        for layer in [self.dec_in, self.dec_h1, self.dec_h2, self.dec_out]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder network.

        Args:
            condition: Latent vector of shape (batch_size, latent_dim+velo_dim)
        Returns:
            Reconstructed next state of shape (batch_size, decode_dim)
        """
        # Process through network with ELU activations
        x = F.elu(self.dec_in(condition))
        x = F.elu(self.dec_h1(x))
        x = F.elu(self.dec_h2(x))
        return self.dec_out(x)

class ActorCritic_HardPACT(nn.Module):
    def __init__(self, 
                 num_actor_obs=57, 
                 num_critic_obs=95, 
                 num_actions=12,
                 actor_layers=[512,256,128],
                 critic_layers=[1024,256,128,64],
                 cenet_in_dim=350,
                 cenet_latent_dim=16,
                 cenet_velo_dim=11, 
                 cenet_enc_layers=[256,128,64],
                 activation="tanh", 
                 init_noise_std=1.0,
                 cenet_explicit_layers=(128, 128),
                 grf_decoder_layers=(128, 128),
                 wrench_decoder_layers=(128, 128),
                 grf_scale_n=None,
                 wrench_scale=None,
                 wrench_qp_clip=None,
                 contact_epsilon=1.0e-2,
                 ablation_features="full"):
        super().__init__()

        # Metadata only: every ablation deliberately constructs the identical
        # module graph and therefore identical state-dict keys.
        from rsl_rl.hard_pact_ablations import resolve_hard_pact_features
        self.hard_pact_features = resolve_hard_pact_features(ablation_features)

        # The history latent is an architecture hyperparameter.  Only the
        # deployment estimator contract is fixed: [base velocity (3), contact
        # logits (4), foot clearances (4)] = 11 values.
        if cenet_velo_dim != 11:
            raise ValueError("HardPACT requires an 11-D explicit estimator")

        # Construct the context encoder network
        self.context_encoder = ContextEncoder(context_input_dim=cenet_in_dim,
                                              context_layer_sizes=cenet_enc_layers,
                                              context_latent_size=cenet_latent_dim,
                                              context_torso_velo_size=cenet_velo_dim,
                                              activation=activation)
        self.explicit_estimator = ExplicitEstimatorDecoder(
            self.context_encoder.feature_dim,
            cenet_explicit_layers, cenet_velo_dim,
            contact_epsilon=contact_epsilon,
        )
        self.physics_estimator = DeploymentPhysicsHeads(
            latent_dim=cenet_latent_dim,
            explicit_dim=cenet_velo_dim,
            grf_hidden_layers=grf_decoder_layers,
            wrench_hidden_layers=wrench_decoder_layers,
            grf_scale_n=grf_scale_n,
            wrench_scale=wrench_scale,
            wrench_qp_clip=wrench_qp_clip,
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
        
        self.act_trunk = nn.Sequential(*shared_trunk_layers)
        
        # Output head for position control
        self.act_pos_out = nn.Linear(actor_layers[-1], num_actions)

        # Output head for torque control
        self.act_tau_out = nn.Linear(actor_layers[-1], num_actions)

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

        # Used to track these values during training....
        #     These values will not be used during inference (sim or real)
        self.cenet_mean = None 
        self.cenet_logvar = None
        self.cenet_z = None
        self.cenet_torso_velo = None

        self.mean_pos = None
        self.mean_tau = None

        self.std = nn.Parameter(init_noise_std * torch.ones(2*num_actions))
        self.num_actions = num_actions
        
        self._std_clip_lwr = 0.20

        self.distribution = None
        
        self.current_obs = None
        
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        nn.init.uniform_(self.act_tau_out.weight, -1.0e-6, 1.0e-6)
        nn.init.zeros_(self.act_tau_out.bias)

    # Currently unused...
    def _init_weights(self):
        # Xavier for linears, zeros for biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # # Optionally set small initial output weights (to reduce initial action magnitude)
        nn.init.uniform_(self.act_pos_out.weight, -3e-2, 3e-2)
        nn.init.uniform_(self.act_tau_out.weight, -3e-6, 3e-6)
        nn.init.zeros_(self.act_pos_out.bias)
        nn.init.zeros_(self.act_tau_out.bias)

    def _init_std(self, std_val=1.00):                
        self.std.data.fill_(std_val)
        

    def get_optim_groups(self, weight_decay: float = 1e-4, strong_decay: float = 1e-1):
        """Separate parameters into groups:
            (1) does not use weight decay
            (2) uses extra strong weight decay (FiLM layers)
            (3) shared parameters with weight decay
            (4) parameters for the position control output with weight decay
            (5) parameters for the torque control output with weight decay
        
        Args:
            weight_decay (float): Weight decay value for regularization. Default: 1e-4.
            
        Returns:
            List of parameter groups for optimizer initialization.
        """
        critic_set = set()
        actor_set = set()
        no_decay  = set()
        encoder = set()
        whitelist = (nn.Linear, nn.MultiheadAttention)
        blacklist = (nn.LayerNorm, nn.Embedding, nn.Parameter)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                if isinstance(m, blacklist):
                    no_decay.add(fpn)
                elif isinstance(m, whitelist):
                    if ("context_encoder" in fpn or "explicit_estimator" in fpn
                            or "physics_estimator" in fpn):
                        encoder.add(fpn)
                    elif "critic" in fpn:
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
        inter_params = actor_set & no_decay & encoder & critic_set
        if inter_params:
            raise ValueError(f"Parameters in all sets: {inter_params}")
        missing_params = param_dict.keys() - (actor_set | no_decay | encoder | critic_set)
        if missing_params:
            raise ValueError(f"Parameters not categorized: {missing_params}")
        # print(f"Parameters with extra strong weight decay{special_decay}")

        params_act = [{"params": [param_dict[pn] for pn in sorted(actor_set)],  "weight_decay": 0.0, "name":"actor"},
                      {"params": [param_dict[pn] for pn in sorted(critic_set)], "weight_decay": 0.0, "name":"critic"},
                      {"params": [param_dict[pn] for pn in sorted(no_decay)],   "weight_decay": 0.0}]
        
        params_enc = [{"params": [param_dict[pn] for pn in sorted(encoder)], "weight_decay": weight_decay, "name":"encoder"}]

        return params_act, params_enc

    def physics_heads(self, latent, explicit, nominal_torque):
        return self.physics_estimator(latent, explicit, nominal_torque)

    def physics_heads_from_history(
        self, obs_history, nominal_torque, latent_noise=None
    ):
        _, _, latent, explicit = self.cenet_enc_forward(
            obs_history, latent_noise=latent_noise
        )
        return self.physics_estimator(latent, explicit, nominal_torque)

    def configure_optimizers(self,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 1e-6,
                             strong_decay: float = 1e-1,
                             betas: Tuple[float, float] = (0.9, 0.999)) -> torch.optim.Optimizer:
        """Configure the AdamW optimizer with parameter groups.

        Standard weights in Linear/Attention layers - weight_decay
        all bias terms - no decay
        FiLM layer parameters - strong_decay
            
        Returns:
            Configured AdamW optimizer.
        """
        opt_groups_act, opt_groups_enc = self.get_optim_groups(weight_decay=weight_decay, strong_decay=strong_decay)

        # PPO and the auxiliary VAE update both optimize the context encoder.
        # These are separate optimizer param-group dictionaries that point at
        # the same underlying nn.Parameter objects.
        ppo_enc_groups = [
            {
                "params": list(group["params"]),
                "weight_decay": group.get("weight_decay", 0.0),
                "name": f"ppo_{group['name']}",
            }
            for group in opt_groups_enc
        ]
        auxiliary_enc_groups = [
            {
                "params": list(group["params"]),
                "weight_decay": group.get("weight_decay", 0.0),
                "name": f"auxiliary_{group['name']}",
            }
            for group in opt_groups_enc
        ]

        act_opt = torch.optim.AdamW([*opt_groups_act, *ppo_enc_groups], lr=learning_rate, betas=betas)
        enc_opt = torch.optim.AdamW(auxiliary_enc_groups, lr=2.0e-4, betas=betas)
        return act_opt, enc_opt

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    # forward methods for the histroical context VAE
    def cenet_enc_forward(self, obs_history, latent_noise=None):
        mean, logvar, features = self.context_encoder.encode_with_features(
            obs_history
        )
        if latent_noise is None:
            latent_noise = torch.randn_like(logvar)
        latent = self.context_encoder.reparameterization_trick(
            mean, logvar, latent_noise
        )
        explicit = self.explicit_estimator(features)
        return mean, logvar, latent, explicit
    
    def cenet_enc_inference(self, obs_history):
        mean, _, features = self.context_encoder.encode_with_features(obs_history)
        return mean, self.explicit_estimator(features)

    # Method for the forward method of the actor network, used mostly as an internal method
    def actor_forward(self, current_obs):
        # We are assuming "current_obs" includes all of the components used in the dreamwaq policy input
        latent = self.act_trunk(current_obs)

        # Now run the final output layers to get both action modalities
        act_pos_act = self.act_pos_out(latent)
        act_tau_act = self.act_tau_out(latent)

        if torch.isnan(act_pos_act).any():
            with torch.no_grad():
                act_pos_act = torch.nan_to_num(act_pos_act, nan=0.0, posinf=0.0, neginf=0.0)

        if torch.isnan(act_tau_act).any():
            with torch.no_grad():
                act_tau_act = torch.nan_to_num(act_tau_act, nan=0.0, posinf=0.0, neginf=0.0)

        return act_pos_act, act_tau_act

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
        mean_pos, mean_tau = self.actor_forward(curr_obs)
        self.mean_pos = mean_pos
        self.mean_tau = mean_tau

        self._clip_std()

        mean = torch.cat([mean_pos, mean_tau], dim=-1)

        self.distribution = Normal(mean, mean * 0.0 + self.std)

    # method used during simulated training
    @torch.jit.ignore
    def act(self, obs, obs_history, latent_noise=None, **kwargs):
        # Call the forward method of the context encoder
        mean, logvar, z, torso_velo = self.cenet_enc_forward(
            obs_history, latent_noise=latent_noise
        )
        
        # Training uses the reparameterized content sample. The explicit
        # estimate is a sibling encoder branch and therefore does not depend
        # on that sampling noise.
        current_obs = torch.cat((obs, z, torso_velo), dim=-1)
        
        # Upated the PPO training distribution
        self.update_distribution(current_obs)
        
        # log context-encoder values to be used in PPO class for calculating context encoder network
        self.cenet_mean = mean
        self.cenet_logvar = logvar
        self.cenet_z = z
        self.cenet_torso_velo = torso_velo

        sample = self.distribution.sample()
        
        # return a sample from the distribution to be executed in simulation
        return sample
    

    # method used during simulated training
    @torch.jit.ignore
    def act_bootmask(self, obs, obs_history, latent_noise=None, **kwargs):
        # Call the forward method of the context encoder
        mean, logvar, z, torso_velo = self.cenet_enc_forward(
            obs_history, latent_noise=latent_noise
        )
        
        # Mask the latent/velo from the encoder with zeros
        boot_mask = torch.zeros((z.shape[0], (z.shape[1] + torso_velo.shape[1])), device=obs.device)

        # create the actors observation
        current_obs = torch.cat((obs,boot_mask), dim=-1)   
        
        # Upated the PPO training distribution
        self.update_distribution(current_obs)
        
        # log context-encoder values to be used in PPO class for calculating context encoder network
        self.cenet_mean = mean
        self.cenet_logvar = logvar
        self.cenet_z = z
        self.cenet_torso_velo = torso_velo

        sample = self.distribution.sample()
        
        # return a sample from the distribution to be executed in simulation
        return sample
    
    # Method using during simulated inference
    @torch.jit.export
    def act_inference(self,obs,obs_history):
        # Call the forward method of the context encoder
        z, torso_velo = self.cenet_enc_inference(obs_history)
        # The environment reuses these policy-rate features for every
        # decimation-substep QP; no second history-encoder pass is required.
        self.cenet_z = z
        self.cenet_torso_velo = torso_velo
        
        # create the actors observation
        current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
                
        # call the actors forward method and return it's results
        actions_pos, actions_tau = self.actor_forward(current_obs)

        total_sample = torch.cat([actions_pos, actions_tau], dim=1)

        return total_sample
    

    def act_inference_recon(self,obs,obs_history):
        # Call the forward method of the context encoder
        z, torso_velo = self.cenet_enc_inference(obs_history)
        
        # create the actors observation
        current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
                
        # call the actors forward method and return it's results
        actions_pos, actions_tau = self.actor_forward(current_obs)

        total_sample = torch.cat([actions_pos, actions_tau], dim=1)

        return total_sample, z, torso_velo

    # Forward method for calculating the value of the current state
    #     using the privilged critic observation
    def evaluate(self, critic_observations, **kwargs):
        val = self.critic(critic_observations)
        return val


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "swish":
        return nn.SiLU()
    else:
        print("invalid activation function!")
        return None
