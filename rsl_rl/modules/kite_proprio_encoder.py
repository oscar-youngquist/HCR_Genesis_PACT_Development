from __future__ import annotations

from typing import Tuple, List, Dict, Any
from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_utils import (
    MLPMixerBlock,
    get_activation,
    init_weights,
)

class ProprioContextMLPMixerKITE(nn.Module):
    """
    VAE-style proprioceptive context encoder using an MLP-Mixer architecture.

    Intended default input:
        X_C: B x 450

    Interpreted as:
        X_C reshaped to B x 45 x 10

    where:
        45 = proprioceptive feature tokens
        10 = history timesteps per feature

    The architecture is:
        B x 45 x 10
            -> per-feature temporal embedding
        B x 45 x hidden_dim
            -> MLP-Mixer blocks
        B x 45 x hidden_dim
            -> token pooling
        B x hidden_dim
            -> VAE-style latent mean/logvar heads

    Args:
        context_input_dim:
            Flattened input dimension. Default 450 = 45 x 10.

        num_tokens:
            Number of proprioceptive feature tokens. Default 45.

        input_dim_per_token:
            History length / channel dimension per token. Default 10.

        hidden_dim:
            Per-token embedding dimension after the initial temporal projection.

        num_mixer_blocks:
            Number of MLP-Mixer blocks.

        token_mlp_dim:
            Hidden dimension inside token-mixing MLP.

        channel_mlp_dim:
            Hidden dimension inside channel-mixing MLP.

        context_latent_size:
            Final latent size.

        activation:
            Activation passed through module_utils.get_activation(...).

        use_layer_norm:
            Whether to use LayerNorm inside mixer blocks.

        logvar_min/logvar_max:
            Bounds for VAE log-variance output.

        use_vae:
            If True, samples z during training. If False, z = mean.

        device:
            Kept for compatibility with your existing encoder API.
    """

    def __init__(
        self,
        context_input_dim: int = 450,
        num_tokens: int = 45,
        input_dim_per_token: int = 10,
        hidden_dim: int = 128,
        num_mixer_blocks: int = 2,
        token_mlp_dim: int = 128,
        channel_mlp_dim: int = 256,
        context_latent_size: int = 16,
        activation: str = "elu",
        use_layer_norm: bool = True,
        logvar_min: float = -5.0,
        logvar_max: float = 5.0,
        use_vae: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        expected_input_dim = num_tokens * input_dim_per_token
        if context_input_dim != expected_input_dim:
            raise ValueError(
                f"context_input_dim must equal num_tokens * input_dim_per_token. "
                f"Got context_input_dim={context_input_dim}, "
                f"but num_tokens * input_dim_per_token = {expected_input_dim}."
            )

        self.context_input_dim = context_input_dim
        self.num_tokens = num_tokens
        self.input_dim_per_token = input_dim_per_token
        self.hidden_dim = hidden_dim
        self.num_mixer_blocks = num_mixer_blocks
        self.token_mlp_dim = token_mlp_dim
        self.channel_mlp_dim = channel_mlp_dim
        self.context_latent_size = context_latent_size
        self.activation_name = activation
        self.use_layer_norm = use_layer_norm
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.use_vae = use_vae
        self.device = device

        self.activation = get_activation(activation)

        # ------------------------------------------------------------------
        # Per-feature temporal embedding:
        #     B x 45 x 10 -> B x 45 x hidden_dim
        # ------------------------------------------------------------------
        self.token_embedding = nn.Linear(input_dim_per_token, hidden_dim)

        # ------------------------------------------------------------------
        # Mixer trunk.
        # ------------------------------------------------------------------
        self.mixer_blocks = nn.ModuleList(
            [
                MLPMixerBlock(
                    num_tokens=num_tokens,
                    hidden_dim=hidden_dim,
                    token_mlp_dim=token_mlp_dim,
                    channel_mlp_dim=channel_mlp_dim,
                    activation=activation,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_mixer_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        # ------------------------------------------------------------------
        # VAE-style latent heads.
        # ------------------------------------------------------------------
        output_hdim = 2 * context_latent_size

        self.ce_h = nn.Linear(hidden_dim, output_hdim)

        self.ce_latmean_h = nn.Linear(output_hdim, output_hdim)
        self.ce_latvar_h = nn.Linear(output_hdim, output_hdim)

        self.ce_out_mean = nn.Linear(output_hdim, context_latent_size)

        self.ce_out_var = nn.Sequential(
            nn.Linear(output_hdim, context_latent_size),
            nn.Hardtanh(min_val=logvar_min, max_val=logvar_max),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialization routine following the style used in your existing
        context encoders.

        Hidden feature layers:
            Kaiming uniform with leaky_relu gain approximation.

        Mean/logvar output heads:
            Kaiming uniform with linear nonlinearity.

        LayerNorm:
            weight = 1, bias = 0.
        """

        # Feature embedding and hidden latent layers.
        for layer in [
            self.token_embedding,
            self.ce_h,
            self.ce_latmean_h,
            self.ce_latvar_h,
        ]:
            nn.init.kaiming_uniform_(
                layer.weight,
                a=1.0,
                mode="fan_in",
                nonlinearity="leaky_relu",
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Mean output head.
        nn.init.kaiming_uniform_(
            self.ce_out_mean.weight,
            a=1.0,
            mode="fan_in",
            nonlinearity="linear",
        )
        if self.ce_out_mean.bias is not None:
            nn.init.zeros_(self.ce_out_mean.bias)

        # Logvar output head inside Sequential.
        for module in self.ce_out_var:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=1.0,
                    mode="fan_in",
                    nonlinearity="linear",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # LayerNorm layers.
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _format_input(self, X_C: torch.Tensor) -> torch.Tensor:
        """
        Supports:
            X_C: B x 450
        or:
            X_C: B x 45 x 10

        Returns:
            X_C: B x 45 x 10
        """

        if X_C.dim() == 3:
            batch_size, num_tokens, input_dim_per_token = X_C.shape

            if num_tokens != self.num_tokens:
                raise ValueError(
                    f"Expected num_tokens={self.num_tokens}, but got {num_tokens}."
                )

            if input_dim_per_token != self.input_dim_per_token:
                raise ValueError(
                    f"Expected input_dim_per_token={self.input_dim_per_token}, "
                    f"but got {input_dim_per_token}."
                )

            return X_C

        if X_C.dim() == 2:
            batch_size, flat_dim = X_C.shape

            if flat_dim != self.context_input_dim:
                raise ValueError(
                    f"Expected flattened context_input_dim={self.context_input_dim}, "
                    f"but got {flat_dim}."
                )

            return X_C.view(
                batch_size,
                self.num_tokens,
                self.input_dim_per_token,
            )

        raise ValueError(
            f"Expected X_C shape Bx450 or Bx45x10, but got {tuple(X_C.shape)}."
        )

    def encode(self, X_C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X_C:
                B x 450
            or:
                B x 45 x 10

        Returns:
            mean:
                B x context_latent_size

            logvar:
                B x context_latent_size
        """

        x = self._format_input(X_C)

        # B x 45 x 10 -> B x 45 x hidden_dim
        x = self.activation(self.token_embedding(x))

        # Mixer trunk.
        for block in self.mixer_blocks:
            x = block(x)

        x = self.final_norm(x)

        # Global pooling over feature tokens.
        # B x 45 x hidden_dim -> B x hidden_dim
        x = x.mean(dim=1)

        # Latent heads.
        x = self.activation(self.ce_h(x))

        lat_mean = self.activation(self.ce_latmean_h(x))
        lat_var = self.activation(self.ce_latvar_h(x))

        mean = self.ce_out_mean(lat_mean)
        logvar = self.ce_out_var(lat_var)

        return mean, logvar

    def reparameterization_trick(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample from latent space using the reparameterization trick.

        Args:
            mean:
                B x context_latent_size

            logvar:
                B x context_latent_size

        Returns:
            z:
                B x context_latent_size
        """
        epsilon = torch.randn_like(logvar).to(logvar.device)
        return mean + torch.exp(0.5 * logvar) * epsilon

    def forward(self, X_C: torch.Tensor):
        """
        Complete forward pass.

        Returns:
            mean:
                B x context_latent_size

            logvar:
                B x context_latent_size

            z:
                B x context_latent_size
        """
        mean, logvar = self.encode(X_C)

        if self.use_vae and self.training:
            z = self.reparameterization_trick(mean, logvar)
        else:
            z = mean

        return mean, logvar, z

    @torch.no_grad()
    def forward_inference(self, X_C: torch.Tensor):
        """
        Deterministic inference path.

        Returns:
            mean:
                B x context_latent_size
        """
        mean, _ = self.encode(X_C)
        return mean


# ###
# #
# #   Context Encoder models used to provide conditioning input
# #
# ###
# class ProprioContextEncoderKITE(nn.Module):
#     """VAE-style encoder for processing context information with velocity prediction.

#     Encodes high-dimensional context into a latent distribution while simultaneously
#     predicting torso velocity. Uses ELU activations and Xavier initialization.

#     Args:
#         context_input_dim (int): Dimension of input context features. Default: 450 (45 x 10).
#         context_layer_sizes (List[int]): Sizes of hidden layers. Default: [256, 128].
#         context_latent_size (int): Dimension of latent space. Default: 16.
#         device (str): Device for tensor operations. Default: "cpu".
#     """
#     def __init__(
#         self,
#         context_input_dim: int = 450,
#         context_layer_sizes: List[int] = [256, 128],
#         context_latent_size: int = 16,
#         activation: str = 'elu',
#         device: str = "cpu"
#     ) -> None:
#         super().__init__()

#         output_hdim = 2*context_latent_size

#         # VAE-style context encoder layers
#         # Input Layer
#         self.ce_in = nn.Linear(context_input_dim, context_layer_sizes[0])
#         # Hidden Layers
#         self.ce_h1 = nn.Linear(context_layer_sizes[0], context_layer_sizes[1])
#         self.ce_h2 = nn.Linear(context_layer_sizes[1], output_hdim)
        
#         # Output Layers
#         self.ce_latmean_h = nn.Linear(output_hdim, output_hdim)
#         self.ce_latvar_h  = nn.Linear(output_hdim, output_hdim)

#         self.ce_out_mean = nn.Linear(output_hdim, context_latent_size)
#         self.ce_out_var = nn.Sequential(
#             nn.Linear(output_hdim, context_latent_size),
#             nn.Hardtanh(min_val=0., max_val=5.))

        
#         self.activation = get_activation(activation)

#         # self.ce_timestep = nn.Linear(context_layer_sizes[2], 1)
#         self.device = device
#         self._initialize_weights()

#     # def _initialize_weights(self) -> None:
#     #     """Initialize all linear layers with Xavier uniform distribution."""
#     #     for layer in [self.ce_in,
#     #                   self.ce_h1,
#     #                   self.ce_h2,
#     #                   self.ce_out_mean,
#     #                   self.ce_latmean_h,
#     #                   self.ce_latvar_h]:
            
#     #         nn.init.xavier_uniform_(layer.weight)
            
#     #         if layer.bias is not None:
#     #             nn.init.zeros_(layer.bias)

#     #     self.ce_out_var.apply(init_weights)

#     def _initialize_weights(self) -> None:
#         """
#         Custom initialization routine targeting all MLP context encoder layers 
#         and VAE projection heads.
#         """
#         # --- 1. Initialize Feature Extraction & Hidden Subspace Layers ---
#         for layer in [self.ce_in, 
#                       self.ce_h1, 
#                       self.ce_h2, 
#                       self.ce_latmean_h, 
#                       self.ce_latvar_h]:
            
#             nn.init.kaiming_uniform_(
#                 layer.weight, 
#                 a=1.0, 
#                 mode='fan_in', 
#                 nonlinearity='leaky_relu' # Closest standard geometric scaling for Swish
#             )
#             if layer.bias is not None:
#                 nn.init.zeros_(layer.bias)

#         # --- 2. Initialize VAE Mean Output Head ---
#         # Set to linear nonlinearity to prevent scaling overshooting
#         nn.init.kaiming_uniform_(self.ce_out_mean.weight, a=1.0, mode='fan_in', nonlinearity='linear')
#         if self.ce_out_mean.bias is not None:
#             nn.init.zeros_(self.ce_out_mean.bias)

#         # --- 3. Initialize VAE Variance Sequential Output Head ---
#         # Loops through the sequential block to isolate and configure the Linear layer
#         for module in self.ce_out_var:
#             if isinstance(module, nn.Linear):
#                 nn.init.kaiming_uniform_(module.weight, a=1.0, mode='fan_in', nonlinearity='linear')
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)

#     def encode(self, X_C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         # Forward pass through encoder
#         x = self.activation(self.ce_in(X_C))
#         x = self.activation(self.ce_h1(x))
#         x = self.activation(self.ce_h2(x))
        
#         lat_mean = self.activation(self.ce_latmean_h(x))
#         lat_var  = self.activation(self.ce_latvar_h(x))

#         return self.ce_out_mean(lat_mean), self.ce_out_var(lat_var)

#     def reparameterization_trick(
#         self,
#         mean: torch.Tensor,
#         logvar: torch.Tensor
#     ) -> torch.Tensor:
#         """Sample from latent space using reparameterization trick.

#         Args:
#             mean: Latent space mean
#             logvar: Latent space log variance

#         Returns:
#             Sampled latent vector
#         """
#         epsilon = torch.randn_like(logvar).to(logvar.device)
#         return mean + torch.exp(0.5 * logvar) * epsilon

#     def forward(self, X_C: torch.Tensor):
#         """Complete forward pass including encoding and sampling.

#         Args:
#             X_C: Input context tensor

#         Returns:
#             Tuple containing:
#                 - mean: Latent space mean
#                 - logvar: Latent space log variance
#                 - z: Sampled latent vector
#         """
#         mean, logvar = self.encode(X_C)
#         z = self.reparameterization_trick(mean, logvar)
#         return mean, logvar, z
    
#     def forward_inference(self, X_C: torch.Tensor):
#         mean, logvar = self.encode(X_C)
#         return mean
