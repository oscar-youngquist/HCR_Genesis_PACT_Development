from __future__ import annotations

from typing import Tuple, List, Dict, Any
from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module_utils import init_weights, get_activation

###
#
#   Context Encoder/Decoder models used to provide conditioning input
#
###
class ProprioContextEncoderKITE(nn.Module):
    """VAE-style encoder for processing context information with velocity prediction.

    Encodes high-dimensional context into a latent distribution while simultaneously
    predicting torso velocity. Uses ELU activations and Xavier initialization.

    Args:
        context_input_dim (int): Dimension of input context features. Default: 450 (45 x 10).
        context_layer_sizes (List[int]): Sizes of hidden layers. Default: [256, 128].
        context_latent_size (int): Dimension of latent space. Default: 16.
        device (str): Device for tensor operations. Default: "cpu".
    """
    def __init__(
        self,
        context_input_dim: int = 450,
        context_layer_sizes: List[int] = [256, 128],
        context_latent_size: int = 16,
        activation: str = 'elu',
        device: str = "cpu"
    ) -> None:
        super().__init__()

        output_hdim = 2*context_latent_size

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
            nn.Hardtanh(min_val=0., max_val=5.))

        
        self.activation = get_activation(activation)

        # self.ce_timestep = nn.Linear(context_layer_sizes[2], 1)
        self.device = device
        self._initialize_weights()

    # def _initialize_weights(self) -> None:
    #     """Initialize all linear layers with Xavier uniform distribution."""
    #     for layer in [self.ce_in,
    #                   self.ce_h1,
    #                   self.ce_h2,
    #                   self.ce_out_mean,
    #                   self.ce_latmean_h,
    #                   self.ce_latvar_h]:
            
    #         nn.init.xavier_uniform_(layer.weight)
            
    #         if layer.bias is not None:
    #             nn.init.zeros_(layer.bias)

    #     self.ce_out_var.apply(init_weights)

    def _initialize_weights(self) -> None:
        """
        Custom initialization routine targeting all MLP context encoder layers 
        and VAE projection heads.
        """
        # --- 1. Initialize Feature Extraction & Hidden Subspace Layers ---
        for layer in [self.ce_in, 
                      self.ce_h1, 
                      self.ce_h2, 
                      self.ce_latmean_h, 
                      self.ce_latvar_h]:
            
            nn.init.kaiming_uniform_(
                layer.weight, 
                a=1.0, 
                mode='fan_in', 
                nonlinearity='leaky_relu' # Closest standard geometric scaling for Swish
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # --- 2. Initialize VAE Mean Output Head ---
        # Set to linear nonlinearity to prevent scaling overshooting
        nn.init.kaiming_uniform_(self.ce_out_mean.weight, a=1.0, mode='fan_in', nonlinearity='linear')
        if self.ce_out_mean.bias is not None:
            nn.init.zeros_(self.ce_out_mean.bias)

        # --- 3. Initialize VAE Variance Sequential Output Head ---
        # Loops through the sequential block to isolate and configure the Linear layer
        for module in self.ce_out_var:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=1.0, mode='fan_in', nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, X_C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass through encoder
        x = self.activation(self.ce_in(X_C))
        x = self.activation(self.ce_h1(x))
        x = self.activation(self.ce_h2(x))
        
        lat_mean = self.activation(self.ce_latmean_h(x))
        lat_var  = self.activation(self.ce_latvar_h(x))

        return self.ce_out_mean(lat_mean), self.ce_out_var(lat_var)

    def reparameterization_trick(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample from latent space using reparameterization trick.

        Args:
            mean: Latent space mean
            logvar: Latent space log variance

        Returns:
            Sampled latent vector
        """
        epsilon = torch.randn_like(logvar).to(logvar.device)
        return mean + torch.exp(0.5 * logvar) * epsilon

    def forward(self, X_C: torch.Tensor):
        """Complete forward pass including encoding and sampling.

        Args:
            X_C: Input context tensor

        Returns:
            Tuple containing:
                - mean: Latent space mean
                - logvar: Latent space log variance
                - z: Sampled latent vector
        """
        mean, logvar = self.encode(X_C)
        z = self.reparameterization_trick(mean, logvar)
        return mean, logvar, z
    
    def forward_inference(self, X_C: torch.Tensor):
        mean, logvar = self.encode(X_C)
        return mean


class ProprioContextDecoderKITE(nn.Module):
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
        self.dec_h3 = nn.Linear(layers[2], layers[3])
        self.dec_out = nn.Linear(layers[3], decode_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize all linear layers with Xavier uniform distribution."""
        for layer in [self.dec_in, self.dec_h1, self.dec_out, self.dec_h2, self.dec_h3]:
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
        x = F.elu(self.dec_h3(x))
        return self.dec_out(x)