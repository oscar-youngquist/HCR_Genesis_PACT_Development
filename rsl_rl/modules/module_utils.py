from __future__ import annotations

from typing import Tuple, List, Dict, Any
from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        # Kaiming uniform initialization for weights
        torch.nn.init.xavier_uniform_(m.weight)
        # Initialize biases to zero if they exist
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# --------------------------------------------------------------------------
# Attention utilities
#     Based on - https://sanjayasubedi.com.np/deeplearning/multihead-attention-from-scratch/, I've found it to be slightly faster than PyTorch built-in
# --------------------------------------------------------------------------

def scaled_dot_product_attention(query, key, value):
    """
    query: B x n_heads x Q x D
    key:   B x n_heads x K x D
    value: B x n_heads x K x D
    """
    assert query.size(-1) == key.size(-1)

    dk = key.size(-1)
    logits = query @ key.transpose(-1, -2) / (dk ** 0.5)
    attn_weights = torch.softmax(logits, dim=-1)
    attn = attn_weights @ value

    return attn, attn_weights


class EfficientMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        projection_bias: bool = False,
    ):
        super().__init__()

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_embed_dim = embed_dim // n_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=projection_bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=projection_bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=projection_bias)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=projection_bias)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B x S x D
        returns: B x n_heads x S x D_head
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.n_heads, self.head_embed_dim)
        return x.transpose(1, 2)

    def forward(self, query, key, value):
        """
        query: B x Q x D
        key:   B x K x D
        value: B x K x D
        """
        batch_size = query.size(0)

        q = self.split_heads(self.W_q(query))
        k = self.split_heads(self.W_k(key))
        v = self.split_heads(self.W_v(value))

        attn, attn_weights = scaled_dot_product_attention(q, k, v)

        # B x n_heads x Q x D_head -> B x Q x D
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch_size, query.size(1), self.embed_dim)

        attn = self.dropout(attn)
        output = self.projection(attn)

        # Mean over attention heads: B x Q x K
        return output, attn_weights.mean(dim=1)


class MixerMLP(nn.Module):
    """
    Basic MLP used inside MLP-Mixer blocks.

    Used for both:
        - token mixing:   mixes across sampled points / tokens
        - channel mixing: mixes across feature channels
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "elu",
    ):
        super().__init__()

        self.activation = get_activation(activation)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(
                layer.weight,
                a=1.0,
                mode="fan_in",
                nonlinearity="leaky_relu",
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class MLPMixerBlock(nn.Module):
    """
    One MLP-Mixer block.

    Input:
        x: B x num_tokens x hidden_dim

    Token mixing:
        mixes information across tokens

    Channel mixing:
        mixes information across latent channels at each token.
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        activation: str = "elu",
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

        self.norm1 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        self.token_mixer = MixerMLP(
            input_dim=num_tokens,
            hidden_dim=token_mlp_dim,
            output_dim=num_tokens,
            activation=activation,
        )

        self.channel_mixer = MixerMLP(
            input_dim=hidden_dim,
            hidden_dim=channel_mlp_dim,
            output_dim=hidden_dim,
            activation=activation,
        )

        self._initialize_norms()

    def _initialize_norms(self) -> None:
        for m in [self.norm1, self.norm2]:
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B x num_tokens x hidden_dim
        """

        # Token mixing.
        y = self.norm1(x)

        # B x num_tokens x hidden_dim
        # -> B x hidden_dim x num_tokens
        y = y.transpose(1, 2)

        # Apply MLP across token dimension.
        y = self.token_mixer(y)

        # B x hidden_dim x num_tokens
        # -> B x num_tokens x hidden_dim
        y = y.transpose(1, 2)

        x = x + y

        # Channel mixing.
        y = self.norm2(x)
        y = self.channel_mixer(y)

        x = x + y

        return x


class ChannelFirstLayerNorm1d(nn.Module):
    """LayerNorm over channels for Conv1d outputs shaped B x C x L."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2).contiguous()


class ChannelFirstLayerNorm2d(nn.Module):
    """LayerNorm over channels for Conv2d outputs shaped B x C x H x W."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


def make_2d_norm(norm_type: str, num_channels: int) -> nn.Module:
    """
    Returns a normalization layer for 2D conv features.

    norm_type:
        "none"  -> Identity
        "batch" -> BatchNorm2d
        "layer" -> LayerNorm over channels
        "group" -> GroupNorm
    """
    norm_type = norm_type.lower()

    if norm_type == "none":
        return nn.Identity()

    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)

    if norm_type == "layer":
        return ChannelFirstLayerNorm2d(num_channels)

    if norm_type == "group":
        num_groups = min(8, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_channels)

    raise ValueError(
        f"Unknown norm_type={norm_type}. Expected one of: 'none', 'batch', 'layer', 'group'."
    )


def make_1d_norm(norm_type: str, num_channels: int) -> nn.Module:
    """
    Returns a normalization layer for 1D conv features.

    norm_type:
        "none"  -> Identity
        "batch" -> BatchNorm1d
        "layer" -> LayerNorm
        "group" -> GroupNorm
    """
    norm_type = norm_type.lower()

    if norm_type == "none":
        return nn.Identity()

    if norm_type == "batch":
        return nn.BatchNorm1d(num_channels)

    if norm_type == "layer":
        return ChannelFirstLayerNorm1d(num_channels)

    if norm_type == "group":
        num_groups = min(8, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_channels)

    raise ValueError(
        f"Unknown norm_type={norm_type}. Expected one of: 'none', 'batch', 'layer', 'group'."
    )


class ConvNormAct(nn.Module):
    """
    2D convolution block with configurable normalization.

    Default:
        Conv2d -> Identity -> activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module,
        norm_type: str = "none",
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            make_2d_norm(norm_type, out_channels),
            activation,
        )

    def forward(self, x):
        return self.block(x)


# --------------------------------------------------------------------------
# Optional contrastive projection head
# --------------------------------------------------------------------------
class ContrastiveProjectionHead(nn.Module):
    """
    Projection head for contrastive alignment.

    Use this for InfoNCE/cosine contrastive losses instead of applying the
    contrastive loss directly to the policy latent.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 32,
        hidden_dim: int = 64,
        activation: str = "elu",
    ):
        super().__init__()

        act = get_activation(activation)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, projection_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=1.0,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        z = self.net(x)
        z = F.normalize(z, p=2, dim=-1, eps=1e-6)
        return z



class ReconDimensionProjectionHead(nn.Module):
    """
    Dimension-matching head for aligning history encoders with privileged decoders.

    Use this for InfoNCE/cosine contrastive losses instead of applying the
    contrastive loss directly to the policy latent.
    """

    def __init__(
        self,
        input_dim: int,
        recon_dim: int = 32,
        activation: str = "elu",
    ):
        super().__init__()

        act = get_activation(activation)
        if hasattr(act, "inplace"):
            act.inplace = False

        self.net = nn.Sequential(
            act,
            nn.Linear(input_dim, recon_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=1.0,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        z = self.net(x)
        return z

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU(inplace=True)
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU(inplace=True)
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
