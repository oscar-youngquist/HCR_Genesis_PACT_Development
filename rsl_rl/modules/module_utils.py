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
