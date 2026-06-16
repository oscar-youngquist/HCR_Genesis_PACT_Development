from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any

from module_utils import get_activation, EfficientMultiHeadAttention, MLPMixerBlock, make_2d_norm, make_1d_norm, ConvNormAct


class TerrainAttentionEncoder(nn.Module):
    """
    Compact terrain encoder using learned-query spatial attention pooling.

    Input:
        x: B x H x W x 4

    Output:
        z: B x latent_dim
    """

    def __init__(
        self,
        height: int,
        width: int,
        in_channels: int = 4,
        latent_dim: int = 32,
        cnn_activation: str = "elu",
        norm_type: str = "none",
        attention_dim: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.norm_type = norm_type

        self.activation = get_activation(cnn_activation)

        self.conv = nn.Sequential(
            ConvNormAct(
                in_channels,
                16,
                activation=self.activation,
                norm_type=norm_type,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ConvNormAct(
                16,
                32,
                activation=self.activation,
                norm_type=norm_type,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ConvNormAct(
                32,
                64,
                activation=self.activation,
                norm_type=norm_type,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ConvNormAct(
                64,
                attention_dim,
                activation=self.activation,
                norm_type=norm_type,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            conv_out = self.conv(dummy)
            self.conv_shape = conv_out.shape[1:]  # C x H' x W'

        self.spatial_attention = EfficientMultiHeadAttention(
            embed_dim=attention_dim,
            n_heads=n_heads
        )

        self.global_query = nn.Parameter(torch.randn(1, 1, attention_dim))

        self.fc = nn.Sequential(
            nn.Linear(attention_dim, latent_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=1.0,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.global_query, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B x H x W x 4
        """
        batch_size = x.shape[0]

        x = x.permute(0, 3, 1, 2)  # B x 4 x H x W

        h = self.conv(x)           # B x C x H' x W'

        b, c, h_sp, w_sp = h.shape
        tokens = h.view(b, c, h_sp * w_sp).transpose(1, 2)  # B x N x C

        q = self.global_query.expand(batch_size, -1, -1)    # B x 1 x C

        pooled, attn_weights = self.spatial_attention(
            query=q,
            key=tokens,
            value=tokens,
        )

        pooled = pooled.squeeze(1)  # B x C

        z = self.fc(pooled)

        return z
    
class TerrainTwoHeadDecoder(nn.Module):
    """
    Two-head terrain decoder.

    Decodes:
        z: B x latent_dim

    Into:
        recon: B x H x W x 4
            channels = [height_hat, normal_x_hat, normal_y_hat, normal_z_hat]

    This decoder keeps the compact size:
        latent_dim -> decoder_hidden_dim -> decoder_channels x h_enc x w_enc
        decoder_channels -> 64 -> 32 -> 16
        height head: 16 -> 1
        normal head: 16 -> 3
    """

    def __init__(
        self,
        height: int,
        width: int,
        latent_dim: int = 32,
        encoded_spatial_shape: tuple[int, int] = (3, 4),
        decoder_hidden_dim: int = 128,
        decoder_channels: int = 64,
        cnn_activation: str = "elu",
        norm_type: str = "none",
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        self.encoded_spatial_shape = encoded_spatial_shape
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_channels = decoder_channels
        self.norm_type = norm_type

        self.activation = get_activation(cnn_activation)

        h_enc, w_enc = encoded_spatial_shape

        self.decoder_spatial_shape = (
            decoder_channels,
            h_enc,
            w_enc,
        )

        decoder_flat_dim = decoder_channels * h_enc * w_enc

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden_dim),
            self.activation,
            nn.Linear(decoder_hidden_dim, decoder_flat_dim),
            self.activation,
        )

        self.decoder_block_1 = ConvNormAct(
            decoder_channels,
            64,
            activation=self.activation,
            norm_type=norm_type,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.decoder_block_2 = ConvNormAct(
            64,
            32,
            activation=self.activation,
            norm_type=norm_type,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.decoder_block_3 = ConvNormAct(
            32,
            16,
            activation=self.activation,
            norm_type=norm_type,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.height_head = nn.Conv2d(
            16,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.normal_head = nn.Conv2d(
            16,
            3,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize decoder layers.

        Conv/Linear:
            Kaiming uniform initialization.
        Biases:
            Zero initialization.
        Norm layers:
            Scale = 1, bias = 0.
        Final reconstruction heads:
            Conservative Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=1.0,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Conservative initialization for final reconstruction heads.
        nn.init.xavier_uniform_(self.height_head.weight, gain=0.5)
        nn.init.zeros_(self.height_head.bias)

        nn.init.xavier_uniform_(self.normal_head.weight, gain=0.5)
        nn.init.zeros_(self.normal_head.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: B x latent_dim

        returns:
            recon: B x H x W x 4
        """
        batch_size = z.shape[0]

        h = self.decoder_fc(z)
        h = h.view(batch_size, *self.decoder_spatial_shape)

        # H' x W' -> 2H' x 2W'
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.decoder_block_1(h)

        # -> 4H' x 4W'
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.decoder_block_2(h)

        # -> 8H' x 8W'
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.decoder_block_3(h)

        # Final resize handles non-power-of-two edge cases.
        h = F.interpolate(
            h,
            size=(self.height, self.width),
            mode="nearest",
        )

        height_hat = self.height_head(h)

        normal_raw = self.normal_head(h)
        normal_hat = F.normalize(
            normal_raw,
            p=2,
            dim=1,
            eps=1e-6,
        )

        recon = torch.cat([height_hat, normal_hat], dim=1)
        recon = recon.permute(0, 2, 3, 1)  # B x H x W x 4

        return recon


class PrviDynamicsMLPMixerKITE(nn.Module):
    """
    Deterministic privileged-dynamics context encoder using an MLP-Mixer architecture.

    This encoder compresses a history of privileged observation features into a
    compact latent state for critic conditioning during asymmetric actor-critic
    training.

    Intended default input:
        X_C: B x 393

    Interpreted as:
        X_C reshaped to B x 131 x 3

    where:
        131 = privileged observation feature tokens
        3   = short history / per-token feature channel dimension

    The architecture is:
        B x 131 x 3
            -> per-token history embedding
        B x 131 x hidden_dim
            -> MLP-Mixer blocks
        B x 131 x hidden_dim
            -> token pooling
        B x hidden_dim
            -> deterministic latent projection
        B x context_latent_size

    The mixer separates:
        token mixing:
            mixes information across privileged observation feature tokens

        channel mixing:
            mixes information within each token's learned hidden embedding

    Args:
        context_input_dim:
            Flattened privileged context dimension.
            Default: 393 = 131 tokens x 3 values per token.

        num_tokens:
            Number of privileged observation feature tokens.
            Default: 131.

        input_dim_per_token:
            Number of values associated with each privileged feature token.
            For the default setting, this is a short 3-step/history channel.
            Default: 3.

        hidden_dim:
            Per-token embedding dimension after the initial projection from
            input_dim_per_token to hidden_dim.

        num_mixer_blocks:
            Number of MLP-Mixer blocks.

        token_mlp_dim:
            Hidden dimension inside the token-mixing MLP, which mixes across
            the num_tokens privileged feature tokens.

        channel_mlp_dim:
            Hidden dimension inside the channel-mixing MLP, which mixes within
            each token's hidden_dim-dimensional representation.

        context_latent_size:
            Final deterministic latent size returned to the critic.

        activation:
            Activation passed through module_utils.get_activation(...).

        use_layer_norm:
            Whether to use LayerNorm inside the mixer blocks and after the
            mixer trunk.

        device:
            Kept for compatibility with the existing encoder API.
    """

    def __init__(
        self,
        context_input_dim: int = 393,
        num_tokens: int = 131,
        input_dim_per_token: int = 3,
        hidden_dim: int = 128,
        num_mixer_blocks: int = 2,
        token_mlp_dim: int = 128,
        channel_mlp_dim: int = 256,
        context_latent_size: int = 16,
        activation: str = "elu",
        use_layer_norm: bool = True,
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
        self.device = device

        self.activation = get_activation(activation)

        # ------------------------------------------------------------------
        # Per-token privileged-history embedding:
        #     B x 131 x 3 -> B x 131 x hidden_dim
        #
        # Each privileged observation feature token has a small associated
        # history/channel vector of length input_dim_per_token.
        # ------------------------------------------------------------------
        self.token_embedding = nn.Linear(input_dim_per_token, hidden_dim)

        # ------------------------------------------------------------------
        # MLP-Mixer trunk:
        #
        # Each block alternates between:
        #   1. token mixing across privileged observation features
        #   2. channel mixing inside each token embedding
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
        # Deterministic latent projection for critic conditioning.
        # ------------------------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 2 * context_latent_size),
            self.activation,
            nn.Linear(2 * context_latent_size, context_latent_size),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialization routine following the style used in the other KITE
        context encoders.

        Feature embedding and hidden latent layers:
            Kaiming uniform with leaky_relu gain approximation.

        Final latent projection:
            Kaiming uniform; final layer uses linear nonlinearity scaling.

        LayerNorm:
            weight = 1, bias = 0.
        """

        # Feature embedding.
        nn.init.kaiming_uniform_(
            self.token_embedding.weight,
            a=1.0,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        if self.token_embedding.bias is not None:
            nn.init.zeros_(self.token_embedding.bias)

        # Latent projection MLP.
        for i, module in enumerate(self.fc):
            if isinstance(module, nn.Linear):
                is_final_layer = i == len(self.fc) - 1

                nn.init.kaiming_uniform_(
                    module.weight,
                    a=1.0,
                    mode="fan_in",
                    nonlinearity="linear" if is_final_layer else "leaky_relu",
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
            X_C: B x 393
        or:
            X_C: B x 131 x 3

        Returns:
            X_C: B x 131 x 3
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
            f"Expected X_C shape Bx{self.context_input_dim} "
            f"or Bx{self.num_tokens}x{self.input_dim_per_token}, "
            f"but got {tuple(X_C.shape)}."
        )

    def encode(self, X_C: torch.Tensor) -> torch.Tensor:
        """
        Encodes privileged observation history into a deterministic latent.

        Args:
            X_C:
                B x 393 flattened privileged context
            or:
                B x 131 x 3 structured privileged context

        Returns:
            z:
                B x context_latent_size
        """

        x = self._format_input(X_C)

        # B x 131 x 3 -> B x 131 x hidden_dim
        x = self.activation(self.token_embedding(x))

        # Mixer trunk.
        for block in self.mixer_blocks:
            x = block(x)

        x = self.final_norm(x)

        # Global pooling over privileged feature tokens.
        # B x 131 x hidden_dim -> B x hidden_dim
        x = x.mean(dim=1)

        # Deterministic latent projection.
        z = self.fc(x)

        return z

    def forward(self, X_C: torch.Tensor) -> torch.Tensor:
        """
        Complete deterministic forward pass.

        Args:
            X_C:
                B x 393 flattened privileged context
            or:
                B x 131 x 3 structured privileged context

        Returns:
            z:
                B x context_latent_size
        """
        return self.encode(X_C)

class PrivDynamicsDecoder(nn.Module):
    """Decoder network for reconstructing next state from latent representation and velocity.
    
    Takes a latent vector and torso velocity as input, processes through two ELU-activated
    hidden layers, and outputs a predicted next state. Uses Xavier uniform initialization.
    """

    def __init__(
            self,
            input_dim: int = 16,
            layers: List[int] = [32,128,256,512],
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