from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from module_utils import get_activation, EfficientMultiHeadAttention


# --------------------------------------------------------------------------
# Helper blocks
# --------------------------------------------------------------------------

def make_2d_norm(norm_type: str, num_channels: int) -> nn.Module:
    """
    Returns normalization layer for 2D conv features.

    norm_type:
        "none"  -> Identity
        "batch" -> BatchNorm2d
        "group" -> GroupNorm
    """
    norm_type = norm_type.lower()

    if norm_type == "none":
        return nn.Identity()

    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)

    if norm_type == "group":
        num_groups = min(8, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_channels)

    raise ValueError(
        f"Unknown norm_type={norm_type}. Expected one of: 'none', 'batch', 'group'."
    )


class ConvNormAct(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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

    @torch.no_grad()
    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
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

