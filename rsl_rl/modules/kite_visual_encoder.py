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
    Returns a normalization layer for 2D conv features.

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


def make_1d_norm(norm_type: str, num_channels: int) -> nn.Module:
    """
    Returns a normalization layer for 1D conv features.

    norm_type:
        "none"  -> Identity
        "batch" -> BatchNorm1d
        "group" -> GroupNorm
    """
    norm_type = norm_type.lower()

    if norm_type == "none":
        return nn.Identity()

    if norm_type == "batch":
        return nn.BatchNorm1d(num_channels)

    if norm_type == "group":
        num_groups = min(8, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_channels)

    raise ValueError(
        f"Unknown norm_type={norm_type}. Expected one of: 'none', 'batch', 'group'."
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


class Conv1dNormAct(nn.Module):
    """
    1D convolution block with configurable normalization.

    Default:
        Conv1d -> Identity -> activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module,
        norm_type: str = "none",
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            make_1d_norm(norm_type, out_channels),
            activation,
        )

    def forward(self, x):
        return self.block(x)


def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick.

    mean:   B x latent_dim
    logvar: B x latent_dim
    """
    eps = torch.randn_like(mean)
    return mean + torch.exp(0.5 * logvar) * eps


# --------------------------------------------------------------------------
# Motion-robust depth image encoder
# --------------------------------------------------------------------------

class MotionRobustDepthEncoder(nn.Module):
    """
    Encodes a single depth image into a latent vector.

    Inputs:
        depth_image: B x 1 x H x W
        torso_state: B x 8
            [roll, pitch, v_x, v_y, v_z, gyro_x, gyro_y, gyro_z]

    Outputs:
        mean:   B x latent_dim
        logvar: B x latent_dim
        z:      B x latent_dim
        aux:    dict
    """

    def __init__(
        self,
        depth_image_resolution=(48, 64),
        cnn_input_channel: int = 1,
        target_latent_dim: int = 16,
        cnn_activation: str = "elu",
        norm_type: str = "none",
        use_vae: bool = True,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        self.depth_image_resolution = depth_image_resolution
        self.target_latent_dim = target_latent_dim
        self.norm_type = norm_type
        self.use_vae = use_vae

        in_height, in_width = depth_image_resolution
        self.activation = get_activation(cnn_activation)

        y_grid = torch.linspace(-1, 1, in_height).view(1, 1, in_height, 1)
        y_grid = y_grid.repeat(1, 1, 1, in_width)

        x_grid = torch.linspace(-1, 1, in_width).view(1, 1, 1, in_width)
        x_grid = x_grid.repeat(1, 1, in_height, 1)

        self.register_buffer(
            "base_coord_grid",
            torch.cat([x_grid, y_grid], dim=1),
        )  # 1 x 2 x H x W

        self.imu_projector = nn.Sequential(
            nn.Linear(8, 16),
            self.activation,
            nn.Linear(16, 4),
        )

        self.conv1 = ConvNormAct(
            cnn_input_channel + 2,
            8,
            self.activation,
            norm_type=norm_type,
            stride=2,
        )
        self.conv2 = ConvNormAct(
            8,
            16,
            self.activation,
            norm_type=norm_type,
            stride=2,
        )
        self.conv3 = ConvNormAct(
            16,
            32,
            self.activation,
            norm_type=norm_type,
            stride=2,
        )

        self.spatial_attention = EfficientMultiHeadAttention(
            embed_dim=32,
            n_heads=4,
            dropout=attention_dropout,
        )

        self.global_query = nn.Parameter(torch.randn(1, 1, 32))

        self.fc = nn.Sequential(
            nn.Linear(32, target_latent_dim),
            self.activation,
        )

        self.mean_out = nn.Linear(target_latent_dim, target_latent_dim)

        self.logvar_out = nn.Sequential(
            nn.Linear(target_latent_dim, target_latent_dim),
            nn.Hardtanh(min_val=-5.0, max_val=5.0),
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

        # Initialize IMU projector to identity transform.
        final_imu_layer = self.imu_projector[-1]
        nn.init.zeros_(final_imu_layer.weight)

        with torch.no_grad():
            final_imu_layer.bias.copy_(
                torch.tensor(
                    [1.0, 0.0, 0.0, 1.0],
                    dtype=final_imu_layer.bias.dtype,
                    device=final_imu_layer.bias.device,
                )
            )

        for proj_layer in [
            self.spatial_attention.W_q,
            self.spatial_attention.W_k,
            self.spatial_attention.W_v,
            self.spatial_attention.projection,
        ]:
            nn.init.kaiming_uniform_(
                proj_layer.weight,
                a=1.0,
                mode="fan_in",
                nonlinearity="linear",
            )
            if proj_layer.bias is not None:
                nn.init.zeros_(proj_layer.bias)

    
    def compute_motion_conditioned_coords(
        self,
        depth_image: torch.Tensor,
        torso_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = depth_image.size(0)
        height = depth_image.size(2)
        width = depth_image.size(3)

        transform_matrices = self.imu_projector(torso_state).view(batch_size, 2, 2)

        flat_grid = self.base_coord_grid.expand(batch_size, -1, -1, -1)
        flat_grid = flat_grid.view(batch_size, 2, -1)

        stabilized_flat_grid = torch.bmm(transform_matrices, flat_grid)
        stabilized_coords = stabilized_flat_grid.view(batch_size, 2, height, width)

        return stabilized_coords, transform_matrices

    
    def encode(self, depth_image: torch.Tensor, torso_state: torch.Tensor):
        batch_size = depth_image.size(0)

        stabilized_coords, transform_matrices = self.compute_motion_conditioned_coords(
            depth_image,
            torso_state,
        )

        x = torch.cat([depth_image, stabilized_coords], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        b, c, h, w = x.shape
        spatial_sequence = x.view(b, c, h * w).transpose(1, 2)

        q = self.global_query.expand(batch_size, -1, -1)

        attn_out, attn_weights = self.spatial_attention(
            query=q,
            key=spatial_sequence,
            value=spatial_sequence,
        )

        attn_out = attn_out.squeeze(1)
        latent = self.fc(attn_out)

        mean = self.mean_out(latent)
        logvar = self.logvar_out(latent)

        aux = {
            "attention_weights": attn_weights,
            "transform_matrices": transform_matrices,
        }

        return mean, logvar, aux

    def forward(self, depth_image: torch.Tensor, torso_state: torch.Tensor):
        mean, logvar, aux = self.encode(depth_image, torso_state)

        if self.use_vae and self.training:
            z = reparameterize(mean, logvar)
        else:
            z = mean

        return mean, logvar, z, aux

    @torch.no_grad()
    def forward_inference(self, depth_image: torch.Tensor, torso_state: torch.Tensor):
        mean, _, _ = self.encode(depth_image, torso_state)
        return mean


# --------------------------------------------------------------------------
# Motion-robust depth decoder
# --------------------------------------------------------------------------

class MotionRobustDepthDecoder(nn.Module):
    """
    Decodes a latent vector back into a reconstructed depth image.

    Uses interpolation + Conv2d blocks rather than ConvTranspose2d.
    """

    def __init__(
        self,
        depth_image_resolution=(48, 64),
        cnn_output_channel: int = 1,
        target_latent_dim: int = 16,
        cnn_activation: str = "elu",
        norm_type: str = "none",
    ):
        super().__init__()

        self.depth_image_resolution = depth_image_resolution
        self.norm_type = norm_type

        in_height, in_width = depth_image_resolution
        self.activation = get_activation(cnn_activation)

        y_grid = torch.linspace(-1, 1, in_height).view(1, 1, in_height, 1)
        y_grid = y_grid.repeat(1, 1, 1, in_width)

        x_grid = torch.linspace(-1, 1, in_width).view(1, 1, 1, in_width)
        x_grid = x_grid.repeat(1, 1, in_height, 1)

        self.register_buffer(
            "base_coord_grid",
            torch.cat([x_grid, y_grid], dim=1),
        )

        self.imu_projector = nn.Sequential(
            nn.Linear(8, 16),
            self.activation,
            nn.Linear(16, 4),
        )

        self.fc = nn.Sequential(
            nn.Linear(target_latent_dim, 32 * 6 * 8),
            self.activation,
        )

        self.conv_6_8 = ConvNormAct(
            32 + 2,
            32,
            self.activation,
            norm_type=norm_type,
            stride=1,
        )
        self.conv_12_16 = ConvNormAct(
            32 + 2,
            16,
            self.activation,
            norm_type=norm_type,
            stride=1,
        )
        self.conv_24_32 = ConvNormAct(
            16 + 2,
            8,
            self.activation,
            norm_type=norm_type,
            stride=1,
        )

        self.final_head = nn.Conv2d(
            8 + 2,
            cnn_output_channel,
            kernel_size=3,
            stride=1,
            padding=1,
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

        final_imu_layer = self.imu_projector[-1]
        nn.init.zeros_(final_imu_layer.weight)

        with torch.no_grad():
            final_imu_layer.bias.copy_(
                torch.tensor(
                    [1.0, 0.0, 0.0, 1.0],
                    dtype=final_imu_layer.bias.dtype,
                    device=final_imu_layer.bias.device,
                )
            )

    def compute_motion_conditioned_coords(
        self,
        batch_size: int,
        torso_state: torch.Tensor,
    ):
        height, width = self.depth_image_resolution

        transform_matrices = self.imu_projector(torso_state).view(batch_size, 2, 2)

        flat_grid = self.base_coord_grid.expand(batch_size, -1, -1, -1)
        flat_grid = flat_grid.view(batch_size, 2, -1)

        stabilized_flat_grid = torch.bmm(transform_matrices, flat_grid)
        coords_full = stabilized_flat_grid.view(batch_size, 2, height, width)

        coords_6_8 = F.interpolate(
            coords_full,
            size=(6, 8),
            mode="bilinear",
            align_corners=False,
        )
        coords_12_16 = F.interpolate(
            coords_full,
            size=(12, 16),
            mode="bilinear",
            align_corners=False,
        )
        coords_24_32 = F.interpolate(
            coords_full,
            size=(24, 32),
            mode="bilinear",
            align_corners=False,
        )

        return {
            "coords_6_8": coords_6_8,
            "coords_12_16": coords_12_16,
            "coords_24_32": coords_24_32,
            "coords_full": coords_full,
            "transform_matrices": transform_matrices,
        }

    def forward(self, z: torch.Tensor, torso_state: torch.Tensor):
        batch_size = z.size(0)

        coord_dict = self.compute_motion_conditioned_coords(batch_size, torso_state)

        x = self.fc(z)
        x = x.view(batch_size, 32, 6, 8)

        x = torch.cat([x, coord_dict["coords_6_8"]], dim=1)
        x = self.conv_6_8(x)

        x = F.interpolate(x, size=(12, 16), mode="nearest")
        x = torch.cat([x, coord_dict["coords_12_16"]], dim=1)
        x = self.conv_12_16(x)

        x = F.interpolate(x, size=(24, 32), mode="nearest")
        x = torch.cat([x, coord_dict["coords_24_32"]], dim=1)
        x = self.conv_24_32(x)

        height, width = self.depth_image_resolution
        x = F.interpolate(x, size=(height, width), mode="nearest")
        x = torch.cat([x, coord_dict["coords_full"]], dim=1)

        reconstructed_depth_image = self.final_head(x)

        return reconstructed_depth_image, {
            "transform_matrices": coord_dict["transform_matrices"],
        }


# --------------------------------------------------------------------------
# Depth sequence encoder
# --------------------------------------------------------------------------

class DepthSequenceEncoder(nn.Module):
    """
    Encodes a short sequence of per-frame depth latents.

    Input:
        x: B x T x feature_dim

    Output:
        mean:   B x output_dim
        logvar: B x output_dim
        z:      B x output_dim
    """

    def __init__(
        self,
        feature_dim: int = 16,
        sequence_length: int = 5,
        output_dim: int = 16,
        activation: str = "elu",
        norm_type: str = "none",
        use_vae: bool = True,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.norm_type = norm_type
        self.use_vae = use_vae

        self.activation = get_activation(activation)

        self.convs = nn.ModuleList()

        current_len = sequence_length
        current_channels = feature_dim
        max_channels = feature_dim * 4

        while current_len > 1:
            kernel_size = 3 if current_len >= 3 else 2
            next_channels = min(current_channels * 2, max_channels)

            self.convs.append(
                Conv1dNormAct(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    activation=self.activation,
                    norm_type=norm_type,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=0,
                )
            )

            current_len = current_len - (kernel_size - 1)
            current_channels = next_channels

        assert current_len == 1, "Temporal conv stack should reduce sequence length to 1."

        self.fc = nn.Sequential(
            nn.Linear(current_channels, output_dim),
            self.activation,
        )

        self.mean_out = nn.Linear(output_dim, output_dim)

        self.logvar_out = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Hardtanh(min_val=-5.0, max_val=5.0),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=1.0,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor):
        """
        x: B x T x feature_dim
        """
        x = x.permute(0, 2, 1)

        for conv in self.convs:
            x = conv(x)

        x = x.flatten(start_dim=1)

        latent = self.fc(x)

        mean = self.mean_out(latent)
        logvar = self.logvar_out(latent)

        return mean, logvar

    def forward(self, x: torch.Tensor):
        mean, logvar = self.encode(x)

        if self.use_vae and self.training:
            z = reparameterize(mean, logvar)
        else:
            z = mean

        return mean, logvar, z

    @torch.no_grad()
    def forward_inference(self, x: torch.Tensor):
        mean, _ = self.encode(x)
        return mean


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