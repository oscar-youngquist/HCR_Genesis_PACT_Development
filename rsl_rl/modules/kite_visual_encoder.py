from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_utils import (
    ConvNormAct,
    Conv1dNormAct,
    make_1d_norm,
    EfficientMultiHeadAttention,
    get_activation,
    SmoothClampLayer,
)

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
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        self.depth_image_resolution = depth_image_resolution
        self.target_latent_dim = target_latent_dim
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
        )  # 1 x 2 x H x W

        self.imu_projector = nn.Sequential(
            nn.Linear(8, 16),
            self.activation,
            nn.Linear(16, 4),
        )

        kernel_x = torch.tensor(
            [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0

        kernel_y = torch.tensor(
            [[-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0

        self.register_buffer("depth_grad_kernel_x", kernel_x)
        self.register_buffer("depth_grad_kernel_y", kernel_y)

        self.conv1 = ConvNormAct(
            cnn_input_channel + 3 + 2,    # depth image, depth-image graident info, CoordConv grid
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

        # Attention pooling returns B x C, not a Conv1d-style B x C x L
        # tensor. Use ordinary LayerNorm for that vector case while preserving
        # the existing BatchNorm/GroupNorm behavior for other norm choices.
        self.att_norm = (
            nn.LayerNorm(32)
            if norm_type.lower() == "layer"
            else make_1d_norm(norm_type=norm_type, num_channels=32)
        )

        output_hdim = 2 * target_latent_dim

        self.fc = nn.Sequential(
            nn.Linear(32, output_hdim),
            self.activation,
            nn.Linear(output_hdim, output_hdim),
            self.activation,
        )        
        
        self.latent_h1 = nn.Linear(output_hdim, output_hdim)
        self.latent_out = nn.Linear(output_hdim, target_latent_dim)

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

        nn.init.normal_(self.latent_out.weight, mean=0.0, std=1.0e-3)
        if self.latent_out.bias is not None:
            nn.init.zeros_(self.latent_out.bias)

        # LayerNorm layers.
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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


    def compute_depth_gradients(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        depth_image: B x 1 x H x W

        Returns:
            depth_grads: B x 2 x H x W
                channels are [grad_x, grad_y]
        """
        depth_padded = F.pad(depth_image, (1, 1, 1, 1), mode="replicate")

        grad_x = F.conv2d(depth_padded, self.depth_grad_kernel_x)
        grad_y = F.conv2d(depth_padded, self.depth_grad_kernel_y)

        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-6)
    
        return torch.cat([grad_x, grad_y, grad_mag], dim=1)
    
    def encode(
        self,
        depth_image: torch.Tensor,
        torso_state: torch.Tensor,
    ):
        batch_size = depth_image.size(0)

        stabilized_coords, transform_matrices = self.compute_motion_conditioned_coords(
            depth_image,
            torso_state,
        )

        # x = torch.cat([depth_image, stabilized_coords], dim=1)

        depth_grads = self.compute_depth_gradients(depth_image)

        x = torch.cat([depth_image, depth_grads, stabilized_coords], dim=1)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        b, c, h, w = x3.shape
        spatial_sequence = x3.view(b, c, h * w).transpose(1, 2)

        q = self.global_query.expand(batch_size, -1, -1)

        attn_out, attn_weights = self.spatial_attention(
            query=q,
            key=spatial_sequence,
            value=spatial_sequence,
        )

        attn_out = self.att_norm(attn_out.squeeze(1))

        latent = self.fc(attn_out)

        latent_h1 = self.activation(self.latent_h1(latent))
        z = self.latent_out(latent_h1)

        aux = {
            "attention_weights": attn_weights,
            "transform_matrices": transform_matrices,
        }

        return z, aux
    

    def encode_inf(self, depth_image: torch.Tensor, torso_state: torch.Tensor):
        batch_size = depth_image.size(0)

        stabilized_coords, _ = self.compute_motion_conditioned_coords(
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

        attn_out, _ = self.spatial_attention(
            query=q,
            key=spatial_sequence,
            value=spatial_sequence,
        )

        attn_out = self.att_norm(attn_out.squeeze(1))
        latent = self.fc(attn_out)

        latent_h1 = self.activation(self.latent_h1(latent))
        z = self.latent_out(latent_h1)

        return z

    def forward(
        self,
        depth_image: torch.Tensor,
        torso_state: torch.Tensor,
    ):
        z, aux = self.encode(
            depth_image,
            torso_state,
        )

        return z, aux

    @torch.no_grad()
    def forward_inference(self, depth_image: torch.Tensor, torso_state: torch.Tensor):
        return self.encode_inf(depth_image, torso_state)


# --------------------------------------------------------------------------
# Depth sequence encoder
# --------------------------------------------------------------------------
class ConvDepthSequenceEncoder(nn.Module):
    """
    Encodes a short sequence of per-frame depth latents.

    Input:
        x: B x T x feature_dim

    The temporal Conv1D branch compresses the full latent history into a sequence-level
    representation. Optionally, a weighted residual skip connection projects the most
    recent depth latent x[:, -1, :] directly into the output latent space and adds it
    to the temporal representation:

        latent = temporal_latent + latest_skip_scale * latest_skip(x[:, -1, :])

    This skip connection helps preserve the most recent perception information while
    allowing the Conv1D branch to learn temporal smoothing, motion trends, and history-
    based corrections. The learnable scalar latest_skip_scale controls how strongly the
    current-frame latent contributes to the final sequence representation.

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
        std_min: float = 0.01,
        std_max: float = 1.5,
        conf_min: float = 0.1,
        conf_mask_scale: float = 0.2,
        use_latest_skip: bool = True,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.norm_type = norm_type
        self.use_latest_skip = use_latest_skip

        self.conf_min = conf_min
        self.conf_mask_scale = conf_mask_scale

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

        output_hdim = 2 * output_dim

        self.fc = nn.Sequential(
            nn.Linear(current_channels, output_hdim),
            self.activation,
            nn.Linear(output_hdim, output_hdim),
            self.activation,
        )

        self.latest_skip = nn.Linear(feature_dim, output_hdim)
        self.latest_skip_scale = nn.Parameter(torch.ones(output_hdim)* 0.1)

        self.skip_norm = nn.LayerNorm(output_hdim)
        
        self.mean_h1 = nn.Linear(output_hdim, output_hdim)
        self.logvar_h1 = nn.Linear(output_hdim, output_hdim)

        self.mean_out = nn.Linear(output_hdim, output_dim)

        self.logvar_out = nn.Sequential(
            nn.Linear(output_hdim, output_dim),
            SmoothClampLayer(min_val=2.0*math.log(std_min), max_val=2.0*math.log(std_max)),
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

        # Initialize near zero so the initial posterior mean is close to the
        # N(0, I) prior. This keeps the initial KL_mu term small.
        nn.init.normal_(self.mean_out.weight, mean=0.0, std=1.0e-3)
        if self.mean_out.bias is not None:
            nn.init.zeros_(self.mean_out.bias)


        # Important: because SmoothClampLayer uses a sigmoid bound, logvar = 0
        # may be the upper asymptote when std_max == 1.0, so we target a small
        # negative value instead.
        smooth_bound = self.logvar_out[1]
        min_logvar = smooth_bound.min_val
        max_logvar = smooth_bound.max_val

        target_logvar = -0.05  # std ≈ exp(-0.025) ≈ 0.975, KL near zero

        p = (target_logvar - min_logvar) / (max_logvar - min_logvar)
        p = min(max(p, 1.0e-6), 1.0 - 1.0e-6)
        init_raw_logvar_bias = math.log(p / (1.0 - p))

        # Make the hidden logvar branch initially neutral.
        nn.init.zeros_(self.logvar_h1.weight)
        if self.logvar_h1.bias is not None:
            nn.init.zeros_(self.logvar_h1.bias)

        # Make the final raw-logvar head output a constant near unit std.
        nn.init.zeros_(self.logvar_out[0].weight)
        if self.logvar_out[0].bias is not None:
            nn.init.constant_(self.logvar_out[0].bias, init_raw_logvar_bias)

        # Optional: initialize the skip more conservatively so it does not
        # dominate the temporal branch at the start of training.
        nn.init.xavier_uniform_(self.latest_skip.weight, gain=0.5)
        if self.latest_skip.bias is not None:
            nn.init.zeros_(self.latest_skip.bias)

    def encode(self, x: torch.Tensor):
        """
        x: B x T x feature_dim
        """
        depth_std = x.std(dim=1, unbiased=False)  # B x feature_dim
        depth__mean = x.mean(dim=1)                    # B x D

        latest_depth_latent = x[:, -1, :]  # B x feature_dim

        # Filter-out/down-weight temporal outliers from the most recent latent
        latest_deviation = torch.abs(latest_depth_latent - depth__mean) / (depth_std + 1e-4)
        conf = 1.0 - torch.tanh(latest_deviation)
        conf = torch.clamp(conf, max=1.0, min=0.01)

        h = x.permute(0, 2, 1)  # B x feature_dim x T

        for conv in self.convs:
            h = conv(h)

        h = h.flatten(start_dim=1)

        temporal_latent = self.fc(h)
        
        latest_latent = self.latest_skip(latest_depth_latent * conf)
        latent = temporal_latent + self.latest_skip_scale * latest_latent
        latent = self.skip_norm(latent)
        latent = self.activation(latent)
        
        mean_h1 = self.activation(self.mean_h1(latent))
        logvar_h1 = self.activation(self.logvar_h1(latent))
        
        mean = self.mean_out(mean_h1)
        logvar = self.logvar_out(logvar_h1)

        return mean, logvar

    def forward(self, x: torch.Tensor):
        mean, logvar = self.encode(x)

        z = reparameterize(mean, logvar)

        return mean, logvar, z

    @torch.no_grad()
    def forward_inference(self, x: torch.Tensor):
        mean, _ = self.encode(x)
        return mean
