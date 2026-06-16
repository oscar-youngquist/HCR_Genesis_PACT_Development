from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any

from .module_utils import (
    MLPMixerBlock,
    get_activation,
    make_1d_norm,
    make_2d_norm,
)


class MultimodalMixerVAE(nn.Module):
    """
    Variational two-token MLP-Mixer for fusing depth and proprioceptive latents.

    Inputs:
        z_depth_seq:
            B x depth_latent_dim

        z_proprio:
            B x proprio_latent_dim

    The two modality latents are first normalized independently with LayerNorm,
    then projected into a shared hidden dimension and stacked as two modality
    tokens:

        LayerNorm(z_depth_seq) -> depth projection   -> B x hidden_dim
        LayerNorm(z_proprio)   -> proprio projection -> B x hidden_dim

        tokens = [depth_token, proprio_token]

        B x 2 x hidden_dim

    The architecture is:

        B x 2 x hidden_dim
            -> MLP-Mixer block
            -> MLP-Mixer block
            -> modality-token pooling
            -> VAE-style mean/logvar heads

    Token mixing:
        mixes information between the depth and proprioceptive modalities.

    Channel mixing:
        mixes within each modality token's hidden representation.

    Output:
        mean:
            B x output_dim

        logvar:
            B x output_dim

        z:
            B x output_dim
    """

    def __init__(
        self,
        depth_latent_dim: int = 16,
        proprio_latent_dim: int = 16,
        hidden_dim: int = 64,
        num_mixer_blocks: int = 2,
        token_mlp_dim: int = 16,
        channel_mlp_dim: int = 128,
        output_dim: int = 16,
        velo_dim: int = 3,           # [v_x, v_y, v_x] - estimated torso velocity for current time-step
        feet_state_dim: int = 20,    # [feet_contact_state (4), feet_height (4), surface_norm_under_feet (4x3=12)]
        velo_hidden: int = 64,
        feet_hidden: int = 64,
        activation: str = "elu",
        use_layer_norm: bool = True,
        logvar_min: float = -5.0,
        logvar_max: float = 5.0,
        use_vae: bool = True,
    ) -> None:
        super().__init__()

        self.depth_latent_dim = depth_latent_dim
        self.proprio_latent_dim = proprio_latent_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = 2
        self.num_mixer_blocks = num_mixer_blocks
        self.token_mlp_dim = token_mlp_dim
        self.channel_mlp_dim = channel_mlp_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.use_layer_norm = use_layer_norm
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.use_vae = use_vae

        self.activation = get_activation(activation)

        # ------------------------------------------------------------------
        # Per-modality input normalization.
        #
        # These normalize each latent before projecting into the shared
        # multimodal token space. This helps prevent one modality from
        # dominating purely because its latent scale is larger.
        # ------------------------------------------------------------------
        self.depth_input_norm = nn.LayerNorm(depth_latent_dim)
        self.proprio_input_norm = nn.LayerNorm(proprio_latent_dim)

        # ------------------------------------------------------------------
        # Modality-specific projections into a shared token embedding space.
        # ------------------------------------------------------------------
        self.depth_projection = nn.Linear(depth_latent_dim, hidden_dim)
        self.proprio_projection = nn.Linear(proprio_latent_dim, hidden_dim)

        # Optional learned modality embeddings. These let the mixer distinguish
        # the depth token from the proprioceptive token even after projection.
        self.modality_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, hidden_dim)
        )

        # ------------------------------------------------------------------
        # Two-token MLP-Mixer trunk.
        # ------------------------------------------------------------------
        self.mixer_blocks = nn.ModuleList(
            [
                MLPMixerBlock(
                    num_tokens=self.num_tokens,
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
        # Fused latent trunk.
        # ------------------------------------------------------------------
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim, 2 * output_dim),
            self.activation,
        )

        self.latmean_h = nn.Linear(2 * output_dim, 2 * output_dim)
        self.latvar_h = nn.Linear(2 * output_dim, 2 * output_dim)

        self.out_mean = nn.Linear(2 * output_dim, output_dim)

        self.out_logvar = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.Hardtanh(min_val=logvar_min, max_val=logvar_max),
        )

        # ------------------------------------------------------------------
        # State estimation heads
        # ------------------------------------------------------------------
        self.velo_est_hidden = nn.Linear(output_dim, velo_hidden)
        self.velo_est_out = nn.Linear(velo_hidden, velo_dim)

        self.feet_est_hidden = nn.Linear(output_dim, feet_hidden)
        self.feet_est_out = nn.Linear(feet_hidden, feet_state_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialization routine following the same style as the other KITE
        encoder modules.

        Hidden feature layers:
            Kaiming uniform with leaky_relu gain approximation.

        Mean/logvar output heads:
            Kaiming uniform with linear nonlinearity.

        LayerNorm:
            weight = 1, bias = 0.

        Modality embedding:
            Small normal initialization.
        """

        # Modality input norms.
        for norm in [self.depth_input_norm, self.proprio_input_norm]:
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

        # Modality projections.
        for layer in [self.depth_projection, self.proprio_projection]:
            nn.init.kaiming_uniform_(
                layer.weight,
                a=1.0,
                mode="fan_in",
                nonlinearity="leaky_relu",
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Small modality identity embeddings.
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)

        # Fusion trunk.
        for module in self.fusion_fc:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=1.0,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Hidden VAE heads.
        for layer in [self.latmean_h, self.latvar_h]:
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
            self.out_mean.weight,
            a=1.0,
            mode="fan_in",
            nonlinearity="linear",
        )
        if self.out_mean.bias is not None:
            nn.init.zeros_(self.out_mean.bias)

        # Logvar output head.
        for module in self.out_logvar:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=1.0,
                    mode="fan_in",
                    nonlinearity="linear",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # All LayerNorm layers, including those inside mixer blocks.
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Auxiliary velocity-estimation head.
        nn.init.kaiming_uniform_(
            self.velo_est_hidden.weight,
            a=1.0,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        if self.velo_est_hidden.bias is not None:
            nn.init.zeros_(self.velo_est_hidden.bias)

        nn.init.kaiming_uniform_(
            self.velo_est_out.weight,
            a=1.0,
            mode="fan_in",
            nonlinearity="linear",
        )
        if self.velo_est_out.bias is not None:
            nn.init.zeros_(self.velo_est_out.bias)

        # Auxiliary feet-state-estimation head.
        nn.init.kaiming_uniform_(
            self.feet_est_hidden.weight,
            a=1.0,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        if self.feet_est_hidden.bias is not None:
            nn.init.zeros_(self.feet_est_hidden.bias)

        nn.init.kaiming_uniform_(
            self.feet_est_out.weight,
            a=1.0,
            mode="fan_in",
            nonlinearity="linear",
        )
        if self.feet_est_out.bias is not None:
            nn.init.zeros_(self.feet_est_out.bias)

    def _check_inputs(
        self,
        z_depth_seq: torch.Tensor,
        z_proprio: torch.Tensor,
    ) -> None:
        if z_depth_seq.dim() != 2:
            raise ValueError(
                f"Expected z_depth_seq shape B x {self.depth_latent_dim}, "
                f"but got {tuple(z_depth_seq.shape)}."
            )

        if z_proprio.dim() != 2:
            raise ValueError(
                f"Expected z_proprio shape B x {self.proprio_latent_dim}, "
                f"but got {tuple(z_proprio.shape)}."
            )

        if z_depth_seq.shape[0] != z_proprio.shape[0]:
            raise ValueError(
                f"Batch size mismatch: z_depth_seq has batch {z_depth_seq.shape[0]}, "
                f"but z_proprio has batch {z_proprio.shape[0]}."
            )

        if z_depth_seq.shape[1] != self.depth_latent_dim:
            raise ValueError(
                f"Expected depth_latent_dim={self.depth_latent_dim}, "
                f"but got {z_depth_seq.shape[1]}."
            )

        if z_proprio.shape[1] != self.proprio_latent_dim:
            raise ValueError(
                f"Expected proprio_latent_dim={self.proprio_latent_dim}, "
                f"but got {z_proprio.shape[1]}."
            )

    def encode(
        self,
        z_depth_seq: torch.Tensor,
        z_proprio: torch.Tensor,
    ):
        """
        Encodes depth and proprioceptive latents into a fused latent distribution.

        Args:
            z_depth_seq:
                B x depth_latent_dim

            z_proprio:
                B x proprio_latent_dim

        Returns:
            mean:
                B x output_dim

            logvar:
                B x output_dim
        """

        self._check_inputs(z_depth_seq, z_proprio)

        # Normalize each modality latent before shared-space projection.
        z_depth_seq = self.depth_input_norm(z_depth_seq)
        z_proprio = self.proprio_input_norm(z_proprio)

        depth_token = self.activation(self.depth_projection(z_depth_seq))
        proprio_token = self.activation(self.proprio_projection(z_proprio))

        # B x hidden_dim -> B x 1 x hidden_dim
        depth_token = depth_token.unsqueeze(1)
        proprio_token = proprio_token.unsqueeze(1)

        # B x 2 x hidden_dim
        x = torch.cat([depth_token, proprio_token], dim=1)

        # Add learned modality identity embeddings.
        x = x + self.modality_embedding

        # Mixer trunk.
        for block in self.mixer_blocks:
            x = block(x)

        x = self.final_norm(x)

        # Pool across the two modality tokens:
        # B x 2 x hidden_dim -> B x hidden_dim
        x = x.mean(dim=1)

        x = self.fusion_fc(x)

        lat_mean = self.activation(self.latmean_h(x))
        lat_var = self.activation(self.latvar_h(x))

        mean = self.out_mean(lat_mean)
        logvar = self.out_logvar(lat_var)

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
                B x output_dim

            logvar:
                B x output_dim

        Returns:
            z:
                B x output_dim
        """
        epsilon = torch.randn_like(logvar).to(logvar.device)
        return mean + torch.exp(0.5 * logvar) * epsilon

    def forward(
        self,
        z_depth_seq: torch.Tensor,
        z_proprio: torch.Tensor,
    ):
        """
        Complete variational forward pass.

        Args:
            z_depth_seq:
                B x depth_latent_dim

            z_proprio:
                B x proprio_latent_dim

        Returns:
            mean:
                B x output_dim

            logvar:
                B x output_dim

            z:
                B x output_dim
        """

        mean, logvar = self.encode(z_depth_seq, z_proprio)

        if self.use_vae and self.training:
            z = self.reparameterization_trick(mean, logvar)
        else:
            z = mean

        body_velo_est = self.velo_est_out(self.activation(self.velo_est_hidden(z)))
        feet_state_est = self.feet_est_out(self.activation(self.feet_est_hidden(z)))

        return mean, logvar, z, body_velo_est, feet_state_est

    @torch.no_grad()
    def forward_inference(
        self,
        z_depth_seq: torch.Tensor,
        z_proprio: torch.Tensor,
    ):
        """
        Deterministic inference path.

        Args:
            z_depth_seq:
                B x depth_latent_dim

            z_proprio:
                B x proprio_latent_dim

        Returns:
            mean:
                B x output_dim
        """
        mean, _ = self.encode(z_depth_seq, z_proprio)
        
        body_velo_est = self.velo_est_out(self.activation(self.velo_est_hidden(mean)))
        feet_state_est = self.feet_est_out(self.activation(self.feet_est_hidden(mean)))
        
        return mean, body_velo_est, feet_state_est
