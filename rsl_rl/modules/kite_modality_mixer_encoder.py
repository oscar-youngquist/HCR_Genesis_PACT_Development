from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any
import math

from .module_utils import (
    get_activation,
    SmoothClampLayer,
    init_weights
)

# class MultimodalFusionVAE(nn.Module):
#     """
#     Variational gated-fusion module for fusing depth and proprioceptive latents.

#     Inputs:
#         z_depth_seq:
#             B x depth_latent_dim

#         z_proprio:
#             B x proprio_latent_dim

#     The two modality latents are first normalized independently with LayerNorm,
#     then projected into a shared hidden dimension:

#         LayerNorm(z_depth_seq) -> depth projection   -> B x hidden_dim
#         LayerNorm(z_proprio)   -> proprio projection -> B x hidden_dim

#     A learned feature-wise gate determines how much to use depth vs proprio:

#         gate = sigmoid(W [depth_token, proprio_token])

#         fused = gate * depth_token + (1 - gate) * proprio_token

#     Output:
#         mean:
#             B x output_dim

#         logvar:
#             B x output_dim

#         z:
#             B x output_dim
#     """

#     def __init__(
#         self,
#         depth_latent_dim: int = 32,
#         proprio_latent_dim: int = 16,
#         hidden_dims = [128, 64],
#         output_dim: int = 16,
#         velo_dim: int = 3,
#         feet_state_dim: int = 20,
#         velo_hidden: int = 32,
#         feet_hidden: int = 32,
#         activation: str = "elu",
#         use_layer_norm: bool = True,
#         std_min: float = 0.01,
#         std_max: float = 1.50,
#     ) -> None:
#         super().__init__()

#         self.depth_latent_dim = depth_latent_dim
#         self.proprio_latent_dim = proprio_latent_dim
#         self.num_tokens = 2
#         self.output_dim = output_dim
#         self.activation_name = activation
#         self.use_layer_norm = use_layer_norm

#         self.activation = get_activation(activation)

#         # ------------------------------------------------------------------
#         # Per-modality input normalization.
#         # ------------------------------------------------------------------
#         self.depth_input_norm = nn.LayerNorm(depth_latent_dim)
#         self.proprio_input_norm = nn.LayerNorm(proprio_latent_dim)

#         # ------------------------------------------------------------------
#         # Modality-specific projections into a shared embedding space.
#         # ------------------------------------------------------------------
#         embed_dim = int(hidden_dims[0] / 2)
#         self.depth_projection = nn.Linear(depth_latent_dim, embed_dim)
#         self.proprio_projection = nn.Linear(proprio_latent_dim, embed_dim)

#         # ------------------------------------------------------------------
#         # Modality-fusion layer.
#         # ------------------------------------------------------------------
#         self.merge_fc = nn.Linear(2 * embed_dim, hidden_dims[0])

#         # ------------------------------------------------------------------
#         # Fused latent trunk.
#         # ------------------------------------------------------------------
#         fusion_layers = []

#         for l in range(len(hidden_dims)-1):
#             fusion_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l+1]))
#             fusion_layers.append(self.activation)

#         fusion_layers.append(nn.Linear(hidden_dims[-1], 2 * output_dim))
#         fusion_layers.append(self.activation)

#         self.fusion_fc  = nn.Sequential(*fusion_layers)

#         self.latmean_h = nn.Linear(2 * output_dim, 2 * output_dim)
#         self.latvar_h = nn.Linear(2 * output_dim, 2 * output_dim)

#         self.out_mean = nn.Linear(2 * output_dim, output_dim)

#         self.out_logvar = nn.Sequential(
#             nn.Linear(2 * output_dim, output_dim),
#             # SmoothClampLayer(
#             #     min_val=2.0 * math.log(std_min),
#             #     max_val=2.0 * math.log(std_max),
#             # ),
#             SmoothClampLayer(
#                 min_val=-5.0,
#                 max_val=5.0,
#             ),
#         )

#         # ------------------------------------------------------------------
#         # State estimation heads.
#         # ------------------------------------------------------------------
#         self.velo_est_hidden = nn.Linear(output_dim, velo_hidden)
#         self.velo_est_out = nn.Linear(velo_hidden, velo_dim)

#         self.feet_est_hidden = nn.Linear(output_dim, feet_hidden)
#         # self.feet_est_out = nn.Linear(feet_hidden, feet_state_dim)

#         self.feet_contact_est = nn.Linear(feet_hidden, 4)
#         self.feet_height_est  = nn.Linear(feet_hidden, 4)
#         self.feet_normal_est  = nn.Linear(feet_hidden, 12)

#         self._initialize_weights()

#     def _initialize_weights(self) -> None:
#         """
#         Initialization routine following the same style as the other KITE
#         encoder modules.

#         Hidden feature layers:
#             Kaiming uniform with leaky_relu gain approximation.

#         Mean/logvar output heads:
#             Mean initialized near zero.
#             Logvar initialized near zero, corresponding to std near one.

#         LayerNorm:
#             weight = 1, bias = 0.

#         Modality embedding:
#             Small normal initialization.

#         Gating:
#             Initialized to produce gate < 0.5, so the initial fusion is
#             slightly weight towards proprioception during early training.
#         """
#         # Modality projections.
#         for layer in [self.depth_projection, self.proprio_projection]:
#             # nn.init.kaiming_uniform_(
#             #     layer.weight,
#             #     a=1.0,
#             #     mode="fan_in",
#             #     nonlinearity="leaky_relu",
#             # )
#             # if layer.bias is not None:
#             #     nn.init.zeros_(layer.bias)
#             nn.init.xavier_uniform_(layer.weight)

#             if layer.bias is not None:
#                 nn.init.zeros_(layer.bias)

#         # Fusion trunk.
#         for layer in self.fusion_fc:
#             if isinstance(layer, nn.Linear):
#                 # nn.init.kaiming_uniform_(
#                 #     module.weight,
#                 #     a=1.0,
#                 #     mode="fan_in",
#                 #     nonlinearity="leaky_relu",
#                 # )
#                 # if module.bias is not None:
#                 #     nn.init.zeros_(module.bias)
#                 nn.init.xavier_uniform_(layer.weight)

#                 if layer.bias is not None:
#                     nn.init.zeros_(layer.bias)

#         # Hidden VAE heads.
#         for layer in [self.latmean_h, self.latvar_h, self.merge_fc, self.out_mean]:
#             nn.init.kaiming_uniform_(
#                 layer.weight,
#                 a=1.0,
#                 mode="fan_in",
#                 nonlinearity="leaky_relu",
#             )
#             if layer.bias is not None:
#                 nn.init.zeros_(layer.bias)

#         self.out_logvar.apply(init_weights)

#         # # Initialize near zero so the initial posterior mean is close to the
#         # # N(0, I) prior. This keeps the initial KL_mu term small.
#         # nn.init.normal_(self.out_mean.weight, mean=0.0, std=1.0e-3)
#         # if self.out_mean.bias is not None:
#         #     nn.init.zeros_(self.out_mean.bias)

#         # # Initialize the variance branch to output near-zero logvar, i.e.
#         # # std ≈ 1, so the initial KL contribution is near zero.
#         # smooth_bound = self.out_logvar[1]
#         # min_logvar = smooth_bound.min_val
#         # max_logvar = smooth_bound.max_val

#         # target_logvar = -0.05  # std ≈ exp(-0.025) ≈ 0.975, KL near zero

#         # p = (target_logvar - min_logvar) / (max_logvar - min_logvar)
#         # p = min(max(p, 1.0e-6), 1.0 - 1.0e-6)
#         # init_raw_logvar_bias = math.log(p / (1.0 - p))

#         # # Make the hidden logvar branch initially neutral.
#         # nn.init.zeros_(self.latvar_h.weight)
#         # if self.latvar_h.bias is not None:
#         #     nn.init.zeros_(self.latvar_h.bias)

#         # # Make the final raw-logvar head output a constant near unit std.
#         # nn.init.zeros_(self.out_logvar[0].weight)
#         # if self.out_logvar[0].bias is not None:
#         #     nn.init.constant_(self.out_logvar[0].bias, init_raw_logvar_bias)

#         # All LayerNorm layers.
#         for module in self.modules():
#             if isinstance(module, nn.LayerNorm):
#                 nn.init.ones_(module.weight)
#                 nn.init.zeros_(module.bias)

#         # Auxiliary velocity-estimation head.
#         nn.init.kaiming_uniform_(
#             self.velo_est_hidden.weight,
#             a=1.0,
#             mode="fan_in",
#             nonlinearity="leaky_relu",
#         )
#         if self.velo_est_hidden.bias is not None:
#             nn.init.zeros_(self.velo_est_hidden.bias)

#         nn.init.kaiming_uniform_(
#             self.velo_est_out.weight,
#             a=1.0,
#             mode="fan_in",
#             nonlinearity="linear",
#         )
#         if self.velo_est_out.bias is not None:
#             nn.init.zeros_(self.velo_est_out.bias)

#         # Auxiliary feet-state-estimation head.
#         nn.init.kaiming_uniform_(
#             self.feet_est_hidden.weight,
#             a=1.0,
#             mode="fan_in",
#             nonlinearity="leaky_relu",
#         )
#         if self.feet_est_hidden.bias is not None:
#             nn.init.zeros_(self.feet_est_hidden.bias)

#         # nn.init.kaiming_uniform_(
#         #     self.feet_est_out.weight,
#         #     a=1.0,
#         #     mode="fan_in",
#         #     nonlinearity="linear",
#         # )
#         # if self.feet_est_out.bias is not None:
#         #     nn.init.zeros_(self.feet_est_out.bias)

#         nn.init.kaiming_uniform_(
#             self.feet_normal_est.weight,
#             a=1.0,
#             mode="fan_in",
#             nonlinearity="linear",
#         )
#         if self.feet_normal_est.bias is not None:
#             nn.init.zeros_(self.feet_normal_est.bias)

#         nn.init.kaiming_uniform_(
#             self.feet_height_est.weight,
#             a=1.0,
#             mode="fan_in",
#             nonlinearity="linear",
#         )
#         if self.feet_height_est.bias is not None:
#             nn.init.zeros_(self.feet_height_est.bias)

#         nn.init.kaiming_uniform_(
#             self.feet_contact_est.weight,
#             a=1.0,
#             mode="fan_in",
#             nonlinearity="linear",
#         )
#         if self.feet_contact_est.bias is not None:
#             nn.init.zeros_(self.feet_contact_est.bias)

#     def _check_inputs(
#         self,
#         z_depth_seq: torch.Tensor,
#         z_proprio: torch.Tensor,
#     ) -> None:
#         if z_depth_seq.dim() != 2:
#             raise ValueError(
#                 f"Expected z_depth_seq shape B x {self.depth_latent_dim}, "
#                 f"but got {tuple(z_depth_seq.shape)}."
#             )

#         if z_proprio.dim() != 2:
#             raise ValueError(
#                 f"Expected z_proprio shape B x {self.proprio_latent_dim}, "
#                 f"but got {tuple(z_proprio.shape)}."
#             )

#         if z_depth_seq.shape[0] != z_proprio.shape[0]:
#             raise ValueError(
#                 f"Batch size mismatch: z_depth_seq has batch {z_depth_seq.shape[0]}, "
#                 f"but z_proprio has batch {z_proprio.shape[0]}."
#             )

#         if z_depth_seq.shape[1] != self.depth_latent_dim:
#             raise ValueError(
#                 f"Expected depth_latent_dim={self.depth_latent_dim}, "
#                 f"but got {z_depth_seq.shape[1]}."
#             )

#         if z_proprio.shape[1] != self.proprio_latent_dim:
#             raise ValueError(
#                 f"Expected proprio_latent_dim={self.proprio_latent_dim}, "
#                 f"but got {z_proprio.shape[1]}."
#             )

#     def encode(
#         self,
#         z_depth_seq: torch.Tensor,
#         z_proprio: torch.Tensor,
#     ):
#         """
#         Encodes depth and proprioceptive latents into a fused latent distribution.

#         Args:
#             z_depth_seq:
#                 B x depth_latent_dim

#             z_proprio:
#                 B x proprio_latent_dim

#         Returns:
#             mean:
#                 B x output_dim

#             logvar:
#                 B x output_dim
#         """

#         self._check_inputs(z_depth_seq, z_proprio)

#         # Normalize each modality latent before shared-space projection.
#         z_depth_seq = self.depth_input_norm(z_depth_seq)
#         z_proprio = self.proprio_input_norm(z_proprio)

#         depth_token = self.activation(self.depth_projection(z_depth_seq))
#         proprio_token = self.activation(self.proprio_projection(z_proprio))

#         ##
#         # Feature-wise gated fusion.
#         ##
#         # gate close to 1.0 -> rely more on depth
#         # gate close to 0.0 -> rely more on proprioception
#         # gate_input = torch.cat([depth_token, proprio_token], dim=-1)
#         # gate = torch.sigmoid(self.gate_fc(gate_input))

#         # # Perform gated fusion
#         # x = gate * depth_token + (1.0 - gate) * proprio_token

#         # # normalize fused results
#         # x = self.final_norm(x)

#         merge_input = torch.cat([depth_token, proprio_token], dim=-1)
#         x = self.activation(self.merge_fc(merge_input))

#         # run through fusion backbone
#         x = self.fusion_fc(x)

#         # run log-variance head
#         lat_mean = self.activation(self.latmean_h(x))
#         lat_var = self.activation(self.latvar_h(x))

#         # run mean head
#         mean = self.out_mean(lat_mean)
#         logvar = self.out_logvar(lat_var)

#         return mean, logvar

#     def reparameterization_trick(
#         self,
#         mean: torch.Tensor,
#         logvar: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Sample from latent space using the reparameterization trick.

#         Args:
#             mean:
#                 B x output_dim

#             logvar:
#                 B x output_dim

#         Returns:
#             z:
#                 B x output_dim
#         """
#         epsilon = torch.randn_like(logvar).to(logvar.device)
#         return mean + torch.exp(0.5 * logvar) * epsilon

#     def est_feet(self, z_sample):
#         feet_hidden = self.activation(self.feet_est_hidden(z_sample))

#         feet_contact = self.feet_contact_est(feet_hidden)
#         feet_height = self.feet_height_est(feet_hidden)
#         feet_normal = F.normalize(self.feet_normal_est(feet_hidden), p=2, dim=1, eps=1e-6)

#         feet_state_est = torch.cat((feet_contact, feet_height, feet_normal), dim=-1)

#         return feet_state_est

#     def forward(
#         self,
#         z_depth_seq: torch.Tensor,
#         z_proprio: torch.Tensor,
#     ):
#         """
#         Complete variational forward pass.

#         Args:
#             z_depth_seq:
#                 B x depth_latent_dim

#             z_proprio:
#                 B x proprio_latent_dim

#         Returns:
#             mean:
#                 B x output_dim

#             logvar:
#                 B x output_dim

#             z:
#                 B x output_dim
#         """

#         mean, logvar = self.encode(z_depth_seq, z_proprio)

#         z = self.reparameterization_trick(mean, logvar)

#         body_velo_est = self.velo_est_out(self.activation(self.velo_est_hidden(z)))
#         # feet_state_est = self.feet_est_out(self.activation(self.feet_est_hidden(z)))

#         feet_state_est = self.est_feet(z)

#         return mean, logvar, z, body_velo_est, feet_state_est

#     @torch.no_grad()
#     def forward_inference(
#         self,
#         z_depth_seq: torch.Tensor,
#         z_proprio: torch.Tensor,
#     ):
#         """
#         Deterministic inference path.

#         Args:
#             z_depth_seq:
#                 B x depth_latent_dim

#             z_proprio:
#                 B x proprio_latent_dim

#         Returns:
#             mean:
#                 B x output_dim
#         """
#         mean, _ = self.encode(z_depth_seq, z_proprio)

#         body_velo_est = self.velo_est_out(self.activation(self.velo_est_hidden(mean)))
#         # feet_state_est = self.feet_est_out(self.activation(self.feet_est_hidden(mean)))

#         feet_state_est = self.est_feet(mean)

#         return mean, body_velo_est, feet_state_est


# class MultimodalFusionVAE(nn.Module):
#     """
#     Variational gated-fusion module for fusing depth and proprioceptive latents.

#     Inputs:
#         z_depth_seq:
#             B x depth_latent_dim

#         z_proprio:
#             B x proprio_latent_dim

#     The two modality latents are first normalized independently with LayerNorm,
#     then projected into a shared hidden dimension:

#         LayerNorm(z_depth_seq) -> depth projection   -> B x hidden_dim
#         LayerNorm(z_proprio)   -> proprio projection -> B x hidden_dim

#     A learned feature-wise gate determines how much to use depth vs proprio:

#         gate = sigmoid(W [depth_token, proprio_token])

#         fused = gate * depth_token + (1 - gate) * proprio_token

#     Output:
#         mean:
#             B x output_dim

#         logvar:
#             B x output_dim

#         z:
#             B x output_dim
#     """

#     def __init__(
#         self,
#         depth_latent_dim: int = 32,
#         proprio_latent_dim: int = 16,
#         hidden_dims = [128, 64],
#         output_dim: int = 16,
#         velo_dim: int = 3,
#         feet_state_dim: int = 20,
#         velo_hidden: int = 32,
#         feet_hidden: int = 32,
#         activation: str = "elu",
#         use_layer_norm: bool = True,
#         std_min: float = 0.01,
#         std_max: float = 1.50,
#     ) -> None:
#         super().__init__()

#         self.depth_latent_dim = depth_latent_dim
#         self.proprio_latent_dim = proprio_latent_dim
#         self.num_tokens = 2
#         self.output_dim = output_dim
#         self.activation_name = activation
#         self.use_layer_norm = use_layer_norm

#         self.velo_dim = velo_dim
#         self.feet_state_dim = feet_state_dim

#         self.activation = get_activation(activation)

#         shared_latent_dim = depth_latent_dim

#         # ------------------------------------------------------------------
#         # Per-modality input normalization.
#         # ------------------------------------------------------------------
#         self.depth_input_norm = nn.LayerNorm(depth_latent_dim)
#         self.proprio_input_norm = nn.LayerNorm(proprio_latent_dim)

#         # ------------------------------------------------------------------
#         # Modality-specific projections into a shared embedding space.
#         # ------------------------------------------------------------------
#         embed_dim = int(hidden_dims[0] / 2)
#         self.depth_projection = nn.Linear(depth_latent_dim, embed_dim)
#         self.proprio_projection = nn.Linear(proprio_latent_dim, embed_dim)

#         # ------------------------------------------------------------------
#         # Modality-fusion layer.
#         # ------------------------------------------------------------------
#         self.merge_fc = nn.Linear(2 * embed_dim, hidden_dims[0])

#         # ------------------------------------------------------------------
#         # Fused latent trunk.
#         # ------------------------------------------------------------------
#         fusion_layers = []

#         for l in range(len(hidden_dims)-1):
#             fusion_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l+1]))
#             fusion_layers.append(self.activation)

#         fusion_layers.append(nn.Linear(hidden_dims[-1], shared_latent_dim))
#         fusion_layers.append(self.activation)

#         self.fusion_fc  = nn.Sequential(*fusion_layers)


#         # These skip connections assume the latent inputs are the same dimension
#         self.pre_terrain_skip_h1 = nn.Linear(shared_latent_dim, shared_latent_dim)
#         self.pre_dynamics_skip_h1 = nn.Linear(shared_latent_dim, shared_latent_dim)

#         self.pre_terrain_skip_h2 = nn.Linear(shared_latent_dim, shared_latent_dim)
#         self.pre_dynamics_skip_h2 = nn.Linear(shared_latent_dim, shared_latent_dim)

#         self.post_skips_h = nn.Linear(shared_latent_dim, 2 * output_dim)

#         self.latmean_h = nn.Linear(2 * output_dim, 2 * output_dim)
#         self.latvar_h = nn.Linear(2 * output_dim, 2 * output_dim)

#         self.out_mean = nn.Linear(2 * output_dim, output_dim)

#         self.out_logvar = nn.Sequential(
#             nn.Linear(2 * output_dim, output_dim),
#             # SmoothClampLayer(
#             #     min_val=2.0 * math.log(std_min),
#             #     max_val=2.0 * math.log(std_max),
#             # ),
#             SmoothClampLayer(
#                 min_val=-5.0,
#                 max_val=5.0,
#             ),
#         )

#         self.velo_est_hidden = nn.Linear(output_dim, velo_hidden)
#         self.velo_est_out = nn.Linear(velo_hidden, velo_dim)

#         self.feet_est_hidden = nn.Linear(output_dim, feet_hidden)
#         self.feet_contact_est = nn.Linear(feet_hidden, 4)
#         self.feet_height_est = nn.Linear(feet_hidden, 4)
#         self.feet_normal_est = nn.Linear(feet_hidden, 12)

#         self._initialize_weights()

#     def _initialize_weights(self) -> None:
#         """
#         Initialization routine following the same style as the other KITE
#         encoder modules.

#         Hidden feature layers:
#             Kaiming uniform with leaky_relu gain approximation.

#         Mean/logvar output heads:
#             Mean initialized near zero.
#             Logvar initialized near zero, corresponding to std near one.

#         LayerNorm:
#             weight = 1, bias = 0.

#         Modality embedding:
#             Small normal initialization.

#         Gating:
#             Initialized to produce gate < 0.5, so the initial fusion is
#             slightly weight towards proprioception during early training.
#         """
#         # Modality projections.
#         for layer in [self.depth_projection, self.proprio_projection]:
#             # nn.init.kaiming_uniform_(
#             #     layer.weight,
#             #     a=1.0,
#             #     mode="fan_in",
#             #     nonlinearity="leaky_relu",
#             # )
#             # if layer.bias is not None:
#             #     nn.init.zeros_(layer.bias)
#             nn.init.xavier_uniform_(layer.weight)

#             if layer.bias is not None:
#                 nn.init.zeros_(layer.bias)

#         # Fusion trunk.
#         for layer in self.fusion_fc:
#             if isinstance(layer, nn.Linear):
#                 # nn.init.kaiming_uniform_(
#                 #     module.weight,
#                 #     a=1.0,
#                 #     mode="fan_in",
#                 #     nonlinearity="leaky_relu",
#                 # )
#                 # if module.bias is not None:
#                 #     nn.init.zeros_(module.bias)
#                 nn.init.xavier_uniform_(layer.weight)

#                 if layer.bias is not None:
#                     nn.init.zeros_(layer.bias)

#         # Hidden VAE heads.
#         for layer in [self.latmean_h, self.latvar_h, self.merge_fc, self.out_mean,
#                       self.pre_terrain_skip_h1, self.pre_dynamics_skip_h1,
#                       self.pre_terrain_skip_h2, self.pre_dynamics_skip_h2,
#                       self.post_skips_h]:
#             # nn.init.kaiming_uniform_(
#             #     layer.weight,
#             #     a=1.0,
#             #     mode="fan_in",
#             #     nonlinearity="leaky_relu",
#             # )
#             # if layer.bias is not None:
#             #     nn.init.zeros_(layer.bias)
#             nn.init.xavier_uniform_(layer.weight)

#             if layer.bias is not None:
#                 nn.init.zeros_(layer.bias)

#         self.out_logvar.apply(init_weights)

#         # All LayerNorm layers.
#         for module in self.modules():
#             if isinstance(module, nn.LayerNorm):
#                 nn.init.ones_(module.weight)
#                 nn.init.zeros_(module.bias)

#         for layer in [self.velo_est_hidden, self.feet_est_hidden]:
#             nn.init.kaiming_uniform_(
#                 layer.weight,
#                 a=1.0,
#                 mode="fan_in",
#                 nonlinearity="leaky_relu",
#             )
#             if layer.bias is not None:
#                 nn.init.zeros_(layer.bias)

#         for layer in [
#             self.velo_est_out,
#             self.feet_contact_est,
#             self.feet_height_est,
#             self.feet_normal_est,
#         ]:
#             nn.init.kaiming_uniform_(
#                 layer.weight,
#                 a=1.0,
#                 mode="fan_in",
#                 nonlinearity="linear",
#             )
#             if layer.bias is not None:
#                 nn.init.zeros_(layer.bias)

#     def _check_inputs(
#         self,
#         z_depth_seq: torch.Tensor,
#         z_proprio: torch.Tensor,
#     ) -> None:
#         if z_depth_seq.dim() != 2:
#             raise ValueError(
#                 f"Expected z_depth_seq shape B x {self.depth_latent_dim}, "
#                 f"but got {tuple(z_depth_seq.shape)}."
#             )

#         if z_proprio.dim() != 2:
#             raise ValueError(
#                 f"Expected z_proprio shape B x {self.proprio_latent_dim}, "
#                 f"but got {tuple(z_proprio.shape)}."
#             )

#         if z_depth_seq.shape[0] != z_proprio.shape[0]:
#             raise ValueError(
#                 f"Batch size mismatch: z_depth_seq has batch {z_depth_seq.shape[0]}, "
#                 f"but z_proprio has batch {z_proprio.shape[0]}."
#             )

#         if z_depth_seq.shape[1] != self.depth_latent_dim:
#             raise ValueError(
#                 f"Expected depth_latent_dim={self.depth_latent_dim}, "
#                 f"but got {z_depth_seq.shape[1]}."
#             )

#         if z_proprio.shape[1] != self.proprio_latent_dim:
#             raise ValueError(
#                 f"Expected proprio_latent_dim={self.proprio_latent_dim}, "
#                 f"but got {z_proprio.shape[1]}."
#             )

#     def encode(
#         self,
#         z_depth_seq: torch.Tensor,
#         z_proprio: torch.Tensor,
#     ):
#         """
#         Encodes depth and proprioceptive latents into a fused latent distribution.

#         Args:
#             z_depth_seq:
#                 B x depth_latent_dim

#             z_proprio:
#                 B x proprio_latent_dim

#         Returns:
#             mean:
#                 B x output_dim

#             logvar:
#                 B x output_dim
#         """

#         self._check_inputs(z_depth_seq, z_proprio)

#         # Normalize each modality latent before shared-space projection.
#         z_depth_seq = self.depth_input_norm(0.0*z_depth_seq)
#         z_proprio = self.proprio_input_norm(z_proprio)

#         depth_token = self.activation(self.depth_projection(z_depth_seq))
#         proprio_token = self.activation(self.proprio_projection(z_proprio))

#         ##
#         # Feature-wise gated fusion.
#         ##
#         # gate close to 1.0 -> rely more on depth
#         # gate close to 0.0 -> rely more on proprioception
#         # gate_input = torch.cat([depth_token, proprio_token], dim=-1)
#         # gate = torch.sigmoid(self.gate_fc(gate_input))

#         # # Perform gated fusion
#         # x = gate * depth_token + (1.0 - gate) * proprio_token

#         # # normalize fused results
#         # x = self.final_norm(x)

#         merge_input = torch.cat([depth_token, proprio_token], dim=-1)
#         x = self.activation(self.merge_fc(merge_input))

#         # run through fusion backbone
#         x = self.fusion_fc(x)

#         # self.pre_terrain_skip_h1 = nn.Linear(shared_latent_dim, shared_latent_dim)
#         # self.pre_dynamics_skip_h1 = nn.Linear(shared_latent_dim, shared_latent_dim)

#         # self.pre_terrain_skip_h2 = nn.Linear(shared_latent_dim, shared_latent_dim)
#         # self.pre_dynamics_skip_h2 = nn.Linear(shared_latent_dim, shared_latent_dim)

#         # self.post_skips_h = nn.Linear(shared_latent_dim, 2 * output_dim)

#         # Input latent skip/residual connections
#         x = self.pre_terrain_skip_h1(x)
#         x = self.activation(x + z_depth_seq)
#         x = self.pre_dynamics_skip_h1(x)
#         x = self.activation(x + z_proprio)

#         x = self.pre_terrain_skip_h2(x)
#         x = self.activation(x + z_depth_seq)
#         x = self.pre_dynamics_skip_h2(x)
#         x = self.activation(x + z_proprio)

#         x = self.activation(self.post_skips_h(x))

#         # run log-variance head
#         lat_mean = self.activation(self.latmean_h(x))
#         lat_var = self.activation(self.latvar_h(x))

#         # run mean head
#         mean = self.out_mean(lat_mean)
#         logvar = self.out_logvar(lat_var)

#         return mean, logvar

#     def reparameterization_trick(
#         self,
#         mean: torch.Tensor,
#         logvar: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Sample from latent space using the reparameterization trick.

#         Args:
#             mean:
#                 B x output_dim

#             logvar:
#                 B x output_dim

#         Returns:
#             z:
#                 B x output_dim
#         """
#         epsilon = torch.randn_like(logvar).to(logvar.device)
#         return mean + torch.exp(0.5 * logvar) * epsilon

#     def est_feet(self, z_sample: torch.Tensor) -> torch.Tensor:
#         feet_hidden = self.activation(self.feet_est_hidden(z_sample))
#         feet_contact = self.feet_contact_est(feet_hidden)
#         feet_height = self.feet_height_est(feet_hidden)
#         feet_normal = F.normalize(
#             self.feet_normal_est(feet_hidden),
#             p=2,
#             dim=1,
#             eps=1e-6,
#         )
#         return torch.cat((feet_contact, feet_height, feet_normal), dim=-1)

#     def forward(
#         self,
#         z_depth_seq: torch.Tensor,
#         z_proprio: torch.Tensor,
#     ):
#         """
#         Complete variational forward pass.

#         Args:
#             z_depth_seq:
#                 B x depth_latent_dim

#             z_proprio:
#                 B x proprio_latent_dim

#         Returns:
#             mean:
#                 B x output_dim

#             logvar:
#                 B x output_dim

#             z:
#                 B x output_dim
#         """
#         mean, logvar = self.encode(z_depth_seq, z_proprio)

#         z = self.reparameterization_trick(mean, logvar)

#         body_velo_est = self.velo_est_out(self.activation(self.velo_est_hidden(z)))
#         feet_state_est = self.est_feet(z)

#         return mean, logvar, z, body_velo_est, feet_state_est

#     @torch.no_grad()
#     def forward_inference(
#         self,
#         z_depth_seq: torch.Tensor,
#         z_proprio: torch.Tensor,
#     ):
#         """
#         Deterministic inference path.

#         Args:
#             z_depth_seq:
#                 B x depth_latent_dim

#             z_proprio:
#                 B x proprio_latent_dim

#         Returns:
#             mean:
#                 B x output_dim
#         """
#         mean, _ = self.encode(z_depth_seq, z_proprio)
#         body_velo_est = self.velo_est_out(self.activation(self.velo_est_hidden(mean)))
#         feet_state_est = self.est_feet(mean)

#         return mean, body_velo_est, feet_state_est




class MultimodalFusionVAE(nn.Module):
    """
    Variational gated-fusion module for fusing depth and proprioceptive latents.

    Inputs:
        z_depth_seq:
            B x depth_latent_dim

        z_proprio:
            B x proprio_latent_dim

    The two modality latents are first normalized independently with LayerNorm,
    then projected into a shared hidden dimension:

        LayerNorm(z_depth_seq) -> depth projection   -> B x hidden_dim
        LayerNorm(z_proprio)   -> proprio projection -> B x hidden_dim

    A learned feature-wise gate determines how much to use depth vs proprio:

        gate = sigmoid(W [depth_token, proprio_token])

        fused = gate * depth_token + (1 - gate) * proprio_token

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
        depth_latent_dim: int = 32,
        proprio_latent_dim: int = 32,
        hidden_dims = [128, 64],
        output_dim: int = 32,
        velo_dim: int = 3,
        feet_state_dim: int = 20,
        velo_hidden: int = 128,
        feet_hidden: int = 256,
        activation: str = "elu",
        use_layer_norm: bool = True,
        std_min: float = 0.01,
        std_max: float = 1.50,
    ) -> None:
        super().__init__()

        self.depth_latent_dim = depth_latent_dim
        self.proprio_latent_dim = proprio_latent_dim
        self.num_tokens = 2
        self.output_dim = output_dim
        self.activation_name = activation
        self.use_layer_norm = use_layer_norm

        self.velo_dim = velo_dim
        self.feet_state_dim = feet_state_dim

        self.activation = get_activation(activation)

        # ------------------------------------------------------------------
        # Per-modality input normalization.
        # ------------------------------------------------------------------
        self.depth_input_norm = nn.LayerNorm(depth_latent_dim)
        self.proprio_input_norm = nn.LayerNorm(proprio_latent_dim)

        # ------------------------------------------------------------------
        # Modality-specific projections into a shared embedding space.
        # ------------------------------------------------------------------
        embed_dim = hidden_dims[0]
        self.depth_projection = nn.Linear(depth_latent_dim, embed_dim)
        self.proprio_projection = nn.Linear(proprio_latent_dim, embed_dim)

        # ------------------------------------------------------------------
        # Modality-fusion layer.
        # ------------------------------------------------------------------
        joint_dim = 2 * embed_dim

        self.depth_delta = nn.Sequential(
            nn.Linear(joint_dim, embed_dim),
            self.activation,
            nn.Linear(embed_dim, embed_dim),
        )

        self.depth_gate = nn.Sequential(
            nn.Linear(joint_dim, embed_dim),
            self.activation,
            nn.Linear(embed_dim, embed_dim),
        )

        self.latmean_h = nn.Linear(embed_dim, 2 * output_dim)
        self.latvar_h = nn.Linear(embed_dim, 2 * output_dim)

        self.out_mean = nn.Linear(2 * output_dim, output_dim)

        self.out_logvar = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            SmoothClampLayer(
                min_val=2.0 * math.log(std_min),
                max_val=2.0 * math.log(std_max),
            ),
            # SmoothClampLayer(
            #     min_val=-5.0,
            #     max_val=5.0,
            # ),
        )

        self.velo_est_hidden = nn.Linear(output_dim, velo_hidden)
        self.velo_est_out = nn.Linear(velo_hidden, velo_dim)

        self.feet_est_hidden = nn.Linear(output_dim, feet_hidden)
        self.feet_contact_est = nn.Linear(feet_hidden, 4)
        self.feet_height_est = nn.Linear(feet_hidden, 4)
        self.feet_normal_est = nn.Linear(feet_hidden, 12)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialization routine following the same style as the other KITE
        encoder modules.

        Hidden feature layers:
            Kaiming uniform with leaky_relu gain approximation.

        Mean/logvar output heads:
            Mean initialized near zero.
            Logvar initialized near zero, corresponding to std near one.

        LayerNorm:
            weight = 1, bias = 0.

        Modality embedding:
            Small normal initialization.

        Gating:
            Initialized to produce gate < 0.5, so the initial fusion is
            slightly weight towards proprioception during early training.
        """
        for layer in [self.depth_projection, self.proprio_projection, self.latmean_h, self.latvar_h, self.out_mean]:
            nn.init.xavier_uniform_(layer.weight)

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        self.out_logvar.apply(init_weights)

        self.depth_delta.apply(init_weights)
        nn.init.normal_(self.depth_delta[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.depth_delta[-1].bias)

        self.depth_gate.apply(init_weights)
        nn.init.zeros_(self.depth_gate[-1].weight)
        nn.init.constant_(self.depth_gate[-1].bias, -2.0)

        # All LayerNorm layers.
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for layer in [self.velo_est_hidden, self.feet_est_hidden]:
            nn.init.kaiming_uniform_(
                layer.weight,
                a=1.0,
                mode="fan_in",
                nonlinearity="leaky_relu",
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        for layer in [
            self.velo_est_out,
            self.feet_contact_est,
            self.feet_height_est,
            self.feet_normal_est,
        ]:
            nn.init.kaiming_uniform_(
                layer.weight,
                a=1.0,
                mode="fan_in",
                nonlinearity="linear",
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

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

        merge_input = torch.cat([depth_token, proprio_token], dim=-1)

        ##
        # Feature-wise gated fusion.
        ##
        # gate close to 1.0 -> rely more on depth
        # gate close to 0.0 -> rely more on proprioception
        # gate_input = torch.cat([depth_token, proprio_token], dim=-1)
        delta_d = self.depth_delta(merge_input)
        g = torch.sigmoid(self.depth_gate(merge_input))

        # # Perform gated fusion
        x = proprio_token + g * delta_d

        # run log-variance head
        lat_mean = self.activation(self.latmean_h(x))
        lat_var = self.activation(self.latvar_h(x))

        # run mean head
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

    def est_feet(self, z_sample: torch.Tensor) -> torch.Tensor:
        feet_hidden = self.activation(self.feet_est_hidden(z_sample))
        feet_contact = self.feet_contact_est(feet_hidden)
        feet_height = self.feet_height_est(feet_hidden)
        
        feet_normal = self.feet_normal_est(feet_hidden)
        feet_normal = feet_normal.view(-1, 4, 3)
        feet_normal = F.normalize(
            feet_normal,
            p=2,
            dim=-1,
            eps=1e-6,
        )
        feet_normal = feet_normal.reshape(-1, 12)
        
        return torch.cat((feet_contact, feet_height, feet_normal), dim=-1)

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

        z = self.reparameterization_trick(mean, logvar)

        body_velo_est = self.velo_est_out(self.activation(self.velo_est_hidden(z)))
        feet_state_est = self.est_feet(z)

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
        feet_state_est = self.est_feet(mean)

        return mean, body_velo_est, feet_state_est
class MixerLatentDecoder(nn.Module):
    """Decoder network for reconstructing next state from latent representation and velocity.

    Takes a latent vector and torso velocity as input, processes through two ELU-activated
    hidden layers, and outputs a predicted next state. Uses Xavier uniform initialization.
    """

    def __init__(
            self,
            input_dim: int = 16,
            layers: List[int] = [32, 128, 256],
            decode_dim: int = 64) -> None:
        super().__init__()


        # Network architecture
        self.dec_in = nn.Linear(input_dim, layers[0])
        self.dec_h1 = nn.Linear(layers[0], layers[1])
        self.dec_h2 = nn.Linear(layers[1], layers[2])
        self.dec_out = nn.Linear(layers[2], decode_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize all linear layers with Xavier uniform distribution."""
        for layer in [self.dec_in, self.dec_h1, self.dec_out, self.dec_h2]:
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
        # x = F.elu(self.dec_h3(x))
        return self.dec_out(x)
