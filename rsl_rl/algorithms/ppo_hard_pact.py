# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import time
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random

from rsl_rl.utils import print_class_attributes

from rsl_rl.storage import RolloutStoragePACT
from rsl_rl.modules.hard_pact_physics import normalized_huber_loss

from legged_gym.dynamics import (
    create_go2_dynamics,
    fixed_mechanics_forward_dynamics,
    wrench_at_point,
)
from rsl_rl.hard_pact_ablations import resolve_hard_pact_features

from .pc_grad import PCGrad
from .hard_pact_bard import corrected_bard_inverse_dynamics_loss
from .hard_pact_bard import differentiable_bard_rollout_loss
from .hard_pact_qp import (
    HardPACTDifferentiableQP,
    HardPACTQPConfig,
    projection_loss,
)


@dataclass(frozen=True)
class _DetachedMechanicsCache:
    """Flat, rollout-indexed mechanics with no autograd ownership.

    ``actual`` caches realized randomized mechanics for PINN supervision.
    ``deployment`` caches nominal mechanics at the selected QP substep.  The
    two instances are deliberately distinct so a projection can never obtain
    privileged mass, CoM, armature, or passive parameters by accident.
    """

    kind: str
    mass_matrix: torch.Tensor
    bias: torch.Tensor
    foot_jacobians: torch.Tensor
    base_jacobian: torch.Tensor
    pre_v_canonical: torch.Tensor | None = None
    post_v_canonical: torch.Tensor | None = None
    mass_com_wrench_world: torch.Tensor | None = None
    foot_acceleration_bias: torch.Tensor | None = None

    def index(self, indices):
        return _DetachedMechanicsCache(
            kind=self.kind,
            mass_matrix=self.mass_matrix[indices],
            bias=self.bias[indices],
            foot_jacobians=self.foot_jacobians[indices],
            base_jacobian=self.base_jacobian[indices],
            pre_v_canonical=(None if self.pre_v_canonical is None
                             else self.pre_v_canonical[indices]),
            post_v_canonical=(None if self.post_v_canonical is None
                              else self.post_v_canonical[indices]),
            mass_com_wrench_world=(
                None if self.mass_com_wrench_world is None
                else self.mass_com_wrench_world[indices]
            ),
            foot_acceleration_bias=(
                None if self.foot_acceleration_bias is None
                else self.foot_acceleration_bias[indices]
            ),
        )

    def as_context(self, dynamics):
        """Expose the small interface consumed by the rollout loss."""
        from types import SimpleNamespace
        return SimpleNamespace(
            dynamics=dynamics,
            mass_matrix=self.mass_matrix,
            bias=self.bias,
            foot_jacobians=self.foot_jacobians,
            base_jacobian=self.base_jacobian,
            pre_v_canonical=self.pre_v_canonical,
            post_v_canonical=self.post_v_canonical,
            mass_com_wrench_world=self.mass_com_wrench_world,
            forward_dynamics=lambda generalized_force:
                fixed_mechanics_forward_dynamics(
                    self.mass_matrix, self.bias, generalized_force
                ),
        )

def _yaw_local_to_world(values, quaternion_xyzw):
    """Rotate batched ``[..., xyz]`` vectors by base yaw only."""
    x, y, z, w = quaternion_xyzw.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    result = values.clone()
    result[..., 0] = cosine[:, None] * values[..., 0] - sine[:, None] * values[..., 1]
    result[..., 1] = sine[:, None] * values[..., 0] + cosine[:, None] * values[..., 1]
    return result


def _body_point_to_world(point_body, q_xyzw):
    """Rotate a body-frame point offset into world-aligned coordinates."""
    x, y, z, w = q_xyzw[:, 3:7].unbind(-1)
    rotation = torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(-1, 3, 3)
    return torch.einsum("bij,bj->bi", rotation, point_body)


def disjoint_qp_epoch_mask(
    rollout_indices,
    anchor_indices,
    *,
    epoch,
    num_epochs,
    passes_per_iteration=1,
    shard_percentage=None,
    stratify_by_anchor=True,
    seed=1,
    iteration=0,
):
    """Return a deterministic, exactly covered PPO-QP shard.

    Every row is assigned a reproducible base epoch from its flat rollout
    index, run seed, and iteration.  Independent ranking within each stored
    physics anchor keeps every epoch approximately balanced over anchors.
    Selecting ``passes_per_iteration`` consecutive cyclic epochs gives every
    row exactly that many evaluations; the default one pass is a true
    disjoint partition.
    """
    if not 0 <= int(epoch) < int(num_epochs):
        raise ValueError("epoch must lie in [0, num_epochs)")
    if not 1 <= int(passes_per_iteration) <= int(num_epochs):
        raise ValueError("passes_per_iteration must lie in [1, num_epochs]")
    expected_percentage = 100.0 * int(passes_per_iteration) / int(num_epochs)
    if (
        shard_percentage is not None
        and abs(float(shard_percentage) - expected_percentage) > 1.0e-6
    ):
        raise ValueError(
            "exact disjoint QP coverage requires shard_percentage == "
            "100 * passes_per_iteration / num_epochs"
        )
    rollout_indices = rollout_indices.reshape(-1).to(torch.int64)
    anchors = anchor_indices.reshape(-1).to(
        device=rollout_indices.device, dtype=torch.int64
    )
    if anchors.numel() != rollout_indices.numel():
        raise ValueError("rollout and anchor index counts must match")
    selected = torch.zeros_like(rollout_indices, dtype=torch.bool)
    groups = (0, 1, 2, 3) if stratify_by_anchor else (None,)
    # Fixed integer hashing avoids mutable RNG state, making resume at the
    # same iteration exactly reproducible without enlarging checkpoints.
    salt = int(seed) + 104729 * int(iteration)
    for anchor in groups:
        rows = torch.arange(
            rollout_indices.numel(), device=rollout_indices.device
        )
        if anchor is not None:
            rows = rows[anchors == anchor]
        if rows.numel() == 0:
            continue
        keys = torch.remainder(
            rollout_indices[rows] * 1103515245 + salt, 2147483647
        )
        order = torch.argsort(keys, stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(order.numel(), device=order.device)
        base_epoch = torch.remainder(rank, int(num_epochs))
        cyclic_distance = torch.remainder(int(epoch) - base_epoch, int(num_epochs))
        selected[rows] = cyclic_distance < int(passes_per_iteration)
    return selected

class PPO_HardPACT:
    r"""Self-contained PACT PPO for the HardPACT aliases.

    The legacy PPO rollout, clipped surrogate/value losses, entropy schedule,
    bootstrapping decision, spectral clipping, and checkpoint-facing optimizer
    attributes are retained locally. HardPACT adds two training phases per
    minibatch:

    1. PPO and the corrected BARD objective share one PCGrad backward pass.
       The B1Z1 PACT ownership boundary is used, so this optimizer contains the
       policy, critic, history pathway, privileged decoder, and physics heads.
    2. A newly recomputed graph forms the single auxiliary objective

       ``L_aux = lambda_priv L_priv + beta L_KL + lambda_e L_e``
       ``        + lambda_F L_F + lambda_Wa L_W_active``
       ``        + lambda_Wn L_W_neutral``.

       One auxiliary optimizer step updates the shared history/decoder
       boundary. Actor, critic, and action-noise parameters are excluded.

    Recomputing phase two is important: PCGrad consumes the first autograd
    graph, and retaining it across optimizer steps would both waste memory and
    make the gradients depend on stale parameters.
    """
    actor_critic: nn.Module
    decoder_network: nn.Module
    def __init__(self,
                 actor_critic,
                 decoder_network,
                 num_priv_obs,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.99,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 use_spo=False,
                 pinn_lambda=0.001,
                 pinn_encoder_weight=0.05,
                 pinn_warmup=1000,
                 pinn_init_steps=500,
                 num_encoder_epochs=1, # number of epochs for hybrid encoder via supervised learning
                 vae_kld_weight=2.0,   # weight of KL divergence loss in VAE
                 use_adaptive_entropy=True,
                 adaptive_ent_bounds=[0.01, 0.001],
                 adaptive_ent_lin_threshold=0.75,
                 adaptive_ent_ang_threshold=0.35,
                 adaptive_ent_ter_threshold=5.0,
                 adaptive_ent_softmax_temp=2.0,
                 reconstruction_indices=None,
                 auxiliary_learning_rate=2.0e-4,
                 privileged_loss_weight=1.0,
                 explicit_loss_weight=1.0,
                 grf_loss_weight=1.0,
                 active_wrench_loss_weight=1.0,
                 neutral_wrench_loss_weight=0.25,
                 bard_enabled=True,
                 bard_randomize_base_inertia=True,
                 bard_scale_rotational_inertia=True,
                 bard_urdf_path="resources/robots/go2/urdf/go2.urdf",
                 bard_batch_capacity=4096,
                 dynamics_backend="bard",
                 pinocchio_num_workers=None,
                 bard_inverse_enabled=True,
                 bard_rollout_enabled=True,
                 lambda_inverse=1.0,
                 lambda_rollout=1.0,
                 lambda_projection=1.0e-3,
                 lambda_soft_constraint=1.0e-3,
                 profile_bard_timing=False,
                 console_debug=False,
                 pcgrad_diagnostics_enabled=False,
                 pcgrad_diagnostics_start_iteration=0,
                 pcgrad_diagnostics_interval=50,
                 cache_rollout_mechanics=True,
                 ppo_qp_sampling="disjoint_epoch_partition",
                 ppo_qp_passes_per_iteration=1,
                 ppo_qp_shard_percentage=None,
                 ppo_qp_stratify_by_anchor=True,
                 ppo_qp_sampling_seed=1,
                 ppo_qp_sampling_logging_enabled=True,
                 ablation_variant="full",
                 hard_pact_qp=None,
                 grf_observation_scale=0.01,
                 base_wrench_observation_scale=0.01,
                 action_clip=100.0,
                 ):
        
        self.device = device

        self.num_priv_obs = num_priv_obs
        self.reconstruction_indices = reconstruction_indices

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.num_enc_epochs = num_encoder_epochs
        self.vae_beta = vae_kld_weight
        self.privileged_loss_weight = float(privileged_loss_weight)
        self.explicit_loss_weight = float(explicit_loss_weight)
        self.grf_loss_weight = float(grf_loss_weight)
        self.active_wrench_loss_weight = float(active_wrench_loss_weight)
        self.neutral_wrench_loss_weight = float(neutral_wrench_loss_weight)

        # ``pinn_encoder_weight`` remains in the constructor solely so legacy
        # configs instantiate unchanged. B1Z1 PCGrad replaced that historical
        # manual gradient-injection mechanism.

        # Adaptive entropy coefficent algorithm values
        self.use_adaptive_entropy = use_adaptive_entropy
        self.entropy_coef_bounds = adaptive_ent_bounds
        self.ent_linvelo_threshold = adaptive_ent_lin_threshold
        self.ent_angvelo_threshold = adaptive_ent_ang_threshold
        self.ent_terrain_threshold = adaptive_ent_ter_threshold
        self.ent_softmax_temperature = adaptive_ent_softmax_temp
        
        self.current_entropy_coef = entropy_coef
        self.entropy_coef = entropy_coef

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later

        # Match the B1Z1 PACT ownership plan. PPO/PINN PCGrad owns the complete
        # differentiable policy path, including context and deployment
        # decoders. A second optimizer performs the temporally separate,
        # combined auxiliary update on that shared context/decoder boundary.
        actor_groups, context_groups = actor_critic.get_optim_groups()
        decoder_group = {
            "params": list(decoder_network.parameters()),
            "weight_decay": context_groups[0].get("weight_decay", 0.0),
            "name": "privileged_decoder",
        }
        shared_groups = [*context_groups, decoder_group]
        ppo_shared_groups = [
            {**group, "params": list(group["params"]), "name": f"ppo_{group['name']}"}
            for group in shared_groups
        ]
        auxiliary_groups = [
            {**group, "params": list(group["params"]), "name": f"auxiliary_{group['name']}"}
            for group in shared_groups
        ]
        self.act_optimizer = PCGrad(
            optim.AdamW([*actor_groups, *ppo_shared_groups], lr=learning_rate),
            reduction="sum",
        )
        self.auxiliary_optimizer = optim.AdamW(
            auxiliary_groups, lr=auxiliary_learning_rate
        )
        self.enc_optimizer = self.auxiliary_optimizer
        self.decoder_optimizer = self.auxiliary_optimizer
        self.transition = RolloutStoragePACT.Transition()

        # # We want to reduce the LR of the critic
        for param_group in self.act_optimizer.optimizer.param_groups:
            # specifically modifies the learning rate of the crtic specific parameters
            if "name" in param_group.keys():
                if "critic" in param_group["name"]:
                    param_group['lr'] = (learning_rate / 3.0)

        self.decoder = decoder_network

        self.ppo_parameters = list(dict.fromkeys(
            parameter for group in self.act_optimizer.optimizer.param_groups
            for parameter in group["params"]
        ))
        self.auxiliary_parameters = list(dict.fromkeys(
            parameter for group in self.auxiliary_optimizer.param_groups
            for parameter in group["params"]
        ))

        self.boot_mult = 1.0
        self.use_boot = False

        self.pinn_weight_final = pinn_lambda
        self.pinn_weight = 0.0
        self.pinn_warmup_steps = pinn_warmup
        self.pinn_init = pinn_init_steps

        self.num_pinn_updates = 0

        self.hard_pact_features = resolve_hard_pact_features(ablation_variant)
        # The single canonical profile is authoritative. Constructor booleans
        # remain accepted only for strict compatibility with old checkpoints
        # and launch configs.
        self.bard_enabled = self.hard_pact_features.needs_bard
        self.bard_inverse_enabled = self.hard_pact_features.inverse_loss
        self.bard_rollout_enabled = self.hard_pact_features.rollout_loss
        self.lambda_inverse = float(lambda_inverse)
        self.lambda_rollout = float(lambda_rollout)
        self.lambda_projection = float(lambda_projection)
        self.lambda_soft_constraint = float(lambda_soft_constraint)
        # Opt-in benchmark instrumentation. CUDA events avoid synchronizing
        # between the shared-context inverse and rollout calculations; one
        # synchronization at the end of update materializes all durations.
        self.profile_bard_timing = bool(profile_bard_timing)
        self.console_debug = bool(console_debug)
        self.pcgrad_diagnostics_enabled = bool(pcgrad_diagnostics_enabled)
        self.pcgrad_diagnostics_start_iteration = int(
            pcgrad_diagnostics_start_iteration
        )
        self.pcgrad_diagnostics_interval = int(pcgrad_diagnostics_interval)
        self.cache_rollout_mechanics = bool(cache_rollout_mechanics)
        self.ppo_qp_sampling = str(ppo_qp_sampling)
        self.ppo_qp_passes_per_iteration = int(
            ppo_qp_passes_per_iteration
        )
        self.ppo_qp_shard_percentage = (
            None if ppo_qp_shard_percentage is None
            else float(ppo_qp_shard_percentage)
        )
        self.ppo_qp_stratify_by_anchor = bool(ppo_qp_stratify_by_anchor)
        self.ppo_qp_sampling_seed = int(
            1 if ppo_qp_sampling_seed is None else ppo_qp_sampling_seed
        )
        self.ppo_qp_sampling_logging_enabled = bool(
            ppo_qp_sampling_logging_enabled
        )
        if self.ppo_qp_sampling not in ("all", "disjoint_epoch_partition"):
            raise ValueError(
                "ppo_qp_sampling must be 'all' or "
                "'disjoint_epoch_partition'"
            )
        configured_learning_epochs = int(num_learning_epochs)
        if not 1 <= self.ppo_qp_passes_per_iteration <= configured_learning_epochs:
            raise ValueError(
                "ppo_qp_passes_per_iteration must be in [1, num_learning_epochs]"
            )
        expected_percentage = (
            100.0 * self.ppo_qp_passes_per_iteration
            / configured_learning_epochs
        )
        if self.ppo_qp_shard_percentage is None:
            self.ppo_qp_shard_percentage = expected_percentage
        if (
            self.ppo_qp_sampling == "disjoint_epoch_partition"
            and abs(self.ppo_qp_shard_percentage - expected_percentage) > 1.0e-6
        ):
            raise ValueError(
                "exact disjoint QP coverage requires "
                "ppo_qp_shard_percentage == "
                "100 * ppo_qp_passes_per_iteration / num_learning_epochs"
            )
        if self.pcgrad_diagnostics_interval < 1:
            raise ValueError("pcgrad_diagnostics_interval must be positive")
        self._rollout_actual_mechanics = None
        self._rollout_deployment_qp_mechanics = None
        self._pcgrad_audit_ran = False
        self._bard_timing_records = {
            "inverse": [], "rollout": [], "dynamics": [],
            "auxiliary": [], "pcgrad": [],
        }
        if min(
            self.lambda_inverse, self.lambda_rollout, self.lambda_projection,
            self.lambda_soft_constraint,
        ) < 0.0:
            raise ValueError(
                "physics loss weights must be nonnegative"
            )
        self.qp_config = replace(
            HardPACTQPConfig(**(hard_pact_qp or {})),
            enabled=self.hard_pact_features.execution_qp,
        )
        self.hard_pact_qp = None
        self.last_qp_metrics = {}
        self._qp_full_audit_inputs = None
        self.grf_observation_scale = float(grf_observation_scale)
        self.base_wrench_observation_scale = float(base_wrench_observation_scale)
        self.action_clip = float(action_clip)
        if self.action_clip <= 0.0:
            raise ValueError("action_clip must be positive")
        self.last_inverse_dynamics_metrics = {}
        self.last_rollout_dynamics_metrics = {}
        self.last_physics_gradient_metrics = {}
        self.last_auxiliary_metrics = {}
        self.last_physics_loss_metrics = {}
        # Reporting keeps the configured inverse/rollout/soft composition but
        # intentionally excludes the outer PINN curriculum weight.  The
        # optimization objective remains unchanged.
        self.last_unweighted_pinn_loss = torch.zeros((), device=self.device)
        self.dynamics_backend = str(dynamics_backend).lower()
        if self.dynamics_backend not in ("bard", "pinocchio"):
            raise ValueError("dynamics_backend must be 'bard' or 'pinocchio'")
        self.physics_dynamics = None
        if self.bard_enabled:
            # Configuration paths are repository-relative, while launchers run
            # from legged_gym/scripts. Resolve relative URDFs against the repo
            # root derived from this source file so training is cwd-independent.
            resolved_bard_urdf_path = bard_urdf_path
            if not os.path.isabs(resolved_bard_urdf_path):
                repository_root = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), "..", ".."
                ))
                resolved_bard_urdf_path = os.path.join(
                    repository_root, resolved_bard_urdf_path
                )
            backend_kwargs = {}
            if self.dynamics_backend == "pinocchio":
                backend_kwargs.update(
                    num_workers=pinocchio_num_workers,
                    profile_timing=self.profile_bard_timing,
                )
            self.physics_dynamics = create_go2_dynamics(
                self.dynamics_backend,
                os.path.abspath(resolved_bard_urdf_path),
                device=self.device,
                batch_capacity=bard_batch_capacity,
                randomize_base_inertia=bard_randomize_base_inertia,
                scale_rotational_inertia=bard_scale_rotational_inertia,
                **backend_kwargs,
            )
        # Compatibility alias for older runners and test fixtures. It points
        # to the selected implementation and never creates a second model.
        self.bard_dynamics = self.physics_dynamics

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        if self.console_debug:
            print_class_attributes(self)

    def configure_hard_pact_qp(
        self, torque_limits, position_limits, velocity_limits
    ):
        """Bind backend hard limits once without copying them per transition."""
        position_limits = torch.as_tensor(position_limits)
        # Genesis returns [num_envs, 2, 12], whereas Isaac-style backends use
        # [12, 2]. Limits are identical across environments, so retain one
        # table and normalize both layouts to canonical [12, lower/upper].
        if position_limits.ndim == 3:
            position_limits = position_limits[0]
        if position_limits.shape == (2, 12):
            position_limits = position_limits.transpose(0, 1)
        if position_limits.shape != (12, 2):
            raise ValueError(
                "HardPACT position limits must resolve to [12,2], got "
                f"{tuple(position_limits.shape)}"
            )
        velocity_limits = torch.as_tensor(velocity_limits).reshape(-1)
        if velocity_limits.numel() == 0:
            # Genesis has no velocity-limit query and legacy PACT intentionally
            # leaves its asset list empty. Keep the alias configuration exact
            # and bind the repository's calibrated Go2 URDF limits only to the
            # HardPACT QP backend object.
            velocity_limits = position_limits.new_tensor(
                [30.1, 30.1, 15.7] * 4
            )
        if velocity_limits.numel() < 12:
            raise ValueError("HardPACT requires 12 joint velocity limits")
        self.hard_pact_qp = HardPACTDifferentiableQP(
            self.qp_config,
            torch.as_tensor(torque_limits).reshape(-1)[:12],
            position_limits[:12, 0],
            position_limits[:12, 1],
            velocity_limits[:12],
        )
        
        
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, priv_obs_shape, obs_hist_shape, action_shape, torso_velo_shape, grf_shape, wb_shape):
        self.storage = RolloutStoragePACT(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, priv_obs_shape, obs_hist_shape, \
                                              action_shape, torso_velo_shape, grf_shape, wb_shape, self.device,
                                              store_legacy_pinn_dynamics=False)

    def test_mode(self):
        self.actor_critic.test()

    def _set_std_clip_lwr(self, clip_val=0.1):
        self.actor_critic._set_std_clip_lwr(clip_val)

    def set_entropy_coef(self, coef=1e-3):
        if self.use_adaptive_entropy: 
            self.current_entropy_coef = coef
        else:
            self.entropy_coef = coef        
        
    def update_adaptive_entropy_coef(self, performance_metrics):
        lin_vel_tracking = performance_metrics.get('lin_vel_tracking', 0.0)
        ang_vel_tracking = performance_metrics.get('ang_vel_tracking', 0.0)
        terrain_level = performance_metrics.get('terrain_level', 0)
        
        lin_vel_gap = max(0, self.ent_linvelo_threshold - lin_vel_tracking)
        ang_vel_gap = max(0, self.ent_angvelo_threshold - ang_vel_tracking)
        terrain_gap = max(0, self.ent_terrain_threshold - terrain_level)
        
        norm_lin_gap = lin_vel_gap /  self.ent_linvelo_threshold if self.ent_linvelo_threshold > 0 else 0
        norm_ang_gap = ang_vel_gap / self.ent_angvelo_threshold if self.ent_angvelo_threshold > 0 else 0
        norm_terrain_gap = terrain_gap / self.ent_terrain_threshold if self.ent_terrain_threshold > 0 else 0
        
        gaps = torch.tensor([norm_lin_gap, norm_ang_gap, norm_terrain_gap], dtype=torch.float32)
        
        weights = F.softmax(gaps / self.ent_softmax_temperature, dim=0)
        
        weighted_gap = torch.sum(weights * gaps).item()
        
        self.current_entropy_coef = self.entropy_coef_bounds[0] + weighted_gap * (self.entropy_coef_bounds[1] - self.entropy_coef_bounds[0])
        
        return self.current_entropy_coef
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, obs_history, prev_obs, prev_obs_hist, pprev_obs, pprev_obs_hist):
        if self.use_boot:
            all_actions = self.actor_critic.act(obs,obs_history).detach()
        else:
            all_actions = self.actor_critic.act_bootmask(obs,obs_history).detach()

        # Compute the actions and values
        #  - Position Control
        self.transition.actions =  all_actions
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # Save the standardized draw, not a reparameterized tensor. During
        # PPO, a_current = mu_current + sigma_current * epsilon_stored gives a
        # differentiable stochastic replay while log probability continues to
        # use the original raw sample in `transition.actions`.
        self.transition.action_noise = (
            (all_actions - self.transition.action_mean)
            / self.transition.action_sigma.clamp_min(1.0e-8)
        ).detach()
        
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.observation_history = obs_history
        self.transition.critic_observations = critic_obs

        # PINN stuff
        self.transition.prev_obs      = prev_obs
        self.transition.prev_obs_hist = prev_obs_hist
        self.transition.pprev_obs      = pprev_obs
        self.transition.pprev_obs_hist = pprev_obs_hist
        
        return all_actions
    
    def process_env_step(self, rewards, dones, infos, grf_labels, obs_labels, explicit_labels, gt_forces, mass_mats, bias_vecs, torso_acc):
        self.transition.rewards = rewards.clone()
        
        self.transition.dones = dones
        # Values from the next-time step used as labels for the decoder network
        self.transition.grf_targets = grf_labels

        # This is now the stack of critic observations, we want to prune off the last one
        reconstruction_target = obs_labels[:, -self.num_priv_obs:]
        if self.reconstruction_indices is not None:
            reconstruction_target = reconstruction_target[:, self.reconstruction_indices]
        self.transition.obs_targets = reconstruction_target

        self.transition.explicit_labels = explicit_labels
        self.transition.hard_pact = infos.get("hard_pact_transition")
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # PINN stuff
        self.transition.wb_contact_forces = gt_forces
        self.transition.wb_mass_mat = mass_mats
        self.transition.wb_bias_vec = bias_vecs
        self.transition.torso_acc = torso_acc
        
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)


    def spectral_normalization(
        self,
        model: nn.Module,
        sigma_max: float = 1.0,
        n_power_iters: int = 1,
    ):
        """
        Spectral-norm clip all Linear layers except selected output layers.

        Args:
            model: network to normalize in-place
            sigma_max: maximum allowed spectral norm
            n_power_iters: number of power iterations for sigma estimate
        """

        whitelist = (nn.Linear,)

        # lazily create persistent power-iteration vectors
        if not hasattr(self, "_spec_u"):
            self._spec_u = {}

        for module_name, module in model.named_modules():
            if not isinstance(module, whitelist):
                continue

            # skip known output layers
            if module_name.endswith("out") or module_name.endswith("mean") or module_name.endswith("var") or "critic" in module_name:
                continue

            for param_name, param in module.named_parameters(recurse=False):
                if param_name != "weight" or param.ndim != 2:
                    continue

                full_name = f"{module_name}.{param_name}" if module_name else param_name
                W = param.data  # [out_dim, in_dim]

                # initialize persistent u vector once per parameter
                if full_name not in self._spec_u or self._spec_u[full_name].shape[0] != W.shape[0]:
                    u = torch.randn(W.shape[0], device=W.device, dtype=W.dtype)
                    u = u / (u.norm() + 1e-12)
                    self._spec_u[full_name] = u

                u = self._spec_u[full_name]

                with torch.no_grad():
                    # power iteration
                    for _ in range(n_power_iters):
                        v = W.t().mv(u)
                        v = v / (v.norm() + 1e-12)

                        u = W.mv(v)
                        u = u / (u.norm() + 1e-12)

                    # sigma ~= u^T W v
                    sigma = torch.dot(u, W.mv(v))

                    # save updated u for next call
                    self._spec_u[full_name] = u

                    # clip only if above threshold
                    if sigma > sigma_max:
                        param.data.mul_(sigma_max / (sigma + 1e-12))
    
    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)  

    @staticmethod
    def _flat_fields(fields):
        return {
            name: value.flatten(0, 1) for name, value in fields.items()
        }

    def _materialize_mechanics_cache(
        self, *, kind, q, v, parameters, post_v=None,
        mass_com_wrench_world=None, need_qp=False,
    ):
        """Evaluate detached backend mechanics once in bounded chunks."""
        names = ["mass_matrix", "bias", "foot_jacobians", "base_jacobian"]
        if need_qp:
            names.append("foot_acceleration_bias")
        values = {name: [] for name in names}
        pre_velocity, post_velocity = [], []
        capacity = self.physics_dynamics.batch_capacity
        for start in range(0, q.shape[0], capacity):
            stop = min(start + capacity, q.shape[0])
            sl = slice(start, stop)
            context = self.physics_dynamics.build_context(
                q[sl], v[sl],
                parameters={name: value[sl] for name, value in parameters.items()},
                post_v_world=None if post_v is None else post_v[sl],
                mass_com_wrench_world=(
                    None if mass_com_wrench_world is None
                    else mass_com_wrench_world[sl]
                ),
                need_jacobians=True,
                need_qp=need_qp,
                need_forward_dynamics=not need_qp,
            )
            for name in names:
                values[name].append(getattr(context, name).detach())
            if not need_qp:
                pre_velocity.append(context.pre_v_canonical.detach())
                post_velocity.append(context.post_v_canonical.detach())
        merged = {
            name: torch.cat(chunks, dim=0) for name, chunks in values.items()
        }
        return _DetachedMechanicsCache(
            kind=kind,
            **merged,
            pre_v_canonical=(
                None if need_qp else torch.cat(pre_velocity, dim=0)
            ),
            post_v_canonical=(
                None if need_qp else torch.cat(post_velocity, dim=0)
            ),
            mass_com_wrench_world=(
                None if mass_com_wrench_world is None
                else mass_com_wrench_world.detach()
            ),
        )

    def _prepare_rollout_mechanics_cache(
        self, default_pose, *, need_actual=True, need_qp=True
    ):
        """Create actual and deployment caches for exactly one PPO update."""
        self._rollout_actual_mechanics = None
        self._rollout_deployment_qp_mechanics = None
        fields = self.storage.hard_pact_fields
        if fields is None or self.physics_dynamics is None:
            return
        flat = self._flat_fields(fields)
        self.physics_dynamics.default_joint_position = torch.as_tensor(
            default_pose, device=flat["pre_q"].device,
            dtype=flat["pre_q"].dtype,
        ).detach()
        if need_actual and (self.bard_inverse_enabled or self.bard_rollout_enabled):
            actual = {
                "added_base_mass": flat["realized_added_mass"].detach(),
                "base_com_shift": flat["realized_com_shift_body"].detach(),
                "joint_armature": flat["joint_armature"].detach(),
                "joint_friction": flat["joint_friction"].detach(),
                "joint_stiffness": flat["joint_stiffness"].detach(),
                "joint_damping": flat["joint_damping"].detach(),
            }
            self._rollout_actual_mechanics = self._materialize_mechanics_cache(
                kind="actual",
                q=flat["pre_q"], v=flat["pre_v"], parameters=actual,
                post_v=flat["post_v"],
                mass_com_wrench_world=flat[
                    "equivalent_mass_com_wrench_world"
                ],
            )
        qp_ready = (
            self.hard_pact_features.execution_qp
            and self.hard_pact_qp is not None
        )
        if need_qp and qp_ready:
            sample_q = flat.get("sampled_qp_q", flat["pre_q"])
            sample_v = flat.get("sampled_qp_v", flat["pre_v"])
            # The deployed projection has no access to realized randomization.
            # Empty parameters mean nominal URDF inertia and zero passive
            # deltas in both BARD and Pinocchio adapters.
            self._rollout_deployment_qp_mechanics = (
                self._materialize_mechanics_cache(
                    kind="deployment", q=sample_q, v=sample_v,
                    parameters={}, need_qp=True,
                )
            )

    def _clear_rollout_mechanics_cache(self):
        self._rollout_actual_mechanics = None
        self._rollout_deployment_qp_mechanics = None

    def _qp_rows_for_epoch(self, epoch, iteration):
        """Select QP-only rows without changing the full PPO/BARD batch."""
        batch = self.storage.current_hard_pact_batch
        if (
            not self.hard_pact_features.execution_qp
            or self.hard_pact_qp is None
            or batch is None
        ):
            return None
        count = next(iter(batch.values())).shape[0]
        if self.ppo_qp_sampling == "all":
            return torch.arange(count, device=self.device)
        anchors = batch["sampled_qp_substep_index"].reshape(-1)
        mask = disjoint_qp_epoch_mask(
            self.storage.current_batch_indices,
            anchors,
            epoch=epoch,
            num_epochs=self.num_learning_epochs,
            passes_per_iteration=self.ppo_qp_passes_per_iteration,
            shard_percentage=self.ppo_qp_shard_percentage,
            stratify_by_anchor=self.ppo_qp_stratify_by_anchor,
            seed=self.ppo_qp_sampling_seed,
            iteration=iteration,
        )
        rows = mask.nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            raise RuntimeError(
                "PPO QP shard is empty; increase minibatch size or disable "
                "anchor stratification"
            )
        return rows

    def _pcgrad_diagnostics_due(self, iteration):
        return (
            self.pcgrad_diagnostics_enabled
            and iteration >= self.pcgrad_diagnostics_start_iteration
            and (
                (iteration - self.pcgrad_diagnostics_start_iteration)
                % self.pcgrad_diagnostics_interval == 0
            )
        )

    def _update_pinn_weight_for_iteration(self, iteration):
        """Apply the one shared PINN warm-up schedule for every ablation.

        Feature selection determines which terms are inside the physics
        objective, never when that objective starts. Keeping this schedule in
        one variant-independent helper prevents ``soft`` and ``full`` from
        drifting when their expensive paths differ.
        """
        if (
            iteration > self.pinn_init
            and self.num_pinn_updates < self.pinn_warmup_steps + 1
        ):
            if self.pinn_weight_final < 0:
                self.pinn_weight = 1.0
            else:
                self.pinn_weight = (
                    float(self.num_pinn_updates)
                    / float(self.pinn_warmup_steps)
                ) * self.pinn_weight_final
            if self.console_debug:
                print(self.pinn_weight)
        return self.pinn_weight

    def update(self, action_func, fb_func, dt, itr, default_pose, qvel_scale):
        metric_zero = torch.zeros((), device=self.device)
        mean_value_loss = metric_zero.clone()
        mean_surrogate_loss = metric_zero.clone()
        mean_autoenc_loss = metric_zero.clone()
        mean_vel_loss = metric_zero.clone()
        mean_recon_loss = metric_zero.clone()
        mean_kld_loss = metric_zero.clone()
        mean_decoder_loss = metric_zero.clone()
        mean_pinn_loss = metric_zero.clone()
        self._bard_timing_records = {
            "inverse": [], "rollout": [], "dynamics": [],
            "auxiliary": [], "pcgrad": [],
        }

        boot_count = 0
        boot_sum_x = None
        boot_sum_x2 = None
        boot_sum_recon_sqerr = torch.zeros(
            (), device=self.device, dtype=torch.float64
        )

        self._update_pinn_weight_for_iteration(itr)

        qp_iteration_ready = (
            self.hard_pact_features.execution_qp
            and self.hard_pact_qp is not None
        )
        pinn_iteration_ready = self.pinn_weight > 0.0 and (
            self.bard_inverse_enabled
            or self.bard_rollout_enabled
            or self.hard_pact_features.soft_constraint_penalty
        )
        if (
            (pinn_iteration_ready or qp_iteration_ready)
            and self.cache_rollout_mechanics
        ):
            dynamics_timing = self._start_bard_timing(
                "dynamics", self.storage.observations
            )
            self._prepare_rollout_mechanics_cache(
                default_pose,
                need_actual=pinn_iteration_ready,
                # Legacy "all" reuses every transition across epochs. The
                # disjoint mode instead constructs mechanics only after its
                # rows are sliced, so unsampled rows never enter QP work.
                need_qp=(
                    qp_iteration_ready and self.ppo_qp_sampling == "all"
                ),
            )
            self._stop_bard_timing("dynamics", dynamics_timing)

        self._pcgrad_audit_ran = self._pcgrad_diagnostics_due(itr)

        qp_sampled_count = torch.zeros((), device=self.device)
        qp_full_count = torch.zeros((), device=self.device)
        qp_target_count = torch.zeros((), device=self.device)
        qp_valid_count = torch.zeros((), device=self.device)
        qp_anchor_zero_count = torch.zeros((), device=self.device)
        qp_anchor_two_count = torch.zeros((), device=self.device)
        qp_gradient_square_sum = torch.zeros((), device=self.device)
        self._qp_sampling_gradient_inputs = None

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch_number, (terminated_batch, obs_batch, critic_obs_batch, obs_hist_batch, explicit_labels_batch, \
            grf_target, obs_target, actions_batch, target_values_batch, \
            advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, \
            old_sigma_batch, _prev_obs_batch, _prev_obs_hist_batch, gt_forces_batch, _mass_mat_batch, \
            _bias_vec_batch, _torso_accs_batch, _pprev_obs_batch, _pprev_obs_hist_batch) in enumerate(generator):
            ppo_epoch = batch_number // self.num_mini_batches
            
            self.actor_critic.train()
            self.act_optimizer.zero_grad()

            # Phase 1: reproduce the legacy PPO objective and retain its mean
            # action, which supplies the differentiable nominal-torque input
            # to the BARD force prediction path.
            ppo_loss, surrogate_loss, value_loss, current_actions, policy_features = self._compute_rl_loss(obs_batch, obs_hist_batch, actions_batch,
                                                                                          critic_obs_batch, old_sigma_batch, old_mu_batch,
                                                                                          old_actions_log_prob_batch,
                                                                                          advantages_batch, target_values_batch, returns_batch)

            # Rebuild the same stochastic action path under the current
            # policy, then select the source chosen by the rollout's exact
            # delay. This is the only nominal torque used by HardPACT physics.
            replay = self._replay_action_path(
                current_actions, obs_batch,
                self.storage.current_hard_pact_batch,
                action_func, fb_func, default_pose, qvel_scale,
            )

            
            # BARD retains the legacy warmup, while projection is deliberately
            # active from iteration zero and receives only this epoch's shard.
            qp_rows = self._qp_rows_for_epoch(ppo_epoch, itr)
            self._qp_sampling_gradient_inputs = None
            if self.ppo_qp_sampling_logging_enabled and qp_rows is not None:
                qp_batch = self.storage.current_hard_pact_batch
                anchors = qp_batch["sampled_qp_substep_index"][qp_rows].reshape(-1)
                valid_qp = ~(
                    qp_batch["push_event_mask"][qp_rows].bool()
                    | qp_batch["reset_mask"][qp_rows].bool()
                    | qp_batch["timeout_mask"][qp_rows].bool()
                    | qp_batch["teleport_mask"][qp_rows].bool()
                )
                qp_sampled_count.add_(float(qp_rows.numel()))
                qp_valid_count.add_(valid_qp.float().sum())
                qp_anchor_zero_count.add_((anchors == 0).float().sum())
                qp_anchor_two_count.add_((anchors == 2).float().sum())
                if ppo_epoch == 0:
                    full = float(obs_batch.shape[0])
                    qp_full_count.add_(full)
                    expected = (
                        self.num_learning_epochs
                        if self.ppo_qp_sampling == "all"
                        else self.ppo_qp_passes_per_iteration
                    )
                    qp_target_count.add_(full * expected)
            pinn_loss = None
            if pinn_iteration_ready or qp_rows is not None:
                pinn_loss = self._compute_bard_loss(
                    replay["nominal_torque"], obs_batch, obs_hist_batch,
                    gt_forces_batch, default_pose,
                    replay["desired_position"], replay["feedforward_torque"],
                    fb_func, policy_features=policy_features,
                    qp_rows=qp_rows, compute_pinn=pinn_iteration_ready,
                )
            optimize_physics = (
                pinn_iteration_ready
                or (
                    qp_rows is not None
                    and self.hard_pact_features.projection_loss
                )
            )
            ppo_losses = (
                [ppo_loss, pinn_loss]
                if optimize_physics and pinn_loss is not None
                else [ppo_loss]
            )
            # Full diagnostics measure the real PCGrad backward containing
            # qpth's implicit KKT derivative. qpth does not expose a timer for
            # its backward alone, so this metric is named accordingly. No
            # synchronization or memory reset occurs at other levels.
            qp_audit_inputs = self._qp_full_audit_inputs
            qp_backward_start = None
            qp_backward_memory_start = None
            if qp_audit_inputs is not None:
                audit_device = next(iter(qp_audit_inputs.values())).device
                if audit_device.type == "cuda":
                    torch.cuda.synchronize(audit_device)
                    torch.cuda.reset_peak_memory_stats(audit_device)
                    qp_backward_memory_start = torch.cuda.memory_allocated(
                        audit_device
                    )
                qp_backward_start = time.perf_counter()

            # PCGrad treats reward learning as the primary objective and
            # removes the reward-parallel component of the BARD gradient.
            pcgrad_timing = self._start_bard_timing("pcgrad", obs_batch)
            if optimize_physics and self.pinn_weight_final >= 0:
                self.act_optimizer.pc_backward_pinn(
                    ppo_losses, record_diagnostics=self._pcgrad_audit_ran
                )
            elif optimize_physics and self.pinn_weight_final < 0:
                self.act_optimizer.pc_backward_ppgrad(
                    ppo_losses, record_diagnostics=self._pcgrad_audit_ran
                )
            else:
                self.act_optimizer.pc_backward(
                    ppo_losses, record_diagnostics=self._pcgrad_audit_ran
                )
            self._stop_bard_timing("pcgrad", pcgrad_timing)
            if (
                self.ppo_qp_sampling_logging_enabled
                and self._qp_sampling_gradient_inputs is not None
            ):
                for value in self._qp_sampling_gradient_inputs:
                    if value.grad is not None:
                        qp_gradient_square_sum.add_(
                            value.grad.detach().float().square().sum()
                        )
            if self.hard_pact_qp is not None:
                self.last_qp_metrics.update(
                    self.hard_pact_qp._last_gradient_metrics
                )
            if qp_audit_inputs is not None:
                audit_device = next(iter(qp_audit_inputs.values())).device
                if audit_device.type == "cuda":
                    torch.cuda.synchronize(audit_device)
                self.last_qp_metrics["qp/full/pcgrad_backward_time_ms"] = (
                    torch.tensor(
                        (time.perf_counter() - qp_backward_start) * 1000.0,
                        device=audit_device, dtype=torch.float32,
                    )
                )
                if audit_device.type == "cuda":
                    peak = (
                        torch.cuda.max_memory_allocated(audit_device)
                        - qp_backward_memory_start
                    ) / (1024.0 ** 2)
                    self.last_qp_metrics[
                        "qp/full/pcgrad_backward_peak_cuda_mib"
                    ] = torch.tensor(
                        peak, device=audit_device, dtype=torch.float32
                    )
                for name, value in qp_audit_inputs.items():
                    gradient = value.grad
                    if gradient is None:
                        norm = value.new_zeros((), dtype=torch.float32)
                        finite_fraction = value.new_ones((), dtype=torch.float32)
                    else:
                        finite = torch.isfinite(gradient)
                        norm = torch.linalg.vector_norm(
                            torch.where(finite, gradient, torch.zeros_like(gradient))
                        ).detach().float()
                        finite_fraction = finite.float().mean().detach()
                    self.last_qp_metrics[f"qp/full/gradient_norm/{name}"] = norm
                    self.last_qp_metrics[
                        f"qp/full/gradient_finite_fraction/{name}"
                    ] = finite_fraction
                self._qp_full_audit_inputs = None
            if len(ppo_losses) == 2 and self._pcgrad_audit_ran:
                reward_gradient, physics_gradient = (
                    self.act_optimizer.last_objective_grads
                )
                finite = torch.isfinite(physics_gradient)
                reward_finite = torch.isfinite(reward_gradient)
                safe_reward = torch.where(
                    reward_finite, reward_gradient, torch.zeros_like(reward_gradient)
                )
                safe_physics = torch.where(
                    finite, physics_gradient, torch.zeros_like(physics_gradient)
                )
                cosine = torch.dot(safe_reward, safe_physics) / (
                    safe_reward.norm() * safe_physics.norm()
                ).clamp_min(1.0e-12)
                self.last_physics_gradient_metrics = {
                    "physics_gradient/finite_fraction": finite.float().mean().detach(),
                    "physics_gradient/nonfinite_count": (~finite).sum().detach(),
                    "physics_gradient/finite_norm": safe_physics.norm().detach(),
                    "grad/objective/ppo_norm": safe_reward.norm().detach(),
                    "grad/objective/physics_norm": safe_physics.norm().detach(),
                    "grad/objective/physics_zero_fraction": (
                        safe_physics == 0
                    ).float().mean().detach(),
                    "grad/pcgrad/cosine": cosine.detach(),
                    "grad/pcgrad/conflict_fraction": (
                        cosine < 0
                    ).float().detach(),
                }
            else:
                nan = torch.full((), float("nan"), device=obs_batch.device)
                self.last_physics_gradient_metrics = {
                    "grad/pcgrad/audit_ran": torch.zeros(
                        (), device=obs_batch.device
                    ),
                    "grad/pcgrad/cosine": nan,
                    "grad/pcgrad/conflict_fraction": nan,
                }

            # Module/group norms are computed from the merged gradients that
            # will actually be stepped. They are reductions on-device; only
            # the scalar norms reach TensorBoard.
            for group in (
                self.act_optimizer.optimizer.param_groups
                if self._pcgrad_audit_ran else ()
            ):
                gradients = [
                    parameter.grad.reshape(-1) for parameter in group["params"]
                    if parameter.grad is not None
                ]
                if gradients:
                    flat = torch.cat(gradients)
                    finite = torch.isfinite(flat)
                    name = group.get("name", "unnamed").replace("/", "_")
                    self.last_physics_gradient_metrics[
                        f"grad/module/{name}_norm"
                    ] = torch.where(finite, flat, torch.zeros_like(flat)).norm().detach()
                    self.last_physics_gradient_metrics[
                        f"grad/module/{name}_nonfinite_fraction"
                    ] = (~finite).float().mean().detach()
            if self._pcgrad_audit_ran:
                self.last_physics_gradient_metrics[
                    "grad/pcgrad/audit_ran"
                ] = torch.ones((), device=obs_batch.device)
            
            nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm)
            self.act_optimizer.step()

            # Perform some logging
            mean_value_loss += value_loss.detach()
            mean_surrogate_loss += surrogate_loss.detach()
            if pinn_loss is not None:
                # ``pinn_loss`` is the optimization objective and therefore
                # contains ``pinn_weight`` (and potentially the independent
                # QP projection term).  Console/TensorBoard should expose the
                # underlying PINN magnitude, not its scheduled contribution.
                mean_pinn_loss += self.last_unweighted_pinn_loss


            # Phase 2: recompute and aggregate every auxiliary term before a
            # single backward/step, following the B1Z1 adaptation phase.
            for enc_epoch in range(self.num_enc_epochs):
                self.actor_critic.train()
                self.decoder.train()

                auxiliary_timing = self._start_bard_timing(
                    "auxiliary", obs_batch
                )
                aux = self._compute_auxiliary_loss(
                    obs_hist_batch, obs_target, explicit_labels_batch, grf_target,
                    terminated_batch, replay["nominal_torque"].detach(),
                    self.storage.current_hard_pact_batch,
                )
                self.auxiliary_optimizer.zero_grad(set_to_none=True)
                aux["loss"].backward()
                nn.utils.clip_grad_norm_(self.auxiliary_parameters, self.max_grad_norm)
                self.auxiliary_optimizer.step()
                self._stop_bard_timing("auxiliary", auxiliary_timing)
                vae_loss, kl_div = aux["loss"], aux["kl"]
                recon_error, vel_pred_error = aux["privileged"], aux["explicit"]
                dec_loss = aux["privileged"]
                decode_targets, recons = obs_target, aux["reconstruction"]
                self.last_auxiliary_metrics = {
                    name: value.detach() for name, value in aux.items()
                    if name not in ("loss", "reconstruction")
                }

                # Log the decode targets and recons for computing boot-probability
                with torch.no_grad():
                    x = decode_targets * terminated_batch
                    r = recons * terminated_batch

                    # flatten batch dimension only; keep feature dim
                    # assumes x shape [B, D]
                    if boot_sum_x is None:
                        boot_sum_x = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float64)
                        boot_sum_x2 = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float64)

                    x64 = x.to(torch.float64)
                    r64 = r.to(torch.float64)

                    boot_sum_x += x64.sum(dim=0)
                    boot_sum_x2 += (x64 * x64).sum(dim=0)

                    # scalar sum over all elements
                    boot_sum_recon_sqerr += ((r64 - x64) ** 2).sum()

                    boot_count += x.shape[0]

                # Log losses
                mean_autoenc_loss += vae_loss.detach()
                mean_vel_loss += vel_pred_error.detach()
                mean_recon_loss += recon_error.detach()
                mean_kld_loss += kl_div.detach()
                mean_decoder_loss += dec_loss.detach()

            # Keeps the interaction of incoming data with layer wieghts below the threashold that 
            #     saturates the tanh activation function.
            self.spectral_normalization(self.actor_critic, sigma_max=6.0)

        if itr > self.pinn_init:
            self.num_pinn_updates += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_pinn_loss /= num_updates

        mean_autoenc_loss /= (num_updates * self.num_enc_epochs)
        mean_decoder_loss /= (num_updates * self.num_enc_epochs)
        mean_kld_loss /= (num_updates * self.num_enc_epochs)
        mean_vel_loss /= (num_updates * self.num_enc_epochs)
        mean_recon_loss /= (num_updates * self.num_enc_epochs)

        if self.profile_bard_timing:
            self._finalize_bard_timing(num_updates)

        if self.ppo_qp_sampling_logging_enabled:
            sampled_denominator = qp_sampled_count.clamp_min(1.0)
            self.last_qp_metrics.update({
                "qp/minimal/sampling/mode_disjoint": metric_zero.new_tensor(
                    float(self.ppo_qp_sampling == "disjoint_epoch_partition")
                ),
                "qp/minimal/sampling/mode_all": metric_zero.new_tensor(
                    float(self.ppo_qp_sampling == "all")
                ),
                "qp/minimal/sampling/shard_percentage": metric_zero.new_tensor(
                    self.ppo_qp_shard_percentage
                ),
                "qp/minimal/sampling/passes_per_iteration": metric_zero.new_tensor(
                    float(self.ppo_qp_passes_per_iteration)
                ),
                "qp/minimal/sampling/sampled_count": qp_sampled_count,
                "qp/minimal/sampling/full_count": qp_full_count,
                "qp/minimal/sampling/coverage": (
                    qp_sampled_count / qp_target_count.clamp_min(1.0)
                ),
                "qp/minimal/sampling/anchor_0_fraction": (
                    qp_anchor_zero_count / sampled_denominator
                ),
                "qp/minimal/sampling/anchor_2_fraction": (
                    qp_anchor_two_count / sampled_denominator
                ),
                "qp/minimal/sampling/valid_fraction": (
                    qp_valid_count / sampled_denominator
                ),
                "qp/minimal/sampling/gradient_norm": (
                    qp_gradient_square_sum.sqrt()
                ),
            })

        # Calculate the total bootstrapping probability over the performance of the autoencoder on all of the above
        #      total number of scalar elements per sample vector
        feat_dim = boot_sum_x.shape[0]

        mean_pred = boot_sum_x / boot_count                     # [D]
        ex2 = boot_sum_x2 / boot_count                          # [D]
        var = torch.clamp(ex2 - mean_pred**2, min=0.0)          # [D]
        mean_pred_error = var.mean()
        actual_pred_error = boot_sum_recon_sqerr / (boot_count * feat_dim)
        ratio = mean_pred_error / (actual_pred_error * self.boot_mult + 1e-8)
        # This is the update's single mandatory scalar synchronization. All
        # minibatch losses and diagnostics stayed on-device until now.
        pboot = float(torch.tanh(ratio).detach().cpu())

        # Use the (scaled) ratio of mean-prediction performance to actual prediction performance
        #     to determine if encoder bootstrapping is performed.
        self.use_boot = random.random() < pboot
        if self.console_debug:
            print("Use bootstrapped Encoder Dynamics: ", self.use_boot)

        self._clear_rollout_mechanics_cache()
        self.storage.clear()
        results = torch.stack((
            mean_value_loss, mean_surrogate_loss, mean_autoenc_loss,
            mean_decoder_loss, mean_vel_loss, mean_recon_loss,
            mean_kld_loss, mean_pinn_loss,
        )).detach().cpu().tolist()
        return tuple(results)

    def _compute_rl_loss(self, obs_batch, obs_hist_batch,
                         actions_batch, critic_obs_batch,
                         old_sigma_batch, old_mu_batch,
                         old_actions_log_prob_batch,
                         advantages_batch, target_values_batch, returns_batch,
                         latent_noise=None):
        if self.use_boot:
            self.actor_critic.act(
                obs_batch, obs_hist_batch, latent_noise=latent_noise
            )
        else:
            self.actor_critic.act_bootmask(
                obs_batch, obs_hist_batch, latent_noise=latent_noise
            )

        # Pull out the current actions for use later
        current_actions = torch.cat([self.actor_critic.mean_pos, self.actor_critic.mean_tau], dim=-1)

        # PPO stuff
        #    - Position Control
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        value_batch            = self.actor_critic.evaluate(critic_obs_batch)
        mu_batch               = self.actor_critic.action_mean
        sigma_batch            = self.actor_critic.action_std
        entropy_batch          = self.actor_critic.entropy

        # Now calculate the PPO/SPO losses
        # KL
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-6, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                
                for param_group in self.act_optimizer.optimizer.param_groups:
                    # specifically modifies the learning rate of the actor-control specific parameters
                    if "name" in param_group.keys():
                        if "actor" in param_group["name"]:
                            param_group['lr'] = self.learning_rate

        # PPO Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # PPO stuff
        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        if self.use_adaptive_entropy: 
            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.current_entropy_coef * entropy_batch.mean()
        else:
            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()        

        policy_features = (
            self.actor_critic.cenet_z,
            self.actor_critic.cenet_torso_velo,
        )
        return (
            ppo_loss, surrogate_loss, value_loss, current_actions,
            policy_features,
        )

    @staticmethod
    def _masked_mse(prediction, target, mask):
        per_sample = (prediction - target).square().mean(dim=-1)
        weights = mask.reshape(-1).to(per_sample.dtype)
        return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)

    @staticmethod
    def _masked_explicit_loss(prediction, target, mask):
        """MSE for continuous estimates and BCE for contact probabilities."""
        if prediction.shape[-1] != 11 or target.shape[-1] != 11:
            raise ValueError("HardPACT explicit estimates and labels must be 11-D")
        target = target.detach()
        element_loss = torch.cat((
            (prediction[:, :3] - target[:, :3]).square(),
            F.binary_cross_entropy(
                prediction[:, 3:7], target[:, 3:7], reduction="none"
            ),
            (prediction[:, 7:11] - target[:, 7:11]).square(),
        ), dim=-1)
        per_sample = element_loss.mean(dim=-1)
        weights = mask.reshape(-1).to(per_sample.dtype)
        return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)

    def _start_bard_timing(self, name, reference):
        """Start one asynchronous forward-only PINN-loss measurement."""
        if not self.profile_bard_timing:
            return None
        if reference.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return start, end
        return time.perf_counter()

    def _stop_bard_timing(self, name, token):
        if token is None:
            return
        if isinstance(token, tuple):
            token[1].record()
        else:
            token = (time.perf_counter() - token) * 1000.0
        self._bard_timing_records[name].append(token)

    def _finalize_bard_timing(self, num_updates):
        """Publish summed and per-minibatch timings after one synchronization."""
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for name, records in self._bard_timing_records.items():
            total_ms = 0.0
            for token in records:
                if isinstance(token, tuple):
                    total_ms += token[0].elapsed_time(token[1])
                else:
                    total_ms += token
            value = torch.tensor(total_ms, device=self.device, dtype=torch.float32)
            self.last_physics_loss_metrics[
                f"physics/timing/{name}_forward_ms_per_update"
            ] = value
            self.last_physics_loss_metrics[
                f"physics/timing/{name}_forward_ms_per_minibatch"
            ] = value / max(int(num_updates), 1)
            self.last_physics_loss_metrics[
                f"physics/timing/{name}_chunk_count"
            ] = torch.tensor(
                len(records), device=self.device, dtype=torch.float32
            )
        dynamics_total = self.last_physics_loss_metrics.get(
            "physics/timing/dynamics_forward_ms_per_update",
            torch.zeros((), device=self.device),
        )
        transfer_total = 0.0
        if self.physics_dynamics is not None and hasattr(
            self.physics_dynamics, "timing_metrics"
        ):
            backend = self.physics_dynamics.timing_metrics(reset=True)
            dynamics_total = backend["dynamics_ms"]
            transfer_total = backend["transfer_ms"]
            self.last_physics_loss_metrics[
                "physics/timing/dynamics_call_count"
            ] = torch.tensor(
                backend["calls"], device=self.device, dtype=torch.float32
            )
        self.last_physics_loss_metrics[
            "physics/timing/dynamics_total_ms_per_update"
        ] = torch.as_tensor(
            dynamics_total, device=self.device, dtype=torch.float32
        )
        self.last_physics_loss_metrics[
            "physics/timing/pinocchio_transfer_ms_per_update"
        ] = torch.as_tensor(
            transfer_total, device=self.device, dtype=torch.float32
        )

    def _compute_auxiliary_loss(
        self, history, privileged_target, explicit_target, grf_target,
        valid, nominal_torque, transition,
    ):
        r"""Compute every decoder term on one shared stochastic VAE graph.

        During training, privileged reconstruction, explicit estimation, and
        both deployment heads share ``z = mu + exp(logvar/2) eps``.  The heads
        internally stop gradients through ``e = D_e(z)`` while retaining
        gradients through ``z`` (and through nominal torque for the GRF head).
        Runtime policy conditioning and inference remain deterministic on
        ``mu``.
        """
        mean, logvar = self.actor_critic.context_encoder(history)
        sample = self.actor_critic.context_encoder.reparameterization_trick(
            mean, logvar
        )
        explicit = self.actor_critic.explicit_estimator(sample)
        reconstruction = self.decoder(torch.cat((sample, explicit), dim=-1))
        heads = self.actor_critic.physics_heads(sample, explicit, nominal_torque)

        privileged = self._masked_mse(
            reconstruction, privileged_target.detach(), valid
        )
        explicit_loss = self._masked_explicit_loss(
            explicit, explicit_target.detach(), valid
        )
        per_sample_kl = -0.5 * torch.sum(
            1 + logvar - mean.square() - logvar.exp(), dim=-1
        )
        weights = valid.reshape(-1).to(per_sample_kl.dtype)
        kl = (per_sample_kl * weights).sum() / weights.sum().clamp_min(1.0)

        grf = normalized_huber_loss(
            heads.grf_yaw_scaled, grf_target.detach(),
            self.actor_critic.physics_estimator.grf_scale, valid,
        )
        if transition is None:
            wrench_target = heads.base_wrench_yaw_scaled.detach().new_zeros(
                heads.base_wrench_yaw_scaled.shape
            )
            active = torch.zeros_like(valid, dtype=torch.bool)
        else:
            wrench_target = transition[
                "total_external_wrench_label_yaw_scaled"
            ].detach()
            active = transition["sustained_wrench_active_mask"].bool()
        active_mask = valid.bool() & active
        neutral_mask = valid.bool() & ~active
        wrench_active = normalized_huber_loss(
            heads.base_wrench_yaw_scaled, wrench_target,
            self.actor_critic.physics_estimator.wrench_scale, active_mask,
        )
        wrench_neutral = normalized_huber_loss(
            heads.base_wrench_yaw_scaled, wrench_target,
            self.actor_critic.physics_estimator.wrench_scale, neutral_mask,
        )
        loss = (
            self.privileged_loss_weight * privileged
            + self.vae_beta * kl
            + self.explicit_loss_weight * explicit_loss
            + self.grf_loss_weight * grf
            + self.active_wrench_loss_weight * wrench_active
            + self.neutral_wrench_loss_weight * wrench_neutral
        )
        return {
            "loss": loss,
            "privileged": privileged,
            "kl": kl,
            "explicit": explicit_loss,
            "grf": grf,
            "wrench_active": wrench_active,
            "wrench_neutral": wrench_neutral,
            "reconstruction": reconstruction,
        }

    @staticmethod
    def _nominal_torque(actions, observations, action_func, fb_func,
                        default_pose, qvel_scale):
        desired_position, feedforward_torque = action_func(actions)
        joint_position = observations[:, 9:21] + default_pose
        joint_velocity = observations[:, 21:33] / qvel_scale
        return feedforward_torque + fb_func(
            desired_position, joint_position, joint_velocity
        )

    def _unweighted_pinn_loss(
        self, inverse_loss, rollout_loss, soft_constraint_loss=None,
    ):
        """Compose PINN terms without the outer curriculum/training weight."""
        if soft_constraint_loss is None:
            soft_constraint_loss = inverse_loss * 0.0
        return (
            self.lambda_inverse * inverse_loss
            + self.lambda_rollout * rollout_loss
            + self.lambda_soft_constraint * soft_constraint_loss
        )

    def _combine_bard_losses(
        self, inverse_loss, rollout_loss, projection_loss_value=None,
        soft_constraint_loss=None, pinn_weight=1.0,
    ):
        r"""Apply the PINN schedule only to soft dynamics objectives.

        ``L_phys = w_PINN*(lambda_ID L_ID + lambda_roll L_roll
        + lambda_soft L_soft) + lambda_proj L_proj``.  In particular, the
        projection objective is active from iteration zero independently of
        the legacy PINN warmup.
        """
        if projection_loss_value is None:
            projection_loss_value = inverse_loss * 0.0
        if soft_constraint_loss is None:
            soft_constraint_loss = inverse_loss * 0.0
        return (
            float(pinn_weight) * self._unweighted_pinn_loss(
                inverse_loss, rollout_loss, soft_constraint_loss
            )
            + self.lambda_projection * projection_loss_value
        )

    def _soft_constraint_loss(self, grf_world):
        """Cheap differentiable force-cone penalty used only by its ablation.

        This path intentionally does not build BARD state or a QP.  It softly
        enforces ``fz >= 0`` and the same fixed-normal friction pyramid
        ``|fx|,|fy| <= mu*fz`` used by the hard projection.
        """
        mu = float(self.qp_config.friction_coefficient)
        normal = grf_world[..., 2]
        violations = torch.stack((
            F.relu(-normal),
            F.relu(grf_world[..., 0].abs() - mu * normal),
            F.relu(grf_world[..., 1].abs() - mu * normal),
        ), dim=-1)
        return (violations / float(self.qp_config.force_scale_n)).square().mean()

    def _policy_raw_action_from_noise(self, observation, history, noise):
        """Reparameterize one stored policy draw under current parameters."""
        # Action replay follows the deterministic policy-conditioning path.
        # The stored VAE noise is still consumed by the auxiliary/physics
        # heads, but is deliberately absent from the policy mean.
        latent, _ = self.actor_critic.context_encoder(history)
        explicit = self.actor_critic.explicit_estimator(latent)
        if self.use_boot:
            conditioning = torch.cat((observation, latent, explicit), dim=-1)
        else:
            conditioning = torch.cat((
                observation,
                torch.zeros_like(torch.cat((latent, explicit), dim=-1)),
            ), dim=-1)
        mean_position, mean_torque = self.actor_critic.actor_forward(conditioning)
        mean = torch.cat((mean_position, mean_torque), dim=-1)
        return mean + self.actor_critic.std.unsqueeze(0) * noise.detach()

    def _replay_action_path(
        self, current_mean, observation, transition,
        action_func, fb_func, default_pose, qvel_scale,
    ):
        r"""Replay sampling, clipping, discrete delay, and nominal torque.

        The raw current draw is ``mu_current + sigma_current * epsilon_t``.
        The delayed source is reconstructed from the source observation/history
        resolved by rollout storage and its own stored noise. Both are clipped
        exactly as in the environment before the stored delay choice is
        applied. Invalid reset/boundary queue entries replay as constant zeros.
        """
        if transition is None:
            raise RuntimeError("HardPACT action replay requires named fields")
        current_raw = (
            current_mean
            + self.actor_critic.std.unsqueeze(0)
            * transition["standardized_action_noise"].detach()
        )
        # Delay zero points at the policy evaluation already performed for
        # PPO, so reuse that exact graph. Only delayed rows require a second
        # history/actor evaluation; the max-delay-zero task avoids it wholly.
        configured_delay = getattr(
            getattr(self, "storage", None), "max_action_delay", None
        )
        if configured_delay == 0:
            source_raw = current_raw
        else:
            sampled_delay = transition.get("sampled_action_delay")
            delayed_rows = (
                torch.ones(
                    current_raw.shape[0], device=current_raw.device,
                    dtype=torch.bool,
                )
                if sampled_delay is None else sampled_delay.reshape(-1).ne(0)
            )
            source_raw = current_raw.clone()
            source_raw[delayed_rows] = self._policy_raw_action_from_noise(
                transition["delayed_source_observation"][delayed_rows].detach(),
                transition["delayed_source_history"][delayed_rows].detach(),
                transition["delayed_source_noise"][delayed_rows],
            )
        current_transformed = torch.clamp(
            current_raw, -self.action_clip, self.action_clip
        )
        source_transformed = torch.clamp(
            source_raw, -self.action_clip, self.action_clip
        )
        source_valid = transition["delayed_action_source_valid"].bool()
        delayed_action = torch.where(
            source_valid, source_transformed,
            torch.zeros_like(source_transformed),
        )
        desired_position, feedforward_torque = action_func(delayed_action)
        if feedforward_torque.shape[-1] == 0:
            feedforward_torque = torch.zeros_like(desired_position)
        joint_position = observation.detach()[:, 9:21] + default_pose
        joint_velocity = observation.detach()[:, 21:33] / qvel_scale
        nominal_torque = feedforward_torque + fb_func(
            desired_position, joint_position, joint_velocity
        )
        return {
            "raw_action": current_raw,
            "transformed_action": current_transformed,
            "delayed_action": delayed_action,
            "desired_position": desired_position,
            "feedforward_torque": feedforward_torque,
            "nominal_torque": nominal_torque,
        }

    def _compute_bard_loss(
        self, nominal_torque, obs_batch, obs_hist_batch,
        measured_generalized_contact_force, default_pose,
        desired_position=None, feedforward_torque=None, fb_func=None,
        policy_features=None, qp_rows=None, compute_pinn=True,
    ):
        r"""Evaluate interval BARD losses and one sampled substep projection.

        Simulator states are first mapped to canonical BARD coordinates.  The
        same context installs realized randomized dynamics, updates kinematics,
        and computes Jacobians exactly once for both PINN objectives.  The
        inverse objective observes

            vdot_obs = (v_{t+1}^BARD - v_t^BARD) / Delta t.

        BARD then evaluates RNEA with the realized inertial/passive parameters.
        With actuation ``tau_a=[0_6,tau_exec]``, learned contact force ``F``
        and applied wrench ``W_applied=W_total-W_mass/CoM``, the residual is

            r_ID = RNEA(q_t,v_t,vdot_obs;theta_rand)
                   - tau_a - J_f^T F - J_b^T W_applied.

        Measured quantities and BARD terms are detached; gradients reach only
        the learned GRF and total-wrench outputs.

        The rollout objective instead forms

            g = S^T tau_control + J_f^T F_hat + J_b^T W_hat_applied,
            M_eff = CRBA(q_t;theta_rand) + D_armature,
            b = RNEA(q_t,v_t,0;theta_rand),
            Delta v_hat = Delta t * M_eff^{-1}(g-b),
            Delta v_obs = v_{t+1} - v_t.

        Its base-linear, base-angular, and joint residual blocks are normalized
        by ``Delta t * [10,20,100]``. Each block contributes its residual RMS
        divided by ``1 + stopgrad(observed-motion RMS)``, and the three block
        scores are averaged so the 12 joint coordinates cannot dominate only
        because that block is wider.

        It retains gradients through ``tau_control`` and both physics heads.
        The detached 18x18 solve uses an RHS-only custom VJP; official BARD
        ABA remains a test/reference path. Before minibatch iteration, one
        detached actual-mechanics cache materializes ``M,h,J_f,J_b`` at every
        control transition. A distinct nominal/deployment cache materializes
        the selected substep QP mechanics. Shuffled PPO epochs only index these
        tensors; no BARD/Pinocchio work or autograd graph is repeated. The QP
        itself is still solved exactly once per sampled minibatch transition.
        """
        if not (self.bard_enabled or self.hard_pact_features.soft_constraint_penalty):
            zero = nominal_torque.sum() * 0.0
            self.last_unweighted_pinn_loss = zero.detach()
            return zero
        qp_ready = (
            self.hard_pact_features.execution_qp
            and self.hard_pact_qp is not None
            and qp_rows is not None
        )
        if not ((compute_pinn and (
                self.bard_inverse_enabled or self.bard_rollout_enabled)) or qp_ready
                or self.hard_pact_features.soft_constraint_penalty):
            zero = nominal_torque.sum() * 0.0
            self.last_unweighted_pinn_loss = zero.detach()
            return zero
        batch = self.storage.current_hard_pact_batch
        if batch is None:
            raise RuntimeError("HardPACT BARD loss requires named transition fields")
        control_dt = batch["control_dt"].detach().clamp_min(1.0e-8)
        mass_wrench = batch["equivalent_mass_com_wrench_world"].detach()
        zero = nominal_torque.sum() * 0.0
        if policy_features is None:
            _, _, latent, explicit = self.actor_critic.cenet_enc_forward(
                obs_hist_batch
            )
        else:
            latent, explicit = policy_features
        grf_world = None
        applied_at_base = None
        total_at_base = None
        wrench_yaw = None

        def world_wrench(q, label_mass_wrench, wrench_prediction, com_shift):
            total = torch.cat((
                _yaw_local_to_world(wrench_prediction[:, :3].unsqueeze(1), q[:, 3:7]).squeeze(1),
                _yaw_local_to_world(wrench_prediction[:, 3:].unsqueeze(1), q[:, 3:7]).squeeze(1),
            ), dim=-1)
            base = q[:, :3].detach()
            com = base + _body_point_to_world(
                com_shift.detach(), q.detach()
            )
            applied = wrench_at_point(total - label_mass_wrench, com, base)
            return applied, applied + label_mass_wrench

        if compute_pinn:
            grf_yaw = self.actor_critic.physics_estimator.predict_grf(
                latent, explicit, nominal_torque
            ).reshape(-1, 4, 3)
            grf_world = _yaw_local_to_world(
                grf_yaw / self.grf_observation_scale,
                batch["pre_q"][:, 3:7],
            )
            if self.bard_inverse_enabled or self.bard_rollout_enabled:
                wrench_yaw = self.actor_critic.physics_estimator.predict_wrench(
                    latent, explicit
                ) / self.base_wrench_observation_scale
                applied_at_base, total_at_base = world_wrench(
                    batch["pre_q"], mass_wrench, wrench_yaw,
                    batch["realized_com_shift_body"],
                )

        # BARD's interval objectives intentionally retain their control-rate
        # state and logged interval-average executed torque.  A straight-
        # through value preserves the earlier rollout gradient contract while
        # making the forward-dynamics value exactly the torque Genesis executed.
        inverse_loss = zero
        rollout_loss = zero
        self.last_inverse_dynamics_metrics = {}
        self.last_rollout_dynamics_metrics = {}
        if compute_pinn and (
            self.bard_inverse_enabled or self.bard_rollout_enabled
        ):
            flat_indices = self.storage.current_batch_indices
            if self._rollout_actual_mechanics is None:
                # Direct unit-test/compatibility calls may bypass update().
                # Production update() always materializes the full cache once.
                actual_parameters = {
                    "added_base_mass": batch["realized_added_mass"].detach(),
                    "base_com_shift": batch["realized_com_shift_body"].detach(),
                    "joint_armature": batch["joint_armature"].detach(),
                    "joint_friction": batch["joint_friction"].detach(),
                    "joint_stiffness": batch["joint_stiffness"].detach(),
                    "joint_damping": batch["joint_damping"].detach(),
                }
                dynamics_timing = self._start_bard_timing(
                    "dynamics", nominal_torque
                )
                mechanics_root = self._materialize_mechanics_cache(
                    kind="actual", q=batch["pre_q"], v=batch["pre_v"],
                    parameters=actual_parameters, post_v=batch["post_v"],
                    mass_com_wrench_world=mass_wrench,
                )
                self._stop_bard_timing("dynamics", dynamics_timing)
                mechanics_indices = None
            else:
                mechanics_root = self._rollout_actual_mechanics
                mechanics_indices = flat_indices
            valid_all = ~(
                batch["push_event_mask"].bool() | batch["reset_mask"].bool()
                | batch["timeout_mask"].bool() | batch["teleport_mask"].bool()
            )
            valid_total = valid_all.reshape(-1).sum().to(control_dt.dtype)
            inverse_numerator = zero
            rollout_numerator = zero
            inverse_metric_sums = {}
            rollout_metric_sums = {}
            capacity = self.physics_dynamics.batch_capacity
            for start in range(0, nominal_torque.shape[0], capacity):
                stop = min(start + capacity, nominal_torque.shape[0])
                sl = slice(start, stop)
                chunk = mechanics_root.index(
                    sl if mechanics_indices is None else mechanics_indices[sl]
                )
                context = chunk.as_context(self.physics_dynamics)
                count = valid_all[sl].reshape(-1).sum().to(control_dt.dtype)
                if self.bard_inverse_enabled:
                    observed_acceleration = (
                        chunk.post_v_canonical - chunk.pre_v_canonical
                    ) / control_dt[sl]
                    timing = self._start_bard_timing("inverse", nominal_torque)
                    # RNEA(q,v,a)=M_eff(q)*a+h(q,v) for fixed realized
                    # mechanics. M_eff already includes armature and h already
                    # includes the configured passive joint forces.
                    required_force = (
                        torch.einsum(
                            "bij,bj->bi", chunk.mass_matrix,
                            observed_acceleration,
                        ) + chunk.bias
                    )
                    inverse = corrected_bard_inverse_dynamics_loss(
                        required_generalized_force=required_force,
                        foot_jacobians=context.foot_jacobians,
                        base_jacobian=context.base_jacobian,
                        interval_executed_torque=batch["interval_executed_torque"][sl],
                        interval_grf_world=grf_world[sl],
                        total_wrench_world=total_at_base[sl],
                        mass_com_wrench_world=chunk.mass_com_wrench_world,
                        measured_generalized_contact_force=measured_generalized_contact_force[sl],
                        push_event_mask=batch["push_event_mask"][sl],
                        reset_mask=batch["reset_mask"][sl],
                        timeout_mask=batch["timeout_mask"][sl],
                        teleport_mask=batch["teleport_mask"][sl],
                    )
                    self._stop_bard_timing("inverse", timing)
                    inverse_numerator = inverse_numerator + inverse.loss * count
                    for name, value in inverse.metrics.items():
                        inverse_metric_sums[name] = (
                            inverse_metric_sums.get(name, zero) + value * count
                        )
                if self.bard_rollout_enabled:
                    timing = self._start_bard_timing("rollout", nominal_torque)
                    rollout = differentiable_bard_rollout_loss(
                        context=context,
                        control_torque=(
                            nominal_torque[sl]
                            + batch["interval_executed_torque"][sl].detach()
                            - nominal_torque[sl].detach()
                        ),
                        interval_grf_world=grf_world[sl],
                        applied_wrench_world=applied_at_base[sl],
                        control_dt=control_dt[sl],
                        push_event_mask=batch["push_event_mask"][sl],
                        reset_mask=batch["reset_mask"][sl],
                        timeout_mask=batch["timeout_mask"][sl],
                        teleport_mask=batch["teleport_mask"][sl],
                    )
                    self._stop_bard_timing("rollout", timing)
                    rollout_numerator = rollout_numerator + rollout.loss * count
                    for name, value in rollout.metrics.items():
                        rollout_metric_sums[name] = (
                            rollout_metric_sums.get(name, zero) + value * count
                        )
            denominator = valid_total.clamp_min(1.0)
            if self.bard_inverse_enabled:
                inverse_loss = inverse_numerator / denominator
                self.last_inverse_dynamics_metrics = {
                    name: value / denominator
                    for name, value in inverse_metric_sums.items()
                }
            if self.bard_rollout_enabled:
                rollout_loss = rollout_numerator / denominator
                self.last_rollout_dynamics_metrics = {
                    name: value / denominator
                    for name, value in rollout_metric_sums.items()
                }

        soft_constraint_loss = zero
        if compute_pinn and self.hard_pact_features.soft_constraint_penalty:
            soft_constraint_loss = self._soft_constraint_loss(grf_world)

        # ---------------- sampled differentiable substep QP ---------------
        # Rollout solved every substep under no_grad.  PPO stores one compact
        # stratified-uniform sample per environment and rebuilds only that
        # sample's detached BARD matrices.  No [T,D,M,J] tensors are retained,
        # which is the principal VRAM saving.  Since P(k)=1/D, its normalized
        # correction-plus-slack value directly estimates mean_k L_proj,k.
        qp_loss = zero
        if qp_ready:
            # Slice before every QP-only head/mechanics operation. PPO, BARD,
            # and auxiliary objectives above continue to use the full batch.
            qp_batch = {name: value[qp_rows] for name, value in batch.items()}
            qp_latent = latent[qp_rows]
            qp_explicit = explicit[qp_rows]
            # q_K and v_K are rollout measurements at the selected substep K;
            # detaching makes them constants in the implicit QP derivative.
            sample_q = qp_batch.get("sampled_qp_q", qp_batch["pre_q"]).detach()
            sample_v = qp_batch.get("sampled_qp_v", qp_batch["pre_v"]).detach()
            # All hard rate/integration constraints use physics Delta t.
            sample_dt = qp_batch.get(
                "physics_dt", control_dt[qp_rows]
            ).detach()
            if desired_position is not None and fb_func is not None:
                # Replay the current stochastic/delayed policy into held q_d
                # and tau_ff, then evaluate the sampled-state PD law:
                # tau_nom,K=Kp(q_d-q_K)-Kd*qdot_K+tau_ff.
                sampled_nominal = feedforward_torque[qp_rows] + fb_func(
                    desired_position[qp_rows],
                    sample_q[:, 7:], sample_v[:, 6:]
                )
            else:
                # Compatibility path for direct legacy unit/integration calls.
                sampled_nominal = nominal_torque[qp_rows]
            # Recompute only the GRF head at K because its input includes
            # tau_nom,K. z_t and e_t remain the current policy-step features.
            sample_grf_yaw = self.actor_critic.physics_estimator.predict_grf(
                qp_latent, qp_explicit, sampled_nominal
            ).reshape(-1, 4, 3)
            # Undo observation scaling and rotate yaw-local Newtons to the
            # world-axis convention of the reconstructed sampled J_f.
            sample_grf_world = _yaw_local_to_world(
                sample_grf_yaw / self.grf_observation_scale,
                sample_q[:, 3:7],
            )
            # Projection is a deployment operation: nominal mechanics and the
            # total predicted wrench are the only quantities available on the
            # robot. In particular, neither realized randomization nor its
            # label-only mass/CoM wrench enters this path.
            qp_wrench_yaw = (
                wrench_yaw[qp_rows]
                if wrench_yaw is not None
                else self.actor_critic.physics_estimator.predict_wrench(
                    qp_latent, qp_explicit
                ) / self.base_wrench_observation_scale
            )
            sample_applied = torch.cat((
                _yaw_local_to_world(
                    qp_wrench_yaw[:, :3].unsqueeze(1), sample_q[:, 3:7]
                ).squeeze(1),
                _yaw_local_to_world(
                    qp_wrench_yaw[:, 3:].unsqueeze(1), sample_q[:, 3:7]
                ).squeeze(1),
            ), dim=-1)
            flat_indices = self.storage.current_batch_indices[qp_rows]
            if self._rollout_deployment_qp_mechanics is None:
                sampled_mechanics = self._materialize_mechanics_cache(
                    kind="deployment", q=sample_q, v=sample_v,
                    parameters={}, need_qp=True,
                )
            else:
                sampled_mechanics = (
                    self._rollout_deployment_qp_mechanics.index(flat_indices)
                )
            # qpth solves
            #   min_x 1/2*x^TQx+p^Tx,  Gx<=h, Ax=b,
            #   x=[qdd_18,f_12,tau_safe_12,s_12].
            # Gradients enter through sampled_nominal, sample_grf_world, and
            # sample_applied; all state/mechanics/rate-center tensors detach.
            sample_contact_probability = qp_explicit[:, 3:7]
            differentiate_qp = self.hard_pact_features.differentiable_qp
            qp_arguments = dict(
                # Equality coefficient M_K.
                mass_matrix=sampled_mechanics.mass_matrix,
                # Equality RHS contribution -b_K.
                bias=sampled_mechanics.bias,
                # Dynamics -J_f,K^T and contact-acceleration J_f,K blocks.
                foot_jacobians=sampled_mechanics.foot_jacobians,
                # Wrench generalized force J_b,K^T*W_hat_applied.
                base_jacobian=sampled_mechanics.base_jacobian,
                # Known affine contact acceleration Jdot_f,K*v_K.
                foot_acceleration_bias=sampled_mechanics.foot_acceleration_bias,
                # Differentiable torque tracking center.
                tau_nom=(sampled_nominal if differentiate_qp else sampled_nominal.detach()),
                # Differentiable force tracking center in world Newtons.
                force_pred_world=(sample_grf_world if differentiate_qp else sample_grf_world.detach()),
                # Differentiable world [force,moment] equality input.
                wrench_pred_world=(sample_applied if differentiate_qp else sample_applied.detach()),
                # The estimator's eps-bounded sigmoid was applied exactly
                # once; the QP consumes that probability without clipping.
                contact_probability=(sample_contact_probability if differentiate_qp else sample_contact_probability.detach()),
                # Exact rollout tau_safe,K-1 reproduces the sampled rate box.
                previous_torque=qp_batch.get(
                    "sampled_qp_previous_torque",
                    qp_batch.get(
                        "previous_executed_torque",
                        qp_batch["interval_executed_torque"],
                    ),
                ),
                # Sampled joint state defines hard one-step q/qdot boxes.
                joint_position=sample_q[:, 7:], joint_velocity=sample_v[:, 6:],
                dt=sample_dt,
                proximal_reference=qp_batch.get("sampled_qp_proximal_reference"),
            )
            if qp_arguments["proximal_reference"] is None:
                qp_arguments.pop("proximal_reference")
            if differentiate_qp:
                qp_result = self.hard_pact_qp.solve(
                    differentiable=True, **qp_arguments
                )
            else:
                # stopgrad pays only for the required forward metric. It does
                # not retain qpth's KKT graph or any physics-head activation.
                with torch.no_grad():
                    qp_result = self.hard_pact_qp.solve(
                        differentiable=False, **qp_arguments
                    )
            # m_physics=not(push or reset or timeout or teleport). Sustained
            # wrench and randomized-mass transitions deliberately remain valid.
            valid = ~(
                qp_batch["push_event_mask"].bool()
                | qp_batch["reset_mask"].bool()
                | qp_batch["timeout_mask"].bool()
                | qp_batch["teleport_mask"].bool()
            )
            # Normalize correction by the same backend torque limits used by G.
            torque_limits = self.hard_pact_qp.torque_limits.to(
                sampled_nominal.device, sampled_nominal.dtype
            )
            # For K~Uniform{0,...,D-1}, this direct sampled loss satisfies
            # E[L_K]=(1/D)sum_k L_k. There is no decimation multiplier.
            # Stage-2 rows are excluded because they have no qpth KKT graph.
            qp_loss = projection_loss(
                qp_result.tau_safe, sampled_nominal, torque_limits, valid,
                qp_result.differentiated_mask,
                contact_slack=qp_result.contact_slack,
                slack_scale=self.qp_config.slack_scale_m_s2,
            )
            # stopgrad deliberately computes and reports exactly this metric,
            # but neither it nor any QP output participates in optimization.
            if not self.hard_pact_features.projection_loss:
                qp_loss = qp_loss.detach()
            if (
                self.ppo_qp_sampling_logging_enabled
                and differentiate_qp
                and self.hard_pact_features.projection_loss
            ):
                self._qp_sampling_gradient_inputs = (
                    sampled_nominal, sample_grf_world,
                    sample_applied, sample_contact_probability,
                )
                for value in self._qp_sampling_gradient_inputs:
                    value.retain_grad()
            # Log only the newly recomputed sampled solve, whose status can
            # differ from rollout after policy/head parameters update.
            self.last_qp_metrics = dict(qp_result.metrics or {})
            self.last_qp_metrics["qp/minimal/projection_loss"] = qp_loss.detach()
            correction = (qp_result.tau_safe - sampled_nominal).detach()
            intervention = correction.abs().amax(dim=-1) > 1.0e-6
            self.last_qp_metrics["qp/minimal/intervention_fraction"] = (
                intervention.float().mean()
            )
            intervention_weights = intervention.to(correction.dtype)
            self.last_qp_metrics[
                "qp/minimal/intervention_torque_correction_rms"
            ] = (
                correction.square().mean(dim=-1).mul(intervention_weights).sum()
                / intervention_weights.sum().clamp_min(1.0)
            ).sqrt()
            audit_ran = self.hard_pact_qp._full_audit_due(differentiate_qp)
            if audit_ran:
                # Retain only four compact learned QP inputs, and only on a
                # periodic full audit. This exposes actual autograd routing
                # without retaining additional matrices during normal runs.
                self._qp_full_audit_inputs = {
                    "tau_nom": sampled_nominal,
                    "grf": sample_grf_world,
                    "wrench": sample_applied,
                    "contact": sample_contact_probability,
                }
                for value in self._qp_full_audit_inputs.values():
                    value.retain_grad()
        else:
            self.last_qp_metrics = {}
        # One weighted physics objective enters the existing PCGrad path:
        # L_phys=w_PINN*(lambda_ID*L_ID+lambda_roll*L_roll)
        #        +lambda_proj*L_proj (plus the existing soft ablation term).
        unweighted_pinn_loss = self._unweighted_pinn_loss(
            inverse_loss, rollout_loss, soft_constraint_loss
        )
        self.last_unweighted_pinn_loss = unweighted_pinn_loss.detach()
        self.last_physics_loss_metrics = {
            "physics/loss/inverse": inverse_loss.detach(),
            "physics/loss/rollout": rollout_loss.detach(),
            "physics/loss/soft_constraint": soft_constraint_loss.detach(),
            "physics/loss/pinn_unweighted": self.last_unweighted_pinn_loss,
            # This scalar makes it explicit whether the transition mask—not
            # a disabled objective—is responsible for a zero physics loss.
            "physics/valid_fraction": (
                valid_all.float().mean().detach()
                if compute_pinn and (
                    self.bard_inverse_enabled or self.bard_rollout_enabled
                )
                else zero.detach()
            ),
        }
        return self._combine_bard_losses(
            inverse_loss, rollout_loss,
            qp_loss if self.hard_pact_features.projection_loss else zero,
            soft_constraint_loss,
            pinn_weight=self.pinn_weight,
        )
