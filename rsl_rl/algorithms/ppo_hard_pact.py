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
from dataclasses import replace

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
    BardGo2Dynamics,
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
                 vae_kld_weight=1.0,   # weight of KL divergence loss in VAE
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
                 bard_inverse_enabled=True,
                 bard_rollout_enabled=True,
                 lambda_inverse=1.0,
                 lambda_rollout=1.0,
                 lambda_projection=1.0e-3,
                 lambda_soft_constraint=1.0e-3,
                 profile_bard_timing=False,
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
        self._bard_timing_records = {"inverse": [], "rollout": []}
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
        self.bard_dynamics = None
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
            self.bard_dynamics = BardGo2Dynamics(
                os.path.abspath(resolved_bard_urdf_path),
                device=self.device,
                batch_capacity=bard_batch_capacity,
                randomize_base_inertia=bard_randomize_base_inertia,
                scale_rotational_inertia=bard_scale_rotational_inertia,
            )

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

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
                                              action_shape, torso_velo_shape, grf_shape, wb_shape, self.device)

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

    def update(self, action_func, fb_func, dt, itr, default_pose, qvel_scale):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_autoenc_loss = 0
        mean_vel_loss = 0
        mean_recon_loss = 0
        mean_kld_loss = 0
        mean_decoder_loss = 0
        mean_pinn_loss = 0
        self._bard_timing_records = {"inverse": [], "rollout": []}

        boot_count = 0
        boot_sum_x = None
        boot_sum_x2 = None
        boot_sum_recon_sqerr = 0.0

        if itr > self.pinn_init and self.num_pinn_updates < (self.pinn_warmup_steps+1):
            if self.pinn_weight_final < 0:
                self.pinn_weight = 1.0
            else:
                self.pinn_weight = (float(self.num_pinn_updates)/float(self.pinn_warmup_steps))*self.pinn_weight_final

            print(self.pinn_weight)

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for terminated_batch, obs_batch, critic_obs_batch, obs_hist_batch, explicit_labels_batch, \
            grf_target, obs_target, actions_batch, target_values_batch, \
            advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, \
            old_sigma_batch, _prev_obs_batch, _prev_obs_hist_batch, gt_forces_batch, _mass_mat_batch, \
            _bias_vec_batch, _torso_accs_batch, _pprev_obs_batch, _pprev_obs_hist_batch in generator:
            
            self.actor_critic.train()
            self.act_optimizer.zero_grad()

            # Phase 1: reproduce the legacy PPO objective and retain its mean
            # action, which supplies the differentiable nominal-torque input
            # to the BARD force prediction path.
            ppo_loss, surrogate_loss, value_loss, current_actions = self._compute_rl_loss(obs_batch, obs_hist_batch, actions_batch,
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

            
            # BARD is warmed up using the same schedule as the legacy PINN.
            pinn_loss = None
            if self.pinn_weight > 0.0:
                pinn_loss = self._compute_bard_loss(
                    replay["nominal_torque"], obs_batch, obs_hist_batch,
                    gt_forces_batch, default_pose,
                    replay["desired_position"], replay["feedforward_torque"],
                    fb_func,
                )
                
            if self.pinn_weight > 0.0 and self.pinn_weight_final > 0:
                ppo_losses = [ppo_loss, self.pinn_weight * pinn_loss]
            elif self.pinn_weight > 0.0 and self.pinn_weight_final < 0:
                ppo_losses = [ppo_loss, pinn_loss]
            else:
                ppo_losses = [ppo_loss]
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
            if self.pinn_weight > 0 and self.pinn_weight_final > 0 and pinn_loss is not None:    # just being extra cautious
                self.act_optimizer.pc_backward_pinn(ppo_losses)
            elif self.pinn_weight_final < 0 and pinn_loss is not None:
                self.act_optimizer.pc_backward_ppgrad(ppo_losses)
            else:
                self.act_optimizer.pc_backward(ppo_losses)
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
            if len(ppo_losses) == 2:
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
                self.last_physics_gradient_metrics = {}

            # Module/group norms are computed from the merged gradients that
            # will actually be stepped. They are reductions on-device; only
            # the scalar norms reach TensorBoard.
            for group in self.act_optimizer.optimizer.param_groups:
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
            
            nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm)
            self.act_optimizer.step()

            # Perform some logging
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            if self.pinn_weight > 0.0:
                mean_pinn_loss += pinn_loss.item()
            else:
                mean_pinn_loss += 0.0


            # Phase 2: recompute and aggregate every auxiliary term before a
            # single backward/step, following the B1Z1 adaptation phase.
            for enc_epoch in range(self.num_enc_epochs):
                self.actor_critic.train()
                self.decoder.train()

                aux = self._compute_auxiliary_loss(
                    obs_hist_batch, obs_target, explicit_labels_batch, grf_target,
                    terminated_batch, replay["nominal_torque"].detach(),
                    self.storage.current_hard_pact_batch,
                )
                self.auxiliary_optimizer.zero_grad(set_to_none=True)
                aux["loss"].backward()
                nn.utils.clip_grad_norm_(self.auxiliary_parameters, self.max_grad_norm)
                self.auxiliary_optimizer.step()
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
                    boot_sum_recon_sqerr += ((r64 - x64) ** 2).sum().item()

                    boot_count += x.shape[0]

                # Log losses
                mean_autoenc_loss += vae_loss.item()
                mean_vel_loss += vel_pred_error.item()
                mean_recon_loss += recon_error.item()
                mean_kld_loss += kl_div.item()
                mean_decoder_loss += dec_loss.item()

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

        # Calculate the total bootstrapping probability over the performance of the autoencoder on all of the above
        #      total number of scalar elements per sample vector
        feat_dim = boot_sum_x.shape[0]

        mean_pred = boot_sum_x / boot_count                     # [D]
        ex2 = boot_sum_x2 / boot_count                          # [D]
        var = torch.clamp(ex2 - mean_pred**2, min=0.0)          # [D]
        mean_pred_error = var.mean().item()
        actual_pred_error = boot_sum_recon_sqerr / (boot_count * feat_dim)
        ratio = mean_pred_error / (actual_pred_error * self.boot_mult + 1e-8)
        pboot = np.tanh(ratio)

        # Use the (scaled) ratio of mean-prediction performance to actual prediction performance
        #     to determine if encoder bootstrapping is performed.
        self.use_boot = random.random() < pboot
        print("Use bootstrapped Encoder Dynamics: ", self.use_boot)

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_autoenc_loss, mean_decoder_loss, \
               mean_vel_loss, mean_recon_loss, mean_kld_loss, mean_pinn_loss

    def _compute_rl_loss(self, obs_batch, obs_hist_batch,
                         actions_batch, critic_obs_batch,
                         old_sigma_batch, old_mu_batch,
                         old_actions_log_prob_batch,
                         advantages_batch, target_values_batch, returns_batch):
        if self.use_boot:
            self.actor_critic.act(obs_batch, obs_hist_batch)
        else:
            self.actor_critic.act_bootmask(obs_batch, obs_hist_batch)

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

        return ppo_loss, surrogate_loss, value_loss, current_actions

    @staticmethod
    def _masked_mse(prediction, target, mask):
        per_sample = (prediction - target).square().mean(dim=-1)
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

    def _compute_auxiliary_loss(
        self, history, privileged_target, explicit_target, grf_target,
        valid, nominal_torque, transition,
    ):
        r"""Compute every decoder term on one shared stochastic VAE graph.

        ``z = mu + exp(logvar/2) eps`` is used only for privileged
        reconstruction. Runtime actor conditioning and both deployment heads
        use the deterministic mean ``mu``. The heads internally stop gradients
        through ``e = D_e(mu)`` while retaining gradients through ``mu`` (and
        through nominal torque for the GRF head).
        """
        mean, logvar = self.actor_critic.context_encoder(history)
        sample = self.actor_critic.context_encoder.reparameterization_trick(
            mean, logvar
        )
        explicit = self.actor_critic.explicit_estimator(mean)
        reconstruction = self.decoder(torch.cat((sample, explicit), dim=-1))
        heads = self.actor_critic.physics_heads(mean, explicit, nominal_torque)

        privileged = self._masked_mse(
            reconstruction, privileged_target.detach(), valid
        )
        explicit_loss = self._masked_mse(
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

    def _combine_bard_losses(
        self, inverse_loss, rollout_loss, projection_loss_value=None,
        soft_constraint_loss=None,
    ):
        """Apply independent physics weights before the PCGrad objective."""
        if projection_loss_value is None:
            projection_loss_value = inverse_loss * 0.0
        if soft_constraint_loss is None:
            soft_constraint_loss = inverse_loss * 0.0
        return (
            self.lambda_inverse * inverse_loss
            + self.lambda_rollout * rollout_loss
            + self.lambda_projection * projection_loss_value
            + self.lambda_soft_constraint * soft_constraint_loss
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
        _, _, latent, explicit = self.actor_critic.cenet_enc_forward(history)
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
        source_raw = self._policy_raw_action_from_noise(
            transition["delayed_source_observation"].detach(),
            transition["delayed_source_history"].detach(),
            transition["delayed_source_noise"],
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
        ABA remains a test/reference path. The inverse and rollout losses share their one
        control-rate context.  A second context is unavoidable for projection:
        it represents the sampled physics-substep state, not ``pre_q/pre_v``.
        Only that sampled context builds CRBA/QP terms; no per-decimation
        matrices are stored. RNEA is evaluated once for each enabled inverse
        or rollout bias term, CRBA once for rollout, and qpth exactly once per
        minibatch transition.
        """
        if not (self.bard_enabled or self.hard_pact_features.soft_constraint_penalty):
            return nominal_torque.sum() * 0.0
        qp_ready = self.hard_pact_features.execution_qp and self.hard_pact_qp is not None
        if not (self.bard_inverse_enabled or self.bard_rollout_enabled or qp_ready
                or self.hard_pact_features.soft_constraint_penalty):
            return nominal_torque.sum() * 0.0
        batch = self.storage.current_hard_pact_batch
        if batch is None:
            raise RuntimeError("HardPACT BARD loss requires named transition fields")
        control_dt = batch["control_dt"].detach().clamp_min(1.0e-8)
        self.bard_dynamics.default_joint_position = torch.as_tensor(
            default_pose, device=batch["pre_q"].device,
            dtype=batch["pre_q"].dtype,
        ).detach()
        parameters = {
            "added_base_mass": batch["realized_added_mass"].detach(),
            "base_com_shift": batch["realized_com_shift_body"].detach(),
            "joint_armature": batch["joint_armature"].detach(),
            "joint_friction": batch["joint_friction"].detach(),
            "joint_stiffness": batch["joint_stiffness"].detach(),
            "joint_damping": batch["joint_damping"].detach(),
        }
        mass_wrench = batch["equivalent_mass_com_wrench_world"].detach()
        zero = nominal_torque.sum() * 0.0
        _, _, latent, explicit = self.actor_critic.cenet_enc_forward(obs_hist_batch)
        grf_yaw = self.actor_critic.physics_estimator.predict_grf(
            latent, explicit, nominal_torque
        ).reshape(-1, 4, 3)
        wrench_yaw = self.actor_critic.physics_estimator.predict_wrench(
            latent, explicit
        ) / self.base_wrench_observation_scale

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

        grf_world = _yaw_local_to_world(
            grf_yaw / self.grf_observation_scale, batch["pre_q"][:, 3:7]
        )
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
        if self.bard_inverse_enabled or self.bard_rollout_enabled:
            # A PPO minibatch is often much larger than the fixed BARD Data
            # workspace (4096 envs x 64 steps / 4 minibatches = 65536 rows).
            # Stream detached mechanics through that reusable GPU workspace.
            # Losses are reweighted by their exact valid-row counts, so this
            # is mathematically identical to one monolithic masked reduction.
            valid_all = ~(
                batch["push_event_mask"].bool() | batch["reset_mask"].bool()
                | batch["timeout_mask"].bool() | batch["teleport_mask"].bool()
            )
            valid_total = valid_all.reshape(-1).sum().to(control_dt.dtype)
            inverse_numerator = zero
            rollout_numerator = zero
            inverse_metric_sums = {}
            rollout_metric_sums = {}
            capacity = self.bard_dynamics.batch_capacity
            for start in range(0, nominal_torque.shape[0], capacity):
                stop = min(start + capacity, nominal_torque.shape[0])
                sl = slice(start, stop)
                chunk_parameters = {
                    name: value[sl] for name, value in parameters.items()
                }
                context = self.bard_dynamics.build_context(
                    batch["pre_q"][sl], batch["pre_v"][sl],
                    parameters=chunk_parameters,
                    post_v_world=batch["post_v"][sl],
                    mass_com_wrench_world=mass_wrench[sl],
                    need_jacobians=True, need_qp=False,
                    need_forward_dynamics=self.bard_rollout_enabled,
                )
                count = valid_all[sl].reshape(-1).sum().to(control_dt.dtype)
                if self.bard_inverse_enabled:
                    observed_acceleration = (
                        context.post_v_bard - context.v_bard
                    ) / control_dt[sl]
                    timing = self._start_bard_timing("inverse", nominal_torque)
                    inverse = corrected_bard_inverse_dynamics_loss(
                        required_generalized_force=context.rnea(observed_acceleration),
                        foot_jacobians=context.foot_jacobians,
                        base_jacobian=context.base_jacobian,
                        interval_executed_torque=batch["interval_executed_torque"][sl],
                        interval_grf_world=grf_world[sl],
                        total_wrench_world=total_at_base[sl],
                        mass_com_wrench_world=context.mass_com_wrench_world,
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
        if self.hard_pact_features.soft_constraint_penalty:
            soft_constraint_loss = self._soft_constraint_loss(grf_world)

        # ---------------- sampled differentiable substep QP ---------------
        # Rollout solved every substep under no_grad.  PPO stores one compact
        # stratified-uniform sample per environment and rebuilds only that
        # sample's detached BARD matrices.  No [T,D,M,J] tensors are retained,
        # which is the principal VRAM saving.  Since P(k)=1/D, its normalized
        # correction-plus-slack value directly estimates mean_k L_proj,k.
        qp_loss = zero
        if qp_ready:
            # q_K and v_K are rollout measurements at the selected substep K;
            # detaching makes them constants in the implicit QP derivative.
            sample_q = batch.get("sampled_qp_q", batch["pre_q"]).detach()
            sample_v = batch.get("sampled_qp_v", batch["pre_v"]).detach()
            # All hard rate/integration constraints use physics Delta t.
            sample_dt = batch.get("physics_dt", control_dt).detach()
            if desired_position is not None and fb_func is not None:
                # Replay the current stochastic/delayed policy into held q_d
                # and tau_ff, then evaluate the sampled-state PD law:
                # tau_nom,K=Kp(q_d-q_K)-Kd*qdot_K+tau_ff.
                sampled_nominal = feedforward_torque + fb_func(
                    desired_position, sample_q[:, 7:], sample_v[:, 6:]
                )
            else:
                # Compatibility path for direct legacy unit/integration calls.
                sampled_nominal = nominal_torque
            # Recompute only the GRF head at K because its input includes
            # tau_nom,K. z_t and e_t remain the current policy-step features.
            sample_grf_yaw = self.actor_critic.physics_estimator.predict_grf(
                latent, explicit, sampled_nominal
            ).reshape(-1, 4, 3)
            # Undo observation scaling and rotate yaw-local Newtons to the
            # world-axis convention of the reconstructed sampled J_f.
            sample_grf_world = _yaw_local_to_world(
                sample_grf_yaw / self.grf_observation_scale,
                sample_q[:, 3:7],
            )
            # This detached label removes the inertial gravity contribution
            # already represented by sampled randomized BARD inertia.
            sample_mass_wrench = batch.get(
                "sampled_qp_mass_com_wrench_world", mass_wrench
            ).detach()
            # Reuse the current wrench-head output, rotate it with sampled yaw,
            # subtract W_mass/CoM once, and shift it to J_b's reference point.
            sample_applied, _ = world_wrench(
                sample_q, sample_mass_wrench, wrench_yaw,
                batch["realized_com_shift_body"],
            )
            # Reconstruct M_K,b_K,J_f,K,J_b,K,Jdot_f,K*v_K from compact state
            # and realized parameters. These mechanics are detached by BARD.
            # Reuse the same fixed-capacity BARD workspace instead of asking
            # it to allocate for the flattened PPO minibatch. The resulting
            # detached terms are concatenated once; HardPACTQP subsequently
            # consumes them in its independently configured PPO chunks.
            sampled_terms = {
                name: [] for name in (
                    "mass_matrix", "bias", "foot_jacobians",
                    "base_jacobian", "foot_acceleration_bias",
                )
            }
            capacity = self.bard_dynamics.batch_capacity
            for start in range(0, sample_q.shape[0], capacity):
                stop = min(start + capacity, sample_q.shape[0])
                sl = slice(start, stop)
                sample_context = self.bard_dynamics.build_context(
                    sample_q[sl], sample_v[sl],
                    parameters={
                        name: value[sl] for name, value in parameters.items()
                    },
                    need_qp=True,
                )
                for name in sampled_terms:
                    sampled_terms[name].append(getattr(sample_context, name))
            sampled_terms = {
                name: torch.cat(values, dim=0)
                for name, values in sampled_terms.items()
            }
            # qpth solves
            #   min_x 1/2*x^TQx+p^Tx,  Gx<=h, Ax=b,
            #   x=[qdd_18,f_12,tau_safe_12,s_12].
            # Gradients enter through sampled_nominal, sample_grf_world, and
            # sample_applied; all state/mechanics/rate-center tensors detach.
            sample_contact_probability = explicit[:, 3:7]
            differentiate_qp = self.hard_pact_features.differentiable_qp
            qp_arguments = dict(
                # Equality coefficient M_K.
                mass_matrix=sampled_terms["mass_matrix"],
                # Equality RHS contribution -b_K.
                bias=sampled_terms["bias"],
                # Dynamics -J_f,K^T and contact-acceleration J_f,K blocks.
                foot_jacobians=sampled_terms["foot_jacobians"],
                # Wrench generalized force J_b,K^T*W_hat_applied.
                base_jacobian=sampled_terms["base_jacobian"],
                # Known affine contact acceleration Jdot_f,K*v_K.
                foot_acceleration_bias=sampled_terms["foot_acceleration_bias"],
                # Differentiable torque tracking center.
                tau_nom=(sampled_nominal if differentiate_qp else sampled_nominal.detach()),
                # Differentiable force tracking center in world Newtons.
                force_pred_world=(sample_grf_world if differentiate_qp else sample_grf_world.detach()),
                # Differentiable world [force,moment] equality input.
                wrench_pred_world=(sample_applied if differentiate_qp else sample_applied.detach()),
                # c_eff keeps every row nondegenerate while retaining the
                # differentiable contact-probability path into the estimator.
                contact_probability=(sample_contact_probability if differentiate_qp else sample_contact_probability.detach()),
                # Exact rollout tau_safe,K-1 reproduces the sampled rate box.
                previous_torque=batch.get(
                    "sampled_qp_previous_torque",
                    batch.get("previous_executed_torque", batch["interval_executed_torque"]),
                ),
                # Sampled joint state defines hard one-step q/qdot boxes.
                joint_position=sample_q[:, 7:], joint_velocity=sample_v[:, 6:],
                dt=sample_dt,
            )
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
                batch["push_event_mask"].bool() | batch["reset_mask"].bool()
                | batch["timeout_mask"].bool() | batch["teleport_mask"].bool()
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
            # Log only the newly recomputed sampled solve, whose status can
            # differ from rollout after policy/head parameters update.
            self.last_qp_metrics = dict(qp_result.metrics or {})
            self.last_qp_metrics["qp/minimal/projection_loss"] = qp_loss.detach()
            correction = (qp_result.tau_safe - sampled_nominal).detach()
            intervention = correction.abs().amax(dim=-1) > 1.0e-6
            self.last_qp_metrics["qp/minimal/intervention_fraction"] = (
                intervention.float().mean()
            )
            if intervention.any():
                self.last_qp_metrics[
                    "qp/minimal/intervention_torque_correction_rms"
                ] = correction[intervention].square().mean().sqrt()
            else:
                self.last_qp_metrics[
                    "qp/minimal/intervention_torque_correction_rms"
                ] = correction.new_zeros(())
            audit_ran = self.last_qp_metrics.get("qp/full/audit_ran")
            if audit_ran is not None and bool(audit_ran.detach().item()):
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
        # All three weighted physics terms enter the existing PCGrad objective;
        # no extra optimizer owns actor/trunk/head parameters:
        # L_phys=lambda_ID*L_ID+lambda_roll*L_roll+lambda_proj*L_proj.
        self.last_physics_loss_metrics = {
            "physics/loss/inverse": inverse_loss.detach(),
            "physics/loss/rollout": rollout_loss.detach(),
            "physics/loss/soft_constraint": soft_constraint_loss.detach(),
        }
        return self._combine_bard_losses(
            inverse_loss, rollout_loss,
            qp_loss if self.hard_pact_features.projection_loss else zero,
            soft_constraint_loss,
        )
