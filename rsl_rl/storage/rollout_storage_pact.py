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

import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class RolloutStoragePACT:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.observation_history = None
            self.dones = None

            self.explicit_labels = None  # same timestep as observations, used by encoder output
            self.grf_targets = None  # next time-step from observations, used by decoder output
            self.obs_targets = None  # next time-step from observations, used by decoder output

            self.actions = None
            self.rewards = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            # HardPACT stores epsilon=(a-mu)/sigma once. Raw actions, policy
            # observations, and histories already have canonical storage
            # above and are deliberately not duplicated for replay.
            self.action_noise = None
            self.context_latent_noise = None

            #  PINN stuff
            self.prev_obs      = None
            self.prev_obs_hist = None
            self.pprev_obs      = None
            self.pprev_obs_hist = None

            self.wb_contact_forces = None
            self.wb_mass_mat = None
            self.wb_bias_vec = None
            self.torso_acc = None
            
            self.hidden_states = None
        
        def clear(self):
            self.__init__()

    # We want all of the actions and associated data formatted in the Model kinematic definition - [FR, FL, RR, RL]
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, sinle_critc_obs_shape, obs_hist_shape, actions_shape, explicit_shape, grf_shape, wb_shape, device="cpu", *, store_legacy_pinn_dynamics=True):

        self.device = device

        self.obs_shape        = obs_shape
        self.critic_obs_shape = critic_obs_shape
        self.actions_shape    = actions_shape

        # Core
        self.observations        = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.critic_observations = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device)
        self.observation_history = torch.zeros(num_transitions_per_env, num_envs, *obs_hist_shape, device=self.device)    
        self.dones               = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        
        # specific to DreamWaQ style history encoder...
        self.explicit_labels = torch.zeros(num_transitions_per_env, num_envs, *explicit_shape, device=self.device)
        self.grf_targets = torch.zeros(num_transitions_per_env, num_envs, *grf_shape, device=self.device)
        self.observation_targets = torch.zeros(num_transitions_per_env, num_envs, *sinle_critc_obs_shape, device=self.device)

        
        # For PPO
        # Need a set of these for each "task" (position control and torque control)
        self.rewards          = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions          = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values           = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns          = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages       = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu               = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma            = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        #  Shared
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PINN specific stuff
        self.prev_obs       = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.prev_obs_hist  = torch.zeros(num_transitions_per_env, num_envs, *obs_hist_shape, device=self.device)
        self.pprev_obs      = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.pprev_obs_hist = torch.zeros(num_transitions_per_env, num_envs, *obs_hist_shape, device=self.device)

        self.wb_contact_forces = torch.zeros(
            num_transitions_per_env, num_envs, *wb_shape, device=self.device
        )
        # Legacy PACT consumes these Pinocchio tensors directly. HardPACT
        # reconstructs detached mechanics once per rollout update, so keeping
        # T x N copies would waste both VRAM and gather bandwidth.
        self.store_legacy_pinn_dynamics = bool(store_legacy_pinn_dynamics)
        if self.store_legacy_pinn_dynamics:
            self.wb_mass_mats = torch.zeros(
                num_transitions_per_env, num_envs, *wb_shape, *wb_shape,
                device=self.device,
            )
            self.wb_bias_vecs = torch.zeros(
                num_transitions_per_env, num_envs, *wb_shape,
                device=self.device,
            )
            self.torso_accelerations = torch.zeros(
                num_transitions_per_env, num_envs, 6, device=self.device
            )
        else:
            self.wb_mass_mats = None
            self.wb_bias_vecs = None
            self.torso_accelerations = None

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0
        # HardPACT adds named physics fields lazily. Legacy PACT never creates
        # these tensors and therefore retains its exact storage behavior.
        self.hard_pact_fields = None
        self.current_hard_pact_batch = None
        self.current_batch_indices = None
        self.action_noise = None
        self.max_action_delay = None
        self._action_replay_boundary_observations = None
        self._action_replay_boundary_history = None
        self._action_replay_boundary_noise = None
        self.context_latent_noise = None
        self._action_replay_boundary_context_latent_noise = None

    def configure_action_replay(self, max_action_delay, context_latent_dim=16):
        """Allocate compact GPU-only stochastic-delay replay metadata."""
        maximum = int(max_action_delay)
        if maximum < 0 or maximum >= self.num_transitions_per_env:
            raise ValueError(
                "max_action_delay must be nonnegative and shorter than a rollout"
            )
        self.max_action_delay = maximum
        self.action_noise = torch.zeros_like(self.actions)
        self.context_latent_noise = torch.zeros(
            self.num_transitions_per_env, self.num_envs,
            int(context_latent_dim), device=self.device,
        )
        # Only sources crossing a rollout boundary need extra observation
        # storage. All in-rollout sources are gathered from the existing core
        # observation/history tensors by index.
        self._action_replay_boundary_observations = torch.zeros(
            maximum, self.num_envs, *self.obs_shape, device=self.device
        )
        self._action_replay_boundary_history = torch.zeros(
            maximum, self.num_envs, *self.observation_history.shape[2:],
            device=self.device,
        )
        self._action_replay_boundary_noise = torch.zeros(
            maximum, self.num_envs, *self.actions_shape, device=self.device
        )
        self._action_replay_boundary_context_latent_noise = torch.zeros(
            maximum, self.num_envs, int(context_latent_dim), device=self.device
        )

    def add_transitions(self, transition: Transition):
        
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        
        self.observations[self.step].copy_(transition.observations)
        self.critic_observations[self.step].copy_(transition.critic_observations)
        self.observation_history[self.step].copy_(transition.observation_history)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        
        # Specific to DreamWaQ style history encoder
        self.explicit_labels[self.step].copy_(transition.explicit_labels)
        self.grf_targets[self.step].copy_(transition.grf_targets)
        self.observation_targets[self.step].copy_(transition.obs_targets)
        
        # Need a set for each "task"
        #  - Position Control
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        if self.action_noise is not None:
            if transition.action_noise is None:
                raise RuntimeError("HardPACT action replay requires stored noise")
            self.action_noise[self.step].copy_(transition.action_noise)
            if transition.context_latent_noise is None:
                raise RuntimeError(
                    "HardPACT context replay requires stored latent noise"
                )
            self.context_latent_noise[self.step].copy_(
                transition.context_latent_noise
            )

        #  - PINN stuff
        self.prev_obs[self.step].copy_(transition.prev_obs)
        self.prev_obs_hist[self.step].copy_(transition.prev_obs_hist)

        self.pprev_obs[self.step].copy_(transition.pprev_obs)
        self.pprev_obs_hist[self.step].copy_(transition.pprev_obs_hist)
        
        self.wb_contact_forces[self.step].copy_(transition.wb_contact_forces)
        if self.store_legacy_pinn_dynamics:
            self.wb_mass_mats[self.step].copy_(transition.wb_mass_mat)
            self.wb_bias_vecs[self.step].copy_(transition.wb_bias_vec)
            self.torso_accelerations[self.step].copy_(transition.torso_acc)

        hard_pact = getattr(transition, "hard_pact", None)
        if hard_pact is not None:
            if self.hard_pact_fields is None:
                self.hard_pact_fields = {
                    name: torch.zeros(
                        self.num_transitions_per_env,
                        self.num_envs,
                        *value.shape[1:],
                        device=self.device,
                        dtype=value.dtype,
                    )
                    for name, value in hard_pact.items()
                }
            for name, value in hard_pact.items():
                self.hard_pact_fields[name][self.step].copy_(value)
        
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        if self.max_action_delay:
            delay = self.max_action_delay
            self._action_replay_boundary_observations.copy_(
                self.observations[-delay:]
            )
            self._action_replay_boundary_history.copy_(
                self.observation_history[-delay:]
            )
            self._action_replay_boundary_noise.copy_(self.action_noise[-delay:])
            self._action_replay_boundary_context_latent_noise.copy_(
                self.context_latent_noise[-delay:]
            )
        self.step = 0
        self.current_hard_pact_batch = None
        self.current_batch_indices = None

    def _action_replay_sources(self, batch_idx, delay):
        """Resolve delayed sources on GPU without duplicating rollout history."""
        timestep = torch.div(
            batch_idx, self.num_envs, rounding_mode="floor"
        )
        environment = batch_idx.remainder(self.num_envs)
        source_timestep = timestep - delay
        source_observation = self.observations.new_empty(
            batch_idx.shape[0], *self.obs_shape
        )
        source_history = self.observation_history.new_empty(
            batch_idx.shape[0], *self.observation_history.shape[2:]
        )
        source_noise = self.action_noise.new_empty(
            batch_idx.shape[0], *self.actions_shape
        )
        source_context_noise = self.context_latent_noise.new_empty(
            batch_idx.shape[0], self.context_latent_noise.shape[-1]
        )
        current = source_timestep >= 0
        t, e = source_timestep[current], environment[current]
        source_observation[current] = self.observations[t, e]
        source_history[current] = self.observation_history[t, e]
        source_noise[current] = self.action_noise[t, e]
        source_context_noise[current] = self.context_latent_noise[t, e]
        boundary = ~current
        index = self.max_action_delay + source_timestep[boundary]
        e = environment[boundary]
        source_observation[boundary] = (
            self._action_replay_boundary_observations[index, e]
        )
        source_history[boundary] = (
            self._action_replay_boundary_history[index, e]
        )
        source_noise[boundary] = self._action_replay_boundary_noise[index, e]
        source_context_noise[boundary] = (
            self._action_replay_boundary_context_latent_noise[index, e]
        )
        return source_observation, source_history, source_noise, source_context_noise

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
    
    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()
    
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        critic_observations = self.critic_observations.flatten(0, 1)
        obs_history = self.observation_history.flatten(0,1)

        explicit_labels = self.explicit_labels.flatten(0,1)
        grf_labels = self.grf_targets.flatten(0,1)
        obs_targets = self.observation_targets.flatten(0,1)

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        dones = self.dones.flatten(0, 1)

        # PINN stuff
        prev_obs      = self.prev_obs.flatten(0, 1)
        prev_obs_hist = self.prev_obs_hist.flatten(0, 1)
        pprev_obs      = self.pprev_obs.flatten(0, 1)
        pprev_obs_hist = self.pprev_obs_hist.flatten(0, 1)

        gt_forces     = self.wb_contact_forces.flatten(0,1)
        wb_mass_mats = (
            self.wb_mass_mats.flatten(0, 1)
            if self.wb_mass_mats is not None else None
        )
        wb_bias_vecs = (
            self.wb_bias_vecs.flatten(0, 1)
            if self.wb_bias_vecs is not None else None
        )
        torso_accs = (
            self.torso_accelerations.flatten(0, 1)
            if self.torso_accelerations is not None else None
        )

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]
                self.current_batch_indices = batch_idx

                # Baseline PPO stuff
                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                obs_hist_batch = obs_history[batch_idx]

                # DreamWaQ Style History Encoder stuff
                explicit_labels_batch = explicit_labels[batch_idx]
                grf_labels_batch = grf_labels[batch_idx]
                obs_labels_batch = obs_targets[batch_idx]

                # Position Control RL Task
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                terminated_batch = 1.0 - dones[batch_idx]


                # PINN stuff
                prev_obs_batch      = prev_obs[batch_idx]
                prev_obs_hist_batch = prev_obs_hist[batch_idx]
                gt_forces_batch     = gt_forces[batch_idx]
                mass_mat_batch = (
                    wb_mass_mats[batch_idx] if wb_mass_mats is not None else None
                )
                bias_vec_batch = (
                    wb_bias_vecs[batch_idx] if wb_bias_vecs is not None else None
                )
                torso_accs_batch = (
                    torso_accs[batch_idx] if torso_accs is not None else None
                )

                pprev_obs_batch = pprev_obs[batch_idx]
                pprev_obs_hist_batch = pprev_obs_hist[batch_idx]

                if self.hard_pact_fields is not None:
                    self.current_hard_pact_batch = {
                        name: value.flatten(0, 1)[batch_idx]
                        for name, value in self.hard_pact_fields.items()
                    }
                    if self.action_noise is not None:
                        delay = self.current_hard_pact_batch[
                            "sampled_action_delay"
                        ].reshape(-1).long()
                        (source_obs, source_history, source_noise,
                         source_context_noise) = (
                            self._action_replay_sources(batch_idx, delay)
                        )
                        # Raw actions and their source observations already
                        # live in core rollout tensors; these aliases expose
                        # them to the named HardPACT replay interface without
                        # allocating duplicate persistent buffers.
                        self.current_hard_pact_batch.update({
                            "raw_sampled_action": actions_batch,
                            "standardized_action_noise": self.action_noise.flatten(
                                0, 1
                            )[batch_idx],
                            "context_latent_noise": self.context_latent_noise.flatten(
                                0, 1
                            )[batch_idx],
                            "action_source_observation": obs_batch,
                            "action_source_history": obs_hist_batch,
                            "delayed_source_observation": source_obs,
                            "delayed_source_history": source_history,
                            "delayed_source_noise": source_noise,
                            "delayed_source_context_latent_noise": (
                                source_context_noise
                            ),
                        })

                
                yield terminated_batch, obs_batch, critic_observations_batch, obs_hist_batch, explicit_labels_batch, \
                        grf_labels_batch, obs_labels_batch, actions_batch, target_values_batch, \
                        advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, \
                        old_sigma_batch, prev_obs_batch, prev_obs_hist_batch, gt_forces_batch, mass_mat_batch, \
                        bias_vec_batch, torso_accs_batch, pprev_obs_batch, pprev_obs_hist_batch
