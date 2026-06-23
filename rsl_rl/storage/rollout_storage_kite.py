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

class RolloutStorageKITE:
    class Transition:
        def __init__(self):
            self.observations = None
            self.observation_history = None
            self.privileged_observation_history = None
            self.depth_images = None
            self.depth_latent_history = None
            self.depth_torso_state = None
            self.terrain_map = None
            self.dones = None

            self.explicit_labels = None  # same timestep as observations, used by encoder output
            self.obs_targets = None  # next time-step from observations, used by decoder output

            self.actions = None
            self.rewards = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            
            self.hidden_states = None
        
        def clear(self):
            self.__init__()

    # We want all of the actions and associated data formatted in the Model kinematic definition - [FR, FL, RR, RL]
    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        single_critic_obs_shape,
        obs_hist_shape,
        actions_shape,
        explicit_shape,
        depth_image_shape,
        depth_latent_history_shape,
        depth_torso_state_shape,
        terrain_map_shape,
        priv_obs_history_shape,
        contrastive_anchor_shape,
        device="cpu",
        storage_dtype=torch.bfloat16,
        store_action_distribution=True,
    ):

        self.device = device
        self.storage_dtype = storage_dtype
        self.store_action_distribution = store_action_distribution

        self.obs_shape        = obs_shape
        self.actions_shape    = actions_shape

        # Core PPO tensors.
        self.observations        = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device, dtype=self.storage_dtype)
        self.observation_history = torch.zeros(num_transitions_per_env, num_envs, *obs_hist_shape, device=self.device, dtype=self.storage_dtype)    
        # Raw privileged history is kept so PPO can rebuild critic inputs from
        # current privileged encoders during the update step.
        self.privileged_observation_history = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *priv_obs_history_shape,
            device=self.device,
            dtype=self.storage_dtype,
        )
        # Store only the newest processed depth image. Previous visual context
        # is stored as depth-frame latents to reduce rollout VRAM.
        self.depth_images = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *depth_image_shape,
            device=self.device,
            dtype=self.storage_dtype,
        )
        self.depth_latent_history = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *depth_latent_history_shape,
            device=self.device,
            dtype=self.storage_dtype,
        )
        # 8D body-motion conditioning vector for the depth-frame encoder.
        self.depth_torso_state = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *depth_torso_state_shape,
            device=self.device,
            dtype=self.storage_dtype,
        )
        # Privileged terrain supervision map: height plus surface normal field.
        self.terrain_maps = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *terrain_map_shape,
            device=self.device,
            dtype=self.storage_dtype,
        )
        self.dones               = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        
        # Auxiliary supervision targets used by the KITE encoder losses.
        self.explicit_labels = torch.zeros(num_transitions_per_env, num_envs, *explicit_shape, device=self.device, dtype=self.storage_dtype)
        self.observation_targets = torch.zeros(num_transitions_per_env, num_envs, *single_critic_obs_shape, device=self.device, dtype=self.storage_dtype)

        
        # PPO rollout tensors.
        self.rewards          = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions          = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values           = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns          = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages       = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu               = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device) if self.store_action_distribution else None
        self.sigma            = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device) if self.store_action_distribution else None

        #  Shared
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        
        self.observations[self.step].copy_(transition.observations)
        self.observation_history[self.step].copy_(transition.observation_history)
        # These tensors are captured before stepping the environment, so they
        # line up with the action/log-prob/value stored for the same timestep.
        self.privileged_observation_history[self.step].copy_(transition.privileged_observation_history)
        self.depth_images[self.step].copy_(transition.depth_images)
        self.depth_latent_history[self.step].copy_(transition.depth_latent_history)
        self.depth_torso_state[self.step].copy_(transition.depth_torso_state)
        self.terrain_maps[self.step].copy_(transition.terrain_map)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        
        # Auxiliary labels for explicit-state and dynamics reconstruction heads.
        self.explicit_labels[self.step].copy_(transition.explicit_labels)
        self.observation_targets[self.step].copy_(transition.obs_targets)
        
        # Action distribution tensors saved from the policy that generated the rollout.
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        if self.store_action_distribution:
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)
        
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
        self.step = 0

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
        obs_history = self.observation_history.flatten(0,1)
        # Flatten time and environment dimensions for random mini-batches.
        privileged_obs_history = self.privileged_observation_history.flatten(0, 1)
        depth_images = self.depth_images.flatten(0, 1)
        depth_latent_history = self.depth_latent_history.flatten(0, 1)
        depth_torso_state = self.depth_torso_state.flatten(0, 1)
        terrain_maps = self.terrain_maps.flatten(0, 1)

        explicit_labels = self.explicit_labels.flatten(0,1)
        obs_targets = self.observation_targets.flatten(0,1)

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1) if self.store_action_distribution else None
        old_sigma = self.sigma.flatten(0, 1) if self.store_action_distribution else None

        dones = self.dones.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                # Keep the random minibatch membership, but sort indices so
                # large rollout tensor gathers are more contiguous on GPU.
                batch_idx = indices[start:end].sort().values

                # Baseline PPO stuff
                obs_batch = observations[batch_idx].float()
                obs_hist_batch = obs_history[batch_idx].float()
                privileged_obs_history_batch = privileged_obs_history[batch_idx].float()
                depth_images_batch = depth_images[batch_idx].float()
                depth_latent_history_batch = depth_latent_history[batch_idx].float()
                depth_torso_state_batch = depth_torso_state[batch_idx].float()
                terrain_maps_batch = terrain_maps[batch_idx].float()

                # Auxiliary KITE encoder targets.
                explicit_labels_batch = explicit_labels[batch_idx].float()
                obs_labels_batch = obs_targets[batch_idx].float()

                # PPO action/value tensors.
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx] if old_mu is not None else None
                old_sigma_batch = old_sigma[batch_idx] if old_sigma is not None else None

                terminated_batch = 1.0 - dones[batch_idx]
  
                yield (
                    terminated_batch,
                    obs_batch,
                    obs_hist_batch,
                    privileged_obs_history_batch,
                    depth_images_batch,
                    depth_latent_history_batch,
                    depth_torso_state_batch,
                    terrain_maps_batch,
                    explicit_labels_batch,
                    obs_labels_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                )
