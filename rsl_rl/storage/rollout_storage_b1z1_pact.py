"""Rollout storage for B1/Z1 PACT PPO and its temporal auxiliary targets."""

from __future__ import annotations

import torch


class RolloutStorageB1Z1PACT:
    class Transition:
        def __init__(self):
            # Match UniFP's explicit transition lifecycle. PACT adds only the
            # fields required by its history-conditioned actor and dynamics objectives.
            self.observations = None
            self.critic_observations = None
            self.histories = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.log_probs = None
            self.mu = None
            self.sigma = None
            self.explicit_targets = None
            self.next_privileged = None
            self.dynamics_state = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, steps, obs_dim, critic_dim, history_dim, action_dim, explicit_dim, next_privileged_dim, state_dim, device):
        self.device, self.num_envs, self.steps, self.step = device, num_envs, steps, 0
        def zeros(dim): return torch.zeros(steps, num_envs, dim, device=device)
        self.observations, self.critic_observations, self.histories = zeros(obs_dim), zeros(critic_dim), zeros(history_dim)
        self.actions, self.mu, self.sigma = zeros(action_dim), zeros(action_dim), zeros(action_dim)
        self.values, self.rewards, self.returns, self.advantages = zeros(1), zeros(1), zeros(1), zeros(1)
        self.log_probs, self.dones = zeros(1), torch.zeros(steps, num_envs, 1, dtype=torch.bool, device=device)
        # Ground-truth explicit_t is supervision only; policy conditioning is predicted.
        self.explicit_targets = zeros(explicit_dim)
        # Decoder supervision is one privileged frame, while the critic sees
        # a temporal stack. This frame now also owns force supervision.
        self.next_privileged = zeros(next_privileged_dim)
        # One 180-D post-action state per rollout step. Retaining v_t and
        # v_(t+1) lets PPO construct a transition-aligned acceleration instead
        # of differentiating a policy action sequence in isolation.
        self.dynamics_state = zeros(state_dim)

    def add(self, transition):
        if self.step >= self.steps:
            raise AssertionError("Rollout buffer overflow")
        # Copy rather than keep references: PPO shuffles the flattened rollout
        # later, but every PINN field must remain paired with its own action_t.
        for name in (
            "observations", "critic_observations", "histories", "actions", "mu", "sigma", "values",
            "log_probs", "explicit_targets",
            "next_privileged", "dynamics_state",
        ):
            getattr(self, name)[self.step].copy_(getattr(transition, name))
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1).bool())
        self.step += 1

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for index in reversed(range(self.steps)):
            next_value = last_values if index == self.steps - 1 else self.values[index + 1]
            alive = 1.0 - self.dones[index].float()
            delta = self.rewards[index] + alive * gamma * next_value - self.values[index]
            advantage = delta + alive * gamma * lam * advantage
            self.returns[index] = advantage + self.values[index]
        # Keep UniFP's exact advantage construction and normalization.
        self.advantages = self.returns - self.values
        self.advantages = (
            self.advantages - self.advantages.mean()
        ) / (self.advantages.std() + 1e-8)

    def mini_batches(self, mini_batches, epochs):
        count = self.steps * self.num_envs
        batch_size = count // mini_batches
        # UniFP draws one rollout permutation and reuses it across PPO epochs.
        # Truncate exactly as UniFP does if the rollout is not evenly divisible.
        indices = torch.randperm(
            mini_batches * batch_size, requires_grad=False, device=self.device
        )
        # Flatten time/environment jointly, applying identical random indices
        # to actions, transition states, privileged labels, and terminal masks.
        flat = {name: getattr(self, name).flatten(0, 1) for name in (
            "observations", "critic_observations", "histories", "actions", "mu", "sigma", "values", "returns", "advantages",
            "log_probs", "dones", "explicit_targets",
            "next_privileged", "dynamics_state",
        )}
        for _ in range(epochs):
            for mini_batch in range(mini_batches):
                start = mini_batch * batch_size
                end = (mini_batch + 1) * batch_size
                idx = indices[start:end]
                yield {name: value[idx] for name, value in flat.items()}

    def clear(self):
        self.step = 0
