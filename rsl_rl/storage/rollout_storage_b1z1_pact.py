"""Rollout storage for B1/Z1 PACT PPO and its temporal auxiliary targets."""

from __future__ import annotations

import torch


class RolloutStorageB1Z1PACT:
    class Transition:
        def clear(self):
            self.__dict__.clear()

    def __init__(self, num_envs, steps, obs_dim, critic_dim, history_dim, action_dim, explicit_dim, force_dim, next_privileged_dim, state_dim, device):
        self.device, self.num_envs, self.steps, self.step = device, num_envs, steps, 0
        def zeros(dim): return torch.zeros(steps, num_envs, dim, device=device)
        self.observations, self.critic_observations, self.histories = zeros(obs_dim), zeros(critic_dim), zeros(history_dim)
        self.actions, self.mu, self.sigma = zeros(action_dim), zeros(action_dim), zeros(action_dim)
        self.values, self.rewards, self.returns, self.advantages = zeros(1), zeros(1), zeros(1), zeros(1)
        self.log_probs, self.dones = zeros(1), torch.zeros(steps, num_envs, 1, dtype=torch.bool, device=device)
        # Both snapshots are explicit_t. Separate fields enforce their PPO
        # reconstruction and current-state auxiliary roles.
        self.actor_explicit_labels = zeros(explicit_dim)
        self.explicit_targets = zeros(explicit_dim)
        # Alpha is frozen for a rollout and stored per transition so PPO always
        # reconstructs the exact explicit blend that generated action_t.
        self.explicit_blend_alpha = zeros(1)
        # Normalized next-state target: four GRFs, EE force, and base wrench.
        # The PINN may use predicted versions only after the reliability gate
        # says this decoder has learned the measurement relationship.
        self.force_targets = zeros(force_dim)
        # Decoder supervision is one privileged frame, while the critic sees
        # a temporal stack of those frames.
        self.next_privileged = zeros(next_privileged_dim)
        # One 180-D post-action state per rollout step. Retaining v_t and
        # v_(t+1) lets PPO construct a transition-aligned acceleration instead
        # of differentiating a policy action sequence in isolation.
        self.dynamics_state = zeros(state_dim)

    def add(self, transition):
        if self.step >= self.steps:
            raise RuntimeError("B1Z1 PACT rollout storage overflow")
        # Copy rather than keep references: PPO shuffles the flattened rollout
        # later, but every PINN field must remain paired with its own action_t.
        for name in (
            "observations", "critic_observations", "histories", "actions", "mu", "sigma", "values", "rewards",
            "log_probs", "actor_explicit_labels", "explicit_targets",
            "force_targets", "next_privileged", "dynamics_state",
        ):
            getattr(self, name)[self.step].copy_(getattr(transition, name))
        # The storage class is shared with coupled PACT, whose discrete
        # explicit bootstrap does not populate this PACT-position-only field.
        if hasattr(transition, "explicit_blend_alpha"):
            self.explicit_blend_alpha[self.step].copy_(transition.explicit_blend_alpha)
        else:
            self.explicit_blend_alpha[self.step].zero_()
        self.dones[self.step].copy_(transition.dones.view(-1, 1).bool())
        self.step += 1

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0.0
        for index in reversed(range(self.steps)):
            next_value = last_values if index == self.steps - 1 else self.values[index + 1]
            alive = 1.0 - self.dones[index].float()
            delta = self.rewards[index] + alive * gamma * next_value - self.values[index]
            advantage = delta + alive * gamma * lam * advantage
            self.returns[index] = advantage + self.values[index]
        self.advantages.copy_(self.returns - self.values)
        self.advantages.sub_(self.advantages.mean()).div_(self.advantages.std().clamp_min(1e-8))

    def mini_batches(self, mini_batches, epochs):
        count = self.steps * self.num_envs
        batch_size = count // mini_batches
        # Flatten time/environment jointly, applying identical random indices
        # to actions, transition states, force labels, and terminal masks.
        flat = {name: getattr(self, name).flatten(0, 1) for name in (
            "observations", "critic_observations", "histories", "actions", "mu", "sigma", "values", "returns", "advantages",
            "log_probs", "dones", "actor_explicit_labels", "explicit_targets", "explicit_blend_alpha",
            "force_targets", "next_privileged", "dynamics_state",
        )}
        for _ in range(epochs):
            indices = torch.randperm(count, device=self.device)
            for start in range(0, count, batch_size):
                idx = indices[start:start + batch_size]
                yield {name: value[idx] for name, value in flat.items()}

    def clear(self):
        self.step = 0
