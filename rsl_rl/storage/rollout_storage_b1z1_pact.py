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
            self.rollout_initial_state = None
            self.latent_noise = None
            self.nominal_torque = None
            self.physics_source = None
            self.interval_torque = None
            self.mass_wrench = None
            self.physics_invalid = None
            self.actor_physics = None

        def clear(self):
            self.__init__()

    def __init__(
        self, num_envs, steps, obs_dim, critic_dim, history_dim, action_dim,
        explicit_dim, next_privileged_dim, state_dim,
        policy_distribution_dim=None, rollout_state_dim=0, device="cpu", latent_dim=64,
    ):
        self.device, self.num_envs, self.steps, self.step = device, num_envs, steps, 0
        self.actor_physics = {}  # Allocated only when the optional actor objective is enabled.
        def zeros(dim): return torch.zeros(steps, num_envs, dim, device=device)
        self.observations, self.critic_observations, self.histories = zeros(obs_dim), zeros(critic_dim), zeros(history_dim)
        # Coupled policies normally use one width for stored actions and their
        # Gaussian. PACT-Pos stores its deterministic torque head too, while
        # PPO statistics remain defined only for the stochastic position half.
        distribution_dim = action_dim if policy_distribution_dim is None else policy_distribution_dim
        self.actions = zeros(action_dim)
        # Replay the same reparameterization noise for PPO likelihood ratios.
        self.latent_noise, self.nominal_torque = zeros(latent_dim), zeros(19)
        self.interval_torque, self.mass_wrench, self.physics_invalid = zeros(19), zeros(6), zeros(1)
        self.mu, self.sigma = zeros(distribution_dim), zeros(distribution_dim)
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
        # Optional q_t,v_t packet used only by the BARD one-step rollout.
        self.rollout_initial_state = (
            zeros(rollout_state_dim) if rollout_state_dim else None
        )
        self.physics_source = (
            zeros(obs_dim + history_dim + latent_dim + action_dim + 1)
            if rollout_state_dim else None
        )

    def add(self, transition):
        if self.step >= self.steps:
            raise AssertionError("Rollout buffer overflow")
        if transition.actor_physics is not None:
            for name, value in transition.actor_physics.items():
                if name not in self.actor_physics:
                    # Collection runs in inference_mode; training must save these
                    # constants for autograd's actor-facing dynamics backward.
                    with torch.inference_mode(False):
                        self.actor_physics[name] = value.new_zeros((self.steps, *value.shape))
                self.actor_physics[name][self.step].copy_(value.detach())
        # Copy rather than keep references: PPO shuffles the flattened rollout
        # later, but every PINN field must remain paired with its own action_t.
        for name in (
            "observations", "critic_observations", "histories", "actions", "mu", "sigma", "values",
            "log_probs", "explicit_targets", "latent_noise", "nominal_torque",
            "next_privileged", "dynamics_state",
        ):
            getattr(self, name)[self.step].copy_(getattr(transition, name))
        if self.rollout_initial_state is not None:
            self.rollout_initial_state[self.step].copy_(
                transition.rollout_initial_state
            )
            # Direct algorithm tests can omit delay replay; runners always supply it.
            source = transition.physics_source
            if source is None:
                source = torch.cat((transition.observations, transition.histories,
                    transition.latent_noise,
                    (transition.actions - transition.mu) / transition.sigma.clamp_min(1e-8),
                    torch.ones_like(transition.mu[:, :1])), dim=-1)
            self.physics_source[self.step].copy_(source)
        for name in ("interval_torque", "mass_wrench", "physics_invalid"):
            value = getattr(transition, name)
            if value is not None:
                getattr(self, name)[self.step].copy_(value)
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
            "log_probs", "dones", "explicit_targets", "latent_noise", "nominal_torque",
            "next_privileged", "dynamics_state", "interval_torque", "mass_wrench", "physics_invalid",
        )}
        if self.rollout_initial_state is not None:
            flat["rollout_initial_state"] = self.rollout_initial_state.flatten(0, 1)
            flat["physics_source"] = self.physics_source.flatten(0, 1)
        flat.update({"actor_phys_" + name: value.flatten(0, 1)
                     for name, value in self.actor_physics.items()})
        for _ in range(epochs):
            for mini_batch in range(mini_batches):
                start = mini_batch * batch_size
                end = (mini_batch + 1) * batch_size
                idx = indices[start:end]
                yield {**{name: value[idx] for name, value in flat.items()}, "indices": idx}

    def clear(self):
        self.step = 0
