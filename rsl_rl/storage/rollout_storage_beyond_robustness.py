"""TS-style owned, timestamp-aligned feed-forward storage and reset-safe history."""
import torch
from .rollout_storage import RolloutStorage
from rsl_rl.beyond_robustness_schema import BeyondRobustnessInputSchema


class ProprioceptionHistory:
    def __init__(self, num_envs, length, device, observation_dim=45):
        self.frames = torch.zeros(num_envs, length, observation_dim, device=device)

    def preview(self, observation, reset, env_ids=None):
        # Zero padding represents unavailable history; never copy the preceding episode.
        frames = self.frames if env_ids is None else self.frames[env_ids]
        previous = torch.where(reset[:, None, None], 0., frames[:, 1:])
        return torch.cat((previous, observation[:, None]), dim=1)

    def append(self, observation, reset):
        self.frames.copy_(self.preview(observation, reset))
        return self.frames.flatten(1)


class RolloutStorageBeyondRobustness(RolloutStorage):
    def __init__(self, num_envs, steps, history_length, device="cpu", schema=None):
        self.schema = schema or BeyondRobustnessInputSchema()
        super().__init__(num_envs, steps, (self.schema.observation_dim,), (self.schema.privileged_dim,), (12,), device)
        self.histories = torch.zeros(steps, num_envs, history_length * self.schema.observation_dim, device=device)
        self.load_labels = torch.zeros(steps, num_envs, self.schema.load_dim, device=device)
        self.supervision_valid = torch.ones(steps, num_envs, dtype=torch.bool, device=device)
        self.timeouts = torch.zeros_like(self.dones, dtype=torch.bool)

    @torch.no_grad()
    def add(self, snapshot, rewards, dones, timeouts, valid):
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError("Rollout buffer overflow")
        k = self.step
        for target, source in ((self.observations, "observations"), (self.histories, "history"),
                               (self.actions, "actions"), (self.values, "values"),
                               (self.actions_log_prob, "log_prob"), (self.mu, "mu"),
                               (self.sigma, "sigma"), (self.load_labels, "load_labels")):
            target[k].copy_(snapshot[source])
        self.privileged_observations[k].copy_(torch.cat((snapshot["state"], snapshot["parameters"]), -1))
        self.rewards[k].copy_(rewards.reshape(-1, 1))
        self.dones[k].copy_(dones.reshape(-1, 1))
        self.timeouts[k].copy_(timeouts.reshape(-1, 1))
        self.supervision_valid[k].copy_(valid)
        self.step += 1

    @torch.no_grad()
    def compute_returns(self, last_values, gamma, lam):
        if self.step != self.num_transitions_per_env:
            raise RuntimeError("Finish rollout before computing returns")
        # Timeouts are bootstrapped into rewards using explicit terminal values by PPO;
        # done still cuts GAE so neither failures nor timeouts connect different episodes.
        advantage = torch.zeros_like(last_values)
        for k in reversed(range(self.step)):
            next_value = last_values if k == self.step - 1 else self.values[k + 1]
            alive = 1. - self.dones[k].float()
            delta = self.rewards[k] + gamma * alive * next_value - self.values[k]
            advantage = delta + gamma * lam * alive * advantage
            self.returns[k] = advantage + self.values[k]
        raw = self.returns - self.values
        self.advantages.copy_((raw - raw.mean()) / (raw.std(unbiased=False) + 1e-8))

    def batches(self, num_batches, epochs=1):
        size = self.step * self.num_envs
        if not 0 < num_batches <= size:
            raise ValueError("num_mini_batches must be between 1 and rollout row count")
        fields = dict(observations=self.observations, history=self.histories,
                      state=self.privileged_observations[..., :self.schema.state_dim],
                      parameters=self.privileged_observations[..., self.schema.state_dim:],
                      load_labels=self.load_labels, valid=self.supervision_valid,
                      actions=self.actions, values=self.values, returns=self.returns,
                      advantages=self.advantages, log_prob=self.actions_log_prob,
                      mu=self.mu, sigma=self.sigma)
        for _ in range(epochs):
            # Unlike the legacy TS generator, keep the remainder and reshuffle each epoch.
            for indices in torch.randperm(size, device=self.device).tensor_split(num_batches):
                yield {key: value[:self.step].flatten(0, 1)[indices] for key, value in fields.items()}
