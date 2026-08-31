"""Named rollout storage for Go2 HardPACT transitions."""

from __future__ import annotations

from typing import Dict

import torch

from rsl_rl.go2_hard_pact_schema import (
    TRANSITION_FIELD_DIMS,
    validate_transition,
)


class RolloutStorageGo2HardPACT:
    class Transition:
        """PACT transition plus the additional HardPACT physics payload."""

        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.observation_history = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hard_pact = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_steps,
        observation_dim=57,
        critic_observation_dim=198,
        history_dim=57 * 20,
        *,
        position_pretraining=False,
        device="cpu",
    ):
        self.num_envs = int(num_envs)
        self.num_steps = int(num_steps)
        self.position_pretraining = bool(position_pretraining)
        self.device = torch.device(device)
        prefix = (self.num_steps, self.num_envs)
        self.data: Dict[str, torch.Tensor] = {
            "observation": torch.zeros(*prefix, observation_dim, device=self.device),
            "critic_observation": torch.zeros(*prefix, critic_observation_dim, device=self.device),
            "history": torch.zeros(*prefix, history_dim, device=self.device),
            "reward": torch.zeros(*prefix, 1, device=self.device),
            "done": torch.zeros(*prefix, 1, device=self.device, dtype=torch.bool),
            "value": torch.zeros(*prefix, 1, device=self.device),
            "return": torch.zeros(*prefix, 1, device=self.device),
            "advantage": torch.zeros(*prefix, 1, device=self.device),
            "raw_action_log_probability": torch.zeros(*prefix, 1, device=self.device),
            "raw_action": torch.zeros(
                *prefix, 12 if position_pretraining else 24, device=self.device
            ),
            "action_mean": torch.zeros(
                *prefix, 12 if position_pretraining else 24, device=self.device
            ),
            "action_std": torch.zeros(
                *prefix, 12 if position_pretraining else 24, device=self.device
            ),
        }
        for name, width in TRANSITION_FIELD_DIMS.items():
            dtype = torch.long if name in {
                "qp_fallback", "qp_status",
            } else torch.float32
            self.data[name] = torch.zeros(*prefix, width, device=self.device, dtype=dtype)
        self.step = 0

    def add_transitions(self, transition):
        if self.step >= self.num_steps:
            raise RuntimeError("HardPACT rollout storage overflow")
        hard_pact = transition.hard_pact
        if hard_pact is None:
            raise ValueError("HardPACT transition payload is required")
        validate_transition(hard_pact)
        core = {
            "observation": transition.observations,
            "critic_observation": transition.critic_observations,
            "history": transition.observation_history,
            "reward": transition.rewards,
            "done": transition.dones,
            "value": transition.values,
            "raw_action": transition.actions,
            "raw_action_log_probability": transition.actions_log_prob,
            "action_mean": transition.action_mean,
            "action_std": transition.action_sigma,
        }
        missing = [name for name, value in core.items() if value is None]
        if missing:
            raise ValueError(f"unset PACT transition fields: {missing}")
        for name, value in core.items():
            if name in {"reward", "done", "raw_action_log_probability"} and value.ndim == 1:
                value = value.unsqueeze(-1)
            self.data[name][self.step].copy_(value)
        for name in TRANSITION_FIELD_DIMS:
            self.data[name][self.step].copy_(hard_pact[name])
        self.step += 1

    def compute_returns(self, last_value, gamma=0.99, lam=0.95):
        advantage = torch.zeros_like(last_value)
        for step in reversed(range(self.num_steps)):
            next_value = last_value if step == self.num_steps - 1 else self.data["value"][step + 1]
            not_done = ~self.data["done"][step]
            delta = self.data["reward"][step] + gamma * not_done * next_value - self.data["value"][step]
            advantage = delta + gamma * lam * not_done * advantage
            self.data["return"][step] = advantage + self.data["value"][step]
        raw_advantage = self.data["return"] - self.data["value"]
        self.data["advantage"] = (
            raw_advantage - raw_advantage.mean()
        ) / raw_advantage.std().clamp_min(1.0e-8)

    def mini_batches(self, num_mini_batches, num_epochs=1):
        batch_size = self.num_steps * self.num_envs
        mini_batch_size = batch_size // int(num_mini_batches)
        if mini_batch_size == 0:
            raise ValueError("more mini-batches than samples")
        flattened = {
            name: value.flatten(0, 1) for name, value in self.data.items()
        }
        # Match legacy PACT: one rollout permutation is reused across epochs.
        indices = torch.randperm(
            mini_batch_size * int(num_mini_batches), device=self.device
        )
        for _ in range(int(num_epochs)):
            for start in range(0, mini_batch_size * int(num_mini_batches), mini_batch_size):
                selected = indices[start:start + mini_batch_size]
                yield {name: value[selected] for name, value in flattened.items()}

    def clear(self):
        self.step = 0
