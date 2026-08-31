"""Named rollout storage for Go2 HardPACT transitions."""

from __future__ import annotations

from typing import Dict, Mapping

import torch

from rsl_rl.go2_hard_pact_schema import (
    TRANSITION_FIELD_DIMS,
    validate_transition,
)


class RolloutStorageGo2HardPACT:
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
            "action_mean": torch.zeros(
                *prefix, 12 if position_pretraining else 24, device=self.device
            ),
            "action_std": torch.zeros(
                *prefix, 12 if position_pretraining else 24, device=self.device
            ),
        }
        for name, width in TRANSITION_FIELD_DIMS.items():
            if position_pretraining and name in {"raw_action", "exploration_noise"}:
                width = 12
            dtype = torch.long if name in {"qp_fallback", "qp_status"} else torch.float32
            self.data[name] = torch.zeros(*prefix, width, device=self.device, dtype=dtype)
        self.step = 0

    def add(self, core: Mapping[str, torch.Tensor], transition: Mapping[str, torch.Tensor]):
        if self.step >= self.num_steps:
            raise RuntimeError("HardPACT rollout storage overflow")
        validate_transition(transition, position_pretraining=self.position_pretraining)
        required = {
            "observation", "critic_observation", "history", "reward", "done",
            "value", "raw_action_log_probability", "action_mean", "action_std",
        }
        missing = required - set(core)
        if missing:
            raise KeyError(f"missing rollout core fields: {sorted(missing)}")
        for name in required:
            value = core[name]
            if name in {"reward", "done", "raw_action_log_probability"} and value.ndim == 1:
                value = value.unsqueeze(-1)
            self.data[name][self.step].copy_(value)
        for name in TRANSITION_FIELD_DIMS:
            self.data[name][self.step].copy_(transition[name])
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
        flattened = {name: value.flatten(0, 1) for name, value in self.data.items()}
        for _ in range(int(num_epochs)):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, mini_batch_size * int(num_mini_batches), mini_batch_size):
                selected = indices[start:start + mini_batch_size]
                yield {name: value[selected] for name, value in flattened.items()}

    def clear(self):
        self.step = 0
