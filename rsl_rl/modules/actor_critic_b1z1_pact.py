"""Coupled position/torque actor-critic for the standalone B1/Z1 PACT task."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Kaiming uniform initialization for weights
        torch.nn.init.xavier_uniform_(m.weight)
        # Initialize biases to zero if they exist
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def _activation(name: str) -> nn.Module:
    return {
        "elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh(), "swish": nn.SiLU(),
    }.get(name, nn.ELU())


def _mlp(in_dim: int, widths: Sequence[int], out_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for width in widths:
        layers.extend((nn.Linear(last, width), _activation(activation)))
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class B1Z1PACTContextEncoder(nn.Module):
    """History encoder predicting the deployment-time disturbance state."""

    def __init__(self, input_dim: int, latent_dim: int, hidden: Sequence[int], activation: str):
        super().__init__()
        trunk_dim = hidden[-1]

        hidden_dim = 2 * latent_dim

        self.trunk = _mlp(input_dim, hidden[:-1], trunk_dim, activation)

        # Stack some additional processing layers
        self.latent_mean = nn.Sequential(nn.Linear(trunk_dim, hidden_dim),
                                         _activation(activation),
                                         nn.Linear(hidden_dim, latent_dim))


        self.latent_logvar = nn.Sequential(nn.Linear(trunk_dim, hidden_dim),
                                           _activation(activation),
                                           nn.Linear(hidden_dim, latent_dim),
                                           nn.Hardtanh(-5.0, 5.0))

        self.base_velocity = nn.Sequential(nn.Linear(trunk_dim, hidden_dim),
                                           _activation(activation),
                                           nn.Linear(hidden_dim, 3))

        self.base_wrench = nn.Sequential(nn.Linear(trunk_dim, hidden_dim),
                                         _activation(activation),
                                         nn.Linear(hidden_dim, 6))

        self.ee_force = nn.Sequential(nn.Linear(trunk_dim, hidden_dim),
                                      _activation(activation),
                                      nn.Linear(hidden_dim, 3))

        # Binary logits are trained with BCE. Contact probabilities are kept
        # out of the actor interface and used only by explicit reconstruction.
        self.foot_contact_logits = nn.Sequential(nn.Linear(trunk_dim, hidden_dim),
                                                 _activation(activation),
                                                 nn.Linear(hidden_dim, 4))

    def _initialize_weights(self) -> None:
        """Initialize all linear layers with Xavier uniform distribution."""
        self.trunk.apply(init_weights)
        self.latent_mean.apply(init_weights)
        self.base_velocity.apply(init_weights)
        self.base_wrench.apply(init_weights)
        self.ee_force.apply(init_weights)
        self.foot_contact_logits.apply(init_weights)
        self.latent_logvar.apply(init_weights)


    def forward(self, history: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        feature = self.trunk(history)
        mean, logvar = self.latent_mean(feature), self.latent_logvar(feature)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if sample else mean
        return {
            "mean": mean, "logvar": logvar, "z": z,
            "base_velocity": self.base_velocity(feature),
            "base_wrench": self.base_wrench(feature),
            "ee_force": self.ee_force(feature),
            "foot_contact_logits": self.foot_contact_logits(feature),
        }

    def forward_inf(self, history: torch.Tensor):
        feature = self.trunk(history)
        z = self.latent_mean(feature)
        return {"z": z,
                "base_velocity": self.base_velocity(feature),
                "base_wrench": self.base_wrench(feature),
                "ee_force": self.ee_force(feature),
                "foot_contact_logits": self.foot_contact_logits(feature),
                }

class FiLM(nn.Module):
    """Near-identity DreamFLEX-style modulation of the shared actor state."""

    def __init__(self, condition_dim: int, feature_dim: int, hidden_dim: int, activation: str):
        super().__init__()
        self.network = _mlp(condition_dim, [hidden_dim, hidden_dim], 2 * feature_dim, activation)

        self.network.apply(init_weights)

        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.network(condition).chunk(2, dim=-1)
        return features * (1.0 + gamma) + beta, gamma.abs().mean(dim=-1)


class B1Z1PACTDecoder(nn.Module):
    """Independent decoder head used for force or privileged-state targets."""

    def __init__(self, input_dim: int, output_dim: int, hidden=(128, 256, 128), activation: str = "elu"):
        super().__init__()
        self.network = _mlp(input_dim, hidden, output_dim, activation)
        self.network.apply(init_weights)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.network(condition)


class ActorCriticB1Z1PACT(nn.Module):
    """PACT actor with 17 position and 17 feedforward-torque outputs."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        history_dim: int,
        latent_dim: int = 16,
        actor_layers=(512, 256, 128),
        critic_layers=(1024, 256, 128),
        context_layers=(256, 128),
        film_hidden_dim: int = 64,
        activation: str = "elu",
        init_noise_std: float = 0.65,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.context_encoder = B1Z1PACTContextEncoder(history_dim, latent_dim, context_layers, activation)
        # The actor consumes the four estimated foot-contact probabilities;
        # FiLM intentionally does not receive them.
        actor_input = num_actor_obs + latent_dim + 3 + 6 + 3 + 4
        self.actor_trunk = _mlp(actor_input, actor_layers[:-1], actor_layers[-1], activation)

        # FiLM may use only predicted disturbances, tracking errors, and z.
        self.film = FiLM(latent_dim + 6 + 3 + 3 + 3, actor_layers[-1], film_hidden_dim, activation)

        self.position_head = nn.Linear(actor_layers[-1], num_actions)
        self.torque_head = nn.Linear(actor_layers[-1], num_actions)
        nn.init.uniform_(self.torque_head.weight, -1.0e-6, 1.0e-6)
        nn.init.zeros_(self.torque_head.bias)

        self.critic = _mlp(num_critic_obs, critic_layers, 1, activation)

        self.std = nn.Parameter(torch.full((2 * num_actions,), init_noise_std))

        self._std_clip_lwr = 0.20

        self.distribution: Normal | None = None

        self.last_context: dict[str, torch.Tensor] | None = None

        self.last_film_magnitude: torch.Tensor | None = None

        Normal.set_default_validate_args = False

    def _bootmasked_context(
        self,
        context: dict[str, torch.Tensor],
        explicit_labels: torch.Tensor | None,
        mask_latent: bool,
        mask_explicit: bool,
    ) -> dict[str, torch.Tensor]:
        """Apply independent PACT masks to latent and explicit context."""
        if not mask_latent and not mask_explicit:
            return context
        if mask_explicit and explicit_labels is None:
            raise ValueError("B1Z1 PACT boot masking requires explicit labels")

        masked = {key: value for key, value in context.items()}
        if mask_latent:
            # No privileged target exists for z, so its masked replacement is zero.
            masked["z"] = torch.zeros_like(context["z"])
        if mask_explicit:
            # Explicit state has privileged simulator labels during training.
            masked["base_velocity"] = explicit_labels[:, :3]
            masked["base_wrench"] = explicit_labels[:, 3:9]
            masked["ee_force"] = explicit_labels[:, 9:12]
        return masked

    def _actor_inputs(self, obs: torch.Tensor, context: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Actor observation ends with [vx, vy, yaw_rate, radius, pitch, yaw].
        command = obs[:, -6:]
        base_error = command[:, :3] - context["base_velocity"]
        # The environment appends FK EE tracking error immediately before commands.
        ee_error = obs[:, -9:-6]
        contact_probability = torch.sigmoid(context["foot_contact_logits"])
        actor_input = torch.cat(
            (obs, context["z"], context["base_velocity"], context["base_wrench"], context["ee_force"], contact_probability),
            dim=-1,
        )
        # Contact state is excluded from FiLM by design.
        film_condition = torch.cat((context["z"], context["base_wrench"], context["ee_force"], base_error, ee_error), dim=-1)
        return actor_input, film_condition

    def actor_forward(self, obs: torch.Tensor, context: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        actor_input, film_condition = self._actor_inputs(obs, context)
        features = self.actor_trunk(actor_input)
        features, magnitude = self.film(features, film_condition)
        self.last_film_magnitude = magnitude
        return self.position_head(features), self.torque_head(features)

    def update_distribution(
        self,
        obs: torch.Tensor,
        history: torch.Tensor,
        sample_context: bool = True,
        explicit_labels: torch.Tensor | None = None,
        mask_latent: bool = False,
        mask_explicit: bool = False,
    ) -> None:
        context = self.context_encoder(history, sample=sample_context)
        actor_context = self._bootmasked_context(context, explicit_labels, mask_latent, mask_explicit)
        position, torque = self.actor_forward(obs, actor_context)
        mean = torch.cat((position, torque), dim=-1)
        self.std.data.clamp_(self._std_clip_lwr, 5.0)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        self.last_context = context

    def act(
        self, obs: torch.Tensor, history: torch.Tensor,
        explicit_labels: torch.Tensor | None = None, mask_latent: bool = False, mask_explicit: bool = False,
    ) -> torch.Tensor:
        # Keep the rollout policy deterministic with respect to its latent
        # estimate. PPO's stored action distribution then remains comparable
        # during the update; exploration is supplied by the action Gaussian.
        self.update_distribution(
            obs, history, sample_context=False, explicit_labels=explicit_labels,
            mask_latent=mask_latent, mask_explicit=mask_explicit,
        )
        return self.distribution.sample()

    def act_inference(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        context = self.context_encoder.forward_inf(history)
        position, torque = self.actor_forward(obs, context)
        actions = torch.cat((position, torque), dim=-1)
        return actions

    def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None) -> None:
        return None

    def get_optim_groups(self, weight_decay: float = 1e-6, strong_decay: float = 1e-1):
        """Return disjoint AdamW groups for this B1/Z1 actor-critic.

        The legacy PACT implementation recognized actor tensors by an ``act``
        substring. This model instead names them ``actor_trunk``, ``film``,
        ``position_head``, and ``torque_head``. Group by the real top-level
        ownership so callers can safely reuse the encoder groups in a shared
        context/decoder optimizer.
        """
        groups = {
            "actor": [], "actor_no_decay": [], "film": [],
            "critic": [], "critic_no_decay": [],
            "encoder": [], "encoder_no_decay": [],
        }
        for name, parameter in self.named_parameters():
            if name.startswith("context_encoder."):
                owner = "encoder"
            elif name.startswith("critic."):
                owner = "critic"
            else:
                owner = "actor"
            # Do not decay biases or the learned Gaussian exploration scale.
            if name.endswith(".bias") or name == "std":
                groups[f"{owner}_no_decay"].append(parameter)
            elif name.startswith("film."):
                groups["film"].append(parameter)
            else:
                groups[owner].append(parameter)

        def group(parameters, decay, name):
            return {"params": parameters, "weight_decay": decay, "name": name}

        actor_groups = [
            group(groups["actor"], weight_decay, "actor"),
            group(groups["film"], strong_decay, "film"),
            group(groups["critic"], weight_decay, "critic"),
            group(groups["actor_no_decay"] + groups["critic_no_decay"], 0.0, "actor_critic_no_decay"),
        ]
        encoder_groups = [
            group(groups["encoder"], weight_decay, "encoder"),
            group(groups["encoder_no_decay"], 0.0, "encoder_no_decay"),
        ]
        # Empty groups confuse some PyTorch versions and indicate no missing
        # parameters only when they are filtered here, not silently ignored.
        return [item for item in actor_groups if item["params"]], [item for item in encoder_groups if item["params"]]
