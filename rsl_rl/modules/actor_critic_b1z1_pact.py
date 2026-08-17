"""Coupled position/torque actor-critic for the standalone B1/Z1 PACT task."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal
from .module_utils import init_weights

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
    """UniFP-sized history VAE encoder producing only the compact latent."""

    def __init__(self, input_dim: int, latent_dim: int, hidden: Sequence[int], activation: str):
        super().__init__()
        trunk_dim = hidden[-1]
        
        self.trunk = nn.Sequential(_mlp(input_dim, hidden[:-1], trunk_dim, activation),
                                   _activation(activation))
        # Match UniFP: independent linear mean/log-variance projections from
        # the shared 128-D history feature, with bounded log variance.
        self.latent_mean = nn.Linear(trunk_dim, latent_dim)
        self.latent_logvar = nn.Sequential(
            nn.Linear(trunk_dim, latent_dim), nn.Hardtanh(-5.0, 5.0)
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Match UniFP's Xavier initialization order exactly."""
        self.trunk.apply(init_weights)
        self.latent_logvar.apply(init_weights)
        # UniFP initializes the mean projection after the log-variance head;
        # retaining that order also aligns seeded random-number consumption.
        self.latent_mean.apply(init_weights)


    def forward(self, history: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        feature = self.trunk(history)
        mean, logvar = self.latent_mean(feature), self.latent_logvar(feature)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if sample else mean
        return {"mean": mean, "logvar": logvar, "z": z}

    def forward_inf(self, history: torch.Tensor):
        feature = self.trunk(history)
        return {"z": self.latent_mean(feature)}

class FiLM(nn.Module):
    """Near-identity DreamFLEX-style modulation of the shared actor state."""

    def __init__(self, condition_dim: int, feature_dim: int, hidden_dim: int, activation: str):
        super().__init__()
        self.network = _mlp(condition_dim, [hidden_dim, hidden_dim], 2 * feature_dim, activation)

        self.network.apply(init_weights)

        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma, beta = self.network(condition).chunk(2, dim=-1)
        # gamma=0 and beta=0 are exactly the identity feature transform.
        identity_deviation = gamma.square().mean(dim=-1) + beta.square().mean(dim=-1)
        return features * (1.0 + gamma) + beta, gamma.abs().mean(dim=-1), identity_deviation


class B1Z1PACTDecoder(nn.Module):
    """MLP decoder used for explicit or privileged-state reconstruction."""

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
        context_layers=(512, 256, 128),
        explicit_decoder_layers=(128, 64),
        explicit_dim: int = 23,
        film_hidden_dim: int = 64,
        activation: str = "elu",
        init_noise_std: float | Sequence[float] = 0.65,
        min_noise_std: float | Sequence[float] = 0.20,
        max_noise_std: float | Sequence[float] = 5.0,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.context_encoder = B1Z1PACTContextEncoder(history_dim, latent_dim, context_layers, activation)
        # As in UniFP, all deployment-time explicit estimates are decoded from
        # z rather than branching directly from the history encoder trunk.
        self.explicit_decoder = B1Z1PACTDecoder(
            latent_dim, explicit_dim, hidden=explicit_decoder_layers, activation=activation
        )
        # The actor consumes estimated contact probabilities and foot heights;
        # FiLM intentionally receives neither terrain/contact signal.
        actor_input = num_actor_obs + latent_dim + 3 + 3 + 6 + 3 + 4 + 4
        self.actor_trunk = nn.Sequential(_mlp(actor_input, actor_layers[:-1], actor_layers[-1], activation),
                                         _activation(activation))


        # FiLM sees only predicted external disturbances and command errors.
        self.film = FiLM(6 + 3 + 3 + 3, actor_layers[-1], film_hidden_dim, activation)

        self.position_head = nn.Linear(actor_layers[-1], num_actions)
        self.torque_head = nn.Linear(actor_layers[-1], num_actions)
        nn.init.uniform_(self.torque_head.weight, -1.0e-6, 1.0e-6)
        nn.init.zeros_(self.torque_head.bias)

        self.critic = _mlp(num_critic_obs, critic_layers, 1, activation)

        action_dim = 2 * num_actions
        self.std = nn.Parameter(self._std_config_tensor(init_noise_std, action_dim, "init_noise_std"))
        self.register_buffer(
            "_std_clip_lwr",
            self._std_config_tensor(min_noise_std, action_dim, "min_noise_std"),
        )
        self.register_buffer(
            "_std_clip_upr",
            self._std_config_tensor(max_noise_std, action_dim, "max_noise_std"),
        )

        self.distribution: Normal | None = None

        self.last_context: dict[str, torch.Tensor] | None = None

        self.last_film_magnitude: torch.Tensor | None = None
        self.last_film_identity_deviation: torch.Tensor | None = None
        self.last_tracking_error_sq: torch.Tensor | None = None

        Normal.set_default_validate_args = False

    def decode_context(self, context: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Decode and name the 23-D explicit state used by actor and FiLM."""
        explicit = self.explicit_decoder(context["z"])
        return {
            **context,
            "explicit_prediction": explicit,
            "base_velocity": explicit[:, 0:3],
            "ee_position": explicit[:, 3:6],
            "base_wrench": explicit[:, 6:12],
            "ee_force": explicit[:, 12:15],
            "foot_contact_logits": explicit[:, 15:19],
            "foot_height": explicit[:, 19:23],
        }

    @staticmethod
    def _std_config_tensor(value, action_dim: int, name: str) -> torch.Tensor:
        """Expand a scalar or validate a per-action exploration profile."""
        tensor = torch.as_tensor(value, dtype=torch.float)
        if tensor.ndim == 0:
            return tensor.repeat(action_dim)
        if tensor.ndim != 1 or tensor.numel() != action_dim:
            raise ValueError(
                f"{name} must be a scalar or a flat list of {action_dim} values; "
                f"got shape {tuple(tensor.shape)}"
            )
        return tensor.clone()

    def _actor_inputs(self, obs: torch.Tensor, context: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Actor observation ends with [vx, vy, yaw_rate, radius, pitch, yaw].
        command = obs[:, -6:]
        # Both terms are component-wise normalized as [vx*lin, vy*lin,
        # yaw_rate*ang], so FiLM never compares unlike physical scales.
        base_error = command[:, :3] - context["base_velocity"]
        # Command and prediction share UniFP's scaled spherical representation.
        ee_error = command[:, 3:6] - context["ee_position"]
        # Retain the six-dimensional tracking error used by FiLM so PPO can
        # regularize modulation strength as a function of tracking quality.
        self.last_tracking_error_sq = torch.cat((base_error, ee_error), dim=-1).square().mean(dim=-1)
        contact_probability = torch.sigmoid(context["foot_contact_logits"])
        actor_input = torch.cat(
            (
                obs, context["z"], context["base_velocity"], context["ee_position"],
                context["base_wrench"], context["ee_force"],
                contact_probability, context["foot_height"],
            ),
            dim=-1,
        )
        # Contact state and foot height are excluded from FiLM by design.
        film_condition = torch.cat((context["base_wrench"], context["ee_force"], base_error, ee_error), dim=-1)
        return actor_input, film_condition

    def actor_forward(self, obs: torch.Tensor, context: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        actor_input, film_condition = self._actor_inputs(obs, context)
        features = self.actor_trunk(actor_input)
        features, magnitude, identity_deviation = self.film(features, film_condition)
        self.last_film_magnitude = magnitude
        self.last_film_identity_deviation = identity_deviation
        return self.position_head(features), self.torque_head(features)

    def update_distribution(
        self,
        obs: torch.Tensor,
        history: torch.Tensor,
        sample_context: bool = True,
    ) -> None:
        context = self.decode_context(self.context_encoder(history, sample=sample_context))
        # Boot masking is intentionally disabled: training and deployment
        # always condition on latent z and every predicted explicit output.
        position, torque = self.actor_forward(obs, context)
        mean = torch.cat((position, torque), dim=-1)
        self.std.data.copy_(
            torch.maximum(torch.minimum(self.std.data, self._std_clip_upr), self._std_clip_lwr)
        )
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        self.last_context = context

    def act(
        self, obs: torch.Tensor, history: torch.Tensor,
    ) -> torch.Tensor:
        # Keep the rollout policy deterministic with respect to its latent
        # estimate. PPO's stored action distribution then remains comparable
        # during the update; exploration is supplied by the action Gaussian.
        self.update_distribution(obs, history, sample_context=False)
        return self.distribution.sample()

    def act_inference(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        context = self.decode_context(self.context_encoder.forward_inf(history))
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
            if name.startswith(("context_encoder.", "explicit_decoder.")):
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
