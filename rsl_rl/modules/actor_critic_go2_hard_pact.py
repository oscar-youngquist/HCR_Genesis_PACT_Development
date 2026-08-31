"""Actor/critic and deployment physics estimator for Go2 HardPACT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal


def _activation(name: str):
    options = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "swish": nn.SiLU,
    }
    try:
        return options[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported activation {name!r}") from exc


def _mlp(input_dim: int, hidden: Sequence[int], output_dim: int, activation: str):
    act = _activation(activation)
    layers = []
    last = input_dim
    for width in hidden:
        layers.extend((nn.Linear(last, width), act()))
        last = width
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


@dataclass
class EncoderOutput:
    latent_mean: torch.Tensor
    latent_log_variance: torch.Tensor
    explicit: torch.Tensor
    latent: torch.Tensor


@dataclass
class PhysicsReferences:
    grf_yaw_scaled: torch.Tensor
    base_wrench_yaw_scaled: torch.Tensor


class Go2HistoryEncoder(nn.Module):
    """History encoder with a deterministic 16-D policy latent and 11-D estimator."""

    def __init__(self, history_dim=57 * 20, latent_dim=16, explicit_dim=11, hidden=(256, 128), activation="elu"):
        super().__init__()
        if latent_dim != 16 or explicit_dim != 11:
            raise ValueError("Go2 HardPACT requires latent_dim=16 and explicit_dim=11")
        act = _activation(activation)
        layers = []
        last = history_dim
        for width in hidden:
            layers.extend((nn.Linear(last, width), act()))
            last = width
        self.trunk = nn.Sequential(*layers)
        self.latent_mean = nn.Linear(last, latent_dim)
        self.latent_log_variance = nn.Linear(last, latent_dim)
        self.explicit_head = nn.Linear(last, explicit_dim)

    @staticmethod
    def _explicit_transform(raw: torch.Tensor) -> torch.Tensor:
        velocity = raw[:, :3]
        contact_probability = raw[:, 3:7].sigmoid()
        clearance = raw[:, 7:11]
        return torch.cat((velocity, contact_probability, clearance), dim=-1)

    def forward(self, history, *, sample_for_auxiliary=False) -> EncoderOutput:
        features = self.trunk(history)
        mean = self.latent_mean(features)
        log_variance = self.latent_log_variance(features).clamp(-10.0, 5.0)
        explicit = self._explicit_transform(self.explicit_head(features))
        if sample_for_auxiliary:
            latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        else:
            # Policy collection, policy updates, and deployment all use means.
            latent = mean
        return EncoderOutput(mean, log_variance, explicit, latent)


class DeploymentPhysicsEstimator(nn.Module):
    """Non-privileged GRF/wrench heads evaluated before the QP."""

    def __init__(
        self,
        latent_dim=16,
        explicit_dim=11,
        hidden=(128, 128),
        activation="elu",
        grf_scale=(1.2, 1.2, 2.5) * 4,
        wrench_scale=(0.6, 0.6, 0.9924, 0.12, 0.12, 0.12),
    ):
        super().__init__()
        self.grf_head = _mlp(latent_dim + explicit_dim + 12, hidden, 12, activation)
        self.wrench_head = _mlp(latent_dim + explicit_dim, hidden, 6, activation)
        self.register_buffer("grf_scale", torch.tensor(
            grf_scale, dtype=torch.float32
        ))
        self.register_buffer("wrench_scale", torch.tensor(
            wrench_scale, dtype=torch.float32
        ))
        if self.grf_scale.shape != (12,):
            raise ValueError("grf_scale must contain twelve scaled GRF limits")
        if self.wrench_scale.shape != (6,):
            raise ValueError("wrench_scale must contain six force/torque scales")

    def forward(self, latent, explicit, nominal_torque) -> PhysicsReferences:
        if nominal_torque.shape[-1] != 12:
            raise ValueError("nominal torque must be 12-D")
        stopped_explicit = explicit.detach()
        grf = self.grf_head(torch.cat((latent, stopped_explicit, nominal_torque), dim=-1))
        wrench = self.wrench_head(torch.cat((latent, stopped_explicit), dim=-1))
        return PhysicsReferences(
            grf_yaw_scaled=grf * self.grf_scale,
            base_wrench_yaw_scaled=wrench * self.wrench_scale,
        )


class ActorCriticGo2HardPACT(nn.Module):
    """Concatenation-conditioned coupled policy shared with HardPACTPos.

    The two actor heads exist in both stages. HardPACT samples all 24 outputs;
    HardPACTPos samples the position head and clone-supervises the feedforward
    head in pre-motor units for strict transfer.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs=57,
        num_critic_obs=198,
        num_actions=12,
        history_dim=57 * 20,
        latent_dim=16,
        explicit_dim=11,
        actor_layers=(512, 256, 128),
        critic_layers=(512, 256, 128),
        encoder_layers=(256, 128),
        physics_head_layers=(128, 128),
        grf_scale=(1.2, 1.2, 2.5) * 4,
        wrench_scale=(0.6, 0.6, 0.9924, 0.12, 0.12, 0.12),
        activation="elu",
        init_noise_std=1.0,
        position_pretraining=False,
        **unused,
    ):
        super().__init__()
        if num_actor_obs != 57 or num_actions != 12:
            raise ValueError("Go2 HardPACT requires 57 observations and 12 joints")
        self.position_pretraining = bool(position_pretraining)
        self.num_joints = 12
        self.action_dim = 12 if self.position_pretraining else 24
        self.history_encoder = Go2HistoryEncoder(
            history_dim, latent_dim, explicit_dim, encoder_layers, activation
        )
        act = _activation(activation)
        actor_input = num_actor_obs + latent_dim + explicit_dim
        trunk = []
        last = actor_input
        for width in actor_layers:
            trunk.extend((nn.Linear(last, width), act()))
            last = width
        self.actor_trunk = nn.Sequential(*trunk)
        self.position_head = nn.Linear(last, 12)
        # This exact head is present and clone-trained in the position stage.
        self.feedforward_head = nn.Linear(last, 12)
        self.critic = _mlp(num_critic_obs, critic_layers, 1, activation)
        self.physics_estimator = DeploymentPhysicsEstimator(
            latent_dim, explicit_dim, physics_head_layers, activation,
            grf_scale=grf_scale,
            wrench_scale=wrench_scale,
        )
        self.privileged_decoder = _mlp(
            latent_dim + explicit_dim, (128, 256, 512), 79, activation
        )
        self.std = nn.Parameter(torch.full((self.action_dim,), float(init_noise_std)))
        self.distribution = None
        self._last_encoder_output = None
        self._last_feedforward_mean = None
        self._physics_evaluations = 0
        nn.init.uniform_(self.position_head.weight, -3.0e-2, 3.0e-2)
        nn.init.uniform_(self.feedforward_head.weight, -3.0e-6, 3.0e-6)
        nn.init.zeros_(self.position_head.bias)
        nn.init.zeros_(self.feedforward_head.bias)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def feedforward_mean(self):
        return self._last_feedforward_mean

    def encode_policy_history(self, history):
        return self.history_encoder(history, sample_for_auxiliary=False)

    def encode_auxiliary_history(self, history, *, sample=True):
        return self.history_encoder(history, sample_for_auxiliary=sample)

    def actor_outputs(self, observation, encoder_output: EncoderOutput):
        conditioning = torch.cat((
            observation, encoder_output.latent, encoder_output.explicit
        ), dim=-1)
        features = self.actor_trunk(conditioning)
        return self.position_head(features), self.feedforward_head(features)

    def update_distribution(self, observation, history):
        encoded = self.encode_policy_history(history)
        position, feedforward = self.actor_outputs(observation, encoded)
        self._last_feedforward_mean = feedforward
        mean = position if self.position_pretraining else torch.cat((position, feedforward), dim=-1)
        std = self.std.clamp_min(1.0e-4).expand_as(mean)
        self.distribution = Normal(mean, std)
        self._last_encoder_output = encoded
        return encoded

    def act(self, observation, history, **kwargs):
        self.update_distribution(observation, history)
        return self.distribution.sample()

    def act_inference(self, observation, history):
        self.update_distribution(observation, history)
        return self.distribution.mean

    def evaluate_actions(self, observation, history, raw_sampled_action):
        self.update_distribution(observation, history)
        return (
            self.distribution.log_prob(raw_sampled_action).sum(dim=-1),
            self.entropy,
        )

    def evaluate(self, critic_observation):
        return self.critic(critic_observation)

    def physics_references(self, history, nominal_torque, *, encoded=None):
        # Call exactly once at each control/QP site; corrected torque is never an input.
        if encoded is None:
            encoded = self.encode_policy_history(history)
        self._physics_evaluations += 1
        return self.physics_estimator(encoded.latent, encoded.explicit, nominal_torque)

    def reconstruct_privileged(self, history, *, sample_for_auxiliary=False):
        encoded = self.encode_auxiliary_history(history, sample=sample_for_auxiliary)
        prediction = self.privileged_decoder(torch.cat((
            encoded.latent, encoded.explicit
        ), dim=-1))
        return prediction, encoded

    def get_auxiliary_optim_groups(self):
        """B1Z1-style adaptation ownership for the second optimizer."""
        encoder_parameters = [
            parameter
            for name, parameter in self.history_encoder.named_parameters()
            if not name.startswith("explicit_head.")
        ]
        return [
            {"params": encoder_parameters, "name": "encoder"},
            {
                "params": list(self.privileged_decoder.parameters()),
                "name": "decoder",
            },
            {
                "params": list(self.history_encoder.explicit_head.parameters()),
                "name": "estimator",
            },
            {
                "params": list(self.physics_estimator.parameters()),
                "name": "force_estimator",
            },
        ]

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        return None


class ActorCriticGo2HardPACTPos(ActorCriticGo2HardPACT):
    def __init__(self, *args, **kwargs):
        kwargs["position_pretraining"] = True
        super().__init__(*args, **kwargs)


@dataclass
class MigrationReport:
    loaded: tuple[str, ...]
    reinitialized: tuple[str, ...]
    ignored_source: tuple[str, ...]


def migrate_hard_pact_pos_checkpoint(
    target: ActorCriticGo2HardPACT,
    checkpoint: Mapping[str, object],
) -> MigrationReport:
    """Strictly migrate the architecture-compatible position checkpoint.

    Only the second 12 entries of the exploration standard deviation may be
    initialized when moving from a 12-D position distribution to the 24-D
    coupled distribution. Every model tensor is then loaded with ``strict=True``.
    """
    if target.position_pretraining:
        raise ValueError("migration target must be a coupled HardPACT policy")
    source = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(source, Mapping):
        raise TypeError("checkpoint must contain a model state mapping")
    target_state = target.state_dict()
    migrated = {}
    loaded = []
    reinitialized = []
    configured_buffers = {
        "physics_estimator.grf_scale",
        "physics_estimator.wrench_scale",
    }
    for key, target_value in target_state.items():
        if key not in source:
            raise RuntimeError(f"undocumented missing HardPACTPos key: {key}")
        source_value = source[key]
        if not torch.is_tensor(source_value):
            raise TypeError(f"checkpoint value {key} is not a tensor")
        if key in configured_buffers and source_value.shape == target_value.shape:
            # Force/wrench output units and ranges are part of the current
            # task contract, not learned checkpoint state.  This documented
            # migration lets pre-payload position checkpoints initialize the
            # expanded vertical torso-force range safely.
            migrated[key] = target_value
            if torch.equal(source_value.to(target_value), target_value):
                loaded.append(key)
            else:
                reinitialized.append(key)
        elif source_value.shape == target_value.shape:
            migrated[key] = source_value
            loaded.append(key)
        elif key == "std" and source_value.shape == (12,) and target_value.shape == (24,):
            value = target_value.clone()
            value[:12].copy_(source_value)
            migrated[key] = value
            reinitialized.append("std[12:24]")
        else:
            raise RuntimeError(
                f"undocumented HardPACTPos shape mismatch for {key}: "
                f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}"
            )
    allowed_source = {"position_pretraining"}
    ignored = tuple(sorted(set(source) - set(target_state)))
    undocumented = set(ignored) - allowed_source
    if undocumented:
        raise RuntimeError(f"undocumented extra HardPACTPos checkpoint keys: {sorted(undocumented)}")
    target.load_state_dict(migrated, strict=True)
    return MigrationReport(tuple(loaded), tuple(reinitialized), ignored)
