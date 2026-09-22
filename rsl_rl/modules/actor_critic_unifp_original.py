"""Upstream UniFP deterministic architecture; independent of the extended VAE."""
import torch
from torch import nn
from torch.distributions import Normal


def mlp(widths):
    layers = []
    for index, (source, target) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(source, target))
        if index < len(widths) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class ActorCriticUniFPOriginal(nn.Module):
    is_recurrent = False
    schema = ("base_velocity", "ee_spherical_position", "ee_external_force", "base_external_force")
    architecture = "unifp_original_73x32_z64_explicit12_v1"
    adaptation_labels = schema
    adaptation_dims = (3, 3, 3, 3)
    adaptation_weights = (.2, .2, 1., 1.)

    def __init__(self, num_obs, num_privileged_obs, num_obs_pred=12,
                 num_single_obs=73, num_actions=17):
        super().__init__()
        if (num_obs, num_single_obs, num_obs_pred, num_actions) != (2336, 73, 12, 17):
            raise ValueError("Original UniFP requires history=32*73, labels=12, actions=17")
        self.num_obs_now = 73
        self.adaptation_encoder_module = mlp([2336, 512, 256, 128, 64])
        self.adaptation_decoder_module = mlp([64, 128, 64, 12])
        self.actor_body = mlp([137, 512, 256, 128, 17])
        self.critic_body = mlp([num_privileged_obs, 512, 256, 128, 1])
        self.std = nn.Parameter(torch.ones(17))
        self.distribution = None
        self.last_prediction = None

    def update_distribution(self, history):
        z = self.adaptation_encoder_module(history)
        # Estimation never conditions the actor; cache it for the external adapter.
        self.last_prediction = self.adaptation_decoder_module(z)
        mean = self.actor_body(torch.cat((history[:, -73:], z), -1))
        self.distribution = Normal(mean, self.std.expand_as(mean), validate_args=False)

    def act(self, history, **kwargs):
        self.update_distribution(history)
        return self.distribution.sample()

    def act_inference(self, observations, policy_info=None):
        history = observations["obs"] if isinstance(observations, dict) else observations
        self.update_distribution(history)
        if policy_info is not None:
            policy_info["latents"] = self.last_prediction.detach()
        return self.action_mean

    def evaluate(self, observations):
        return self.critic_body(observations)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def reset(self, dones=None):
        pass

    def load_state_dict(self, state_dict, strict=True):
        expected = self.state_dict()
        if set(state_dict) != set(expected) or any(
                state_dict[k].shape != expected[k].shape for k in expected if k in state_dict):
            raise RuntimeError("Incompatible UniFP architecture/schema: expected deterministic "
                               "73x32 history, z64, 12-D [velocity, EE position, EE force, base force]. "
                               "Extended UniFP/VAE checkpoints cannot be loaded into this baseline.")
        return super().load_state_dict(state_dict, strict=strict)
