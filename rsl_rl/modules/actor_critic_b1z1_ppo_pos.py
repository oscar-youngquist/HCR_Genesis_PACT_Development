"""PACT MLPs without context, FiLM, or the feedforward-torque branch."""
import torch
from torch import nn
from torch.distributions import Normal
from .actor_critic_b1z1_pact import _mlp, _activation


class ActorCriticB1Z1PPOPos(nn.Module):
    is_recurrent = False

    def __init__(self, num_actor_obs, num_critic_obs, num_actions, *, actor_layers,
                 critic_layers, activation, init_noise_std, min_noise_std,
                 max_noise_std, **unused):
        super().__init__()
        self.actor_trunk = nn.Sequential(
            _mlp(num_actor_obs, actor_layers[:-1], actor_layers[-1], activation),
            _activation(activation))
        self.position_head = nn.Linear(actor_layers[-1], num_actions)
        self.critic = _mlp(num_critic_obs, critic_layers, 1, activation)

        def position_profile(value, name):
            value = torch.as_tensor(value, dtype=torch.float)
            if value.ndim == 1 and value.numel() == 2 * num_actions:
                value = value[:num_actions]  # PACT's position exploration profile.
            if value.ndim == 0:
                return value.repeat(num_actions)
            if value.shape != (num_actions,):
                raise ValueError(f"{name} must be scalar or a position/coupled action profile")
            return value.clone()

        self.std = nn.Parameter(position_profile(init_noise_std, "init_noise_std"))
        self.register_buffer("_std_clip_lwr", position_profile(min_noise_std, "min_noise_std"))
        self.register_buffer("_std_clip_upr", position_profile(max_noise_std, "max_noise_std"))
        self.distribution = None

    def act_inference(self, observations):
        return self.position_head(self.actor_trunk(observations))

    def update_distribution(self, observations):
        with torch.no_grad():
            self.std.copy_(torch.maximum(torch.minimum(self.std, self._std_clip_upr), self._std_clip_lwr))
        mean = self.act_inference(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None):
        pass

    def get_optim_groups(self, weight_decay=1e-6):
        # Match PACT: decay MLP parameters, but not Gaussian exploration scale.
        return [
            {"params": [p for name, p in self.named_parameters() if name != "std"],
             "weight_decay": weight_decay},
            {"params": [self.std], "weight_decay": 0.},
        ], []
