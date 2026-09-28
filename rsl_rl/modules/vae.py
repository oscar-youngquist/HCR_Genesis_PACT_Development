"""DreamWaQ VAE with deterministic, typed explicit estimates.

Contact logits are supervised directly; only probabilities reach the actor and
reconstruction decoder. The implicit latent alone is Gaussian.
"""
from typing import NamedTuple

import torch
from torch import nn
from .actor_critic import get_activation


class ExplicitEstimatorOutput(NamedTuple):
    contact_logits: torch.Tensor
    contact_probability: torch.Tensor
    explicit_for_policy: torch.Tensor


class VAE(nn.Module):
    def __init__(self, num_history_input, num_latent_dims, num_explicit_dims,
                 num_decoder_output, activation='elu',
                 encoder_hidden_dims=(256, 128), decoder_hidden_dims=(256, 128),
                 contact_epsilon=1.e-6):
        super().__init__()
        if num_explicit_dims != 11:
            raise ValueError('DreamWaQ explicit estimates must be 11-D: velocity/contact/height')
        if not 0 <= contact_epsilon < 0.5:
            raise ValueError('contact_epsilon must lie in [0, 0.5)')
        self.num_explicit_dims = num_explicit_dims
        self.num_latent_dims = num_latent_dims
        self.contact_epsilon = float(contact_epsilon)
        layers = []
        width = num_history_input
        for hidden in encoder_hidden_dims:
            layers.extend((nn.Linear(width, hidden), get_activation(activation)))
            width = hidden
        self.encoder = nn.Sequential(*layers)
        self.latent_mu = nn.Linear(width, num_latent_dims)
        self.latent_var = nn.Sequential(nn.Linear(width, num_latent_dims), nn.Hardtanh(-5., 5.))
        self.explicit_head = nn.Linear(width, 11)
        layers = []
        width = num_latent_dims + num_explicit_dims
        for hidden in decoder_hidden_dims:
            layers.extend((nn.Linear(width, hidden), get_activation(activation)))
            width = hidden
        layers.append(nn.Linear(width, num_decoder_output))
        self.decoder = nn.Sequential(*layers)

    def encode(self, obs_history):
        encoded = self.encoder(obs_history)
        raw = self.explicit_head(encoded)
        logits = raw[:, 3:7]
        probability = self.contact_epsilon + (1 - 2 * self.contact_epsilon) * logits.sigmoid()
        explicit = ExplicitEstimatorOutput(
            logits, probability, torch.cat((raw[:, :3], probability, raw[:, 7:11]), dim=-1))
        return self.latent_mu(encoded), self.latent_var(encoded), explicit

    def decode(self, z, explicit):
        return self.decoder(torch.cat((z, explicit), dim=-1))

    def forward(self, obs_history):
        mu, logvar, explicit = self.encode(obs_history)
        return (self.reparameterize(mu, logvar), explicit.explicit_for_policy), (mu, logvar, explicit)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def sample(self, obs_history):
        return self.forward(obs_history)

    def inference(self, obs_history):
        mu, _, explicit = self.encode(obs_history)
        return torch.cat((mu, explicit.explicit_for_policy), dim=-1)
