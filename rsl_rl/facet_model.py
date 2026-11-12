"""FACET ordinary PyTorch API, used by the FACET-only Stage 5 runner.

Ported from btx0424/active-adaptation c0d6f5aa6146814803090b5f6f19aafb8ce6f792
(ppo_facet.py, common.py, vecnorm.py, distributions.py). See FACET_LICENSE.txt.
"""
import torch
from torch import nn
from torch.distributions import Independent, Normal


def mlp(input_dim, widths):
    layers = []
    for width in widths:
        layers.extend((nn.Linear(input_dim, width), nn.LayerNorm(width), nn.Mish()))
        input_dim = width
    return nn.Sequential(*layers)


class FACETVecNorm(nn.Module):
    """Upstream decay=1 statistics, including its initial zero pseudo-sample.

    Only update() mutates statistics. Evaluation and repeated PPO queries never
    update, even in train mode; the collector owns the once/batch update.
    """
    def __init__(self, width):
        super().__init__()
        self.register_buffer("sum", torch.zeros(width))
        self.register_buffer("ssq", torch.zeros(width))
        self.register_buffer("count", torch.tensor(1.))
        self.eps = 1e-5

    @torch.no_grad()
    def update(self, value):
        value = value.detach().reshape(-1, self.sum.numel())
        if not len(value):
            return
        self.count.add_(len(value))
        weight = len(value) / self.count
        self.sum.lerp_(value.mean(0), weight)
        self.ssq.lerp_(value.square().mean(0), weight)

    def forward(self, value):
        return (value-self.sum) / (self.ssq-self.sum.square()).clamp_min(self.eps).sqrt()


class FACETActor(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.mlp = mlp(observation_dim+128+32, (256, 256, 256))
        self.mean = nn.Linear(256, action_dim)
        self.std = nn.Parameter(torch.ones(action_dim))

    def forward(self, observation, privileged_features, force_features):
        mean = self.mean(self.mlp(torch.cat((observation, privileged_features, force_features), -1)))
        # Clamp only at distribution construction, not the learned parameter.
        return Independent(Normal(mean, self.std.clamp_min(1e-6).expand_as(mean)), 1)


class FACETEstimator(nn.Module):
    def __init__(self, observation_dim, wrench_dim):
        super().__init__()
        self.mlp = mlp(observation_dim, (128, 128))
        self.gru = nn.GRUCell(128, 128)
        self.norm = nn.LayerNorm(128)
        self.out = nn.Linear(128, 128+32+wrench_dim)
        self.split = (128, 32, wrench_dim)

    def forward(self, observation, hidden, is_init):
        """[B,D] or [B,T,D]; hidden is raw GRU [B,128], never normalized.

        Unlike upstream's framework-dependent single-step path, BOTH paths reset
        before consuming an initial observation. Sequence returns final hidden.
        """
        features = self.mlp(observation)
        if observation.ndim == 2:
            hidden = self.gru(features, hidden * (~is_init.bool()).reshape(-1, 1))
            recurrent = self.norm(hidden)
        elif observation.ndim == 3:
            outputs = []
            reset = is_init.bool().reshape(*observation.shape[:2], 1)
            for step in range(observation.shape[1]):
                hidden = self.gru(features[:, step], hidden * (~reset[:, step]))
                outputs.append(hidden)
            recurrent = self.norm(torch.stack(outputs, 1))
        else:
            raise ValueError("FACET estimator expects [B,D] or [B,T,D]")
        return (*self.out(features+recurrent).split(self.split, -1), hidden)


class FACETModel(nn.Module):
    """No optimizer, rollout storage, simulator, or implicit normalizer updates."""
    def __init__(self, observation_dim, privileged_dim, wrench_dim=9, action_dim=17):
        super().__init__()
        self.obs_norm = FACETVecNorm(observation_dim)
        self.priv_norm = FACETVecNorm(privileged_dim)
        self.wrench_norm = FACETVecNorm(wrench_dim)
        self.priv_encoder = nn.Sequential(mlp(privileged_dim, (128,)), nn.Linear(128, 128))
        self.force_encoder = nn.Sequential(mlp(wrench_dim, (32,)), nn.Linear(32, 32))
        self.teacher = FACETActor(observation_dim, action_dim)
        self.student = FACETActor(observation_dim, action_dim)
        self.estimator = FACETEstimator(observation_dim, wrench_dim)
        self.critic = nn.Sequential(mlp(observation_dim+privileged_dim+wrench_dim, (512, 256, 128)), nn.Linear(128, 1))

        def initialize(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, .01)
                nn.init.zeros_(module.bias)
        # Upstream deliberately leaves the student Linear defaults and GRUCell
        # initialization intact. No blanket self.apply() here.
        for module in (self.teacher, self.critic, self.priv_encoder, self.force_encoder, self.estimator):
            module.apply(initialize)

    @torch.no_grad()
    def update_normalizers(self, observation, privileged, wrench):
        if not self.training:
            raise RuntimeError("FACET normalizer updates are disabled in evaluation mode")
        for norm, value in ((self.obs_norm, observation), (self.priv_norm, privileged), (self.wrench_norm, wrench)):
            norm.update(value)

    def teacher_distribution(self, observation, privileged, wrench):
        return self.teacher(self.obs_norm(observation), self.priv_encoder(self.priv_norm(privileged)),
                            self.force_encoder(self.wrench_norm(wrench)))

    def estimate(self, observation, hidden, is_init):
        """Separate 9-D wrench reconstruction head, not the 32-D latent; no loss/calibration yet."""
        return self.estimator(self.obs_norm(observation), hidden, is_init)

    def student_distribution(self, observation, hidden, is_init):
        priv, force, reconstruction, hidden = self.estimate(observation, hidden, is_init)
        return self.student(self.obs_norm(observation), priv, force), hidden, reconstruction

    def teacher_action(self, observation, privileged, wrench, deterministic=True):
        distribution = self.teacher_distribution(observation, privileged, wrench)
        return distribution.mean if deterministic else distribution.sample()

    def student_action(self, observation, hidden, is_init, deterministic=True):
        distribution, hidden, reconstruction = self.student_distribution(observation, hidden, is_init)
        return (distribution.mean if deterministic else distribution.sample()), hidden, reconstruction

    def value(self, observation, privileged, wrench):
        # CatTensors' default sort=True: _ext_normed, _obs_normed, _priv_normed.
        return self.critic(torch.cat((self.wrench_norm(wrench), self.obs_norm(observation), self.priv_norm(privileged)), -1))
