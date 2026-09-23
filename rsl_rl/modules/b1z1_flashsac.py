"""FlashSAC numerical primitives for B1Z1 PACT.

Ported from Holiday-Robot/FlashSAC, commit 87edc9061150ae9e962dd84e6544e27a1554b3ab.
See FLASH_SAC_LICENSE for the upstream MIT license. No framework dependencies.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

class EnsembleUnitLinear(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))
        for i in range(num_ensemble):
            nn.init.orthogonal_(self.weight.data[i], gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, B, in] @ [N, in, out] -> [N, B, out]
        return torch.einsum("nbi,noi->nbo", x, self.weight)

    def normalize_parameters(self) -> None:
        self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, num_ensemble: int, input_dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_mean", torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_var", torch.ones(num_ensemble, input_dim))

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        if training:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                B = x.shape[1]
                # Cast to float32 for running stats (BatchNorm uses float32 even in AMP)
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                self.running_var.lerp_((var.squeeze(1) * (B / (B - 1))).float(), self.momentum)
            x = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(self.running_var.unsqueeze(1) + self.eps)
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    def normalize_parameters(self) -> None:
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class EnsembleUnitRMSNorm(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.unsqueeze(1)

    def normalize_parameters(self) -> None:
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class EnsembleFlashSACEmbedder(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = EnsembleUnitBatchNorm(num_ensemble, input_dim)
        self.w = EnsembleUnitLinear(num_ensemble, input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.norm(x, training=training)
        x = self.w(x)
        return x


class EnsembleFlashSACBlock(nn.Module):
    def __init__(self, num_ensemble: int, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.w1 = EnsembleUnitLinear(num_ensemble, hidden_dim, hidden_dim * expansion)
        self.w2 = EnsembleUnitLinear(num_ensemble, hidden_dim * expansion, hidden_dim)
        self.norm1 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim * expansion)
        self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = F.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = F.relu(x)
        x = x + residual
        return x


class EnsembleCategoricalValue(nn.Module):
    bin_values: torch.Tensor

    def __init__(
        self,
        num_ensemble: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
    ):
        super().__init__()
        self.w = EnsembleUnitLinear(num_ensemble, hidden_dim, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_ensemble, num_bins))
        self.register_buffer(
            "bin_values",
            torch.linspace(start=min_v, end=max_v, steps=num_bins, dtype=torch.float32).reshape(1, 1, -1),
        )

    def forward(
        self,
        x: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        value = self.w(x) + self.bias.unsqueeze(1)
        log_prob = F.log_softmax(value, dim=-1)
        value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=-1)
        info: dict[str, torch.Tensor] = {"log_prob": log_prob}
        return value, info


class FlashSACDoubleCritic(nn.Module):
    """
    Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3

    Fuses N parallel critic networks into single batched operations.
    All internal computation uses (N, batch, dim) tensor layout.
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
    ):
        super().__init__()
        self.num_qs = num_qs

        self.embedder = EnsembleFlashSACEmbedder(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList([EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat((observations, actions), dim=-1)  # [B, in_dim]
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)  # [num_qs, B, in_dim]
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        qs, infos = self.predictor(x, training)
        return qs, infos


class FlashSACTemperature(nn.Module):
    def __init__(self, initial_value: float = 0.01):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([math.log(initial_value)], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temp)


def _select_min_q_log_probs(
    next_qs: torch.Tensor,  # (2, B)
    next_q_log_probs: torch.Tensor,  # (2, B, num_bins)
) -> torch.Tensor:
    """Select log-probs from the min-Q critic and return the next-obs half (batch, num_bins)."""
    num_bins = next_q_log_probs.shape[-1]
    min_indices = next_qs.argmin(dim=0)  # (B,)
    selected = torch.gather(
        next_q_log_probs,
        dim=0,
        index=min_indices[None, :, None].expand(1, -1, num_bins),
    )[
        0
    ]  # (B, num_bins)
    return selected


def _compute_categorical_td_target(
    target_log_probs: torch.Tensor,  # (B, num_bins)
    reward: torch.Tensor,  # (B,)
    done: torch.Tensor,  # (B,)
    actor_entropy: torch.Tensor,  # (B,)
    gamma: float,
    num_bins: int,
    min_v: float,
    max_v: float,
) -> torch.Tensor:
    batch_size = reward.shape[0]

    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)

    # Compute target value buckets
    bin_width = (max_v - min_v) / (num_bins - 1)
    bin_values = torch.linspace(
        min_v, max_v, num_bins, device=target_log_probs.device, dtype=target_log_probs.dtype
    ).view(1, -1)

    # target_bin_values
    target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

    # update indices
    b = (target_bin_values - min_v) / bin_width
    lower = torch.floor(b).long()
    upper = torch.clamp(lower + 1, 0, num_bins - 1)

    frac = b - lower.float()

    # Compute target probabilities using exp
    target_probs_exp = target_log_probs.exp()
    m_l = target_probs_exp * (1.0 - frac)
    m_u = target_probs_exp * frac

    # Allocate output tensor
    target_probs = torch.zeros(batch_size, num_bins, dtype=target_probs_exp.dtype, device=target_probs_exp.device)

    # Scatter operations
    target_probs.scatter_add_(1, lower, m_l)
    target_probs.scatter_add_(1, upper, m_u)

    return target_probs



def _update_reward_stats(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    G_r: torch.Tensor,
    G_r_max: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    done = torch.logical_or(terminated, truncated).float()
    new_G_r = gamma * (1.0 - done) * G_r + reward
    new_G_r_max = torch.maximum(G_r_max, torch.max(torch.abs(new_G_r)))
    return new_G_r, new_G_r_max


def _scale_reward(
    rewards: torch.Tensor,
    G_var: torch.Tensor,
    G_r_max: torch.Tensor,
    G_max: float,
    eps: float,
) -> torch.Tensor:
    var_denominator = torch.sqrt(G_var + eps)
    min_required_denominator = G_r_max / G_max
    denominator = torch.maximum(var_denominator, min_required_denominator)
    return rewards / denominator


def _update_mean_var_count_from_moments(
    samples: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    running_count: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    sample_mean = torch.mean(samples, dim=0)
    sample_var = torch.var(samples, dim=0, unbiased=False)
    sample_count = float(samples.shape[0])

    delta = sample_mean - running_mean
    total_count = running_count + sample_count
    ratio = sample_count / total_count

    new_mean = running_mean + delta * ratio
    m_a = running_var * (running_count + epsilon)
    m_b = sample_var * sample_count
    M2 = m_a + m_b + torch.square(delta) * running_count * ratio
    new_var = M2 / total_count

    return (
        new_mean,
        new_var,
        total_count,
    )



class RewardNormalizer(nn.Module):
    def __init__(self, gamma=.95, maximum=5.):
        super().__init__()
        self.gamma, self.maximum = gamma, maximum
        for name, value in (("returns", 0.), ("maximum_return", 0.), ("mean", 0.), ("variance", 1.), ("count", 0.)):
            self.register_buffer(name, torch.tensor(value))

    @torch.no_grad()
    def update(self, reward, terminated, truncated):
        self.returns, self.maximum_return = _update_reward_stats(
            reward.flatten(), terminated.flatten(), truncated.flatten(),
            self.returns, self.maximum_return, self.gamma)
        self.mean, self.variance, self.count = _update_mean_var_count_from_moments(
            self.returns, self.mean, self.variance, self.count, 1e-4)

    def forward(self, rewards):
        return _scale_reward(rewards, self.variance, self.maximum_return, self.maximum, 1e-8)

    def load_state_dict(self, state_dict, strict=True):
        self.returns = torch.empty_like(state_dict["returns"], device=self.mean.device)
        return super().load_state_dict(state_dict, strict=strict)


@torch.no_grad()
def normalize_weights(network):
    for module in network.modules():
        if hasattr(module, "normalize_parameters"):
            module.normalize_parameters()


class RepeatedNoise:
    """Independent per-environment truncated-zeta repeat counters."""
    def __init__(self, mu=2., maximum=16):
        probabilities = torch.arange(1, maximum + 1).float().pow(-mu)
        self.cdf = probabilities.cumsum(0) / probabilities.sum()
        self.noise = self.remaining = None

    def sample(self, mean):
        if self.noise is None or self.noise.shape != mean.shape:
            self.noise = torch.zeros_like(mean)
            self.remaining = torch.zeros(len(mean), device=mean.device, dtype=torch.long)
        refresh = self.remaining <= 0
        noise = torch.randn_like(mean)
        repeat = torch.searchsorted(self.cdf.to(mean.device), torch.rand(len(mean), device=mean.device)) + 1
        self.noise = torch.where(refresh[:, None], noise, self.noise)
        self.remaining = torch.where(refresh, repeat, self.remaining) - 1
        return self.noise

    def reset(self, dones):
        if self.remaining is not None:
            self.remaining[dones.flatten().bool()] = 0
