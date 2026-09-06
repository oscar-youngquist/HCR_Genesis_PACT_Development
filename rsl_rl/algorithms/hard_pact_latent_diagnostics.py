"""Side-effect-free, GPU-resident HardPACT latent diagnostics."""

from __future__ import annotations

import torch


def deterministic_subsample(tensor, count):
    """Return bounded, evenly spaced row indices without consuming RNG."""
    size = tensor.shape[0]
    count = min(int(count), size)
    if count == size:
        return torch.arange(size, device=tensor.device)
    return torch.div(
        torch.arange(count, device=tensor.device) * size,
        count,
        rounding_mode="floor",
    )


def deterministic_permutation(size, device):
    """A fixed cyclic permutation with no generator or host synchronization."""
    return (torch.arange(size, device=device) + max(size // 2, 1)) % size


def policy_distribution_without_side_effects(
    actor, observation, history, latent_noise, use_boot=True,
):
    """Recompute a diagonal-Gaussian policy without touching actor caches."""
    mean, logvar, features = actor.context_encoder.encode_with_features(history)
    latent = actor.context_encoder.reparameterization_trick(
        mean, logvar, latent_noise
    )
    estimator = actor.explicit_estimator(features)
    explicit = estimator.explicit_for_policy
    if use_boot:
        conditioning = torch.cat((observation, latent, explicit), dim=-1)
    else:
        conditioning = torch.cat((
            observation, torch.zeros_like(torch.cat((latent, explicit), dim=-1)),
        ), dim=-1)
    position, torque = actor.actor_forward(conditioning)
    action_mean = (
        torch.cat((position, torque), dim=-1)
        if actor.std.numel() == position.shape[-1] + torque.shape[-1]
        else position
    )
    sigma = actor.std.unsqueeze(0).expand_as(action_mean)
    return action_mean, sigma, mean, logvar, features, explicit


def diagonal_gaussian_kl(mean_p, sigma_p, mean_q, sigma_q):
    """KL(N_p || N_q), summed over action dimensions per sample."""
    sigma_p = sigma_p.clamp_min(1.0e-8)
    sigma_q = sigma_q.clamp_min(1.0e-8)
    return (
        torch.log(sigma_q / sigma_p)
        + (sigma_p.square() + (mean_p - mean_q).square())
        / (2.0 * sigma_q.square())
        - 0.5
    ).sum(dim=-1)


def diagonal_gaussian_log_prob(actions, mean, sigma):
    """Log probability of stored actions without constructing actor state."""
    sigma = sigma.clamp_min(1.0e-8)
    return (-0.5 * (
        ((actions - mean) / sigma).square()
        + 2.0 * torch.log(sigma)
        + torch.log(actions.new_tensor(2.0 * torch.pi))
    )).sum(dim=-1)


class LatentDiagnosticsAccumulator:
    """Accumulate sample-weighted scalar and per-latent-dimension metrics."""

    def __init__(self, variance_threshold):
        self.variance_threshold = float(variance_threshold)
        self.sums = {}
        self.counts = {}

    @torch.no_grad()
    def add(self, name, values):
        values = values.detach()
        self.sums[name] = self.sums.get(name, torch.zeros_like(values.sum(dim=0))) + values.sum(dim=0)
        count = values.shape[0]
        self.counts[name] = self.counts.get(name, 0) + count

    @torch.no_grad()
    def add_ppo(self, log_ratio, clip_param):
        log_ratio = log_ratio.detach().reshape(-1)
        ratio = log_ratio.exp()
        self.add("ppo/approx_kl", ((ratio - 1.0) - log_ratio).unsqueeze(-1))
        self.add("ppo/clip_fraction", ((ratio - 1.0).abs() > clip_param).float().unsqueeze(-1))
        self.add("ppo/ratio", ratio.unsqueeze(-1))
        self.add("ppo/ratio_square", ratio.square().unsqueeze(-1))

    @torch.no_grad()
    def add_latent(self, mean, logvar):
        kl = -0.5 * (1.0 + logvar - mean.square() - logvar.exp())
        std = torch.exp(0.5 * logvar)
        self.add("latent/kl_per_dim", kl)
        self.add("latent/mu", mean)
        self.add("latent/mu_square", mean.square())
        self.add("latent/posterior_std_per_dim", std)

    @torch.no_grad()
    def finalize(self):
        result = {
            name: total / float(self.counts[name])
            for name, total in self.sums.items()
        }
        if "ppo/ratio" in result:
            variance = (result.pop("ppo/ratio_square") - result["ppo/ratio"].square()).clamp_min(0.0)
            result["ppo/ratio_mean"] = result.pop("ppo/ratio").squeeze()
            result["ppo/ratio_std"] = variance.sqrt().squeeze()
            result["ppo/approx_kl"] = result["ppo/approx_kl"].squeeze()
            result["ppo/clip_fraction"] = result["ppo/clip_fraction"].squeeze()
        if "latent/mu" in result:
            mu_variance = (result.pop("latent/mu_square") - result.pop("latent/mu").square()).clamp_min(0.0)
            result["latent/mu_variance_per_dim"] = mu_variance
            kl = result["latent/kl_per_dim"]
            std = result["latent/posterior_std_per_dim"]
            result["latent/kl_total"] = kl.sum()
            result["latent/posterior_std_mean"] = std.mean()
            result["latent/posterior_std_min"] = std.min()
            result["latent/posterior_std_max"] = std.max()
            active = (mu_variance > self.variance_threshold).to(std.dtype)
            result["latent/active_unit_count"] = active.sum()
            result["latent/active_unit_fraction"] = active.mean()
        return result


@torch.no_grad()
def latent_ablation_metrics(actor, decoder, observation, history, target,
                            nominal_torque=None):
    """Compare deterministic mean conditioning to zero/permuted baselines."""
    mean, _, features = actor.context_encoder.encode_with_features(history)
    explicit = actor.explicit_estimator(features).explicit_for_policy
    permutation = deterministic_permutation(mean.shape[0], mean.device)
    variants = {"zero": torch.zeros_like(mean), "permuted": mean[permutation]}

    def outputs(latent):
        policy_input = torch.cat((observation, latent, explicit), dim=-1)
        position, torque = actor.actor_forward(policy_input)
        policy = torch.cat((position, torque), dim=-1) if actor.std.numel() > position.shape[-1] else position
        reconstruction = decoder(torch.cat((latent, explicit), dim=-1))
        result = {"policy": policy, "reconstruction": reconstruction, "explicit": explicit}
        if hasattr(actor, "physics_estimator") and nominal_torque is not None:
            heads = actor.physics_heads(latent, explicit, nominal_torque)
            result["grf"] = heads.grf_normalized
            result["wrench"] = heads.wrench_raw_normalized
        return result

    reference = outputs(mean)
    metrics = {}
    reference_error = (reference["reconstruction"] - target).square().mean(dim=-1).sqrt()
    for variant_name, latent in variants.items():
        variant = outputs(latent)
        for output_name in reference:
            metrics[f"latent/ablation/{variant_name}/{output_name}_rms_change"] = (
                variant[output_name] - reference[output_name]
            ).square().mean().sqrt()
        variant_error = (variant["reconstruction"] - target).square().mean(dim=-1).sqrt()
        metrics[f"latent/ablation/{variant_name}/reconstruction_error_rms_change"] = (
            variant_error - reference_error
        ).square().mean().sqrt()
    return metrics
