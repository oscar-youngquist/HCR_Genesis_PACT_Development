"""Shared VAE KL warmup and primal-dual rate-band controller."""

import math

import torch


def update_duals_from_mean(
    controller, raw_kl_sum, raw_kl_count, iteration, device, enabled=True,
    use_cosine_warmup=True,
):
    """Apply one dual update to the detached mean KL from a PPO iteration."""
    if raw_kl_count == 0:
        return None
    mean_raw_kl = torch.tensor(raw_kl_sum / raw_kl_count, device=device)
    if enabled:
        if use_cosine_warmup:
            controller.update_duals(mean_raw_kl, iteration)
        else:
            controller.update_duals(
                mean_raw_kl, iteration, use_cosine_warmup=False
            )
    return mean_raw_kl


class KLRateBandController:
    """Build a differentiable KL penalty while keeping dual state detached."""

    METRIC_NAMES = (
        "kl_raw", "kl_ema", "kl_reg_loss", "kl_warmup_beta",
        "kl_band_warmup_scale",
        "kl_low_violation", "kl_high_violation", "kl_lambda_low",
        "kl_lambda_high", "kl_dual_effective_beta", "kl_effective_coef",
        "kl_base_beta", "kl_band_active",
        "kl_rate_band_enabled", "kl_warmup_enabled",
    )

    def __init__(
        self, *, warmup_iters, warmup_beta_max, rate_min, rate_max,
        dual_lr, augmented_rho, ema_decay, band_warmup_iters=0,
    ):
        self.warmup_iters = int(warmup_iters)
        self.band_warmup_iters = int(band_warmup_iters)
        self.warmup_beta_max = float(warmup_beta_max)
        self.rate_min = float(rate_min)
        self.rate_max = float(rate_max)
        self.dual_lr = float(dual_lr)
        self.augmented_rho = float(augmented_rho)
        self.ema_decay = float(ema_decay)
        if self.warmup_iters < 0 or self.band_warmup_iters < 0 or self.warmup_beta_max < 0.0:
            raise ValueError("KL warmup iterations and beta must be nonnegative")
        if not 0.0 <= self.rate_min <= self.rate_max:
            raise ValueError("KL rate bounds must satisfy 0 <= min <= max")
        if self.dual_lr < 0.0 or self.augmented_rho < 0.0:
            raise ValueError("KL dual learning rate and augmented rho must be nonnegative")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("kl_ema_decay must lie in [0, 1)")

        # Python scalars cannot enter optimizer parameter groups or acquire gradients.
        self.lambda_low = 0.0
        self.lambda_high = 0.0
        self.kl_ema = None

    def band_active(self, iteration, use_cosine_warmup=True):
        return not use_cosine_warmup or int(iteration) >= self.warmup_iters

    def warmup_beta(self, iteration, use_cosine_warmup=True):
        if not use_cosine_warmup or self.warmup_iters == 0:
            return self.warmup_beta_max
        progress = min(max(float(iteration) / self.warmup_iters, 0.0), 1.0)
        return 0.5 * self.warmup_beta_max * (1.0 - math.cos(math.pi * progress))

    def band_warmup_scale(self, iteration, use_cosine_warmup=True):
        """Cosine-ramp rate-band pressure after the base KL warmup."""
        if not self.band_active(iteration, use_cosine_warmup):
            return 0.0
        if self.band_warmup_iters == 0:
            return 1.0
        band_start = self.warmup_iters if use_cosine_warmup else 0
        progress = min(
            max(float(iteration - band_start) / self.band_warmup_iters, 0.0),
            1.0,
        )
        return 0.5 * (1.0 - math.cos(math.pi * progress))

    def loss(
        self, raw_kl, iteration, use_rate_band=True,
        use_cosine_warmup=True,
    ):
        """Return rate-band KL or fixed-weight standard KL regularization."""
        if not use_rate_band:
            return self.warmup_beta(iteration, use_cosine_warmup) * raw_kl
        if not self.band_active(iteration, use_cosine_warmup):
            return self.warmup_beta(iteration, use_cosine_warmup) * raw_kl
        g_low = self.rate_min - raw_kl
        g_high = raw_kl - self.rate_max
        band_scale = self.band_warmup_scale(iteration, use_cosine_warmup)
        band_penalty = (
            self.lambda_low * g_low
            + self.lambda_high * g_high
            + 0.5 * self.augmented_rho * torch.relu(g_low).square()
            + 0.5 * self.augmented_rho * torch.relu(g_high).square()
        )
        return self.warmup_beta_max * raw_kl + band_scale * band_penalty

    def update_duals(self, raw_kl, iteration, use_cosine_warmup=True):
        """Update detached EMA/duals after the auxiliary optimizer step."""
        with torch.no_grad():
            rate = float(raw_kl.detach().item())
            if math.isfinite(rate):
                self.kl_ema = rate if self.kl_ema is None else (
                    self.ema_decay * self.kl_ema + (1.0 - self.ema_decay) * rate
                )
                if self.band_active(iteration, use_cosine_warmup):
                    dual_step = self.dual_lr * self.band_warmup_scale(
                        iteration, use_cosine_warmup
                    )
                    self.lambda_low = max(
                        0.0, self.lambda_low + dual_step * (self.rate_min - self.kl_ema)
                    )
                    self.lambda_high = max(
                        0.0, self.lambda_high + dual_step * (self.kl_ema - self.rate_max)
                    )

    def metrics(
        self, raw_kl, reg_loss, iteration, use_rate_band=True,
        use_cosine_warmup=True,
    ):
        """Return detached tensors suitable for existing metric reducers."""
        rate = float(raw_kl.detach().item())
        base_beta = self.warmup_beta(iteration, use_cosine_warmup)
        if not use_rate_band:
            values = {
                "kl_raw": rate,
                "kl_ema": rate,
                "kl_reg_loss": float(reg_loss.detach().item()),
                "kl_warmup_beta": base_beta,
                "kl_band_warmup_scale": 0.0,
                "kl_low_violation": 0.0,
                "kl_high_violation": 0.0,
                "kl_lambda_low": 0.0,
                "kl_lambda_high": 0.0,
                "kl_dual_effective_beta": 0.0,
                "kl_effective_coef": base_beta,
                "kl_base_beta": self.warmup_beta_max,
                "kl_band_active": 0.0,
                "kl_rate_band_enabled": 0.0,
                "kl_warmup_enabled": float(use_cosine_warmup),
            }
            return {name: raw_kl.new_tensor(value) for name, value in values.items()}
        low_violation = max(self.rate_min - rate, 0.0)
        high_violation = max(rate - self.rate_max, 0.0)
        active = self.band_active(iteration, use_cosine_warmup)
        band_scale = self.band_warmup_scale(iteration, use_cosine_warmup)
        effective_coef = (
            self.warmup_beta_max + band_scale * (
                self.lambda_high - self.lambda_low
                + self.augmented_rho * (high_violation - low_violation)
            )
            if active else base_beta
        )
        values = {
            "kl_raw": rate,
            "kl_ema": rate if self.kl_ema is None else self.kl_ema,
            "kl_reg_loss": float(reg_loss.detach().item()),
            "kl_warmup_beta": base_beta,
            "kl_band_warmup_scale": band_scale,
            "kl_low_violation": low_violation,
            "kl_high_violation": high_violation,
            "kl_lambda_low": self.lambda_low,
            "kl_lambda_high": self.lambda_high,
            "kl_dual_effective_beta": band_scale * (self.lambda_high - self.lambda_low),
            "kl_effective_coef": effective_coef,
            "kl_base_beta": self.warmup_beta_max,
            "kl_band_active": float(active),
            "kl_rate_band_enabled": 1.0,
            "kl_warmup_enabled": float(use_cosine_warmup),
        }
        return {name: raw_kl.new_tensor(value) for name, value in values.items()}

    def state_dict(self):
        return {
            "lambda_low": self.lambda_low,
            "lambda_high": self.lambda_high,
            "kl_ema": self.kl_ema,
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.lambda_low = max(0.0, float(state.get("lambda_low", 0.0)))
        self.lambda_high = max(0.0, float(state.get("lambda_high", 0.0)))
        ema = state.get("kl_ema")
        self.kl_ema = None if ema is None else float(ema)
