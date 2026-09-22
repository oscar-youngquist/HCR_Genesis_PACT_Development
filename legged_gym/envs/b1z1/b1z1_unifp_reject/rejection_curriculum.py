"""Prediction-gated rejection schedule; privileged labels never generate commands."""
import math
import warnings

import torch

from legged_gym.envs.b1z1.force_task_utils import B1Z1StagedForceCurriculum


class RejectionCurriculum(B1Z1StagedForceCurriculum):
    def __init__(self, cfg, device):
        super().__init__(cfg)
        self.cfg = cfg
        self.beta = 0.0
        self.full_iteration = -1
        self.force_error_ema = [None, None]
        # Per stream: squared error, squared target magnitude, active sample count.
        self.samples = torch.zeros(2, 3, device=device)
        self.enabled = (cfg.apply_ee_external_forces, cfg.apply_base_external_forces)
        if not 0 < cfg.reject_initial_external_scale <= 1:
            raise ValueError("reject_initial_external_scale must be in (0, 1]")
        if min(cfg.reject_warmup_iterations, cfg.reject_external_ramp_iterations) < 0:
            raise ValueError("Rejection durations must be nonnegative")
        if cfg.reject_compensation_ramp_iterations < 1 or cfg.reject_min_active_samples < 1:
            raise ValueError("Rejection ramp and minimum sample count must be positive")
        if cfg.reject_active_force_threshold <= 0 or cfg.reject_force_nrmse_threshold <= 0:
            raise ValueError("Rejection force thresholds must be positive")

    @torch.no_grad()
    def observe(self, prediction, target, scales):
        """Use current-time active-force labels only for detached gate statistics."""
        for i, scale in enumerate(scales):
            p = prediction[:, 6 + 3*i:9 + 3*i].detach() / scale
            t = target[:, 6 + 3*i:9 + 3*i].detach() / scale
            active = t.square().sum(-1) > self.cfg.reject_active_force_threshold**2
            error = (p - t).square().sum(-1)
            # Nonfinite active predictions fail the gate, rather than disappearing.
            self.samples[i] += torch.stack((
                torch.where(active, error, 0.).sum(),
                torch.where(active, t.square().sum(-1), 0.).sum(), active.sum(),
            ))

    def command_scale(self, iteration):
        return self.beta

    def external_scale(self, iteration):
        initial = self.cfg.reject_initial_external_scale
        progress = 0.0 if self.full_iteration < 0 else self._linear_ramp(
            iteration, self.full_iteration, self.cfg.reject_external_ramp_iterations)
        return initial + (1.0 - initial) * progress

    def update(self, iteration, ee_l1=None, roll_termination_rate=None, mean_episode_length=None):
        if iteration == self.last_update_iteration:
            return
        if iteration < self.last_update_iteration:
            raise ValueError("Rejection iterations must be monotonically increasing")
        self.last_update_iteration = iteration
        samples = self.samples.tolist()  # One device synchronization per PPO iteration.
        self.samples.zero_()
        force_ok = True
        for i, (error, magnitude, count) in enumerate(samples):
            if not self.enabled[i]:
                continue
            valid = count >= self.cfg.reject_min_active_samples and math.isfinite(error)
            if valid:
                value = math.sqrt(error / max(magnitude, 1e-12))
                self.force_error_ema[i] = self._update_ema(self.force_error_ema[i], value)
            force_ok &= valid and self.force_error_ema[i] is not None and (
                self.force_error_ema[i] <= self.cfg.reject_force_nrmse_threshold)
        values = (ee_l1, roll_termination_rate, mean_episode_length)
        complete = all(v is not None and math.isfinite(float(v)) for v in values)
        self.ee_l1_ema = self._update_ema(self.ee_l1_ema, ee_l1)
        self.roll_termination_ema = self._update_ema(self.roll_termination_ema, roll_termination_rate)
        self.episode_length_ema = self._update_ema(self.episode_length_ema, mean_episode_length)
        stable = complete and self.ee_l1_ema < self.ee_l1_threshold and (
            self.roll_termination_ema < self.roll_threshold and
            self.episode_length_ema > self.episode_length_threshold)
        passed = force_ok and stable and iteration >= self.cfg.reject_warmup_iterations
        self.gate_patience = self.gate_patience + 1 if passed else 0
        # No timeout/oracle fallback. Pause compensation growth if quality deteriorates.
        if self.gate_patience >= self.required_patience and self.beta < 1:
            self.beta = min(1., self.beta + 1. / self.cfg.reject_compensation_ramp_iterations)
            if self.beta >= 1. - 1e-12:
                self.beta = 1.
                self.full_iteration = iteration + 1

    def metrics(self, iteration):
        return {
            "Rejection/beta": self.beta,
            "Rejection/external_scale": self.external_scale(iteration),
            "Rejection/stage": 1 if self.beta == 0 else (2 if self.beta < 1 else 3),
            "Rejection/gate_patience": self.gate_patience,
            **{f"Rejection/{name}_active_nrmse_ema": value
               for name, value in zip(("ee", "base"), self.force_error_ema) if value is not None},
        }

    def state_dict(self):
        return dict(super().state_dict(), rejection_version=1, beta=self.beta,
                    full_iteration=self.full_iteration, force_error_ema=self.force_error_ema)

    def load_state_dict(self, state):
        if not state or state.get("rejection_version") != 1:
            warnings.warn("No rejection curriculum state; starting estimator warmup.")
            return
        super().load_state_dict(state)
        self.beta = float(state["beta"])
        self.full_iteration = int(state["full_iteration"])
        self.force_error_ema = list(state["force_error_ema"])
