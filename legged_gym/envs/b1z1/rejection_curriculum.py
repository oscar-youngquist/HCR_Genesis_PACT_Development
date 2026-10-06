"""Performance-gated rejection schedule; privileged labels never generate commands."""
import math
import warnings

import torch

from legged_gym.envs.b1z1.force_task_utils import B1Z1StagedForceCurriculum


class RejectionCurriculumDefaults:
    """One set of rejection-stage defaults for UniFP-Reject and PACT."""
    use_shared_rejection_curriculum = True
    reject_initial_external_scale = 0.25
    reject_warmup_iterations = 1600  # Legacy field; force_curriculum_gate_start_iteration controls gating.
    reject_compensation_ramp_iterations = 400
    reject_external_ramp_iterations = 3200
    force_curriculum_gate_start_iteration = 1600
    force_curriculum_gate_patience = 400
    force_curriculum_use_latest_start_fallback = True
    force_curriculum_latest_start_iteration = 6400
    reject_active_force_threshold = 1.0  # Diagnostic active-force cutoff [N]; not an advancement gate.
    
    reject_force_nrmse_threshold = 0.25  # Legacy field; force accuracy no longer gates advancement.
    reject_min_active_samples = 32
    reject_require_force_quality = False  # Legacy compatibility only; all tasks use the performance gate.

    # # VRAM SMOKE TEST: assignments below override normal defaults above.
    # # Comment only these overrides to restore normal timing; keep diagnostics configured.
    # # Also enable the PACT policy/algorithm/runner smoke blocks for a fresh 35-iteration run.
    # # Hold 0.25 for 10 iterations; ramp rejection for 10, then disturbances for 10.
    # # Performance fallback forces the start at 10; force-prediction accuracy is not required.
    # force_curriculum_gate_start_iteration = 10
    # force_curriculum_gate_patience = 1
    # force_curriculum_use_latest_start_fallback = True
    # force_curriculum_latest_start_iteration = 10
    # reject_compensation_ramp_iterations = 10
    # reject_external_ramp_iterations = 10


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
        """Use current-time active-force labels only for detached diagnostic statistics."""
        for i, scale in enumerate(scales):
            p = prediction[:, 6 + 3*i:9 + 3*i].detach() / scale
            t = target[:, 6 + 3*i:9 + 3*i].detach() / scale
            active = t.square().sum(-1) > self.cfg.reject_active_force_threshold**2
            error = (p - t).square().sum(-1)
            # Preserve nonfinite active errors so update() rejects the diagnostic sample.
            self.samples[i] += torch.stack((
                torch.where(active, error, 0.).sum(),
                torch.where(active, t.square().sum(-1), 0.).sum(), active.sum(),
            ))

    @torch.no_grad()
    def observe_forces(self, predicted_ee, predicted_base, target_ee, target_base):
        """Physical forces in matching frames; no observation-layout assumptions."""
        for i, (p, t) in enumerate(((predicted_ee, target_ee), (predicted_base, target_base))):
            p, t = p.detach(), t.detach()
            active = t.square().sum(-1) > self.cfg.reject_active_force_threshold**2
            self.samples[i] += torch.stack((
                torch.where(active, (p-t).square().sum(-1), 0.).sum(),
                torch.where(active, t.square().sum(-1), 0.).sum(), active.sum()))

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
        # Preserve the original performance gate, patience, and latest-start fallback.
        super().update(iteration, ee_l1, roll_termination_rate, mean_episode_length)
        samples = self.samples.tolist()  # One device synchronization per PPO iteration.
        self.samples.zero_()
        for i, (error, magnitude, count) in enumerate(samples):
            if not self.enabled[i]:
                continue
            valid = count >= self.cfg.reject_min_active_samples and math.isfinite(error)
            if valid:
                value = math.sqrt(error / max(magnitude, 1e-12))
                self.force_error_ema[i] = self._update_ema(self.force_error_ema[i], value)
        # Force-error EMAs are diagnostics only; the shared performance gate owns advancement.
        if self.gate_latched:
            self.beta = self._linear_ramp(iteration, self.trigger_iteration,
                                          self.cfg.reject_compensation_ramp_iterations)
            if self.beta >= 1 and self.full_iteration < 0:
                self.full_iteration = iteration

    def metrics(self, iteration):
        return {
            "Rejection/beta": self.beta,
            "Rejection/progress": self.beta,
            "ForceCurriculum/command_scale": self.beta,
            "ForceCurriculum/external_scale": self.external_scale(iteration),
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
        # Resume an older rejection checkpoint without restarting its progress.
        if self.beta > 0 and not self.gate_latched:
            self.gate_latched = True
            self.trigger_iteration = self.last_update_iteration - round(
                self.beta * self.cfg.reject_compensation_ramp_iterations)
