"""Independent QP schedules; no domain-randomization state or RNG is used."""
from collections import deque
from dataclasses import replace
import math

import torch


def execution_torque(base, candidate, accepted, alpha):
    """Blend accepted total commands only; rejected candidates are fallbacks.

    A blended command is NOT certified by the candidate's certificate. Endpoint
    branches avoid roundoff and preserve recovery's intentionally softened rate.
    """
    if alpha == 1.0:
        return candidate
    blended = base if alpha == 0.0 else base + alpha * (candidate - base)
    return torch.where(accepted[:, None], blended, candidate)


class QPCurriculum:
    def __init__(self, cfg):
        self.cfg = cfg
        self.origin = int(cfg.warmup_iterations) + int(cfg.correction_ramp_start_offset)
        self.start = (max(cfg.warmup_iterations, self.origin + cfg.correction_ramp_duration)
                      if cfg.objective_curriculum_start is None and cfg.correction_ramp_enabled
                      else cfg.warmup_iterations if cfg.objective_curriculum_start is None
                      else int(cfg.objective_curriculum_start))
        self.progress = 0.0
        self.ema = None
        self.threshold = None
        self.history = deque(maxlen=cfg.objective_curriculum_window)
        self.last_iteration = -1
        self.last_step = -10**9
        self.snapshot_iteration = None
        self.snapshot = None
        self.advanced = False
        self.activation_iteration = int(cfg.warmup_iterations)
        self.pre_qp_reference = None
        self.baseline_frozen = False
        self.recovery_counter = 0
        self.block_reason = 0
        if (cfg.correction_ramp_duration < 0 or cfg.objective_curriculum_window < 1
                or cfg.objective_curriculum_step_interval < 1
                or not 0 < cfg.objective_curriculum_ema_alpha <= 1
                or not 0 <= cfg.objective_curriculum_quantile <= 1
                or not 0 < cfg.objective_curriculum_progress_delta <= 1
                or cfg.objective_curriculum_min_samples < 1
                or cfg.objective_curriculum_recovery_iterations < 1):
            raise ValueError("Invalid QP curriculum schedule")
        override = cfg.objective_curriculum_baseline_override
        if override is not None and (not math.isfinite(override) or override < 0):
            raise ValueError('QP baseline override must be finite and nonnegative')

    def _freeze_baseline(self):
        if self.baseline_frozen:
            return
        self.baseline_frozen = True
        self.pre_qp_reference = self.cfg.objective_curriculum_baseline_override
        if self.pre_qp_reference is None and self.history:
            ordered = sorted(self.history)
            self.pre_qp_reference = ordered[int(self.cfg.objective_curriculum_quantile*(len(ordered)-1))]
        self._set_threshold()

    def _set_threshold(self):
        self.threshold = (None if self.pre_qp_reference is None else
            max(self.cfg.objective_curriculum_min_tracking,
                self.cfg.objective_curriculum_recovery_ratio*self.pre_qp_reference))

    def begin(self, iteration):
        """Freeze weights/alpha for the complete rollout plus all PPO epochs."""
        if iteration == self.snapshot_iteration:
            return self.snapshot
        c = self.cfg
        if iteration >= self.activation_iteration:
            self._freeze_baseline()  # BEFORE the first QP, even with alpha=0
        alpha = (1.0 if not c.correction_ramp_enabled or c.correction_ramp_duration == 0
                 else min(1.0, max(0.0, (iteration-self.origin)/c.correction_ramp_duration)))
        weights = {}
        for name in ("contact_acceleration_weight", "attitude_weight", "height_weight"):
            # Component weights are authoritative endpoints. Deprecated *_final
            # fields remain loadable but cannot override the configured objective.
            final = getattr(c, name)
            initial = getattr(c, name + "_initial")
            initial = .25 * final if initial is None else initial
            if not all(math.isfinite(v) and v >= 0 for v in (initial, final)):
                raise ValueError("QP curriculum weights must be finite and nonnegative")
            scheduled=c.objective_curriculum_enabled and (name=='contact_acceleration_weight' or c.torso_stability_curriculum_enabled)
            weights[name] = (initial + self.progress*(final-initial) if scheduled else final)
        self.snapshot_iteration = int(iteration)
        self.snapshot_progress = self.progress
        self.snapshot = (replace(c, **weights), alpha)
        return self.snapshot

    def finish(self, iteration, performance, count):
        """Evidence collected before reward scaling; advance only the next snapshot."""
        if iteration <= self.last_iteration:
            return
        self.advanced = False
        if iteration != self.last_iteration+1:
            self.recovery_counter = 0
        self.last_iteration = int(iteration)
        c = self.cfg
        if (not c.objective_curriculum_enabled or count < c.objective_curriculum_min_samples
                or performance is None or not math.isfinite(performance)):
            self.recovery_counter = 0
            self.block_reason = 1  # disabled/invalid evidence
            return
        self.ema = (performance if self.ema is None else
                    (1-c.objective_curriculum_ema_alpha)*self.ema+c.objective_curriculum_ema_alpha*performance)
        if iteration < self.activation_iteration and not self.baseline_frozen:
            self.history.append(self.ema)
            self.block_reason = 2  # collecting pre-QP baseline
            return
        self._freeze_baseline()
        if self.pre_qp_reference is None:
            self.recovery_counter = 0
            self.block_reason = 3  # baseline unavailable: explicit override required
            return
        if iteration < self.start:
            self.recovery_counter = 0
            self.block_reason = 4  # earliest start
            return
        if self.ema <= self.threshold:
            self.recovery_counter = 0
            self.block_reason = 5  # performance below frozen threshold
            return
        self.recovery_counter += 1
        self.block_reason = 6  # sustained recovery or interval pending
        if (self.recovery_counter >= c.objective_curriculum_recovery_iterations
                and iteration-self.last_step >= c.objective_curriculum_step_interval and self.progress < 1):
            self.progress = min(1., self.progress+c.objective_curriculum_progress_delta)
            self.last_step = int(iteration)
            self.advanced = True
            self.recovery_counter = 0
            self.block_reason = 0
        elif self.progress >= 1:
            self.recovery_counter = 0
            self.block_reason = 7  # complete

    def metrics(self):
        cfg, alpha = self.snapshot
        return dict(alpha=alpha, weight_progress=self.snapshot_progress,
                    next_weight_progress=self.progress,
                    contact_acceleration_weight=cfg.contact_acceleration_weight,
                    attitude_weight=cfg.attitude_weight,
                    height_weight=cfg.height_weight,
                    **{'torso_stability/attitude_weight':cfg.attitude_weight,
                       'torso_stability/height_weight':cfg.height_weight,
                       'torso_stability/progress':self.snapshot_progress},
                    tracking_ema=float('nan') if self.ema is None else self.ema,
                    tracking_threshold=float('nan') if self.threshold is None else self.threshold,
                    frozen_pre_qp_reference=float('nan') if self.pre_qp_reference is None else self.pre_qp_reference,
                    baseline_frozen=int(self.baseline_frozen),
                    baseline_samples=len(self.history),recovery_counter=self.recovery_counter,
                    block_reason=self.block_reason,
                    advanced=int(self.advanced))

    def state_dict(self):
        return {"version": 2, **{k: getattr(self, k) for k in
                ("origin", "start", "progress", "ema", "threshold", "last_iteration", "last_step",
                 "activation_iteration", "pre_qp_reference", "baseline_frozen", "recovery_counter", "block_reason")},
                "history": list(self.history)}

    def load_state_dict(self, state):
        if state["version"] not in (1,2):
            raise ValueError("Unsupported QP curriculum checkpoint version")
        for key in ("origin", "start", "progress", "ema", "threshold", "last_iteration", "last_step"):
            setattr(self, key, state[key])
        self.history.clear()
        self.history.extend(state["history"])
        if state['version'] == 2:
            for key in ('activation_iteration','pre_qp_reference','baseline_frozen','recovery_counter','block_reason'):
                setattr(self,key,state[key])
        elif self.last_iteration >= self.activation_iteration:
            # v1 history is contaminated by post-QP performance. Never use it
            # to manufacture a permissive baseline; retain objective progress.
            self.history.clear()
            self.baseline_frozen = True
            self.pre_qp_reference = None
        if self.pre_qp_reference is None and self.cfg.objective_curriculum_baseline_override is not None:
            self.pre_qp_reference = self.cfg.objective_curriculum_baseline_override
        self._set_threshold()
        self.snapshot_iteration = self.snapshot = None
