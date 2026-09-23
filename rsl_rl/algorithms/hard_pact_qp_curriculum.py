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
        if (cfg.correction_ramp_duration < 0 or cfg.objective_curriculum_window < 1
                or cfg.objective_curriculum_step_interval < 1
                or not 0 < cfg.objective_curriculum_ema_alpha <= 1
                or not 0 <= cfg.objective_curriculum_quantile <= 1
                or not 0 < cfg.objective_curriculum_progress_delta <= 1
                or cfg.objective_curriculum_min_samples < 1):
            raise ValueError("Invalid QP curriculum schedule")

    def begin(self, iteration):
        """Freeze weights/alpha for the complete rollout plus all PPO epochs."""
        if iteration == self.snapshot_iteration:
            return self.snapshot
        c = self.cfg
        alpha = (1.0 if not c.correction_ramp_enabled or c.correction_ramp_duration == 0
                 else min(1.0, max(0.0, (iteration-self.origin)/c.correction_ramp_duration)))
        weights = {}
        for name in ("contact_acceleration_weight", "attitude_weight"):
            final = getattr(c, name + "_final")
            final = getattr(c, name) if final is None else final
            initial = getattr(c, name + "_initial")
            initial = .25 * final if initial is None else initial
            if not all(math.isfinite(v) and v >= 0 for v in (initial, final)):
                raise ValueError("QP curriculum weights must be finite and nonnegative")
            weights[name] = (initial + self.progress*(final-initial)
                             if c.objective_curriculum_enabled else getattr(c, name))
        self.snapshot_iteration = int(iteration)
        self.snapshot_progress = self.progress
        self.snapshot = (replace(c, **weights), alpha)
        return self.snapshot

    def finish(self, iteration, performance, count):
        """Evidence collected before reward scaling; advance only the next snapshot."""
        self.advanced = False
        if iteration <= self.last_iteration:
            return
        self.last_iteration = int(iteration)
        c = self.cfg
        if (not c.objective_curriculum_enabled or count < c.objective_curriculum_min_samples
                or performance is None or not math.isfinite(performance)):
            return
        self.ema = (performance if self.ema is None else
                    (1-c.objective_curriculum_ema_alpha)*self.ema+c.objective_curriculum_ema_alpha*performance)
        self.history.append(self.ema)
        ordered = sorted(self.history)
        reference = ordered[int(c.objective_curriculum_quantile*(len(ordered)-1))]
        self.threshold = max(c.objective_curriculum_min_tracking,
                             c.objective_curriculum_recovery_ratio*reference)
        if (iteration >= self.start and iteration-self.last_step >= c.objective_curriculum_step_interval
                and self.ema >= self.threshold and self.progress < 1):
            self.progress = min(1., self.progress+c.objective_curriculum_progress_delta)
            self.last_step = int(iteration)
            self.advanced = True

    def metrics(self):
        cfg, alpha = self.snapshot
        return dict(alpha=alpha, weight_progress=self.snapshot_progress,
                    next_weight_progress=self.progress,
                    contact_acceleration_weight=cfg.contact_acceleration_weight,
                    attitude_weight=cfg.attitude_weight,
                    tracking_ema=float('nan') if self.ema is None else self.ema,
                    tracking_threshold=float('nan') if self.threshold is None else self.threshold,
                    advanced=int(self.advanced))

    def state_dict(self):
        return {"version": 1, **{k: getattr(self, k) for k in
                ("origin", "start", "progress", "ema", "threshold", "last_iteration", "last_step")},
                "history": list(self.history)}

    def load_state_dict(self, state):
        if state["version"] != 1:
            raise ValueError("Unsupported QP curriculum checkpoint version")
        for key in ("origin", "start", "progress", "ema", "threshold", "last_iteration", "last_step"):
            setattr(self, key, state[key])
        self.history.clear()
        self.history.extend(state["history"])
        self.snapshot_iteration = self.snapshot = None
