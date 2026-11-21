"""Simulator-neutral, checkpointable Go2 HardPACT domain randomization.

Genesis PACT is the source of truth for ranges and phase ordering.  This module
owns *progression and sampling only*; backend adapters remain responsible for
installing sampled SI values through their simulator APIs.
"""

from __future__ import annotations

from collections import deque
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple

import torch


Range = Tuple[float, float]


@dataclass(frozen=True)
class DomainRandFeature:
    name: str
    initial: Range
    final: Range
    phase: str
    enabled: bool
    update_mode: str
    unit: str

    def range_at(self, progress: float) -> Range:
        p = min(1.0, max(0.0, float(progress)))
        return tuple(a + p * (b - a) for a, b in zip(self.initial, self.final))


def go2_pact_domain_rand_schema(cfg) -> Mapping[str, DomainRandFeature]:
    """Build the immutable schema from the legacy Go2 PACT configuration."""
    d = cfg.domain_rand
    enabled = lambda name: bool(getattr(d, name, False))
    positive_z = bool(getattr(d, 'com_displacement_z_positive', getattr(d, 'com_rand_z_positive', False)))
    z_floor = float(getattr(d, 'com_displacement_z_min_pos', 0.0)) if positive_z else None
    for axis in ('x', 'y', 'z'):
        low, high = (float(getattr(d, 'com_displacement_'+axis+'_'+end)) for end in ('min','max'))
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
            raise ValueError('Invalid CoM displacement bounds: '+axis)
    if positive_z and not (math.isfinite(z_floor) and 0 <= z_floor < float(d.com_displacement_z_min)):
        raise ValueError('Positive CoM Z minimum must be below initial upper bound; use 0 for [0, z_max]')
    # Older configs retain the original isotropic torque envelope.
    yaw_initial = float(getattr(d, "persistent_yaw_torque_min_nm", d.persistent_torque_min_nm))
    yaw_final = float(getattr(d, "persistent_yaw_torque_max_nm", d.persistent_torque_max_nm))
    if not 0.0 <= yaw_initial <= yaw_final:
        raise ValueError("Yaw torque magnitudes must satisfy 0 <= min <= max")
    for name in ("persistent_force_min_n", "persistent_force_max_n",
                 "persistent_vertical_force_min_n", "persistent_vertical_force_max_n"):
        if float(getattr(d, name)) < 0:
            raise ValueError(name + " must be a nonnegative force magnitude")
    return {
        "ground_friction": DomainRandFeature("ground_friction", tuple(d.friction_range), tuple(d.friction_range), "fixed", enabled("randomize_friction"), "reset", "coefficient"),
        "added_base_mass": DomainRandFeature("added_base_mass", (float(d.added_mass_min), float(d.min_added_mass_max)), (float(d.added_mass_min), float(d.max_added_mass_max)), "mass_com", enabled("randomize_base_mass"), "reset", "kg"),
        "base_com_x": DomainRandFeature("base_com_x", (-float(d.com_displacement_x_min), float(d.com_displacement_x_min)), (-float(d.com_displacement_x_max), float(d.com_displacement_x_max)), "mass_com", enabled("randomize_com_displacement"), "reset", "m"),
        "base_com_y": DomainRandFeature("base_com_y", (-float(d.com_displacement_y_min), float(d.com_displacement_y_min)), (-float(d.com_displacement_y_max), float(d.com_displacement_y_max)), "mass_com", enabled("randomize_com_displacement"), "reset", "m"),
        "base_com_z": DomainRandFeature("base_com_z", (z_floor if positive_z else -float(d.com_displacement_z_min), float(d.com_displacement_z_min)), (z_floor if positive_z else -float(d.com_displacement_z_max), float(d.com_displacement_z_max)), "mass_com", enabled("randomize_com_displacement"), "reset", "m"),
        "control_delay": DomainRandFeature("control_delay", tuple(map(float, d.ctrl_delay_step_range)), tuple(map(float, d.ctrl_delay_step_range)), "fixed", enabled("randomize_ctrl_delay"), "reset", "control_steps"),
        "kp_scale": DomainRandFeature("kp_scale", tuple(d.kp_range), tuple(d.kp_range), "fixed", enabled("randomize_pd_gain"), "reset", "ratio"),
        "kd_scale": DomainRandFeature("kd_scale", tuple(d.kd_range), tuple(d.kd_range), "fixed", enabled("randomize_pd_gain"), "reset", "ratio"),
        "motor_strength": DomainRandFeature("motor_strength", tuple(d.motor_strength_range), tuple(d.motor_strength_range), "fixed", enabled("randomize_motor_strength"), "reset", "ratio"),
        "armature": DomainRandFeature("armature", tuple(d.joint_armature_range), tuple(d.joint_armature_range), "fixed", enabled("randomize_joint_armature"), "reset", "kg*m^2"),
        "joint_friction": DomainRandFeature("joint_friction", tuple(d.joint_friction_range_start), tuple(d.joint_friction_range_end), "joint_dynamics", enabled("randomize_joint_friction"), "reset", "N*m"),
        "joint_stiffness": DomainRandFeature("joint_stiffness", tuple(d.joint_stiffness_range_start), tuple(d.joint_stiffness_range_end), "joint_dynamics", enabled("randomize_joint_stiffness"), "reset", "N*m/rad"),
        "joint_damping": DomainRandFeature("joint_damping", tuple(d.joint_damping_range_start), tuple(d.joint_damping_range_end), "joint_dynamics", enabled("randomize_joint_damping"), "reset", "N*m*s/rad"),
        "push_xy": DomainRandFeature("push_xy", (-float(d.min_push_vel_xy), float(d.min_push_vel_xy)), (-float(d.max_push_vel_xy), float(d.max_push_vel_xy)), "disturbance", enabled("push_robots"), "runtime", "m/s"),
        "push_z": DomainRandFeature("push_z", (-float(d.min_vertical_push), 0.0), (-float(d.max_vertical_push), 0.0), "disturbance", enabled("push_robots"), "runtime", "m/s"),
        "push_angular": DomainRandFeature("push_angular", (-float(d.min_push_torque), float(d.min_push_torque)), (-float(d.max_push_torque), float(d.max_push_torque)), "disturbance", enabled("push_robots"), "runtime", "rad/s"),
        "persistent_force": DomainRandFeature("persistent_force", (-float(d.persistent_force_min_n), float(d.persistent_force_min_n)), (-float(d.persistent_force_max_n), float(d.persistent_force_max_n)), "disturbance", enabled("persistent_disturbance"), "runtime", "N"),
        "persistent_vertical_force": DomainRandFeature("persistent_vertical_force", (-float(d.persistent_vertical_force_min_n), 0.0), (-float(d.persistent_vertical_force_max_n), 0.0), "disturbance", enabled("persistent_disturbance"), "runtime", "N"),
        "persistent_torque": DomainRandFeature("persistent_torque", (-float(d.persistent_torque_min_nm), float(d.persistent_torque_min_nm)), (-float(d.persistent_torque_max_nm), float(d.persistent_torque_max_nm)), "disturbance", enabled("persistent_disturbance"), "runtime", "N*m"),
        "persistent_yaw_torque": DomainRandFeature("persistent_yaw_torque", (-yaw_initial, yaw_initial), (-yaw_final, yaw_final), "disturbance", enabled("persistent_disturbance"), "runtime", "N*m"),
    }


class PostInterventionDisturbanceGate:
    """HardPACT-only completion latch, settling and best-window survival gate.

    Uses the DR tracking EMA/quantile convention, but references only improve
    after initialization. Decisions affect subsequent rollouts; no reset clock.
    """
    WAIT_QP, SETTLING, EVIDENCE, TRACKING, FAILURE, OPEN, WARMUP, INTERVAL, COMPLETE = range(9)

    def __init__(self, cfg):
        self.settling = int(cfg.disturbance_settling_iterations)
        self.window = int(cfg.disturbance_gate_window_iterations)
        self.min_exposure = float(cfg.disturbance_gate_min_env_seconds)
        self.tracking_ratio = float(cfg.recovery_ratio)
        self.min_tracking = float(cfg.min_reward_to_step)
        self.quantile = float(cfg.best_reward_quantile)
        self.failure_ratio = float(cfg.disturbance_survival_ratio)
        self.failure_allowance = float(cfg.disturbance_failure_allowance)
        if (self.settling != cfg.disturbance_settling_iterations or self.settling < 0
                or self.window != cfg.disturbance_gate_window_iterations or self.window < 1
                or not math.isfinite(self.min_exposure) or self.min_exposure <= 0
                or not 0 < self.tracking_ratio <= 1 or not 0 <= self.quantile <= 1
                or not math.isfinite(self.min_tracking) or self.min_tracking < 0
                or not math.isfinite(self.failure_ratio) or self.failure_ratio < 1
                or not math.isfinite(self.failure_allowance) or self.failure_allowance < 0):
            raise ValueError('Invalid HardPACT disturbance gate configuration')
        self.completion_iteration = None
        self.last_iteration = -1
        self.tracking_reference = self.failure_reference = None
        self.tracking_threshold = self.failure_threshold = None
        self.failure_rate = None
        self.history = []
        self.window_iterations = 0
        self.failures = self.exposure = 0.
        self.window_valid = True
        self.gate = False
        self.reason = self.WAIT_QP

    def finish(self, iteration, qp_complete, ema, count, failures, exposure):
        consecutive = self.last_iteration < 0 or iteration == self.last_iteration + 1
        self.last_iteration = iteration
        if self.completion_iteration is None:
            if not qp_complete:
                self.reason = self.WAIT_QP
                return False
            self.completion_iteration = iteration
            self.reason = self.SETTLING
            return False  # evidence starts AFTER the completing rollout
        valid = (consecutive and all(v is not None and math.isfinite(v)
                 for v in (ema, count, failures, exposure))
                 and count > 0 and exposure > 0 and failures >= 0)
        # Use the previous complete-window gate, never this iteration's decision.
        ready = (valid and self.gate and iteration-self.completion_iteration >= self.settling)
        if self.gate:
            self.reason = self.OPEN
        self.window_iterations += 1
        self.window_valid &= valid
        if valid:
            self.history.append(ema)
            self.failures += failures
            self.exposure += exposure
        else:
            self.gate = False
            self.reason = self.EVIDENCE
        if self.window_iterations >= self.window:
            if self.window_valid and self.exposure >= self.min_exposure:
                reference = sorted(self.history)[int(self.quantile*(len(self.history)-1))]
                self.tracking_reference = (reference if self.tracking_reference is None
                                           else max(self.tracking_reference, reference))
                self.failure_rate = self.failures/self.exposure
                self.failure_reference = (self.failure_rate if self.failure_reference is None
                                          else min(self.failure_reference, self.failure_rate))
                self.tracking_threshold = max(self.min_tracking, self.tracking_ratio*self.tracking_reference)
                self.failure_threshold = self.failure_ratio*self.failure_reference+self.failure_allowance
                good_tracking = ema >= self.tracking_threshold
                good_failure = self.failure_rate <= self.failure_threshold
                self.gate = good_tracking and good_failure
                self.reason = self.TRACKING if not good_tracking else self.FAILURE if not good_failure else self.OPEN
            else:
                self.gate = False
                self.reason = self.EVIDENCE
            self.history = []
            self.window_iterations = 0
            self.failures = self.exposure = 0.
            self.window_valid = True
        if iteration-self.completion_iteration < self.settling:
            self.reason = self.SETTLING
        return ready

    def metrics(self, iteration):
        available = lambda v: float('nan') if v is None else v
        elapsed = 0 if self.completion_iteration is None else max(0,iteration-self.completion_iteration)
        return dict(settling_iterations=min(elapsed,self.settling), gate_open=int(self.gate),
            pause_reason=self.reason, tracking_reference=available(self.tracking_reference),
            tracking_threshold=available(self.tracking_threshold), failure_reference=available(self.failure_reference),
            failure_threshold=available(self.failure_threshold), failures_per_env_second=available(self.failure_rate))

    def state_dict(self):
        return dict(version=1, **deepcopy(vars(self)))

    def load_state_dict(self, state):
        if state.get('version') != 1 or set(state) != {'version', *vars(self)}:
            raise ValueError('Incompatible HardPACT disturbance sequence state')
        self.__dict__.update(deepcopy({k:v for k,v in state.items() if k != 'version'}))


class HardPACTDomainRandCurriculum:
    """Deterministic three-phase state advanced once per PPO iteration."""

    phases = ("joint_dynamics", "mass_com", "disturbance")

    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.schema = go2_pact_domain_rand_schema(cfg)
        d = cfg.domain_rand
        self.progress = {phase: 0.0 for phase in self.phases}
        self.deltas = {
            "joint_dynamics": float(d.joint_dynamics_progress_delta),
            "mass_com": float(d.mass_com_progress_delta),
            "disturbance": float(d.disturbance_progress_delta),
        }
        self.enabled_phases = tuple(
            p for p, flag in zip(self.phases, (
                d.use_joint_dynamics_curriculum,
                d.use_mass_com_curriculum,
                d.use_disturbance_curriculum,
            )) if flag
        )
        self.disabled_curriculum_use_final_bounds = bool(
            getattr(d, "disabled_curriculum_use_final_bounds", False))
        self._pin_disabled_phase_bounds()
        self.phase_index = 0
        self.last_iteration = -1
        self.last_step_iteration = -10**9
        self.reward_ema = None
        self.best_reward_ema = -float("inf")
        self.required_reward = 0.0
        self.reward_history = deque(maxlen=int(d.best_reward_window))
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.sequence = (PostInterventionDisturbanceGate(d)
                         if getattr(d,'disturbance_sequence_enabled',False) else None)

    def _pin_disabled_phase_bounds(self):
        # Use the same progress for backend ranges and persistent-wrench
        # sampling, which reads phase progress directly. Enabled phases retain
        # their schedule/checkpoint progress; feature enable flags are untouched.
        for phase in self.phases:
            if phase not in self.enabled_phases:
                self.progress[phase] = float(self.disabled_curriculum_use_final_bounds)

    @property
    def phase(self):
        return self.enabled_phases[self.phase_index] if self.phase_index < len(self.enabled_phases) else "complete"

    def effective_ranges(self) -> Dict[str, Range]:
        return {
            name: spec.range_at(self.progress.get(spec.phase, 0.0))
            for name, spec in self.schema.items()
        }

    def sample(self, name: str, shape, *, device="cpu", dtype=torch.float32):
        low, high = self.effective_ranges()[name]
        value = torch.rand(shape, generator=self.generator, dtype=dtype)
        return (value * (high - low) + low).to(device)

    def advance(self, iteration: int, mean_reward=None, *, qp_complete=None,
                tracking_count=0, failures=None, exposure=None) -> bool:
        """Advance at most once for an iteration; return whether state changed."""
        iteration = int(iteration)
        if iteration <= self.last_iteration:
            return False
        self.last_iteration = iteration
        d = self.cfg.domain_rand
        if mean_reward is not None and math.isfinite(float(mean_reward)):
            reward = float(mean_reward)
            alpha = float(d.reward_ema_alpha)
            self.reward_ema = reward if self.reward_ema is None else (1-alpha)*self.reward_ema + alpha*reward
            self.reward_history.append(self.reward_ema)
            ordered = sorted(self.reward_history)
            qindex = min(len(ordered)-1, int(float(d.best_reward_quantile) * max(0, len(ordered)-1)))
            self.best_reward_ema = ordered[qindex]
            self.required_reward = float(d.recovery_ratio) * self.best_reward_ema
            can_step = self.reward_ema >= max(float(d.min_reward_to_step), self.required_reward)
        else:
            can_step = mean_reward is None
        if self.sequence is not None:
            can_step = self.sequence.finish(iteration, qp_complete,
                self.reward_ema if mean_reward is not None and math.isfinite(float(mean_reward)) else None,
                tracking_count, failures, exposure)
            self.required_reward = (self.sequence.tracking_threshold
                                    if self.sequence.tracking_threshold is not None else 0.)
        if iteration <= int(d.push_warmup) or not can_step:
            if self.sequence is not None and can_step: self.sequence.reason = self.sequence.WARMUP
            return False
        if iteration - self.last_step_iteration < int(d.step_interval) or self.phase == "complete":
            if self.sequence is not None:
                self.sequence.reason = self.sequence.COMPLETE if self.phase == 'complete' else self.sequence.INTERVAL
            return False
        phase = self.phase
        self.progress[phase] = min(1.0, self.progress[phase] + self.deltas[phase])
        self.last_step_iteration = iteration
        if self.progress[phase] >= 1.0:
            self.phase_index += 1
        return True

    def state_dict(self):
        return {
            "progress": dict(self.progress), "phase_index": self.phase_index,
            "last_iteration": self.last_iteration,
            "last_step_iteration": self.last_step_iteration,
            "reward_ema": self.reward_ema,
            "best_reward_ema": self.best_reward_ema,
            "required_reward": self.required_reward,
            "reward_history": list(self.reward_history),
            "generator_state": self.generator.get_state(),
            "enabled_phases": list(self.enabled_phases),
            "sequence": None if self.sequence is None else self.sequence.state_dict(),
        }

    def load_state_dict(self, state):
        if 'enabled_phases' in state and tuple(state['enabled_phases']) != self.enabled_phases:
            raise ValueError('Incompatible domain curriculum phase configuration')
        self.progress = {k: float(v) for k, v in state["progress"].items()}
        # Current explicit config chooses disabled endpoints even for older
        # checkpoints, which did not expose this setting.
        self._pin_disabled_phase_bounds()
        # Legacy positional indices cannot survive a disabled/skipped phase.
        # Recover from named attained progress, never infer from global time.
        self.phase_index = (int(state['phase_index']) if 'enabled_phases' in state else
                            next((i for i,p in enumerate(self.enabled_phases)
                                  if self.progress[p] < 1.), len(self.enabled_phases)))
        self.last_iteration = int(state["last_iteration"])
        self.last_step_iteration = int(state["last_step_iteration"])
        self.reward_ema = state["reward_ema"]
        self.best_reward_ema = float(state["best_reward_ema"])
        self.required_reward = float(state["required_reward"])
        self.reward_history.clear(); self.reward_history.extend(state["reward_history"])
        self.generator.set_state(state["generator_state"].cpu())
        if self.sequence is not None and state.get('sequence') is not None:
            self.sequence.load_state_dict(state['sequence'])

    def report(self, capabilities: Mapping[str, bool]):
        effective = self.effective_ranges()
        capabilities = dict(capabilities)
        capabilities.setdefault("persistent_vertical_force", capabilities.get("persistent_force", False))
        capabilities.setdefault("persistent_yaw_torque", capabilities.get("persistent_torque", False))
        return {
            name: {
                **asdict(spec),
                "requested_range": effective[name],
                "effective_range": effective[name] if capabilities.get(name, False) else None,
                "supported": bool(capabilities.get(name, False)),
                **({"maximum_planar_norm_n": effective[name][1], "sampling": "uniform world-XY disk"}
                   if name == "persistent_force" else {}),
            }
            for name, spec in self.schema.items()
        }
