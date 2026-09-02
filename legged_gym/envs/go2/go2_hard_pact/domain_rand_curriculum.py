"""Simulator-neutral, checkpointable Go2 HardPACT domain randomization.

Genesis PACT is the source of truth for ranges and phase ordering.  This module
owns *progression and sampling only*; backend adapters remain responsible for
installing sampled SI values through their simulator APIs.
"""

from __future__ import annotations

from collections import deque
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
    return {
        "ground_friction": DomainRandFeature("ground_friction", tuple(d.friction_range), tuple(d.friction_range), "fixed", enabled("randomize_friction"), "reset", "coefficient"),
        "added_base_mass": DomainRandFeature("added_base_mass", (float(d.added_mass_min), float(d.min_added_mass_max)), (float(d.added_mass_min), float(d.max_added_mass_max)), "mass_com", enabled("randomize_base_mass"), "reset", "kg"),
        "base_com_x": DomainRandFeature("base_com_x", (-float(d.com_displacement_x_min), float(d.com_displacement_x_min)), (-float(d.com_displacement_x_max), float(d.com_displacement_x_max)), "mass_com", enabled("randomize_com_displacement"), "reset", "m"),
        "base_com_y": DomainRandFeature("base_com_y", (-float(d.com_displacement_y_min), float(d.com_displacement_y_min)), (-float(d.com_displacement_y_max), float(d.com_displacement_y_max)), "mass_com", enabled("randomize_com_displacement"), "reset", "m"),
        "base_com_z": DomainRandFeature("base_com_z", (-float(d.com_displacement_z_min), float(d.com_displacement_z_min)), (-float(d.com_displacement_z_max), float(d.com_displacement_z_max)), "mass_com", enabled("randomize_com_displacement"), "reset", "m"),
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
        "persistent_torque": DomainRandFeature("persistent_torque", (-float(d.persistent_torque_min_nm), float(d.persistent_torque_min_nm)), (-float(d.persistent_torque_max_nm), float(d.persistent_torque_max_nm)), "disturbance", enabled("persistent_disturbance"), "runtime", "N*m"),
    }


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
        self.phase_index = 0
        self.last_iteration = -1
        self.last_step_iteration = -10**9
        self.reward_ema = None
        self.best_reward_ema = -float("inf")
        self.required_reward = 0.0
        self.reward_history = deque(maxlen=int(d.best_reward_window))
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

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

    def advance(self, iteration: int, mean_reward=None) -> bool:
        """Advance at most once for an iteration; return whether state changed."""
        iteration = int(iteration)
        if iteration <= self.last_iteration:
            return False
        self.last_iteration = iteration
        d = self.cfg.domain_rand
        if mean_reward is not None:
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
            can_step = True
        if iteration <= int(d.push_warmup) or not can_step:
            return False
        if iteration - self.last_step_iteration < int(d.step_interval) or self.phase == "complete":
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
        }

    def load_state_dict(self, state):
        self.progress = {k: float(v) for k, v in state["progress"].items()}
        self.phase_index = int(state["phase_index"])
        self.last_iteration = int(state["last_iteration"])
        self.last_step_iteration = int(state["last_step_iteration"])
        self.reward_ema = state["reward_ema"]
        self.best_reward_ema = float(state["best_reward_ema"])
        self.required_reward = float(state["required_reward"])
        self.reward_history.clear(); self.reward_history.extend(state["reward_history"])
        self.generator.set_state(state["generator_state"].cpu())

    def report(self, capabilities: Mapping[str, bool]):
        effective = self.effective_ranges()
        return {
            name: {
                **asdict(spec),
                "requested_range": effective[name],
                "effective_range": effective[name] if capabilities.get(name, False) else None,
                "supported": bool(capabilities.get(name, False)),
            }
            for name, spec in self.schema.items()
        }
