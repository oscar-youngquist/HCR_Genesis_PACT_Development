"""Detached FACET equations (8--9); SI units, world axes, no physical actuation."""
from dataclasses import dataclass, field
import math

import torch

from legged_gym.envs.b1z1.directional_compliance import rotate_yaw


@dataclass
class FACETReferenceCfg:
    enabled: bool = True
    # Representative fixed values within upstream B1Z1 sampling ranges, not
    # fitted robot inertia. Upstream EE mass=2 and yaw inertia=1.5.
    base_mass: float = 4.0
    ee_mass: float = 2.0
    yaw_inertia: float = 1.5
    base_stiffness: float = 16.0
    base_damping: float = 7.2
    ee_stiffness: float = 16.0
    ee_damping: float = 7.2
    yaw_stiffness: float = 16.0
    yaw_damping: float = 7.2
    alpha: float = 1.0
    seed: int = 1701
    # Optional {parameter: [low, high]}; independently uniform at reset only.
    sample_ranges: dict = field(default_factory=dict)
    max_substep_seconds: float = 0.005


PARAMETERS = ("base_mass", "ee_mass", "yaw_inertia", "base_stiffness",
              "base_damping", "ee_stiffness", "ee_damping", "yaw_stiffness",
              "yaw_damping", "alpha")


def ee_setpoint(base_xy, base_velocity_xy, yaw, yaw_rate, local, local_velocity, height):
    """Mount-inclusive local XY, terrain-referenced world height; no base-Z transport."""
    offset, velocity = rotate_yaw(local, yaw), rotate_yaw(local_velocity, yaw)
    position = offset + torch.cat((base_xy, height[:, None]), -1)
    transport = torch.stack((-offset[:, 1], offset[:, 0], torch.zeros_like(yaw)), -1)
    velocity = velocity + torch.cat((base_velocity_xy, torch.zeros_like(height[:, None])), -1)
    return position, velocity + yaw_rate[:, None] * transport


def applied_loads(commands, ee_force, torso_source, base_origin, application_point):
    """Held physical loads ON robot; source couple is at its application point.

    Capture the point with the interval-start state. A base-referenced source
    passes base_origin as its point, so the lever is added exactly once.
    """
    force, couple, _ = torso_source
    ee = ee_force if commands.push_gripper_stators and commands.apply_ee_external_forces else torch.zeros_like(ee_force)
    force = force if commands.push_robot_base and commands.apply_base_external_forces else torch.zeros_like(force)
    couple = couple if commands.push_robot_base and getattr(commands, "apply_base_external_torques", False) else torch.zeros_like(couple)
    moment = couple + torch.linalg.cross(application_point - base_origin, force)
    return ee.detach().clone(), force.detach().clone(), moment[:, 2].detach().clone()


class FACETReference:
    """Batched virtual state, with reset-only RNG and copy-on-read snapshots.

    State columns: base XY/velocity XY, yaw/rate, EE XYZ/velocity XYZ.
    Neither reads nor advance draw random numbers or mutate caller tensors.
    """
    def __init__(self, count, device, cfg, dtype=torch.float32):
        self.cfg = cfg
        self._state = torch.zeros(count, 12, device=device, dtype=dtype)
        self._initialized = torch.zeros(count, device=device, dtype=torch.bool)
        self._ready = False
        self._rng = torch.Generator(device=device).manual_seed(cfg.seed)
        unknown = set(cfg.sample_ranges) - set(PARAMETERS)
        if unknown:
            raise ValueError(f"Unknown FACET parameters: {sorted(unknown)}")
        self._ranges = {}
        for name in PARAMETERS:
            lo, hi = cfg.sample_ranges.get(name, (getattr(cfg, name),) * 2)
            positive = name in ("base_mass", "ee_mass", "yaw_inertia")
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi and
                    (lo > 0 if positive else lo >= 0) and (name != "alpha" or hi <= 1)):
                raise ValueError(f"Invalid FACET {name} range: {(lo, hi)}")
            self._ranges[name] = (lo, hi)
        if not math.isfinite(cfg.max_substep_seconds) or cfg.max_substep_seconds <= 0:
            raise ValueError("FACET max_substep_seconds must be positive")
        self._parameters = {name: self._state.new_full((count, 1), lo)
                            for name, (lo, _) in self._ranges.items()}
        # Conservative fixed stability budget includes translational coupling.
        mass = min(self._ranges[n][0] for n in ("base_mass", "ee_mass", "yaw_inertia"))
        stiffness = sum(self._ranges[n][1] for n in PARAMETERS if "stiffness" in n)
        damping = sum(self._ranges[n][1] for n in PARAMETERS if "damping" in n)
        rate = 2 * (math.sqrt(stiffness / mass) + damping / mass)
        self._max_dt = min(cfg.max_substep_seconds, 0.5 / max(rate, 1e-12))
        self._diagnostics = self._state.new_zeros(count, 15)
        self._elapsed = self._state.new_zeros(count)

    @torch.no_grad()
    def reset(self, ids, base_position, base_velocity, yaw, yaw_rate, ee_position, ee_velocity):
        """Measured state arrays have len(ids) rows; untouched rows retain state/parameters."""
        self._state[ids] = torch.cat((base_position[:, :2], base_velocity[:, :2],
            yaw[:, None], yaw_rate[:, None], ee_position, ee_velocity), -1)
        for name, (lo, hi) in self._ranges.items():
            values = lo if lo == hi else lo + (hi-lo) * torch.rand(
                (len(ids), 1), device=self._state.device, dtype=self._state.dtype, generator=self._rng)
            self._parameters[name][ids] = values
        self._diagnostics[ids] = 0
        self._elapsed[ids] = 0
        self._initialized[ids] = True
        self._ready = bool(self._initialized.all())  # Reset-time check, no per-step GPU synchronization.

    def _rhs(self, state, time, inputs, loads):
        p = self._parameters
        ee_force, base_force, yaw_moment = loads
        measured_xy, measured_yaw, command_world, yaw_command, local, local_rate, height = inputs
        xy, velocity, yaw, rate = state[:, :2], state[:, 2:4], state[:, 4], state[:, 5]
        # First-order hold of the sampled nominal trajectory over one control dt.
        target, target_velocity = ee_setpoint(xy, velocity, yaw, rate,
            local + time * local_rate, local_rate, height)
        spring_ee = p["ee_stiffness"] * (target-state[:, 6:9]) + p["ee_damping"] * (target_velocity-state[:, 9:12])
        spring_base = p["base_stiffness"]*(measured_xy-xy) + p["base_damping"]*(command_world-velocity)
        reaction = -p["alpha"]*spring_ee[:, :2]
        yaw_error = torch.atan2(torch.sin(measured_yaw-yaw), torch.cos(measured_yaw-yaw))
        spring_yaw = p["yaw_stiffness"][:, 0]*yaw_error + p["yaw_damping"][:, 0]*(yaw_command-rate)
        derivative = torch.cat((velocity, (spring_base+base_force[:, :2]+reaction)/p["base_mass"],
            rate[:, None], ((spring_yaw+yaw_moment)/p["yaw_inertia"][:, 0])[:, None],
            state[:, 9:12], (spring_ee+ee_force)/p["ee_mass"]), -1)
        # Interval-average force components below use the same RK quadrature.
        forces = torch.cat((spring_base, spring_ee, reaction, base_force,
                            spring_yaw[:, None], yaw_moment[:, None], ee_force), -1)
        return derivative, forces

    @torch.no_grad()
    def advance(self, dt, measured_xy, measured_yaw, command_world, yaw_command,
                ee_local, ee_local_velocity, height, ee_force, base_force, yaw_moment):
        """One completed control interval. Inputs are frozen interval-start snapshots."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("FACET control dt must be positive")
        if not self._ready:
            raise RuntimeError("Initialize every FACET row before advancing")
        inputs = (measured_xy, measured_yaw, command_world, yaw_command, ee_local, ee_local_velocity, height)
        loads = (ee_force, base_force, yaw_moment)
        steps = max(1, math.ceil(dt / self._max_dt))
        h = dt / steps
        self._diagnostics.zero_()
        for i in range(steps):  # Fixed batch-wide numerical substeps, never per-robot loops.
            s, t = self._state, i*h
            k1, f1 = self._rhs(s, t, inputs, loads)
            k2, f2 = self._rhs(s+h*k1/2, t+h/2, inputs, loads)
            k3, f3 = self._rhs(s+h*k2/2, t+h/2, inputs, loads)
            k4, f4 = self._rhs(s+h*k3, t+h, inputs, loads)
            self._state = s+h*(k1+2*k2+2*k3+k4)/6
            self._diagnostics.add_((f1+2*f2+2*f3+f4)/(6*steps))
        self._elapsed.add_(dt)

    def read(self):
        """Owned detached copies: editing a result cannot edit the reference."""
        s, d = self._state, self._diagnostics
        values = dict(base_position_xy=s[:, :2], base_velocity_xy=s[:, 2:4],
            yaw=s[:, 4], yaw_rate=s[:, 5], ee_position_world=s[:, 6:9], ee_velocity_world=s[:, 9:12],
            base_spring_force_xy=d[:, :2], ee_spring_force_world=d[:, 2:5],
            ee_reaction_xy=d[:, 5:7], applied_base_force_world=d[:, 7:10],
            yaw_spring_moment=d[:, 10], applied_yaw_moment=d[:, 11],
            applied_ee_force_world=d[:, 12:15], elapsed_seconds=self._elapsed)
        values.update({"parameter_"+name: value for name, value in self._parameters.items()})
        return {name: value.detach().clone() for name, value in values.items()}
