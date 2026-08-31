"""Independent instantaneous-push and sustained-wrench disturbances."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .schema import body_to_world, world_to_yaw_local


def _uniform(shape, low: float, high: float, *, device, dtype):
    return torch.rand(shape, device=device, dtype=dtype) * (high - low) + low


def added_mass_gravity_wrench_world(
    added_mass: torch.Tensor,
    gravity_world: torch.Tensor,
    com_offset_body: torch.Tensor,
    base_quaternion_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Equivalent persistent torso wrench relative to the nominal mass.

    The Genesis and Isaac Lab mass randomizers store a signed mass delta and
    apply it to the base link.  Its persistent external-force equivalent is
    ``delta_mass * gravity``.  ``com_offset_body`` is the known randomized COM
    displacement from the nominal torso wrench origin.  It is rotated into
    the world frame and contributes the moment ``r_world x force_world``.

    The returned linear force and moment are both expressed in world axes at
    the nominal torso origin.  A zero COM displacement recovers the previous
    force-only label.
    """
    if added_mass.ndim != 2 or added_mass.shape[-1] != 1:
        raise ValueError("added_mass must be (batch,1)")
    gravity_world = torch.as_tensor(
        gravity_world, device=added_mass.device, dtype=added_mass.dtype
    )
    if gravity_world.ndim == 1:
        gravity_world = gravity_world.unsqueeze(0).expand(added_mass.shape[0], -1)
    if gravity_world.shape != (added_mass.shape[0], 3):
        raise ValueError("gravity_world must be (3,) or (batch,3)")
    com_offset_body = torch.as_tensor(
        com_offset_body, device=added_mass.device, dtype=added_mass.dtype
    )
    if com_offset_body.shape != (added_mass.shape[0], 3):
        raise ValueError("com_offset_body must be (batch,3)")
    base_quaternion_xyzw = torch.as_tensor(
        base_quaternion_xyzw,
        device=added_mass.device,
        dtype=added_mass.dtype,
    )
    if base_quaternion_xyzw.shape != (added_mass.shape[0], 4):
        raise ValueError("base_quaternion_xyzw must be (batch,4)")
    force_world = added_mass * gravity_world
    com_offset_world = body_to_world(
        com_offset_body, base_quaternion_xyzw
    )
    moment_world = torch.cross(com_offset_world, force_world, dim=-1)
    return torch.cat((force_world, moment_world), dim=-1)


def torso_wrench_scale_from_ranges(
    force_bounds_n,
    torque_bounds_nm,
    added_mass_range,
    gravity_world,
    *,
    com_offset_ranges=None,
    include_sustained_force=True,
    include_sustained_torque=True,
    include_added_mass=True,
) -> tuple[float, ...]:
    """Compute component-wise head/loss scales from effective DR ranges."""
    if len(force_bounds_n) != 2 or len(torque_bounds_nm) != 2:
        raise ValueError("force and torque bounds must each contain two values")
    if len(added_mass_range) != 2:
        raise ValueError("added_mass_range must contain two values")
    gravity = torch.as_tensor(gravity_world, dtype=torch.float64)
    if gravity.shape != (3,):
        raise ValueError("gravity_world must contain three values")
    sustained_force = (
        max(abs(float(value)) for value in force_bounds_n)
        if include_sustained_force else 0.0
    )
    sustained_torque = (
        max(abs(float(value)) for value in torque_bounds_nm)
        if include_sustained_torque else 0.0
    )
    force_scale = torch.full((3,), sustained_force, dtype=torch.float64)
    torque_scale = torch.full((3,), sustained_torque, dtype=torch.float64)
    if include_added_mass:
        maximum_mass_delta = max(
            abs(float(value)) for value in added_mass_range
        )
        force_scale += maximum_mass_delta * gravity.abs()
        if com_offset_ranges is not None:
            if len(com_offset_ranges) != 3 or any(
                len(axis_range) != 2 for axis_range in com_offset_ranges
            ):
                raise ValueError(
                    "com_offset_ranges must contain three two-value ranges"
                )
            maximum_offset = torch.tensor(
                [
                    max(abs(float(value)) for value in axis_range)
                    for axis_range in com_offset_ranges
                ],
                dtype=torch.float64,
            )
            # The COM offset is body-local, so its world direction changes
            # with torso orientation. Use a rotation-invariant conservative
            # bound for every world moment component.
            maximum_moment = (
                maximum_mass_delta
                * torch.linalg.vector_norm(gravity)
                * torch.linalg.vector_norm(maximum_offset)
            )
            torque_scale += maximum_moment
    return tuple(force_scale.tolist()) + tuple(torque_scale.tolist())


@dataclass(frozen=True)
class InstantaneousPushConfig:
    enabled: bool = True
    probability: float = 0.30
    interval_steps_min: int = 250
    interval_steps_max: int = 750
    planar_delta_v: tuple = (-1.20, 1.20)
    downward_delta_vz: tuple = (-0.50, 0.0)
    angular_delta_v: tuple = (-1.50, 1.50)


class InstantaneousPushes:
    def __init__(self, num_envs, device, dtype, config: InstantaneousPushConfig):
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        self.config = config
        self.actual_delta_world = torch.zeros(num_envs, 6, device=device, dtype=dtype)
        self.event_mask = torch.zeros(num_envs, 1, device=device, dtype=torch.bool)
        self.next_event_step = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.reset(torch.arange(num_envs, device=device))

    def _sample_interval(self, count: int) -> torch.Tensor:
        cfg = self.config
        if cfg.interval_steps_min <= 0 or cfg.interval_steps_min > cfg.interval_steps_max:
            raise ValueError("instantaneous-push interval bounds are invalid")
        return torch.randint(
            cfg.interval_steps_min, cfg.interval_steps_max + 1,
            (count,), device=self.device,
        )

    def sample(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.actual_delta_world.zero_()
        self.event_mask.zero_()
        if not self.config.enabled:
            return self.actual_delta_world, self.event_mask
        due = step >= self.next_event_step
        event = due & (torch.rand(self.num_envs, device=self.device) < self.config.probability)
        ids = due.nonzero(as_tuple=False).flatten()
        if ids.numel():
            self.next_event_step[ids] = step + self._sample_interval(ids.numel())
        event_ids = event.nonzero(as_tuple=False).flatten()
        if event_ids.numel():
            cfg = self.config
            self.actual_delta_world[event_ids, :2] = _uniform(
                (event_ids.numel(), 2), *cfg.planar_delta_v, device=self.device, dtype=self.dtype
            )
            # The vertical component is constrained to be non-positive.
            vz_low, vz_high = cfg.downward_delta_vz
            if vz_high > 0.0 or vz_low > vz_high:
                raise ValueError("downward push bounds must be ordered and non-positive")
            self.actual_delta_world[event_ids, 2:3] = _uniform(
                (event_ids.numel(), 1), vz_low, vz_high, device=self.device, dtype=self.dtype
            )
            self.actual_delta_world[event_ids, 3:] = _uniform(
                (event_ids.numel(), 3), *cfg.angular_delta_v, device=self.device, dtype=self.dtype
            )
            self.event_mask[event_ids] = True
        return self.actual_delta_world, self.event_mask

    def reset(self, env_ids: torch.Tensor, current_step: int = 0) -> None:
        self.actual_delta_world[env_ids] = 0.0
        self.event_mask[env_ids] = False
        self.next_event_step[env_ids] = current_step + self._sample_interval(env_ids.numel())


@dataclass(frozen=True)
class SustainedWrenchConfig:
    enabled: bool = True
    force_probability: float = 0.30
    torque_probability: float = 0.30
    interval_steps: tuple = (250, 750)
    duration_steps: tuple = (75, 250)
    force_interval_steps: tuple | None = None
    torque_interval_steps: tuple | None = None
    force_duration_steps: tuple | None = None
    torque_duration_steps: tuple | None = None
    ramp_fraction: float = 0.25
    force_bounds_n: tuple = (-60.0, 60.0)
    torque_bounds_nm: tuple = (-12.0, 12.0)
    force_normalizer_n: float = 60.0
    torque_normalizer_nm: float = 12.0


class SustainedBaseWrench:
    """Per-environment ramp-up, hold, ramp-down force/torque profiles."""

    def __init__(self, num_envs, device, dtype, config: SustainedWrenchConfig):
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        self.config = config
        self.current_world = torch.zeros(num_envs, 6, device=device, dtype=dtype)
        self.target_world = torch.zeros_like(self.current_world)
        self.component_active = torch.zeros(
            num_envs, 2, device=device, dtype=torch.bool
        )
        self.active_mask = torch.zeros(num_envs, 1, device=device, dtype=torch.bool)
        self.start_step = torch.zeros(num_envs, 2, device=device, dtype=torch.long)
        self.end_step = torch.zeros_like(self.start_step)
        self.duration = torch.ones_like(self.start_step)
        self.next_event_step = torch.zeros_like(self.start_step)
        self.reset(torch.arange(num_envs, device=device))

    def _rand_int(self, bounds, count):
        low, high = map(int, bounds)
        if low <= 0 or low > high:
            raise ValueError(f"invalid sustained-wrench bounds {bounds}")
        return torch.randint(low, high + 1, (count,), device=self.device)

    def _component_bounds(self, component: int):
        cfg = self.config
        if component == 0:
            return (
                cfg.force_interval_steps or cfg.interval_steps,
                cfg.force_duration_steps or cfg.duration_steps,
                cfg.force_probability,
                cfg.force_bounds_n,
                slice(0, 3),
            )
        return (
            cfg.torque_interval_steps or cfg.interval_steps,
            cfg.torque_duration_steps or cfg.duration_steps,
            cfg.torque_probability,
            cfg.torque_bounds_nm,
            slice(3, 6),
        )

    def _start_component(
        self, env_ids: torch.Tensor, step: int, component: int
    ) -> None:
        if not env_ids.numel():
            return
        interval_bounds, duration_bounds, probability, target_bounds, target_slice = (
            self._component_bounds(component)
        )
        count = env_ids.numel()
        duration = self._rand_int(duration_bounds, count)
        self.start_step[env_ids, component] = step
        self.duration[env_ids, component] = duration
        self.end_step[env_ids, component] = step + duration
        enabled = torch.rand(count, device=self.device) < probability
        targets = _uniform(
            (count, 3), *target_bounds, device=self.device, dtype=self.dtype
        ) * enabled.unsqueeze(-1)
        self.target_world[env_ids, target_slice] = targets
        self.component_active[env_ids, component] = enabled
        self.next_event_step[env_ids, component] = step + self._rand_int(
            interval_bounds, count
        )

    def step(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.current_world.zero_()
        if not self.config.enabled:
            self.active_mask.zero_()
            return self.current_world, self.active_mask
        running_components = torch.zeros_like(self.component_active)
        for component, target_slice in ((0, slice(0, 3)), (1, slice(3, 6))):
            due = (
                (step >= self.next_event_step[:, component])
                & (step >= self.end_step[:, component])
            )
            due_ids = due.nonzero(as_tuple=False).flatten()
            self._start_component(due_ids, step, component)
            running = (
                (step >= self.start_step[:, component])
                & (step < self.end_step[:, component])
                & self.component_active[:, component]
            )
            running_components[:, component] = running
            ids = running.nonzero(as_tuple=False).flatten()
            if ids.numel():
                elapsed = (step - self.start_step[ids, component]).to(self.dtype)
                duration = self.duration[ids, component].to(self.dtype).clamp_min(1.0)
                ramp = (duration * self.config.ramp_fraction).clamp_min(1.0)
                up = (elapsed / ramp).clamp(0.0, 1.0)
                down = ((duration - elapsed) / ramp).clamp(0.0, 1.0)
                amplitude = torch.minimum(up, down)
                self.current_world[ids, target_slice] = (
                    amplitude.unsqueeze(-1) * self.target_world[ids, target_slice]
                )
            finished = (
                (step >= self.end_step[:, component])
                & self.component_active[:, component]
            )
            self.component_active[finished, component] = False
            self.target_world[finished, target_slice] = 0.0
        self.active_mask.copy_(running_components.any(dim=-1, keepdim=True))
        return self.current_world, self.active_mask

    def yaw_local(self, base_quat_xyzw: torch.Tensor) -> torch.Tensor:
        force = world_to_yaw_local(self.current_world[:, :3], base_quat_xyzw)
        torque = world_to_yaw_local(self.current_world[:, 3:], base_quat_xyzw)
        return torch.cat((force, torque), dim=-1)

    def yaw_local_normalized(self, base_quat_xyzw: torch.Tensor) -> torch.Tensor:
        local = self.yaw_local(base_quat_xyzw)
        return torch.cat((
            local[:, :3] / max(float(self.config.force_normalizer_n), 1.0e-8),
            local[:, 3:] / max(float(self.config.torque_normalizer_nm), 1.0e-8),
        ), dim=-1)

    def reset(self, env_ids: torch.Tensor, current_step: int = 0) -> None:
        self.current_world[env_ids] = 0.0
        self.target_world[env_ids] = 0.0
        self.component_active[env_ids] = False
        self.active_mask[env_ids] = False
        self.start_step[env_ids] = current_step
        self.end_step[env_ids] = current_step
        self.duration[env_ids] = 1
        for component in range(2):
            interval_bounds = self._component_bounds(component)[0]
            self.next_event_step[env_ids, component] = current_step + self._rand_int(
                interval_bounds, env_ids.numel()
            )


def physics_transition_mask(reset, timeout, teleport, instantaneous_push) -> torch.Tensor:
    """Mask only discontinuous transitions; sustained wrenches remain valid."""
    return ~(reset.bool() | timeout.bool() | teleport.bool() | instantaneous_push.bool())
