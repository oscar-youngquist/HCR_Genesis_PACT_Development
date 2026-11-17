"""Detached FACET equations (8--9); SI units, world axes, no physical actuation."""
from dataclasses import dataclass, field
import math

import torch

from legged_gym.envs.b1z1.directional_compliance import rotate_yaw
from .bounds import restoring_force, guard, geometry
from .recovery_diagnostics import checked, SeedDiagnostics


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
    sampling_mode: str = 'fixed'  # 'fixed' (legacy sample_ranges) or 'upstream_b1z1'.
    # Optional {parameter: [low, high]}; independently uniform at reset only.
    sample_ranges: dict = field(default_factory=dict)
    max_substep_seconds: float = 0.005
    # Version 2 is a local bounded extension, not released FACET behavior.
    formulation_version: int = 1
    bounded: bool = False
    yaw_reaction: bool = False
    damping_ratio: float = 1.0
    radius: tuple = (.4, .9)
    pitch: tuple = (-math.pi/2.5, math.pi/3)
    azimuth: tuple = (-1.2, 1.2)
    center_xy: tuple = (.3, 0.)
    box_low: tuple = (-.9, -.2, -.7)
    box_high: tuple = (.1, .2, -.05)
    ground: float = -.7
    max_offset: float = .20
    boundary_margin: float = .04
    boundary_stiffness: float = 300.
    boundary_damping: float = 30.
    base_speed_limit: float = 2.
    yaw_rate_limit: float = 2.
    # Additional motion relative to the moving nominal, NOT nominal arm motion.
    compliant_offset_speed_limit: float = 1.
    compile_constraints: bool = True
    recovery_diagnostics: bool = True  # Reporting only; never a dynamics/eligibility switch.


PARAMETERS = ("base_mass", "ee_mass", "yaw_inertia", "base_stiffness",
              "base_damping", "ee_stiffness", "ee_damping", "yaw_stiffness",
              "yaw_damping", "alpha")


def coupled_impulse(relative_change_world, lever, parameters, yaw_reaction, normal_only=False):
    """Solve J M^-1 J^T impulse = relative velocity change (SI, world axes).

    J=[I,-I_xy,-z_cross_r]. Alpha scales the base reaction only. The diagonal
    plus rank-one inverse below is Sherman-Morrison, not an iterative solve.
    At alpha=1 equal/opposite planar impulses conserve virtual momentum.
    """
    p = parameters
    inverse_ee = torch.ones_like(relative_change_world)/p['ee_mass']
    diagonal = inverse_ee.clone()
    diagonal[..., :2] += p['alpha']/p['base_mass']
    c = torch.stack((-lever[..., 1], lever[..., 0], torch.zeros_like(lever[..., 0])), -1)
    if normal_only:
        # A frictionless scalar constraint uses n.T W n, not a three-axis
        # velocity servo. This avoids injecting tangential kinetic energy when
        # yaw makes W non-diagonal. Zero correction gives exactly zero impulse.
        length = relative_change_world.norm(dim=-1, keepdim=True)
        n = relative_change_world/length.clamp_min(1e-12)
        effective = (diagonal*n*n).sum(-1, keepdim=True)
        if yaw_reaction:
            effective += p['alpha']/p['yaw_inertia']*(c*n).sum(-1, keepdim=True).square()
        return n*length/effective.clamp_min(1e-12)
    solution = relative_change_world/diagonal
    if yaw_reaction:
        dc = c/diagonal
        scale = p['alpha']/p['yaw_inertia']
        solution = solution-dc*scale*(c*solution).sum(-1, keepdim=True)/(1+scale*(c*dc).sum(-1, keepdim=True))
    return solution


def ee_setpoint(base_xy, base_velocity_xy, yaw, yaw_rate, local, local_velocity, height):
    """Mount-inclusive local XY, terrain-referenced world height; no base-Z transport."""
    # Broadcast views, not repeated copies of each robot's interval packet.
    offset = rotate_yaw(local.expand(*yaw.shape, 3), yaw)
    velocity = rotate_yaw(local_velocity.expand(*yaw.shape, 3), yaw)
    position = offset + torch.cat((base_xy, height.expand_as(yaw)[..., None]), -1)
    transport = torch.stack((-offset[..., 1], offset[..., 0], torch.zeros_like(yaw)), -1)
    velocity = velocity + torch.cat((base_velocity_xy, torch.zeros_like(base_velocity_xy[..., :1])), -1)
    return position, velocity + yaw_rate[..., None] * transport


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
        mode = getattr(cfg, 'sampling_mode', 'fixed')
        if mode not in ('fixed', 'upstream_b1z1', 'bounded_b1z1'):
            raise ValueError('Unknown FACET impedance sampling mode')
        if mode == 'upstream_b1z1' and cfg.sample_ranges:
            raise ValueError('upstream_b1z1 preserves correlated sampling; use fixed for sample_ranges')
        ranges = dict(cfg.sample_ranges)
        if mode == 'bounded_b1z1':
            ranges.update(base_mass=(2., 8.), ee_mass=(2., 2.), yaw_inertia=(1.5, 1.5),
                base_stiffness=(2., 30.), yaw_stiffness=(2., 30.), ee_stiffness=(60., 120.),
                base_damping=(2*math.sqrt(4), 2*math.sqrt(240)),
                yaw_damping=(2*math.sqrt(3), 2*math.sqrt(45)),
                ee_damping=(2*math.sqrt(120), 2*math.sqrt(240)))
            if cfg.sample_ranges:
                raise ValueError('bounded_b1z1 uses correlated impedance sampling')
        if mode == 'upstream_b1z1':
            # Pinned impedance_manip.py: randi(1,4) is upper-exclusive;
            # virtual_mass_range is stored but not consumed by its kernels.
            ranges.update(base_mass=(2., 8.), ee_mass=(2., 2.), yaw_inertia=(1.5, 1.5),
                base_stiffness=(2., 30.), yaw_stiffness=(2., 30.), ee_stiffness=(4., 30.),
                base_damping=(1.8*math.sqrt(2), 1.8*math.sqrt(30)),
                yaw_damping=(1.8*math.sqrt(2), 1.8*math.sqrt(30)),
                ee_damping=(3.6, 1.8*math.sqrt(30)))
        unknown = set(cfg.sample_ranges) - set(PARAMETERS)
        if unknown:
            raise ValueError(f"Unknown FACET parameters: {sorted(unknown)}")
        self._ranges = {}
        for name in PARAMETERS:
            lo, hi = ranges.get(name, (getattr(cfg, name),) * 2)
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
        if cfg.bounded:
            if any(not math.isfinite(x) or x <= 0 for x in (cfg.max_offset, cfg.boundary_margin,
                    cfg.base_speed_limit, cfg.yaw_rate_limit, cfg.compliant_offset_speed_limit, cfg.damping_ratio)):
                raise ValueError('FACET bounds/speeds/damping must be positive')
            if any(len(b) != 2 or b[0] >= b[1] for b in (cfg.radius, cfg.pitch, cfg.azimuth)):
                raise ValueError('Invalid FACET workspace bounds')
            if any(not math.isfinite(x) or x < 0 for x in (cfg.boundary_stiffness, cfg.boundary_damping)):
                raise ValueError('FACET boundary spring/damping must be nonnegative')
            if cfg.radius[0] <= 0 or not (-math.pi/2 < cfg.pitch[0] < cfg.pitch[1] < math.pi/2):
                raise ValueError('FACET spherical workspace requires positive radius and nonpolar pitch')
            # RK4's negative-real stability interval reaches ~2.78. Use 1.0
            # with a three-face local bound, rather than treating nine mutually
            # exclusive surfaces as simultaneously active critically damped axes.
            rate = 2*math.sqrt((stiffness+3*cfg.boundary_stiffness)/mass)+(damping+3*cfg.boundary_damping)/mass
        self._max_dt = min(cfg.max_substep_seconds, (1.0 if cfg.bounded else .5) / max(rate, 1e-12))
        self._diagnostics = self._state.new_zeros(count, 15)
        self._elapsed = self._state.new_zeros(count)
        self._guard_impulse = self._state.new_zeros(count, 3)
        self._guard_invalid = torch.zeros(count, device=device, dtype=torch.bool)
        self._offset = self._state.new_zeros(count, 3)
        self._relative_speed = self._state.new_zeros(count)
        self._boundary_active = torch.zeros(count, device=device, dtype=torch.bool)
        self._last_constraints = {}
        self._integration_times = {}
        # Fixed-shape fusion avoids hundreds of small geometry kernels per bank.
        self._constraint_step = self._constrain
        self._dynamics_step = self._rhs
        self._checked_step = self._checked
        if cfg.bounded and cfg.compile_constraints and self._state.is_cuda:
            self._constraint_step = torch.compile(self._constrain, fullgraph=True, dynamic=False)
            self._dynamics_step = torch.compile(self._rhs, fullgraph=True, dynamic=False)
            if cfg.recovery_diagnostics:
                self._checked_step = torch.compile(self._checked, fullgraph=True, dynamic=False)

    @torch.no_grad()
    def reset(self, ids, base_position, base_velocity, yaw, yaw_rate, ee_position, ee_velocity):
        """Measured state arrays have len(ids) rows; untouched rows retain state/parameters."""
        self._state[ids] = torch.cat((base_position[:, :2], base_velocity[:, :2],
            yaw[:, None], yaw_rate[:, None], ee_position, ee_velocity), -1)
        for name, (lo, hi) in self._ranges.items():
            values = lo if lo == hi else lo + (hi-lo) * torch.rand(
                (len(ids), 1), device=self._state.device, dtype=self._state.dtype, generator=self._rng)
            self._parameters[name][ids] = values
        if getattr(self.cfg, 'sampling_mode', 'fixed') == 'upstream_b1z1':
            p = self._parameters
            exponent = torch.randint(1, 4, (len(ids), 1), device=self._state.device, generator=self._rng)
            p['base_mass'][ids] = 2.**exponent.to(self._state.dtype)
            p['base_damping'][ids] = 1.8*p['base_stiffness'][ids].sqrt()
            p['yaw_stiffness'][ids] = p['base_stiffness'][ids]
            p['yaw_damping'][ids] = p['base_damping'][ids]
            p['ee_damping'][ids] = 1.8*p['ee_stiffness'][ids].sqrt()
        if self.cfg.sampling_mode == 'bounded_b1z1':
            p = self._parameters
            exponent = torch.randint(1, 4, (len(ids), 1), device=self._state.device, generator=self._rng)
            p['base_mass'][ids] = 2.**exponent.to(self._state.dtype)
            p['yaw_stiffness'][ids] = p['base_stiffness'][ids]
            for body, mass in (('base', 'base_mass'), ('ee', 'ee_mass'), ('yaw', 'yaw_inertia')):
                p[body+'_damping'][ids] = 2*self.cfg.damping_ratio*(p[mass][ids]*p[body+'_stiffness'][ids]).sqrt()
        self._diagnostics[ids] = 0
        self._elapsed[ids] = 0
        self._guard_impulse[ids] = 0
        self._guard_invalid[ids] = False
        self._offset[ids] = 0
        self._relative_speed[ids] = 0
        self._boundary_active[ids] = False
        self._initialized[ids] = True
        self._ready = bool(self._initialized.all())  # Reset-time check, no per-step GPU synchronization.

    def _rhs(self, state, time, inputs, loads, parameters=None, diagnostics=True):
        p = self._parameters if parameters is None else parameters
        ee_force, base_force, yaw_moment = loads
        measured_xy, measured_yaw, command_world, yaw_command, local, local_rate, height = inputs
        xy, velocity, yaw, rate = state[..., :2], state[..., 2:4], state[..., 4], state[..., 5]
        # First-order hold of the sampled nominal trajectory over one control dt.
        target, target_velocity = ee_setpoint(xy, velocity, yaw, rate,
            local + time * local_rate, local_rate, height)
        spring_ee = p["ee_stiffness"] * (target-state[..., 6:9]) + p["ee_damping"] * (target_velocity-state[..., 9:12])
        if self.cfg.bounded:
            q, nominal, relative_velocity = self._relative(state, time, inputs)
            spring_ee = spring_ee + rotate_yaw(restoring_force(q, nominal, relative_velocity, self.cfg, inputs[5]), yaw)
        spring_base = p["base_stiffness"]*(measured_xy-xy) + p["base_damping"]*(command_world-velocity)
        reaction = -p["alpha"]*spring_ee[..., :2]
        yaw_error = torch.atan2(torch.sin(measured_yaw-yaw), torch.cos(measured_yaw-yaw))
        spring_yaw = p["yaw_stiffness"][..., 0]*yaw_error + p["yaw_damping"][..., 0]*(yaw_command-rate)
        if self.cfg.yaw_reaction:
            # Same reference-EE application point as the constraint Jacobian.
            # Only internal reaction is transmitted, never the external EE force.
            lever = state[..., 6:8]-xy
            spring_yaw = spring_yaw+lever[..., 0]*reaction[..., 1]-lever[..., 1]*reaction[..., 0]
        derivative = torch.cat((velocity, (spring_base+base_force[..., :2]+reaction)/p["base_mass"],
            rate[..., None], ((spring_yaw+yaw_moment)/p["yaw_inertia"][..., 0])[..., None],
            state[..., 9:12], (spring_ee+ee_force)/p["ee_mass"]), -1)
        if not diagnostics:
            return derivative, None
        # Interval-average force components below use the same RK quadrature.
        forces = torch.cat((spring_base, spring_ee, reaction, base_force,
                            spring_yaw[..., None], yaw_moment[..., None], ee_force), -1)
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
        self._state, self._diagnostics = self.integrate(self._state, dt, inputs, loads)
        if self.cfg.bounded:
            q, nominal, v = self._relative(self._state, dt, inputs)
            self._offset.copy_(q-nominal)
            self._relative_speed.copy_(v.norm(dim=-1))
            self._boundary_active.copy_((geometry(q, nominal, self.cfg)[0] < self.cfg.boundary_margin).any(-1))
        self._elapsed.add_(dt)

    @torch.no_grad()
    def integrate(self, state, dt, inputs, loads, parameters=None, diagnostics=True):
        """Same RK4 for one start or a broadcast robot/start bank; no sampling."""
        steps = max(1, math.ceil(dt / self._max_dt))
        h = dt / steps
        # Tensor times avoid one compiled specialization per RK stage. Cache
        # device scalars once per control dt, not a CPU->GPU transfer per stage.
        times = None
        if self.cfg.bounded and self.cfg.compile_constraints and state.is_cuda:
            key = (dt, steps)
            if key not in self._integration_times:
                self._integration_times[key] = tuple((torch.arange(2*steps+1, device=state.device, dtype=state.dtype)*(h/2)).unbind())
            times = self._integration_times[key]
        forces = torch.zeros_like(self._diagnostics) if diagnostics else None
        detailed = self.cfg.bounded and self.cfg.recovery_diagnostics and not diagnostics
        if detailed:
            any_flags = torch.zeros((*state.shape[:-1], 6), device=state.device, dtype=torch.bool)
        if self.cfg.bounded:
            initial_q, initial_nominal, _ = self._relative(state, 0., inputs)
            recovery = (geometry(initial_q, initial_nominal, self.cfg)[0] < -1e-4).any(-1)
            total_impulse = torch.zeros_like(state[..., 9:12])
            saturation_impulse = torch.zeros_like(total_impulse)
            incompatible = torch.zeros_like(state[..., 4], dtype=torch.bool)
            saturation = torch.zeros_like(incompatible)
        for i in range(steps):  # Fixed batch-wide numerical substeps, never per-robot loops.
            s = state
            t, middle, end = (i*h, (i+.5)*h, (i+1)*h) if times is None else times[2*i:2*i+3]
            k1, f1 = self._dynamics_step(s, t, inputs, loads, parameters, diagnostics)
            k2, f2 = self._dynamics_step(s+h*k1/2, middle, inputs, loads, parameters, diagnostics)
            k3, f3 = self._dynamics_step(s+h*k2/2, middle, inputs, loads, parameters, diagnostics)
            k4, f4 = self._dynamics_step(s+h*k3, end, inputs, loads, parameters, diagnostics)
            state = s+h*(k1+2*k2+2*k3+k4)/6
            if self.cfg.bounded:
                # Speed saturation is dissipative, not an internal reaction.
                saturation |= ((state[..., 2:4].norm(dim=-1) > self.cfg.base_speed_limit)
                    | (state[..., 5].abs() > self.cfg.yaw_rate_limit))
                incoming = state
                state, impulse, invalid = self._constraint_step(s, state, end, h, inputs, parameters)
                p = self._parameters if parameters is None else parameters
                removed = p['ee_mass']*(state[..., 9:12]-incoming[..., 9:12])-impulse
                removed[..., :2] += p['base_mass']*(state[..., 2:4]-incoming[..., 2:4])+p['alpha']*impulse[..., :2]
                saturation_impulse += removed
                saturation |= removed.norm(dim=-1) > 1e-4
                total_impulse += impulse
                incompatible |= invalid
                if detailed:
                    checked_end = self._checked_step(state, end, inputs)
                    any_flags |= checked_end[0]
                if diagnostics:
                    if i == 0:
                        self._guard_impulse.zero_()
                        self._guard_invalid.zero_()
                    self._guard_impulse.add_(impulse)
                    self._guard_invalid.logical_or_(invalid)
            if diagnostics:
                forces.add_((f1+2*f2+2*f3+f4)/(6*steps))
        if self.cfg.bounded:
            q, nominal, velocity = self._relative(state, dt, inputs)
            self._last_constraints = dict(
                boundary_active=(geometry(q, nominal, self.cfg)[0] < self.cfg.boundary_margin).any(-1),
                valid=~incompatible, incompatible=incompatible,
                recovery=recovery, speed_saturation=saturation,
                correction_impulse_Ns=total_impulse.norm(dim=-1),
                saturation_impulse_Ns=saturation_impulse.norm(dim=-1),
                accepted_offset_m=(q-nominal).norm(dim=-1),
                compliant_offset_speed_mps=(velocity-inputs[5]).norm(dim=-1),
                arm_relative_speed_mps=velocity.norm(dim=-1),
                ee_world_speed_mps=state[..., 9:12].norm(dim=-1),
                base_speed_mps=state[..., 2:4].norm(dim=-1), yaw_rate_rps=state[..., 5].abs())
        if detailed:
            self._last_checked = checked_end, any_flags
        return state, forces

    def _checked(self, state, time, inputs):
        return checked(self, state, time, inputs)

    def _relative(self, state, time, inputs):
        yaw = state[..., 4]
        origin = torch.cat((state[..., :2], inputs[6].expand_as(yaw)[..., None]), -1)
        offset = state[..., 6:9]-origin
        transport = state[..., 5, None]*torch.stack((-offset[..., 1], offset[..., 0], torch.zeros_like(yaw)), -1)
        vb = torch.cat((state[..., 2:4], torch.zeros_like(yaw[..., None])), -1)
        center = state.new_tensor((*self.cfg.center_xy, 0.))
        return (rotate_yaw(offset, -yaw)-center,
                inputs[4]+time*inputs[5]-center,
                rotate_yaw(state[..., 9:12]-vb-transport, -yaw))

    def _constrain(self, previous, state, time, dt, inputs, parameters):
        p = self._parameters if parameters is None else parameters
        cfg = self.cfg
        state = state.clone()
        old_q, _, _ = self._relative(previous, time-dt, inputs)
        impulse = torch.zeros_like(state[..., 9:12])
        # Semi-implicit accepted positions use the same corrected velocities.
        # Fixed geometric sweeps handle curved/intersecting surfaces; they are
        # not IK. No projection teleports an infeasible measured history seed.
        for _ in range(4):
            v = state[..., 2:4]
            state[..., 2:4] = v*(cfg.base_speed_limit/v.norm(dim=-1, keepdim=True).clamp_min(cfg.base_speed_limit))
            state[..., 5] = state[..., 5].clamp(-cfg.yaw_rate_limit, cfg.yaw_rate_limit)
            state[..., :2] = previous[..., :2]+dt*state[..., 2:4]
            state[..., 4] = previous[..., 4]+dt*state[..., 5]
            state[..., 6:9] = previous[..., 6:9]+dt*state[..., 9:12]
            q, nominal, velocity = self._relative(state, time, inputs)
            limited_q, _, _ = guard(q, nominal, velocity, old_q, cfg, inputs[5], refinements=1, project_velocity=False)
            desired = velocity+(limited_q-q)/dt
            # The moving nominal is not slowed by this compliance-rate limit.
            delta = desired-inputs[5]
            desired = inputs[5]+delta*(cfg.compliant_offset_speed_limit/delta.norm(dim=-1, keepdim=True).clamp_min(cfg.compliant_offset_speed_limit))
            change = rotate_yaw(desired-velocity, state[..., 4])
            lever = state[..., 6:9]-torch.cat((state[..., :2], inputs[6].expand_as(state[..., 4])[..., None]), -1)
            correction = coupled_impulse(change, lever, p, cfg.yaw_reaction, normal_only=True)
            state[..., 9:12] += correction/p['ee_mass']
            state[..., 2:4] -= p['alpha']*correction[..., :2]/p['base_mass']
            if cfg.yaw_reaction:
                state[..., 5] -= p['alpha'][..., 0]*torch.linalg.cross(lever, correction)[..., 2]/p['yaw_inertia'][..., 0]
            impulse += correction
        # Do not reconstruct EE velocity from the changed base velocity: that
        # would apply the same internal momentum exchange a second time.
        v = state[..., 2:4]
        state[..., 2:4] = v*(cfg.base_speed_limit/v.norm(dim=-1, keepdim=True).clamp_min(cfg.base_speed_limit))
        state[..., 5] = state[..., 5].clamp(-cfg.yaw_rate_limit, cfg.yaw_rate_limit)
        state[..., :2] = previous[..., :2]+dt*state[..., 2:4]
        state[..., 4] = previous[..., 4]+dt*state[..., 5]
        state[..., 6:9] = previous[..., 6:9]+dt*state[..., 9:12]
        # Final rate saturation is explicitly non-internal. Account for its
        # changed endpoint lever: d(relative v)=(I-dt*omega*Z)d(world v).
        # This exact 2x2 inverse keeps both the speed bound and x+=dt*v true.
        _, _, velocity = self._relative(state, time, inputs)
        delta = velocity-inputs[5]
        desired = inputs[5]+delta*(cfg.compliant_offset_speed_limit/delta.norm(dim=-1, keepdim=True).clamp_min(cfg.compliant_offset_speed_limit))
        dv = rotate_yaw(desired-velocity, state[..., 4])
        w = dt*state[..., 5]
        correction = torch.stack(((dv[..., 0]-w*dv[..., 1])/(1+w*w),
                                  (dv[..., 1]+w*dv[..., 0])/(1+w*w), dv[..., 2]), -1)
        state[..., 9:12] += correction
        state[..., 6:9] = previous[..., 6:9]+dt*state[..., 9:12]
        q, nominal, velocity = self._relative(state, time, inputs)
        invalid = ((geometry(q, nominal, cfg)[0] < -1e-4).any(-1)
            | ((velocity-inputs[5]).norm(dim=-1) > cfg.compliant_offset_speed_limit+1e-4)
            | (state[..., 2:4].norm(dim=-1) > cfg.base_speed_limit+1e-4)
            | (state[..., 5].abs() > cfg.yaw_rate_limit+1e-4))
        return state, impulse, invalid

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
        values['guard_impulse_world'] = self._guard_impulse
        values['guard_incompatible'] = self._guard_invalid
        values['ee_offset_task'] = self._offset
        values['ee_relative_speed'] = self._relative_speed
        values['boundary_active'] = self._boundary_active
        return {name: value.detach().clone() for name, value in values.items()}


def measured_reference_state(base_position, base_velocity, yaw, yaw_rate, ee_position, ee_velocity):
    """Pack a measured boundary state in the integrator's world-frame ordering."""
    return torch.cat((base_position[..., :2], base_velocity[..., :2], yaw[..., None],
                      yaw_rate[..., None], ee_position, ee_velocity), -1).detach()


class FACETReferenceBank:
    """Ring of measured starts ending at one common time, not future predictions.

    A seed at X[n] first has age 1 after integrating packet n -> n+1. The
    age-h target therefore starts at X[n+1-h]. No historical packet is replayed.
    """
    def __init__(self, reference, horizons=(8, 16, 32)):
        if not horizons or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons) or len(set(horizons)) != len(horizons):
            raise ValueError("FACET horizons must be distinct positive control-step ages")
        self.reference = reference
        self.horizons = tuple(horizons)
        self.capacity = max(horizons)
        self._horizons = torch.tensor(horizons, device=reference._state.device)
        self._state = reference._state.new_zeros(len(reference._state), self.capacity, 12)
        self._ages = torch.full(self._state.shape[:2], -1, device=self._state.device, dtype=torch.long)
        # Views share the robot's episode parameters, including partial resamples.
        self._parameters = {key: value[:, None] for key, value in reference._parameters.items()}
        self._cursor = 0
        self._constraints = {}
        self._seed_diagnostics = (SeedDiagnostics(self._state)
            if reference.cfg.bounded and reference.cfg.recovery_diagnostics else None)
        self._snapshot = self._select()

    @torch.no_grad()
    def clear(self, ids):
        if self._seed_diagnostics is not None:
            self._seed_diagnostics.retire(ids, self._ages[ids] >= 0, 1)
            for value in self._seed_diagnostics.data.values():
                value[ids] = 0
        self._state[ids] = 0
        self._ages[ids] = -1
        for value in self._constraints.values():
            value[ids] = 0
        # Do not clear the completed-transition snapshot: terminal diagnostics
        # can read it after the parent has initialized the next physical episode.

    def _select(self):
        columns = (self._cursor-self._horizons) % self.capacity
        ages = self._ages.index_select(1, columns)
        result = dict(targets=self._state.index_select(1, columns).detach(), ages=ages,
                      valid=ages == self._horizons[None])
        # Age eligibility stays independent of constraint validity: invalid
        # targets remain counted in diagnostics and retain existing reward use.
        result.update({'constraint_'+key: value.index_select(1, columns).detach()
                       for key, value in self._constraints.items()})
        if self._seed_diagnostics is not None:
            result.update({'diagnostic_'+key: value.index_select(1, columns).detach()
                           for key, value in self._seed_diagnostics.data.items()})
        return result

    @torch.no_grad()
    def advance(self, seed, dt, inputs, loads, applied_torque_world=None):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("FACET control dt must be positive")
        if not self.reference._ready:
            raise RuntimeError("Initialize episode impedance before advancing histories")
        if self._seed_diagnostics is not None:
            self._seed_diagnostics.retire((slice(None), self._cursor), self._ages[:, self._cursor] >= 0, 0)
            self._seed_diagnostics.seed(self._cursor, self.reference._checked_step(seed, 0., inputs))
        self._state[:, self._cursor] = seed
        self._ages[:, self._cursor] = 0
        active = self._ages >= 0
        state, _ = self.reference.integrate(self._state, dt, tuple(v[:, None] for v in inputs),
            tuple(v[:, None] for v in loads), self._parameters, diagnostics=False)
        # Fixed-shape batched arithmetic includes unused slots, but they never
        # advance or become valid. No full-bank roll/copy to insert a new seed.
        self._state = torch.where(active[..., None], state, 0.)
        self._constraints = {key: torch.where(active, value, 0).detach()
                             for key, value in self.reference._last_constraints.items()}
        if self._seed_diagnostics is not None:
            # Actual held forces/couples, including injected loads. Once loaded,
            # a trajectory remains in the loaded group through force removal.
            loaded = (loads[0].norm(dim=-1) > 1e-6) | (loads[1].norm(dim=-1) > 1e-6) | (loads[2].abs() > 1e-6)
            if applied_torque_world is not None:
                loaded |= applied_torque_world.norm(dim=-1) > 1e-6
            torque_norm = loads[2].abs() if applied_torque_world is None else applied_torque_world.norm(dim=-1)
            self._seed_diagnostics.exposure += torch.stack((torch.ones_like(torque_norm), loaded,
                loads[0].norm(dim=-1), loads[1].norm(dim=-1), torque_norm), -1).sum(0)*dt
            self._seed_diagnostics.update(active, dt, *self.reference._last_checked, loaded)
        self._ages.add_(active)
        self._cursor = (self._cursor+1) % self.capacity
        self._snapshot = self._select()  # Only requested ages are copied, once per interval.

    def read(self):
        return {key: value.detach().clone() for key, value in self._snapshot.items()}

    def read_current(self):
        """Current-episode view for observations; cleared rows never expose terminal targets."""
        return {key: value.detach().clone() for key, value in self._select().items()}
