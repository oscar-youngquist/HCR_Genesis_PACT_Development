"""One FACET disturbance source; outputs are held for exactly one control dt."""
from dataclasses import dataclass
import math
import torch


@dataclass
class FACETForceCfg:
    enabled: bool = True
    seed: int = 2718
    mixture: tuple = (.25, .25, .25, .25)  # none, EE, base, simultaneous
    cycle_seconds: float = 8.
    start_seconds: float = 2.
    ee_duration: float = 4.
    pulse_duration: tuple = (.4, .6)
    horizontal_peak: tuple = (40., 200.)
    vertical_peak: tuple = (0., 20.)
    # Equivalent equilibrium displacement, not independent +/-50 N samples.
    interior_offset: float = .04
    boundary_offset: float = .24
    ramp_control_steps: int = 3200*24


class FACETForces:
    def __init__(self, count, device, cfg):
        self.cfg = cfg
        values = (*cfg.mixture, cfg.cycle_seconds, cfg.start_seconds, cfg.ee_duration,
                  *cfg.pulse_duration, *cfg.horizontal_peak, *cfg.vertical_peak,
                  cfg.interior_offset, cfg.boundary_offset, cfg.ramp_control_steps)
        if not all(math.isfinite(x) for x in values):
            raise ValueError('FACET force settings must be finite')
        if len(cfg.mixture) != 4 or min(cfg.mixture) < 0 or sum(cfg.mixture) <= 0:
            raise ValueError('FACET force mixture must have four nonnegative weights')
        if not 0 <= cfg.start_seconds < cfg.cycle_seconds or not 0 < cfg.ee_duration <= cfg.cycle_seconds-cfg.start_seconds:
            raise ValueError('Invalid FACET force timing')
        for bounds in (cfg.pulse_duration, cfg.horizontal_peak, cfg.vertical_peak):
            if len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1]:
                raise ValueError('Invalid FACET pulse range')
        if cfg.pulse_duration[0] <= 0 or cfg.pulse_duration[1] > cfg.cycle_seconds-cfg.start_seconds:
            raise ValueError('FACET pulse must fit inside the cycle and have positive duration')
        if not 0 <= cfg.interior_offset <= cfg.boundary_offset or cfg.ramp_control_steps < 0:
            raise ValueError('Invalid FACET displacement curriculum')
        self.rng = torch.Generator(device=device).manual_seed(cfg.seed)
        self.weights = torch.tensor(cfg.mixture, device=device)
        self.family = torch.zeros(count, dtype=torch.long, device=device)
        self.cycle = torch.full((count,), -1, dtype=torch.long, device=device)
        self.direction = torch.zeros(count, 3, device=device)
        self.base_peak = torch.zeros_like(self.direction)
        self.duration = torch.ones(count, device=device)

    def reset(self, ids):
        self.cycle[ids] = -1

    @torch.no_grad()
    def sample(self, seconds, stiffness, control_steps):
        c = self.cfg
        cycle = torch.floor(seconds/c.cycle_seconds).long()
        ids = (cycle != self.cycle).nonzero().flatten()
        if len(ids):
            self.cycle[ids] = cycle[ids]
            self.family[ids] = torch.multinomial(self.weights, len(ids), replacement=True, generator=self.rng)
            r = torch.rand(len(ids), 8, device=seconds.device, generator=self.rng)
            direction = 2*r[:, :3]-1
            self.direction[ids] = direction/direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            signs = torch.where(r[:, 3:6] < .5, -1., 1.)
            magnitude = c.horizontal_peak[0]+(c.horizontal_peak[1]-c.horizontal_peak[0])*(2*r[:, 3:6]-1).abs()
            magnitude[:, 2] = c.vertical_peak[0]+(c.vertical_peak[1]-c.vertical_peak[0])*r[:, 6]
            self.base_peak[ids] = signs*magnitude
            self.duration[ids] = c.pulse_duration[0]+(c.pulse_duration[1]-c.pulse_duration[0])*r[:, 7]
        phase = seconds-cycle*c.cycle_seconds-c.start_seconds
        progress = min(max(control_steps/max(c.ramp_control_steps, 1), 0.), 1.)
        offset = c.interior_offset+(c.boundary_offset-c.interior_offset)*progress
        ee_active = ((self.family == 1) | (self.family == 3)) & (phase >= 0) & (phase < c.ee_duration)
        ee = self.direction*stiffness*offset*ee_active[:, None]
        triangle = (1-(2*phase/self.duration-1).abs()).clamp_min(0)
        base = self.base_peak*triangle[:, None]*((self.family == 2) | (self.family == 3))[:, None]
        return ee, base


class FACETExposure:
    """Compatibility surface for existing diagnostics, with no reward gate."""
    def __init__(self):
        self.iteration = 0
    def external_scale(self, iteration):
        return 1.
    def command_scale(self, iteration):
        return 0.
    def update(self, iteration, *unused):
        self.iteration = int(iteration)
    def metrics(self, iteration):
        return {'ForceCurriculum/external_scale': 1., 'ForceCurriculum/command_scale': 0.}
    def state_dict(self):
        return {'facet_force_version': 2, 'iteration': self.iteration}
    def load_state_dict(self, state):
        if state and state.get('facet_force_version') != 2:
            raise ValueError('Cannot resume rejection-force curriculum as FACET mixture')
        self.iteration = int((state or {}).get('iteration', 0))
