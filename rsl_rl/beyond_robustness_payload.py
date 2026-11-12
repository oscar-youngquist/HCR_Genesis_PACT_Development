"""BR rigid-payload contract and tensor math (no simulator imports).

Dimensions use a seeded creation-time size bank across environments, before
collision cooking. Mass/friction/spawn samples persist for the per-env reset
cadence; a disturbance-range change makes them due on the next reset. SI units.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class SlidingPayloadConfig:
    tray_size: tuple = (.50, .34, .016)
    tray_height: float = .11  # floor centre relative to torso link origin
    wall_height: float = .045
    wall_thickness: float = .012
    tray_mass: float = .50
    tray_friction: float = 1.0  # multiply combine => effective mu = box mu
    box_size_min: tuple = (.10, .08, .06)
    box_size_max: tuple = (.14, .12, .10)
    size_bank_count: int = 16
    mass_initial: tuple = (.20, .40)
    mass_final: tuple = (1., 3.)
    friction_range: tuple = (.10, .60)
    displacement_initial: float = .01
    displacement_final: float = .10  # per-axis, limited by usable footprint
    velocity_initial: float = .0
    velocity_final: float = .40  # body-relative horizontal m/s
    spawn_clearance: float = .003
    loss_horizontal_tolerance: float = .08
    loss_below_floor_tolerance: float = .12
    seed: int = 0

    def __post_init__(self):
        for name in ('tray_size','box_size_min','box_size_max','mass_initial','mass_final','friction_range'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        if not isinstance(self.size_bank_count,int) or self.size_bank_count<1:
            raise ValueError('size_bank_count must be positive')
        for name in ('tray_size','box_size_min','box_size_max'):
            value=getattr(self,name)
            if len(value)!=3 or not all(math.isfinite(x) and x>0 for x in value):
                raise ValueError(f'{name} must contain three positive finite lengths')
        for lo, hi in zip(self.box_size_min, self.box_size_max):
            if not 0 < lo <= hi: raise ValueError('Positive ordered box sizes required')
        for name in ('mass_initial', 'mass_final', 'friction_range'):
            lo, hi = getattr(self, name)
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
                raise ValueError(f'Invalid {name}')
        if self.tray_mass <= 0 or self.tray_friction <= 0 or self.spawn_clearance < 0:
            raise ValueError('Invalid tray properties')
        for name in ('tray_height','wall_height','wall_thickness','tray_mass','tray_friction',
                     'displacement_initial','displacement_final','velocity_initial','velocity_final',
                     'spawn_clearance','loss_horizontal_tolerance','loss_below_floor_tolerance'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if any(b + 2*self.spawn_clearance >= t - 2*self.wall_thickness
               for b, t in zip(self.box_size_max[:2], self.tray_size[:2])):
            raise ValueError('Box must fit inside tray')

    def ranges(self, progress):
        if not math.isfinite(progress): raise ValueError('Nonfinite payload curriculum progress')
        p = min(1., max(0., float(progress)))
        return dict(mass=tuple(a+p*(b-a) for a,b in zip(self.mass_initial,self.mass_final)),
                    displacement=(0., self.displacement_initial+p*(self.displacement_final-self.displacement_initial)),
                    velocity=(0., self.velocity_initial+p*(self.velocity_final-self.velocity_initial)),
                    friction=self.friction_range)


def rotate_wxyz(q, v):
    """Body-to-world unit quaternion, batched vectors; does not mutate inputs."""
    xyz = q[..., 1:]
    t = 2*torch.cross(xyz, v, dim=-1)
    return v + q[..., :1]*t + torch.cross(xyz, t, dim=-1)


def inverse_rotate_wxyz(q, v):
    return rotate_wxyz(torch.cat((q[..., :1], -q[..., 1:]), -1), v)


def load_labels(q, base_origin, base_origin_velocity, angular_velocity_world,
                box_com, box_com_velocity, mass):
    r = inverse_rotate_wxyz(q, box_com-base_origin)
    omega = inverse_rotate_wxyz(q, angular_velocity_world)
    dr = inverse_rotate_wxyz(q, box_com_velocity-base_origin_velocity) - torch.cross(omega,r,dim=-1)
    return torch.cat((r, dr, mass.reshape(-1,1)), -1)


def spawn_state(q, base_origin, base_origin_velocity, angular_velocity_world, relative_pos, relative_vel):
    offset = rotate_wxyz(q, relative_pos)
    position = base_origin+offset
    # Use the realized float32 world lever arm, not its pre-addition value.
    # Otherwise large terrain coordinates introduce a spurious relative twist.
    offset = position-base_origin
    velocity = base_origin_velocity + torch.cross(angular_velocity_world,offset,dim=-1) + rotate_wxyz(q,relative_vel)
    return torch.cat((position,q,velocity,angular_velocity_world),-1)


def lost_payload(relative_pos, cfg):
    # Separation above the floor alone is NOT loss. Entire CoM outside an
    # expanded footprint or below the floor indicates escape/drop.
    footprint = relative_pos.new_tensor(cfg.tray_size[:2])/2 + cfg.loss_horizontal_tolerance
    return ((relative_pos[:,:2].abs()>footprint).any(-1) |
            (relative_pos[:,2] < cfg.tray_height-cfg.loss_below_floor_tolerance) |
            ~torch.isfinite(relative_pos).all(-1))


def termination_masks(failure, episode_steps, max_steps):
    timeout = (episode_steps >= max_steps) & ~failure
    return failure | timeout, timeout


def named_indices(available, required):
    if len(set(available)) != len(available): raise ValueError('Duplicate simulator names')
    missing = set(required)-set(available)
    if missing: raise ValueError(f'Missing simulator names: {sorted(missing)}')
    return [available.index(name) for name in required]


def cuboid_inertia(mass, size):
    size = torch.as_tensor(size, device=mass.device, dtype=mass.dtype)
    squared = size.square()
    return torch.diag_embed(mass[:,None]*(squared.sum(-1,keepdim=True)-squared)/12)


def payload_geometry(cfg, num_envs):
    """Seeded bank, assigned cyclically in canonical environment order.

    Collision dimensions are immutable for the scene lifetime. Using a bank
    avoids thousands of separately cooked shapes; mass/inertia remain per-env.
    """
    rng=torch.Generator().manual_seed(cfg.seed)
    bank=torch.tensor(cfg.box_size_min)+torch.rand(min(cfg.size_bank_count,num_envs),3,generator=rng)*(
        torch.tensor(cfg.box_size_max)-torch.tensor(cfg.box_size_min))
    ids=torch.arange(num_envs)%len(bank)
    return bank,ids
