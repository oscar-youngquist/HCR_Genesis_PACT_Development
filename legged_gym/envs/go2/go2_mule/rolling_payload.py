"""MULE rolling-load geometry/sampling, SI units, no simulator imports.

Paper: four balls, 250 g tray, reset U[0,1] kg/ball, every 4 s U[0,2.5].
User-requested override: periodic maximum 2.0 kg per ball.
Geometry/materials, numerical mass floor, severity interpolation and initial
twists are explicit reproduction assumptions. Geometry is creation-time only.
"""
from dataclasses import dataclass
import math
import torch
from rsl_rl.beyond_robustness_payload import rotate_wxyz, inverse_rotate_wxyz, spawn_state


@dataclass(frozen=True)
class RollingPayloadConfig:
    mode: str = 'rolling'  # nominal creates NO sphere bodies; tray remains.
    seed: int = 1
    tray_size: tuple = (.50, .34, .016)
    tray_height: float = .11
    wall_height: float = .045
    wall_thickness: float = .012
    tray_mass: float = .250
    tray_friction: float = 1.
    radius_range: tuple = (.035, .045)
    size_bank_count: int = 8
    mass_floor_kg: float = .01
    reset_mass_range: tuple = (0., 1.)
    periodic_mass_range: tuple = (0., 2.0)
    light_mass_range: tuple = (.01, .05)
    mass_change_interval_s: float = 4.  # zero disables periodic changes
    friction_range: tuple = (.15, .6)
    light_friction_range: tuple = (.6, .6) # endpoint choice, not a monotonic difficulty claim
    restitution_range: tuple = (0., .1)
    spawn_clearance: float = .003
    spawn_jitter_m: float = .025
    relative_linear_velocity_max: float = .1
    relative_angular_velocity_max: float = 1.
    initial_severity: float = 1.
    loss_horizontal_margin: float = .08
    loss_below_floor_margin: float = .12

    def __post_init__(self):
        if self.mode not in ('nominal','rolling'): raise ValueError('Payload mode must be nominal or rolling')
        for key in ('tray_size','radius_range','reset_mass_range','periodic_mass_range',
                    'light_mass_range','friction_range','light_friction_range','restitution_range'):
            v=tuple(getattr(self,key)); object.__setattr__(self,key,v)
            if not all(math.isfinite(x) and x>=0 for x in v): raise ValueError('Invalid '+key)
            if len(v)!=(3 if key=='tray_size' else 2) or (key!='tray_size' and v[0]>v[1]): raise ValueError('Invalid '+key)
        for key in ('tray_height','wall_height','wall_thickness','tray_mass','tray_friction','mass_floor_kg',
                    'spawn_clearance','spawn_jitter_m','relative_linear_velocity_max','relative_angular_velocity_max',
                    'mass_change_interval_s','loss_horizontal_margin','loss_below_floor_margin','initial_severity'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<0: raise ValueError('Invalid '+key)
        if min(self.tray_size)<=0 or min(self.radius_range)<=0 or min(self.tray_mass,self.mass_floor_kg)<=0:
            raise ValueError('Positive geometry, tray mass and numerical floor required')
        if not 0<=self.initial_severity<=1 or self.restitution_range[1]>1: raise ValueError('Invalid severity/restitution')
        if max(self.periodic_mass_range[1],self.light_mass_range[1],self.mass_floor_kg)>2.:
            raise ValueError('Resampled ball mass must not exceed the requested 2.0 kg maximum')
        if self.mass_floor_kg>min(self.reset_mass_range[1],self.periodic_mass_range[1],self.light_mass_range[1]):
            raise ValueError('Mass floor exceeds configured mass range')
        if not isinstance(self.size_bank_count,int) or self.size_bank_count<1: raise ValueError('Invalid size bank')
        # Four disjoint rectangular cells guarantee non-overlap without rejection
        # sampling or per-env loops, even at the largest configured radius.
        if min(self.tray_size[:2])/4-self.wall_thickness-self.radius_range[1]-self.spawn_clearance<=0:
            raise ValueError('Four spheres do not fit the tray footprint')

    def mass_range(self, severity, *, periodic=False):
        if not math.isfinite(severity) or not 0<=severity<=1: raise ValueError('Severity must lie in [0,1]')
        target=self.periodic_mass_range if periodic else self.reset_mass_range
        return tuple(max(self.mass_floor_kg,a+severity*(b-a)) for a,b in zip(self.light_mass_range,target))

    def friction_at(self,severity):
        self.mass_range(severity) # validate progress
        return tuple(a+severity*(b-a) for a,b in zip(self.light_friction_range,self.friction_range))


def sphere_inertia(mass, radius):
    return torch.diag_embed((.4*mass*radius.square())[...,None].expand(*mass.shape,3))


def sphere_geometry(cfg, count):
    rng=torch.Generator().manual_seed(cfg.seed)
    bank=cfg.radius_range[0]+torch.rand(min(count,cfg.size_bank_count),4,generator=rng)*(cfg.radius_range[1]-cfg.radius_range[0])
    ids=torch.arange(count)%len(bank)
    return bank,ids


def sample_offsets(cfg, radii, generator, severity):
    centres=radii.new_tensor([[1,1],[1,-1],[-1,1],[-1,-1]])*radii.new_tensor(cfg.tray_size[:2])/4
    room=radii.new_tensor(cfg.tray_size[:2])/4-cfg.wall_thickness-radii[...,None]-cfg.spawn_clearance
    jitter=room.clamp(max=cfg.spawn_jitter_m*severity)
    xy=centres+(2*torch.rand((*radii.shape,2),device=radii.device,generator=generator)-1)*jitter
    z=cfg.tray_height+cfg.tray_size[2]/2+radii+cfg.spawn_clearance
    return torch.cat((xy,z[...,None]),-1)


def ball_spawn_state(q, pos, vel, omega, offsets, relative_vel, relative_omega):
    n,k=offsets.shape[:2]
    expand=lambda x:x[:,None].expand(-1,k,-1).reshape(n*k,-1)
    state=spawn_state(expand(q),expand(pos),expand(vel),expand(omega),offsets.reshape(-1,3),relative_vel.reshape(-1,3)).reshape(n,k,13)
    state[...,10:13]+=rotate_wxyz(q[:,None].expand(-1,k,-1),relative_omega)
    return state


def ball_motion(q,pos,vel,omega,state):
    q=q[:,None].expand(-1,4,-1)
    r=inverse_rotate_wxyz(q,state[...,:3]-pos[:,None])
    relative=inverse_rotate_wxyz(q,state[...,7:10]-vel[:,None])-torch.cross(
        inverse_rotate_wxyz(q,omega[:,None].expand(-1,4,-1)),r,dim=-1)
    return r,relative


def lost_balls(relative, radii, cfg):
    # Any sphere whose entire extent has left the expanded footprint, or fallen
    # below the floor, terminates. Upward motion within the tray is never loss.
    outside=(relative[...,:2].abs()-radii[...,None]>
             relative.new_tensor(cfg.tray_size[:2])/2+cfg.loss_horizontal_margin).any(-1)
    below=relative[...,2]+radii<cfg.tray_height-cfg.loss_below_floor_margin
    return (outside|below|~torch.isfinite(relative).all(-1)).any(-1)


def due_mass_changes(elapsed, dt, interval):
    """Per-env physical timer; no curriculum/iteration clock. Remainder preserved."""
    new=elapsed+dt
    if interval<=0: return torch.zeros_like(elapsed,dtype=torch.bool),new
    due=new>=interval-1e-6
    return due,torch.where(due,(new-interval).clamp_min(0).remainder(interval),new)
