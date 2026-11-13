"""Go2 MULE v2 tensor contracts. No simulator/HardPACT observation packing.

Body axes: x forward, y left, z up. FR/FL/RR/RL, hip/thigh/calf.
Physical inputs are scaled only by pack(); stored actor/history tensors are
already scaled. Previous action is the clipped combined *issued* action,
before delay and before position scaling, zero on reset (not a raw sample).
"""
from dataclasses import asdict, dataclass
import math
import torch


FEET = ('FR', 'FL', 'RR', 'RL')
JOINTS = tuple(f'{leg}_{joint}_joint' for leg in FEET for joint in ('hip', 'thigh', 'calf'))


@dataclass(frozen=True)
class MULESchema:
    version: int = 2
    history_length: int = 20
    terrain_points_x: tuple = ()
    terrain_points_y: tuple = ()
    joint_order: tuple = JOINTS
    angular_velocity_scale: float = 1.
    linear_velocity_scale: float = 1.
    joint_position_scale: float = 1.
    joint_velocity_scale: float = 1.
    terrain_height_scale: float = 1.
    # Independent proxy/critic force normalization, not a HardPACT GRF head.
    force_scale: float = .01
    clip_observations: float = 100.

    def __post_init__(self):
        for key in ('terrain_points_x', 'terrain_points_y', 'joint_order'):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if self.version != 2 or self.joint_order != JOINTS:
            raise ValueError('Incompatible MULE schema/joint order')
        if not isinstance(self.history_length, int) or self.history_length < 1:
            raise ValueError('history_length must be positive')
        for key in ('angular_velocity_scale', 'linear_velocity_scale', 'joint_position_scale',
                    'joint_velocity_scale', 'terrain_height_scale', 'force_scale', 'clip_observations'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not all(math.isfinite(x) for x in self.terrain_points_x + self.terrain_points_y):
            raise ValueError('Terrain coordinates must be finite')

    @property
    def observation_dim(self): return 45
    @property
    def terrain_dim(self): return len(self.terrain_points_x) * len(self.terrain_points_y)
    @property
    def critic_dim(self): return 45 + 3 + 3 + self.terrain_dim
    @property
    def history_dim(self): return 45 * self.history_length
    @property
    def adaptive_history_dim(self): return 57 * self.history_length

    def fields(self, group):
        # (name, width, physical units, multiplier); q is ABSOLUTE, not q-q0.
        obs = (('angular_velocity_B', 3, 'rad/s', self.angular_velocity_scale),
               ('gravity_B', 3, 'unit vector', 1.),
               ('command_B', 3, 'm/s,m/s,rad/s', (self.linear_velocity_scale,)*2+(self.angular_velocity_scale,)),
               ('joint_position', 12, 'rad absolute', self.joint_position_scale),
               ('joint_velocity', 12, 'rad/s', self.joint_velocity_scale),
               ('previous_action', 12, 'clipped combined policy units', 1.))
        if group == 'actor': return obs
        if group == 'critic':
            # DreamWaQ full state [o,v,disturbance force,terrain]; no arbitrary payload fields.
            extra = (('linear_velocity_B', 3, 'm/s', self.linear_velocity_scale),
                     ('disturbance_force_B', 3, 'N at torso origin', self.force_scale))
            terrain = (('terrain_heights', self.terrain_dim, 'm base_z-ground_z; x-major', self.terrain_height_scale),) if self.terrain_dim else ()
            return obs + extra + terrain
        raise ValueError(f'Unknown MULE field group: {group}')

    def pack(self, group, physical):
        fields = self.fields(group)
        if set(physical) != {f[0] for f in fields}:
            raise ValueError(f'{group}: missing/extra fields; privileged actor inputs are forbidden')
        batch = physical[fields[0][0]].shape[0]
        result = []
        for name, width, _, scale in fields:
            x = physical[name]
            if x.shape != (batch, width) or not x.is_floating_point():
                raise ValueError(f'{name}: expected floating [{batch},{width}]')
            result.append(x.detach() * x.new_tensor(scale))
        return torch.cat(result, -1)

    def contract(self):
        self.__post_init__()  # Also reject old unpickled objects before use.
        return dict(**asdict(self), actor_fields=self.fields('actor'), critic_fields=self.fields('critic'),
                    history='H frames oldest first including current; zero preceding frames on reset; append once per action',
                    auxiliary='velocity_B(t) physical m/s; packed next observation(t+1); same episode only',
                    previous_action='clipped combined issued command before action scale/delay; reset zero',
                    normalization='actor: fixed scales then cached noise then +/-clip_observations before history and next-observation targets; critic and physical labels remain clean/unclipped; no running statistics',
                    force='pinv(J_leg_B.T) @ applied_tau; signed endpoint-force proxy, not simulator GRF')


@dataclass(frozen=True)
class FootForceEstimate:
    force_b: torch.Tensor  # [N,4,3], N, torque-equivalent endpoint force in body axes
    valid: torch.Tensor    # [N,4]; invalid/unobservable feet have zero proxy


@torch.no_grad()
def estimate_foot_forces(applied_torque_nm, leg_jacobians_b, rtol=1e-5):
    """Deployable inputs only: measured/applied torque and named leg kinematics.

    J = d(foot position in B)/dq_leg [m/rad], holding the base fixed.
    Implements MULE's positive tau=J.T*f convention literally. In quasistatic
    balance the environment-on-robot reaction has the opposite sign (ignoring
    leg gravity/inertia). This is NOT a dynamically corrected GRF observer.
    Nonfinite/rank-deficient inputs are invalid, not silently valid zero force.
    """
    if applied_torque_nm.ndim != 2 or applied_torque_nm.shape[1] != 12:
        raise ValueError('Expected torque [N,12]')
    if leg_jacobians_b.shape != (applied_torque_nm.shape[0], 4, 3, 3):
        raise ValueError('Expected canonical body-frame leg Jacobians [N,4,3,3]')
    if not math.isfinite(rtol) or rtol <= 0: raise ValueError('rtol must be positive')
    tau = applied_torque_nm.detach().reshape(-1, 4, 3)
    jt = leg_jacobians_b.detach().transpose(-1, -2)
    finite = torch.isfinite(jt).all((-1, -2)) & torch.isfinite(tau).all(-1)
    jt = torch.where(finite[..., None, None], jt, torch.eye(3, device=jt.device, dtype=jt.dtype))
    tau = torch.where(finite[..., None], tau, 0.)
    u, s, vh = torch.linalg.svd(jt, full_matrices=False)
    keep = s > rtol * s[..., :1]
    inv = torch.where(keep, 1. / s.clamp_min(torch.finfo(s.dtype).tiny), 0.)
    force = (vh.transpose(-1, -2) @ (inv * (u.transpose(-1, -2) @ tau[..., None]).squeeze(-1))[..., None]).squeeze(-1)
    residual = (jt @ force[..., None]).squeeze(-1) - tau
    valid = finite & keep.all(-1) & torch.isfinite(force).all(-1)
    valid &= residual.norm(dim=-1) <= 10 * rtol * (1 + tau.norm(dim=-1))
    return FootForceEstimate(torch.where(valid[..., None], force, 0.), valid)


class MULEHistory:
    """Owned deployable history, NOT PPO storage. No implicit append on inference.

    timestamp is a monotonically increasing global control-step ID; episode IDs
    are per-environment. A changed episode clears only that row before append.
    Save these buffers separately for exact environment continuation; a fresh
    physical reset must construct/reset history instead of restoring stale data.
    """
    def __init__(self, schema, num_envs, device='cpu'):
        self.schema = schema
        self.observations = torch.zeros(num_envs, schema.history_length, 45, device=device)
        self.adaptive = torch.zeros(num_envs, schema.history_length, 57, device=device)
        self.episode_ids = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.timestamp = -1

    @torch.no_grad()
    def append(self, observation, forces, episode_ids, timestamp):
        if timestamp <= self.timestamp: raise ValueError('History must advance exactly once per action timestamp')
        n = self.observations.shape[0]
        if observation.shape != (n, 45) or episode_ids.shape != (n,):
            raise ValueError('History observation/episode shape mismatch')
        if forces.force_b.shape != (n, 4, 3) or forces.valid.shape != (n, 4):
            raise ValueError('Expected deployable force estimates [N,4,3] with validity')
        reset = episode_ids != self.episode_ids
        self.observations[reset] = 0.; self.adaptive[reset] = 0.
        self.observations[:, :-1] = self.observations[:, 1:].clone()
        self.adaptive[:, :-1] = self.adaptive[:, 1:].clone()
        self.observations[:, -1] = observation
        force = torch.where(forces.valid[..., None], forces.force_b, 0.).flatten(1)
        self.adaptive[:, -1] = torch.cat((observation, force*self.schema.force_scale), -1)
        self.episode_ids.copy_(episode_ids); self.timestamp = int(timestamp)

    def snapshot(self):
        return self.observations.flatten(1).clone(), self.adaptive.flatten(1).clone()


@dataclass(frozen=True)
class MULEAuxiliaryBatch:
    history: torch.Tensor
    velocity_b: torch.Tensor  # clean physical label at t, not t+1
    next_observation: torch.Tensor  # packed o(t+1), owned pre-reset snapshot if available
    valid: torch.Tensor
    episode_id: torch.Tensor
    next_episode_id: torch.Tensor

    def valid_rows(self):
        n = self.history.shape[0]
        if any(x.shape != (n,) for x in (self.valid, self.episode_id, self.next_episode_id)):
            raise ValueError('Auxiliary masks/episode IDs must be [N]')
        if self.valid.dtype != torch.bool: raise ValueError('valid must be boolean')
        return self.valid & (self.episode_id == self.next_episode_id)


@dataclass(frozen=True)
class MULECommand:
    nominal_raw: torch.Tensor
    adaptive_raw: torch.Tensor
    combined_raw: torch.Tensor
    combined_clipped: torch.Tensor
    position_target: torch.Tensor  # issued absolute target, not yet delay-selected/executed


@dataclass(frozen=True)
class MULEExecution:
    position_target: torch.Tensor  # actual delay-selected target at application
    source_transition_id: torch.Tensor
    applied_torque_nm: torch.Tensor  # actual actuator torque, input to force proxy at next timestamp
