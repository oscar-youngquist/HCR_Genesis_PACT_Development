"""FACET-only control-boundary packets; legacy observations remain untouched."""
from dataclasses import dataclass
import torch

from legged_gym.envs.b1z1.directional_compliance import rotate_yaw, sphere_basis
from legged_gym.utils.math_utils import quat_rotate_inverse
from .reference import PARAMETERS


JOINTS = tuple(f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
               for joint in ("hip", "thigh", "calf")) + (
    "z1_waist", "z1_shoulder", "z1_elbow", "z1_wrist_angle", "z1_forearm_roll",
    "z1_wrist_rotate", "z1_jointGripper")
ACTION_JOINTS = JOINTS[:17]


@dataclass
class FACETObservationCfg:
    enabled: bool = True
    noise_seed: int = 1704
    observed_joint_names: tuple = JOINTS


def nominal_command(env):
    """Nominal mixed-task XYZ and derivative: heading XY, terrain-based world Z.

    The derivative is trajectory-only; no true base velocity enters student
    observations. Virtual-base translation/yaw transport remains in Stage 2.
    """
    u = (env.goal_timer/env.traj_timesteps).clamp(0, 1)
    rate = (env.ee_goal_sphere-env.ee_start_sphere) * (
        30*u.square()*(1-u).square()/env.traj_timesteps/env.dt)[:, None]
    r, p = env.curr_ee_goal_sphere[:, 0], env.curr_ee_goal_sphere[:, 1]
    scale = torch.stack((torch.ones_like(r), r, r*p.cos()), -1)
    velocity = ((rate*scale)[..., None]*sphere_basis(env.curr_ee_goal_sphere)).sum(1)
    position = env.curr_ee_goal_cart.clone()
    position[:, :2] += env.ee_goal_center_offset[:, :2]
    position[:, 2] += env.simulator.measured_heights.mean(1) + env.cfg.goal_ee.sphere_center.z_invariant_offset
    return torch.cat((env.commands[:, :3], position, velocity), -1)


def external_wrench(env, origin, yaw):
    """Held loads at this boundary: base F, EE F, base M in heading axes.

    Source M is a free couple at the supplied application point. Shift once to
    the measured base link origin (never randomized CoM). No spring reactions.
    These are currently held loads, not the next interval's unsampled forces.
    """
    sim, cfg = env.simulator, env.cfg.commands
    force, couple, point = sim.applied_torso_wrench_source()
    force = force if cfg.push_robot_base and cfg.apply_base_external_forces else torch.zeros_like(force)
    couple = couple if cfg.push_robot_base and getattr(cfg, "apply_base_external_torques", False) else torch.zeros_like(couple)
    ee = sim._ee_force_world if cfg.push_gripper_stators and cfg.apply_ee_external_forces else torch.zeros_like(force)
    moment = couple + torch.linalg.cross(point-origin, force)
    return torch.cat(tuple(rotate_yaw(value, -yaw) for value in (force, ee, moment)), -1)


class FACETObservations:
    """Owned cache. Only collect() advances; read() is copy-only and RNG-free."""
    def __init__(self, env):
        cfg = env.cfg.facet_observations
        names = tuple(env.cfg.asset.dof_names)
        if names[:17] != ACTION_JOINTS:
            raise ValueError("FACET position-action ordering differs from the existing Pos controller")
        observed = tuple(cfg.observed_joint_names)
        if len(set(observed)) != len(observed) or not set(JOINTS[12:18]).issubset(observed):
            raise ValueError("FACET observed joints must be unique and include all six FK joints")
        self.joint_ids = torch.tensor([names.index(n) for n in observed], device=env.device)
        self.arm_ids = torch.tensor([observed.index(n) for n in JOINTS[12:18]], device=env.device)
        self.rng = torch.Generator(device=env.device).manual_seed(cfg.noise_seed)
        n, j = env.num_envs, len(observed)
        self.gravity = torch.zeros(n, 4, 3, device=env.device)
        self.q = torch.zeros(n, 4, j, device=env.device)
        self.qd = self.q.clone()
        self.actions = torch.zeros(n, 3, 17, device=env.device)
        self.skip = torch.zeros(n, device=env.device, dtype=torch.bool)
        self.packet = dict(observation=torch.zeros(n, 9+10+12+8*j+51+3, device=env.device),
            privileged=torch.zeros(n, 13+13*len(env.cfg.facet_tracking.horizons), device=env.device),
            wrench=torch.zeros(n, 9, device=env.device), is_init=torch.ones(n, 1, dtype=torch.bool, device=env.device))
        # Ordered blocks are the serialization contract; all values are raw SI.
        self.schema = dict(version=1, observed_joint_names=observed, action_joint_names=ACTION_JOINTS,
            foot_names=tuple(env.cfg.asset.foot_name), parameter_names=PARAMETERS,
            observation_blocks=(('nominal_commands', 9), ('impedance', 10), ('gravity_history', 12),
                                ('joint_position_history', 4*j), ('joint_velocity_history', 4*j),
                                ('issued_action_history', 51), ('ee_position_trunk', 3)),
            privileged_blocks=(('base_velocity_heading', 3), ('ee_velocity_heading', 3),
                               ('angular_velocity_heading', 3), ('foot_contacts', 4),
                               ('references_and_validity', 13*len(env.cfg.facet_tracking.horizons))),
            wrench_blocks=('base_force_heading', 'ee_force_heading', 'base_moment_heading'),
            horizons=tuple(env.cfg.facet_tracking.horizons), history_order='newest first')

    def _noise(self, value, amplitude):
        return value + (2*torch.rand(value.shape, device=value.device, dtype=value.dtype, generator=self.rng)-1)*amplitude

    @torch.no_grad()
    def collect(self, env, ids, issued_action=None, reset=False):
        if not len(ids):
            return
        sim = env.simulator
        position, velocity, yaw, _, _, ee_velocity = env._facet_measured_state()
        # Read reset-fresh articulation data, not the simulator's terminal cache.
        data = getattr(getattr(sim, '_robot', None), 'data', None)
        if data is not None:
            quat = data.root_link_quat_w[:, [1, 2, 3, 0]]
            q, qd = data.joint_pos[:, sim._dof_indices], data.joint_vel[:, sim._dof_indices]
            angular = data.root_link_ang_vel_w
        else:
            quat, q, qd, angular = sim.base_quat, sim.dof_pos, sim.dof_vel, sim._base_world_ang_vel
        gravity = quat_rotate_inverse(quat[ids], q.new_tensor([0., 0., -1.]).expand(len(ids), 3))
        q, qd = q[ids].index_select(1, self.joint_ids), qd[ids].index_select(1, self.joint_ids)
        if env.cfg.noise.add_noise:
            scales, level = env.cfg.noise.noise_scales, env.cfg.noise.noise_level
            gravity = self._noise(gravity, scales.gravity*level)
            q = self._noise(q, scales.dof_pos*level)
            qd = self._noise(qd, scales.dof_vel*level)
        for buffer, current in ((self.gravity, gravity), (self.q, q), (self.qd, qd)):
            if reset:
                buffer[ids] = current[:, None]  # Repeat first measurement; no terminal history.
            else:
                buffer[ids, 1:] = buffer[ids, :-1]
                buffer[ids, 0] = current
        if reset:
            self.actions[ids] = 0
        elif issued_action is not None:
            self.actions[ids, 1:] = self.actions[ids, :-1]
            self.actions[ids, 0] = issued_action[ids]
        reference = getattr(env, 'facet_reference', None)
        params = (torch.cat([reference._parameters[name][ids] for name in PARAMETERS], -1) if reference is not None
                  else q.new_tensor([getattr(env.cfg.facet_reference, name) for name in PARAMETERS]).expand(len(ids), -1))
        self.packet['observation'][ids] = torch.cat((nominal_command(env)[ids], params,
            self.gravity[ids].flatten(1), self.q[ids].flatten(1), self.qd[ids].flatten(1), self.actions[ids].flatten(1),
            env.compute_z1_arm_fk(q.index_select(1, self.arm_ids))), -1)
        # Only this separate view is safe after partial reset. read() deliberately
        # retains terminal diagnostics in the Stage 3 bank.
        bank = getattr(env, 'facet_history', None)
        refs = q.new_zeros(len(ids), len(env.cfg.facet_tracking.horizons), 13)
        if bank is not None:
            current = bank.read_current()
            target, valid = current['targets'][ids], current['valid'][ids]
            zero = torch.zeros_like(target[..., :1])
            def planar(v):
                return rotate_yaw(torch.cat((v, zero), -1), -yaw[ids, None])[..., :2]
            angle = target[..., 4]-yaw[ids, None]
            refs[..., :12] = torch.cat((planar(target[..., :2]-position[ids, None, :2]), planar(target[..., 2:4]),
                torch.atan2(angle.sin(), angle.cos())[..., None], target[..., 5:6],
                rotate_yaw(target[..., 6:9]-position[ids, None], -yaw[ids, None]),
                rotate_yaw(target[..., 9:12], -yaw[ids, None])), -1)
            refs[..., :12] *= valid[..., None]
            refs[..., 12] = valid
        contacts = sim.foot_contacts[ids].float()
        if reset:
            contacts = torch.zeros_like(contacts)  # Contact sensor can retain the pre-reset interval.
        self.packet['privileged'][ids] = torch.cat((rotate_yaw(velocity[ids], -yaw[ids]),
            rotate_yaw(ee_velocity[ids], -yaw[ids]), rotate_yaw(angular[ids], -yaw[ids]), contacts, refs.flatten(1)), -1)
        self.packet['wrench'][ids] = 0 if reset else external_wrench(env, position, yaw)[ids]
        self.packet['is_init'][ids] = reset

    def read(self):
        return {key: value.detach().clone() for key, value in self.packet.items()}
