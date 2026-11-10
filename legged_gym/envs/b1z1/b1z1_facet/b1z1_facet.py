"""FACET-only reference hooks; Pos physics, observations and rewards stay intact."""
import torch

from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos import B1Z1PPOPos
from legged_gym.envs.b1z1.directional_compliance import heading, rotate_yaw, sphere_basis
from .reference import FACETReference, applied_loads


class B1Z1FACET(B1Z1PPOPos):
    """Stage 2: virtual references only, no FACET controller or teacher/student."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.facet_reference = None
        self._facet_interval = None
        if self.cfg.facet_reference.enabled:
            self.facet_reference = FACETReference(self.num_envs, self.device, self.cfg.facet_reference)
            self._reset_facet_reference(self.all_env_ids)

    def _facet_measured_state(self):
        sim = self.simulator
        # Isaac Lab's articulation buffers are updated by reset writes; the
        # simulator's policy-step EE/base cache may still describe a terminal state.
        if hasattr(getattr(sim, "_robot", None), "data"):
            data = sim._robot.data
            quat = data.root_link_quat_w[:, [1, 2, 3, 0]]  # wxyz -> xyzw
            yaw, rate = heading(quat, data.root_link_ang_vel_w)
            return (data.root_link_pos_w, data.root_link_lin_vel_w, yaw, rate,
                    data.body_link_pos_w[:, sim._gripper_index],
                    data.body_link_vel_w[:, sim._gripper_index, :3])
        yaw, rate = heading(sim.base_quat, sim._base_world_ang_vel)
        return sim.base_pos, sim._base_world_lin_vel, yaw, rate, sim.ee_pos, sim.ee_vel

    @torch.no_grad()
    def _reset_facet_reference(self, ids):
        state = self._facet_measured_state()
        self.facet_reference.reset(ids, *(value[ids] for value in state))

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if getattr(self, "facet_reference", None) is not None and len(env_ids):
            self._reset_facet_reference(env_ids)

    @torch.no_grad()
    def _pre_sim_step(self, actions):
        executed = super()._pre_sim_step(actions)
        if getattr(self, "facet_reference", None) is None:
            return executed
        position, _, yaw, _, _, _ = self._facet_measured_state()
        u = (self.goal_timer / self.traj_timesteps).clamp(0, 1)
        sphere_rate = ((self.ee_goal_sphere-self.ee_start_sphere) *
            (30*u.square()*(1-u).square()/self.traj_timesteps/self.dt)[:, None])
        radius, elevation = self.curr_ee_goal_sphere[:, 0], self.curr_ee_goal_sphere[:, 1]
        scales = torch.stack((torch.ones_like(radius), radius, radius*elevation.cos()), -1)
        cart_rate = ((sphere_rate*scales)[:, :, None]*sphere_basis(self.curr_ee_goal_sphere)).sum(1)
        local = self.curr_ee_goal_cart.clone()
        local[:, :2] += self.ee_goal_center_offset[:, :2]
        height = self.simulator.measured_heights.mean(1) + self.cfg.goal_ee.sphere_center.z_invariant_offset
        velocity_command = rotate_yaw(torch.cat((self.commands[:, :2], torch.zeros_like(yaw[:, None])), -1), yaw)
        _, _, point = self.simulator.applied_torso_wrench_source()
        # Commands, geometry and nominal derivatives at t, not the resampled
        # successor commands. Terrain height and measured base anchor are held
        # during this dt; virtual base motion transports the attached EE setpoint.
        self._facet_interval = tuple(v.detach().clone() for v in
            (position, yaw, velocity_command[:, :2], self.commands[:, 2], local, cart_rate, height, point))
        return executed

    def post_physics_step(self):
        if getattr(self, "facet_reference", None) is not None and self._facet_interval is not None:
            position, yaw, command, yaw_command, local, local_rate, height, point = self._facet_interval
            # These buffers are the actual sampled/gated loads held throughout
            # the just-completed interval. Read before parent callbacks can reset
            # or resample; use the captured start lever, never successor geometry.
            ee, base, moment = applied_loads(self.cfg.commands, self.simulator._ee_force_world,
                self.simulator.applied_torso_wrench_source(), position, point)
            self.facet_reference.advance(self.dt, position[:, :2], yaw, command, yaw_command,
                                         local, local_rate, height, ee, base, moment)
            self._facet_interval = None  # Reads/post-step calls cannot advance twice.
        super().post_physics_step()

    def get_facet_references(self):
        """Copy-on-read diagnostic API; no observations, reward targets or histories."""
        return {} if self.facet_reference is None else self.facet_reference.read()
