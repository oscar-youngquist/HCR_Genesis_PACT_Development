import math
import torch

from legged_gym.envs.base.legged_robot_dreamwaq import LeggedRobotDreamwaq
from legged_gym.utils.math_utils import wrap_to_pi, quat_apply, torch_rand_float, quat_rotate_inverse

class Go2Dreamwaq(LeggedRobotDreamwaq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_reward_curriculum(0)

    def _init_buffers(self):
        super()._init_buffers()
        queue_length = self.action_queue.shape[1] if hasattr(self, 'action_queue') else 1
        self._raw_action_queue = torch.zeros(
            self.num_envs, queue_length, self.num_actions, device=self.device)

    def _pre_sim_step(self, actions):
        delayed_actions = super()._pre_sim_step(actions)
        # Preserve the pre-clipping request with exactly the same delay as execution.
        self._raw_action_queue[:, 1:] = self._raw_action_queue[:, :-1].clone()
        self._raw_action_queue[:, 0] = actions.detach().to(self.device)
        delay = self.action_delay if self.cfg.domain_rand.randomize_ctrl_delay else 0
        self.simulator.raw_delayed_actions.copy_(self._raw_action_queue[
            torch.arange(self.num_envs, device=self.device), delay])
        return delayed_actions

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._raw_action_queue[env_ids] = 0

    def _parse_cfg(self, cfg):
        # Derive critic width from the configured sensors rather than a stale
        # hard-coded height-grid size. Actor history remains proprioceptive.
        height_count = (len(cfg.terrain.measured_points_x) * len(cfg.terrain.measured_points_y)
                        if cfg.terrain.measure_heights else 0)
        contacts = len(cfg.asset.contact_state_link_names) if cfg.asset.obtain_link_contact_states else 0
        cfg.env.single_critic_obs_len = cfg.env.num_observations + 31 + 3 + height_count + contacts
        cfg.env.num_privileged_obs = cfg.env.c_frame_stack * cfg.env.single_critic_obs_len
        cfg.domain_rand.push_interval_s = cfg.control.decimation * cfg.sim.dt
        super()._parse_cfg(cfg)

    def step_reward_curriculum(self, iteration):
        if not self.cfg.rewards.use_reward_curriculum:
            return
        curriculum = self.cfg.rewards.reward_curriculum
        fraction = min(1., max(0., (iteration - curriculum.warmup_steps) / max(curriculum.curr_steps, 1)))
        ramp = 0.5 * (1 - math.cos(math.pi * fraction))
        for name, (low, high) in curriculum.curr_reward_bounds.items():
            if name not in self.reward_scales:
                raise ValueError(f"Curriculum reward {name!r} must be enabled")
            self.reward_scales[name] = (low + (high - low) * ramp) * self.dt


    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                                         # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,  # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
            self.actions                                                    # num_actions
        ), dim=-1)
        
        domain_randomization_info = torch.cat((
                    (self.simulator._friction_values - 
                    self.friction_value_offset),            # 1
                    self.simulator._added_base_mass,        # 1
                    self.simulator._base_com_bias,          # 3
                    self.simulator._rand_push_vels[:, :2],  # 2
                    (self.simulator._kp_scale - 
                     self.kp_scale_offset),                 # num_actions
                    (self.simulator._kd_scale - 
                     self.kd_scale_offset),                 # num_actions
            ), dim=-1)
        
        # Critic observation
        critic_obs = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,                   # 3
            self.obs_buf,                 # num_observations
            domain_randomization_info,    # 34
        ), dim=-1)
        
        ## add link contact states
        if self.cfg.asset.obtain_link_contact_states:
            critic_obs = torch.cat(
                (
                    critic_obs,                         # previous
                    self.simulator.link_contact_states,  # 17
                ),
                dim=-1,
            )
        ## add measured terrain heights
        if self.cfg.terrain.measure_heights: # 81
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            critic_obs = torch.cat((critic_obs, heights), dim=-1)
        
        self.critic_obs_deque.append(critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) -
                             1) * self.noise_scale_vec

        # push obs_buf to obs_history
        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
        
        # next state
        self.next_state_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                                         # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,  # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
            self.actions,  # same units as the clean proprioceptive observation
        ), dim=-1)
        
        # explicit info labels
        self.explicit_labels_buf = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,  # body-frame velocity, 3
            (torch.linalg.vector_norm(self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices], dim=-1)
             > self.cfg.rewards.contact_force_threshold).float(),  # FR, FL, RR, RL
            self.simulator.feet_pos[:, :, 2] -
                torch.mean(self.simulator.height_around_feet, dim=-1) -
                self.cfg.rewards.foot_height_offset,  # 4
        ), dim=-1)
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_pos[:, [0, 3, 6, 9]] = self.simulator.default_dof_pos[:, [0, 3, 6, 9]] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), 4), self.device)
        dof_pos[:, [1, 4, 7, 10]] = self.simulator.default_dof_pos[:, [1, 4, 7, 10]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)
        dof_pos[:, [2, 5, 8, 11]] = self.simulator.default_dof_pos[:, [2, 5, 8, 11]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)

        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[9:21] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.  # previous actions
        return noise_vec
    
    def _reward_dof_vel_limits(self):
        # Cap each joint's excess at 1 rad/s, as in the base locomotion convention.
        excess = (self.simulator.dof_vel.abs()
                  - self.simulator.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit)
        return excess.clamp(min=0., max=1.).sum(dim=-1)

    def _reward_front_foot_overreach(self):
        sim = self.simulator
        front_x = torch.stack([
            quat_rotate_inverse(sim.base_quat, sim.feet_pos[:, i] - sim.base_pos)[:, 0]
            for i in (0, 1)], dim=-1)
        excess = (front_x - self.cfg.rewards.overreach_x_max).clamp_min(0.)
        contact = (sim.link_contact_forces[:, sim.feet_contact_indices[:2], 2]
                   > self.cfg.rewards.overreach_contact_force_threshold)
        penalty = (contact * excess.square()).sum(dim=-1)
        # Preserve PACT's payload-dependent front-foot penalty scaling.
        total_mass = sim._robot_mass + sim._added_base_mass.clamp_min(0.)
        scale = 1. - .5 * (sim._robot_mass / total_mass).squeeze(-1)
        return scale * penalty

    def _reward_rear_foot_overreach(self):
        sim = self.simulator
        rear_x = torch.stack([
            quat_rotate_inverse(sim.base_quat, sim.feet_pos[:, i] - sim.base_pos)[:, 0]
            for i in (2, 3)], dim=-1)
        excess = ((rear_x - self.cfg.rewards.rear_foot_x_nominal).abs()
                  - self.cfg.rewards.rear_foot_x_margin).clamp_min(0.)
        contact = (sim.link_contact_forces[:, sim.feet_contact_indices[2:4], 2]
                   > self.cfg.rewards.overreach_contact_force_threshold)
        return (contact * excess.square()).sum(dim=-1)

    def _reward_torque_limits(self):
        excess = (self.simulator.requested_torques.abs()
                  - self.simulator.torque_limits * self.cfg.rewards.soft_torque_limit)
        return excess.clamp_min(0).sum(dim=-1)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        
        Attention: using torch.max(self.simulator.height_around_feet) will cause reward value jumping, bad for learning
        """
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(
                self.simulator.feet_pos[:, :, 2] - torch.mean(self.simulator.height_around_feet, dim=-1) -
                self.cfg.rewards.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)
    
    def _reward_hip_pos(self):
        """ Reward for the hip joint position close to default position
        """
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(torch.square(
            self.simulator.dof_pos[:, hip_joint_indices] - 
            self.simulator.default_dof_pos[:, hip_joint_indices]), dim=-1)
        return dof_pos_error
