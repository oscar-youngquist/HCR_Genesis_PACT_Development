from legged_gym import *
from time import time
import numpy as np
import os
from legged_gym.utils.math_utils import *

import torch
from torch import Tensor
from typing import Tuple, Dict
import random
from collections import deque

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math_utils import wrap_to_pi, torch_rand_float, quat_apply
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from ...base.legged_robot_config import LeggedRobotCfg
import torch.nn.functional as F

class Go1PACTPos(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params: dict, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.init_done = False
        self._parse_cfg(self.cfg, sim_device)
        super().__init__(self.cfg, sim_params, sim_device, headless)
        
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def get_observations(self):
        return self.obs_buf, self.obs_history, self.privileged_obs_buf, self.explicit_labels_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = self._pre_sim_step(actions)
        
        self.simulator.step(actions)
        
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs)
        
        return self.obs_buf, self.privileged_obs_buf, self.obs_history, self.explicit_labels_buf, \
            self.rew_buf, self.reset_buf, self.extras, (self.simulator._grfs_buf * self.obs_scales.grf)

    def get_failure_idx(self):
        return self.reset_buf * ~self.time_out_buf
    
    def get_scaled_pos_actions(self):
                # control_type = 'P'
        # Pull out the position control actions
        pos_actions = self.actions[:,0:12]
        
        # Scale the position actions        
        actions_scaled = pos_actions * self.cfg.control.action_scale + self.simulator.default_dof_pos

        return actions_scaled
    
    def create_async_pino_workers(self):
        self.simulator._create_async_pino_workers()

    def shutdown_asynic_pino_workers(self):
        self.simulator._shutdown_asynic_pino_workers()

    def get_pinn_wb_dynamics(self):
        return self.simulator._get_pinn_wb_dynamics()
    
    def _get_pinn_feedback(self, pos_actions, dof_pos, dof_vel):
        return self.simulator._get_pinn_feedback(pos_actions, dof_pos, dof_vel)

    def _get_pinn_actions(self, actions):
        # Pull out the position control actions
        pos_actions = actions[:,0:12]
        # pull out the torque control actions
        tau_actions = actions[:,12:24]
        
        # Scale and shift the position actions
        actions_scaled = pos_actions * self.cfg.control.action_scale
        target_dof_pos = actions_scaled + self.simulator.default_dof_pos

        # Scale and shift the torque actions
        feedforward_torques = tau_actions * self.cfg.control.torque_scale

        return target_dof_pos, feedforward_torques

    def get_prev_obs(self):
        return self.last_obs_buf, self.last_obs_hist, self.llast_obs_buf, self.llast_obs_hist

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self.simulator.draw_debug_vis() if needed
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.simulator.post_physics_step()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        
        self.compute_reward()
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        if self.cfg.sensor.add_depth:
            self.simulator.update_depth_images()
        
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        if self.debug:
            self.simulator.draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        fail_buf = torch.any(
            torch.norm(self.simulator.link_contact_forces[:, self.simulator.termination_contact_indices, :], dim=-1)
            > 10.0, dim=1)
        # print(f"contact termination: {fail_buf}")
        # fail_buf |= self.simulator.projected_gravity[:, 2] > self.cfg.rewards.max_projected_gravity
        # print(f"gravity termination: {self.simulator.projected_gravity[:, 2] > self.cfg.rewards.max_projected_gravity}")
        
        if hasattr(self.cfg, "termination"):
            # more sophisticated termination conditions
            rpy = self.simulator._base_euler
            r, p = wrap_to_pi(rpy[:,0]), wrap_to_pi(rpy[:,1])
            base_height = torch.mean(self.simulator.base_pos[:, 2].unsqueeze(1) - self.simulator.measured_heights, dim=1)
            
            if "roll" in self.cfg.termination.termination_terms:
                r_term_buff = torch.abs(r) > self.cfg.termination.roll_threshold
                self.fail_buf |= r_term_buff
            if "pitch" in self.cfg.termination.termination_terms:
                p_term_buff = torch.abs(p) > self.cfg.termination.pitch_threshold
                self.fail_buf |= p_term_buff
            if "height_min" in self.cfg.termination.termination_terms:
                height_term_buff = base_height < self.cfg.termination.height_min
                self.fail_buf |= height_term_buff
            if "height_max" in self.cfg.termination.termination_terms:
                height_term_buff = base_height > self.cfg.termination.height_max
                self.fail_buf |= height_term_buff
        
        self.fail_buf += fail_buf
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
        )

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length ==0):
            self._update_command_curriculum(env_ids)

        # Update the position/torque control tradeoff curriculum 
        if self.use_tradeoff:
            self.step_tradeoff_curriculum(env_ids)

        self._resample_commands(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.simulator.reset_idx(env_ids)

        # after base pose/orientation has been reset:
        self.phi_prev_orientation[env_ids] = self._potential_orientation()[env_ids]

        self.phi_prev_height[env_ids] = self._potential_height()[env_ids]

        # reset buffers
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0

        # clear obs history for the envs that are reset
        self.last_obs_buf[env_ids] = 0.
        self.llast_obs_buf[env_ids] = 0.

        # clear history
        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.simulator.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # reset action queue and delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.
            self.action_queue[env_ids] = 0.
            self.action_delay[env_ids] = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                                       self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination(
            ) * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        # Update previous observations
        self.llast_obs_buf = self.last_obs_buf.clone().detach()
        self.last_obs_buf = self.obs_buf.clone().detach()

        # Compute new observation
        self.obs_buf = torch.cat((self.commands[:, :3] * self.commands_scale,                 # velocity commands     3
                                  self.simulator.projected_gravity,                           # projected gravity vec 3
                                  self.simulator.base_ang_vel * self.obs_scales.ang_vel,      # angular velocity      3
                                  (self.simulator.dof_pos - self.simulator.default_dof_pos)   
                                      * self.obs_scales.dof_pos,                              # joint pose            12
                                    self.simulator.dof_vel * self.obs_scales.dof_vel,         # joint velocity        12
                                    self.actions[:,0:12],                                     # joint pose actions    12
                                    self.simulator.feedback_torques * (1.0/float(self.cfg.control.torque_scale)),    # joint torque actions  12
                                    ), dim=-1)                                                # 57

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # build the explicit labels buffer
        self.explicit_labels_buf = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,                     # 3  - torso linear velocity
            self.simulator.link_contact_states[:,self.simulator.feet_indices],         # 4  - contact states of feet
            torch.clip(self.simulator.feet_pos[:, :, 2] -
                torch.mean(self.simulator.height_around_feet, dim=-1) -
                self.cfg.rewards.foot_height_offset, -1, 1.),                          # 4  - feet height
            torch.mean(self.simulator.base_pos[:, 2].unsqueeze(1) - 
                       self.simulator.measured_heights, dim=1, keepdim=True) - 
                       self.cfg.rewards.base_height_target,                            # 1  - base height error
            self.simulator._added_base_mass,                                           # 1  - payload mass
            self.simulator._base_com_bias,                                             # 3  - CoM shift
        ), dim=-1)

        # track history buffer
        self.llast_obs_hist = self.last_obs_hist.clone().detach()
        self.last_obs_hist = self.obs_history.clone().detach()
        
        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i] for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )

        # build up privlieged domain randomization buffer
        domain_randomization_info = torch.cat((
            (self.simulator._friction_values - self.friction_value_offset),  # 1
            self.simulator._added_base_mass,                                 # 1
            self.simulator._base_com_bias,                                   # 3
            self.simulator._rand_push_vels,                                  # 3
            self.simulator._rand_wrench_vels,                                # 3
            (self.simulator._kp_scale - self.kp_scale_offset),               # num_actions
            (self.simulator._kd_scale - self.kd_scale_offset),               # num_actions
            self.simulator._motor_strength,                                  # num_actions
            self.simulator._joint_armature,                                  # 1
            self.simulator._joint_friction,                                  # 1
            self.simulator._joint_damping,                                   # 1
            ), dim=-1)                                                       # 50

        critic_obs = torch.cat(
            (
                self.obs_buf,                                                          # 57
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,                 # 3  - base linear velocity
                torch.mean(self.simulator.base_pos[:, 2].unsqueeze(1) - 
                           self.simulator.measured_heights, dim=1, keepdim=True),      # 1  - base height
                self.simulator._grfs_buf * self.obs_scales.grf,                        # 12 - measured ground reaction forces (GRFs)
                self.simulator.normal_vector_around_feet.reshape(self.num_envs, -1),   # 12 - terrain info around feet
                self.simulator.link_contact_states[:,self.simulator.feet_indices],     # 4  - contact states of feet
                torch.clip(self.simulator.feet_pos[:, :, 2] -
                    torch.mean(self.simulator.height_around_feet, dim=-1) -
                    self.cfg.rewards.foot_height_offset, -1, 1.),                      # 4 - feet height
                self.simulator.feedforward_tau_weight,                    # 1
                self.simulator.feedback_tau_weight,                       # 1
                domain_randomization_info                                 # 50
            ),
            dim=-1,
        ) # 145 (grf) / 133 (no-grf)

        # add hieght measurements to asymmetric critic if approperiate
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(1) - 0.5 \
                                 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements # 81
            heights *= self.height_noise_vec
            critic_obs = torch.cat((critic_obs, heights), dim=-1) # 207

        self.critic_obs_deque.append(critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )

    def set_viewer_camera(self, pos, lookat):
        """ Set viewer camera position and direction
        """
        self.simulator.set_viewer_camera(eye=pos, target=lookat)

    # ------------- Callbacks (Protected Function) --------------
    
    def _pre_sim_step(self, actions):
        """ Callback called at the beginning of the step function, before stepping the simulation
        """
        clip_actions = self.cfg.normalization.clip_actions
        actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # update history of actions
        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.actions[:] = actions[:]
        
        # apply action delay by using an action queue
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            actions = self.action_queue[torch.arange(self.num_envs), self.action_delay].clone()
        
        # # during training, the camera follows the first environment
        # if not self.debug and not self.headless:
        #     pos = self.simulator.base_pos[0].cpu().numpy() + np.array(self.cfg.viewer.pos)
        #     lookat = self.simulator.base_pos[0].cpu().numpy() + np.array(self.cfg.viewer.lookat)
        #     self.set_viewer_camera(pos, lookat)
        
        return actions
    
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(
            self.simulator.base_pos[env_ids, :2] - self.simulator.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.simulator._terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(
            self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        
        self.simulator.update_terrain_curriculum(env_ids, move_up, move_down)
    
    def _reset_dofs(self, env_ids):
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_pos[:, :] = self.simulator.default_dof_pos[:] + \
            torch_rand_float(-0.3, 0.3, (len(env_ids), self.num_actions), self.device)        

        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def _reset_root_states(self, env_ids):
        # base pos
        if self.simulator.custom_origins:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
            base_pos[:, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
        # base quat
        base_quat = self.simulator.base_init_quat.reshape(1, -1).repeat(len(env_ids), 1)
        # base lin vel
        base_lin_vel = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        # base ang vel
        base_ang_vel = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        
        self.simulator.reset_root_states(env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.simulator.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), self.cfg.commands.ranges.ang_vel_yaw[0], 
                                                                 self.cfg.commands.ranges.ang_vel_yaw[1])

        if self.cfg.domain_rand.push_robots:
            self.simulator.push_robots()
        
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids),1), self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids),1), self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        # set small commands to zero
        self.commands[env_ids, :3] *= (torch.norm(
            self.commands[env_ids, :3], dim=1) > 0.2).unsqueeze(1)

    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > \
                self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel"]:
            # self.command_ranges["lin_vel_x"][0] = np.clip(
            #     self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            # self.command_ranges["lin_vel_x"][1] = np.clip(
            #     self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_y"][0] = np.clip(
                self.command_ranges["lin_vel_y"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            
            self.command_ranges["ang_vel_yaw"][0] = np.clip(
                self.command_ranges["ang_vel_yaw"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            
            
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
            self.command_ranges["lin_vel_y"][1] = np.clip(
                self.command_ranges["lin_vel_y"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
            
            self.command_ranges["ang_vel_yaw"][1] = np.clip(
                self.command_ranges["ang_vel_yaw"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])

        self.height_noise_vec = torch.zeros(self.simulator._num_height_points, device=self.device)

        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # input commands
        noise_vec[:3] = 0.
        # projected gravity vector
        noise_vec[3:6] = noise_scales.gravity * noise_level
        # angular velocity
        noise_vec[6:9] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        # leg joint positions
        noise_vec[9:21] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        # leg joint velocities
        noise_vec[21:33] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        
        # previous joint position actions
        noise_vec[33:45] = 0.
        # previous joint torque actions
        noise_vec[45:57] = 0.

        if self.cfg.terrain.measure_heights:
            self.height_noise_vec[:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        
        # PBRS orientation reward
        self.phi_prev_orientation = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self.phi_prev_height = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self.forward_vec[:, 0] = 1.0
        self.fail_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        
        self.commands = torch.zeros(
            (self.num_envs, self.cfg.commands.num_commands), device=self.device, dtype=torch.float)
        
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, dtype=torch.float,
                                           requires_grad=False)
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.last_actions = torch.zeros_like(self.actions)
        self.llast_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # last last actions
        
        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.simulator.feet_indices)), device=self.device, dtype=torch.float)
        self.last_contacts = torch.zeros((self.num_envs, len(self.simulator.feet_indices)), device=self.device, dtype=torch.int)

        # history of single observations
        self.last_obs_buf = torch.zeros(
            (self.num_envs, self.cfg.env.num_observations),
            dtype=torch.float,
            device=self.device,
        )

        self.llast_obs_buf = torch.zeros(
            (self.num_envs, self.cfg.env.num_observations),
            dtype=torch.float,
            device=self.device,
        )

        # observation history buffer
        self.obs_history_deque = deque(maxlen=self.cfg.env.num_obs_hist)

        self.obs_history = torch.zeros(
            (self.num_envs, self.num_obs * self.num_obs_hist), device=self.device, dtype=torch.float)
        
        self.last_obs_hist = torch.zeros(
            (self.num_envs, self.num_obs * self.num_obs_hist), device=self.device, dtype=torch.float)
        
        self.llast_obs_hist = torch.zeros(
            (self.num_envs, self.num_obs * self.num_obs_hist), device=self.device, dtype=torch.float)

        for _ in range(self.cfg.env.num_obs_hist):
            self.obs_history_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_observations,
                    dtype=torch.float,
                    device=self.device,
                )
            )

        # dqueue of critic observations. -> used to stablize critic predictions
        self.critic_obs_deque = deque(maxlen=self.cfg.env.num_priv_stack)
        for _ in range(self.cfg.env.num_priv_stack):
            self.critic_obs_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_privileged_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )

        # explicit recon obs buffer (torso lin.velo, feet contact state, feet height)
        self.explicit_labels_buf = torch.zeros(
            (self.num_envs, self.cfg.env.num_explicit_recon_obs),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # randomize action delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(
                self.num_envs, self.cfg.domain_rand.ctrl_delay_step_range[1]+1, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                              self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)
            

    def step_reward_curriculum(self, num_iters):
        # Safety catch
        if not self.use_reward_curriculum:
            return
        
        # initialize the policy with fixed-lower bound
        if num_iters < self.reward_warmup_steps:
            for key in self.reward_curr_keys:
                if key in self.reward_scales.keys():
                    self.reward_scales[key] = self.reward_curr_bounds[key][0] * self.dt
                    # print("Reward - ", key, " scale - ", self.reward_scales[key])
        # Gradually increase the regularization strength
        elif num_iters > self.reward_warmup_steps and (num_iters - self.reward_warmup_steps) < self.reward_curr_steps:
            print("Stepping Reward Curriculum")
            adjusted_iter = num_iters - self.reward_warmup_steps
            for key in self.reward_curr_keys:
                if key in self.reward_scales.keys():
                    low, high = self.reward_curr_bounds[key]

                    alpha = adjusted_iter / self.reward_curr_steps
                    alpha = np.clip(alpha, 0.0, 1.0)
                    print(alpha)
                    ramp = 0.5 * (1.0 - np.cos(np.pi * alpha))

                    self.reward_scales[key] = (low + (high - low) * ramp) * self.dt
                    # self.reward_scales[key] = ((float(adjusted_iter)/float(self.reward_curr_steps))*self.reward_bound_diffs[key] + self.reward_curr_bounds[key][0])*self.dt
                    # print("Reward - ", key, " scale - ", self.reward_scales[key])
        # Fix the regularization strength to the upper-bound
        else:
            # by default set the reward to the upper bound
            for key in self.reward_curr_keys:
                if key in self.reward_scales.keys():
                    self.reward_scales[key] = self.reward_curr_bounds[key][1] * self.dt


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale ==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name =="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def step_tradeoff_curriculum(self, env_ids):
        # # If the tracking reward is above XX% of the maximum, increase the tradeoff        
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > \
                self.cfg.control.tradeoff_threshold * self.reward_scales["tracking_lin_vel"]:
            
            # print("^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*")
            # print(torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length)
            # print(self.cfg.control.tradeoff_threshold * self.reward_scales["tracking_lin_vel"])

            # Increment the tradeoff step-counter for these successful envs.
            self.tradeoff_step_ctr[env_ids] += 1.0

            # Check if this increased the step count of any env beyond the maximum and then reset
            max_step_mask = self.tradeoff_step_ctr > self.tradeoff_num_steps
            self.tradeoff_step_ctr[max_step_mask] = self.tradeoff_num_steps

        # apply the curriculum scaling
        self.simulator.feedforward_tau_weight[env_ids]  = self.tradeoff_step_ctr[env_ids] *float(1.0/self.tradeoff_num_steps)*self.bound_diff[0] + self.tradeoff_lowerbounds[0]
        self.simulator.feedback_tau_weight[env_ids]     = self.tradeoff_step_ctr[env_ids] *float(1.0/self.tradeoff_num_steps)*self.bound_diff[1] + self.tradeoff_lowerbounds[1]

        random_smaple = random.random()
        
        if random_smaple <= 0.25:  # 20% of the time reduce to lower bound
            self.simulator.feedforward_tau_weight[env_ids] = self.tradeoff_lowerbounds[0]
            self.simulator.feedback_tau_weight[env_ids]    = self.tradeoff_lowerbounds[1]
        elif random_smaple > 0.25 and random_smaple <= 0.50: # ~25% of the time, sample a random value between the lower and current upper bound
            # step_ctr * (1.0/num_steps) -> is the per-env upper bound. Multipled by a random float between [0,1)
            random_step_size = self.tradeoff_step_ctr*float(1.0/self.tradeoff_num_steps) * torch.rand((self.num_envs, 1))

            self.simulator.feedforward_tau_weight[env_ids] = random_step_size[env_ids]*self.bound_diff[0] + self.tradeoff_lowerbounds[0]
            self.simulator.feedback_tau_weight[env_ids]    = random_step_size[env_ids]*self.bound_diff[1] + self.tradeoff_lowerbounds[1]

    def _parse_cfg(self, cfg, sim_device):
        self.dt = self.cfg.control.dt
        self.debug = self.cfg.env.debug
        
        self.num_exp_labels = self.cfg.env.num_explicit_recon_obs
        self.num_crit_obs_stack = self.cfg.env.num_priv_stack

        # use self-implemented pd controller
        self.obs_scales = self.cfg.normalization.obs_scales
        
        # Reward stuff (scale, curriculum scales/schedule)
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        
        self.use_reward_curriculum = self.cfg.rewards.use_reward_curriculum

        self.reward_curr_keys = self.cfg.rewards.reward_curriculum.curr_reward_keys
        self.reward_curr_bounds = self.cfg.rewards.reward_curriculum.curr_reward_bounds
        self.reward_curr_steps = self.cfg.rewards.reward_curriculum.curr_steps
        self.reward_warmup_steps = self.cfg.rewards.reward_curriculum.warmup_steps

        self.reward_bound_diffs = {}
        for key in self.reward_curr_keys:
            self.reward_bound_diffs[key] = self.reward_curr_bounds[key][1] - self.reward_curr_bounds[key][0]        
        
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        
        
        # determine privileged observation offset to normalize privileged observations
        self.friction_value_offset = (self.cfg.domain_rand.friction_range[0] + 
                                      self.cfg.domain_rand.friction_range[1]) / 2  # mean value
        
        self.kp_scale_offset = (self.cfg.domain_rand.kp_range[0] +
                                self.cfg.domain_rand.kp_range[1]) / 2  # mean value
        
        self.kd_scale_offset = (self.cfg.domain_rand.kd_range[0] +
                                self.cfg.domain_rand.kd_range[1]) / 2  # mean value
        
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

        # load some PACT specific values
        #     some PACT specific dimension values
        self.wb_dim = self.cfg.env.whole_body_dim
        self.grf_dim = self.cfg.env.grf_dim
        self.num_obs_hist = self.cfg.env.num_obs_hist

        # Tradeoff curriculum stuff...
        self.tradeoff_lowerbounds = np.array(self.cfg.control.tradeoff_init_weights)
        self.tradeoff_upperbounds = np.array(self.cfg.control.tradeoff_final_weights)
        self.tradeoff_num_steps = self.cfg.control.tradeoff_steps
        self.bound_diff = self.tradeoff_upperbounds - self.tradeoff_lowerbounds 
        self.use_tradeoff = self.cfg.control.use_tradeoff_curriculum
        
        print("self.use_tradeoff - ", self.use_tradeoff)

        # We want to be at the full bounds right away, but we want to skip back sometimes for exploration
        self.tradeoff_step_ctr = torch.zeros((self.cfg.env.num_envs, 1), device=sim_device, dtype=torch.float)

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.simulator.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.simulator.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.simulator.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.simulator.base_pos[:, 2].unsqueeze(
            1) - self.simulator.measured_heights, dim=1)
        # print(f"base height: {base_height}")
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        return rew

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.simulator.torques), dim=1)

    def _reward_feedback_torques(self):
        return torch.sum(torch.square(self.simulator.first_loop_feedback),dim=1)
    
    def _reward_feedforward_torques(self):
        return torch.sum(torch.square(self.simulator.feedforward_torques),dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.simulator.dof_vel), dim=1)
    
    def _reward_dof_power(self):
        # Penalize power consumption
        return torch.sum(torch.abs(self.simulator.torques * self.simulator.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.simulator.last_dof_vel - 
                                       self.simulator.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_pos_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,0:12] - self.actions[:,0:12]), dim=1)
    
    def _reward_tau_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,12:24] - self.actions[:,12:24]), dim=1)

    def _reward_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(
            self.actions - 2*self.last_actions + self.llast_actions), dim=-1)
        return action_smoothness_cost

    def _reward_pos_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(
            self.actions[:,0:12] - 2*self.last_actions[:,0:12] + self.llast_actions[:,0:12]), dim=-1)
        return action_smoothness_cost
    
    def _reward_tau_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(
            self.actions[:,12:24] - 2*self.last_actions[:,12:24] + self.llast_actions[:,12:24]), dim=-1)
        return action_smoothness_cost

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        # print(f"contacts: {(torch.norm(self.simulator.link_contact_forces[0, self.simulator.penalized_contact_indices, :], dim=-1) > 0.1)}")
        rew = torch.sum(1.*(torch.norm(
            self.simulator.link_contact_forces[:, self.simulator.penalized_contact_indices, :], 
            dim=-1) > 0.1), dim=1)
        # print(f"collision reward: {rew[0]}")
        return rew

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # def _reward_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(self.simulator.torques) - self.simulator.torques_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_feet_spread_pairwise_axes(self):
        """
        Pair-aware, axis-separated MIN-SEPARATION penalty.

        Only penalizes when distances are BELOW thresholds:
        - Side pairs (left/right): (FL,FR), (RL,RR) -> penalize ONLY if |dx| < x_min  (ignore y)
        - Lateral pairs (front/back on same side): (FL,RL), (FR,RR) -> penalize ONLY if |dy| < y_min (ignore x)
        - Diagonal pairs: (FL,RR), (FR,RL) -> penalize if below thresholds on x and/or y (weighted blend)

        Returns:
        p: (N,) >= 0  (0 when constraints satisfied). You typically SUBTRACT this from total reward,
            e.g. self.rew_buf -= scale * p  (or set reward_scales[...] negative).
        """
        # -------------------- hyperparameters (tune) --------------------
        y_min = 0.35      # min lateral separation target [m]
        x_min = 0.40      # min fore-aft separation target [m]
        alpha_diag = 0.8  # diagonal pair blend: weight on lateral term (0..1)
        fz_thr = 5.0      # contact threshold [N] if using forces

        # contact weighting mode:
        #   "both"  -> w_ij = c_i * c_j  (strict stance-only)
        #   "blend" -> w_ij = 0.5*(c_i + c_j) (some signal when one foot swings)
        contact_mode = "blend"
        # ----------------------------------------------------------------

        # Assumed foot order: 0=FL, 1=FR, 2=RL, 3=RR (common in RSL-RL / legged_gym setups)
        # FL, FR, RL, RR = 0, 1, 2, 3
        FR, FL, RR, RL = 0, 1, 2, 3

        # Trasnform feet pose into base-frame
        feet_base_01 = quat_rotate_inverse(self.simulator.base_quat, (self.simulator.feet_pos[:,0,:] - self.simulator.base_pos))
        feet_base_02 = quat_rotate_inverse(self.simulator.base_quat, (self.simulator.feet_pos[:,1,:] - self.simulator.base_pos))
        feet_base_03 = quat_rotate_inverse(self.simulator.base_quat, (self.simulator.feet_pos[:,2,:] - self.simulator.base_pos))
        feet_base_04 = quat_rotate_inverse(self.simulator.base_quat, (self.simulator.feet_pos[:,3,:] - self.simulator.base_pos))

        feet_stack_base = torch.stack([feet_base_01, feet_base_02, feet_base_03, feet_base_04], dim=1)  # (N,4,3)
        
        feet_xy = feet_stack_base[:,:,:2]  # (N,4,2) use XY for support polygon in ground plane

        # contact mask (N,4) in {0,1}
        c = (self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > fz_thr).float()

        # Define pair groups per your constraint
        side_pairs   = [(FL, FR), (RL, RR)]      # left/right
        lateral_pairs = [(FL, RL), (FR, RR)]     # front/back on same side
        diag_pairs   = [(FL, RR), (FR, RL)]      # diagonals

        def w_pair(i, j):
            if contact_mode == "blend":
                return 0.5 * (c[:, i] + c[:, j])
            return c[:, i] * c[:, j]

        # Hinge penalty: >0 only when below threshold, 0 otherwise
        def hinge(thresh_minus_val):
            return torch.square(torch.relu(thresh_minus_val))

        p = torch.zeros(feet_xy.shape[0], device=feet_xy.device)
        denom = torch.zeros_like(p)

        # Side pairs: penalize ONLY if |dx| < x_min (ignore y)
        for i, j in side_pairs:
            dxy = feet_xy[:, i] - feet_xy[:, j]
            dx = torch.abs(dxy[:, 0])
            wij = w_pair(i, j)
            pij = hinge(x_min - dx)            # (N,)
            p += wij * pij
            denom += wij

        # Lateral pairs: penalize ONLY if |dy| < y_min (ignore x)
        for i, j in lateral_pairs:
            dxy = feet_xy[:, i] - feet_xy[:, j]
            dy = torch.abs(dxy[:, 1])
            wij = w_pair(i, j)
            pij = hinge(y_min - dy)            # (N,)
            p += wij * pij
            denom += wij

        # Diagonal pairs: penalize shortfall on both axes (weighted blend)
        for i, j in diag_pairs:
            dxy = feet_xy[:, i] - feet_xy[:, j]
            dx = torch.abs(dxy[:, 0])
            dy = torch.abs(dxy[:, 1])
            wij = w_pair(i, j)

            px = hinge(x_min - dx)
            py = hinge(y_min - dy)
            pij = alpha_diag * py + (1.0 - alpha_diag) * px
            p += wij * pij
            denom += wij

        # Average over active pairs; if none active, penalty 0
        p = torch.where(denom > 0.0, p / (denom + 1e-6), torch.zeros_like(p))

        # Optional: normalize by thresholds to make it roughly scale-free:
        p = p / (alpha_diag * y_min + (1.0 - alpha_diag) * x_min + 1e-6)

        return p


    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.simulator.torques) - self.simulator.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.simulator.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-4.0*lin_vel_error)

    def _reward_dof_act_limits(self):
        pos_actions = self.actions[:,0:12]
        # actions_scaled = pos_actions * self.cfg.control.action_scale
        actions_scaled = pos_actions * self.cfg.control.action_scale + self.simulator.default_dof_pos
        
        # Penalize dof positions too close to the limit
        out_of_limits = -(actions_scaled - self.simulator.dof_pos_limits_hard[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (actions_scaled - \
                          self.simulator.dof_pos_limits_hard[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.simulator.base_ang_vel[:, 2])
        return torch.exp(-4.0*ang_vel_error)

    def _reward_joint_power(self):
        # penalize large amounts of motor power
        return torch.sum(torch.abs(self.simulator.dof_vel * self.simulator.torques), dim=1)

    def _reward_joint_power_dist(self):
        # Penalize uneven distributions of motor power
        return torch.var(self.simulator.torques*self.simulator.dof_vel, dim=1)

    def _reward_foot_slip(self):
        # penalize feet that are in-contact for any movement in the x/y direction
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.
        return  torch.sum(torch.square(contact * torch.sum(self.simulator.feet_vel[:,:,:2], dim=-1)), dim=-1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_dof_vel_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.simulator.dof_vel), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_dof_pos_stand_still(self):
        # Penalize position deviation at zero commands
        return torch.sum(torch.square(self.simulator.dof_pos - self.simulator.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.1)
    
    def _reward_stand_still_contact(self):
        # Encourage feet contact with the ground at zero commands
        contacts = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 0.1
        full_contact = torch.sum(1.*contacts, dim=1)==len(self.simulator.feet_indices)
        return 1.0*full_contact * (torch.norm(self.commands[:, :3], dim=1) < 0.1)
    
    def _reward_dof_close_to_default(self):
        # Penalize dof position deviation from default
        return torch.sum(torch.square(self.simulator.dof_pos - self.simulator.default_dof_pos), dim=1)

    def _reward_alive_bonus(self):
        return ~self.reset_buf

    def _reward_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        """
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        # print(f"feet pos: {self.simulator.feet_pos[:, :, 2]}")
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(
                self.simulator.feet_pos[:, :, 2] -
                self.cfg.rewards.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)
    
    def _reward_foot_clearance_terrain_aware(self):
        """
        Encourage swing feet to reach a terrain-aware desired height,
        while softly discouraging excessive swing height.

        Assumes:
            self.simulator.feet_pos           : (N, 4, 3)
            self.simulator.feet_vel           : (N, 4, 3)
            self.simulator._height_around_feet: (N, 4, 3, 3) or (N, 4, 9)

        Uses:
            - terrain-aware target height
            - horizontal foot velocity weighting (same style as original reward)
            - excess-height penalty to prevent over-swinging
        """

        feet_z = self.simulator.feet_pos[:, :, 2]                       # (N,4)
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)  # (N,4)

        # Flatten 3x3 terrain patch if needed, then take local max height near each foot
        h_patch = self.simulator._height_around_feet
        if h_patch.ndim == 4:   # (N,4,3,3)
            h_patch = h_patch.view(h_patch.shape[0], h_patch.shape[1], -1)  # (N,4,9)

        local_terrain_h = torch.max(h_patch, dim=-1)[0]                # (N,4)

        # Terrain-aware desired foot height
        z_des = (
            self.cfg.rewards.foot_clearance_target
            + self.cfg.rewards.foot_height_offset
            + local_terrain_h
        )                                                               # (N,4)

        # Main tracking error: encourage feet to reach desired terrain-aware height
        track_err = torch.square(feet_z - z_des)                        # (N,4)

        # Soft over-swing penalty: only penalize when foot goes too far above desired height
        # Margin gives some freedom to overshoot a little during learning
        excess_margin = 0.04  # [m], tune: 0.03 - 0.06
        excess = torch.relu(feet_z - (z_des + excess_margin))           # (N,4)
        excess_err = torch.square(excess)

        # Weight excess penalty less than main tracking term
        excess_weight = 0.25  # tune: 0.1 - 0.5

        total_err = torch.sum(
            foot_vel_xy_norm * (track_err + excess_weight * excess_err),
            dim=-1
        )                                                               # (N,)

        return torch.exp(-total_err / self.cfg.rewards.foot_clearance_tracking_sigma)


    def _reward_front_foot_overreach(self):
        # Assumed order is FR/L, FL/R....
        front_1 = quat_rotate_inverse(
            self.simulator.base_quat,
            self.simulator.feet_pos[:, 0, :] - self.simulator.base_pos
        )  # (N,3)

        front_2 = quat_rotate_inverse(
            self.simulator.base_quat,
            self.simulator.feet_pos[:, 1, :] - self.simulator.base_pos
        )  # (N,3)

        front_x_1 = front_1[:, 0]   # (N,)
        front_x_2 = front_2[:, 0]   # (N,)

        front_x = torch.stack([front_x_1, front_x_2], dim=1)   # (N,2)

        
        overreach = torch.relu(front_x - self.cfg.rewards.overreach_x_max)
        
        # stance/contact gating
        contact = (
            self.simulator.link_contact_forces[:, self.simulator.feet_indices[:2], 2] > 5.0
        ).float()

        penalty = torch.sum(contact * overreach ** 2, dim=1)

        # total_mass = self.simulator._robot_mass + torch.clamp(self.simulator._added_base_mass, min=0.0)
        # _scales = (self.simulator._robot_mass / total_mass).squeeze(-1)
        
        # scales = 1.0 - 0.5 * _scales

        # return scales * penalty
        return penalty
    

    def _reward_support_polygon(self):
        """
        Positive support-region stability reward that supports:
        - 4/3 stance feet: polygon-like support using stance-center distance
        - 2 stance feet: line-segment support using distance to support line
        - <2 stance feet: reward = 0

        Assumptions:
            self.simulator.base_pos:            (N, 3)
            self.simulator.feet_pos:            (N, 4, 3)
            self.simulator.link_contact_forces: (N, n_links, 3)
            self.simulator.feet_indices:        4 foot link ids
            self.cfg.rewards.support_polygon_sigma: float
        """
        fz_thr = 5.0
        sigma = self.cfg.rewards.support_polygon_sigma

        base_xy = self.simulator.base_pos[:, :2]                      # (N,2)
        feet_xy = self.simulator.feet_pos[:, :, :2]                  # (N,4,2)

        contact = (
            self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > fz_thr
        ).float()                                                    # (N,4)

        n_stance = torch.sum(contact, dim=1)                         # (N,)

        rew = torch.zeros(base_xy.shape[0], device=base_xy.device)

        # ------------------------------------------------------------------
        # Case 1: 3 or more stance feet -> center/radius approximation
        # ------------------------------------------------------------------
        mask_poly = n_stance >= 3
        if torch.any(mask_poly):
            contact_poly = contact[mask_poly]
            feet_poly = feet_xy[mask_poly]
            base_poly = base_xy[mask_poly]

            stance_center = torch.sum(contact_poly.unsqueeze(-1) * feet_poly, dim=1) / (
                n_stance[mask_poly].unsqueeze(-1) + 1e-6
            )                                                        # (M,2)

            center_dist = torch.norm(base_poly - stance_center, dim=1)  # (M,)

            support_radius = torch.sum(
                contact_poly * torch.norm(feet_poly - stance_center.unsqueeze(1), dim=-1),
                dim=1
            ) / (n_stance[mask_poly] + 1e-6)                        # (M,)

            margin_frac = 0.60
            desired_dist = margin_frac * support_radius
            excess = torch.relu(center_dist - desired_dist)

            rew[mask_poly] = torch.exp(-(excess ** 2) / (sigma + 1e-6))

        # ------------------------------------------------------------------
        # Case 2: exactly 2 stance feet -> distance to support line segment
        # ------------------------------------------------------------------
        mask_line = n_stance == 2
        if torch.any(mask_line):
            contact_line = contact[mask_line]                       # (K,4)
            feet_line = feet_xy[mask_line]                         # (K,4,2)
            base_line = base_xy[mask_line]                         # (K,2)

            # indices of the two stance feet for each env
            idx = torch.nonzero(contact_line, as_tuple=False)      # (2K,2): [env_idx, foot_idx]
            foot_ids = idx[:, 1].view(-1, 2)                       # (K,2)

            batch_ids = torch.arange(feet_line.shape[0], device=feet_line.device)
            p1 = feet_line[batch_ids, foot_ids[:, 0]]              # (K,2)
            p2 = feet_line[batch_ids, foot_ids[:, 1]]              # (K,2)

            seg = p2 - p1                                          # (K,2)
            seg_len_sq = torch.sum(seg ** 2, dim=1, keepdim=True)  # (K,1)

            # projection of base onto line segment
            t = torch.sum((base_line - p1) * seg, dim=1, keepdim=True) / (seg_len_sq + 1e-6)
            t = torch.clamp(t, 0.0, 1.0)

            proj = p1 + t * seg                                    # (K,2)
            line_dist = torch.norm(base_line - proj, dim=1)        # (K,)

            # allow some lateral deviation relative to segment length
            seg_len = torch.sqrt(seg_len_sq.squeeze(-1) + 1e-6)    # (K,)
            desired_dist = 0.30 * seg_len
            excess = torch.relu(line_dist - desired_dist)

            rew[mask_line] = torch.exp(-(excess ** 2) / (sigma + 1e-6))

        # <2 stance feet -> reward stays 0
        return rew

    def _reward_pd_target_torque_limit(self):
        """
        Penalize joint position targets that would induce PD torques
        exceeding tau_max.

        Uses a quadratic hinge outside the admissible target range.
        """

        tau_max = [25.0, 25.0, 35.0, 25.0, 25.0, 35.0, 25.0, 25.0, 35.0, 25.0, 25.0, 35.0]  # [Nm]

        q      = self.simulator.dof_pos        # (N, dof)
        qdot   = self.simulator.dof_vel        # (N, dof)
        q_des  = self.actions[:, :12] * self.cfg.control.action_scale + self.simulator.default_dof_pos  # (N, dof)
    
        Kp = self.simulator._kp_scale * self.simulator._p_gains  # (N, dof)
        Kd = self.simulator._kd_scale * self.simulator._d_gains  # (N, dof)

        # compute admissible bounds
        q_lower = (Kd * qdot - tau_max) / Kp + q
        q_upper = (Kd * qdot + tau_max) / Kp + q

        # hinge penalties
        over_upper = F.softplus(q_des - q_upper)
        under_lower = F.softplus(q_lower - q_des)

        penalty = over_upper**2 + under_lower**2

        return torch.sum(penalty, dim=1)

    def _reward_foot_landing_vel(self):
        z_vels = self.simulator.feet_vel[:, :, 2]
        contacts = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 0.1
        about_to_land = ((self.simulator.feet_pos[:, :, 2] -
                          self.cfg.rewards.foot_height_offset) <
                         self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(
            about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward
    
    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
    
    def _reward_foot_acc(self):
        '''reward for foot acceleration'''
        foot_acc = (self.simulator.feet_vel - self.simulator.last_feet_vel) / self.dt
        return torch.sum(torch.square(foot_acc), dim=(1, 2))

    def _reward_sparse_contacts(self):
        fz = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2]
        contact_prob = torch.sigmoid(10.0*(fz - 10.0))
        num_contacts = torch.sum(contact_prob, dim=-1)
        
        return torch.exp(-torch.square(num_contacts - 2.0)) 
    
        # Curriculum, sensitive torque ratio reward and robust to several potential "failure" modes
    
    def _reward_aligned_torques(self):
        # prevent feedforward (or possibly PD) torques from learning to simply dominate and "cancel out"
        #    the torques produced from feedback control. 
        cos_sim = F.cosine_similarity(self.simulator.feedforward_torques, self.simulator.feedback_torques, dim=-1)
        aligned_torques_gate = torch.clamp(cos_sim, min=0.0)  # only reward non-conflicting torques

        return torch.exp(-4.0*aligned_torques_gate)
    
    def _reward_dof_tracking(self):
        # control_type = 'P'
        # Pull out the position control actions
        pos_actions = self.actions[:,0:12]

        # Scale the position actions
        actions_ = pos_actions * self.cfg.control.action_scale + self.simulator.default_dof_pos
        error = torch.sum(torch.square(actions_ - self.simulator.dof_pos), dim=-1)
        return torch.exp(-4.0*error)

    def _reward_hip_pos(self):
        """ Reward for the hip joint position close to default position
        """
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(torch.square(
            self.simulator.dof_pos[:, hip_joint_indices] - 
            self.simulator.default_dof_pos[:, hip_joint_indices]), dim=-1)
        return dof_pos_error
    
    def _potential_orientation(self):
        roll_pitch = self.simulator.projected_gravity[:, :2]
        return torch.exp(-torch.sum(roll_pitch**2, dim=1) / 0.5)
    
    def _potential_height(self):
        base_height = torch.mean(self.simulator.base_pos[:, 2].unsqueeze(
            1) - self.simulator.measured_heights, dim=1)
        h_po = torch.square(base_height - self.cfg.rewards.base_height_target)
        return torch.exp(-h_po / 0.5)

    def _reward_pbrs_orientation(self):
        phi_next = self._potential_orientation()
        shaping = phi_next - self.phi_prev_orientation

        # If env will reset after this step, terminate the telescoping sum cleanly.
        # Use zero potential for the absorbing terminal state.
        terminal_mask = self.reset_buf & (~self.time_out_buf)
        shaping[terminal_mask] = -self.phi_prev_orientation[terminal_mask]

        self.phi_prev_orientation = phi_next
        return shaping

    def _reward_pbrs_height(self):
        phi_next = self._potential_height()
        shaping = phi_next - self.phi_prev_height

        terminal_mask = self.reset_buf & (~self.time_out_buf)
        shaping[terminal_mask] = -self.phi_prev_height[terminal_mask]

        self.phi_prev_height = phi_next
        return shaping
    
    def _compute_vhip_angle(self):
        com_pos = self.simulator.base_pos[:,0:3]  # B x 3

        foot_contact_forces = self.simulator._link_contact_forces[:, self.simulator.feet_indices, :]    # B, num_feet, 3
        foot_positions = self.simulator.feet_pos

        normal_forces = foot_contact_forces[:,:,2:3]  # B. num_feet, 1

        total_force = torch.sum(normal_forces, dim=1).clamp(min=1e-6)  # B x 1  

        cop_pos =  torch.sum(foot_positions * normal_forces, dim=1) / total_force  # B x 3
        pendulum_length = torch.norm(com_pos - cop_pos, dim=1).clamp(min=1e-6)  # B x 1

        com_z = com_pos[:,2]   # B x 1
        cos_theta = torch.clamp(com_z / pendulum_length, -1.0, 1.0)
        theta = torch.acos(cos_theta)

        return theta
    
    def _compute_vhip_acceleration(self):
        theta = self._compute_vhip_angle()
        
        com_pos = self.simulator.base_pos[:,0:3]  # B x 3

        foot_contact_forces = self.simulator._link_contact_forces[:, self.simulator.feet_indices, :]    # B, num_feet, 3
        foot_positions = self.simulator.feet_pos

        normal_forces = foot_contact_forces[:,:,2:3]  # B. num_feet, 1

        total_force = torch.sum(normal_forces, dim=1).clamp(min=1e-6)  # B x 1  

        cop_pos =  torch.sum(foot_positions * normal_forces, dim=1) / total_force  # B x 3

        pendulum_length = torch.norm(com_pos - cop_pos, dim=1).clamp(min=1e-6)  # B x 1

        g = 9.81

        theta_ddot = -(g / pendulum_length) * torch.sin(theta)

        return theta_ddot

    def _reward_vhip_angle(self):
        theta = self._compute_vhip_angle()

        angle_error = torch.abs(theta) - 0.1

        angle_rew = torch.clamp(angle_error, min=0)

        return angle_rew
    
    def _reward_vhip_angular_acc(self):
        theta_ddot = self._compute_vhip_acceleration()

        acc_error = torch.abs(theta_ddot) - 0.001

        acc_rew = torch.clamp(acc_error, min=0.0)

        return acc_rew
    
    def _reward_rear_foot_overreach(self):
        """
        Penalize rear feet for being too far from their nominal rear-foot x location
        in the base frame, in either direction.

        Penalizes:
        - rear foot too far forward
        - rear foot too far backward

        Assumed foot order: FR, FL, RR, RL
        """

        # Rear feet in base frame
        rear_1 = quat_rotate_inverse(
            self.simulator.base_quat,
            self.simulator.feet_pos[:, 2, :] - self.simulator.base_pos
        )  # RR, (N,3)

        rear_2 = quat_rotate_inverse(
            self.simulator.base_quat,
            self.simulator.feet_pos[:, 3, :] - self.simulator.base_pos
        )  # RL, (N,3)

        rear_x = torch.stack([rear_1[:, 0], rear_2[:, 0]], dim=1)  # (N,2)

        # Nominal rear-foot x location in base frame.
        # This should usually be negative, e.g. -0.20 to -0.25 m.
        rear_x_nominal = self.cfg.rewards.rear_foot_x_nominal

        # Allowed deviation around nominal rear-foot x location.
        # Example: 0.08 m allows rear_x in [nominal - 0.08, nominal + 0.08].
        rear_x_margin = self.cfg.rewards.rear_foot_x_margin

        # Penalize both too far forward and too far backward relative to nominal.
        x_error = torch.abs(rear_x - rear_x_nominal)
        overreach = torch.relu(x_error - rear_x_margin)

        # Contact gate rear feet only: feet_indices[2:4], not [:2]
        contact = (
            self.simulator.link_contact_forces[:, self.simulator.feet_indices[2:4], 2] > 5.0
        ).float()  # (N,2)

        penalty = torch.sum(contact * overreach ** 2, dim=1)

        return penalty