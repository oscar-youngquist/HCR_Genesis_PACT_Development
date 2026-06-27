from legged_gym.envs.go2.go2_kite.kite_reward_helpers import _eig_desc, _geom_mean, _interval_reward, _safe_inv, _safe_normalize, _safe_pinv, _sanitize_tensor, _skew, _upper_reward
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
from legged_gym.envs.go2.kite_depth_mixin import KITEDepthMixin
from legged_gym.utils.math_utils import wrap_to_pi, torch_rand_float, quat_apply
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from ...base.legged_robot_config import LeggedRobotCfg
import torch.nn.functional as F

class Go2KITE(KITEDepthMixin, BaseTask):
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
        
        self.command_lin_tracking_ema = None
        self.command_ang_tracking_ema = None
        self.command_lin_best_tracking = 0.0
        self.command_ang_best_tracking = 0.0
        self.command_lin_required_tracking = self.cfg.commands.curriculum_min_lin_tracking
        self.command_ang_required_tracking = self.cfg.commands.curriculum_min_ang_tracking
        self.command_lin_tracking_history = deque(
            maxlen=self.cfg.commands.curriculum_best_window
        )
        self.command_ang_tracking_history = deque(
            maxlen=self.cfg.commands.curriculum_best_window
        )
        self.last_lin_update_idx = 0
        self.last_ang_update_idx = 0
        
        self._init_buffers()
        self._prepare_reward_function()
        self._print_forward_only_command_env_count()
        self.init_done = True

    def get_observations(self):
        # KITE training consumes proprioception plus separate visual/terrain
        # tensors. Keeping terrain maps separate prevents raw terrain from
        # being appended to the privileged critic observation.
        return self.obs_buf, self.obs_history, self.privileged_obs_buf, self.explicit_labels_buf, self.get_depth_observations(), self.depth_torso_state_buf, self.terrain_map_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, *_ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self._pre_depth_step()
        actions = self._pre_sim_step(actions)
        
        self.simulator.step(actions)
        
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs)
        
        # Return the next-step tensors used both for PPO rollout storage and
        # auxiliary encoder targets.
        return self.obs_buf, self.privileged_obs_buf, self.obs_history, self.explicit_labels_buf, self.get_depth_observations(), self.depth_torso_state_buf, self.terrain_map_buf, self.rew_buf, self.reset_buf, self.extras

    def set_camera(self, pos, lookat):
        """ Set camera position and direction
        """
        self.simulator._floating_camera.set_pose(
            pos=pos,
            lookat=lookat
        )

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

    def _cached_step_value(self, key, compute_fn):
        """Return a reward helper tensor recomputed at most once per physics step."""
        if self._reward_cache_step != self.common_step_counter:
            self._reward_cache.clear()
            self._reward_cache_step = self.common_step_counter
        if key not in self._reward_cache:
            self._reward_cache[key] = compute_fn()
        return self._reward_cache[key]

    def _feet_contact_fz(self):
        return self._cached_step_value(
            "feet_contact_fz",
            lambda: self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2],
        )

    def _feet_contact_mask(self):
        return self._cached_step_value(
            "feet_contact_mask",
            lambda: self._feet_contact_fz() > self.cfg.rewards.contact_force_threshold,
        )

    def _gait_order_tensor(self, values):
        """Return foot tensors in gait reward order [FL, FR, RL, RR]."""
        return values[:, self.gait_foot_order, ...]

    def _gait_contact_mask(self):
        contact = torch.abs(self._feet_contact_fz()) > self.cfg.rewards.contact_force_threshold
        return self._gait_order_tensor(contact)

    def _gait_terrain_relative_foot_height(self):
        feet_height = self.simulator.feet_pos[:, :, 2]
        if hasattr(self, "_local_terrain_height_under_feet"):
            feet_height = feet_height - self._local_terrain_height_under_feet()
        return self._gait_order_tensor(feet_height)

    def _update_gait_swing_statistics(self):
        """Update persistent per-env swing statistics once per env step."""
        contact = self._gait_contact_mask()
        swing = (~contact).float()

        alpha = self.cfg.rewards.swing_ema_alpha
        self.feet_swing_ema[:] = alpha * self.feet_swing_ema + (1.0 - alpha) * swing

        foot_height = self._gait_terrain_relative_foot_height()
        self.feet_swing_peak_height[:] = torch.where(
            swing.bool(),
            torch.maximum(self.feet_swing_peak_height, foot_height),
            self.feet_swing_peak_height,
        )

        touchdown = (~self.prev_feet_contact) & contact
        height_alpha = self.cfg.rewards.swing_height_ema_alpha
        updated_height_ema = (
            height_alpha * self.feet_swing_height_ema
            + (1.0 - height_alpha) * self.feet_swing_peak_height
        )
        self.feet_swing_height_ema[:] = torch.where(
            touchdown,
            updated_height_ema,
            self.feet_swing_height_ema,
        )
        self.feet_swing_peak_height[:] = torch.where(
            touchdown,
            torch.zeros_like(self.feet_swing_peak_height),
            self.feet_swing_peak_height,
        )
        self.prev_feet_contact[:] = contact

    def _command_norm(self, cols):
        return self._cached_step_value(
            f"command_norm_{cols}",
            lambda: torch.norm(self.commands[:, :cols], dim=1),
        )

    def _lin_vel_tracking_error(self):
        return self._cached_step_value(
            "lin_vel_tracking_error",
            lambda: torch.sum(
                torch.square(
                    self.commands[:, :2] - self.simulator.base_lin_vel[:, :2]
                ),
                dim=1,
            ),
        )

    def _ang_vel_tracking_error(self):
        return self._cached_step_value(
            "ang_vel_tracking_error",
            lambda: torch.square(
                self.commands[:, 2] - self.simulator.base_ang_vel[:, 2]
            ),
        )

    def _base_height_over_terrain(self):
        return self._cached_step_value(
            "base_height_over_terrain",
            lambda: torch.mean(
                self.simulator.base_pos[:, 2].unsqueeze(1)
                - self.simulator.measured_heights,
                dim=1,
            ),
        )

    def _scaled_pos_actions(self):
        return self._cached_step_value(
            "scaled_pos_actions",
            lambda: self.actions[:, 0:12] * self.cfg.control.action_scale
            + self.simulator.default_dof_pos,
        )

    def _joint_power_per_dof(self):
        return self._cached_step_value(
            "joint_power_per_dof",
            lambda: self.simulator.torques * self.simulator.dof_vel,
        )

    def _feet_vel_xy_norm(self):
        return self._cached_step_value(
            "feet_vel_xy_norm",
            lambda: torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1),
        )

    def _local_terrain_height_under_feet(self):
        def compute():
            h_patch = self.simulator._height_around_feet
            if h_patch.ndim == 4:
                h_patch = h_patch.view(h_patch.shape[0], h_patch.shape[1], -1)
            return torch.max(h_patch, dim=-1)[0]
        return self._cached_step_value("local_terrain_height_under_feet", compute)

    def _terrain_aware_foot_target_height(self):
        return self._cached_step_value(
            "terrain_aware_foot_target_height",
            lambda: (
                self.cfg.rewards.foot_clearance_target
                + self.cfg.rewards.foot_height_offset
                + self._local_terrain_height_under_feet()
            ),
        )

    def _feet_pos_base_frame(self):
        def compute():
            feet_rel = self.simulator.feet_pos - self.simulator.base_pos[:, None, :]
            return torch.stack(
                [
                    quat_rotate_inverse(self.simulator.base_quat, feet_rel[:, foot_id, :])
                    for foot_id in range(feet_rel.shape[1])
                ],
                dim=1,
            )
        return self._cached_step_value("feet_pos_base_frame", compute)

    def _vhip_terms(self):
        def compute():
            com_pos = self.simulator.base_pos[:, :3]
            normal_forces = self.simulator.link_contact_forces[
                :, self.simulator.feet_indices, 2:3
            ]
            total_force = torch.sum(normal_forces, dim=1).clamp(min=1e-6)
            cop_pos = torch.sum(self.simulator.feet_pos * normal_forces, dim=1) / total_force
            pendulum_length = torch.norm(com_pos - cop_pos, dim=1).clamp(min=1e-6)
            cos_theta = torch.clamp(com_pos[:, 2] / pendulum_length, -1.0, 1.0)
            theta = torch.acos(cos_theta)
            angular_acc = -(9.81 / pendulum_length) * torch.sin(theta)
            return theta, angular_acc
        return self._cached_step_value("vhip_terms", compute)

    def _feet_near_edge_mask(self):
        return self._cached_step_value(
            "feet_near_edge",
            lambda: self.simulator.calc_feet_near_edge(),
        )

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
        
        self._update_gait_swing_statistics()
        self.compute_reward()
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        if self.cfg.sensor.add_depth:
            self.simulator.update_depth_images()
            self._update_depth_observations()
        
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        # KITE specific update
        self.leg_jacobians[:] = self.compute_all_leg_jacobians(self.simulator.dof_pos.view(-1, 4, 3))
        
        if (
            self.debug
            or (
                not self.headless
                and (
                    self.cfg.sensor.depth_camera_config.debug_draw_camera_position
                    or self.cfg.terrain.debug_draw_measured_surface_normals
                )
            )
        ):
            self.simulator.draw_debug_vis()

    def compute_all_leg_jacobians(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute translational Jacobians for all 4 legs.

        Args:
            q:
                Joint angles for all legs.
                Shape (N, 4, 3), ordered per leg as [abad, hip, knee].

        Returns:
            J:
                Batched translational Jacobians in the base frame.
                Shape (N, 4, 3, 3)

        Required config entries
        -----------------------
        self.cfg.robot.abad_link_length
        self.cfg.robot.hip_link_length
        self.cfg.robot.knee_link_length
        self.cfg.robot.knee_link_y_offset
        self.cfg.robot.side_signs   # e.g. [1.0, -1.0, 1.0, -1.0]
        """
        assert q.ndim == 3 and q.shape[1:] == (4, 3), f"Expected q shape (N, 4, 3), got {q.shape}"
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

        dtype = q.dtype
        device = q.device
        N = q.shape[0]

        l1 = torch.as_tensor(self.cfg.asset.abad_link_length, device=device, dtype=dtype)
        l2 = torch.as_tensor(self.cfg.asset.hip_link_length, device=device, dtype=dtype)
        l3 = torch.as_tensor(self.cfg.asset.knee_link_length, device=device, dtype=dtype)
        l4 = torch.as_tensor(self.cfg.asset.knee_link_y_offset, device=device, dtype=dtype)

        side_sign = torch.as_tensor(
            self.cfg.asset.side_signs,
            device=device,
            dtype=dtype,
        ).view(1, 4).expand(N, 4)  # (N, 4)
        side_sign = torch.nan_to_num(side_sign, nan=1.0, posinf=1.0, neginf=-1.0)

        q0 = q[:, :, 0]  # abad / hip roll
        q1 = q[:, :, 1]  # thigh / hip pitch
        q2 = q[:, :, 2]  # calf / knee

        s0 = torch.sin(q0)
        c0 = torch.cos(q0)

        s1 = torch.sin(q1)
        c1 = torch.cos(q1)

        s12 = torch.sin(q1 + q2)
        c12 = torch.cos(q1 + q2)

        C = l2 * c1 + l3 * c12
        S = l2 * s1 + l3 * s12

        J = torch.zeros(N, 4, 3, 3, device=device, dtype=dtype)

        # x row
        J[:, :, 0, 0] = 0.0
        J[:, :, 0, 1] = -C
        J[:, :, 0, 2] = -l3 * c12

        # y row
        J[:, :, 1, 0] = -side_sign * l1 * s0 + C * c0
        J[:, :, 1, 1] = -S * s0
        J[:, :, 1, 2] = -l3 * s12 * s0

        # z row
        J[:, :, 2, 0] = side_sign * l1 * c0 + C * s0
        J[:, :, 2, 1] = S * c0
        J[:, :, 2, 2] = l3 * s12 * c0

        return torch.nan_to_num(J, nan=0.0, posinf=1e6, neginf=-1e6)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        fail_buf = torch.any(
            torch.norm(self.simulator.link_contact_forces[:, self.simulator.termination_contact_indices, :], dim=-1)
            > self.cfg.rewards.contact_force_threshold, dim=1)
        # print(f"contact termination: {fail_buf}")
        # fail_buf |= self.simulator.projected_gravity[:, 2] > self.cfg.rewards.max_projected_gravity
        # print(f"gravity termination: {self.simulator.projected_gravity[:, 2] > self.cfg.rewards.max_projected_gravity}")
        
        if hasattr(self.cfg, "termination"):
            # more sophisticated termination conditions
            rpy = self.simulator._base_euler
            r, p = wrap_to_pi(rpy[:,0]), wrap_to_pi(rpy[:,1])
            base_height = self._base_height_over_terrain()
            
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

        self.gap_reset_buf = self._check_unrecoverable_gap()
        
        self.fail_buf += fail_buf
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
            | self.gap_reset_buf
        )

    def _check_unrecoverable_gap(self):
        if (
            not getattr(self.cfg.termination, "reset_unrecoverable_gaps", False)
            or self.cfg.terrain.mesh_type not in ("heightfield", "trimesh")
            or not self.cfg.terrain.obtain_terrain_info_around_feet
        ):
            self.gap_fall_counter.zero_()
            return torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        support_height = self.simulator.env_origins[:, 2].unsqueeze(1)
        deep_void = self.simulator.gap_void_under_feet
        fallen_feet = deep_void & (
            self.simulator.feet_pos[:, :, 2]
            < support_height - self.cfg.termination.gap_foot_drop_threshold
        )
        enough_fallen_feet = (
            fallen_feet.sum(dim=1)
            >= self.cfg.termination.gap_min_fallen_feet
        )
        base_fallen = deep_void.any(dim=1) & (
            self.simulator.base_pos[:, 2]
            < self.simulator.env_origins[:, 2]
            - self.cfg.termination.gap_base_drop_threshold
        )
        falling_into_gap = enough_fallen_feet | base_fallen

        self.gap_fall_counter = torch.where(
            falling_into_gap,
            self.gap_fall_counter + 1,
            torch.zeros_like(self.gap_fall_counter),
        )
        return (
            self.gap_fall_counter
            >= self.cfg.termination.gap_reset_steps
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
        if self.cfg.commands.curriculum:
            self._update_command_curriculum(env_ids)

        # Update the position/torque control tradeoff curriculum 
        if self.use_tradeoff:
            self.step_tradeoff_curriculum(env_ids)

        self._resample_commands(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.simulator.reset_idx(env_ids)
        self._reset_depth_buffers(env_ids)

        self.phi_prev_orientation[env_ids] = self._potential_orientation()[env_ids]
        self.phi_prev_height[env_ids] = self._potential_height()[env_ids]

        # reset buffers
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.feet_swing_ema[env_ids] = 0.
        self.feet_swing_peak_height[env_ids] = 0.
        self.feet_swing_height_ema[env_ids] = 0.
        self.prev_feet_contact[env_ids] = True
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        self.gap_fall_counter[env_ids] = 0

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
            self.extras["episode"]["max_command_y"] = self.command_ranges["lin_vel_y"][1]
            self.extras["episode"]["max_command_yaw"] = self.command_ranges["ang_vel_yaw"][1]
            self.extras["episode"]["command_lin_tracking_ema"] = (
                self.command_lin_tracking_ema or 0.0
            )
            self.extras["episode"]["command_ang_tracking_ema"] = (
                self.command_ang_tracking_ema or 0.0
            )
            self.extras["episode"]["command_lin_best_tracking"] = (
                self.command_lin_best_tracking
            )
            self.extras["episode"]["command_ang_best_tracking"] = (
                self.command_ang_best_tracking
            )
            self.extras["episode"]["command_lin_required_tracking"] = (
                self.command_lin_required_tracking
            )
            self.extras["episode"]["command_ang_required_tracking"] = (
                self.command_ang_required_tracking
            )
            self.extras["episode"]["command_lin_vel_x_bias_progress"] = (
                self._get_lin_vel_x_bias_progress()
            )
            self.extras["episode"]["command_lin_vel_x_forward_bias"] = (
                self._get_lin_vel_x_forward_bias()
            )
            self.extras["episode"]["command_lin_vel_x_high_speed_power"] = (
                self._get_lin_vel_x_high_speed_power()
            )
        self.extras["episode"]["gap_reset"] = self.gap_reset_buf[env_ids].float().mean()
        if self.cfg.domain_rand.use_domainrand_curriculum:
            phase_to_idx = {
                "joint_dynamics": 0.0,
                "mass_com": 1.0,
                "disturbance": 2.0,
                "complete": 3.0,
            }
            self.extras["episode"]["domain_rand_phase"] = phase_to_idx.get(
                self.simulator.domain_rand_phase, -1.0
            )
            self.extras["episode"]["domain_rand_joint_dynamics_progress"] = (
                self.simulator.domain_rand_joint_dynamics_progress
            )
            self.extras["episode"]["domain_rand_mass_com_progress"] = (
                self.simulator.domain_rand_mass_com_progress
            )
            self.extras["episode"]["domain_rand_disturbance_progress"] = (
                self.simulator.domain_rand_disturbance_progress
            )
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
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,                     # torso linear velocity         3
            self.simulator.link_contact_states[:,self.simulator.feet_indices],         # contact states of feet        4
            torch.clip(self.simulator.feet_pos[:, :, 2] -
                torch.mean(self.simulator.height_around_feet, dim=-1) -
                self.cfg.rewards.foot_height_offset, -1, 1.),                              # feet height               4
            self.simulator.normal_vector_around_feet.reshape(self.num_envs, -1)        # 12 - terrain info around feet
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
            self.simulator._joint_stiffness,                                 # 1
            ), dim=-1)                                                       # 51

        critic_obs = torch.cat(
            (
                self.obs_buf,                                                          # 45 - standard policy observation
                torch.mean(self.simulator.base_pos[:, 2].unsqueeze(1) - 
                           self.simulator.measured_heights, dim=1, keepdim=True),      # 1  - base height
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,                 # 3  - base linear velocity 
                self.simulator._grfs_buf * self.obs_scales.grf,                        # 12 - ground reaction forces experienced by feet
                self.simulator.normal_vector_around_feet.reshape(self.num_envs, -1),   # 12 - terrain (surface normals) around feet
                self.simulator.link_contact_states[:,self.simulator.feet_indices],     # 4  - contact states of feet
                torch.clip(self.simulator.feet_pos[:, :, 2] -
                    torch.mean(self.simulator.height_around_feet, dim=-1) -
                    self.cfg.rewards.foot_height_offset, -1, 1.),                      # 4  - feet height
                domain_randomization_info                                              # 51 - privileged domain randomization values
            ),
            dim=-1,
        ) # 132

        # add hieght measurements to asymmetric critic if approperiate
        self._update_privileged_terrain_map()
        self._update_depth_torso_state()

        self.critic_obs_deque.append(critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )

    def _update_privileged_terrain_map(self):
        """Build the privileged terrain target map for encoder supervision.

        The map is B x H x W x 4 with channels:
            [height, normal_x, normal_y, normal_z].
        """
        if not self.cfg.terrain.measure_heights:
            self.terrain_map_buf.zero_()
            self.terrain_map_buf[..., 3] = 1.0
            return

        num_x = len(self.cfg.terrain.measured_points_x)
        num_y = len(self.cfg.terrain.measured_points_y)
        heights = torch.clip(
            self.simulator.base_pos[:, 2].unsqueeze(1)
            - 0.5
            - self.simulator.measured_heights,
            -1,
            1,
        ) * self.obs_scales.height_measurements
        heights *= self.height_noise_vec
        height_map = heights.view(self.num_envs, num_x, num_y, 1)
        normal_map = self.simulator.measured_surface_normals.view(
            self.num_envs,
            num_x,
            num_y,
            3,
        )
        self.terrain_map_buf = torch.cat((height_map, normal_map), dim=-1)

    def _update_depth_torso_state(self):
        """Collect IMU fields for the depth encoder's 8D torso state.

        Columns 2:5 start as simulator ground-truth linear velocity. The
        runner overwrites them with the latest modality-mixer estimate only
        when the reconstruction boot gate says the learned state is reliable.
        Final order is:
            [roll, pitch, v_x, v_y, v_z, gyro_x, gyro_y, gyro_z].
        """
        roll_pitch = get_euler_xyz(self.simulator.base_quat)[:, :2]
        self.depth_torso_state_buf = torch.cat(
            (
                roll_pitch,
                self.simulator.base_lin_vel,
                self.simulator.base_ang_vel,
            ),
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
        move_up = distance > (self.simulator._terrain.env_length / 3.0)
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
        base_quat = quat_from_euler_xyz(
            torch_rand_float(
                -self.cfg.init_state.roll_random_scale,
                self.cfg.init_state.roll_random_scale,
                (len(env_ids), 1),
                self.device,
            ).squeeze(1),
            torch_rand_float(
                -self.cfg.init_state.pitch_random_scale,
                self.cfg.init_state.pitch_random_scale,
                (len(env_ids), 1),
                self.device,
            ).squeeze(1),
            torch_rand_float(
                -self.cfg.init_state.yaw_random_scale,
                self.cfg.init_state.yaw_random_scale,
                (len(env_ids), 1),
                self.device,
            ).squeeze(1),
        )
        # base lin vel
        base_lin_vel = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        # base ang vel
        base_ang_vel = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        
        self.simulator.reset_root_states(env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            self.episode_length_buf % self.command_resample_timeouts == 0
        ).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.simulator.base_quat, self.forward_vec)
            self.heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - self.heading), self.cfg.commands.ranges.ang_vel_yaw[0], 
                                                                 self.cfg.commands.ranges.ang_vel_yaw[1])

        if self.cfg.domain_rand.push_robots:
            self.simulator.push_robots()
        self._resample_depth_latency()
        
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = self._sample_lin_vel_x_commands(env_ids)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids),1), self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        forward_only_mask = self._get_forward_only_command_mask(env_ids)
        if forward_only_mask.any():
            forward_only_env_ids = env_ids[forward_only_mask]
            self.commands[forward_only_env_ids, 0] = self._sample_forward_only_lin_vel_x_commands(
                len(forward_only_env_ids)
            )
            self.commands[forward_only_env_ids, 1] = 0.0
            if self.cfg.commands.heading_command:
                self.commands[forward_only_env_ids, 3] = torch_rand_float(-0.349, 0.349, (len(forward_only_env_ids), 1), device=self.device).squeeze(1)  # +/- 20 degrees
            else:
                self.commands[forward_only_env_ids, 2] = 0.0

        # Set small commands to zero using the same threshold that gates
        # stand-still rewards, so the policy sees one consistent definition of
        # "zero command" during command sampling and reward computation.
        zero_command_threshold = self.cfg.commands.zero_command_threshold
        command_is_active = (
            torch.norm(self.commands[env_ids, :3], dim=1)
            > zero_command_threshold
        )
        command_is_active = command_is_active | forward_only_mask
        self.commands[env_ids, :3] *= command_is_active.unsqueeze(1)

        self.command_resample_timeouts[env_ids] = self._sample_command_resample_timeouts(len(env_ids))

    def _get_forward_only_command_mask(self, env_ids):
        """Return envs on obstacle terrain columns that should only receive forward commands."""
        if (
            len(env_ids) == 0
            or self.forward_only_command_terrain_kind_ids.numel() == 0
        ):
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        terrain_kind_ids = getattr(self.simulator, "terrain_kind_ids", None)
        if terrain_kind_ids is not None:
            env_terrain_kind_ids = terrain_kind_ids[env_ids]
            return (
                env_terrain_kind_ids.unsqueeze(1)
                == self.forward_only_command_terrain_kind_ids.unsqueeze(0)
            ).any(dim=1)

        if (
            self.forward_only_command_terrain_types.numel() == 0
            or not hasattr(self.simulator, "terrain_types")
        ):
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        terrain_types = self.simulator.terrain_types[env_ids]
        return (
            terrain_types.unsqueeze(1)
            == self.forward_only_command_terrain_types.unsqueeze(0)
        ).any(dim=1)

    def _sample_forward_only_lin_vel_x_commands(self, num_envs):
        """Sample strictly positive forward speed commands for obstacle-focused terrains."""
        lin_vel_x_max = max(
            float(self.command_ranges["lin_vel_x"][1]),
            self.forward_only_min_lin_vel_x,
        )
        return torch_rand_float(
            self.forward_only_min_lin_vel_x,
            lin_vel_x_max,
            (num_envs, 1),
            self.device,
        ).squeeze(1)

    def _print_forward_only_command_env_count(self):
        """Print how many envs use obstacle-focused forward-only commands."""
        if self.num_envs == 0:
            return
        env_ids = torch.arange(self.num_envs, device=self.device)
        forward_only_mask = self._get_forward_only_command_mask(env_ids)
        num_forward_only = int(forward_only_mask.sum().item())
        percent = 100.0 * num_forward_only / max(1, self.num_envs)
        print(
            "KITE forward-only command envs: "
            f"{num_forward_only}/{self.num_envs} ({percent:.1f}%) "
            "will sample positive x velocity, zero y velocity, and minimal heading change.",
            flush=True,
        )

    def _build_forward_only_command_terrain_kind_ids(self, sim_device):
        """Return terrain generator branch ids that use obstacle-focused commands."""
        return torch.tensor(
            Terrain.FORWARD_ONLY_COMMAND_KIND_IDS,
            device=sim_device,
            dtype=torch.long,
        )

    def _build_forward_only_command_terrain_types(self, sim_device):
        """Map terrain curriculum columns to obstacle terrain branches."""
        terrain_names = (
            "gap_terrain",
            "pit_terrain",
            "multiple_high_platforms_terrain",
            "high_platform_gaps_terrain",
        )
        if getattr(self.cfg.terrain, "selected", False):
            terrain_type = getattr(self.cfg.terrain, "terrain_kwargs", {}).get(
                "type", ""
            )
            if any(name in terrain_type for name in terrain_names):
                return torch.arange(
                    self.cfg.terrain.num_cols, device=sim_device, dtype=torch.long
                )
            return torch.empty(0, device=sim_device, dtype=torch.long)

        obstacle_terrain_indices = set(Terrain.FORWARD_ONLY_COMMAND_KIND_IDS)
        proportions = np.cumsum(self.cfg.terrain.terrain_proportions)
        terrain_type_ids = []
        lower_bound = 0.0
        for terrain_idx, upper_bound in enumerate(proportions):
            if terrain_idx in obstacle_terrain_indices:
                for terrain_col in range(self.cfg.terrain.num_cols):
                    choice = terrain_col / self.cfg.terrain.num_cols + 0.001
                    if lower_bound <= choice < upper_bound:
                        terrain_type_ids.append(terrain_col)
            lower_bound = upper_bound

        return torch.tensor(
            sorted(set(terrain_type_ids)), device=sim_device, dtype=torch.long
        )

    def _get_lin_vel_x_bias_progress(self):
        initial_max = self.initial_command_ranges["lin_vel_x"][1]
        final_max = self.cfg.commands.max_curriculum
        span = max(final_max - initial_max, 1e-6)
        progress = (self.command_ranges["lin_vel_x"][1] - initial_max) / span
        return float(np.clip(progress, 0.0, 1.0))

    def _get_lin_vel_x_forward_bias(self):
        if not getattr(self.cfg.commands, "bias_lin_vel_x_with_curriculum", False):
            return 0.0
        progress = self._get_lin_vel_x_bias_progress()
        final_bias = getattr(
            self.cfg.commands, "lin_vel_x_forward_bias_final", 0.0
        )
        return float(np.clip(progress * final_bias, 0.0, 1.0))

    def _get_lin_vel_x_high_speed_power(self):
        progress = self._get_lin_vel_x_bias_progress()
        final_power = getattr(
            self.cfg.commands,
            "lin_vel_x_high_speed_bias_power_final",
            1.0,
        )
        return float((1.0 - progress) + progress * final_power)

    def _sample_lin_vel_x_commands(self, env_ids):
        num_envs = len(env_ids)
        lin_vel_x_min = self.command_ranges["lin_vel_x"][0]
        lin_vel_x_max = self.command_ranges["lin_vel_x"][1]
        samples = torch_rand_float(
            lin_vel_x_min,
            lin_vel_x_max,
            (num_envs, 1),
            self.device,
        ).squeeze(1)

        forward_bias = self._get_lin_vel_x_forward_bias()
        if forward_bias <= 0.0 or lin_vel_x_max <= 0.0:
            return samples

        positive_min = max(0.0, lin_vel_x_min)
        high_speed_power = self._get_lin_vel_x_high_speed_power()
        biased_unit = torch.rand(num_envs, device=self.device).pow(
            high_speed_power
        )
        biased_samples = positive_min + (
            lin_vel_x_max - positive_min
        ) * biased_unit
        use_biased_sample = torch.rand(num_envs, device=self.device) < forward_bias
        return torch.where(use_biased_sample, biased_samples, samples)

    def _update_command_curriculum(self, env_ids):
        """Update linear and angular command ranges using independent EMA gates."""
        if not self.init_done:
            return

        # Ignore very short episodes: their partial reward sums are too noisy to
        # represent sustained command-tracking performance.
        min_episode_steps = (
            self.cfg.commands.curriculum_min_episode_fraction
            * self.max_episode_length
        )
        valid_ids = env_ids[self.episode_length_buf[env_ids] >= min_episode_steps]
        if len(valid_ids) == 0:
            return

        def update_tracking_state(reward_name, ema, history, minimum):
            scale = self.reward_scales.get(reward_name, 0.0)
            if scale <= 0.0:
                return ema, 0.0, minimum

            # Remove episode length and the active reward scale. Since each raw
            # tracking reward is bounded by 1, this is the attained fraction of
            # the maximum possible tracking reward for each episode.
            episode_steps = self.episode_length_buf[valid_ids].float().clamp(min=1.0)
            normalized = (
                self.episode_sums[reward_name][valid_ids]
                / (episode_steps * scale)
            ).clamp(0.0, 1.0)
            sample = normalized.mean().item()

            # Smooth reset-batch performance so a single unusually good or bad
            # batch cannot immediately advance or stall the curriculum.
            alpha = self.cfg.commands.curriculum_ema_alpha
            ema = sample if ema is None else (1.0 - alpha) * ema + alpha * sample
            history.append(ema)

            # Estimate demonstrated best performance robustly. Use the maximum
            # during startup, then a high quantile to reject isolated spikes.
            values = np.asarray(history, dtype=np.float32)
            if len(values) < 10:
                best = float(values.max())
            else:
                best = float(
                    np.quantile(
                        values, self.cfg.commands.curriculum_best_quantile
                    )
                )

            # Require both an absolute level of competence and recovery toward
            # the best EMA retained from the current/recent command ranges.
            required = max(
                minimum,
                self.cfg.commands.curriculum_recovery_ratio * best,
            )
            return ema, best, required

        def command_curriculum_terrain_gate_open(command_name, cutoff_name, resume_name):
            """Pause command growth after a cutoff until terrain difficulty catches up."""
            cutoff = getattr(
                self.cfg.commands,
                cutoff_name,
                float("inf"),
            )
            resume_level = getattr(
                self.cfg.commands,
                resume_name,
                0.0,
            )
            if self.command_ranges[command_name][1] < cutoff:
                return True
            if not hasattr(self.simulator, "terrain_levels"):
                return True
            terrain_level = torch.mean(self.simulator.terrain_levels.float()).item()
            return terrain_level >= resume_level

        # Linear and angular tracking have separate EMAs and recovery targets,
        # allowing one command family to progress without waiting for the other.
        (
            self.command_lin_tracking_ema,
            self.command_lin_best_tracking,
            self.command_lin_required_tracking,
        ) = update_tracking_state(
            "tracking_lin_vel",
            self.command_lin_tracking_ema,
            self.command_lin_tracking_history,
            self.cfg.commands.curriculum_min_lin_tracking,
        )
        (
            self.command_ang_tracking_ema,
            self.command_ang_best_tracking,
            self.command_ang_required_tracking,
        ) = update_tracking_state(
            "tracking_ang_vel",
            self.command_ang_tracking_ema,
            self.command_ang_tracking_history,
            self.cfg.commands.curriculum_min_ang_tracking,
        )

        # Expand linear ranges only after performance has recovered and the
        # control-timestep cooldown since the previous linear update has elapsed.
        update_interval = self.cfg.commands.curriculum_update_interval_steps
        if (
            self.command_lin_tracking_ema is not None
            and self.command_lin_tracking_ema
            >= self.command_lin_required_tracking
            and self.common_step_counter - self.last_lin_update_idx
            >= update_interval
            and command_curriculum_terrain_gate_open(
                "lin_vel_x",
                "lin_vel_x_terrain_gate_cutoff",
                "lin_vel_x_terrain_gate_resume_level",
            )
        ):
            self.last_lin_update_idx = self.common_step_counter
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0]
                - self.cfg.commands.lin_vel_x_step,
                -self.cfg.commands.max_curriculum,
                0.0,
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1]
                + self.cfg.commands.lin_vel_x_step,
                0.0,
                self.cfg.commands.max_curriculum,
            )
            self.command_ranges["lin_vel_y"][0] = np.clip(
                self.command_ranges["lin_vel_y"][0]
                - self.cfg.commands.lin_vel_y_step,
                -self.cfg.commands.max_lin_vel_y,
                0.0,
            )
            self.command_ranges["lin_vel_y"][1] = np.clip(
                self.command_ranges["lin_vel_y"][1]
                + self.cfg.commands.lin_vel_y_step,
                0.0,
                self.cfg.commands.max_lin_vel_y,
            )

        # Apply the same two-part gate independently to the yaw-rate range.
        if (
            self.command_ang_tracking_ema is not None
            and self.command_ang_tracking_ema
            >= self.command_ang_required_tracking
            and self.common_step_counter - self.last_ang_update_idx
            >= update_interval
            and command_curriculum_terrain_gate_open(
                "ang_vel_yaw",
                "ang_vel_yaw_terrain_gate_cutoff",
                "ang_vel_yaw_terrain_gate_resume_level",
            )
        ):
            self.last_ang_update_idx = self.common_step_counter
            self.command_ranges["ang_vel_yaw"][0] = np.clip(
                self.command_ranges["ang_vel_yaw"][0]
                - self.cfg.commands.ang_vel_yaw_step,
                -self.cfg.commands.max_ang_vel_yaw,
                0.0,
            )
            self.command_ranges["ang_vel_yaw"][1] = np.clip(
                self.command_ranges["ang_vel_yaw"][1]
                + self.cfg.commands.ang_vel_yaw_step,
                0.0,
                self.cfg.commands.max_ang_vel_yaw,
            )


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
        # # previous joint torque actions
        # noise_vec[45:57] = 0.

        if self.cfg.terrain.measure_heights:
            self.height_noise_vec[:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.common_step_counter = 0
        self.extras = {}
        self._reward_cache_step = -1
        self._reward_cache = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.phi_prev_orientation = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.phi_prev_height = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        
        self.forward_vec[:, 0] = 1.0
        self.fail_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.gap_fall_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.gap_reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        
        self.commands = torch.zeros(
            (self.num_envs, self.cfg.commands.num_commands), device=self.device, dtype=torch.float)
        self.command_resample_timeouts = self._sample_command_resample_timeouts(self.num_envs)
        
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, dtype=torch.float,
                                           requires_grad=False)
        self.pd_target_tau_max = torch.as_tensor(
            self.cfg.rewards.pd_target_tau_max,
            device=self.device,
            dtype=torch.float,
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.last_actions = torch.zeros_like(self.actions)
        self.llast_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # last last actions
        
        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.simulator.feet_indices)), device=self.device, dtype=torch.float)
        self.last_contacts = torch.zeros((self.num_envs, len(self.simulator.feet_indices)), device=self.device, dtype=torch.int)
        foot_names = list(getattr(self.cfg.asset, "foot_name", ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]))
        gait_order_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        gait_order = [foot_names.index(name) for name in gait_order_names]
        self.gait_foot_order = torch.as_tensor(gait_order, device=self.device, dtype=torch.long)
        self.feet_swing_ema = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float
        )
        self.feet_swing_peak_height = torch.zeros_like(self.feet_swing_ema)
        self.feet_swing_height_ema = torch.zeros_like(self.feet_swing_ema)
        self.prev_feet_contact = torch.ones(
            (self.num_envs, 4), device=self.device, dtype=torch.bool
        )

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
        
        self.leg_jacobians = torch.zeros((self.num_envs, 4, 3, 3), dtype=torch.float, device=self.device)
        # IMU/depth-conditioning state for MotionRobustDepthEncoder. Columns
        # 2:5 are simulator linear velocity unless the runner's boot gate
        # replaces them with the latest modality-mixer velocity estimate.
        self.depth_torso_state_buf = torch.zeros(
            (self.num_envs, 8),
            dtype=torch.float,
            device=self.device,
        )
        # Privileged terrain target used by TerrainAttentionEncoder and
        # TerrainTwoHeadDecoder. Channel order is height + XYZ normal.
        self.terrain_map_buf = torch.zeros(
            (
                self.num_envs,
                len(self.cfg.terrain.measured_points_x),
                len(self.cfg.terrain.measured_points_y),
                4,
            ),
            dtype=torch.float,
            device=self.device,
        )
        self.terrain_map_buf[..., 3] = 1.0

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
        self._init_depth_processing()
            

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
        # Gradually increase the regularization strength via cosine annealing.
        elif num_iters >= self.reward_warmup_steps and (num_iters - self.reward_warmup_steps) < self.reward_curr_steps:
            print("Stepping Reward Curriculum")
            adjusted_iter = num_iters - self.reward_warmup_steps
            for key in self.reward_curr_keys:
                if key in self.reward_scales.keys():
                    start, end = self.reward_curr_bounds[key]
                    alpha = np.clip(adjusted_iter / self.reward_curr_steps, 0.0, 1.0)
                    ramp = 0.5 * (1.0 - np.cos(np.pi * alpha))
                    self.reward_scales[key] = (start + (end - start) * ramp) * self.dt
        # Fix the regularization strength to the upper-bound
        else:
            # by default set the reward to the upper bound
            for key in self.reward_curr_keys:
                if key in self.reward_scales.keys():
                    self.reward_scales[key] = self.reward_curr_bounds[key][1] * self.dt

    def step_command_resampling_time_curriculum(self, num_iters):
        if not self.use_command_resampling_time_curriculum:
            self.randomize_command_resampling_time = self.enable_command_resampling_time_randomization
            return

        self.randomize_command_resampling_time = (
            num_iters >= self.command_resampling_time_warmup_iters
        )


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
        self.initial_command_ranges = {
            key: value.copy() if isinstance(value, list) else value
            for key, value in self.command_ranges.items()
        }
        self.enable_command_resampling_time_randomization = getattr(
            self.cfg.commands, "randomize_resampling_time", False
        )
        self.use_command_resampling_time_curriculum = getattr(
            self.cfg.commands, "use_command_resampling_time_curriculum", False
        )
        self.command_resampling_time_warmup_iters = getattr(
            self.cfg.commands, "command_resampling_time_warmup_iters", 0
        )
        self.randomize_command_resampling_time = self.enable_command_resampling_time_randomization
        self.command_resampling_time_min = getattr(
            self.cfg.commands, "resampling_time_min", self.cfg.commands.resampling_time
        )
        self.command_resampling_time_max = getattr(
            self.cfg.commands, "resampling_time_max", self.cfg.commands.resampling_time
        )
        self.forward_only_min_lin_vel_x = getattr(
            self.cfg.commands, "forward_only_min_lin_vel_x", 0.05
        )
        self.forward_only_command_terrain_kind_ids = self._build_forward_only_command_terrain_kind_ids(
            sim_device
        )
        self.forward_only_command_terrain_types = self._build_forward_only_command_terrain_types(
            sim_device
        )
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

    def _sample_command_resample_timeouts(self, num_envs):
        if self.randomize_command_resampling_time:
            command_timeouts_s = torch_rand_float(
                self.command_resampling_time_min,
                self.command_resampling_time_max,
                (num_envs, 1),
                self.device,
            ).squeeze(1)
        else:
            command_timeouts_s = torch.full(
                (num_envs,),
                self.cfg.commands.resampling_time,
                device=self.device,
                dtype=torch.float,
            )

        return torch.clamp(
            torch.round(command_timeouts_s / self.dt).long(),
            min=1,
        )
        
        print("self.use_tradeoff - ", self.use_tradeoff)

        # We want to be at the full bounds right away, but we want to skip back sometimes for exploration
        self.tradeoff_step_ctr = torch.zeros((self.cfg.env.num_envs, 1), device=sim_device, dtype=torch.float)

    # ------------ reward functions----------------
    
    def _reward_swing_vel_ellipsoid_terrain(self):
        """Reward swing-foot velocity ellipsoids that are:
        (1) large,
        (2) aligned so their dominant axes lie in the local terrain tangent plane,
        (3) mildly anisotropic, favoring tangential over normal velocity authority.
        Assumes:
        - self._get_swing_leg_jacobians() -> (B, 4, 3, nj)
        - self._terrain_normals_under_feet -> (B, 4, 3)
        - self._swing_mask -> (B, 4) in {0,1}
        """
        eps = 1e-6
        J = _sanitize_tensor(self.leg_jacobians)                                            # (B, 4, 3, nj)
        n = _sanitize_tensor(self.simulator._normal_vector_around_feet.reshape(-1, 4, 3))   # (B, 4, 3)
        swing = self._feet_contact_mask().float()                                           # (B, 4)

        # velocity ellipsoid matrix: M = J J^T
        M = J @ J.transpose(-1, -2)                             # (B, 4, 3, 3)
        I3 = torch.eye(3, device=M.device, dtype=M.dtype).view(1, 1, 3, 3)
        M = _sanitize_tensor(0.5 * (M + M.transpose(-1, -2)) + eps * I3)

        # principal axes / sizes
        eigvals, eigvecs = torch.linalg.eigh(M)                 # ascending
        eigvals = torch.clamp(_sanitize_tensor(eigvals), min=eps)
        eigvecs = _safe_normalize(eigvecs, dim=-2, eps=eps)
        axes = torch.sqrt(eigvals)                              # semi-axis lengths

        v_small = eigvecs[..., 0]                               # (B, 4, 3)
        v_mid   = eigvecs[..., 1]
        v_large = eigvecs[..., 2]

        # normalize terrain normals
        n = _safe_normalize(n, dim=-1, eps=eps)
        default_n = torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype).view(1, 1, 3)
        n_norm = torch.linalg.norm(n, dim=-1, keepdim=True)
        n = torch.where(n_norm > eps, n, default_n)

        # alignment: largest two axes tangent to terrain, smallest axis normal to terrain
        align_large = 1.0 - (torch.sum(v_large * n, dim=-1) ** 2)
        align_mid   = 1.0 - (torch.sum(v_mid   * n, dim=-1) ** 2)
        align_small =       (torch.sum(v_small * n, dim=-1) ** 2)
        align = torch.clamp(0.5 * align_large + 0.25 * align_mid + 0.25 * align_small, 0.0, 1.0)

        # size: geometric mean of ellipsoid semi-axis lengths
        # size_raw = torch.exp(torch.mean(torch.log(axes), dim=-1))
        size_raw = _geom_mean(axes)
        size = 1.0 - torch.exp(-self.cfg.rewards.kite_rewards.ellipsoid_force_size_scale * size_raw)
        size = torch.clamp(_sanitize_tensor(size), 0.0, 1.0)

        # directional authority in normal vs tangent plane
        normal_auth = torch.clamp(_sanitize_tensor(torch.einsum('bfi,bfij,bfj->bf', n, M, n)), min=0.0)

        # plane authority = trace(M) - normal authority
        plane_auth = torch.clamp(_sanitize_tensor(torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1) - normal_auth), min=0.0)

        # mild anisotropy: prefer tangent-plane authority slightly above normal
        # target plane_auth / normal_auth ~= 2.2 / 1.0
        ratio = torch.clamp(_sanitize_tensor(plane_auth / torch.clamp(normal_auth, min=eps)), min=eps, max=1e6)
        target_ratio = torch.tensor(2.2, device=M.device, dtype=M.dtype)
        anis = torch.exp(-torch.abs(torch.log(ratio) - torch.log(target_ratio)))
        anis = torch.clamp(_sanitize_tensor(anis), 0.0, 1.0)

        # combine over swing feet only
        per_foot = 0.6 * align + 0.25 * size + 0.15 * anis
        reward = (per_foot * swing).sum(dim=1) / torch.clamp(swing.sum(dim=1), min=1.0)

        return torch.clamp(_sanitize_tensor(reward), 0.0, 1.0)
    
    def _reward_torso_force_wrench_ellipsoid(self):
        """
        Force / wrench ellipsoid reward using:
            J_legs: (N, 4, 3, 3)

        Expected tensors on self
        ------------------------
        self.foot_jacobian_b        : (N, 4, 3, 3)  # from compute_all_leg_jacobians(...)
        self.applied_leg_torques    : (N, 4, 3)
        self.torque_limits_leg      : (N, 4, 3) or (1, 4, 3)
        self.contact_forces         : (N, n_bodies, 3)
        self.rigid_body_states      : (N, n_bodies, >=3)
        self.root_states[:, 0:3]    : (N, 3)
        self.base_quat              : (N, 4) xyzw
        self.foot_indices           : len 4
        self.terrain_normals_w      : (N, 4, 3)

        Suggested cfg.rewards fields
        ----------------------------
        ellipsoid_main_weight = 1.0
        ellipsoid_force_aux_weight = 0.25
        ellipsoid_wrench_aux_weight = 0.10
        ellipsoid_tau_margin_weight = 0.10
        ellipsoid_tau_grf_consistency_weight = 0.20
        ellipsoid_friction_weight = 0.15

        ellipsoid_wrench_length_scale = 0.30
        ellipsoid_force_size_scale = 0.75
        ellipsoid_wrench_size_scale = 0.20

        ellipsoid_force_z_ratio_min = 1.2
        ellipsoid_force_z_ratio_max = 4.0
        ellipsoid_force_xy_ratio_max = 2.0
        ellipsoid_wrench_cond_max = 6.0

        ellipsoid_mu_friction = 0.6
        ellipsoid_normal_force_margin = 5.0
        ellipsoid_tangential_force_margin = 2.0
        """
        device = self.simulator.base_pos.device
        dtype = self.simulator.base_pos.dtype
        N = self.simulator.base_pos.shape[0]

        # --------------------------------------------------
        # state
        # --------------------------------------------------
        J_b = _sanitize_tensor(self.leg_jacobians)                                                # (N,4,3,3)
        tau_leg = _sanitize_tensor(self.simulator._dof_tau.view(-1, 4, 3))                        # (N,4,3)
        tau_max = torch.clamp(_sanitize_tensor(self.simulator._torque_limits.view(-1, 4, 3)).abs(), min=1e-6)                  # (N,4,3) or broadcastable

        foot_pos_w = _sanitize_tensor(self.simulator.feet_pos)          # (N,4,3)
        base_pos_w = _sanitize_tensor(self.simulator.base_pos)          # (N,3)
        foot_pos_rel_base_w = foot_pos_w - base_pos_w[:, None, :]               # (N,4,3)

        grf_w = _sanitize_tensor(self.simulator._grfs_buf.view(-1, 4, 3))                         # (N,4,3)
        normals_w = _safe_normalize(self.simulator._normal_vector_around_feet.view(-1,4,3), dim=-1, eps=1e-6)      # (N,4,3)
        default_n = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 3)
        normals_norm = torch.linalg.norm(normals_w, dim=-1, keepdim=True)
        normals_w = torch.where(normals_norm > 1e-6, normals_w, default_n)

        # stance mask from normal contact force
        fn_meas = torch.sum(grf_w * normals_w, dim=-1)                          # (N,4)
        stance_mask = fn_meas > self.cfg.rewards.contact_force_threshold
        stance_f = stance_mask.to(dtype)

        # --------------------------------------------------
        # rotate Jacobians from base frame -> world frame
        # --------------------------------------------------
        base_quat = _safe_normalize(self.simulator.base_quat, dim=-1, eps=1e-6)
        x, y, z, w = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        R_wb = torch.stack([
            torch.stack([1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)], dim=-1),
            torch.stack([    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)], dim=-1),
            torch.stack([    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)], dim=-1),
        ], dim=-2)                                                               # (N,3,3)

        J_w = _sanitize_tensor(torch.matmul(R_wb[:, None, :, :], J_b))                             # (N,4,3,3)

        # --------------------------------------------------
        # primary capability matrices
        #   M_i = J_i diag(tau_max_i^2) J_i^T
        #   S_F = sum_i M_i^{-1}
        #   S_W = sum_i A_i M_i^{-1} A_i^T
        # --------------------------------------------------
        Lw = max(float(self.cfg.rewards.kite_rewards.ellipsoid_wrench_length_scale), 1e-6)

        S_F = torch.zeros(N, 3, 3, device=device, dtype=dtype)
        S_W = torch.zeros(N, 6, 6, device=device, dtype=dtype)
        I3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(N, 3, 3)

        for i in range(4):
            J_i = J_w[:, i]                                                      # (N,3,3)
            tau_max_i = tau_max[:, i]                                            # (N,3)

            Wtau_inv_i = torch.diag_embed(tau_max_i ** 2)                        # (N,3,3)
            M_i = J_i @ Wtau_inv_i @ J_i.transpose(-1, -2)                       # (N,3,3)
            M_i = _sanitize_tensor(0.5 * (M_i + M_i.transpose(-1, -2)))
            M_i_inv = _safe_inv(M_i, eps=1e-5)

            mask_i = stance_f[:, i].view(N, 1, 1)
            S_F = S_F + mask_i * M_i_inv

            A_i = torch.cat([
                I3,
                _skew(foot_pos_rel_base_w[:, i]) / Lw,
            ], dim=-2)                                                           # (N,6,3)

            S_W = S_W + mask_i * (A_i @ M_i_inv @ A_i.transpose(-1, -2))

        S_F = _sanitize_tensor(0.5 * (S_F + S_F.transpose(-1, -2)))
        S_W = _sanitize_tensor(0.5 * (S_W + S_W.transpose(-1, -2)))

        # --------------------------------------------------
        # force ellipsoid reward
        #   F^T S_F^{-1} F <= 1
        # --------------------------------------------------
        evals_F, evecs_F = _eig_desc(S_F)
        lam1, lam2, lam3 = evals_F[:, 0], evals_F[:, 1], evals_F[:, 2]

        force_size = _geom_mean(evals_F)
        r_force_size = 1.0 - torch.exp(-self.cfg.rewards.kite_rewards.ellipsoid_force_size_scale * force_size)
        r_force_size = torch.clamp(_sanitize_tensor(r_force_size), 0.0, 1.0)

        z_ratio = torch.clamp(_sanitize_tensor(lam1 / torch.clamp(0.5 * (lam2 + lam3), min=1e-8)), min=0.0, max=1e6)
        xy_ratio = torch.clamp(_sanitize_tensor(lam2 / torch.clamp(lam3, min=1e-8)), min=0.0, max=1e6)

        r_force_aniso = (
            _interval_reward(
                z_ratio,
                self.cfg.rewards.kite_rewards.ellipsoid_force_z_ratio_min,
                self.cfg.rewards.kite_rewards.ellipsoid_force_z_ratio_max,
                sharpness=2.0,
            )
            *
            _upper_reward(
                xy_ratio,
                self.cfg.rewards.kite_rewards.ellipsoid_force_xy_ratio_max,
                sharpness=2.0,
            )
        )

        ez = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 3)
        u1 = evecs_F[:, :, 0]
        u2 = evecs_F[:, :, 1]
        u3 = evecs_F[:, :, 2]

        align_z = torch.abs(torch.sum(u1 * ez, dim=-1))
        align_xy = 1.0 - 0.5 * (u2[:, 2].abs() + u3[:, 2].abs())
        align_xy = torch.clamp(align_xy, 0.0, 1.0)

        r_force_align = torch.clamp(_sanitize_tensor(align_z * align_xy), 0.0, 1.0)
        r_force_main = 0.6 * r_force_align + 0.25 * r_force_size + 0.15 * r_force_aniso
        r_force_main = torch.clamp(_sanitize_tensor(r_force_main), 0.0, 1.0)

        # --------------------------------------------------
        # wrench ellipsoid reward
        #   W_tilde^T S_W^{-1} W_tilde <= 1
        # --------------------------------------------------
        evals_W, _ = _eig_desc(S_W)

        n_stance = stance_mask.sum(dim=1).clamp(min=1)                           # (N,)
        k_active = torch.clamp(3 * n_stance, max=6)                              # (N,)

        idx6 = torch.arange(6, device=device).view(1, 6)
        active_mask = idx6 < k_active.unsqueeze(-1)

        wrench_log = torch.where(
            active_mask,
            torch.log(torch.clamp(evals_W, min=1e-8)),
            torch.zeros_like(evals_W),
        ).sum(dim=1)

        wrench_size = torch.exp(wrench_log / k_active.to(dtype))
        r_wrench_size = 1.0 - torch.exp(-self.cfg.rewards.kite_rewards.ellipsoid_wrench_size_scale * wrench_size)
        r_wrench_size = torch.clamp(_sanitize_tensor(r_wrench_size), 0.0, 1.0)

        lam_max = evals_W[:, 0]
        lam_min_active = torch.gather(evals_W, 1, (k_active - 1).long().unsqueeze(-1)).squeeze(-1)
        cond_W = torch.clamp(_sanitize_tensor(lam_max / torch.clamp(lam_min_active, min=1e-8)), min=0.0, max=1e6)

        r_wrench_cond = _upper_reward(
            cond_W,
            self.cfg.rewards.kite_rewards.ellipsoid_wrench_cond_max,
            sharpness=1.0,
        )

        r_wrench_main = 0.6 * r_wrench_size + 0.4 * r_wrench_cond
        r_wrench_main = torch.clamp(_sanitize_tensor(r_wrench_main), 0.0, 1.0)

        # --------------------------------------------------
        # auxiliary: realized force/wrench uses strong ellipsoid directions
        # --------------------------------------------------
        grf_stance = grf_w * stance_f.unsqueeze(-1)
        F_meas = grf_stance.sum(dim=1)                                           # (N,3)
        M_meas = torch.cross(foot_pos_rel_base_w, grf_stance, dim=-1).sum(dim=1)
        W_meas = torch.cat([F_meas, M_meas / Lw], dim=-1)                        # (N,6)

        H_F = _safe_pinv(S_F, eps=1e-5)
        F_hat = _safe_normalize(F_meas, dim=-1, eps=1e-6)
        force_dir_cost = torch.clamp(_sanitize_tensor(torch.einsum("bi,bij,bj->b", F_hat, H_F, F_hat)), min=0.0, max=1e6)
        r_force_use = torch.clamp(_sanitize_tensor(torch.exp(-2.0 * force_dir_cost)), 0.0, 1.0)

        r_force_principal_use = torch.clamp(_sanitize_tensor(torch.abs(torch.sum(F_hat * u1, dim=-1))), 0.0, 1.0)

        H_W = _safe_pinv(S_W, eps=1e-5)
        W_hat = _safe_normalize(W_meas, dim=-1, eps=1e-6)
        wrench_dir_cost = torch.clamp(_sanitize_tensor(torch.einsum("bi,bij,bj->b", W_hat, H_W, W_hat)), min=0.0, max=1e6)
        r_wrench_use = torch.clamp(_sanitize_tensor(torch.exp(-2.0 * wrench_dir_cost)), 0.0, 1.0)

        # # --------------------------------------------------
        # # auxiliary: torque limits and torque/GRF consistency
        # # --------------------------------------------------
        # F_tau = torch.zeros(N, 4, 3, device=device, dtype=dtype)
        # tau_margin_terms = []

        # for i in range(4):
        #     J_i = J_w[:, i]                                                      # (N,3,3)
        #     tau_i = tau_leg[:, i]                                                # (N,3)
        #     tau_max_i = tau_max[:, i]                                            # (N,3)

        #     F_tau_i = (_safe_pinv(J_i @ J_i.transpose(-1, -2), eps=1e-5) @ (J_i @ tau_i.unsqueeze(-1))).squeeze(-1)
        #     F_tau_i = F_tau_i * stance_f[:, i:i+1]
        #     F_tau[:, i] = F_tau_i

        #     tau_norm_i = tau_i / torch.clamp(tau_max_i, min=1e-6)
        #     tau_util_i = torch.linalg.norm(tau_norm_i, dim=-1) / (tau_i.shape[-1] ** 0.5)
        #     tau_margin_i = torch.exp(-3.0 * tau_util_i**2) * stance_f[:, i]
        #     tau_margin_terms.append(tau_margin_i)

        # tau_margin_terms = torch.stack(tau_margin_terms, dim=1)                  # (N,4)
        # r_tau_margin = tau_margin_terms.sum(dim=1) / torch.clamp(stance_f.sum(dim=1), min=1.0)

        # diff_F = F_tau - grf_w
        # err_F = torch.linalg.norm(diff_F, dim=-1) / torch.clamp(torch.linalg.norm(grf_w, dim=-1), min=1e-6)
        # r_tau_grf_consistency = (
        #     torch.exp(-5.0 * err_F**2) * stance_f
        # ).sum(dim=1) / torch.clamp(stance_f.sum(dim=1), min=1.0)

        # --------------------------------------------------
        # auxiliary: terrain awareness via friction cone
        # --------------------------------------------------
        fn = _sanitize_tensor(torch.sum(grf_w * normals_w, dim=-1))                                # (N,4)
        ft_vec = grf_w - fn.unsqueeze(-1) * normals_w
        ft = _sanitize_tensor(torch.linalg.norm(ft_vec, dim=-1))

        mu = self.cfg.rewards.kite_rewards.ellipsoid_mu_friction
        fn_margin = self.cfg.rewards.kite_rewards.ellipsoid_normal_force_margin
        ft_margin = self.cfg.rewards.kite_rewards.ellipsoid_tangential_force_margin

        r_normal_support = torch.clamp(_sanitize_tensor(torch.exp(-2.0 * torch.relu(fn_margin - fn)**2)), 0.0, 1.0)
        cone_violation = _sanitize_tensor(torch.relu(ft - mu * torch.relu(fn - ft_margin)))
        r_friction = torch.clamp(_sanitize_tensor(torch.exp(-0.5 * cone_violation**2)), 0.0, 1.0)

        r_contact_cone = (
            (r_normal_support * r_friction) * stance_f
        ).sum(dim=1) / torch.clamp(stance_f.sum(dim=1), min=1.0)

        # --------------------------------------------------
        # final reward
        # --------------------------------------------------
        r_main = 0.5 * r_force_main + 0.5 * r_wrench_main
        r_aux = (
            self.cfg.rewards.kite_rewards.ellipsoid_force_aux_weight
            * (0.5 * r_force_use + 0.5 * r_force_principal_use)
            +
            self.cfg.rewards.kite_rewards.ellipsoid_wrench_aux_weight
            * r_wrench_use
            +
            # self.cfg.rewards.ellipsoid_tau_margin_weight
            # * r_tau_margin
            # +
            # self.cfg.rewards.ellipsoid_tau_grf_consistency_weight
            # * r_tau_grf_consistency
            # +
            self.cfg.rewards.kite_rewards.ellipsoid_friction_weight
            * r_contact_cone
        )

        reward = self.cfg.rewards.kite_rewards.ellipsoid_main_weight * r_main + (1.0 - self.cfg.rewards.kite_rewards.ellipsoid_main_weight) * r_aux
        reward = reward * (n_stance > 0).to(dtype)
        reward = torch.clamp(_sanitize_tensor(reward), 0.0, 1.0)

        # # optional logging
        # self.extras["ellipsoid_force_size"] = force_size.mean()
        # self.extras["ellipsoid_force_z_ratio"] = z_ratio.mean()
        # self.extras["ellipsoid_force_xy_ratio"] = xy_ratio.mean()
        # self.extras["ellipsoid_force_align"] = r_force_align.mean()
        # self.extras["ellipsoid_wrench_cond"] = cond_W.mean()
        # self.extras["ellipsoid_force_use"] = r_force_use.mean()
        # self.extras["ellipsoid_wrench_use"] = r_wrench_use.mean()
        # self.extras["ellipsoid_tau_margin"] = r_tau_margin.mean()
        # self.extras["ellipsoid_tau_grf_consistency"] = r_tau_grf_consistency.mean()
        # self.extras["ellipsoid_contact_cone"] = r_contact_cone.mean()

        return reward    

    
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
        base_height = self._base_height_over_terrain()
        # print(f"base height: {base_height}")
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        return rew

    def _potential_orientation(self):
        roll_pitch = self.simulator.projected_gravity[:, :2]
        return torch.exp(-torch.sum(roll_pitch ** 2, dim=1) / 0.5)

    def _potential_height(self):
        base_height = self._base_height_over_terrain()
        height_error = torch.square(
            base_height - self.cfg.rewards.base_height_target
        )
        return torch.exp(-height_error / 0.5)

    def _reward_pbrs_orientation(self):
        phi_next = self._potential_orientation()
        shaping = phi_next - self.phi_prev_orientation
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
        return torch.sum(torch.abs(self._joint_power_per_dof()), dim=1)

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
        # print(f"contacts: {(torch.norm(self.simulator.link_contact_forces[0, self.simulator.penalized_contact_indices, :], dim=-1) > self.cfg.rewards.contact_force_threshold)}")
        rew = torch.sum(1.*(torch.norm(
            self.simulator.link_contact_forces[:, self.simulator.penalized_contact_indices, :], 
            dim=-1) > self.cfg.rewards.contact_force_threshold), dim=1)
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
        y_min = self.cfg.rewards.feet_spread_y_min
        x_min = self.cfg.rewards.feet_spread_x_min
        alpha_diag = self.cfg.rewards.feet_spread_alpha_diag
        fz_thr = self.cfg.rewards.contact_force_threshold

        # contact weighting mode:
        #   "both"  -> w_ij = c_i * c_j  (strict stance-only)
        #   "blend" -> w_ij = 0.5*(c_i + c_j) (some signal when one foot swings)
        contact_mode = self.cfg.rewards.feet_spread_contact_mode
        # ----------------------------------------------------------------

        # Assumed foot order: 0=FL, 1=FR, 2=RL, 3=RR (common in RSL-RL / legged_gym setups)
        # FL, FR, RL, RR = 0, 1, 2, 3
        FR, FL, RR, RL = 0, 1, 2, 3

        feet_xy = self._feet_pos_base_frame()[:, :, :2]  # (N,4,2)

        # contact mask (N,4) in {0,1}
        c = (self._feet_contact_fz() > fz_thr).float()

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


    def _reward_tracking_lin_vel_base(self):
        lin_vel_error = self._lin_vel_tracking_error()
        return torch.exp(-self.cfg.rewards.tracking_lin_vel_error_scale * lin_vel_error)

    def _reward_tracking_lin_vel(self):
        reward = self._reward_tracking_lin_vel_base()

        if self.cfg.commands.heading_command:
            heading_error = torch.abs(self.heading - self.commands[:, 3])
            heading_coef = (1.0 + torch.cos(heading_error)) / 2.0
            reward = reward * heading_coef

        return reward

    def _reward_tracking_lin_vel_penalty(self):
        return 1.0 - self._reward_tracking_lin_vel_base()

    def _reward_dof_act_limits(self):
        actions_scaled = self._scaled_pos_actions()
        
        # Penalize dof positions too close to the limit
        out_of_limits = -(actions_scaled - self.simulator.dof_pos_limits_hard[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (actions_scaled - \
                          self.simulator.dof_pos_limits_hard[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = self._ang_vel_tracking_error()
        return torch.exp(-self.cfg.rewards.tracking_ang_vel_error_scale * ang_vel_error)

    def _reward_tracking_ang_vel_penalty(self):
        # Bounded penalty that grows as yaw-rate command tracking fails.
        return 1.0 - self._reward_tracking_ang_vel()

    def _reward_joint_power(self):
        # penalize large amounts of motor power
        return torch.sum(torch.abs(self._joint_power_per_dof()), dim=1)

    def _reward_joint_power_dist(self):
        # Penalize uneven distributions of motor power
        return torch.var(self._joint_power_per_dof(), dim=1)

    def _reward_foot_slip(self):
        # penalize feet that are in-contact for any movement in the x/y direction
        contact = self._feet_contact_mask()
        return  torch.sum(torch.square(contact * torch.sum(self.simulator.feet_vel[:,:,:2], dim=-1)), dim=-1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_near_edge(self):
        feet_near_edge = self._feet_near_edge_mask()
        feet_contact = self._feet_contact_mask()
        return torch.sum(feet_near_edge & feet_contact, dim=-1).float()

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self._feet_contact_mask()
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= self._command_norm(2) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_swing_participation_balance(self):
        # Penalize uneven long-horizon swing participation across feet.
        mean_swing = self.feet_swing_ema.mean(dim=1, keepdim=True)
        return -torch.mean(torch.square(self.feet_swing_ema - mean_swing), dim=1)

    def _reward_diagonal_pair_balance(self):
        # Gait order is [FL, FR, RL, RR], so diagonals are FL/RR and FR/RL.
        diag_a = 0.5 * (self.feet_swing_ema[:, 0] + self.feet_swing_ema[:, 3])
        diag_b = 0.5 * (self.feet_swing_ema[:, 1] + self.feet_swing_ema[:, 2])
        return -torch.square(diag_a - diag_b)

    def _reward_completed_swing_height_balance(self):
        # Penalize uneven completed-swing peak heights across feet.
        mean_height = self.feet_swing_height_ema.mean(dim=1, keepdim=True)
        return -torch.mean(
            torch.square(self.feet_swing_height_ema - mean_height),
            dim=1,
        )

    def _reward_dof_vel_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.simulator.dof_vel), dim=1) * (
            self._command_norm(3) < self.cfg.commands.zero_command_threshold
        )

    def _reward_dof_pos_stand_still(self):
        # Penalize position deviation at zero commands
        return torch.sum(
            torch.square(self.simulator.dof_pos - self.simulator.default_dof_pos),
            dim=1,
        ) * (self._command_norm(3) < self.cfg.commands.zero_command_threshold)
    
    def _reward_stand_still_contact(self):
        # Encourage feet contact with the ground at zero commands
        contacts = self._feet_contact_mask()
        full_contact = torch.sum(1.*contacts, dim=1)==len(self.simulator.feet_indices)
        return 1.0 * full_contact * (
            self._command_norm(3) < self.cfg.commands.zero_command_threshold
        )
    
    def _reward_dof_close_to_default(self):
        # Penalize dof position deviation from default
        return torch.sum(torch.square(self.simulator.dof_pos - self.simulator.default_dof_pos), dim=1)

    def _reward_alive_bonus(self):
        return ~self.reset_buf

    def _reward_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        """
        foot_vel_xy_norm = self._feet_vel_xy_norm()
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
        foot_vel_xy_norm = self._feet_vel_xy_norm()  # (N,4)
        z_des = self._terrain_aware_foot_target_height()  # (N,4)

        # Main tracking error: encourage feet to reach desired terrain-aware height
        track_err = torch.square(feet_z - z_des)                        # (N,4)

        # Soft over-swing penalty: only penalize when foot goes too far above desired height
        # Margin gives some freedom to overshoot a little during learning
        excess_margin = self.cfg.rewards.foot_clearance_excess_margin
        excess = torch.relu(feet_z - (z_des + excess_margin))           # (N,4)
        excess_err = torch.square(excess)

        excess_weight = self.cfg.rewards.foot_clearance_excess_weight

        total_err = torch.sum(
            foot_vel_xy_norm * (track_err + excess_weight * excess_err),
            dim=-1
        )                                                               # (N,)

        return torch.exp(-total_err / self.cfg.rewards.foot_clearance_tracking_sigma)


    def _reward_front_foot_overreach(self):
        # Assumed order is FR/L, FL/R....
        front_x = self._feet_pos_base_frame()[:, :2, 0]

        
        overreach = torch.relu(front_x - self.cfg.rewards.overreach_x_max)
        
        # stance/contact gating
        contact = (
            self._feet_contact_fz()[:, :2] > self.cfg.rewards.contact_force_threshold
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
        fz_thr = self.cfg.rewards.contact_force_threshold
        sigma = self.cfg.rewards.support_polygon_sigma

        base_xy = self.simulator.base_pos[:, :2]                      # (N,2)
        feet_xy = self.simulator.feet_pos[:, :, :2]                  # (N,4,2)

        contact = (self._feet_contact_fz() > fz_thr).float()         # (N,4)

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

    def _compute_vhip_angle(self):
        return self._vhip_terms()[0]

    def _compute_vhip_acceleration(self):
        return self._vhip_terms()[1]

    def _reward_vhip_angle(self):
        return torch.clamp(
            torch.abs(self._compute_vhip_angle()) - self.cfg.rewards.vhip_angle_deadband,
            min=0.0,
        )

    def _reward_vhip_angular_acc(self):
        return torch.clamp(
            torch.abs(self._compute_vhip_acceleration()) - self.cfg.rewards.vhip_acc_deadband,
            min=0.0,
        )

    def _reward_rear_foot_overreach(self):
        rear_x = self._feet_pos_base_frame()[:, 2:4, 0]
        x_error = torch.abs(rear_x - self.cfg.rewards.rear_foot_x_nominal)
        overreach = torch.relu(x_error - self.cfg.rewards.rear_foot_x_margin)
        contact = (
            self._feet_contact_fz()[:, 2:4] > self.cfg.rewards.contact_force_threshold
        ).float()
        return torch.sum(contact * overreach ** 2, dim=1)

    def _reward_pd_target_torque_limit(self):
        """
        Penalize joint position targets that would induce PD torques
        exceeding tau_max.

        Uses a quadratic hinge outside the admissible target range.
        """

        tau_max = self.pd_target_tau_max.to(dtype=self.simulator.dof_pos.dtype)

        q      = self.simulator.dof_pos        # (N, dof)
        qdot   = self.simulator.dof_vel        # (N, dof)
        q_des  = self._scaled_pos_actions()  # (N, dof)
    
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
        contacts = self._feet_contact_mask()
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
        fz = self._feet_contact_fz()
        contact_prob = torch.sigmoid(10.0*(fz - self.cfg.rewards.contact_force_threshold))
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
        error = torch.sum(torch.square(self._scaled_pos_actions() - self.simulator.dof_pos), dim=-1)
        return torch.exp(-4.0*error)

    def _reward_hip_pos(self):
        """ Reward for the hip joint position close to default position
        """
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(torch.square(
            self.simulator.dof_pos[:, hip_joint_indices] - 
            self.simulator.default_dof_pos[:, hip_joint_indices]), dim=-1)
        return dof_pos_error

    def _reward_stumble(self):
        """
        Penalize feet colliding with vertical surfaces / obstacles during swing.
        """

        contact_forces = self.simulator.link_contact_forces[:, self.simulator.feet_indices, :]  # (N,4,3)

        horizontal_force = torch.norm(contact_forces[:, :, :2], dim=2)
        vertical_force = torch.abs(contact_forces[:, :, 2])

        contact = vertical_force > 1.0
        swing = (~contact).float()

        stumble = (horizontal_force > 4.0 * vertical_force) & (horizontal_force > 5.0)

        return torch.sum(swing * stumble.float(), dim=1)
