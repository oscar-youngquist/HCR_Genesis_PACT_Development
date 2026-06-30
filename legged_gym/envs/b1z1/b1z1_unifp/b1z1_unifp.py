from collections import deque

import numpy as np
import torch

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math_utils import (
    quat_apply,
    quat_apply_yaw,
    quat_from_euler_xyz,
    quat_rotate_inverse,
    torch_rand_float,
    wrap_to_pi,
)


def sphere2cart(sphere_coords):
    radius = sphere_coords[:, 0]
    pitch = sphere_coords[:, 1]
    yaw = sphere_coords[:, 2]
    return torch.stack(
        (
            radius * torch.cos(pitch) * torch.cos(yaw),
            radius * torch.cos(pitch) * torch.sin(yaw),
            radius * torch.sin(pitch),
        ),
        dim=-1,
    )


def cart2sphere(cart_coords):
    radius = torch.norm(cart_coords, dim=-1).clamp_min(1.0e-6)
    pitch = torch.asin((cart_coords[:, 2] / radius).clamp(-1.0, 1.0))
    yaw = torch.atan2(cart_coords[:, 1], cart_coords[:, 0])
    return torch.stack((radius, pitch, yaw), dim=-1)


class B1Z1UniFP(BaseTask):
    """Genesis port of the original UniFP B1/Z1 position-force baseline."""

    def __init__(self, cfg, sim_params, sim_device, headless):
        self.cfg = cfg
        self.init_done = False
        self._parse_cfg(cfg, sim_device)
        super().__init__(cfg, sim_params, sim_device, headless)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def get_observations(self):
        return self.obs_buf, self.obs_history, self.privileged_obs_buf, self.explicit_labels_buf

    def reset(self):
        """Reset all robots."""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _, _, _ = self.step(
            torch.zeros(
                self.num_envs,
                self.num_actions,
                device=self.device,
                requires_grad=False,
            )
        )
        return obs, privileged_obs

    def step(self, actions):
        """Apply actions, simulate, and return the PACT-style env tuple."""
        actions = self._pre_sim_step(actions)
        if self.force_randomization_active and self.cfg.commands.push_gripper_stators:
            self._push_gripper(torch.arange(self.num_envs, device=self.device))
        if self.force_randomization_active and self.cfg.commands.push_robot_base:
            self._push_robot_base(torch.arange(self.num_envs, device=self.device))
        self.simulator.step(actions)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.obs_history,
            self.explicit_labels_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.simulator._grfs_buf * self.obs_scales.grf,
        )

    def get_failure_idx(self):
        return self.reset_buf * ~self.time_out_buf

    def get_prev_obs(self):
        return self.last_obs_buf, self.last_obs_hist, self.llast_obs_buf, self.llast_obs_hist

    @property
    def force_randomization_active(self):
        return self.common_step_counter > self.cfg.commands.force_start_step * self.cfg.runner_steps_per_iter

    def post_physics_step(self):
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.simulator.post_physics_step()
        self._post_physics_step_callback()
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        if getattr(self.cfg, "sensor", None) is not None and self.cfg.sensor.add_depth:
            self.simulator.update_depth_images()
        self.compute_observations()
        if self.debug:
            self.simulator.draw_debug_vis()

    def check_termination(self):
        self.fail_buf[:] = 0
        if len(self.simulator.termination_contact_indices) > 0:
            self.fail_buf |= torch.any(
                torch.norm(
                    self.simulator.link_contact_forces[:, self.simulator.termination_contact_indices, :],
                    dim=-1,
                )
                > 10.0,
                dim=1,
            )
        rpy = self.simulator.base_euler
        base_height = torch.mean(
            self.simulator.base_pos[:, 2].unsqueeze(1) - self.simulator.measured_heights,
            dim=1,
        )
        if "roll" in self.cfg.termination.termination_terms:
            self.fail_buf |= torch.abs(wrap_to_pi(rpy[:, 0])) > self.cfg.termination.roll_threshold
        if "pitch" in self.cfg.termination.termination_terms:
            self.fail_buf |= torch.abs(wrap_to_pi(rpy[:, 1])) > self.cfg.termination.pitch_threshold
        if "height_min" in self.cfg.termination.termination_terms:
            self.fail_buf |= base_height < self.cfg.termination.height_min
        if "height_max" in self.cfg.termination.termination_terms:
            self.fail_buf |= base_height > self.cfg.termination.height_max

        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.fail_buf | self.time_out_buf

    def reset_idx(self, env_ids):
        """Reset selected environments and fill PACT-style episode logs."""
        if len(env_ids) == 0:
            return
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum and self.common_step_counter % self.max_episode_length == 0:
            self._update_command_curriculum(env_ids)

        episode_ee_goal_sphere = self.curr_ee_goal_sphere[env_ids].clone()
        episode_ee_force_cmd_norm = torch.mean(torch.norm(self.current_Fxyz_gripper_cmd[env_ids], dim=1))
        episode_ee_force_ext_norm = torch.mean(torch.norm(self.ee_force_ext_world[env_ids], dim=1))
        episode_base_force_cmd_norm = torch.mean(torch.norm(self.current_Fxyz_base_cmd[env_ids], dim=1))
        episode_base_force_ext_norm = torch.mean(torch.norm(self.base_force_ext_world[env_ids], dim=1))

        self._resample_commands(env_ids)
        self._resample_ee_goal(env_ids, is_init=True)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.simulator.reset_idx(env_ids)

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.llast_actions[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        self.ee_force_ext_world[env_ids] = 0.0
        self.base_force_ext_world[env_ids] = 0.0
        self.current_Fxyz_gripper_cmd[env_ids] = 0.0
        self.current_Fxyz_base_cmd[env_ids] = 0.0
        self._reset_force_events(env_ids)
        self._randomize_force_gains(env_ids)
        if hasattr(self.simulator, "apply_ee_force"):
            self.simulator.apply_ee_force(self.ee_force_ext_world)
        if hasattr(self.simulator, "apply_base_force"):
            self.simulator.apply_base_force(self.base_force_ext_world)

        self.last_obs_buf[env_ids] = 0.0
        self.llast_obs_buf[env_ids] = 0.0
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0.0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0.0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.simulator.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.domain_rand.use_domainrand_curriculum:
            phase_to_idx = {
                "joint_dynamics": 0.0,
                "mass_com": 1.0,
                "disturbance": 2.0,
                "complete": 3.0,
            }
            self.extras["episode"]["domain_rand_phase"] = phase_to_idx.get(
                self.simulator.domain_rand_phase,
                -1.0,
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
        self.extras["episode"]["ee_goal_radius"] = torch.mean(episode_ee_goal_sphere[:, 0])
        self.extras["episode"]["ee_goal_pitch"] = torch.mean(episode_ee_goal_sphere[:, 1])
        self.extras["episode"]["ee_goal_yaw"] = torch.mean(episode_ee_goal_sphere[:, 2])
        self.extras["episode"]["ee_force_cmd_norm"] = episode_ee_force_cmd_norm
        self.extras["episode"]["ee_force_ext_norm"] = episode_ee_force_ext_norm
        self.extras["episode"]["base_force_cmd_norm"] = episode_base_force_cmd_norm
        self.extras["episode"]["base_force_ext_norm"] = episode_base_force_ext_norm
        self.extras["episode"]["force_randomization_active"] = float(self.force_randomization_active)
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.0
            self.action_delay[env_ids] = torch.randint(
                self.cfg.domain_rand.ctrl_delay_step_range[0],
                self.cfg.domain_rand.ctrl_delay_step_range[1] + 1,
                (len(env_ids),),
                device=self.device,
            )

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        for i, name in enumerate(self.reward_names):
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf, min=0.0)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        self.llast_obs_buf = self.last_obs_buf.clone().detach()
        self.last_obs_buf = self.obs_buf.clone().detach()

        base_quat = self.simulator.base_quat
        base_rpy = self.simulator.base_euler
        base_yaw_quat = quat_from_euler_xyz(
            torch.zeros_like(base_rpy[:, 0]),
            torch.zeros_like(base_rpy[:, 1]),
            base_rpy[:, 2],
        )
        ee_center = self.get_ee_goal_spherical_center(base_yaw_quat)
        ee_local_cart = quat_rotate_inverse(base_yaw_quat, self.simulator.ee_pos - ee_center)
        self.ee_pos_sphe_arm = cart2sphere(ee_local_cart)

        ee_force_local = quat_rotate_inverse(base_yaw_quat, self.ee_force_ext_world)
        base_force_local = quat_rotate_inverse(base_yaw_quat, self.base_force_ext_world)
        force_offset_world = self.ee_force_ext_world + quat_apply(base_yaw_quat, self.current_Fxyz_gripper_cmd)
        ee_goal_offset_local = quat_rotate_inverse(
            base_yaw_quat,
            self.curr_ee_goal_cart_world + force_offset_world / self.gripper_force_kps - ee_center,
        )
        ee_goal_offset_sphere = cart2sphere(ee_goal_offset_local)

        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        body_orientation = self.get_body_orientation()
        dof_pos_err = (self.simulator.dof_pos[:, :17] - self.simulator.default_dof_pos[:, :17]) * self.obs_scales.dof_pos
        dof_vel = self.simulator.dof_vel[:, :17] * self.obs_scales.dof_vel

        self.obs_buf = torch.cat(
            (
                body_orientation,
                self.simulator.base_ang_vel * self.obs_scales.ang_vel,
                dof_pos_err,
                dof_vel,
                self.actions,
                sin_pos,
                cos_pos,
                self.commands * self.commands_scale,
            ),
            dim=-1,
        )
        if self.add_noise:
            self.obs_buf += (2.0 * torch.rand_like(self.obs_buf) - 1.0) * self.noise_scale_vec

        self.explicit_labels_buf = torch.cat(
            (
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,
                self.ee_pos_sphe_arm * self.ee_sphere_scale,
                ee_force_local * self.obs_scales.ee_force,
                base_force_local * self.obs_scales.base_force,
            ),
            dim=-1,
        )

        mass_params = torch.zeros(self.num_envs, 22, device=self.device)
        mass_params[:, 0:1] = self.simulator._added_base_mass
        mass_params[:, 1:4] = self.simulator._base_com_bias
        stance_mask = self._get_gait_phase()
        contact_mask = (self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 5.0).float()
        critic_obs = torch.cat(
            (
                self.explicit_labels_buf,
                self.simulator.dof_pos[:, :12] - self.ref_dof_pos,
                mass_params,
                self.simulator._friction_values - self.friction_value_offset,
                self.simulator._motor_strength[:, :17] - 1.0,
                stance_mask,
                contact_mask,
                self.simulator.projected_gravity,
                self.simulator.base_ang_vel * self.obs_scales.ang_vel,
                dof_pos_err,
                dof_vel,
                self.actions,
                sin_pos,
                cos_pos,
                self.commands * self.commands_scale,
                ee_goal_offset_sphere * self.ee_sphere_scale,
            ),
            dim=-1,
        )
        if critic_obs.shape[1] != self.cfg.env.num_privileged_obs:
            raise RuntimeError(
                f"B1Z1 UniFP privileged observation size mismatch: "
                f"got {critic_obs.shape[1]}, expected {self.cfg.env.num_privileged_obs}"
            )
        self.critic_obs_deque.append(critic_obs[:, : self.cfg.env.num_privileged_obs])
        self.privileged_obs_buf = torch.cat([self.critic_obs_deque[i] for i in range(self.critic_obs_deque.maxlen)], dim=-1)

        self.llast_obs_hist = self.last_obs_hist.clone().detach()
        self.last_obs_hist = self.obs_history.clone().detach()
        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat([self.obs_history_deque[i] for i in range(self.obs_history_deque.maxlen)], dim=-1)

    def set_viewer_camera(self, pos, lookat):
        """Set viewer camera position and direction."""
        self.simulator.set_viewer_camera(eye=pos, target=lookat)

    def _pre_sim_step(self, actions):
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions).to(self.device)
        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.actions[:] = actions[:]
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            actions = self.action_queue[torch.arange(self.num_envs, device=self.device), self.action_delay].clone()
        return actions

    def _post_physics_step_callback(self):
        env_ids = (
            self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0
        ).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self.update_curr_ee_goal()
        if self.cfg.domain_rand.push_robots:
            self.simulator.push_robots()

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids, 0] = torch_rand_float(*self.command_ranges["lin_vel_x"], (len(env_ids), 1), self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(*self.command_ranges["lin_vel_y"], (len(env_ids), 1), self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(*self.command_ranges["ang_vel_yaw"], (len(env_ids), 1), self.device).squeeze(1)
        zero_prob = self.cfg.commands.zero_vel_cmd_prob_after_force if self.force_randomization_active else self.cfg.commands.zero_vel_cmd_prob
        zero_mask = torch.rand(len(env_ids), device=self.device) < zero_prob
        self.commands[env_ids[zero_mask], :3] = 0.0
        self._resample_ee_goal(env_ids)

    def _resample_ee_goal(self, env_ids, is_init=False):
        if len(env_ids) == 0:
            return
        if is_init:
            self.ee_goal_sphere[env_ids] = self.init_end_ee_sphere
        else:
            self.ee_goal_sphere[env_ids, 0] = torch_rand_float(*self.cfg.goal_ee.ranges.pos_l, (len(env_ids), 1), self.device).squeeze(1)
            self.ee_goal_sphere[env_ids, 1] = torch_rand_float(*self.cfg.goal_ee.ranges.pos_p, (len(env_ids), 1), self.device).squeeze(1)
            self.ee_goal_sphere[env_ids, 2] = torch_rand_float(*self.cfg.goal_ee.ranges.pos_y, (len(env_ids), 1), self.device).squeeze(1)
        self.ee_start_sphere[env_ids] = self.curr_ee_goal_sphere[env_ids]
        self.goal_timer[env_ids] = 0.0
        self.traj_timesteps[env_ids] = torch_rand_float(*self.cfg.goal_ee.traj_time, (len(env_ids), 1), self.device).squeeze(1) / self.dt
        self.traj_total_timesteps[env_ids] = self.traj_timesteps[env_ids] + (
            torch_rand_float(*self.cfg.goal_ee.hold_time, (len(env_ids), 1), self.device).squeeze(1) / self.dt
        )

    def update_curr_ee_goal(self):
        self.goal_timer += 1
        done = self.goal_timer > self.traj_total_timesteps
        if torch.any(done):
            self._resample_ee_goal(done.nonzero(as_tuple=False).flatten())
        ratio = (self.goal_timer / self.traj_timesteps).clamp(0.0, 1.0).unsqueeze(1)
        self.curr_ee_goal_sphere = self.ee_start_sphere + ratio * (self.ee_goal_sphere - self.ee_start_sphere)
        self.curr_ee_goal_cart = sphere2cart(self.curr_ee_goal_sphere)
        base_yaw_quat = quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            self.simulator.base_euler[:, 2],
        )
        self.curr_ee_goal_cart_world = self.get_ee_goal_spherical_center(base_yaw_quat) + quat_apply(
            base_yaw_quat,
            self.curr_ee_goal_cart,
        )
        self.commands[:, 3:6] = self.curr_ee_goal_sphere
        self.commands[:, 9:12] = self.current_Fxyz_gripper_cmd
        self.commands[:, 12:15] = self.current_Fxyz_base_cmd

    def get_ee_goal_spherical_center(self, base_yaw_quat):
        return self.simulator.base_pos + quat_apply(base_yaw_quat, self.ee_goal_center_offset)

    def get_body_orientation(self):
        return self.simulator.base_euler[:, :2]

    def _get_phase(self):
        return (self.episode_length_buf.float() * self.dt / self.cfg.rewards.cycle_time) % 1.0

    def _get_gait_phase(self):
        phase = self._get_phase()
        stance = torch.zeros(self.num_envs, 4, device=self.device)
        stance[:, 0] = phase < 0.5
        stance[:, 3] = phase < 0.5
        stance[:, 1] = phase >= 0.5
        stance[:, 2] = phase >= 0.5
        return stance

    def _rand_force_interval(self, min_steps, max_steps, shape):
        low = int(min_steps)
        high = max(int(max_steps), low + 1)
        return torch.randint(low, high, shape, device=self.device)

    def _sample_force_target(self, env_ids, force_range, *, zero_z=False, z_scale=1.0):
        force_min, force_max = force_range
        target = torch_rand_float(force_min, force_max, (len(env_ids), 3), self.device)
        if zero_z:
            target[:, 2] = 0.0
        else:
            target[:, 2] *= z_scale
        return target

    def _update_force_stream(
        self,
        env_ids_all,
        *,
        interval,
        interval_min,
        interval_max,
        duration,
        duration_min,
        duration_max,
        settling_time,
        forced_prob,
        selected,
        freed,
        target,
        output,
        force_range,
        zero_z=False,
        z_scale=1.0,
    ):
        new_env_ids = env_ids_all[(self.episode_length_buf[env_ids_all] % interval[env_ids_all]) == 0]
        if len(new_env_ids) > 0:
            freed[new_env_ids] = torch.rand(len(new_env_ids), device=self.device) > forced_prob
            target[new_env_ids] = self._sample_force_target(new_env_ids, force_range, zero_z=zero_z, z_scale=z_scale)
            sampled_duration = torch_rand_float(duration_min, duration_max, (len(new_env_ids), 1), self.device).view(len(new_env_ids))
            max_duration = ((interval[new_env_ids].float() - settling_time) / 2.0).clamp_min(1.0)
            sampled_duration = torch.minimum(sampled_duration, max_duration)
            duration[new_env_ids] = sampled_duration
            selected[new_env_ids] = True
            self._force_push_end_time_for(output)[new_env_ids] = self.episode_length_buf[new_env_ids] + sampled_duration

        selected_env_ids = env_ids_all[selected[env_ids_all]]
        if len(selected_env_ids) > 0:
            end_time = self._force_push_end_time_for(output)
            before_end = self.episode_length_buf[selected_env_ids] < end_time[selected_env_ids].to(torch.int32)
            step1_env_ids = selected_env_ids[before_end]
            if len(step1_env_ids) > 0:
                dur = duration[step1_env_ids].unsqueeze(-1)
                elapsed = self.episode_length_buf[step1_env_ids].unsqueeze(-1) - (end_time[step1_env_ids].unsqueeze(-1) - dur)
                output[step1_env_ids] = (target[step1_env_ids] / dur) * torch.clamp(elapsed, torch.zeros_like(dur), dur)

            after_settling = self.episode_length_buf[selected_env_ids] > (end_time[selected_env_ids] + settling_time).to(torch.int32)
            step2_env_ids = selected_env_ids[after_settling]
            if len(step2_env_ids) > 0:
                dur = duration[step2_env_ids].unsqueeze(-1)
                elapsed = self.episode_length_buf[step2_env_ids].unsqueeze(-1) - (end_time[step2_env_ids].unsqueeze(-1) + settling_time)
                output[step2_env_ids] = target[step2_env_ids] - (target[step2_env_ids] / dur) * torch.clamp(
                    elapsed,
                    torch.zeros_like(dur),
                    dur,
                )

            finished = self.episode_length_buf[selected_env_ids] >= (
                end_time[selected_env_ids] + settling_time + duration[selected_env_ids]
            ).to(torch.int32)
            finished_env_ids = selected_env_ids[finished]
            if len(finished_env_ids) > 0:
                selected[finished_env_ids] = False
                target[finished_env_ids] = 0.0
                output[finished_env_ids] = 0.0
                end_time[finished_env_ids] = 0.0
                duration[finished_env_ids] = 0.0
                interval[finished_env_ids] = self._rand_force_interval(
                    interval_min,
                    interval_max,
                    (len(finished_env_ids),),
                )

        if torch.any(freed):
            selected[freed] = False
            target[freed] = 0.0
            output[freed] = 0.0
            self._force_push_end_time_for(output)[freed] = 0.0
            duration[freed] = 0.0

    def _force_push_end_time_for(self, output):
        if output is self.current_Fxyz_gripper_cmd:
            return self.push_end_time_gripper_cmd
        if output is self.ee_force_ext_world:
            return self.push_end_time_gripper_ext
        if output is self.current_Fxyz_base_cmd:
            return self.push_end_time_base_cmd
        return self.push_end_time_base_ext

    def _push_gripper(self, env_ids_all):
        self._update_force_stream(
            env_ids_all,
            interval=self.push_interval_gripper_cmd,
            interval_min=self.push_interval_gripper_cmd_min,
            interval_max=self.push_interval_gripper_cmd_max,
            duration=self.push_duration_gripper_cmd,
            duration_min=self.push_duration_gripper_cmd_min,
            duration_max=self.push_duration_gripper_cmd_max,
            settling_time=self.settling_time_force_gripper,
            forced_prob=self.cfg.commands.gripper_forced_prob_cmd,
            selected=self.selected_env_ids_gripper_cmd,
            freed=self.freed_envs_gripper_cmd,
            target=self.force_target_gripper_cmd,
            output=self.current_Fxyz_gripper_cmd,
            force_range=self.cfg.commands.max_push_force_xyz_gripper_cmd,
        )
        self._update_force_stream(
            env_ids_all,
            interval=self.push_interval_gripper_ext,
            interval_min=self.push_interval_gripper_ext_min,
            interval_max=self.push_interval_gripper_ext_max,
            duration=self.push_duration_gripper_ext,
            duration_min=self.push_duration_gripper_ext_min,
            duration_max=self.push_duration_gripper_ext_max,
            settling_time=self.settling_time_force_gripper,
            forced_prob=self.cfg.commands.gripper_forced_prob_ext,
            selected=self.selected_env_ids_gripper_ext,
            freed=self.freed_envs_gripper_ext,
            target=self.force_target_gripper_ext,
            output=self.ee_force_ext_world,
            force_range=self.cfg.commands.max_push_force_xyz_gripper_ext,
        )
        if hasattr(self.simulator, "apply_ee_force"):
            self.simulator.apply_ee_force(self.ee_force_ext_world)

    def _push_robot_base(self, env_ids_all):
        self._update_force_stream(
            env_ids_all,
            interval=self.push_interval_base_cmd,
            interval_min=self.push_interval_base_cmd_min,
            interval_max=self.push_interval_base_cmd_max,
            duration=self.push_duration_base_cmd,
            duration_min=self.push_duration_base_cmd_min,
            duration_max=self.push_duration_base_cmd_max,
            settling_time=self.settling_time_force_base,
            forced_prob=self.cfg.commands.base_forced_prob_cmd,
            selected=self.selected_env_ids_base_cmd,
            freed=self.freed_envs_base_cmd,
            target=self.force_target_base_cmd,
            output=self.current_Fxyz_base_cmd,
            force_range=self.cfg.commands.max_push_force_xyz_base_cmd,
            zero_z=True,
        )
        self._update_force_stream(
            env_ids_all,
            interval=self.push_interval_base_ext,
            interval_min=self.push_interval_base_ext_min,
            interval_max=self.push_interval_base_ext_max,
            duration=self.push_duration_base_ext,
            duration_min=self.push_duration_base_ext_min,
            duration_max=self.push_duration_base_ext_max,
            settling_time=self.settling_time_force_base,
            forced_prob=self.cfg.commands.base_forced_prob_ext,
            selected=self.selected_env_ids_base_ext,
            freed=self.freed_envs_base_ext,
            target=self.force_target_base_ext,
            output=self.base_force_ext_world,
            force_range=self.cfg.commands.max_push_force_xyz_base_ext,
            z_scale=self.cfg.commands.force_z_base_ext_scale,
        )
        if hasattr(self.simulator, "apply_base_force"):
            self.simulator.apply_base_force(self.base_force_ext_world)

    def _reset_dofs(self, env_ids):
        dof_pos = self.simulator.default_dof_pos.repeat(len(env_ids), 1)
        dof_pos += torch_rand_float(-0.3, 0.3, dof_pos.shape, self.device)
        dof_vel = torch.zeros_like(dof_pos)
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)

    def _reset_root_states(self, env_ids):
        base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
        base_pos += self.simulator._env_origins[env_ids]
        base_pos[:, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), self.device)
        base_quat = self.simulator.base_init_quat.reshape(1, -1).repeat(len(env_ids), 1)
        base_lin_vel = torch_rand_float(-0.1, 0.1, (len(env_ids), 3), self.device)
        base_ang_vel = torch_rand_float(-0.1, 0.1, (len(env_ids), 3), self.device)
        self.simulator.reset_root_states(env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel)

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return
        distance = torch.norm(self.simulator.base_pos[env_ids, :2] - self.simulator._env_origins[env_ids, :2], dim=1)
        move_up = distance > self.simulator._terrain.env_length / 2
        move_down = distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        self.simulator.update_terrain_curriculum(env_ids, move_up, move_down & ~move_up)

    def _update_command_curriculum(self, env_ids):
        if "tracking_lin_vel_force_world" not in self.episode_sums:
            return
        mean_tracking = torch.mean(self.episode_sums["tracking_lin_vel_force_world"][env_ids]) / self.max_episode_length
        if mean_tracking > self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel_force_world"]:
            for key in ["lin_vel_x", "lin_vel_y", "ang_vel_yaw"]:
                self.command_ranges[key][0] = np.clip(self.command_ranges[key][0] - 0.1, -self.cfg.commands.max_curriculum, 0.0)
                self.command_ranges[key][1] = np.clip(self.command_ranges[key][1] + 0.1, 0.0, self.cfg.commands.max_curriculum)

    def step_reward_curriculum(self, num_iters):
        if not self.use_reward_curriculum:
            return
        if num_iters < self.reward_warmup_steps:
            alpha = 0.0
        else:
            alpha = np.clip((num_iters - self.reward_warmup_steps) / max(1, self.reward_curr_steps), 0.0, 1.0)
            alpha = 0.5 * (1.0 - np.cos(np.pi * alpha))
        for key in self.reward_curr_keys:
            if key in self.reward_scales:
                low, high = self.reward_curr_bounds[key]
                self.reward_scales[key] = (low + (high - low) * alpha) * self.dt

    def _prepare_reward_function(self):
        for key in list(self.reward_scales.keys()):
            if self.reward_scales[key] == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name in self.reward_scales.keys():
            if name == "termination":
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, "_reward_" + name))
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self.reward_scales.keys()
        }

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(self.cfg.env.num_observations, device=self.device)
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:2] = noise_scales.gravity * noise_level
        noise_vec[2:5] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[5:22] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[22:39] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        return noise_vec

    def _init_buffers(self):
        self.common_step_counter = 0
        self.extras = {}
        self.forward_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.forward_vec[:, 0] = 1.0
        self.fail_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, device=self.device)
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.ee_sphe_radius_cmd,
                self.obs_scales.ee_sphe_pitch_cmd,
                self.obs_scales.ee_sphe_yaw_cmd,
                1.0,
                1.0,
                1.0,
                self.obs_scales.ee_force,
                self.obs_scales.ee_force,
                self.obs_scales.ee_force,
                self.obs_scales.base_force,
                self.obs_scales.base_force,
                self.obs_scales.base_force,
            ],
            device=self.device,
        )
        self.ee_sphere_scale = torch.tensor(
            [
                self.obs_scales.ee_sphe_radius_cmd,
                self.obs_scales.ee_sphe_pitch_cmd,
                self.obs_scales.ee_sphe_yaw_cmd,
            ],
            device=self.device,
        )
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.llast_actions = torch.zeros_like(self.actions)
        self.feet_air_time = torch.zeros(self.num_envs, len(self.simulator.feet_indices), device=self.device)
        self.last_contacts = torch.zeros_like(self.feet_air_time)

        self.obs_history_deque = deque(maxlen=self.cfg.env.num_obs_hist)
        for _ in range(self.cfg.env.num_obs_hist):
            self.obs_history_deque.append(torch.zeros(self.num_envs, self.cfg.env.num_observations, device=self.device))
        self.obs_history = torch.zeros(self.num_envs, self.cfg.env.num_observations * self.cfg.env.num_obs_hist, device=self.device)
        self.last_obs_buf = torch.zeros_like(self.obs_buf)
        self.llast_obs_buf = torch.zeros_like(self.obs_buf)
        self.last_obs_hist = torch.zeros_like(self.obs_history)
        self.llast_obs_hist = torch.zeros_like(self.obs_history)

        self.critic_obs_deque = deque(maxlen=self.cfg.env.num_priv_stack)
        for _ in range(self.cfg.env.num_priv_stack):
            self.critic_obs_deque.append(torch.zeros(self.num_envs, self.cfg.env.num_privileged_obs, device=self.device))
        self.explicit_labels_buf = torch.zeros(self.num_envs, self.cfg.env.num_explicit_recon_obs, device=self.device)

        self.ref_dof_pos = self.simulator.default_dof_pos[:, :12].repeat(self.num_envs, 1)
        self.ee_goal_center_offset = torch.tensor(
            [
                self.cfg.goal_ee.sphere_center.x_offset,
                self.cfg.goal_ee.sphere_center.y_offset,
                self.cfg.goal_ee.sphere_center.z_invariant_offset,
            ],
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.init_end_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_end, device=self.device).unsqueeze(0)
        self.ee_goal_sphere = self.init_end_ee_sphere.repeat(self.num_envs, 1)
        self.curr_ee_goal_sphere = self.ee_goal_sphere.clone()
        self.ee_start_sphere = self.ee_goal_sphere.clone()
        self.curr_ee_goal_cart = sphere2cart(self.curr_ee_goal_sphere)
        self.curr_ee_goal_cart_world = torch.zeros_like(self.curr_ee_goal_cart)
        self.ee_pos_sphe_arm = torch.zeros_like(self.curr_ee_goal_cart)
        self.traj_timesteps = torch.ones(self.num_envs, device=self.device) / self.dt
        self.traj_total_timesteps = self.traj_timesteps.clone()
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)

        self.ee_force_ext_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_force_ext_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.current_Fxyz_gripper_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.current_Fxyz_base_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.freed_envs_gripper_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.freed_envs_gripper_ext = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.selected_env_ids_gripper_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.selected_env_ids_gripper_ext = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.push_interval_gripper_cmd = self._rand_force_interval(
            self.push_interval_gripper_cmd_min,
            self.push_interval_gripper_cmd_max,
            (self.num_envs,),
        )
        self.push_interval_gripper_ext = self._rand_force_interval(
            self.push_interval_gripper_ext_min,
            self.push_interval_gripper_ext_max,
            (self.num_envs,),
        )
        self.push_end_time_gripper_cmd = torch.zeros(self.num_envs, device=self.device)
        self.push_end_time_gripper_ext = torch.zeros(self.num_envs, device=self.device)
        self.push_duration_gripper_cmd = torch.zeros(self.num_envs, device=self.device)
        self.push_duration_gripper_ext = torch.zeros(self.num_envs, device=self.device)
        self.force_target_gripper_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.force_target_gripper_ext = torch.zeros(self.num_envs, 3, device=self.device)

        self.freed_envs_base_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.freed_envs_base_ext = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.selected_env_ids_base_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.selected_env_ids_base_ext = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.push_interval_base_cmd = self._rand_force_interval(
            self.push_interval_base_cmd_min,
            self.push_interval_base_cmd_max,
            (self.num_envs,),
        )
        self.push_interval_base_ext = self._rand_force_interval(
            self.push_interval_base_ext_min,
            self.push_interval_base_ext_max,
            (self.num_envs,),
        )
        self.push_end_time_base_cmd = torch.zeros(self.num_envs, device=self.device)
        self.push_end_time_base_ext = torch.zeros(self.num_envs, device=self.device)
        self.push_duration_base_cmd = torch.zeros(self.num_envs, device=self.device)
        self.push_duration_base_ext = torch.zeros(self.num_envs, device=self.device)
        self.force_target_base_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.force_target_base_ext = torch.zeros(self.num_envs, 3, device=self.device)
        self.gripper_force_kps = torch_rand_float(*self.cfg.commands.gripper_force_kp_range, (self.num_envs, 1), self.device)
        self.gripper_force_kds = torch_rand_float(*self.cfg.commands.gripper_force_kd_range, (self.num_envs, 1), self.device)
        self.base_force_kps = torch_rand_float(*self.cfg.commands.base_force_kp_range, (self.num_envs, 1), self.device)
        self.base_force_kds = torch_rand_float(*self.cfg.commands.base_force_kd_range, (self.num_envs, 1), self.device)
        self._randomize_force_gains(torch.arange(self.num_envs, device=self.device))

        self.noise_scale_vec = self._get_noise_scale_vec()
        self.add_noise = self.cfg.noise.add_noise

        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(
                self.num_envs,
                self.cfg.domain_rand.ctrl_delay_step_range[1] + 1,
                self.num_actions,
                device=self.device,
            )
            self.action_delay = torch.randint(
                self.cfg.domain_rand.ctrl_delay_step_range[0],
                self.cfg.domain_rand.ctrl_delay_step_range[1] + 1,
                (self.num_envs,),
                device=self.device,
            )
    def _parse_cfg(self, cfg, sim_device):
        self.dt = cfg.control.dt
        self.debug = cfg.env.debug
        self.num_obs_hist = cfg.env.num_obs_hist
        self.num_crit_obs_stack = cfg.env.num_priv_stack
        self.num_pred_obs = cfg.env.num_pred_obs
        self.num_exp_labels = cfg.env.num_explicit_recon_obs
        self.obs_scales = cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(cfg.rewards.scales)
        self.use_reward_curriculum = cfg.rewards.use_reward_curriculum
        self.reward_curr_keys = cfg.rewards.reward_curriculum.curr_reward_keys
        self.reward_curr_bounds = cfg.rewards.reward_curriculum.curr_reward_bounds
        self.reward_curr_steps = cfg.rewards.reward_curriculum.curr_steps
        self.reward_warmup_steps = cfg.rewards.reward_curriculum.warmup_steps
        self.command_ranges = class_to_dict(cfg.commands.ranges)
        self.max_episode_length_s = cfg.env.episode_length_s
        self.max_episode_length = int(np.ceil(self.max_episode_length_s / self.dt))
        self.friction_value_offset = (cfg.domain_rand.friction_range[0] + cfg.domain_rand.friction_range[1]) / 2
        self.kp_scale_offset = (cfg.domain_rand.kp_range[0] + cfg.domain_rand.kp_range[1]) / 2
        self.kd_scale_offset = (cfg.domain_rand.kd_range[0] + cfg.domain_rand.kd_range[1]) / 2
        cfg.domain_rand.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)
        cfg.runner_steps_per_iter = 24
        self.push_interval_gripper_cmd_min = np.ceil(cfg.commands.push_gripper_interval_s_cmd[0] / self.dt)
        self.push_interval_gripper_cmd_max = np.ceil(cfg.commands.push_gripper_interval_s_cmd[1] / self.dt)
        self.push_interval_gripper_ext_min = np.ceil(cfg.commands.push_gripper_interval_s_ext[0] / self.dt)
        self.push_interval_gripper_ext_max = np.ceil(cfg.commands.push_gripper_interval_s_ext[1] / self.dt)
        self.push_duration_gripper_cmd_min = np.ceil(cfg.commands.push_gripper_duration_s_cmd[0] / self.dt)
        self.push_duration_gripper_cmd_max = np.ceil(cfg.commands.push_gripper_duration_s_cmd[1] / self.dt)
        self.push_duration_gripper_ext_min = np.ceil(cfg.commands.push_gripper_duration_s_ext[0] / self.dt)
        self.push_duration_gripper_ext_max = np.ceil(cfg.commands.push_gripper_duration_s_ext[1] / self.dt)
        self.settling_time_force_gripper = np.ceil(cfg.commands.settling_time_force_gripper_s / self.dt)
        self.push_interval_base_cmd_min = np.ceil(cfg.commands.push_base_interval_s_cmd[0] / self.dt)
        self.push_interval_base_cmd_max = np.ceil(cfg.commands.push_base_interval_s_cmd[1] / self.dt)
        self.push_interval_base_ext_min = np.ceil(cfg.commands.push_base_interval_s_ext[0] / self.dt)
        self.push_interval_base_ext_max = np.ceil(cfg.commands.push_base_interval_s_ext[1] / self.dt)
        self.push_duration_base_cmd_min = np.ceil(cfg.commands.push_base_duration_s_cmd[0] / self.dt)
        self.push_duration_base_cmd_max = np.ceil(cfg.commands.push_base_duration_s_cmd[1] / self.dt)
        self.push_duration_base_ext_min = np.ceil(cfg.commands.push_base_duration_s_ext[0] / self.dt)
        self.push_duration_base_ext_max = np.ceil(cfg.commands.push_base_duration_s_ext[1] / self.dt)
        self.settling_time_force_base = np.ceil(cfg.commands.settling_time_force_base_s / self.dt)

        self.wb_dim = cfg.env.whole_body_dim
        self.grf_dim = cfg.env.grf_dim

    # Rewards
    def _reward_tracking_lin_vel_force_world(self):
        base_yaw_quat = quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            self.simulator.base_euler[:, 2],
        )
        force_offset = quat_rotate_inverse(base_yaw_quat, self.base_force_ext_world) + self.current_Fxyz_base_cmd
        force_offset = force_offset[:, :2] / self.base_force_kds
        error = torch.sum(torch.square(self.commands[:, :2] + force_offset - self.simulator.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        return torch.exp(-torch.square(self.commands[:, 2] - self.simulator.base_ang_vel[:, 2]) / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ee_force_world(self):
        base_yaw_quat = quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            self.simulator.base_euler[:, 2],
        )
        force_offset = (self.ee_force_ext_world + quat_apply(base_yaw_quat, self.current_Fxyz_gripper_cmd)) / self.gripper_force_kps
        target = self.curr_ee_goal_cart_world + force_offset
        error = torch.sum(torch.square(target - self.simulator.ee_pos), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_ee_sigma)

    def _reward_termination(self):
        return (self.reset_buf.bool() & ~self.time_out_buf.bool()).float()

    def _reward_alive(self):
        return torch.ones(self.num_envs, device=self.device)

    def _reward_lin_vel_z(self):
        return torch.square(self.simulator.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.simulator.base_ang_vel[:, :2]), dim=1)

    def _reward_roll(self):
        return torch.square(self.simulator.base_euler[:, 0])

    def _reward_base_height(self):
        base_height = torch.mean(self.simulator.base_pos[:, 2].unsqueeze(1) - self.simulator.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        return torch.sum(torch.square(self.simulator.torques[:, :17]), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.simulator.dof_vel[:, :17]), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.simulator.last_dof_vel[:, :12] - self.simulator.dof_vel[:, :12]) / self.dt), dim=1)

    def _reward_dof_vel_arm(self):
        return torch.sum(torch.square(self.simulator.dof_vel[:, 12:17]), dim=1)

    def _reward_dof_acc_arm(self):
        return torch.sum(torch.square((self.simulator.last_dof_vel[:, 12:17] - self.simulator.dof_vel[:, 12:17]) / self.dt), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions[:, :12] - self.actions[:, :12]), dim=1)

    def _reward_action_rate_arm(self):
        return torch.sum(torch.square(self.last_actions[:, 12:17] - self.actions[:, 12:17]), dim=1)

    def _reward_collision(self):
        if len(self.simulator.penalized_contact_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(
            (torch.norm(self.simulator.link_contact_forces[:, self.simulator.penalized_contact_indices, :], dim=-1) > 0.1).float(),
            dim=1,
        )

    def _reward_dof_pos_limits(self):
        limits = self.simulator.dof_pos_limits
        if limits.ndim == 3:
            limits = limits[0]
        out = -(self.simulator.dof_pos[:, :17] - limits[:17, 0]).clip(max=0.0)
        out += (self.simulator.dof_pos[:, :17] - limits[:17, 1]).clip(min=0.0)
        return torch.sum(out, dim=1)

    def _reward_torque_limits(self):
        limits = self.simulator.torque_limits
        if limits.ndim == 2:
            limits = limits[0]
        return torch.sum((torch.abs(self.simulator.torques[:, :17]) - limits[:17] * self.cfg.rewards.soft_torque_limit).clip(min=0.0), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.simulator.dof_pos[:, [0, 3, 6, 9]]), dim=1)

    def _reward_feet_contact_forces(self):
        return torch.sum((torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.0), dim=1)

    def _reward_feet_air_time(self):
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.0
        self.feet_air_time += self.dt
        rew = torch.sum((self.feet_air_time - 0.5) * contact.float(), dim=1)
        self.feet_air_time *= ~contact
        return rew

    def _reward_feet_height(self):
        return torch.mean(torch.square(self.simulator.feet_pos[:, :, 2] - 0.08), dim=1)

    def _reward_feet_height_high(self):
        return torch.mean((self.simulator.feet_pos[:, :, 2] - 0.18).clip(min=0.0), dim=1)

    def _reward_feet_drag(self):
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.0
        return torch.sum(torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1) * contact.float(), dim=1)

    def _reward_feet_pos_xy(self):
        return torch.sum(torch.square(self.simulator.feet_pos[:, :, :2] - self.simulator.base_pos[:, None, :2]), dim=(1, 2))

    def _reward_feet_contact_number(self):
        contact = (self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.0).float()
        return torch.exp(-torch.square(torch.sum(contact, dim=1) - 2.0))

    def _reward_stand_still(self):
        moving = torch.norm(self.commands[:, :3], dim=1) > 0.1
        return torch.sum(torch.abs(self.simulator.dof_pos[:, :12] - self.simulator.default_dof_pos[:, :12]), dim=1) * (~moving)

    def _reward_ref_dof_leg(self):
        return torch.exp(-torch.mean(torch.square(self.simulator.dof_pos[:, :12] - self.ref_dof_pos), dim=1))
