import math

import numpy as np
import torch

from legged_gym.envs.b1z1.b1z1_unifp.b1z1_unifp import B1Z1UniFP
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math_utils import quat_rotate_inverse, torch_rand_float


class B1UniFP(B1Z1UniFP):
    """Arm-free UniFP locomotion problem for the Unitree B1.

    The policy controls only the twelve leg joints. Its command is torso planar
    velocity/yaw plus a 3-D torso force command. The force-aware velocity target,
    external torso disturbances, gait shaping, terrain, and randomization retain
    the UniFP torso-force formulation without manipulation state or control.
    """

    def step(self, actions):
        actions = self._pre_sim_step(actions)
        self._diagnostic_previous_applied_actions.copy_(self._diagnostic_applied_actions)
        self._diagnostic_applied_actions.copy_(actions.detach())

        # actions = torch.zeros_like(actions)

        if self.force_randomization_active and self.cfg.commands.push_robot_base:
            self._push_robot_base(self.all_env_ids)
        self.simulator.step(actions)
        self.post_physics_step()
        clip = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip, clip)
        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip, clip)
        return (
            self.obs_buf, self.privileged_obs_buf, self.obs_history,
            self.explicit_labels_buf, self.rew_buf, self.reset_buf, self.extras,
            self.simulator._grfs_buf * self.obs_scales.grf,
        )

    def post_physics_step(self):
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.simulator.post_physics_step()
        self._post_physics_step_callback()
        self.compute_all_leg_jacobians(
            self.simulator.dof_pos.view(-1, 4, 3), out=self.leg_jacobians
        )
        self.compute_ref_state()
        self.check_termination()
        self.compute_reward()
        self._accumulate_rollout_diagnostics()
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        if self.cfg.sensor.add_depth:
            self.simulator.update_depth_images()
        self.compute_observations()
        if self.debug:
            self.simulator.draw_debug_vis()

    def _post_physics_step_callback(self):
        resample = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
        self._resample_commands(resample.nonzero(as_tuple=False).flatten())
        if self.cfg.commands.heading_command:
            self._update_heading_command()
        self._step_contact_targets()

    def begin_rollout_diagnostics(self):
        """Reset detached physical/gait aggregates for one policy rollout."""
        self._rollout_diagnostic_sums = {}
        self._rollout_diagnostic_counts = {}
        self._rollout_diagnostic_maxima = {}

    def _rollout_diagnostic_add(self, name, values, mask=None, square=False):
        values = values.detach().float()
        if mask is not None:
            mask = mask.detach().bool()
            values = values[mask]
        if values.numel() == 0:
            return
        if square:
            values = values.square()
        self._rollout_diagnostic_sums[name] = self._rollout_diagnostic_sums.get(
            name, values.new_zeros(())
        ) + values.sum()
        self._rollout_diagnostic_counts[name] = self._rollout_diagnostic_counts.get(
            name, values.new_zeros(())
        ) + values.new_tensor(values.numel())

    def _rollout_diagnostic_max(self, name, values):
        value = values.detach().float().max()
        self._rollout_diagnostic_maxima[name] = torch.maximum(
            self._rollout_diagnostic_maxima.get(name, value.new_zeros(())), value
        )

    @torch.no_grad()
    def _accumulate_rollout_diagnostics(self):
        """Collect control and gait measurements from the pre-reset simulator state."""
        if not hasattr(self, "_rollout_diagnostic_sums"):
            self.begin_rollout_diagnostics()

        actions = self._diagnostic_applied_actions[:, :12]
        motor_strength = self.simulator._motor_strength[:, :12]
        target = (
            self.simulator.default_dof_pos[:, :12]
            + motor_strength * actions * self.cfg.control.action_scale
        )
        previous_target = (
            self.simulator.default_dof_pos[:, :12]
            + motor_strength * self._diagnostic_previous_applied_actions[:, :12]
            * self.cfg.control.action_scale
        )
        dof_velocity = self.simulator.dof_vel[:, :12]
        dof_acceleration = (
            dof_velocity - self.simulator._last_dof_vel[:, :12]
        ) / self.dt
        torque = self.simulator.unclipped_torques[:, :12]
        torque_limits = self.simulator._torque_limits
        torque_limits = (
            torque_limits[:12].unsqueeze(0)
            if torque_limits.ndim == 1 else torque_limits[:, :12]
        )

        self._rollout_diagnostic_add(
            "Control/joint_target_tracking_rmse", target - self.simulator.dof_pos[:, :12], square=True
        )
        self._rollout_diagnostic_add(
            "Control/joint_target_delta_rms", target - previous_target, square=True
        )
        self._rollout_diagnostic_add("Control/dof_velocity_rms", dof_velocity, square=True)
        self._rollout_diagnostic_add("Control/dof_acceleration_rms", dof_acceleration, square=True)
        self._rollout_diagnostic_add("Control/torque_rms", torque, square=True)
        self._rollout_diagnostic_max("Control/torque_abs_max", torque.abs())
        at_limit = torque.abs() >= torque_limits.clamp_min(1.0e-12)
        self._rollout_diagnostic_add("Control/torque_limit_fraction", at_limit)
        # Absolute joint power is a useful shaking/energy diagnostic even when
        # positive and negative work alternate within one control interval.
        self._rollout_diagnostic_add(
            "Control/mechanical_power_mean", (torque * dof_velocity).abs()
        )

        contact = (
            self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 5.0
        )
        stance = self._get_gait_phase().bool()
        walking = self.get_walking_cmd_mask()
        contact_match = contact == stance
        foot_speed = torch.norm(self.simulator.feet_vel, dim=-1)
        foot_xy_speed = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        stance_contact = stance & contact
        valid_swing = (~stance) & (~contact) & walking.unsqueeze(1)

        if hasattr(self.simulator, "_height_around_feet"):
            terrain = self.simulator.height_around_feet
            if terrain.ndim == 4:
                terrain = terrain.flatten(2)
            terrain_height = terrain.max(dim=-1).values
        else:
            terrain_height = torch.zeros_like(self.simulator.feet_pos[:, :, 2])
        desired_foot_height = terrain_height + (
            self.cfg.rewards.foot_height_offset + self.cfg.rewards.foot_clearance_target
        )
        clearance_success = self.simulator.feet_pos[:, :, 2] >= desired_foot_height

        self._rollout_diagnostic_add("Gait/contact_match_fraction", contact_match)
        self._rollout_diagnostic_add(
            "Gait/stance_slip_speed_mean", foot_xy_speed, stance_contact
        )
        self._rollout_diagnostic_add(
            "Gait/swing_clearance_success_fraction", clearance_success, valid_swing
        )
        self._rollout_diagnostic_add("Gait/swing_foot_speed_mean", foot_speed, valid_swing)

        command_xy = self.commands[:, :2]
        command_speed = torch.norm(command_xy, dim=1)
        directed = command_speed > 1.0e-6
        command_direction = command_xy / command_speed.clamp_min(1.0e-6).unsqueeze(1)
        aligned_velocity = torch.sum(
            self.simulator.base_lin_vel[:, :2] * command_direction, dim=1
        )
        self._rollout_diagnostic_add(
            "Gait/backward_motion_fraction", aligned_velocity < 0.0, directed
        )
        vx = self.simulator.base_lin_vel[:, 0]
        self._rollout_diagnostic_add("Gait/vx_mean", vx)
        self._rollout_diagnostic_add("_Gait/vx_square", vx, square=True)
        self._rollout_diagnostic_add(
            "Gait/vx_tracking_mae", torch.abs(vx - self.commands[:, 0])
        )
        self._rollout_diagnostic_sums["_Gait/aligned_progress"] = (
            self._rollout_diagnostic_sums.get("_Gait/aligned_progress", vx.new_zeros(()))
            + (aligned_velocity[directed] * self.dt).sum()
        )
        self._rollout_diagnostic_sums["_Gait/commanded_progress"] = (
            self._rollout_diagnostic_sums.get("_Gait/commanded_progress", vx.new_zeros(()))
            + (command_speed[directed] * self.dt).sum()
        )

        diagonal_agreement = torch.stack(
            (contact[:, 0] == contact[:, 3], contact[:, 1] == contact[:, 2]), dim=1
        )
        self._rollout_diagnostic_add("Gait/diagonal_contact_agreement", diagonal_agreement)
        for foot_index, foot_name in enumerate(self.cfg.asset.foot_name):
            prefix = f"GaitPerFoot/{foot_name}"
            self._rollout_diagnostic_add(f"{prefix}/contact_duty_factor", contact[:, foot_index])
            self._rollout_diagnostic_add(
                f"{prefix}/contact_match_fraction", contact_match[:, foot_index]
            )
            self._rollout_diagnostic_add(
                f"{prefix}/stance_slip_speed_mean",
                foot_xy_speed[:, foot_index], stance_contact[:, foot_index],
            )
            self._rollout_diagnostic_add(
                f"{prefix}/clearance_success_fraction",
                clearance_success[:, foot_index], valid_swing[:, foot_index],
            )

        phase = self._get_phase()
        phase_wrapped = phase < self._diagnostic_previous_phase
        completed_cycle = phase_wrapped & self._diagnostic_cycle_active & walking
        self._rollout_diagnostic_add(
            "Gait/net_displacement_per_cycle",
            self._diagnostic_cycle_progress,
            completed_cycle,
        )
        self._diagnostic_cycle_progress[phase_wrapped] = 0.0
        self._diagnostic_cycle_active[phase_wrapped & walking] = True
        starts_cycle = (
            walking
            & (~self._diagnostic_cycle_active)
            & (phase <= self.dt / self.cfg.rewards.cycle_time + 1.0e-6)
        )
        self._diagnostic_cycle_active[starts_cycle] = True
        active_progress = self._diagnostic_cycle_active & directed
        self._diagnostic_cycle_progress[active_progress] += (
            aligned_velocity[active_progress] * self.dt
        )
        stopped = ~walking
        self._diagnostic_cycle_active[stopped] = False
        self._diagnostic_cycle_progress[stopped] = 0.0
        self._diagnostic_previous_phase.copy_(phase)

    @torch.no_grad()
    def get_rollout_diagnostics(self):
        """Finalize rollout aggregates as finite Python scalars for logging."""
        rms_metrics = {
            "Control/joint_target_tracking_rmse",
            "Control/joint_target_delta_rms",
            "Control/dof_velocity_rms",
            "Control/dof_acceleration_rms",
            "Control/torque_rms",
        }
        metrics = {}
        for name, total in self._rollout_diagnostic_sums.items():
            if name.startswith("_"):
                continue
            count = self._rollout_diagnostic_counts.get(name)
            value = total / count.clamp_min(1.0) if count is not None else total
            if name in rms_metrics:
                value = torch.sqrt(value.clamp_min(0.0))
            metrics[name] = value.item() if torch.isfinite(value) else 0.0
        for name, value in self._rollout_diagnostic_maxima.items():
            metrics[name] = value.item() if torch.isfinite(value) else 0.0

        vx_count = self._rollout_diagnostic_counts.get("Gait/vx_mean")
        if vx_count is not None and vx_count.item() > 0:
            vx_mean = self._rollout_diagnostic_sums["Gait/vx_mean"] / vx_count
            vx_second = self._rollout_diagnostic_sums["_Gait/vx_square"] / vx_count
            metrics["Gait/vx_std"] = torch.sqrt(
                (vx_second - vx_mean.square()).clamp_min(0.0)
            ).item()
        else:
            metrics["Gait/vx_std"] = 0.0
        actual = self._rollout_diagnostic_sums.get("_Gait/aligned_progress")
        desired = self._rollout_diagnostic_sums.get("_Gait/commanded_progress")
        metrics["Gait/aligned_progress_ratio"] = (
            (actual / desired.clamp_min(1.0e-12)).item()
            if actual is not None and desired is not None and desired.item() > 0.0 else 0.0
        )
        required = (
            "Control/joint_target_tracking_rmse", "Control/joint_target_delta_rms",
            "Control/dof_velocity_rms", "Control/dof_acceleration_rms",
            "Control/torque_rms", "Control/torque_abs_max",
            "Control/torque_limit_fraction", "Control/mechanical_power_mean",
            "Gait/contact_match_fraction", "Gait/stance_slip_speed_mean",
            "Gait/swing_clearance_success_fraction", "Gait/swing_foot_speed_mean",
            "Gait/backward_motion_fraction", "Gait/vx_mean", "Gait/vx_std",
            "Gait/vx_tracking_mae", "Gait/aligned_progress_ratio",
            "Gait/net_displacement_per_cycle", "Gait/diagonal_contact_agreement",
        )
        for name in required:
            metrics.setdefault(name, 0.0)
        for foot_name in self.cfg.asset.foot_name:
            for metric_name in (
                "contact_duty_factor", "contact_match_fraction",
                "stance_slip_speed_mean", "clearance_success_fraction",
            ):
                metrics.setdefault(f"GaitPerFoot/{foot_name}/{metric_name}", 0.0)
        return metrics

    def compute_ref_state(self):
        """Build the UniFP swing pose, then add the optimized thigh sweep."""
        # Keep the inherited thigh/calf swing trajectory and gait/contact clock
        # exactly intact. The sweep phase lead applies only to this added term.
        super().compute_ref_state()
        phase = self._get_phase()
        sweep_phase = torch.remainder(
            phase + self.cfg.rewards.sweep_phase_lead,
            1.0,
        )
        vx_command = self.commands[:, 0]
        # The gain maps signed commanded velocity directly to joint-angle
        # amplitude; the clamp bounds the reference at larger commands.
        raw_sweep = (
            self.cfg.rewards.sweep_velocity_gain
            * vx_command
            * torch.cos(2.0 * torch.pi * sweep_phase)
        )
        max_amplitude = self.cfg.rewards.max_sweep_amplitude
        sweep = torch.clamp(
            raw_sweep,
            min=-max_amplitude,
            max=max_amplitude,
        )

        idx = self.leg_dof_indices
        self.ref_dof_pos[:, idx["FR_thigh_joint"]] += sweep
        self.ref_dof_pos[:, idx["RL_thigh_joint"]] += sweep
        self.ref_dof_pos[:, idx["FL_thigh_joint"]] -= sweep
        self.ref_dof_pos[:, idx["RR_thigh_joint"]] -= sweep

    def set_training_iteration(self, iteration):
        """Set the true PPO iteration used by gait-guidance schedules."""
        iteration = int(iteration)
        if iteration < 0:
            raise ValueError("training iteration must be nonnegative")
        self.training_iteration = iteration

    def _get_gait_guidance_multiplier(self, initial, final):
        """Geometrically interpolate one guidance weight over PPO iterations."""
        if not self.cfg.rewards.gait_guidance_decay_enabled:
            return 1.0

        duration = self.cfg.rewards.gait_guidance_decay_iterations
        progress = min(max(self.training_iteration / duration, 0.0), 1.0)
        if progress <= 0.0:
            return float(initial)
        if progress >= 1.0:
            return float(final)

        # Epsilon permits a zero endpoint while preserving exact endpoint values.
        epsilon = 1.0e-12
        effective_initial = max(float(initial), epsilon)
        effective_final = max(float(final), epsilon)
        return effective_initial * math.exp(
            math.log(effective_final / effective_initial) * progress
        )

    def get_gait_guidance_multipliers(self):
        """Expose current schedule values for runner/TensorBoard diagnostics."""
        rewards = self.cfg.rewards
        return {
            "ref_dof_leg": self._get_gait_guidance_multiplier(
                rewards.ref_dof_leg_initial_multiplier,
                rewards.ref_dof_leg_final_multiplier,
            ),
            "feet_contact": self._get_gait_guidance_multiplier(
                rewards.feet_contact_initial_multiplier,
                rewards.feet_contact_final_multiplier,
            ),
        }

    def _reward_ref_dof_leg(self):
        raw_reward = super()._reward_ref_dof_leg()
        return raw_reward * self.get_gait_guidance_multipliers()["ref_dof_leg"]

    def _reward_feet_contact_number(self):
        raw_reward = super()._reward_feet_contact_number()
        return raw_reward * self.get_gait_guidance_multipliers()["feet_contact"]

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids, 0] = torch_rand_float(*self.command_ranges["lin_vel_x"], (len(env_ids), 1), self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(*self.command_ranges["lin_vel_y"], (len(env_ids), 1), self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(*self.command_ranges["ang_vel_yaw"], (len(env_ids), 1), self.device).squeeze(1)
        zero = torch.rand(len(env_ids), device=self.device) < self.cfg.commands.zero_vel_cmd_prob
        zero_ids = env_ids[zero]
        self.commands[zero_ids, :3] = 0.0

        self._reset_progress_statistics(env_ids)

    def _reset_progress_statistics(self, env_ids):
        if len(env_ids) == 0:
            return

        self.progress_delta_buffer[env_ids] = 0.0
        self.progress_desired_buffer[env_ids] = 0.0
        self.progress_valid_steps[env_ids] = 0

        self.last_progress_base_pos[env_ids] = (
            self.simulator.base_pos[env_ids, :2]
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.terrain.curriculum and self.init_done:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum and self.common_step_counter % self.max_episode_length == 0:
            self._update_command_curriculum(env_ids)

        base_force_cmd = torch.mean(torch.norm(self.current_Fxyz_base_cmd[env_ids], dim=1))
        base_force_ext = torch.mean(torch.norm(self.base_force_ext_world[env_ids], dim=1))
        contact_fail = torch.mean(self.contact_fail_buf[env_ids].float())
        self._resample_commands(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._reset_progress_statistics(env_ids)

        self.simulator.reset_idx(env_ids)

        for buf in (self.actions, self.last_actions, self.llast_actions):
            buf[env_ids] = 0.0
        if hasattr(self, "_diagnostic_applied_actions"):
            self._diagnostic_applied_actions[env_ids] = 0.0
            self._diagnostic_previous_applied_actions[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.feet_stance_time[env_ids] = 0.0
        self.valid_swing[env_ids] = False
        self.step_liftoff_pos[env_ids] = 0.0
        self.step_direction_world[env_ids] = 0.0
        self.step_max_progress[env_ids] = 0.0
        self.gait_indices[env_ids] = 0.0
        if hasattr(self, "_diagnostic_cycle_progress"):
            self._diagnostic_cycle_progress[env_ids] = 0.0
            self._diagnostic_previous_phase[env_ids] = 0.0
            self._diagnostic_cycle_active[env_ids] = False
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = False
        self.contact_fail_buf[env_ids] = False
        self.termination_contact_counter[env_ids] = 0
        self.base_force_ext_world[env_ids] = 0.0
        self.current_Fxyz_base_cmd[env_ids] = 0.0
        self.estimated_base_force_local[env_ids] = 0.0
        self.progress_vel_ema[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self._reset_force_events(env_ids)
        self._randomize_force_gains(env_ids)
        self.simulator.apply_base_force(self.base_force_ext_world)

        self.last_obs_buf[env_ids] = 0.0
        self.llast_obs_buf[env_ids] = 0.0
        for slot in self.obs_history_slots + self.critic_obs_slots:
            slot[env_ids] = 0.0
        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["base_force_cmd_norm"] = base_force_cmd
        self.extras["episode"]["base_force_ext_norm"] = base_force_ext
        self.extras["episode"]["contact_fail_rate"] = contact_fail
        self.extras["episode"]["force_randomization_active"] = float(self.force_randomization_active)
        if hasattr(self.simulator, "terrain_levels"):
            terrain_levels = self.simulator.terrain_levels.float()
            self.extras["episode"]["terrain_level_max"] = torch.max(terrain_levels)
            self.extras["episode"]["terrain_level_mean"] = torch.mean(terrain_levels)
        for log_name, range_name in (
            ("command_range_abs_max_x", "lin_vel_x"),
            ("command_range_abs_max_y", "lin_vel_y"),
            ("command_range_abs_max_yaw", "ang_vel_yaw"),
        ):
            command_min, command_max = self.command_ranges[range_name]
            self.extras["episode"][log_name] = max(abs(command_min), abs(command_max))
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] = 0.0
            self.action_delay[env_ids] = torch.randint(
                self.cfg.domain_rand.ctrl_delay_step_range[0],
                self.cfg.domain_rand.ctrl_delay_step_range[1] + 1,
                (len(env_ids),), device=self.device,
            )

    def _reset_dofs(self, env_ids):
        low, high = self.cfg.init_state.leg_dof_pos_perturb_range
        dof_pos = self.simulator.default_dof_pos.repeat(len(env_ids), 1)

        dof_pos += torch_rand_float(low, high, dof_pos.shape, self.device)

        # dof_pos += torch_rand_float(low, high, dof_pos.shape, self.device)

        self.simulator.reset_dofs(env_ids, dof_pos, torch.zeros_like(dof_pos))

    def compute_observations(self):
        self.llast_obs_buf.copy_(self.last_obs_buf)
        self.last_obs_buf.copy_(self.obs_buf)
        base_force_local = quat_rotate_inverse(self._get_base_yaw_quat(), self.base_force_ext_world)

        # # Express each world-frame foot position relative to the translating
        # # and rotating B1 torso. The flattened transform gives
        # # quat_rotate_inverse one quaternion-vector pair per foot.
        # feet_from_base_world = self.simulator.feet_pos - self.simulator.base_pos.unsqueeze(1)
        # base_quat_per_foot = self.simulator.base_quat.unsqueeze(1).expand(
        #     -1, feet_from_base_world.shape[1], -1
        # )
        # feet_pos_base = quat_rotate_inverse(
        #     base_quat_per_foot.reshape(-1, 4),
        #     feet_from_base_world.reshape(-1, 3),
        # ).view(self.num_envs, -1, 3)

        # print("Feet positions in base frame:", feet_pos_base[0])

        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        dof_pos_err = (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos
        dof_vel = self.simulator.dof_vel * self.obs_scales.dof_vel
        torch.cat((
            self.get_body_orientation(), 
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            dof_pos_err, 
            dof_vel,
            sin_pos,
            cos_pos, 
            self.actions, 
            self.commands * self.commands_scale,
        ), dim=-1, out=self.obs_buf)
        if self.add_noise:
            self.obs_buf += (2.0 * torch.rand_like(self.obs_buf) - 1.0) * self.noise_scale_vec
        contacts = (self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 5.0).float()
        # Terrain-relative foot clearance matches the PACT estimator target:
        # h_foot - mean(local terrain patch) - nominal clearance offset.
        foot_heights = torch.clip(
            self.simulator.feet_pos[:, :, 2]
            - torch.mean(self.simulator.height_around_feet, dim=-1)
            - self.cfg.rewards.foot_height_offset,
            -1.0,
            1.0,
        )
        # Explicit target ordering is shared with ActorCriticB1UniFP:
        # base velocity (3), base force (3), contacts (4), foot heights (4).
        torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            base_force_local * self.obs_scales.base_force,
            contacts,
            foot_heights,
        ), dim=-1, out=self.explicit_labels_buf)

        mass_params = self._mass_params_buf
        mass_params.zero_()
        mass_params[:, 0:1] = self.simulator._added_base_mass
        mass_params[:, 1:4] = self.simulator._base_com_bias
        stance = self._get_gait_phase()
        critic_components = (
            ("adaptation_labels", self.explicit_labels_buf),
            ("reference_dof_error", self.simulator.dof_pos - self.ref_dof_pos),
            ("mass_com", mass_params),
            ("friction_offset", self.simulator._friction_values - self.friction_value_offset),
            ("stance", stance),
            # Contacts are already carried by adaptation_labels above.
            ("phase_sin", sin_pos),
            ("phase_cos", cos_pos),
            ("projected_gravity", self.simulator.projected_gravity),
            ("base_angular_velocity", self.simulator.base_ang_vel * self.obs_scales.ang_vel),
            ("dof_position_error", dof_pos_err),
            ("dof_velocity", dof_vel),
            ("actions", self.actions),
            ("commands", self.commands * self.commands_scale),
            ("push_velocity", self.simulator._rand_push_vels),
            ("wrench_velocity", self.simulator._rand_wrench_vels),
            ("kp_scale", self.simulator._kp_scale - self.kp_scale_offset),
            ("kd_scale", self.simulator._kd_scale - self.kd_scale_offset),
            ("motor_strength", self.simulator._motor_strength),
            ("joint_armature", self.simulator._joint_armature),
            ("joint_friction", self.simulator._joint_friction),
            ("joint_damping", self.simulator._joint_damping),
        )
        critic_state = torch.cat([value for _, value in critic_components], dim=-1)
        if critic_state.shape[1] != self.cfg.env.num_critic_state_obs:
            layout = ", ".join(f"{name}={value.shape[1]}" for name, value in critic_components)
            raise RuntimeError(
                f"B1 UniFP critic state is {critic_state.shape[1]}D, expected "
                f"{self.cfg.env.num_critic_state_obs}D ({layout})"
            )
        self._critic_obs_buf.copy_(critic_state)

        critic_obs = self._critic_obs_buf

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.simulator.base_pos[:, 2:3] - 0.5 - self.simulator.measured_heights, -1.0, 1.0
            ) * self.obs_scales.height_measurements
            if heights.shape[1] != self.cfg.env.num_height_obs:
                raise RuntimeError(
                    f"B1 UniFP terrain observation is {heights.shape[1]}D, expected "
                    f"{self.cfg.env.num_height_obs}D"
                )
            self._critic_height_obs.copy_(heights)

        if self._critic_height_obs.shape[1] > 0:
            critic_obs = torch.cat((critic_obs, self._critic_height_obs), dim=-1)

        if critic_obs.shape[1] != self.cfg.env.num_privileged_obs:
            raise RuntimeError(f"B1 UniFP critic observation is {critic_obs.shape[1]}D, expected {self.cfg.env.num_privileged_obs}D")

        self._critic_obs_slot = (self._critic_obs_slot + 1) % len(self.critic_obs_slots)
        self.critic_obs_slots[self._critic_obs_slot].copy_(critic_obs)
        ordered = self.critic_obs_slots[self._critic_obs_slot + 1:] + self.critic_obs_slots[:self._critic_obs_slot + 1]
        torch.cat(ordered, dim=-1, out=self.privileged_obs_buf)
        self.llast_obs_hist.copy_(self.last_obs_hist)
        self.last_obs_hist.copy_(self.obs_history)
        self._obs_history_slot = (self._obs_history_slot + 1) % len(self.obs_history_slots)
        self.obs_history_slots[self._obs_history_slot].copy_(self.obs_buf)
        ordered = self.obs_history_slots[self._obs_history_slot + 1:] + self.obs_history_slots[:self._obs_history_slot + 1]
        torch.cat(ordered, dim=-1, out=self.obs_history)

    def set_impedance_force_estimates(self, obs_pred):
        if obs_pred.shape[1] != self.cfg.env.num_explicit_recon_obs:
            raise RuntimeError(
                f"Expected {self.cfg.env.num_explicit_recon_obs} B1 UniFP labels, got {obs_pred.shape[1]}"
            )
        self.estimated_base_force_local[:] = obs_pred[:, 3:6] / self.obs_scales.base_force

    def _randomize_force_gains(self, env_ids):
        if len(env_ids) and self.cfg.commands.randomize_base_force_gains:
            self.base_force_kps[env_ids] = torch_rand_float(*self.cfg.commands.base_force_kp_range, (len(env_ids), 1), self.device)
            self.base_force_kds[env_ids] = torch_rand_float(*self.cfg.commands.base_force_kd_range, (len(env_ids), 1), self.device)

    def _force_push_end_time_for(self, output):
        """Map B1 torso-force streams without probing manipulation buffers."""
        if output is self.current_Fxyz_base_cmd:
            return self.push_end_time_base_cmd
        if output is self.base_force_ext_world:
            return self.push_end_time_base_ext
        raise ValueError("Unknown B1 torso-force stream output tensor")

    def _reset_force_events(self, env_ids):
        for buf in (self.freed_envs_base_cmd, self.freed_envs_base_ext,
                    self.selected_env_ids_base_cmd, self.selected_env_ids_base_ext):
            buf[env_ids] = False
        for buf in (self.force_target_base_cmd, self.force_target_base_ext,
                    self.current_Fxyz_base_cmd, self.base_force_ext_world):
            buf[env_ids] = 0.0
        for buf in (self.push_end_time_base_cmd, self.push_end_time_base_ext,
                    self.push_duration_base_cmd, self.push_duration_base_ext):
            buf[env_ids] = 0.0
        self.push_interval_base_cmd[env_ids] = self._rand_force_interval(self.push_interval_base_cmd_min, self.push_interval_base_cmd_max, (len(env_ids),))
        self.push_interval_base_ext[env_ids] = self._rand_force_interval(self.push_interval_base_ext_min, self.push_interval_base_ext_max, (len(env_ids),))

    def _get_noise_scale_vec(self):
        vec = torch.zeros(self.cfg.env.num_observations, device=self.device)
        scales, level = self.cfg.noise.noise_scales, self.cfg.noise.noise_level
        vec[0:2] = scales.gravity * level
        vec[2:5] = scales.ang_vel * level * self.obs_scales.ang_vel
        vec[5:17] = scales.dof_pos * level * self.obs_scales.dof_pos
        vec[17:29] = scales.dof_vel * level * self.obs_scales.dof_vel
        self.height_noise_vec = torch.zeros(self.simulator._num_height_points, device=self.device)
        return vec

    def _init_buffers(self):
        self.common_step_counter = 0
        self.extras = {}
        self.all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.forward_vec = torch.zeros(self.num_envs, 3, device=self.device); self.forward_vec[:, 0] = 1.0
        self.fail_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_fail_buf = torch.zeros_like(self.fail_buf)
        self.termination_contact_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.commands = torch.zeros(self.num_envs, 6, device=self.device)
        self.heading_commands = torch.zeros(self.num_envs, device=self.device)

        self.commands_scale = torch.tensor([
            self.obs_scales.lin_vel, 
            self.obs_scales.lin_vel, 
            self.obs_scales.ang_vel,
            self.obs_scales.base_force, 
            self.obs_scales.base_force, 
            self.obs_scales.base_force,
        ], device=self.device)

        self.actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.last_actions = torch.zeros_like(self.actions); self.llast_actions = torch.zeros_like(self.actions)
        self._diagnostic_applied_actions = torch.zeros_like(self.actions)
        self._diagnostic_previous_applied_actions = torch.zeros_like(self.actions)
        self.feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        # The inherited air-time/early-swing rewards share these stateful
        # buffers and cache their update once per control step.
        self.feet_stance_time = torch.zeros_like(self.feet_air_time)
        self.valid_swing = torch.zeros_like(self.last_contacts)
        self._feet_stats_update_step = -1
        self._feet_stats = {}

        self.progress_vel_ema = torch.zeros(self.num_envs, 2, device=self.device)

        self.progress_window_s = 0.40
        self.progress_window_steps = max(
            1, int(round(self.progress_window_s / self.dt))
        )

        self.progress_delta_buffer = torch.zeros(
            self.num_envs,
            self.progress_window_steps,
            2,
            device=self.device,
        )

        self.progress_desired_buffer = torch.zeros_like(
            self.progress_delta_buffer
        )

        self.progress_valid_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self.progress_buffer_index = 0
        self.last_progress_base_pos = self.simulator.base_pos[:, :2].clone()
        self._progress_update_step = -1

        self.step_liftoff_pos = torch.zeros(
            self.num_envs, 4, 2, device=self.device
        )
        self.step_direction_world = torch.zeros_like(
            self.step_liftoff_pos
        )
        self.step_max_progress = torch.zeros(
            self.num_envs, 4, device=self.device
        )

        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self._diagnostic_cycle_progress = torch.zeros(self.num_envs, device=self.device)
        self._diagnostic_previous_phase = torch.zeros(self.num_envs, device=self.device)
        self._diagnostic_cycle_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.begin_rollout_diagnostics()
        # Play/evaluation defaults to iteration zero until a runner supplies one.
        self.training_iteration = 0
        self.obs_history_slots = [torch.zeros_like(self.obs_buf) for _ in range(self.cfg.env.num_obs_hist)]
        self._obs_history_slot = len(self.obs_history_slots) - 1
        self.obs_history = torch.zeros(self.num_envs, self.num_obs * self.num_obs_hist, device=self.device)
        self.last_obs_buf = torch.zeros_like(self.obs_buf); self.llast_obs_buf = torch.zeros_like(self.obs_buf)
        self.last_obs_hist = torch.zeros_like(self.obs_history); self.llast_obs_hist = torch.zeros_like(self.obs_history)
        self.critic_obs_slots = [torch.zeros(self.num_envs, self.cfg.env.num_privileged_obs, device=self.device) for _ in range(self.cfg.env.num_priv_stack)]
        self._critic_obs_slot = len(self.critic_obs_slots) - 1
        expected_privileged_obs = self.cfg.env.num_critic_state_obs + self.cfg.env.num_height_obs
        if expected_privileged_obs != self.cfg.env.num_privileged_obs:
            raise ValueError(
                "B1 privileged-observation config is inconsistent: "
                f"state ({self.cfg.env.num_critic_state_obs}) + heights "
                f"({self.cfg.env.num_height_obs}) != total ({self.cfg.env.num_privileged_obs})"
            )
        configured_height_points = len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y)
        if configured_height_points != self.cfg.env.num_height_obs:
            raise ValueError(
                f"B1 terrain grid contains {configured_height_points} points, expected "
                f"{self.cfg.env.num_height_obs}"
            )
        self._critic_obs_buf = torch.zeros(
            self.num_envs, self.cfg.env.num_critic_state_obs, device=self.device
        )
        self._critic_height_obs = torch.zeros(
            self.num_envs,
            self.cfg.env.num_height_obs,
            device=self.device,
        )
        self._mass_params_buf = torch.zeros(self.num_envs, 22, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.cfg.env.num_privileged_obs * self.cfg.env.num_priv_stack, device=self.device)
        self.explicit_labels_buf = torch.zeros(
            self.num_envs, self.cfg.env.num_explicit_recon_obs, device=self.device
        )
        self.leg_dof_indices = {name: self.cfg.asset.dof_names.index(name) for name in (
            "FL_thigh_joint", "FL_calf_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_thigh_joint", "RL_calf_joint", "RR_thigh_joint", "RR_calf_joint")}
        self.ref_dof_pos = self.simulator.default_dof_pos.repeat(self.num_envs, 1)
        self._base_yaw_quat_buf = torch.zeros(self.num_envs, 4, device=self.device); self._base_yaw_quat_buf[:, 3] = 1.0
        self._base_yaw_quat_subset_buf = torch.zeros_like(self._base_yaw_quat_buf); self._base_yaw_quat_subset_buf[:, 3] = 1.0
        self.base_force_ext_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.current_Fxyz_base_cmd = torch.zeros_like(self.base_force_ext_world)
        self.estimated_base_force_local = torch.zeros_like(self.base_force_ext_world)
        for name in ("freed_envs_base_cmd", "freed_envs_base_ext", "selected_env_ids_base_cmd", "selected_env_ids_base_ext"):
            setattr(self, name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        for name in ("push_end_time_base_cmd", "push_end_time_base_ext", "push_duration_base_cmd", "push_duration_base_ext"):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))
        self.push_interval_base_cmd = self._rand_force_interval(self.push_interval_base_cmd_min, self.push_interval_base_cmd_max, (self.num_envs,))
        self.push_interval_base_ext = self._rand_force_interval(self.push_interval_base_ext_min, self.push_interval_base_ext_max, (self.num_envs,))
        self.force_target_base_cmd = torch.zeros_like(self.base_force_ext_world); self.force_target_base_ext = torch.zeros_like(self.base_force_ext_world)
        self.base_force_kps = torch.ones(self.num_envs, 1, device=self.device); self.base_force_kds = torch.ones_like(self.base_force_kps)
        self._randomize_force_gains(self.all_env_ids)
        self.leg_jacobians = torch.zeros(self.num_envs, 4, 3, 3, device=self.device)
        self._abad_link_length = torch.tensor(self.cfg.asset.abad_link_length, device=self.device)
        self._hip_link_length = torch.tensor(self.cfg.asset.hip_link_length, device=self.device)
        self._knee_link_length = torch.tensor(self.cfg.asset.knee_link_length, device=self.device)
        self._leg_side_sign = torch.tensor(self.cfg.asset.side_signs, device=self.device).view(1, 4)
        self.noise_scale_vec = self._get_noise_scale_vec(); self.add_noise = self.cfg.noise.add_noise
        if self.cfg.domain_rand.randomize_ctrl_delay:
            max_delay = self.cfg.domain_rand.ctrl_delay_step_range[1]
            self.action_queue = torch.zeros(self.num_envs, max_delay + 1, 12, device=self.device)
            self.action_delay = torch.randint(0, max_delay + 1, (self.num_envs,), device=self.device)

    def _parse_cfg(self, cfg, sim_device):
        if cfg.rewards.cycle_time <= 0.0:
            raise ValueError("rewards.cycle_time must be positive")
        if cfg.rewards.sweep_velocity_gain < 0.0:
            raise ValueError("rewards.sweep_velocity_gain must be nonnegative")
        if cfg.rewards.max_sweep_amplitude < 0.0:
            raise ValueError("rewards.max_sweep_amplitude must be nonnegative")
        if cfg.rewards.gait_guidance_decay_iterations <= 0:
            raise ValueError("rewards.gait_guidance_decay_iterations must be positive")
        for name in (
            "ref_dof_leg_initial_multiplier",
            "ref_dof_leg_final_multiplier",
            "feet_contact_initial_multiplier",
            "feet_contact_final_multiplier",
        ):
            if getattr(cfg.rewards, name) < 0.0:
                raise ValueError(f"rewards.{name} must be nonnegative")
        self.dt = cfg.control.dt; self.debug = cfg.env.debug
        self.num_obs_hist = cfg.env.num_obs_hist; self.num_crit_obs_stack = cfg.env.num_priv_stack
        self.num_pred_obs = cfg.env.num_pred_obs; self.num_exp_labels = cfg.env.num_explicit_recon_obs
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
        self.friction_value_offset = sum(cfg.domain_rand.friction_range) / 2
        self.kp_scale_offset = sum(cfg.domain_rand.kp_range) / 2
        self.kd_scale_offset = sum(cfg.domain_rand.kd_range) / 2
        cfg.domain_rand.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)
        cfg.runner_steps_per_iter = cfg.env.num_steps_per_env
        for kind in ("cmd", "ext"):
            setattr(self, f"push_interval_base_{kind}_min", np.ceil(getattr(cfg.commands, f"push_base_interval_s_{kind}")[0] / self.dt))
            setattr(self, f"push_interval_base_{kind}_max", np.ceil(getattr(cfg.commands, f"push_base_interval_s_{kind}")[1] / self.dt))
            setattr(self, f"push_duration_base_{kind}_min", np.ceil(getattr(cfg.commands, f"push_base_duration_s_{kind}")[0] / self.dt))
            setattr(self, f"push_duration_base_{kind}_max", np.ceil(getattr(cfg.commands, f"push_base_duration_s_{kind}")[1] / self.dt))
        self.settling_time_force_base = np.ceil(cfg.commands.settling_time_force_base_s / self.dt)
        self.wb_dim = cfg.env.whole_body_dim; self.grf_dim = cfg.env.grf_dim
