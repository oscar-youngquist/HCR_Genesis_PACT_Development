from collections import deque

import numpy as np
import torch

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math_utils import (
    quat_apply,
    quat_apply_yaw,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
    torch_rand_float,
    wrap_to_pi,
)

from legged_gym.envs.b1z1.b1z1_unifp.b1z1_reward_helpers import _eig_desc, _geom_mean, _interval_reward, _safe_inv, _safe_normalize, _safe_pinv, _sanitize_tensor, _skew, _upper_reward, _rot_x, _rot_y, _rot_z


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
    """Genesis port of the original UniFP B1/Z1 position-force baseline.

    The environment keeps the UniFP learning problem intact:
    - 17 learned position actions for 12 B1 leg joints + 5 Z1 arm joints.
    - A 15-D command vector: base velocity, EE spherical position, reservedself._reset_progress_statistics()
      orientation slots, EE force command, and base force command.
    - A CSE/adaptation target (`explicit_labels_buf`) containing base velocity,
      EE spherical state, EE external force, and base external force.
    - UniFP force-randomization streams that separately sample commanded force
      offsets and physically applied external disturbances.

    The surrounding API is shaped like the Genesis/PACT environments so the
    existing HCR runner, logging, and simulator wrappers can train the baseline.
    """

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
        self.reset_idx(self.all_env_ids)
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

        # UniFP delays force randomization until `force_start_step * steps_per_iter`.
        # After that point, commanded force offsets become part of the policy input,
        # and optional external forces are applied through the Genesis simulator.
        if self.force_randomization_active and self.cfg.commands.push_gripper_stators:
            self._push_gripper(self.all_env_ids)
        if self.force_randomization_active and self.cfg.commands.push_robot_base:
            self._push_robot_base(self.all_env_ids)

        # Play/eval can optionally mimic the paper's external impedance wrapper.
        # It must use estimator outputs, not simulator ground-truth forces.
        if self.cfg.commands.use_external_impedance_compensation:
            self._apply_external_impedance_compensation()

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
        # Original UniFP gates the force curriculum by PPO iteration count
        # (`force_start_step * 24`). `runner_steps_per_iter` makes that explicit.
        return self.common_step_counter > self.cfg.commands.force_start_step * self.cfg.runner_steps_per_iter

    def post_physics_step(self):
        # Match the Isaac-Gym UniFP order: refresh simulator state, resample
        # commands/goals, terminate, reward, reset, then build next observations.
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.simulator.post_physics_step()
        self._post_physics_step_callback()

        # Leg main-ip update specific update
        self.compute_all_leg_jacobians(
            self.simulator.dof_pos[:, 0:12].view(-1, 4, 3),
            out=self.leg_jacobians,
        )

        self._compute_z1_arm_jacobian_buffer()

        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        if getattr(self.cfg, "sensor", None) is not None and self.cfg.sensor.add_depth:
            self.simulator.update_depth_images()
        self.compute_observations()
        if self.debug:
            self.simulator.draw_debug_vis()

    def compute_z1_arm_jacobian(self, q_arm: torch.Tensor) -> torch.Tensor:
        """
        Compute translational EE Jacobian for the Z1 arm.

        Args:
            q_arm: shape (N, 6), ordered as:
                [
                    z1_waist,
                    z1_shoulder,
                    z1_elbow,
                    z1_wrist_angle,
                    z1_forearm_roll,
                    z1_wrist_rotate,
                ]

        Returns:
            J: shape (N, 3, 6)
        """
        assert q_arm.ndim == 2 and q_arm.shape[1] == 6, (
            f"Expected q_arm shape (N, 6), got {q_arm.shape}"
        )

        q_arm = torch.nan_to_num(q_arm, nan=0.0, posinf=0.0, neginf=0.0)

        N = q_arm.shape[0]
        device = q_arm.device
        dtype = q_arm.dtype

        # Cast constants only if needed.
        joint_offsets = self.z1_joint_offsets.to(device=device, dtype=dtype)
        joint_axes = self.z1_joint_axes.to(device=device, dtype=dtype)
        link00_offset = self.z1_link00_offset.to(device=device, dtype=dtype)
        ee_offset = self.z1_ee_offset.to(device=device, dtype=dtype)

        # Current transform from base to active frame.
        R = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3).clone()
        p = link00_offset.view(1, 3).expand(N, 3).clone()

        joint_pos = torch.empty(N, 6, 3, device=device, dtype=dtype)
        joint_axis_world = torch.empty(N, 6, 3, device=device, dtype=dtype)

        for i in range(6):
            # p = p + R @ offset_i
            p = p + torch.einsum("nij,j->ni", R, joint_offsets[i])

            # store joint origin
            joint_pos[:, i, :] = p

            # axis_world = R @ axis_local
            joint_axis_world[:, i, :] = torch.einsum("nij,j->ni", R, joint_axes[i])

            qi = q_arm[:, i]
            c = torch.cos(qi)
            s = torch.sin(qi)

            R_next = R.clone()

            if i == 0 or i == 4:
                # local z rotation
                # R = R @ Rz(q)
                r0 = R[:, :, 0].clone()
                r1 = R[:, :, 1].clone()

                R_next[:, :, 0] = c[:, None] * r0 + s[:, None] * r1
                R_next[:, :, 1] = -s[:, None] * r0 + c[:, None] * r1
                R_next[:, :, 2] = R[:, :, 2]

            elif i == 1 or i == 2 or i == 3:
                # local y rotation
                # R = R @ Ry(q)
                r0 = R[:, :, 0].clone()
                r2 = R[:, :, 2].clone()

                R_next[:, :, 0] = c[:, None] * r0 - s[:, None] * r2
                R_next[:, :, 1] = R[:, :, 1]
                R_next[:, :, 2] = s[:, None] * r0 + c[:, None] * r2

            else:
                # i == 5, local x rotation
                # R = R @ Rx(q)
                r1 = R[:, :, 1].clone()
                r2 = R[:, :, 2].clone()

                R_next[:, :, 0] = R[:, :, 0]
                R_next[:, :, 1] = c[:, None] * r1 + s[:, None] * r2
                R_next[:, :, 2] = -s[:, None] * r1 + c[:, None] * r2

            R = R_next

        p_ee = p + torch.einsum("nij,j->ni", R, ee_offset)

        r = p_ee[:, None, :] - joint_pos  # [N, 6, 3]

        # cross(axis, r), then transpose to [N, 3, 6]
        J_cols = torch.cross(joint_axis_world, r, dim=2)

        J = J_cols.transpose(1, 2).contiguous()

        return torch.nan_to_num(J, nan=0.0, posinf=1e6, neginf=-1e6)

    def _compute_z1_arm_jacobian_buffer(self):
        q_arm = self.simulator.dof_pos[:, self.simulator._arm_dof_cfg_ids]
        self.z1_arm_jacobian[:] = self.compute_z1_arm_jacobian(q_arm)

    def compute_all_leg_jacobians(self, q: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
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

        l1 = self._abad_link_length
        l2 = self._hip_link_length
        l3 = self._knee_link_length
        side_sign = self._leg_side_sign.expand(N, 4)
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

        J = self.leg_jacobians if out is None else out

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

        return torch.nan_to_num(J, nan=0.0, posinf=1e6, neginf=-1e6, out=J)

    def check_termination(self):
        self.fail_buf[:] = 0
        self.contact_fail_buf[:] = 0
        if len(self.simulator.termination_contact_indices) > 0:
            # Contact termination is based on net link contact force for the
            # configured termination links. A patience counter avoids resetting
            # on one-frame spikes.
            contact_force_norm = torch.norm(
                self.simulator.link_contact_forces[:, self.simulator.termination_contact_indices, :],
                dim=-1,
            )
            undesired_contact = torch.any(
                contact_force_norm > self.cfg.termination.contact_force_threshold,
                dim=1,
            )
            self.termination_contact_counter = torch.where(
                undesired_contact,
                self.termination_contact_counter + 1,
                torch.zeros_like(self.termination_contact_counter),
            )
            contact_patience_steps = max(1, int(self.cfg.termination.contact_patience_steps))
            self.contact_fail_buf |= self.termination_contact_counter >= contact_patience_steps
            self.fail_buf |= self.contact_fail_buf
        else:
            self.termination_contact_counter[:] = 0
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

        # HCR curricula run from reset-time episode statistics. The original
        # UniFP command curriculum is usually disabled for B1Z1, but the hook is
        # kept so this port can run the same machinery when enabled.
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum and self.common_step_counter % self.max_episode_length == 0:
            self._update_command_curriculum(env_ids)

        # Snapshot episode-level force/goal statistics before state is zeroed.
        episode_ee_goal_sphere = self.curr_ee_goal_sphere[env_ids].clone()
        episode_ee_force_cmd_norm = torch.mean(torch.norm(self.current_Fxyz_gripper_cmd[env_ids], dim=1))
        episode_ee_force_ext_norm = torch.mean(torch.norm(self.ee_force_ext_world[env_ids], dim=1))
        episode_base_force_cmd_norm = torch.mean(torch.norm(self.current_Fxyz_base_cmd[env_ids], dim=1))
        episode_base_force_ext_norm = torch.mean(torch.norm(self.base_force_ext_world[env_ids], dim=1))
        episode_contact_fail_rate = torch.mean(self.contact_fail_buf[env_ids].float())

        self._resample_commands(env_ids)
        self._resample_ee_goal(env_ids, is_init=True)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._reset_progress_statistics(env_ids)
        # Simulator reset applies Genesis-side domain randomization and writes
        # root/DOF state into the Genesis entity.
        self.simulator.reset_idx(env_ids)
        # The reset EE command starts from init_pos_start; refresh the world
        # cache after Genesis has accepted the randomized root pose/yaw.
        self.curr_ee_goal_cart[env_ids] = sphere2cart(self.curr_ee_goal_sphere[env_ids])
        self._refresh_curr_ee_goal_world(env_ids)

        # Clear action histories, force streams, and estimator state so the next
        # rollout segment starts with a clean UniFP command/force state.
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.llast_actions[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.feet_stance_time[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.valid_swing[env_ids] = False
        self.step_liftoff_pos[env_ids] = 0.0
        self.step_direction_world[env_ids] = 0.0
        self.step_max_progress[env_ids] = 0.0
        self.gait_indices[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        self.contact_fail_buf[env_ids] = 0
        self.termination_contact_counter[env_ids] = 0
        self.ee_force_ext_world[env_ids] = 0.0
        self.base_force_ext_world[env_ids] = 0.0
        self.current_Fxyz_gripper_cmd[env_ids] = 0.0
        self.current_Fxyz_base_cmd[env_ids] = 0.0
        self.estimated_ee_force_local[env_ids] = 0.0
        self.estimated_base_force_local[env_ids] = 0.0
        self.prev_ee_error[env_ids] = 0.0
        self.progress_vel_ema[env_ids] = 0.0

        self._reset_force_events(env_ids)
        self._randomize_force_gains(env_ids)

        # Force buffers live in the env, but Genesis consumes them in the
        # simulator. Push zeros immediately so reset environments do not keep
        # stale external forces for one control step.
        if hasattr(self.simulator, "apply_ee_force"):
            self.simulator.apply_ee_force(self.ee_force_ext_world)
        if hasattr(self.simulator, "apply_base_force"):
            self.simulator.apply_base_force(self.base_force_ext_world)

        self.last_obs_buf[env_ids] = 0.0
        self.llast_obs_buf[env_ids] = 0.0
        for obs_slot in self.obs_history_slots:
            obs_slot[env_ids] = 0.0
        for critic_slot in self.critic_obs_slots:
            critic_slot[env_ids] = 0.0

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
        self.extras["episode"]["contact_fail_rate"] = episode_contact_fail_rate
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
        """Build UniFP actor, critic, and adaptation-target observations.

        Actor input is the stacked noisy proprioceptive history. The adaptation
        decoder is trained to predict `explicit_labels_buf`, matching original
        UniFP's `obs_pred`: base velocity, EE spherical position, EE force, and
        base force. The critic gets a privileged stack containing these labels,
        domain-randomization variables, contact/gait state, and the same current
        policy-facing commands.
        """
        self.llast_obs_buf.copy_(self.last_obs_buf)
        self.last_obs_buf.copy_(self.obs_buf)

        base_quat = self.simulator.base_quat
        base_rpy = self.simulator.base_euler
        base_yaw_quat = self._get_base_yaw_quat()
        ee_center = self.get_ee_goal_spherical_center(base_yaw_quat)

        # UniFP represents the EE target/state in a yaw-aligned spherical frame
        # centered at the Z1 waist/workspace origin, not in raw world XYZ.
        ee_local_cart = quat_rotate_inverse(base_yaw_quat, self.simulator.ee_pos - ee_center)
        self.ee_pos_sphe_arm = cart2sphere(ee_local_cart)

        # Force labels are also yaw-frame quantities. These are privileged during
        # training and become estimator predictions at deployment/play time.
        ee_force_local = quat_rotate_inverse(base_yaw_quat, self.ee_force_ext_world)
        base_force_local = quat_rotate_inverse(base_yaw_quat, self.base_force_ext_world)

        # The commanded EE force acts like an external impedance offset: the
        # target position is shifted by force / virtual stiffness.
        force_offset_world = self.ee_force_ext_world + quat_apply(base_yaw_quat, self.current_Fxyz_gripper_cmd)
        ee_goal_offset_local = quat_rotate_inverse(
            base_yaw_quat,
            self.curr_ee_goal_cart_world + force_offset_world / self.gripper_force_kps - ee_center,
        )
        ee_goal_offset_sphere = cart2sphere(ee_goal_offset_local)

        phase = self._get_phase()
        self.compute_ref_state()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        
        body_orientation = self.get_body_orientation()
        dof_pos_err = (self.simulator.dof_pos[:, :17] - self.simulator.default_dof_pos[:, :17]) * self.obs_scales.dof_pos
        dof_vel = self.simulator.dof_vel[:, :17] * self.obs_scales.dof_vel

        # Current single-frame actor observation. The runner stacks this over
        # `num_obs_hist` frames before feeding the UniFP adaptation encoder.
        torch.cat(
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
            out=self.obs_buf,
        )
        if self.add_noise:
            self.obs_buf += (2.0 * torch.rand_like(self.obs_buf) - 1.0) * self.noise_scale_vec

        # Supervised target for the UniFP CSE/adaptation decoder.
        torch.cat(
            (
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,
                self.ee_pos_sphe_arm * self.ee_sphere_scale,
                ee_force_local * self.obs_scales.ee_force,
                base_force_local * self.obs_scales.base_force,
            ),
            dim=-1,
            out=self.explicit_labels_buf,
        )

        mass_params = self._mass_params_buf
        mass_params.zero_()
        mass_params[:, 0:1] = self.simulator._added_base_mass
        mass_params[:, 1:4] = self.simulator._base_com_bias
        stance_mask = self._get_gait_phase()
        contact_mask = (self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 5.0).float()

        # Privileged critic observation mirrors the original UniFP ordering:
        # estimator labels first, then randomized dynamics/contact/gait state,
        # current robot state, commands, and force-shifted EE target.
        critic_obs = self._critic_obs_buf
        torch.cat(
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
                # sin_pos,
                # cos_pos,
                self.commands * self.commands_scale,
                ee_goal_offset_sphere * self.ee_sphere_scale,
                self.simulator._rand_push_vels,                                  # 3
                self.simulator._rand_wrench_vels,                                # 3
                (self.simulator._kp_scale - self.kp_scale_offset),               # num_actions
                (self.simulator._kd_scale - self.kd_scale_offset),               # num_actions
                self.simulator._motor_strength,                                  # num_actions
                self.simulator._joint_armature,                                  # 1
                self.simulator._joint_friction,                                  # 1
                self.simulator._joint_damping,                                   # 1
            ),
            dim=-1,
            out=critic_obs,
        )

        # add height measurements to asymmetric critic if approperiate
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(1) - 0.5 \
                                 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements # 81

            if self.add_noise:
                heights *= self.height_noise_vec

            critic_obs = torch.cat((critic_obs, heights), dim=-1) # 207


        if critic_obs.shape[1] != self.cfg.env.num_privileged_obs:
            raise RuntimeError(
                f"B1Z1 UniFP privileged observation size mismatch: "
                f"got {critic_obs.shape[1]}, expected {self.cfg.env.num_privileged_obs}"
            )
        self._critic_obs_slot = (self._critic_obs_slot + 1) % len(self.critic_obs_slots)
        self.critic_obs_slots[self._critic_obs_slot].copy_(critic_obs[:, : self.cfg.env.num_privileged_obs])
        # Preserve deque semantics: concatenate slots in oldest -> newest order.
        ordered_critic_slots = self.critic_obs_slots[self._critic_obs_slot + 1 :] + self.critic_obs_slots[: self._critic_obs_slot + 1]
        torch.cat(ordered_critic_slots, dim=-1, out=self.privileged_obs_buf)

        self.llast_obs_hist.copy_(self.last_obs_hist)
        self.last_obs_hist.copy_(self.obs_history)
        self._obs_history_slot = (self._obs_history_slot + 1) % len(self.obs_history_slots)
        self.obs_history_slots[self._obs_history_slot].copy_(self.obs_buf)
        # Preserve deque semantics: concatenate slots in oldest -> newest order.
        ordered_obs_slots = self.obs_history_slots[self._obs_history_slot + 1 :] + self.obs_history_slots[: self._obs_history_slot + 1]
        torch.cat(ordered_obs_slots, dim=-1, out=self.obs_history)

    def _get_base_yaw_quat(self, env_ids=None):
        """Return yaw-only xyzw quaternions without full Euler conversion."""
        if env_ids is None:
            yaw = self.simulator.base_euler[:, 2]
            out = self._base_yaw_quat_buf
        else:
            n = len(env_ids)
            yaw = self.simulator.base_euler[env_ids, 2]
            out = self._base_yaw_quat_subset_buf[:n]
        half_yaw = 0.5 * yaw
        out[:, 0:2] = 0.0
        out[:, 2] = torch.sin(half_yaw)
        out[:, 3] = torch.cos(half_yaw)
        return out

    def set_viewer_camera(self, pos, lookat):
        """Set viewer camera position and direction."""
        self.simulator.set_viewer_camera(eye=pos, target=lookat)

    def set_camera(self, pos, lookat):
        """Set the play-script camera using the available Genesis camera path."""
        floating_camera = getattr(self.simulator, "_floating_camera", None)
        if floating_camera is not None:
            floating_camera.set_pose(pos=pos, lookat=lookat)
        else:
            self.set_viewer_camera(pos, lookat)

    def draw_ee_goal_debug_vis(self, env_ids=None):
        """Draw UniFP-style EE target markers in the Genesis viewer."""
        if self.headless or not self.cfg.env.render_ee_goal_debug:
            return
        scene = getattr(self.simulator, "_scene", None)
        if scene is None:
            return

        if env_ids is None:
            rendered_envs = getattr(self.cfg.viewer, "rendered_envs_idx", [0])
            env_ids = rendered_envs[: min(len(rendered_envs), 16)]
        if len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        base_yaw_quat = self._get_base_yaw_quat(env_ids)
        # ee_center = self.simulator.base_pos[env_ids] + quat_apply(base_yaw_quat, self.ee_goal_center_offset[env_ids])
        ee_center = self.get_ee_goal_spherical_center(base_yaw_quat, env_ids)
        forces_cmd_world = quat_apply(base_yaw_quat, self.current_Fxyz_gripper_cmd[env_ids])
        force_offset = (self.ee_force_ext_world[env_ids] + forces_cmd_world) / self.gripper_force_kps[env_ids]
        ee_goal_offset_world = self.curr_ee_goal_cart_world[env_ids] + force_offset
        ee_pos = self.simulator.ee_pos[env_ids]

        scene.clear_debug_objects()
        scene.draw_debug_spheres(
            self.curr_ee_goal_cart_world[env_ids],
            radius=0.05,
            color=(1.0, 1.0, 0.0, 0.8),
        )
        scene.draw_debug_spheres(
            ee_goal_offset_world,
            radius=0.05,
            color=(1.0, 0.0, 1.0, 0.8),
        )
        scene.draw_debug_spheres(
            ee_pos,
            radius=0.05,
            color=(0.0, 0.0, 1.0, 0.8),
        )
        scene.draw_debug_spheres(
            ee_center,
            radius=0.05,
            color=(0.0, 1.0, 1.0, 0.8),
        )
        if self.cfg.env.render_ee_frame_debug:
            tcp_from_link06 = self._get_debug_tcp_from_link06(env_ids)
            if tcp_from_link06 is not None:
                scene.draw_debug_spheres(
                    tcp_from_link06,
                    radius=0.035,
                    color=(1.0, 0.5, 0.0, 0.9),
                )

        force_vec = force_offset.detach().cpu().numpy()
        goal_pos = self.curr_ee_goal_cart_world[env_ids].detach().cpu().numpy()
        for pos, vec in zip(goal_pos, force_vec):
            if np.linalg.norm(vec) > 1.0e-4:
                scene.draw_debug_arrow(
                    pos,
                    vec=vec,
                    radius=0.01,
                    color=(1.0, 0.0, 1.0, 0.8),
                )

    def _get_debug_tcp_from_link06(self, env_ids):
        """Reconstruct the URDF TCP from link06 for checking fixed-link frame alignment."""
        robot = getattr(self.simulator, "_robot", None)
        if robot is None:
            return None
        link06_index = getattr(self, "_debug_link06_index", None)
        if link06_index is None:
            link06_index = None
            for link in robot.links:
                if link.name == "link06":
                    link06_index = link.idx - robot.link_start
                    break
            self._debug_link06_index = link06_index
        if link06_index is None:
            return None

        link06_pos = robot.get_links_pos()[:, link06_index, :][env_ids]
        link06_quat_gs = robot.get_links_quat()[:, link06_index, :][env_ids]
        link06_quat = torch.empty_like(link06_quat_gs)
        link06_quat[:, :3] = link06_quat_gs[:, 1:4]
        link06_quat[:, 3] = link06_quat_gs[:, 0]
        tcp_offset = torch.tensor(
            self.cfg.goal_ee.debug_tcp_from_link06_offset,
            device=self.device,
            dtype=link06_pos.dtype,
        ).repeat(len(env_ids), 1)
        return link06_pos + quat_apply(link06_quat, tcp_offset)

    def set_impedance_force_estimates(self, obs_pred):
        """Set estimator-predicted local force values used by the play-time impedance controller."""
        if isinstance(obs_pred, np.ndarray):
            obs_pred = torch.as_tensor(obs_pred, device=self.device, dtype=torch.float)
        else:
            obs_pred = obs_pred.to(device=self.device, dtype=torch.float)
        if obs_pred.shape[1] < 12:
            raise RuntimeError(f"Expected at least 12 predicted UniFP labels, got {obs_pred.shape[1]}")

        # Decoder target layout: [base_vel(3), ee_sphere(3), ee_force(3), base_force(3)].
        self.estimated_ee_force_local[:] = obs_pred[:, 6:9] / self.obs_scales.ee_force
        self.estimated_base_force_local[:] = obs_pred[:, 9:12] / self.obs_scales.base_force

    def _apply_external_impedance_compensation(self):
        """Cancel force offsets using estimator-predicted local forces, not simulator force buffers."""
        if self.cfg.commands.compensate_ee_external_force:
            self.current_Fxyz_gripper_cmd[:] = -self.estimated_ee_force_local
        if self.cfg.commands.compensate_base_external_force:
            self.current_Fxyz_base_cmd[:] = -self.estimated_base_force_local
        self.commands[:, 9:12] = self.current_Fxyz_gripper_cmd
        self.commands[:, 12:15] = self.current_Fxyz_base_cmd

    def _pre_sim_step(self, actions):
        """Clip/store actor actions and optionally apply actuator delay."""
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions).to(self.device)
        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.actions[:] = actions[:]
        if self.cfg.domain_rand.randomize_ctrl_delay:
            # Genesis port inherits the HCR delay queue: the policy sees the
            # current observation but the simulator receives an older action.
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            actions = self.action_queue[self.all_env_ids, self.action_delay].clone()
        return actions

    def _post_physics_step_callback(self):
        """Periodic UniFP command updates after simulator state refresh."""
        env_ids = (
            self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0
        ).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._randomize_force_gains(env_ids)
        self._update_heading_command()
        self._step_contact_targets()
        self.update_curr_ee_goal()
        if self.cfg.domain_rand.push_robots:
            self.simulator.push_robots()

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        # Command slots 0:3 are base x/y/yaw velocity commands.
        self.commands[env_ids, 0] = torch_rand_float(*self.command_ranges["lin_vel_x"], (len(env_ids), 1), self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(*self.command_ranges["lin_vel_y"], (len(env_ids), 1), self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            # UniFP slots 3:6 are already EE spherical commands, so desired
            # heading is kept in a side buffer and converted to yaw rate.
            self.heading_commands[env_ids] = torch_rand_float(
                *self.command_ranges["heading"],
                (len(env_ids), 1),
                self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(*self.command_ranges["ang_vel_yaw"], (len(env_ids), 1), self.device).squeeze(1)

        # UniFP samples many standing commands; after force randomization starts
        # that probability increases so the force policy sees static balancing
        # cases under disturbance.
        zero_prob = self.cfg.commands.zero_vel_cmd_prob_after_force if self.force_randomization_active else self.cfg.commands.zero_vel_cmd_prob
        zero_mask = torch.rand(len(env_ids), device=self.device) < zero_prob
        zero_env_ids = env_ids[zero_mask]
        self.commands[zero_env_ids, :2] = 0.0
        if self.cfg.commands.heading_command:
            self.heading_commands[zero_env_ids] = self.simulator.base_euler[zero_env_ids, 2]
        else:
            self.commands[zero_env_ids, 2] = 0.0

        # Original UniFP snaps small sampled base commands to a full stop so
        # "standing" and "walking" are cleanly separated.
        if self.cfg.commands.heading_command:
            non_stop = (
                (torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip)
                | (torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_y_clip)
            )
        else:
            non_stop = (
                (torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip)
                | (torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_y_clip)
                | (torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip)
            )
        stop_env_ids = env_ids[~non_stop]
        self.commands[stop_env_ids, :2] = 0.0
        if self.cfg.commands.heading_command:
            self.heading_commands[stop_env_ids] = self.simulator.base_euler[stop_env_ids, 2]
        else:
            self.commands[stop_env_ids, 2] = 0.0
        self._resample_ee_goal(env_ids)

        self._reset_progress_statistics(env_ids)

    def _update_heading_command(self):
        """PACT-style desired heading conversion without using UniFP command slot 3."""
        if not self.cfg.commands.heading_command:
            return
        forward = quat_apply(self.simulator.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        normalized_heading_error = wrap_to_pi(self.heading_commands - heading) / torch.pi
        yaw_rate_min, yaw_rate_max = self.command_ranges["ang_vel_yaw"]

        # Map [-pi, 0, pi] heading error to [min yaw rate, 0, max yaw rate].
        # The piecewise mapping preserves a zero command for zero error while
        # supporting asymmetric limits and curriculum-adjusted command ranges.
        self.commands[:, 2] = torch.where(
            normalized_heading_error >= 0.0,
            normalized_heading_error * yaw_rate_max,
            -normalized_heading_error * yaw_rate_min,
        ).clamp(min=yaw_rate_min, max=yaw_rate_max)

    def _resample_ee_goal(self, env_ids, is_init=False):
        """Sample a new EE spherical target and timing profile.

        UniFP commands the EE as radius/pitch/yaw around a yaw-aligned arm
        workspace origin. Non-initial samples are rejected if the interpolated
        path passes through the configured B1 body collision box or underground.
        """
        if len(env_ids) == 0:
            return
        init_env_ids = env_ids.clone()
        if is_init:
            # Original UniFP reset starts each episode from init_pos_start and
            # interpolates toward init_pos_end instead of beginning at the final
            # target immediately.
            self.ee_start_sphere[env_ids] = self.init_start_ee_sphere
            self.ee_goal_sphere[env_ids] = self.init_end_ee_sphere
            self.curr_ee_goal_sphere[env_ids] = self.init_start_ee_sphere
            self.commands[env_ids, 3:6] = self.curr_ee_goal_sphere[env_ids]
        else:
            self.ee_start_sphere[env_ids] = self.curr_ee_goal_sphere[env_ids]
            remaining_env_ids = env_ids
            for _ in range(10):
                self._resample_ee_goal_sphere_once(remaining_env_ids)
                rejection_mask = self.collision_check(remaining_env_ids)
                # Keep only rejected envs in the loop; accepted envs retain the
                # most recent sample and stop resampling.
                remaining_env_ids = remaining_env_ids[rejection_mask]
                if len(remaining_env_ids) == 0:
                    break
        self.goal_timer[env_ids] = 0.0
        self.traj_timesteps[env_ids] = torch_rand_float(*self.cfg.goal_ee.traj_time, (len(env_ids), 1), self.device).squeeze(1) / self.dt
        self.traj_total_timesteps[env_ids] = self.traj_timesteps[env_ids] + (
            torch_rand_float(*self.cfg.goal_ee.hold_time, (len(env_ids), 1), self.device).squeeze(1) / self.dt
        )
        self.curr_ee_goal_cart[init_env_ids] = sphere2cart(self.curr_ee_goal_sphere[init_env_ids])
        self._refresh_curr_ee_goal_world(init_env_ids)

    def _resample_ee_goal_sphere_once(self, env_ids):
        if len(env_ids) == 0:
            return
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(*self.cfg.goal_ee.ranges.pos_l, (len(env_ids), 1), self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(*self.cfg.goal_ee.ranges.pos_p, (len(env_ids), 1), self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(*self.cfg.goal_ee.ranges.pos_y, (len(env_ids), 1), self.device).squeeze(1)

    def collision_check(self, env_ids):
        """Return True for EE target paths that should be rejected."""
        if len(env_ids) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        # Check the whole linear spherical interpolation, not just the final
        # target, because intermediate EE targets can clip through the torso.
        ee_target_all_sphere = torch.lerp(
            self.ee_start_sphere[env_ids, :, None],
            self.ee_goal_sphere[env_ids, :, None],
            self.collision_check_t,
        ).squeeze(-1)
        ee_target_cart = sphere2cart(
            torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)
        ).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(
            torch.logical_and(
                torch.all(ee_target_cart < self.collision_upper_limits, dim=-1),
                torch.all(ee_target_cart > self.collision_lower_limits, dim=-1),
            ),
            dim=0,
        )
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    def update_curr_ee_goal(self):
        """Advance the EE trajectory and publish UniFP command slots."""
        self.goal_timer += 1
        done = self.goal_timer > self.traj_total_timesteps
        if torch.any(done):
            self._resample_ee_goal(done.nonzero(as_tuple=False).flatten())
        ratio = (self.goal_timer / self.traj_timesteps).clamp(0.0, 1.0).unsqueeze(1)
        self.curr_ee_goal_sphere = self.ee_start_sphere + ratio * (self.ee_goal_sphere - self.ee_start_sphere)
        self.curr_ee_goal_cart = sphere2cart(self.curr_ee_goal_sphere)
        self._refresh_curr_ee_goal_world()

        # Command vector layout used by the actor:
        # 0:3 base velocity, 3:6 EE spherical target, 6:9 reserved orientation,
        # 9:12 EE force command, 12:15 base force command.
        self.commands[:, 3:6] = self.curr_ee_goal_sphere
        self.commands[:, 9:12] = self.current_Fxyz_gripper_cmd
        self.commands[:, 12:15] = self.current_Fxyz_base_cmd

    def _refresh_curr_ee_goal_world(self, env_ids=None):
        """Refresh cached world-frame EE target positions from spherical commands."""
        if env_ids is None:
            env_ids = self.all_env_ids
        if len(env_ids) == 0:
            return
        base_yaw_quat = self._get_base_yaw_quat(env_ids)
        # UniFP spherical commands are yaw-aligned around the arm workspace
        # center, so only base yaw rotates the local target into world space.
        self.curr_ee_goal_cart_world[env_ids] = self.get_ee_goal_spherical_center(base_yaw_quat, env_ids) + quat_apply(
            base_yaw_quat,
            self.curr_ee_goal_cart[env_ids],
        )

    def get_ee_goal_spherical_center(self, base_yaw_quat, env_ids=None):
        """Return the Genesis-correct arm workspace center for EE spherical goals."""
        center = None
        if env_ids is None:
            center = torch.cat([self.simulator.base_pos[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
            rotated_offset = quat_apply(base_yaw_quat, self.ee_goal_center_offset)
            center[:, :2] = center[:,:2] + rotated_offset[:,:2]
            terrain_z = torch.mean(self.simulator.measured_heights,dim=1)
            center[:, 2] = terrain_z + self.cfg.goal_ee.sphere_center.z_invariant_offset
            return center
        
        center = torch.cat([self.simulator.base_pos[env_ids, :2], torch.zeros(len(env_ids), 1, device=self.device)], dim=1)
        rotated_offset = quat_apply(base_yaw_quat, self.ee_goal_center_offset[env_ids])
        center[:, :2] = center[:,:2] + rotated_offset[:,:2]
        terrain_z = torch.mean(self.simulator.measured_heights[env_ids],dim=1)
        center[:, 2] = terrain_z + self.cfg.goal_ee.sphere_center.z_invariant_offset
        
        return center

    def get_body_orientation(self):
        return self.simulator.base_euler[:, :2]

    def _normalize_quat(self, quat):
        """Normalize xyzw quaternions defensively before orientation rewards."""
        return quat / torch.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-6)

    def _quat_conjugate(self, quat):
        """Return the conjugate of an xyzw quaternion batch."""
        quat_conj = quat.clone()
        quat_conj[:, :3] *= -1.0
        return quat_conj

    def _capture_default_ee_local_quat(self):
        """Capture canonical EE orientation relative to the UniFP base-yaw frame."""
        base_yaw_quat = self._get_base_yaw_quat()
        ee_quat = self._normalize_quat(self.simulator.ee_quat)
        return self._normalize_quat(quat_mul(self._quat_conjugate(base_yaw_quat), ee_quat))

    def get_walking_cmd_mask(self, env_ids=None, return_all=False):
        """Return envs whose base command is large enough to count as walking."""
        if env_ids is None:
            env_ids = self.all_env_ids
        walking_mask0 = torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip
        walking_mask1 = torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_y_clip
        walking_mask2 = torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2
        if return_all:
            return walking_mask0, walking_mask1, walking_mask2, walking_mask
        return walking_mask

    def _step_contact_targets(self):
        """Advance the UniFP gait clock and pin it to phase zero while standing."""
        cycle_time = self.cfg.rewards.cycle_time
        standing_mask = ~self.get_walking_cmd_mask()
        self.gait_indices = torch.remainder(self.gait_indices + self.dt / cycle_time, 1.0)
        self.gait_indices[standing_mask] = 0.0

    def _get_phase(self):
        """Return the persistent UniFP gait phase in [0, 1)."""
        return self.gait_indices

    def _get_gait_phase(self):
        """Return the original UniFP diagonal-trot stance mask.

        The mask is float-valued with 1 for stance and 0 for swing. The
        `target_joint_pos_thd` offset creates a double-support window around
        phase transitions, matching the Isaac Gym UniFP implementation.
        """
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone() + self.cfg.rewards.target_joint_pos_thd
        sin_pos_r = sin_pos.clone() - self.cfg.rewards.target_joint_pos_thd

        stance_mask = torch.zeros((self.num_envs, 4), device=self.device)
        # Feet are ordered [FR, FL, RR, RL] in this Genesis port. Original UniFP
        # pairs FL/RR and FR/RL as diagonal stance groups.
        stance_mask[:, 1] = sin_pos_l >= 0
        stance_mask[:, 2] = sin_pos_l >= 0
        stance_mask[:, 0] = sin_pos_r < 0
        stance_mask[:, 3] = sin_pos_r < 0
        return stance_mask

    def compute_ref_state(self):
        """Compute the original UniFP gait-phase reference leg pose."""
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone() + self.cfg.rewards.target_joint_pos_thd
        sin_pos_r = sin_pos.clone() - self.cfg.rewards.target_joint_pos_thd

        self.ref_dof_pos = self.simulator.default_dof_pos[:, :12].repeat(self.num_envs, 1).clone()
        scale_1 = self.cfg.rewards.target_joint_pos_scale / (1 - self.cfg.rewards.target_joint_pos_thd)
        scale_2 = scale_1 * 2
        idx = self.leg_dof_indices

        # Original UniFP zeros the stance half of each diagonal before applying
        # the sinusoidal swing offsets to thigh/calf reference joints.
        sin_pos_l[sin_pos_l > 0] = sin_pos_l[sin_pos_l > 0] * (1 - self.cfg.rewards.target_joint_pos_thd) / (1 + self.cfg.rewards.target_joint_pos_thd) * 0.0
        self.ref_dof_pos[:, idx["FL_thigh_joint"]] -= sin_pos_l * scale_1
        self.ref_dof_pos[:, idx["FL_calf_joint"]] += sin_pos_l * scale_2
        self.ref_dof_pos[:, idx["RR_thigh_joint"]] -= sin_pos_l * scale_1
        self.ref_dof_pos[:, idx["RR_calf_joint"]] += sin_pos_l * scale_2

        sin_pos_r[sin_pos_r < 0] = sin_pos_r[sin_pos_r < 0] * (1 - self.cfg.rewards.target_joint_pos_thd) / (1 + self.cfg.rewards.target_joint_pos_thd) * 0.0
        self.ref_dof_pos[:, idx["FR_thigh_joint"]] += sin_pos_r * scale_1
        self.ref_dof_pos[:, idx["FR_calf_joint"]] -= sin_pos_r * scale_2
        self.ref_dof_pos[:, idx["RL_thigh_joint"]] += sin_pos_r * scale_1
        self.ref_dof_pos[:, idx["RL_calf_joint"]] -= sin_pos_r * scale_2

    def _randomize_force_gains(self, env_ids):
        """Randomize virtual impedance gains used to turn force into offsets."""
        if len(env_ids) == 0:
            return
        if self.cfg.commands.randomize_gripper_force_gains:
            self.gripper_force_kps[env_ids] = torch_rand_float(
                *self.cfg.commands.gripper_force_kp_range,
                (len(env_ids), 1),
                self.device,
            )
            if self.cfg.commands.gripper_prop_kd > 0:
                self.gripper_force_kds[env_ids] = self.gripper_force_kps[env_ids] * self.cfg.commands.gripper_prop_kd
            else:
                self.gripper_force_kds[env_ids] = torch_rand_float(
                    *self.cfg.commands.gripper_force_kd_range,
                    (len(env_ids), 1),
                    self.device,
                )
        if self.cfg.commands.randomize_base_force_gains:
            self.base_force_kps[env_ids] = torch_rand_float(
                *self.cfg.commands.base_force_kp_range,
                (len(env_ids), 1),
                self.device,
            )
            self.base_force_kds[env_ids] = torch_rand_float(
                *self.cfg.commands.base_force_kd_range,
                (len(env_ids), 1),
                self.device,
            )

    def _reset_force_events(self, env_ids):
        """Clear and resample per-env timers for all four UniFP force streams."""
        if len(env_ids) == 0:
            return
        self.freed_envs_gripper_cmd[env_ids] = False
        self.freed_envs_gripper_ext[env_ids] = False
        self.selected_env_ids_gripper_cmd[env_ids] = False
        self.selected_env_ids_gripper_ext[env_ids] = False
        self.force_target_gripper_cmd[env_ids] = 0.0
        self.force_target_gripper_ext[env_ids] = 0.0
        self.push_end_time_gripper_cmd[env_ids] = 0.0
        self.push_end_time_gripper_ext[env_ids] = 0.0
        self.push_duration_gripper_cmd[env_ids] = 0.0
        self.push_duration_gripper_ext[env_ids] = 0.0
        self.push_interval_gripper_cmd[env_ids] = self._rand_force_interval(
            self.push_interval_gripper_cmd_min,
            self.push_interval_gripper_cmd_max,
            (len(env_ids),),
        )
        self.push_interval_gripper_ext[env_ids] = self._rand_force_interval(
            self.push_interval_gripper_ext_min,
            self.push_interval_gripper_ext_max,
            (len(env_ids),),
        )

        self.freed_envs_base_cmd[env_ids] = False
        self.freed_envs_base_ext[env_ids] = False
        self.selected_env_ids_base_cmd[env_ids] = False
        self.selected_env_ids_base_ext[env_ids] = False
        self.force_target_base_cmd[env_ids] = 0.0
        self.force_target_base_ext[env_ids] = 0.0
        self.push_end_time_base_cmd[env_ids] = 0.0
        self.push_end_time_base_ext[env_ids] = 0.0
        self.push_duration_base_cmd[env_ids] = 0.0
        self.push_duration_base_ext[env_ids] = 0.0
        self.push_interval_base_cmd[env_ids] = self._rand_force_interval(
            self.push_interval_base_cmd_min,
            self.push_interval_base_cmd_max,
            (len(env_ids),),
        )
        self.push_interval_base_ext[env_ids] = self._rand_force_interval(
            self.push_interval_base_ext_min,
            self.push_interval_base_ext_max,
            (len(env_ids),),
        )

    def _rand_force_interval(self, min_steps, max_steps, shape):
        low = int(min_steps)
        high = max(int(max_steps), low + 1)
        return torch.randint(low, high, shape, device=self.device)

    def _sample_force_target(self, env_ids, force_range, *, zero_z=False, z_scale=1.0):
        """Sample a force target for command or external-force streams."""
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
        """Update one triangular UniFP force profile.

        A stream owns its own interval, duration, selected/free masks, target,
        and output tensor. The profile ramps 0 -> target, holds through a
        settling interval, then ramps target -> 0. The same helper is used for:
        EE commanded force, EE external force, base commanded force, and base
        external force.
        """
        new_env_ids = env_ids_all[(self.episode_length_buf[env_ids_all] % interval[env_ids_all]) == 0]
        if len(new_env_ids) > 0:
            # `forced_prob` controls whether this env receives the sampled
            # profile or is marked free/zero for this interval.
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

            # Ramp up until `end_time`.
            before_end = self.episode_length_buf[selected_env_ids] < end_time[selected_env_ids].to(torch.int32)
            step1_env_ids = selected_env_ids[before_end]
            if len(step1_env_ids) > 0:
                dur = duration[step1_env_ids].unsqueeze(-1)
                elapsed = self.episode_length_buf[step1_env_ids].unsqueeze(-1) - (end_time[step1_env_ids].unsqueeze(-1) - dur)
                output[step1_env_ids] = (target[step1_env_ids] / dur) * torch.clamp(elapsed, torch.zeros_like(dur), dur)

            # Ramp down after the configured settling plateau.
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

            # Once the full profile is over, clear the stream and sample a new
            # interval before the next force event.
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
            # Free envs are explicitly zeroed so old targets cannot leak across
            # intervals or resets.
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
        """Update EE commanded-force and external-force streams."""
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
        if self.cfg.commands.apply_ee_external_forces:
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
        else:
            self.ee_force_ext_world[env_ids_all] = 0.0
            self.selected_env_ids_gripper_ext[env_ids_all] = False
            self.force_target_gripper_ext[env_ids_all] = 0.0
        if hasattr(self.simulator, "apply_ee_force"):
            self.simulator.apply_ee_force(self.ee_force_ext_world)

    def _push_robot_base(self, env_ids_all):
        """Update base commanded-force and external-force streams."""
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
        if self.cfg.commands.apply_base_external_forces:
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
        else:
            self.base_force_ext_world[env_ids_all] = 0.0
            self.selected_env_ids_base_ext[env_ids_all] = False
            self.force_target_base_ext[env_ids_all] = 0.0
        if hasattr(self.simulator, "apply_base_force"):
            self.simulator.apply_base_force(self.base_force_ext_world)

    def _reset_dofs(self, env_ids):
        """Reset B1/Z1 DOFs with original UniFP perturbation structure."""
        dof_pos = self.simulator.default_dof_pos.repeat(len(env_ids), 1)
        leg_low, leg_high = self.cfg.init_state.leg_dof_pos_perturb_range
        arm_low, arm_high = self.cfg.init_state.arm_dof_pos_perturb_range
        # UniFP perturbs legs multiplicatively around their default crouch pose.
        dof_pos[:, :12] = self.simulator.default_dof_pos[:, :12].repeat(len(env_ids), 1) + torch_rand_float(
            leg_low,
            leg_high,
            (len(env_ids), 12),
            self.device,
        )
        # Arm joints are perturbed additively; gripper joints remain at default.
        dof_pos[:, 12:17] += torch_rand_float(
            arm_low,
            arm_high,
            (len(env_ids), self.num_actions - 12),
            self.device,
        )
        dof_vel = torch.zeros_like(dof_pos)
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)

    def _reset_root_states(self, env_ids):
        """Reset base pose at the terrain origin and sample initial velocities."""
        base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
        base_pos += self.simulator._env_origins[env_ids]
        base_pos[:, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), self.device)
        # Original UniFP randomizes initial yaw while keeping roll/pitch at zero.
        rand_yaw = self.cfg.init_state.rand_yaw_range * torch_rand_float(
            -1.0,
            1.0,
            (len(env_ids), 1),
            self.device,
        ).squeeze(1)
        base_quat = quat_from_euler_xyz(
            torch.zeros_like(rand_yaw),
            torch.zeros_like(rand_yaw),
            rand_yaw,
        )
        base_lin_vel = torch_rand_float(-0.10, 0.10, (len(env_ids), 3), self.device)
        base_ang_vel = torch_rand_float(-0.1, 0.1, (len(env_ids), 3), self.device)
        self.simulator.reset_root_states(env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel)

    def _update_terrain_curriculum(self, env_ids):
        """PACT/legged-gym terrain curriculum retained for Genesis training."""
        if not self.init_done:
            return
        distance = torch.norm(self.simulator.base_pos[env_ids, :2] - self.simulator._env_origins[env_ids, :2], dim=1)
        move_up = distance > self.simulator._terrain.env_length / 2
        move_down = distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        self.simulator.update_terrain_curriculum(env_ids, move_up, move_down & ~move_up)

    def _update_command_curriculum(self, env_ids):
        """Expand base velocity command ranges when tracking is good enough."""
        if "tracking_lin_vel_force_world" not in self.episode_sums:
            return
        mean_tracking = torch.mean(self.episode_sums["tracking_lin_vel_force_world"][env_ids]) / self.max_episode_length
        if mean_tracking > self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel_force_world"]:
            for key in ["lin_vel_x", "lin_vel_y"]:
                self.command_ranges[key][0] = np.clip(self.command_ranges[key][0] - 0.2, -self.cfg.commands.max_curriculum, 0.0)
                self.command_ranges[key][1] = np.clip(self.command_ranges[key][1] + 0.2, 0.0, self.cfg.commands.max_curriculum)

    def step_reward_curriculum(self, num_iters):
        """Cosine-ramp selected reward scales during training."""
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

        self.height_noise_vec = torch.zeros(self.simulator._num_height_points, device=self.device)

        if self.cfg.terrain.measure_heights:
            self.height_noise_vec[:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        return noise_vec

    def _init_buffers(self):
        self.common_step_counter = 0
        self.extras = {}
        self.all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.forward_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.forward_vec[:, 0] = 1.0
        self.fail_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_fail_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Counts consecutive control steps with termination contact.
        # Clearing this counter when contact disappears implements patience.
        self.termination_contact_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, device=self.device)
        self.heading_commands = torch.zeros(self.num_envs, device=self.device)
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

        # Boolean previous-step foot-contact latch. Keeping this bool avoids
        # ambiguous reward dtype changes when contact rewards reuse the buffer.
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.simulator.feet_indices),
            dtype=torch.bool,
            device=self.device,
        )
        self.feet_air_time = torch.zeros(self.num_envs, len(self.simulator.feet_indices), device=self.device)
        self.feet_stance_time = torch.zeros_like(self.feet_air_time)
        self.valid_swing = torch.zeros_like(self.last_contacts)

        # Prevent multiple reward functions from updating the statistics in one step.
        self._feet_stats_update_step = -1
        self._feet_stats = {}

        self.progress_vel_ema = torch.zeros(self.num_envs, 2, device=self.device)

        self.step_liftoff_pos = torch.zeros(
            self.num_envs, 4, 2, device=self.device
        )
        self.step_direction_world = torch.zeros_like(
            self.step_liftoff_pos
        )
        self.step_max_progress = torch.zeros(
            self.num_envs, 4, device=self.device
        )


        # Original UniFP uses a persistent phase variable instead of deriving
        # phase directly from episode time. The phase only advances while a
        # walking command is active and is reset to zero for standing commands.
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.obs_history_slots = [
            torch.zeros(self.num_envs, self.cfg.env.num_observations, device=self.device)
            for _ in range(self.cfg.env.num_obs_hist)
        ]
        self._obs_history_slot = self.cfg.env.num_obs_hist - 1
        self.obs_history = torch.zeros(self.num_envs, self.cfg.env.num_observations * self.cfg.env.num_obs_hist, device=self.device)
        self.last_obs_buf = torch.zeros_like(self.obs_buf)
        self.llast_obs_buf = torch.zeros_like(self.obs_buf)
        self.last_obs_hist = torch.zeros_like(self.obs_history)
        self.llast_obs_hist = torch.zeros_like(self.obs_history)

        self.critic_obs_slots = [
            torch.zeros(self.num_envs, self.cfg.env.num_privileged_obs, device=self.device)
            for _ in range(self.cfg.env.num_priv_stack)
        ]
        self._critic_obs_slot = self.cfg.env.num_priv_stack - 1
        self._critic_obs_buf = torch.zeros(self.num_envs, self.cfg.env.num_privileged_obs, device=self.device)
        self._mass_params_buf = torch.zeros(self.num_envs, 22, device=self.device)
        self.privileged_obs_buf = torch.zeros(
            self.num_envs,
            self.cfg.env.num_privileged_obs * self.cfg.env.num_priv_stack,
            device=self.device,
        )
        self.explicit_labels_buf = torch.zeros(self.num_envs, self.cfg.env.num_explicit_recon_obs, device=self.device)

        self.leg_dof_indices = {
            name: self.cfg.asset.dof_names.index(name)
            for name in (
                "FL_thigh_joint",
                "FL_calf_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
            )
        }
        self.ref_dof_pos = self.simulator.default_dof_pos[:, :12].repeat(self.num_envs, 1)
        self.ee_goal_center_offset = torch.tensor(
            [
                self.cfg.goal_ee.sphere_center.x_offset,
                self.cfg.goal_ee.sphere_center.y_offset,
                self.cfg.goal_ee.sphere_center.z_invariant_offset,
            ],
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.init_start_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_start, device=self.device).unsqueeze(0)
        self.init_end_ee_sphere = torch.tensor(self.cfg.goal_ee.ranges.init_pos_end, device=self.device).unsqueeze(0)
        self.ee_goal_sphere = self.init_end_ee_sphere.repeat(self.num_envs, 1)
        self.curr_ee_goal_sphere = self.ee_goal_sphere.clone()
        self.ee_start_sphere = self.ee_goal_sphere.clone()
        self.curr_ee_goal_cart = sphere2cart(self.curr_ee_goal_sphere)
        self.curr_ee_goal_cart_world = torch.zeros_like(self.curr_ee_goal_cart)
        self._base_yaw_quat_buf = torch.zeros(self.num_envs, 4, device=self.device)
        self._base_yaw_quat_buf[:, 3] = 1.0
        self._base_yaw_quat_subset_buf = torch.zeros_like(self._base_yaw_quat_buf)
        self._base_yaw_quat_subset_buf[:, 3] = 1.0
        # Default EE orientation is captured once from the canonical initialized
        # robot and stored in the base-yaw frame used by UniFP EE commands.
        self.default_ee_local_quat = self._capture_default_ee_local_quat()
        self.ee_pos_sphe_arm = torch.zeros_like(self.curr_ee_goal_cart)
        self.collision_lower_limits = torch.tensor(
            self.cfg.goal_ee.collision_lower_limits,
            device=self.device,
            dtype=torch.float,
        )
        self.collision_upper_limits = torch.tensor(
            self.cfg.goal_ee.collision_upper_limits,
            device=self.device,
            dtype=torch.float,
        )
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(
            0.0,
            1.0,
            self.num_collision_check_samples,
            device=self.device,
        )[None, None, :]
        self.traj_timesteps = torch.ones(self.num_envs, device=self.device) / self.dt
        self.traj_total_timesteps = self.traj_timesteps.clone()
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)

        self.ee_force_ext_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_force_ext_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.current_Fxyz_gripper_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.current_Fxyz_base_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.estimated_ee_force_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.estimated_base_force_local = torch.zeros(self.num_envs, 3, device=self.device)
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
        self._randomize_force_gains(self.all_env_ids)

        self.z1_arm_jacobian = torch.zeros(
            self.num_envs,
            3,
            6,
            device=self.device,
            dtype=torch.float,
        )

        self.leg_jacobians = torch.zeros((self.num_envs, 4, 3, 3), dtype=torch.float, device=self.device)
        self._abad_link_length = torch.tensor(self.cfg.asset.abad_link_length, device=self.device, dtype=torch.float)
        self._hip_link_length = torch.tensor(self.cfg.asset.hip_link_length, device=self.device, dtype=torch.float)
        self._knee_link_length = torch.tensor(self.cfg.asset.knee_link_length, device=self.device, dtype=torch.float)
        self._leg_side_sign = torch.tensor(self.cfg.asset.side_signs, device=self.device, dtype=torch.float).view(1, 4)

        self.prev_ee_error = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )

        # Taken from b1z1 URDF, hardcoded for now...
        self.z1_link00_offset = torch.tensor(
            [0.3000, 0.0000, 0.0900],
            device=self.device,
            dtype=torch.float,
        )
        self.z1_joint_offsets = torch.tensor(
            [
                [0.0000, 0.0000, 0.0585],
                [0.0000, 0.0000, 0.0450],
                [-0.3500, 0.0000, 0.0000],
                [0.2180, 0.0000, 0.0570],
                [0.0700, 0.0000, 0.0000],
                [0.0492, 0.0000, 0.0000],
            ],
            device=self.device,
            dtype=torch.float,
        )
        self.z1_joint_axes = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            device=self.device,
            dtype=torch.float,
        )
        self.z1_ee_offset = torch.tensor(
            [0.0510 + 0.1350, 0.0000, 0.0000],
            device=self.device,
            dtype=torch.float,
        )

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
        cfg.runner_steps_per_iter = cfg.env.num_steps_per_env
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
        base_yaw_quat = self._get_base_yaw_quat()
        force_offset = quat_rotate_inverse(base_yaw_quat, self.base_force_ext_world) + self.current_Fxyz_base_cmd
        force_offset = force_offset[:, :2] / self.base_force_kds
        base_lin_vel_offset = self.commands[:, :2] + force_offset
        
        non_stop_sign = (torch.abs(base_lin_vel_offset[:, 0]) > self.cfg.commands.lin_vel_x_clip) | (torch.abs(base_lin_vel_offset[:, 1]) > self.cfg.commands.lin_vel_y_clip) | (torch.abs(self.commands[:, 2]) > self.cfg.commands.ang_vel_yaw_clip)
        base_lin_vel_offset[:, :3] *= non_stop_sign.unsqueeze(1)

        error = torch.sum(torch.square(base_lin_vel_offset - self.simulator.base_lin_vel[:, :2]), dim=1)

        return torch.exp(-error / self.cfg.rewards.tracking_sigma)

        # # Remove the reward that comes from simply standing still IF the robot should be moving
        # #     help remove the incentive to stand-still to accumulate positive reward
        # target_mag = torch.norm(base_lin_vel_offset, dim=1)

        # tracking = torch.exp(-error / self.cfg.rewards.tracking_sigma)
        # standing_baseline = torch.exp(
        #     -torch.sum(torch.square(base_lin_vel_offset), dim=1) / self.cfg.rewards.tracking_sigma
        # )

        # moving = target_mag > 0.1

        # max_improvement = (1.0 - standing_baseline).clamp_min(1e-4)

        # moving_reward = (
        #     tracking - standing_baseline
        # ) / max_improvement

        # moving_reward = torch.clamp(moving_reward, -1.0, 1.0)

        # reward = torch.where(
        #     moving,
        #     moving_reward,
        #     tracking,
        # )

        # return reward.clamp_(min=0.0)


    def _reward_tracking_ang_vel(self):
        return torch.exp(-torch.square(self.commands[:, 2] - self.simulator.base_ang_vel[:, 2]) / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ee_force_world(self):
        base_yaw_quat = self._get_base_yaw_quat()
        force_offset = (self.ee_force_ext_world + quat_apply(base_yaw_quat, self.current_Fxyz_gripper_cmd)) / self.gripper_force_kps
        target = self.curr_ee_goal_cart_world + force_offset
        error = torch.sum(torch.abs(target - self.simulator.ee_pos), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_ee_sigma)

    def _reward_upright(self):
        tilt_sq = torch.sum(
            torch.square(self.simulator.projected_gravity[:, :2]),
            dim=1,
        )
        return torch.exp(-tilt_sq / 0.10)


    def _reward_tracking_ee_orientation_default(self):
        """Reward keeping the EE frame at its default yaw-aligned orientation."""
        base_yaw_quat = self._get_base_yaw_quat()
        # The reference is the default EE frame expressed in the base-yaw frame.
        # Rotating it by current yaw avoids punishing normal commanded turning.
        target_ee_quat = self._normalize_quat(quat_mul(base_yaw_quat, self.default_ee_local_quat))
        ee_quat = self._normalize_quat(self.simulator.ee_quat)
        quat_dot = torch.sum(ee_quat * target_ee_quat, dim=1).abs().clamp(0.0, 1.0)
        orientation_error = 1.0 - torch.square(quat_dot)
        return torch.exp(-orientation_error / self.cfg.rewards.tracking_ee_orientation_sigma)

    def _reward_termination(self):
        return (self.reset_buf.bool() & ~self.time_out_buf.bool()).float()

    def _reward_alive(self):
        return torch.ones(self.num_envs, device=self.device)

    def _reward_lin_vel_z(self):
        return torch.square(self.simulator.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.simulator.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.simulator.projected_gravity[:, :2]), dim=1)

    def _reward_roll(self):
        return torch.square(self.simulator.base_euler[:, 0])

    def _reward_pitch(self):
        return torch.square(self.simulator.base_euler[:, 1])

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

    def _reward_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(
            self.actions[:,:12] - 2*self.last_actions[:,:12] + self.llast_actions[:,:12]), dim=-1)
        return action_smoothness_cost

    def _reward_action_rate_arm(self):
        return torch.sum(torch.square(self.last_actions[:, 12:17] - self.actions[:, 12:17]), dim=1)
    
    def _reward_action_smoothness_arm(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(
            self.actions[:,12:17] - 2*self.last_actions[:,12:17] + self.llast_actions[:,12:17]), dim=-1)
        return action_smoothness_cost

    def _reward_joint_power(self):
        # penalize large amounts of motor power
        return torch.sum(torch.abs(self.simulator.dof_vel[:,:12] * self.simulator.torques[:,:12]), dim=1)

    def _reward_joint_power_dist(self):
        # Penalize uneven distributions of motor power
        return torch.var(self.simulator.torques[:,:12]*self.simulator.dof_vel[:,:12], dim=1)
    
    def _reward_joint_power_arm(self):
        # penalize large amounts of motor power
        return torch.sum(torch.abs(self.simulator.dof_vel[:,12:17] * self.simulator.torques[:,12:17]), dim=1)

    def _reward_joint_power_dist_arm(self):
        # Penalize uneven distributions of motor power
        return torch.var(self.simulator.torques[:,12:17]*self.simulator.dof_vel[:,12:17], dim=1)

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
        return torch.sum((torch.abs(self.simulator.unclipped_torques[:, :17]) - limits[:17] * self.cfg.rewards.soft_torque_limit).clip(min=0.0), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.simulator.dof_pos[:, [0, 3, 6, 9]]), dim=1)

    def _reward_feet_contact_forces(self):
        return torch.sum((torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.0), dim=1)

    def _update_feet_air_time_stats(self):
        """
        Update and cache foot contact, stance-time, and air-time statistics.

        This function may be called by multiple rewards, but updates the stateful
        buffers at most once per environment step.
        """
        current_step = int(self.common_step_counter)

        if self._feet_stats_update_step == current_step:
            return self._feet_stats

        contact = (
            self.simulator.link_contact_forces[
                :, self.simulator.feet_indices, 2
            ] > 1.0
        )

        previous_contact = self.last_contacts.clone()

        # Two-sample contact filtering suppresses single-frame contact losses.
        contact_filt = torch.logical_or(contact, previous_contact)

        previous_air_time = self.feet_air_time.clone()
        previous_stance_time = self.feet_stance_time.clone()

        # Raw contact-to-noncontact transition.
        liftoff = previous_contact & (~contact)

        # Only accept liftoff after a meaningful stance interval. This reduces
        # reward exploitation through rapid contact chatter.
        valid_liftoff = (
            liftoff
            & (previous_stance_time >= 0.08)
        )

        # Preserve valid-swing status until the foot contacts again.
        self.valid_swing &= ~contact
        self.valid_swing |= valid_liftoff

        # Advance air time before detecting touchdown, matching the original
        # air-time reward convention.
        self.feet_air_time += self.dt

        first_contact = (
            (previous_air_time > 0.0)
            & contact_filt
        )

        # Save duration before resetting feet that have contacted.
        touchdown_air_time = self.feet_air_time.clone()

        # Air time is retained only while filtered contact is absent.
        self.feet_air_time *= (~contact_filt).float()

        # Update stance duration.
        self.feet_stance_time += self.dt
        self.feet_stance_time *= contact_filt.float()

        # Store the newest raw contact state without replacing the buffer.
        self.last_contacts.copy_(contact)

        self._feet_stats = {
            "contact": contact,
            "contact_filt": contact_filt,
            "first_contact": first_contact,
            "liftoff": liftoff,
            "valid_liftoff": valid_liftoff,
            "valid_swing": self.valid_swing.clone(),
            "air_time": self.feet_air_time.clone(),
            "touchdown_air_time": touchdown_air_time,
            "stance_time": self.feet_stance_time.clone(),
        }
        self._feet_stats_update_step = current_step

        return self._feet_stats

    # def _reward_feet_air_time(self):
    #     contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.0
    #     contact_filt = torch.logical_or(contact, self.last_contacts)
    #     # Preserve the buffer object and dtype so other rewards cannot inherit
    #     # accidental bool/float conversions through aliasing.
    #     self.last_contacts.copy_(contact)
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.12) * first_contact, dim=1)  # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1  # no reward for zero command
    #     self.feet_air_time *= (~contact_filt).float()
    #     return rew_airTime


    def _reward_sparse_contacts(self):
        fz = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2]
        contact_prob = torch.sigmoid(10.0*(fz - 10.0))
        num_contacts = torch.sum(contact_prob, dim=-1)

        moving = self.get_walking_cmd_mask()
        
        return torch.exp(-torch.square(num_contacts - 2.0)) * moving.float()

    def _reward_feet_air_time(self):
        """
        Reward sufficiently long swing periods when the foot touches down.
        """
        stats = self._update_feet_air_time_stats()

        first_contact = stats["first_contact"]
        touchdown_air_time = stats["touchdown_air_time"]

        error = (touchdown_air_time - 0.30)
        error[:,0:2] *= 2                 # give twice the weight to the front feet

        reward = torch.sum(error * first_contact.float(), dim=1)

        reward *= self.get_walking_cmd_mask().float()

        return reward

    def _reward_early_swing(self):
        """
        Reward upward foot motion during the beginning of a valid swing.

        A swing is valid only when initiated after at least 0.08 seconds of
        continuous stance. This helps suppress rapid tapping exploits.
        """
        stats = self._update_feet_air_time_stats()

        contact = stats["contact"]
        contact_filt = stats["contact_filt"]
        air_time = stats["air_time"]
        valid_swing = stats["valid_swing"]

        # Apply guidance during the first 100 ms of a valid swing.
        early_swing = (
            (~contact_filt)
            & valid_swing
            & (air_time > 0.0)
            & (air_time <= 0.18)
        )

        foot_vel_z = self.simulator.feet_vel[:, :, 2]

        upward_velocity_reward = torch.clamp(
            foot_vel_z / 0.40,
            min=0.0,
            max=1.0,
        )

        per_foot_reward = (
            early_swing.float()
            * upward_velocity_reward
        )

        # Prevent aerial hopping from earning swing-initiation reward.
        support_gate = contact.sum(dim=1) >= 2

        # Two swing feet constitute a full reward.
        reward = per_foot_reward.sum(dim=1) / 2.0
        reward *= support_gate.float()
        reward *= self.get_walking_cmd_mask().float()

        return reward

    def _reward_feet_height(self):
        feet_height = self.simulator.feet_pos[:, :, 2]
        errors = torch.relu(feet_height - 0.10)
        moving = self.get_walking_cmd_mask()
        return torch.mean(torch.square(errors), dim=1) * moving.float()

    def _reward_feet_height_high(self):
        return torch.mean((self.simulator.feet_pos[:, :, 2] - 0.18).clip(min=0.0), dim=1)

    def _reward_feet_drag(self):
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.0
        return torch.sum(torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1) * contact.float(), dim=1)

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

    def _reward_feet_pos_xy(self):
        # Original UniFP penalizes each foot drifting far from its matching
        # thigh in world XY, rather than measuring foot spread from the base.
        feet_pos_xy = self.simulator.feet_pos[:, :, :2]
        thigh_pos_xy = self.simulator.thigh_pos[:, :, :2]
        diff = torch.norm(feet_pos_xy - thigh_pos_xy, dim=2).view(self.num_envs, -1)
        return torch.mean(diff, dim=1)

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
        
        # overreach = torch.relu(front_x - self.cfg.rewards.overreach_x_max)

        # Nominal rear-foot x location in base frame.
        # This should usually be negative, e.g. -0.20 to -0.25 m.
        front_x_nominal = self.cfg.rewards.front_foot_x_nominal

        # Allowed deviation around nominal rear-foot x location.
        # Example: 0.08 m allows rear_x in [nominal - 0.08, nominal + 0.08].
        front_x_margin = self.cfg.rewards.foot_x_margin

        # Penalize both too far forward and too far backward relative to nominal.
        x_error = torch.abs(front_x - front_x_nominal)
        overreach = torch.relu(x_error - front_x_margin)

        # stance/contact gating
        contact = (
            self.simulator.link_contact_forces[:, self.simulator.feet_indices[:2], 2] > 1.0
        ).float()

        penalty = torch.sum(contact * overreach ** 2, dim=1)

        return penalty

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
        rear_x_nominal = -self.cfg.rewards.rear_foot_x_nominal

        # Allowed deviation around nominal rear-foot x location.
        # Example: 0.08 m allows rear_x in [nominal - 0.08, nominal + 0.08].
        rear_x_margin = self.cfg.rewards.foot_x_margin

        # Penalize both too far forward and too far backward relative to nominal.
        x_error = torch.abs(rear_x - rear_x_nominal)
        overreach = torch.relu(x_error - rear_x_margin)

        # Contact gate rear feet only: feet_indices[2:4], not [:2]
        contact = (
            self.simulator.link_contact_forces[:, self.simulator.feet_indices[2:4], 2] > 5.0
        ).float()  # (N,2)

        penalty = torch.sum(contact * overreach ** 2, dim=1)

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

        contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 5.0
        swing = ~contact

        # desired_swing = 1.0 - self._get_gait_phase()
        num_swing = swing.sum(dim=1)

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
        excess_margin = 0.10  # [m], tune: 0.03 - 0.06
        excess = torch.relu(feet_z - (z_des + excess_margin))           # (N,4)
        # excess = F.softplus(feet_z - (z_des + excess_margin))           # (N,4)
        excess_err = torch.square(excess)

        # Weight excess penalty less than main tracking term
        excess_weight = 0.25  # tune: 0.1 - 0.5

        # total_err = torch.sum(
            # foot_vel_xy_norm * (track_err + excess_weight * excess_err) * swing.float(),
            # dim=-1
        # )                                                               # (N,)

        # return torch.exp(-total_err / self.cfg.rewards.foot_clearance_tracking_sigma) * moving.float()

        # rew = torch.exp(-total_err / self.cfg.rewards.foot_clearance_tracking_sigma)

        # rew *= (num_swing > 0).float()     # gate reward for robots with no swigning feet.

        height_quality = torch.exp(-(track_err + excess_weight * excess_err) / self.cfg.rewards.foot_clearance_tracking_sigma)

        motion_quality = torch.clamp(foot_vel_xy_norm / 0.30, 0.0, 1.0)

        per_foot = (
            swing.float()
            * height_quality
            * (0.25 + 0.75 * motion_quality)
        )

        reward = per_foot.sum(dim=1) / swing.sum(dim=1).clamp_min(1)
        reward *= self.get_walking_cmd_mask().float()

        return reward

    def _reward_feet_regulation(self):
        base_height = torch.mean(
            self.simulator.base_pos[:, 2].unsqueeze(1) - self.simulator.measured_heights,
            dim=1,
        )
        delta_feet = self.simulator.feet_pos - self.simulator.base_pos.unsqueeze(1)
        feet_to_base_height = (delta_feet * self.simulator.projected_gravity.unsqueeze(1)).sum(-1)
        feet_height = torch.clamp(base_height.unsqueeze(1) - feet_to_base_height, min=0.0)
        feet_xy_speed_sq = self.simulator.feet_vel[:, :, :2].pow(2).sum(-1)
        return (feet_xy_speed_sq * torch.exp(
            -feet_height / (0.025 * self.cfg.rewards.base_height_target)
        )).sum(-1)

    def _reward_feet_contact_number(self):
        """
        Reward foot contacts that match the UniFP gait-phase stance mask.

        This replaces the temporary "exactly two contacts" reward with the
        original UniFP phase-conditioned version.
        """
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_phase().bool()

        # Reward values must be floating point even though contacts and stance
        # masks are boolean phase/contact predicates.
        matched = contact == stance_mask
        reward = torch.where(
            matched,
            1.0,
            -1.0,
        )

        return torch.mean(reward, dim=1)

    def _reward_walking_ref_dof(self):
        """Reward tracking the gait-phase reference leg pose while walking."""
        self.compute_ref_state()
        joint_pos = self.simulator.dof_pos[:, :12].clone()
        pos_target = self.ref_dof_pos.clone()
        dof_error = torch.sum(torch.abs(joint_pos - pos_target), dim=1)
        rew = torch.exp(-dof_error * 0.2)
        rew[~self.get_walking_cmd_mask()] = 0.0
        return rew

    def _reward_walking_ref_swing_dof(self):
        """Reward only swing-leg joints tracking the gait-phase reference pose."""
        self.compute_ref_state()

        joint_pos = self.simulator.dof_pos[:, :12].clone()
        pos_target = self.ref_dof_pos.clone()

        stance_mask = self._get_gait_phase()
        stance_mask = torch.stack([stance_mask, stance_mask, stance_mask], 2).reshape(self.num_envs, 12)

        dof_error = torch.square(joint_pos - pos_target)
        dof_error[stance_mask == 1] = 0.0

        rew = torch.exp(-torch.sum(dof_error, dim=1) * 0.05)
        rew[~self.get_walking_cmd_mask()] = 0.0

        return rew

    def _reward_stand_still(self):
        moving = torch.norm(self.commands[:, :3], dim=1) > 0.1
        return torch.sum(torch.square(self.simulator.dof_pos[:, :12] - self.simulator.default_dof_pos[:, :12]), dim=1) * (~moving)

    def _reward_dof_close_to_default(self):
        # Penalize dof position deviation from default
        return torch.sum(torch.square(self.simulator.dof_pos[:,:12] - self.simulator.default_dof_pos[:,:12]), dim=1)

    def _reward_stand_still_contact(self):
        # Encourage feet contact with the ground at zero commands
        contacts = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.0
        full_contact = torch.sum(1.*contacts.float(), dim=1)==len(self.simulator.feet_indices)
        return 1.0*full_contact * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_ref_dof_leg(self):
        return torch.exp(-torch.sum(torch.abs(self.simulator.dof_pos[:, :12] - self.ref_dof_pos), dim=1) * 0.1)

    def _reward_torso_force_wrench_ellipsoid(self):
        """
        Force / wrench ellipsoid reward using:
            J_legs: (N, 4, 3, 3)

        """
        device = self.simulator.base_pos.device
        dtype = self.simulator.base_pos.dtype
        N = self.simulator.base_pos.shape[0]

        # --------------------------------------------------
        # state
        # --------------------------------------------------
        J_b = _sanitize_tensor(self.leg_jacobians)                                                # (N,4,3,3)
        tau_leg = _sanitize_tensor(self.simulator._dof_tau[:,0:12].view(-1, 4, 3))                        # (N,4,3)
        # Genesis stores static torque limits as a 1-D DOF vector, while some
        # simulator/domain-rand paths may provide per-env limits as (N, dof).
        torque_limits = self.simulator._torque_limits
        if torque_limits.ndim == 1:
            tau_max = torque_limits[:12].view(1, 4, 3).expand(N, 4, 3)
        else:
            tau_max = torque_limits[:, :12].view(-1, 4, 3)
        tau_max = torch.clamp(_sanitize_tensor(tau_max).abs(), min=1e-6)                  # (N,4,3) or broadcastable

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
        Lw = max(float(self.cfg.rewards.manip_rewards.ellipsoid_wrench_length_scale), 1e-6)

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
        r_force_size = 1.0 - torch.exp(-self.cfg.rewards.manip_rewards.ellipsoid_force_size_scale * force_size)
        r_force_size = torch.clamp(_sanitize_tensor(r_force_size), 0.0, 1.0)

        z_ratio = torch.clamp(_sanitize_tensor(lam1 / torch.clamp(0.5 * (lam2 + lam3), min=1e-8)), min=0.0, max=1e6)
        xy_ratio = torch.clamp(_sanitize_tensor(lam2 / torch.clamp(lam3, min=1e-8)), min=0.0, max=1e6)

        r_force_aniso = (
            _interval_reward(
                z_ratio,
                self.cfg.rewards.manip_rewards.ellipsoid_force_z_ratio_min,
                self.cfg.rewards.manip_rewards.ellipsoid_force_z_ratio_max,
                sharpness=2.0,
            )
            *
            _upper_reward(
                xy_ratio,
                self.cfg.rewards.manip_rewards.ellipsoid_force_xy_ratio_max,
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
        r_wrench_size = 1.0 - torch.exp(-self.cfg.rewards.manip_rewards.ellipsoid_wrench_size_scale * wrench_size)
        r_wrench_size = torch.clamp(_sanitize_tensor(r_wrench_size), 0.0, 1.0)

        lam_max = evals_W[:, 0]
        lam_min_active = torch.gather(evals_W, 1, (k_active - 1).long().unsqueeze(-1)).squeeze(-1)
        cond_W = torch.clamp(_sanitize_tensor(lam_max / torch.clamp(lam_min_active, min=1e-8)), min=0.0, max=1e6)

        r_wrench_cond = _upper_reward(
            cond_W,
            self.cfg.rewards.manip_rewards.ellipsoid_wrench_cond_max,
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

        # --------------------------------------------------
        # auxiliary: terrain awareness via friction cone
        # --------------------------------------------------
        fn = _sanitize_tensor(torch.sum(grf_w * normals_w, dim=-1))                                # (N,4)
        ft_vec = grf_w - fn.unsqueeze(-1) * normals_w
        ft = _sanitize_tensor(torch.linalg.norm(ft_vec, dim=-1))

        mu = self.cfg.rewards.manip_rewards.ellipsoid_mu_friction
        fn_margin = self.cfg.rewards.manip_rewards.ellipsoid_normal_force_margin
        ft_margin = self.cfg.rewards.manip_rewards.ellipsoid_tangential_force_margin

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
            self.cfg.rewards.manip_rewards.ellipsoid_force_aux_weight
            * (0.5 * r_force_use + 0.5 * r_force_principal_use)
            +
            self.cfg.rewards.manip_rewards.ellipsoid_wrench_aux_weight
            * r_wrench_use
            +
            self.cfg.rewards.manip_rewards.ellipsoid_friction_weight
            * r_contact_cone
        )

        reward = self.cfg.rewards.manip_rewards.ellipsoid_main_weight * r_main + (1.0 - self.cfg.rewards.manip_rewards.ellipsoid_main_weight) * r_aux
        reward = reward * (n_stance > 0).to(dtype)
        reward = torch.clamp(_sanitize_tensor(reward), 0.0, 1.0)

        return reward
    
    def _reward_arm_ee_force_manipulability(self):
        """
        Reward Z1 arm configurations that support large, isotropic EE force generation.

        Uses the translational EE Jacobian:

            J_arm: (N, 3, 6)

        and arm torque limits:

            tau_max_arm: (N, 6) or (6,)

        We construct a torque-weighted velocity manipulability matrix

            M = J diag(tau_max^2) J^T

        and use its inverse as the force ellipsoid shape matrix:

            S_F = M^{-1}

        Large eigenvalues of S_F indicate large force capability.
        Isotropy is encouraged by keeping the condition number small.

        Returns:
            reward: (N,), bounded in [0, 1]
        """
        device = self.z1_arm_jacobian.device
        dtype = self.z1_arm_jacobian.dtype
        N = self.z1_arm_jacobian.shape[0]

        # --------------------------------------------------
        # State
        # --------------------------------------------------
        J_arm = _sanitize_tensor(self.z1_arm_jacobian)  # (N, 3, 6)

        # Arm torque limits.
        #
        # Recommended:
        #   self._arm_dof_cfg_ids: indices into self._torque_limits
        #   self._arm_dof_ids:     absolute simulator DOF ids for indexing dof_pos
        #
        # self._torque_limits is usually ordered like self._cfg.asset.dof_names,
        # so prefer _arm_dof_cfg_ids if available.
        torque_limits = self.simulator._torque_limits
        if torque_limits.ndim == 1:
            tau_max_arm = torque_limits[self.simulator._arm_dof_cfg_ids]
        else:
            tau_max_arm = torque_limits[:, self.simulator._arm_dof_cfg_ids]

        tau_max_arm = _sanitize_tensor(tau_max_arm).abs().to(device=device, dtype=dtype)

        if tau_max_arm.ndim == 1:
            tau_max_arm = tau_max_arm.view(1, 6).expand(N, 6)
        elif tau_max_arm.ndim == 2 and tau_max_arm.shape[0] == 1:
            tau_max_arm = tau_max_arm.expand(N, 6)

        tau_max_arm = torch.clamp(tau_max_arm, min=1e-6)
        # --------------------------------------------------
        # Torque-weighted arm manipulability
        # --------------------------------------------------
        W_tau = torch.diag_embed(tau_max_arm ** 2)  # (N, 6, 6)

        M = J_arm @ W_tau @ J_arm.transpose(-1, -2)  # (N, 3, 3)
        M = _sanitize_tensor(0.5 * (M + M.transpose(-1, -2)))

        # Force ellipsoid shape matrix.
        #
        # If M is close to singular, _safe_inv regularizes it.
        # S_F eigenvalues are the force-capability ellipsoid axes squared,
        # up to the chosen torque-limit metric.
        S_F = _safe_inv(M, eps=self.cfg.rewards.manip_rewards.arm_ellipsoid_inv_eps)
        S_F = _sanitize_tensor(0.5 * (S_F + S_F.transpose(-1, -2)))

        # --------------------------------------------------
        # Eigenvalues of force ellipsoid
        # --------------------------------------------------
        evals_F, _ = _eig_desc(S_F)  # descending: lam1 >= lam2 >= lam3
        evals_F = torch.clamp(_sanitize_tensor(evals_F), min=1e-8, max=1e8)

        lam1 = evals_F[:, 0]
        lam2 = evals_F[:, 1]
        lam3 = evals_F[:, 2]

        # --------------------------------------------------
        # (1) Large force ellipsoid reward
        # --------------------------------------------------
        force_size = _geom_mean(evals_F)

        r_force_size = 1.0 - torch.exp(
            -self.cfg.rewards.manip_rewards.arm_ellipsoid_force_size_scale
            * force_size
        )
        r_force_size = torch.clamp(_sanitize_tensor(r_force_size), 0.0, 1.0)

        # --------------------------------------------------
        # (2) Isotropic force ellipsoid reward
        # --------------------------------------------------
        force_cond = torch.clamp(
            _sanitize_tensor(lam1 / torch.clamp(lam3, min=1e-8)),
            min=1.0,
            max=1e6,
        )

        r_force_iso_cond = _upper_reward(
            force_cond,
            self.cfg.rewards.manip_rewards.arm_ellipsoid_force_cond_max,
            sharpness=self.cfg.rewards.manip_rewards.arm_ellipsoid_iso_sharpness,
        )
        r_force_iso_cond = torch.clamp(_sanitize_tensor(r_force_iso_cond), 0.0, 1.0)

        # Optional smoother isotropy term based on log-eigenvalue spread.
        # This avoids relying only on lam1 / lam3.
        log_evals = torch.log(evals_F)
        log_mean = log_evals.mean(dim=1, keepdim=True)
        log_spread = torch.mean(torch.square(log_evals - log_mean), dim=1)

        r_force_iso_spread = torch.exp(
            -self.cfg.rewards.manip_rewards.arm_ellipsoid_log_iso_scale
            * log_spread
        )
        r_force_iso_spread = torch.clamp(_sanitize_tensor(r_force_iso_spread), 0.0, 1.0)

        r_force_iso = (
            self.cfg.rewards.manip_rewards.arm_ellipsoid_cond_iso_weight
            * r_force_iso_cond
            +
            (1.0 - self.cfg.rewards.manip_rewards.arm_ellipsoid_cond_iso_weight)
            * r_force_iso_spread
        )
        r_force_iso = torch.clamp(_sanitize_tensor(r_force_iso), 0.0, 1.0)

        # --------------------------------------------------
        # Final reward
        # --------------------------------------------------
        reward = (
            self.cfg.rewards.manip_rewards.arm_ellipsoid_size_weight
            * r_force_size
            +
            self.cfg.rewards.manip_rewards.arm_ellipsoid_iso_weight
            * r_force_iso
        )

        norm = (
            self.cfg.rewards.manip_rewards.arm_ellipsoid_size_weight
            +
            self.cfg.rewards.manip_rewards.arm_ellipsoid_iso_weight
        )
        reward = reward / max(float(norm), 1e-6)

        reward = torch.clamp(_sanitize_tensor(reward), 0.0, 1.0)

        return reward

    def _reward_arm_progress_before_torso(self):
        """Reward reducing EE error while keeping torso upright."""
        base_yaw_quat = self._get_base_yaw_quat()
        force_offset = (self.ee_force_ext_world + quat_apply(base_yaw_quat, self.current_Fxyz_gripper_cmd)) / self.gripper_force_kps
        target = self.curr_ee_goal_cart_world + force_offset
        ee_error = torch.norm(target - self.simulator.ee_pos, dim=1)

        ee_progress = torch.clamp(
            self.prev_ee_error - ee_error,
            min=0.0,
        )

        torso_tilt = torch.norm(self.simulator.projected_gravity[:, :2], dim=1)

        upright_gate = torch.exp(
            -self.cfg.rewards.upright_gate_sigma * torch.square(torso_tilt)
        )

        self.prev_ee_error[:] = ee_error

        return ee_progress * upright_gate

    def _reward_early_torso_tilt(self):
        """Penalize torso tilt before the EE has reached the target."""
        base_yaw_quat = self._get_base_yaw_quat()
        force_offset = (self.ee_force_ext_world + quat_apply(base_yaw_quat, self.current_Fxyz_gripper_cmd)) / self.gripper_force_kps
        target = self.curr_ee_goal_cart_world + force_offset
        ee_error = torch.norm(target - self.simulator.ee_pos, dim=1)

        torso_tilt = torch.norm(self.simulator.projected_gravity[:, :2], dim=1)

        not_reached_gate = torch.sigmoid(
            self.cfg.rewards.arm_before_torso_gate_sharpness
            * (ee_error - self.cfg.rewards.arm_before_torso_ee_thresh)
        )

        deadband = self.cfg.rewards.torso_tilt_deadband

        excess_tilt = torch.clamp(
            torch.square(torso_tilt) - deadband ** 2,
            min=0.0,
        )

        return not_reached_gate * excess_tilt

    def _reward_no_progress(self):
        """
        Return a bounded penalty in [0, 1]:

            0: sufficient locomotion progress
            1: stationary, reversing, unsupported, or insufficient progress

        Configure with a negative reward scale.
        """

        target_vel = self.commands[:, :2]
        target_mag = torch.norm(target_vel, dim=1)

        moving_cmd = target_mag > 0.15
        target_dir = target_vel / target_mag.unsqueeze(1).clamp_min(1e-6)

        measured_vel = self.simulator.base_lin_vel[:, :2]

        # Signed velocity EMA suppresses bouncing and alternating motion.
        smoothing_time = 0.25
        alpha = self.dt / (smoothing_time + self.dt)

        self.progress_vel_ema.lerp_(
            measured_vel,
            alpha,
        )

        aligned_speed = torch.sum(
            self.progress_vel_ema * target_dir,
            dim=1,
        )

        # Require at least 40% of commanded speed before considering progress adequate.
        required_speed = 0.40 * target_mag
        progress_ratio = aligned_speed / required_speed.clamp_min(1e-6)

        # Smooth transition from no progress to sufficient progress.
        engagement = progress_ratio.clamp(0.0, 1.0)
        engagement = engagement.square() * (3.0 - 2.0 * engagement)

        # Do not count falling, jumping, or unsupported motion as locomotion.
        contact = (
            self.simulator.link_contact_forces[
                :, self.simulator.feet_indices, 2
            ] > 5.0
        )
        supported = contact.sum(dim=1) >= 2

        # Confirm the sign convention: Genesis commonly gives approximately -1 here
        # when the torso is upright.
        upright = self.simulator.projected_gravity[:, 2] < -0.8

        valid_locomotion = supported & upright
        engagement *= valid_locomotion.float()

        no_progress = 1.0 - engagement

        # Avoid penalizing the policy immediately after reset.
        grace_complete = self.episode_length_buf * self.dt > 0.30

        return (
            no_progress
            * moving_cmd.float()
            * grace_complete.float()
        )

    def _reset_progress_statistics(self, env_ids):
        if len(env_ids) == 0:
            return

        self.progress_delta_buffer[env_ids] = 0.0
        self.progress_desired_buffer[env_ids] = 0.0
        self.progress_valid_steps[env_ids] = 0

        self.last_progress_base_pos[env_ids] = (
            self.simulator.base_pos[env_ids, :2]
        )

    def _update_progress_statistics(self):
        """Update rolling actual and commanded world-frame displacement."""

        current_step = int(self.common_step_counter)
        if self._progress_update_step == current_step:
            return

        current_pos = self.simulator.base_pos[:, :2]

        # Per-step physical displacement.
        delta_pos = current_pos - self.last_progress_base_pos
        self.last_progress_base_pos.copy_(current_pos)

        # Reject reset/teleportation spikes.
        delta_pos = torch.where(
            torch.norm(delta_pos, dim=1, keepdim=True) < 0.25,
            delta_pos,
            torch.zeros_like(delta_pos),
        )

        # Convert yaw-frame velocity commands into world-frame displacement.
        command_local = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        command_local[:, :2] = self.commands[:, :2]

        command_world = quat_apply(
            self._get_base_yaw_quat(),
            command_local,
        )[:, :2]

        desired_delta = command_world * self.dt

        index = self.progress_buffer_index
        self.progress_delta_buffer[:, index] = delta_pos
        self.progress_desired_buffer[:, index] = desired_delta

        self.progress_valid_steps.add_(1)
        self.progress_valid_steps.clamp_(
            max=self.progress_window_steps
        )

        self.progress_buffer_index = (
            index + 1
        ) % self.progress_window_steps

        self._progress_update_step = current_step


    def _reward_no_physical_progress(self):
        """
        Bounded penalty in [0, 1].

        0: sufficient net physical progress
        1: standing, shuffling, reversing, or unsupported motion

        Configure with a negative reward scale.
        """
        self._update_progress_statistics()

        actual_displacement = self.progress_delta_buffer.sum(dim=1)
        desired_displacement = self.progress_desired_buffer.sum(dim=1)

        desired_distance = torch.norm(
            desired_displacement,
            dim=1,
        )

        desired_direction = (
            desired_displacement
            / desired_distance.unsqueeze(1).clamp_min(1e-6)
        )

        aligned_progress = torch.sum(
            actual_displacement * desired_direction,
            dim=1,
        )

        # Require 40% of commanded displacement over the window.
        required_progress = 0.40 * desired_distance

        progress_ratio = (
            aligned_progress
            / required_progress.clamp_min(1e-6)
        )

        # Smooth bounded progress score.
        engagement = progress_ratio.clamp(0.0, 1.0)
        engagement = engagement.square() * (
            3.0 - 2.0 * engagement
        )

        # Only apply after enough history has accumulated.
        minimum_history = max(
            1, int(round(0.20 / self.dt))
        )
        history_ready = (
            self.progress_valid_steps >= minimum_history
        )

        moving_command = (
            torch.norm(self.commands[:, :2], dim=1) > 0.15
        )

        # Do not count falling or aerial motion as locomotion.
        contact = (
            self.simulator.link_contact_forces[
                :, self.simulator.feet_indices, 2
            ] > 5.0
        )
        supported = contact.sum(dim=1) >= 2
        upright = self.simulator.projected_gravity[:, 2] < -0.8

        engagement *= (supported & upright).float()

        penalty = 1.0 - engagement

        return (
            penalty
            * moving_command.float()
            * history_ready.float()
        )


    def _reward_swing_foot_direction(self):
        """Reward swing feet moving relative to the torso in the travel direction."""

        command_xy = self.commands[:, :2]
        command_mag = torch.norm(command_xy, dim=1)

        moving = command_mag > 0.15
        command_dir = (
            command_xy
            / command_mag.unsqueeze(1).clamp_min(1e-6)
        )

        contact = (
            self.simulator.link_contact_forces[
                :, self.simulator.feet_indices, 2
            ] > 5.0
        )
        actual_swing = ~contact

        desired_swing = 1.0 - self._get_gait_phase()
        active_swing = actual_swing.float() * desired_swing

        # World-frame foot velocity relative to the torso.
        foot_vel_relative = (
            self.simulator.feet_vel
            - self.simulator.base_lin_vel.unsqueeze(1)
        )

        # Convert relative foot velocity to the yaw-aligned command frame.
        yaw_quat = self._get_base_yaw_quat()
        yaw_quat_feet = yaw_quat.unsqueeze(1).expand(-1, 4, -1)

        foot_vel_local = quat_rotate_inverse(
            yaw_quat_feet.reshape(-1, 4),
            foot_vel_relative.reshape(-1, 3),
        ).reshape(self.num_envs, 4, 3)

        aligned_velocity = torch.sum(
            foot_vel_local[:, :, :2]
            * command_dir.unsqueeze(1),
            dim=2,
        )

        # Signed bounded reward: forward swing positive, reverse swing negative.
        per_foot_reward = torch.tanh(
            aligned_velocity / 0.4
        )

        # Require ground support to avoid rewarding aerial leg motion.
        supported = contact.sum(dim=1) >= 2

        reward = torch.sum(
            active_swing.float() * per_foot_reward,
            dim=1,
        ) / 2.0

        reward *= supported.float()
        reward *= moving.float()

        return reward

    def _reward_step_progress(self):
        """
        Reward new swing-foot displacement in the commanded travel direction.
        """
        stats = self._update_feet_air_time_stats()

        valid_liftoff = stats["valid_liftoff"]
        valid_swing = stats["valid_swing"]
        first_contact = stats["first_contact"]
        contact = stats["contact"]

        feet_xy = self.simulator.feet_pos[:, :, :2]

        # Convert command direction into the world frame.
        command_local = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        command_local[:, :2] = self.commands[:, :2]

        command_world = quat_apply(
            self._get_base_yaw_quat(),
            command_local,
        )[:, :2]

        command_mag = torch.norm(command_world, dim=1)
        command_direction = (
            command_world
            / command_mag.unsqueeze(1).clamp_min(1e-6)
        )

        # Capture position and command direction at valid liftoff.
        liftoff_mask = valid_liftoff.unsqueeze(-1)

        self.step_liftoff_pos = torch.where(
            liftoff_mask,
            feet_xy,
            self.step_liftoff_pos,
        )

        self.step_direction_world = torch.where(
            liftoff_mask,
            command_direction.unsqueeze(1).expand(-1, 4, -1),
            self.step_direction_world,
        )

        self.step_max_progress[valid_liftoff] = 0.0

        foot_displacement = feet_xy - self.step_liftoff_pos

        aligned_progress = torch.sum(
            foot_displacement * self.step_direction_world,
            dim=2,
        )

        # Reward only improvements beyond the previous maximum.
        new_max = torch.maximum(
            self.step_max_progress,
            aligned_progress,
        )

        incremental_progress = torch.relu(
            new_max - self.step_max_progress
        )

        self.step_max_progress.copy_(new_max)

        # Approximately 10 cm of swing progress gives a full cumulative reward.
        target_step_length = 0.10

        dense_reward = torch.clamp(
            incremental_progress / target_step_length,
            min=0.0,
            max=1.0,
        )

        dense_reward *= valid_swing.float()

        # Optional touchdown bonus for completing a useful step.
        completed_progress = torch.clamp(
            self.step_max_progress / target_step_length,
            min=0.0,
            max=1.0,
        )

        touchdown_bonus = (
            first_contact.float() * completed_progress
        )

        supported = contact.sum(dim=1) >= 2
        upright = self.simulator.projected_gravity[:, 2] < -0.8
        moving = command_mag > 0.15

        reward = (
            dense_reward.sum(dim=1) / 2.0
            + 0.25 * touchdown_bonus.sum(dim=1) / 2.0
        )

        reward *= supported.float()
        reward *= upright.float()
        reward *= moving.float()

        return reward