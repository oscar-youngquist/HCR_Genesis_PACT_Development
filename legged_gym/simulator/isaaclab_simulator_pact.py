"""Isaac Lab adapter for PACT and HardPACT tasks.

The base :mod:`isaaclab_simulator` intentionally remains the legacy simulator.
This subclass owns the PACT-specific coupled action, canonical state/force
contract, physics-rate contact sensing, transition buffers, and curriculum
randomization compatibility required by HardPACT and HardPACTPos.
"""

import torch

from legged_gym.utils.math_utils import quat_apply_yaw
from legged_gym.simulator.isaaclab_simulator import IsaacLabSimulator


class IsaacLabSimulator_PACT(IsaacLabSimulator):
    """Thin Isaac Lab backend adapter for the shared PACT task logic."""

    def __init__(self, cfg, sim_params: dict, device, headless):
        self.first_loop = True
        self.first_loop_feedback = None
        super().__init__(cfg, sim_params, device, headless)

    # ------------------------------------------------------------------
    # Canonical HardPACT backend boundary
    # ------------------------------------------------------------------
    def hard_pact_configuration(self):
        return torch.cat((
            self._robot.data.root_link_pos_w,
            self.hard_pact_base_quat_xyzw(), self.dof_pos,
        ), dim=-1)

    def hard_pact_velocity_world(self):
        return torch.cat((
            self._robot.data.root_link_lin_vel_w,
            self._robot.data.root_link_ang_vel_w, self.dof_vel,
        ), dim=-1)

    def hard_pact_foot_forces_world(self):
        return self._contact_sensors.data.net_forces_w[
            :, self._feet_contact_indices, :
        ]

    def hard_pact_apply_base_wrench_world(self, wrench):
        # Isaac Lab retains these world-frame values until write_data_to_sim.
        self._robot.set_external_force_and_torque(
            wrench[:, :3].unsqueeze(1), wrench[:, 3:].unsqueeze(1),
            body_ids=[self._base_link_index],
        )

    def hard_pact_capabilities(self):
        features = {
            name: True for name in (
                "ground_friction", "added_base_mass", "base_com_x",
                "base_com_y", "base_com_z", "control_delay", "kp_scale",
                "kd_scale", "armature", "joint_friction", "joint_stiffness",
                "joint_damping", "push_xy", "persistent_force",
                "persistent_torque",
            )
        }
        # These capabilities remain false until real-backend parity is proven.
        features.update(motor_strength=False, push_z=False, push_angular=False)
        return {
            "backend": "isaaclab",
            "supports_domain_rand_curriculum": False,
            "features": features,
        }

    # The corrected HardPACT losses use BARD. These compatibility methods keep
    # the pre-existing transition/storage schema without creating CPU workers.
    def _create_async_pino_workers(self):
        return None

    def _shutdown_asynic_pino_workers(self):
        return None

    def _get_pinn_wb_dynamics(self):
        return (
            self._contact_forces_buff, self._wb_mass_mat_buff,
            self._wb_bias_vec_buff, self._torso_6dof_acceleration,
        )

    def _get_pinn_feedback(self, position, dof_pos, dof_vel):
        return self._p_gains * (position - dof_pos) - self._d_gains * dof_vel

    # ------------------------------------------------------------------
    # Isaac Lab construction hooks
    # ------------------------------------------------------------------
    def _parse_cfg(self):
        super()._parse_cfg()
        self.use_domainrand_curriculum = bool(
            getattr(self._cfg.domain_rand, "use_domainrand_curriculum", False)
        )

    def _contact_sensor_update_period(self):
        # GRFs must be sampled once per physics substep, not once per policy
        # step, so interval conditioning matches the Genesis reference.
        return self._sim_params["dt"]

    def _contact_sensor_history_length(self):
        return 2

    def _resolve_feet_names(self):
        names = list(self._cfg.asset.foot_name)
        if len(names) != 4:
            raise ValueError(
                "PACT requires four configured feet in FR, FL, RR, RL order"
            )
        return names

    @staticmethod
    def _ordered_indices(available, requested):
        indices = []
        for name in requested:
            matches = [
                i for i, actual in enumerate(available)
                if name == actual or name in actual
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one match for {name!r}; got {matches}")
            indices.append(matches[0])
        return indices

    def _resolve_feet_indices(self, find_contact_indices, find_body_indices):
        contact = self._ordered_indices(
            self._contact_sensors.body_names, self._feet_names
        )
        bodies = self._ordered_indices(self._robot.body_names, self._feet_names)
        if len(contact) != 4 or len(bodies) != 4:
            raise ValueError("Could not resolve four unique PACT feet")
        return contact, bodies

    def _create_envs(self):
        super()._create_envs()
        if getattr(self._cfg.domain_rand, "randomize_joint_stiffness", False):
            self._randomize_joint_stiffness(torch.arange(self._num_envs))

    def _init_buffers(self):
        super()._init_buffers()
        self._base_world_lin_vel = torch.zeros_like(
            self._robot.data.root_link_lin_vel_w
        )
        self._base_world_ang_vel = torch.zeros_like(
            self._robot.data.root_link_ang_vel_w
        )
        self._last_base_world_lin_vel = torch.zeros_like(self._base_world_lin_vel)
        self._last_base_world_ang_vel = torch.zeros_like(self._base_world_ang_vel)
        self._torques = torch.zeros(
            self._num_envs, self._num_actions, device=self._device
        )
        self._grfs_buf = torch.zeros(self._num_envs, 12, device=self._device)
        self._contact_forces_buff = torch.zeros(
            self._num_envs, 18, device=self._device
        )
        self._wb_mass_mat_buff = torch.zeros(
            self._num_envs, 18, 18, device=self._device
        )
        self._wb_bias_vec_buff = torch.zeros(
            self._num_envs, 18, device=self._device
        )
        self._torso_6dof_acceleration = torch.zeros(
            self._num_envs, 6, device=self._device
        )
        if (self._cfg.terrain.measure_heights and
                self._cfg.terrain.obtain_terrain_info_around_feet):
            self._max_height_ahead_feet = torch.zeros(
                self._num_envs, len(self.feet_indices), device=self._device
            )
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = torch.zeros(
                self._num_envs, len(self._robot.body_names), device=self._device
            )

    # ------------------------------------------------------------------
    # PACT stepping and coupled action path
    # ------------------------------------------------------------------
    def step(self, actions):
        self.first_loop = True
        self._last_base_lin_vel[:] = self._base_lin_vel
        self._last_base_ang_vel[:] = self._base_ang_vel
        self._last_base_world_lin_vel[:] = self._base_world_lin_vel
        self._last_base_world_ang_vel[:] = self._base_world_ang_vel
        self._last_feet_vel[:] = self._robot.data.body_link_vel_w[
            :, self.feet_indices, :3
        ]
        self._last_dof_vel[:] = self._robot.data.joint_vel
        for _ in range(self._cfg.control.decimation):
            self._torques = self._compute_torques(actions)
            callback = getattr(self, "_hard_pact_pre_physics_substep", None)
            if callback is not None:
                callback()
            self._robot.set_joint_effort_target(
                torch.clip(self._torques, -self.torque_limits, self.torque_limits),
                self._dof_indices,
            )
            self._robot.write_data_to_sim()
            self._sim.step(render=False)
            self._robot.update(self._sim_params["dt"])
            self._contact_sensors.update(self._sim_params["dt"])
            callback = getattr(self, "_hard_pact_grf_post_physics_substep", None)
            if callback is not None:
                callback()
        if not self._headless:
            self._sim.render()

    def _compute_torques(self, actions):
        if actions.shape[-1] == 2 * self._num_actions:
            position = actions[:, :self._num_actions] * self._cfg.control.action_scale
            feedforward = (
                actions[:, self._num_actions:] * self._cfg.control.torque_scale
            )
            self.feedback_torques = (
                self._kp_scale * self._p_gains
                * (position + self.default_dof_pos - self.dof_pos)
                - self._kd_scale * self._d_gains * self.dof_vel
            )
            if self.first_loop:
                self.first_loop = False
                self.first_loop_feedback = self.feedback_torques.clone()
            self.feedforward_torques = feedforward
            self._unweighted_torques = self._motor_strength * (
                feedforward + self.feedback_torques
            )
            return self._motor_strength * (
                self.feedforward_tau_weight * feedforward
                + self.feedback_tau_weight * self.feedback_torques
            )

        actions_scaled = actions * self._cfg.control.action_scale
        if self._cfg.control.control_type == "P":
            torques = (
                self._kp_scale * self._p_gains
                * (actions_scaled + self.default_dof_pos - self.dof_pos)
                - self._kd_scale * self._d_gains * self.dof_vel
            )
        elif self._cfg.control.control_type == "V":
            torques = (
                self._kp_scale * self._p_gains * (actions_scaled - self.dof_vel)
                - self._kd_scale * self._d_gains
                * self._robot.data.joint_acc[:, self._dof_indices]
            )
        elif self._cfg.control.control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {self._cfg.control.control_type}")

        # HardPACTPos records the actually executed PD torque for history and
        # clone supervision even though it has no feed-forward action half.
        self.feedback_torques = torques
        self.feedforward_torques = torch.zeros_like(torques)
        self._unweighted_torques = torques
        if self.first_loop:
            self.first_loop = False
            self.first_loop_feedback = torques.clone()
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def post_physics_step(self):
        super().post_physics_step()
        self._base_world_lin_vel[:] = self._robot.data.root_link_lin_vel_w
        self._base_world_ang_vel[:] = self._robot.data.root_link_ang_vel_w
        self._grfs_buf[:] = self.hard_pact_foot_forces_world().reshape(
            self._num_envs, 12
        )
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = (
                torch.norm(self._contact_sensors.data.net_forces_w, dim=-1) > 1.0
            ).float()

    def reset_idx(self, env_ids):
        if getattr(self._cfg.domain_rand, "randomize_joint_stiffness", False):
            self._randomize_joint_stiffness(env_ids)
        super().reset_idx(env_ids)
        self._grfs_buf[env_ids] = 0.0
        self._last_base_world_lin_vel[env_ids] = 0.0
        self._last_base_world_ang_vel[env_ids] = 0.0
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = (
                torch.norm(self._contact_sensors.data.net_forces_w, dim=-1) > 1.0
            ).float()

    # ------------------------------------------------------------------
    # Terrain targets used by legacy PACT rewards/labels
    # ------------------------------------------------------------------
    def _calc_terrain_info_around_feet(self):
        super()._calc_terrain_info_around_feet()
        foot_points = (
            (self._feet_pos + self._cfg.terrain.border_size)
            / self._cfg.terrain.horizontal_scale
        ).long()
        px = foot_points[:, :, 0].reshape(-1).clamp(
            0, self._height_samples.shape[0] - 2
        )
        py = foot_points[:, :, 1].reshape(-1).clamp(
            0, self._height_samples.shape[1] - 2
        )
        self._update_max_height_ahead_feet(px, py)

    def _update_max_height_ahead_feet(self, px, py):
        direction_base = torch.cat((
            self._base_lin_vel[:, :2],
            torch.zeros(self._num_envs, 1, device=self._device),
        ), dim=-1)
        direction_world = quat_apply_yaw(self._base_quat, direction_base)[:, :2]
        norm = torch.linalg.norm(direction_world, dim=-1, keepdim=True)
        heading = quat_apply_yaw(
            self._base_quat,
            torch.tensor([1.0, 0.0, 0.0], device=self._device).repeat(
                self._num_envs, 1
            ),
        )[:, :2]
        direction_world = torch.where(
            (norm.squeeze(-1) > 1e-4).unsqueeze(-1),
            direction_world / norm.clamp_min(1e-6), heading,
        )
        forward = direction_world.repeat_interleave(len(self._feet_indices), 0)
        lateral = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        maximum = None
        for fwd in self._cfg.rewards.edge_clearance_forward_cells:
            for lat in self._cfg.rewards.edge_clearance_lateral_cells:
                offset = torch.round(forward * fwd + lateral * lat).long()
                sx = (px + offset[:, 0]).clamp(0, self._height_samples.shape[0] - 1)
                sy = (py + offset[:, 1]).clamp(0, self._height_samples.shape[1] - 1)
                sample = self._height_samples[sx, sy]
                maximum = sample if maximum is None else torch.maximum(maximum, sample)
        self._max_height_ahead_feet[:] = maximum.view(
            self._num_envs, -1
        ) * self._cfg.terrain.vertical_scale

    def calc_feet_near_edge(self):
        if self._cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self._num_envs, len(self._feet_indices), device=self._device,
                dtype=torch.bool,
            )
        feet_xy = self._feet_pos[:, :, :2]
        points = ((feet_xy + self._cfg.terrain.border_size) /
                  self._cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].clamp(0, self._edge_mask.shape[0] - 1)
        py = points[:, :, 1].clamp(0, self._edge_mask.shape[1] - 1)
        near_edge = torch.zeros_like(px, dtype=torch.bool)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                ex = (px + dx).clamp(0, self._edge_mask.shape[0] - 1)
                ey = (py + dy).clamp(0, self._edge_mask.shape[1] - 1)
                edge_xy = torch.stack((
                    ex.float() * self._cfg.terrain.horizontal_scale
                    - self._cfg.terrain.border_size,
                    ey.float() * self._cfg.terrain.horizontal_scale
                    - self._cfg.terrain.border_size,
                ), dim=-1)
                near_edge |= self._edge_mask[ex, ey] & (
                    torch.norm(feet_xy - edge_xy, dim=-1)
                    < self._cfg.rewards.feet_edge_threshold
                )
        return near_edge

    def _create_trimesh(self):
        super()._create_trimesh()
        self._edge_mask = torch.as_tensor(
            self._terrain.edge_mask, device=self._device, dtype=torch.bool
        )

    # ------------------------------------------------------------------
    # PACT curriculum-compatible realized randomization
    # ------------------------------------------------------------------
    def _init_domain_params(self):
        super()._init_domain_params()
        self._robot_mass = float(self._robot.data.default_mass[0].sum().item())
        self._rand_wrench_vels = torch.zeros(
            self._num_envs, 3, device=self._device
        )
        self._joint_stiffness = torch.zeros(
            self._num_envs, 1, device=self._device
        )
        self._motor_strength = torch.ones(
            self._num_envs, self._num_actions, device=self._device
        )
        self.feedforward_tau_weight = torch.ones(
            self._num_envs, 1, device=self._device
        )
        self.feedback_tau_weight = torch.ones(
            self._num_envs, 1, device=self._device
        )
        self.feedforward_tau_weight_clean = self.feedforward_tau_weight.clone()
        self.feedback_tau_weight_clean = self.feedback_tau_weight.clone()

    def _randomize_base_mass(self, env_ids):
        configured = getattr(self._cfg.domain_rand, "added_mass_range", (-1.0, 1.0))
        original = self._cfg.domain_rand.added_mass_range
        minimum = float(getattr(self._cfg.domain_rand, "added_mass_min", configured[0]))
        default_max = getattr(self._cfg.domain_rand, "min_added_mass_max", configured[1])
        maximum = float(getattr(self, "mass_max_value", default_max))
        self._cfg.domain_rand.added_mass_range = (minimum, maximum)
        try:
            super()._randomize_base_mass(env_ids)
        finally:
            self._cfg.domain_rand.added_mass_range = original

    def _randomize_com_displacement(self, env_ids):
        if not hasattr(self, "com_delta_x_value"):
            return super()._randomize_com_displacement(env_ids)
        original = (
            self._cfg.domain_rand.com_pos_x_range,
            self._cfg.domain_rand.com_pos_y_range,
            self._cfg.domain_rand.com_pos_z_range,
        )
        self._cfg.domain_rand.com_pos_x_range = (
            -self.com_delta_x_value, self.com_delta_x_value
        )
        self._cfg.domain_rand.com_pos_y_range = (
            -self.com_delta_y_value, self.com_delta_y_value
        )
        self._cfg.domain_rand.com_pos_z_range = self.com_delta_z_val_bounds
        try:
            return super()._randomize_com_displacement(env_ids)
        finally:
            (self._cfg.domain_rand.com_pos_x_range,
             self._cfg.domain_rand.com_pos_y_range,
             self._cfg.domain_rand.com_pos_z_range) = original

    def _randomize_joint_friction(self, env_ids):
        original = self._cfg.domain_rand.joint_friction_range
        self._cfg.domain_rand.joint_friction_range = getattr(
            self, "joint_friction_bound_current", original
        )
        try:
            return super()._randomize_joint_friction(env_ids)
        finally:
            self._cfg.domain_rand.joint_friction_range = original

    def _randomize_joint_damping(self, env_ids):
        original = self._cfg.domain_rand.joint_damping_range
        self._cfg.domain_rand.joint_damping_range = getattr(
            self, "joint_damping_bound_current", original
        )
        try:
            return super()._randomize_joint_damping(env_ids)
        finally:
            self._cfg.domain_rand.joint_damping_range = original

    def _randomize_joint_stiffness(self, env_ids):
        if len(env_ids) == 0:
            return
        bounds = getattr(
            self, "joint_stiffness_bound_current",
            self._cfg.domain_rand.joint_stiffness_range_start,
        )
        stiffness = torch.rand((len(env_ids), 1), device=self._device)
        stiffness = stiffness * (bounds[1] - bounds[0]) + bounds[0]
        self._joint_stiffness[env_ids] = stiffness
        self._robot.write_joint_stiffness_to_sim(
            stiffness.repeat(1, self._num_actions), self._dof_indices, env_ids
        )

    @property
    def dof_pos_limits_hard(self):
        return self._robot.data.joint_pos_limits[0, self._dof_indices, :]

    @property
    def _link_contact_forces(self):
        return self.link_contact_forces
