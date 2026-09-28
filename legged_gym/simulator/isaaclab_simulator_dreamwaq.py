"""IsaacLab Go2 DreamWaQ adapter with separate body and sensor index spaces."""
import torch
from .isaaclab_simulator import IsaacLabSimulator
from .dreamwaq_adapter import DreamWaQAdapter


class IsaacLabSimulatorDreamWaQ(DreamWaQAdapter, IsaacLabSimulator):
    def __init__(self, cfg, sim_params, device, headless):
        super().__init__(cfg, sim_params, device, headless)
        if headless:
            handle = getattr(self._sim, '_app_control_on_stop_handle', None)
            if handle is not None:
                handle.unsubscribe()
                self._sim._app_control_on_stop_handle = None

    def _create_envs(self):
        super()._create_envs()
        self._robot_mass = self._robot.data.default_mass[0].sum().to(self._device)
        self._configure_contact_indices()

    @staticmethod
    def _indices_for_names(available, requested, source):
        """Resolve exact names in the requested order; reject ambiguous assets."""
        if len(set(requested)) != len(requested):
            raise ValueError(f"Duplicate requested names for {source}: {requested}")
        indices = []
        for name in requested:
            matches = [i for i, actual in enumerate(available) if actual == name]
            if len(matches) != 1:
                raise ValueError(f"Expected one {source} entry for {name!r}; found {len(matches)}")
            indices.append(matches[0])
        return indices

    def _configure_contact_indices(self):
        # Articulation data and ContactSensor data have independent body orders.
        # Never reorder the public force tensor: inherited rewards index it with
        # feet_contact_indices / penalized_contact_indices / termination_contact_indices.
        body_names = self._robot.body_names
        sensor_names = self._contact_sensors.body_names
        feet_names = self._cfg.asset.feet_names
        if len(feet_names) != 4:
            raise ValueError("Go2 requires four configured feet in FR, FL, RR, RL order")
        self._feet_indices = self._indices_for_names(body_names, feet_names, 'articulation')
        self._feet_contact_indices = self._indices_for_names(sensor_names, feet_names, 'contact sensor')
        self._contact_state_link_indices = self._indices_for_names(
            sensor_names, self._cfg.asset.contact_state_link_names, 'contact sensor')
        self._termination_contact_indices = [i for i, name in enumerate(sensor_names)
            if any(part in name for part in self._cfg.asset.terminate_after_contacts_on)]
        self._penalized_contact_indices = [i for i, name in enumerate(sensor_names)
            if any(part in name for part in self._cfg.asset.penalize_contacts_on)]

    def _init_buffers(self):
        super()._init_buffers()
        # Parent gains are constructed in articulation order, not policy order.
        self._p_gains = self._p_gains[self._dof_indices]
        self._d_gains = self._d_gains[self._dof_indices]
        self._torques = torch.zeros(self._num_envs, self._num_actions, device=self._device)

    @property
    def last_dof_vel(self):
        return self._last_dof_vel[:, self._dof_indices]

    @property
    def torque_limits(self):
        if not hasattr(self, '_dreamwaq_torque_limits'):
            sim_limits = self._robot.data.joint_effort_limits[0]
            motor_limits = torch.full_like(sim_limits, float('nan'))
            for actuator in self._robot.actuators.values():
                motor_limits[actuator.joint_indices] = actuator.effort_limit[0]
            limits = torch.minimum(sim_limits, motor_limits)[self._dof_indices]
            if not torch.all(torch.isfinite(limits) & (limits > 0)):
                raise ValueError('Invalid Go2 motor effort limits')
            self._dreamwaq_torque_limits = limits
        return self._dreamwaq_torque_limits

    @property
    def dof_vel_limits(self):
        # Explicit actuator limits may be lower than PhysX's solver limits.
        if not hasattr(self, '_dreamwaq_velocity_limits'):
            sim_limits = self._robot.data.joint_vel_limits[0]
            motor_limits = torch.full_like(sim_limits, float('nan'))
            for actuator in self._robot.actuators.values():
                motor_limits[actuator.joint_indices] = actuator.velocity_limit[0]
            limits = torch.minimum(sim_limits, motor_limits)[self._dof_indices]
            if not torch.all(torch.isfinite(limits) & (limits > 0)):
                raise ValueError('Invalid Go2 motor velocity limits')
            self._dreamwaq_velocity_limits = limits
        return self._dreamwaq_velocity_limits

    @property
    def torques(self):
        # Aligned PACT rewards use the bounded command (before motor-model clipping).
        return self._torques

    def _compute_torques(self, actions):
        self._torques = self._bounded_position_torques(actions)
        self._robot.set_joint_effort_target(self._torques, self._dof_indices)

    def _install_joint_stiffness(self, env_ids, values):
        self._robot.write_joint_stiffness_to_sim(values, self._dof_indices, env_ids)

    def _apply_velocity_push(self, env_ids, linear, angular):
        velocity = self._robot.data.root_link_vel_w[env_ids].clone()
        velocity[:, :3] += linear
        velocity[:, 3:6] += angular
        self._robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

    def _update_surrounding_heights(self):
        if self._cfg.terrain.mesh_type == 'plane':
            self._measured_heights.zero_()
            return
        super()._update_surrounding_heights()

    def _calc_terrain_info_around_feet(self):
        if self._cfg.terrain.mesh_type == 'plane':
            self._height_around_feet.zero_()
            return
        super()._calc_terrain_info_around_feet()
