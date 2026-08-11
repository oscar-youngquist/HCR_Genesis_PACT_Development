"""Isaac Gym adapters for the three B1Z1 training approaches.

The task environments keep their Genesis-era simulator contract.  This file
translates that contract to Isaac Gym while sharing rigid-body state, external
wrench, and domain-randomization plumbing across UniFP, PACT-Pos, and PACT.
Only their action-to-torque mappings differ.
"""

from collections import deque

import numpy as np
import torch

from legged_gym import SIMULATOR
from legged_gym.utils.math_utils import quat_rotate_inverse, torch_rand_float
from .isaacgym_simulator import IsaacGymSimulator

if "isaacgym" in SIMULATOR:
    from isaacgym import gymapi, gymtorch


class _IsaacGymSimulatorB1Z1(IsaacGymSimulator):
    """Shared Isaac Gym implementation of the B1Z1 simulator-facing API."""

    def __init__(self, cfg, sim_params, sim_device="cuda:0", headless=False):
        # The Genesis B1Z1 configs define the authoritative control period.
        # Make the Isaac physics period agree with it before SimParams is built.
        sim_params = dict(sim_params)
        sim_params["dt"] = cfg.control.dt / cfg.control.decimation
        self._enable_dof_force_sensors = True
        super().__init__(cfg, sim_params, sim_device, headless)
        self.first_loop = True
        self.first_loop_feedback = None

    # ---------------------------------------------------------------------
    # Construction and state
    # ---------------------------------------------------------------------
    def _parse_cfg(self):
        super()._parse_cfg()
        self._control_dt = self._cfg.control.dt
        self._num_learned_actions = self._cfg.env.num_actions
        self._wb_dim = self._cfg.env.whole_body_dim
        self._grf_dim = self._cfg.env.grf_dim

        dr = self._cfg.domain_rand
        self.n_digits = 2
        self.vert_interval_min = dr.vert_interval_min
        self.vert_interval_max = dr.vert_interval_max
        self.push_interval_min = dr.push_interval_min
        self.push_interval_max = dr.push_interval_max
        self.wrench_timeout_min = dr.wrench_timeout_min
        self.wrench_timeout_max = dr.wrench_timeout_max

        self.use_domainrand_curriculum = dr.use_domainrand_curriculum
        # Isaac Gym GPU actors cannot accept these physical-property updates
        # after construction. Select either curriculum endpoint up front.
        self.isaacgym_use_final_domain_rand_ranges = getattr(
            dr, "isaacgym_use_final_domain_rand_ranges", False
        )
        construction_range_index = (
            1 if self.isaacgym_use_final_domain_rand_ranges else 0
        )
        self.com_rand_z_positive = dr.com_rand_z_positive
        self.num_push_steps = dr.num_push_steps
        self.push_warmup_step = dr.push_warmup

        self.push_bounds = [dr.min_push_vel_xy, dr.max_push_vel_xy]
        self.vert_bounds = [dr.min_vertical_push, dr.max_vertical_push]
        self.wrench_bounds = [dr.min_push_torque, dr.max_push_torque]
        self.push_diff = self.push_bounds[1] - self.push_bounds[0]
        self.vert_diff = self.vert_bounds[1] - self.vert_bounds[0]
        self.wrench_diff = self.wrench_bounds[1] - self.wrench_bounds[0]
        self.push_value = self.push_bounds[0]
        self.vert_value = self.vert_bounds[0]
        self.wrench_value = self.wrench_bounds[0]

        self.mass_min = dr.added_mass_min
        self.max_mass_bounds = [dr.min_added_mass_max, dr.max_added_mass_max]
        self.mass_bounds_diff = self.max_mass_bounds[1] - self.max_mass_bounds[0]
        self.mass_max_value = self.max_mass_bounds[construction_range_index]

        self._has_gripper = hasattr(self._cfg.asset, "gripper_name")
        if self._has_gripper:
            self.grip_mass_min = dr.gripper_mass_min
            self.grip_max_mass_bounds = [
                dr.min_gripper_added_mass_max,
                dr.max_gripper_added_mass_max,
            ]
            self.grip_mass_bounds_diff = (
                self.grip_max_mass_bounds[1] - self.grip_max_mass_bounds[0]
            )
            self.grip_mass_max_value = self.grip_max_mass_bounds[
                construction_range_index
            ]

        self.com_delta_x_bounds = [dr.com_displacement_x_min, dr.com_displacement_x_max]
        self.com_delta_y_bounds = [dr.com_displacement_y_min, dr.com_displacement_y_max]
        z_min = dr.com_displacement_z_min_pos if self.com_rand_z_positive else dr.com_displacement_z_min
        self.com_delta_z_bounds = [z_min, dr.com_displacement_z_max]
        self.com_delta_x_diff = self.com_delta_x_bounds[1] - self.com_delta_x_bounds[0]
        self.com_delta_y_diff = self.com_delta_y_bounds[1] - self.com_delta_y_bounds[0]
        self.com_delta_z_diff = self.com_delta_z_bounds[1] - self.com_delta_z_bounds[0]
        self.com_delta_x_value = self.com_delta_x_bounds[construction_range_index]
        self.com_delta_y_value = self.com_delta_y_bounds[construction_range_index]
        self.com_delta_z_value = (
            self.com_delta_z_bounds[1]
            if self.isaacgym_use_final_domain_rand_ranges
            else dr.com_displacement_z_min
        )
        self.com_delta_z_val_bounds = (
            [-dr.com_displacement_z_min, self.com_delta_z_value]
            if self.com_rand_z_positive
            else [-self.com_delta_z_value, self.com_delta_z_value]
        )

        self.joint_stiffness_bounds_start = np.asarray(dr.joint_stiffness_range_start)
        self.joint_stiffness_range = np.asarray(dr.joint_stiffness_range_end) - self.joint_stiffness_bounds_start
        self.joint_stiffness_bound_current = (
            np.asarray(dr.joint_stiffness_range_end)
            if self.isaacgym_use_final_domain_rand_ranges
            else self.joint_stiffness_bounds_start.copy()
        )
        self.joint_damping_bounds_start = np.asarray(dr.joint_damping_range_start)
        self.joint_damping_range = np.asarray(dr.joint_damping_range_end) - self.joint_damping_bounds_start
        self.joint_damping_bound_current = (
            np.asarray(dr.joint_damping_range_end)
            if self.isaacgym_use_final_domain_rand_ranges
            else self.joint_damping_bounds_start.copy()
        )
        self.joint_friction_bounds_start = np.asarray(dr.joint_friction_range_start)
        self.joint_friction_range = np.asarray(dr.joint_friction_range_end) - self.joint_friction_bounds_start
        self.joint_friction_bound_current = (
            np.asarray(dr.joint_friction_range_end)
            if self.isaacgym_use_final_domain_rand_ranges
            else self.joint_friction_bounds_start.copy()
        )
        self._init_domain_rand_curriculum_state()

    def _create_envs(self):
        super()._create_envs()

        def body_indices(patterns):
            if isinstance(patterns, str):
                patterns = [patterns]
            indices = []
            for pattern in patterns:
                indices.extend(i for i, name in enumerate(self._body_names) if pattern in name)
            return indices

        self._thigh_names = getattr(
            self._cfg.asset, "thigh_name", ["FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"]
        )
        self._thigh_indices = torch.tensor(
            body_indices(self._thigh_names), dtype=torch.long, device=self._device
        )
        if len(self._thigh_indices) != len(self._feet_indices):
            raise RuntimeError("B1Z1 Isaac adapter could not resolve all four thigh links")

        if self._has_gripper:
            gripper_indices = body_indices(self._cfg.asset.gripper_name)
            if len(gripper_indices) != 1:
                raise RuntimeError(
                    f"Expected one gripper link matching '{self._cfg.asset.gripper_name}', "
                    f"found {gripper_indices}"
                )
            self._gripper_index = gripper_indices[0]

        arm_names = [
            "z1_waist",
            "z1_shoulder",
            "z1_elbow",
            "z1_wrist_angle",
            "z1_forearm_roll",
            "z1_wrist_rotate",
        ]
        self._arm_dof_cfg_ids = torch.tensor(
            [self._cfg.asset.dof_names.index(name) for name in arm_names],
            dtype=torch.long,
            device=self._device,
        )
        self._dof_indices_tensor = torch.tensor(
            self._dof_indices, dtype=torch.long, device=self._device
        )

    def _init_domain_params(self):
        n, d, device = self._num_envs, self._num_dof, self._device
        self._friction_values = torch.zeros(n, 1, device=device)
        self._added_base_mass = torch.zeros(n, 1, device=device)
        self._added_gripper_mass = torch.zeros(n, 1, device=device)
        self._base_com_bias = torch.zeros(n, 3, device=device)
        self._rand_push_vels = torch.zeros(n, 3, device=device)
        self._rand_wrench_vels = torch.zeros(n, 3, device=device)
        # UniFP privileged observations use one scalar per environment for
        # each passive joint-property randomization.
        self._joint_armature = torch.zeros(n, 1, device=device)
        self._joint_friction = torch.zeros(n, 1, device=device)
        self._joint_stiffness = torch.zeros(n, 1, device=device)
        self._joint_damping = torch.zeros(n, 1, device=device)
        self._kp_scale = torch.ones(n, d, device=device)
        self._kd_scale = torch.ones(n, d, device=device)
        self._motor_strength = torch.ones(n, d, device=device)

    def _process_rigid_shape_props(self, props, env_id):
        if self._cfg.domain_rand.randomize_friction:
            low, high = self._cfg.domain_rand.friction_range
            friction = float(np.random.uniform(low, high))
            for prop in props:
                prop.friction = friction
            self._friction_values[env_id, 0] = friction
        return props

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self._dof_pos_limits = torch.zeros(self._num_dof, 2, device=self._device)
            self._dof_vel_limits = torch.zeros(self._num_dof, device=self._device)
            self._torque_limits = torch.zeros(self._num_dof, device=self._device)
            for i in range(self._num_dof):
                self._dof_pos_limits[i] = torch.tensor(
                    [props["lower"][i], props["upper"][i]], device=self._device
                )
                self._dof_vel_limits[i] = float(props["velocity"][i])
                self._torque_limits[i] = float(props["effort"][i])
                midpoint = self._dof_pos_limits[i].mean()
                span = self._dof_pos_limits[i, 1] - self._dof_pos_limits[i, 0]
                soft = self._cfg.rewards.soft_dof_pos_limit
                self._dof_pos_limits[i, 0] = midpoint - 0.5 * span * soft
                self._dof_pos_limits[i, 1] = midpoint + 0.5 * span * soft

        dr = self._cfg.domain_rand
        if dr.randomize_joint_armature:
            value = float(np.random.uniform(*dr.joint_armature_range))
            props["armature"][:] = value
            self._joint_armature[env_id, 0] = value
        if dr.randomize_joint_friction:
            value = float(np.random.uniform(*self.joint_friction_bound_current))
            props["friction"][:] = value
            self._joint_friction[env_id, 0] = value
        if dr.randomize_joint_damping:
            value = float(np.random.uniform(*self.joint_damping_bound_current))
            props["damping"][:] = value
            self._joint_damping[env_id, 0] = value
        if dr.randomize_joint_stiffness:
            value = float(np.random.uniform(*self.joint_stiffness_bound_current))
            props["stiffness"][:] = value
            self._joint_stiffness[env_id, 0] = value
        return props

    def _process_rigid_body_props(self, props, env_id):
        dr = self._cfg.domain_rand
        base_index = self._body_names.index(self._cfg.asset.base_link_name)
        if dr.randomize_base_mass:
            requested = float(np.random.uniform(self.mass_min, self.mass_max_value))
            old_mass = props[base_index].mass
            props[base_index].mass = max(1.0e-4, old_mass + requested)
            self._added_base_mass[env_id, 0] = props[base_index].mass - old_mass
        if dr.randomize_com_displacement:
            displacement = np.array(
                [
                    np.random.uniform(-self.com_delta_x_value, self.com_delta_x_value),
                    np.random.uniform(-self.com_delta_y_value, self.com_delta_y_value),
                    np.random.uniform(*self.com_delta_z_val_bounds),
                ],
                dtype=np.float32,
            )
            props[base_index].com.x += float(displacement[0])
            props[base_index].com.y += float(displacement[1])
            props[base_index].com.z += float(displacement[2])
            self._base_com_bias[env_id] = torch.as_tensor(displacement, device=self._device)
        if self._has_gripper and dr.randomize_gripper_mass:
            gripper_index = self._body_names.index(self._cfg.asset.gripper_name)
            requested = float(np.random.uniform(self.grip_mass_min, self.grip_mass_max_value))
            old_mass = props[gripper_index].mass
            props[gripper_index].mass = max(1.0e-4, old_mass + requested)
            self._added_gripper_mass[env_id, 0] = props[gripper_index].mass - old_mass
        return props

    def _init_buffers(self):
        super()._init_buffers()
        self.common_step_counter = 0
        self._actuation_torques = torch.zeros_like(self._torques)
        self.unclipped_torques = torch.zeros_like(self._torques)
        self.executed_torques = torch.zeros_like(self._torques)
        self.feedback_torques = torch.zeros_like(self._torques)
        self.feedforward_torques = torch.zeros_like(self._torques)
        self.combined_feedback_torques = torch.zeros_like(self._torques)
        self.combined_feedforward_torques = torch.zeros_like(self._torques)

        dof_force_tensor = self._gym.acquire_dof_force_tensor(self._sim)
        self._dof_force_raw = gymtorch.wrap_tensor(dof_force_tensor).view(
            self._num_envs, self._num_dof
        )
        self._dof_tau = torch.zeros_like(self._dof_force_raw)
        self._thigh_pos = self._rigid_body_states[:, self._thigh_indices, 0:3]
        self._ee_pos = self._rigid_body_states[:, self._gripper_index, 0:3]
        self._ee_quat = self._rigid_body_states[:, self._gripper_index, 3:7]
        self._ee_vel = self._rigid_body_states[:, self._gripper_index, 7:10]
        self._grfs_buf = torch.zeros(self._num_envs, self._grf_dim, device=self._device)
        self._base_world_lin_vel = self._root_states[:, 7:10]
        self._base_world_ang_vel = self._root_states[:, 10:13]
        self._last_base_world_lin_vel = torch.zeros_like(self._base_world_lin_vel)
        self._last_base_world_ang_vel = torch.zeros_like(self._base_world_ang_vel)

        self._ee_force_world = torch.zeros(self._num_envs, 3, device=self._device)
        self._base_force_world = torch.zeros_like(self._ee_force_world)
        self._base_torque_world = torch.zeros_like(self._ee_force_world)
        self._external_force_world = torch.zeros(
            self._num_envs, self._num_bodies, 3, device=self._device
        )
        self._external_torque_world = torch.zeros_like(self._external_force_world)

        self.feedforward_tau_weight = torch.ones(self._num_envs, 1, device=self._device)
        self.feedback_tau_weight = torch.ones(self._num_envs, 1, device=self._device)
        self.wrench_timeouts = torch_rand_float(
            self.wrench_timeout_min, self.wrench_timeout_max, (self._num_envs, 1), self._device
        )
        self.push_timeouts = torch_rand_float(
            self.push_interval_min, self.push_interval_max, (self._num_envs, 1), self._device
        )
        self.vert_timeouts = torch_rand_float(
            self.vert_interval_min, self.vert_interval_max, (self._num_envs, 1), self._device
        )

        all_env_ids = torch.arange(self._num_envs, device=self._device)
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(all_env_ids)
        if self._cfg.domain_rand.randomize_motor_strength:
            self._randomize_motor_strength(all_env_ids)
        self.post_physics_step()

    def step(self, actions):
        self._render()
        self._last_base_lin_vel.copy_(self._base_lin_vel)
        self._last_base_ang_vel.copy_(self._base_ang_vel)
        self._last_base_world_lin_vel.copy_(self._base_world_lin_vel)
        self._last_base_world_ang_vel.copy_(self._base_world_ang_vel)
        self._last_feet_vel.copy_(self._feet_vel)
        self._last_dof_vel.copy_(self._dof_vel)
        self.first_loop = True

        for _ in range(self._cfg.control.decimation):
            torques_cfg = self._compute_torques(actions)
            limits = self.torque_limits
            self.executed_torques = torch.clamp(torques_cfg, -1.1 * limits, 1.1 * limits)
            self._torques.copy_(self.executed_torques)
            self._actuation_torques.zero_()
            self._actuation_torques[:, self._dof_indices_tensor] = self.executed_torques
            self._gym.set_dof_actuation_force_tensor(
                self._sim, gymtorch.unwrap_tensor(self._actuation_torques)
            )
            self._apply_external_forces()
            self._gym.simulate(self._sim)
            self._gym.fetch_results(self._sim, True)
            self._gym.refresh_dof_state_tensor(self._sim)

    def post_physics_step(self):
        super().post_physics_step()
        self.common_step_counter += 1
        self._gym.refresh_dof_force_tensor(self._sim)
        self._dof_tau.copy_(self._dof_force_raw[:, self._dof_indices_tensor])
        self._thigh_pos = self._rigid_body_states[:, self._thigh_indices, 0:3]
        self._ee_pos = self._rigid_body_states[:, self._gripper_index, 0:3]
        self._ee_quat = self._rigid_body_states[:, self._gripper_index, 3:7]
        self._ee_vel = self._rigid_body_states[:, self._gripper_index, 7:10]
        self._grfs_buf.copy_(
            self._link_contact_forces[:, self._feet_indices, :].reshape(self._num_envs, -1)
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self._cfg.domain_rand.randomize_motor_strength:
            self._randomize_motor_strength(env_ids)
        self._dof_tau[env_ids] = 0.0
        self._grfs_buf[env_ids] = 0.0
        self._last_base_world_lin_vel[env_ids] = 0.0
        self._last_base_world_ang_vel[env_ids] = 0.0

    def reset_dofs(self, env_ids, dof_pos, dof_vel):
        raw_pos = self._dof_pos[env_ids].clone()
        raw_vel = self._dof_vel[env_ids].clone()
        raw_pos[:, self._dof_indices_tensor] = dof_pos
        raw_vel[:, self._dof_indices_tensor] = dof_vel
        self._dof_pos[env_ids] = raw_pos
        self._dof_vel[env_ids] = raw_vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self._gym.set_dof_state_tensor_indexed(
            self._sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    # ---------------------------------------------------------------------
    # External disturbances
    # ---------------------------------------------------------------------
    def apply_ee_force(self, force_world):
        self._ee_force_world.copy_(force_world)

    def apply_base_force(self, force_world):
        self._base_force_world.copy_(force_world)

    def apply_base_torque(self, torque_world):
        self._base_torque_world.copy_(torque_world)

    def _apply_external_forces(self):
        commands = self._cfg.commands
        apply_ee = (
            self._has_gripper
            and commands.push_gripper_stators
            and commands.apply_ee_external_forces
        )
        apply_base = commands.push_robot_base and commands.apply_base_external_forces
        apply_torque = (
            commands.push_robot_base
            and getattr(commands, "apply_base_external_torques", False)
        )
        if not (apply_ee or apply_base or apply_torque):
            return

        self._external_force_world.zero_()
        self._external_torque_world.zero_()
        if apply_ee:
            self._external_force_world[:, self._gripper_index] = self._ee_force_world
        if apply_base:
            self._external_force_world[:, self._base_link_index] = self._base_force_world
        if apply_torque:
            self._external_torque_world[:, self._base_link_index] = self._base_torque_world
        self._gym.apply_rigid_body_force_tensors(
            self._sim,
            gymtorch.unwrap_tensor(self._external_force_world),
            gymtorch.unwrap_tensor(self._external_torque_world),
            gymapi.ENV_SPACE,
        )

    def push_robots(self):
        """Mirror UniFP's intermittent base linear/angular velocity impulses."""
        push_steps = torch.clamp((self.push_timeouts / self._control_dt).long().view(-1), min=1)
        wrench_steps = torch.clamp((self.wrench_timeouts / self._control_dt).long().view(-1), min=1)
        vert_steps = torch.clamp((self.vert_timeouts / self._control_dt).long().view(-1), min=1)
        push_mask = (self.common_step_counter % push_steps) == 0
        wrench_mask = (self.common_step_counter % wrench_steps) == 0
        vert_mask = (self.common_step_counter % vert_steps) == 0

        if torch.any(push_mask):
            count = int(push_mask.sum())
            values = torch_rand_float(-self.push_value, self.push_value, (count, 2), self._device)
            self._rand_push_vels[push_mask, :2] = values
            self._root_states[push_mask, 7:9] += values
        if torch.any(wrench_mask):
            count = int(wrench_mask.sum())
            values = torch_rand_float(-self.wrench_value, self.wrench_value, (count, 3), self._device)
            self._rand_wrench_vels[wrench_mask] = values
            self._root_states[wrench_mask, 10:13] += values
        if torch.any(vert_mask):
            count = int(vert_mask.sum())
            values = torch_rand_float(-self.vert_value, 0.0, (count, 1), self._device)
            self._rand_push_vels[vert_mask, 2:3] = values
            self._root_states[vert_mask, 9:10] += values
        if torch.any(push_mask | wrench_mask | vert_mask):
            self._gym.set_actor_root_state_tensor(
                self._sim, gymtorch.unwrap_tensor(self._root_states)
            )

    # ---------------------------------------------------------------------
    # Domain-randomization controller state
    # ---------------------------------------------------------------------
    def _randomize_pd_gain(self, env_ids):
        self._kp_scale[env_ids] = torch_rand_float(
            *self._cfg.domain_rand.kp_range,
            (len(env_ids), self._num_dof),
            self._device,
        )
        self._kd_scale[env_ids] = torch_rand_float(
            *self._cfg.domain_rand.kd_range,
            (len(env_ids), self._num_dof),
            self._device,
        )

    def _randomize_motor_strength(self, env_ids):
        self._motor_strength[env_ids] = torch_rand_float(
            *self._cfg.domain_rand.motor_strength_range,
            (len(env_ids), self._num_dof),
            self._device,
        )

    # Isaac Gym GPU simulations cannot safely rewrite these actor properties
    # after preparation. They are sampled per environment during construction.
    def _randomize_friction(self, env_ids):
        return None

    def _randomize_base_mass(self, env_ids):
        return None

    def _randomize_gripper_mass(self, env_ids):
        return None

    def _randomize_com_displacement(self, env_ids):
        return None

    def _randomize_joint_armature(self, env_ids):
        return None

    def _randomize_joint_friction(self, env_ids):
        return None

    def _randomize_joint_stiffness(self, env_ids):
        return None

    def _randomize_joint_damping(self, env_ids):
        return None

    def _init_domain_rand_curriculum_state(self):
        dr = self._cfg.domain_rand
        self.domain_rand_joint_dynamics_progress = 0.0
        self.domain_rand_mass_com_progress = 0.0
        self.domain_rand_disturbance_progress = 0.0
        self.domain_rand_curriculum_phases = []
        if getattr(dr, "use_joint_dynamics_curriculum", True):
            self.domain_rand_curriculum_phases.append("joint_dynamics")
        if getattr(dr, "use_mass_com_curriculum", True):
            self.domain_rand_curriculum_phases.append("mass_com")
        if getattr(dr, "use_disturbance_curriculum", True):
            self.domain_rand_curriculum_phases.append("disturbance")
        self.domain_rand_phase = (
            self.domain_rand_curriculum_phases[0]
            if self.domain_rand_curriculum_phases
            else "complete"
        )
        self.domain_rand_joint_dynamics_delta = getattr(dr, "joint_dynamics_progress_delta", 0.002)
        self.domain_rand_mass_com_delta = getattr(dr, "mass_com_progress_delta", 0.002)
        self.domain_rand_disturbance_delta = getattr(dr, "disturbance_progress_delta", 0.001)
        self.domain_rand_frozen = False
        self.domain_rand_reward_ema = None
        self.domain_rand_best_reward_ema = -float("inf")
        self.domain_rand_recovery_ratio = getattr(dr, "recovery_ratio", 0.70)
        self.domain_rand_min_reward = getattr(dr, "min_reward_to_step", 12.0)
        self.domain_rand_step_interval = getattr(dr, "step_interval", 100)
        self.domain_rand_last_step_iter = -10**9
        self.domain_rand_ema_alpha = getattr(dr, "reward_ema_alpha", 0.05)
        self.domain_rand_reward_ema_hist = deque(maxlen=getattr(dr, "best_reward_window", 500))
        self.domain_rand_best_quantile = getattr(dr, "best_reward_quantile", 0.90)
        self.required_reward = 0.0

    def _advance_domain_rand_phase(self):
        if self.domain_rand_phase not in self.domain_rand_curriculum_phases:
            self.domain_rand_phase = "complete"
            return
        index = self.domain_rand_curriculum_phases.index(self.domain_rand_phase) + 1
        self.domain_rand_phase = (
            self.domain_rand_curriculum_phases[index]
            if index < len(self.domain_rand_curriculum_phases)
            else "complete"
        )
        self.domain_rand_best_reward_ema = self.domain_rand_reward_ema

    def _step_domian_rand(self, num_iters, mean_reward=None):
        """Advance the existing performance-gated UniFP curriculum state.

        Controller and disturbance stages take effect immediately. Isaac Gym
        mass/COM and rigid-property samples take effect on the next simulator
        construction because GPU actors cannot mutate those properties safely.
        """
        if not self.use_domainrand_curriculum:
            return
        if mean_reward is not None:
            mean_reward = float(mean_reward)
            if self.domain_rand_reward_ema is None:
                self.domain_rand_reward_ema = mean_reward
            else:
                alpha = self.domain_rand_ema_alpha
                self.domain_rand_reward_ema = (1.0 - alpha) * self.domain_rand_reward_ema + alpha * mean_reward
            self.domain_rand_reward_ema_hist.append(self.domain_rand_reward_ema)
            history = np.asarray(self.domain_rand_reward_ema_hist, dtype=np.float32)
            self.domain_rand_best_reward_ema = float(
                history.max() if len(history) < 10 else np.quantile(history, self.domain_rand_best_quantile)
            )
            self.required_reward = self.domain_rand_recovery_ratio * self.domain_rand_best_reward_ema
            can_step = (
                self.domain_rand_reward_ema >= self.required_reward
                and self.domain_rand_reward_ema >= self.domain_rand_min_reward
            )
        else:
            can_step = True
        enough_time = num_iters - self.domain_rand_last_step_iter >= self.domain_rand_step_interval
        if num_iters <= self.push_warmup_step or not can_step or not enough_time:
            self.domain_rand_frozen = True
            return

        self.domain_rand_frozen = False
        self.domain_rand_last_step_iter = num_iters
        progress_name = {
            "joint_dynamics": "domain_rand_joint_dynamics_progress",
            "mass_com": "domain_rand_mass_com_progress",
            "disturbance": "domain_rand_disturbance_progress",
        }.get(self.domain_rand_phase)
        delta_name = {
            "joint_dynamics": "domain_rand_joint_dynamics_delta",
            "mass_com": "domain_rand_mass_com_delta",
            "disturbance": "domain_rand_disturbance_delta",
        }.get(self.domain_rand_phase)
        if progress_name is not None:
            progress = min(1.0, getattr(self, progress_name) + getattr(self, delta_name))
            setattr(self, progress_name, progress)
            if progress >= 1.0:
                self._advance_domain_rand_phase()

        p_mass = self.domain_rand_mass_com_progress
        p_dist = self.domain_rand_disturbance_progress
        p_joint = self.domain_rand_joint_dynamics_progress
        self.mass_max_value = self.max_mass_bounds[0] + p_mass * self.mass_bounds_diff
        if self._has_gripper:
            self.grip_mass_max_value = self.grip_max_mass_bounds[0] + p_mass * self.grip_mass_bounds_diff
        self.com_delta_x_value = self.com_delta_x_bounds[0] + p_mass * self.com_delta_x_diff
        self.com_delta_y_value = self.com_delta_y_bounds[0] + p_mass * self.com_delta_y_diff
        self.com_delta_z_value = self.com_delta_z_bounds[0] + p_mass * self.com_delta_z_diff
        self.push_value = self.push_bounds[0] + p_dist * self.push_diff
        self.vert_value = self.vert_bounds[0] + p_dist * self.vert_diff
        self.wrench_value = self.wrench_bounds[0] + p_dist * self.wrench_diff
        self.joint_stiffness_bound_current = self.joint_stiffness_bounds_start + p_joint * self.joint_stiffness_range
        self.joint_damping_bound_current = self.joint_damping_bounds_start + p_joint * self.joint_damping_range
        self.joint_friction_bound_current = self.joint_friction_bounds_start + p_joint * self.joint_friction_range
        self.com_delta_z_val_bounds = (
            [-self._cfg.domain_rand.com_displacement_z_min, self.com_delta_z_value]
            if self.com_rand_z_positive
            else [-self.com_delta_z_value, self.com_delta_z_value]
        )

    @property
    def thigh_indices(self):
        return self._thigh_indices

    @property
    def thigh_pos(self):
        return self._thigh_pos

    @property
    def ee_quat(self):
        return self._ee_quat


class IsaacGymSimulatorB1Z1UniFP(_IsaacGymSimulatorB1Z1):
    """UniFP position-residual controller with 17 learned B1/Z1 actions."""

    def _compute_torques(self, actions):
        if actions.shape[-1] != self._num_learned_actions:
            raise RuntimeError(f"Expected {self._num_learned_actions} UniFP actions, got {actions.shape[-1]}")
        q = self.dof_pos
        qd = self.dof_vel
        p_gains = self._p_gains[self._dof_indices_tensor]
        d_gains = self._d_gains[self._dof_indices_tensor]
        target_offset = torch.zeros_like(q)
        target_offset[:, :self._num_learned_actions] = (
            self._motor_strength[:, :self._num_learned_actions]
            * actions
            * self._cfg.control.action_scale
        )
        self.feedback_torques = (
            self._kp_scale * p_gains * (self.default_dof_pos + target_offset - q)
            - self._kd_scale * d_gains * qd
        )
        if self.first_loop:
            self.first_loop = False
            self.first_loop_feedback = self.feedback_torques.clone()
        self.unclipped_torques = self.feedback_torques.clone()
        return self.feedback_torques


class IsaacGymSimulatorB1Z1PACTPos(_IsaacGymSimulatorB1Z1):
    """PACT-Pos position feedback controller with motor-strength scaling."""

    def _compute_torques(self, actions):
        if actions.shape[-1] != self._num_learned_actions:
            raise RuntimeError(f"Expected {self._num_learned_actions} PACT-Pos actions, got {actions.shape[-1]}")
        q = self.dof_pos
        qd = self.dof_vel
        p_gains = self._p_gains[self._dof_indices_tensor]
        d_gains = self._d_gains[self._dof_indices_tensor]
        target_offset = torch.zeros_like(q)
        target_offset[:, :self._num_learned_actions] = actions * self._cfg.control.action_scale
        self.feedback_torques = (
            self._kp_scale * p_gains * (self.default_dof_pos + target_offset - q)
            - self._kd_scale * d_gains * qd
        )
        self.feedforward_torques.zero_()
        self.combined_feedback_torques = self._motor_strength * self.feedback_torques
        self.combined_feedforward_torques.zero_()
        self.unclipped_torques = self.combined_feedback_torques.clone()
        return self.combined_feedback_torques


class IsaacGymSimulatorB1Z1PACT(_IsaacGymSimulatorB1Z1):
    """PACT controller combining 17 position and 17 direct-torque actions."""

    def _compute_torques(self, actions):
        expected = 2 * self._num_learned_actions
        if actions.shape[-1] != expected:
            raise RuntimeError(f"Expected {expected} coupled PACT actions, got {actions.shape[-1]}")
        position_actions = actions[:, :self._num_learned_actions]
        torque_actions = actions[:, self._num_learned_actions:]
        q = self.dof_pos
        qd = self.dof_vel
        p_gains = self._p_gains[self._dof_indices_tensor]
        d_gains = self._d_gains[self._dof_indices_tensor]
        target_offset = torch.zeros_like(q)
        target_offset[:, :self._num_learned_actions] = position_actions * self._cfg.control.action_scale
        self.feedback_torques = (
            self._kp_scale * p_gains * (self.default_dof_pos + target_offset - q)
            - self._kd_scale * d_gains * qd
        )
        self.feedforward_torques.zero_()
        self.feedforward_torques[:, :self._num_learned_actions] = (
            self._motor_strength[:, :self._num_learned_actions]
            * torque_actions
            * self._cfg.control.torque_scale
        )
        self.combined_feedback_torques = self.feedback_tau_weight * self.feedback_torques
        self.combined_feedforward_torques = self.feedforward_tau_weight * self.feedforward_torques
        torques = self.combined_feedback_torques + self.combined_feedforward_torques
        self.unclipped_torques = torques.clone()
        return torques
