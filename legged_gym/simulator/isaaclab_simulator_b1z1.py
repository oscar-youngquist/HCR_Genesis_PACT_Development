"""Isaac Lab B1/Z1 adapter preserving the existing UniFP and PACT contracts."""

from types import SimpleNamespace

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.math_utils import get_euler_xyz, quat_rotate_inverse, torch_rand_float
from legged_gym.utils.terrain import Terrain
from .isaaclab_simulator import IsaacLabSimulator
from .b1z1_lab_assets import prepare_lab_urdf
from legged_gym.envs.go2.go2_hard_pact.grf import GRFProcessingConfig, IntervalGRFProcessor
from .isaacgym_simulator_b1z1 import (
    _IsaacGymSimulatorB1Z1,
    IsaacGymSimulatorB1Z1UniFP,
    IsaacGymSimulatorB1Z1PACT,
    IsaacGymSimulatorB1Z1PACTPos,
)

GROUND_PATH = "/World/ground"


class _IsaacLabSimulatorB1Z1(IsaacLabSimulator):
    """Lab articulation and contact sensors behind the Isaac Gym B1Z1 API."""

    def __init__(self, cfg, sim_params, device="cuda:1", headless=False):
        params = dict(sim_params)
        params["dt"] = cfg.control.dt / cfg.control.decimation
        super().__init__(cfg, params, device, headless)
        self.first_loop = True
        self.first_loop_feedback = None
        if headless:
            # A STOP event cannot be resumed by a viewer in headless training.
            handle = getattr(self._sim, "_app_control_on_stop_handle", None)
            if handle is not None:
                handle.unsubscribe()
                self._sim._app_control_on_stop_handle = None

    def _parse_cfg(self):
        super()._parse_cfg()
        # Lab can update physical properties at reset; an enabled curriculum
        # must begin at its initial bounds, not Gym's immutable final bounds.
        active = self._cfg.domain_rand.use_domainrand_curriculum
        _IsaacGymSimulatorB1Z1._parse_b1z1_cfg(self, use_final_ranges=False if active else None)
        self._domain_rand_last_iteration = -1

    _init_domain_rand_curriculum_state = _IsaacGymSimulatorB1Z1._init_domain_rand_curriculum_state
    _advance_domain_rand_phase = _IsaacGymSimulatorB1Z1._advance_domain_rand_phase
    _update_domain_rand_bounds = _IsaacGymSimulatorB1Z1._update_domain_rand_bounds
    def _step_domian_rand(self, num_iters, mean_reward=None):
        """Retain B1Z1 reward gating, advancing at most once per PPO iteration."""
        if num_iters <= self._domain_rand_last_iteration:
            return
        self._domain_rand_last_iteration = num_iters
        _IsaacGymSimulatorB1Z1._step_domian_rand(self, num_iters, mean_reward)

    def domain_rand_curriculum_state_dict(self):
        """Preserve progression and reward gating across training resumes."""
        state = {name: value for name, value in vars(self).items()
                 if name.startswith("domain_rand_") and name != "domain_rand_reward_ema_hist"}
        state["reward_history"] = list(self.domain_rand_reward_ema_hist)
        state["last_iteration"] = self._domain_rand_last_iteration
        state["required_reward"] = self.required_reward
        return state

    def load_domain_rand_curriculum_state_dict(self, state):
        if not state:
            return  # Older checkpoints retain configured initial bounds.
        for name, value in state.items():
            if name.startswith("domain_rand_") and hasattr(self, name):
                setattr(self, name, value)
        self.domain_rand_reward_ema_hist.clear()
        self.domain_rand_reward_ema_hist.extend(state["reward_history"])
        self._domain_rand_last_iteration = state["last_iteration"]
        self.required_reward = state["required_reward"]
        if self.use_domainrand_curriculum:
            self._update_domain_rand_bounds()

    def _create_sim(self):
        # The existing Lab terrain helper already accepts a trimesh. Convert
        # the configured B1Z1 heightfield only at the physics boundary.
        if self._cfg.terrain.mesh_type != "heightfield":
            return super()._create_sim()
        from isaaclab.app import AppLauncher
        import carb
        self._app_launcher = AppLauncher(self._lab_launcher_args())
        import isaaclab.sim as sim_utils
        from isaacsim.core.utils.stage import get_current_stage
        from isaaclab.terrains.utils import create_prim_from_mesh

        physx = self._sim_params["physx"]
        sim_cfg = sim_utils.SimulationCfg(
            device=self._device, dt=self._sim_params["dt"],
            render_interval=self._cfg.control.decimation,
            physx=sim_utils.PhysxCfg(
                solver_type=physx["solver_type"],
                max_position_iteration_count=physx["num_position_iterations"],
                max_velocity_iteration_count=physx["num_velocity_iterations"],
                bounce_threshold_velocity=physx["bounce_threshold_velocity"],
                gpu_max_rigid_contact_count=physx["max_gpu_contact_pairs"],
            ),
        )
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._stage = get_current_stage()
        carb.settings.get_settings().set_bool("/app/runLoops/main/rateLimitEnabled", False)
        terrain_cfg = SimpleNamespace(**{
            name: getattr(self._cfg.terrain, name)
            for name in dir(self._cfg.terrain) if not name.startswith("_")
        })
        terrain_cfg.mesh_type = "trimesh"
        self._terrain = Terrain(terrain_cfg)
        # Terrain() produces the same height samples and origins for either
        # representation; only the collision representation changes here.
        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=self._cfg.terrain.static_friction,
            dynamic_friction=self._cfg.terrain.dynamic_friction,
            restitution=self._cfg.terrain.restitution,
        )
        create_prim_from_mesh(
            GROUND_PATH + "/terrain", self._terrain.terrain_mesh,
            physics_material=material,
            translation=(-self._cfg.terrain.border_size - self._cfg.terrain.horizontal_scale / 2,
                         -self._cfg.terrain.border_size - self._cfg.terrain.horizontal_scale / 2, 0.0),
        )
        self._height_samples = torch.as_tensor(self._terrain.heightsamples, device=self._device)
        if not self._headless:
            self._build_lights()
        terrain_cfg = self._cfg.terrain
        self._terrain_x_range = torch.tensor(
            [-terrain_cfg.border_size + 1.0,
             terrain_cfg.border_size + terrain_cfg.num_rows * terrain_cfg.terrain_length - 1.0],
            device=self._device,
        )
        self._terrain_y_range = torch.tensor(
            [-terrain_cfg.border_size + 1.0,
             terrain_cfg.border_size + terrain_cfg.num_cols * terrain_cfg.terrain_width - 1.0],
            device=self._device,
        )

    def _create_trimesh(self):
        import isaaclab.sim as sim_utils
        from isaaclab.terrains.utils import create_prim_from_mesh

        cfg = self._cfg.terrain
        create_prim_from_mesh(
            GROUND_PATH + "/terrain", self._terrain.terrain_mesh,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=cfg.static_friction,
                dynamic_friction=cfg.dynamic_friction,
                restitution=cfg.restitution,
            ),
            translation=(-cfg.border_size - cfg.horizontal_scale / 2,
                         -cfg.border_size - cfg.horizontal_scale / 2, 0.0),
        )
        self._height_samples = torch.as_tensor(self._terrain.heightsamples, device=self._device)

    def _create_envs(self):
        from isaacsim.core.cloner import Cloner
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from pxr import PhysxSchema

        cfg = self._cfg
        self._cloner = Cloner(self._stage)
        paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        self._stage.DefinePrim(paths[0], "Xform")
        spawn = sim_utils.UrdfFileCfg(
            asset_path=prepare_lab_urdf(getattr(cfg.asset, "isaacgym_file", cfg.asset.file).format(
                LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)),
            fix_base=cfg.asset.fix_base_link,
            merge_fixed_joints=cfg.asset.collapse_fixed_joints,
            replace_cylinders_with_capsules=cfg.asset.replace_cylinder_with_capsule,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=self._sim_params["physx"]["max_depenetration_velocity"],
                angular_damping=cfg.asset.angular_damping,
                linear_damping=cfg.asset.linear_damping,
                disable_gravity=cfg.asset.disable_gravity,
                max_linear_velocity=cfg.asset.max_linear_velocity,
                max_angular_velocity=cfg.asset.max_angular_velocity,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=cfg.asset.self_collisions,
                fix_root_link=cfg.asset.fix_base_link,
                solver_position_iteration_count=self._sim_params["physx"]["num_position_iterations"],
                solver_velocity_iteration_count=self._sim_params["physx"]["num_velocity_iterations"],
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
            activate_contact_sensors=True,
        )
        self._robot = Articulation(ArticulationCfg(
            prim_path=f"/World/envs/env_.*/{cfg.asset.name}", spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=cfg.init_state.pos,
                rot=(cfg.init_state.rot[3], *cfg.init_state.rot[:3]),
                joint_pos=cfg.init_state.default_joint_angles,
            ),
            actuators={"all": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=0.0, damping=0.0,
            )},
            soft_joint_pos_limit_factor=cfg.rewards.soft_dof_pos_limit,
        ))
        self._cloner.clone(
            source_prim_path=paths[0], prim_paths=paths,
            copy_from_source=False, replicate_physics=True,
            base_env_path="/World/envs", enable_env_ids=True,
        )
        self._contact_sensors = ContactSensor(ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/{cfg.asset.name}/.*",
            update_period=self._sim_params["dt"], history_length=2, debug_vis=False,
        ))
        scene_path = next(
            prim.GetPrimPath().pathString for prim in self._stage.Traverse()
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI)
        )
        self._cloner.filter_collisions(
            scene_path, "/World/collisions", paths, global_paths=[GROUND_PATH]
        )
        self._sim.reset()
        self._get_env_origins()
        self._dof_names = list(self._robot.joint_names)
        self._dof_indices = [self._dof_names.index(name) for name in cfg.asset.dof_names]
        self._num_dof = len(self._dof_names)
        self._num_bodies = len(self._robot.body_names)
        self._body_names = list(self._robot.body_names)
        self._feet_names = list(cfg.asset.foot_name)
        self._feet_indices = torch.tensor(
            [self._body_names.index(name) for name in self._feet_names],
            device=self._device, dtype=torch.long,
        )
        self._thigh_indices = torch.tensor(
            [self._body_names.index(name) for name in cfg.asset.thigh_name],
            device=self._device, dtype=torch.long,
        )
        self._gripper_index = self._body_names.index(cfg.asset.gripper_name)
        self._base_link_index = self._body_names.index(cfg.asset.base_link_name)
        self._resolve_contact_indices()
        self._termination_contact_indices = torch.tensor(
            [i for i, name in enumerate(self._body_names)
             if any(pattern in name for pattern in cfg.asset.terminate_after_contacts_on)],
            device=self._device, dtype=torch.long,
        )
        self._penalized_contact_indices = torch.tensor(
            [i for i, name in enumerate(self._body_names)
             if any(pattern in name for pattern in cfg.asset.penalize_contacts_on)],
            device=self._device, dtype=torch.long,
        )
        self._contact_state_link_indices = torch.tensor(
            [i for i, name in enumerate(self._body_names)
             if any(pattern in name for pattern in cfg.asset.contact_state_link_names)],
            device=self._device, dtype=torch.long,
        )
        self._arm_dof_cfg_ids = torch.tensor(
            [cfg.asset.dof_names.index(name) for name in (
                "z1_waist", "z1_shoulder", "z1_elbow", "z1_wrist_angle",
                "z1_forearm_roll", "z1_wrist_rotate")],
            device=self._device, dtype=torch.long,
        )
        self._dof_indices_tensor = torch.tensor(self._dof_indices, device=self._device)
        self._init_domain_params()

    def _resolve_contact_indices(self):
        """Keep sensor indices distinct from articulation-order contact buffers."""
        names = list(self._contact_sensors.body_names)
        if len(set(names)) != len(names) or any(name not in names for name in self._body_names):
            raise ValueError("B1Z1 contact sensor must uniquely cover every articulation body")
        self._contact_indices = torch.tensor(
            [names.index(name) for name in self._body_names],
            device=self._device, dtype=torch.long,
        )
        self._feet_contact_indices = self._contact_indices[self._feet_indices]

    def _init_domain_params(self):
        n, d = self._num_envs, len(self._dof_indices)
        device = self._device
        for name, width in (
            ("_friction_values", 1), ("_added_base_mass", 1),
            ("_added_gripper_mass", 1), ("_base_com_bias", 3),
            ("_rand_push_vels", 3), ("_rand_wrench_vels", 3),
            ("_joint_armature", 1), ("_joint_friction", 1),
            ("_joint_stiffness", 1), ("_joint_damping", 1),
        ):
            setattr(self, name, torch.zeros(n, width, device=device))
        self._kp_scale = torch.ones(n, d, device=device)
        self._kd_scale = torch.ones(n, d, device=device)
        self._motor_strength = torch.ones(n, d, device=device)

    def _randomize_physical_properties(self, env_ids):
        dr = self._cfg.domain_rand
        if dr.randomize_friction:
            self._randomize_friction(env_ids)
        if dr.randomize_base_mass or dr.randomize_gripper_mass:
            self._randomize_mass(env_ids)
        if dr.randomize_com_displacement:
            self._randomize_com_displacement(env_ids)
        for enabled, method in (
            (dr.randomize_joint_armature, self._randomize_joint_armature),
            (dr.randomize_joint_friction, self._randomize_joint_friction),
            (dr.randomize_joint_stiffness, self._randomize_joint_stiffness),
            (dr.randomize_joint_damping, self._randomize_joint_damping),
        ):
            if enabled:
                method(env_ids)

    def _randomize_friction(self, env_ids):
        dr = self._cfg.domain_rand
        values = torch_rand_float(*dr.friction_range, (len(env_ids), 1), self._device)
        self._friction_values[env_ids] = values
        props = self._robot.root_physx_view.get_material_properties()
        props[env_ids.cpu(), :, 0:2] = values.cpu().unsqueeze(-1)
        self._robot.root_physx_view.set_material_properties(
            props, torch.arange(self._num_envs, device="cpu")
        )

    def _randomize_mass(self, env_ids):
        view = self._robot.root_physx_view
        ids = env_ids.cpu()
        masses = view.get_masses()
        inertias = view.get_inertias()
        defaults = self._robot.data.default_mass.cpu()
        default_inertias = self._robot.data.default_inertia.cpu()
        for enabled, index, low, high, output in (
            (self._cfg.domain_rand.randomize_base_mass, self._base_link_index,
             self.mass_min, self.mass_max_value, self._added_base_mass),
            (self._cfg.domain_rand.randomize_gripper_mass, self._gripper_index,
             self.grip_mass_min, self.grip_mass_max_value, self._added_gripper_mass),
        ):
            if enabled:
                delta = torch_rand_float(low, high, (len(env_ids), 1), self._device)
                masses[ids, index] = (defaults[ids, index] + delta.cpu().flatten()).clamp_min(1e-4)
                output[env_ids] = (masses[ids, index] - defaults[ids, index]).to(self._device).unsqueeze(1)
                # Match Lab's recompute_inertia convention: fixed shape, uniform
                # density change. Start from defaults to avoid compounding resets.
                ratio = masses[ids, index] / defaults[ids, index]
                inertias[ids, index] = default_inertias[ids, index] * ratio.unsqueeze(-1)
        view.set_masses(masses, ids)
        view.set_inertias(inertias, ids)

    def _randomize_com_displacement(self, env_ids):
        ranges = [(-self.com_delta_x_value, self.com_delta_x_value),
                  (-self.com_delta_y_value, self.com_delta_y_value),
                  tuple(self.com_delta_z_val_bounds)]
        delta = torch.cat([
            torch_rand_float(low, high, (len(env_ids), 1), self._device)
            for low, high in ranges
        ], dim=1)
        self._base_com_bias[env_ids] = delta
        coms = self._robot.root_physx_view.get_coms()
        defaults = self._default_com_pos.cpu()
        coms[env_ids.cpu(), self._base_link_index, :3] = defaults[env_ids.cpu()] + delta.cpu()
        self._robot.root_physx_view.set_coms(coms, torch.arange(self._num_envs, device="cpu"))

    def _randomize_joint_armature(self, env_ids):
        values = torch_rand_float(*self._cfg.domain_rand.joint_armature_range,
                                  (len(env_ids), 1), self._device)
        self._joint_armature[env_ids] = values
        self._robot.write_joint_armature_to_sim(values.expand(-1, self._num_dof), env_ids=env_ids)

    def _randomize_joint_friction(self, env_ids):
        values = torch_rand_float(*self.joint_friction_bound_current,
                                  (len(env_ids), 1), self._device)
        self._joint_friction[env_ids] = values
        self._robot.write_joint_friction_coefficient_to_sim(
            values.expand(-1, self._num_dof), joint_ids=None, env_ids=env_ids
        )

    def _randomize_joint_stiffness(self, env_ids):
        values = torch_rand_float(*self.joint_stiffness_bound_current,
                                  (len(env_ids), 1), self._device)
        self._joint_stiffness[env_ids] = values
        self._robot.write_joint_stiffness_to_sim(values.expand(-1, self._num_dof), env_ids=env_ids)

    def _randomize_joint_damping(self, env_ids):
        values = torch_rand_float(*self.joint_damping_bound_current,
                                  (len(env_ids), 1), self._device)
        self._joint_damping[env_ids] = values
        self._robot.write_joint_damping_to_sim(values.expand(-1, self._num_dof), env_ids=env_ids)

    def _update_surrounding_heights(self):
        if self._cfg.terrain.mesh_type == "plane":
            self._measured_heights.zero_()
        else:
            super()._update_surrounding_heights()

    def _calc_terrain_info_around_feet(self):
        if self._cfg.terrain.mesh_type == "plane":
            self._height_around_feet.zero_()
            self._normal_vector_around_feet.zero_()
            self._normal_vector_around_feet[:, 2::3] = -1.0
        else:
            super()._calc_terrain_info_around_feet()

    def _randomize_pd_gain(self, env_ids):
        return _IsaacGymSimulatorB1Z1._randomize_pd_gain(self, env_ids)

    def _randomize_motor_strength(self, env_ids):
        return _IsaacGymSimulatorB1Z1._randomize_motor_strength(self, env_ids)

    def _init_buffers(self):
        n, device = self._num_envs, self._device
        self.common_step_counter = 0
        self._base_pos = torch.zeros(n, 3, device=device)
        self._base_quat = torch.zeros(n, 4, device=device)
        self._base_euler = torch.zeros(n, 3, device=device)
        self._base_lin_vel = torch.zeros(n, 3, device=device)
        self._base_ang_vel = torch.zeros(n, 3, device=device)
        self._projected_gravity = torch.zeros(n, 3, device=device)
        self._global_gravity = torch.tensor([0., 0., -1.], device=device).expand(n, -1)
        self._last_base_lin_vel = torch.zeros(n, 3, device=device)
        self._last_base_ang_vel = torch.zeros(n, 3, device=device)
        self._last_base_world_lin_vel = torch.zeros(n, 3, device=device)
        self._last_base_world_ang_vel = torch.zeros(n, 3, device=device)
        self._last_dof_vel = torch.zeros(n, self._num_dof, device=device)
        self._p_gains = torch.zeros(self._num_dof, device=device)
        self._d_gains = torch.zeros_like(self._p_gains)
        # PACT labels and the shared torque controller consume configured order.
        for i, name in enumerate(self._cfg.asset.dof_names):
            for key, value in self._cfg.control.stiffness.items():
                if key in name:
                    self._p_gains[i] = value
                    self._d_gains[i] = self._cfg.control.damping[key]
        self._torque_limits = self.torque_limits.clone()
        self._torques = torch.zeros(n, self._num_dof, device=device)
        self.unclipped_torques = torch.zeros_like(self._torques)
        self.executed_torques = torch.zeros_like(self._torques)
        self.feedback_torques = torch.zeros_like(self._torques)
        self.feedforward_torques = torch.zeros_like(self._torques)
        self.combined_feedback_torques = torch.zeros_like(self._torques)
        self.combined_feedforward_torques = torch.zeros_like(self._torques)
        self._dof_tau = torch.zeros_like(self._torques)
        self._grfs_buf = torch.zeros(n, self._grf_dim, device=device)
        self._configure_grf_processing()
        self._grf_processor = IntervalGRFProcessor(
            n, len(self._feet_names), device, self._grfs_buf.dtype,
            GRFProcessingConfig(
                vertical_deadband_n=self._grf_deadband,
                clip_min_n=self._grf_clip_min, clip_max_n=self._grf_clip_max,
                ema_alpha=self._grf_ema_alpha,
                contact_threshold_n=self._foot_contact_force_threshold,
            ),
        )
        # Keep existing B1Z1 diagnostics/labels as views of the shared processor.
        self._grfs_raw_buf = self._grf_processor.raw.flatten(1)
        self._grfs_deadband_buf = self._grf_processor.complete.flatten(1)
        self._grfs_clipped_buf = self._grf_processor.clipped.flatten(1)
        self._grfs_smoothed_buf = self._grf_processor.ema.flatten(1)
        self._grfs_interval_buf = self._grf_processor.interval_average.flatten(1)
        self._feet_pos = torch.zeros(n, 4, 3, device=device)
        self._feet_vel = torch.zeros_like(self._feet_pos)
        self._last_feet_vel = torch.zeros_like(self._feet_pos)
        self._thigh_pos = torch.zeros_like(self._feet_pos)
        self._ee_pos = torch.zeros(n, 3, device=device)
        self._ee_quat = torch.zeros(n, 4, device=device)
        self._ee_vel = torch.zeros(n, 3, device=device)
        self._link_contact_forces = torch.zeros(n, self._num_bodies, 3, device=device)
        self._base_force_world = torch.zeros(n, 3, device=device)
        self._ee_force_world = torch.zeros_like(self._base_force_world)
        self._base_torque_world = torch.zeros_like(self._base_force_world)
        self._external_force_world = torch.zeros(n, self._num_bodies, 3, device=device)
        self._external_torque_world = torch.zeros_like(self._external_force_world)
        self.feedback_tau_weight = torch.ones(n, 1, device=device)
        self.feedforward_tau_weight = torch.ones(n, 1, device=device)
        self.wrench_timeouts = torch_rand_float(self.wrench_timeout_min, self.wrench_timeout_max, (n, 1), device)
        self.push_timeouts = torch_rand_float(self.push_interval_min, self.push_interval_max, (n, 1), device)
        self.vert_timeouts = torch_rand_float(self.vert_interval_min, self.vert_interval_max, (n, 1), device)
        self._default_com_pos = self._robot.root_physx_view.get_coms()[:, self._base_link_index, :3].to(device).clone()
        self._randomize_physical_properties(torch.arange(n, device=device))
        self._init_height_points()
        self._measured_heights = torch.zeros(n, self._num_height_points, device=device)
        self._height_around_feet = torch.zeros(n, 4, 9, device=device)
        self._normal_vector_around_feet = torch.zeros(n, 12, device=device)
        ids = torch.arange(n, device=device)
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(ids)
        if self._cfg.domain_rand.randomize_motor_strength:
            self._randomize_motor_strength(ids)
        self.post_physics_step()

    def step(self, actions):
        self._last_base_lin_vel.copy_(self._base_lin_vel)
        self._last_base_ang_vel.copy_(self._base_ang_vel)
        self._last_base_world_lin_vel.copy_(self._base_world_lin_vel)
        self._last_base_world_ang_vel.copy_(self._base_world_ang_vel)
        self._last_feet_vel.copy_(self._feet_vel)
        # This buffer is permuted into configured order by last_dof_vel.
        self._last_dof_vel.copy_(self._robot.data.joint_vel)
        self.first_loop = True
        self._apply_external_forces()
        self._grf_processor.begin_interval()
        for _ in range(self._cfg.control.decimation):
            torque = self._compute_torques(actions)
            self.executed_torques = torch.clamp(torque, -1.1 * self.torque_limits, 1.1 * self.torque_limits)
            self._torques.copy_(self.executed_torques)
            self._robot.set_joint_effort_target(self.executed_torques, joint_ids=self._dof_indices)
            self._robot.write_data_to_sim()
            self._sim.step(render=False)
            self._robot.update(self._sim_params["dt"])
            self._contact_sensors.update(self._sim_params["dt"])
            # Sensor order is independent of articulation order. Read with
            # sensor indices exactly once, preserving configured FR/FL/RR/RL.
            self._grf_processor.update_substep(self._sensor_foot_forces_world())
        self._grf_processor.end_interval()
        self._grfs_buf.copy_(self._grfs_smoothed_buf)
        if not self._headless:
            self._sim.render()

    def post_physics_step(self):
        data = self._robot.data
        self._base_pos.copy_(data.root_link_pos_w)
        if self._cfg.terrain.mesh_type != "plane":
            self._check_base_pos_out_of_bound()
            self._base_pos.copy_(data.root_link_pos_w)
        self._base_quat.copy_(data.root_link_quat_w[:, (1, 2, 3, 0)])
        self._base_euler = get_euler_xyz(self._base_quat)
        self._base_world_lin_vel = data.root_link_lin_vel_w
        self._base_world_ang_vel = data.root_link_ang_vel_w
        self._base_lin_vel.copy_(quat_rotate_inverse(self._base_quat, self._base_world_lin_vel))
        self._base_ang_vel.copy_(quat_rotate_inverse(self._base_quat, self._base_world_ang_vel))
        self._projected_gravity.copy_(quat_rotate_inverse(self._base_quat, self._global_gravity))
        self._feet_pos = data.body_link_pos_w[:, self._feet_indices]
        self._feet_vel = data.body_link_vel_w[:, self._feet_indices, :3]
        self._thigh_pos = data.body_link_pos_w[:, self._thigh_indices]
        self._ee_pos = data.body_link_pos_w[:, self._gripper_index]
        self._ee_quat = data.body_link_quat_w[:, self._gripper_index, (1, 2, 3, 0)]
        self._ee_vel = data.body_link_vel_w[:, self._gripper_index, :3]
        self._link_contact_forces = self._contact_sensors.data.net_forces_w[:, self._contact_indices]
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = (
                self._link_contact_forces[:, self._contact_state_link_indices].norm(dim=-1)
                > self._foot_contact_force_threshold
            ).float()
        # GRFs were conditioned at physics rate in step(); do not EMA twice.
        # Projected joint forces are solver-reported; applied_torque is only
        # the command and would change force-manipulability reward semantics.
        self._dof_tau.copy_(
            self._robot.root_physx_view.get_dof_projected_joint_forces()
            .to(self._device)[:, self._dof_indices]
        )
        if self._cfg.terrain.measure_heights:
            self._update_surrounding_heights()
            if self._cfg.terrain.obtain_terrain_info_around_feet:
                self._calc_terrain_info_around_feet()
        self.common_step_counter += 1

    def reset_idx(self, env_ids):
        if env_ids.numel() == 0:
            return
        self._robot.reset(env_ids)
        self._contact_sensors.reset(env_ids)
        # Install current curriculum bounds only at episode boundaries.
        self._randomize_physical_properties(env_ids)
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(env_ids)
        if self._cfg.domain_rand.randomize_motor_strength:
            self._randomize_motor_strength(env_ids)
        self._last_dof_vel[env_ids] = 0
        self._last_base_lin_vel[env_ids] = 0
        self._last_base_ang_vel[env_ids] = 0
        self._last_base_world_lin_vel[env_ids] = 0
        self._last_base_world_ang_vel[env_ids] = 0
        self._last_feet_vel[env_ids] = 0
        self._dof_tau[env_ids] = 0
        self._reset_grf_buffer(env_ids)
        self._grf_processor.reset(env_ids)

    def _sensor_foot_forces_world(self):
        return self._contact_sensors.data.net_forces_w[:, self._feet_contact_indices]

    def get_grf_metrics(self):
        """Current control-interval force diagnostics, in world-frame Newtons."""
        metrics = {}
        for stage, forces in self._grf_processor.flattened_stages().items():
            if stage == "complete":
                continue  # Alias of deadbanded.
            feet = forces.reshape(self._num_envs, len(self._feet_names), 3)
            metrics[f"GRF/{stage}_norm_mean"] = feet.norm(dim=-1).mean()
            for index, name in enumerate(self._feet_names):
                metrics[f"GRF/{stage}_{name}_fz"] = feet[:, index, 2].mean()
        metrics["GRF/contact_fraction"] = self._grf_processor.contacts.float().mean()
        return metrics

    def reset_dofs(self, env_ids, dof_pos, dof_vel):
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, self._dof_indices, env_ids)

    def apply_ee_force(self, force_world):
        self._ee_force_world.copy_(force_world)

    def apply_base_force(self, force_world):
        self._base_force_world.copy_(force_world)

    def apply_base_torque(self, torque_world):
        self._base_torque_world.copy_(torque_world)

    def _apply_external_forces(self):
        commands = self._cfg.commands
        self._external_force_world.zero_()
        self._external_torque_world.zero_()
        if commands.push_gripper_stators and commands.apply_ee_external_forces:
            self._external_force_world[:, self._gripper_index] = self._ee_force_world
        if commands.push_robot_base and commands.apply_base_external_forces:
            self._external_force_world[:, self._base_link_index] = self._base_force_world
        if commands.push_robot_base and getattr(commands, "apply_base_external_torques", False):
            self._external_torque_world[:, self._base_link_index] = self._base_torque_world
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._external_force_world,
            torques=self._external_torque_world,
            is_global=True,
        )

    def push_robots(self):
        push_steps = torch.clamp((self.push_timeouts / self._control_dt).long().flatten(), min=1)
        wrench_steps = torch.clamp((self.wrench_timeouts / self._control_dt).long().flatten(), min=1)
        vert_steps = torch.clamp((self.vert_timeouts / self._control_dt).long().flatten(), min=1)
        push = (self.common_step_counter % push_steps) == 0
        wrench = (self.common_step_counter % wrench_steps) == 0
        vertical = (self.common_step_counter % vert_steps) == 0
        if not torch.any(push | wrench | vertical):
            return
        velocity = self._robot.data.root_link_vel_w.clone()
        if torch.any(push):
            value = torch_rand_float(-self.push_value, self.push_value,
                                     (int(push.sum()), 2), self._device)
            velocity[push, :2] += value
            self._rand_push_vels[push, :2] = value
        if torch.any(wrench):
            value = torch_rand_float(-self.wrench_value, self.wrench_value,
                                     (int(wrench.sum()), 3), self._device)
            velocity[wrench, 3:6] += value
            self._rand_wrench_vels[wrench] = value
        if torch.any(vertical):
            value = torch_rand_float(-self.vert_value, 0.0,
                                     (int(vertical.sum()), 1), self._device)
            velocity[vertical, 2:3] += value
            self._rand_push_vels[vertical, 2:3] = value
        self._robot.write_root_link_velocity_to_sim(velocity)

    @property
    def dof_pos(self):
        return self._robot.data.joint_pos[:, self._dof_indices]

    @property
    def dof_vel(self):
        return self._robot.data.joint_vel[:, self._dof_indices]

    @property
    def last_dof_vel(self):
        return self._last_dof_vel[:, self._dof_indices]

    @property
    def default_dof_pos(self):
        # The environment repeats this single nominal pose for reset env_ids.
        return self._robot.data.default_joint_pos[0, self._dof_indices].unsqueeze(0)

    @property
    def torques(self):
        return self._torques

    @property
    def torque_limits(self):
        if not hasattr(self, "_lab_torque_limits"):
            limits = self._robot.data.joint_effort_limits[0, self._dof_indices]
            if not torch.all(torch.isfinite(limits) & (limits > 0)):
                raise ValueError("B1Z1 IsaacLab requires finite positive joint effort limits")
            self._lab_torque_limits = limits.detach().clone()
        return self._lab_torque_limits

    @property
    def link_contact_forces(self):
        return self._link_contact_forces

    @property
    def thigh_pos(self):
        return self._thigh_pos

    @property
    def ee_quat(self):
        return self._ee_quat


class IsaacLabSimulatorB1Z1UniFP(_IsaacLabSimulatorB1Z1):
    _compute_torques = IsaacGymSimulatorB1Z1UniFP._compute_torques


class IsaacLabSimulatorB1Z1PACTPos(_IsaacLabSimulatorB1Z1):
    _compute_torques = IsaacGymSimulatorB1Z1PACTPos._compute_torques


class IsaacLabSimulatorB1Z1PACT(_IsaacLabSimulatorB1Z1):
    _compute_torques = IsaacGymSimulatorB1Z1PACT._compute_torques
