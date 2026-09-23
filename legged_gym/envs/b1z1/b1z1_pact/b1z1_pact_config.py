import numpy as np

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class B1Z1PACTCfg(LeggedRobotCfg):
    seed = 1                                                                            # Random seed for reproducible initialization.

    class env:
        num_envs = 1024                                                                # Parallel simulation instances.
        # num_envs = 5120                                                                  # Parallel simulation instances.
        # 2 body-orientation + 3 angular velocity + 17 joint positions +
        # 17 joint velocities + 34 coupled PACT actions + 6 commands. EE pose
        # is estimated from history instead of exposed through an FK error.
        num_observations = 81
        # Both variants share the same critic state. Coupled PACT's additional
        # Pinocchio packet is stored separately by the rollout storage.
        num_privileged_force_obs = 21                                                   # GRFs (12), torso wrench (6), and EE force (3).
        privileged_force_start = 23                                                     # Start index of the critic's normalized force block.
        num_critic_state_obs = 213 + num_privileged_force_obs + 17 - 19                 # Single-frame critic state width, excluding terrain.
        num_height_obs = 187                                                            # Terrain-height samples supplied to the critic.
        num_privileged_obs = num_critic_state_obs + num_height_obs                      # Total single-frame critic input width.
        # Reconstruct only the non-terrain state. Height samples remain
        # privileged critic inputs but are not an encoder/decoder objective.
        num_privileged_recon_obs = num_critic_state_obs
        num_priv_stack = 5                                                              # Number of stacked critic frames.
        # Base velocity (3), spherical EE pose (3), base wrench (6), EE force
        # (3), foot contacts (4), and terrain-relative foot heights (4).
        num_explicit_recon_obs = 23
        assert privileged_force_start == num_explicit_recon_obs
        num_pred_obs = 23                                                               # Explicit prediction width retained for interface compatibility.
        num_actions = 17                                                                # Learned joints per action head.
        num_policy_actions = 34                                                         # Position and torque outputs combined.
        num_gripper_joints = 2                                                          # Additional joints held by the default controller.
        num_obs_hist = 25                                                               # Proprioceptive history length for the context encoder.
        env_spacing = 0.5                                                               # Environment placement spacing [m].
        episode_length_s = 20                                                           # Maximum episode duration [s].
        grf_dim = 12                                                                    # Flattened four-foot XYZ force width.
        whole_body_dim = 25                                                             # Floating-base (6) plus joint (19) dynamics coordinates.
        fail_to_terminal_time_s = 0.0                                                   # Delay between failure detection and termination [s].
        send_timeouts = True                                                            # Report timeouts for value-function bootstrapping.
        debug = False                                                                   # Enable environment debugging.
        debug_viz = False                                                               # Enable debug rendering.
        debug_draw_height_points_around_base = False                                    # Draw torso terrain probes.
        debug_draw_height_points_around_feet = False                                    # Draw foot terrain probes.
        debug_draw_terrain_height_points = False                                        # Draw terrain-height samples.
        render_ee_goal_debug = False                                                    # Draw desired EE target markers.
        render_ee_frame_debug = False                                                   # Draw EE frame markers.

    class goal_ee:
        num_commands = 3
        max_ee_force_offset = 0.05                                                      # Bound force-adjusted target displacement [m].
        project_force_adjusted_ee_target = True                                         # Project force-adjusted targets into the allowed workspace.
        force_target_radius_limits = [0.30, 0.90]                                       # Allowed projected EE radius range [m].
        force_target_projection_samples = 21                                            # Samples used when projecting an adjusted target.
        traj_time = [1.875, 5.625]                                                      # EE target transition duration range [s].
        hold_time = [0.5, 2.0]                                                          # EE target hold duration range [s].
        command_mode = "sphere"                                                         # EE target coordinate representation.
        collision_upper_limits = [0.1, 0.2, -0.05]                                      # Upper corner of the EE trajectory exclusion box [m].
        collision_lower_limits = [-0.9, -0.2, -0.7]                                     # Lower corner of the EE trajectory exclusion box [m].
        underground_limit = -0.7                                                        # Minimum allowed EE trajectory height [m].
        num_collision_check_samples = 10                                                # Samples checked along each candidate EE trajectory.
        arm_induced_pitch = 0.38                                                        # Reference pitch correction associated with the arm [rad].

        class sphere_center:
            # Genesis URDF z1_waist origin: base_static_joint [0.3, 0, 0.09]
            # plus z1_waist joint [0, 0, 0.0585].
            x_offset = 0.3
            y_offset = 0.0
            # The 0.55 term is the configured nominal B1 base height; 0.1485
            # is the Z1 waist height in the base frame after the lowered mount.
            z_invariant_offset = 0.55 + 0.1485

        class ranges:
            init_pos_start = [0.5, np.pi / 8, 0.0]                                      # Initial spherical trajectory start [radius, pitch, yaw].
            init_pos_end = [0.7, 0.0, 0.0]                                              # Initial spherical trajectory end [radius, pitch, yaw].
            pos_l = [0.40, 0.90]                                                        # Sampled EE radius limits [m].
            pos_p = [-1.0 * np.pi / 2.5, np.pi / 3.0]                                   # Sampled EE spherical pitch limits [rad].
            pos_y = [-1.2, 1.2]                                                         # Sampled EE spherical yaw limits [rad].
            delta_orn_r = [-0.5, 0.5]                                                   # EE orientation roll perturbation range [rad].
            delta_orn_p = [-0.5, 0.5]                                                   # EE orientation pitch perturbation range [rad].
            delta_orn_y = [-0.5, 0.5]                                                   # EE orientation yaw perturbation range [rad].

        sphere_error_scale = [1.0, 1.0, 1.0]                                            # Per-coordinate spherical tracking-error weights.
        orn_error_scale = [1.0, 1.0, 1.0]                                               # Per-axis orientation-error weights.
        debug_tcp_from_link06_offset = [0.186, 0.0, 0.0]                                # Debug TCP offset from link06 [m].

    class init_state:
        pos = [0.0, 0.0, 0.6]
        rot = [0.0, 0.0, 0.0, 1.0]                                                      # Initial root quaternion in xyzw order.
        lin_vel = [0.0, 0.0, 0.0]                                                       # Initial root linear velocity [m/s].
        ang_vel = [0.0, 0.0, 0.0]                                                       # Initial root angular velocity [rad/s].
        roll_random_scale = 0.0                                                         # Initial roll perturbation scale.
        pitch_random_scale = 0.0                                                        # Initial pitch perturbation scale.
        yaw_random_scale = 0.0                                                          # Initial yaw perturbation scale.
        default_joint_angles = {                                                        # Nominal joint positions and residual-action reference [rad].
            "FR_hip_joint": -0.15,
            "FR_thigh_joint": 0.67,
            "FR_calf_joint": -1.32,

            "FL_hip_joint": 0.15,
            "FL_thigh_joint": 0.67,
            "FL_calf_joint": -1.32,

            "RR_hip_joint": -0.15,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.32,

            "RL_hip_joint": 0.15,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.32,

            "z1_waist": 0.0,
            "z1_shoulder": 1.48,
            "z1_elbow": -0.63,
            "z1_wrist_angle": -0.84,
            "z1_forearm_roll": 0.0,
            "z1_wrist_rotate": 1.57,
            "z1_jointGripper": -0.785,
        }
        yaw_angle_range = [0.0, 3.14]                                                   # Reset heading range [rad].
        rand_yaw_range = np.pi / 2                                                      # Reset yaw perturbation bound [rad].
        origin_perturb_range = 0.5                                                      # Reset position perturbation bound [m].
        init_vel_perturb_range = 0.1                                                    # Reset velocity perturbation bound.
        leg_dof_pos_perturb_range = [-0.15, 0.15]                                       # Leg reset joint-position perturbations [rad].
        arm_dof_pos_perturb_range = [-0.1, 0.1]                                         # Arm reset joint-position perturbations [rad].

    class asset:
        name = "b1z1"
        # Genesis robot asset path.
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/b1z1_current/urdf/b1z1_genesis.urdf"
        # Isaac Gym/Lab robot asset path.
        isaacgym_file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/b1z1_current/urdf/b1z1.urdf"
        base_name = "trunk"                                                             # Base-body name used by asset lookup.
        base_mass_name = "trunk"                                                        # Body selected for added torso mass.
        base_com_name = "trunk"                                                         # Body selected for torso CoM displacement.
        dof_names = [                                                                   # Canonical joint order; simulator indices are resolved by name.
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "z1_waist",
            "z1_shoulder",
            "z1_elbow",
            "z1_wrist_angle",
            "z1_forearm_roll",
            "z1_wrist_rotate",
            "z1_jointGripper",
        ]
        foot_name = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]                        # Canonical contact-force order: FR, FL, RR, RL.
        thigh_name = ["FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"]                   # Thigh links used for geometry and diagnostics.
        gripper_name = "ee_gripper_link"                                                # End-effector reference link.
        penalize_contacts_on = ["trunk", "thigh", "hip", "calf"]                        # Link-name patterns for collision penalties.
        terminate_after_contacts_on = []                                                # Link-name patterns for contact termination.
        # Fixed links that must remain available after import.
        links_to_keep = ["FR_foot", "FL_foot", "RR_foot", "RL_foot", "ee_gripper_link"]
        self_collisions = True                                                          # Enable robot self-collisions; backend import semantics apply.
        flip_visual_attachments = False                                                 # Asset visual-frame compatibility option.
        fix_base_link = False                                                           # Fix the floating base when enabled.
        obtain_link_contact_states = True                                               # Collect link-level contact state.
        # Link patterns included in contact-state lookup.
        contact_state_link_names = ["thigh", "calf", "foot", "trunk", "ee_gripper_link"]
        base_link_name = "trunk"                                                        # Reference torso link.
        disable_gravity = False                                                         # Disable gravity on the robot asset.
        collapse_fixed_joints = True                                                    # Merge fixed joints during asset import.
        default_dof_drive_mode = 3                                                      # Isaac Gym drive mode; 3 selects effort.
        replace_cylinder_with_capsule = False                                           # Use capsule collision shapes for cylinders.
        density = 0.001                                                                 # Fallback asset density for importer-generated inertias.
        angular_damping = 0.0                                                           # Asset-level angular damping.
        linear_damping = 0.0                                                            # Asset-level linear damping.
        max_angular_velocity = 1000.0                                                   # Asset angular-velocity cap [rad/s].
        max_linear_velocity = 1000.0                                                    # Asset linear-velocity cap [m/s].
        armature = 0.0                                                                  # Nominal added joint inertia.

        thickness = 0.01                                                                # Asset collision thickness [m].
        dof_vel_limits = []                                                             # Optional joint-velocity limit overrides.

        abad_link_length = 0.12675                                                      # Hip abduction link length [m].
        hip_link_length = 0.35                                                          # Upper-leg length [m].
        knee_link_length = 0.35                                                         # Lower-leg length [m].
        knee_link_y_offset = 0.0                                                        # Lower-leg lateral offset [m].
        side_signs = [-1.0, 1.0, -1.0, 1.0]                                             # FR, FL, RR, RL

    class terrain:
        mesh_type = "trimesh"                                                           # Terrain collision representation.
        simplify_mesh = True                                                            # Simplify generated terrain meshes.
        plane_length = 200.0                                                            # Flat-plane extent [m].
        horizontal_scale = 0.1                                                          # Terrain grid spacing [m].
        vertical_scale = 0.005                                                          # Heightfield quantization [m].
        static_friction = 1.0                                                           # Nominal static ground friction.
        dynamic_friction = 1.0                                                          # Nominal sliding ground friction.
        restitution = 0.0                                                               # Ground coefficient of restitution.
        border_size = 5.0                                                               # Terrain border width [m].
        border_height = 1.0                                                             # Terrain border height [m].
        curriculum = False
        obtain_terrain_info_around_feet = True                                          # Query terrain around each foot.
        measure_heights = True                                                          # Provide terrain-height observations.
        # Forward offsets of torso height probes [m].
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # Lateral offsets of torso height probes [m].
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False                                                                # Use an explicitly selected terrain generator.
        terrain_kwargs = None                                                           # Arguments for the selected terrain generator.
        max_init_terrain_level = 2                                                      # Highest terrain level used at initialization.
        terrain_length = 8.0                                                            # Individual terrain tile length [m].
        terrain_width = 8.0                                                             # Individual terrain tile width [m].
        platform_size = 4.0                                                             # Flat platform size within a terrain tile [m].
        num_rows = 10                                                                   # Terrain difficulty levels.
        num_cols = 20                                                                   # Terrain columns/types.
        terrain_proportions = [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.0, 0.0, 0.0]  # Sampling proportions in the terrain generator's type order.
        terrain_curriculum_difficulty = {                                               # Expressions mapping difficulty to terrain geometry.
            "slope": "difficulty * 0.4",
            "step_height": "0.04 + 0.16 * difficulty",
            "discrete_height": "0.04 + 0.16 * difficulty",
            "stepping_stones_params": {
                "stone_length": "1.5 * (1.05 - difficulty)",
                "stone_width": "1.5 * (1.05 - difficulty)",
                "stone_distance_x": "0.05 if difficulty == 0 else 0.1",
                "stone_distance_y": "0.05 if difficulty == 0 else 0.1",
                "max_height": "0.0",
            },
            "gap_size": "difficulty",
            "pit_depth": "0.3 * difficulty",
        }
        slope_treshold = 0.75                                                           # Slope cutoff for vertical-face mesh correction.

    class sim:
        suppress_backend_warnings = True                                                # Suppress selected known backend startup warnings.
        dt = 0.002
        substeps = 1                                                                    # Physics substeps within a simulation step.
        max_collision_pairs = 100                                                       # Backend collision-pair capacity.
        IK_max_targets = 2                                                              # Backend inverse-kinematics target capacity.
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1                                                                     # Isaac Gym up-axis selector; 1 means Z.
        use_gpu_pipeline = True                                                         # Keep supported simulator tensors on the GPU.

        class grf:
            use_substep_filtering = False                                               # IsaacLab: False applies deadband/clipping/EMA once per control step.
            # Per-foot force conditioning in physical Newtons. The vertical
            # component gates the whole XYZ vector before clipping and EMA.
            deadband = 15.0                                                             # Reject an entire foot force when vertical force is below this [N].
            clip_min = -2000.0                                                          # Lower per-component GRF bound [N].
            clip_max = 2000.0                                                           # Upper per-component GRF bound [N].
            ema_alpha = 0.20                                                            # Weight of the latest conditioned GRF sample.
            contact_threshold = 40.0                                                    # Force threshold for binary foot contact [N].

        class physx:
            use_gpu = True                                                              # Enable GPU PhysX.
            num_subscenes = 0                                                           # PhysX subscene partition count.
            num_threads = 10                                                            # CPU physics worker count.
            solver_type = 1                                                             # PhysX solver selector; 1 means TGS.
            num_position_iterations = 4                                                 # Position-constraint solver iterations.
            num_velocity_iterations = 0                                                 # Velocity-constraint solver iterations.
            contact_offset = 0.01                                                       # Contact generation distance [m].
            rest_offset = 0.0                                                           # Resting contact separation [m].
            bounce_threshold_velocity = 0.5                                             # Minimum impact speed for restitution [m/s].
            max_depenetration_velocity = 1.0                                            # Maximum overlap-correction speed [m/s].
            max_gpu_contact_pairs = 2**23                                               # GPU contact-pair capacity.
            default_buffer_size_multiplier = 5                                          # PhysX contact-buffer allocation multiplier.
            contact_collection = 2                                                      # Isaac Gym contact collection mode; 2 collects all substeps.

    class control:
        # Coupled PACT control: the actor emits both position and feedforward
        # torque branches. The simulator combines them before clipping.
        control_type = "P"
        # stiffness = {                                                                   # Joint-name PD proportional gains [N m/rad].
        #     "hip": 250.0,
        #     "thigh": 250.0,
        #     "calf": 400.0,
        #     "z1_waist": 64.0,
        #     "z1_shoulder": 128.0,
        #     "z1_elbow": 64.0,
        #     "z1_wrist_angle": 64.0,
        #     "z1_forearm_roll": 64.0,
        #     "z1_wrist_rotate": 64.0,
        #     "z1_jointGripper": 64.0,
        # }
        # damping = {                                                                     # Joint-name PD derivative gains [N m s/rad].
        #     "hip": 6.25,
        #     "thigh": 6.25,
        #     "calf": 10.0,
        #     "z1_waist": 1.5,
        #     "z1_shoulder": 3.0,
        #     "z1_elbow": 1.5,
        #     "z1_wrist_angle": 1.5,
        #     "z1_forearm_roll": 1.5,
        #     "z1_wrist_rotate": 1.5,
        #     "z1_jointGripper": 1.5,
        # }

        stiffness = {"joint":100.0, "z1": 30.0,}
        damping = {"joint": 5.0,"z1": 0.70,}

        action_scale = 0.25                                                             # Convert normalized position actions to joint offsets [rad].
        torque_scale = 30.0                                                             # Convert normalized feedforward actions to torque [N m].
        torque_scale_overrides = {                                                      # Exact learned-joint overrides [N m/action].
            "z1_waist": 10.0,
            "z1_shoulder": 10.0,
            "z1_elbow": 10.0,
            "z1_wrist_angle": 10.0,
            "z1_forearm_roll": 10.0,
        }
        dt = 0.02
        decimation = 4

        use_tradeoff_curriculum = False                                                 # Schedule feedback/feedforward torque-branch weights.
        tradeoff_init_weights = [0.40, 1.60]                                            # Initial [feedback, feedforward] torque weights.
        tradeoff_final_weights = [1.0, 1.0]                                             # Final [feedback, feedforward] torque weights.
        tradeoff_steps = 10                                                             # Number of torque-weight curriculum stages.
        tradeoff_threshold = 0.70                                                       # Performance threshold for advancing torque weights.

        # This is not a curriculum: each reset slightly biases the coupled
        # output toward either feedback or feedforward torque. A subset stays
        # at the clean 1:1 contribution used by PACT-Pos pretraining.
        randomize_pact_weights = True                                                   # Randomize feedback/feedforward balance at reset.
        pact_weight_bias_min = 0.0                                                      # Minimum sampled branch-weight bias.
        pact_weight_bias_max = 0.20                                                     # Maximum sampled branch-weight bias.
        pact_balanced_prob = 0.25                                                       # Probability of keeping a balanced 1:1 torque split.

    class commands:
        curriculum = False
        max_curriculum = 0.8                                                            # Maximum command-curriculum range.
        # UniFP convention inside PACT training: the last three slots retain
        # the yaw-aligned spherical EE target, but all force-command slots are
        # removed.
        num_commands = 6

        resampling_time = 10.0                                                          # Base command resampling period [s].

        heading_command = False                                                         # Convert desired heading into yaw-rate commands.

        curriculum_threshold = 0.8                                                      # Performance threshold for command expansion.

        ang_vel_yaw_clip = 0.1                                                          # Yaw-command dead zone.
        ang_vel_pitch_clip = 0.5                                                        # Pitch-command dead zone where supported.

        lin_vel_x_clip = 0.05                                                           # Forward-command dead zone [m/s].
        lin_vel_y_clip = 0.05                                                           # Lateral-command dead zone [m/s].

        zero_vel_cmd_prob = 0.2                                                         # Standing-command probability before force-stage activation.
        zero_vel_cmd_prob_after_force = 0.6                                             # Standing-command probability after force-stage activation.

        # Shared B1Z1 schedule. PACT has no force-command observation channel,
        # but retains the common command stage before disturbances are enabled.
        force_curriculum_command_start_env_step = 0  # Start of the shared command-force stage. Units: completed per-environment control steps.
        force_curriculum_command_ramp_env_steps = 0  # Command-force ramp duration [completed env steps]. Units: completed per-environment control steps.
        force_curriculum_gate_start_env_step = 192000  # Earliest env step for the external-force performance gate. Units: completed per-environment control steps.
        force_curriculum_external_ramp_env_steps = 192000  # External-force ramp duration after activation [completed env steps]. Units: completed per-environment control steps.



        force_curriculum_ee_l1_threshold = 0.25                                         # Maximum EE tracking error for force-stage advancement.
        force_curriculum_roll_termination_threshold = 0.05                              # Maximum roll-termination rate for advancement.
        force_curriculum_episode_length_threshold = 950.0                               # Minimum episode length for advancement [control steps].
        force_curriculum_gate_patience_env_steps = 9600  # Consecutive qualifying env steps before advancement. Units: completed per-environment control steps.
        force_curriculum_metric_ema_alpha = 0.05                                        # New-sample weight for force-curriculum metrics.
        force_curriculum_use_latest_start_fallback = True                               # Allow time-based activation if the performance gate stalls.
        force_curriculum_latest_start_env_step = 192000  # Latest allowed external-force start env step. Units: completed per-environment control steps.


        push_gripper_stators = True                                                     # Enable the EE disturbance-event scheduler.
        apply_ee_external_forces = True                                                 # Actually apply scheduled EE forces.
        push_gripper_interval_s_ext = [3.5, 9.0]                                        # EE force-event interval range [s].
        push_gripper_duration_s_ext = [1.0, 3.0]                                        # EE force-event duration range [s].
        gripper_forced_prob_ext = 0.8                                                   # Probability of an active EE force event.

        max_push_force_xyz_gripper_ext = [-50.0, 50.0]                                  # Full-strength per-axis EE force range [N].
        randomize_gripper_force_gains = False                                           # Randomize EE force-feedback gains.
        gripper_force_kp_range = [200.0, 200.0]                                         # EE force proportional-gain range.
        gripper_force_kd_range = [3.0, 3.0]                                             # EE force derivative-gain range.
        gripper_prop_kd = 0.1                                                           # EE force derivative-gain proportional factor.
        settling_time_force_gripper_s = 1.0                                             # EE force settling/ramp timescale [s].

        push_robot_base = True                                                          # Enable the base disturbance-event scheduler.
        apply_base_external_forces = True                                               # Actually apply scheduled torso forces.
        push_base_interval_s_ext = [6.0, 12.0]                                          # Torso force-event interval range [s].
        push_base_duration_s_ext = [1.0, 3.0]                                           # Torso force-event duration range [s].
        base_forced_prob_ext = 0.8                                                      # Probability of an active torso force event.
        max_push_force_xyz_base_ext = [-20.0, 20.0]                                     # Full-strength per-axis torso force range [N].
        apply_base_external_torques = True                                              # Apply scheduled torso moments.
        base_torque_forced_prob_ext = 0.8                                               # Probability of an active torso torque event.
        # The shared force curriculum scales this world-frame moment range.
        max_push_torque_xyz_base_ext = [-10.0, 10.0]                                    # Full-strength per-axis torso moment range [N m].
        randomize_base_force_gains = False                                              # Randomize torso force-feedback gains.
        base_force_kp_range = [200.0, 200.0]                                            # Torso force proportional-gain range.
        base_force_kd_range = [200.0, 200.0]                                            # Torso force derivative-gain range.
        base_prop_kd = 0.1                                                              # Torso force derivative-gain proportional factor.
        force_z_base_ext_scale = 0.1                                                    # Extra scale on vertical torso disturbances.
        settling_time_force_base_s = 3.0                                                # Torso force settling/ramp timescale [s].
        # The impedance relation is a reward only. It never changes commands.

        class ranges:
            lin_vel_x = [-0.8, 0.8]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class normalization:
        class obs_scales:
            lin_vel = 2.0                                                               # Initial root linear velocity [m/s].
            ang_vel = 0.25                                                              # Initial root angular velocity [rad/s].
            dof_pos = 1.0
            dof_vel = 0.05
            grf = 0.001
            height_measurements = 5.0
            ee_sphe_radius_cmd = 0.5
            ee_sphe_pitch_cmd = 1.0
            ee_sphe_yaw_cmd = 1.3
            ee_force = 0.01
            base_force = 0.01
        clip_observations = 100.0                                                       # Symmetric bound on normalized observations.
        clip_actions = 50.0                                                             # Symmetric bound on normalized policy actions.

    class domain_rand:
        use_domainrand_curriculum = False                                               # Progressively expand supported randomization bounds.
        # Isaac Gym must choose immutable physical randomization ranges when
        # actors are built. False uses the curriculum starts; True uses ends.
        isaacgym_use_final_domain_rand_ranges = True
        randomize_friction = True                                                       # Randomize friction.
        friction_range = [0.3, 2.0]                                                     # Ground/robot friction randomization bounds.

        randomize_base_mass = True                                                      # Randomize base mass.
        added_mass_min = -2.0                                                           # Minimum added torso mass [kg].
        min_added_mass_max = 5.0                                                        # Initial upper bound on added torso mass [kg].
        max_added_mass_max = 10.0                                                       # Final upper bound on added torso mass [kg].

        randomize_gripper_mass = True                                                   # Randomize gripper mass.
        gripper_mass_min = -0.01                                                        # Minimum added gripper mass [kg].
        min_gripper_added_mass_max = 0.05                                                # Initial upper bound on added gripper mass [kg].
        max_gripper_added_mass_max = 0.10                                               # Final upper bound on added gripper mass [kg].

        randomize_com_displacement = True                                               # Randomize com displacement.
        com_rand_z_positive = False                                                     # Restrict sampled vertical CoM shifts to positive values.
        com_displacement_x_min = 0.15
        com_displacement_x_max = 0.15

        com_displacement_y_min = 0.15
        com_displacement_y_max = 0.15

        com_displacement_z_min = 0.15
        com_displacement_z_min_pos = 0.15
        com_displacement_z_max = 0.15

        push_robots = True                                                              # Generic push settings; separate from commands' sustained force events.
        push_interval_s = 8.0

        push_interval_min = 5.0
        push_interval_max = 15.0

        max_push_vel_xy = 0.8
        min_push_vel_xy = 0.2

        max_vertical_push = 0.10
        min_vertical_push = 0.0
        vert_interval_min = 5.0
        vert_interval_max = 15.0

        max_push_torque = 0.50
        min_push_torque = 0.0
        wrench_timeout_min = 5.0
        wrench_timeout_max = 15.0

        # Delay is tracked by source-action replay; BARD uses measured interval torque.
        randomize_ctrl_delay = True                                                     # Randomize ctrl delay.
        ctrl_delay_step_range = [0, 1]                                                  # Action delay bounds [control steps].

        randomize_pd_gain = True                                                        # Randomize pd gain.
        kp_range = [0.8, 1.2]                                                           # Multipliers on nominal proportional gains.
        kd_range = [0.8, 1.2]                                                           # Multipliers on nominal derivative gains.

        randomize_motor_strength = True                                                 # Randomize motor strength.
        motor_strength_range = [0.85, 1.15]                                             # Multipliers on motor torque authority.

        randomize_joint_armature = True                                                 # Randomize joint armature.
        joint_armature_range = [0.0, 0.03]                                              # Added joint-inertia range.

        randomize_joint_friction = True                                                 # Randomize joint friction.
        joint_friction_range_start = [0.0, 0.05]                                        # Initial joint-friction bounds.
        joint_friction_range_end = [0.0, 0.20]                                          # Final joint-friction bounds.

        randomize_joint_stiffness = False                                               # Randomize joint stiffness.
        joint_stiffness_range_start = [0.0, 0.0]                                        # Initial passive joint-stiffness bounds.
        joint_stiffness_range_end = [0.0, 0.0]                                          # Final passive joint-stiffness bounds.

        randomize_joint_damping = True                                                  # Randomize joint damping.
        joint_damping_range_start = [0.30, 0.40]                                        # Initial passive joint-damping bounds.
        joint_damping_range_end = [0.00, 0.80]                                          # Final passive joint-damping bounds.

        num_push_steps = 500
        push_warmup_env_steps = 312000  # Warmup before disturbance-curriculum advancement. Units: completed per-environment control steps.


        best_reward_window = 200                                                        # Reward-metric samples retained for curriculum decisions (fixed env-step cadence).
        best_reward_quantile = 0.90                                                     # Reference reward quantile for curriculum advancement.
        recovery_ratio = 0.90                                                           # Fraction of reference performance required to recover.
        step_interval_env_steps = 240  # Curriculum evaluation interval. Units: completed per-environment control steps.
        reward_ema_alpha = 0.05                                                         # New-sample weight for smoothed curriculum reward.
        min_reward_to_step = 0.60                                                       # Minimum reward required for curriculum advancement.
        joint_dynamics_progress_delta = 0.02                                            # Joint-dynamics progress increment per stage.
        mass_com_progress_delta = 0.01                                                  # Mass/CoM progress increment per stage.
        disturbance_progress_delta = 0.01                                               # Disturbance progress increment per stage.
        use_joint_dynamics_curriculum = True                                            # Enable the joint-dynamics curriculum branch.
        use_mass_com_curriculum = True                                                  # Enable the mass/CoM curriculum branch.
        use_disturbance_curriculum = False                                              # Enable the generic disturbance curriculum branch.

    class noise:
        add_noise = True                                                                # Inject noise into policy observations.
        noise_level = 1.0                                                               # Global observation-noise multiplier.
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            ang_vel = 0.5                                                               # Initial root angular velocity [rad/s].
            gravity = 0.06
            height_measurements = 0.1

    class arm:
        mount_offset = [0.3, 0.0, 0.09]                                                 # Arm mount translation relative to the torso [m].
        init_target_ee_base = [0.2, 0.0, 0.2]                                           # Initial Cartesian EE target in the base frame [m].
        grasp_offset = 0.08                                                             # Grasp-point offset [m].

    class termination:
        termination_terms = ["roll", "pitch", "height_min", "height_max"]               # Enabled non-timeout termination checks.
        roll_threshold = 1.0                                                            # Maximum torso roll magnitude [rad].
        pitch_threshold = 1.2                                                           # Maximum torso pitch magnitude [rad].
        height_min = 0.10                                                               # Minimum allowed base height [m].
        height_max = 2.00                                                               # Maximum allowed base height [m].
        contact_force_threshold = 1.0                                                   # Contact magnitude required for contact termination [N].
        contact_patience_steps = 5                                                      # Consecutive undesired-contact steps before termination.

    class constraints:
        class limits:
            pass

    class rewards:
        force_neutral_threshold = 1.0e-3                                                # Force magnitude below which standing rewards are allowed.
        only_positive_rewards = False                                                    # Clamp the summed reward to nonnegative values.
        use_reward_curriculum = True                                                    # Schedule selected reward coefficients.

        tracking_sigma = 0.25                                                           # Velocity tracking error scale.
        tracking_ee_sigma = 1.00                                                        # EE position tracking error scale.

        tracking_ee_orientation_sigma = 0.02                                            # Default EE orientation tracking error scale.
        impedance_virtual_mass = [1.0, 1.0, 1.0]                                        # Per-axis virtual inertia in the soft impedance residual.
        impedance_virtual_damping = [400.0, 400.0, 400.0]                               # Per-axis virtual damping in the impedance residual.
        impedance_virtual_stiffness = [40.0, 40.0, 40.0]                                # Per-axis virtual stiffness in the impedance residual.
        impedance_residual_weights = [1.0, 1.0, 1.0]                                    # Per-axis weights on the impedance residual.
        impedance_filter_alpha = 0.2                                                    # Smoothing weight for impedance kinematics.
        impedance_sigma = 2500.0                                                        # Impedance residual reward scale.

        sigma_force = 1.0 / 50.0                                                        # Force tracking error coefficient.

        soft_dof_pos_limit = 0.8                                                        # Fraction of joint-position range before soft penalties.
        soft_dof_vel_limit = 1.0                                                        # Fraction of velocity limits before soft penalties.
        soft_torque_limit = 0.9                                                         # Fraction of torque limits before soft penalties.

        base_height_target = 0.55                                                       # Desired torso height [m].

        max_contact_force = 600.0                                                       # Foot-force magnitude above which penalties apply [N].

        foot_clearance_target = 0.10                                                    # desired foot clearance above ground [m]
        foot_height_offset = 0.02                                                       # Foot reference/sole height offset [m].
        foot_clearance_tracking_sigma = 0.01                                            # Terrain-aware foot-clearance error scale.

        # Gait-phase guidance settings
        # cycle_time = 0.48
        # sweep_phase_lead = 0.175
        # sweep_velocity_gain = 0.28
        # max_sweep_amplitude = 0.18
        # target_joint_pos_scale = 0.29
        # target_joint_pos_thd = 0.35

        cycle_time = 0.64                                                               # Unshifted reference-gait cycle duration [s].
        target_joint_pos_scale = 0.17                                                   # Reference thigh/calf swing amplitude scale [rad].
        target_joint_pos_thd = 0.50                                                     # Reference swing-phase shaping threshold.
        sweep_phase_lead = 0.0                                                          # Phase lead for fore-aft thigh sweep only [cycles].
        max_sweep_amplitude = 0.0                                                       # Maximum fore-aft thigh offset [rad].
        # Commanded forward velocity to sweep amplitude [rad/(m/s)].
        sweep_velocity_gain = 0.0                                                       # disable the B1-only fore-aft sweep initially

        gait_guidance_decay_enabled = False                                             # Decay reference-pose and stance guidance during learning.
        gait_guidance_decay_env_steps = 240000  # Duration of gait-guidance multiplier decay [completed env steps]. Units: completed per-environment control steps.
        ref_dof_leg_initial_multiplier = 1.0                                            # Initial reference-pose reward multiplier.
        ref_dof_leg_final_multiplier = 0.30                                             # Final reference-pose reward multiplier.
        feet_contact_initial_multiplier = 1.0                                           # Initial stance-matching reward multiplier.
        feet_contact_final_multiplier = 0.30                                            # Final stance-matching reward multiplier.

        upright_gate_sigma = 10.0                                                       # Uprightness gate sharpness.

        arm_before_torso_ee_thresh = 0.08                                               # EE error threshold for arm-before-torso shaping.
        arm_before_torso_gate_sharpness = 40.0                                          # Sharpness of the arm-before-torso gate.

        overreach_x_max = 0.42                                                          # Forward foot overreach bound [m].
        front_foot_x_nominal = 0.34                                                     # Nominal front-foot forward position [m].
        rear_foot_x_nominal = 0.47                                                      # The rear reward negates this magnitude to obtain its base-frame x.
        foot_x_margin = 0.10                                                            # Allowed fore-aft foot-position margin [m].

        support_polygon_sigma = 0.01                                                    # Support-polygon reward error scale.

        torso_tilt_deadband = 0.10                                                      # Unpenalized torso tilt range [rad].

        # Allow small corrective opposition between the coupled action heads;
        # cancellation beyond this fraction of each joint's torque limit is
        # penalized by ``_reward_torque_cancellation``.
        torque_cancellation_deadband = 0.03
        class scales:
            termination = 0.0                                                           # Constraints
            collision = -5.0
            dof_pos_limits = -10.0
            torque_limits = -0.01
            dof_close_to_default = 0.0

            # Add in close to default reward
            stand_still         = -0.5
            stand_still_contact = 0.5

            alive = 1.00

            # tracking
            tracking_lin_vel_force_world = 2.0
            tracking_ang_vel = 1.0

            no_physical_progress = -0.50

            tracking_ee_force_world = 2.0
            tracking_ee_orientation_default = 0.0
            impedance_consistency = 0.5

            # Discourage the position-PD and direct-torque heads from wasting
            # authority by producing large opposing torques on the same joint.
            torque_cancellation = -0.10

            arm_progress_before_torso = 0.5                                             # Style rewards encouraging using the arm
            early_torso_tilt = -0.2
            # feet_contact_number = 0.01
            # arm_progress_before_torso = 0.0
            # early_torso_tilt = 0.0


            ref_dof_leg = 1.0                                                           # gait-phase based leg posture shaping
            walking_ref_dof = 0.0
            walking_ref_swing_dof = 0.0
            feet_contact_number = 1.00
            hip_pos = -0.30

            base_height = -5.0                                                          # Base
            lin_vel_z   = -1.0
            ang_vel_xy  = -0.02
            roll        = -0.2
            orientation = -0.2

            dof_acc           = -2.5e-7                                                 # Legs
            dof_vel           = -1e-4
            joint_power       = -2.e-5
            joint_power_dist  = -1.e-6

            dof_acc_arm = -4.5e-7                                                       # Arm
            dof_vel_arm = -2e-4
            joint_power_arm = -2.e-5
            joint_power_dist_arm = -2.e-6

            # Raw-action temporal penalties are split by coupled head and by
            # leg/arm coordinates while retaining the previous coefficients.
            leg_feedback_action_rate = -0.02
            leg_feedback_action_smoothness = -0.01

            arm_feedback_action_rate = -0.045
            arm_feedback_action_smoothness = -0.01

            leg_feedforward_action_rate = -0.02
            leg_feedforward_action_smoothness = -0.01

            arm_feedforward_action_rate = -0.045
            arm_feedforward_action_smoothness = -0.01

            # Penalize the two applied torque branches independently so their
            # costs remain visible for leg and arm actuation in TensorBoard.
            leg_feedback_torques = -2.0e-5
            arm_feedback_torques = -2.0e-5
            leg_feedforward_torques = -1.0e-5
            arm_feedforward_torques = -1.0e-5

            front_foot_overreach = -1.0                                                 # I developed these
            rear_foot_overreach = -1.0

            # Taken from "Stable Imitation of Multigait and Bipedal Motions for Quadrupedal Robots Over Uneven Terrains" paper
            support_polygon = 0.2                                                       # encourages well condition foot-placement realtive to the base CoM
            vhip_angle = -0.1                                                           # Use a Variable-Height Inverted Pendulum (VHIP) model to penalize unstable torso orientation w.r.t. ground contact
            vhip_angular_acc = -0.01                                                    # Use a Variable-Height Inverted Pendulum (VHIP) model to penalize moving torwards and unstable torso orientation w.r.t. ground contact

            feet_drag = -0.0001                                                         # Gait shaping
            feet_regulation = -0.1
            feet_pos_xy = -0.1
            stumble = -0.1
            feet_contact_forces = -0.001
            feet_air_time = 1.00
            foot_clearance_terrain_aware = 0.70                                         # tracking reward for feet reaching the desired clearance responsive to terrain height

            arm_ee_force_manipulability = 0.0                                           # Leg and Arm Posture Conditioning
            torso_force_wrench_ellipsoid = 0.0

        class manip_rewards():
            ellipsoid_main_weight = 0.6                                                 # Leg Posture Conditioning
            ellipsoid_force_aux_weight = 0.35
            ellipsoid_wrench_aux_weight = 0.35
            ellipsoid_friction_weight = 0.30

            ellipsoid_wrench_length_scale = 1.125
            ellipsoid_force_size_scale = 0.50
            ellipsoid_wrench_size_scale = 0.50

            ellipsoid_force_z_ratio_min = 1.2
            ellipsoid_force_z_ratio_max = 4.0
            ellipsoid_force_xy_ratio_max = 2.0
            ellipsoid_wrench_cond_max = 6.0

            ellipsoid_mu_friction = 0.6
            ellipsoid_normal_force_margin = 5.0
            ellipsoid_tangential_force_margin = 2.0

            # Arm Posture Conditioning
            # Numerical regularization
            arm_ellipsoid_inv_eps = 1e-5

            # Size reward:
            # Larger values make the size reward saturate faster.
            arm_ellipsoid_force_size_scale = 0.80

            # Isotropy reward:
            # cond = lam_max / lam_min.
            arm_ellipsoid_force_cond_max = 4.0                                          # A value near 1 is perfectly isotropic.
            arm_ellipsoid_iso_sharpness = 1.0

            # Log-spread isotropy:
            # Larger values penalize nonuniform eigenvalues more strongly.
            arm_ellipsoid_log_iso_scale = 1.0

            arm_ellipsoid_cond_iso_weight = 0.5                                         # Blend condition-number isotropy and log-spread isotropy.

            arm_ellipsoid_size_weight = 0.5                                             # Final blend between large and isotropic.
            arm_ellipsoid_iso_weight = 0.5

            leg_feedback_action_rate = -0.02
            leg_feedback_action_smoothness = -0.02
            arm_feedback_action_rate = -0.02
            arm_feedback_action_smoothness = -0.02
            leg_feedforward_action_rate = -0.045
            leg_feedforward_action_smoothness = -0.045
            arm_feedforward_action_rate = -0.045
            arm_feedforward_action_smoothness = -0.045


        class reward_curriculum:
            curr_reward_keys = [                                                        # Reward terms whose coefficients are scheduled.
                                "torque_limits",
                                "dof_pos_limits",
                                "dof_vel",
                                # "collision",
                                # "feet_contact_forces",
                                # "lin_vel_z",
                                # "arm_ee_force_manipulability",
                                # "torso_force_wrench_ellipsoid",
                                "leg_feedback_action_rate",
                                "leg_feedback_action_smoothness",
                                "arm_feedback_action_rate",
                                "arm_feedback_action_smoothness",
                                "leg_feedforward_action_rate",
                                "leg_feedforward_action_smoothness",
                                "arm_feedforward_action_rate",
                                "arm_feedforward_action_smoothness",
                                ]
            curr_reward_bounds = {                                                      # Initial and final coefficients for scheduled rewards.
                "torque_limits":[-0.001, -1.0],
                "dof_pos_limits":[-1.0, -10.0],
                "dof_vel":[-1e-6, -1e-4],
                # "collision":[-0.5, -5.0],
                # "feet_contact_forces":[-1.0e-5, -1.0e-4],
                # "lin_vel_z":[-1.00, -2.0],
                # "arm_ee_force_manipulability":[0.2, 0.5],
                # "torso_force_wrench_ellipsoid":[0.2, 0.5],
                "leg_feedback_action_rate":[-0.002, -0.02],
                "leg_feedback_action_smoothness":[-0.002, -0.02],
                "arm_feedback_action_rate":[-0.003, -0.03],
                "arm_feedback_action_smoothness":[-0.003, -0.03],
                "leg_feedforward_action_rate":[-0.004, -0.04],
                "leg_feedforward_action_smoothness":[-0.004, -0.04],
                "arm_feedforward_action_rate":[-0.006, -0.06],
                "arm_feedforward_action_smoothness":[-0.006, -0.06],
            }
            warmup_env_steps = 720000  # Reward-curriculum warmup. Units: completed per-environment control steps.
            curr_env_steps = 240000  # Reward-curriculum ramp duration. Units: completed per-environment control steps.


    class viewer:
        ref_env = 0                                                                     # Environment followed by the viewer.
        pos = [1, 2, 2]
        lookat = [0.0, 0.0, 0.0]                                                        # Viewer camera target [world m].
        num_rendered_envs = 25                                                          # Number of environments displayed.
        rendered_envs_idx = np.random.choice(                                           # Subset of environments displayed.
            np.arange(4000),
            size=num_rendered_envs,
            replace=False,
        )
        add_camera = False                                                              # Create an auxiliary rendering camera.

    class sensor:
        add_depth = False                                                               # Enable depth sensing.
        use_warp = False                                                                # Use Warp-based depth rendering.
        class depth_camera_config:
            num_sensors = 1                                                             # Number of depth sensors per robot.
            num_history = 1                                                             # Stored depth frames.
            near_clip = 0.1                                                             # Near depth clipping distance [m].
            far_clip = 10.0                                                             # Far depth clipping distance [m].
            near_plane = 0.1                                                            # Camera near plane [m].
            far_plane = 10.0                                                            # Camera far plane [m].
            resolution = (80, 60)                                                       # Depth image width and height [pixels].
            horizontal_fov_deg = 75                                                     # Horizontal camera field of view [degrees].
            pos = (0.3, 0.0, 0.1)
            euler = (0.0, 0.0, 0.0)                                                     # Camera mounting Euler angles [rad].
            decimation = 5
            calculate_depth = True                                                      # Return depth images.
            segmentation_camera = False                                                 # Enable segmentation output.
            return_pointcloud = False                                                   # Return a depth-derived point cloud.
            pointcloud_in_world_frame = False                                           # Express point-cloud coordinates in world axes.


class B1Z1PACTCfgPPO(LeggedRobotCfgPPO):
    seed = 1                                                                            # Random seed for reproducible initialization.
    runner_class_name = "B1Z1PACTRunner"                                                # Training runner selected by task registration.

    class policy:

        # Model architecture
        # Actor hidden-layer widths.
        actor_layers = [512, 256, 128]
        critic_layers = [1024, 512, 256, 128]                                           # Critic hidden-layer widths.
        actor_hidden_dims = actor_layers                                                # Compatibility alias for actor widths.
        critic_hidden_dims = critic_layers                                              # Compatibility alias for critic widths.

        activation = "elu"                                                              # MLP activation function.

        # Apply the UniFP per-joint exploration profile independently to both
        # coupled heads. Both action branches are normalized policy outputs.
        init_noise_std = ([0.80, 1.00, 1.00] * 4 + [0.85] * 5) * 2                      # Initial Gaussian action standard deviation per output.
        min_noise_std = ([0.15, 0.25, 0.25] * 4 + [0.15] * 5) * 2                       # Lower bound on action standard deviation per output.
        max_noise_std = 1.1                                                             # Upper bound on action standard deviation.

        # Match UniFP's history VAE and latent-only explicit estimator sizes.
        cenet_enc_layers = [512, 256, 128]                                              # History encoder hidden-layer widths.
        explicit_decoder_layers = [128, 128]                                            # Deterministic history-feature estimator, matching HardPACT.
        force_decoder_layers = [128, 128]                                               # Torso-wrench/EE-force decoder hidden-layer widths.
        grf_decoder_layers = [128, 128]                                                 # Torque-conditioned GRF decoder hidden-layer widths.
        grf_torque_scale = 100.0                                                        # Divide detached physical Nm before GRF conditioning.
        grf_decoder_weight = 1.0                                                        # Supervised normalized GRF reconstruction coefficient.
        cenet_latent_dim = 64                                                           # Dimension of sampled VAE context z.
        cenet_base_vel_dim = 3                                                          # Explicit base-velocity output width.
        cenet_base_wrench_dim = 6                                                       # Torso force/moment output width.
        cenet_ee_force_dim = 3                                                          # EE force output width.
        # Dedicated explicit/force heads are excluded from this reconstruction.
        privileged_decoder_layers = [128, 256, 512]                                     # Hidden widths for non-overlapping privileged reconstruction.
        film_hidden_dim = 64                                                            # FiLM conditioning-network hidden width.
        # Encourage FiLM to be an identity transform near a well-tracked
        # command. The pressure decays exponentially with mean squared base
        # velocity and EE-pose tracking error.
        film_identity_loss_weight = 1.0e-3                                              # Penalty on non-identity FiLM modulation near the target.
        film_identity_error_scale = 1.0                                                 # Tracking-error scale controlling identity-penalty decay.

        # Optional weights-only PACT-Pos hot start generated by
        # rsl_rl/modules/b1z1_pact_pos_to_pact_checkpoint.ipynb. Relative
        # paths are resolved from LEGGED_GYM_ROOT_DIR. None disables loading.
        pretrained_path = None                                                          # Old monolithic-estimator checkpoints are incompatible.
        # pretrained_path = None

        # Loss weights
        # Base-velocity reconstruction coefficient.
        explicit_base_vel_weight = 1.0
        explicit_ee_position_weight = 1.0                                               # Spherical EE-position reconstruction coefficient.
        explicit_base_wrench_weight = 1.0                                               # Normalized torso-wrench reconstruction coefficient.
        explicit_ee_force_weight = 1.0                                                  # Normalized EE-force reconstruction coefficient.
        explicit_foot_contact_weight = 0.1                                              # Binary foot-contact BCE coefficient.
        explicit_foot_height_weight = 1.0                                               # Foot-height reconstruction coefficient.
        privileged_decoder_weight = 1.0                                                 # Remaining next-state reconstruction coefficient.
        vae_kld_weight = 1.00                                                           # Default VAE KL coefficient, separate from physics PINNs.

        use_cosine_kl_warmup = True                                                     # Independently cosine-ramp the base KL coefficient to its maximum.
        kl_warmup_env_steps = 24000  # Cosine VAE KL warmup duration [completed env steps]. Units: completed per-environment control steps.
        kl_warmup_beta_max = vae_kld_weight                                             # Baseline VAE KL coefficient after cosine warmup.
        kl_band_warmup_env_steps = 12000  # Cosine ramp duration for the KL-rate band after base warmup. Units: completed per-environment control steps.

        use_kl_rate_band = False                                                        # Disable the rate band; optional cosine warmup still applies.
        kl_r_min = 2.00                                                                 # Lower target on raw VAE KL rate.
        kl_r_max = 6.00                                                                 # Upper target on raw VAE KL rate.
        kl_dual_lr = 1.0e-3                                                             # Projected dual-variable step size per PPO update.
        kl_aug_rho = 0.1                                                                # Strength of squared KL-band violations.
        kl_ema_decay = 0.99                                                             # Previous-value weight in the KL-rate EMA.
        adaptation_learning_rate = 2.0e-4                                               # HardPACT encoder/decoder learning rate.

        pinn_loss_weight = -1.0                                                         # Magnitude scales PINNs; sign: + PINN / - PPGrad; 0 disables.
        pinn_warmup_env_steps = 12000  # Ramp duration after PINN activation [completed env steps]. Units: completed per-environment control steps.
        pinn_start_env_step = 0  # Completed control steps before the PINN ramp. Units: completed per-environment control steps.
        use_pinn_rollout_loss = True                                                    # Enable the rollout term in addition to inverse dynamics.
        pinn_inverse_weight = 0.5                                                       # Inverse-dynamics coefficient inside the combined physics objective.
        pinn_rollout_weight = 0.5                                                       # Rollout coefficient inside the combined physics objective.

        # Legacy velocity-loss scales only. BARD uses dt * [10, 20, 100]
        # for base translation, rotation, and all joints; these four are unused there.
        pinn_rollout_base_linear_scale = 1.0                                            # Pinocchio-path base-linear velocity error scale [m/s].
        pinn_rollout_base_angular_scale = 1.0                                           # Pinocchio-path base-angular velocity error scale [rad/s].
        pinn_rollout_leg_velocity_scale = 10.0                                          # Pinocchio-path leg velocity error scale [rad/s].
        pinn_rollout_arm_velocity_scale = 5.0                                           # Pinocchio-path arm velocity error scale [rad/s].

        # LEGACYYYYYY BELOWWWW
        # Legacy force mixing controls; BARD trains directly on predicted forces.
        # Gate statistics may still be logged, but do not gate BARD PINN gradients.
        predicted_force_detach = False                                                  # Legacy Pinocchio: stop gradients through predicted forces.
        force_gate_ema_alpha = 0.05                                                     # New-error weight in force-reliability EMAs.
        force_gate_threshold = 0.075                                                    # Legacy force-blend target for normalized reconstruction MSE.
        force_gate_hysteresis = 0.10                                                    # Legacy aggregate gate setting; event-specific limits govern gating.
        force_gate_patience_env_steps = 240  # Qualifying completed env steps required before opening the force gate. Units: completed per-environment control steps.
        # No event mask is stored in the rollout, so normalized target norms
        # identify physical EE/base disturbance events for reliability gating.
        force_gate_ee_event_norm_threshold = 0.05                                       # Normalized EE-force norm separating active/neutral samples.
        force_gate_base_event_norm_threshold = 0.05                                     # Normalized torso-wrench norm separating active/neutral samples.
        force_gate_grf_threshold = 0.075                                                # GRF MSE required to open the reliability gate.
        force_gate_ee_active_threshold = 0.075                                          # Active-EE MSE required to open the gate.
        force_gate_ee_neutral_threshold = 0.075                                         # Neutral-EE MSE required to open the gate.
        force_gate_base_active_threshold = 0.075                                        # Active-torso MSE required to open the gate.
        force_gate_base_neutral_threshold = 0.075                                       # Neutral-torso MSE required to open the gate.
        force_gate_grf_hysteresis = 0.10                                                # GRF MSE allowed while the gate is open.
        force_gate_ee_active_hysteresis = 0.10                                          # Active-EE MSE allowed while the gate is open.
        force_gate_ee_neutral_hysteresis = 0.10                                         # Neutral-EE MSE allowed while the gate is open.
        force_gate_base_active_hysteresis = 0.10                                        # Active-torso MSE allowed while the gate is open.
        force_gate_base_neutral_hysteresis = 0.10                                       # Neutral-torso MSE allowed while the gate is open.
        force_gate_grf_min_samples = 32                                                 # Minimum GRF samples for a reliability decision.
        force_gate_ee_active_min_samples = 32                                           # Minimum active-EE samples for a reliability decision.
        force_gate_ee_neutral_min_samples = 32                                          # Minimum neutral-EE samples for a reliability decision.
        force_gate_base_active_min_samples = 32                                         # Minimum active-torso samples for a reliability decision.
        force_gate_base_neutral_min_samples = 32                                        # Minimum neutral-torso samples for a reliability decision.
        # Minimum predicted-force contribution before reconstruction reaches
        # the reliability threshold; the remainder comes from measurements.
        force_blend_min_alpha = 0.01                                                    # Legacy Pinocchio: minimum predicted-force blend fraction.

    class algorithm:
        # Select PPO_B1Z1PACT in runner.algorithm_class_name for the baseline.
        sac_position_action_range = 3.0                                                # All 17 position outputs: multiply normalized actions before position_action_scale; finite, positive, <= clip_actions.
        sac_leg_torque_action_range = 3.0                                              # 12 leg torque outputs: multiply normalized actions before physical torque scales; finite, positive, <= clip_actions.
        sac_arm_torque_action_range = 2.0                                              # 5 learned arm torque outputs: multiply normalized actions before physical torque scales; finite, positive, <= clip_actions.
        sac_gamma = 0.95                                                               # Discount per simulator control step; independent of the PPO gamma below.
        sac_batch_size = 2048                                                          # Replay transitions sampled per gradient update, across all environments.
        sac_updates_per_step = 2                                                       # Gradient updates per vectorized simulator step after warm-up; accumulated by the runner.
        sac_actor_period = 2                                                           # Update actor and temperature every this many critic updates, starting at update zero.
        sac_tau = 0.01                                                                 # Target-Q EMA fraction: target = (1 - tau) * target + tau * online Q.
        sac_num_bins = 101                                                             # Number of categorical Q-value support bins, including both endpoints.
        sac_min_v = -5.0                                                               # Lower categorical Q-support endpoint in normalized return units.
        sac_max_v = 5.0                                                                # Upper Q-support endpoint; also the reward normalizer's maximum-return scale.
        sac_critic_width = 256                                                         # Hidden feature width of each fused Q critic; does not change the PACT actor.
        sac_critic_blocks = 2                                                          # Residual blocks per Q critic; online and target critics share this architecture.
        sac_initial_temperature = 0.01                                                 # Initial learned entropy coefficient alpha; larger values favor exploration.
        sac_target_sigma = 0.15                                                        # Gaussian std used to derive target entropy H = action_dim/2 * log(2*pi*e*sigma^2).
        sac_learning_rate = 3e-4                                                       # Initial Adam learning rate for actor, Q critic and temperature; auxiliary rates are separate.
        sac_lr_end = 1.5e-4                                                            # Final learning rate reached by each SAC optimizer's cosine decay schedule.
        sac_lr_decay_updates = 1000000                                                 # Cosine decay duration in each optimizer's own steps; delayed actor/temperature advance slower.
        sac_use_amp = False                                                            # Enable CUDA FP16 autocast/scaling for critic updates; actor PCGrad and physics remain FP32.
        sac_use_compile = False                                                        # Compile online/target critic forwards with torch.compile; defaults off for correctness checks.
        replay_capacity = 2500000                                                      # Maximum individual transitions in the ring across all environments; default uses about 634.6 MiB.
        replay_warmup = 10000                                                          # Stored transitions required before optimization; must be between 1 and replay_capacity.
        replay_device = "cpu"                                                          # Replay storage device, independent of the learner device; CPU avoids allocating replay on the GPU.
        replay_pin_memory = True                                                       # Pin CPU replay storage when CUDA is available; ignored for non-CPU storage.
        replay_persistence = False                                                     # Include replay contents in checkpoints and restore when present; increases checkpoint size.
        n_step = 1                                                                     # Bellman backup horizon; only 1 is supported to prevent cross-environment n-step mixing.

        # Actor-facing task prediction; independent of representation-PINN weights.
        actor_phys_enabled = True                         # Opt in only for coupled PACT/BARD.
        actor_phys_coef = 0.1                              # Overall actor auxiliary coefficient.
        actor_phys_vel_weight = 1.0                         # Reachable planar velocity/yaw tracking.
        actor_phys_ee_weight = 1.0                          # Next scheduled, compliant EE target.
        actor_phys_q_weight = 0.1                           # Joint-position safety barrier.
        actor_phys_qd_weight = 0.1                          # Joint-velocity safety barrier.
        actor_phys_velocity_time_constant = 0.25            # Reachable command response time [s].
        actor_phys_q_margin = 0.05                          # Position safety margin [rad].
        actor_phys_qd_margin = 0.5                          # Velocity safety margin [rad/s].
        actor_phys_softplus_temperature = 0.05              # Barrier temperature in normalized units.
        actor_phys_huber_delta = 1.0                        # Huber transition in normalized units.
        actor_phys_ee_scale = 0.1                           # EE-error normalization [m].
        actor_phys_q_scale = 1.0                            # Position-barrier normalization [rad].
        actor_phys_qd_scale = 10.0                          # Velocity-barrier normalization [rad/s].
        actor_phys_require_force_gate = False               # Optional existing force-quality gate.

        actor_phys_pos_fk_enabled = True                   # Direct arm position-command FK objective.
        actor_phys_pos_fk_weight = 0.1                      # Inside the scheduled actor-physics coefficient.
        actor_phys_pos_fk_huber_delta = 1.0                 # Huber threshold after EE-error normalization.
        actor_phys_pos_fk_axis_weights = [1.0, 1.0, 1.0]     # Arm-root Cartesian weights; uses actor_phys_ee_scale.
        actor_phys_pos_fk_deadband = 0.05                    # Per-axis tolerance [m].

        value_loss_coef = 1.0                                                           # Critic regression coefficient.
        use_clipped_value_loss = True                                                   # Apply PPO-style clipping to critic updates.
        clip_param = 0.2                                                                # PPO probability-ratio clipping width.
        entropy_coef = 0.01                                                             # Policy entropy bonus coefficient.
        learning_rate = 3.0e-4                                                          # Actor/critic optimizer learning rate.
        # learning_rate = 3.75e-4                                                          # Actor/critic optimizer learning rate.
        # Learning-rate schedule.
        schedule = "adaptive"                                                           # adaptive
        gamma = 0.99                                                                    # Reward discount factor.
        lam = 0.95                                                                      # Generalized advantage estimation smoothing factor.
        desired_kl = 0.01                                                               # Policy KL target for adaptive learning rate; not VAE KL.
        max_grad_norm = 1.0                                                             # Gradient-norm cap per optimizer ownership group.
        num_learning_epochs = 5                                                         # PPO passes over each rollout.
        num_mini_batches = 4                                                            # Minibatches per PPO epoch.


        # BARD is the differentiable GPU backend; Pinocchio remains available
        # as the numerical reference/fallback.
        dynamics_backend = "bard"                                                       # Select BARD auxiliary PINNs or the legacy Pinocchio path.
        bard_batch_capacity = 0                                                         # Maximum BARD batch; zero derives capacity from rollout size.
        # Persistent CPU workers evaluate the Pinocchio observed-state terms.
        # A zero capacity selects the rollout/minibatch-derived capacity.
        pino_num_workers = 8                                                            # Persistent Pinocchio worker processes.
        pino_batch_capacity = 0                                                         # Pinocchio batch capacity; zero selects an automatic size.
        pino_worker_start_method = "spawn"                                              # Multiprocessing start method for Pinocchio workers.
        use_spo = False                                                                 # Compatibility option for the alternative policy objective.
        use_adaptive_entropy = True                                                    # Adapt entropy coefficient from tracking performance.
        adaptive_ent_bounds = [0.001, 0.01]                                             # Minimum and maximum adaptive entropy coefficients.
        adaptive_ent_lin_threshold = 1.5                                                # Linear-tracking threshold for adaptive entropy.
        adaptive_ent_ang_threshold = 0.70                                               # Angular-tracking threshold for adaptive entropy.
        adaptive_ent_ter_threshold = 6.0                                                # Terrain-related adaptive entropy threshold.
        adaptive_ent_softmax_temp = 2.0                                                 # Temperature for adaptive entropy weighting.

    class runner:
        enable_additional_diagnostics = True                                            # Disable expensive, non-training rollout and PPO-consistency diagnostics.
        policy_class_name = "ActorCriticB1Z1PACT"                                       # Actor-critic implementation selected by the runner.
        algorithm_class_name = "FlashSAC_B1Z1PACT"                                      # Select FlashSAC_B1Z1PACT or PPO_B1Z1PACT for the experimental baseline.
        curriculum_metrics_interval_env_steps = 24                                      # Completed control steps between reward-gated force/domain-rand decisions; independent of rollout length.
        num_steps_per_env = 24                                                          # Control transitions collected per environment per update.
        grf_dim = 12                                                                    # Flattened four-foot XYZ force width.

        max_iterations = 70000                                                          # Total learning iterations (run length only; curricula use env steps).
        # max_iterations = 56000                                                          # Total learning iterations (run length only; curricula use env steps).

        save_interval = 1000                                                            # Checkpoint interval [learning iterations]; does not advance curricula.
        run_name = "b1z1_pact_improved"                                                  # Run label used in output directories.
        experiment_name = "b1z1_pact_lab_flashsac"                                               # Experiment/log directory group.
        sync_wandb = False                                                              # Synchronize supported logs to Weights & Biases.
        resume = False                                                                  # Resume a previous training checkpoint.
        load_run = "Jul14_11-16-03_unifp_baseline"                                      # Run directory selected when resuming.
        checkpoint = -1                                                                 # Checkpoint index; -1 selects the latest.
        resume_path = None                                                              # Explicit checkpoint path override.
