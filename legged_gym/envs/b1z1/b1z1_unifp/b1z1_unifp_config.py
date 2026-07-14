import numpy as np


class B1Z1UniFPCfg:
    seed = 1

    class env:
        num_envs = 6000
        num_observations = 71
        num_privileged_obs = 143
        num_priv_stack = 3
        num_explicit_recon_obs = 12
        num_pred_obs = 12
        num_actions = 17
        num_gripper_joints = 2
        num_obs_hist = 32
        env_spacing = 0.5
        episode_length_s = 20
        grf_dim = 12
        whole_body_dim = 25
        fail_to_terminal_time_s = 0.0
        send_timeouts = True
        debug = False
        debug_viz = False
        debug_draw_height_points_around_base = False
        debug_draw_height_points_around_feet = False
        debug_draw_terrain_height_points = False
        render_ee_goal_debug = False
        render_ee_frame_debug = False

    class goal_ee:
        num_commands = 3
        traj_time = [1.0, 3.0]
        hold_time = [0.5, 2.0]
        command_mode = "sphere"
        collision_upper_limits = [0.1, 0.2, -0.05]
        collision_lower_limits = [-0.9, -0.2, -0.7]
        underground_limit = -0.7
        num_collision_check_samples = 10
        arm_induced_pitch = 0.38

        class sphere_center:
            # Genesis URDF z1_waist origin: base_static_joint [0.3, 0, 0.09]
            # plus z1_waist joint [0, 0, 0.0585].
            x_offset = 0.3
            y_offset = 0.0
            # NOTE - the 0.60 corresponds to the desired base height.
            z_invariant_offset = 0.55 + 0.1485

        class ranges:
            init_pos_start = [0.5, np.pi / 8, 0.0]
            init_pos_end = [0.7, 0.0, 0.0]
            pos_l = [0.40, 0.95]
            pos_p = [-1.0 * np.pi / 2.5, np.pi / 3.0]
            pos_y = [-1.2, 1.2]
            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]

        sphere_error_scale = [1.0, 1.0, 1.0]
        orn_error_scale = [1.0, 1.0, 1.0]
        debug_tcp_from_link06_offset = [0.186, 0.0, 0.0]

    class init_state:
        pos = [0.0, 0.0, 0.6]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        roll_random_scale = 0.0
        pitch_random_scale = 0.0
        yaw_random_scale = 0.0
        default_joint_angles = {
            "FR_hip_joint": -0.2,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
         
            "FL_hip_joint": 0.2,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
         
            "RR_hip_joint": -0.2,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.5,
         
            "RL_hip_joint": 0.2,
            "RL_thigh_joint": 0.8,
            "RL_calf_joint": -1.5,
         
            "z1_waist": 0.0,
            "z1_shoulder": 1.48,
            "z1_elbow": -0.63,
            "z1_wrist_angle": -0.84,
            "z1_forearm_roll": 0.0,
            "z1_wrist_rotate": 1.57,
            "z1_jointGripper": -0.785,
        }
        yaw_angle_range = [0.0, 3.14]
        rand_yaw_range = np.pi / 2
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1
        leg_dof_pos_perturb_range = [0.5, 1.5]
        arm_dof_pos_perturb_range = [-0.5, 0.5]

    class asset:
        name = "b1z1"
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/b1z1_current/urdf/b1z1_genesis.urdf"
        base_name = "trunk"
        base_mass_name = "trunk"
        base_com_name = "trunk"
        dof_names = [
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
        foot_name = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        gripper_name = "ee_gripper_link"
        penalize_contacts_on = ["trunk", "thigh", "calf", "wrist", "shoulder", "elbow", "forearm"]
        terminate_after_contacts_on = []
        links_to_keep = ["FR_foot", "FL_foot", "RR_foot", "RL_foot", "ee_gripper_link"]
        self_collisions = False
        flip_visual_attachments = False
        fix_base_link = False
        obtain_link_contact_states = True
        contact_state_link_names = ["thigh", "calf", "foot", "trunk", "ee_gripper_link"]
        base_link_name = "trunk"
        disable_gravity = False
        collapse_fixed_joints = True
        default_dof_drive_mode = 3
        replace_cylinder_with_capsule = False
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        
        thickness = 0.01
        dof_vel_limits = []

        abad_link_length = 0.12675
        hip_link_length = 0.35
        knee_link_length = 0.35
        knee_link_y_offset = 0.0
        side_signs = [-1.0, 1.0, -1.0, 1.0]  # FR, FL, RR, RL

    class terrain:
        mesh_type = "heightfield"
        simplify_mesh = True
        plane_length = 200.0
        horizontal_scale = 0.1
        vertical_scale = 0.005
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        border_size = 5.0
        border_height = 1.0
        curriculum = False
        obtain_terrain_info_around_feet = True
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.0
        terrain_width = 8.0
        platform_size = 4.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.00, 1.00, 0.00, 0.00, 0.00, 0.00]
        terrain_curriculum_difficulty = {
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
        slope_treshold = 0.75

    class sim:
        dt = 0.002
        substeps = 1
        max_collision_pairs = 100
        IK_max_targets = 2
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1
        use_gpu_pipeline = True

        class physx:
            use_gpu = True
            num_subscenes = 0
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class control:
        control_type = "P"
        stiffness = {
            "hip": 175.0,
            "thigh": 175.0,
            "calf": 300.0,
            "z1_waist": 64.0,
            "z1_shoulder": 128.0,
            "z1_elbow": 64.0,
            "z1_wrist_angle": 64.0,
            "z1_forearm_roll": 64.0,
            "z1_wrist_rotate": 64.0,
            "z1_jointGripper": 64.0,
        }
        damping = {
            "hip": 7.5,
            "thigh": 7.5,
            "calf": 12.5,
            "z1_waist": 1.5,
            "z1_shoulder": 3.0,
            "z1_elbow": 1.5,
            "z1_wrist_angle": 1.5,
            "z1_forearm_roll": 1.5,
            "z1_wrist_rotate": 1.5,
            "z1_jointGripper": 1.5,
        }

        # stiffness = {"joint":100.0, "z1": 64.0,}
        # damping = {"joint": 5.0,"z1": 1.5,}

        action_scale = 0.25
        dt = 0.02
        decimation = 4
        
        use_tradeoff_curriculum = False
        tradeoff_init_weights = [1.0, 1.0]
        tradeoff_final_weights = [1.0, 1.0]
        tradeoff_steps = 1
        tradeoff_threshold = 1.0

    class commands:
        curriculum = True
        max_curriculum = 1.0
        num_commands = 15
        
        resampling_time = 5.0
        
        heading_command = False
        
        curriculum_threshold = 0.6
        
        ang_vel_yaw_clip = 0.2
        ang_vel_pitch_clip = 0.5
        
        lin_vel_x_clip = 0.1
        lin_vel_y_clip = 0.1

        zero_vel_cmd_prob = 0.3
        zero_vel_cmd_prob_after_force = 0.8
        
        force_start_step = 8000

        push_gripper_stators = True
        apply_ee_external_forces = True
        push_gripper_interval_s_cmd = [3.5, 9.0]
        push_gripper_duration_s_cmd = [1.0, 3.0]
        gripper_forced_prob_cmd = 0.8
        
        push_gripper_interval_s_ext = [3.5, 9.0]
        push_gripper_duration_s_ext = [1.0, 3.0]
        gripper_forced_prob_ext = 0.8
        
        randomize_gripper_force_gains = True
        max_push_force_xyz_gripper_cmd = [-60.0, 60.0]
        max_push_force_xyz_gripper_ext = [-60.0, 60.0]

        gripper_force_kp_range = [200.0, 200.0]
        gripper_force_kd_range = [3.0, 3.0]
        gripper_prop_kd = 0.1
        settling_time_force_gripper_s = 1.0
        
        push_robot_base = True
        apply_base_external_forces = True
        push_base_interval_s_cmd = [3.5, 9.0]
        push_base_duration_s_cmd = [1.0, 3.0]
        base_forced_prob_cmd = 0.8
        push_base_interval_s_ext = [6.0, 12.0]
        push_base_duration_s_ext = [1.0, 3.0]
        base_forced_prob_ext = 0.8
        randomize_base_force_gains = True
        max_push_force_xyz_base_cmd = [-50.0, 50.0]
        max_push_force_xyz_base_ext = [-50.0, 50.0]
        
        base_force_kp_range = [200.0, 200.0]
        base_force_kd_range = [200.0, 200.0]
        base_prop_kd = 0.1
        force_z_base_ext_scale = 0.1
        settling_time_force_base_s = 3.0
        use_external_impedance_compensation = False
        compensate_ee_external_force = True
        compensate_base_external_force = True

        class ranges:
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            grf = 0.01
            height_measurements = 5.0
            ee_sphe_radius_cmd = 0.5
            ee_sphe_pitch_cmd = 1.0
            ee_sphe_yaw_cmd = 1.3
            ee_force = 0.01
            base_force = 0.01
        clip_observations = 100.0
        clip_actions = 100.0

    class domain_rand:
        use_domainrand_curriculum = True
        randomize_friction = True  
        friction_range = [0.3, 2.0]

        randomize_base_mass = True
        added_mass_min = -5.0  
        min_added_mass_max = 5.0
        max_added_mass_max = 15.0
  
        randomize_gripper_mass = True
        gripper_mass_min = -0.1
        min_gripper_added_mass_max = 0.1
        max_gripper_added_mass_max = 0.25

        randomize_com_displacement = True
        com_rand_z_positive = False
        com_displacement_x_min = 0.15
        com_displacement_x_max = 0.15
   
        com_displacement_y_min = 0.15
        com_displacement_y_max = 0.15
   
        com_displacement_z_min = 0.15
        com_displacement_z_min_pos = 0.15
        com_displacement_z_max = 0.15
     
        push_robots = True
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
     
        randomize_ctrl_delay = True
        ctrl_delay_step_range = [0, 3]
     
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
     
        randomize_motor_strength = True
        motor_strength_range = [0.85, 1.15]
        
        randomize_joint_armature = True
        joint_armature_range = [0.0, 0.03]
        
        randomize_joint_friction = True
        joint_friction_range_start = [0.0, 0.02]
        joint_friction_range_end = [0.0, 0.04]
        
        randomize_joint_stiffness = False
        joint_stiffness_range_start = [0.0, 0.0]
        joint_stiffness_range_end = [0.0, 0.0]
        
        randomize_joint_damping = True
        joint_damping_range_start = [0.30, 0.40]
        joint_damping_range_end = [0.00, 0.50]
        
        num_push_steps = 500
        push_warmup = 10000
        
        best_reward_window = 200
        best_reward_quantile = 0.90
        recovery_ratio = 0.90
        step_interval = 10
        reward_ema_alpha = 0.05
        min_reward_to_step = 0.60
        joint_dynamics_progress_delta = 0.02
        mass_com_progress_delta = 0.01
        disturbance_progress_delta = 0.01
        use_joint_dynamics_curriculum = True
        use_mass_com_curriculum = True
        use_disturbance_curriculum = False

    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            ang_vel = 0.5
            gravity = 0.06
            height_measurements = 0.1

    class arm:
        mount_offset = [0.3, 0.0, 0.09]
        init_target_ee_base = [0.2, 0.0, 0.2]
        grasp_offset = 0.08

    class termination:
        termination_terms = ["roll", "pitch", "height_min", "height_max"]
        roll_threshold = 0.8
        pitch_threshold = 1.0
        height_min = 0.10
        height_max = 2.00
        contact_force_threshold = 10.0
        contact_patience_steps = 25

    class constraints:
        class limits:
            pass

    class rewards:
        only_positive_rewards = False
        use_reward_curriculum = True
        tracking_sigma = 0.25
        tracking_ee_sigma = 0.50
        tracking_ee_orientation_sigma = 0.05
        sigma_force = 1.0 / 50.0
        
        soft_dof_pos_limit = 0.8
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.9
        
        base_height_target = 0.55
        
        max_contact_force = 400.0
        contact_force_threshold = 1.0
        
        foot_clearance_target = 0.09 # desired foot clearance above ground [m]
        foot_height_offset = 0.02
        foot_clearance_tracking_sigma = 0.01
        
        cycle_time = 0.64
        
        target_joint_pos_scale = 0.17
        target_joint_pos_thd = 0.5

        ee_tracking_sigma = 25.0
        upright_gate_sigma = 10.0

        arm_before_torso_ee_thresh = 0.08
        arm_before_torso_gate_sharpness = 40.0

        overreach_x_max = 0.42   # cm
        rear_foot_x_nominal = -0.37
        rear_foot_x_margin = 0.10
        
        support_polygon_sigma = 0.01

        torso_tilt_deadband = 0.10
        class scales:
            # Constraints
            termination = 0.0
            collision = -5.0
            dof_pos_limits = -10.0
            torque_limits = -0.001
            dof_close_to_default = -0.0001

            # Add in close to default reward
            stand_still         = -0.5
            stand_still_contact = 0.5

            alive = 1.5

            # tracking
            tracking_lin_vel_force_world = 2.0
            tracking_ang_vel = 1.0
            
            tracking_ee_force_world = 2.0
            tracking_ee_orientation_default = 0.0

            # Style rewards encouraging using the arm
            arm_progress_before_torso = 0.3
            early_torso_tilt = -0.2
            feet_contact_number = 0.01

            # Base
            base_height = -5.0
            lin_vel_z   = -2.0
            ang_vel_xy  = -0.20
            roll        = -0.25

            # Legs
            dof_acc           = -2.5e-7
            action_rate       = -0.01
            action_smoothness = -0.01
            joint_power       = -2.e-5
            joint_power_dist  = -1.e-5

            # Arm
            dof_acc_arm = -4.5e-7
            action_rate_arm = -0.02
            action_smoothness_arm = -0.02
            joint_power_arm = -4.e-5
            joint_power_dist_arm = -2.e-5

            # Leg posture shaping
            ref_dof_leg = 0.0

            # I developed these
            front_foot_overreach = -10.0
            rear_foot_overreach = -1.0

            # Taken from "Stable Imitation of Multigait and Bipedal Motions for Quadrupedal Robots Over Uneven Terrains" paper
            support_polygon = 0.2             # encourages well condition foot-placement realtive to the base CoM
            vhip_angle = -0.1                 # Use a Variable-Height Inverted Pendulum (VHIP) model to penalize unstable torso orientation w.r.t. ground contact 
            vhip_angular_acc = -0.001         # Use a Variable-Height Inverted Pendulum (VHIP) model to penalize moving torwards and unstable torso orientation w.r.t. ground contact

            # Gait shaping
            feet_drag = -0.01
            feet_contact_forces = -0.0001
            feet_air_time = 1.00
            foot_clearance_terrain_aware = 1.00  # tracking reward for feet reaching the desired clearance responsive to terrain height            

            # Leg and Arm Posture Conditioning
            arm_ee_force_manipulability = 0.2
            torso_force_wrench_ellipsoid = 0.2
            hip_pos = -0.30

        class manip_rewards():
            # Leg Posture Conditioning
            ellipsoid_main_weight = 0.6
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
            # A value near 1 is perfectly isotropic.
            arm_ellipsoid_force_cond_max = 4.0
            arm_ellipsoid_iso_sharpness = 1.0

            # Log-spread isotropy:
            # Larger values penalize nonuniform eigenvalues more strongly.
            arm_ellipsoid_log_iso_scale = 1.0

            # Blend condition-number isotropy and log-spread isotropy.
            arm_ellipsoid_cond_iso_weight = 0.5

            # Final blend between large and isotropic.
            arm_ellipsoid_size_weight = 0.5
            arm_ellipsoid_iso_weight = 0.5

        class reward_curriculum:
            curr_reward_keys = ["collision", 
                                "action_rate", 
                                "action_rate_arm",
                                "action_smoothness",
                                "action_smoothness_arm",
                                "dof_acc", 
                                "dof_acc_arm",
                                "dof_pos_limits",
                                "feet_contact_forces",
                                "base_height",
                                "lin_vel_z",
                                ]
            curr_reward_bounds = {
                "collision": [-2.0, -10.0],
                "action_rate": [-0.001, -0.01],
                "action_rate_arm": [-0.002, -0.02],
                "action_smoothness":[-0.001, -0.01],
                "action_smoothness_arm":[-0.002, -0.02],
                "dof_acc": [-5.0e-8, -2.5e-7],
                "dof_acc_arm": [-1.0e-8, -4.5e-7],
                "dof_pos_limits":[-2.0, -10.0],
                "feet_contact_forces":[-1.0e-5, -1.0e-4],
                "base_height":[-2.0, -5.0],
                "lin_vel_z":[-1.00, -2.0],
            }
            warmup_steps = 12000
            curr_steps = 6000

    class viewer:
        ref_env = 0
        pos = [1, 2, 2]
        lookat = [0.0, 0.0, 0.0]
        num_rendered_envs = 20
        rendered_envs_idx = np.random.choice(
            np.arange(4000),
            size=num_rendered_envs,
            replace=False,
        )
        add_camera = False

    class sensor:
        add_depth = False
        use_warp = False
        class depth_camera_config:
            num_sensors = 1
            num_history = 1
            near_clip = 0.1
            far_clip = 10.0
            near_plane = 0.1
            far_plane = 10.0
            resolution = (80, 60)
            horizontal_fov_deg = 75
            pos = (0.3, 0.0, 0.1)
            euler = (0.0, 0.0, 0.0)
            decimation = 5
            calculate_depth = True
            segmentation_camera = False
            return_pointcloud = False
            pointcloud_in_world_frame = False


class B1Z1UniFPCfgPPO:
    seed = 1
    runner_class_name = "UniFPRunner"

    class policy:
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        init_noise_std = 1.0

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.001
        learning_rate = 3.0e-4
        schedule = "adaptive"  # adaptive
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        num_learning_epochs = 5
        num_mini_batches = 4
        use_spo = False

    class runner:
        policy_class_name = "ActorCriticUniFP"
        algorithm_class_name = "PPO_UniFP"
        num_steps_per_env = 24
        
        max_iterations = 20000
        
        save_interval = 500
        run_name = "unifp_baseline"
        experiment_name = "b1z1_unifp_genesis"
        sync_wandb = False
        resume = False
        load_run = "Jul12_22-59-32_unifp_baseline"
        checkpoint = -1
        resume_path = None
