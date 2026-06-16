from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import SIMULATOR

class GO2KITECfg( LeggedRobotCfg ):
    
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 45 + (51 + 33) + 143 # robot_state + privilged info + terrain_heights (143)
        num_priv_stack = 5
        num_explicit_recon_obs = 3 + 4 + 4 + 12 # torso lin-velo, feet contact states, feet height
        # num_explicit_recon_obs = 3 + 4 + 4 # torso lin-velo, feet contact states, feet height
        num_actions = 12
        env_spacing = 0.5
        num_obs_hist = 10
        grf_dim = 12
        whole_body_dim = 18
        debug = False # if debugging, visualize contacts, 
        debug_viz = False # draw debug visualizations

    
    class terrain( LeggedRobotCfg.terrain ):
        # mesh_type = 'plane' # plane, heightfield, trimesh
        # plane_length = 200.0 # [m]. plane size is 200x200x10 by default
        # horizontal_scale = 0.1 # [m] distance between height samples in x and y direction
        # vertical_scale = 0.005 # [m] distance between height samples in z direction
        # border_size = 5 # [m] length of the border surrounding the terrain
        # border_height = 1.0 # [m] height of the border surrounding the terrain
        # curriculum = False # whether to use terrain curriculum, starting from easier terrains and gradually increasing the difficulty
        # static_friction = 1.0 # coefficient of static friction of the terrain
        # dynamic_friction = 1.0 # coefficient of dynamic friction of the terrain
        # restitution = 0. # coefficient of restitution of the terrain
        # obtain_terrain_info_around_feet = True

        # rough terrain only:
        mesh_type = "heightfield"
        static_friction = 1.0 # coefficient of static friction of the terrain
        dynamic_friction = 1.0 # coefficient of dynamic friction of the terrain
        restitution = 0. # coefficient of restitution of the terrain
        border_size = 5.0 # [m]
        curriculum = True
        # obtain terrain height information around feet (default: 9 points around feet), measure_
        # x  x   x
        # x F(x) x
        # x  x   x (x: height point, F: foot position)
        obtain_terrain_info_around_feet = True
        measure_heights = True # obtain height measurements

        # Optional visualization of the robot-centric height and normal maps.
        debug_draw_measured_surface_normals = False
        debug_height_map_env_id = 0
        debug_height_point_radius = 0.015
        debug_height_point_color = (0.0, 1.0, 0.0, 1.0)
        debug_height_visualization_offset = 0.02
        debug_surface_normal_length = 0.12
        debug_surface_normal_radius = 0.003
        debug_surface_normal_color = (1.0, 0.8, 0.0, 1.0)
        debug_surface_normal_refresh_steps = 5
        
        # positions of the sampling height around the base (relative to the base of the robot) 11x18 = 198
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2] #  rows
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]                          #  cols
        
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 0 # starting curriculum level
        
        terrain_length = 8.0 # [m] length of each subterrain, X direction
        terrain_width = 8.0 # [m] width of each subterrain, Y direction
        platform_size = 3.0 # [m] size of the flat platform at the center of each subterrain
        num_rows = 10  # number of terrain rows (levels), X direction
        num_cols = 5  # number of terrain cols (types), Y direction
        num_subterrains = num_rows * num_cols
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, wave]
        # terrain_proportions = [0.10, 0.15, 0.20, 0.20, 0.20, 0.15]
        
        # slope, random-rough, stairs-down, stairs-up, discrete, stepping-stone, gap (jump), pit (climb-up), high-platform (climb), platform+gap (climb and jump) 
        # terrain_proportions = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.05, 0.10, 0.10]
        terrain_proportions = [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00]
        simplify_mesh = True
        
        terrain_curriculum_difficulty = {
            "slope": "difficulty * 0.6",
            "step_height": "0.05 + 0.2 * difficulty",
            "discrete_height": "0.05 + 0.2 * difficulty",
            "stepping_stones_params": {
                "stone_length": "np.random.uniform(0.4, 1.2)",
                "stone_width": "np.random.uniform(0.4, 1.2)",
                "stone_distance_x": "0.1 + 0.7 * difficulty",
                "stone_distance_y": "np.random.uniform(0.2, 0.8)",
                "max_height": "0.20",
            },
            "gap_size": "0.1 + difficulty * 0.8",
            "pit_depth": "0.1 + 0.3 * difficulty",
            "high_platform_params": {
                "high_platform_height": "0.1 + 0.3 * difficulty",
                "high_platform_length": "np.random.uniform(0.6, 1.6)",
                "high_platform_width": "np.random.uniform(1.0, 2.0)",
                "high_platform_interval": "np.random.uniform(1.0, 2.0)",
            },
            "high_platform_gaps_params": {
                "high_platform_height": "0.1 + 0.3 * difficulty",
                "high_platform_length": "np.random.uniform(1.6, 2.0)",
                "high_platform_width": "np.random.uniform(1.0, 2.0)",
                "high_platform_distance_y": "np.random.uniform(0.2, 2.0)",
                "gap_size": "0.1 + difficulty * 0.8",
            },
        }
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class sim:
        # Common
        dt = 0.02                 # 500 Hz
        substeps = 1
        # For Genesis
        max_collision_pairs = 100  # More collision pairs will occupy more GPU memory and slow down the simulation
        IK_max_targets = 2         # Fewer IK targets will lead to fewer memory usage

    class init_state( LeggedRobotCfg.init_state ):
        leg_joint_limits = [[-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721]]
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,     # [rad]
            'RL_hip_joint': 0.1,     # [rad]
            'FR_hip_joint': -0.1 ,    # [rad]
            'RR_hip_joint': -0.1,     # [rad]

            'FL_thigh_joint': 0.8,   # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'FR_thigh_joint': 0.8,   # [rad]
            'RR_thigh_joint': 0.8,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,   # [rad]
            'FR_calf_joint': -1.5,   # [rad]
            'RR_calf_joint': -1.5,   # [rad]
        }

        default_joint_torques = { # = target joint torques [nM] when action = 0.0
            'FR_hip_joint':  0.0,   # [nM]
            'FL_hip_joint':  0.0,   # [nM]
            'RR_hip_joint':  0.0,   # [nM]
            'RL_hip_joint':  0.0,   # [nM]

            'FL_thigh_joint': 0.0,  # [nM]
            'RL_thigh_joint': 0.0,  # [nM]
            'FR_thigh_joint': 0.0,  # [nM]
            'RR_thigh_joint': 0.0,  # [nM]

            'FL_calf_joint': 0.0,   # [nM]
            'RL_calf_joint': 0.0,   # [nM]
            'FR_calf_joint': 0.0,   # [nM]
            'RR_calf_joint': 0.0,   # [nM]
        }
        # initial state randomization
        roll_random_scale = 0.1
        pitch_random_scale = 0.1
        yaw_random_scale = 3.14

    class normalization (LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_tau = 0.05               # in collected data the magnitude of the DOF's velocity and torques are roughly comparable 
            grf = 0.01
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 50.

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_domainrand_curriculum = True
        com_rand_z_positive = False
        num_push_steps = 1000  # number of steps to increase the domain randomization ranges
        push_warmup = 4000     # number of steps with initial values held constant
        
        # Randomize Friction
        randomize_friction = True
        friction_range = [0.2, 1.8]

        # What changes with finetuning round
        # Randomized 6DOF torso wrench
        push_robots = True
        push_interval_max = 15.0
        push_interval_min = 5.00
        max_push_vel_xy = 1.00
        min_push_vel_xy = 0.50

        max_vertical_push = 0.50
        min_vertical_push = 0.20
        vert_interval_max = 15.0
        vert_interval_min = 5.00

        max_push_torque = 1.00
        min_push_torque = 0.50
        wrench_timeout_min = 5.00
        wrench_timeout_max = 15.0
        
        # Randomized base mass, applied at COM
        randomize_base_mass = True
        min_added_mass_max = 3.0
        max_added_mass_max = 4.0
        added_mass_min = -1.0
        
        # COM displacement crap
        randomize_com_displacement = True
        com_displacement_x_min = 0.05
        com_displacement_x_max = 0.075
        
        com_displacement_y_min = 0.05
        com_displacement_y_max = 0.075
        
        com_displacement_z_positive = False
        com_displacement_z_min_pos = 0.1
        com_displacement_z_min = 0.05
        com_displacement_z_max = 0.075
        
        # Control delay
        randomize_ctrl_delay = True
        ctrl_delay_step_range = [0, 2]

        # PD-gain randomization
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]

        # Motor strength randomization
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        
        randomize_joint_armature = True
        joint_armature_range = [0.00, 0.015]         # [N*m*s/rad]
        
        randomize_joint_friction = True
        joint_friction_range_end   = [0.00, 0.2]
        joint_friction_range_start = [0.00, 0.05]
        
        randomize_joint_stiffness = True
        joint_stiffness_range_end   = [0.0, 0.02]
        joint_stiffness_range_start = [0.0, 0.005]
        
        randomize_joint_damping = True
        joint_damping_range_end   = [0.00, 0.80]
        joint_damping_range_start = [0.20, 0.60]
        
        
        # new domain randomization curriculum parameters
        best_reward_window = 400        # amount of history used to capture recent performance.
        best_reward_quantile = 0.90     # quantile for determining "max" performance over history window.

        recovery_ratio = 0.90           # allowable deviation from quantile of history window
        step_interval = 10              # minimum number of iterations before taking next domain rand step

        reward_ema_alpha = 0.05         # EMA value for tracking
        min_reward_to_step = 0.60       # absolute performance floor for curriculum steps

        joint_dynamics_progress_delta = 0.02 # domain rand step delta for stepping joint-level dynamics parameters
        mass_com_progress_delta = 0.01       # domain rand step delta for stepping payload parameters
        disturbance_progress_delta = 0.01    # domain rand step delta for external disturbance parameters
        use_joint_dynamics_curriculum = True # set False to skip joint stiffness/damping/friction curriculum updates
        use_mass_com_curriculum = True       # set False to skip payload and CoM curriculum updates
        use_disturbance_curriculum = True    # set False to skip push/wrench curriculum updates

        randomize_camera_pos = True
        camera_com_displacement_range = [0.01, 0.0025, 0.03]
        randomize_camera_euler = True
        # Roll/yaw vary by about 3.3 degrees; pitch varies the nominal
        # 10-degree downward view angle by +/-5 degrees.
        camera_euler_offset_range = [0.0577, 0.08726646259971647, 0.0577]

    class noise (LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            dof_tau = 0.5
            lin_vel = 0.1
            ang_vel = 0.5
            gravity = 0.06
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [2, 2, 2]       # [m]
        lookat = [0., 0, 1.]  # [m]
        # rendered_envs_idx = [1500]
        rendered_envs_idx = [i for i in range(0, 3, 1)]  # number of environments to be rendered
        rendered_envs_idx.extend([i for i in range(500, 503, 1)])  # number of environments to be rendered
        rendered_envs_idx.extend([i for i in range(900, 903, 1)])  # number of environments to be rendered

        rendered_envs_idx.extend([i for i in range(1500, 1503, 1)])
        rendered_envs_idx.extend([i for i in range(3500, 3503, 1)])
        rendered_envs_idx.extend([i for i in range(4000, 4003, 1)])

        rendered_envs_idx.extend([i for i in range(1700, 1703, 1)])
        rendered_envs_idx.extend([i for i in range(2200, 2203, 1)])
        rendered_envs_idx.extend([i for i in range(3900, 3903, 1)])
        # rendered_envs_idx = [0, 1000, 3500]
        add_camera = False

    class sensor:
        add_depth = SIMULATOR == "genesis_kite_depth"
        # add_depth = False
        use_warp = add_depth
        class depth_camera_config:
            num_sensors = 1
            num_history = 1        # history frames for depth images

            near_clip = 0.0
            far_clip = 3.0
            near_plane = 0.05
            far_plane = 4.0
            resolution = (120, 160)
            resized_resolution = (48, 64)
            crop_top_bottom = (12, 0)
            crop_left_right = (7, 9)
            horizontal_fov_deg = 88
            
            pos = (0.32, 0.0, 0.07)
            # Warp camera rays point along local +Z. A +105 degree pitch maps
            # that axis 15 degrees downward from the robot's forward +X axis.
            euler = (0.0, 1.74533, 0.0)
            decimation = 5
            latency_range = (0.08, 0.142)
            latency_resampling_time = 5.0
            refresh_duration = 0.1
            
            calculate_depth = True
            segmentation_camera = False
            return_pointcloud = False
            pointcloud_in_world_frame = False
            
            stereo_min_distance = 0.175
            stereo_far_distance = 1.2
            stereo_far_noise_std = 0.08
            stereo_near_noise_std = 0.02
            stereo_half_block_spark_prob = 0.02
            sky_artifacts_prob = 0.001
            sky_artifacts_far_distance = 2.0
            sky_artifacts_values = (0.6, 1.0, 1.2, 1.5, 1.8)

            # Debug rendering controls.
            debug_render_depth_image = False
            debug_camera_env_id = 0
            debug_draw_camera_position = False
            debug_camera_marker_radius = 0.03
            debug_camera_marker_color = (1.0, 0.0, 0.0, 1.0)
            debug_draw_camera_direction = False
            debug_camera_direction_length = 0.15
            debug_camera_direction_radius = 0.002
            debug_camera_direction_color = (1.0, 0.0, 0.0, 1.0)
            debug_print_depth_stats = False
            debug_depth_stats_interval = 25

    class asset( LeggedRobotCfg.asset ):
        name = "go2"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        dof_names = [        # specify the sequence of actions
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',]
        # foot_name = "foot"
        foot_name = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        penalize_contacts_on = ["thigh", "hip", "calf", "base", "Head"]
        terminate_after_contacts_on = ["base","Head"]
        links_to_keep = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        self_collisions = True
        obtain_link_contact_states = True
        contact_state_link_names = ["thigh", "calf", "foot", "base", "hip"]
        
        abad_link_length = 0.0955
        hip_link_length = 0.213
        knee_link_length = 0.213
        knee_link_y_offset = 0.0
        side_signs = [-1.0, 1.0, -1.0, 1.0]   # FR, FL, RR, RL
  
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        # Much smaller values than typical... only used for feedback control
        stiffness = {'joint': 30.0}   # [N*m/rad]
        damping   = {'joint': 0.60}     # [N*m*s/rad]
        
        action_scale = 0.25   # action scale: target angle = action_scale * pose_action + defaultAngle
        torque_scale = 10.00   # action scale:  target torque = torque_scale * tau_action + defaultTorque
        
        
        dt =  0.02     # control frequency 200Hz
        decimation = 4  # decimation: Number of control action updates @ sim DT per policy DT

        # Assumed order - tau_ff, tau_fb
        # tradeoff_init_weights  = [0.80, 1.4]
        tradeoff_init_weights  = [1.00, 1.00]
        tradeoff_final_weights = [1.00, 1.00]
        tradeoff_steps = 10
        tradeoff_threshold = 0.60
        use_tradeoff_curriculum = False

    class termination:
        termination_terms = ["roll", "pitch", "height_min", "height_max"]
        roll_threshold    = 0.7  # [rad] ~ 40 degrees
        pitch_threshold   = 1.0  # [rad] ~ 40 degrees
        height_min = 0.20       # [m]
        height_max = 1.50       # [m]

        # Reset after a foot or the base falls into terrain marked as a deep void.
        reset_unrecoverable_gaps = True
        gap_terrain_depth_threshold = 1.0  # void height below the environment origin [m]
        gap_foot_drop_threshold = 0.25     # foot height below the environment origin [m]
        gap_base_drop_threshold = 0.30     # base height below the environment origin [m]
        gap_min_fallen_feet = 1
        gap_reset_steps = 4

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.90
        soft_torque_limit = 0.90
        base_height_target = 0.33
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        
        foot_clearance_target = 0.09 # desired foot clearance above ground [m]
        foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
        
        overreach_x_max = 0.36
        rear_foot_x_nominal = -0.20
        rear_foot_x_margin = 0.16
        support_polygon_sigma = 0.01

        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = True

        use_reward_curriculum = True

        max_contact_force = 200.0
        feet_edge_threshold = 0.05
        class scales( LeggedRobotCfg.rewards.scales ):
            # General
            termination           = 0.0
            collision             = -10.0
            dof_pos_limits        = -2.0
            dof_close_to_default  = -0.01
            torque_limits         = -0.0001

            alive_bonus           = 0.001

            stand_still_contact = -0.5
            dof_pos_stand_still = -0.1
            # dof_vel_stand_still = -0.5

            # command tracking
            tracking_lin_vel  = 1.0
            tracking_ang_vel  = 0.5
            
            dof_tracking      = 0.00
            aligned_torques   = 0.00
            sparse_contacts   = 0.01         
            
            # smoothness and stability
            lin_vel_z        = -2.0
            base_height      = -1.0
            ang_vel_xy       = -0.05
            orientation      = -0.2
            dof_acc          = -2.0e-7
            joint_power      = -2.0e-5
            joint_power_dist = -1.0e-5
            torques          = 0.0     # don't need to use this when we already have joint power above...

            # Zero out some values that are used in the individual reward classes below
            action_rate       = -0.01
            action_smoothness = -0.01

            pos_action_rate       = 0.0
            pos_action_smoothness = 0.0

            tau_action_rate       = 0.0
            tau_action_smoothness = 0.0

            feedforward_torques   = 0.0
            feedback_torques      = 0.0
            dof_act_limits        = 0.0

            # Potential-based recovery rewards
            pbrs_orientation = 10.0
            pbrs_height = 10.0

            # Stability rewards
            support_polygon = 0.2
            vhip_angle = -0.1
            vhip_angular_acc = -0.001

            # Foot-placement limits
            front_foot_overreach = -10000.0
            rear_foot_overreach = -10.0

            # gait
            feet_air_time    = 0.70            # tracking reward for long steps
            # foot_clearance   = 0.20            # tracking reward for feet reaching the desired clearance
            foot_clearance_terrain_aware = 0.70  # tracking reward for feet reaching the desired clearance responsive to terrain height
            hip_pos = -0.20
            
            foot_slip        = -0.01           # penalty for feet slipping
            feet_contact_forces = -1.0e-2     # penalty for high contact forces on the feet
            feet_near_edge = -1.0
            feet_spread_pairwise_axes = 0.0

            torso_force_wrench_ellipsoid = 0.2
            swing_vel_ellipsoid_terrain  = 0.1

        # KITE reward terms
        class kite_rewards():
            ellipsoid_main_weight = 0.6
            ellipsoid_force_aux_weight = 0.35
            ellipsoid_wrench_aux_weight = 0.35
            ellipsoid_friction_weight = 0.30

            ellipsoid_wrench_length_scale = 0.40
            ellipsoid_force_size_scale = 0.50
            ellipsoid_wrench_size_scale = 0.50

            ellipsoid_force_z_ratio_min = 1.2
            ellipsoid_force_z_ratio_max = 4.0
            ellipsoid_force_xy_ratio_max = 2.0
            ellipsoid_wrench_cond_max = 6.0

            ellipsoid_mu_friction = 0.6
            ellipsoid_normal_force_margin = 5.0
            ellipsoid_tangential_force_margin = 2.0

        class reward_curriculum():
            # curr_reward_keys = ["orientation", "ang_vel_xy"]
            
            # curr_reward_bounds = {
            #                       "orientation":[-0.2,-1.0],
            #                       "ang_vel_xy":[-0.05, -0.2]
            #                      }

            curr_reward_keys = [
                                "torso_force_wrench_ellipsoid", 
                                "swing_vel_ellipsoid_terrain"
                                ]
            
            curr_reward_bounds = {
                                  "torso_force_wrench_ellipsoid":[0.2, 0.6],
                                  "swing_vel_ellipsoid_terrain":[0.1, 0.4]
                                 }

            curr_steps = 1000
            warmup_steps = 3000

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 2.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]

        curriculum_ema_alpha = 0.05
        curriculum_best_window = 400
        curriculum_best_quantile = 0.90
        curriculum_recovery_ratio = 0.90
        curriculum_min_lin_tracking = 0.60
        curriculum_min_ang_tracking = 0.60
        curriculum_min_episode_fraction = 0.25
        curriculum_update_interval_steps = 10000

        lin_vel_x_step = 0.10
        lin_vel_y_step = 0.05
        ang_vel_yaw_step = 0.10
        max_lin_vel_y = 0.30
        max_ang_vel_yaw = 3.0
        
        randomize_resampling_time = False
        resampling_time_min = 1.0
        resampling_time_max = 10.0
        use_command_resampling_time_curriculum = False
        command_resampling_time_warmup_iters = 5000
        
        heading_command = True # if true: compute ang vel command from heading error
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

class GO2KITECfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = "KITERunner" # Teacher-Student Runner
    
    class policy( LeggedRobotCfgPPO.policy ):
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid, swish (SiLU)
        init_noise_std = 1.00
        
        # Context encoder
        cenet_enc_layers=[512,256,128]
        cenet_enc_latent_dim = 16
        cenet_velo_dim = 3 + 4 + 4 + 12   # torso velocity, foot-contact indicator, foot-height 
        # cenet_velo_dim = 3 + 4 + 4   # torso velocity, foot-contact indicator, foot-height 

        # Context Decoder
        cenet_dec_input_dim = 27 + 12
        # cenet_dec_input_dim = 27
        cenet_dec_layers = [32,128,256,512]
        cenet_dec_out_dim = 45 + (51 + 33) + 143     # next obs (45) + grf_dim (12)

        # Actor/critic
        actor_layers = [512,256,128]
        critic_layers = [1024,256,128]
        
        # Shared
        dropout = 0.1

        pinn_loss_weight = 0.01
        pinn_warmup = 10000
        pinn_init_steps = 0

        # pretrained_path = "../../rsl_rl/modules/pretrained_models/rl_pos/Jan17_17-39-51_unimodel_grf_01_100hz_tanh_pos/model_1000.pt"
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # learning_rate = 1.0e-3 #
        learning_rate = 3.0e-4 #
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam   = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

        # Adaptive entropy coefficient curriculum
        entropy_coef = 0.01
        use_adaptive_entropy = True
        adaptive_ent_bounds = [0.005, 0.01]
        adaptive_ent_lin_threshold = 0.75
        adaptive_ent_ang_threshold = 0.35
        adaptive_ent_ter_threshold = 6.0
        adaptive_ent_softmax_temp = 2.0

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic_KITE'
        algorithm_class_name = 'PPO_KITE'
        num_steps_per_env = 24 # per iteration
        max_iterations = 7000 # number of policy updates
        grf_dim = 12
        
        # debug_warmpinn_wb
        run_name = '50hz_spec_jointrand_stairs'
        experiment_name = 'go2_kite_rough'
        save_interval = 500
        
        
        # load_run = "Apr15_11-53-35_50hz_spec_jointrand_stairs"
        load_run = "Apr15_21-58-54_50hz_spec_jointrand_stairs_baseline"
        checkpoint = -1
        resume = False
        exp_data_path = "exp_data/kite_feasibility/baseline_model_stairs.csv"
