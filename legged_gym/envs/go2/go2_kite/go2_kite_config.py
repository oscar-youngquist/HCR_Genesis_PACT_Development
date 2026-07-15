from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import SIMULATOR

class GO2KITECfg( LeggedRobotCfg ):
    
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 132
        num_priv_stack = 5
        num_explicit_recon_obs = 3 + 4 + 4 + 12 # torso lin-velo, feet contact states, feet height
        num_actions = 12
        env_spacing = 0.5
        num_obs_hist = 10
        grf_dim = 12
        whole_body_dim = 18
        debug = True # if debugging, visualize contacts,
        debug_viz = False # draw debug visualizations

    
    class terrain( LeggedRobotCfg.terrain ):
        # mesh_type = 'plane' # plane, heightfield, trimesh
        # plane_length = 200.0 # [m]. plane size is 200x200x10 by default
        horizontal_scale = 0.1  # [m] distance between height samples in x and y direction
        vertical_scale = 0.005  # [m] distance between height samples in z direction
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
        move_down_by_accumulated_xy_command = True
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

        debug_draw_edge_mask = False
        debug_edge_mask_env_id = 0
        debug_edge_mask_refresh_steps = 20
        debug_edge_mask_stride_cells = 1
        debug_edge_mask_max_points = 0
        debug_edge_mask_radius = 0.01
        debug_edge_mask_height_offset = 0.035
        debug_edge_mask_color = (1.0, 0.1, 0.1, 0.9)
        debug_edge_non_edge_color = (0.1, 0.35, 1.0, 0.35)

        # positions of the sampling height around the base (relative to the base of the robot) 11x18 = 153
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2] #  rows
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]                          #  cols

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 2 # starting curriculum level
        
        terrain_length = 8.0 # [m] length of each subterrain, X direction
        terrain_width = 8.0 # [m] width of each subterrain, Y direction
        platform_size = 3.0 # [m] size of the flat platform at the center of each subterrain
        num_rows = 10  # number of terrain rows (levels), X direction
        num_cols = 20  # number of terrain cols (types), Y direction
        num_subterrains = num_rows * num_cols
        # Order: slope, rough, stairs down, stairs up, discrete, wave,
        # stepping stones, gap, pit, platforms, platforms and gaps.
        terrain_proportions = [0.15, 0.15, 0.25, 0.25, 0.10, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00]

        # terrain_proportions = [0.09, 0.09, 0.09, 0.09, 0.09, 0.10, 0.09, 0.09, 0.09, 0.09, 0.09]
        # terrain_proportions = [0.10, 0.10, 0.15, 0.15, 0.10, 0.00, 0.15, 0.10, 0.15, 0.00, 0.00]
        # terrain_proportions = [0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.40, 0.40, 0.00]
        simplify_mesh = True

        edge_mask_dilation_cells = 0

        add_terrain_roughness = True
        terrain_roughness_height_range = [0.0, 0.04]
        terrain_roughness_step = 0.005
        terrain_roughness_downsampled_scale = 0.30
        terrain_roughness_protect_edges = True
        terrain_roughness_edge_clearance = 0.20
        terrain_roughness_border_clearance = 0.20
        # None applies roughness to all terrain kinds when enabled. To restrict
        # it, use terrain kind ids from legged_gym.utils.terrain.Terrain.
        terrain_roughness_kind_ids = [0,  # slope
                                      5,  # wave
                                      ]
        # 2,  # stairs down
        # 3,  # stairs up
        # 4,  # discrete obstacles
        # 6  # stepping stones
        # 7  # gap
        # 8  # pit
        # 9  # multiple high platforms
        # 10 # high platform gaps
        
        terrain_curriculum_difficulty = {
            "slope": "difficulty * 0.4",
            "step_height": "0.05 + 0.20 * difficulty",
            "discrete_height": "0.05 + 0.20 * difficulty",
            "wave_params": {
                "num_waves": "1",
                "amplitude": "0.05 + 0.10 * difficulty",
            },
            "stepping_stones_params": {
                "stone_length": "max(0.20, np.random.uniform(0.5, 0.8) - 0.3 * difficulty)",
                "stone_width": "max(0.20, np.random.uniform(0.5, 0.8) - 0.3 * difficulty)",
                "stone_distance_x": "0.1 + 0.3 * difficulty",
                "stone_distance_y": "np.random.uniform(0.1, 0.4)",
                "max_height": "0.10",
                "min_stone_length": "0.20",
                "min_stone_width": "0.20",
                "stepping_stone_edge_clearance": "0.4",
            },
            "gap_size": "0.1 + difficulty * 0.4",
            "pit_depth": "0.1 + 0.3 * difficulty",
            "high_platform_params": {
                "high_platform_height": "0.1 + 0.3 * difficulty",
                "high_platform_length": "np.random.uniform(0.6, 1.6)",
                "high_platform_width": "np.random.uniform(6.0, 8.0)",
                "high_platform_interval": "np.random.uniform(1.0, 2.0)",
                "min_high_platform_interval": "0.8",
                "min_high_platform_edge_clearance": "0.8",
            },
            "high_platform_gaps_params": {
                "high_platform_height": "0.1 + 0.3 * difficulty",
                "high_platform_length": "np.random.uniform(1.6, 2.0)",
                "high_platform_width": "np.random.uniform(6.0, 8.0)",
                "high_platform_distance_y": "np.random.uniform(0.2, 2.0)",
                "gap_size": "0.1 + difficulty * 0.4",
                "min_high_platform_track_width": "0.35",
                "min_high_platform_edge_clearance": "0.8",
            },
        }
        # trimesh only:
        slope_treshold = 1.5 # slopes above this threshold will be corrected to vertical surfaces

    class sim:
        # Common
        dt = 0.005                 # 500 Hz
        substeps = 2
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
            'RL_thigh_joint': 1.0,   # [rad]
            'FR_thigh_joint': 0.8,   # [rad]
            'RR_thigh_joint': 1.0,   # [rad]

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
        clip_actions = 100.

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
        pos = [0.5, 1.5, 0.5]       # [m]
        lookat = [0., 0, 0.0]  # [m]
        # rendered_envs_idx = [1500]
        rendered_envs_idx = [i for i in range(0, 3, 1)]  # number of environments to be rendered
        rendered_envs_idx.extend([i for i in range(500, 503, 1)])  # number of environments to be rendered
        rendered_envs_idx.extend([i for i in range(900, 903, 1)])  # number of environments to be rendered

        # rendered_envs_idx.extend([i for i in range(1500, 1503, 1)])
        # rendered_envs_idx.extend([i for i in range(3500, 3503, 1)])
        # rendered_envs_idx.extend([i for i in range(4000, 4003, 1)])

        # rendered_envs_idx.extend([i for i in range(1700, 1703, 1)])
        # rendered_envs_idx.extend([i for i in range(2200, 2203, 1)])
        # rendered_envs_idx.extend([i for i in range(3900, 3903, 1)])
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
        damping   = {'joint': 0.75}     # [N*m*s/rad]
        
        action_scale = 0.25   # action scale: target angle = action_scale * pose_action + defaultAngle
        torque_scale = 10.00   # action scale:  target torque = torque_scale * tau_action + defaultTorque
        
        
        dt =  0.02     # control frequency 50Hz
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
        pitch_threshold   = 1.39  # [rad] ~ 80 degrees - larger to allow for climbing
        height_min = 0.20       # [m]
        height_max = 1.50       # [m]

        # Reset after a foot or the base falls into terrain marked as a deep void.
        reset_unrecoverable_gaps = False
        gap_terrain_depth_threshold = 1.0  # void height below the environment origin [m]
        gap_foot_drop_threshold = 0.75     # foot height below the environment origin [m]
        gap_base_drop_threshold = 0.75     # base height below the environment origin [m]
        gap_min_fallen_feet = 1
        gap_reset_steps = 4
        gap_terrain_projection_max_distance = 1.5
        gap_terrain_projection_stride_cells = 3

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.90
        soft_torque_limit = 0.90
        base_height_target = 0.40
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)

        # Order: slope, rough, stairs down, stairs up, discrete, wave,
        # stepping stones, gap, pit, platforms, platforms and gaps.
        dynamic_sigma = {
            "min_lin_vel": 0.5,
            "max_lin_vel": 1.5,
            "min_ang_vel": 1.0,
            "max_ang_vel": 2.0,
            "max_sigma": [1/4, 1/4, 1/2, 1/2, 3/4, 5/12, 1, 1, 1, 1, 1],  # parkour envs are most relaxed
        }

        #### Currently unused
        tracking_lin_vel_error_scale = 4.0
        tracking_ang_vel_error_scale = 4.0
        ####

        foot_clearance_target = 0.09 # desired foot clearance above ground [m]
        foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
        foot_clearance_excess_margin = 0.04
        foot_clearance_excess_weight = 0.25
        
        overreach_x_max = 0.36
        rear_foot_x_nominal = -0.20
        rear_foot_x_margin = 0.16
        
        support_polygon_sigma = 0.01
        
        vhip_angle_deadband = 0.1
        vhip_acc_deadband = 0.001

        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = False

        use_reward_curriculum = True

        max_contact_force       = 200.0
        contact_force_threshold = 5.0

        # Used to prevent exploting "sloped" edges (artifact of heighfield's being unable to generate vertical faces)
        feet_edge_threshold     = 0.05             # [m]
        edge_clearance_lateral_cells = (-1, 0, 1)
        edge_clearance_forward_cells = (0, 1, 2)
        edge_swing_clearance_margin = 0.04         # [m]
        swing_collision_max_normal_z = 0.85
        swing_collision_min_speed = 0.05

        # Position-based progress reward/penalty
        position_progress_min_cmd = 0.05
        position_no_progress_fraction = 0.75

        # Used to create "balanced" swing-leg participation
        swing_ema_alpha         = 0.97
        swing_height_ema_alpha  = 0.95
        completed_swing_min_height_weight = 1.0
        class scales( LeggedRobotCfg.rewards.scales ):
            # General
            termination           = 0.0
            collision             = -1.0
            dof_pos_limits        = -2.0
            dof_close_to_default  = -0.0001
            torque_limits         = -0.0001

            alive_bonus           = 0.01

            stand_still_contact = 0.5
            dof_pos_stand_still = -0.5
            # dof_vel_stand_still = -0.5

            # Psuedo Potential rewards -> command tracking
            #    positve pulls towards tracking
            tracking_lin_vel  = 1.00
            tracking_ang_vel  = 0.50
            
            #    negative pushes away from not tracking
            tracking_lin_vel_penalty = 0.0
            tracking_ang_vel_penalty = 0.0

            # Asymmetric PB reward for making progress in the desired direction
            position_progress    =  0.1
            position_no_progress = -0.1

            dof_tracking      = 0.00
            aligned_torques   = 0.00
            sparse_contacts   = 0.01
            heading_error     = -0.1 
            
            # smoothness and stability
            lin_vel_z        = -1.0
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
            front_foot_overreach = -1000.0
            rear_foot_overreach = -10.0

            # gait
            feet_air_time                = 0.70    # (-) tracking reward for long steps
            foot_clearance_terrain_aware = 0.30    # (+) tracking reward for feet reaching the desired clearance responsive to terrain height            
            
            foot_slip                    = -0.01   # penalty for feet slipping
            feet_contact_forces          = -1.0e-4 # penalty for high contact forces on the feet
            
            feet_near_edge               = -1.0    # penalty for feet being in contact near any edge terrain  
            stumble                      = -1.0   # penalty for making horizontal contact during swing phase.
            
            edge_swing_clearance         = -2.0    # penalty for not lifitng foot above edges of vertical surfaces
            swing_foot_collision_edge    = -1.0    # penlaty for swigning feet into the artifical "slope" surfaces around edges in heightfield terrains
            feet_regulation              = -0.1   # penalty for learning a (1) lift (2) swing (3) touchdown swing-leg cycle
            
            # Targted posture regularization
            hip_pos                      = -0.05   # hip joints specifically should be close to default. Ued to avoid learning unnecessarily wide gaits.
            x_command_hip_symmetry       = -0.01    # hip joints should be symmetric when moving forwards fast

            # Added these gait balance rewards to discourage observed behavior of diagonals pairs of feet behaving differently.
            swing_participation_balance    = 0.10   # (-) encourages all feet to swing for roughly the same amount of time an episode
            diagonal_pair_balance          = 0.10   # (-) encourages diagonal pairs of feet specifically to wing the same amount of time per episode
            completed_swing_height_balance = 0.10   # (-) enocurages all swing feet to reach the desired height throughout a swing

            # Novel KITE-specific whole-body posture + terrain regualrization/alignment rewards
            torso_force_wrench_ellipsoid = 0.2     # (+) encourage whole-body postures that align well conditioned force generation with the terrain
            swing_vel_ellipsoid_terrain  = 0.1     # (+) align swing feet with the terrain underfoot.

        # KITE reward paramaters
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

            curr_reward_keys = ["torque_limits",
                                "dof_pos_limits",
                                "collision",
                                "joint_power",
                                "action_rate",
                                "action_smoothness",
                                "dof_acc",
                                
                                "torso_force_wrench_ellipsoid", 
                                "swing_vel_ellipsoid_terrain",
                                
                                "stumble",
                                "front_foot_overreach",
                                "rear_foot_overreach",
                                "feet_regulation",
                                "edge_swing_clearance",
                                "swing_foot_collision_edge",

                                "position_no_progress",
                                "position_progress",

                                "lin_vel_z",
                                "base_height",
                                
                                "swing_participation_balance",
                                "diagonal_pair_balance",
                                "completed_swing_height_balance",
                                ]
            
            curr_reward_bounds = {
                                  # these get bigger to enforce platform saftey for physical deployment   
                                  "torque_limits":[-1.0e-4, -1.0e-2],
                                  "dof_pos_limits":[-1.0, -10.0],
                                  "collision":[-1.0, -5.0],

                                  # These increase from a relative small starting value and decay slightly to allow for more aggresive behaviors
                                  "joint_power":[-2.0e-5, -2.0e-6],
                                  "action_rate":[-0.01, -0.0001],
                                  "action_smoothness":[-0.01, -0.0001],
                                  "dof_acc":[-2.0e-8, -2.0e-9],
                                  
                                  "torso_force_wrench_ellipsoid":[0.1, 0.50],
                                  "swing_vel_ellipsoid_terrain": [0.05, 0.40],
                                  
                                  "front_foot_overreach":[-10.0, -100.0],
                                  "rear_foot_overreach":[-1.0, -10.0],
                                  
                                  # these get stronger after initial gait discovery in order to prevent maladapted 
                                  #    swing-beahviors that transfer poorly to the real world   
                                  "stumble":[-0.2, -1.0],
                                  "feet_regulation":[-0.01, -0.10],
                                  "edge_swing_clearance":[-0.1, -1.0],
                                  "swing_foot_collision_edge":[-0.1, -1.0],
                                  
                                  # relax these to allow for more aggressive locomotion   
                                  "lin_vel_z":[-2.0, -0.2],
                                  "base_height":[-1.0, -0.6],
                                 
                                  # Later in training, exact velo-command following may be difficult,
                                  # so, increase these values that are about rewarding/penalizing 
                                  # position-based progress in the commanded direction, not strict tracking performance.
                                  "position_no_progress":[-0.0, -0.2],
                                  "position_progress":[0.00, 0.5],

                                  # These enforce a balance gait and at the swing-level  
                                  "swing_participation_balance":[0.0, 0.10],
                                  "diagonal_pair_balance":[0.0, 0.10],
                                  "completed_swing_height_balance":[0.0, 0.10],
                                 }

            curr_steps = 10000
            warmup_steps = 10000

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 3.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]

        curriculum_ema_alpha = 0.05
        curriculum_best_window = 400
        curriculum_best_quantile = 0.90
        curriculum_recovery_ratio = 0.70
        
        curriculum_min_lin_tracking = 0.65
        curriculum_min_ang_tracking = 0.30
        
        curriculum_min_episode_fraction = 0.25
        curriculum_update_interval_steps = 10000
        
        lin_vel_x_terrain_gate_cutoff = 1.0
        lin_vel_x_terrain_gate_resume_level = 3.0
        ang_vel_yaw_terrain_gate_cutoff = 1.5
        ang_vel_yaw_terrain_gate_resume_level = 3.0

        lin_vel_x_step = 0.25
        lin_vel_y_step = 0.05
        ang_vel_yaw_step = 0.25
        max_lin_vel_y = 0.30
        max_ang_vel_yaw = 3.0
        bias_lin_vel_x_with_curriculum = True
        
        lin_vel_x_forward_bias_final = 0.60
        lin_vel_x_high_speed_bias_power_final = 0.50
        zero_command_threshold = 0.05
        zero_command_prob = 0.10
        # Warm up sampling pure zero commands, let the policy learn to walk first
        zero_command_curriculum = {'start_iter': 0, 'end_iter': 2000, 'start_value': 0.0, 'end_value': 0.1}
        limit_ang_vel_at_zero_command_prob = 0.10
        
        randomize_resampling_time = False
        resampling_time_min = 1.0
        resampling_time_max = 10.0
        use_command_resampling_time_curriculum = False
        command_resampling_time_warmup_iters = 5000
        
        heading_command = True # if true: compute ang vel command from heading error
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.6, 0.6] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

class GO2KITECfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = "KITERunner" # Teacher-Student Runner

    class policy( LeggedRobotCfgPPO.policy ):
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid, swish (SiLU)
        init_noise_std = 1.00

        # Latent dimensions
        privileged_terrain_latent_dim = depth_sequence_outdim = 32
        privileged_dynamics_latent_dim = proprio_latent_dim = 32
        depth_image_latent_dim = 32
        mixer_latent_dim = 16
        
        # Privileged Encoder/Decoder
        priv_activation = 'elu'

        cnn_norm_type                 = "layer"
        terrain_encoder_attention_dim = 128
        terrain_encoder_n_heads       = 4
        terrain_decoder_hidden_dim    = 128
        terrain_decoder_channels      = 64
        terrain_decoder_encoded_spatial_dim = (3,4)


        # Old privileged dynamics MLP-Mixer encoder params.
        # priv_mixer_num_blocks     = 3
        # priv_mixer_hidden_dim     = 64
        # priv_mixer_token_dim      = 64
        # priv_mixer_channel_dim    = 128
        # priv_mixer_use_layer_norm = False
        privileged_dynamics_context_layer_sizes = [256, 128]
        privileged_dynamics_decoder_layers = [128,256,512]
        privileged_terrain_std_min = 0.10
        privileged_terrain_std_max = 1.0
        privileged_dynamics_std_min = 0.10
        privileged_dynamics_std_max = 1.0

        # Depth Image/Sequence Models
        depth_image_norm                    = "layer"
        depth_sequence_length               = 5

        depth_sequence_norm            = "layer"
        depth_sequence_std_min         = 0.1
        depth_sequence_std_max         = 1.0
        depth_sequence_conf_min        = 0.1
        depth_sequence_conf_mask_scale = 0.2
        cnn_activation                 = 'elu'

        # Proprioceptive Context encoder
        # proprio_in_dim      = 570
        # proprio_in_dim      = 225
        proprio_in_dim      = 450
        proprio_context_layer_sizes = [256, 128]
        proprio_std_min     = 0.10
        proprio_std_max     = 1.0
        # Old proprioceptive MLP-Mixer encoder params.
        # proprio_use_norm    = False
        # proprio_num_blocks  = 3
        # proprio_hidden_dim  = 64
        # proprio_token_dim   = 64
        # proprio_channel_dim = 128

        # Modality Mixer Network
        mixer_velo_dim       = 3             # torso velocity state [v_x, v_y, v_z]
        mixer_feet_state_dim = 20            # [feet-contact-state (4), feet-height (4), surface-normal under feet (12)]
        mixer_use_norm       = False
        mixer_hidden_dims    = [128, 64, 32]
        mixer_velo_hidden    = 128
        mixer_feet_hidden    = 256
        mixer_std_min        = 0.10
        mixer_std_max        = 1.0

        # Actor/critic
        actor_layers  = [512,256,128]
        critic_layers = [1024,256,128]

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

        # Enables pytorch anomaly detecting in the runner-class
        debug_autograd_anomaly = False
        
        # Enables expensive CUDA synchronizations/cache clears for profiling
        # and OOM debugging. Keep False for normal training speed.
        gpu_debugging = False

        # Prints synchronized phase timings for collection and learning.
        # Keep disabled for normal training because CUDA syncs perturb runtime.
        profile_training = False
        profile_learning = False
        profile_iterations = 1
        profile_warmup_iterations = 0
        profile_sync_cuda = True
        
        # When False, log only compact per-model auxiliary loss totals.
        # Enable for the full detailed encoder-loss breakdown.
        log_detailed_encoder_losses = True

        # Adaptive entropy coefficient curriculum
        entropy_coef = 0.01
        use_adaptive_entropy = True
        adaptive_ent_bounds = [0.005, 0.012]
        adaptive_ent_lin_threshold = 0.80
        adaptive_ent_ang_threshold = 0.40
        adaptive_ent_ter_threshold = 6.0
        adaptive_ent_softmax_temp = 2.0
        
        # Adaptive beta scheduling for VAE KL terms:
        # beta <- clamp(exp(delta * (tau - recon_loss_ema)) * beta).
        use_adaptive_kl_beta = True
        adaptive_kl_beta_delta = 0.05
        adaptive_kl_beta_ema_alpha = 0.05
       
        #     loss weights for sequence of latent-depth-images encoder
        depth_sequence_kl_weight = 0.1
        depth_sequence_kl_recon_target = 0.05
        depth_sequence_kl_beta_min = 0.10
        depth_sequence_kl_beta_max = 2.0
        
        #     loss weights for proprioceptive history context encoder
        proprio_kl_weight = 1.0
        proprio_kl_recon_target = 0.075
        proprio_kl_beta_min = 1.0
        proprio_kl_beta_max = 2.0

        #     loss weights for privileged teacher VAEs
        privileged_terrain_kl_weight = 0.1
        privileged_terrain_kl_recon_target = 0.05
        privileged_terrain_kl_beta_min = 0.10
        privileged_terrain_kl_beta_max = 2.0

        privileged_dynamics_kl_weight = 1.0
        privileged_dynamics_kl_recon_target = 0.05
        privileged_dynamics_kl_beta_min = 1.0
        privileged_dynamics_kl_beta_max = 2.0
        
        #     reconstruction losses attached to the underlying student encoders
        depth_sequence_terrain_weight  = 1.0    # terrain reconstruction loss
        proprio_dynamics_weight = 1.0           # privileged obs reconstruction loss

        #     loss weights for depth+proprio modality mixing encoder
        modality_explicit_weight = 1.0    # torso-velo + feet-state estimation reconstruction loss
        versatility_weight = 0.001         # latent versatility loss
        versatility_lambda_e = 1.0        # weight of KL-regularization on the versility loss
        
        mixer_kl_weight = 1.0
        mixer_kl_recon_target = 0.075
        mixer_kl_beta_min = 1.0
        mixer_kl_beta_max = 2.0

        #     shared weights for contrastive loss used between variational encoder and privileged counter-parts.
        contrastive_weight = 0.01
        contrastive_lambda = 0.5
        contrastive_margin = 1.0

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic_KITE'
        algorithm_class_name = 'PPO_KITE'
        num_steps_per_env = 24 # per iteration
        max_iterations = 20000 # number of policy updates
        grf_dim = 12
        
        # debug_warmpinn_wb
        run_name = '50hz_nogap_parkour'
        # run_name = 'terrain_debug_test'
        experiment_name = 'go2_kite'
        save_interval = 500
        
        
        # load_run = "Apr15_11-53-35_50hz_spec_jointrand_stairs"
        load_run = "Jul14_19-06-40_50hz_nogap_parkour"
        checkpoint = -1
        resume = False
        exp_data_path = "exp_data/kite_feasibility/baseline_model_stairs.csv"
