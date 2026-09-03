"""Standalone Go2 HardPACT configuration.

All retained Go2 PACT settings and HardPACT-specific values are declared in
their owning classes below.  This module does not import a legacy PACT config.
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from .transition import DISTURBANCE_CRITIC_DIM

class GO2HardPACTCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 57
        num_privileged_obs = 57 + (50 + 38) + 143 + DISTURBANCE_CRITIC_DIM
        num_priv_stack = 5
        num_explicit_recon_obs = 11
        num_actions = 12
        env_spacing = 0.5
        num_obs_hist = 20
        grf_dim = 12
        whole_body_dim = 18
        debug = False       # if debugging, visualize contacts,
        debug_viz = False    # draw debug visualizations

        # Added for PACT experiment collection
        lateral_push_only = False

        # stuff for drawing the surface normal visulations
        debug_draw_swing_planes = False
        debug_viz_env                 = 0
        debug_viz_plane_size          = (0.16, 0.16)
        debug_viz_plane_color         = (0.2, 0.7, 1.0, 0.35)
        debug_viz_frame_axis_length   = 0.05
        debug_viz_frame_origin_size   = 0.008
        debug_viz_frame_axis_radius   = 0.003
        debug_viz_sample_point_radius = 0.01
        debug_viz_sample_point_color  = (1.0, 0.0, 0.0, 1.0)
        debug_viz_plane_offset        = 0.01


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
        # restitution = 0. # coefficient of restitution of the terrainr
        # obtain_terrain_info_around_feet = True

        # rough terrain only:
        mesh_type = "heightfield"
        static_friction = 1.0 # coefficient of static friction of the terrain
        dynamic_friction = 1.0 # coefficient of dynamic friction of the terrain
        restitution = 0. # coefficient of restitution of the terrain
        border_size = 20.0 # [m]
        curriculum = True
        # obtain terrain height information around feet (default: 9 points around feet), measure_
        # x  x   x
        # x F(x) x
        # x  x   x (x: height point, F: foot position)
        obtain_terrain_info_around_feet = True
        measure_heights = True # obtain height measurements

        # positions of the sampling height around the base (relative to the base of the robot)
        measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # 11x13 = 143
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 1 # starting curriculum level

        terrain_length = 8.0 # [m] length of each subterrain, X direction
        terrain_width = 8.0 # [m] width of each subterrain, Y direction
        platform_size = 4.0 # [m] size of the flat platform at the center of each subterrain
        num_rows = 10  # number of terrain rows (levels), X direction
        num_cols = 20  # number of terrain cols (types), Y direction
        num_subterrains = num_rows * num_cols
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, wave]
        terrain_proportions = [0.10, 0.10, 0.25, 0.25, 0.20, 0.10]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

        # Added for PACT experiment collection
        reset_out_of_bounds = False

    class sim:
        # Common
        # Assigned from control.dt / control.decimation after the complete
        # environment config class is defined. Keep one timing source of truth.
        dt = None
        substeps = 1
        # For Genesis
        max_collision_pairs = 100  # More collision pairs will occupy more GPU memory and slow down the simulation
        IK_max_targets = 2         # Fewer IK targets will lead to fewer memory usage
        console_debug = False
        suppress_backend_warnings = True

        class grf:
            prediction_scale_n = [120.0, 120.0, 250.0]
            vertical_deadband_n = 3.0
            clip_min_n = -250.0
            clip_max_n = 250.0
            ema_alpha = 0.30
            contact_threshold_n = 5.0
            use_ema_grfs_buf = True

    class init_state( LeggedRobotCfg.init_state ):
        leg_joint_limits = [[-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721]]
        pos = [0.0, 0.0, 0.44] # x,y,z [m]
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
        yaw_angle_range = [0., 3.14] # min max [rad]

    class normalization (LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_tau = 0.05               # in collected data the magnitude of the DOF's velocity and torques are roughly comparable
            grf = 0.01
            height_measurements = 5.0
            base_wrench = 0.01
        clip_observations = 100.
        clip_actions = 50.

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_domainrand_curriculum = True
        com_rand_z_positive = False
        num_push_steps = 1000  # number of steps to increase the domain randomization ranges
        push_warmup = 2000     # number of steps with initial values held constant
        num_jumps = 10

        # Randomize Friction
        randomize_friction = True
        friction_range = [0.2, 1.25]

        # What changes with finetuning round
        # Randomized 6DOF torso wrench
        push_robots = True
        push_interval_max = 10.0
        push_interval_min = 5.00
        max_push_vel_xy = 1.00
        min_push_vel_xy = 0.50

        max_vertical_push = 0.50
        min_vertical_push = 0.10
        vert_interval_max = 10.0
        vert_interval_min = 5.00

        max_push_torque = 1.00
        min_push_torque = 0.50
        wrench_timeout_max = 10.0
        wrench_timeout_min = 5.00

        # Randomized base mass, applied at COM
        randomize_base_mass = True
        min_added_mass_max = 2.0
        max_added_mass_max = 4.0
        added_mass_min = -1.0

        # COM displacement crap
        randomize_com_displacement = True
        com_displacement_x_min = 0.05
        com_displacement_x_max = 0.10

        com_displacement_y_min = 0.05
        com_displacement_y_max = 0.10

        com_displacement_z_positive = False
        com_displacement_z_min_pos = 0.1
        com_displacement_z_min = 0.05
        com_displacement_z_max = 0.10

        # Control delay
        randomize_ctrl_delay = True
        ctrl_delay_step_range = [0, 1]

        # PD-gain randomization
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]

        # Motor strength randomization
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        # Unused more complicated dynamics randomization
        randomize_joint_armature = True
        joint_armature_range = [0.00, 0.015]         # [N*m*s/rad]

        randomize_joint_friction = True
        joint_friction_range_end   = [0.00, 0.20]
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

        recovery_ratio = 0.90           # allowable deivation from quantile of history window
        step_interval = 10              # minimum number of iterations before taking next domain rand step

        reward_ema_alpha = 0.05         # ema value for tracking
        min_reward_to_step = 0.60       # minimum reward threashold for stepping (i.e. the performance must always be above this for a step to occur, regardless of the historical performance.)

        joint_dynamics_progress_delta = 0.02 # domain rand step delta for stepping joint-level dynamics parameters
        mass_com_progress_delta = 0.01       # domain rand step delta for stepping payload parameters
        disturbance_progress_delta = 0.01    # domain rand step delta for external disturbance parameters
        use_joint_dynamics_curriculum = True # set False to skip joint stiffness/damping/friction curriculum updates
        use_mass_com_curriculum = True       # set False to skip payload and CoM curriculum updates
        use_disturbance_curriculum = True    # set False to skip push/wrench curriculum updates

        persistent_disturbance = True
        persistent_force_probability = 0.30
        persistent_torque_probability = 0.30
        persistent_force_interval_range_s = [5.0, 15.0]
        persistent_torque_interval_range_s = [5.0, 15.0]
        persistent_force_duration_range_s = [2.0, 6.0]
        persistent_torque_duration_range_s = [2.0, 6.0]
        persistent_ramp_fraction = 0.25
        persistent_force_min_n = 10.0
        persistent_force_max_n = 60.0
        persistent_torque_min_nm = 3.0
        persistent_torque_max_nm = 12.0


    # Taken from the Go2 config class in -
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
        rendered_envs_idx = [i for i in range(0, 3, 1)]  # number of environments to be rendered
        # rendered_envs_idx.extend([i for i in range(200, 203, 1)])  # number of environments to be rendered
        # rendered_envs_idx.extend([i for i in range(500, 503, 1)])  # number of environments to be rendered
        # # rendered_envs_idx.extend([i for i in range(750, 753, 1)])  # number of environments to be rendered
        # rendered_envs_idx.extend([i for i in range(900, 903, 1)])  # number of environments to be rendered

        # rendered_envs_idx.extend([i for i in range(1500, 1503, 1)])
        # # rendered_envs_idx.extend([i for i in range(1900, 1903, 1)])
        # rendered_envs_idx.extend([i for i in range(3500, 3503, 1)])
        # rendered_envs_idx.extend([i for i in range(4000, 4003, 1)])

        # rendered_envs_idx.extend([i for i in range(1700, 1703, 1)])
        # # rendered_envs_idx.extend([i for i in range(2200, 2203, 1)])
        # # rendered_envs_idx.extend([i for i in range(3700, 3703, 1)])
        # rendered_envs_idx.extend([i for i in range(3900, 3903, 1)])
        # rendered_envs_idx = [0, 1000, 3500]
        add_camera = False

    class sensor:
        add_depth = False
        use_warp = False       # whether to use warp-based model
        class depth_camera_config:
            num_sensors = 1
            num_history = 1        # history frames for depth images

            near_clip = 0.1
            far_clip = 10.0
            near_plane = 0.1
            far_plane = 10.0
            resolution = (80, 60)
            horizontal_fov_deg = 75
            pos =   (0.3, 0.0, 0.1)
            euler = (0.0, 0.0, 0.0)
            decimation = 5
            # Warp only
            calculate_depth = True
            segmentation_camera = False
            return_pointcloud = False
            pointcloud_in_world_frame = False

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

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        # Much smaller values than typical... only used for feedback control
        stiffness = {'joint': 30.0}   # [N*m/rad]
        damping   = {'joint': 0.60}     # [N*m*s/rad]

        action_scale = 0.25   # action scale: target angle = action_scale * pose_action + defaultAngle
        torque_scale = 10.0   # action scale:  target torque = torque_scale * tau_action + defaultTorque


        dt =  0.02     # control frequency 200Hz
        decimation = 4  # decimation: Number of control action updates @ sim DT per policy DT

        # Assumed order - tau_ff, tau_fb
        # tradeoff_init_weights  = [0.20, 1.16]
        tradeoff_init_weights  = [0.40, 1.60]
        tradeoff_final_weights = [1.00, 1.00]
        tradeoff_steps = 10
        tradeoff_threshold = 0.70
        use_tradeoff_curriculum = False

        # not a tradeoff curriculum, but just slightly randomizing how much each branch contributes
        randomize_pact_weights = True
        pact_weight_bias_min = 0.0       # minimum output bias
        pact_weight_bias_max = 0.20      # maximum output bias
        pact_balanced_prob = 0.25        # % of envs that are guaranteed to have a 1-1 "balanced" output contribution


    class termination:
        termination_terms = ["roll", "pitch", "height_min", "height_max"]
        roll_threshold    = 0.70  # [rad] ~ 40 degrees
        pitch_threshold   = 1.0  # [rad] ~ 30 degrees
        height_min = 0.20       # [m]
        height_max = 1.50        # [m]

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.90
        soft_torque_limit = 0.90
        base_height_target = 0.38
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)

        foot_clearance_target = 0.09 # desired foot clearance above ground [m]
        foot_height_offset = 0.022    # height of the foot coordinate origin above ground [m]

        overreach_x_max = 0.28
        rear_foot_x_nominal = -0.25
        rear_foot_x_margin = 0.08
        support_polygon_sigma = 0.01

        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = True

        use_reward_curriculum = True

        max_contact_force = 200.0
        contact_force_threshold = 5.0

        feet_edge_threshold = 0.05
        edge_clearance_lateral_cells = (-1, 0, 1)
        edge_clearance_forward_cells = (0, 1, 2)
        edge_swing_clearance_margin = 0.04
        swing_collision_max_normal_z = 0.85
        swing_collision_min_speed = 0.05

        ff_ratio_target = 0.50
        ff_ratio_width  = 0.20
        class scales( LeggedRobotCfg.rewards.scales ):
            # General
            termination           = 0.0
            collision             = -10.0
            dof_pos_limits        = -2.0
            dof_close_to_default  = -0.01
            torque_limits         = -0.01
            pd_target_torque_limit = 0.0

            alive_bonus           = 0.001

            dof_vel_stand_still = 0.0
            stand_still_contact = 0.5
            dof_pos_stand_still = -0.1

            # command tracking
            tracking_lin_vel  = 1.0
            tracking_ang_vel  = 0.5
            dof_tracking      = 0.1
            # sparse_contacts   = 0.1

            # coupled output specific rewards
            torque_conflict_symmetric = 0.0
            torque_alignment = 0.4               # encourage a positive cosine-similarity between the ff and fb torques
            ff_ratio = 0.0                       # encourage the feeforward torques explaining more of the final torque

            # smoothness and stability
            lin_vel_z        = -2.0
            base_height      = -2.0
            ang_vel_xy       = -0.05
            orientation      = -0.2
            dof_acc          = -2.5e-7
            joint_power      = -2.e-5
            joint_power_dist = -1.e-5
            torques          = 0.0     # don't need to use this when we already have joint power above...

            # Zero out some values that are used in the individual reward classes below
            action_rate       = 0.0
            action_smoothness = 0.0

            pos_action_rate       = -0.001
            pos_action_smoothness = -0.001

            tau_action_rate       = -0.001
            tau_action_smoothness = -0.001

            # feedforward_torques   = -2.5e-5
            # feedback_torques      = -2.0e-5

            feedforward_torques_scaled = -1.0e-5       # penalize magnitude of ff torques, scales down with added payload mass (ff assumes MORE responsibility when transporting)
            feedback_torques           = -2.0e-5       # make using large PD torques 1.5x as expensive as ff torques.
            dof_act_limits             = 0.0


            # Taken from MIT benchmarking PBRS for humanoid locomotion paper
            pbrs_orientation = 10.0         # potiential reward for encouraging orientation recovery
            pbrs_height = 10.0              # potiential reward for encouraging height change recovery

            # Taken from "Stable Imitation of Multigait and Bipedal Motions for Quadrupedal Robots Over Uneven Terrains" paper
            support_polygon = 0.2             # encourages well condition foot-placement realtive to the base CoM
            vhip_angle = -0.1                 # Use a Variable-Height Inverted Pendulum (VHIP) model to penalize unstable torso orientation w.r.t. ground contact
            vhip_angular_acc = -0.001         # Use a Variable-Height Inverted Pendulum (VHIP) model to penalize moving torwards and unstable torso orientation w.r.t. ground contact

            # I developed these
            front_foot_overreach = -10000.0
            rear_foot_overreach = -10.0

            # gait
            feet_air_time    = 0.70            # tracking reward for long steps
            # foot_clearance   = 0.2            # tracking reward for feet reaching the desired clearance
            foot_clearance_terrain_aware = 0.70  # tracking reward for feet reaching the desired clearance responsive to terrain height
            hip_pos = -0.2

            foot_slip        = -0.01          # penalty for feet slipping
            stumble          = -4.0
            feet_contact_forces = -1.0e-2     # penalty for high contact forces on the feet
            feet_near_edge = -0.5
            edge_swing_clearance = -1.0
            swing_foot_collision_edge = -1.0
            feet_regulation = -0.1
            torque_cancellation = -0.10

        class reward_curriculum():
            curr_reward_keys = ["ang_vel_xy",
                                "orientation",
                                "torque_limits",
                                "hip_pos",
                                "pos_action_rate",
                                "pos_action_smoothness",
                                "tau_action_rate",
                                "tau_action_smoothness",
                                ]

            curr_reward_bounds = {
                                  "ang_vel_xy":[-0.05, -0.2],
                                  "orientation":[-0.2,-2.0],
                                  "torque_limits":[-1.0e-2, -1.0],
                                  "hip_pos":[-0.2, -0.4],
                                  "pos_action_rate":[-0.001, -0.01],
                                  "pos_action_smoothness":[-0.001,-0.01],
                                  "tau_action_rate":[-0.002, -0.02],
                                  "tau_action_smoothness":[-0.002,-0.02],
                                 }

            curr_steps = 500
            warmup_steps = 6000
        torque_cancellation_deadband = 0.03
        foot_clearance_excess_margin = 0.10
        foot_clearance_excess_weight = 0.25

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class deployment_physics:
        sustained_force_bounds_n = [-60.0, 60.0]
        sustained_torque_bounds_nm = [-12.0, 12.0]
        planned_added_mass_range_kg = [-1.0, 4.0]
        wrench_margin_absolute = 0.0
        wrench_margin_relative = 0.0
        wrench_learning_offset = [0.0] * 6
        contact_probability_epsilon = 1.0e-2
        contact_observation_offset = 0.0
        contact_observation_scale = 1.0

# A nested ``sim`` class is declared before ``control`` above, so derive its
# physics timestep only after Python has finished constructing GO2HardPACTCfg.
GO2HardPACTCfg.sim.dt = GO2HardPACTCfg.control.dt / GO2HardPACTCfg.control.decimation


class GO2HardPACTCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = "PACTRunner" # Teacher-Student Runner

    class policy( LeggedRobotCfgPPO.policy ):
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid, swish (SiLU)
        init_noise_std = 1.00

        # Context encoder
        cenet_enc_layers=[256,128]
        cenet_enc_latent_dim = 16
        cenet_velo_dim = 11

        # Context Decoder
        cenet_dec_input_dim = 16 + 11
        cenet_dec_layers = [128,256,512]
        cenet_dec_out_dim = 133

        # Actor/critic
        actor_layers = [512,256,128]
        critic_layers = [1024,256,128]

        pinn_loss_weight = 0.01
        pinn_warmup = 10
        pinn_init_steps = 0

        # pretrained_path = "../../rsl_rl/modules/pretained_checkpoints/rl_pos/pact_corl/go2_pact_pos_rough/May09_19-14-36_pact_posboot_100hz_grf/model_3000_converted.pt"
        # pretrained_path = "../../rsl_rl/modules/pretained_checkpoints/rl_pos/pact_coral/go2_pact_pos_rough/Apr23_00-50-42_pact_posboot_100hz_spec_grf/model_5000_converted.pt"
        # pretrained_path = "../../rsl_rl/modules/pretained_checkpoints/rl_pos/pact_corl/go2_pact_pos_rough/May10_16-17-52_pact_posboot_100hz_grf/model_3000_converted.pt"
        pretrained_path = ""
        cenet_explicit_layers = [128, 128]
        grf_decoder_layers = [128, 128]
        wrench_decoder_layers = [128, 128]

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

        # adaptive entropy coefficent algorithm parameters
        entropy_coef = 0.01                      # initial entropy value
        use_adaptive_entropy = True              # weather or not to use the adaptive entropy coef alg.
        adaptive_ent_bounds = [0.005, 0.01]      # entropy coefficent bands
        adaptive_ent_lin_threshold = 0.75        # minimum linear velocity tracking target
        adaptive_ent_ang_threshold = 0.35        # minimum angular velocity tracking target
        adaptive_ent_ter_threshold = 6.0         # minimum avg. terrain curriculum progress target
        adaptive_ent_softmax_temp = 2.0          # temperature (sharpness) of the softmax operation used in the alg.
        auxiliary_learning_rate = 2.0e-4
        privileged_loss_weight = 1.0
        explicit_loss_weight = 1.0
        grf_loss_weight = 1.0
        active_wrench_loss_weight = 1.0
        neutral_wrench_loss_weight = 0.25
        bard_enabled = True
        dynamics_backend = "bard"
        pinocchio_num_workers = None
        bard_randomize_base_inertia = True
        bard_scale_rotational_inertia = True
        bard_batch_capacity = 4096
        bard_inverse_enabled = True
        bard_rollout_enabled = True
        lambda_inverse = 0.01
        lambda_rollout = 0.01
        lambda_projection = 1.0e-3
        profile_bard_timing = True
        console_debug = False
        pcgrad_diagnostics_enabled = False
        pcgrad_diagnostics_start_iteration = 0
        pcgrad_diagnostics_interval = 50
        cache_rollout_mechanics = True
        ppo_qp_sampling = "disjoint_epoch_partition"
        ppo_qp_passes_per_iteration = 1
        ppo_qp_shard_percentage = 20.0
        ppo_qp_stratify_by_anchor = True
        ppo_qp_sampling_seed = None
        ppo_qp_sampling_logging_enabled = True
        hard_pact_qp = {
            "enabled": True,
            "qp_update_mode": "two_anchor_held_correction",
            # qpth remains the verified default. Optional GPU-native solvers
            # are selected explicitly and never trigger a hidden CPU/backend
            # fallback when their dependency is unavailable.
            "qp_solver": "qpth",
            "rollout_qp_solver": None,
            "ppo_qp_solver": None,
            "allow_solver_mismatch": False,
            "cupiqp_mode": "dense",
            "cupiqp_cuda_graph": False,
            "rollout_eps_abs": 1.0e-4,
            "rollout_eps_rel": 1.0e-4,
            "rollout_max_iter": 20,
            "rollout_feasibility_tolerance": 1.0e-3,
            "rollout_duality_gap_abs": 1.0e-3,
            "rollout_duality_gap_rel": 1.0e-3,
            "rollout_duality_gap_policy": "report",
            "ppo_eps_abs": 3.0e-6,
            "ppo_eps_rel": 3.0e-6,
            "ppo_max_iter": 30,
            "ppo_feasibility_tolerance": 1.0e-3,
            "ppo_duality_gap_abs": 3.0e-6,
            "ppo_duality_gap_rel": 3.0e-6,
            "ppo_duality_gap_policy": "require",
            # Opt-in because measured warm-start speed depends on the contact
            # regime. False preserves exact legacy cold-qpth execution; the
            # converged/certified QP is identical when enabled.
            # Real per-substep robot states are strongly temporally coherent;
            # repository-local qpth warm starts reduce rollout time markedly.
            # The generic solver default remains cold and this can still be
            # disabled for exact legacy/cold-reference experiments.
            "qpth_warm_start": True,
            "friction_coefficient": 0.6,
            "torque_rate_limit_nm_s": 1000.0,
            "contact_acceleration_limit_m_s2": 0.0,
            "interior_margin": 1.0e-3,
            "contact_probability_floor": 1.0e-2,
            "qdd_scale": 50.0,
            "force_scale_n": 250.0,
            "torque_scale_nm": 40.0,
            "slack_scale_m_s2": 50.0,
            "torque_tracking_weight": 20.0,
            "force_tracking_weight": 5.0,
            "slack_weight": 200.0,
            "qdd_regularization": 1.0e-3,
            "force_regularization": 1.0e-4,
            "torque_regularization": 1.0e-4,
            "q_regularization": 1.0e-7,
            "proximal_rho": 0.10,
            "proximal_block_weights": (1.0, 1.0, 1.0, 1.0),
            "elastic_recovery_enabled": True,
            "elastic_dynamics_weight": 1.0e4,
            "gradient_scale_tau": 1.0,
            "gradient_scale_grf": 1.0,
            "gradient_scale_wrench": 1.0,
            "gradient_scale_contact": 1.0,
            "gradient_clip_tau": 0.0,
            "gradient_clip_grf": 0.0,
            "gradient_clip_wrench": 0.0,
            "gradient_clip_contact": 0.0,
            "normalized_feasibility_tolerance_float32": 1.0e-3,
            "normalized_feasibility_tolerance_float64": 1.0e-6,
            "kkt_tolerance": 1.0e-1,
            "active_tolerance": 1.0e-1,
            "eps_float32": 1.0e-5,
            "eps_float64": 1.0e-9,
            "max_iter": 30,
            "not_improved_limit": 6,
            "check_q_spd": True,
            "check_equality_rank": True,
            "solver_dtype": "auto",
            "verbose": 0,
            # Production logging computes only mandatory normalized primal
            # certification and compact fallback summaries. Use "physical"
            # for physical-unit constraint statistics or "full" for periodic
            # sampled matrix/KKT/timing/memory/gradient audits.
            "diagnostics_level": "minimal",
            "full_audit_period": 1000,
            "full_audit_sample_size": 8,
            "rollout_chunk_size": 4096,
            "ppo_chunk_size": 8000,
            # Genesis and both PhysX backends advance position with the new
            # velocity (semi-implicit Euler), hence q+=dt*v+dt^2*qdd.
            "position_integration_coefficient": 1.0,
        }
        grf_observation_scale = GO2HardPACTCfg.normalization.obs_scales.grf
        base_wrench_observation_scale = GO2HardPACTCfg.normalization.obs_scales.base_wrench
        action_clip = GO2HardPACTCfg.normalization.clip_actions

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCritic_HardPACT"
        algorithm_class_name = "PPO_HardPACT"
        num_steps_per_env = 24
        max_iterations = 6000
        grf_dim = 12

        # debug_warmpinn_wb
        run_name = 'pact_100hz_spec_smartcurr_stricterer'
        experiment_name = 'go2_pact_rough'
        save_interval = 500


        # load_run = "May01_16-41-42_pact_100hz_spec_smartcurr"
        # load_run = "May06_20-34-35_pact_100hz_spec_smartcurr"
        # load_run = "May07_18-30-49_pact_100hz_spec_smartcurr"   # this is the most promising model/one with collected data, vhip but no rear-overreach
        # load_run = "May07_18-49-22_pact_100hz_spec_smartcurr_e2e"
        # load_run = "May09_01-26-21_pact_100hz_spec_smartcurr"    # weaker pos-boot, rear-overreach
        # load_run = "May10_02-07-54_pact_100hz_spec_smartcurr"    # most recent model with strong boot and rear-overreah
        # load_run = "May10_20-41-46_pact_100hz_spec_smartcurr"    # most recent model with strong boot and rear-overreah, 3000 pos-boot start
        # load_run = "May11_21-55-58_pact_100hz_spec_smartcurr"    # best performing aligned model
        # load_run = "May14_18-35-56_pact_100hz_spec_smartcurr_stricterer"
        load_run = "Aug01_18-27-22_pact_100hz_spec_smartcurr_stricterer"
        checkpoint = -1
        resume = False
        exp_data_path = "exp_data/corl_tests_01/pact_stairs_12-16kg.csv"
        console_iteration = True
        console_model_summary = False
        console_reward_terms = True
        console_detailed_losses = False
        console_pinn_timing = True
        console_qp_timing = True
