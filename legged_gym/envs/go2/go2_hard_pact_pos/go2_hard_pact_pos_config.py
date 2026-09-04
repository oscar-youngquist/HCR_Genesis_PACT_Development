"""Standalone Go2 HardPACTPos configuration with all retained PACTPos settings inline."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.go2.go2_hard_pact.transition import DISTURBANCE_CRITIC_DIM

class GO2HardPACTPosCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 57
        num_privileged_obs = 57 + (50 + 38) + 143 + DISTURBANCE_CRITIC_DIM
        num_priv_stack = 5
        num_explicit_recon_obs = 11
        num_actions = 12
        env_spacing = 0.5
        num_obs_hist = 10
        grf_dim = 12
        whole_body_dim = 18
        debug = False
        debug_viz = False
        # Match HardPACT's disturbance contract.  False preserves the legacy
        # PACT/PactPos planar push sampling (both world x and y components).
        lateral_push_only = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'heightfield'
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        border_size = 20.0
        curriculum = True
        obtain_terrain_info_around_feet = True
        measure_heights = True
        measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 1
        terrain_length = 8.0
        terrain_width = 8.0
        platform_size = 4.0
        num_rows = 10
        num_cols = 20
        num_subterrains = num_rows * num_cols
        terrain_proportions = [0.10, 0.20, 0.20, 0.20, 0.20, 0.10]
        slope_treshold = 0.75

    class sim:
        dt = None
        substeps = 1
        max_collision_pairs = 100
        IK_max_targets = 2
        console_debug = False
        suppress_backend_warnings = True

        class grf:
            prediction_scale_n = [250.0, 250.0, 500.0]
            vertical_deadband_n = 3.0
            clip_min_n = -500.0
            clip_max_n = 500.0
            ema_alpha = 0.2
            contact_threshold_n = 5.0
            use_ema_grfs_buf = False

    class init_state(LeggedRobotCfg.init_state):
        leg_joint_limits = [[-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721], [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721], [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721], [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721]]
        pos = [0.0, 0.0, 0.44]
        default_joint_angles = {'FL_hip_joint': 0.1, 
                                'RL_hip_joint': 0.1, 
                                'FR_hip_joint': -0.1, 
                                'RR_hip_joint': -0.1, 
                                'FL_thigh_joint': 0.8, 
                                'RL_thigh_joint': 0.8, 
                                'FR_thigh_joint': 0.8, 
                                'RR_thigh_joint': 0.8, 
                                'FL_calf_joint': -1.5, 
                                'RL_calf_joint': -1.5, 
                                'FR_calf_joint': -1.5, 
                                'RR_calf_joint': -1.5}
        default_joint_torques = {'FR_hip_joint': 0.0, 'FL_hip_joint': 0.0, 'RR_hip_joint': 0.0, 'RL_hip_joint': 0.0, 'FL_thigh_joint': 0.0, 'RL_thigh_joint': 0.0, 'FR_thigh_joint': 0.0, 'RR_thigh_joint': 0.0, 'FL_calf_joint': 0.0, 'RL_calf_joint': 0.0, 'FR_calf_joint': 0.0, 'RR_calf_joint': 0.0}
        yaw_angle_range = [0.0, 3.14]

    class normalization(LeggedRobotCfg.normalization):

        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_tau = 0.05
            grf = 0.01
            height_measurements = 5.0
            base_wrench = 0.01
        clip_observations = 100.0
        clip_actions = 50.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_domainrand_curriculum = True
        com_rand_z_positive = False
        num_push_steps = 500
        push_warmup = 3000

        randomize_friction = True
        friction_range = [0.2, 1.8]

        push_interval_max = 10.0
        push_interval_min = 5.0
        max_push_vel_xy = 0.5
        min_push_vel_xy = 0.5

        max_vertical_push = 0.1
        min_vertical_push = 0.1
        vert_interval_max = 10.0
        vert_interval_min = 5.0

        max_push_torque = 0.5
        min_push_torque = 0.5
        wrench_timeout_min = 5.0
        wrench_timeout_max = 10.0

        randomize_base_mass = True
        min_added_mass_max = 2.0
        max_added_mass_max = 2.0
        added_mass_min = -1.0

        randomize_com_displacement = True
        com_displacement_x_min = 0.05
        com_displacement_x_max = 0.05

        com_displacement_y_min = 0.05
        com_displacement_y_max = 0.05

        com_displacement_z_positive = False
        com_displacement_z_min_pos = 0.1
        com_displacement_z_min = 0.05
        com_displacement_z_max = 0.05

        randomize_ctrl_delay = True
        ctrl_delay_step_range = [0, 1]

        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_motor_strength = True

        motor_strength_range = [0.9, 1.1]

        randomize_joint_armature = True
        joint_armature_range = [0.0, 0.02]

        randomize_joint_friction = True
        joint_friction_range_end = [0.0, 0.02]
        joint_friction_range_start = [0.0, 0.01]

        randomize_joint_stiffness = False
        joint_stiffness_range_end = [0.0, 0.0]
        joint_stiffness_range_start = [0.0, 0.0]

        randomize_joint_damping = True
        joint_damping_range_end = [0.25, 0.5]
        joint_damping_range_start = [0.3, 0.4]

        best_reward_window = 200
        best_reward_quantile = 0.9
        recovery_ratio = 0.9
        step_interval = 10
        reward_ema_alpha = 0.05
        min_reward_to_step = 0.6
        joint_dynamics_progress_delta = 0.02
        mass_com_progress_delta = 0.01
        disturbance_progress_delta = 0.01
        use_joint_dynamics_curriculum = True
        use_mass_com_curriculum = False
        use_disturbance_curriculum = False

        persistent_disturbance = True
        persistent_force_probability = 0.1
        persistent_torque_probability = 0.1
        persistent_force_interval_range_s = [5.0, 10.0]
        persistent_torque_interval_range_s = [5.0, 10.0]
        persistent_force_duration_range_s = [2.0, 4.0]
        persistent_torque_duration_range_s = [2.0, 4.0]
        persistent_ramp_fraction = 0.25
        persistent_force_min_n = 5.0
        persistent_force_max_n = 5.0
        persistent_torque_min_nm = 3.0
        persistent_torque_max_nm = 4.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

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
        pos = [2, 2, 2]
        lookat = [0.0, 0, 1.0]
        rendered_envs_idx = [i for i in range(0, 3, 1)]
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

    class asset(LeggedRobotCfg.asset):
        name = 'go2'
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
        foot_name = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        penalize_contacts_on = ['thigh', 'hip', 'calf', 'base', 'Head']
        terminate_after_contacts_on = ['base', 'Head']
        links_to_keep = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        self_collisions = True
        obtain_link_contact_states = True
        contact_state_link_names = ['thigh', 'calf', 'foot', 'base', 'hip']

    class control(LeggedRobotCfg.control):
        stiffness = {'joint': 30.0}
        damping = {'joint': 0.6}
        action_scale = 0.25
        torque_scale = 10.0
        dt = 0.02
        decimation = 4
        tradeoff_init_weights = [1.0, 1.0]
        tradeoff_final_weights = [1.0, 1.0]
        tradeoff_steps = 10
        tradeoff_threshold = 0.6
        use_tradeoff_curriculum = False

    class termination:
        termination_terms = ['roll', 'pitch', 'height_min', 'height_max']
        roll_threshold = 0.7
        pitch_threshold = 1.0
        height_min = 0.2
        height_max = 1.5

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        soft_torque_limit = 0.9

        base_height_target = 0.38

        tracking_sigma = 0.25

        foot_clearance_target = 0.09
        foot_height_offset = 0.022

        # Legacy
        overreach_x_max = 0.28
        rear_foot_x_nominal = -0.20
        rear_foot_x_margin = 0.08

        support_polygon_sigma = 0.01

        front_foot_x_nominal = 0.20
        rear_foot_x_nominal = 0.20
        foot_x_margin = 0.10


        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = True
        use_reward_curriculum = True

        max_contact_force = 400.0
        contact_force_threshold = 5.0

        feet_edge_threshold = 0.05
        edge_clearance_lateral_cells = (-1, 0, 1)
        edge_clearance_forward_cells = (0, 1, 2)
        edge_swing_clearance_margin = 0.04
        swing_collision_max_normal_z = 0.85
        swing_collision_min_speed = 0.05

        # Parameters consumed by the shared HardPACT implementations of the
        # B1Z1 torque-conflict and terrain-aware foot-clearance rewards.  The
        # clearance values retain HardPACTPos's former in-function constants.
        torque_cancellation_deadband = 0.03
        foot_clearance_excess_margin = 0.04
        foot_clearance_excess_weight = 0.25

        class scales( LeggedRobotCfg.rewards.scales ):
            # General
            termination           = 0.0
            collision             = -1.0
            dof_pos_limits        = -2.0
            dof_close_to_default  = -0.01
            torque_limits         = -0.0001

            alive_bonus           = 0.001

            stand_still_contact = -0.5
            dof_pos_stand_still = -0.1
            dof_vel_stand_still = -0.0

            # command tracking
            tracking_lin_vel  = 1.0
            tracking_ang_vel  = 0.5
            
            dof_tracking      = 0.00
            aligned_torques   = 0.00
            sparse_contacts   = 0.01
            
            # smoothness and stability
            lin_vel_z        = -2.0
            base_height      = -2.0
            ang_vel_xy       = -0.05
            orientation      = -0.2
            dof_acc          = -2.0e-7
            joint_power      = -2.0e-5
            joint_power_dist = -1.0e-5
            torques          = -1.0e-5     # don't need to use this when we already have joint power above...

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

            # Taken from MIT benchmarking PBRS for humanoid locomotion paper
            pbrs_orientation = 10.0           # potiential reward for encouraging orientation recovery
            pbrs_height = 10.0                # potiential reward for encouraging height change recovery

            # Taken from "Stable Imitation of Multigait and Bipedal Motions for Quadrupedal Robots Over Uneven Terrains" paper
            support_polygon = 0.2             # encourages well condition foot-placement realtive to the base CoM (and vice-versa)
            vhip_angle = -0.1                 # Use a Variable-Height Inverted Pendulum (VHIP) model to penalize unstable torso orientation w.r.t. ground contact 
            vhip_angular_acc = -0.001         # Use a Variable-Height Inverted Pendulum (VHIP) model to penalize moving torwards and unstable torso orientation w.r.t. ground contact
            
            # I developed these
            front_foot_overreach = -10.0
            rear_foot_overreach = -10.0

            # gait
            feet_air_time    = 0.70            # tracking reward for long steps
            # foot_clearance   = 0.20            # tracking reward for feet reaching the desired clearance
            foot_clearance_terrain_aware = 0.30  # tracking reward for feet reaching the desired clearance responsive to terrain height    
            hip_pos = -0.05
            
            foot_slip        = -0.01           # penalty for feet slipping
            stumble          = -0.2
            feet_contact_forces = -1.0e-2     # penalty for high contact forces on the feet

            feet_near_edge = -0.1
            edge_swing_clearance = -0.0
            swing_foot_collision_edge = -0.0
            feet_regulation = -0.01

        class reward_curriculum:
            curr_reward_keys = ['orientation',
                                'ang_vel_xy', 
                                'dof_close_to_default', 
                                'torque_limits']
            curr_reward_bounds = {'orientation': [-0.2, -1.0], 
                                  'ang_vel_xy': [-0.05, -0.1], 
                                  'dof_close_to_default': [-0.05, -0.1], 
                                  'torque_limits': [-0.0001, -0.01]}
            curr_steps = 1
            warmup_steps = 4000

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class deployment_physics:
        sustained_force_bounds_n = [-60.0, 60.0]
        sustained_torque_bounds_nm = [-12.0, 12.0]
        planned_added_mass_range_kg = [-1.0, 4.0]

GO2HardPACTPosCfg.sim.dt = GO2HardPACTPosCfg.control.dt / GO2HardPACTPosCfg.control.decimation

class GO2HardPACTPosCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'PACTPosRunner'

    class policy(LeggedRobotCfgPPO.policy):
        activation = 'elu'
        init_noise_std = 1.0

        cenet_enc_layers = [256, 128]
        cenet_enc_latent_dim = 32
        cenet_velo_dim = 11
        cenet_dec_input_dim = 32 + 11
        cenet_dec_layers = [128, 256, 512]
        cenet_dec_out_dim = 133

        actor_layers = [512, 256, 128]
        critic_layers = [1024, 256, 128]

        dropout = 0.1

        pinn_loss_weight = 0.01
        pinn_warmup = 10000
        pinn_init_steps = 0

        cenet_explicit_layers = [128, 128]
        grf_decoder_layers = [128, 128]
        wrench_decoder_layers = [128, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 0.0003
        # Weight beta on the latent KL term in the combined auxiliary loss.
        vae_kld_weight = 1.0
        vae_kl_initial_weight = 0.001
        vae_kl_warmup_start = 0
        vae_kl_warmup_iterations = 1000

        privileged_loss_weight = 1.0
        explicit_loss_weight = 1.0
        grf_loss_weight = 1.0
        active_wrench_loss_weight = 1.0
        neutral_wrench_loss_weight = 0.25
        value_loss_coef = 1.0

        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4

        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95

        desired_kl = 0.01
        max_grad_norm = 1.0

        # adaptive entropy coefficent algorithm parameters
        entropy_coef = 0.01                      # initial entropy value
        use_adaptive_entropy = False              # weather or not to use the adaptive entropy coef alg.
        adaptive_ent_bounds = [0.01, 0.02]      # entropy coefficent bands
        adaptive_ent_lin_threshold = 0.75        # minimum linear velocity tracking target
        adaptive_ent_ang_threshold = 0.35        # minimum angular velocity tracking target
        adaptive_ent_ter_threshold = 6.0         # minimum avg. terrain curriculum progress target
        adaptive_ent_softmax_temp = 2.0          # temperature (sharpness) of the softmax operation used in the alg. 

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic_HardPACT_Pos'
        algorithm_class_name = 'PPO_PACT_Pos'
        num_steps_per_env = 24
        max_iterations = 5000
        grf_dim = 12
        run_name = 'pact_posboot_100hz_grf'
        experiment_name = 'go2_pact_pos_rough'
        save_interval = 500
        load_run = 'May08_17-10-01_pact_posboot_100hz_nogrf'
        checkpoint = -1
        resume = False
        exp_data_path = 'exp_data/pact_pos_tests/spec_0_01_4-6kg_stairs.csv'
        export_hard_pact_start = True
        hard_pact_start_filename = 'hard_pact_start_model_{iteration}.pt'
        # Match HardPACT's concise console policy while retaining reward
        # summaries. These flags do not affect TensorBoard logging.
        console_iteration = True
        console_model_summary = False
        console_reward_terms = True
        console_detailed_losses = False
        console_pinn_timing = True
        console_qp_timing = True
