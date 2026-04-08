from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class A1PACTCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 57
        num_privileged_obs = 57 + 66 + 2 + 81 # robot_state + privilged info + tradeoff curriculum weights + terrain_heights (81)
        num_priv_stack = 5
        num_explicit_recon_obs = 3 + 4 + 4 # torso lin-velo, feet contact states, feet height
        num_actions = 12
        env_spacing = 0.5
        num_obs_hist = 10
        grf_dim = 12
        whole_body_dim = 18
        debug = False
        debug_viz = False

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "heightfield"
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        border_size = 20.0 # [m]
        curriculum = True
        obtain_terrain_info_around_feet = True
        measure_heights = True

        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4] # 9x9=81
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]

        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 1

        terrain_length = 8.0
        terrain_width = 8.0
        platform_size = 4.0
        num_rows = 20
        num_cols = 10
        num_subterrains = num_rows * num_cols
        terrain_proportions = [0.20, 0.10, 0.20, 0.20, 0.20, 0.1]
        slope_treshold = 0.75

    class sim:
        dt = 0.002                 # 500 Hz
        substeps = 1
        max_collision_pairs = 100
        IK_max_targets = 2

    class init_state( LeggedRobotCfg.init_state ):
        leg_joint_limits = [[-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721]]
        pos = [0.0, 0.0, 0.34]
        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RR_hip_joint': 0.0,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

        default_joint_torques = {
            'FR_hip_joint':  0.0,
            'FL_hip_joint':  0.0,
            'RR_hip_joint':  0.0,
            'RL_hip_joint':  0.0,
            'FL_thigh_joint': 0.0,
            'RL_thigh_joint': 0.0,
            'FR_thigh_joint': 0.0,
            'RR_thigh_joint': 0.0,
            'FL_calf_joint': 0.0,
            'RL_calf_joint': 0.0,
            'FR_calf_joint': 0.0,
            'RR_calf_joint': 0.0,
        }
        yaw_angle_range = [0., 3.14]

    class normalization (LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_tau = 0.05
            grf = 0.01
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 50.

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_domainrand_curriculum = True
        com_rand_z_positive = True
        num_push_steps = 2000
        push_warmup = 2000

        randomize_friction = True
        friction_range = [0.2, 1.8]

        push_robots = True
        push_interval_max = 15.0
        push_interval_min = 0.1
        max_push_vel_xy = 1.00
        min_push_vel_xy = 0.5

        max_vertical_push = 0.40
        min_vertical_push = 0.00
        vert_interval_max = 10.0
        vert_interval_min = 0.1

        max_push_torque = 1.5
        min_push_torque = 0.50
        wrench_timeout_min = 0.01
        wrench_timeout_max = 10.0

        randomize_base_mass = True
        min_added_mass_max = 4.0
        max_added_mass_max = 8.0
        added_mass_min = -1.0

        randomize_com_displacement = True
        com_displacement_x_min = 0.075
        com_displacement_x_max = 0.16

        com_displacement_y_min = 0.075
        com_displacement_y_max = 0.12

        com_displacement_z_positive = False
        com_displacement_z_min_pos = 0.1
        com_displacement_z_min = 0.05
        com_displacement_z_max = 0.25

        randomize_ctrl_delay = True
        ctrl_delay_step_range = [0, 2]

        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]

        randomize_joint_armature = False
        joint_armature_range = [0.015, 0.025]
        randomize_joint_stiffness = False
        joint_stiffness_range = [0.01, 0.02]
        randomize_joint_damping = False
        joint_damping_range = [0.25, 0.3]

    class noise (LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.0006
            dof_vel = 0.02
            dof_tau = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.06
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [2, 2, 2]
        lookat = [0., 0, 1.]
        rendered_envs_idx = [i for i in range(0, 3, 1)]
        rendered_envs_idx.extend([i for i in range(500, 503, 1)])
        rendered_envs_idx.extend([i for i in range(900, 903, 1)])
        rendered_envs_idx.extend([i for i in range(1500, 1503, 1)])
        rendered_envs_idx.extend([i for i in range(3500, 3503, 1)])
        rendered_envs_idx.extend([i for i in range(4000, 4003, 1)])
        rendered_envs_idx.extend([i for i in range(1700, 1703, 1)])
        rendered_envs_idx.extend([i for i in range(2200, 2203, 1)])
        rendered_envs_idx.extend([i for i in range(3900, 3903, 1)])
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
            pos =   (0.3, 0.0, 0.1)
            euler = (0.0, 0.0, 0.0)
            decimation = 5
            calculate_depth = True
            segmentation_camera = False
            return_pointcloud = False
            pointcloud_in_world_frame = False

    class asset( LeggedRobotCfg.asset ):
        name = "a1"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        dof_names = [
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
        foot_name = "foot"
        penalize_contacts_on = ["hip", "thigh", "calf"]
        terminate_after_contacts_on = ["base", "trunk", "hip"]
        links_to_keep = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        self_collisions = True
        obtain_link_contact_states = True
        contact_state_link_names = ["thigh", "calf", "foot", "base", "hip"]

    class control( LeggedRobotCfg.control ):
        stiffness = {'joint': 40.0}
        damping   = {'joint': 0.80}

        action_scale = 0.25
        torque_scale = 10.0

        dt =  0.01
        decimation = 5

        tradeoff_init_weights  = [1.00, 1.00]
        tradeoff_final_weights = [1.00, 1.00]
        tradeoff_steps = 50
        tradeoff_threshold = 0.60
        use_tradeoff_curriculum = False

    class termination:
        termination_terms = ["roll", "pitch", "height_min", "height_max"]
        roll_threshold    = 1.5
        pitch_threshold   = 1.5
        height_min = 0.20
        height_max = 1.50

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.90
        soft_torque_limit = 0.90
        base_height_target = 0.30
        tracking_sigma = 0.25

        foot_clearance_target = 0.075
        foot_height_offset = 0.022

        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = False

        use_reward_curriculum = False

        max_contact_force = 400.0
        class scales( LeggedRobotCfg.rewards.scales ):
            termination           = 0.0
            collision             = -1.0
            dof_pos_limits        = -1.0
            dof_close_to_default  = 0.0
            torque_limits         = -0.01

            alive_bonus           = 0.11

            stand_still_contact = -0.1
            dof_pos_stand_still = -0.1

            tracking_lin_vel  = 1.0
            tracking_ang_vel  = 0.5
            dof_tracking      = 0.1
            sparse_contacts   = 0.1

            torque_conflict_symmetric = -0.1
            torque_alignment = 0.4

            lin_vel_z        = -2.0
            base_height      = -1.0
            ang_vel_xy       = -0.05
            orientation      = -0.2
            dof_acc          = -2.5e-7
            joint_power      = -2.e-5
            joint_power_dist = -1.e-5
            torques          = 0.0

            action_rate       = 0.0
            action_smoothness = 0.0

            pos_action_rate       = -0.01
            pos_action_smoothness = -0.01

            tau_action_rate       = -0.02
            tau_action_smoothness = -0.02

            feedforward_torques   = -2.5e-7
            feedback_torques      = -2.0e-7
            dof_act_limits        = -1.0

            feet_air_time    = 0.75
            foot_clearance   = 0.2
            hip_pos = -0.05

            foot_slip        = -0.1
            feet_contact_forces = -1.0e-3
            feet_spread_pairwise_axes = 0.0

        class reward_curriculum():
            curr_reward_keys = ["ang_vel_xy", "orientation",
                                "dof_act_limits", "torque_limits",
                                "tau_action_rate", "tau_action_smoothness",
                                "feedforward_torques","feedback_torques",
                                "feet_contact_forces",
                                "dof_tracking"]

            curr_reward_bounds = {
                                  "ang_vel_xy":[-0.05, -0.1],
                                  "orientation":[-0.2,-0.4],
                                  "dof_act_limits":[-0.1, -0.2],
                                  "torque_limits":[-0.01, -0.05],
                                  "tau_action_rate":[-0.01, -0.05],
                                  "tau_action_smoothness":[-0.01, -0.05],
                                  "feedback_torques":[-2.0e-7, -2.5e-6],
                                  "feedforward_torques":[-2.0e-7, -2.0e-6],
                                  "feet_contact_forces":[-1.0e-4, -1.0e-2],
                                  "dof_tracking":[0.1, 0.25],
                                 }

            curr_steps = 1505
            warmup_steps = 1500

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.
        num_commands = 3
        resampling_time = 5.
        heading_command = False
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-3.14, 3.14]

class A1PACTCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = "PACTRunner"

    class policy( LeggedRobotCfgPPO.policy ):
        activation = 'tanh'
        init_noise_std = 0.50

        cenet_enc_layers=[256,128]
        cenet_enc_latent_dim = 16
        cenet_velo_dim = 3 + 4 + 4

        cenet_dec_input_dim = 27
        cenet_dec_layers = [128,256]
        cenet_dec_out_dim = 57 + 12

        actor_layers = [512,256,128]
        critic_layers = [1024,256,128,64]

        pinn_loss_weight = 0.01
        pinn_warmup = 10
        pinn_init_steps = 0

        pretrained_path = "rsl_rl/modules/pretained_checkpoints/rl_pos/a1_pact_pos_rough/model_2000_converted.pt"

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 3.0e-4
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4
        schedule = 'adaptive'
        gamma = 0.99
        lam   = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic_PACT'
        algorithm_class_name = 'PPO_PACT'
        num_steps_per_env = 100
        max_iterations = 6000
        grf_dim = 12

        run_name = 'pact_100hz'
        experiment_name = 'a1_pact_rough'
        save_interval = 100

        load_run = ""
        checkpoint = -1
        resume = False
        exp_data_path = ""
