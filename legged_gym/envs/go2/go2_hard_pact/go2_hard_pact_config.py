"""Standalone Go2 HardPACT configuration with all retained PACT settings inline."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from .transition import DISTURBANCE_CRITIC_DIM

class GO2HardPACTCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
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
        debug = False
        debug_viz = False
        lateral_push_only = False
        debug_draw_swing_planes = False
        debug_viz_env = 0
        debug_viz_plane_size = (0.16, 0.16)
        debug_viz_plane_color = (0.2, 0.7, 1.0, 0.35)
        debug_viz_frame_axis_length = 0.05
        debug_viz_frame_origin_size = 0.008
        debug_viz_frame_axis_radius = 0.003
        debug_viz_sample_point_radius = 0.01
        debug_viz_sample_point_color = (1.0, 0.0, 0.0, 1.0)
        debug_viz_plane_offset = 0.01

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
        terrain_proportions = [0.10, 0.20, 0.20, 0.20, 0.20, 0.1]
        slope_treshold = 0.75
        reset_out_of_bounds = False

    class sim:
        dt = None
        substeps = 1
        max_collision_pairs = 100
        IK_max_targets = 2
        console_debug = False
        suppress_backend_warnings = True

        class grf:
            prediction_scale_n = [250.0, 250.0, 250.0]
            vertical_deadband_n = 3.0
            clip_min_n = -250.0
            clip_max_n = 250.0
            ema_alpha = 0.3
            contact_threshold_n = 5.0
            use_ema_grfs_buf = True

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
        num_push_steps = 1000
        push_warmup = 2000
        num_jumps = 10

        randomize_friction = True
        friction_range = [0.2, 1.25]

        push_robots = True
        push_interval_max = 10.0
        push_interval_min = 5.0
        max_push_vel_xy = 1.0
        min_push_vel_xy = 0.5

        max_vertical_push = 0.5
        min_vertical_push = 0.1
        vert_interval_max = 10.0
        vert_interval_min = 5.0

        max_push_torque = 1.0
        min_push_torque = 0.5
        wrench_timeout_max = 10.0
        wrench_timeout_min = 5.0

        randomize_base_mass = True
        min_added_mass_max = 2.0
        max_added_mass_max = 4.0
        added_mass_min = -1.0

        randomize_com_displacement = True
        com_displacement_x_min = 0.05
        com_displacement_x_max = 0.1

        com_displacement_y_min = 0.05
        com_displacement_y_max = 0.1

        com_displacement_z_positive = False
        com_displacement_z_min_pos = 0.1
        com_displacement_z_min = 0.05
        com_displacement_z_max = 0.1

        randomize_ctrl_delay = True
        ctrl_delay_step_range = [0, 1]
        
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_joint_armature = True
        joint_armature_range = [0.0, 0.015]

        randomize_joint_friction = True
        joint_friction_range_end = [0.0, 0.2]
        joint_friction_range_start = [0.0, 0.05]

        randomize_joint_stiffness = True
        joint_stiffness_range_end = [0.0, 0.02]
        joint_stiffness_range_start = [0.0, 0.005]

        randomize_joint_damping = True
        joint_damping_range_end = [0.0, 0.8]
        joint_damping_range_start = [0.2, 0.6]

        best_reward_window = 400
        best_reward_quantile = 0.9
        recovery_ratio = 0.9
        step_interval = 10
        reward_ema_alpha = 0.05
        min_reward_to_step = 0.6
        joint_dynamics_progress_delta = 0.02
        mass_com_progress_delta = 0.01
        disturbance_progress_delta = 0.01
        use_joint_dynamics_curriculum = True
        use_mass_com_curriculum = True
        use_disturbance_curriculum = True

        persistent_disturbance = True
        persistent_force_probability = 0.3
        persistent_torque_probability = 0.3
        persistent_force_interval_range_s = [5.0, 15.0]
        persistent_torque_interval_range_s = [5.0, 15.0]
        persistent_force_duration_range_s = [2.0, 6.0]
        persistent_torque_duration_range_s = [2.0, 6.0]
        persistent_ramp_fraction = 0.25
        persistent_force_min_n = 10.0
        persistent_force_max_n = 60.0
        persistent_torque_min_nm = 3.0
        persistent_torque_max_nm = 12.0

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
        pos = [0.5, 1.5, 0.5]
        lookat = [0.0, 0, 0.0]
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
        dof_names = ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
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
        tradeoff_init_weights = [0.4, 1.6]
        tradeoff_final_weights = [1.0, 1.0]
        tradeoff_steps = 10
        tradeoff_threshold = 0.7
        use_tradeoff_curriculum = False
        randomize_pact_weights = True
        pact_weight_bias_min = 0.0
        pact_weight_bias_max = 0.2
        pact_balanced_prob = 0.25

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
        overreach_x_max = 0.28
        front_foot_x_nominal = 0.20
        foot_x_margin = 0.10
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
        ff_ratio_target = 0.5
        ff_ratio_width = 0.2

        torque_cancellation_deadband = 0.03
        foot_clearance_excess_margin = 0.1
        foot_clearance_excess_weight = 0.25
        class scales(LeggedRobotCfg.rewards.scales):
            termination = 0.0
            collision = -10.0
            dof_pos_limits = -2.0
            dof_close_to_default = -0.01
            torque_limits = -0.01
            pd_target_torque_limit = 0.0

            alive_bonus = 0.001

            dof_vel_stand_still = 0.0
            stand_still_contact = 0.5
            dof_pos_stand_still = -0.1

            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            dof_tracking = 0.1

            torque_conflict_symmetric = 0.0
            torque_alignment = 0.0
            ff_ratio = 0.0
            torque_cancellation = -0.1

            lin_vel_z = -2.0
            base_height = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2

            dof_acc = -2.5e-07
            joint_power = -2e-05
            joint_power_dist = -1e-05
            torques = 0.0

            action_rate = 0.0
            action_smoothness = 0.0

            pos_action_rate = -0.001
            pos_action_smoothness = -0.001

            tau_action_rate = -0.001
            tau_action_smoothness = -0.001

            feedforward_torques_scaled = -1e-05
            feedback_torques = -2e-05

            dof_act_limits = 0.0

            pbrs_orientation = 10.0
            pbrs_height = 10.0

            support_polygon = 0.2
            vhip_angle = -0.1
            vhip_angular_acc = -0.001

            front_foot_overreach = -100.0
            rear_foot_overreach = -10.0

            feet_air_time = 0.7
            foot_clearance_terrain_aware = 0.3
            hip_pos = -0.2
            foot_slip = -0.01
            stumble = -1.0
            feet_contact_forces = -0.01
            feet_near_edge = -0.5
            edge_swing_clearance = -0.5
            swing_foot_collision_edge = -1.0
            feet_regulation = -0.1

        class reward_curriculum:
            curr_reward_keys = ['ang_vel_xy', 
                                'orientation', 
                                'torque_limits',
                                'hip_pos',
                                'pos_action_rate', 
                                'pos_action_smoothness', 
                                'tau_action_rate', 
                                'tau_action_smoothness']
            curr_reward_bounds = {'ang_vel_xy': [-0.05, -0.2], 
                                  'orientation': [-0.2, -2.0], 
                                  'torque_limits': [-0.01, -1.0], 
                                  'hip_pos': [-0.2, -0.4], 
                                  'pos_action_rate': [-0.001, -0.01], 
                                  'pos_action_smoothness': [-0.001, -0.01], 
                                  'tau_action_rate': [-0.002, -0.02], 
                                  'tau_action_smoothness': [-0.002, -0.02]}
            curr_steps = 1000
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
        wrench_scale = [100.0, 100.0, 100.0, 25.0, 25.0, 25.0]
        wrench_qp_clip = [150.0, 150.0, 150.0, 40.0, 40.0, 40.0]
GO2HardPACTCfg.sim.dt = GO2HardPACTCfg.control.dt / GO2HardPACTCfg.control.decimation

class GO2HardPACTCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'PACTRunner'

    class policy(LeggedRobotCfgPPO.policy):
        activation = 'elu'
        init_noise_std = 1.0
        cenet_enc_layers = [256, 128]
        cenet_enc_latent_dim = 16
        cenet_velo_dim = 11
        # Bounds runtime contact probabilities to [epsilon, 1-epsilon].
        contact_epsilon = 0.01
        cenet_dec_input_dim = 16 + 11
        cenet_dec_layers = [128, 256, 512]
        cenet_dec_out_dim = 133
        actor_layers = [512, 256, 128]
        critic_layers = [1024, 256, 128]
        pinn_loss_weight = 0.01
        pinn_warmup = 10
        pinn_init_steps = 0
        pretrained_path = ''
        cenet_explicit_layers = [128, 128]
        grf_decoder_layers = [128, 128]
        wrench_decoder_layers = [128, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 0.0003
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

        entropy_coef = 0.01
        use_adaptive_entropy = True
        adaptive_ent_bounds = [0.005, 0.01]
        adaptive_ent_lin_threshold = 0.75
        adaptive_ent_ang_threshold = 0.35
        adaptive_ent_ter_threshold = 6.0
        adaptive_ent_softmax_temp = 2.0

        auxiliary_learning_rate = 0.0002
        # Weight beta on the latent KL term in the combined auxiliary loss.
        vae_kld_weight = 2.0
        privileged_loss_weight = 1.0
        explicit_loss_weight = 1.0
        # Multiplies contact BCE inside the collective explicit-estimator loss.
        contact_probability_loss_weight = 1.0
        ppo_latent_diagnostics_enabled = False
        ppo_latent_diagnostics_interval = 100
        ppo_latent_diagnostics_sample_count = 256
        latent_active_unit_variance_threshold = 1e-2
        grf_loss_weight = 1.0
        active_wrench_loss_weight = 1.0
        neutral_wrench_loss_weight = 0.25
        # Detailed physical GRF/base-wrench TensorBoard reductions. Decoder
        # losses remain logged when this is disabled.
        force_decoder_diagnostics_enabled = True

        bard_enabled = True
        dynamics_backend = 'bard'
        pinocchio_num_workers = None
        bard_randomize_base_inertia = True
        bard_scale_rotational_inertia = True
        bard_batch_capacity = 4096
        bard_inverse_enabled = True
        bard_rollout_enabled = True

        lambda_inverse = 0.01
        lambda_rollout = 0.01
        lambda_projection = 0.001

        profile_bard_timing = False
        console_debug = False

        pcgrad_diagnostics_enabled = False
        pcgrad_diagnostics_start_iteration = 0
        pcgrad_diagnostics_interval = 50
        cache_rollout_mechanics = True

        ppo_qp_sampling = 'disjoint_epoch_partition'
        ppo_qp_passes_per_iteration = 1
        ppo_qp_shard_percentage = 20.0
        ppo_qp_stratify_by_anchor = True
        ppo_qp_sampling_seed = None
        ppo_qp_sampling_logging_enabled = True

        hard_pact_qp = {'enabled': True, 
                        'qp_update_mode': 'two_anchor_held_correction', 
                        'qp_solver': 'qpth', 
                        'rollout_qp_solver': None, 
                        'ppo_qp_solver': None, 
                        'allow_solver_mismatch': False, 
                        'cupiqp_mode': 'dense', 
                        'cupiqp_cuda_graph': False, 
                        'rollout_eps_abs': 0.0001, 
                        'rollout_eps_rel': 0.0001, 
                        'rollout_max_iter': 20, 
                        'rollout_feasibility_tolerance': 0.001, 
                        'rollout_duality_gap_abs': 0.001, 
                        'rollout_duality_gap_rel': 0.001, 
                        'rollout_duality_gap_policy': 'report', 
                        'ppo_eps_abs': 3e-06, 
                        'ppo_eps_rel': 3e-06, 
                        'ppo_max_iter': 30, 
                        'ppo_feasibility_tolerance': 0.001, 
                        'ppo_duality_gap_abs': 3e-06, 
                        'ppo_duality_gap_rel': 3e-06, 
                        'ppo_duality_gap_policy': 'require', 
                        'qpth_warm_start': True, 
                        'friction_coefficient': 0.6, 
                        'torque_rate_limit_nm_s': 1000.0, 
                        'contact_acceleration_limit_m_s2': 0.0, 
                        'interior_margin': 0.001, 
                        'qdd_scale': 50.0, 
                        'force_scale_n': 250.0, 
                        'torque_scale_nm': 40.0, 
                        'slack_scale_m_s2': 50.0, 
                        'torque_tracking_weight': 20.0, 
                        'force_tracking_weight': 5.0, 
                        'slack_weight': 200.0, 
                        'qdd_regularization': 0.001, 
                        'force_regularization': 0.0001, 
                        'torque_regularization': 0.0001, 
                        'q_regularization': 1e-07, 
                        'proximal_rho': 0.1, 
                        'proximal_block_weights': (1.0, 1.0, 1.0, 1.0), 
                        'elastic_recovery_enabled': True, 
                        'elastic_dynamics_weight': 10000.0, 
                        'gradient_scale_tau': 1.0, 
                        'gradient_scale_grf': 1.0, 
                        'gradient_scale_wrench': 1.0, 
                        'gradient_scale_contact': 1.0, 
                        'gradient_clip_tau': 0.0, 
                        'gradient_clip_grf': 0.0, 
                        'gradient_clip_wrench': 0.0, 
                        'gradient_clip_contact': 0.0, 
                        'normalized_feasibility_tolerance_float32': 0.001, 
                        'normalized_feasibility_tolerance_float64': 1e-06, 
                        'kkt_tolerance': 0.1, 
                        'active_tolerance': 0.1, 
                        'eps_float32': 1e-05, 
                        'eps_float64': 1e-09, 
                        'max_iter': 30, 
                        'not_improved_limit': 6, 
                        'check_q_spd': True, 
                        'check_equality_rank': True, 
                        'solver_dtype': 'auto', 
                        'verbose': 0, 
                        'diagnostics_level': 'minimal', 
                        'full_audit_period': 1000, 
                        'full_audit_sample_size': 8, 
                        'rollout_chunk_size': 4096, 
                        'ppo_chunk_size': 8000, 
                        'position_integration_coefficient': 1.0}

        action_clip = GO2HardPACTCfg.normalization.clip_actions

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic_HardPACT'
        algorithm_class_name = 'PPO_HardPACT'
        num_steps_per_env = 24
        max_iterations = 8000
        grf_dim = 12
        run_name = 'pact_100hz_spec_smartcurr_stricterer'
        experiment_name = 'go2_pact_rough'
        save_interval = 500
        load_run = 'Aug01_18-27-22_pact_100hz_spec_smartcurr_stricterer'
        checkpoint = -1
        resume = False
        exp_data_path = 'exp_data/corl_tests_01/pact_stairs_12-16kg.csv'
        console_iteration = True
        console_model_summary = False
        console_reward_terms = True
        console_detailed_losses = False
        console_pinn_timing = True
        console_qp_timing = True
