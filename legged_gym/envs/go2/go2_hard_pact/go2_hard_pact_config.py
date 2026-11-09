"""Standalone Go2 HardPACT configuration with all retained PACT settings inline."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from .transition import DISTURBANCE_CRITIC_DIM

class GO2HardPACTCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 57
        num_privileged_obs = 57 + (50 + 38) + 143 + DISTURBANCE_CRITIC_DIM
        num_priv_stack = 5
        num_explicit_recon_obs = 12
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
        curriculum_upward_delay_iterations = 0  # Absolute PPO iteration; 0 disables delay.
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
            force_mode = "total"  # Isaac Lab: normal + friction; normal_only explicitly reproduces legacy labels.
            contact_capacity_per_body = 8
            prediction_scale_n = [100.0, 100.0, 100.0]
            vertical_deadband_n = 3.0
            clip_min_n = -500.0
            clip_max_n = 500.0
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
        # Isaac Lab: hold each environment's reset-sampled properties for this
        # many completed episodes. 0/1 = every reset. Changed ranges bypass it.
        reset_resample_episodes = 25
        use_domainrand_curriculum = True
        com_rand_z_positive = True
        num_push_steps = 1000
        push_warmup = 10000
        num_jumps = 10

        randomize_friction = True
        friction_range = [0.2, 1.25]

        push_robots = True
        push_interval_max = 10.0
        push_interval_min = 5.0
        max_push_vel_xy = 0.5
        min_push_vel_xy = 0.25

        max_vertical_push = 0.3
        min_vertical_push = 0.1
        vert_interval_max = 10.0
        vert_interval_min = 5.0

        max_push_torque = 0.5
        min_push_torque = 0.25
        wrench_timeout_max = 10.0
        wrench_timeout_min = 5.0

        randomize_base_mass = True
        min_added_mass_max = 1.0
        max_added_mass_max = 1.0
        added_mass_min = -1.0

        randomize_com_displacement = True
        com_displacement_x_min = 0.05
        com_displacement_x_max = 0.15

        com_displacement_y_min = 0.05
        com_displacement_y_max = 0.15

        com_displacement_z_positive = True
        com_displacement_z_min_pos = 0.05
        com_displacement_z_min = 0.05
        com_displacement_z_max = 0.20

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

        best_reward_window = 200
        best_reward_quantile = 0.9
        recovery_ratio = 0.9
        step_interval = 10
        reward_ema_alpha = 0.05
        min_reward_to_step = 0.50
        joint_dynamics_progress_delta = 0.02
        mass_com_progress_delta = 0.01
        disturbance_progress_delta = 0.01
        use_joint_dynamics_curriculum = False
        use_mass_com_curriculum = True
        use_disturbance_curriculum = True
        # Disabled sub-curricula: True uses final bounds; False uses initial bounds.
        disabled_curriculum_use_final_bounds = True

        persistent_disturbance = True
        persistent_force_probability = 0.4
        persistent_torque_probability = 0.4
        persistent_force_interval_range_s = [5.0, 15.0]
        persistent_torque_interval_range_s = [5.0, 15.0]
        persistent_force_duration_range_s = [1.0, 5.0]
        persistent_torque_duration_range_s = [1.0, 5.0]
        persistent_ramp_fraction = 0.30
        # Initial -> final planar FORCE NORM cap (sqrt(Fx²+Fy²)), in N.
        # The disturbance curriculum expands the planar disk and downward Fz
        # after push_warmup and the joint-dynamics/mass-CoM phases. Existing
        # events finish their waveform; only new events use expanded bounds.
        persistent_force_min_n = 4.0
        persistent_force_max_n = 10.0
        # Positive magnitudes; actual world Fz is sampled in [-limit, 0].
        persistent_vertical_force_min_n = 15.0
        persistent_vertical_force_max_n = 70.0
        persistent_torque_min_nm = 2.0
        persistent_torque_max_nm = 15.0
        persistent_yaw_torque_min_nm = 1.0  # Initial world-Z torque cap (+/- N*m).
        persistent_yaw_torque_max_nm = 4.0  # Final cap at full disturbance curriculum.

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
        rendered_envs_count = 250  # Isaac Lab random startup subset; None uses rendered_envs_idx.
        rendered_envs_seed = 0     # Independent display RNG; None chooses a fresh subset each launch.
        # Isaac Lab: None shows all robots; [] hides all (physics stays active).
        debug_vis = False  # Isaac Lab contact markers; True shows markers for all sensed robots.
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

        dof_vel_limits = [30.1, 30.1, 15.7, 
                          30.1, 30.1, 15.7, 
                          30.1, 30.1, 15.7, 
                          30.1, 30.1, 15.7]

    class control(LeggedRobotCfg.control):
        clip_torque_rate_without_qp = False
        torque_rate_limit_nm_s = 2000.0  # Shared by QP, fallback, and optional non-QP clipping.
        stiffness = {'joint': 30.0}
        damping = {'joint': 0.75}
        action_scale = 0.25
        torque_scale = 10.0
        dt = 0.02
        decimation = 4
        tradeoff_init_weights = [1.0, 1.6]
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
        soft_dof_vel_limit = 0.9  # fraction of asset.dof_vel_limits (rad/s)
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

        only_positive_rewards = False
        use_reward_curriculum = True

        max_contact_force = 400.0
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
        foot_clearance_excess_margin = 0.04
        foot_clearance_excess_weight = 0.25
        class scales(LeggedRobotCfg.rewards.scales):
            termination = 0.0
            collision = -1.0
            dof_pos_limits = -2.0
            dof_vel_limits = -0.1  # 0 disables the soft joint-speed penalty
            dof_close_to_default = -0.01
            torque_limits = -0.01
            torque_rate_limits = 0.0  # Substep-mean squared excess / (rate*physics_dt)^2; 0 disables.
            pd_target_torque_limit = 0.0

            alive_bonus = 0.1

            dof_vel_stand_still = 0.0
            stand_still_contact = 0.5
            dof_pos_stand_still = -0.1

            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.50
            dof_tracking = 0.10

            torque_conflict_symmetric = -0.0
            torque_alignment = 0.00
            ff_ratio = 0.0
            torque_cancellation = -0.2

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

            pos_action_rate = -0.01
            pos_action_smoothness = -0.01

            tau_action_rate = -0.02
            tau_action_smoothness = -0.02

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
            feet_near_edge = -0.1
            edge_swing_clearance = -0.0
            swing_foot_collision_edge = -0.0
            feet_regulation = -0.0

        class reward_curriculum:
            curr_reward_keys = ['ang_vel_xy',
                                'lin_vel_z', 
                                'orientation', 
                                'torque_limits',
                                'hip_pos',
                                # 'pos_action_rate', 
                                # 'pos_action_smoothness', 
                                # 'tau_action_rate', 
                                # 'tau_action_smoothness',
                                'dof_vel_limits',
                                # 'dof_acc'
                                ]
            curr_reward_bounds = {'ang_vel_xy': [-0.01, -0.1], 
                                  'lin_vel_z':[-0.5, -1.0],
                                  'orientation': [-0.2, -1.0], 
                                  'torque_limits': [-0.01, -1.0], 
                                  'hip_pos': [-0.2, -0.4], 
                                #   'pos_action_rate': [-0.005, -0.01], 
                                #   'pos_action_smoothness': [-0.005, -0.01], 
                                #   'tau_action_rate': [-0.01, -0.02], 
                                #   'tau_action_smoothness': [-0.01, -0.02],
                                  'dof_vel_limits':[-0.1, -1.0],
                                #   'dof_acc':[-2.5e-08, -2.5e-07]
                                  }
            curr_steps = 16000
            warmup_steps = 0  


    class commands(LeggedRobotCfg.commands):
        curriculum = True
        curriculum_threshold = 0.8  # raw linear-tracking mean, not weighted reward
        curriculum_patience_iterations = 50  # consecutive PPO rollouts; 0 = legacy reset-based updates
        max_curriculum = 1.2
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class deployment_physics:
        # Gate only the QP GRF reference; raw supervised predictions stay intact.
        grf_swing_gating_enabled = True
        grf_swing_contact_threshold = 0.5
        # Independent, GRF-decoder-only physical-zero consistency penalty.
        grf_swing_loss_weight = 0.1
        wrench_scale = [100.0, 100.0, 100.0, 25.0, 25.0, 25.0]
        wrench_qp_clip = [150.0, 150.0, 150.0, 40.0, 40.0, 40.0]
GO2HardPACTCfg.sim.dt = GO2HardPACTCfg.control.dt / GO2HardPACTCfg.control.decimation

class GO2HardPACTCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'PACTRunner'

    class policy(LeggedRobotCfgPPO.policy):
        activation = 'elu'
        # Normalized action std: hip/thigh/calf in FR, FL, RR, RL order,
        # repeated independently for position and feedforward outputs.
        init_noise_std = [0.80, 1.00, 1.00] * 4 + [0.40, 0.45, 0.45] * 4
        min_noise_std = ([0.15, 0.25, 0.25] * 4) * 2
        max_noise_std = 1.10

        cenet_enc_layers = [256, 128]
        cenet_enc_latent_dim = 32
        cenet_velo_dim = 12

        # Bounds runtime contact probabilities to [epsilon, 1-epsilon].
        contact_epsilon = 0.01
        cenet_dec_input_dim = cenet_enc_latent_dim + cenet_velo_dim
        cenet_dec_layers = [128, 256, 512]

        cenet_dec_out_dim = 276  # 133 retained features + 143 terrain heights

        actor_layers = [512, 256, 128]
        critic_layers = [1024, 256, 128]

        pinn_loss_weight = -1.0
        pinn_warmup = 10
        pinn_init_steps = 0

        # pretrained_path = '../../rsl_rl/modules/pretrained_checkpoints/go2_hard_pact/Sep08_hard_pact_start_model_5000_pos_std_mean_torque.pt'
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
        adaptive_ent_bounds = [0.003, 0.01]
        adaptive_ent_lin_threshold = 0.75
        adaptive_ent_ang_threshold = 0.35
        adaptive_ent_ter_threshold = 6.0
        adaptive_ent_softmax_temp = 2.0

        auxiliary_learning_rate = 0.0002
        # Weight beta on the latent KL term in the combined auxiliary loss.
        vae_kld_weight = 0.001
        vae_kl_initial_weight = 0.001
        vae_kl_warmup_start = 0  # absolute PPO iteration (also on resume)
        vae_kl_warmup_iterations = 0  # 0 disables the curriculum

        # Multiplies contact BCE inside the collective explicit-estimator loss.
        contact_probability_loss_weight = 0.1
        privileged_loss_weight = 1.0
        explicit_loss_weight = 1.0
        grf_loss_weight = 1.0
        active_wrench_loss_weight = 1.0
        neutral_wrench_loss_weight = 1.0

        # Opt-in policy recomputation/latent ablations; basic KL/loss logging
        # remains available without these diagnostic-only forward passes.
        ppo_latent_diagnostics_enabled = True
        action_channel_usage_enabled = True  # Physical mean-FF usage; no extra forwards when disabled.
        action_channel_usage_interval = 100   # Compare before/after complete updates at this cadence.
        action_channel_usage_sample_count = 256  # Fixed bounded rollout sample; deterministic latent mean.
        ppo_latent_diagnostics_interval = 100
        ppo_latent_diagnostics_sample_count = 256
        latent_active_unit_variance_threshold = 1e-2

        # Detailed physical GRF/base-wrench TensorBoard reductions. Decoder
        # losses remain logged when this is disabled.
        force_decoder_diagnostics_enabled = False

        bard_enabled = True
        dynamics_backend = 'bard'
        pinocchio_num_workers = None
        bard_randomize_base_inertia = True
        bard_scale_rotational_inertia = True
        bard_batch_capacity = 4096
        # Independent PINN representation-loss switches for encoder/decoder updates.
        # False skips the corresponding loss; QP and supervised losses are unchanged.
        # Named ablations (including explicit full) override both switches;
        # normal go2_hard_pact_<backend> tasks inherit these settings unchanged.
        bard_inverse_enabled = False
        bard_rollout_enabled = False

        lambda_inverse = 0.2
        lambda_rollout = 0.8
        lambda_projection = 0.25
        # Cosine 0->1 from QP activation, in absolute PPO iterations; 0 disables.
        # Scales actor projection/velocity/FK/load losses, not encoder/decoder PINNs.
        actor_physics_warmup_iterations = 200
        lambda_qp_velocity_xy = 1.0  # inherit lambda_projection; 0 disables
        lambda_qp_velocity_yaw = 1.0  # independent nonnegative override

        profile_bard_timing = False
        # Optional actor-only losses; zero skips computation and FK storage.
        # Shared actor weights can affect both outputs. Load compensation does
        # not identify disturbance-specific incremental effort or a PD/FF ratio.
        actor_regularizers = {
            'lambda_fk': 0.10,  # Foot-position consistency, position-command path only.
            'fk_position_scale_m': 0.10,  # Body-relative XYZ normalization [m].
            'fk_deadband_m': 0.02,  # Per-coordinate zero-error band [m].
            
            'lambda_load': 0.10,  # Full accepted QP candidate, FF-command path only.
            'load_preferred_fraction': 0.80,  # Preferred share of compensation demand.
            'load_min_normalized_demand': 0.02,  # Ignore negligible stance-joint RMS demand.
            'load_allocation_regularization': 1e-6,  # Detached 6x6 contact allocation ridge.
            'load_balance_relative_tolerance': 0.05,  # Floating-base balance residual tolerance.
            'load_force_scale_n': 100.0,  # Allocation force-row scale [N].
            'load_moment_scale_nm': 25.0,  # Allocation moment-row scale [Nm].
            'load_epsilon': 1e-8,  # Direction normalization safeguard.
        }
        console_debug = False

        # Keep PCGrad optimization, but skip diagnostic gradient clones,
        # per-module norms/cosines, and nonfinite scans in production.
        pcgrad_diagnostics_enabled = True
        pcgrad_diagnostics_start_iteration = 0
        pcgrad_diagnostics_interval = 500
        # Preserve the reference gradient; remove only opposing physics components.
        # "legacy" restores sign-selected orthogonal/orthogonal-balanced projection.
        pinn_projection_mode = "conflict_only"
        pinn_projection_norm_balance = True
        cache_rollout_mechanics = True

        ppo_qp_sampling = 'disjoint_epoch_partition'
        ppo_qp_passes_per_iteration = 1
        ppo_qp_shard_percentage = 20.0
        ppo_qp_stratify_by_anchor = True
        ppo_qp_sampling_seed = None
        ppo_qp_sampling_logging_enabled = False

        hard_pact_qp = {
            # Activation and rollout execution
            'enabled': True,                                 # Master QP switch; ablation gates still apply.
            'warmup_iterations': 0,                        # Absolute PPO iteration at which rollout/replay QPs become eligible.
            'qp_update_mode': 'first_plus_random_substep',   # first_plus_random_substep - First plus balanced later solve; every_substep=4, random_one_substep=1 per env/interval.
            'correction_ramp_enabled': True,                 # Blend accepted QP corrections into execution; full QP losses remain unchanged.
            'correction_ramp_start_offset': 0,             # Iterations after QP activation before the execution ramp starts.
            'correction_ramp_duration': 100,                # Iterations to reach full correction; 0 applies full correction immediately.

            # Physical constraints and prediction convention
            'friction_coefficient': 0.6,                     # Friction-pyramid mu: abs(Fx), abs(Fy) <= mu*Fz for stance feet.
            'contact_threshold': 0.5,                        # Detached contact-probability cutoff for the stance mask.
            'endpoint_torque_constraints': True,             # Enforce hard predicted endpoint PD+FF torque bounds, including recovery.
            'position_integration_coefficient': 1.0,         # Beta in q_next=q+dt*v+beta*dt^2*a; 1 is semi-implicit Euler.
            'torque_rate_constraint_weight': 0.0,            # 0 removes QP rate constraints; any positive value enables the hard rate bound.

            # Inner objective: nominal torque / GRF tracking and stance acceleration
            'torque_tracking_weight': 15.0,                  # Inner normalized total-torque tracking weight.
            'torque_scale_nm': 40.0,                         # Torque variable/reference normalization scale [Nm].
            'force_tracking_weight': 1.0,                    # Inner normalized predicted-GRF tracking weight.
            'force_scale_n': 250.0,                          # Force variable/reference normalization scale [N].
            'contact_acceleration_weight': 0.50,             # Inner stance-acceleration weight; authoritative final curriculum target.
            'contact_acceleration_scale_m_s2': 50.0,         # Stance-acceleration residual normalization [m/s^2].

            # Inner objective: torso attitude and terrain-relative height
            'attitude_weight': 5.0,                          # Inner roll/pitch stabilization weight; final curriculum target.
            'attitude_kp': 20.0,                             # Tilt-error gain in desired roll/pitch angular acceleration [1/s^2].
            'attitude_kd': 5.0,                              # Angular-rate damping gain [1/s].
            'attitude_acceleration_scale_rad_s2': 20.0,      # Roll/pitch acceleration residual normalization [rad/s^2].
            'height_weight': 5.0,                            # Inner height-regulation weight; final curriculum target; 0 skips this objective.
            'height_kp': 20.0,                               # Terrain-relative height-error gain [1/s^2].
            'height_kd': 5.0,                                # Predicted world-vertical velocity damping gain [1/s].
            'height_target': None,                           # Target torso height [m]; None uses rewards.base_height_target.
            'height_acceleration_scale': 20.0,               # Vertical-acceleration residual normalization [m/s^2].

            # Velocity tracking: independent inner-QP and outer-loss horizons
            'planar_velocity_weight': 20.0,                  # Inner normalized body-frame planar-velocity tracking weight.
            'yaw_rate_weight': 10.0,                          # Inner normalized body-frame yaw-rate tracking weight.
            'planar_velocity_scale_m_s': 1.0,                # Planar-velocity error normalization [m/s].
            'yaw_rate_scale_rad_s': 1.0,                     # Yaw-rate error normalization [rad/s].
            'qp_velocity_objective_horizon_s': 0.020,        # Inner xy/yaw constant-derivative horizon [s]; None uses physics dt.
            'qp_velocity_loss_horizon_s': 0.02,              # Outer xy/yaw loss horizon [s]; independent of inner horizon and constraint dt.

            # Outer projection losses (overall lambda_projection is configured above)
            'projection_contact_acceleration_weight': 0.10,  # Fixed outer stance-residual coefficient; not scheduled with inner weight.
            'recovery_projection_weight': 0.5,               # Outer recovery correction/slack multiplier before lambda_projection.
            'recovery_projection_slack_weight': 1.0,         # Relative outer penalty on normalized joint-envelope recovery slack.
            'recovery_projection_rate_slack_weight': 1.0,    # Relative outer rate-slack penalty; absent when QP rate constraints are off.

            # Recovery QP: soften joint envelope / optional rate bounds, not absolute torque
            'soft_joint_recovery_enabled': True,             # Retry failed primary rows with a softened joint position/velocity envelope.
            'soft_joint_recovery_weight': 200.0,             # Inner quadratic penalty on normalized joint-envelope slack.
            'soft_joint_recovery_scale_rad_s2': 100.0,       # Joint-envelope slack normalization in acceleration units [rad/s^2].
            'soft_rate_recovery_weight': 200.0,              # Inner quadratic rate-slack penalty; used only with QP rate constraints.
            'soft_rate_recovery_scale_nm': 10.0,             # Torque-rate slack normalization [Nm], not Nm/s.

            # Performance-gated inner stance / torso-stability curriculum
            'objective_curriculum_enabled': True,            # Schedule inner weights using frozen pre-QP tracking-reference recovery.
            'torso_stability_curriculum_enabled': True,      # Include attitude/height in the objective curriculum; stance remains independent.
            'contact_acceleration_weight_initial': 0.25,     # Initial stance weight; None uses 25% of contact_acceleration_weight.
            'attitude_weight_initial': None,                 # Initial attitude weight; None uses 25% of attitude_weight.
            'height_weight_initial': None,                   # Initial height weight; None uses 25% of height_weight.
            'objective_curriculum_start': None,              # Earliest absolute iteration for progression; None follows execution-ramp completion.
            'objective_curriculum_progress_delta': 0.05,     # Fraction of initial-to-final weight range added per successful advancement.
            'objective_curriculum_step_interval': 10,        # Minimum PPO iterations between objective-weight increases.
            'objective_curriculum_ema_alpha': 0.05,          # Current tracking EMA update coefficient.
            'objective_curriculum_window': 200,              # Maximum valid pre-QP EMA observations retained for the frozen baseline.
            'objective_curriculum_quantile': 0.9,            # Quantile of pre-QP EMA history used as the frozen tracking reference.
            'objective_curriculum_recovery_ratio': 0.90,     # Threshold multiplier on frozen baseline, subject to min_tracking.
            'objective_curriculum_recovery_iterations': 50,  # Consecutive valid above-threshold iterations required per increase.
            'objective_curriculum_baseline_override': None,  # Explicit frozen tracking reference; needed for resumes missing a baseline.
            'objective_curriculum_min_tracking': 0.5,        # Absolute lower floor for the tracking recovery threshold.
            'objective_curriculum_min_samples': 1,           # Minimum tracking evidence count required for a valid iteration.

            # Solver selection, precision, and shared numerical regularization
            'qp_solver': 'cupiqp',                           # Default numerical backend for both rollout and PPO.
            'rollout_qp_solver': None,                       # Optional rollout backend override; None inherits qp_solver.
            'ppo_qp_solver': None,                           # Optional differentiable PPO backend override; None inherits qp_solver.
            'allow_solver_mismatch': False,                  # Permit intentionally different rollout/PPO numerical backends.
            'solver_dtype': 'auto',                          # auto selects CUDA float32 / CPU float64; explicit precision may override.
            'q_regularization': 1e-07,                       # Positive solver-space diagonal Hessian ridge for numerical conditioning.

            # cuPIQP memory reuse and batching
            'cupiqp_mode': 'dense',                          # cuPIQP matrix representation: dense or sparse.
            'cupiqp_cuda_graph': False,                      # Opt-in cuPIQP CUDA graph execution; disabled on this training path.
            'cupiqp_rollout_capacity_reuse': True,           # Reuse rollout/fallback solver capacity buckets.
            'cupiqp_rollout_cache_size': 16,                 # Maximum retained rollout capacity-cache entries; idle entries are evictable.
            'cupiqp_ppo_reuse': True,                        # Reuse exclusively leased PPO solvers; False uses fresh instances.
            'cupiqp_ppo_capacity_reuse': True,               # Capacity-bucket reuse for PPO recovery batches, with padded rows excluded.
            'cupiqp_ppo_pool_size': 8,                       # Bounded PPO solver pool; outstanding autograd leases are not evicted.
            'cache_rollout_qp_mechanics': True,              # Own the selected rollout anchor's nominal mechanics for PPO reuse.
            'scheduled_compact_diagnostics': True,          # Physical health summaries sample diagnostic iterations; outcome counts remain continuous.
            'cupiqp_selective_backward_copy': True,         # Clone only requested implicit gradients; False is the copy-all reference.
            'rollout_chunk_size': 4096,                      # Maximum rows per rollout solver chunk.
            'ppo_chunk_size': 8000,                          # Maximum rows per differentiable PPO solver chunk.

            # Rollout solver convergence and acceptance profile
            'rollout_eps_abs': 0.0001,                       # Rollout solver absolute convergence tolerance.
            'rollout_eps_rel': 0.0001,                       # Rollout solver relative convergence tolerance.
            'rollout_max_iter': 20,                          # Rollout numerical-solver iteration cap.
            'rollout_feasibility_tolerance': 0.001,          # Normalized primal-residual acceptance tolerance for rollout candidates.
            'rollout_duality_gap_abs': 0.001,                # Absolute duality-gap threshold for rollout gap checks/reporting.
            'rollout_duality_gap_rel': 0.001,                # Relative duality-gap threshold for rollout gap checks/reporting.
            'rollout_duality_gap_policy': 'report',          # report logs gap without rejecting on it; require also gates acceptance.

            # PPO solver convergence and acceptance profile
            'ppo_eps_abs': 0.0001,                           # Differentiable PPO solver absolute convergence tolerance.
            'ppo_eps_rel': 0.0001,                           # Differentiable PPO solver relative convergence tolerance.
            'ppo_max_iter': 30,                              # Differentiable PPO numerical-solver iteration cap.
            'ppo_feasibility_tolerance': 0.001,              # Normalized primal-residual acceptance tolerance for PPO candidates.
            'ppo_duality_gap_abs': 0.001,                    # Absolute duality-gap threshold for PPO gap checks.
            'ppo_duality_gap_rel': 0.001,                    # Relative duality-gap threshold for PPO gap checks.
            'ppo_duality_gap_policy': 'require',             # require gates accepted gradients on gap checks; report only records gaps.

            # qpth/reference numerical settings (separate from cuPIQP profiles)
            'qpth_warm_start': True,                         # Enable qpth warm starts and its existing cold-retry path.
            'eps_float32': 1e-05,                            # Dtype-specific qpth convergence tolerance for float32.
            'eps_float64': 1e-09,                            # Dtype-specific qpth convergence tolerance for float64.
            'max_iter': 30,                                  # qpth interior-point iteration cap.
            'not_improved_limit': 6,                         # qpth stagnation limit before early termination.
            'normalized_feasibility_tolerance_float32': 0.001,  # Default float32 normalized primal acceptance tolerance (qpth path).
            'normalized_feasibility_tolerance_float64': 1e-06,  # Default float64 normalized primal acceptance tolerance (qpth path).
            'check_q_spd': True,                             # Request qpth positive-definite Hessian validation.
            'check_equality_rank': True,                     # Compatibility flag; current reduced QP has no equality rows.
            'interior_margin': 0.001,                        # Retained compatibility setting; unused by the current reduced assembly.
            'kkt_tolerance': 0.1,                            # Retained compatibility setting; not a current primal-acceptance threshold.
            'verbose': 0,                                    # Solver console verbosity; 0 requests quiet output.

            # Implicit-solver input gradients; 0 clipping means no clipping
            'gradient_scale_tau': 1.0,                       # Multiplier on QP gradients into nominal torque.
            'gradient_scale_grf': 1.0,                       # Multiplier on QP gradients into the GRF reference.
            'gradient_scale_wrench': 1.0,                    # Multiplier on QP gradients into the predicted wrench.
            'gradient_scale_contact': 1.0,                   # Compatibility multiplier; current detached binary stance mask has no contact VJP.
            'gradient_clip_tau': 0.0,                        # Per-row nominal-torque gradient norm cap; 0 disables.
            'gradient_clip_grf': 0.0,                        # Per-row GRF-reference gradient norm cap; 0 disables.
            'gradient_clip_wrench': 0.0,                     # Per-row wrench-input gradient norm cap; 0 disables.
            'gradient_clip_contact': 0.0,                    # Compatibility cap; current detached binary stance mask has no contact VJP.

            # Diagnostics, bounded audits, profiling, and failure capture
            'diagnostics_level': 'minimal',                  # minimal acceptance summaries, physical residuals, or full sampled matrix/KKT audits.
            'tensorboard_diagnostics_enabled': True,         # Enable periodic aggregated QP TensorBoard diagnostics.
            'tensorboard_diagnostics_interval': 200,         # QP diagnostic reporting interval in PPO iterations.
            'active_tolerance': 0.1,                         # Residual/slack threshold used to classify diagnostically active constraints.
            'full_audit_period': 0,                          # Solver-call interval for full-profile audits; 0 disables expensive audits.
            'full_audit_sample_size': 256,                   # Maximum environments sampled per full audit.
            'cuda_event_profiling': False,                   # Opt-in CUDA event timings with iteration-end synchronization.
            'exception_capture_enabled': True,               # Save bounded solver exception packets for debugging.
            'exception_capture_limit': 1,                    # Maximum captured exceptions per solver instance.
            'exception_capture_dir': '/tmp/hard_pact_qp_failures',  # Directory for saved solver-failure packets.
        }

        action_clip = GO2HardPACTCfg.normalization.clip_actions

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic_HardPACT'
        algorithm_class_name = 'PPO_HardPACT'
        num_steps_per_env = 24
        max_iterations = 20000
        grf_dim = 12
        run_name = 'hardpact_50hz_noboot'
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


# conda activate lr_lab_cupiqp

# SIMULATOR=isaaclab PYTHONPATH=. \
# python legged_gym/scripts/play_go2_hard_pact_pos.py \
#   --task go2_hard_pact_baseline_isaaclab \
#   --load_run Sep17_14-33-42_hard_pact_full_isaaclab \
#   --ckpt -1 \
#   --num_envs 10 \
#   --gpu cuda:0


# ./go2_hard_pact.sh \
#   --task go2_hard_pact_isaaclab \
#   --headless \
#   --dynamics_backend bard \
#   --qp_solver cupiqp \
#   --num_envs 4096 \
#   --gpu cuda:0


# cd /home/oyoungquist/Research/Genesis_Development/HCR_HardPact
# conda activate lr_lab_cupiqp

# SIMULATOR=isaaclab PYTHONPATH="$PWD" \
# python legged_gym/scripts/play_go2_hard_pact_pos.py \
#   --task go2_hard_pact_baseline_isaaclab \
#   --load_run /home/oyoungquist/Research/Genesis_Development/HCR_Genesis_PACT_Development/logs/hardpact_iclr/go2_pact_rough/Oct07_10-22-08_hard_pact_full_isaaclab \
#   --ckpt 20000 \
#   --gpu cuda:0 \
#   --num_envs 10
