"""Shared configuration for every Go2 HardPACT approach and ablation."""

from legged_gym import SIMULATOR
from legged_gym.envs.go2.go2_pact.go2_pact_config import GO2PACTCfg, GO2PACTCfgPPO


def hard_pact_terrain_mesh_type(backend=SIMULATOR):
    """Use the backend-native terrain representation without task subclasses."""
    if backend == "genesis":
        return "heightfield"
    if backend == "isaaclab":
        return "trimesh"
    # HardPACT is intentionally not constructed for Isaac Gym: official BARD
    # does not support that backend's Python 3.8 environment.
    raise ValueError(f"Go2 HardPACT supports Genesis and Isaac Lab, not {backend!r}")


class GO2HardPACTCfg(GO2PACTCfg):
    """ICLR HardPACT profile selected by ``SIMULATOR`` at launch time."""

    class env(GO2PACTCfg.env):
        num_observations = 57
        num_explicit_recon_obs = 11
        num_reconstruction_obs = 79
        num_privileged_obs = 198
        num_priv_stack = 1
        num_actions = 12
        num_obs_hist = 20
        grf_dim = 12
        whole_body_dim = 18

    class terrain(GO2PACTCfg.terrain):
        # Genesis uses heightfields; Isaac Lab uses its supported triangle mesh
        # import. Both are generated from the exact same suite below.
        mesh_type = hard_pact_terrain_mesh_type()
        hard_pact_terrain_suite = True
        # flat, slope, rough, stair-down, stair-up, discrete
        terrain_proportions = [0.15, 0.20, 0.20, 0.15, 0.15, 0.15]
        obtain_terrain_info_around_feet = True  # labels/rewards only
        measure_heights = True                  # critic/rewards only

    class sim(GO2PACTCfg.sim):
        use_hard_pact_simulator = True
        backend = SIMULATOR

        class grf:
            prediction_scale_n = [120.0, 120.0, 250.0]
            vertical_deadband_n = 3.0
            clip_min_n = -250.0
            clip_max_n = 250.0
            ema_alpha = 0.20
            contact_threshold_n = 5.0

    class asset(GO2PACTCfg.asset):
        base_link_name = "base"
        # Genesis does not expose these limits; values are from go2.urdf and
        # follow the verified canonical joint-name order.
        dof_vel_limits = [
            30.1, 30.1, 15.7,
            30.1, 30.1, 15.7,
            30.1, 30.1, 15.7,
            30.1, 30.1, 15.7,
        ]

    class domain_rand(GO2PACTCfg.domain_rand):
        # Preserve the current Go2 Genesis request in a backend-neutral schema.
        friction_range = [0.20, 1.25]
        added_mass_range = [-1.0, 4.0]
        added_mass_min = -1.0
        min_added_mass_max = 4.0
        max_added_mass_max = 4.0
        com_pos_x_range = [-0.20, 0.20]
        com_pos_y_range = [-0.15, 0.15]
        com_pos_z_range = [-0.15, 0.15]
        ctrl_delay_step_range = [0, 2]
        kp_range = [0.80, 1.20]
        kd_range = [0.80, 1.20]
        motor_strength_range = [0.90, 1.10]
        joint_armature_range = [0.00, 0.015]
        joint_friction_range_start = [0.00, 0.05]
        joint_friction_range_end = [0.00, 0.20]
        joint_friction_range = [0.00, 0.20]
        joint_stiffness_range_start = [0.00, 0.005]
        joint_stiffness_range_end = [0.00, 0.02]
        joint_stiffness_range = [0.00, 0.02]
        joint_damping_range_start = [0.20, 0.60]
        joint_damping_range_end = [0.00, 0.80]
        joint_damping_range = [0.00, 0.80]
        # The shared core owns atomic, fully recorded disturbance deltas. Do
        # not also invoke the legacy backend push callback.
        push_robots = False
        randomize_instantaneous_disturbances = True
        instantaneous_planar_delta_v_range = [-1.20, 1.20]
        instantaneous_downward_delta_vz_range = [-0.50, 0.0]
        instantaneous_angular_delta_v_range = [-1.50, 1.50]
        # Runtime curricula are capability-gated; currently validated only in
        # Genesis. Isaac Lab keeps reset-time randomization.
        use_domainrand_curriculum = SIMULATOR == "genesis"

    class control(GO2PACTCfg.control):
        use_direct_safe_torque = True
        ideal_torque_tracking = True
        randomize_pact_weights = False
        use_tradeoff_curriculum = False
        torque_rate_limit = 1000.0  # Nm/s, further capped by actuator limit

    class normalization(GO2PACTCfg.normalization):
        class obs_scales(GO2PACTCfg.normalization.obs_scales):
            # All learned force references use observation-scaled units.
            grf = 0.01
            base_wrench = 0.01

    class disturbances:
        class instantaneous:
            enabled = True
            probability = 0.30
            interval_steps_min = 250
            interval_steps_max = 750
            planar_delta_v = [-1.20, 1.20]
            downward_delta_vz = [-0.50, 0.0]
            angular_delta_v = [-1.50, 1.50]

        class sustained_wrench:
            enabled = True
            force_probability = 0.30
            torque_probability = 0.30
            interval_steps = [250, 750]
            duration_steps = [75, 250]
            force_interval_steps = [250, 750]
            torque_interval_steps = [250, 750]
            force_duration_steps = [75, 250]
            torque_duration_steps = [75, 250]
            ramp_fraction = 0.25
            force_bounds_n = [-60.0, 60.0]
            torque_bounds_nm = [-12.0, 12.0]
            force_normalizer_n = 60.0
            torque_normalizer_nm = 12.0

    class features:
        physics_parameter_source = "realized_randomized"
        supervised_physics_head_pretraining = True
        grf_supervision_weight = 1.0
        active_wrench_supervision_weight = 1.0
        neutral_wrench_supervision_weight = 0.25
        feedforward_clone_weight = 1.0
        use_bard_inverse_loss = True
        use_bard_rollout_loss = True
        use_qp = True
        differentiate_qp = True
        stop_gradient_qp = False
        use_soft_projection_penalty = False

    class bard:
        enabled = True
        required = True
        batch_capacity = 4096
        dtype = "float32"
        lambda_inverse = 0.01
        lambda_rollout = 0.01
        lambda_projection = 0.001

    class qp:
        enabled = True
        friction = 0.45
        solver = "auto"
        reference_dtype = "float64"
        gpu_dtype = "float32"
        use_float32_gpu = True
        gpu_chunk_size = 1024
        acceleration_weight = 1.0e-3
        grf_tracking_weight = 1.0
        torque_tracking_weight = 4.0
        contact_slack_weight = 2.0e3
        hessian_regularization = 1.0e-7

    class rewards(GO2PACTCfg.rewards):
        torque_cancellation_deadband = 0.03

        class scales(GO2PACTCfg.rewards.scales):
            torque_cancellation = -0.10
            # Retained as valid keys for old checkpoints/configs, but disabled
            # to avoid applying duplicate conflict penalties.
            torque_conflict_symmetric = 0.0
            torque_conflict_symmetric_scaled = 0.0
            torque_alignment = 0.0
            aligned_torques = 0.0

    class logging:
        paper_metrics = True
        log_grf_stages = True
        log_gradient_modules = True
        log_qp_conditioned_statistics = True
        log_hardware_metadata = True


class GO2HardPACTCfgPPO(GO2PACTCfgPPO):
    seed = 1
    runner_class_name = "Go2HardPACTRunner"

    class policy(GO2PACTCfgPPO.policy):
        activation = "elu"
        init_noise_std = 1.0
        history_dim = 57 * 20
        latent_dim = 16
        explicit_dim = 11
        encoder_layers = [256, 128]
        physics_head_layers = [128, 128]
        # Observation-scaled physics-head output gains for the main profile.
        # The corresponding physical wrench envelope is approximately
        # [60, 60, 99.24] N and [23.4403, 23.4403, 23.4403] Nm. The latter is
        # 12 Nm of sustained disturbance plus the conservative 4 kg payload
        # moment at the configured [0.20, 0.15, 0.15] m COM envelope. The
        # runner recomputes these values from the effective DR ranges so edits
        # to mass, COM, gravity, or normalization remain authoritative.
        grf_scale = [1.20, 1.20, 2.50] * 4
        wrench_scale = [
            0.60, 0.60, 0.9924,
            0.234403276, 0.234403276, 0.234403276,
        ]
        actor_layers = [512, 256, 128]
        critic_layers = [512, 256, 128]
        position_pretraining = False

    class algorithm(GO2PACTCfgPPO.algorithm):
        supervised_physics_head_pretraining = True
        use_bard_inverse_loss = True
        use_bard_rollout_loss = True
        use_qp = True
        lambda_inverse = 0.01
        lambda_rollout = 0.01
        lambda_projection = 0.001
        grf_loss_weight = 1.0
        active_wrench_loss_weight = 1.0
        neutral_wrench_loss_weight = 0.25
        reliability_ema_alpha = 0.05
        adaptation_learning_rate = 1.0e-5

    class runner(GO2PACTCfgPPO.runner):
        policy_class_name = "ActorCriticGo2HardPACT"
        algorithm_class_name = "PPOGo2HardPACT"
        experiment_name = "go2_hard_pact"
        run_name = "full"
        checkpoint_migration = "strict_hard_pact_pos"
        save_interval = 500
