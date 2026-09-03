"""Legacy-compatible HardPACT configuration aliases."""

from legged_gym.envs.go2.go2_pact.go2_pact_config import (
    GO2PACTCfg,
    GO2PACTCfgPPO,
)
from .transition import DISTURBANCE_CRITIC_DIM


class GO2HardPACTCfg(GO2PACTCfg):
    """Legacy PACT configuration plus calibrated Go2 GRF conditioning."""

    class env(GO2PACTCfg.env):
        num_explicit_recon_obs = 11
        num_privileged_obs = GO2PACTCfg.env.num_privileged_obs + DISTURBANCE_CRITIC_DIM

    class sim(GO2PACTCfg.sim):
        # Console controls are scoped to HardPACT. Legacy PACT configurations
        # do not define them and retain their existing output unchanged.
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

    class normalization(GO2PACTCfg.normalization):
        class obs_scales(GO2PACTCfg.normalization.obs_scales):
            base_wrench = 0.01

    class domain_rand(GO2PACTCfg.domain_rand):
        # Instantaneous pushes use the legacy PACT curriculum and callback.
        # Persistent external wrenches share its disturbance progress scalar.
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

    class deployment_physics:
        sustained_force_bounds_n = [-60.0, 60.0]
        sustained_torque_bounds_nm = [-12.0, 12.0]
        # Stable planned envelope used for the entire run. It intentionally
        # does not follow the simulator's instantaneous curriculum progress.
        planned_added_mass_range_kg = [-1.0, 4.0]

class GO2HardPACTCfgPPO(GO2PACTCfgPPO):
    """Legacy PACT architecture with the reduced explicit estimator."""

    class policy(GO2PACTCfgPPO.policy):
        cenet_velo_dim = 11
        cenet_explicit_layers = [128, 128]
        grf_decoder_layers = [128, 128]
        wrench_decoder_layers = [128, 128]
        cenet_dec_input_dim = 16 + 11
        cenet_dec_out_dim = 133
        pretrained_path = ""

    class runner(GO2PACTCfgPPO.runner):
        policy_class_name = "ActorCritic_HardPACT"
        algorithm_class_name = "PPO_HardPACT"
        console_iteration = True
        console_model_summary = False
        console_reward_terms = True
        console_detailed_losses = False
        console_pinn_timing = True
        console_qp_timing = True

        num_steps_per_env = 24 # per iteration
        max_iterations = 6000 # number of policy updates
        grf_dim = 12

        # debug_warmpinn_wb
        run_name = 'pact_100hz_spec_smartcurr_stricterer'
        experiment_name = 'go2_pact_rough'
        save_interval = 500

    class algorithm(GO2PACTCfgPPO.algorithm):
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
        # Keep lightweight CUDA-event timing enabled for the requested
        # training profile. Console presentation is controlled separately by
        # runner.console_pinn_timing.
        profile_bard_timing = True
        console_debug = False
        # PCGrad itself remains active every minibatch. These flags only gate
        # expensive cloned-gradient/module-cosine diagnostics.
        pcgrad_diagnostics_enabled = False
        pcgrad_diagnostics_start_iteration = 0
        pcgrad_diagnostics_interval = 50
        cache_rollout_mechanics = True

        # qpth/OptNet safety projection.  The decision vector is always
        # [qdd_18, world_grf_12, tau_safe_12, contact_slack_12]. CPU references
        # use float64 while CUDA uses float32 to halve retained
        # KKT VRAM; either can be forced. Chunking bounds peak graph size.
        hard_pact_qp = {
            "enabled": True,
            # qpth remains the verified default. Optional GPU-native solvers
            # are selected explicitly and never trigger a hidden CPU/backend
            # fallback when their dependency is unavailable.
            "qp_solver": "qpth",
            "rollout_qp_solver": None,
            "ppo_qp_solver": None,
            "allow_solver_mismatch": False,
            "cupiqp_mode": "dense",
            "cupiqp_cuda_graph": False,
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
            "ppo_chunk_size": 4096,
            # Genesis and both PhysX backends advance position with the new
            # velocity (semi-implicit Euler), hence q+=dt*v+dt^2*qdd.
            "position_integration_coefficient": 1.0,
        }
        grf_observation_scale = GO2HardPACTCfg.normalization.obs_scales.grf
        base_wrench_observation_scale = (
            GO2HardPACTCfg.normalization.obs_scales.base_wrench
        )
        action_clip = GO2HardPACTCfg.normalization.clip_actions
