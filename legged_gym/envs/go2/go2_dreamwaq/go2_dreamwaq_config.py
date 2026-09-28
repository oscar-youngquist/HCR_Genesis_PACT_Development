from legged_gym import SIMULATOR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2DreamwaqCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 12
        num_observations = 45  # num_obs
        frame_stack = 5    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = 16
        num_explicit_dims = 11  # torso velocity, foot contact probabilities, foot heights
        num_decoder_output = num_observations
        c_frame_stack = 5
        single_critic_obs_len = num_observations + 31 + 187 + 17 + 3
        num_privileged_obs = c_frame_stack * single_critic_obs_len
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector

    class sim(LeggedRobotCfg.sim):
        use_dreamwaq_adapter = True

    class terrain(LeggedRobotCfg.terrain):
        if SIMULATOR in ["isaacgym", "isaaclab"]:
            mesh_type = "trimesh"
        else:
            mesh_type = "heightfield"
        border_size = 20.0 # [m]
        curriculum = True
        # rough terrain only:
        obtain_terrain_info_around_feet = True
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 11x17 = 187
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        terrain_length = 8.0
        terrain_width = 8.0
        platform_size = 4.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.1, 0.25, 0.25, 0.2]
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8, # [rad]
            'RL_thigh_joint': 0.8, # [rad]
            'FR_thigh_joint': 0.8, # [rad]
            'RR_thigh_joint': 0.8, # [rad]

            'FL_calf_joint': -1.5, # [rad]
            'RL_calf_joint': -1.5, # [rad]
            'FR_calf_joint': -1.5, # [rad]
            'RR_calf_joint': -1.5, # [rad]
        }
    class control(LeggedRobotCfg.control):
        stiffness = {'joint': 20.}   # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        dt = 0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT
    class asset(LeggedRobotCfg.asset):
        name = "go2"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        foot_name = "foot"
        # full name of the base link
        base_link_name = "base"
        dof_names = [           # align with the real robot
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
            "RL_calf_joint"
        ]
        # For Genesis
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        dof_vel_limits = [30.1, 30.1, 15.7,
                          30.1, 30.1, 15.7,
                          30.1, 30.1, 15.7,
                          30.1, 30.1, 15.7]
        obtain_link_contact_states = True
        penalize_contacts_on = ["thigh", "calf", "base", "Head", "hip"]
        terminate_after_contacts_on = []
        feet_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        contact_state_link_names = ["base"] + [
            leg + "_" + link for leg in ("FR", "FL", "RR", "RL")
            for link in ("hip", "thigh", "calf", "foot")]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        foot_clearance_target = 0.09 # desired foot clearance above ground [m]
        foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = True
        soft_dof_vel_limit = 0.9
        overreach_x_max = 0.28
        rear_foot_x_nominal = -0.25
        rear_foot_x_margin = 0.08
        overreach_contact_force_threshold = 5.0
        use_reward_curriculum = True
        soft_torque_limit = 0.90  # PACT torque-limit penalty starts before saturation.
        contact_force_threshold = 1.0  # N, foot-force norm
        class scales(LeggedRobotCfg.rewards.scales):
            dof_pos_limits = -2.0
            dof_vel_limits = -1.0
            torque_limits = -0.01
            collision = -1.0

            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5

            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            hip_pos = -0.2
            
            dof_power = -2.e-4
            dof_acc = -2.e-7

            action_rate = -0.01
            action_smoothness = -0.01

            feet_air_time = 1.0
            foot_clearance = 0.2
            feet_contact_stand_still = 0.5

            front_foot_overreach = -100.0
            rear_foot_overreach = -10.0
        class reward_curriculum:
            warmup_steps = 0
            curr_steps = 5000
            curr_reward_bounds = {
                "ang_vel_xy": (-0.05, -0.2),
                "orientation": (-0.2, -1.0),
                "torque_limits": (-0.01, -1.0),
                "hip_pos": (-0.2, -0.4),
                "action_rate": (-0.001, -0.01),
                "action_smoothness": (-0.001, -0.01),
            }

    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_domainrand_curriculum = True
        reset_resample_episodes = 25  # Per environment; 0/1 resamples every reset.
        com_rand_z_positive = False
        push_warmup = 2000     # number of steps with initial values held constant

        # Randomize Friction
        randomize_friction = True
        friction_range = [0.2, 1.25]

        # Torso linear/angular velocity impulses
        push_robots = True
        push_interval_max = 15.0
        push_interval_min = 5.00
        max_push_vel_xy = 1.00
        min_push_vel_xy = 0.50

        max_vertical_push = 0.50
        min_vertical_push = 0.10
        vert_interval_max = 15.0
        vert_interval_min = 5.00

        max_push_torque = 1.00
        min_push_torque = 0.50
        wrench_timeout_max = 15.0
        wrench_timeout_min = 5.00

        # Randomized base mass, applied at COM
        randomize_base_mass = True
        min_added_mass_max = 2.0
        max_added_mass_max = 4.0
        added_mass_min = -1.0

        # CoM displacement
        randomize_com_displacement = True
        com_displacement_x_min = 0.05
        com_displacement_x_max = 0.05

        com_displacement_y_min = 0.05
        com_displacement_y_max = 0.05

        com_displacement_z_min = 0.05
        com_displacement_z_max = 0.05

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

        # Joint dynamics
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


        # Legacy backend adapters consume the current ranges below.
        added_mass_range = [-1., 4.]
        com_pos_x_range = [-0.075, 0.075]
        com_pos_y_range = [-0.075, 0.075]
        com_pos_z_range = [-0.075, 0.075]
        joint_friction_range = joint_friction_range_start
        joint_damping_range = joint_damping_range_start
        # The adapter owns individual event timers, ticked every control step.
        push_interval_s = 0.02

class Go2DreamwaqCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "DreamWaQRunner"
    class policy( LeggedRobotCfgPPO.policy ):
        critic_hidden_dims = [1024, 256, 128]
        encoder_hidden_dims = [256, 128]
        decoder_hidden_dims = [256, 128]
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        encoder_lr = 2.e-4
        num_encoder_epochs = 1
        vae_kld_weight = 2.0
        explicit_loss_weight = 1.0
        contact_probability_loss_weight = 0.10
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticDreamWaQ"
        algorithm_class_name = "PPO_DreamWaQ"
        run_name = 'dreamwaq'
        if SIMULATOR == "genesis":
            run_name += "_genesis"
        elif SIMULATOR == "isaacgym":
            run_name += "_isaacgym"
        elif SIMULATOR == "isaaclab":
            run_name += "_isaaclab"
        experiment_name = 'go2_rough'
        save_interval = 500
        max_iterations = 10000

        load_run = -1
        checkpoint = -1
        resume = False
