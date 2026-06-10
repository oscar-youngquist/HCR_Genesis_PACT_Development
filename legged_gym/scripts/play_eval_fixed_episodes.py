from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *

import numpy as np
import torch
from legged_gym.scripts.joystick import Joystick
from legged_gym.utils.exp_data_logger import ExpLogger
import argparse

def _fmt_bound_value(x):
    """Format numeric bounds so they are filename-friendly."""
    return f"{float(x):g}".replace("-", "neg").replace(".", "p")


def _fmt_bounds(prefix, bounds):
    """Create filename-safe bound string like payload_neg3_12 or push_2_1_2."""
    return prefix + "_" + "_".join(_fmt_bound_value(v) for v in bounds)

def override_configs(env_cfg, args):
    """Override some environment configuration parameters for testing

    Args:
        env_cfg: environment configuration
        args: command line arguments
    """
    task_name = args.task
    # override some parameters for testing
    env_cfg.env.num_envs = args.num_envs                               # number of environments
    env_cfg.viewer.rendered_envs_idx = list(range(args.num_envs))      # render all robots if rendering
    
    ###
    #   Terrain Stuff
    ###
    #     shared terrain parameters (enforcing to be safe)
    env_cfg.terrain.horizontal_scale = 0.1                    # [m] distance between height samples in x and y direction
    env_cfg.terrain.vertical_scale = 0.005                    # [m] distance between height samples in z direction
    env_cfg.terrain.static_friction = 1.0                     # coefficient of static friction of the terrain
    env_cfg.terrain.dynamic_friction = 1.0                    # coefficient of dynamic friction of the terrain
    env_cfg.terrain.restitution = 0.                          # coefficient of restitution of the terrain
    env_cfg.terrain.curriculum = False                        # whether to generate terrain curriculum (no)
    
    # Terrain construction logic
    if args.terrain_type == "plane":
        env_cfg.terrain.mesh_type = 'plane'                     # plane, heightfield, trimesh
        env_cfg.terrain.plane_length = 200.0                    # [m]. plane size is 200x200x10 by default
        env_cfg.terrain.measure_heights = False                 # obtain height measurements
        env_cfg.terrain.obtain_terrain_info_around_feet = True  # whether to capture terrain info around feet (yes)
        
        env_cfg.rewards.scales.foot_clearance_terrain_aware = 0.0
        env_cfg.rewards.scales.foot_clearance = 0.3

        print("Adding Plane Terrain")
    else:
        env_cfg.terrain.mesh_type = "heightfield"
        env_cfg.terrain.num_rows = args.terrain_rows
        env_cfg.terrain.num_cols = args.terrain_cols
        env_cfg.terrain.border_size = 1.0
        env_cfg.terrain.selected   = True
        env_cfg.terrain.measure_heights = True # obtain height measurements
        env_cfg.terrain.obtain_terrain_info_around_feet = True    # whether to capture terrain info around feet (yes)

        
        # Keep all of the below fixed for the large-scale tests. Disturbances change, not the terrains.
        if args.terrain_type == "rough":                                                             # random uniform terrain
            env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.random_uniform_terrain", 
                                              "min_height" : -0.08, "max_height": 0.08, 
                                              "step":0.005, "downsampled_scale" : 0.2}
            print("Adding Rough Terrain")
        elif args.terrain_type =="slope":                                                            # slope terrain
            env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_sloped_terrain",
                                              "slope": -0.4, "platform_size": 3.0}
            print("Adding Slope Terrain")
        elif args.terrain_type == "stairs":                                                           # stairs terrain
            env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_stairs_terrain",
                                            "step_width": 0.40, "step_height": -0.10, "platform_size": 3.0}
            print("Adding Stairs Terrain")
        elif args.terrain_type == "discrete":                                                         # discrete obstacles terrain
            env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.discrete_obstacles_terrain",
                                            "max_height": 0.1,
                                            "min_size": 1.0,
                                            "max_size": 2.0,
                                            "num_rects": 20,
                                            "platform_size": 3.0}
            print("Adding Discrete Terrain")
        elif args.terrain_type == "wave":
            env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.wave_terrain", 
                                            "amplitude": 0.2, "num_waves": 2}
            print("Adding Wave Terrain")
        else:
            f"Terrain {args.terrain_type} is not supported. Please select one of the following - [plane, rough, slope, stairs, discrete, wave]"
        
            
    ###
    #   Cmd sampling/joystick stuff
    ###
    # Command sampling stuff
    if args.use_joystick:
        env_cfg.commands.heading_command = False
    else:
        env_cfg.commands.ranges.lin_vel_x   = [-1.0, 1.0]
        env_cfg.commands.ranges.lin_vel_y   = [-1.0, 1.0]
        env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]
        env_cfg.commands.ranges.heading     = [-3.14, 3.14]
    env_cfg.commands.resampling_time = 5.0

    # Slightly relaxed from training termination conditions
    env_cfg.termination.roll_threshold = 1.57
    env_cfg.termination.pitch_threshold = 1.57
    env_cfg.termination.height_min = 0.0
    # env_cfg.asset.terminate_after_contacts_on = ["base","trunk"]
    env_cfg.asset.terminate_after_contacts_on = ["base","trunk","hip"]

    # Turn off/on domain randomization elements
    env_cfg.noise.add_noise = True
    # Disable some of the domain randomization (our payload will handle that now)
    if args.more_rand:
        env_cfg.domain_rand.randomize_pd_gain = True
        env_cfg.domain_rand.randomize_motor_strength = True
    else:
        env_cfg.domain_rand.randomize_pd_gain = False
        env_cfg.domain_rand.randomize_motor_strength = False
    
    # Just sample a value right in the middle of the training ranges
    env_cfg.domain_rand.joint_friction_range_end    = [0.1, 0.1]
    env_cfg.domain_rand.joint_friction_range_start  = [0.1, 0.1]
    
    env_cfg.domain_rand.joint_armature_range        = [0.0075, 0.0075]
    
    env_cfg.domain_rand.randomize_joint_stiffness = False
    env_cfg.domain_rand.joint_stiffness_range_start = [0.01, 0.01]
    env_cfg.domain_rand.joint_stiffness_range_end   = [0.01, 0.01]
    
    env_cfg.domain_rand.joint_damping_range_start   = [0.40, 0.40]
    env_cfg.domain_rand.joint_damping_range_end     = [0.40, 0.40]


    # Enable/disable disturbances as requested
    if args.disturbance_type == "none":
        print("Adding No Disturbances")
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_com_displacement = False
        env_cfg.domain_rand.randomize_base_mass = False
    elif args.disturbance_type == "payload":
        print("Adding Randomized Payloads!")
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_base_mass = True

        env_cfg.domain_rand.min_added_mass_max = args.payload_bounds[1]
        env_cfg.domain_rand.max_added_mass_max = args.payload_bounds[1]
        env_cfg.domain_rand.added_mass_min = args.payload_bounds[0]

        if args.shift_com:                                               # shift CoM with payload?
            print("Adding CoM Randomization!")
            print("Adding CoM Randomization!")
            env_cfg.domain_rand.randomize_com_displacement = True
            # COM displacement crap
            env_cfg.domain_rand.com_displacement_x_min = args.com_bounds[0]
            env_cfg.domain_rand.com_displacement_x_max = args.com_bounds[0]
            
            env_cfg.domain_rand.com_displacement_y_min = args.com_bounds[1]
            env_cfg.domain_rand.com_displacement_y_max = args.com_bounds[1]
            
            env_cfg.domain_rand.com_displacement_z_positive = False
            env_cfg.domain_rand.com_displacement_z_min_pos = 0.1
            env_cfg.domain_rand.com_displacement_z_min = args.com_bounds[2]
            env_cfg.domain_rand.com_displacement_z_max = args.com_bounds[2]
        else:
            env_cfg.domain_rand.randomize_com_displacement = False

    elif args.disturbance_type == "push":
        print("Adding external pushes!")
        env_cfg.domain_rand.push_robots = True
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_com_displacement = False

        # Random impulse time ranges
        env_cfg.domain_rand.push_interval_max = 5.0
        env_cfg.domain_rand.push_interval_min = 1.0
        env_cfg.domain_rand.vert_interval_max = 5.0
        env_cfg.domain_rand.vert_interval_min = 1.0
        env_cfg.domain_rand.wrench_timeout_min = 5.0
        env_cfg.domain_rand.wrench_timeout_max = 1.0
        
        # Random impulse magnitudes
        env_cfg.domain_rand.max_push_vel_xy = args.push_bounds[0]
        env_cfg.domain_rand.min_push_vel_xy = args.push_bounds[0]
        env_cfg.domain_rand.max_vertical_push = args.push_bounds[1]
        env_cfg.domain_rand.min_vertical_push = args.push_bounds[1]
        env_cfg.domain_rand.max_push_torque = args.push_bounds[2]
        env_cfg.domain_rand.min_push_torque = args.push_bounds[2]

    # Training artifact unique to PACT that needs to be disabled.
    if args.rand_pact:
        env_cfg.control.randomize_pact_weights = True
    else:
        env_cfg.control.randomize_pact_weights = False

    # Ensure debugging visulization stuff is disabled.
    env_cfg.asset.fix_base_link = False
    env_cfg.env.debug_viz = False
    env_cfg.env.debug = False
    env_cfg.env.debug_draw_terrain_height_points = False

    # Add extra floating camera to scene
    if args.record_frames or args.follow_robot:
        env_cfg.viewer.add_camera = True  # use a extra camera for moving

    # Construct the log-file output path
    # args.output_path = os.path.join(args.log_path, f"{args.task}_{args.terrain_type}_{args.disturbance_type}_episodes.csv")
    if args.disturbance_type == "payload":
        disturbance_tag = _fmt_bounds("payload", args.payload_bounds)

        if args.shift_com:
            disturbance_tag += "_" + _fmt_bounds("com", args.com_bounds)

    elif args.disturbance_type == "push":
        disturbance_tag = _fmt_bounds("push", args.push_bounds)

    else:
        disturbance_tag = args.disturbance_type

    args.output_path = os.path.join(
        args.log_path,
        f"{args.task}_{args.terrain_type}_{disturbance_tag}_episodes.csv"
    )
    

def print_debug_info(env, robot_index):
    """Print debug information while interacting

    Args:
        env: environment object
        robot_index (int): index of the robot to print info for
    """
    # print debug info
    # print("base lin vel: ", env.simulator.base_lin_vel[robot_index, :].cpu().numpy())
    # print("base yaw angle: ", env.simulator.base_euler[robot_index, 2].item())
    # print("base height: ", env.simulator.base_pos[robot_index, 2].cpu().numpy())
    # print("foot_height: ", env.simulator.feet_pos[robot_index, :, 2].cpu().numpy())
    # print(f"ankle pitch: {env.simulator.dof_pos[robot_index, [3,7]].cpu().numpy()}")
    pass


def _to_bool_tensor(value, device, num_envs=None):
    """Convert a scalar/list/np array/torch tensor into a bool tensor on device."""
    if value is None:
        if num_envs is None:
            raise ValueError("num_envs must be provided when value is None")
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    if torch.is_tensor(value):
        return value.detach().to(device=device).bool().view(-1)
    return torch.as_tensor(value, device=device).bool().view(-1)


def _get_timeout_mask(env, infos, dones, pre_step_episode_lengths=None):
    """
    Robustly infer which reset events were caused by episode timeouts.

    RSL-RL/legged_gym variants commonly expose timeout information in one of:
        - infos["time_outs"]
        - infos["timeouts"]
        - env.time_out_buf

    If none are available, fall back to comparing the pre-step episode length
    against env.max_episode_length. This fallback should work when the env uses
    episode_length_buf to trigger time-limit resets.
    """
    device = dones.device
    num_envs = dones.numel()

    timeout_value = None
    if isinstance(infos, dict):
        for key in ("time_outs", "timeouts", "time_out", "timeout"):
            if key in infos:
                timeout_value = infos[key]
                break

    if timeout_value is None and hasattr(env, "time_out_buf"):
        timeout_value = getattr(env, "time_out_buf")

    if timeout_value is not None:
        timeout_mask = _to_bool_tensor(timeout_value, device, num_envs=num_envs)
        if timeout_mask.numel() == 1:
            timeout_mask = timeout_mask.repeat(num_envs)
        return timeout_mask[:num_envs] & dones

    if pre_step_episode_lengths is not None and hasattr(env, "max_episode_length"):
        # Most legged_gym envs increment episode_length_buf during step and then
        # reset environments whose length reaches max_episode_length.
        return ((pre_step_episode_lengths.to(device).view(-1) + 1) >= env.max_episode_length) & dones

    return torch.zeros_like(dones, dtype=torch.bool)


def _get_failure_reset_mask(env, dones, timeout_mask):
    """
    Robustly infer which reset events were caused by failure instead of timeout.

    Prefer env.get_failure_idx() when available. If it is unavailable or has an
    unexpected shape, use the conservative definition: done and not timeout.
    """
    try:
        failure_mask = _to_bool_tensor(env.get_failure_idx(), dones.device, num_envs=dones.numel())
        if failure_mask.numel() == dones.numel():
            return failure_mask & dones & (~timeout_mask)
    except Exception:
        pass
    return dones & (~timeout_mask)


def _get_observations_for_task(env, task_name):
    """Fetch initial observations using the same task-specific conventions as the original play script."""
    if "ts" in task_name or "cat" in task_name:
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
        return {"obs_buf": obs_buf, "privileged_obs_buf": privileged_obs_buf,
                "obs_history": obs_history, "critic_obs": critic_obs}
    elif "ee" in task_name:
        estimator_features, _, _ = env.get_observations()
        return {"estimator_features": estimator_features}
    elif "dreamwaq" in task_name:
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states = env.get_observations()
        return {"obs_buf": obs_buf, "privileged_obs_buf": privileged_obs_buf,
                "obs_history": obs_history, "explicit_labels": explicit_labels,
                "next_states": next_states}
    elif "pact" in task_name:
        obs_buf, obs_history, privileged_obs_buf, explicit_labels = env.get_observations()
        return {"obs_buf": obs_buf, "obs_history": obs_history,
                "privileged_obs_buf": privileged_obs_buf, "explicit_labels": explicit_labels}
    elif "abl" in task_name:
        obs_buf, obs_history, privileged_obs_buf, explicit_labels = env.get_observations()
        return {"obs_buf": obs_buf, "obs_history": obs_history,
                "privileged_obs_buf": privileged_obs_buf, "explicit_labels": explicit_labels}
    elif "pos" in task_name and "pact" not in task_name:
        obs_buf, obs_history, privileged_obs_buf, explicit_labels = env.get_observations()
        return {"obs_buf": obs_buf, "obs_history": obs_history,
                "privileged_obs_buf": privileged_obs_buf, "explicit_labels": explicit_labels}
    elif "tau" in task_name:
        obs_buf, obs_history, privileged_obs_buf, explicit_labels = env.get_observations()
        return {"obs_buf": obs_buf, "obs_history": obs_history,
                "privileged_obs_buf": privileged_obs_buf, "explicit_labels": explicit_labels}
    elif "rl2ac" in task_name:
        obs_buf, obs_history, privileged_obs_buf, explicit_labels = env.get_observations()
        return {"obs_buf": obs_buf, "obs_history": obs_history,
                "privileged_obs_buf": privileged_obs_buf, "explicit_labels": explicit_labels}
    else:
        obs = env.get_observations()
        return {"obs": obs}


def _policy_step_for_task(env, policy, task_name, state):
    """Run one policy/env step and update the task-specific observation state dict."""
    if "ts" in task_name or "cat" in task_name:
        actions = policy(state["obs_buf"], state["obs_history"])
        obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
        state.update(obs_buf=obs_buf, privileged_obs_buf=privileged_obs_buf,
                     obs_history=obs_history, critic_obs=critic_obs)
    elif "ee" in task_name:
        actions = policy(state["estimator_features"].detach())
        estimator_features, estimator_labels, _, rews, dones, infos = env.step(actions.detach())
        state.update(estimator_features=estimator_features, estimator_labels=estimator_labels)
    elif "waq" in task_name:
        actions = policy(state["obs_buf"], state["obs_history"])
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos = env.step(actions.detach())
        state.update(obs_buf=obs_buf, privileged_obs_buf=privileged_obs_buf,
                     obs_history=obs_history, explicit_labels=explicit_labels,
                     next_states=next_states)
    elif "pact" in task_name:
        actions = policy(state["obs_buf"], state["obs_history"])
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, rews, dones, infos, grfs = env.step(actions.detach())
        state.update(obs_buf=obs_buf, privileged_obs_buf=privileged_obs_buf,
                     obs_history=obs_history, explicit_labels=explicit_labels, grfs=grfs)
    elif "abl" in task_name:
        actions = policy(state["obs_buf"], state["obs_history"])
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, rews, dones, infos, grfs = env.step(actions.detach())
        state.update(obs_buf=obs_buf, privileged_obs_buf=privileged_obs_buf,
                     obs_history=obs_history, explicit_labels=explicit_labels, grfs=grfs)
    elif "pos" in task_name and "pact" not in task_name:
        actions = policy(state["obs_buf"], state["obs_history"])
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, rews, dones, infos = env.step(actions.detach())
        state.update(obs_buf=obs_buf, privileged_obs_buf=privileged_obs_buf,
                     obs_history=obs_history, explicit_labels=explicit_labels)
    elif "tau" in task_name:
        actions = policy(state["obs_buf"], state["obs_history"])
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, rews, dones, infos = env.step(actions.detach())
        state.update(obs_buf=obs_buf, privileged_obs_buf=privileged_obs_buf,
                     obs_history=obs_history, explicit_labels=explicit_labels)
    elif "rl2ac" in task_name:
        actions, qref = policy(state["obs_buf"], state["obs_history"])
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, rews, dones, infos, grfs = env.step(actions.detach(), qref)
        state.update(obs_buf=obs_buf, privileged_obs_buf=privileged_obs_buf,
                     obs_history=obs_history, explicit_labels=explicit_labels, grfs=grfs,
                     qref=qref)
    else:
        actions = policy(state["obs"].detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        state.update(obs=obs)

    return state, rews, dones, infos


def interaction_loop(train_cfg, env, policy, args):
    """Run evaluation for a fixed number of completed episodes, not fixed timesteps.

    Episode accounting is global across vectorized environments. For example,
    with --num_envs 16 and --num_eps 100, the script exits once 100 total
    env-episodes have completed. Each row is tagged with:
        - episode: global episode id assigned to that env rollout
        - episode_step: 1-indexed transition index within that episode
        - done: whether this transition ended the episode
        - time_out: whether the reset was induced by episode time limit
        - failure_reset: whether the reset was induced by failure
        - valid_episode: whether the row belongs to one of the requested episodes

    Because vectorized envs can finish multiple episodes on the same step, the
    final logged step may include rows with valid_episode=0 or episode ids beyond
    the requested count. Filter valid_episode==1 in downstream analysis.
    """

    robot_index = 0  # which robot is used for camera following
    task_name = args.task
    target_episodes = int(args.num_eps)
    if target_episodes <= 0:
        raise ValueError(f"--num_eps must be positive, got {args.num_eps}")

    logger = None
    if args.log:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        logger = ExpLogger(args.output_path)

    state = _get_observations_for_task(env, task_name)

    if args.use_joystick:
        joystick = Joystick(joystick_type=args.joystick_type)

    if args.record_frames:
        env.simulator._floating_camera.start_recording()

    num_envs = env.num_envs if hasattr(env, "num_envs") else args.num_envs
    device = env.device

    # Assign global episode ids to the first wave of vectorized env rollouts.
    episode_id = torch.arange(num_envs, dtype=torch.long, device=device)
    episode_active = episode_id < target_episodes
    next_episode_id = int(min(num_envs, target_episodes))
    completed_episodes = 0
    episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    global_step = 0

    print(f"Collecting {target_episodes} completed episodes across {num_envs} envs.")

    while completed_episodes < target_episodes:
        global_step += 1

        if hasattr(env, "episode_length_buf"):
            pre_step_episode_lengths = env.episode_length_buf.detach().clone()
        else:
            pre_step_episode_lengths = None

        # Update commands from joystick.
        if args.use_joystick:
            joystick.update()
            env.commands[:, 0] = -joystick.ly
            env.commands[:, 1] = -joystick.lx
            env.commands[:, 2] = -joystick.rx

        # Override with fixed command, if requested.
        if args.fixed_cmd is not None:
            env.commands[:, 0] = args.fixed_cmd[0]
            env.commands[:, 1] = args.fixed_cmd[1]
            env.commands[:, 2] = args.fixed_cmd[2]
            env.commands[:, 3] = args.fixed_cmd[3]

        if args.follow_robot:
            pos = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.pos, dtype=np.float32)
            lookat = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.lookat, dtype=np.float32)
            env.set_camera(pos, lookat)
            env.simulator._floating_camera.render()

        # Step policy and environment.
        state, rews, dones, infos = _policy_step_for_task(env, policy, task_name, state)
        dones = _to_bool_tensor(dones, device, num_envs=num_envs)

        # These labels refer to the transition that was just executed.
        log_episode_id = episode_id.detach().clone()
        log_episode_step = episode_step.detach().clone() + 1
        log_valid_episode = episode_active.detach().clone()

        timeout_mask = _get_timeout_mask(env, infos, dones, pre_step_episode_lengths)
        failure_reset_mask = _get_failure_reset_mask(env, dones, timeout_mask)
        completed_mask = dones & episode_active

        if args.log:
            logger.log_states(
                {
                    'episode': log_episode_id.detach().cpu().numpy().tolist(),
                    'episode_step': log_episode_step.detach().cpu().numpy().tolist(),
                    'global_step': [global_step] * num_envs,
                    'valid_episode': list(map(int, log_valid_episode.detach().cpu().numpy().tolist())),
                    'done': list(map(int, dones.detach().cpu().numpy().tolist())),
                    'time_out': list(map(int, timeout_mask.detach().cpu().numpy().tolist())),
                    'failure_reset': list(map(int, failure_reset_mask.detach().cpu().numpy().tolist())),
                    
                    'base_cmd': env.commands.detach().cpu().numpy().tolist(),
                    'base_pose': env.simulator.base_pos.detach().cpu().numpy().tolist(),
                    'base_rpy': env.simulator.base_euler.detach().cpu().numpy().tolist(),
                    'dof_pose': env.simulator.dof_pos.detach().cpu().numpy().tolist(),
                    'base_lin_vel': env.simulator.base_lin_vel.detach().cpu().numpy().tolist(),
                    'base_ang_vel': env.simulator.base_ang_vel.detach().cpu().numpy().tolist(),
                    'dof_vel': env.simulator.dof_vel.detach().cpu().numpy().tolist(),
                    'proj_grav': env.simulator.projected_gravity.detach().cpu().numpy().tolist(),
                    'feet_pos': env.simulator.feet_pos.detach().cpu().numpy().tolist(),
                    'tau_act': env.simulator._dof_tau.detach().cpu().numpy().tolist(),
                    'grf': env.simulator._grfs_buf.detach().cpu().numpy().tolist(),
                    'q_des': env.get_scaled_pos_actions().detach().cpu().numpy().tolist(),
                    'tau_ff': env.simulator.feedforward_torques.detach().cpu().numpy().tolist(),
                    'tau_pd': env.simulator.first_loop_feedback.detach().cpu().numpy().tolist(),
                    
                    # Kept for backward compatibility with your previous CSV format.
                    'failure': list(map(int, failure_reset_mask.detach().cpu().numpy().tolist())),
                    'payload': env.simulator._added_base_mass.detach().cpu().numpy().tolist(),
                    'com_shift': env.simulator._base_com_bias.detach().cpu().numpy().tolist(),
                    'rand_push': env.simulator._rand_push_vels.detach().cpu().numpy().tolist(),
                    'rand_wrench': env.simulator._rand_wrench_vels.detach().cpu().numpy().tolist()
                }
            )

        # Advance per-env episode step counters for active rollouts.
        episode_step[episode_active] += 1

        # Retire completed episodes and assign fresh global episode ids to envs
        # that reset while more requested episodes remain.
        done_env_ids = torch.nonzero(dones, as_tuple=False).flatten().detach().cpu().tolist()
        for env_id in done_env_ids:
            if bool(episode_active[env_id].item()):
                completed_episodes += 1

            episode_step[env_id] = 0

            if next_episode_id < target_episodes:
                episode_id[env_id] = next_episode_id
                episode_active[env_id] = True
                next_episode_id += 1
            else:
                episode_id[env_id] = -1
                episode_active[env_id] = False

        if args.progress_interval > 0 and (global_step % args.progress_interval == 0 or completed_mask.any()):
            print(f"Completed {completed_episodes}/{target_episodes} episodes after {global_step} vectorized steps.")

    if logger is not None:
        logger.save_log()
        print(f"Saved episode-based evaluation log to: {args.output_path}")

def export_policy(alg_runner, path: str, args, env_cfg, train_cfg):
    """export the policy as jit script according to different task types

    Args:
        alg_runner: algorithm runner
        path (str): path to which the policy is exported
        args: command line arguments
        env_cfg: environment configuration
        train_cfg: training configuration
    """
    task_name = args.task
    if "ts" in task_name or "cat" in task_name:
        exporter = PolicyExporterTS(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif "ee" in task_name:
        exporter = PolicyExporterEE(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif "dreamwaq" in task_name:
        exporter = PolicyExporterWaQ(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif "pact" in task_name:
        exporter = PolicyExporterPACT(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, train_cfg)
    else:
        exporter = PolicyExporter(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    
    print('Exported policy as jit script to: ', path)
    if args.export_onnx:
        print('Exported policy as onnx to: ', path)
    

def play(args):
    """Main function to run the play script

    Args:
        args (_type_): command line arguments
    """
    print_experiment_settings(args)
    
    if "genesis" in SIMULATOR:
        init_genesis(args, gs)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, args=args)
    override_configs(env_cfg, args)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++ or python)
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'pact_corl', train_cfg.runner.experiment_name, 
                            train_cfg.runner.load_run, 'exported')
    # export_policy(ppo_runner, path, args, env_cfg, train_cfg)

    interaction_loop(train_cfg, env, policy, args)

    if args.record_frames:
        try:
            filename_mp4 = f"{args.task}_{args.terrain_type}_{args.disturbance_type}_video.mp4"
        except:
            from datetime import datetime
            filename_mp4 = f"{datetime.now().timestamp()}"
        
        env.simulator._floating_camera.stop_recording(save_to_filename=filename_mp4, fps=30)
        print("Saved recording to " + filename_mp4)
    
def print_experiment_settings(args):
    """Pretty print all command-line experiment settings."""
    print("\n===== Experiment Settings =====")
    for k, v in vars(args).items():
        print(f"{k:20s}: {v}")
    print("================================\n")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',             type=str, default='go2', help="task name")
    parser.add_argument('--headless',         action='store_true', default=False, help="enable visualization by default")
    parser.add_argument('--cpu',              action='store_true', default=False, help="use CPU instead of CUDA")
    parser.add_argument('--gpu',              type=str, default='cuda:0', help="which GPU to use (default: cuda:0)")
    parser.add_argument('--num_envs',         type=int, default=1, help="number of parallel environments")
    parser.add_argument('--max_iterations',   type=int, default=None, help="max number of training iterations")
    parser.add_argument('--resume',           action='store_true', default=False, help="resume training from specified checkpoint")
    parser.add_argument('--sync_wandb',       action='store_true', default=False, help="synchronize training log with wandb")
    parser.add_argument('--export_onnx',      action='store_true', default=False, help="export policy as onnx (besides jit)")
    parser.add_argument('--debug',            action='store_true', default=False, help="enable debug mode")
    parser.add_argument('--load_run',         type=str, default=None, help="run to load, default: last run")
    parser.add_argument('--ckpt',             type=int, default=-1, help="checkpoint to load, -1 means latest")
    parser.add_argument('--use_joystick',     action='store_true', default=False, help="use joystick to provide commands")
    parser.add_argument('--joystick_type',    type=str, default='xbox', help="type of joystick: xbox, switch")
    parser.add_argument('--follow_robot',     action='store_true', default=False, help="whether the camera follows the robot during play")
    parser.add_argument('--record_frames',    action='store_true', default=False, help="whether to record the camera")

    parser.add_argument('--seed',             type=int, default=1, help="int seed for random sampling (default 1)")

    # PACT PINN specific thing.
    parser.add_argument('--pinn_loss_weight', type=float, default=0.01, help="float for weight of PINN loss (default 0.01)")

    # large scale experiment specific arguments
    parser.add_argument('--log',              action='store_true', default=False, help="log results to csv file.")
    
    # Terrain selection parameters
    parser.add_argument('--terrain_type',     type=str, default='plane', help="Terrain type to be evaluted (options - plane, rough, slope, stairs, discrete, waves. Default - plane)")
    parser.add_argument('--terrain_rows',     type=int, default=4, help="Number of rows of rough terrains to generate (default - 2)")
    parser.add_argument('--terrain_cols',     type=int, default=4, help="Number of cols of rough terrains to generate (default - 2)")

    # Disturbance parameters
    parser.add_argument('--disturbance_type', type=str, default='none', help="Type of disturbance applied to robot (options - none, payload, push. Default - none)")
    parser.add_argument('--payload_bounds',   type=float, nargs='+', default=[-3.0, 12.0], help="min and max payload sample range (default - [-3.0, 12.0])")
    parser.add_argument('--shift_com',        action='store_true', default=False, help="whether or not to randomize the CoM when transporting payloads. (default - False)")
    parser.add_argument('--com_bounds',       type=float, nargs='+', default=[0.25, 0.20, 0.20], help="combined min/max COM-shift values [x, y, z] (default - [0.25, 0.20, 0.20])")
    parser.add_argument('--push_bounds',      type=float, nargs='+', default=[2.0, 1.0, 2.0], help="combined min/max external push velo. values [planer, vertical, wrench] (default - [1.0, 0.5, 1.0])")

    # Fixed command execution
    parser.add_argument('--fixed_cmd',        type=float, nargs='+', default=None, help="A fixed command to be executed throughout the experiment [x, y, ang, heading] (default: None)")

    parser.add_argument('--log_path',         type=str, default="exp_data/output", help="path to experiment output folder (default - 'exp_data/output')")

    parser.add_argument('--num_eps',          type=int, default=5, help="Number of completed evaluation episodes to collect globally across all envs (default - 5)")
    parser.add_argument('--progress_interval', type=int, default=10000, help="Print episode collection progress every N vectorized steps. Use 0 to disable.")

    parser.add_argument('--rand_pact',        action='store_true', default=False)
    parser.add_argument('--more_rand',        action='store_true', default=False)

    play(configure_runtime_device(parser.parse_args()))
