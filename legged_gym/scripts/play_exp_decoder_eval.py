from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *

import numpy as np
import torch
from legged_gym.scripts.joystick import Joystick
from legged_gym.scripts.play_exp_largescale import override_configs, print_experiment_settings
from legged_gym.utils.exp_data_logger import ExpLogger
import argparse


def interaction_loop(train_cfg, env, alg_runner, args):
    """Run PACT/ABL3 evaluation while logging decoder predictions and latest privileged obs."""
    robot_index = 0
    task_name = args.task

    if "pact" not in task_name and "abl3" not in task_name:
        raise ValueError("play_exp_decoder_eval.py is intended for go1_pact and go1_abl3-style tasks.")

    logger = ExpLogger(args.output_path, ref_key="gt_priv_obs_latest") if args.log else None

    obs_buf, obs_history, privileged_obs_buf, explicit_labels = env.get_observations()

    actor_critic = alg_runner.alg.actor_critic
    decoder = alg_runner.alg.decoder
    actor_critic.eval()
    decoder.eval()
    actor_critic.to(env.device)
    decoder.to(env.device)

    num_priv_obs = alg_runner.alg.num_priv_obs



    if args.use_joystick:
        joystick = Joystick(joystick_type=args.joystick_type)

    if args.record_frames:
        env.simulator._floating_camera.start_recording()

    for _ in range(int(args.num_eps * env.max_episode_length)):
        if args.use_joystick:
            joystick.update()
            env.commands[:, 0] = -joystick.ly
            env.commands[:, 1] = -joystick.lx
            env.commands[:, 2] = -joystick.rx

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

        with torch.inference_mode():
            actions, context_latent, context_torso_velo = actor_critic.act_inference_recon(obs_buf, obs_history)
            decoder_input = torch.cat((context_latent, context_torso_velo), dim=-1)
            
            if args.more_rand:
                decoder_input += 0.01 * torch.randn_like(decoder_input)
            
            decoder_pred = decoder(decoder_input)

        obs_buf, privileged_obs_buf, obs_history, explicit_labels, rews, dones, infos, grfs = env.step(actions.detach())
        privileged_obs_latest = privileged_obs_buf[:, -num_priv_obs:]

        if logger is not None:
            logger.log_states(
                {
                    # 'base_cmd': env.commands.detach().cpu().numpy().tolist(),
                    # 'base_pose': env.simulator.base_pos.detach().cpu().numpy().tolist(),
                    # 'base_rpy': env.simulator.base_euler.detach().cpu().numpy().tolist(),
                    # 'dof_pose': env.simulator.dof_pos.detach().cpu().numpy().tolist(),
                    # 'base_lin_vel': env.simulator.base_lin_vel.detach().cpu().numpy().tolist(),
                    # 'base_ang_vel': env.simulator.base_ang_vel.detach().cpu().numpy().tolist(),
                    # 'dof_vel': env.simulator.dof_vel.detach().cpu().numpy().tolist(),
                    # 'proj_grav': env.simulator.projected_gravity.detach().cpu().numpy().tolist(),
                    # 'feet_pos': env.simulator.feet_pos.detach().cpu().numpy().tolist(),
                    # 'tau_act': env.simulator._dof_tau.detach().cpu().numpy().tolist(),
                    # 'grf': env.simulator._grfs_buf.detach().cpu().numpy().tolist(),
                    # 'q_des': env.get_scaled_pos_actions().detach().cpu().numpy().tolist(),
                    # 'tau_ff': env.simulator.feedforward_torques.detach().cpu().numpy().tolist(),
                    # 'tau_pd': env.simulator.first_loop_feedback.detach().cpu().numpy().tolist(),
                    'failure': list(map(int, env.get_failure_idx().detach().cpu().numpy().tolist())),
                    # 'payload': env.simulator._added_base_mass.detach().cpu().numpy().tolist(),
                    # 'com_shift': env.simulator._base_com_bias.detach().cpu().numpy().tolist(),
                    # 'rand_push': env.simulator._rand_push_vels.detach().cpu().numpy().tolist(),
                    # 'rand_wrench': env.simulator._rand_wrench_vels.detach().cpu().numpy().tolist(),
                    'decoder_pred_priv_obs': decoder_pred.detach().cpu().numpy().tolist(),
                    'gt_priv_obs_latest': privileged_obs_latest.detach().cpu().numpy().tolist(),
                    # 'context_latent': context_latent.detach().cpu().numpy().tolist(),
                    # 'context_torso_velo': context_torso_velo.detach().cpu().numpy().tolist(),
                }
            )

    if logger is not None:
        logger.save_log()


def play(args):
    print_experiment_settings(args)

    if "genesis" in SIMULATOR:
        init_genesis(args, gs)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, args=args)
    override_configs(env_cfg, args)
    
    env_cfg.noise.add_noise = False
    
    args.output_path = os.path.join(
        args.log_path,
        f"{args.task}_{args.terrain_type}_{args.disturbance_type}_decoder_eval.csv",
    )

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    interaction_loop(train_cfg, env, ppo_runner, args)

    if args.record_frames:
        filename_mp4 = f"{args.task}_{args.terrain_type}_{args.disturbance_type}_decoder_eval_video.mp4"
        env.simulator._floating_camera.stop_recording(save_to_filename=filename_mp4, fps=30)
        print("Saved recording to " + filename_mp4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='go1_pact', help="task name")
    parser.add_argument('--headless', action='store_true', default=False, help="enable visualization by default")
    parser.add_argument('--cpu', action='store_true', default=False, help="use CPU instead of CUDA")
    parser.add_argument('--gpu', type=str, default='cuda:0', help="which GPU to use (default: cuda:0)")
    parser.add_argument('--num_envs', type=int, default=1, help="number of parallel environments")
    parser.add_argument('--max_iterations', type=int, default=None, help="max number of training iterations")
    parser.add_argument('--resume', action='store_true', default=False, help="resume training from specified checkpoint")
    parser.add_argument('--sync_wandb', action='store_true', default=False, help="synchronize training log with wandb")
    parser.add_argument('--export_onnx', action='store_true', default=False, help="export policy as onnx (besides jit)")
    parser.add_argument('--debug', action='store_true', default=False, help="enable debug mode")
    parser.add_argument('--load_run', type=str, default=None, help="run to load, default: last run")
    parser.add_argument('--ckpt', type=int, default=-1, help="checkpoint to load, -1 means latest")
    parser.add_argument('--use_joystick', action='store_true', default=False, help="use joystick to provide commands")
    parser.add_argument('--joystick_type', type=str, default='xbox', help="type of joystick: xbox, switch")
    parser.add_argument('--follow_robot', action='store_true', default=False, help="whether the camera follows the robot during play")
    parser.add_argument('--record_frames', action='store_true', default=False, help="whether to record the camera")
    parser.add_argument('--seed', type=int, default=1, help="int seed for random sampling (default 1)")
    parser.add_argument('--pinn_loss_weight', type=float, default=0.01, help="float for weight of PINN loss (default 0.01)")
    parser.add_argument('--log', action='store_true', default=False, help="log results to csv file.")
    
    parser.add_argument('--terrain_type', type=str, default='plane', help="Terrain type to be evaluated")
    parser.add_argument('--terrain_rows', type=int, default=4, help="Number of rows of rough terrains to generate")
    parser.add_argument('--terrain_cols', type=int, default=4, help="Number of cols of rough terrains to generate")
    
    parser.add_argument('--disturbance_type', type=str, default='none', help="Type of disturbance applied to robot")
    
    parser.add_argument('--payload_bounds', type=float, nargs='+', default=[10.0, 10.0], help="min and max payload sample range")
    parser.add_argument('--shift_com', action='store_true', default=False, help="whether to randomize CoM when transporting payloads")
    
    parser.add_argument('--com_bounds', type=float, nargs='+', default=[0.20, 0.15, 0.15], help="COM-shift values [x, y, z]")
    
    parser.add_argument('--push_bounds', type=float, nargs='+', default=[2.00, 1.0, 2.00], help="push values [planar, vertical, wrench]")
    
    parser.add_argument('--fixed_cmd', type=float, nargs='+', default=None, help="fixed command [x, y, ang, heading]")
    
    parser.add_argument('--log_path', type=str, default="exp_data/output", help="path to experiment output folder")
    
    parser.add_argument('--num_eps', type=float, default=5.0, help="Number of data collection episodes to run")
    
    parser.add_argument('--rand_pact', action='store_true', default=False)
    parser.add_argument('--more_rand', action='store_true', default=False)

    play(configure_runtime_device(parser.parse_args()))
