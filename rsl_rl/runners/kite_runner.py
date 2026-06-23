# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
import inspect
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO_KITE
from rsl_rl.modules import ActorCritic_KITE, export_kite_async_deployment_pipelines
from rsl_rl.env import VecEnv
from rsl_rl.utils import pretty_print_module


# ---------------- 4090 / Ada Lovelace performance knobs ----------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


class OnPolicyRunnerKITE:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.debug_autograd_anomaly = self.alg_cfg.get(
            "debug_autograd_anomaly", False
        )
        torch.autograd.set_detect_anomaly(self.debug_autograd_anomaly)
        
        self.device = device
        self.env = env
        
        # The critic sees raw privileged dynamics plus compact privileged
        # terrain/dynamics latents. Raw terrain maps stay outside critic_obs.
        latest_privileged_obs_dim = (
            self.env.num_privileged_obs
            if self.env.num_privileged_obs is not None
            else self.env.num_obs
        )
        num_critic_obs = (
            latest_privileged_obs_dim
            + self.policy_cfg.get("privileged_terrain_latent_dim", 32)
            + self.policy_cfg.get("privileged_dynamics_latent_dim", 16)
        )

        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        

        actor_critic: ActorCritic_KITE = actor_critic_class(self.env.num_obs,
                                                            self.env.num_obs_hist,
                                                            num_critic_obs,
                                                            self.env.depth_output_resolution,
                                                            self.policy_cfg["depth_image_latent_dim"],
                                                            self.policy_cfg["depth_image_norm"],
                                                            self.policy_cfg.get("depth_sequence_length", 5),
                                                            self.policy_cfg["depth_sequence_norm"],
                                                            
                                                            self.policy_cfg["proprio_in_dim"],
                                                            self.policy_cfg["proprio_latent_dim"],
                                                            self.policy_cfg["proprio_use_norm"],
                                                            self.policy_cfg["proprio_num_blocks"],
                                                            self.policy_cfg["proprio_hidden_dim"],
                                                            self.policy_cfg["proprio_token_dim"],
                                                            self.policy_cfg["proprio_channel_dim"],
                                                            
                                                            self.policy_cfg["mixer_velo_dim"],
                                                            self.policy_cfg["mixer_feet_state_dim"],
                                                            self.policy_cfg["mixer_latent_dim"],
                                                            self.policy_cfg["mixer_use_norm"],
                                                            self.policy_cfg.get("mixer_num_blocks", self.policy_cfg.get("mixer_blocks", 2)),
                                                            self.policy_cfg["mixer_hidden_dim"],
                                                            self.policy_cfg["mixer_token_dim"],
                                                            self.policy_cfg["mixer_channel_dim"],
                                                            self.env.num_actions,
                                                            self.policy_cfg["actor_layers"],
                                                            self.policy_cfg["critic_layers"],
                                                            self.policy_cfg["activation"],
                                                            self.policy_cfg["init_noise_std"],
                                                            ).to(self.device)        
        print("Created Actor-Critic Model")
        pretty_print_module(actor_critic)

        self.use_adaptive_entropy = self.alg_cfg.get("use_adaptive_entropy", False)

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        
        # Terrain maps are stored as height + XYZ normal channels and used by
        # PPO-owned privileged encoders/decoders.
        terrain_map_shape = (
            len(self.env.cfg.terrain.measured_points_x),
            len(self.env.cfg.terrain.measured_points_y),
            4,
        )
        alg_init_params = set(inspect.signature(alg_class.__init__).parameters)
        alg_cfg = {
            key: value
            for key, value in self.alg_cfg.items()
            if key in alg_init_params
        }
        self.alg: PPO_KITE = alg_class(
            actor_critic,
            latest_privileged_obs_dim,
            terrain_map_shape=terrain_map_shape,
            num_priv_obs_history=self.env.num_crit_obs_stack,
            privileged_terrain_latent_dim=self.policy_cfg.get("privileged_terrain_latent_dim", 32),
            privileged_dynamics_latent_dim=self.policy_cfg.get("privileged_dynamics_latent_dim", 16),
            priv_activation_func=self.policy_cfg.get("priv_activation", "elu"),
            cnn_norm_type=self.policy_cfg.get("cnn_norm_type", "layer"),
            terrain_encoder_attention_dim=self.policy_cfg.get("terrain_encoder_attention_dim", 128),
            terrain_encoder_n_heads=self.policy_cfg.get("terrain_encoder_n_heads", 4),
            terrain_decoder_hidden_dim=self.policy_cfg.get("terrain_decoder_hidden_dim", 128),
            terrain_decoder_encoded_spatial_dim=self.policy_cfg.get("terrain_decoder_encoded_spatial_dim", (3,4)),
            terrain_decoder_channels=self.policy_cfg.get("terrain_decoder_channels", 64),
            privileged_dynamics_decoder_layers=self.policy_cfg.get("privileged_dynamics_decoder_layers", [32,64,128,256]),
            priv_mixer_num_blocks=self.policy_cfg.get("priv_mixer_num_blocks", 2),
            priv_mixer_hidden_dim=self.policy_cfg.get("priv_mixer_hidden_dim", 128),
            priv_mixer_token_dim=self.policy_cfg.get("priv_mixer_token_dim", 128),
            priv_mixer_channel_dim=self.policy_cfg.get("priv_mixer_channel_dim", 256),
            priv_mixer_use_layer_norm=self.policy_cfg.get("priv_mixer_use_layer_norm", True),
            device=self.device,
            **alg_cfg,
        )
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        depth_h, depth_w = getattr(self.env, "depth_output_resolution", (48, 64))
        depth_sequence_length = self.policy_cfg.get("depth_sequence_length", 5)
        depth_latent_dim = self.policy_cfg["depth_image_latent_dim"]
        
        # Rollout-time visual memory: only the newest depth image is stored
        #     raw; previous frames are kept as compact depth-frame latents.
        self.depth_latent_history = torch.zeros(
            self.env.num_envs,
            depth_sequence_length - 1,
            depth_latent_dim,
            device=self.device,
        )

        # Latest modality-mixer torso linear velocity estimate. The runner
        # only injects this into depth_torso_state when the velocity-specific
        # boot gate trusts the estimate; otherwise env ground truth remains.
        self.depth_torso_lin_vel_est = torch.zeros(
            self.env.num_envs,
            3,
            device=self.device,
        )
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_obs_hist * self.env.num_obs],
                              [self.env.num_actions], [self.env.num_exp_labels], [1, depth_h, depth_w], [depth_sequence_length - 1, depth_latent_dim], [8], list(terrain_map_shape),
                                [self.env.num_crit_obs_stack * self.env.num_privileged_obs], [depth_latent_dim],)

        if "pretrained_path" in self.policy_cfg.keys():
            self._load_pretrained_model()

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # self.env.create_async_pino_workers()

        _, _ = self.env.reset()

    # Construct the torso-state (roll, pitch, v_x, v_y, v_z, w_roll, w_pitch, w_yaw) used by the 
    #     learned coordconv coordinate transform
    def _build_depth_torso_state(self, imu_depth_torso_state):
        """Gate predicted velocity versus simulator velocity for depth input."""
        if not self.alg.use_depth_vel_boot:
            return imu_depth_torso_state

        depth_torso_state = imu_depth_torso_state.clone()
        depth_torso_state[:, 2:5] = self.depth_torso_lin_vel_est.to(
            device=depth_torso_state.device,
            dtype=depth_torso_state.dtype,
        )
        return depth_torso_state

    # function to load a boot-strap initial model and reset the std
    def _load_pretrained_model(self):
        pretrained_path = self.policy_cfg["pretrained_path"]
        print("Loading boot-strapping model from - ", pretrained_path)
        loaded_dict = torch.load(pretrained_path)
        # Load the pretrained action-network and encoder
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
                    
        obs, obs_hist, privileged_obs, exp_labels, depth_obs, depth_torso_state, terrain_map = self.env.get_observations()

        if self.env.use_reward_curriculum:
            self.env.step_reward_curriculum(0)
        self.env.step_command_resampling_time_curriculum(0)
        
        obs = obs.to(self.device)
        obs_hist = obs_hist.to(self.device)
        privileged_obs = privileged_obs.to(self.device) if privileged_obs is not None else obs
        exp_labels = exp_labels.to(self.device)
        depth_image = depth_obs[:, 0:1].to(self.device)
        depth_torso_state = self._build_depth_torso_state(depth_torso_state.to(self.device))
        terrain_map = terrain_map.to(self.device)
        # Critic observations are rebuilt through the current privileged
        # encoders so value targets match the latest auxiliary representation.
        critic_obs = self.alg.build_critic_obs(privileged_obs, terrain_map)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # Call the algorithms act() method to store current transition data and predict actions
                    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        # alg.act stores the pre-step transition and returns
                        # the newest frame latent used to advance visual memory.
                        actions, latest_depth_latent, latest_torso_lin_vel_est = self.alg.act(obs, critic_obs, obs_hist, privileged_obs, depth_image, self.depth_latent_history, depth_torso_state, terrain_map) # obs_t, (obs_t-1)
                         
                    # Submit the predicted action and extract the resulting state... 
                    obs, privileged_obs, obs_hist, exp_labels, depth_obs, depth_torso_state, terrain_map, rewards, dones, infos = self.env.step(actions)  # obs_t+1  (obs_t)
                    
                    # move everything to the correct device
                    obs = obs.to(self.device)
                    obs_hist = obs_hist.to(self.device)
                    privileged_obs = privileged_obs.to(self.device) if privileged_obs is not None else obs
                    exp_labels = exp_labels.to(self.device)
                    depth_image = depth_obs[:, 0:1].to(self.device)
                    self.depth_torso_lin_vel_est = latest_torso_lin_vel_est     # copy this for use in the _build_depth_torso_state function
                    if dones.any():
                        self.depth_torso_lin_vel_est[dones.bool()] = 0.0
                    depth_torso_state = self._build_depth_torso_state(
                        depth_torso_state.to(self.device)
                    )
                    terrain_map = terrain_map.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    critic_obs = self.alg.build_critic_obs(privileged_obs, terrain_map)
                    
                    # Advance depth latent history after the action generated
                    #     from the previous history has been stored.
                    if self.depth_latent_history.shape[1] > 0:
                        if self.depth_latent_history.shape[1] > 1:
                            self.depth_latent_history[:, :-1].copy_(
                                self.depth_latent_history[:, 1:].clone()
                            )
                        self.depth_latent_history[:, -1].copy_(
                            latest_depth_latent.detach()
                        )
                    if dones.any():
                        self.depth_latent_history[dones.bool()] = 0.0

                    # Log the labels associated with the context decoder as well as the typical stuff
                    # Pass next privileged obs as the dynamics reconstruction
                    # target; critic_obs itself contains learned latents.
                    self.alg.process_env_step(rewards, dones, infos, privileged_obs, exp_labels)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, mean_aux_loss, mean_aux_decoder_loss, mean_explicit_loss, mean_reconstruction_loss, mean_kl_loss, mean_aux_loss_details = self.alg.update()
            
            # Step the reward curriculum if we are doing that
            if self.env.use_reward_curriculum:
                self.env.step_reward_curriculum(it)
            self.env.step_command_resampling_time_curriculum(it)
            
            # Step domain randomization when tracking performance is available.
            if self.env.simulator.use_domainrand_curriculum:
                tracking_values = []
                for ep_info in ep_infos:
                    if "rew_tracking_lin_vel" not in ep_info:
                        continue
                    value = ep_info["rew_tracking_lin_vel"]
                    if isinstance(value, torch.Tensor):
                        value = value.float().mean().item()
                    tracking_values.append(float(value))

                if tracking_values:
                    mean_tracking_lin_vel = statistics.mean(tracking_values)
                    self.env.simulator._step_domian_rand(
                        it, mean_tracking_lin_vel
                    )

                if self.writer is not None:
                    reward_ema = self.env.simulator.domain_rand_reward_ema
                    required_reward = self.env.simulator.required_reward
                    self.writer.add_scalar(
                        "Values/domain_rand_reward_ema",
                        reward_ema if reward_ema is not None else 0.0,
                        it,
                    )
                    self.writer.add_scalar(
                        "Values/required_reward",
                        required_reward if required_reward is not None else 0.0,
                        it,
                    )
                    self.writer.add_scalar(
                        "Values/domain_rand_joint_dynamics_progress",
                        self.env.simulator.domain_rand_joint_dynamics_progress,
                        it,
                    )
                    self.writer.add_scalar(
                        "Values/domain_rand_mass_com_progress",
                        self.env.simulator.domain_rand_mass_com_progress,
                        it,
                    )
                    self.writer.add_scalar(
                        "Values/domain_rand_disturbance_progress",
                        self.env.simulator.domain_rand_disturbance_progress,
                        it,
                    )

            entropy_coef = self.alg.current_entropy_coef
            if ep_infos and self.use_adaptive_entropy:
                performance_metrics = {}
                metric_names = (
                    ("command_lin_tracking_ema", "lin_vel_tracking_pct"),
                    ("command_ang_tracking_ema", "ang_vel_tracking_pct"),
                    ("terrain_level", "terrain_level"),
                )
                for episode_key, metric_key in metric_names:
                    values = []
                    for ep_info in ep_infos:
                        if episode_key not in ep_info:
                            continue
                        value = ep_info[episode_key]
                        if isinstance(value, torch.Tensor):
                            value = value.float().max().item()
                        values.append(float(value))
                    if values:
                        performance_metrics[metric_key] = max(values)

                entropy_coef = self.alg.update_adaptive_entropy_coef(
                    performance_metrics
                )
            
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
                self.writer.add_scalar("Policy/entropy_coef", entropy_coef, it)

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

        # Learning is done, shutdown the async. pinocchio workers
        # self.env.shutdown_asynic_pino_workers()

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
        mean_std = self.alg.actor_critic.std.mean()
        
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        
        # Reconstruction stuff
        self.writer.add_scalar('Loss/aux_total', locs['mean_aux_loss'], locs['it'])
        self.writer.add_scalar('Loss/aux_explicit_state', locs['mean_explicit_loss'], locs['it'])
        self.writer.add_scalar('Loss/aux_reconstruction', locs['mean_reconstruction_loss'], locs['it'])
        self.writer.add_scalar('Loss/aux_kl', locs['mean_kl_loss'], locs['it'])
        self.writer.add_scalar('Loss/aux_decoder', locs['mean_aux_decoder_loss'], locs['it'])
        for name, value in locs['mean_aux_loss_details'].items():
            self.writer.add_scalar(f'Loss/aux_detail/{name}', value, locs['it'])

        # RL stuff
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])        
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Auxiliary total loss:':>{pad}} {locs['mean_aux_loss']:.4f}\n"""
                          f"""{'Explicit-state loss:':>{pad}} {locs['mean_explicit_loss']:.4f}\n"""
                          f"""{'Reconstruction loss:':>{pad}} {locs['mean_reconstruction_loss']:.4f}\n"""
                          f"""{'Auxiliary KL loss:':>{pad}} {locs['mean_kl_loss']:.4f}\n"""
                          f"""{'Auxiliary decoder loss:':>{pad}} {locs['mean_aux_decoder_loss']:.4f}\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Auxiliary total loss:':>{pad}} {locs['mean_aux_loss']:.4f}\n"""
                          f"""{'Explicit-state loss:':>{pad}} {locs['mean_explicit_loss']:.4f}\n"""
                          f"""{'Reconstruction loss:':>{pad}} {locs['mean_reconstruction_loss']:.4f}\n"""
                          f"""{'Auxiliary KL loss:':>{pad}} {locs['mean_kl_loss']:.4f}\n"""
                          f"""{'Auxiliary decoder loss:':>{pad}} {locs['mean_aux_decoder_loss']:.4f}\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean pos action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def _enc_optimizer_uses_bundle(self):
        return hasattr(self.alg.enc_optimizer, "optimizers")

    def _load_enc_optimizer_state(self, enc_optimizer_state):
        if enc_optimizer_state is None:
            print("No encoder optimizer state found in checkpoint.")
            return

        if not self._enc_optimizer_uses_bundle():
            self.alg.enc_optimizer.load_state_dict(enc_optimizer_state)
            return

        expected_names = set(self.alg.enc_optimizer.optimizers.keys())
        if isinstance(enc_optimizer_state, dict) and expected_names.issubset(enc_optimizer_state.keys()):
            self.alg.enc_optimizer.load_state_dict(enc_optimizer_state)
            return

        is_legacy_single_optimizer = (
            isinstance(enc_optimizer_state, dict)
            and "state" in enc_optimizer_state
            and "param_groups" in enc_optimizer_state
        )
        if is_legacy_single_optimizer:
            print(
                "Skipping legacy single encoder optimizer state because the "
                "current KITE policy uses an OptimizerBundle with separate "
                "sub-optimizers. Model weights were loaded; encoder optimizer "
                "states will be reinitialized."
            )
            return

        raise KeyError(
            "Encoder optimizer state does not match the current OptimizerBundle. "
            f"Expected keys: {sorted(expected_names)}; got keys: "
            f"{sorted(enc_optimizer_state.keys()) if isinstance(enc_optimizer_state, dict) else type(enc_optimizer_state)}"
        )

    def save(self, path, infos=None):
        checkpoint = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'act_optimizer_state_dict': self.alg.act_optimizer.state_dict(),
            'enc_optimizer_state_dict': self.alg.enc_optimizer.state_dict(),
            'enc_optimizer_format': 'bundle' if self._enc_optimizer_uses_bundle() else 'single',
            'depth_decoder_state_dict': self.alg.depth_decoder.state_dict(),
            'depth_decoder_opt_state_dict': self.alg.depth_decoder_optimizer.state_dict(),
            'priv_terrain_encoder_state_dict': self.alg.priv_terrain_encoder.state_dict(),
            'priv_terrain_decoder_state_dict': self.alg.priv_terrain_decoder.state_dict(),
            'priv_dynamics_encoder_state_dict': self.alg.priv_dynamics_encoder.state_dict(),
            'priv_dynamics_decoder_state_dict': self.alg.priv_dynamics_decoder.state_dict(),
            'privileged_optimizer_state_dict': self.alg.privileged_optimizer.state_dict(),
            'depth_to_terrain_latent_state_dict': self.alg.depth_to_terrain_latent.state_dict(),
            'proprio_to_dynamics_latent_state_dict': self.alg.proprio_to_dynamics_latent.state_dict(),
            'mixer_to_terrain_latent_state_dict': self.alg.mixer_to_terrain_latent.state_dict(),
            'mixer_to_dynamics_latent_state_dict': self.alg.mixer_to_dynamics_latent.state_dict(),
            'terrain_contrastive_head_depth_state_dict': self.alg.terrain_contrastive_head_depth.state_dict(),
            'dynamics_contrastive_head_proprio_state_dict': self.alg.dynamics_contrastive_head_proprio.state_dict(),
            'terrain_contrastive_head_mixer_state_dict': self.alg.terrain_contrastive_head_mixer.state_dict(),
            'dynamics_contrastive_head_mixer_state_dict': self.alg.dynamics_contrastive_head_mixer.state_dict(),
            'aux_projection_optimizer_state_dict': self.alg.aux_projection_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        torch.save(checkpoint, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        # Load actor/critic model(s)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        # Load optimizer(s)
        if load_optimizer:
            self.alg.act_optimizer.load_state_dict(loaded_dict['act_optimizer_state_dict'])
            self._load_enc_optimizer_state(loaded_dict.get('enc_optimizer_state_dict'))
            if 'depth_decoder_opt_state_dict' in loaded_dict:
                self.alg.depth_decoder_optimizer.load_state_dict(loaded_dict['depth_decoder_opt_state_dict'])
            if 'privileged_optimizer_state_dict' in loaded_dict:
                self.alg.privileged_optimizer.load_state_dict(loaded_dict['privileged_optimizer_state_dict'])
            if 'aux_projection_optimizer_state_dict' in loaded_dict:
                try:
                    self.alg.aux_projection_optimizer.load_state_dict(loaded_dict['aux_projection_optimizer_state_dict'])
                except ValueError:
                    print(
                        "Skipping auxiliary projection optimizer state because "
                        "its parameter groups do not match the current KITE "
                        "auxiliary projection modules. Model weights were "
                        "loaded; this optimizer will be reinitialized."
                    )
        if 'depth_decoder_state_dict' in loaded_dict:
            self.alg.depth_decoder.load_state_dict(loaded_dict['depth_decoder_state_dict'])
        if 'priv_terrain_encoder_state_dict' in loaded_dict:
            self.alg.priv_terrain_encoder.load_state_dict(loaded_dict['priv_terrain_encoder_state_dict'])
        if 'priv_terrain_decoder_state_dict' in loaded_dict:
            self.alg.priv_terrain_decoder.load_state_dict(loaded_dict['priv_terrain_decoder_state_dict'])
        if 'priv_dynamics_encoder_state_dict' in loaded_dict:
            self.alg.priv_dynamics_encoder.load_state_dict(loaded_dict['priv_dynamics_encoder_state_dict'])
        if 'priv_dynamics_decoder_state_dict' in loaded_dict:
            self.alg.priv_dynamics_decoder.load_state_dict(loaded_dict['priv_dynamics_decoder_state_dict'])
        if 'depth_to_terrain_latent_state_dict' in loaded_dict:
            self.alg.depth_to_terrain_latent.load_state_dict(loaded_dict['depth_to_terrain_latent_state_dict'])
        if 'proprio_to_dynamics_latent_state_dict' in loaded_dict:
            self.alg.proprio_to_dynamics_latent.load_state_dict(loaded_dict['proprio_to_dynamics_latent_state_dict'])
        if 'mixer_to_terrain_latent_state_dict' in loaded_dict:
            self.alg.mixer_to_terrain_latent.load_state_dict(loaded_dict['mixer_to_terrain_latent_state_dict'])
        if 'mixer_to_dynamics_latent_state_dict' in loaded_dict:
            self.alg.mixer_to_dynamics_latent.load_state_dict(loaded_dict['mixer_to_dynamics_latent_state_dict'])
        if 'terrain_contrastive_head_depth_state_dict' in loaded_dict:
            self.alg.terrain_contrastive_head_depth.load_state_dict(loaded_dict['terrain_contrastive_head_depth_state_dict'])
        if 'dynamics_contrastive_head_proprio_state_dict' in loaded_dict:
            self.alg.dynamics_contrastive_head_proprio.load_state_dict(loaded_dict['dynamics_contrastive_head_proprio_state_dict'])
        if 'terrain_contrastive_head_mixer_state_dict' in loaded_dict:
            self.alg.terrain_contrastive_head_mixer.load_state_dict(loaded_dict['terrain_contrastive_head_mixer_state_dict'])
        if 'dynamics_contrastive_head_mixer_state_dict' in loaded_dict:
            self.alg.dynamics_contrastive_head_mixer.load_state_dict(loaded_dict['dynamics_contrastive_head_mixer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_async_inference_pipelines(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        depth_pipeline = self.alg.actor_critic.build_depth_async_pipeline().eval()
        actor_pipeline = self.alg.actor_critic.build_actor_async_pipeline().eval()
        if device is not None:
            depth_pipeline.to(device)
            actor_pipeline.to(device)
        return depth_pipeline, actor_pipeline

    def export_async_inference_pipelines(self, path, device="cpu"):
        self.alg.actor_critic.eval()
        return export_kite_async_deployment_pipelines(
            self.alg.actor_critic,
            path,
            device=device,
        )
