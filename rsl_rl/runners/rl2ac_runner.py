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

import math
import time
import os
from collections import deque
import statistics
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO_RL2AC
from rsl_rl.modules import ActorCritic_RL2AC, ContextDecoder
from rsl_rl.env import VecEnv



# ---------------- 4090 / Ada Lovelace performance knobs ----------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


class OnPolicyRunnerRL2AC:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        torch.autograd.set_detect_anomaly(True)
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        
        self.device = device
        self.env = env
        
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        
        # check if we are using a history of critic observations
        if self.env.num_crit_obs_stack is not None:
            num_critic_obs *= self.env.num_crit_obs_stack

        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        
        cenet_input_dim = self.env.num_obs * self.env.num_obs_hist

        actor_critic: ActorCritic_PosTau = actor_critic_class(self.env.num_obs,
                                                                num_critic_obs,
                                                                self.env.num_actions,
                                                                self.policy_cfg["actor_layers"],
                                                                self.policy_cfg["critic_layers"],
                                                                cenet_input_dim,
                                                                self.policy_cfg["cenet_enc_latent_dim"],
                                                                self.policy_cfg["cenet_velo_dim"],
                                                                self.policy_cfg["cenet_enc_layers"],
                                                                self.policy_cfg["activation"],
                                                                self.policy_cfg["init_noise_std"]).to(self.device)
        
        
        # actor_critic = torch.compile(actor_critic)
        
        print(actor_critic)
        
        decoder = ContextDecoder(self.policy_cfg["cenet_dec_input_dim"],
                                 self.policy_cfg["cenet_dec_layers"],
                                 self.policy_cfg["cenet_dec_out_dim"]
                                 ).to(self.device)
        
        # decoder = torch.compile(decoder)

        print("Created Parallel Actor-Critic Model. Parameter Count: ", np.sum(p.numel() for p in actor_critic.parameters() if p.requires_grad))

        print("\t Actor Trunk Parameter Count: ", np.sum(p.numel() for p in actor_critic.act_trunk.parameters() if p.requires_grad))

        print("\t Encoder Parameter Count: ", np.sum(p.numel() for p in actor_critic.context_encoder.parameters() if p.requires_grad))

        print("\t Critic Parameter Count: ", np.sum(p.numel() for p in actor_critic.critic.parameters() if p.requires_grad))


        self._init_entropy_coef = self.alg_cfg["entropy_coef"]

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO


        
        self.alg: PPO_RL2AC = alg_class(actor_critic, decoder, self.env.num_privileged_obs,
                                           device=self.device, **self.alg_cfg)
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_crit_obs_stack*self.env.num_privileged_obs], \
                              [self.env.num_privileged_obs], [self.env.num_obs_hist*self.env.num_obs], \
                              [self.env.num_actions], [self.env.num_exp_labels])

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


    # function to load a boot-strap initial model and reset the std
    def _load_pretrained_model(self):
        pretrained_path = self.policy_cfg["pretrained_path"]
        pretrained_std = self.policy_cfg["pretrained_std"]
        print(pretrained_path)
        loaded_dict = torch.load(pretrained_path)
        # Load the pretrained action-network and encoder
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        # Reset the 
        self.alg.actor_critic._init_std(pretrained_std)
        # Load the pretrained decoder network
        self.alg.decoder.load_state_dict(loaded_dict['decoder_state_dict'])


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
            
        # self.alg.actor_critic.std.data.fill_(self.policy_cfg["init_noise_std"]*0.45)
        
        obs,obs_hist,privileged_obs,exp_labels = self.env.get_observations()

        if self.env.use_reward_curriculum:
            self.env.step_reward_curriculum(0)
        
        critic_obs = privileged_obs if privileged_obs is not None else obs
        
        obs, critic_obs, obs_hist, exp_labels = obs.to(self.device), critic_obs.to(self.device),obs_hist.to(self.device), exp_labels.to(self.device)
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
                        actions = self.alg.act(obs, critic_obs, obs_hist) # obs_t, (obs_t-1)
                         
                    # Submit the predicted action and extract the resulting state... 
                    #     passing through actions twice, second one is unused. Just saving a place for pasing through actual RL2AC q-ref targets during eval
                    obs, privileged_obs, obs_hist, exp_labels, rewards, dones, infos, dof_labels = self.env.step(actions, actions)  # obs_t+1  (obs_t)
                    
                    # Create privileged obs
                    critic_obs = privileged_obs if privileged_obs is not None else obs

                    # move everything to the correct device
                    obs, critic_obs, obs_hist, exp_labels, rewards, dones, dof_labels = obs.to(self.device), critic_obs.to(self.device), \
                        obs_hist.to(self.device), exp_labels.to(self.device), rewards.to(self.device), dones.to(self.device), dof_labels.to(self.device)

                    # Log the labels associated with the context decoder as well as the typical stuff
                    # self.alg.process_env_step(rewards, dones, infos, grfs, obs, exp_labels, gt_forces, mass_mats, bias_vecs, torso_acc)
                    self.alg.process_env_step(rewards, dones, infos, critic_obs, exp_labels, dof_labels)

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
            
            mean_value_loss, mean_surrogate_loss, mean_autoenc_loss, mean_decoder_loss, mean_vel_loss, \
                    mean_recon_loss, mean_kld_loss, mean_qref_loss \
                    = self.alg.update()

            # self.env.step_tradeoff_curriculum()
            # print("Avg - Curriculum Step: ", torch.mean(self.env.tradeoff_step_ctr).item())
            # print("Max - self.feedforward_tau_weight: ", torch.max(self.env.simulator.feedforward_tau_weight).item())
            # print("Min - self.feedforward_tau_weight: ", torch.min(self.env.simulator.feedforward_tau_weight).item())
            # print("Avg - self.feedforward_tau_weight: ", torch.mean(self.env.simulator.feedforward_tau_weight).item())
            # print("Max - self.feedback_tau_weight: ", torch.max(self.env.simulator.feedback_tau_weight).item())
            # print("Min - self.feedback_tau_weight: ", torch.min(self.env.simulator.feedback_tau_weight).item())
            # print("Avg - self.feedback_tau_weight: ", torch.mean(self.env.simulator.feedback_tau_weight).item())
            
            # Step the reward curriculum if we are doing that
            if self.env.use_reward_curriculum:
                self.env.step_reward_curriculum(it)
            
            # Step the domain randomization if approperiate
            if self.env.simulator.use_domainrand_curriculum:
                self.env.simulator._step_domian_rand(it)

            # if it > 1000:
            #     self.alg.set_entropy_coef(1.0e-3)
            
            entropy_coef = self._init_entropy_coef
            std_lwr = 0.40

            half_coef = self._init_entropy_coef * 0.5
            tenth_coef = self._init_entropy_coef * 0.1
            
            # if it < 4000:
            #     entropy_coef = self._init_entropy_coef
            # elif it < 4500:
            #     alpha = (it - 4000) / 500.0
            #     entropy_coef = half_coef + 0.5 * (self._init_entropy_coef - half_coef) * (1 + math.cos(math.pi * alpha))
            # elif it < 5000:
            #     entropy_coef = half_coef
            # elif it < 5500:
            #     alpha = (it - 5000) / 500.0
            #     entropy_coef = tenth_coef + 0.5 * (half_coef - tenth_coef) * (1 + math.cos(math.pi * alpha))
            # else:
            #     entropy_coef = tenth_coef

            if it < 6000 and it >= 5000:
                alpha = (it - 5000) / 1000.0
                entropy_coef = tenth_coef + 0.5 * (self._init_entropy_coef - tenth_coef) * (1 + math.cos(math.pi * alpha))
            
            entropy_coef = max(entropy_coef, 0.0001)

            print("entropy_coef - ", entropy_coef)
            # print("std_lwr - ", std_lwr)

            self.alg.set_entropy_coef(entropy_coef)
            # self.alg._set_std_clip_lwr(std_lwr)


            # if self.env.cfg.rewards.only_positive_rewards and it > 1000:
            #     self.env.cfg.rewards.only_positive_rewards = False
            
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
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
        self.writer.add_scalar('Loss/autoenc_function', locs['mean_autoenc_loss'], locs['it'])
        self.writer.add_scalar('Loss/velo_pred', locs['mean_vel_loss'], locs['it'])
        self.writer.add_scalar('Loss/recon', locs['mean_recon_loss'], locs['it'])
        self.writer.add_scalar('Loss/kl_div', locs['mean_kld_loss'], locs['it'])
        self.writer.add_scalar('Loss/qref', locs['mean_qref_loss'], locs['it'])
        self.writer.add_scalar('Loss/decoder_function', locs['mean_decoder_loss'], locs['it'])
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
                          f"""{'Autoenc function loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
                          f"""{'Torso Velo. Pred loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                          f"""{'Reconstruction   loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"""
                          f"""{'KL Divergence    loss:':>{pad}} {locs['mean_kld_loss']:.4f}\n"""
                          f"""{'Q-Reference      loss:':>{pad}} {locs['mean_qref_loss']:.4f}\n"""
                          f"""{'Decoder function loss:':>{pad}} {locs['mean_decoder_loss']:.4f}\n"""
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
                          f"""{'Autoenc function loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
                          f"""{'Torso Velo. Pred loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                          f"""{'Reconstruction   loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"""
                          f"""{'KL Divergence    loss:':>{pad}} {locs['mean_kld_loss']:.4f}\n"""
                          f"""{'Q-Reference      loss:':>{pad}} {locs['mean_qref_loss']:.4f}\n"""
                          f"""{'Decoder function loss:':>{pad}} {locs['mean_decoder_loss']:.4f}\n"""
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

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'act_optimizer_state_dict': self.alg.act_optimizer.state_dict(),
            'enc_optimizer_state_dict': self.alg.enc_optimizer.state_dict(),
            'decoder_state_dict': self.alg.decoder.state_dict(),
            'decoder_opt_state_dict': self.alg.decoder_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        # Load actor/critic model(s)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        # Load optimizer(s)
        if load_optimizer:
            self.alg.act_optimizer.load_state_dict(loaded_dict['act_optimizer_state_dict'])
            self.alg.enc_optimizer.load_state_dict(loaded_dict['enc_optimizer_state_dict'])
            self.alg.decoder_optimizer.load_state_dict(loaded_dict['decoder_opt_state_dict'])
        # Load the VAE decoder model...
        self.alg.decoder.load_state_dict(loaded_dict['decoder_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        self.current_learning_iteration = 0
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
