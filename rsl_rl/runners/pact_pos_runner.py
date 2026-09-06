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

from rsl_rl.algorithms import PPO_PACT_Pos
from rsl_rl.modules import ActorCritic_PACT_Pos, ActorCritic_HardPACT_Pos, ContextDecoder
from rsl_rl.env import VecEnv
from rsl_rl.utils import pretty_print_module
from rsl_rl.hard_pact_logging import (
    collect_contact_estimator_scalars,
    collect_force_decoder_scalars,
    collect_latent_diagnostics_scalars,
)
from legged_gym.envs.go2.go2_hard_pact.deployment import (
    RECONSTRUCTION_DIM,
    RECONSTRUCTION_INDICES,
    build_deployment_contract,
    calculate_physics_head_gains,
    write_deployment_contract_once,
)


# ---------------- 4090 / Ada Lovelace performance knobs ----------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def build_hard_pact_start_checkpoint(
    pos_actor_state_dict,
    decoder_state_dict,
    iteration,
    infos=None,
):
    """Convert HardPACTPos weights into a strict HardPACT bootstrap.

    The actor, critic, encoder, estimators, and physics heads share identical
    keys and shapes. HardPACT's only state-dict shape difference is ``std``:
    position pretraining samples 12 actions, while HardPACT samples 12
    position plus 12 feed-forward-torque actions. The HardPACT exploration
    standard deviation is deliberately reset to a fresh 24-D vector of ones.

    Optimizer states are intentionally omitted because this artifact starts a
    new HardPACT run rather than resuming the HardPACTPos optimizer.
    """
    if "std" not in pos_actor_state_dict:
        raise KeyError("HardPACTPos actor state dict is missing 'std'")
    pos_std = pos_actor_state_dict["std"]
    if pos_std.ndim != 1 or pos_std.numel() != 12:
        raise ValueError(
            "HardPACTPos exploration std must have shape (12,), got "
            f"{tuple(pos_std.shape)}"
        )

    model_state = type(pos_actor_state_dict)(
        (key, value.detach().clone())
        for key, value in pos_actor_state_dict.items()
    )
    model_state["std"] = pos_std.new_ones(24)
    decoder_state = type(decoder_state_dict)(
        (key, value.detach().clone())
        for key, value in decoder_state_dict.items()
    )
    return {
        "model_state_dict": model_state,
        "decoder_state_dict": decoder_state,
        "iter": int(iteration),
        "infos": infos,
        "hard_pact_start": True,
        "source_task": "go2_hard_pact_pos",
    }

class OnPolicyRunnerPACTPos:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        # Keep legacy PACTPos diagnostics intact, but give HardPACTPos the
        # same throughput-safe anomaly/debug defaults as HardPACT.
        torch.autograd.set_detect_anomaly(bool(
            self.alg_cfg.get(
                "detect_anomaly",
                self.cfg.get("policy_class_name") != "ActorCritic_HardPACT_Pos",
            )
        ))
        
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
        self.is_hard_pact_pos = actor_critic_class is ActorCritic_HardPACT_Pos
        self.console_debug = bool(self.cfg.get("console_debug", False))
        self.console_iteration = bool(self.cfg.get("console_iteration", True))
        self.console_model_summary = bool(
            self.cfg.get("console_model_summary", not self.is_hard_pact_pos)
        )
        self.console_reward_terms = bool(
            self.cfg.get("console_reward_terms", not self.is_hard_pact_pos)
        )
        self.console_detailed_losses = bool(
            self.cfg.get("console_detailed_losses", not self.is_hard_pact_pos)
        )
        gain_spec = None
        actor_extra_kwargs = {}
        reconstruction_indices = None
        reconstruction_dim = self.env.num_privileged_obs
        if actor_critic_class is ActorCritic_HardPACT_Pos:
            gain_spec = calculate_physics_head_gains(self.env.cfg)
            actor_extra_kwargs = {
                "cenet_explicit_layers": self.policy_cfg["cenet_explicit_layers"],
                "grf_decoder_layers": self.policy_cfg["grf_decoder_layers"],
                "wrench_decoder_layers": self.policy_cfg["wrench_decoder_layers"],
                "grf_scale_n": gain_spec.grf_scale_n,
                "wrench_scale": gain_spec.wrench_scale_n_nm,
                "wrench_qp_clip": gain_spec.wrench_qp_clip_n_nm,
                "contact_epsilon": self.policy_cfg["contact_epsilon"],
            }
            reconstruction_indices = RECONSTRUCTION_INDICES
            reconstruction_dim = RECONSTRUCTION_DIM
        
        cenet_input_dim = self.env.num_obs * self.env.num_obs_hist

        actor_critic: ActorCritic_PACT_Pos = actor_critic_class(self.env.num_obs,
                                                                num_critic_obs,
                                                                self.env.num_actions,
                                                                self.policy_cfg["actor_layers"],
                                                                self.policy_cfg["critic_layers"],
                                                                cenet_input_dim,
                                                                self.policy_cfg["cenet_enc_latent_dim"],
                                                                self.policy_cfg["cenet_velo_dim"],
                                                                self.policy_cfg["cenet_enc_layers"],
                                                                self.policy_cfg["activation"],
                                                                self.policy_cfg["init_noise_std"],
                                                                **actor_extra_kwargs).to(self.device)
                        
        decoder = ContextDecoder(self.policy_cfg["cenet_dec_input_dim"],
                                 self.policy_cfg["cenet_dec_layers"],
                                 self.policy_cfg["cenet_dec_out_dim"]
                                 ).to(self.device)
        

        if self.console_model_summary:
            print("Created Parallel Actor-Critic Model")
            pretty_print_module(actor_critic)
            pretty_print_module(decoder)

        self._init_entropy_coef = self.alg_cfg["entropy_coef"]
        self.use_adaptive_entropy = self.alg_cfg["use_adaptive_entropy"]



        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        
        self.alg: PPO_PACT_Pos = alg_class(actor_critic, 
                                           decoder, 
                                           self.env.num_privileged_obs,
                                           reconstruction_indices=reconstruction_indices,
                                           device=self.device, 
                                           **self.alg_cfg)
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_crit_obs_stack*self.env.num_privileged_obs], \
                              [reconstruction_dim], [self.env.num_obs_hist*self.env.num_obs], \
                              [self.env.num_actions], [self.env.num_exp_labels], [self.cfg["grf_dim"]])

        if self.policy_cfg.get("pretrained_path"):
            self._load_pretrained_model()

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        if gain_spec is not None:
            self.deployment_contract = build_deployment_contract(
                self.env.cfg, self.alg.actor_critic, gain_spec
            )
            write_deployment_contract_once(self.log_dir, self.deployment_contract)

        # self.env.create_async_pino_workers()

        _, _ = self.env.reset()


    # function to load a boot-strap initial model and reset the std
    def _load_pretrained_model(self):
        pretrained_path = self.policy_cfg["pretrained_path"]
        pretrained_std = self.policy_cfg["pretrained_std"]
        if self.console_model_summary:
            print(pretrained_path)
        loaded_dict = torch.load(pretrained_path)
        # Load the pretrained action-network and encoder
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        # Reset the 
        # self.alg.actor_critic._init_std(pretrained_std)
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
                    obs, privileged_obs, obs_hist, exp_labels, rewards, dones, infos, grfs = self.env.step(actions)  # obs_t+1  (obs_t)
                    
                    # Create privileged obs
                    critic_obs = privileged_obs if privileged_obs is not None else obs

                    # move everything to the correct device
                    obs, critic_obs, obs_hist, exp_labels, rewards, dones, grfs = obs.to(self.device), critic_obs.to(self.device), \
                        obs_hist.to(self.device), exp_labels.to(self.device), rewards.to(self.device), dones.to(self.device), grfs.to(self.device)

                    # Log the labels associated with the context decoder as well as the typical stuff
                    self.alg.process_env_step(rewards, dones, infos, grfs, critic_obs, exp_labels)

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
                    mean_recon_loss, mean_kld_loss, mean_tau_loss \
                    = self.alg.update(self.env._get_pinn_actions, self.env._get_pinn_feedback, self.env.dt, it, self.env.simulator.default_dof_pos, self.env.obs_scales.dof_vel)
            
            # Step the reward curriculum if we are doing that
            if self.env.use_reward_curriculum:
                self.env.step_reward_curriculum(it)
            
            # Step the domain randomization if approperiate
            if self.env.simulator.use_domainrand_curriculum:
                # self.env.simulator._step_domian_rand(it)
                mean_tracking_lin_vel = None

                if len(ep_infos) > 0 and "rew_tracking_lin_vel" in ep_infos[0]:
                    vals = []
                    for ep_info in ep_infos:
                        v = ep_info["rew_tracking_lin_vel"]
                        if not isinstance(v, torch.Tensor):
                            v = torch.tensor([v], device=self.device)
                        vals.append(v.float().mean().to(self.device))

                # mean_reward = statistics.mean(rewbuffer) if len(rewbuffer) > 0 else None
                if len(ep_infos) > 0 and "rew_tracking_lin_vel" in ep_infos[0] and vals:
                    mean_tracking_lin_vel = torch.stack(vals).mean().item()
                if self.is_hard_pact_pos and hasattr(
                    self.env, "step_domain_rand_curriculum"
                ):
                    self.env.step_domain_rand_curriculum(
                        it, mean_tracking_lin_vel
                    )
                else:
                    self.env.simulator._step_domian_rand(
                        it, mean_tracking_lin_vel
                    )

                if self.env.simulator.domain_rand_reward_ema is not None:
                    self.writer.add_scalar('Values/domain_rand_reward_ema',self.env.simulator.domain_rand_reward_ema,it) 
                else:
                    self.writer.add_scalar('Values/domain_rand_reward_ema',0.0,it) 
                self.writer.add_scalar('Values/required_reward',self.env.simulator.required_reward,it) 
                self.writer.add_scalar('Values/domain_rand_joint_dynamics_progress',self.env.simulator.domain_rand_joint_dynamics_progress,it)
                self.writer.add_scalar('Values/domain_rand_mass_com_progress',self.env.simulator.domain_rand_mass_com_progress,it) 
                self.writer.add_scalar('Values/domain_rand_disturbance_progress',self.env.simulator.domain_rand_disturbance_progress,it) 

            performance_metrics = {}
            if ep_infos and self.use_adaptive_entropy:
                lin_vel_tracking = 0.0
                ang_vel_tracking = 0.0
                terrain_level = 0
                
                for ep_info in ep_infos:
                    if 'rew_tracking_lin_vel' in ep_info:
                        lin_vel_tracking = max(lin_vel_tracking, ep_info['rew_tracking_lin_vel'])
                    if 'rew_tracking_ang_vel' in ep_info:
                        ang_vel_tracking = max(ang_vel_tracking, ep_info['rew_tracking_ang_vel'])
                    if 'terrain_level' in ep_info:
                        terrain_level = max(terrain_level, ep_info['terrain_level'])
                
                performance_metrics = {
                    'lin_vel_tracking': lin_vel_tracking,
                    'ang_vel_tracking': ang_vel_tracking,
                    'terrain_level': terrain_level
                }
            
                entropy = self.alg.update_adaptive_entropy_coef(performance_metrics)
                if self.console_debug:
                    print(entropy)
                self.writer.add_scalar('Values/entropy',entropy,it)
            
            # entropy_coef = 0.01
            # half_coef = self._init_entropy_coef * 0.5
            # tenth_coef = self._init_entropy_coef * 0.1
            # if it < 2500:
            #     entropy_coef = self._init_entropy_coef
            # elif it < 3000:
            #     alpha = (it - 2500) / 500.0
            #     entropy_coef = half_coef + 0.5 * (self._init_entropy_coef - half_coef) * (1 + math.cos(math.pi * alpha))
            # elif it < 3500:
            #     entropy_coef = half_coef
            # elif it < 4000:
            #     alpha = (it - 4000) / 500.0
            #     entropy_coef = tenth_coef + 0.5 * (half_coef - tenth_coef) * (1 + math.cos(math.pi * alpha))
            # else:
            #     entropy_coef = tenth_coef
            # entropy_coef = max(entropy_coef, 0.001)
            # print("entropy_coef - ", entropy_coef)
            # self.alg.set_entropy_coef(entropy_coef)

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
        final_checkpoint = os.path.join(
            self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)
        )
        self.save(final_checkpoint)
        if self.is_hard_pact_pos and self.cfg.get(
            "export_hard_pact_start", True
        ):
            filename = self.cfg.get(
                "hard_pact_start_filename",
                "hard_pact_start_model_{iteration}.pt",
            ).format(iteration=self.current_learning_iteration)
            self.save_hard_pact_start(os.path.join(self.log_dir, filename))

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
                if self.console_reward_terms:
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
        mean_std = self.alg.actor_critic.std.mean()
        
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        self.writer.add_scalar('Loss/autoenc_function', locs['mean_autoenc_loss'], locs['it'])
        self.writer.add_scalar('Loss/velo_pred', locs['mean_vel_loss'], locs['it'])
        self.writer.add_scalar('Loss/recon', locs['mean_recon_loss'], locs['it'])
        self.writer.add_scalar('Loss/kl_div', locs['mean_kld_loss'], locs['it'])
        self.writer.add_scalar(
            'Loss/vae_kl_effective_weight',
            self.alg.current_vae_beta,
            locs['it'],
        )
        self.writer.add_scalar('Loss/decoder_function', locs['mean_decoder_loss'], locs['it'])
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/tau_loss', locs['mean_tau_loss'], locs['it'])
        if self.is_hard_pact_pos:
            # Keep the legacy aggregate key above and add stable, descriptive
            # component keys for the combined HardPACTPos auxiliary update.
            self.writer.add_scalar(
                'Loss/autoencoder_total', locs['mean_autoenc_loss'], locs['it']
            )
            for name, value in self.alg.last_auxiliary_metrics.items():
                if name == "total":
                    continue
                self.writer.add_scalar(
                    f'Loss/autoencoder_{name}', value, locs['it']
                )
            self.writer.add_scalar(
                "physics/force_decoder_diagnostics_enabled",
                float(self.alg.force_decoder_diagnostics_enabled), locs['it'],
            )
            for name, value in collect_force_decoder_scalars(
                self.alg.last_auxiliary_metrics
            ).items():
                self.writer.add_scalar(name, value, locs['it'])
            for name, value in collect_contact_estimator_scalars(
                self.alg.last_auxiliary_metrics
            ).items():
                self.writer.add_scalar(name, value, locs['it'])
            if self.alg.ppo_latent_diagnostics_enabled:
                for name, value in collect_latent_diagnostics_scalars(
                    self.alg.last_latent_diagnostics
                ).items():
                    self.writer.add_scalar(name, value, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])        
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        
        detailed = ""
        if self.console_detailed_losses:
            detailed = (
                f"{'Autoenc function loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"
                f"{'Torso Velo. Pred loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"
                f"{'Reconstruction loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"
                f"{'KL Divergence loss:':>{pad}} {locs['mean_kld_loss']:.4f}\n"
                f"{'Effective VAE KL weight:':>{pad}} {self.alg.current_vae_beta:.6f}\n"
                f"{'Decoder function loss:':>{pad}} {locs['mean_decoder_loss']:.4f}\n"
            )
            if self.is_hard_pact_pos:
                for name, value in self.alg.last_auxiliary_metrics.items():
                    detailed += (
                        f"{('Aux ' + name + ':'):>{pad}} {float(value):.4f}\n"
                    )

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
                          f"""{'Tau loss:':>{pad}} {locs['mean_tau_loss']:.4f}\n"""
                          f"""{detailed}"""
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
                          f"""{'Tau loss:':>{pad}} {locs['mean_tau_loss']:.4f}\n"""
                          f"""{detailed}"""
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
        if self.console_iteration:
            print(log_string)

    def save(self, path, infos=None):
        checkpoint = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'act_optimizer_state_dict': self.alg.act_optimizer.optimizer.state_dict(),
            'enc_optimizer_state_dict': self.alg.enc_optimizer.state_dict(),
            'decoder_state_dict': self.alg.decoder.state_dict(),
            'decoder_opt_state_dict': self.alg.decoder_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if self.is_hard_pact_pos and hasattr(
            self.env, "domain_rand_curriculum_state_dict"
        ):
            checkpoint["hard_pact_domain_rand_curriculum"] = (
                self.env.domain_rand_curriculum_state_dict()
            )
        torch.save(checkpoint, path)

    def save_hard_pact_start(self, path, infos=None):
        """Write a strict-loadable HardPACT weight initialization checkpoint."""
        checkpoint = build_hard_pact_start_checkpoint(
            self.alg.actor_critic.state_dict(),
            self.alg.decoder.state_dict(),
            self.current_learning_iteration,
            infos=infos,
        )
        torch.save(checkpoint, path)
        return path

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        # Load actor/critic model(s)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        # Load optimizer(s)
        if load_optimizer:
            self.alg.act_optimizer.optimizer.load_state_dict(loaded_dict['act_optimizer_state_dict'])
            self.alg.enc_optimizer.load_state_dict(loaded_dict['enc_optimizer_state_dict'])
            self.alg.decoder_optimizer.load_state_dict(loaded_dict['decoder_opt_state_dict'])
        # Load the VAE decoder model...
        self.alg.decoder.load_state_dict(loaded_dict['decoder_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        curriculum = loaded_dict.get("hard_pact_domain_rand_curriculum")
        if curriculum is not None and hasattr(
            self.env, "load_domain_rand_curriculum_state_dict"
        ):
            self.env.load_domain_rand_curriculum_state_dict(curriculum)
        else:
            self.current_learning_iteration = 0
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
