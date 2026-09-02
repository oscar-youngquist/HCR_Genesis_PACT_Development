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
import platform
import subprocess
import json
from importlib import metadata as importlib_metadata
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO_PACT, PPO_HardPACT
from rsl_rl.modules import ActorCritic_PACT, ActorCritic_HardPACT, ContextDecoder
from rsl_rl.env import VecEnv
from rsl_rl.utils import pretty_print_module
from legged_gym.envs.go2.go2_hard_pact.deployment import (
    RECONSTRUCTION_DIM,
    RECONSTRUCTION_INDICES,
    build_deployment_contract,
    calculate_physics_head_gains,
    write_deployment_contract_once,
)
from legged_gym.envs.go2.go2_hard_pact.ablations import resolve_hard_pact_features
from rsl_rl.hard_pact_logging import collect_hard_pact_scalars



# ---------------- 4090 / Ada Lovelace performance knobs ----------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

class OnPolicyRunnerPACT:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        torch.autograd.set_detect_anomaly(True)
        self.cfg=train_cfg["runner"]
        self.train_cfg = train_cfg
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
        gain_spec = None
        actor_extra_kwargs = {}
        reconstruction_indices = None
        reconstruction_dim = self.env.num_privileged_obs
        if actor_critic_class is ActorCritic_HardPACT:
            self.hard_pact_features = resolve_hard_pact_features(
                self.alg_cfg.get("ablation_variant", "full")
            )
            gain_spec = calculate_physics_head_gains(self.env.cfg)
            actor_extra_kwargs = {
                "cenet_explicit_layers": self.policy_cfg["cenet_explicit_layers"],
                "grf_decoder_layers": self.policy_cfg["grf_decoder_layers"],
                "wrench_decoder_layers": self.policy_cfg["wrench_decoder_layers"],
                "grf_scale": gain_spec.model_grf,
                "wrench_scale": gain_spec.model_wrench,
                "ablation_features": self.hard_pact_features,
            }
            reconstruction_indices = RECONSTRUCTION_INDICES
            reconstruction_dim = RECONSTRUCTION_DIM
        
        cenet_input_dim = self.env.num_obs * self.env.num_obs_hist

        actor_critic: ActorCritic_PACT = actor_critic_class(self.env.num_obs,
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
        
        # decoder = torch.compile(decoder)

        # print("Created Parallel Actor-Critic Model. Parameter Count: ", np.sum(p.numel() for p in actor_critic.parameters() if p.requires_grad))
        # print("\t Actor Trunk Parameter Count: ", np.sum(p.numel() for p in actor_critic.act_trunk.parameters() if p.requires_grad))
        # print("\t Encoder Parameter Count: ", np.sum(p.numel() for p in actor_critic.context_encoder.parameters() if p.requires_grad))
        # print("\t Critic Parameter Count: ", np.sum(p.numel() for p in actor_critic.critic.parameters() if p.requires_grad))
        # print("\t Decoder Parameter Count: ", np.sum(p.numel() for p in decoder.parameters() if p.requires_grad))
        # print(actor_critic)
        # print(decoder)

        print("Created Parallel Actor-Critic Model")
        pretty_print_module(actor_critic)
        pretty_print_module(decoder)

        self._init_entropy_coef = self.alg_cfg["entropy_coef"]
        self.use_adaptive_entropy = self.alg_cfg["use_adaptive_entropy"]


        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
                
        self.alg: PPO_PACT = alg_class(actor_critic, decoder, self.env.num_privileged_obs,
                                       pinn_lambda=self.policy_cfg["pinn_loss_weight"], 
                                       pinn_warmup=self.policy_cfg["pinn_warmup"], 
                                       pinn_init_steps=self.policy_cfg["pinn_init_steps"],
                                       reconstruction_indices=reconstruction_indices,
                                       device=self.device, **self.alg_cfg)
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_crit_obs_stack*self.env.num_privileged_obs], \
                              [reconstruction_dim], [self.env.num_obs_hist*self.env.num_obs], \
                              [2*self.env.num_actions], [self.env.num_exp_labels], [self.cfg["grf_dim"]], [self.env.wb_dim])
        if actor_critic_class is ActorCritic_HardPACT:
            delay_range = self.env.cfg.domain_rand.ctrl_delay_step_range
            self.alg.storage.configure_action_replay(int(delay_range[1]))
            # Limits are backend properties, not learned transition data.
            # Bind them once so PPO minibatches carry only the previous torque
            # needed by the rate constraint, minimizing persistent GPU memory.
            if self.hard_pact_features.execution_qp:
                simulator = self.env.simulator
                self.alg.configure_hard_pact_qp(
                    simulator._torque_limits,
                    simulator.dof_pos_limits_hard,
                    simulator._dof_vel_limits,
                )
                # Reuse the same BARD model and qpth layer for rollout and PPO;
                # duplicating either would waste substantial GPU memory.
                self.env.configure_hard_pact_substep_qp(
                    self.alg.actor_critic,
                    self.alg.bard_dynamics,
                    self.alg.hard_pact_qp,
                )

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

        self.env.create_async_pino_workers()

        _, _ = self.env.reset()

    # function to load a boot-strap initial model and reset the std
    def _load_pretrained_model(self):
        pretrained_path = self.policy_cfg["pretrained_path"]
        print("Loading boot-strapping model from - ", pretrained_path)
        loaded_dict = torch.load(pretrained_path)
        # Load the pretrained action-network and encoder
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        # Load the pretrained decoder network
        self.alg.decoder.load_state_dict(loaded_dict['decoder_state_dict'])


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            self._log_hard_pact_run_metadata()

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
            self._rollout_qp_metric_sums = {}
            self._rollout_qp_metric_count = 0
            self._rollout_disturbance_active_sum = torch.zeros(
                (), device=self.device
            )
            self._rollout_disturbance_metric_count = 0
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # extract previous observations (obs_{t-1}) BEFORE running env.step()
                    #     as env.step() takes the obs_t (used below to get a_t) and sets obs_{t-1} = obs_t before computing obs_{t+1}
                    #     NOTE - tracking the last obs across episode resets is handled in the env class
                    prev_obs, prev_obs_hist, pprev_obs, pprev_obs_hist = self.env.get_prev_obs()
                    prev_obs, prev_obs_hist = prev_obs.to(self.device), prev_obs_hist.to(self.device)
                    pprev_obs, pprev_obs_hist = pprev_obs.to(self.device), pprev_obs_hist.to(self.device)

                    # Call the algorithms act() method to store current transition data and predict actions
                    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        actions = self.alg.act(obs, critic_obs, obs_hist, prev_obs, prev_obs_hist, pprev_obs, pprev_obs_hist) # obs_t, (obs_t-1)
                         
                    # Submit the predicted action and extract the resulting state... 
                    obs, privileged_obs, obs_hist, exp_labels, rewards, dones, infos, grfs = self.env.step(actions)  # obs_t+1  (obs_t)
                    self._accumulate_rollout_qp_metrics(infos)
                    
                    # Create privileged obs
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    
                    # get the PINN specific data
                    gt_forces, mass_mats, bias_vecs, torso_acc = self.env.get_pinn_wb_dynamics()
                    gt_forces, mass_mats, bias_vecs, torso_acc = gt_forces.to(self.device), mass_mats.to(self.device), bias_vecs.to(self.device), torso_acc.to(self.device)

                    # move everything to the correct device
                    obs, critic_obs, obs_hist, exp_labels, rewards, dones, grfs = obs.to(self.device), critic_obs.to(self.device), \
                        obs_hist.to(self.device), exp_labels.to(self.device), rewards.to(self.device), dones.to(self.device), grfs.to(self.device)

                    # Log the labels associated with the context decoder as well as the typical stuff
                    # self.alg.process_env_step(rewards, dones, infos, grfs, obs, exp_labels, gt_forces, mass_mats, bias_vecs, torso_acc)
                    self.alg.process_env_step(rewards, dones, infos, grfs, critic_obs, exp_labels, gt_forces, mass_mats, bias_vecs, torso_acc)

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
                    mean_recon_loss, mean_kld_loss, mean_pinn_loss \
                    = self.alg.update(self.env._get_pinn_actions, self.env._get_pinn_feedback, self.env.dt, it, self.env.simulator.default_dof_pos, self.env.obs_scales.dof_vel)

            # self.env.step_tradeoff_curriculum()
            print("Avg - Curriculum Step: ", torch.mean(self.env.tradeoff_step_ctr).item())
            print("Max - self.feedforward_tau_weight: ", torch.max(self.env.simulator.feedforward_tau_weight).item())
            print("Min - self.feedforward_tau_weight: ", torch.min(self.env.simulator.feedforward_tau_weight).item())
            print("Avg - self.feedforward_tau_weight: ", torch.mean(self.env.simulator.feedforward_tau_weight).item())
            print("Max - self.feedback_tau_weight: ", torch.max(self.env.simulator.feedback_tau_weight).item())
            print("Min - self.feedback_tau_weight: ", torch.min(self.env.simulator.feedback_tau_weight).item())
            print("Avg - self.feedback_tau_weight: ", torch.mean(self.env.simulator.feedback_tau_weight).item())
            
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
                mean_tracking_lin_vel = torch.stack(vals).mean().item()
                self.env.simulator._step_domian_rand(it, mean_tracking_lin_vel)

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
                print(entropy)
                self.writer.add_scalar('Values/entropy',entropy,it)

            # entropy_coef = self._init_entropy_coef
            # half_coef = self._init_entropy_coef * 0.5
            # tenth_coef = self._init_entropy_coef * 0.1
            # if it < 6500:
            #     entropy_coef = self._init_entropy_coef
            # elif it < 7500:
            #     alpha = (it - 6500) / 1000.0
            #     entropy_coef = tenth_coef + 0.5 * (self._init_entropy_coef - tenth_coef) * (1 + math.cos(math.pi * alpha))
            # else:
            #     entropy_coef = tenth_coef
            # print("entropy_coef - ", entropy_coef)
            # entropy_coef = max(entropy_coef, 0.00001)            
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
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

        # Learning is done, shutdown the async. pinocchio workers
        self.env.shutdown_asynic_pino_workers()

    def _log_qp_metrics(self, iteration):
        """Transfer only aggregated QP scalars to TensorBoard."""
        for name, value in getattr(self.alg, "last_qp_metrics", {}).items():
            if not name.startswith(("qp/minimal/", "qp/physical/", "qp/full/")):
                continue
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise ValueError(
                    f"QP runner metric {name!r} must be one scalar tensor"
                )
            self.writer.add_scalar(name, value.item(), iteration)

    def _accumulate_rollout_qp_metrics(self, infos):
        """Reduce per-environment rollout QP diagnostics on the live device."""
        transition = infos.get("hard_pact_transition") if isinstance(infos, dict) else None
        if transition and "sustained_wrench_active_mask" in transition:
            self._rollout_disturbance_active_sum += transition[
                "sustained_wrench_active_mask"
            ].float().mean()
            self._rollout_disturbance_metric_count += 1
        interval = infos.get("hard_pact_qp_interval") if isinstance(infos, dict) else None
        if not interval:
            return
        stage = interval["interval_qp_stage_fractions"]
        correction = interval["interval_qp_correction"].abs().reshape(-1)
        slack = interval["interval_qp_contact_slack"].abs().reshape(-1)
        residual = interval["interval_qp_residuals"].abs()
        timing = interval["interval_qp_timing_ms"]
        metrics = {
            "qp/minimal/full_fraction": stage[:, 0].mean(),
            "qp/minimal/relaxed_fraction": stage[:, 1].mean(),
            "qp/minimal/fallback_fraction": stage[:, 2].mean(),
            "qp/minimal/normalized_equality_residual_max": residual[:, 0].max(),
            "qp/minimal/normalized_inequality_violation_max": residual[:, 1].max(),
            "qp/minimal/rollout_timing_ms": timing.mean(),
            "qp/minimal/rollout_correction_mean": correction.mean(),
            "qp/minimal/rollout_correction_p95": torch.quantile(correction, 0.95),
            "qp/minimal/rollout_correction_max": correction.max(),
            "qp/minimal/rollout_slack_mean": slack.mean(),
            "qp/minimal/rollout_slack_p95": torch.quantile(slack, 0.95),
            "qp/minimal/rollout_slack_max": slack.max(),
        }
        for key, value in metrics.items():
            self._rollout_qp_metric_sums[key] = (
                self._rollout_qp_metric_sums.get(key, torch.zeros_like(value)) + value
            )
        self._rollout_qp_metric_count += 1

    @staticmethod
    def _scalar(value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("HardPACT logging accepts aggregated scalars only")
            return value.item()
        return float(value)

    def _log_hard_pact_run_metadata(self):
        """Write immutable provenance once; no per-iteration synchronization."""
        if not hasattr(self, "hard_pact_features"):
            return
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            sha = "unknown"
        metadata = {
            "variant": self.hard_pact_features.variant_id,
            "backend": self.cfg.get("task_backend", "unknown"),
            "seed": str(self.train_cfg.get("seed", "configured-by-task")),
            "git_sha": sha,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "dtype": str(next(self.alg.actor_critic.parameters()).dtype),
            "hardware": torch.cuda.get_device_name(self.device)
            if str(self.device).startswith("cuda") and torch.cuda.is_available()
            else platform.processor() or "cpu",
        }
        backend = metadata["backend"]
        distribution = {
            "genesis": "genesis-world", "isaacgym": "isaacgym",
            "isaaclab": "isaaclab",
        }.get(backend, backend)
        try:
            metadata["backend_version"] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            metadata["backend_version"] = "unknown"
        for key, value in metadata.items():
            self.writer.add_text(f"system/{key}", str(value), 0)
        self.writer.add_text(
            "system/config", json.dumps(self.train_cfg, sort_keys=True, default=str), 0
        )

    def _log_stable_hard_pact_metrics(self, locs):
        if not hasattr(self, "hard_pact_features"):
            return
        metrics = collect_hard_pact_scalars(self.alg, self.hard_pact_features)
        count = getattr(self, "_rollout_qp_metric_count", 0)
        if count:
            metrics.update({
                key: value / count
                for key, value in self._rollout_qp_metric_sums.items()
            })
        disturbance_count = getattr(self, "_rollout_disturbance_metric_count", 0)
        if disturbance_count:
            metrics["disturbance/persistent_active_fraction"] = (
                self._rollout_disturbance_active_sum / disturbance_count
            )
        simulator = self.env.simulator
        for attr, key in (
            ("domain_rand_joint_dynamics_progress", "domain_rand/joint_dynamics_progress"),
            ("domain_rand_mass_com_progress", "domain_rand/mass_com_progress"),
            ("domain_rand_disturbance_progress", "domain_rand/disturbance_progress"),
        ):
            if hasattr(simulator, attr):
                value = getattr(simulator, attr)
                metrics[key] = value.float().mean() if isinstance(
                    value, torch.Tensor
                ) and value.numel() != 1 else value
        metrics["system/timesteps"] = float(self.tot_timesteps)
        metrics["system/wall_time_s"] = float(self.tot_time)
        metrics["system/iteration_time_s"] = float(
            locs["collection_time"] + locs["learn_time"]
        )
        for name, enabled in self.hard_pact_features.scalar_flags().items():
            metrics[f"system/features/{name}"] = enabled
        for name, value in metrics.items():
            self.writer.add_scalar(name, self._scalar(value), locs["it"])

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        stable_episode_values = {}
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
                stable_episode_values[key] = value
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                # Stable reward/tracking namespaces coexist with legacy keys.
                stable_prefix = "tracking" if "tracking" in key else "reward"
                self.writer.add_scalar(f"{stable_prefix}/{key}", value, locs['it'])
                if "success" in key:
                    self.writer.add_scalar("train/success", value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        # These keys never disappear when an episode boundary is absent.
        canonical_episode = {
            "train/success": ("success",),
            "tracking/linear_velocity": ("rew_tracking_lin_vel", "tracking_lin_vel"),
            "tracking/angular_velocity": ("rew_tracking_ang_vel", "tracking_ang_vel"),
            "tracking/terrain_level": ("terrain_level",),
        }
        for output_key, candidates in canonical_episode.items():
            value = next((stable_episode_values[name] for name in candidates
                          if name in stable_episode_values), float("nan"))
            self.writer.add_scalar(output_key, value, locs['it'])
        # Reward-scale names are configuration-stable across all variants.
        for reward_name in getattr(self.env, "reward_scales", {}):
            episode_key = f"rew_{reward_name}"
            value = stable_episode_values.get(episode_key, float("nan"))
            self.writer.add_scalar(f"reward/{reward_name}", value, locs['it'])
        
        mean_std = self.alg.actor_critic.std.mean()
        
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        self.writer.add_scalar('Loss/autoenc_function', locs['mean_autoenc_loss'], locs['it'])
        self.writer.add_scalar('Loss/velo_pred', locs['mean_vel_loss'], locs['it'])
        self.writer.add_scalar('Loss/recon', locs['mean_recon_loss'], locs['it'])
        self.writer.add_scalar('Loss/kl_div', locs['mean_kld_loss'], locs['it'])
        self.writer.add_scalar('Loss/decoder_function', locs['mean_decoder_loss'], locs['it'])
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/pinn_loss', locs['mean_pinn_loss'], locs['it'])
        for name, value in getattr(
            self.alg, "last_inverse_dynamics_metrics", {}
        ).items():
            self.writer.add_scalar(name, value.item(), locs['it'])
        for name, value in getattr(
            self.alg, "last_rollout_dynamics_metrics", {}
        ).items():
            self.writer.add_scalar(name, value.item(), locs['it'])
        for name, value in getattr(
            self.alg, "last_physics_gradient_metrics", {}
        ).items():
            self.writer.add_scalar(name, value.item(), locs['it'])
        # HardPACT QP exposes only already-aggregated device scalars. The
        # runner never receives per-environment residuals or solver matrices;
        # this is the sole device-to-host transfer for QP diagnostics.
        self._log_qp_metrics(locs['it'])
        self._log_stable_hard_pact_metrics(locs)
        for name, value in getattr(self.alg, "last_auxiliary_metrics", {}).items():
            self.writer.add_scalar(f"Loss/auxiliary_{name}", value.item(), locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])        
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
            self.writer.add_scalar('train/return', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('train/episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
        else:
            self.writer.add_scalar('train/return', float("nan"), locs['it'])
            self.writer.add_scalar('train/episode_length', float("nan"), locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'PINN loss:':>{pad}} {locs['mean_pinn_loss']:.4f}\n"""
                          f"""{'Autoenc function loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
                          f"""{'Torso Velo. Pred loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                          f"""{'Reconstruction   loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"""
                          f"""{'KL Divergence    loss:':>{pad}} {locs['mean_kld_loss']:.4f}\n"""
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
                          f"""{'PINN loss:':>{pad}} {locs['mean_pinn_loss']:.4f}\n"""
                          f"""{'Autoenc function loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
                          f"""{'Torso Velo. Pred loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                          f"""{'Reconstruction   loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"""
                          f"""{'KL Divergence    loss:':>{pad}} {locs['mean_kld_loss']:.4f}\n"""
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
            'act_optimizer_state_dict': self.alg.act_optimizer.optimizer.state_dict(),
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
            self.alg.act_optimizer.optimizer.load_state_dict(loaded_dict['act_optimizer_state_dict'])
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
