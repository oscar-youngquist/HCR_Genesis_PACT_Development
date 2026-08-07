"""PACT-style training runner for the standalone B1/Z1 coupled-action task."""

from __future__ import annotations

import os
import math
import time
import statistics
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms.ppo_b1z1_pact_pos import PPO_B1Z1PACTPos
from rsl_rl.modules.actor_critic_b1z1_pact_pos import ActorCriticB1Z1PACTPos, B1Z1PACTDecoder


class B1Z1PACTPosRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.cfg, self.device, self.log_dir = env, train_cfg, device, log_dir

        policy_cfg, algorithm_cfg, runner_cfg = train_cfg["policy"], train_cfg["algorithm"], train_cfg["runner"]

        critic_dim = env.num_privileged_obs * env.num_crit_obs_stack

        history_dim = env.num_obs * env.num_obs_hist

        self.actor_critic = ActorCriticB1Z1PACTPos(
            env.num_obs, critic_dim, env.num_actions, history_dim,
            latent_dim=policy_cfg["cenet_latent_dim"], actor_layers=policy_cfg["actor_layers"],
            critic_layers=policy_cfg["critic_layers"], context_layers=policy_cfg["cenet_enc_layers"],
            explicit_decoder_layers=policy_cfg["explicit_decoder_layers"],
            explicit_dim=env.num_exp_labels,
            film_hidden_dim=policy_cfg["film_hidden_dim"], activation=policy_cfg["activation"],
            init_noise_std=policy_cfg["init_noise_std"],
            min_noise_std=policy_cfg["min_noise_std"],
            max_noise_std=policy_cfg["max_noise_std"],
        ).to(device)

        self.privileged_decoder = B1Z1PACTDecoder(
            # Match UniFP: next-state privileged reconstruction is decoded
            # from z alone, without explicit estimates as side information.
            policy_cfg["cenet_latent_dim"], env.num_privileged_obs,
            hidden=policy_cfg["privileged_decoder_layers"],
            activation=policy_cfg["activation"],
        ).to(device)

        merged = dict(algorithm_cfg)
        merged.update({
            "dt": env.dt, "position_action_scale": env.cfg.control.action_scale,
            "torque_action_scale": env.cfg.control.torque_scale,
            "dof_pos_obs_scale": env.obs_scales.dof_pos,
            "dof_vel_obs_scale": env.obs_scales.dof_vel,
            "privileged_force_start": env.cfg.env.privileged_force_start,
            "privileged_force_dim": env.cfg.env.num_privileged_force_obs,
        })

        merged.update({key: policy_cfg[key] for key in (
            "film_identity_loss_weight", "film_identity_error_scale",
            "torque_clone_target_scale", "torque_clone_loss_weight",
            "explicit_base_vel_weight", "explicit_ee_position_weight", "explicit_base_wrench_weight", "explicit_ee_force_weight", "explicit_foot_contact_weight",
            "explicit_foot_height_weight",
            "privileged_decoder_weight", "vae_kld_weight",
            "adaptation_learning_rate",
        )})

        self.alg = PPO_B1Z1PACTPos(
            self.actor_critic, self.privileged_decoder, merged, device
        )

        self.alg.init_storage(
            env.num_envs, runner_cfg["num_steps_per_env"], env.num_obs, critic_dim,
            history_dim, env.num_actions, env.num_exp_labels, env.num_privileged_obs, 76,
        )

        self.steps, self.save_interval = runner_cfg["num_steps_per_env"], runner_cfg["save_interval"]
        self.enable_additional_diagnostics = runner_cfg.get("enable_additional_diagnostics", True)
        self.alg.enable_additional_diagnostics = self.enable_additional_diagnostics
        self.use_adaptive_entropy = algorithm_cfg.get("use_adaptive_entropy", False)

        self.current_learning_iteration, self.total_timesteps, self.total_time = 0, 0, 0.0

        self.writer = None
        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=self.env.max_episode_length)
        obs, history, privileged, explicit = self.env.get_observations()
        obs, history, privileged, explicit = (value.to(self.device) for value in (obs, history, privileged, explicit))
        rewards, lengths, ep_infos = deque(maxlen=100), deque(maxlen=100), []
        running_reward = torch.zeros(self.env.num_envs, 1, device=self.device)
        running_length = torch.zeros_like(running_reward)
        self.actor_critic.train()
        final_iteration = self.current_learning_iteration + num_learning_iterations
        for iteration in range(self.current_learning_iteration, final_iteration):
            # Reward schedules use the true checkpoint-aware PPO iteration,
            # never an approximation based on environment transitions.
            if hasattr(self.env, "set_training_iteration"):
                self.env.set_training_iteration(iteration)
            if self.enable_additional_diagnostics and hasattr(self.actor_critic, "begin_rollout_diagnostics"):
                self.actor_critic.begin_rollout_diagnostics()
            if self.enable_additional_diagnostics and hasattr(self.env, "begin_rollout_diagnostics"):
                self.env.begin_rollout_diagnostics()
            start = time.time()
            # Rollout collection never needs autograd. Cover environment tensor
            # work and storage copies too, matching the fast UniFP runner path.
            with torch.inference_mode():
                for _ in range(self.steps):
                    actions = self.alg.act(obs, privileged, history, explicit)
                    if self.enable_additional_diagnostics and hasattr(self.actor_critic, "record_rollout_diagnostics"):
                        self.actor_critic.record_rollout_diagnostics(
                            actions, self.env.cfg.normalization.clip_actions
                        )
                    next_obs, next_privileged, next_history, next_explicit, reward, dones, infos, _ = self.env.step(actions)
                    next_obs, next_privileged, next_history, next_explicit, reward, dones = (
                        value.to(self.device) for value in (next_obs, next_privileged, next_history, next_explicit, reward, dones)
                    )
                    self.alg.process_env_step(
                        reward, dones, infos,
                        next_privileged[:, -self.env.num_privileged_obs:],
                        self.env.get_pact_pos_torque_clone_state().to(self.device),
                    )
                    running_reward += reward.view(-1, 1)
                    running_length += 1
                    done_ids = dones.nonzero(as_tuple=False).view(-1)
                    if len(done_ids):
                        rewards.extend(running_reward[done_ids, 0].cpu().tolist())
                        lengths.extend(running_length[done_ids, 0].cpu().tolist())
                        running_reward[done_ids] = 0
                        running_length[done_ids] = 0
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    obs, history, privileged, explicit = next_obs, next_history, next_privileged, next_explicit
                collection_time = time.time() - start
                # The bootstrap value is a rollout target, not a training graph.
                # Keep it in inference mode because the final observation was
                # assembled under this same guard.
                self.alg.compute_returns(privileged)
                policy_diagnostics = (
                    self.actor_critic.get_rollout_diagnostics()
                    if self.enable_additional_diagnostics and hasattr(self.actor_critic, "get_rollout_diagnostics") else {}
                )
                environment_diagnostics = (
                    self.env.get_rollout_diagnostics()
                    if self.enable_additional_diagnostics and hasattr(self.env, "get_rollout_diagnostics") else {}
                )
            update_start = time.time()
            metrics = self.alg.update(iteration)
            learning_time = time.time() - update_start
            if ep_infos and self.use_adaptive_entropy:
                self.alg.update_adaptive_entropy_coef({
                    "lin_vel_tracking": self._episode_metric_max(ep_infos, "rew_tracking_lin_vel_force_world"),
                    "ang_vel_tracking": self._episode_metric_max(ep_infos, "rew_tracking_ang_vel"),
                    "terrain_level": self._episode_metric_max(ep_infos, "terrain_level", "terrain_level_mean"),
                })
            if getattr(self.env, "use_reward_curriculum", False):
                self.env.step_reward_curriculum(iteration)
            self._step_domain_randomization_curriculum(iteration, ep_infos)
            metrics.update(policy_diagnostics)
            metrics.update(environment_diagnostics)
            if self.enable_additional_diagnostics:
                self._validate_diagnostics(metrics)
            self._log(
                iteration, self.current_learning_iteration + num_learning_iterations,
                metrics, collection_time, learning_time, rewards, lengths, ep_infos,
            )
            if self.log_dir and iteration % self.save_interval == 0:
                # Store the next PPO iteration so resumed schedules continue
                # from the first iteration that has not yet been completed.
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"), iteration=iteration + 1)
            ep_infos.clear()
        self.current_learning_iteration = final_iteration
        if hasattr(self.env, "set_training_iteration"):
            self.env.set_training_iteration(self.current_learning_iteration)
        if self.log_dir:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    @staticmethod
    def _episode_metric_max(ep_infos, *names):
        """Reduce vectorized episode summaries to the best completed episode."""
        values = []
        for info in ep_infos:
            name = next((candidate for candidate in names if candidate in info), None)
            if name is not None:
                values.append(torch.as_tensor(info[name]).float().mean().item())
        return max(values, default=0.0)

    def _step_domain_randomization_curriculum(self, iteration, ep_infos):
        """Mirror UniFP's reward-driven simulator domain-randomization update."""
        simulator = self.env.simulator
        if not getattr(simulator, "use_domainrand_curriculum", False):
            return
        has_tracking = any("rew_tracking_lin_vel_force_world" in info for info in ep_infos)
        tracking = self._episode_metric_max(ep_infos, "rew_tracking_lin_vel_force_world") if has_tracking else None
        if tracking is not None:
            simulator._step_domian_rand(iteration, tracking)
        if self.writer is not None:
            reward_ema = simulator.domain_rand_reward_ema
            self.writer.add_scalar("Values/domain_rand_reward_ema", reward_ema if reward_ema is not None else 0.0, iteration)
            self.writer.add_scalar("Values/required_reward", simulator.required_reward, iteration)
            self.writer.add_scalar("Values/domain_rand_joint_dynamics_progress", simulator.domain_rand_joint_dynamics_progress, iteration)
            self.writer.add_scalar("Values/domain_rand_mass_com_progress", simulator.domain_rand_mass_com_progress, iteration)
            self.writer.add_scalar("Values/domain_rand_disturbance_progress", simulator.domain_rand_disturbance_progress, iteration)

    @staticmethod
    def _validate_diagnostics(metrics):
        """Match UniFP's finite-value and bounded-fraction checks."""
        bounded_tokens = (
            "fraction", "contact_match", "contact_duty",
            "clearance_success", "diagonal_contact_agreement",
        )
        for name, value in metrics.items():
            value = float(value)
            if not math.isfinite(value):
                raise RuntimeError(f"nonfinite diagnostic {name}: {value}")
            if any(token in name for token in bounded_tokens) and not 0.0 <= value <= 1.0:
                raise RuntimeError(f"bounded diagnostic {name} outside [0, 1]: {value}")

    def _log(self, iteration, total_iterations, metrics, collection_time, learning_time, rewards, lengths, ep_infos):
        """Print the shared PACT/UniFP training panel plus B1Z1 diagnostics."""
        self.total_timesteps += self.steps * self.env.num_envs
        iteration_time = collection_time + learning_time
        self.total_time += iteration_time
        fps = self.steps * self.env.num_envs / max(iteration_time, 1e-6)
        position_std = self.actor_critic.std[:self.env.num_actions].mean().item()
        mean_reward = statistics.mean(rewards) if rewards else None
        mean_length = statistics.mean(lengths) if lengths else None

        if self.writer:
            for name, value in metrics.items():
                if "/" in name:
                    self.writer.add_scalar(name, value, iteration)
                else:
                    group = "PPO" if name.startswith(("pre_update_", "lr_")) else "Loss"
                    self.writer.add_scalar(f"{group}/{name}", value, iteration)
            self.writer.add_scalar("Policy/position_noise_std", position_std, iteration)
            self.writer.add_scalar("Policy/mean_noise_std", self.actor_critic.std.mean(), iteration)
            self.writer.add_scalar("Values/entropy", self.alg.current_entropy_coef, iteration)
            self.writer.add_scalar("Perf/total_fps", fps, iteration)
            self.writer.add_scalar("Perf/collection time", collection_time, iteration)
            self.writer.add_scalar("Perf/learning_time", learning_time, iteration)
            if hasattr(self.env, "get_gait_guidance_multipliers"):
                for name, multiplier in self.env.get_gait_guidance_multipliers().items():
                    self.writer.add_scalar(f"Values/gait_guidance_{name}_multiplier", multiplier, iteration)
            if mean_reward is not None: self.writer.add_scalar("Train/mean_reward", mean_reward, iteration)
            if mean_length is not None: self.writer.add_scalar("Train/mean_episode_length", mean_length, iteration)
            if mean_reward is not None: self.writer.add_scalar("Train/mean_reward/time", mean_reward, self.total_time)
            if mean_length is not None: self.writer.add_scalar("Train/mean_episode_length/time", mean_length, self.total_time)
            if ep_infos:
                for key in ep_infos[0]:
                    values = [torch.as_tensor(info[key], device=self.device).float().mean() for info in ep_infos if key in info]
                    if values:
                        self.writer.add_scalar(f"Episode/{key}", torch.stack(values).mean(), iteration)

        width, pad = 80, 35
        header = f" \033[1m Learning iteration {iteration}/{total_iterations} \033[0m "
        lines = [
            "#" * width, header.center(width), "",
            f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learning_time:.3f}s)",
            f"{'Torque clone loss:':>{pad}} {metrics['torque_clone']:.4f}",
            f"{'Value function loss:':>{pad}} {metrics['value']:.4f}",
            f"{'Surrogate loss:':>{pad}} {metrics['surrogate']:.4f}",
            f"{'FiLM identity loss:':>{pad}} {metrics['film_identity']:.4f}",
            f"{'Base velocity loss:':>{pad}} {metrics['base_velo']:.4f}",
            f"{'EE position loss:':>{pad}} {metrics['ee_position']:.4f}",
            f"{'Base wrench loss:':>{pad}} {metrics['base_wrench']:.4f}",
            f"{'EE force loss:':>{pad}} {metrics['ee_force']:.4f}",
            f"{'Foot-contact BCE loss:':>{pad}} {metrics['foot_contact']:.4f}",
            f"{'Foot-height loss:':>{pad}} {metrics['foot_height']:.4f}",
            f"{'Privileged reconstruction loss:':>{pad}} {metrics['privileged_decoder']:.4f}",
            f"{'Privileged force loss:':>{pad}} {metrics['privileged_force']:.4f}",
            f"{'KL divergence loss:':>{pad}} {metrics['kl']:.4f}",
            f"{'Position action noise std:':>{pad}} {position_std:.2f}",
            f"{'Entropy coefficient:':>{pad}} {self.alg.current_entropy_coef:.6f}",
            f"{'Latent bootstrap probability:':>{pad}} {metrics['latent_boot_probability']:.4f}",
        ]
        if mean_reward is not None:
            lines.extend((f"{'Mean reward:':>{pad}} {mean_reward:.2f}", f"{'Mean episode length:':>{pad}} {mean_length:.2f}"))
        for key in (ep_infos[0] if ep_infos else {}):
            values = [torch.as_tensor(info[key], device=self.device).float().mean() for info in ep_infos if key in info]
            if values:
                lines.append(f"{f'Mean episode {key}:':>{pad}} {torch.stack(values).mean().item():.4f}")
        eta = self.total_time / max(iteration + 1, 1) * max(total_iterations - iteration, 0)
        lines.extend((
            "-" * width,
            f"{'Total timesteps:':>{pad}} {self.total_timesteps}",
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s",
            f"{'Total time:':>{pad}} {self.total_time:.2f}s",
            f"{'ETA:':>{pad}} {eta:.1f}s",
        ))
        print("\n".join(lines))

    def save(self, path, iteration=None):
        saved_iteration = self.current_learning_iteration if iteration is None else int(iteration)
        torch.save({
            "model_state_dict": self.actor_critic.state_dict(),
            "privileged_decoder_state_dict": self.privileged_decoder.state_dict(),
            "actor_optimizer": self.alg.actor_optimizer.optimizer.state_dict(),
            "auxiliary_optimizer": self.alg.auxiliary_optimizer.state_dict(),
            "iteration": saved_iteration,
            "use_boot_latent": self.alg.use_boot_latent,
            "entropy_coef": self.alg.current_entropy_coef,
        }, path)

    def load(self, path, load_optimizer=True):
        """Restore the policy, reconstruction decoder, and curriculum state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        self.privileged_decoder.load_state_dict(checkpoint["privileged_decoder_state_dict"])
        if load_optimizer:
            self.alg.actor_optimizer.optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.alg.auxiliary_optimizer.load_state_dict(checkpoint["auxiliary_optimizer"])
        self.current_learning_iteration = checkpoint.get("iteration", 0)
        if hasattr(self.env, "set_training_iteration"):
            self.env.set_training_iteration(self.current_learning_iteration)
        # Latent bootstrap masking is retired: resumed policies always consume
        # the encoder's history latent regardless of legacy checkpoint state.
        self.alg.use_boot_latent = True
        self.alg.current_entropy_coef = checkpoint.get("entropy_coef", self.alg.current_entropy_coef)
        return checkpoint.get("iteration", 0)

    def get_inference_policy(self, device=None):
        self.actor_critic.eval()
        if device: self.actor_critic.to(device)
        return self.actor_critic.act_inference
