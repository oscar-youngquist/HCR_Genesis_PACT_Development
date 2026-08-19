"""PACT-style training runner for the standalone B1/Z1 coupled-action task."""

from __future__ import annotations

import os
import time
import math
import statistics
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.dynamics import PinocchioWholeBodyDynamics
from rsl_rl.algorithms.ppo_b1z1_pact import PPO_B1Z1PACT
from rsl_rl.modules.actor_critic_b1z1_pact import ActorCriticB1Z1PACT, B1Z1PACTDecoder
from rsl_rl.utils import RolloutPhaseTimer, log_startup_metadata, startup_metadata


class B1Z1PACTRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.cfg, self.device, self.log_dir = env, train_cfg, device, log_dir

        policy_cfg, algorithm_cfg, runner_cfg = train_cfg["policy"], train_cfg["algorithm"], train_cfg["runner"]

        critic_dim = env.num_privileged_obs * env.num_crit_obs_stack

        history_dim = env.num_obs * env.num_obs_hist

        self.actor_critic = ActorCriticB1Z1PACT(
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
            # Decode the next non-terrain privileged state from z. Terrain
            # heights remain available to the critic but are not reconstructed.
            policy_cfg["cenet_latent_dim"], env.cfg.env.num_privileged_recon_obs,
            hidden=policy_cfg["privileged_decoder_layers"],
            activation=policy_cfg["activation"],
        ).to(device)

        urdf = env.cfg.asset.file.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        rollout_samples = env.num_envs * runner_cfg["num_steps_per_env"]
        default_pino_capacity = math.ceil(rollout_samples / algorithm_cfg["num_mini_batches"])
        pino_capacity = algorithm_cfg.get("pino_batch_capacity", 0) or default_pino_capacity
        self.dynamics = PinocchioWholeBodyDynamics(
            urdf,
            env.cfg.asset.dof_names,
            env.cfg.asset.foot_name,
            env.cfg.asset.gripper_name,
            env.cfg.asset.base_name,
            num_workers=algorithm_cfg.get("pino_num_workers", 0),
            batch_capacity=pino_capacity,
            worker_start_method=algorithm_cfg.get("pino_worker_start_method", "spawn"),
        )

        merged = dict(algorithm_cfg)
        merged.update({
            "dt": env.dt, "position_action_scale": env.cfg.control.action_scale,
            "torque_action_scale": env.cfg.control.torque_scale,
            "grf_scale": env.obs_scales.grf,
            "ee_force_scale": env.obs_scales.ee_force,
            "base_velocity_scale": env.base_velocity_scale.tolist(),
            "base_wrench_scale": env.base_wrench_scale.tolist(),
            "privileged_force_start": env.cfg.env.privileged_force_start,
            "privileged_force_dim": env.cfg.env.num_privileged_force_obs,
        })

        merged.update({key: policy_cfg[key] for key in (
            "film_identity_loss_weight", "film_identity_error_scale",
            "pinn_loss_weight", "pinn_warmup", "pinn_init_steps", "predicted_force_detach",
            "force_gate_ema_alpha", "force_gate_threshold", "force_gate_hysteresis", "force_gate_patience",
            "force_blend_min_alpha",
            "explicit_base_vel_weight", "explicit_ee_position_weight", "explicit_base_wrench_weight", "explicit_ee_force_weight", "explicit_foot_contact_weight",
            "explicit_foot_height_weight",
            "privileged_decoder_weight", "vae_kld_weight",
            "use_kl_rate_band", "use_cosine_kl_warmup", "kl_warmup_iters", "kl_warmup_beta_max", "kl_r_min", "kl_r_max",
            "kl_dual_lr", "kl_aug_rho", "kl_ema_decay",
            "adaptation_learning_rate",
        )})

        self.alg = PPO_B1Z1PACT(self.actor_critic, self.privileged_decoder, self.dynamics, merged, device)

        # Match the original Go2 PACT bootstrap path: initialize model weights
        # from a converted PACT-Pos checkpoint while keeping fresh optimizers,
        # schedules, reliability-gate state, and training iteration counters.
        if policy_cfg.get("pretrained_path"):
            self._load_pretrained_model(policy_cfg["pretrained_path"])

        self.alg.init_storage(
            env.num_envs, runner_cfg["num_steps_per_env"], env.num_obs, critic_dim,
            history_dim, env.cfg.env.num_policy_actions, env.num_exp_labels,
            env.cfg.env.num_privileged_recon_obs, 180,
        )

        self.steps, self.save_interval = runner_cfg["num_steps_per_env"], runner_cfg["save_interval"]
        self.enable_additional_diagnostics = runner_cfg.get("enable_additional_diagnostics", True)
        self.alg.enable_additional_diagnostics = self.enable_additional_diagnostics
        self.env.enable_additional_diagnostics = self.enable_additional_diagnostics
        self.use_adaptive_entropy = algorithm_cfg.get("use_adaptive_entropy", False)

        self.current_learning_iteration, self.total_timesteps, self.total_time = 0, 0, 0.0
        self._startup_metadata_logged = False

        self.writer = None
        _, _ = self.env.reset()

    def _load_pretrained_model(self, pretrained_path):
        """Load a weights-only PACT hot start produced by the conversion notebook."""
        pretrained_path = os.path.expanduser(pretrained_path)
        if not os.path.isabs(pretrained_path):
            pretrained_path = os.path.join(LEGGED_GYM_ROOT_DIR, pretrained_path)
        print(f"Loading PACT-Pos hot-start weights from: {pretrained_path}")
        checkpoint = torch.load(
            pretrained_path, map_location=self.device, weights_only=True
        )
        required = {"model_state_dict", "privileged_decoder_state_dict"}
        missing = required.difference(checkpoint)
        if missing:
            raise KeyError(f"Hot-start checkpoint is missing keys: {sorted(missing)}")
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.privileged_decoder.load_state_dict(
            checkpoint["privileged_decoder_state_dict"], strict=True
        )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=self.env.max_episode_length)
        obs, history, privileged, explicit = self.env.get_observations()
        obs, history, privileged, explicit = (value.to(self.device) for value in (obs, history, privileged, explicit))
        if self.enable_additional_diagnostics and not self._startup_metadata_logged:
            log_startup_metadata(
                startup_metadata(
                    self.env, self.device, (obs, history, privileged, explicit), self.actor_critic
                ),
                self.writer,
            )
            self._startup_metadata_logged = True
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
            rollout_timer = (
                RolloutPhaseTimer(self.device)
                if self.enable_additional_diagnostics else None
            )
            self.env._rollout_phase_timer = rollout_timer
            start = time.time()
            # Rollout collection never needs autograd. Cover simulation,
            # observation/reward construction, and storage copies as well as
            # policy inference, matching the optimized UniFP/Pact-pos path.
            with torch.inference_mode():
                for _ in range(self.steps):
                    policy_start = rollout_timer.start("policy") if rollout_timer is not None else None
                    actions = self.alg.act(obs, privileged, history, explicit)
                    if rollout_timer is not None:
                        rollout_timer.stop("policy", policy_start)
                    if self.enable_additional_diagnostics and hasattr(self.actor_critic, "record_rollout_diagnostics"):
                        self.actor_critic.record_rollout_diagnostics(
                            actions, self.env.cfg.normalization.clip_actions
                        )
                    next_obs, next_privileged, next_history, next_explicit, reward, dones, infos, _ = self.env.step(actions)
                    storage_start = rollout_timer.start("transition_storage") if rollout_timer is not None else None
                    next_obs, next_privileged, next_history, next_explicit, reward, dones = (
                        value.to(self.device) for value in (next_obs, next_privileged, next_history, next_explicit, reward, dones)
                    )
                    self.alg.process_env_step(
                        # Preserve the complete transition-aligned 180-D packet:
                        # unlike PACT-pos, the active PINN consumes its floating
                        # base, force, velocity, and inertial-randomization data.
                        reward, dones, infos,
                        # The final stacked frame is [state, terrain heights].
                        # Store only state as the next-frame decoder target.
                        next_privileged[:, -self.env.num_privileged_obs:][
                            :, :self.env.cfg.env.num_privileged_recon_obs
                        ],
                        self.env.get_pact_dynamics_state().to(self.device),
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
                    if rollout_timer is not None:
                        rollout_timer.stop("transition_storage", storage_start)
                rollout_timing_metrics = rollout_timer.finish() if rollout_timer is not None else {}
                self.env._rollout_phase_timer = None
                collection_time = (
                    rollout_timing_metrics["Perf/Rollout/total_collection_ms"] / 1000.0
                    if rollout_timer is not None else time.time() - start
                )
                # Bootstrap values are rollout targets and the final privileged
                # observation was assembled under this inference guard.
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
            metrics.update(rollout_timing_metrics)
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
        self.dynamics.close()

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
        torque_std = self.actor_critic.std[self.env.num_actions:].mean().item()
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
            self.writer.add_scalar("Policy/torque_noise_std", torque_std, iteration)
            self.writer.add_scalar("Policy/mean_noise_std", self.actor_critic.std.mean(), iteration)
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, iteration)
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
            f"{'PINN loss:':>{pad}} {metrics['pinn']:.4f}",
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
            f"{'Raw KL divergence:':>{pad}} {metrics['kl_raw']:.4f}",
            f"{'Position action noise std:':>{pad}} {position_std:.2f}",
            f"{'Torque action noise std:':>{pad}} {torque_std:.2f}",
            f"{'Entropy coefficient:':>{pad}} {self.alg.current_entropy_coef:.6f}",
            f"{'Privileged force gate:':>{pad}} {metrics['force_gate_active']:.0f}",
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
            "iteration": saved_iteration, "force_ema": self.alg.force_ema,
            "force_gate_active": self.alg.force_gate_active, "force_gate_count": self.alg.force_gate_count,
            "force_blend_start_ema": self.alg.force_blend_start_ema,
            "entropy_coef": self.alg.current_entropy_coef,
            "kl_controller_state": self.alg.kl_controller.state_dict(),
        }, path)

    def load(self, path, load_optimizer=True):
        """Restore learned heads and the privileged-force reliability gate."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        self.privileged_decoder.load_state_dict(checkpoint["privileged_decoder_state_dict"])
        if load_optimizer:
            self.alg.actor_optimizer.optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.alg.auxiliary_optimizer.load_state_dict(checkpoint["auxiliary_optimizer"])
        self.current_learning_iteration = checkpoint.get("iteration", 0)
        if hasattr(self.env, "set_training_iteration"):
            self.env.set_training_iteration(self.current_learning_iteration)
        self.alg.force_ema = checkpoint.get("force_ema")
        self.alg.force_gate_active = checkpoint.get("force_gate_active", False)
        self.alg.force_gate_count = checkpoint.get("force_gate_count", 0)
        self.alg.force_blend_start_ema = checkpoint.get(
            "force_blend_start_ema", self.alg.force_ema
        )
        self.alg.current_entropy_coef = checkpoint.get("entropy_coef", self.alg.current_entropy_coef)
        self.alg.kl_controller.load_state_dict(checkpoint.get("kl_controller_state"))
        return checkpoint.get("iteration", 0)

    def get_inference_policy(self, device=None):
        self.actor_critic.eval()
        if device: self.actor_critic.to(device)
        return self.actor_critic.act_inference

    def __del__(self):
        # Handles interrupted runs before learn() reaches its normal shutdown.
        dynamics = getattr(self, "dynamics", None)
        if dynamics is not None:
            dynamics.close()
