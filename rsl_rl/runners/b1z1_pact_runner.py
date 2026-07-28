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
            film_hidden_dim=policy_cfg["film_hidden_dim"], activation=policy_cfg["activation"],
            init_noise_std=policy_cfg["init_noise_std"],
        ).to(device)

        condition_dim = policy_cfg["cenet_latent_dim"] + 3 + 6 + 3

        self.force_decoder = B1Z1PACTDecoder(condition_dim, 15, activation=policy_cfg["activation"]).to(device)
        self.privileged_decoder = B1Z1PACTDecoder(condition_dim, env.num_privileged_obs, activation=policy_cfg["activation"]).to(device)

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
            "pinn_block_weights": [1.0] * 6 + [1.0] * 12 + [1.0] * 7,
        })

        merged.update({key: policy_cfg[key] for key in (
            "pinn_loss_weight", "pinn_warmup", "pinn_init_steps", "predicted_force_detach",
            "force_gate_ema_alpha", "force_gate_threshold", "force_gate_hysteresis", "force_gate_patience",
            "explicit_base_vel_weight", "explicit_base_wrench_weight", "explicit_ee_force_weight", "explicit_foot_contact_weight",
            "force_decoder_weight", "privileged_decoder_weight", "vae_kld_weight",
        )})

        self.alg = PPO_B1Z1PACT(self.actor_critic, self.force_decoder, self.privileged_decoder, self.dynamics, merged, device)

        self.alg.init_storage(
            env.num_envs, runner_cfg["num_steps_per_env"], env.num_obs, critic_dim,
            history_dim, 2 * env.num_actions, env.num_exp_labels, 15, env.num_privileged_obs, 180,
        )

        self.steps, self.save_interval = runner_cfg["num_steps_per_env"], runner_cfg["save_interval"]

        self.current_learning_iteration, self.total_timesteps, self.total_time = 0, 0, 0.0

        self.writer = SummaryWriter(log_dir, flush_secs=10) if log_dir else None

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=self.env.max_episode_length)
        self.env.reset()
        obs, history, privileged, explicit = self.env.get_observations()
        obs, history, privileged, explicit = (value.to(self.device) for value in (obs, history, privileged, explicit))
        rewards, lengths, ep_infos = deque(maxlen=100), deque(maxlen=100), []
        running_reward = torch.zeros(self.env.num_envs, 1, device=self.device)
        running_length = torch.zeros_like(running_reward)
        for iteration in range(self.current_learning_iteration, self.current_learning_iteration + num_learning_iterations):
            start = time.time()
            for _ in range(self.steps):
                with torch.inference_mode():
                    actions = self.alg.act(obs, privileged, history, explicit)
                next_obs, next_privileged, next_history, next_explicit, reward, dones, infos, _ = self.env.step(actions)
                next_obs, next_privileged, next_history, next_explicit, reward, dones = (
                    value.to(self.device) for value in (next_obs, next_privileged, next_history, next_explicit, reward, dones)
                )
                self.alg.process_env_step(
                    # The auxiliary labels and dynamics state are read after
                    # physics. Thus action_t is paired with x_(t+1), v_(t+1),
                    # and v_t cached by Genesis: the finite-difference PINN
                    # acceleration describes the transition caused by action_t.
                    reward, dones, infos, next_explicit, self.env.get_force_decoder_target().to(self.device),
                    next_privileged[:, -self.env.num_privileged_obs:], self.env.get_pact_dynamics_state().to(self.device),
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
            self.alg.compute_returns(privileged)
            update_start = time.time()
            metrics = self.alg.update(iteration)
            learning_time = time.time() - update_start
            self._log(
                iteration, self.current_learning_iteration + num_learning_iterations,
                metrics, collection_time, learning_time, rewards, lengths, ep_infos,
            )
            if self.log_dir and iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))
            ep_infos.clear()
        self.current_learning_iteration += num_learning_iterations
        if self.log_dir:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.dynamics.close()

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
            for name, value in metrics.items(): self.writer.add_scalar(f"Loss/{name}", value, iteration)
            self.writer.add_scalar("Policy/position_noise_std", position_std, iteration)
            self.writer.add_scalar("Policy/torque_noise_std", torque_std, iteration)
            self.writer.add_scalar("Perf/fps", fps, iteration)
            self.writer.add_scalar("Perf/collection_time", collection_time, iteration)
            self.writer.add_scalar("Perf/learning_time", learning_time, iteration)
            if mean_reward is not None: self.writer.add_scalar("Train/mean_reward", mean_reward, iteration)
            if mean_length is not None: self.writer.add_scalar("Train/mean_episode_length", mean_length, iteration)
            if ep_infos:
                for key, value in ep_infos[-1].items(): self.writer.add_scalar(f"Episode/{key}", float(value), iteration)

        width, pad = 80, 35
        header = f" \033[1m Learning iteration {iteration}/{total_iterations} \033[0m "
        lines = [
            "#" * width, header.center(width), "",
            f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learning_time:.3f}s)",
            f"{'PINN loss:':>{pad}} {metrics['pinn']:.4f}",
            f"{'Value function loss:':>{pad}} {metrics['value']:.4f}",
            f"{'Surrogate loss:':>{pad}} {metrics['surrogate']:.4f}",
            f"{'Base velocity loss:':>{pad}} {metrics['base_velo']:.4f}",
            f"{'Base wrench loss:':>{pad}} {metrics['base_wrench']:.4f}",
            f"{'EE force loss:':>{pad}} {metrics['ee_force']:.4f}",
            f"{'Foot-contact BCE loss:':>{pad}} {metrics['foot_contact']:.4f}",
            f"{'Privileged reconstruction loss:':>{pad}} {metrics['privileged_decoder']:.4f}",
            f"{'Force decoder loss:':>{pad}} {metrics['force_decoder']:.4f}",
            f"{'KL divergence loss:':>{pad}} {metrics['kl']:.4f}",
            f"{'Position action noise std:':>{pad}} {position_std:.2f}",
            f"{'Torque action noise std:':>{pad}} {torque_std:.2f}",
            f"{'Force decoder gate:':>{pad}} {metrics['force_gate_active']:.0f}",
            f"{'Latent bootstrap probability:':>{pad}} {metrics['latent_boot_probability']:.4f}",
            f"{'Explicit bootstrap probability:':>{pad}} {metrics['explicit_boot_probability']:.4f}",
        ]
        if mean_reward is not None:
            lines.extend((f"{'Mean reward:':>{pad}} {mean_reward:.2f}", f"{'Mean episode length:':>{pad}} {mean_length:.2f}"))
        for key, value in (ep_infos[-1].items() if ep_infos else ()):
            lines.append(f"{f'Mean episode {key}:':>{pad}} {float(value):.4f}")
        eta = self.total_time / max(iteration + 1, 1) * max(total_iterations - iteration, 0)
        lines.extend((
            "-" * width,
            f"{'Total timesteps:':>{pad}} {self.total_timesteps}",
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s",
            f"{'Total time:':>{pad}} {self.total_time:.2f}s",
            f"{'ETA:':>{pad}} {eta:.1f}s",
        ))
        print("\n".join(lines))

    def save(self, path):
        torch.save({
            "model_state_dict": self.actor_critic.state_dict(), "force_decoder_state_dict": self.force_decoder.state_dict(),
            "privileged_decoder_state_dict": self.privileged_decoder.state_dict(),
            "actor_optimizer": self.alg.actor_optimizer.optimizer.state_dict(),
            "auxiliary_optimizer": self.alg.auxiliary_optimizer.state_dict(),
            "iteration": self.current_learning_iteration, "force_ema": self.alg.force_ema,
            "force_gate_active": self.alg.force_gate_active, "force_gate_count": self.alg.force_gate_count,
            "use_boot_latent": self.alg.use_boot_latent,
            "use_boot_explicit": self.alg.use_boot_explicit,
        }, path)

    def load(self, path, load_optimizer=True):
        """Restore all learned heads and the force-reliability gate state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        self.force_decoder.load_state_dict(checkpoint["force_decoder_state_dict"])
        self.privileged_decoder.load_state_dict(checkpoint["privileged_decoder_state_dict"])
        if load_optimizer:
            self.alg.actor_optimizer.optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.alg.auxiliary_optimizer.load_state_dict(checkpoint["auxiliary_optimizer"])
        self.current_learning_iteration = checkpoint.get("iteration", 0)
        self.alg.force_ema = checkpoint.get("force_ema")
        self.alg.force_gate_active = checkpoint.get("force_gate_active", False)
        self.alg.force_gate_count = checkpoint.get("force_gate_count", 0)
        self.alg.use_boot_latent = checkpoint.get("use_boot_latent", False)
        self.alg.use_boot_explicit = checkpoint.get("use_boot_explicit", False)
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
