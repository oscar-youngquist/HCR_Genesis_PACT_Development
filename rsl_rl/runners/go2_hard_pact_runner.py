"""On-policy runner for the canonical Go2 HardPACT transition contract."""

from __future__ import annotations

import json
import importlib
import importlib.metadata
import os
import platform
import subprocess
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms import PPOGo2HardPACT
from rsl_rl.modules import (
    ActorCriticGo2HardPACT,
    ActorCriticGo2HardPACTPos,
    migrate_hard_pact_pos_checkpoint,
)


class Go2HardPACTRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.cfg = train_cfg
        self.device = torch.device(device)
        self.log_dir = log_dir
        policy_cfg = dict(train_cfg["policy"])
        policy_class = (
            ActorCriticGo2HardPACTPos
            if policy_cfg.get("position_pretraining", False)
            else ActorCriticGo2HardPACT
        )
        self.actor_critic = policy_class(
            num_actor_obs=env.num_obs,
            num_critic_obs=env.num_privileged_obs,
            num_actions=env.num_actions,
            **policy_cfg,
        )
        algorithm_cfg = dict(train_cfg["algorithm"])
        features = env.cfg.features
        algorithm_cfg.update({
            "supervised_physics_head_pretraining": bool(
                features.supervised_physics_head_pretraining
            ),
            "use_bard_inverse_loss": bool(features.use_bard_inverse_loss),
            "use_bard_rollout_loss": bool(features.use_bard_rollout_loss),
            "use_qp": bool(features.use_qp),
            "differentiate_qp": bool(features.differentiate_qp),
            "stop_gradient_qp": bool(features.stop_gradient_qp),
            "use_soft_projection_penalty": bool(
                features.use_soft_projection_penalty
            ),
            "lambda_inverse": float(env.cfg.bard.lambda_inverse),
            "lambda_rollout": float(env.cfg.bard.lambda_rollout),
            "lambda_projection": float(env.cfg.bard.lambda_projection),
            "grf_loss_weight": float(features.grf_supervision_weight),
            "active_wrench_loss_weight": float(
                features.active_wrench_supervision_weight
            ),
            "neutral_wrench_loss_weight": float(
                features.neutral_wrench_supervision_weight
            ),
            "feedforward_clone_weight": float(
                features.feedforward_clone_weight
            ),
        })
        self.alg = PPOGo2HardPACT(
            self.actor_critic, device=device, **algorithm_cfg
        )
        self.num_steps_per_env = int(train_cfg["runner"]["num_steps_per_env"])
        self.alg.init_storage(
            env.num_envs,
            self.num_steps_per_env,
            env.num_obs,
            env.num_privileged_obs,
            env.num_obs * env.num_obs_hist,
        )
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None
        self.last_migration_report = None
        if log_dir:
            self._write_metadata()

    def _write_metadata(self):
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except (OSError, subprocess.SubprocessError):
            git_sha = "unknown"
        dependency_versions = {}
        for package in ("genesis-world", "qpth", "bard"):
            try:
                dependency_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                dependency_versions[package] = None
        backend_name = self.env.backend.capabilities.name
        try:
            backend_module = importlib.import_module(
                {"genesis": "genesis", "isaaclab": "isaaclab"}[
                    backend_name
                ]
            )
            backend_version = getattr(backend_module, "__version__", "unknown")
        except (ImportError, KeyError):
            backend_version = "unavailable"
        gpu_properties = None
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            gpu_properties = {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        metadata = {
            "config": self.cfg,
            "seed": self.cfg.get("seed"),
            "git_sha": git_sha,
            "backend_contract": self.env.backend.metadata(),
            "backend_version": backend_version,
            "domain_randomization": self.env.domain_randomization_report,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "gpu_properties": gpu_properties,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "dependencies": dependency_versions,
            "solver_dtype": str(self.env.qp.config.solver_dtype),
            "physics_parameter_source": str(
                self.env.cfg.features.physics_parameter_source
            ),
        }
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "hard_pact_metadata.json"), "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, default=str)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self.alg.train_mode()
        obs, critic_obs = self.env.reset()
        obs, history, critic_obs, _ = self.env.get_observations()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )
        obs = obs.to(self.device)
        history = history.to(self.device)
        critic_obs = critic_obs.to(self.device)
        reward_buffer = deque(maxlen=100)
        for iteration in range(
            self.current_learning_iteration,
            self.current_learning_iteration + int(num_learning_iterations),
        ):
            start = time.perf_counter()
            episode_infos = []
            iteration_reward_sum = 0.0
            iteration_reward_count = 0
            for _ in range(self.num_steps_per_env):
                action = self.alg.act(obs, critic_obs, history)
                (
                    next_obs, next_critic, next_history, _, reward, done,
                    infos, _,
                ) = self.env.step(action, physics_estimator=self.actor_critic)
                self.alg.process_env_step(
                    reward, done, infos, self.env.last_transition
                )
                iteration_reward_sum += float(reward.float().sum().item())
                iteration_reward_count += reward.numel()
                obs = next_obs.to(self.device)
                critic_obs = next_critic.to(self.device)
                history = next_history.to(self.device)
                if "episode" in infos:
                    episode_infos.append(infos["episode"])
                    reward_buffer.extend(reward[done.bool()].detach().cpu().tolist())
            self.alg.compute_returns(critic_obs)
            metrics = self.alg.update(
                self.env.recompute_training_outputs,
                self.env.recompute_auxiliary_outputs,
                iteration,
            )
            if episode_infos and self.alg.use_adaptive_entropy:
                performance_metrics = {
                    "lin_vel_tracking": 0.0,
                    "ang_vel_tracking": 0.0,
                    "terrain_level": 0.0,
                }
                episode_keys = {
                    "rew_tracking_lin_vel": "lin_vel_tracking",
                    "rew_tracking_ang_vel": "ang_vel_tracking",
                    "terrain_level": "terrain_level",
                }
                for episode in episode_infos:
                    for episode_key, metric_key in episode_keys.items():
                        if episode_key not in episode:
                            continue
                        value = episode[episode_key]
                        if torch.is_tensor(value):
                            value = value.float().mean().item()
                        performance_metrics[metric_key] = max(
                            performance_metrics[metric_key], float(value)
                        )
                metrics["policy/entropy_coefficient"] = (
                    self.alg.update_adaptive_entropy_coef(performance_metrics)
                )
            elapsed = time.perf_counter() - start
            self.tot_time += elapsed
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            if self.writer:
                for name, value in metrics.items():
                    self.writer.add_scalar(name, value, iteration)
                self.writer.add_scalar("performance/wall_time_iteration", elapsed, iteration)
                self.writer.add_scalar("performance/total_timesteps", self.tot_timesteps, iteration)
                self.writer.add_scalar(
                    "return/mean_step_reward",
                    iteration_reward_sum / max(iteration_reward_count, 1),
                    iteration,
                )
                for episode in episode_infos:
                    for name, value in episode.items():
                        if torch.is_tensor(value):
                            value = value.float().mean().item()
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f"episode/{name}", value, iteration)
                for name, value in self.env.grf_processor.flattened_stages().items():
                    self.writer.add_scalar(f"grf/{name}_magnitude_n", value.norm(dim=-1).mean(), iteration)
                    reshaped = value.reshape(-1, 4, 3)
                    for foot_index, foot_name in enumerate(("FR", "FL", "RR", "RL")):
                        for axis_index, axis in enumerate(("x", "y", "z")):
                            self.writer.add_scalar(
                                f"grf/{name}/{foot_name}_{axis}_mean_n",
                                reshaped[:, foot_index, axis_index].mean(),
                                iteration,
                            )
                self.writer.add_scalar(
                    "grf/contact_fraction",
                    self.env.grf_processor.contacts.float().mean(),
                    iteration,
                )
                self.writer.add_scalar(
                    "disturbance/push_magnitude",
                    self.env.instantaneous_pushes.actual_delta_world.norm(dim=-1).mean(), iteration,
                )
                self.writer.add_scalar(
                    "disturbance/wrench_magnitude",
                    self.env.sustained_wrench.current_world.norm(dim=-1).mean(), iteration,
                )
                transition = self.env.last_transition
                for prefix, field, labels in (
                    (
                        "disturbance/push_world",
                        "instantaneous_push_delta_world",
                        ("dv_x", "dv_y", "dv_z", "domega_roll", "domega_pitch", "domega_yaw"),
                    ),
                    (
                        "disturbance/wrench_world",
                        "sustained_wrench_world",
                        ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"),
                    ),
                ):
                    for index, label in enumerate(labels):
                        self.writer.add_scalar(
                            f"{prefix}/{label}_mean",
                            transition[field][:, index].mean(),
                            iteration,
                        )
                        self.writer.add_scalar(
                            f"{prefix}/{label}_abs_mean",
                            transition[field][:, index].abs().mean(),
                            iteration,
                        )
                for field in (
                    "instantaneous_push_mask", "sustained_wrench_active_mask",
                    "reset_mask", "timeout_mask", "teleport_mask", "physics_valid_mask",
                ):
                    self.writer.add_scalar(
                        f"transition/{field}_fraction",
                        transition[field].float().mean(),
                        iteration,
                    )
                linear_error = (
                    self.env.commands[:, :2]
                    - self.env.simulator.base_lin_vel[:, :2]
                ).norm(dim=-1)
                yaw_error = (
                    self.env.commands[:, 2]
                    - self.env.simulator.base_ang_vel[:, 2]
                ).abs()
                self.writer.add_scalar("tracking/linear_velocity_error", linear_error.mean(), iteration)
                self.writer.add_scalar("tracking/yaw_velocity_error", yaw_error.mean(), iteration)
                self.writer.add_scalar(
                    "success/fraction",
                    ((linear_error < 0.5) & (yaw_error < 0.5) & ~self.env.reset_buf.bool())
                    .float().mean(),
                    iteration,
                )
                terrain_levels = getattr(self.env.simulator, "_terrain_levels", None)
                if terrain_levels is not None:
                    self.writer.add_scalar("terrain/level_mean", terrain_levels.float().mean(), iteration)
                for randomization, report in self.env.domain_randomization_report.items():
                    if not report.get("active") or not report.get("effective_ranges"):
                        continue
                    for range_name, bounds in report["effective_ranges"].items():
                        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                            self.writer.add_scalar(
                                f"domain_rand/{randomization}/{range_name}_min",
                                float(bounds[0]), iteration,
                            )
                            self.writer.add_scalar(
                                f"domain_rand/{randomization}/{range_name}_max",
                                float(bounds[1]), iteration,
                            )
            self.current_learning_iteration = iteration + 1
            save_interval = int(self.cfg["runner"].get("save_interval", 500))
            if self.log_dir and iteration % save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))
        if self.writer:
            self.writer.flush()
        return list(reward_buffer)

    def save(self, path, infos=None):
        torch.save({
            "model_state_dict": self.actor_critic.state_dict(),
            "actor_optimizer": self.alg.actor_optimizer.optimizer.state_dict(),
            "auxiliary_optimizer": self.alg.auxiliary_optimizer.state_dict(),
            "iteration": self.current_learning_iteration,
            "reliability_ema": self.alg.reliability.values,
            "current_entropy_coef": self.alg.current_entropy_coef,
            "infos": infos,
        }, path)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        source_state = checkpoint.get("model_state_dict", checkpoint)
        migrated = False
        if (
            not self.actor_critic.position_pretraining
            and torch.is_tensor(source_state.get("std"))
            and source_state["std"].shape == (12,)
        ):
            self.last_migration_report = migrate_hard_pact_pos_checkpoint(
                self.actor_critic, checkpoint
            )
            migrated = True
        else:
            self.actor_critic.load_state_dict(source_state, strict=True)
        # Position-stage optimizer moments have 12-D distribution tensors and
        # are not part of the documented model-only migration contract.
        if load_optimizer and not migrated:
            if "actor_optimizer" in checkpoint:
                self.alg.actor_optimizer.optimizer.load_state_dict(
                    checkpoint["actor_optimizer"]
                )
                if "auxiliary_optimizer" not in checkpoint:
                    raise KeyError(
                        "two-optimizer HardPACT checkpoint is missing "
                        "auxiliary_optimizer"
                    )
                self.alg.auxiliary_optimizer.load_state_dict(
                    checkpoint["auxiliary_optimizer"]
                )
            elif "optimizer_state_dict" in checkpoint:
                # Documented migration for checkpoints produced before the
                # B1Z1 two-optimizer alignment. Auxiliary moments restart.
                self.alg.actor_optimizer.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"]
                )
        self.current_learning_iteration = int(checkpoint.get("iteration", 0))
        if "current_entropy_coef" in checkpoint:
            self.alg.current_entropy_coef = float(
                checkpoint["current_entropy_coef"]
            )
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        self.actor_critic.eval()
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_inference
