"""On-policy runner for the canonical Go2 HardPACT transition contract."""

from __future__ import annotations

import json
import importlib
import importlib.metadata
import os
import platform
import statistics
import subprocess
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from legged_gym.envs.go2.go2_hard_pact.disturbances import (
    torso_wrench_scale_from_ranges,
)
from rsl_rl.algorithms import PPOGo2HardPACT
from rsl_rl.modules import (
    ActorCriticGo2HardPACT,
    ActorCriticGo2HardPACTPos,
    migrate_hard_pact_pos_checkpoint,
)


# Match the legacy Go2 PACT CUDA settings.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


class Go2HardPACTRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.cfg, self.device, self.log_dir = (
            env, train_cfg, torch.device(device), log_dir
        )

        policy_cfg = dict(train_cfg["policy"])
        algorithm_cfg = dict(train_cfg["algorithm"])
        runner_cfg = train_cfg["runner"]

        critic_dim = env.num_privileged_obs
        if env.num_crit_obs_stack is not None:
            critic_dim *= env.num_crit_obs_stack
        history_dim = env.num_obs * env.num_obs_hist

        wrench_cfg = env.cfg.disturbances.sustained_wrench
        mass_report = env.domain_randomization_report["base_mass"]
        com_report = env.domain_randomization_report["base_com"]
        effective_mass_ranges = mass_report.get("effective_ranges") or {}
        added_mass_range = effective_mass_ranges.get(
            "added_mass_range", (0.0, 0.0)
        )
        # Cover the complete configured curriculum envelope as well as the
        # currently effective reset-time range; the head scale must remain
        # stable while Genesis expands its randomization range.
        if mass_report["active"]:
            added_mass_candidates = [float(value) for value in added_mass_range]
            for name in ("added_mass_min", "max_added_mass_max"):
                if hasattr(env.cfg.domain_rand, name):
                    added_mass_candidates.append(
                        float(getattr(env.cfg.domain_rand, name))
                    )
            added_mass_range = (
                min(added_mass_candidates), max(added_mass_candidates)
            )
        effective_com_ranges = com_report.get("effective_ranges") or {}
        com_offset_ranges = (
            effective_com_ranges.get("com_pos_x_range", (0.0, 0.0)),
            effective_com_ranges.get("com_pos_y_range", (0.0, 0.0)),
            effective_com_ranges.get("com_pos_z_range", (0.0, 0.0)),
        )
        gravity_world = (
            (0.0, 0.0, 0.0)
            if bool(getattr(env.cfg.asset, "disable_gravity", False))
            else getattr(env.cfg.sim, "gravity", (0.0, 0.0, -9.81))
        )
        grf_observation_scale = float(env.obs_scales.grf)
        base_wrench_observation_scale = float(env.obs_scales.base_wrench)
        if grf_observation_scale <= 0.0 or base_wrench_observation_scale <= 0.0:
            raise ValueError("HardPACT force observation scales must be positive")
        physical_grf_scale = tuple(float(value) for value in env.cfg.sim.grf.prediction_scale_n) * 4
        physical_wrench_scale = torso_wrench_scale_from_ranges(
            wrench_cfg.force_bounds_n,
            wrench_cfg.torque_bounds_nm,
            added_mass_range,
            gravity_world,
            com_offset_ranges=com_offset_ranges,
            include_sustained_force=(
                bool(wrench_cfg.enabled)
                and float(wrench_cfg.force_probability) > 0.0
            ),
            include_sustained_torque=(
                bool(wrench_cfg.enabled)
                and float(wrench_cfg.torque_probability) > 0.0
            ),
            include_added_mass=bool(mass_report["active"]),
        )
        policy_cfg["grf_scale"] = tuple(
            value * grf_observation_scale for value in physical_grf_scale
        )
        policy_cfg["wrench_scale"] = tuple(
            value * base_wrench_observation_scale
            for value in physical_wrench_scale
        )
        policy_class = eval(runner_cfg["policy_class_name"])
        self.actor_critic = policy_class(
            num_actor_obs=env.num_obs,
            num_critic_obs=critic_dim,
            num_actions=env.num_actions,
            **policy_cfg,
        )
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
            "grf_observation_scale": grf_observation_scale,
            "base_wrench_observation_scale": base_wrench_observation_scale,
            "grf_normalization_scale": tuple(
                self.actor_critic.physics_estimator.grf_scale.tolist()
            ),
            "wrench_normalization_scale": tuple(
                self.actor_critic.physics_estimator.wrench_scale.tolist()
            ),
            "feedforward_clone_weight": float(
                features.feedforward_clone_weight
            ),
        })
        algorithm_class = eval(runner_cfg["algorithm_class_name"])
        self.alg = algorithm_class(
            self.actor_critic, device=self.device, **algorithm_cfg
        )
        self.alg.init_storage(
            env.num_envs,
            runner_cfg["num_steps_per_env"],
            env.num_obs,
            critic_dim,
            history_dim,
        )

        self.num_steps_per_env = runner_cfg["num_steps_per_env"]
        self.save_interval = runner_cfg["save_interval"]
        self.use_adaptive_entropy = algorithm_cfg.get(
            "use_adaptive_entropy", False
        )

        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.writer = None
        self.last_migration_report = None
        self.deployment_contract = self._deployment_contract()
        if log_dir:
            self._write_metadata()

        _, _ = self.env.reset()

    def _deployment_contract(self):
        """Return the exact force-unit contract carried by this policy."""
        grf_observation_scale = float(self.env.obs_scales.grf)
        wrench_observation_scale = float(self.env.obs_scales.base_wrench)
        grf_scaled = (
            self.actor_critic.physics_estimator.grf_scale.detach().cpu().tolist()
        )
        wrench_scaled = (
            self.actor_critic.physics_estimator.wrench_scale.detach().cpu().tolist()
        )
        return {
            "version": 1,
            "force_frame": "yaw_local",
            "grf_order": [
                "FR_fx", "FR_fy", "FR_fz",
                "FL_fx", "FL_fy", "FL_fz",
                "RR_fx", "RR_fy", "RR_fz",
                "RL_fx", "RL_fy", "RL_fz",
            ],
            "base_wrench_order": ["fx", "fy", "fz", "tx", "ty", "tz"],
            "observation_scales": {
                "grf": grf_observation_scale,
                "base_wrench": wrench_observation_scale,
            },
            "head_output_scales_observation_units": {
                "grf": grf_scaled,
                "base_wrench": wrench_scaled,
            },
            "head_output_scales_physical_units": {
                "grf_n": [
                    value / grf_observation_scale for value in grf_scaled
                ],
                "base_wrench_n_nm": [
                    value / wrench_observation_scale
                    for value in wrench_scaled
                ],
            },
            "conversion": {
                "model_to_physical": "prediction_scaled / observation_scale",
                "physical_to_model": "value_physical * observation_scale",
            },
            "checkpoint_buffer_keys": {
                "grf": "physics_estimator.grf_scale",
                "base_wrench": "physics_estimator.wrench_scale",
            },
        }

    def _write_deployment_contract(self):
        self.deployment_contract = self._deployment_contract()
        path = os.path.join(
            self.log_dir, "hard_pact_deployment_contract.json"
        )
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(self.deployment_contract, stream, indent=2)

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
        backend_name = self.env.backend_capabilities.name
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
            "backend_contract": self.env.backend_metadata(),
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
            "torso_wrench_scale": (
                self.actor_critic.physics_estimator.wrench_scale.tolist()
            ),
            "grf_prediction_scale": (
                self.actor_critic.physics_estimator.grf_scale.tolist()
            ),
            "force_observation_scales": {
                "grf": float(self.env.obs_scales.grf),
                "base_wrench": float(self.env.obs_scales.base_wrench),
            },
            "physics_parameter_source": str(
                self.env.cfg.features.physics_parameter_source
            ),
            "deployment_contract_file": "hard_pact_deployment_contract.json",
        }
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "hard_pact_metadata.json"), "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, default=str)
        self._write_deployment_contract()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # Initialize the writer lazily, as in the legacy PACT runners.
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )
        obs, history, privileged_obs, _ = self.env.get_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, history = (
            obs.to(self.device), critic_obs.to(self.device),
            history.to(self.device),
        )
        self.alg.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros_like(cur_reward_sum)

        tot_iter = self.current_learning_iteration + int(num_learning_iterations)
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout collection follows legacy PACT. HardPACT's only extra
            # step input is the deployment-available force estimator used once
            # before the QP; the resulting named transition is stored directly.
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, history)
                    (
                        obs, privileged_obs, history, _, rewards, dones,
                        infos, _,
                    ) = self.env.step(
                        actions, physics_estimator=self.actor_critic
                    )
                    critic_obs = (
                        privileged_obs
                        if privileged_obs is not None else obs
                    )
                    obs, critic_obs, history, rewards, dones = (
                        obs.to(self.device), critic_obs.to(self.device),
                        history.to(self.device), rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(
                        rewards, dones, infos, self.env.last_transition
                    )

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(
                            as_tuple=False
                        ).view(-1)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids].cpu().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids].cpu().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start
                self.alg.compute_returns(critic_obs)

            start = time.time()
            # The PPO implementation recomputes the delayed-action-conditioned
            # force heads and differentiable QP from the stored sampled action.
            metrics = self.alg.update(
                self.env.recompute_training_outputs,
                self.env.recompute_auxiliary_outputs,
                it,
            )
            learn_time = time.time() - start

            if getattr(self.env, "use_reward_curriculum", False):
                self.env.step_reward_curriculum(it)
            self._step_domain_randomization_curriculum(it, ep_infos)

            if ep_infos and self.use_adaptive_entropy:
                metrics["policy/entropy_coefficient"] = (
                    self.alg.update_adaptive_entropy_coef({
                        "lin_vel_tracking": self._episode_metric_max(
                            ep_infos, "rew_tracking_lin_vel",
                            "rew_tracking_lin_vel_force_world",
                        ),
                        "ang_vel_tracking": self._episode_metric_max(
                            ep_infos, "rew_tracking_ang_vel"
                        ),
                        "terrain_level": self._episode_metric_max(
                            ep_infos, "terrain_level", "terrain_level_mean"
                        ),
                    })
                )

            metrics.update(self._hard_pact_rollout_metrics())
            if self.log_dir is not None:
                self.log(locals())
            if self.log_dir and it % self.save_interval == 0:
                self.save(
                    os.path.join(self.log_dir, f"model_{it}.pt"),
                    iteration=it + 1,
                )
            ep_infos.clear()

        self.current_learning_iteration = tot_iter
        if self.log_dir:
            self.save(os.path.join(self.log_dir, f"model_{tot_iter}.pt"))
        if self.writer:
            self.writer.flush()
        return list(rewbuffer)

    @staticmethod
    def _episode_metric_max(ep_infos, *names):
        values = []
        for info in ep_infos:
            name = next((name for name in names if name in info), None)
            if name is not None:
                values.append(
                    torch.as_tensor(info[name]).float().mean().item()
                )
        return max(values, default=0.0)

    def _step_domain_randomization_curriculum(self, iteration, ep_infos):
        report = self.env.domain_randomization_report[
            "domain_rand_curriculum"
        ]
        simulator = self.env.simulator
        if not report["active"] or not getattr(
            simulator, "use_domainrand_curriculum", False
        ):
            return
        tracking = None
        if ep_infos:
            tracking = self._episode_metric_max(
                ep_infos, "rew_tracking_lin_vel",
                "rew_tracking_lin_vel_force_world",
            )
        simulator._step_domian_rand(iteration, tracking)

    def _hard_pact_rollout_metrics(self):
        """Return only diagnostics specific to the HardPACT transition."""
        metrics = {}
        for name, value in self.env.grf_processor.flattened_stages().items():
            metrics[f"grf/{name}_magnitude_n"] = value.norm(dim=-1).mean()
            reshaped = value.reshape(-1, 4, 3)
            for foot_index, foot_name in enumerate(("FR", "FL", "RR", "RL")):
                for axis_index, axis in enumerate(("x", "y", "z")):
                    metrics[f"grf/{name}/{foot_name}_{axis}_mean_n"] = (
                        reshaped[:, foot_index, axis_index].mean()
                    )
        metrics["grf/contact_fraction"] = (
            self.env.grf_processor.contacts.float().mean()
        )
        metrics["disturbance/push_magnitude"] = (
            self.env.instantaneous_pushes.actual_delta_world.norm(
                dim=-1
            ).mean()
        )
        metrics["disturbance/wrench_magnitude"] = (
            self.env.sustained_wrench.current_world.norm(dim=-1).mean()
        )

        transition = self.env.last_transition
        for prefix, field, labels in (
            (
                "disturbance/push_world",
                "instantaneous_push_delta_world",
                (
                    "dv_x", "dv_y", "dv_z", "domega_roll",
                    "domega_pitch", "domega_yaw",
                ),
            ),
            (
                "disturbance/wrench_world",
                "sustained_wrench_world",
                (
                    "force_x", "force_y", "force_z", "torque_x",
                    "torque_y", "torque_z",
                ),
            ),
            (
                "disturbance/added_mass_wrench_world",
                "added_mass_wrench_world",
                (
                    "force_x", "force_y", "force_z", "torque_x",
                    "torque_y", "torque_z",
                ),
            ),
        ):
            for index, label in enumerate(labels):
                metrics[f"{prefix}/{label}_mean"] = (
                    transition[field][:, index].mean()
                )
                metrics[f"{prefix}/{label}_abs_mean"] = (
                    transition[field][:, index].abs().mean()
                )
        for field in (
            "instantaneous_push_mask", "sustained_wrench_active_mask",
            "reset_mask", "timeout_mask", "teleport_mask",
            "physics_valid_mask",
        ):
            metrics[f"transition/{field}_fraction"] = (
                transition[field].float().mean()
            )
        terrain_levels = getattr(self.env.simulator, "_terrain_levels", None)
        if terrain_levels is None:
            terrain_levels = getattr(self.env.simulator, "terrain_levels", None)
        if terrain_levels is not None:
            metrics["terrain/level_mean"] = terrain_levels.float().mean()
        for randomization, report in self.env.domain_randomization_report.items():
            if not report.get("active") or not report.get("effective_ranges"):
                continue
            for range_name, bounds in report["effective_ranges"].items():
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    metrics[f"domain_rand/{randomization}/{range_name}_min"] = (
                        float(bounds[0])
                    )
                    metrics[f"domain_rand/{randomization}/{range_name}_max"] = (
                        float(bounds[1])
                    )
        return metrics

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        iteration_time = locs["collection_time"] + locs["learn_time"]
        self.tot_time += iteration_time
        fps = int(
            self.num_steps_per_env * self.env.num_envs
            / max(iteration_time, 1.0e-6)
        )
        metrics = locs["metrics"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                values = [
                    torch.as_tensor(info[key], device=self.device).float().mean()
                    for info in locs["ep_infos"] if key in info
                ]
                if values:
                    value = torch.stack(values).mean()
                    self.writer.add_scalar(f"Episode/{key}", value, locs["it"])
                    ep_string += (
                        f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"
                    )

        mean_std = self.alg.actor_critic.std.mean()
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, locs["it"])
        self.writer.add_scalar(
            "Loss/learning_rate", self.alg.learning_rate, locs["it"]
        )
        self.writer.add_scalar(
            "Policy/mean_noise_std", mean_std.item(), locs["it"]
        )
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar(
            "Perf/learning_time", locs["learn_time"], locs["it"]
        )
        if locs["rewbuffer"]:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]), locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]), self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]), self.tot_time,
            )

        header = (
            f" \033[1m Learning iteration {locs['it']}/"
            f"{locs['tot_iter']}"
            " \033[0m "
        )
        lines = [
            "#" * width,
            header.center(width),
            "",
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {locs['collection_time']:.3f}s, "
            f"learning {locs['learn_time']:.3f}s)",
            f"{'Physics loss:':>{pad}} {metrics['loss/physics']:.4f}",
            f"{'Inverse-dynamics PINN:':>{pad}} {metrics['loss/inverse']:.4f}",
            f"{'BARD rollout PINN:':>{pad}} {metrics['loss/rollout']:.4f}",
            f"{'QP projection loss:':>{pad}} {metrics['loss/projection']:.4f}",
            f"{'Value function loss:':>{pad}} {metrics['loss/value']:.4f}",
            f"{'Surrogate loss:':>{pad}} {metrics['loss/surrogate']:.4f}",
            f"{'Auxiliary loss:':>{pad}} {metrics['loss/auxiliary']:.4f}",
            f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}",
            f"{'Entropy coefficient:':>{pad}} "
            f"{self.alg.current_entropy_coef:.6f}",
        ]
        if locs["rewbuffer"]:
            lines.extend((
                f"{'Mean reward:':>{pad}} "
                f"{statistics.mean(locs['rewbuffer']):.2f}",
                f"{'Mean episode length:':>{pad}} "
                f"{statistics.mean(locs['lenbuffer']):.2f}",
            ))
        if ep_string:
            lines.append(ep_string.rstrip())
        eta = (
            self.tot_time / max(locs["it"] + 1, 1)
            * max(locs["tot_iter"] - locs["it"], 0)
        )
        lines.extend((
            "-" * width,
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}",
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s",
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s",
            f"{'ETA:':>{pad}} {eta:.1f}s",
        ))
        print("\n".join(lines))

    def save(self, path, iteration=None, infos=None):
        saved_iteration = (
            self.current_learning_iteration
            if iteration is None else int(iteration)
        )
        torch.save({
            "model_state_dict": self.actor_critic.state_dict(),
            "actor_optimizer": self.alg.actor_optimizer.optimizer.state_dict(),
            "auxiliary_optimizer": self.alg.auxiliary_optimizer.state_dict(),
            "iteration": saved_iteration,
            "reliability_ema": self.alg.reliability.values,
            "reliability_iteration": self.alg.reliability.last_iteration,
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
        # The checkpoint model buffers are authoritative. Keep PPO's loss
        # normalizers synchronized when training is resumed.
        self.alg.grf_normalization_scale = tuple(
            self.actor_critic.physics_estimator.grf_scale.detach().cpu().tolist()
        )
        self.alg.wrench_normalization_scale = tuple(
            self.actor_critic.physics_estimator.wrench_scale.detach().cpu().tolist()
        )
        self.deployment_contract = self._deployment_contract()
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
        self.alg.reliability.values.update(
            checkpoint.get("reliability_ema", {})
        )
        self.alg.reliability.last_iteration = int(
            checkpoint.get(
                "reliability_iteration",
                self.current_learning_iteration - 1,
            )
        )
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
