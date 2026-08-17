import os
import math
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms import PPO_UniFP
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticB1UniFP, ActorCriticUniFP
from rsl_rl.utils import pretty_print_module


class OnPolicyRunnerUniFP:
    """Runner for the faithful UniFP CSE/adaptation baseline.

    HCR environments expose the current observation and stacked history
    separately. The original UniFP actor consumes the stacked actor history, so
    this runner stores and trains on `obs_history` as the actor observation.
    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.use_adaptive_entropy = self.alg_cfg.get("use_adaptive_entropy", False)
        self.enable_additional_diagnostics = self.cfg.get(
            "enable_additional_diagnostics", True
        )

        num_single_obs = self.env.num_obs
        num_actor_obs = self.env.num_obs * self.env.num_obs_hist
        num_critic_obs = self.env.num_privileged_obs
        # B1Z1 excludes the terrain-height tail from next-frame decoding.
        # Other UniFP tasks remain backward compatible and reconstruct their
        # complete single privileged frame when this field is not configured.
        self.num_privileged_recon_obs = getattr(
            self.env.cfg.env,
            "num_privileged_recon_obs",
            self.env.num_privileged_obs,
        )
        if getattr(self.env, "num_crit_obs_stack", None) is not None:
            num_critic_obs *= self.env.num_crit_obs_stack

        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCriticUniFP = actor_critic_class(
            num_actor_obs,
            num_critic_obs,
            self.env.num_pred_obs,
            num_single_obs,
            self.env.num_actions,
            num_privileged_obs_single=self.num_privileged_recon_obs,
            enable_additional_diagnostics=self.enable_additional_diagnostics,
            **self.policy_cfg,
        ).to(self.device)

        if self.enable_additional_diagnostics:
            print("Created UniFP Actor-Critic Model")
            pretty_print_module(actor_critic)

        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO_UniFP = alg_class(
            actor_critic,
            device=self.device,
            enable_additional_diagnostics=self.enable_additional_diagnostics,
            **self.alg_cfg,
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.alg.enable_additional_diagnostics = self.enable_additional_diagnostics
        self.alg.actor_critic.enable_additional_diagnostics = self.enable_additional_diagnostics
        self.env.enable_additional_diagnostics = self.enable_additional_diagnostics
        self.enable_deterministic_diagnostics = self.cfg.get(
            "enable_deterministic_diagnostics", False
        ) and self.enable_additional_diagnostics
        self.deterministic_diagnostics_interval = self.cfg.get(
            "deterministic_diagnostics_interval", 100
        )
        if self.deterministic_diagnostics_interval <= 0:
            raise ValueError("deterministic_diagnostics_interval must be positive")

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [self.env.num_pred_obs],
            [self.num_privileged_recon_obs],
            [self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, obs_history, privileged_obs, obs_pred = self.env.get_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs_history = obs_history.to(self.device)
        critic_obs = critic_obs.to(self.device)
        obs_pred = obs_pred.to(self.device)

        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            # The environment receives the checkpoint-aware PPO iteration, not
            # an approximation based on simulation steps or episode resets.
            if hasattr(self.env, "set_training_iteration"):
                self.env.set_training_iteration(it)
            if self.enable_additional_diagnostics:
                self.alg.actor_critic.begin_rollout_diagnostics()
            if self.enable_additional_diagnostics and hasattr(self.env, "begin_rollout_diagnostics"):
                self.env.begin_rollout_diagnostics()
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_history, critic_obs, obs_pred)
                    if self.enable_additional_diagnostics:
                        self.alg.actor_critic.record_rollout_diagnostics(
                            actions, self.env.cfg.normalization.clip_actions
                        )
                    obs, privileged_obs, obs_history, obs_pred, rewards, dones, infos, _ = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs

                    obs_history = obs_history.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    obs_pred = obs_pred.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # The final stack slot is the post-action, single-frame
                    # privileged target for one-step latent reconstruction.
                    next_privileged = critic_obs[:, -self.env.num_privileged_obs:][
                        :, :self.num_privileged_recon_obs
                    ]
                    self.alg.process_env_step(rewards, dones, infos, next_privileged)

                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    if self.log_dir is not None:
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if len(new_ids):
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                collection_time = time.time() - start
                policy_diagnostics = (
                    self.alg.actor_critic.get_rollout_diagnostics()
                    if self.enable_additional_diagnostics else {}
                )
                environment_diagnostics = (
                    self.env.get_rollout_diagnostics()
                    if self.enable_additional_diagnostics and hasattr(self.env, "get_rollout_diagnostics") else {}
                )
                start = time.time()
                self.alg.compute_returns(critic_obs)

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_adaptation_module_loss,
                mean_adaptation_losses,
                ppo_diagnostics,
            ) = self.alg.update(it)
            latent_resample_diagnostics = {}
            deterministic_diagnostics = {}
            joint_std_diagnostics = {}
            if self.enable_additional_diagnostics:
                diagnostic_observations = obs_history[:min(32, obs_history.shape[0])]
                if (
                    self.enable_deterministic_diagnostics
                    and it % self.deterministic_diagnostics_interval == 0
                ):
                    deterministic_diagnostics = self.alg.actor_critic.deterministic_diagnostics(
                        diagnostic_observations
                    )
                else:
                    latent_resample_diagnostics = self.alg.actor_critic.latent_resample_diagnostics(
                        diagnostic_observations
                    )
                joint_names = self.env.cfg.asset.dof_names[:self.env.num_actions]
                joint_std_diagnostics = {
                    f"PolicyStd/{joint_name}": self.alg.actor_critic.std[index].detach().item()
                    for index, joint_name in enumerate(joint_names)
                }
            diagnostic_metrics = {
                **ppo_diagnostics,
                **policy_diagnostics,
                **environment_diagnostics,
                **latent_resample_diagnostics,
                **deterministic_diagnostics,
                **joint_std_diagnostics,
            }
            if self.enable_additional_diagnostics:
                self._validate_diagnostics(diagnostic_metrics)
            learn_time = time.time() - start

            if ep_infos and self.use_adaptive_entropy:
                performance_metrics = {
                    "lin_vel_tracking": self._episode_metric_max(
                        ep_infos, "rew_tracking_lin_vel_force_world"
                    ),
                    "ang_vel_tracking": self._episode_metric_max(
                        ep_infos, "rew_tracking_ang_vel"
                    ),
                    # B1Z1 emits ``terrain_level`` while older UniFP tasks may
                    # still use ``terrain_level_mean``.
                    "terrain_level": self._episode_metric_max(
                        ep_infos, "terrain_level", "terrain_level_mean"
                    ),
                }
                self.alg.update_adaptive_entropy_coef(performance_metrics)

            if getattr(self.env, "use_reward_curriculum", False):
                self.env.step_reward_curriculum(it)

            if getattr(self.env.simulator, "use_domainrand_curriculum", False):
                tracking = None
                if ep_infos and "rew_tracking_lin_vel_force_world" in ep_infos[0]:
                    vals = [info["rew_tracking_lin_vel_force_world"].float().mean().to(self.device) for info in ep_infos]
                    tracking = torch.stack(vals).mean().item()
                if tracking is not None:
                    self.env.simulator._step_domian_rand(it, tracking)
                if self.writer is not None:
                    reward_ema = self.env.simulator.domain_rand_reward_ema
                    self.writer.add_scalar(
                        "Values/domain_rand_reward_ema",
                        reward_ema if reward_ema is not None else 0.0,
                        it,
                    )
                    self.writer.add_scalar("Values/required_reward", self.env.simulator.required_reward, it)
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

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0 and self.log_dir is not None:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"), iteration=it)
            ep_infos.clear()

        self.current_learning_iteration = tot_iter
        if hasattr(self.env, "set_training_iteration"):
            self.env.set_training_iteration(self.current_learning_iteration)
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    value = ep_info[key]
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor([value], device=self.device)
                    if len(value.shape) == 0:
                        value = value.unsqueeze(0)
                    infotensor = torch.cat((infotensor, value.to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/adaptation", locs["mean_adaptation_module_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        for key, value in locs["mean_adaptation_losses"].items():
            self.writer.add_scalar("Loss/adaptation_" + key, value, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Values/entropy", self.alg.current_entropy_coef, locs["it"])
        for tag, value in locs["diagnostic_metrics"].items():
            self.writer.add_scalar(tag, value, locs["it"])
        if hasattr(self.env, "get_gait_guidance_multipliers"):
            for name, multiplier in self.env.get_gait_guidance_multipliers().items():
                self.writer.add_scalar(
                    f"Values/gait_guidance_{name}_multiplier",
                    multiplier,
                    locs["it"],
                )
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        header = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        log_string = (
            f"{'#' * width}\n"
            f"{header.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
            f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
            f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
            f"{'Adaptation loss:':>{pad}} {locs['mean_adaptation_module_loss']:.4f}\n"
        )
        for key, value in locs["mean_adaptation_losses"].items():
            log_string += f"{f'Adaptation {key} loss:':>{pad}} {value:.4f}\n"
        log_string += f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
        log_string += f"{'Entropy coefficient:':>{pad}} {self.alg.current_entropy_coef:.6f}\n"
        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        log_string += ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"
        )
        print(log_string)

    @staticmethod
    def _episode_metric_max(ep_infos, *keys):
        """Return the largest scalar episode metric, or zero if unavailable."""
        values = []
        for ep_info in ep_infos:
            key = next((candidate for candidate in keys if candidate in ep_info), None)
            if key is None:
                continue
            value = ep_info[key]
            if isinstance(value, torch.Tensor):
                value = value.float().mean().item()
            values.append(float(value))
        return max(values, default=0.0)

    @staticmethod
    def _validate_diagnostics(metrics):
        """Fail early if a diagnostic is nonfinite or a fraction leaves [0, 1]."""
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

    def save(self, path, infos=None, iteration=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration if iteration is None else iteration,
                "entropy_coef": self.alg.current_entropy_coef,
                "kl_controller_state": self.alg.kl_controller.state_dict(),
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        self.alg.current_entropy_coef = loaded_dict.get(
            "entropy_coef", self.alg.current_entropy_coef
        )
        self.alg.kl_controller.load_state_dict(loaded_dict.get("kl_controller_state"))
        if hasattr(self.env, "set_training_iteration"):
            self.env.set_training_iteration(self.current_learning_iteration)
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
