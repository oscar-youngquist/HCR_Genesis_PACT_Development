import os
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

        num_single_obs = self.env.num_obs
        num_actor_obs = self.env.num_obs * self.env.num_obs_hist
        num_critic_obs = self.env.num_privileged_obs
        if getattr(self.env, "num_crit_obs_stack", None) is not None:
            num_critic_obs *= self.env.num_crit_obs_stack

        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCriticUniFP = actor_critic_class(
            num_actor_obs,
            num_critic_obs,
            self.env.num_pred_obs,
            num_single_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        print("Created UniFP Actor-Critic Model")
        pretty_print_module(actor_critic)

        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO_UniFP = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [self.env.num_pred_obs],
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
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_history, critic_obs, obs_pred)
                    obs, privileged_obs, obs_history, obs_pred, rewards, dones, infos, _ = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs

                    obs_history = obs_history.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    obs_pred = obs_pred.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_adaptation_losses = self.alg.update()
            learn_time = time.time() - start

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
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
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
        for key, value in locs["mean_adaptation_losses"].items():
            self.writer.add_scalar("Loss/adaptation_" + key, value, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
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

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
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
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
