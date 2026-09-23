"""Thin standard-PPO adapter retaining PACT's environment and curricula."""
import os
import statistics
import time
from collections import deque
import torch
from .on_policy_runner import OnPolicyRunner
from .b1z1_pact_runner import B1Z1PACTRunner
from rsl_rl.modules.actor_critic_b1z1_ppo_pos import ActorCriticB1Z1PPOPos
from rsl_rl.algorithms.ppo_b1z1_ppo_pos import PPO_B1Z1PPOPos
from rsl_rl.utils.simulator_diagnostics import domain_rand_state, load_domain_rand_state
from legged_gym.envs.b1z1.force_task_utils import (
    update_force_curriculum_from_rollout, staged_force_curriculum_state_dict,
    load_staged_force_curriculum_state_dict,
)


class B1Z1PPOPosRunner(OnPolicyRunner):
    _episode_metric_max = staticmethod(B1Z1PACTRunner._episode_metric_max)
    _step_domain_randomization_curriculum = B1Z1PACTRunner._step_domain_randomization_curriculum

    def _init_agent_and_algo(self):
        # Preserve PACT's privileged mass/wrench labels without a dynamics model.
        self.env.bard_mass_wrench_labels = self.alg_cfg["dynamics_backend"] == "bard"
        critic_dim = self.env.num_privileged_obs * self.env.num_crit_obs_stack
        self.actor_critic = ActorCriticB1Z1PPOPos(
            self.env.num_obs, critic_dim, self.env.num_actions, **self.policy_cfg).to(self.device)
        self.alg = PPO_B1Z1PPOPos(self.actor_critic, self.alg_cfg, self.device)

    def _init_storage(self):
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env,
                              [self.env.num_obs],
                              [self.env.num_privileged_obs * self.env.num_crit_obs_stack],
                              [self.env.num_actions])

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._pre_learn(init_at_random_ep_len)
        obs, _, critic_obs, _ = self.env.get_observations()
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.actor_critic.train()
        rewbuffer, lenbuffer = deque(maxlen=100), deque(maxlen=100)
        reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        episode_length = torch.zeros_like(reward_sum)
        final_iteration = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, final_iteration):
            self.env.set_training_iteration(it)
            ep_infos = []
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, critic_obs, _, _, rewards, dones, infos, _ = self.env.step(actions)
                    obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    reward_sum += rewards
                    episode_length += 1
                    ids = dones.nonzero(as_tuple=False).flatten()
                    rewbuffer.extend(reward_sum[ids].tolist())
                    lenbuffer.extend(episode_length[ids].tolist())
                    reward_sum[ids] = 0
                    episode_length[ids] = 0
                self.alg.compute_returns(critic_obs)
            collection_time = time.time() - start
            start = time.time()
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            learn_time = time.time() - start
            if ep_infos and self.alg.use_adaptive_entropy:
                self.alg.update_adaptive_entropy_coef({
                    "lin_vel_tracking": self._episode_metric_max(ep_infos, "rew_tracking_lin_vel_force_world"),
                    "ang_vel_tracking": self._episode_metric_max(ep_infos, "rew_tracking_ang_vel"),
                    "terrain_level": self._episode_metric_max(ep_infos, "terrain_level", "terrain_level_mean"),
                })
            if getattr(self.env, "use_reward_curriculum", False):
                self.env.step_reward_curriculum(it)
            self._step_domain_randomization_curriculum(it, ep_infos)
            metrics = update_force_curriculum_from_rollout(
                self.env, it, ep_infos, statistics.mean(lenbuffer) if lenbuffer else None)
            if self.writer:
                self.log(locals())
                self.writer.add_scalar("Values/entropy", self.alg.current_entropy_coef, it)
                for name, value in metrics.items():
                    self.writer.add_scalar(name, value, it)
            if self.log_dir and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"), iteration=it + 1)
        self.current_learning_iteration = final_iteration
        self.env.set_training_iteration(final_iteration)
        if self.log_dir:
            self.save(os.path.join(self.log_dir, f"model_{final_iteration}.pt"))

    def save(self, path, infos=None, iteration=None):
        torch.save({
            "architecture": "b1z1_ppo_pos",
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration if iteration is None else iteration,
            "entropy_coef": self.alg.current_entropy_coef,
            "force_curriculum_state": staged_force_curriculum_state_dict(self.env),
            "domain_rand_curriculum_state": domain_rand_state(self.env.simulator),
            "infos": infos,
        }, path)

    def load(self, path, load_optimizer=True):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("architecture") != "b1z1_ppo_pos":
            raise ValueError("Expected a b1z1_ppo_pos checkpoint")
        self.actor_critic.load_state_dict(state["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        self.current_learning_iteration = state["iter"]
        self.alg.current_entropy_coef = self.alg.entropy_coef = state["entropy_coef"]
        load_domain_rand_state(self.env.simulator, state.get("domain_rand_curriculum_state"))
        load_staged_force_curriculum_state_dict(self.env, state.get("force_curriculum_state"))
        self.env.set_training_iteration(self.current_learning_iteration)
        return state.get("infos")
