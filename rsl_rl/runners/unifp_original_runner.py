"""Reuse local collection/logging with a deterministic upstream model and objective."""
import torch
from .unifp_runner import OnPolicyRunnerUniFP
from rsl_rl.modules.actor_critic_unifp_original import ActorCriticUniFPOriginal
from rsl_rl.algorithms.ppo_unifp_original import PPO_UniFPOriginal
from rsl_rl.utils.simulator_diagnostics import domain_rand_state, load_domain_rand_state
from legged_gym.envs.b1z1.force_task_utils import (
    staged_force_curriculum_state_dict, load_staged_force_curriculum_state_dict,
)


class OnPolicyRunnerUniFPOriginal(OnPolicyRunnerUniFP):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.device, self.log_dir = env, device, log_dir
        self.cfg, self.alg_cfg = train_cfg["runner"], train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.use_adaptive_entropy = self.alg_cfg.get("use_adaptive_entropy", False)
        self.enable_additional_diagnostics = False
        self.enable_deterministic_diagnostics = False
        env.enable_additional_diagnostics = False
        self.num_privileged_recon_obs = 0
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        history_dim = env.num_obs * env.num_obs_hist
        critic_dim = env.num_privileged_obs * env.num_crit_obs_stack
        model = ActorCriticUniFPOriginal(history_dim, critic_dim, env.num_pred_obs,
                                        env.num_obs, env.num_actions).to(device)
        self.alg = PPO_UniFPOriginal(model, device=device,
                                    decision_callback=self._publish_estimates, **self.alg_cfg)
        self.alg.init_storage(env.num_envs, self.num_steps_per_env, [history_dim],
                              [critic_dim], [12], [0], [17])
        self.writer = None
        self.tot_timesteps = self.tot_time = self.current_learning_iteration = 0
        self._startup_metadata_logged = False
        env.reset()

    def _publish_estimates(self, prediction):
        if self.env.reject_external_forces:
            if self.alg.actor_critic.training:
                self.env.record_force_prediction_quality(prediction)
            self.env.set_impedance_force_estimates(prediction)

    def get_inference_policy(self, device=None):
        model = self.alg.actor_critic.eval()
        if device is not None:
            model.to(device)

        @torch.no_grad()
        def policy(observations, policy_info=None):
            actions = model.act_inference(observations, policy_info)
            self._publish_estimates(model.last_prediction.detach())
            return actions
        return policy

    def save(self, path, infos=None, iteration=None):
        torch.save({
            "architecture": self.alg.actor_critic.architecture,
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "adaptation_optimizer_state_dict": self.alg.adaptation_module_optimizer.state_dict(),
            "iter": self.current_learning_iteration if iteration is None else iteration,
            "entropy_coef": self.alg.current_entropy_coef,
            "domain_rand_curriculum_state": domain_rand_state(self.env.simulator),
            "force_curriculum_state": staged_force_curriculum_state_dict(self.env)
                if hasattr(self.env, "_staged_force_curriculum") else None,
            "infos": infos,
        }, path)

    def load(self, path, load_optimizer=True):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("architecture") != self.alg.actor_critic.architecture:
            raise RuntimeError("Incompatible architecture/schema: expected a new deterministic UniFP baseline checkpoint")
        self.alg.actor_critic.load_state_dict(state["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.alg.adaptation_module_optimizer.load_state_dict(state["adaptation_optimizer_state_dict"])
        self.current_learning_iteration = state["iter"]
        self.alg.current_entropy_coef = state.get("entropy_coef", self.alg.current_entropy_coef)
        load_domain_rand_state(self.env.simulator, state.get("domain_rand_curriculum_state"))
        self.env.set_training_iteration(self.current_learning_iteration)
        if hasattr(self.env, "_staged_force_curriculum"):
            load_staged_force_curriculum_state_dict(self.env, state.get("force_curriculum_state"))
        return state.get("infos")
