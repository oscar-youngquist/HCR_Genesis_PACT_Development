"""Standard PPO with PACT optimizer groups and adaptive-entropy convention."""
import inspect
import torch
from .ppo import PPO
from .ppo_b1z1_pact import PPO_B1Z1PACT


class PPO_B1Z1PPOPos(PPO):
    def __init__(self, actor_critic, cfg, device):
        accepted = inspect.signature(PPO.__init__).parameters
        super().__init__(actor_critic, device=device,
                         **{k: v for k, v in cfg.items() if k in accepted})
        groups, unused = actor_critic.get_optim_groups()
        assert not unused
        self.optimizer = torch.optim.AdamW(groups, lr=self.learning_rate)
        self.use_adaptive_entropy = cfg["use_adaptive_entropy"]
        self.entropy_coef_bounds = tuple(cfg["adaptive_ent_bounds"])
        self.ent_linvelo_threshold = cfg["adaptive_ent_lin_threshold"]
        self.ent_angvelo_threshold = cfg["adaptive_ent_ang_threshold"]
        self.ent_terrain_threshold = cfg["adaptive_ent_ter_threshold"]
        self.ent_softmax_temperature = cfg["adaptive_ent_softmax_temp"]
        self.current_entropy_coef = self.entropy_coef

    def update_adaptive_entropy_coef(self, metrics):
        value = PPO_B1Z1PACT.update_adaptive_entropy_coef(self, metrics)
        self.entropy_coef = value
        return value

    def act(self, obs, critic_obs):
        # Environment buffers can be overwritten during reset/observation assembly.
        return super().act(obs.detach().clone(), critic_obs.detach().clone())
