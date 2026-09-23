"""Position actions through the inherited PACT PD branch, with zero feedforward."""
from copy import copy
import torch
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
from legged_gym.torque_action_scaling import simulator_torque_action_scale


class B1Z1PPOPos(B1Z1PACT):
    def __init__(self, cfg, *args, **kwargs):
        if cfg.env.num_policy_actions != cfg.env.num_actions:
            raise ValueError("PPO-Pos requires one position action per learned joint")
        super().__init__(cfg, *args, **kwargs)
        self.simulator.collect_b1z1_bard_interval = False

    @property
    def position_history_slice(self):
        return slice(0, self.num_actions)

    @property
    def torque_history_slice(self):
        return slice(self.num_actions, 2 * self.num_actions)

    def _init_buffers(self):
        # PACT observation/reward bookkeeping needs both action-history blocks.
        # Keep only a current-frame compatibility slot, not encoder history.
        cfg = self.cfg
        local_cfg = copy(cfg)
        local_cfg.env = copy(cfg.env)
        local_cfg.env.num_policy_actions = 2 * self.num_actions
        local_cfg.env.num_obs_hist = 1
        self.cfg = local_cfg
        try:
            super()._init_buffers()
        finally:
            self.cfg = cfg

    def _pre_sim_step(self, actions):
        if actions.shape != (self.num_envs, self.cfg.env.num_policy_actions):
            raise ValueError("Expected (num_envs, num_actions) position commands")
        # Parent clipping/delay is unchanged. Padding is a fixed zero, not a head.
        return super()._pre_sim_step(torch.cat((actions, torch.zeros_like(actions)), dim=-1))

    def post_physics_step(self):
        # Record controller-applied torque, not a learned torque command. This
        # happens before inherited rewards, resets, and next-observation assembly.
        self.actions[:, self.torque_history_slice] = (
            self.simulator._torques[:, self.position_history_slice]
            / simulator_torque_action_scale(self.simulator))
        super().post_physics_step()
