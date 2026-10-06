"""Whole-body target tracking without constructing a simulator."""
from types import SimpleNamespace

import torch
import legged_gym.envs  # Initialize task registration before importing the environment.
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg


def environment(delay=False):
    cfg = B1Z1PACTCfg()
    cfg.domain_rand.randomize_ctrl_delay = delay
    count = cfg.env.num_actions
    width = len(cfg.asset.dof_names)
    default = torch.zeros(1, width)
    return SimpleNamespace(cfg=cfg, num_actions=count, device="cpu",
        actions=torch.zeros(2, 2*count), last_actions=torch.zeros(2, 2*count),
        llast_actions=torch.zeros(2, 2*count), dof_tracking_target=default.expand(2, -1).clone(),
        simulator=SimpleNamespace(default_dof_pos=default, dof_pos=default.expand(2, -1).clone()),
        action_queue=torch.zeros(2, 2, 2*count), all_env_ids=torch.arange(2),
        action_delay=torch.ones(2, dtype=torch.long))


def test_positive_whole_body_reward():
    env = environment()
    reward = B1Z1PACT._reward_tracking_dof_pos
    torch.testing.assert_close(reward(env), torch.ones(2))
    # Both a leg and the final held arm/gripper DOF contribute identically.
    env.simulator.dof_pos[0, 0] = .1
    env.simulator.dof_pos[1, -1] = .1
    small = reward(env)
    assert (small > 0).all() and (small < 1).all()
    torch.testing.assert_close(small[0], small[1])
    env.simulator.dof_pos *= 2
    assert (reward(env) < small).all()


def test_targets_use_delayed_position_actions_not_feedforward():
    env = environment(True)
    actions = torch.ones_like(env.actions)
    actions[:, env.num_actions:] = 20.
    B1Z1PACT._pre_sim_step(env, actions)
    assert not env.dof_tracking_target.any()
    B1Z1PACT._pre_sim_step(env, torch.zeros_like(actions))
    torch.testing.assert_close(env.dof_tracking_target[:, :env.num_actions],
                               torch.full((2, env.num_actions), env.cfg.control.action_scale))
    assert not env.dof_tracking_target[:, env.num_actions:].any()
    env.simulator.dof_pos.copy_(env.dof_tracking_target)
    torch.testing.assert_close(B1Z1PACT._reward_tracking_dof_pos(env), torch.ones(2))
