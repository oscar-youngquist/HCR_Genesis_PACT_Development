"""Simulator-free checks for the optional arm compensation RL reward."""
from types import SimpleNamespace

import pytest
import torch

from test_b1z1_actor_feedforward import fixture as actor_fixture
from legged_gym.envs.b1z1.b1z1_pact import arm_feedforward_reward as reward
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
from rsl_rl.algorithms.b1z1_actor_feedforward import compensation_target


def fixture(dtype=torch.float32):
    a, batch, actions, _, acceleration, force, ids, dofs = actor_fixture(dtype=dtype)
    cfg = B1Z1PACTCfg()
    cfg.rewards.scales.tracking_arm_feedforward = .5
    state = batch["actor_phys_state"].clone()
    state[:, 97:97 + dofs] = torch.linspace(.8, 1.2, dofs, dtype=dtype)
    state[:, 155] = .7
    fixed = a.actor_physics_cache
    calls = []
    def evaluate(*args):
        assert not torch.is_inference_mode_enabled() and not args[0].is_inference()
        assert not torch.is_grad_enabled()
        calls.append(tuple(v.clone() for v in args))
        return SimpleNamespace(**{k: v[:len(args[0])] for k, v in vars(fixed).items()})
    backend = SimpleNamespace(batch_capacity=2, device="cpu", evaluate=evaluate)
    n = len(actions)
    env = SimpleNamespace(cfg=cfg, num_actions=cfg.env.num_actions, num_envs=n, dt=.02, device="cpu",
        rew_buf=torch.zeros(n, dtype=dtype), reset_buf=torch.zeros(n, dtype=torch.bool),
        reward_scales={reward.NAME: .5}, get_pact_dynamics_state=lambda: state,
        simulator=SimpleNamespace(_cfg=cfg, _arm_dof_cfg_ids=ids.tolist(),
            _torque_action_scale=actions.new_tensor(a.cfg["torque_action_scale"]),
            dof_pos=state[:, 7:26], dof_vel=state[:, 32:51].clone(),
            _base_world_lin_vel=state[:, 26:29].clone(), _base_world_ang_vel=state[:, 29:32].clone(),
            base_pos=state[:, :3].clone()))
    env._reward_tracking_arm_feedforward = lambda: B1Z1PACT._reward_tracking_arm_feedforward(env)
    reward.configure(env, backend)
    target = compensation_target(fixed, acceleration, force, ids, dofs)[0]
    gain = state[:, 97:97 + dofs].index_select(1, ids) * state[:, 155:156] * env._arm_ff_reward_scale
    actions = torch.zeros_like(actions)
    actions[:, env.num_actions + ids] = target / gain
    return env, state, actions, acceleration, calls, fixed


def complete_step(env, acceleration):
    initial = env._arm_ff_reward_snapshot[0]
    velocity = initial[:, 26:51] + env.dt * acceleration
    env.simulator._base_world_lin_vel = velocity[:, :3]
    env.simulator._base_world_ang_vel = velocity[:, 3:6]
    env.simulator.dof_vel = velocity[:, 6:]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_positive_reward_matches_teacher_and_is_detached(dtype):
    env, state, actions, acceleration, calls, fixed = fixture(dtype)
    for value in (state, actions, acceleration, *vars(fixed).values()):
        value.requires_grad_()
    reward.capture(env, actions)
    complete_step(env, acceleration)
    value = reward.reward(env)
    torch.testing.assert_close(value, torch.ones_like(value))
    assert not value.requires_grad and value.dtype == dtype
    assert [len(call[0]) for call in calls] == [2, 1]
    bad = actions.detach().clone()
    bad[:, env.num_actions + env._arm_ff_reward_indices] += .1
    reward.capture(env, bad)
    slightly_wrong = reward.reward(env)
    bad[:, env.num_actions + env._arm_ff_reward_indices] += .2
    reward.capture(env, bad)
    worse = reward.reward(env)
    assert ((slightly_wrong > 0) & (slightly_wrong < 1)).all()
    assert (worse < slightly_wrong).all()
    assert all(v.grad is None for v in (state, actions, acceleration, *vars(fixed).values()))


def test_uses_captured_forces_randomization_and_delay_selected_action():
    env, state, actions, acceleration, calls, _ = fixture()
    state[:, 175:180] = state.new_tensor([2., .01, -.02, .03, .4])
    with torch.inference_mode():
        reward.capture(env, actions)
        snapshot = env._arm_ff_reward_snapshot[0].clone()
        # Later resampling/observation updates must not alter the interval label.
        state[:, 88:97] = 100.
        state[:, 175:180] = -100.
        actions.zero_()
        complete_step(env, acceleration)
        value = reward.reward(env)
    torch.testing.assert_close(value, torch.ones_like(value))
    torch.testing.assert_close(calls[0][-3], snapshot[:2, 175:176])
    torch.testing.assert_close(calls[0][-2], snapshot[:2, 176:179])
    torch.testing.assert_close(calls[0][-1], snapshot[:2, 179:180])
    # The snapshot contains the action supplied after the environment's delay queue.
    assert env._arm_ff_reward_snapshot[1].abs().sum() > 0


@pytest.mark.parametrize("reason", ["reset", "teleport", "nonfinite", "no_authority"])
def test_invalid_rows_skip_dynamics(reason):
    env, state, actions, acceleration, calls, _ = fixture()
    if reason == "no_authority":
        state[:, 155] = 0.
    reward.capture(env, actions)
    complete_step(env, acceleration)
    if reason == "reset":
        env.reset_buf.fill_(True)
    elif reason == "teleport":
        env.simulator.base_pos += 10.
    elif reason == "nonfinite":
        env.simulator.dof_vel.fill_(float("nan"))
    assert not reward.reward(env).any() and calls == []


def test_nonfinite_teacher_is_not_rewarded():
    env, _, actions, acceleration, _, fixed = fixture()
    fixed.bias.fill_(float("nan"))
    reward.capture(env, actions)
    complete_step(env, acceleration)
    value = reward.reward(env)
    assert torch.isfinite(value).all() and not value.any()


@pytest.mark.parametrize("position_only", [False, True])
def test_zero_scale_and_position_only_have_no_work(position_only):
    cfg = B1Z1PACTCfg()
    if position_only:
        cfg.env.num_policy_actions = cfg.env.num_actions
    env = SimpleNamespace(cfg=cfg, num_actions=cfg.env.num_actions, num_envs=2, dt=.02,
                          reward_scales={reward.NAME: 1. if position_only else 0.},
                          device="cpu", rew_buf=torch.zeros(2))
    B1Z1PACT._prepare_reward_function(env)
    assert reward.NAME not in env.reward_scales and not env.reward_functions
    reward.configure(env, None)
    reward.capture(env, None)
    assert not reward.reward(env).any()
    assert not hasattr(env, "_arm_ff_reward_backend")


def test_reward_enters_normal_total_and_episode_logging():
    env, _, actions, acceleration, _, _ = fixture()
    B1Z1PACT._prepare_reward_function(env)
    reward.capture(env, actions)
    complete_step(env, acceleration)
    B1Z1PACT.compute_reward(env)
    expected = torch.full_like(env.rew_buf, .5 * env.dt)
    torch.testing.assert_close(env.rew_buf, expected)
    torch.testing.assert_close(env.episode_sums[reward.NAME], expected)
    assert env.reward_names == [reward.NAME]


def test_reward_preserved_by_coupled_thin_configs():
    from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs
    for number in range(4, 12):
        cfg_type, _ = make_b1z1_pact_ablation_configs(number)
        cfg = cfg_type()
        env = SimpleNamespace(cfg=cfg, num_actions=cfg.env.num_actions, num_envs=1, device="cpu", dt=.02,
            reward_scales={reward.NAME: .5}, _reward_tracking_arm_feedforward=lambda: torch.ones(1))
        B1Z1PACT._prepare_reward_function(env)
        assert reward.enabled(env) == (cfg.action_mode == "coupled")


def test_validation_and_missing_runner_binding():
    env, _, actions, *_ = fixture()
    for sigma in (0., -1., float("nan"), float("inf")):
        env.cfg.rewards.arm_feedforward_tracking_sigma = sigma
        with pytest.raises(ValueError, match="sigma"):
            reward.configure(env, env._arm_ff_reward_backend)
    del env._arm_ff_reward_backend
    with pytest.raises(RuntimeError, match="runner"):
        reward.capture(env, actions)
