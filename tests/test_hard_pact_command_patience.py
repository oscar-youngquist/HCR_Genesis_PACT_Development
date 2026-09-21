"""Command curriculum tests without simulator initialization."""
from types import SimpleNamespace

import pytest
import torch

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos import Go2HardPACTPos


def make_env(cls, patience=3):
    env = cls.__new__(cls)
    env.device = "cpu"
    env.cfg = SimpleNamespace(commands=SimpleNamespace(
        curriculum=True, curriculum_threshold=0.8,
        curriculum_patience_iterations=patience, max_curriculum=2.0))
    env.command_ranges = {k: [-0.5, 0.5] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")}
    env.commands = torch.zeros(2, 3)
    env.simulator = SimpleNamespace(base_lin_vel=torch.zeros(2, 3))
    env.max_episode_length = 10
    env.reward_scales = {"tracking_lin_vel": 2.0}
    env.episode_sums = {"tracking_lin_vel": torch.full((2,), 19.0)}
    return env


def finish(env, iteration, score):
    env.begin_command_curriculum_iteration()
    env._command_tracking_sum = torch.tensor(score * 2, dtype=torch.float64)
    env._command_tracking_count = 2
    env.finish_command_curriculum_iteration(iteration)


@pytest.mark.parametrize("cls", [Go2HardPACT, Go2HardPACTPos])
def test_zero_uses_exact_legacy_update_only(cls):
    env, reference = make_env(cls, 0), make_env(cls, 0)
    ids = torch.arange(2)
    env.begin_command_curriculum_iteration()
    assert not env._command_tracking_collect
    assert not hasattr(env, "_command_tracking_sum")
    env._update_command_curriculum(ids)
    reference._legacy_task_class._update_command_curriculum(reference, ids)
    assert env.command_ranges == reference.command_ranges
    env.finish_command_curriculum_iteration(0)
    assert env.command_ranges == reference.command_ranges
    assert not hasattr(env, "_command_curriculum_last_iteration")


@pytest.mark.parametrize("cls", [Go2HardPACT, Go2HardPACTPos])
def test_consecutive_iterations_reset_and_resume(cls):
    env = make_env(cls)
    env._update_command_curriculum(torch.arange(2))
    assert env.command_ranges["lin_vel_x"] == [-0.5, 0.5]
    for iteration, score in enumerate([0.9, 0.9, 0.7, 0.9, 0.9]):
        finish(env, iteration, score)
    assert env._command_curriculum_streak == 2
    assert env.command_ranges["lin_vel_x"] == [-0.5, 0.5]
    resumed = make_env(cls)
    resumed.load_command_curriculum_state_dict(env.command_curriculum_state_dict())
    finish(resumed, 5, 0.9)
    assert resumed.command_ranges["lin_vel_x"] == [-1.0, 1.0]
    assert resumed._command_curriculum_streak == 0
    finish(resumed, 5, 0.9)  # Duplicate completion cannot advance the schedule.
    assert resumed._command_curriculum_streak == 0
    finish(resumed, 6, 0.9)
    finish(resumed, 8, 0.9)  # A missing iteration breaks the streak.
    assert resumed._command_curriculum_streak == 1
    finish(resumed, 9, 0.8)
    assert resumed._command_curriculum_streak == 0


@pytest.mark.parametrize("cls", [Go2HardPACT, Go2HardPACTPos])
def test_reward_collection_and_disabled_curriculum(cls):
    env = make_env(cls, 1)
    env.begin_command_curriculum_iteration()
    expected = env._legacy_task_class._reward_tracking_lin_vel(env)
    actual = env._reward_tracking_lin_vel()
    assert torch.equal(actual, expected)
    assert env._command_tracking_count == 2
    env.finish_command_curriculum_iteration(0)
    assert env.command_curriculum_metrics["tracking_mean"] == expected.mean().item()
    assert env.command_ranges["lin_vel_x"] == [-1.0, 1.0]
    env.cfg.commands.curriculum = False
    finish(env, 1, 1.0)
    assert env.command_ranges["lin_vel_x"] == [-1.0, 1.0]


@pytest.mark.parametrize("cls", [Go2HardPACT, Go2HardPACTPos])
def test_empty_nonfinite_and_cap(cls):
    env = make_env(cls, 1)
    env.begin_command_curriculum_iteration()
    env.finish_command_curriculum_iteration(0)
    assert env._command_curriculum_streak == 0
    finish(env, 1, float("nan"))
    assert env.command_ranges["lin_vel_x"] == [-0.5, 0.5]
    for iteration in range(2, 8):
        finish(env, iteration, 1.0)
    assert env.command_ranges["lin_vel_x"] == [-2.0, 2.0]
