"""Consecutive-failure termination for Go2 PACT and both HardPACT tasks."""
from types import SimpleNamespace as NS

import pytest
import torch

from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos import Go2PACTPos
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos import Go2HardPACTPos


def task(cls):
    env = cls.__new__(cls)
    env.cfg = NS(
        termination=NS(termination_terms=["roll", "pitch", "height_min", "height_max"],
                       roll_threshold=.7, pitch_threshold=.7, height_min=.2, height_max=.6),
        terrain=NS(reset_out_of_bounds=True), env=NS(fail_to_terminal_time_s=.02),
    )
    env.dt, env.max_episode_length = .02, 1000
    env.fail_buf = torch.zeros(5, dtype=torch.long)
    env.episode_length_buf = torch.zeros(5, dtype=torch.long)
    env.non_failure_reset_buf = torch.zeros(5, dtype=torch.bool)
    env.simulator = NS(
        link_contact_forces=torch.zeros(5, 1, 3), termination_contact_indices=[0],
        _base_euler=torch.zeros(5, 3), base_pos=torch.tensor([[0., 0., .3]]).repeat(5, 1),
        measured_heights=torch.zeros(5, 1), _base_pos_out_of_bounds_buf=torch.zeros(5, dtype=torch.bool),
    )
    return env


@pytest.mark.parametrize("cls", [Go2PACT, Go2PACTPos, Go2HardPACT, Go2HardPACTPos])
def test_all_failure_predicates_accumulate_and_healthy_rows_clear(cls):
    env = task(cls)
    # Each row violates a different termination condition.
    env.simulator._base_euler[0, 0] = .8
    env.simulator._base_euler[1, 1] = -.8
    env.simulator.base_pos[2, 2] = .1
    env.simulator.base_pos[3, 2] = .7
    env.simulator.link_contact_forces[4, 0, 2] = 11.
    env.check_termination()
    assert env.fail_buf.eq(1).all() and not env.reset_buf.any()
    env.check_termination()
    assert env.fail_buf.eq(2).all() and env.reset_buf.all()
    # Partial recovery clears only those rows' counters, without resetting
    # healthy histories in other environments or accumulating lifetime hits.
    env.simulator._base_euler[0].zero_()
    env.simulator.link_contact_forces[4].zero_()
    env.check_termination()
    torch.testing.assert_close(env.fail_buf, torch.tensor([0, 3, 3, 3, 0]))
    env.simulator.link_contact_forces[4, 0, 2] = 11.
    env.check_termination()
    assert env.fail_buf[4] == 1 and not env.reset_buf[4]


@pytest.mark.parametrize("cls", [Go2PACT, Go2PACTPos, Go2HardPACT, Go2HardPACTPos])
def test_simultaneous_failures_count_once_and_timeouts_are_unchanged(cls):
    env = task(cls)
    env.simulator._base_euler[0, :2] = .8
    env.simulator.base_pos[0, 2] = .1
    env.simulator.link_contact_forces[0, 0, 2] = 11.
    env.episode_length_buf[1:3] = torch.tensor([1000, 1001])
    env.simulator._base_pos_out_of_bounds_buf[3] = True
    env.check_termination()
    assert env.fail_buf[0] == 1 and not env.reset_buf[0]
    assert not env.time_out_buf[1] and env.time_out_buf[2]
    assert env.reset_buf[2]
    # Preserve each legacy task's existing out-of-bounds behavior.
    assert bool(env.reset_buf[3]) == (cls in (Go2PACT, Go2HardPACT))
