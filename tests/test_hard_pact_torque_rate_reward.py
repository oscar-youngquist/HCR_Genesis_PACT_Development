import torch
from test_hard_pact_qp_modes import fixture


def test_requested_slew_penalty_bounds_substep_average_and_partial_reset():
    task,*_=fixture('random_one_substep',n=2)
    task._hard_pact_torque_rate_penalty_sum=torch.zeros(2)
    task._hard_pact_torque_rate_penalty_count=torch.zeros(2)
    # Rate=10 Nm/s, physics dt=.01 s -> allowed delta=.1 Nm.
    previous=torch.full((2,12),2.)
    task._hard_pact_previous_substep_torque.copy_(previous)
    requested=previous.clone();requested[0,0]+=.05;requested[1,0]+=.2
    task._accumulate_torque_rate_penalty(requested)
    torch.testing.assert_close(task._reward_torque_rate_limits(),torch.tensor([0.,1.]),atol=2e-6,rtol=1e-5)
    # Read-only reward collection must not replace actual execution history.
    torch.testing.assert_close(task._hard_pact_previous_substep_torque,previous)
    task._accumulate_torque_rate_penalty(previous)
    torch.testing.assert_close(task._reward_torque_rate_limits(),torch.tensor([0.,.5]),atol=2e-6,rtol=1e-5)
    # A larger unclipped request cannot hide behind the same saturated command.
    requested[1,0]=3.
    task._accumulate_torque_rate_penalty(requested)
    assert task._reward_torque_rate_limits()[1]>20
    task.reset_idx(torch.tensor([1]))
    assert task._reward_torque_rate_limits().eq(0).all()
    assert task._hard_pact_torque_rate_penalty_count.tolist()==[3.,0.]


def test_pos_inherits_same_penalty_and_config():
    from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
    from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos import Go2HardPACTPos
    from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg
    from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import GO2HardPACTPosCfg
    assert Go2HardPACTPos._reward_torque_rate_limits is Go2HardPACT._reward_torque_rate_limits
    assert GO2HardPACTCfg.rewards.scales.torque_rate_limits == -.01
    assert GO2HardPACTPosCfg.rewards.scales.torque_rate_limits == -.01
