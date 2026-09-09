"""PACT must use motor limits, not explicit-actuator PhysX sentinels."""

from types import SimpleNamespace

import pytest
import torch

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.simulator.isaaclab_simulator import IsaacLabSimulator
from legged_gym.simulator.isaaclab_simulator_pact import IsaacLabSimulator_PACT
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig
from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT


def simulator():
    sim = IsaacLabSimulator_PACT.__new__(IsaacLabSimulator_PACT)
    # Articulation groups are hip x4, thigh x4, calf x4, whereas the task
    # requests FR/FL/RR/RL triplets. Neither order may be assumed identical.
    sim._dof_indices = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    sim._robot = SimpleNamespace(
        data=SimpleNamespace(joint_effort_limits=torch.full((2, 12), 1.0e9)),
        actuators={
            "calf": SimpleNamespace(
                joint_indices=[8, 9, 10, 11],
                effort_limit=torch.full((2, 4), 45.43),
            ),
            "hip_thigh": SimpleNamespace(
                joint_indices=list(range(8)),
                effort_limit=torch.full((2, 8), 23.7),
            ),
        },
    )
    return sim


def test_motor_limits_canonical_order_cached_and_base_adapter_unchanged():
    sim = simulator()
    expected = torch.tensor([23.7, 23.7, 45.43] * 4)
    torch.testing.assert_close(sim.torque_limits, expected)
    assert sim.torque_limits.data_ptr() == sim.torque_limits.data_ptr()
    assert not sim.torque_limits.requires_grad
    # The generic Isaac Lab backend deliberately retains its old contract.
    torch.testing.assert_close(
        IsaacLabSimulator.torque_limits.fget(sim), torch.full((12,), 1.0e9)
    )


def test_all_joint_slice_and_tighter_physx_limit():
    sim = simulator()
    sim._robot.actuators = {"all": SimpleNamespace(
        joint_indices=slice(None), effort_limit=torch.full((2, 12), 23.7)
    )}
    sim._robot.data.joint_effort_limits[:, 4] = 15.0
    expected = torch.full((12,), 23.7)
    expected[1] = 15.0
    torch.testing.assert_close(sim.torque_limits, expected)


@pytest.mark.parametrize("bad_limit", [0.0, float("nan"), -1.0])
def test_invalid_motor_limits_fail_clearly(bad_limit):
    sim = simulator()
    sim._robot.actuators["calf"].effort_limit[:, 0] = bad_limit
    with pytest.raises(ValueError, match="actuator effort limits"):
        _ = sim.torque_limits


def test_unmapped_controlled_joints_fail_instead_of_using_sentinel():
    sim = simulator()
    del sim._robot.actuators["calf"]
    with pytest.raises(ValueError, match="actuator effort limits"):
        _ = sim.torque_limits


def test_real_reward_accumulates_nonzero_soft_limit_penalty():
    env = Go2HardPACT.__new__(Go2HardPACT)
    env.simulator = simulator()
    env.cfg = SimpleNamespace(rewards=SimpleNamespace(
        soft_torque_limit=0.9, only_positive_rewards=True,
    ))
    env.simulator._unweighted_torques = torch.zeros(2, 12)
    # First row exceeds the physical soft limits by 1 and 2 Nm. These
    # torques are still below the motor's HARD limits; the second row is safe.
    env.simulator._unweighted_torques[0, 0] = 0.9 * 23.7 + 1.0
    env.simulator._unweighted_torques[0, 2] = -(0.9 * 45.43 + 2.0)
    torch.testing.assert_close(env._reward_torque_limits(), torch.tensor([3.0, 0.0]))
    env.dt = 0.02
    env.num_envs, env.device = 2, "cpu"
    env.reward_scales = {"torque_limits": -0.01}
    env.rew_buf = torch.zeros(2)
    env._prepare_reward_function()
    env.compute_reward()
    torch.testing.assert_close(
        env.episode_sums["torque_limits"], torch.tensor([-0.0006, 0.0])
    )
    # Positive-total-reward clipping must not erase individual logged terms.
    assert env.rew_buf.eq(0).all()
    episode_metric = env.episode_sums["torque_limits"].mean() / 20.0
    torch.testing.assert_close(episode_metric, torch.tensor(-0.000015))


def test_ppo_qp_binding_receives_physical_motor_limits():
    sim = simulator()
    alg = PPO_HardPACT.__new__(PPO_HardPACT)
    alg.qp_config = HardPACTQPConfig()
    alg.configure_hard_pact_qp(
        sim.torque_limits, torch.tensor([[-2.0, 2.0]] * 12),
        torch.full((12,), 30.0),
    )
    torch.testing.assert_close(
        alg.hard_pact_qp.torque_limits, torch.tensor([23.7, 23.7, 45.43] * 4)
    )
