"""PACT must use motor limits, not explicit-actuator PhysX sentinels."""

from types import SimpleNamespace

import pytest
import torch

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos import Go2HardPACTPos
from legged_gym.simulator.isaaclab_simulator import IsaacLabSimulator
from legged_gym.simulator.isaaclab_simulator_pact import IsaacLabSimulator_PACT
from legged_gym.simulator.genesis_simulator_pact import GenesisSimulator_PACT
from legged_gym.simulator.genesis_simulator_pact_pos import GenesisSimulator_PACT_Pos
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig, project_nominal_torque
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


@pytest.mark.parametrize("action_dim", [12, 24])
@pytest.mark.parametrize("qp_active", [False, True])
def test_actual_substep_callback_records_the_clipped_effort_command(action_dim, qp_active):
    """Run the real adapter loop and HardPACT accumulator against a fake API.

    Both Pos PD and coupled controls exercise motor-strength scaling and
    saturation. QP-off covers warmup/ablations; QP-on replaces the command
    at the same hook used by the certified solve/fallback pipeline.
    """
    sim = simulator()
    sim._num_actions = 12
    sim._cfg = SimpleNamespace(control=SimpleNamespace(
        decimation=4, action_scale=.25, torque_scale=10., control_type="P",
    ))
    sim._sim_params = {"dt": .005}
    sim._headless = True
    sim._feet_indices = [0, 1, 2, 3]
    sim._robot.data.joint_vel = torch.zeros(2, 12)
    sim._robot.data.joint_pos = torch.zeros(2, 12)
    sim._robot.data.default_joint_pos = torch.zeros(2, 12)
    sim._robot.data.body_link_vel_w = torch.zeros(2, 4, 6)
    sim._kp_scale = sim._kd_scale = torch.ones(2, 12)
    sim._motor_strength = torch.tensor([[.8], [1.2]])
    sim._p_gains, sim._d_gains = torch.full((12,), 30.), torch.full((12,), .75)
    sim.feedforward_tau_weight = sim.feedback_tau_weight = 1.
    for name in ("_base_lin_vel", "_base_ang_vel", "_base_world_lin_vel", "_base_world_ang_vel"):
        setattr(sim, name, torch.zeros(2, 3))
        setattr(sim, "_last" + name, torch.zeros(2, 3))
    sim._last_feet_vel = torch.zeros(2, 4, 3)
    sim._last_dof_vel = torch.zeros(2, 12)
    commands = []
    sim._robot.set_joint_effort_target = lambda value, ids: commands.append(value.clone())
    sim._robot.write_data_to_sim = lambda: None
    sim._robot.update = lambda dt: None
    sim._sim = SimpleNamespace(step=lambda **kwargs: None)
    sim._contact_sensors = SimpleNamespace(update=lambda dt: None)

    env = Go2HardPACT.__new__(Go2HardPACT)
    env.simulator = sim
    env.cfg = SimpleNamespace(sim=SimpleNamespace(gravity=[0., 0., -9.81]))
    env.obs_scales = SimpleNamespace(base_wrench=.01)
    env._realized_added_mass = torch.zeros(2, 1)
    env._realized_com_shift_body = torch.zeros(2, 3)
    env._current_sustained_wrench_world = torch.zeros(2, 6)
    env._current_base_quat_xyzw = lambda: torch.tensor([[0., 0., 0., 1.]]).expand(2, -1)
    env._apply_sustained_world_wrench = lambda wrench: None
    for suffix in ("sustained", "mass_com", "total", "sustained_yaw_scaled",
                   "mass_com_yaw_scaled", "yaw_scaled", "yaw_physical"):
        setattr(env, "_disturbance_interval_sum_" + suffix, torch.zeros(2, 6))
    env._disturbance_interval_count = torch.zeros(2, 1)
    env._interval_executed_torque_sum = torch.zeros(2, 12)
    env._interval_executed_torque_peak = torch.zeros(2, 12)
    env._interval_executed_torque_count = torch.zeros(2, 1)
    env._hard_pact_rollout_qp_enabled = qp_active
    env._hard_pact_policy_context_ready = True
    env._solve_hard_pact_rollout_qp_substep = lambda *args: sim.hard_pact_set_executed_torque(
        .5 * sim.torque_limits.expand(2, -1)
    )
    sim._hard_pact_pre_physics_substep = env._hard_pact_pre_physics_substep
    sim.step(torch.full((2, action_dim), 10.))
    assert len(commands) == 4
    commands = torch.stack(commands)
    torch.testing.assert_close(
        env._interval_executed_torque_sum / env._interval_executed_torque_count,
        commands.mean(0), rtol=0, atol=0,
    )
    torch.testing.assert_close(env._interval_executed_torque_peak, commands.abs().amax(0))
    assert (commands.abs() <= sim.torque_limits).all()
    # Reward buffers must still detect saturation of the original request.
    assert (sim._unweighted_torques.abs() > sim.torque_limits).all()
    if not qp_active:
        assert (sim._torques.abs() > sim.torque_limits).all()
        torch.testing.assert_close(commands[0], sim.torque_limits.expand(2, -1))

    # Install precisely the HardPACT-only conversion used in production.
    # The legacy assertions above remain unchanged: no generic backend edit.
    if action_dim == 12:
        env.__class__ = Go2HardPACTPos
    env.num_envs, env.num_actions, env.device = 2, 12, "cpu"
    env.actions = torch.zeros(2, action_dim)
    env.cfg.control = sim._cfg.control
    env.cfg.rewards = SimpleNamespace(soft_torque_limit=.9)
    sim._robot.data.default_joint_pos[:] = torch.linspace(-.2, .2, 12)
    sim._robot.data.joint_pos[:] = torch.linspace(-.1, .1, 12)
    sim._robot.data.joint_vel[:] = torch.linspace(-.3, .3, 12)
    sim._kp_scale = torch.linspace(.8, 1.2, 24).reshape(2, 12)
    sim._kd_scale = 1.3 * torch.ones(2, 12)
    sim.feedforward_tau_weight, sim.feedback_tau_weight = .7, 1.1
    clipped_actions = torch.ones(2, action_dim)
    expected_nominal = sim._compute_torques(clipped_actions).clamp(
        -sim.torque_limits, sim.torque_limits)
    genesis = SimpleNamespace(
        _cfg=sim._cfg, _num_envs=2, first_loop=True,
        _kp_scale=sim._kp_scale, _kd_scale=sim._kd_scale,
        _p_gains=sim._p_gains.clone(), _d_gains=sim._d_gains.clone(),
        _default_dof_pos=sim.default_dof_pos, _dof_pos=sim.dof_pos, _dof_vel=sim.dof_vel,
        _motor_strength=sim._motor_strength,
        feedforward_tau_weight=sim.feedforward_tau_weight,
        feedback_tau_weight=sim.feedback_tau_weight,
    )
    genesis_class = GenesisSimulator_PACT_Pos if action_dim == 12 else GenesisSimulator_PACT
    # Both legacy backend controller formulas agree before installing the
    # shared HardPACT hook, including default pose and randomized actuators.
    torch.testing.assert_close(genesis_class._compute_torques(genesis, clipped_actions).clamp(
        -sim.torque_limits, sim.torque_limits), expected_nominal)
    expected_feedback, expected_ff = sim.feedback_torques.clone(), sim.feedforward_torques.clone()
    env._hard_pact_control_parameters = env._capture_control_parameters()
    sim._hard_pact_torque_conversion = env._hard_pact_compute_torques

    # Real simulator substep loop + real transition callback: non-QP,
    # successful-QP and analytic-fallback values must be the API command.
    requested_rewards = []
    for raw_magnitude in (10., 20.):
        env._hard_pact_raw_delayed_action = torch.full((2, action_dim), raw_magnitude)
        for fallback in (False, True) if qp_active else (False,):
            commands = []
            env._interval_executed_torque_sum.zero_()
            env._interval_executed_torque_count.zero_()
            env._interval_executed_torque_peak.zero_()
            env._hard_pact_previous_substep_torque = torch.zeros(2, 12)
            nominal_inputs = []

            def project(*_):
                nominal_inputs.append(env._hard_pact_bounded_nominal_torque.clone())
                if fallback:
                    selected = project_nominal_torque(
                        nominal_inputs[-1],
                        env._hard_pact_previous_substep_torque,
                        sim.torque_limits, 50., .005)
                else:
                    selected = .5 * sim.torque_limits.expand(2, -1)
                sim.hard_pact_set_executed_torque(selected)

            env._solve_hard_pact_rollout_qp_substep = project
            sim.step(clipped_actions)
            commands_tensor = torch.stack(commands)
            assert len(commands) == 4
            torch.testing.assert_close(env._hard_pact_bounded_nominal_torque, expected_nominal)
            for nominal in nominal_inputs:
                torch.testing.assert_close(nominal, expected_nominal)
            if not qp_active:
                torch.testing.assert_close(commands_tensor, expected_nominal.expand(4, -1, -1))
            if fallback:
                delta = torch.diff(commands_tensor, dim=0, prepend=torch.zeros(1, 2, 12))
                assert (delta.abs() <= .25).all()
            torch.testing.assert_close(env._hard_pact_executed_torque, commands[-1])
            torch.testing.assert_close(env._hard_pact_previous_substep_torque, commands[-1])
            torch.testing.assert_close(
                env._interval_executed_torque_sum / env._interval_executed_torque_count,
                commands_tensor.mean(0), rtol=0, atol=0)
            torch.testing.assert_close(sim.feedback_torques, expected_feedback)
            torch.testing.assert_close(sim.feedforward_torques, expected_ff)
            assert (commands_tensor.abs() <= sim.torque_limits).all()
        requested_rewards.append(torch.stack((env._reward_torque_limits(),
            env._reward_feedback_torques(), env._reward_feedforward_torques())))
    assert (requested_rewards[1][:2] > requested_rewards[0][:2]).all()
    if action_dim == 24:
        assert (requested_rewards[1][2] > requested_rewards[0][2]).all()
    else:
        assert requested_rewards[1][2].eq(0).all()
    genesis._hard_pact_torque_conversion = env._hard_pact_compute_torques
    torch.testing.assert_close(genesis_class._compute_torques(genesis, clipped_actions), expected_nominal)
