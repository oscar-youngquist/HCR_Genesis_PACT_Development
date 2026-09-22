"""Feedforward units agree across simulator execution, PINNs and torque cloning."""
from types import SimpleNamespace

import pytest
import torch

from test_b1z1_sampled_context import make_model  # Initialize repository imports.
from legged_gym.torque_action_scaling import resolve_torque_action_scale, simulator_torque_action_scale
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
from legged_gym.simulator.isaacgym_simulator_b1z1 import IsaacGymSimulatorB1Z1PACT
from legged_gym.simulator.genesis_simulator_b1z1_pact import GenesisSimulatorB1Z1PACT
from rsl_rl.algorithms.ppo_b1z1_pact import PPO_B1Z1PACT
from rsl_rl.algorithms.ppo_b1z1_pact_pos import PPO_B1Z1PACTPos


def config(scale=100., overrides=None):
    return SimpleNamespace(asset=SimpleNamespace(dof_names=B1Z1PACTCfg.asset.dof_names),
        env=SimpleNamespace(num_actions=17),
        control=SimpleNamespace(torque_scale=scale, torque_scale_overrides=overrides or {}, action_scale=.25))


def test_scalars_vectors_and_named_order():
    ref = torch.zeros(2, 19, dtype=torch.float64)
    torch.testing.assert_close(resolve_torque_action_scale(config(), ref), ref.new_full((17,), 100.))
    values = list(range(1, 18))
    cfg = config(values, {"z1_elbow": 3., "FR_hip_joint": 6.})
    expected = ref.new_tensor(values)
    expected[0], expected[14] = 6., 3.
    torch.testing.assert_close(resolve_torque_action_scale(cfg, ref), expected)
    scale = resolve_torque_action_scale(config(overrides={name: 10. for name in
        B1Z1PACTCfg.asset.dof_names[12:17]}), ref)
    torch.testing.assert_close(scale, ref.new_tensor([100.]*12+[10.]*5))


@pytest.mark.parametrize("cfg", [config([1.]*16), config(float("nan")), config(0.),
                                 config(overrides={"z1_wrist_rotate": 10.}),
                                 config(overrides={"typo": 10.})])
def test_invalid_scales_rejected(cfg):
    with pytest.raises(ValueError):
        resolve_torque_action_scale(cfg, torch.zeros(1, 19))


def test_execution_and_pinn_agree():
    cfg = config(overrides=B1Z1PACTCfg.control.torque_scale_overrides)
    z, o = torch.zeros(2, 19), torch.ones(2, 19)
    sim = SimpleNamespace(_cfg=cfg, dof_pos=z, dof_vel=z, default_dof_pos=z,
        _dof_pos=z, _dof_vel=z, _default_dof_pos=z, _num_envs=2, _num_learned_actions=17,
        _kp_scale=o, _kd_scale=o, _p_gains=o, _d_gains=o, _motor_strength=o,
        feedback_tau_weight=torch.ones(2, 1), feedforward_tau_weight=torch.ones(2, 1),
        feedforward_torques=z.clone())
    with torch.inference_mode():
        scales = simulator_torque_action_scale(sim)
    assert not scales.is_inference()
    actions = torch.cat((torch.zeros(2, 17), torch.ones(2, 17)), -1).requires_grad_()
    gym = IsaacGymSimulatorB1Z1PACT._compute_torques(sim, actions)
    genesis = GenesisSimulatorB1Z1PACT._compute_torques(sim, actions)
    state = torch.zeros(2, 180)
    state[:, 97:156] = 1.
    alg = SimpleNamespace(cfg={"position_action_scale": .25, "torque_action_scale": scales.tolist()})
    pinn = PPO_B1Z1PACT._coupled_torque(alg, actions, state)
    torch.testing.assert_close(gym, genesis)
    torch.testing.assert_close(gym, pinn)
    torch.testing.assert_close(gym[:, :17], scales.expand(2, -1))
    assert torch.equal(gym[:, 17:], torch.zeros(2, 2))
    pinn.sum().backward()
    torch.testing.assert_close(actions.grad[:, 17:], scales.expand(2, -1))


def test_pos_clone_uses_per_joint_units():
    scales = torch.tensor([100.]*12 + [10.]*5)
    mean = (1 / scales).expand(2, -1).clone().requires_grad_()
    alg = SimpleNamespace(cfg={"torque_action_scale": scales.tolist(), "position_action_scale": .25,
        "dof_pos_obs_scale": 1., "dof_vel_obs_scale": 1., "torque_clone_target_scale": 1.},
        actor_critic=SimpleNamespace(last_torque_mean=mean, last_position_mean=torch.zeros(2, 17)))
    observations = torch.zeros(2, 81)
    observations[:, 5:22] = -1.
    state = torch.cat((torch.ones(2, 57), torch.zeros(2, 19)), -1)
    loss = PPO_B1Z1PACTPos._torque_clone_loss(alg, observations, state)
    torch.testing.assert_close(loss, torch.tensor(0.))
    loss.backward()
    assert torch.isfinite(mean.grad).all()
