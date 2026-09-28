"""Playback configuration and world-force geometry, without starting IsaacLab."""
from types import SimpleNamespace
import numpy as np
import legged_gym.envs
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
from legged_gym.utils.helpers import class_to_dict
from legged_gym.scripts.play_b1z1_pact import configure_play, joystick_command, force_arrow


def test_rejection_config_preserves_trained_ranges_and_goals():
    cfg = B1Z1PACTCfg()
    ranges = class_to_dict(cfg.commands.ranges)
    goals = class_to_dict(cfg.goal_ee)
    configure_play(cfg, SimpleNamespace(use_joystick=False, ee_forces=True, base_forces=True))
    assert class_to_dict(cfg.commands.ranges) == ranges
    assert class_to_dict(cfg.goal_ee) == goals
    assert not cfg.commands.curriculum
    assert not cfg.use_force_shifted_target
    assert not cfg.commands.use_external_impedance_compensation
    assert cfg.commands.apply_ee_external_forces and cfg.commands.apply_base_external_forces
    configure_play(cfg, SimpleNamespace(use_joystick=True, ee_forces=False, base_forces=False))
    assert not cfg.commands.heading_command
    assert not cfg.commands.push_gripper_stators and not cfg.commands.push_robot_base
    assert not cfg.commands.apply_base_external_torques


def test_joystick_bounds():
    for bounds in ((-.3,.8), (0.,.7), (-.9,-.1)):
        for axis in np.linspace(-2,2,30):
            assert bounds[0] <= joystick_command(axis,bounds) <= bounds[1]
    assert joystick_command(-1,(-.3,.8)) == -.3
    assert joystick_command(1,(-.3,.8)) == .8


def test_applied_force_arrow():
    origin = np.array([1.,2.,3.])
    for force in ([50.,0.,0.], [0.,0.,-50.], [2.,-4.,3.]):
        starts, ends = force_arrow(origin,force,.005)
        np.testing.assert_allclose(starts[0],origin)
        np.testing.assert_allclose(np.array(ends[0])-origin,np.array(force)*.005)
        assert np.isfinite(np.array(starts+ends)).all()
    assert force_arrow(origin,[0.,0.,0.],.005) == ([],[])
