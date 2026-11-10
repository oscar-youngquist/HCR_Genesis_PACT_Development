"""FACET-only numerical/lifecycle contracts; no simulator or policy training."""
from dataclasses import replace
from types import SimpleNamespace
import math

import pytest
import torch

from legged_gym.envs.b1z1.b1z1_facet.reference import (
    FACETReference, FACETReferenceCfg, applied_loads, ee_setpoint,
)


def setup(count=2, **kwargs):
    cfg = replace(FACETReferenceCfg(), **kwargs)
    ref = FACETReference(count, "cpu", cfg, torch.float64)
    z = torch.zeros(count, 3, dtype=torch.float64)
    ref.reset(torch.arange(count), z, z, z[:, 0], z[:, 0], z, z)
    return ref, dict(dt=.02, measured_xy=z[:, :2], measured_yaw=z[:, 0],
        command_world=z[:, :2], yaw_command=z[:, 0], ee_local=z,
        ee_local_velocity=z, height=z[:, 0], ee_force=z, base_force=z, yaw_moment=z[:, 0])


@pytest.mark.parametrize("channel", ["base", "ee", "yaw"])
def test_independent_critical_step_and_release(channel):
    ref, args = setup(base_mass=1, ee_mass=1, yaw_inertia=1,
        base_stiffness=4, base_damping=4, ee_stiffness=4, ee_damping=4,
        yaw_stiffness=4, yaw_damping=4, alpha=0)
    # EE follows a moving base, so decouple its spring when testing the base.
    if channel != "ee":
        ref._parameters["ee_stiffness"].zero_()
        ref._parameters["ee_damping"].zero_()
    force_key = {"base": "base_force", "ee": "ee_force", "yaw": "yaw_moment"}[channel]
    load = torch.zeros_like(args[force_key]); load[..., 0] = 1 if load.ndim == 2 else 0
    if load.ndim == 1:
        load.fill_(1)
    args[force_key] = load
    key = {"base": "base_position_xy", "ee": "ee_position_world", "yaw": "yaw"}[channel]
    for step in range(100):
        ref.advance(**args)
        t = (step+1)*args["dt"]
        expected = (1-(1+2*t)*math.exp(-2*t))/4
        value = ref.read()[key]
        if value.ndim == 2:
            value = value[:, 0]
        torch.testing.assert_close(value, torch.full_like(value, expected), atol=2e-9, rtol=1e-7)
    args[force_key] = torch.zeros_like(load)
    for _ in range(500):
        ref.advance(**args)
    assert ref.read()[key].abs().max() < 1e-7


@pytest.mark.parametrize("alpha", [0., 1.])
def test_simultaneous_loads_and_spring_reaction(alpha):
    ref, args = setup(alpha=alpha)
    args["ee_local"] = torch.ones_like(args["ee_local"])*.1
    args["ee_force"] = torch.ones_like(args["ee_force"])*3
    args["base_force"] = torch.ones_like(args["base_force"])*2
    before = ref.read()
    ref.advance(**args)
    out = ref.read()
    torch.testing.assert_close(out["ee_reaction_xy"], -alpha*out["ee_spring_force_world"][:, :2])
    torch.testing.assert_close((out["base_velocity_xy"]-before["base_velocity_xy"])*4/args["dt"],
        out["base_spring_force_xy"]+out["ee_reaction_xy"]+2)
    torch.testing.assert_close((out["ee_velocity_world"]-before["ee_velocity_world"])*2/args["dt"],
        out["ee_spring_force_world"]+3)
    assert out["yaw"].count_nonzero() == 0  # No invented EE-to-yaw lever term.


def test_equilibrium_read_only_and_detached():
    ref, args = setup()
    for name, value in args.items():
        if isinstance(value, torch.Tensor):
            args[name] = value.clone().requires_grad_()
    ref.advance(**args)
    first = ref.read()
    for value in first.values():
        assert not value.requires_grad and value.grad_fn is None
    assert first["ee_position_world"].count_nonzero() == 0
    first["ee_position_world"].fill_(10)
    assert ref.read()["ee_position_world"].count_nonzero() == 0
    torch.testing.assert_close(ref.read()["elapsed_seconds"], torch.full((2,), .02, dtype=torch.float64))


def test_zero_stiffness_velocity_commands_and_yaw_wrap():
    ref, args = setup(base_stiffness=0, yaw_stiffness=0, alpha=0)
    args["command_world"] = torch.full_like(args["command_world"], .4)
    args["yaw_command"] = torch.full_like(args["yaw_command"], .2)
    for _ in range(500):
        ref.advance(**args)
    torch.testing.assert_close(ref.read()["base_velocity_xy"], args["command_world"], atol=1e-7, rtol=0)
    torch.testing.assert_close(ref.read()["yaw_rate"], args["yaw_command"])
    ref, args = setup(yaw_stiffness=10, yaw_damping=0)
    ref._state[:, 4] = math.pi-.01
    args["measured_yaw"] = torch.full_like(args["measured_yaw"], -math.pi+.01)
    ref.advance(**args)
    assert (ref.read()["yaw_rate"] > 0).all()
    assert (ref.read()["yaw_rate"] < .01).all()


def test_moving_anchor_velocity_command():
    ref, args = setup(alpha=0)
    args["command_world"] = torch.full_like(args["command_world"], .3)
    for _ in range(300):
        args["measured_xy"] = ref.read()["base_position_xy"]
        ref.advance(**args)
    # Measured moving anchors, not a fixed initial position or heading target.
    assert (ref.read()["base_velocity_xy"] > .29).all()


def test_setpoint_mount_height_and_yaw_transport_finite_difference():
    dtype = torch.float64
    base = torch.tensor([[1., 2.]], dtype=dtype)
    velocity = torch.tensor([[.3, -.2]], dtype=dtype)
    yaw, rate, height = (torch.tensor([v], dtype=dtype) for v in (.7, .4, .63))
    local = torch.tensor([[.6, .1, .2]], dtype=dtype)
    local_velocity = torch.tensor([[.02, -.03, .04]], dtype=dtype)
    p, v = ee_setpoint(base, velocity, yaw, rate, local, local_velocity, height)
    dt = 1e-6
    next_p, _ = ee_setpoint(base+dt*velocity, velocity, yaw+dt*rate, rate,
                            local+dt*local_velocity, local_velocity, height)
    torch.testing.assert_close((next_p-p)/dt, v, atol=1e-7, rtol=1e-6)
    assert p[0, 2] == pytest.approx(.83)


def test_force_gates_and_application_moment():
    cfg = SimpleNamespace(push_gripper_stators=True, apply_ee_external_forces=True,
        push_robot_base=True, apply_base_external_forces=True, apply_base_external_torques=True)
    force = torch.tensor([[35., 0., 0.]])
    origin = torch.zeros_like(force)
    couple = torch.tensor([[0., 0., 2.]])
    ee, base, moment = applied_loads(cfg, force, (force, couple, origin), origin, origin)
    assert moment.item() == 2  # CoM offsets are not application levers.
    point = torch.tensor([[0., .15, 0.]])
    _, _, moment = applied_loads(cfg, force, (force, couple, point), origin, point)
    assert moment.item() == pytest.approx(2-5.25)
    cfg.apply_base_external_torques = False
    _, _, moment = applied_loads(cfg, force, (force, couple, point), origin, point)
    assert moment.item() == pytest.approx(-5.25)
    cfg.apply_base_external_forces = cfg.apply_ee_external_forces = False
    assert all(value.count_nonzero() == 0 for value in applied_loads(cfg, force, (force, couple, point), origin, point))


def test_rotation_equivariance():
    from legged_gym.envs.b1z1.directional_compliance import rotate_yaw
    ref, args = setup()
    rotated, other = setup()
    angle = torch.full((2,), math.pi/2, dtype=torch.float64)
    args["ee_force"] = torch.tensor([[3., 2., -1.]], dtype=torch.float64).expand(2, -1)
    other["ee_force"] = rotate_yaw(args["ee_force"], angle)
    rotated._state[:, 4] = angle
    other["measured_yaw"] = angle
    for _ in range(10):
        ref.advance(**args); rotated.advance(**other)
    torch.testing.assert_close(rotated.read()["ee_position_world"], rotate_yaw(ref.read()["ee_position_world"], angle))


def test_partial_reset_rng_isolation_and_reproducibility():
    torch.manual_seed(40)
    global_state = torch.random.get_rng_state().clone()
    a, args = setup(sample_ranges={"ee_mass": [1., 3.], "alpha": [0., 1.]})
    b, _ = setup(sample_ranges={"ee_mass": [1., 3.], "alpha": [0., 1.]})
    assert torch.equal(global_state, torch.random.get_rng_state())
    for key in a.read():
        torch.testing.assert_close(a.read()[key], b.read()[key])
    a.advance(**args)
    before = a.read()
    z = torch.ones(1, 3, dtype=torch.float64)
    a.reset(torch.tensor([0]), z, z, z[:, 0], z[:, 0], z, z)
    for key in before:
        torch.testing.assert_close(before[key][1], a.read()[key][1])
    assert a.read()["elapsed_seconds"][0] == 0


def test_timestep_convergence():
    errors = []
    for dt in (.08, .04, .02):
        ref, args = setup(base_stiffness=0, base_damping=0, yaw_stiffness=0, yaw_damping=0,
            ee_mass=1, ee_stiffness=1, ee_damping=2, alpha=0, max_substep_seconds=.5)
        args["dt"] = dt
        args["ee_force"] = torch.ones_like(args["ee_force"])
        for _ in range(round(.8/dt)):
            ref.advance(**args)
        errors.append(abs(ref.read()["ee_position_world"][0, 0].item()-(1-1.8*math.exp(-.8))))
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.parametrize("kwargs", [{"base_mass": 0}, {"ee_damping": -1}, {"alpha": 2},
    {"sample_ranges": {"ee_mass": [2, 1]}}, {"sample_ranges": {"foo": [0, 1]}}, {"max_substep_seconds": 0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        setup(**kwargs)


def test_environment_interval_timing_and_command_adapter(monkeypatch):
    from legged_gym.envs.b1z1.b1z1_facet import B1Z1FACET, B1Z1FACETCfg
    from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos import B1Z1PPOPos
    from legged_gym.envs.b1z1.directional_compliance import cartesian
    monkeypatch.setattr(B1Z1PPOPos, "_pre_sim_step", lambda self, actions: actions)
    env = B1Z1FACET.__new__(B1Z1FACET)
    env.cfg = B1Z1FACETCfg()
    for flag in ("push_gripper_stators", "apply_ee_external_forces", "push_robot_base",
                 "apply_base_external_forces", "apply_base_external_torques"):
        setattr(env.cfg.commands, flag, True)
    env.dt = .02
    ref, args = setup()
    env.facet_reference = ref
    env._facet_interval = None
    z = torch.zeros(2, 3, dtype=torch.float64)
    env.simulator = SimpleNamespace(base_pos=z.clone(), _base_world_lin_vel=z.clone(),
        _base_world_ang_vel=z.clone(), ee_pos=z.clone(), ee_vel=z.clone(),
        base_quat=torch.tensor([[0., 0., math.sqrt(.5), math.sqrt(.5)]], dtype=torch.float64).repeat(2, 1),
        measured_heights=torch.full((2, 4), .2, dtype=torch.float64), _ee_force_world=z.clone())
    torso, couple = z.clone(), z.clone()
    env.simulator.applied_torso_wrench_source = lambda: (torso, couple, env.simulator.base_pos)
    env.commands = torch.tensor([[.4, 0., .1]], dtype=torch.float64).repeat(2, 1)
    env.goal_timer = torch.full((2,), 50., dtype=torch.float64)
    env.traj_timesteps = torch.full((2,), 100., dtype=torch.float64)
    env.ee_start_sphere = torch.tensor([[.5, .1, 0.]], dtype=torch.float64).repeat(2, 1)
    env.ee_goal_sphere = env.ee_start_sphere + torch.tensor([.1, .02, .03], dtype=torch.float64)
    env.curr_ee_goal_sphere = (env.ee_start_sphere+env.ee_goal_sphere)/2
    env.curr_ee_goal_cart = cartesian(env.curr_ee_goal_sphere)
    env.ee_goal_center_offset = torch.tensor([[.2, .01, 99.]], dtype=torch.float64).repeat(2, 1)
    action = torch.zeros(2, 17)
    assert env._pre_sim_step(action) is action
    packet = env._facet_interval
    torch.testing.assert_close(packet[2], torch.tensor([[0., .4]], dtype=torch.float64).repeat(2, 1))
    torch.testing.assert_close(packet[4][:, :2], env.curr_ee_goal_cart[:, :2]+env.ee_goal_center_offset[:, :2])
    assert (packet[6] == .2+env.cfg.goal_ee.sphere_center.z_invariant_offset).all()
    dt = 1e-6
    u = .5+dt/(100*env.dt)
    next_sphere = env.ee_start_sphere+(10*u**3-15*u**4+6*u**5)*(env.ee_goal_sphere-env.ee_start_sphere)
    torch.testing.assert_close(packet[5], (cartesian(next_sphere)-env.curr_ee_goal_cart)/dt, atol=1e-7, rtol=1e-5)
    # Loads are sampled after _pre_sim_step, then held by simulation. Successor
    # geometry/commands must not replace the captured initial anchor or lever.
    env.simulator._ee_force_world.fill_(2)
    torso[:, 0] = 3
    couple[:, 2] = .5
    env.simulator.base_pos.fill_(10)
    def parent_post(self):
        self.commands.fill_(100)
        self.simulator._ee_force_world.zero_()
    monkeypatch.setattr(B1Z1PPOPos, "post_physics_step", parent_post)
    env.post_physics_step()
    out = env.get_facet_references()
    torch.testing.assert_close(out["applied_ee_force_world"], z+2)
    torch.testing.assert_close(out["applied_yaw_moment"], z[:, 0]+.5)
    env.post_physics_step()
    torch.testing.assert_close(out["elapsed_seconds"], env.get_facet_references()["elapsed_seconds"])
