"""Named buffers, causal control histories and current-episode isolation."""
from types import SimpleNamespace
import torch
from legged_gym.envs.b1z1.b1z1_facet.b1z1_facet_config import B1Z1FACETCfg
from legged_gym.envs.b1z1.b1z1_facet.observations import FACETObservations, JOINTS, external_wrench
from legged_gym.envs.b1z1.b1z1_facet.reference import FACETReference, FACETReferenceBank
from legged_gym.envs.b1z1.b1z1_facet.b1z1_facet import B1Z1FACET


def packet_env(n=3, device='cpu'):
    cfg = B1Z1FACETCfg()
    cfg.noise.add_noise = False
    env = SimpleNamespace(cfg=cfg, num_envs=n, device=device, dt=.02)
    z = torch.zeros(n, 3, device=device)
    yaw = torch.zeros(n, device=device)
    env._facet_measured_state = lambda: (z, z, yaw, yaw, z, z)
    env.commands = torch.zeros(n, 6, device=device)
    env.goal_timer = torch.zeros_like(yaw)
    env.traj_timesteps = torch.ones_like(yaw)*100
    env.ee_start_sphere = z.clone()
    env.ee_goal_sphere = z.clone()
    env.curr_ee_goal_sphere = z.clone()
    env.curr_ee_goal_cart = z.clone()
    env.ee_goal_center_offset = z.clone()
    env.compute_z1_arm_fk = lambda q: q[:, :3]  # Isolate observation routing, not FK mathematics.
    env.simulator = SimpleNamespace(base_quat=z.new_tensor([0, 0, 0, 1]).expand(n, -1),
        dof_pos=torch.arange(19, device=device).float().expand(n, -1).clone(),
        dof_vel=torch.arange(19, device=device).float().expand(n, -1).clone()+100,
        _base_world_ang_vel=z.clone(), measured_heights=torch.zeros(n, 1, device=device),
        foot_contacts=torch.ones(n, 4, device=device), _ee_force_world=z.clone(),
        applied_torso_wrench_source=lambda: (z, z, z))
    env.facet_reference = FACETReference(n, device, cfg.facet_reference)
    ids = torch.arange(n, device=device)
    env.facet_reference.reset(ids, *env._facet_measured_state())
    env.facet_history = FACETReferenceBank(env.facet_reference, cfg.facet_tracking.horizons)
    return env


def test_schema_raw_units_history_and_getter_no_mutation():
    env = packet_env()
    c = FACETObservations(env)
    ids = torch.arange(3)
    c.collect(env, ids, reset=True)
    packet = c.read()
    assert packet['observation'].shape == (3, 237)
    assert packet['privileged'].shape == (3, 52)
    assert packet['wrench'].shape == (3, 9)
    assert packet['is_init'].all()
    assert c.schema['observed_joint_names'] == JOINTS
    torch.testing.assert_close(c.q[:, 0], env.simulator.dof_pos)
    torch.testing.assert_close(c.qd[:, 0], env.simulator.dof_vel)
    for step in range(5):
        env.simulator.dof_pos.add_(1)
        c.collect(env, ids, torch.full((3, 17), float(step)))
    torch.testing.assert_close(c.actions[0, :, 0], torch.tensor([4., 3., 2.]))
    torch.testing.assert_close(c.q[0, :, 0], torch.tensor([5., 4., 3., 2.]))
    global_rng, private_rng = torch.random.get_rng_state(), c.rng.get_state()
    before = c.read()
    c.read()['observation'].zero_()
    for key, value in c.read().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert torch.equal(global_rng, torch.random.get_rng_state())
    assert torch.equal(private_rng, c.rng.get_state())


def test_partial_reset_terminal_references_and_histories():
    env = packet_env()
    c = FACETObservations(env)
    ids = torch.arange(3)
    bank = env.facet_history
    bank._state.fill_(123)
    columns = (bank._cursor-bank._horizons) % bank.capacity
    bank._ages[:, columns] = bank._horizons
    bank._snapshot = bank._select()
    c.collect(env, ids, torch.ones(3, 17))
    before = c.read()
    bank.clear(ids[:1])
    c.collect(env, ids[:1], reset=True)
    assert bank.read()['valid'][0].all()  # Terminal diagnostics deliberately retained.
    assert not bank.read_current()['valid'][0].any()
    assert c.read()['privileged'][0, 13:].count_nonzero() == 0
    assert c.read()['wrench'][0].count_nonzero() == c.actions[0].count_nonzero() == 0
    assert c.read()['is_init'][0] and not c.read()['is_init'][1:].any()
    for key, value in c.read().items():
        torch.testing.assert_close(value[1:], before[key][1:], rtol=0, atol=0)


def test_student_packet_has_no_privileged_or_reference_leak():
    env = packet_env()
    c = FACETObservations(env)
    ids = torch.arange(3)
    c.collect(env, ids, reset=True)
    before = c.read()['observation']
    env.facet_history._state.fill_(1000)
    env.simulator._ee_force_world.fill_(100)
    env.simulator._base_world_ang_vel.fill_(99)
    c.collect(env, ids, reset=True)
    torch.testing.assert_close(before, c.read()['observation'], rtol=0, atol=0)


def test_private_noise_rng_and_joint_name_mapping():
    env = packet_env()
    env.cfg.noise.add_noise = True
    env.cfg.facet_observations.observed_joint_names = tuple(reversed(JOINTS))
    state = torch.random.get_rng_state()
    a, b = FACETObservations(env), FACETObservations(env)
    for c in (a, b):
        c.collect(env, torch.arange(3), reset=True)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(a.read()['observation'], b.read()['observation'], rtol=0, atol=0)
    env.cfg.noise.add_noise = False
    a.collect(env, torch.arange(3), reset=True)
    torch.testing.assert_close(a.q[:, 0], env.simulator.dof_pos.flip(-1))
    torch.testing.assert_close(a.read()['observation'][:, -3:], env.simulator.dof_pos[:, 12:15])


def test_external_wrench_gate_frames_and_single_lever_shift():
    env = packet_env(1)
    env.cfg.commands.push_robot_base = env.cfg.commands.apply_base_external_forces = True
    env.cfg.commands.apply_base_external_torques = True
    env.cfg.commands.push_gripper_stators = env.cfg.commands.apply_ee_external_forces = True
    f, m, point = torch.tensor([[35., 0., 0.]]), torch.tensor([[0., 0., 2.]]), torch.tensor([[0., .15, 0.]])
    env.simulator.applied_torso_wrench_source = lambda: (f, m, point)
    env.simulator._ee_force_world[:] = torch.tensor([0., 4., -5.])
    value = external_wrench(env, torch.zeros_like(f), torch.tensor([torch.pi/2]))
    torch.testing.assert_close(value, torch.tensor([[0., -35., 0., 4., 0., -5., 0., 0., -3.25]]), atol=2e-6, rtol=1e-5)
    env.cfg.commands.apply_base_external_forces = False
    env.cfg.commands.apply_base_external_torques = False
    env.cfg.commands.apply_ee_external_forces = False
    assert external_wrench(env, torch.zeros_like(f), torch.zeros(1)).count_nonzero() == 0


def test_issued_actions_captured_before_delay_and_torque_compatibility_slots():
    env = B1Z1FACET.__new__(B1Z1FACET)
    env.cfg, env.device, env.num_envs, env.num_actions = B1Z1FACETCfg(), 'cpu', 2, 17
    env.cfg.domain_rand.randomize_ctrl_delay = True
    env.cfg.normalization.clip_actions = .5
    env.all_env_ids = torch.arange(2)
    env.actions = torch.zeros(2, 34)
    env.last_actions, env.llast_actions = env.actions.clone(), env.actions.clone()
    env.actions[:, 17:] = 999  # Pos's executed-torque compatibility slots.
    env.action_queue = torch.zeros(2, 3, 34)
    env.action_delay = torch.tensor([0, 2])
    env.simulator = SimpleNamespace(default_dof_pos=torch.zeros(1, 19))
    env.dof_tracking_target = torch.zeros(2, 19)
    env._facet_observations = object()  # Only exercise the issued-action hook.
    issued = torch.linspace(-1, 1, 34).reshape(2, 17)
    executed = env._pre_sim_step(issued)
    torch.testing.assert_close(env._facet_issued_action, issued.clamp(-.5, .5))
    assert executed[1].count_nonzero() == 0
    assert executed[:, 17:].count_nonzero() == 0
    assert env._facet_issued_action[1].count_nonzero() > 0
