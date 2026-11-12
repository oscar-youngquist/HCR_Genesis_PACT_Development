"""Stages 6-7: phase budgets, sensor-only deployment, correlated impedance."""
from copy import deepcopy
import json
import os
import subprocess
import sys
import torch
import pytest

from test_b1z1_facet_training import make_runner, snapshot, unchanged
from test_b1z1_facet_observations import packet_env
from rsl_rl.facet_model import FACETModel
from rsl_rl.facet_ppo import FACETPPO
from rsl_rl.facet_contract import contract_from_env
from rsl_rl.facet_deployment import StudentMean, SensorStudent, export_student, load_student
from legged_gym.envs.b1z1.b1z1_facet.reference import FACETReferenceCfg, FACETReference
from legged_gym.envs.b1z1.b1z1_facet.observations import FACETObservations, nominal_command
from legged_gym.envs.b1z1.z1_arm_kinematics import compute_z1_arm_fk


@pytest.mark.parametrize('phase,updates', [('teacher', 1), ('adaptation', 3), ('student', 5)])
def test_phase_resume_and_exact_budget(tmp_path, phase, updates):
    runner = make_runner(2)
    runner.alg.cfg['adaptation_iterations'] = 2
    runner._schedules = lambda iteration, infos: {}
    runner.learn(updates)
    assert runner.alg.phase == phase
    expected = dict(teacher=min(updates, 2), adaptation=min(max(updates-2, 0), 2), student=max(updates-4, 0))
    assert runner.phase_updates == expected
    assert runner.phase_interactions == {k: v*16 for k, v in expected.items()}
    runner.env._staged_force_curriculum.beta = .37
    path = tmp_path/'state.pt'
    runner.save(path)
    resumed = make_runner(999)  # Saved schedule, not constructor defaults, wins.
    resumed.load(path)
    assert resumed.alg.cfg['student_start_iteration'] == 2
    assert resumed.alg.cfg['adaptation_iterations'] == 2
    assert resumed.alg.phase == phase and resumed.phase_interactions == runner.phase_interactions
    assert unchanged(resumed.model, snapshot(runner.model))
    assert unchanged(resumed.alg.ema, snapshot(runner.alg.ema))
    assert resumed.env._staged_force_curriculum.beta == .37
    for name in ('optimizer', 'estimator_optimizer'):
        old, new = getattr(runner.alg, name).state_dict(), getattr(resumed.alg, name).state_dict()
        for key, values in old['state'].items():
            for field, value in values.items():
                torch.testing.assert_close(value, new['state'][key][field])
    resumed._schedules = lambda iteration, infos: {}
    resumed.learn(6)
    assert resumed.phase_interactions == dict(teacher=32, adaptation=32, student=32)
    assert resumed.simulated_seconds == pytest.approx(96*.02)


def test_adaptation_only_estimator_and_online_collection():
    runner = make_runner(1)
    runner.alg.cfg['adaptation_iterations'] = 2
    runner._schedules = lambda *args: {}
    runner.learn(1)
    assert runner.alg.phase == 'adaptation'
    before = {k: snapshot(getattr(runner.model, k)) for k in ('teacher', 'student', 'critic', 'priv_encoder', 'force_encoder', 'obs_norm', 'priv_norm', 'wrench_norm')}
    estimator, ema = snapshot(runner.model.estimator), snapshot(runner.alg.ema)
    optimizer = deepcopy(runner.alg.optimizer.state_dict())
    packet = runner._normalize(runner.env.packet)
    with torch.no_grad():
        p, f, _, _ = runner.model.estimator(packet['obs'], runner.hidden, runner.env.packet['is_init'])
    storage, _ = runner.collect()
    torch.testing.assert_close(storage.data['features'][0], torch.cat((p, f), -1))
    runner.alg.update(storage, 1)
    assert all(unchanged(getattr(runner.model, k), v) for k, v in before.items())
    assert not unchanged(runner.model.estimator, estimator) and not unchanged(runner.alg.ema, ema)
    for key, value in optimizer['state'].items():
        for name, tensor in value.items():
            torch.testing.assert_close(runner.alg.optimizer.state_dict()['state'][key][name], tensor)
    student = snapshot(runner.model.student)
    assert runner.alg.switch_to_student()
    assert unchanged(runner.model.student, student)


def test_legacy_requires_metadata_and_never_recopies_student(tmp_path):
    runner = make_runner(1)
    runner._schedules = lambda *args: {}
    runner.learn(2)
    path = tmp_path/'old.pt'
    runner.save(path)
    state = torch.load(path, weights_only=False)
    contract = state.pop('contract')
    state['version'] = 1
    state['algorithm'].pop('student_initialized')
    state['config']['algorithm'].pop('adaptation_iterations')
    torch.save(state, path)
    resumed = make_runner()
    with pytest.raises(ValueError, match='metadata'):
        resumed.load(path)
    resumed.load(path, legacy_contract=contract)
    assert resumed.alg.student_initialized and unchanged(resumed.model.student, snapshot(runner.model.student))
    assert not resumed.accounting_exact


def test_upstream_sampling_correlations_partial_resets_and_stability():
    shared_rng = torch.random.get_rng_state().clone()
    cfg = FACETReferenceCfg(sampling_mode='upstream_b1z1', seed=42)
    refs = [FACETReference(4096, 'cpu', cfg) for _ in range(2)]
    ids = torch.arange(4096)
    def reset(ref, ids):
        xyz, yaw = torch.zeros(len(ids), 3), torch.zeros(len(ids))
        ref.reset(ids, xyz, xyz, yaw, yaw, xyz, xyz)
    for ref in refs:
        reset(ref, ids)
    p = refs[0]._parameters
    assert set(p['base_mass'].unique().tolist()) == {2., 4., 8.}
    for name in p:
        torch.testing.assert_close(p[name], refs[1]._parameters[name])
    torch.testing.assert_close(p['base_damping'], 1.8*p['base_stiffness'].sqrt())
    torch.testing.assert_close(p['yaw_damping'], p['base_damping'])
    torch.testing.assert_close(p['yaw_stiffness'], p['base_stiffness'])
    torch.testing.assert_close(p['ee_damping'], 1.8*p['ee_stiffness'].sqrt())
    assert (p['ee_mass'] == 2).all() and (p['yaw_inertia'] == 1.5).all()
    assert p['base_stiffness'].min() >= 2 and p['base_stiffness'].max() <= 30
    untouched = {k: v[1:].clone() for k, v in p.items()}
    reset(refs[0], ids[:1])
    for k in p:
        torch.testing.assert_close(untouched[k], p[k][1:])
    assert refs[0]._max_dt <= .005
    assert torch.equal(shared_rng, torch.random.get_rng_state())


def sensor_fixture(n=3, device='cpu'):
    env = packet_env(n, device)
    env.cfg.noise.add_noise = False
    env.cfg.init_state = __import__('types').SimpleNamespace(default_joint_angles={k: 0. for k in env.cfg.asset.dof_names})
    env.cfg.control = __import__('types').SimpleNamespace(dt=.02)
    env.cfg.domain_rand = __import__('types').SimpleNamespace()
    env.cfg.normalization = __import__('types').SimpleNamespace(clip_actions=2.)
    env.z1_joint_offsets = torch.randn(6, 3, device=device)*.1
    env.z1_joint_axes = torch.zeros(6, 3, device=device)
    env.z1_link00_offset = torch.tensor([.3, 0, .09], device=device)
    env.z1_ee_offset = torch.tensor([.186, 0, 0], device=device)
    env.compute_z1_arm_fk = lambda q: compute_z1_arm_fk(q, env.z1_joint_offsets, env.z1_joint_axes, env.z1_link00_offset, env.z1_ee_offset)
    collector = FACETObservations(env)
    env.get_facet_observation_schema = lambda: collector.schema
    contract = contract_from_env(env)
    model = FACETModel(collector.packet['observation'].shape[1], collector.packet['privileged'].shape[1]).to(device)
    model.student.load_state_dict(model.teacher.state_dict())
    module = SensorStudent(StudentMean(model, model.estimator, model.student), contract).to(device)
    return env, collector, contract, model, module


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_sensor_packet_export_and_hidden_parity(tmp_path, device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    env, collector, contract, model, module = sensor_fixture(device=device)
    ids = torch.arange(3, device=device)
    sensors = torch.zeros(3, 4, 41, device=device)
    actions = torch.zeros(3, 3, 17, device=device)
    hidden = torch.randn(3, 128, device=device)
    from legged_gym.envs.b1z1.b1z1_facet.reference import PARAMETERS
    from legged_gym.utils.math_utils import quat_rotate_inverse
    for step in range(4):
        init = torch.full((3, 1), step == 0, device=device, dtype=torch.bool)
        if step == 2:
            init[1] = True
        issued = torch.randn(3, 17, device=device).clamp(-2, 2)
        for reset in (False, True):
            selected = ids[init[:, 0] == reset]
            collector.collect(env, selected, issued, reset=reset)
        sim = env.simulator
        gravity = quat_rotate_inverse(sim.base_quat, torch.tensor([0., 0., -1.], device=device).expand(3, -1))
        params = torch.cat([env.facet_reference._parameters[k] for k in PARAMETERS], -1)
        packet, sensors, actions = module.observation(nominal_command(env), params, gravity, sim.dof_pos, sim.dof_vel, issued, init, sensors, actions)
        torch.testing.assert_close(packet, collector.packet['observation'])
        with torch.no_grad():
            expected, h, _ = model.student_action(packet, hidden, init)
            actual, hh = module.student(packet, hidden, init)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(hh, h)
        hidden = hh
    # Script contains only sensors/EMA/mean; importing or running it needs torch only.
    scripted = torch.jit.script(module.eval())
    path = tmp_path/'sensor.pt'
    scripted.save(str(path))
    loaded = torch.jit.load(str(path), map_location=device)
    a, h = loaded.student(packet, hidden, init)
    b, hh = module.student(packet, hidden, init)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(h, hh)
    assert not any(k.startswith(('teacher', 'critic', 'priv_encoder', 'force_encoder')) for k in loaded.state_dict())


def test_export_guard_and_clean_process(tmp_path):
    env, collector, contract, model, module = sensor_fixture()
    from legged_gym.envs.b1z1.b1z1_facet import B1Z1FACETCfgPPO
    from legged_gym.utils.helpers import class_to_dict
    alg = FACETPPO(model, class_to_dict(B1Z1FACETCfgPPO().algorithm))
    checkpoint = tmp_path/'teacher.pt'
    torch.save(dict(version=2, contract=contract, algorithm=alg.state_dict(), iteration=1,
                    config=class_to_dict(B1Z1FACETCfgPPO()), domain_curriculum={'uninitialized': -float('inf')}), checkpoint)
    before = checkpoint.read_bytes()
    output = tmp_path/'student.pt'
    with pytest.raises(ValueError, match='Uninitialized'):
        export_student(checkpoint, output)
    export_student(checkpoint, output, copy_teacher=True)
    assert checkpoint.read_bytes() == before
    loaded, metadata = load_student(output)
    assert metadata['copied_teacher']
    assert metadata['domain_curriculum']['uninitialized'] is None
    code = '''import torch,sys
m=torch.jit.load(sys.argv[1]); n=2
a,h,s,u=m(torch.zeros(n,9),torch.ones(n,10),torch.zeros(n,3),torch.zeros(n,19),torch.zeros(n,19),torch.zeros(n,17),torch.ones(n,1,dtype=torch.bool),torch.zeros(n,4,41),torch.zeros(n,3,17),torch.zeros(n,128))
assert a.shape==(n,17) and torch.isfinite(a).all()
assert not any(k.startswith(('legged_gym','isaaclab','omni')) for k in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code, str(output)], cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'SIMULATOR')}, check=True)


def test_live_phase_boundaries_keep_both_recurrent_states():
    runner = make_runner(1)
    runner.alg.cfg['adaptation_iterations'] = 2
    runner.collect()
    before = (runner.hidden.clone(), runner.ema_hidden.clone(), runner.env.resets)
    runner.current_learning_iteration = 1
    assert runner.maybe_transition() and runner.alg.phase == 'adaptation'
    torch.testing.assert_close(runner.hidden, before[0])
    torch.testing.assert_close(runner.ema_hidden, before[1])
    assert runner.env.resets == before[2]
    runner.current_learning_iteration = 3
    assert runner.maybe_transition() and runner.alg.phase == 'student'
    torch.testing.assert_close(runner.hidden, before[0])
    torch.testing.assert_close(runner.ema_hidden, before[1])
    assert runner.env.resets == before[2]


def test_resume_restores_budget_and_rejects_incompatible_physics(tmp_path):
    runner = make_runner(2)
    runner.cfg['runner']['max_iterations'] = 12
    runner.cfg['runner']['save_interval'] = 7
    path = tmp_path/'state.pt'
    runner.save(path)
    resumed = make_runner(100)
    resumed.load(path, schedule_overrides={'max_iterations': 20, 'adaptation_iterations': 3})
    assert resumed.cfg['runner']['max_iterations'] == 20 and resumed.save_interval == 7
    assert resumed.alg.cfg['student_start_iteration'] == 2 and resumed.alg.cfg['adaptation_iterations'] == 3
    resumed = make_runner()
    resumed.contract['control']['stiffness']['z1_waist'] = 999
    with pytest.raises(ValueError, match='contract: control'):
        resumed.load(path)


def test_sampled_reference_equilibrium_uses_observed_parameters():
    from test_b1z1_facet_reference import setup
    ref, args = setup(count=3, sampling_mode='upstream_b1z1', alpha=0)
    args['ee_force'] = torch.ones_like(args['ee_force'])*2
    for _ in range(1000):
        ref.advance(**args)
    torch.testing.assert_close(ref.read()['ee_position_world'], args['ee_force']/ref._parameters['ee_stiffness'], atol=2e-5, rtol=2e-5)
    assert not ref.read()['ee_position_world'].requires_grad
