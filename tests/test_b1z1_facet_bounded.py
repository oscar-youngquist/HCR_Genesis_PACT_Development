"""Bounded FACET reference checks, without simulator or learned-performance claims."""
from dataclasses import replace
import pytest
import torch
from test_b1z1_facet_reference import setup
from legged_gym.envs.b1z1.b1z1_facet.reference import FACETReferenceCfg, FACETReferenceBank
from legged_gym.envs.b1z1.b1z1_facet.bounds import geometry, guard
from legged_gym.envs.b1z1.b1z1_facet.forces import FACETForceCfg, FACETForces


def bounded(**kwargs):
    ref, args = setup(bounded=True, formulation_version=2, ee_stiffness=90,
        ee_damping=2*(180)**.5, **kwargs)
    args['ee_local'] = args['ee_local'].clone()+torch.tensor([.9, 0., 0.])
    ref._state[:, 6:9] = args['ee_local']
    return ref, args


def test_bounds_velocity_and_release():
    ref, args = bounded(alpha=0)
    args['ee_force'] = torch.tensor([[0., 40., 0.]], dtype=torch.float64).expand(2, -1)
    for _ in range(40):
        ref.advance(**args)
    q, nominal, v = ref._relative(ref._state, 0, tuple(args[k] for k in (
        'measured_xy', 'measured_yaw', 'command_world', 'yaw_command', 'ee_local', 'ee_local_velocity', 'height')))
    assert geometry(q, nominal, ref.cfg)[0].min() >= -1e-4
    assert (q-nominal).norm(dim=-1).max() <= ref.cfg.max_offset+1e-4
    assert v.norm(dim=-1).max() <= ref.cfg.ee_relative_speed_limit+1e-4
    before = q[:, 1].abs().mean()
    args['ee_force'] = torch.zeros_like(args['ee_force'])
    for _ in range(80):
        ref.advance(**args)
    assert ref._state[:, 7].abs().mean() < before


def test_collision_segment_and_tangential_velocity():
    cfg = FACETReferenceCfg(radius=(.01, 3.), pitch=(-1.5, 1.5), azimuth=(-3.14, 3.14),
                            max_offset=4., box_low=(-.2, -.2, -.2), box_high=(.2, .2, .2))
    start = torch.tensor([[.5, 0., 0.]])
    end = torch.tensor([[-.5, 0., 0.]])
    q, v, bad = guard(end, start, torch.tensor([[-1., .2, 0.]]), start, cfg)
    assert not bad.any() and q[0, 0] >= .2
    assert v[0, 0] == 0 and v[0, 1] == pytest.approx(.2)


def test_yaw_reaction_uses_internal_force_only():
    ref, args = bounded(yaw_reaction=True)
    ref._state[:, 7] = .1
    inputs = tuple(args[k] for k in ('measured_xy', 'measured_yaw', 'command_world',
                                    'yaw_command', 'ee_local', 'ee_local_velocity', 'height'))
    loads = tuple(args[k] for k in ('ee_force', 'base_force', 'yaw_moment'))
    derivative, force = ref._rhs(ref._state, 0, inputs, loads)
    torch.testing.assert_close(derivative[:, 5]*ref.cfg.yaw_inertia, .9*force[:, 6])
    # Changing physical EE loading does not directly add another base moment.
    derivative2, _ = ref._rhs(ref._state, 0, inputs, (loads[0]+20, *loads[1:]))
    torch.testing.assert_close(derivative[:, :6], derivative2[:, :6])


def test_history_all_ages_bounded_and_reset_snapshot():
    ref, args = bounded()
    bank = FACETReferenceBank(ref)
    inputs = tuple(args[k] for k in ('measured_xy', 'measured_yaw', 'command_world',
                                    'yaw_command', 'ee_local', 'ee_local_velocity', 'height'))
    loads = tuple(args[k] for k in ('ee_force', 'base_force', 'yaw_moment'))
    for _ in range(32):
        bank.advance(ref._state.clone(), .02, inputs, loads)
    assert bank.read()['valid'].all()
    saved = bank.read()
    bank.clear(torch.tensor([0]))
    assert not bank.read_current()['valid'][0].any()
    torch.testing.assert_close(saved['targets'], bank.read()['targets'])


def test_force_mixture_reproducibility_and_release():
    cfg = FACETForceCfg()
    a, b = (FACETForces(4096, 'cpu', cfg) for _ in range(2))
    t = torch.full((4096,), 2.25)
    stiffness = torch.full((4096, 1), 90.)
    ee, base = a.sample(t, stiffness, 0)
    other = b.sample(t, stiffness, 0)
    torch.testing.assert_close(ee, other[0]); torch.testing.assert_close(base, other[1])
    assert torch.bincount(a.family).min() > 850
    assert ee.norm(dim=-1).max() == pytest.approx(3.6)
    assert not ee[a.family == 0].any() and not base[a.family == 1].any()
    ee, base = a.sample(t*0+6.1, stiffness, 0)
    assert not ee.any() and not base.any()
    before = a.cycle.clone(); a.reset(torch.tensor([0]))
    torch.testing.assert_close(before[1:], a.cycle[1:])


def test_fixed_ppo_and_preserved_interaction_budget():
    from legged_gym.envs.b1z1.b1z1_facet import B1Z1FACETCfgPPO, B1Z1FACETCfg
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfgPPO
    cfg, env = B1Z1FACETCfgPPO(), B1Z1FACETCfg()
    a = cfg.algorithm
    assert (a.optimizer, a.learning_rate, a.entropy_coef, a.schedule) == ('Adam', 5e-4, .002, 'fixed')
    assert not a.use_adaptive_entropy and not a.use_clipped_value_loss
    assert cfg.runner.num_steps_per_env == 32
    assert cfg.runner.max_iterations*32 == B1Z1PACTCfgPPO.runner.max_iterations*24
    assert a.student_start_iteration*32 == 24000*24
    assert not env.domain_rand.push_robots


def test_sampling_mass_aware_damping():
    ref, _ = setup(sampling_mode='bounded_b1z1')
    for body, mass in (('base', 'base_mass'), ('ee', 'ee_mass'), ('yaw', 'yaw_inertia')):
        p = ref._parameters
        torch.testing.assert_close(p[body+'_damping'], 2*(p[mass]*p[body+'_stiffness']).sqrt())


def test_moving_offset_cap_and_inward_velocity():
    cfg = FACETReferenceCfg()
    nominal = torch.tensor([[.6, 0., 0.]])
    q = nominal+torch.tensor([[0., .2, 0.]])
    moving = torch.tensor([[0., .1, 0.]])
    accepted, velocity, invalid = guard(q, nominal, moving.clone(), q, cfg, moving)
    assert not invalid.any()
    torch.testing.assert_close(velocity, moving)
    _, inward, _ = guard(q, nominal, -moving, q, cfg, moving)
    torch.testing.assert_close(inward, -moving)


def test_force_publication_once_and_gates():
    from types import SimpleNamespace
    from legged_gym.envs.b1z1.b1z1_facet import B1Z1FACET, B1Z1FACETCfg
    env = B1Z1FACET.__new__(B1Z1FACET)
    env.cfg = B1Z1FACETCfg()
    env.dt, env.training_iteration = .02, 0
    env.episode_length_buf = torch.tensor([112, 112])
    env._facet_forces = FACETForces(2, 'cpu', replace(FACETForceCfg(), mixture=(0., 0., 0., 1.)))
    env.facet_reference = SimpleNamespace(_parameters={'ee_stiffness': torch.full((2, 1), 90.)})
    env.ee_force_ext_world, env.base_force_ext_world, env.base_torque_ext_world = (torch.zeros(2, 3) for _ in range(3))
    calls = []
    env.simulator = SimpleNamespace(apply_ee_force=lambda x: calls.append(('ee', x.clone())),
        apply_base_force=lambda x: calls.append(('base', x.clone())),
        apply_base_torque=lambda x: calls.append(('moment', x.clone())))
    env._sample_facet_forces()
    env._push_gripper(torch.arange(2)); env._push_robot_base(torch.arange(2))
    assert [c[0] for c in calls] == ['ee', 'base', 'moment']
    assert calls[0][1].norm() > 0 and calls[1][1].norm() > 0 and not calls[2][1].any()
    env.cfg.commands.apply_ee_external_forces = False
    env.cfg.commands.apply_base_external_forces = False
    env._sample_facet_forces()
    env._push_gripper(torch.arange(2)); env._push_robot_base(torch.arange(2))
    assert not env.ee_force_ext_world.any() and not env.base_force_ext_world.any()
    env.cfg.commands.push_gripper_stators = False
    env.cfg.commands.apply_base_external_forces = True
    env._sample_facet_forces()
    env._push_robot_base(torch.arange(2))
    assert not env.ee_force_ext_world.any() and env.base_force_ext_world.norm() > 0


def test_bounded_speed_and_timestep_convergence():
    results = []
    for dt in (.02, .01):
        ref, args = bounded(yaw_reaction=True, alpha=1.)
        args['base_force'] = torch.full_like(args['base_force'], 200.)
        args['yaw_moment'] = torch.full_like(args['yaw_moment'], 100.)
        args['ee_force'] = torch.full_like(args['ee_force'], 20.)
        args['dt'] = dt
        for _ in range(round(.2/dt)):
            ref.advance(**args)
            out = ref.read()
            assert out['base_velocity_xy'].norm(dim=-1).max() <= 2+1e-6
            assert out['yaw_rate'].abs().max() <= 2+1e-6
            assert out['ee_relative_speed'].max() <= 1+1e-6
        results.append(ref._state)
    torch.testing.assert_close(results[0], results[1], atol=.015, rtol=.03)


def test_vertical_ground_reaction_has_no_planar_escape():
    from legged_gym.envs.b1z1.b1z1_facet.bounds import restoring_force
    cfg = FACETReferenceCfg(radius=(.1, 2.))
    q = torch.tensor([[.6, 0., -.69]])
    force = restoring_force(q, q, torch.tensor([[0., 0., -.2]]), cfg)
    assert force[0, 2] > 0
    assert force[0, :2].count_nonzero() == 0


def test_speed_clip_never_reverses_base_increment():
    ref, args = bounded(alpha=0)
    previous = ref._state.clone()
    proposed = previous.clone()
    proposed[:, 0] = .25; proposed[:, 2] = 100.
    inputs = tuple(args[k] for k in ('measured_xy', 'measured_yaw', 'command_world',
                                    'yaw_command', 'ee_local', 'ee_local_velocity', 'height'))
    accepted, _, _ = ref._constrain(previous, proposed, .005, .005, inputs, None)
    assert (accepted[:, 0] >= 0).all()
    assert (accepted[:, 0] <= .005*2).all()


def test_formulation_resume_rejected_before_training():
    from rsl_rl.facet_contract import validate_contract
    with pytest.raises(ValueError, match='formulation'):
        validate_contract({'reference': {'formulation_version': 1}}, {'reference': {'formulation_version': 2}})


def test_compliant_base_participates_more_under_ee_pull():
    positions = []
    for stiffness in (2., 100.):
        ref, args = bounded(alpha=1., base_stiffness=stiffness, base_damping=2*(4*stiffness)**.5)
        args['ee_force'] = torch.tensor([[12., 0., 0.]], dtype=torch.float64).expand(2, -1)
        for _ in range(30):
            ref.advance(**args)
        positions.append(ref._state[:, 0])
    assert (positions[0] > positions[1]).all()


@pytest.mark.parametrize('kwargs', [{'pulse_duration': (0., 0.)}, {'mixture': (1., float('nan'), 0., 0.)},
    {'interior_offset': .3, 'boundary_offset': .1}, {'ee_duration': 9.}])
def test_invalid_force_configuration(kwargs):
    with pytest.raises(ValueError):
        FACETForces(2, 'cpu', replace(FACETForceCfg(), **kwargs))
