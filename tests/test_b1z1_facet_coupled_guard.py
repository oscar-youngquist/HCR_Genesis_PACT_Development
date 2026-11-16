"""Virtual impulses/accepted-state regressions; no simulator or policy required."""
from dataclasses import asdict
import math
import pytest
import torch
from test_b1z1_facet_bounded import bounded
from legged_gym.envs.b1z1.b1z1_facet.reference import coupled_impulse, FACETReferenceBank


def packet(args):
    return tuple(args[k] for k in ('measured_xy', 'measured_yaw', 'command_world',
        'yaw_command', 'ee_local', 'ee_local_velocity', 'height'))


@pytest.mark.parametrize('alpha', [0., .4, 1.])
def test_boundary_effective_mass_and_accepted_increments(alpha):
    ref, args = bounded(alpha=alpha, max_offset=.2, base_speed_limit=20.)
    # Nominal center-relative x=.6; EE at the +.2 m offset boundary.
    previous = ref._state.clone()
    previous[:, 6] += .2
    incoming = previous.clone()
    incoming[:, 9] = 1.
    out, impulse, invalid = ref._constrain(previous, incoming, .005, .005, packet(args), None)
    expected = alpha*2/(4+alpha*2)
    torch.testing.assert_close(out[:, 9], out[:, 9]*0+expected, atol=1e-10, rtol=0)
    torch.testing.assert_close(out[:, 2], out[:, 9], atol=1e-10, rtol=0)
    torch.testing.assert_close(out[:, 6:9]-previous[:, 6:9], .005*out[:, 9:12])
    torch.testing.assert_close(out[:, :2]-previous[:, :2], .005*out[:, 2:4])
    assert not invalid.any()
    if alpha == 1:
        torch.testing.assert_close(2*out[:, 9:11]+4*out[:, 2:4], incoming[:, 9:11]*2)
        torch.testing.assert_close(out[:, 9], torch.full_like(out[:, 9], 1/3))
    assert (out[:, 9:12].square().sum(-1)+2*out[:, 2:4].square().sum(-1) <= 1+1e-10).all()


def test_yaw_effective_mass_matches_dense_jacobian_solve():
    ref, _ = bounded(alpha=1., yaw_reaction=True)
    lever = torch.tensor([[.8, .2, .3], [.6, -.4, .1]], dtype=torch.float64)
    change = torch.tensor([[.1, -.7, .2], [.3, .2, -.1]], dtype=torch.float64)
    p = ref._parameters
    c = torch.stack((-lever[:, 1], lever[:, 0], lever[:, 0]*0), -1)
    matrix = torch.diag_embed(torch.tensor([.75, .75, .5], dtype=torch.float64).expand(2, -1))+c[..., None]*c[:, None, :]/1.5
    impulse = coupled_impulse(change, lever, p, True)
    torch.testing.assert_close(impulse, torch.linalg.solve(matrix, change))


def test_frictionless_yaw_constraint_does_not_inject_tangential_energy():
    ref, _ = bounded(alpha=1., yaw_reaction=True)
    p = ref._parameters
    lever = torch.tensor([[1.1, .2, 0.], [1.1, .2, 0.]], dtype=torch.float64)
    velocity = torch.tensor([[1., 10., 0.], [1., -10., 0.]], dtype=torch.float64)
    change = torch.tensor([[-1., 0., 0.], [-1., 0., 0.]], dtype=torch.float64)
    impulse = coupled_impulse(change, lever, p, True, normal_only=True)
    ee = velocity+impulse/2
    base = -impulse[:, :2]/4
    yaw = -torch.linalg.cross(lever, impulse)[:, 2]/1.5
    initial_energy = velocity.square().sum(-1)
    final_energy = ee.square().sum(-1)+2*base.square().sum(-1)+.75*yaw.square()
    assert (final_energy <= initial_energy).all()
    torch.testing.assert_close(2*ee[:, :2]+4*base, 2*velocity[:, :2])


def test_fast_nominal_not_subject_to_offset_speed_cap():
    ref, args = bounded(alpha=1., radius=(.4, .95), max_offset=.2)
    args = {k: v.clone() if torch.is_tensor(v) else v for k, v in args.items()}
    args['ee_local'][:, 0] = 1.2  # .9 m from the .3 m workspace center.
    args['ee_local_velocity'][:, 1] = .9*2.4/1.875
    previous = ref._state.clone()
    previous[:, 6:9] = args['ee_local']
    previous[:, 9:12] = args['ee_local_velocity']
    out, _, invalid = ref._constrain(previous, previous.clone(), .001, .001, packet(args), None)
    _, _, velocity = ref._relative(out, .001, packet(args))
    torch.testing.assert_close(velocity, args['ee_local_velocity'])
    assert velocity.norm(dim=-1).min() > 1.
    assert not invalid.any()


@pytest.mark.parametrize('yaw_rate', [0., .6])
def test_fast_spherical_nominal_and_final_world_increments(yaw_rate):
    ref, args = bounded(yaw_reaction=True)
    args = {k: v.clone() if torch.is_tensor(v) else v for k, v in args.items()}
    # Quintic -1.2 -> +1.2 rad, radius .9 m, 1.875 seconds:
    # peak nominal speed 2.16 m/s must NOT be capped at 1 m/s.
    for u in (.1, .3, .5, .7, .9):
        theta = -1.2+2.4*(10*u**3-15*u**4+6*u**5)
        rate = 2.4*30*u*u*(1-u)**2/1.875
        q = torch.tensor([.9*math.cos(theta), .9*math.sin(theta), 0.], dtype=torch.float64)
        v = rate*torch.stack((-q[1], q[0], q[2]))
        args['ee_local'][:] = q+torch.tensor([.3, 0., 0.])
        args['ee_local_velocity'][:] = v
        previous = ref._state.clone()
        previous[:, 6:9] = args['ee_local']
        previous[:, 2] = .2
        previous[:, 5] = yaw_rate
        previous[:, 9:12] = v
        previous[:, 9] += .2-yaw_rate*previous[:, 7]
        previous[:, 10] += yaw_rate*previous[:, 6]
        out, _, invalid = ref._constrain(previous, previous.clone(), .001, .001, packet(args), None)
        _, _, relative = ref._relative(out, .001, packet(args))
        assert (relative-v).norm(dim=-1).max() <= 1.0001
        assert not invalid.any()
        torch.testing.assert_close(out[:, 6:9]-previous[:, 6:9], .001*out[:, 9:12])
        torch.testing.assert_close(out[:, :2]-previous[:, :2], .001*out[:, 2:4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA compiled-kernel parity')
def test_compiled_constraint_and_dynamics_match_eager():
    from legged_gym.envs.b1z1.b1z1_facet.reference import FACETReference
    ref, args = bounded(yaw_reaction=True)
    eager = FACETReference(2, 'cuda:0', ref.cfg)
    state = ref._state.float().cuda()
    inputs = tuple(v.float().cuda() for v in packet(args))
    loads = tuple(args[k].float().cuda()+.1 for k in ('ee_force', 'base_force', 'yaw_moment'))
    time = torch.tensor(.005, device='cuda:0')
    expected = eager._constrain(state, state, time, .005, inputs, None)
    actual = eager._constraint_step(state, state, time, .005, inputs, None)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    expected = eager._rhs(state, time, inputs, loads)
    actual = eager._dynamics_step(state, time, inputs, loads)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_history_constraint_diagnostics_terminal_and_partial_reset():
    ref, args = bounded()
    bank = FACETReferenceBank(ref, (1, 2))
    loads = (args['ee_force'], args['base_force'], args['yaw_moment'])
    for _ in range(2):
        bank.advance(ref._state, .02, packet(args), loads)
    before = bank.read()
    assert before['valid'].all()
    assert before['constraint_valid'].all()
    bank.clear(torch.tensor([0]))
    for key in before:
        torch.testing.assert_close(bank.read()[key], before[key])
    assert not bank.read_current()['valid'][0].any()
    assert bank.read_current()['valid'][1].all()


def test_checkpoint_profiles_and_no_force_release():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'legged_gym/scripts'))
    from eval_b1z1_facet import represented_profiles, scenarios
    from legged_gym.envs.b1z1.b1z1_facet.forces import FACETForceCfg
    ref, _ = bounded(sampling_mode='bounded_b1z1')
    profiles = represented_profiles(asdict(ref.cfg))
    assert [p['ee_stiffness'] for p in profiles] == [60, 90, 120]
    assert [p['base_stiffness'] for p in profiles] == [2, 16, 30]
    for p in profiles:
        assert p['ee_damping'] == pytest.approx(2*(p['ee_mass']*p['ee_stiffness'])**.5)
    cfg = dict(event_times=[3, 7, 11], horizon_seconds=14, impedance_profiles=profiles,
               force_protocol=asdict(FACETForceCfg()))
    batch = scenarios(0, range(48), cfg)
    assert {s.metadata['impedance_level'] for s in batch} == {'low', 'middle', 'high'}
    for s in batch:
        assert not s.at(11)[2].any()
        if s.metadata['condition'] == 'none':
            assert not s.at(3.25)[2].any()
        if s.metadata['condition'] in ('ee-only', 'combined'):
            assert abs(s.at(3.25)[2][:3]).sum() == pytest.approx(s.metadata['stiffness']*.04)
