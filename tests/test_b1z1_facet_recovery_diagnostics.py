"""Seed-aware reporting must not alter the virtual or physical task."""
import json
import torch
from test_b1z1_facet_bounded import bounded
from test_b1z1_facet_coupled_guard import packet
from legged_gym.envs.b1z1.b1z1_facet.reference import FACETReferenceBank
from legged_gym.envs.b1z1.b1z1_facet.recovery_diagnostics import checked, SeedDiagnostics, RecoveryRollout


def test_inherited_offset_recovery_same_trajectory():
    ref, args = bounded(alpha=0.)
    args = {k: v.clone() if torch.is_tensor(v) else v for k, v in args.items()}
    args['ee_local'][:, 1] = -.385
    seed = ref._state.clone()
    seed[:, 7] = .385
    bank = FACETReferenceBank(ref)
    values = {}
    for age in range(1, 33):
        bank.advance(seed, .02, packet(args), (args['ee_force'], args['base_force'], args['yaw_moment']))
        d = bank._seed_diagnostics.data
        assert not d['initial_flags'][:, 0, 1:].any()
        assert not d['end_flags'][:, 0, 1:3].any()
        torch.testing.assert_close(d['initial_offset_m'][:, 0], torch.full((2,), .77, dtype=torch.float64))
        if age in (8, 16, 32):
            values[age] = float(d['offset_m'][0, 0])
    assert .59 < values[8] < .64 and .42 < values[16] < .49
    assert values[32] <= .2001
    assert d['recovered'][:, 0].all() and not d['new_violation'][:, 0].any()
    assert (d['recovery_time_s'][:, 0] > .4).all()


def test_category_separation_and_new_violation_reentry():
    ref, args = bounded()
    state = ref._state.clone()
    flags, offset, excess, nominal = checked(ref, state, 0., packet(args))
    assert not flags.any()
    metadata = SeedDiagnostics(state[:, None])
    metadata.seed(0, (flags, offset, excess, nominal))
    active = torch.ones((2, 1), dtype=torch.bool)
    def update(f):
        metadata.update(active, .02, (f[:, None], offset[:, None], excess[:, None], nominal[:, None]),
                        f[:, None], torch.zeros(2, dtype=torch.bool))
    update(flags)
    assert not metadata.data['new_violation'].any()
    violated = flags.clone(); violated[:, 0] = True
    update(violated)
    assert metadata.data['new_violation'].all()
    metadata.seed(0, (violated, offset+.3, excess+.3, nominal))
    update(violated); update(flags); update(violated)
    assert metadata.data['recovered'].all() and metadata.data['reentered'].all()
    torch.testing.assert_close(metadata.data['recovery_time_s'], torch.full((2, 1), .04, dtype=torch.float64))
    state[:, 6] = 1.4
    assert checked(ref, state, 0., packet(args))[0][:, 1].all()
    state = ref._state.clone(); state[:, 9] = 2
    speed = checked(ref, state, 0., packet(args))[0]
    assert speed[:, 3].all() and not speed[:, :3].any()
    state = ref._state.clone(); state[:, 8] = -1
    assert checked(ref, state, 0., packet(args))[0][:, 2].all()


def test_partial_reset_terminal_reads_censoring_and_empty_groups():
    ref, args = bounded()
    bank = FACETReferenceBank(ref, (1, 2, 4))
    seed = ref._state.clone(); seed[:, 7] += .5
    loads = (args['ee_force'], args['base_force'], args['yaw_moment'])
    bank.advance(seed, .02, packet(args), loads)
    before = bank.read()
    bank.clear(torch.tensor([0]))
    for key, value in before.items():
        torch.testing.assert_close(bank.read()[key], value)
    assert not bank.read_current()['valid'][0].any()
    assert bank.read_current()['valid'][1, 0]
    assert bank._seed_diagnostics.censored.tolist() == [0., 1.]
    reducer = RecoveryRollout(); reducer.add(before)
    report = reducer.report(bank.horizons)
    assert report['1']['all']['count'] == 2
    assert report['2']['all']['count'] == 0
    assert report['2']['all']['end_incompatible_fraction'] is None
    assert report['1']['all']['recovery_time_s_mean'] is None
    assert report['1']['loaded']['count'] == 0
    json.loads(json.dumps(report, allow_nan=False))
    for _ in range(5):
        bank.advance(seed, .02, packet(args), loads)
    assert bank._seed_diagnostics.censored[0] > 0


def test_diagnostics_on_off_exact_reference_and_tracking_parity():
    from legged_gym.envs.b1z1.b1z1_facet.tracking import tracking_values
    ref, args = bounded()
    off, _ = bounded(recovery_diagnostics=False)
    banks = FACETReferenceBank(ref), FACETReferenceBank(off)
    loads = (args['ee_force'].clone()+2, args['base_force'], args['yaw_moment'])
    for i in range(34):
        for bank in banks:
            bank.advance(ref._state, .02, packet(args), loads)
        left, right = (b.read() for b in banks)
        for key in right:
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
        rewards = [tracking_values(p['targets'], ref._state, p['valid'], ref._state.new_ones(6)) for p in (left, right)]
        for a, b in zip(*rewards):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        if i == 12:
            for bank in banks:
                bank.clear(torch.tensor([0]))
    assert banks[0]._seed_diagnostics.data['ever_loaded'][banks[0]._ages >= 0].all()


def test_nominal_motion_and_roll_pitch_load_exposure():
    ref, args = bounded()
    bank = FACETReferenceBank(ref, (1, 2))
    args['ee_local_velocity'] = torch.ones_like(args['ee_local'])*.1
    torque = torch.zeros_like(args['ee_force']); torque[:, 0] = 1
    bank.advance(ref._state, .02, packet(args), (args['ee_force'], args['base_force'], args['yaw_moment']), torque)
    d = bank.read()
    assert d['diagnostic_ever_loaded'][:, 0].all()
    assert (d['diagnostic_nominal_motion_m'][:, 0] > 0).all()
    assert bank._seed_diagnostics.exposure[1] == .04
    assert bank._seed_diagnostics.exposure[4] == .04


def test_hand_computed_rollout_conditioned_denominators():
    ref, args = bounded()
    bank = FACETReferenceBank(ref, (1, 2))
    bank.advance(ref._state, .02, packet(args), (args['ee_force'], args['base_force'], args['yaw_moment']))
    p = bank.read()
    # Two eligible age-1 observations: one initially feasible/newly violated,
    # one initially infeasible/recovered at .02 s. No eligible age-2 entries.
    p['diagnostic_initial_feasible'][1, 0] = False
    p['diagnostic_recovered'][1, 0] = True
    p['diagnostic_recovery_time_s'][1, 0] = .02
    p['diagnostic_recovery_age'][1, 0] = 1
    p['diagnostic_end_flags'][0, 0, 0] = True
    p['diagnostic_any_flags'][0, 0, 0] = True
    p['diagnostic_any_flags'][1, 0, 1] = True  # Workspace violation cleared before endpoint.
    reducer = RecoveryRollout()
    reducer.add(p); reducer.add(p)
    result = reducer.report((1, 2))
    assert result['1']['all']['count'] == 4
    assert result['1']['all']['recovered_count'] == 2
    assert abs(result['1']['all']['recovery_time_s_mean']-.02) < 1e-8
    assert result['1']['initially_feasible']['end_incompatible_fraction'] == 1
    assert result['1']['initially_infeasible']['end_incompatible_fraction'] == 0
    assert result['1']['all']['end_incompatible_fraction'] == .5
    assert result['1']['all']['any_incompatible_fraction'] == 1
    assert result['2']['all']['recovery_time_s_mean'] is None
