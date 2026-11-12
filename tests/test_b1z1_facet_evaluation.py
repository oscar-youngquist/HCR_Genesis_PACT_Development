"""Paired event specifications and failure-inclusive reporting, without simulation."""
from copy import deepcopy
import json
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'legged_gym/scripts'))
from eval_b1z1_facet import scenarios, summarize_trial, aggregate_trials, EVAL_CONFIG
import eval_b1z1_facet as evaluation


def test_event_protocol_is_trial_keyed_and_covers_conditions():
    cfg = dict(event_times=[3., 7., 11.], horizon_seconds=14.)
    batch = scenarios(2, range(36), cfg)
    assert {x.metadata['condition'] for x in batch} == {'ee-only', 'base-only', 'combined'}
    assert {x.metadata['motion'] for x in batch} == {'standing', 'walking'}
    assert {x.metadata['stiffness'] for x in batch} == {4., 16., 28.}
    for item in batch:
        single = scenarios(2, [item.metadata['trial_id']], cfg)[0]
        for t, magnitude in ((2.99, 0), (3., 20), (3.01, 20), (6.99, 20), (7., 35), (7.01, 35), (10.99, 35), (11., 0), (11.01, 0)):
            command, _, load, phase = item.at(t)
            np.testing.assert_array_equal(load, single.at(t)[2])
            expected = magnitude if item.metadata['condition'] != 'base-only' else 0
            assert np.linalg.norm(load[:3]) == pytest.approx(expected)
            assert np.linalg.norm(load[3:6]) == pytest.approx(.7*magnitude if item.metadata['condition'] != 'ee-only' else 0)
            if item.metadata['direction'] == 'opposing':
                assert load[0] <= 0 and load[3] <= 0
            else:
                assert load[1] >= 0 and load[4] >= 0


def test_failures_and_incomplete_release_remain_in_denominators():
    config = deepcopy(EVAL_CONFIG)
    config.update(event_times=[.2, .4, .6], horizon_seconds=1.)
    scenario = scenarios(0, [0], config)[0].metadata
    def row(step):
        return dict(time=(step+1)*.1, interval_start=step*.1, phase=min(step//2, 3),
            ee_error=np.zeros(3), velocity_error=np.zeros(2), yaw_error=0., applied_wrench=np.zeros(9),
            ee_base_position_world=np.ones(6)*step*.1, ee_base_velocity_world=np.ones(6), valid=True)
    rows = [row(i) for i in range(5)]
    rows[-1]['valid'] = False
    result = summarize_trial(rows, scenario, 10, .1, config, 'height')
    assert result['S_track'] == .4 and result['failed'] and result['valid_samples'] == 4
    assert result['ee_error_m'] == 0 and result['release_recovery_seconds'] is None
    assert not result['release_window_complete']
    complete = summarize_trial([row(i) for i in range(10)], scenario, 10, .1, config, '')
    assert not complete['failed'] and complete['S_track'] == 1.
    assert complete['release_recovery_seconds'] is None  # Only .4 s observed, less than dwell.
    zero = summarize_trial([dict(row(0), valid=False)], scenario, 10, .1, config, 'contact')
    assert zero['ee_error_m'] is None and zero['S_track'] == 0.
    combined = aggregate_trials([result, complete, zero])
    assert combined['physical_failure_rate']['mean'] == pytest.approx(2/3)
    assert combined['S_track']['mean'] == pytest.approx(1.4/3)
    assert combined['ee_error_m']['contributing_trials'] == 2
    assert combined['release_recovered_fraction']['mean'] is None
    assert combined['release_recovered_fraction']['censored_unrecovered'] == 1


@pytest.mark.parametrize('field,index', [('applied_wrench', 0), ('applied_wrench', 3),
    ('applied_wrench', 8), ('ee_error', 2), ('velocity_error', 0),
    ('ee_base_position_world', 1), ('ee_base_velocity_world', 4)])
@pytest.mark.parametrize('first_step', [False, True])
def test_nonfinite_failures_write_strict_reports(tmp_path, field, index, first_step):
    config = dict(EVAL_CONFIG, event_times=[.2, .4, .6], horizon_seconds=1.)
    scenario = scenarios(0, [0], config)[0].metadata
    rows = [dict(time=(i+1)*.1, interval_start=i*.1, phase=min(i//2, 3),
        ee_error=np.zeros(3), velocity_error=np.zeros(2), yaw_error=0.,
        applied_wrench=np.array([3., 4., 0., 0., 0., 2., 0., 0., 1.]),
        ee_base_position_world=np.zeros(6), ee_base_velocity_world=np.zeros(6), valid=True)
        for i in range(1 if first_step else 5)]
    rows[-1][field][index] = np.nan if index % 2 == 0 else np.inf
    rows[-1]['valid'] = False
    rows[-1]['failed'] = True
    trial = summarize_trial(rows, scenario, 10, .1, config, 'nonfinite')
    summary = dict(trials=1, failed_trials=1, metrics=aggregate_trials([trial]))
    json.dumps(trial, allow_nan=False)
    # Same writer as the evaluator; raw transition arrays are deliberately not sanitized.
    evaluation.write_reports(tmp_path, [trial], [scenario], summary)
    def reject_constant(value):
        raise AssertionError(value)
    loaded = json.loads((tmp_path/'trials.json').read_text(), parse_constant=reject_constant)[0]
    report = json.loads((tmp_path/'summary.json').read_text(), parse_constant=reject_constant)
    assert (tmp_path/'COMPLETE').exists()
    assert loaded['failed'] and loaded['termination_reason'] == 'nonfinite'
    assert loaded['S_track'] == pytest.approx((len(rows)-1)/10)
    assert report['metrics']['physical_failure_rate'] == dict(mean=1., attempted_trials=1)
    assert loaded['ee_error_m'] == (None if first_step else 0.)
    invalid_channel = index//3 if field == 'applied_wrench' and index < 6 else None
    for channel, prefix, force in ((0, '', 5.), (1, 'base_', 2.)):
        invalid = channel == invalid_channel
        count = len(rows)-int(invalid)
        assert loaded[prefix+'force_peak_N'] == (None if invalid else force)
        assert loaded['force_exposure_N_s'][channel] == (None if invalid else pytest.approx(force*len(rows)*.1))
        assert loaded['force_valid_samples'][channel] == count
        assert loaded['force_observed_seconds'][channel] == pytest.approx(count*.1)
        assert loaded['force_measurements_complete'][channel] == (not invalid)
        observed = loaded['observed_force_exposure_N_s'][channel]
        assert observed == (pytest.approx(force*count*.1) if count else None)
    assert not np.isfinite(rows[-1][field][index])


def test_finite_force_reports_preserve_values(tmp_path):
    cfg = dict(EVAL_CONFIG, event_times=[.2, .4, .6], horizon_seconds=1.)
    scenario = scenarios(0, [0], cfg)[0].metadata
    rows = [dict(time=(i+1)*.1, interval_start=i*.1, phase=min(i//2, 3), valid=True,
        ee_error=np.zeros(3), velocity_error=np.zeros(2), yaw_error=0.,
        applied_wrench=np.array([3., 4., 0., 0., 0., 2., 0., 0., 1.])) for i in range(10)]
    trial = summarize_trial(rows, scenario, 10, .1, cfg, '')
    evaluation.write_reports(tmp_path, [trial], [scenario], aggregate_trials([trial]))
    assert trial['force_peak_N'] == 5. and trial['base_force_peak_N'] == 2.
    assert trial['force_exposure_N_s'] == [5., 2.]
    assert trial['force_valid_samples'] == [10, 10]
    assert not trial['failed'] and trial['S_track'] == 1.


def test_nonfinite_reference_diagnostic_retains_physical_tracking(tmp_path):
    cfg = dict(EVAL_CONFIG, event_times=[.2, .4, .6], horizon_seconds=1.)
    scenario = scenarios(0, [0], cfg)[0].metadata
    errors = np.ones((2, 6))
    errors[0, 0] = np.nan
    row = dict(time=.1, interval_start=0., phase=0, valid=True,
        ee_error=np.zeros(3), velocity_error=np.zeros(2), yaw_error=0.,
        applied_wrench=np.zeros(9), facet_valid=np.ones(2, bool), facet_errors_squared=errors)
    trial = summarize_trial([row], scenario, 10, .1, cfg, '')
    evaluation.write_reports(tmp_path, [trial], [scenario], aggregate_trials([trial]))
    assert trial['facet_horizon_counts'] == [0, 1]
    assert trial['facet_horizon_mse'] == [[None]*6, [1.]*6]
    assert trial['ee_error_m'] == 0 and trial['S_track'] == .1
