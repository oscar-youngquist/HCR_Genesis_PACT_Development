"""Paired event specifications and failure-inclusive reporting, without simulation."""
from copy import deepcopy
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'legged_gym/scripts'))
from eval_b1z1_facet import scenarios, summarize_trial, aggregate_trials, EVAL_CONFIG


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
