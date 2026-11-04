"""Collection contracts independent of Genesis/GPU imports."""
import json
import random
from unittest.mock import patch
import numpy as np
import pytest
import torch
from liquid_collection.payload import supported_bounds, sample_payload, validate_payload, geometry
from liquid_collection.physics import aggregates, coupling_impulse, rotate, shift_moment, CausalClock, isolated_window
from liquid_collection.plan import make_plan, run_plan
from liquid_collection.storage import ChunkWriter, atomic_json, verified_complete


def test_bounds_and_rejection_include_overlapped_lid():
    bounds = supported_bounds()
    assert bounds['rho'] == [730, 1000]
    assert bounds['scale_z'] == [1.2, 2.2]
    rng = random.Random(4)
    for _ in range(100):
        p = sample_payload(rng)
        g = validate_payload(p, .065)
        assert min(g['clearance']) >= .007-1e-12
        np.testing.assert_allclose(g['fill_size'], np.array(g['interior'])-p['offset'])
        assert g['mount_translation_B'][2] == .065
    p['offset'] = .02  # side clearance 10 mm, overlapped lid only 3 mm
    with pytest.raises(ValueError, match='Unsafe'):
        validate_payload(p, .065)
    with pytest.raises(ValueError, match='No safe'):
        sample_payload(rng, safety_margin=1, attempts=5)
    with pytest.raises(ValueError, match='positive|Invalid'):
        validate_payload(p, .065, particle_size=-1)


def test_expansions_never_expand_mounting_bounds():
    expanded = supported_bounds(.10, .02)
    base = supported_bounds()
    assert expanded['mount_x'] == base['mount_x'] == [0, .05]
    assert expanded['mount_y'] == base['mount_y']
    assert expanded['rho'] == [657., 1100.]
    rng = random.Random(9)
    p = sample_payload(rng, property_expansion=.1, geometry_expansion=.02)
    validate_payload(p, .06, property_expansion=.1, geometry_expansion=.02)
    p['mount_offset'][0] = .05001
    with pytest.raises(ValueError, match='mount_x'):
        validate_payload(p, .06, property_expansion=.1, geometry_expansion=.02)


def test_box_walls_match_cavity_and_overlap_lid():
    from liquid_collection.payload import box_parts
    p = sample_payload(random.Random(1))
    g = geometry(p, .06)
    parts = box_parts(p, .06)
    assert len(parts) == 6
    bottom, lid = parts[0], parts[-1]
    assert bottom['pos'][2]+bottom['size'][2]/2 == pytest.approx(g['lower_B'][2])
    assert lid['pos'][2]-lid['size'][2]/2 == pytest.approx(g['upper_B'][2])
    for part in parts[1:-1]:
        axis = 0 if part['name'].startswith('x') else 1
        sign = -1 if '-1' in part['name'] else 1
        face = part['pos'][axis]-sign*part['size'][axis]/2
        assert face == pytest.approx(g['lower_B' if sign < 0 else 'upper_B'][axis])
        assert part['pos'][2]+part['size'][2]/2 > g['upper_B'][2]
    import math
    sx, sy, sz = g['scale']
    expected = 200*((.20*sx)*(.15*sy)*(.20*sz)-(.17*sx)*(.12*sy)*(.185*sz)
                    +(.20*sx)*(.15*sy)*.015)
    actual = sum(math.prod(part['size'])*part.get('rho', 200.) for part in parts)
    assert actual == pytest.approx(expected)


def test_rotation_origin_shift_and_rigidly_carried_angular_momentum():
    q = torch.tensor([[2**-.5, 0, 0, 2**-.5]], dtype=torch.float64)
    x = torch.tensor([[[1., 0, 0], [-1, 0, 0]]], dtype=torch.float64)
    omega = torch.tensor([[0., 0, 2]], dtype=torch.float64)
    uO = torch.tensor([[3., 4, 5]], dtype=torch.float64)
    origin = torch.tensor([[6., 7, 8]], dtype=torch.float64)
    u = uO[:, None]+torch.linalg.cross(omega[:, None].expand_as(x), x)
    a = aggregates(x+origin[:, None], u, torch.tensor([[2., 2.]], dtype=torch.float64), origin, q, uO, omega)
    torch.testing.assert_close(a['r_B'], torch.zeros_like(origin))
    torch.testing.assert_close(a['v_B'], torch.zeros_like(origin))
    torch.testing.assert_close(a['S_B'], torch.tensor([[0., 0, 8.]], dtype=torch.float64))
    torch.testing.assert_close(rotate(q, rotate(q, origin), True), origin)
    force = torch.tensor([[0., 0, -10.]], dtype=torch.float64)
    shifted = shift_moment(force, torch.zeros_like(force), torch.tensor([[1., 0, 0.]], dtype=torch.float64), torch.zeros_like(force))
    torch.testing.assert_close(shifted, torch.tensor([[0., 10., 0.]], dtype=torch.float64))


def test_coupling_static_weight_freefall_and_nonuniform_mass():
    x = torch.tensor([[[1., 0, 0], [2., 0, 0]]])
    mass = torch.tensor([[1., 2.]])
    g = torch.tensor([0., 0, -9.81])
    dt = .001
    before = (g*dt).expand_as(x)
    force, moment, moment_world = coupling_impulse(x, before, torch.zeros_like(x), mass, torch.zeros(1, 3))
    torch.testing.assert_close(force/dt, (3*g)[None])
    torch.testing.assert_close(moment/dt, torch.tensor([[0., 5*9.81, 0]]))
    torch.testing.assert_close(moment, moment_world)
    free, _, _ = coupling_impulse(x, before, before, mass, torch.zeros(1, 3))
    assert free.eq(0).all()


def test_completed_interval_hold_rotation_and_partial_reset():
    v = torch.zeros(2, 6)
    clock = CausalClock(v, 2)
    assert not clock.advance(v+1, .01).any()
    assert clock.held.eq(0).all()
    assert clock.advance(v+2, .01).all()
    torch.testing.assert_close(clock.held, torch.full_like(v, 100))
    clock.advance(v+3, .01)
    torch.testing.assert_close(clock.held, torch.full_like(v, 100))
    clock.reset(torch.tensor([0]), v+7)
    assert clock.held[0].eq(0).all() and clock.held[1].eq(100).all()
    assert clock.episode.tolist() == [1, 0]
    assert clock.valid.tolist() == [False, True]
    assert isolated_window(torch.tensor([[0, 0], [1, 0]]), torch.ones(2, 2, dtype=torch.bool)).tolist() == [False, True]


def test_chunk_bounds_completion_and_corruption(tmp_path):
    w = ChunkWriter(tmp_path, chunk_size=3, max_bytes=40)
    for i in range(5):
        w.append(dict(x=np.ones(4, dtype=np.float32)*i))
        assert w.bytes <= 40
    assert not (tmp_path/'result.json').exists()
    w.finish(dict(status='complete'))
    assert verified_complete(tmp_path)
    values = np.concatenate([np.load(tmp_path/c['file'])['x'] for c in w.chunks])
    np.testing.assert_array_equal(values[:, 0], np.arange(5))
    (tmp_path/w.chunks[0]['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Corrupt'):
        verified_complete(tmp_path)


def test_writer_failure_never_marks_complete(tmp_path):
    w = ChunkWriter(tmp_path, chunk_size=2)
    w.append(dict(x=np.ones(4)))
    with patch('liquid_collection.storage.np.savez', side_effect=OSError('disk full')):
        with pytest.raises(OSError, match='disk full'):
            w.flush()
    with pytest.raises(RuntimeError):
        w.finish(dict(status='complete'))
    assert not (tmp_path/'result.json').exists()


def manifest(tmp_path):
    (tmp_path/'weights').write_bytes(b'weights')
    (tmp_path/'config.py').write_text('# literal config')
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(policies=[dict(id=f'p{i}', robot=robot,
        checkpoint='weights', config='config.py') for i, robot in enumerate(('go1', 'go1', 'go2'))])))
    return path


def test_plan_reproducible_multiple_policies_and_resume(tmp_path):
    source = manifest(tmp_path)
    plan = make_plan(source, rounds=12)
    assert plan == make_plan(source, rounds=12)
    assert len({r['policy']['id'] for r in plan['rounds']}) == 3
    assert plan['rounds'][:2] == make_plan(source, rounds=2)['rounds']
    output = tmp_path/'output'
    output.mkdir()
    atomic_json(output/'plan.json', plan)
    for r in plan['rounds']:
        d = output/f"round-{r['round']:05d}"/'collection'/'attempt-0000'
        w = ChunkWriter(d)
        w.append(dict(x=np.ones(1)))
        w.finish(dict(status='complete'))
    with patch('liquid_collection.plan.subprocess.Popen', side_effect=AssertionError('must skip completed')):
        assert run_plan(plan, output, resume=True) == 0
    plan['seed'] += 1
    with pytest.raises(ValueError, match='differs'):
        run_plan(plan, output, resume=True)


def test_selected_round_preserves_full_plan_and_resumes(tmp_path):
    plan = make_plan(manifest(tmp_path), rounds=3)
    output = tmp_path/'output'
    with patch('liquid_collection.plan.subprocess.Popen') as popen:
        popen.return_value.wait.return_value = 2
        assert run_plan(plan, output, round_index=0) == 2
        assert run_plan(plan, output, resume=True, round_index=1) == 2
        assert popen.call_count == 2
    assert json.loads((output/'plan.json').read_text()) == plan
    assert not (output/'round-00002').exists()
    prior = output/'round-00001/collection/attempt-0000'
    w = ChunkWriter(prior)
    w.append(dict(x=np.ones(1)))
    w.finish(dict(status='complete'))
    with patch('liquid_collection.plan.subprocess.Popen', side_effect=AssertionError('must skip completed')):
        assert run_plan(plan, output, resume=True, round_index=1) == 0
    with pytest.raises(ValueError, match='Round index'):
        run_plan(plan, output, resume=True, round_index=3)


def test_incomplete_resume_preserves_previous_attempt(tmp_path):
    plan = make_plan(manifest(tmp_path), rounds=1)
    output = tmp_path/'output'
    output.mkdir()
    atomic_json(output/'plan.json', plan)
    prior = output/'round-00000/collection/attempt-0000'
    prior.mkdir(parents=True)
    atomic_json(prior/'status.json', dict(status='incomplete'))
    with patch('liquid_collection.plan.subprocess.Popen') as popen:
        popen.return_value.wait.return_value = 2
        assert run_plan(plan, output, resume=True) == 2
    assert json.loads((prior/'status.json').read_text())['status'] == 'incomplete'
    assert (prior.parent/'attempt-0001/status.json').exists()


def test_jvp_checks_every_history_tap_not_just_latest():
    from liquid_collection.audit import interior_mask
    # Candidate .012 is interior itself, but its older .002 tap hits an event.
    mask = interior_mask([.012, .013], [.002], 0, .1, history=3, margin=.00025)
    assert mask.tolist() == [False, True]
    assert not interior_mask([.005], [], 0, .1, history=3).any()
    knots = np.arange(0., 1.5, .0005)
    assert interior_mask([1.24525], knots, 0, 1.5).item()
    assert not interior_mask([1.24525], knots, 0, 1.5, margin=.00026).item()


def test_recorder_retains_terminal_before_partial_reset():
    from types import SimpleNamespace
    from liquid_collection.recorder import NativeRecorder
    r = NativeRecorder.__new__(NativeRecorder)
    r.env = SimpleNamespace(num_envs=2)
    r.device = 'cpu'
    r.completed = torch.zeros(2, dtype=torch.long)
    r.episode_supported = torch.zeros(2, dtype=torch.bool)
    r.forced_reset = False
    r.clock = CausalClock(torch.zeros(2, 6), 2)
    r.state = {'velocity_W': torch.ones(2, 6)}
    events = []
    r.emit = lambda state, **kw: events.append((state['velocity_W'].clone(), r.clock.episode.clone(), kw))
    r.original_reset = lambda ids: events.append('physical_reset')
    r.read_state = lambda: {'velocity_W': torch.zeros(2, 6)}
    r.reset(torch.tensor([1]))
    assert events[0][0].eq(1).all()
    assert events[0][2]['terminal'].tolist() == [False, True]
    assert events[1] == 'physical_reset'
    assert events[2][1].tolist() == [0, 1]
    assert r.completed.tolist() == [0, 1]
    assert r.episode_supported.tolist() == [False, True]


def test_benchmark_enforces_original_player_runtime_budget(tmp_path):
    plan = make_plan(manifest(tmp_path), benchmark=True)
    out = tmp_path/'benchmark'
    out.mkdir()
    atomic_json(out/'plan.json', plan)
    for mode, seconds in (('reference', 1.), ('baseline', .1), ('collection', .2)):
        w = ChunkWriter(out/'round-00000'/mode/'attempt-0000')
        w.finish(dict(status='complete', steady_seconds_per_policy_step=seconds))
    assert run_plan(plan, out, resume=True) == 0
    report = json.loads((out/'round-00000/benchmark.json').read_text())
    assert report['runtime_budget_passed'] and report['relative_to_original_player'] == .2
    reference = out/'round-00000/reference/attempt-0000/result.json'
    data = json.loads(reference.read_text())
    data['steady_seconds_per_policy_step'] = .1
    atomic_json(reference, data)
    assert run_plan(plan, out, resume=True) == 2


def test_streaming_history_audit_excludes_partial_reset_and_keeps_other_env(tmp_path):
    from liquid_collection.audit import audit
    w = ChunkWriter(tmp_path, chunk_size=32)
    ids = np.zeros(2, dtype=np.int64)
    for i in range(300):
        row = dict(time_s=np.array(i*.005), episode_id=ids.copy(),
                   interval_valid=np.ones(2, dtype=bool), state_valid=np.ones(2, dtype=bool),
                   held_valid=np.ones(2, dtype=bool), episode_supported=np.ones(2, dtype=bool),
                   target_grid_event=np.array(True),
                   force_closure_residual_impulse_W=np.zeros((2, 3)),
                   moment_closure_residual_impulse_W=np.zeros((2, 3)))
        for key in ('reset', 'terminal', 'policy_update', 'feature_update'):
            row[key] = np.zeros(2, dtype=bool)
        w.append(row)
        if i == 260:
            ids[0] += 1
            row.update(episode_id=ids.copy(), reset=np.array([True, False]), interval_valid=np.zeros(2, dtype=bool))
            w.append(row)
    w.finish(dict(status='complete'))
    result = audit(tmp_path)
    assert result['valid_249_sample_histories'] == 64
    assert result['valid_history_plus_200ms_windows'] == 12


def interval_row(step, dt=.00025):
    from liquid_collection.cadence import IMPULSES, ALL_FLAGS
    # Translating base with a changing orientation and a known world wrench.
    theta = step*.03
    q = torch.tensor([[np.cos(theta/2), 0., 0., np.sin(theta/2)]], dtype=torch.float64)
    row = dict(mass=torch.ones(1, dtype=torch.float64), episode_id=torch.zeros(1, dtype=torch.long),
               interval_valid=torch.ones(1, dtype=torch.bool),
               domain_contact=torch.zeros(1, dtype=torch.bool),
               max_wall_penetration_m=torch.tensor([step*.001], dtype=torch.float64),
               active_count=torch.tensor([30]), q_WB_wxyz=q,
               x_O_W=torch.tensor([[step*.01, 0., 0.]], dtype=torch.float64),
               velocity_W=torch.full((1, 6), step*dt, dtype=torch.float64))
    row.update({k: torch.ones(1, dtype=torch.bool) for k in ALL_FLAGS})
    row.update({k: torch.zeros(1, 3, dtype=torch.float64) for k in IMPULSES})
    row['coupling_force_impulse_W'][:] = torch.tensor([0., 0., -10.])*dt
    row['coupling_moment_impulse_world_origin_W'][:] = torch.tensor([0., 3., 0.])*dt
    row.update(coupling_origin_W=row['x_O_W'], coupling_q_WB_wxyz=q,
               coupling_moment_impulse_O_W=torch.zeros(1, 3),
               coupling_force_impulse_B=torch.zeros(1, 3), coupling_moment_impulse_O_B=torch.zeros(1, 3),
               base_acceleration_interval_average_W=torch.ones(1, 6))
    return row


def test_200hz_impulses_use_fixed_origin_and_endpoint_frame():
    from liquid_collection.cadence import GridAccumulator
    a = GridAccumulator(.00025)
    for step in range(1, 21):
        row = interval_row(step)
        a.push(row, torch.full((1, 6), (step-1)*.00025, dtype=torch.float64), step, step)
    reduced, tick, sequence = a.pop()
    assert (tick, sequence) == (20, 20)
    assert reduced['interval_native_steps'].item() == 20
    assert reduced['interval_duration_s'].item() == pytest.approx(.005)
    torch.testing.assert_close(reduced['x_O_W'], row['x_O_W'])
    torch.testing.assert_close(reduced['q_WB_wxyz'], row['q_WB_wxyz'])
    force = torch.tensor([[0., 0., -.05]], dtype=torch.float64)
    moment = torch.tensor([[0., .015, 0.]], dtype=torch.float64)-torch.linalg.cross(row['x_O_W'], force)
    torch.testing.assert_close(reduced['coupling_force_impulse_W'], force, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(reduced['coupling_moment_impulse_O_W'], moment, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(reduced['moment_interval_average_O_B'], rotate(row['q_WB_wxyz'], moment, True)/.005,
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(reduced['base_acceleration_interval_average_W'], torch.ones(1, 6, dtype=torch.float64))
    assert a.pop() is None


def test_200hz_keeps_transient_failures_and_partial_interval_duration():
    from liquid_collection.cadence import GridAccumulator
    a = GridAccumulator(.00025)
    for step in range(1, 4):
        row = interval_row(step)
        if step == 2:
            row['contained'][:] = False
            row['interval_valid'][:] = False
            row['domain_contact'][:] = True
            row['max_wall_penetration_m'][:] = .04
            row['active_count'][:] = 29
        a.push(row, torch.zeros(1, 6), step, step)
    partial, *_ = a.pop()
    assert partial['contained'].item()  # endpoint recovered; interval evidence did not
    assert not partial['interval_contained'].item()
    assert not partial['interval_valid'].item()
    assert partial['interval_domain_contact'].item()
    assert partial['interval_max_wall_penetration_m'].item() == pytest.approx(.04)
    assert partial['interval_min_active_count'].item() == 29
    assert partial['interval_duration_s'].item() == pytest.approx(.00075)
    row = interval_row(4)
    row['episode_id'][:] = 1
    a.push(row, row['velocity_W'], 4, 4)
    following, *_ = a.pop()
    assert following['interval_native_steps'].item() == 1
    assert following['interval_valid'].item()
    torch.testing.assert_close(following['coupling_force_impulse_W'], row['coupling_force_impulse_W'])


def test_resume_verifies_auxiliary_event_streams(tmp_path):
    w = ChunkWriter(tmp_path)
    e = ChunkWriter(tmp_path/'events')
    e.append(dict(time_s=np.array(0.)))
    e.finish(dict(status='complete'))
    w.finish(dict(status='complete', auxiliary_streams=['events']))
    assert verified_complete(tmp_path)
    (tmp_path/'events'/e.chunks[0]['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Corrupt'):
        verified_complete(tmp_path)


def test_200hz_recorder_splits_off_grid_reset_and_preserves_event_timing(tmp_path):
    from types import SimpleNamespace
    from liquid_collection.recorder import NativeRecorder
    from liquid_collection.cadence import GridAccumulator
    from liquid_collection.audit import stream_rows, audit
    r = NativeRecorder.__new__(NativeRecorder)
    r.env = SimpleNamespace(num_envs=1, commands=torch.zeros(1, 4))
    r.device, r.dt, r.grid_ticks, r.policy_ticks = 'cpu', .001, 5, 5
    r.save_mode, r.tick, r.sequence = '200hz', 0, 0
    r.writer = ChunkWriter(tmp_path)
    r.event_writer, r.boundary_writer = ChunkWriter(tmp_path/'events'), ChunkWriter(tmp_path/'boundaries')
    r.accumulator = GridAccumulator(r.dt)
    r.feature_phase_by_env, r.feature_phases = np.zeros(1, dtype=np.int64), {0}
    r._bad_counts, r.last_commands, r.snapshot = {}, None, None
    r.gravity = torch.zeros(1, 3, dtype=torch.float64)
    r.completed = torch.zeros(1, dtype=torch.long)
    r.episode_supported = torch.ones(1, dtype=torch.bool)
    r.forced_reset = False
    r.episodes_per_env = 1
    r.state = interval_row(0, r.dt)
    r.clock = CausalClock(r.state['velocity_W'], 2)
    r.emit(r.state, reset=torch.ones(1, dtype=torch.bool))
    r.policy_event()
    r.original_reset = lambda ids: None
    r.read_state = lambda: interval_row(0, r.dt)
    for tick in range(1, 6):
        r.tick = tick
        state = interval_row(tick, r.dt)
        event = r.clock.advance(state['velocity_W'], r.dt)
        r.emit(state, interval=state, feature=event)
        r.state = state
        if tick == 3:
            r.reset(torch.tensor([0]))
    r.finish_streams(dict(status='complete'))
    r.writer.finish(dict(status='complete', auxiliary_streams=['events', 'boundaries']))
    assert verified_complete(tmp_path)
    main = list(stream_rows(tmp_path))
    boundaries = list(stream_rows(tmp_path/'boundaries'))
    assert [float(x['time_s']) for x in main] == [0., .005]
    assert main[-1]['episode_id'].item() == 1
    assert not main[-1]['episode_selected'].item()
    assert not main[-1]['interval_valid'].item()
    assert main[-1]['interval_native_steps'].item() == 2
    assert boundaries[0]['interval_native_steps'].item() == 3
    assert boundaries[0]['episode_id'].item() == 0
    assert boundaries[1]['terminal'].item() and boundaries[2]['reset'].item()
    assert boundaries[1]['episode_selected'].item() and not boundaries[2]['episode_selected'].item()
    impulses = sum((x['coupling_force_impulse_W'] for x in main+boundaries), start=np.zeros((1, 3)))
    np.testing.assert_allclose(impulses, [[0., 0., -.05]], atol=1e-8)
    events = list(stream_rows(tmp_path/'events'))
    assert [float(x['time_s']) for x in events if x['feature_update'].any()] == [.002, .005]
    result = audit(tmp_path)
    assert result['native_intervals'] == 5
    assert result['resets'] == 2 and result['terminals'] == 1


def test_per_environment_budget_cannot_be_met_by_one_fast_environment():
    from liquid_collection.budget import budget_reached, episode_steps
    job = dict(episodes=4, episodes_per_env=2)
    assert not budget_reached(np.array([0, 4]), job)
    assert not budget_reached(np.array([1, 100]), job)
    assert budget_reached(np.array([2, 3]), job)
    assert budget_reached(np.array([0, 4]), dict(episodes=4))
    assert episode_steps(.025, .01) == 2
    assert episode_steps(.01, .01) == 1
    with pytest.raises(ValueError, match='at least one'):
        episode_steps(.005, .01)


def test_campaign_covers_all_platforms_and_resumes_immutable_plans(tmp_path):
    from liquid_collection.launch import campaign_plan, run_campaign
    source = manifest(tmp_path)
    (tmp_path/'config.py').write_text('class RobotCfg:\n    class control:\n        dt = .01\n')
    platforms = ['go1-dact', 'go2-dact']
    with patch.dict('liquid_collection.launch.PLATFORMS', {name: source for name in platforms}):
        campaign = campaign_plan(platforms, 3, 2, 4, .055)
    assert campaign['worker_runs'] == 8 and campaign['requested_episodes'] == 48
    a, b = campaign['plans'].values()
    assert [r['payload'] for r in a['rounds']] == [r['payload'] for r in b['rounds']]
    assert a['episodes_per_env'] == 2 and a['max_steps'] == 15
    assert a['save_mode'] == '200hz'
    out = tmp_path/'campaign'
    with patch('liquid_collection.launch.run_plan', return_value=0) as runner:
        assert run_campaign(campaign, out) == 0
        assert runner.call_count == 8
        assert [(call.args[1].name, call.kwargs['round_index']) for call in runner.call_args_list] == [
            (platform, index) for index in range(4) for platform in platforms]
        assert [call.kwargs['resume'] for call in runner.call_args_list] == [False, False] + [True]*6
        runner.reset_mock()
        assert run_campaign(campaign, out, resume=True) == 0
        assert runner.call_count == 8
        assert all(call.kwargs['resume'] for call in runner.call_args_list)
    changed = dict(campaign, rounds=10)
    with pytest.raises(ValueError, match='differ'):
        run_campaign(changed, out, resume=True)


def test_hardpact_campaign_rejected_before_any_policy_planning():
    from liquid_collection.launch import campaign_plan
    with patch('liquid_collection.launch.make_plan', side_effect=AssertionError('must reject before planning')):
        with pytest.raises(ValueError, match='no HardPACT collection adapter'):
            campaign_plan(['go1-dact', 'go2-hardpact'], 2, 1, 1, 5.)
