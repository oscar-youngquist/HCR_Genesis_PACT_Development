"""CPU-only Exp2 profiles/metrics; never construct a simulator or training runner."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluation2", ROOT / "legged_gym/scripts/eval_b1z1_exp2.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
u, c = e.tasks, e.EVAL_CONFIG


def scenario(profile="pushing", loading="loaded", trial=0):
    return u.Scenario(42, trial, profile, loading, c)


@pytest.mark.parametrize("index_type", ["list", "tuple", "numpy", "tensor"])
def test_metadata_index_containers(index_type):
    import json
    from types import SimpleNamespace

    factories = {"list": list, "tuple": tuple, "numpy": np.asarray}
    factory = pytest.importorskip("torch").tensor if index_type == "tensor" else factories[index_type]
    simulator = SimpleNamespace(
        _body_names=["base", "foot_a", "arm", "foot_b"],
        feet_indices=factory([3, 1]),
        _contact_indices=factory([2, 0, 3, 1]),
        _dof_indices=factory([6, 11, 16, 7]),
    )
    metadata = e.Experiment2.metadata(
        SimpleNamespace(dt=.02, simulator=simulator),
        SimpleNamespace(horizon_seconds=c["finish_s"]), c, {"training_seed": 42},
    )
    assert metadata["nonfoot_body_indices"] == [0, 2]
    assert metadata["articulation_to_contact_sensor_indices"] == [2, 0, 3, 1]
    assert metadata["configured_to_articulation_dof_indices"] == [6, 11, 16, 7]
    assert json.loads(json.dumps(metadata))["configured_to_articulation_dof_indices"] == [6, 11, 16, 7]


@pytest.mark.parametrize("profile,velocity,start,end", [
    ("pushing", .4, .50, .60), ("pulling", -.4, .60, .45), ("carrying", .4, .50, None)])
def test_profile_endpoints(profile, velocity, start, end):
    s = scenario(profile)
    for t in (0., 3., 19.5, 24., 28.):
        np.testing.assert_array_equal(s.at(t)[0], 0.)
    assert s.at(3.25)[0][0] == pytest.approx(velocity/2)
    assert s.at(3.5)[0][0] == velocity
    assert s.at(19.25)[0][0] == pytest.approx(velocity/2)
    assert s.at(7.)[1][0] == start
    if end is not None:
        assert s.at(11.)[1][0] == end
        assert s.at(9.)[1][0] == pytest.approx((start+end)/2)
    else:
        np.testing.assert_allclose(u.sphere_to_cart(s.at(11.)[1])-u.sphere_to_cart(s.at(7.)[1]), [.08, .05, 0.], atol=1e-15)
    np.testing.assert_allclose(s.at(28.)[1], s.at(11.)[1])
    for t in (7., 11.):
        assert np.linalg.norm((s.at(t+1e-4)[1]-s.at(t-1e-4)[1])/2e-4) < 1e-7


@pytest.mark.parametrize("profile,axis,sign", [("pushing", 0, -1), ("pulling", 0, 1), ("carrying", 2, -1)])
def test_load_timing_signs_frames(profile, axis, sign):
    s = scenario(profile)
    for t, force in ((0, 0), (7-1e-8, 0), (7, 25), (11, 45), (15, 65), (19.5, 65), (23-1e-8, 65), (23, 0)):
        load = s.at(t)[2]
        assert load[axis] == sign*force
        assert np.linalg.norm(load[:3]) == force
        assert not load[3:].any()
    s.set_heading(np.pi/2)
    assert s.at(15)[2][1 if axis == 0 else 2] == pytest.approx(sign*65)
    assert sum(s.metadata["events"][2]["exceeds_training_bounds"]) == 1
    assert c["finish_s"]-s.metadata["events"][-1]["time"] >= c["recovery_window_seconds"]


def test_pair_streams_and_subsets():
    selected = u.selected_batches(3, 2, "9,2,20")
    assert selected == [[9, 2], [20]]
    assert u.selected_batches(5, 2) == [[0, 1], [2, 3], [4]]
    for trial in (20, 9, 2):
        a, b = scenario(trial=trial), scenario(loading="unloaded", trial=trial)
        assert a.metadata["pair_id"] == b.metadata["pair_id"]
        assert a.metadata["motion_hash"] == b.metadata["motion_hash"]
        for stream in ("initial", "physical_initial", "noise", "_randomize_friction"):
            assert u.common.seed_for(42, trial, stream, 4) == e.exp1.util.seed_for(42, trial, stream, 4)
        for time in (0, 7, 9, 19, 23):
            np.testing.assert_array_equal(a.at(time)[0], b.at(time)[0])
            np.testing.assert_array_equal(a.at(time)[1], b.at(time)[1])
            assert not b.at(time)[2].any()
    with pytest.raises(ValueError):
        u.selected_batches(3, 2, "1,1,2")


def test_successor_references_and_stage_labels():
    row = {"commands_t": np.array([123.]), "ee_reference_t_world": np.array([456.])}
    e.Experiment2.augment_row(row, scenario(), 3., 3.25)
    assert row["command_nominal_t"][0] == 0
    assert row["command_nominal"][0] == pytest.approx(.2)
    assert row["commands_t"][0] == 123. and row["ee_reference_t_world"][0] == 456.
    e.Experiment2.augment_row(row, scenario(), 6.98, 7.)
    assert row["task_stage_interval"] == 2 and row["task_stage_successor"] == 3
    assert row["load_event_index"] == -1  # Load changes at the next interval, not retroactively.


def test_metrics_failure_and_complete_windows():
    s = scenario()
    metric = u.TrialMetrics(1400, .02, s, c)
    for i in range(1400):
        time = (i+1)*.02
        metric.add(time, [0., 0., 0.], [0., 0.], 0., s.stage(time))
    result = metric.finish("")
    assert result["S_track"] == 1 and not result["physical_failure"]
    assert len(result["event_metrics"]) == 4
    assert all(x["complete_window"] and x["recovered"] for x in result["event_metrics"])
    assert sum(x["prescribed_steps"] for x in result["stage_statistics"].values()) == 1400
    failed = u.TrialMetrics(1400, .02, s, c)
    failed.add(.02, [.05, 0., 0.], [0., 0.], 0., 0)
    failed.add(.04, [9., 0., 0.], [0., 0.], 0., 0, valid=False)
    result = failed.finish("height")
    assert result["S_track"] == 1/1400 and result["ee_error_m"] == .05
    assert all(not x["recovered"] and x["iae_m_s"] is None for x in result["event_metrics"])
    assert result["stage_statistics"]["load65_moving"]["S_track"] == 0.
    assert u.aggregate_seed([result])["42/pushing/loaded/loco-manipulation"]["physical_failure_rate"] == 1.


def test_torque_and_substep_moments():
    pd = np.array([[10., -4.], [-10., 4.]])
    ff = -pd/2
    executed = np.clip(pd+ff, -3, 3)
    stats = u.torque_statistics(pd, ff, executed)
    np.testing.assert_allclose(stats["pd_weighted_motor_mean_square"], [100., 16.])
    np.testing.assert_allclose(stats["executed_torque_mean_square"], [9., 4.])
    np.testing.assert_allclose(stats["opposing_channel_cancellation_mean"], [5., 2.])
    points = np.array([[[1., 0., 0.]], [[-1., 0., 0.]]])
    force = np.array([[[0., 2., 0.]], [[0., -2., 0.]]])
    moments = u.point_force_moment(points, force, np.zeros((2, 3)))
    np.testing.assert_allclose(moments.mean(0), [0., 0., 2.])
    assert not u.point_force_moment(points.mean(0), force.mean(0), np.zeros(3)).any()


def test_configuration_validation_isolation_and_dispatch():
    args = e.parser().parse_args(["--run", "run", "--output-dir", "/tmp/exp2"])
    assert (args.episodes, args.num_envs, args.horizon_seconds) == (50, 10, 28.)
    e.Experiment2.validate(c, args)
    assert e.EVAL_CONFIG["overrides"] == e.exp1.EVAL_CONFIG["overrides"]
    assert e.EVAL_CONFIG["overrides"] is not e.exp1.EVAL_CONFIG["overrides"]
    for task in ("b1z1_pact", "b1z1_ppo_pos", *u.ATTRIBUTION.values()):
        assert e.exp1.simulator_family(task) == "isaaclab_b1z1_pact"
    assert e.exp1.simulator_family("b1z1_unifp_original") == "isaaclab_b1z1_unifp"
    bad = copy.deepcopy(c)
    bad["profiles"]["pushing"]["radius_end"] = 1.
    with pytest.raises(ValueError, match="waypoint"):
        u.validate(bad, 28.)
    bad = copy.deepcopy(c)
    bad["load_times_s"][1] = 6.
    with pytest.raises(ValueError, match="ordered"):
        u.validate(bad, 28.)


def test_substep_recorder_moments_sensor_order_and_torque_identity():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace as NS
    adapter = e.Adapter.__new__(e.Adapter)
    pos = torch.tensor([[[0., 0., .5], [1., 0., 0.], [0., 1., .5]]])
    robot = NS(data=NS(body_com_pos_w=pos, body_link_pos_w=pos,
                      body_com_vel_w=torch.ones(1, 3, 6)))
    # Sensor order deliberately differs from articulation order.
    sensors = NS(data=NS(net_forces_w=torch.tensor([[[0., 0., 10.], [0., 0., 30.], [0., 0., 20.]]])))
    s = NS(_robot=robot, _contact_sensors=sensors, _contact_indices=torch.tensor([2, 0, 1]),
           feet_indices=torch.tensor([1]), _gripper_index=2, _base_link_index=0,
           _foot_contact_force_threshold=1., _external_force_world=torch.zeros(1, 3, 3),
           dof_pos=torch.zeros(1, 2), dof_vel=torch.zeros(1, 2),
           dof_pos_limits=torch.tensor([[-1., 1.], [-1., 1.]]), dof_vel_limits=torch.ones(2),
           torque_limits=torch.ones(2)*10, executed_torques=torch.tensor([[3., 4.]]))
    s._external_force_world[:, 2, 0] = 25.
    s._sensor_foot_forces_world = lambda: sensors.data.net_forces_w[:, [0]]
    adapter.sim, adapter.body_masses = s, torch.tensor([[2., 1., 1.]])
    adapter.pending = {"torque_request_motor_scaled": torch.tensor([[3., 4.]])}
    adapter.sensor_update = lambda *_: None
    adapter.substeps = []
    adapter.contact_update(.005)
    row = adapter.substeps[-1]
    com = (pos*adapter.body_masses[..., None]).sum(1)/4
    torch.testing.assert_close(row["robot_com_world"], com)
    np.testing.assert_allclose(row["ee_load_moment_world"].numpy(), u.point_force_moment(
        pos[:, [2]].numpy(), s._external_force_world[:, [2]].numpy(), com.numpy()))
    torch.testing.assert_close(row["body_contact_forces_raw_world"], sensors.data.net_forces_w[:, [2, 0, 1]])
    pd, ff = torch.tensor([[[10., -4.]], [[-10., 4.]]]), torch.tensor([[[-5., 2.]], [[5., -2.]]])
    stats = u.torque_statistics(pd, ff, (pd+ff).clamp(-3, 3), xp=torch)
    torch.testing.assert_close(stats["opposing_channel_cancellation_mean"], torch.tensor([[5., 2.]]))


def test_exp1_deterministic_fixture_and_config_isolation(monkeypatch):
    config = e.exp1.EVAL_CONFIG
    assert config["event_times"] == [3., 7., 11.]
    assert [config[k] for k in ("ee_tolerance_m", "velocity_tolerance_m_s", "yaw_tolerance_rad_s")] == [.10, .20, .30]
    # Fixed seed/trial fixture guards event stream, force layout, and boundary timing.
    fixture = [
        [63.17778810695938, 32.087173650893504, 60.71882727394734,
         -51.727168394006085, -27.24916517958675, 22.012232262025407,
         12.572858131991339, 12.435928384640444, -4.815043768655327],
        [22.75408388833121, .11191258734464782, 51.49359186040597,
         54.408304238663746, 40.717752940380706, -42.652819053745375,
         -14.149447091300186, -13.247084293112387, 1.7796455766230501],
        [0.]*9,
    ]
    s = e.exp1.util.Scenario(1, 2, "combined-disturbance", "out", "mixed", config)
    np.testing.assert_array_equal(s.at(1.5)[0], [0., 0., 0.])
    np.testing.assert_allclose(s.at(1.5)[1], [.54, .20, .15])
    for i, time in enumerate((3., 7., 11.)):
        np.testing.assert_allclose(s.at(time-1e-6)[2], fixture[i-1] if i else np.zeros(9))
        np.testing.assert_allclose(s.at(time)[2], fixture[i])
        np.testing.assert_allclose(s.at(time+1e-6)[2], fixture[i])
    before = copy.deepcopy(config)
    monkeypatch.setitem(e.EVAL_CONFIG["overrides"]["noise"], "noise_level", 99.)
    monkeypatch.setitem(e.EVAL_CONFIG["overrides"]["domain_rand"], "friction_range", [9., 10.])
    assert e.exp1.EVAL_CONFIG == before  # Exp2 edits cannot mutate Exp1 nested settings.
    local = copy.deepcopy(config)
    isolated = e.exp1.util.Scenario(1, 2, "combined-disturbance", "out", "mixed", local)
    local["event_times"][0] = 99.
    np.testing.assert_allclose(isolated.at(3.)[2], fixture[0])


def limit_trials(experiment, counts, seed=42):
    """Synthetic saved-trial records with deliberately unequal exposure lengths."""
    result = []
    for i, (count, denominator) in enumerate(counts):
        if experiment == 2:
            s = scenario(trial=i)
            metric = u.TrialMetrics(1400, .02, s, c)
        else:
            s = e.exp1.util.Scenario(seed, i, "nominal", "in", "stationary", e.exp1.EVAL_CONFIG)
            metric = e.exp1.util.TrialMetrics(100, .02, s, e.exp1.EVAL_CONFIG)
        row = metric.finish("height" if i else "")
        row.update(seed=seed, S_track=.2 if i == 0 else .8, ee_error_m=.1 if i == 0 else .9)
        for name in ("position_violation", "velocity_violation", "actuator_saturation"):
            row[name+"_count"], row[name+"_denominator"] = count, denominator
            row[name+"_frequency"] = count/denominator if denominator else None
        result.append(row)
    return result


@pytest.mark.parametrize("experiment", [1, 2])
@pytest.mark.parametrize("counts,expected", [([(10, 100), (10, 10)], 20/110),
                                             ([(0, 0), (10, 10)], 1.), ([(0, 0), (0, 0)], None)])
def test_pooled_physical_limits_and_seed_weighting(experiment, counts, expected):
    aggregate = e.exp1.util.aggregate_seed if experiment == 1 else u.aggregate_seed
    summary = aggregate(limit_trials(experiment, counts))
    row = next(iter(summary.values()))
    assert row["S_track"] == .5 and row["ee_error_m"] == .5
    assert row["physical_failure_rate"] == .5
    for name in ("position_violation", "velocity_violation", "actuator_saturation"):
        assert row[name+"_frequency"] == expected
        assert row[name+"_count"] == sum(n for n, d in counts)
        assert row[name+"_denominator"] == sum(d for n, d in counts)
        assert row[name+"_frequency_trial_count"] == sum(d > 0 for n, d in counts)
    other_seed = aggregate(limit_trials(experiment, [(0, 1000)], seed=43))
    across = next(iter(e.exp1.util.aggregate_across_seeds([summary, other_seed]).values()))
    # Across seeds, do NOT pool the 1000 exposures from the second seed.
    assert across["actuator_saturation_frequency"]["mean"] == (expected/2 if expected is not None else 0.)
    assert across["actuator_saturation_frequency"]["seeds"] == (2 if expected is not None else 1)


@pytest.mark.parametrize("experiment", [1, 2])
def test_summary_regeneration_without_raw_changes(tmp_path, experiment):
    import json
    import subprocess
    import sys
    trials = limit_trials(experiment, [(10, 100), (10, 10)])
    (tmp_path / "trials.json").write_text(json.dumps(trials))
    (tmp_path / "metadata.json").write_text(json.dumps({} if experiment == 1 else {"experiment": 2}))
    (tmp_path / "COMPLETE").write_text("2 trials")
    (tmp_path / "transitions-000000.npz").write_bytes(b"raw sentinel: never read or rewritten")
    (tmp_path / "records.json").write_text("raw schema sentinel")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    (tmp_path / "seed_summary.json").write_text('{"old_frequency": 0.55}')
    subprocess.run([sys.executable, str(ROOT / "legged_gym/utils/b1z1_experiment.py"), str(tmp_path)], check=True)
    summary = json.loads((tmp_path / "seed_summary.json").read_text())
    row = next(iter(summary.values()))
    assert row["actuator_saturation_frequency"] == 20/110
    assert ("stages" in row) == (experiment == 2)
    assert {p.name for p in tmp_path.iterdir()} == set(before) | {"seed_summary.json"}
    assert all((tmp_path / name).read_bytes() == content for name, content in before.items())
    # Failed regeneration must not replace the last valid summary.
    last = (tmp_path / "seed_summary.json").read_bytes()
    (tmp_path / "COMPLETE").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        e.exp1.util.regenerate_seed_summary(tmp_path)
    assert (tmp_path / "seed_summary.json").read_bytes() == last
