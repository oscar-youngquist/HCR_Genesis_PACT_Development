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


def test_exp1_configuration_and_scenarios_unchanged():
    import subprocess
    from types import ModuleType
    baseline = ModuleType("exp1_head")
    baseline.__file__ = str(ROOT / "legged_gym/scripts/eval_b1z1_exp1.py")
    source = subprocess.check_output(["git", "show", "HEAD:legged_gym/scripts/eval_b1z1_exp1.py"], cwd=ROOT, text=True)
    exec(compile(source, baseline.__file__, "exec"), baseline.__dict__)
    assert baseline.EVAL_CONFIG == e.exp1.EVAL_CONFIG
    for condition in ("nominal", "ee-disturbance", "combined-disturbance"):
        old = baseline.util.Scenario(1, 2, condition, "out", "mixed", baseline.EVAL_CONFIG)
        new = e.exp1.util.Scenario(1, 2, condition, "out", "mixed", e.exp1.EVAL_CONFIG)
        assert old.metadata == new.metadata
        for time in (0, 3, 7, 11, 12):
            for x, y in zip(old.at(time), new.at(time)):
                np.testing.assert_array_equal(x, y)
